"""Model server endpoints, and switching between them while the app runs.

Two flavours of server speak the same OpenAI-compatible `/v1/chat/completions`
API but differ everywhere else:

* llama.cpp's `llama-server` -- one model per process, exposes `/props` and
  `/slots`, and ignores the "model" field of a request.
* Ollama -- many models in one process, exposes `/api/tags` and `/api/ps`, has no
  `/props`, and REQUIRES the model name in every request.

Every difference between the two is decided in this module, so `app.py` only ever
asks for "the active endpoint" and gets one uniform status dict back.

Endpoints are probed lazily and cached for a couple of seconds: `status()` is
called once per page, per extraction and per verification, and an uncached probe
would add an HTTP round trip to each of those.
"""

import os
import threading
import time

import requests

import config
import settings  # for OLLAMA_SYSTEM only; imports just config and jobs, so no cycle

# (connect, read). A local server accepts a connection immediately, so a short
# connect timeout is what keeps a dead port cheap: an endpoint that is not there
# is probed twice (llama.cpp, then Ollama) and both attempts are paid in full.
# Windows does not always refuse a closed loopback port promptly -- without this
# a switch to an unused port took 12 s.
PROBE_TIMEOUT = (1.5, 4)
CACHE_TTL = 3.0
# An endpoint that is not there is remembered for longer: a dead port costs a
# full connect timeout on every probe, and the page asks for status on each
# render. "Re-check" forces a probe anyway, so nothing waits on this after
# starting a server.
MISS_TTL = 8.0


def clean_url(url: str) -> str:
    """Normalise user input: strip spaces, add a scheme, drop a trailing slash."""
    url = (url or "").strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


def _defaults():
    """Endpoints offered in the picker.

    OCR_ENDPOINTS overrides the list entirely (comma separated). Otherwise the
    two servers this machine actually runs are offered: llama-server on 8080
    (from LLAMA_URL, and first so it stays the default) and Ollama on 11434.
    Anything else is reachable through the picker's "Other address" box.
    """
    listed = [u for u in (os.environ.get("OCR_ENDPOINTS") or "").split(",") if u.strip()]
    if not listed:
        listed = [
            os.environ.get("LLAMA_URL", "http://127.0.0.1:8080"),
            "http://127.0.0.1:11434",
        ]
    seen, out = set(), []
    for url in listed:
        url = clean_url(url)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


_lock = threading.Lock()
_endpoints = _defaults()
_active = _endpoints[0]
# Per endpoint model choice. Only Ollama needs one; llama-server serves whatever
# it was started with, so the entry is left unset there.
_chosen = {}
_cache = {}


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def _probe_llama(url):
    """llama-server if /props answers with its settings object."""
    try:
        res = requests.get(f"{url}/props", timeout=PROBE_TIMEOUT)
        if res.status_code != 200:
            return None
        props = res.json()
    except Exception:
        return None
    if not isinstance(props, dict) or "default_generation_settings" not in props:
        return None

    model = props.get("model_alias") or props.get("model_path") or "unknown"
    vision = bool((props.get("modalities") or {}).get("vision"))
    reason = None
    if not vision:
        reason = (
            f"llama-server is running '{model}' with vision disabled "
            "(modalities.vision=false). It was started without an --mmproj "
            "projector, so it returns 500 for any image. Restart it with "
            "--mmproj <mmproj-...gguf> to enable OCR."
        )
    return {
        "kind": "llama.cpp",
        "reachable": True,
        "model": model,
        "models": [{"name": model, "vision": vision}],
        "vision": vision,
        "slots": props.get("total_slots"),
        "reason": reason,
    }


def _probe_ollama(url):
    """Ollama if /api/tags answers with its model list."""
    try:
        res = requests.get(f"{url}/api/tags", timeout=PROBE_TIMEOUT)
        if res.status_code != 200:
            return None
        body = res.json()
    except Exception:
        return None
    if not isinstance(body, dict) or "models" not in body:
        return None

    models = []
    for entry in body.get("models") or []:
        caps = entry.get("capabilities")
        models.append({
            "name": entry.get("name") or entry.get("model") or "",
            # Older Ollama builds omit capabilities; None means "not stated",
            # which is reported honestly rather than guessed at.
            "vision": ("vision" in caps) if isinstance(caps, list) else None,
            "size_gb": round((entry.get("size") or 0) / 1024 ** 3, 2),
            "family": (entry.get("details") or {}).get("family", ""),
        })
    models = [m for m in models if m["name"]]

    if not models:
        return {"kind": "ollama", "reachable": True, "model": None, "models": [],
                "vision": False, "slots": None,
                "reason": f"Ollama is running at {url} but has no models pulled. "
                          "Pull a vision model, e.g. "
                          "`ollama pull scb10x/typhoon-ocr1.5-3b`."}
    return {"kind": "ollama", "reachable": True, "model": None, "models": models,
            "vision": None, "slots": None, "reason": None}


def probe(url: str, force: bool = False) -> dict:
    """What is listening at `url`. Cached for CACHE_TTL seconds."""
    url = clean_url(url)
    if not force:
        hit = _cache.get(url)
        if hit:
            at, cached = hit
            ttl = CACHE_TTL if cached["reachable"] else MISS_TTL
            if time.time() - at < ttl:
                return cached

    info = _probe_llama(url) or _probe_ollama(url) or {
        "kind": None,
        "reachable": False,
        "model": None,
        "models": [],
        "vision": False,
        "slots": None,
        "reason": (
            f"No model server reachable at {url}. Start llama-server or Ollama "
            "there, or pick another endpoint above."
        ),
    }
    info = {**info, "url": url}
    # Stamped on completion, not on entry: a probe that took three seconds to
    # time out would otherwise be stale the moment it was stored, and the very
    # next caller would pay for it again.
    _cache[url] = (time.time(), info)
    return info


def _resolve_model(url, info):
    """Which model an Ollama endpoint should be asked for.

    The explicit choice wins if it is still installed; otherwise the first model
    that reports vision, otherwise the first model at all -- so a freshly added
    endpoint is usable without picking anything.
    """
    if info["kind"] != "ollama" or not info["models"]:
        return info.get("model")
    names = [m["name"] for m in info["models"]]
    picked = _chosen.get(url)
    if picked in names:
        return picked
    for m in info["models"]:
        if m["vision"]:
            return m["name"]
    return names[0]


def status(url: str = None, force: bool = False) -> dict:
    """Uniform status for one endpoint, defaulting to the active one.

    Shape is deliberately the same for both server kinds: `available`, `vision`,
    `model`, `url` and `reason` are what the page and the request path consume.
    """
    with _lock:
        url = clean_url(url) if url else _active
    info = probe(url, force=force)
    model = _resolve_model(url, info)
    reason = info["reason"]
    vision = info["vision"]

    if info["kind"] == "ollama" and model:
        entry = next((m for m in info["models"] if m["name"] == model), None)
        vision = entry["vision"] if entry else None
        if vision is False:
            reason = (
                f"Ollama model '{model}' has no vision capability, so it cannot "
                "read an image. Choose a vision model above, or pull one, e.g. "
                "`ollama pull scb10x/typhoon-ocr1.5-3b`."
            )
    # vision None means the server did not say. Attempting the read and letting
    # it fail is better than refusing a model that would have worked.
    available = bool(info["reachable"] and vision is not False and model is not None)
    return {
        "available": available,
        "vision": bool(vision),
        "vision_known": vision is not None,
        "kind": info["kind"],
        "model": model,
        "models": info["models"],
        "slots": info["slots"],
        "url": url,
        "reason": reason,
        # Sent with every request, whichever kind is active.
        "num_ctx": num_ctx(),
    }


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def endpoints():
    with _lock:
        return list(_endpoints)


def active_url():
    with _lock:
        return _active


def select(url: str = None, model: str = None) -> dict:
    """Point the app at an endpoint, optionally at one of its models.

    An unknown URL is added to the list rather than rejected, so the page can
    offer a free-text box for a port that was not configured up front.
    """
    global _active
    with _lock:
        if url:
            url = clean_url(url)
            if not url:
                raise ValueError("Empty server URL.")
            if url not in _endpoints:
                _endpoints.append(url)
            _active = url
        else:
            url = _active
        if model:
            _chosen[url] = model
    # Force: the point of switching is to see the new server's real state.
    return status(url, force=True)


def overview(force: bool = False) -> dict:
    """Every configured endpoint plus the active one, for the picker."""
    active = active_url()
    listed = []
    for url in endpoints():
        info = probe(url, force=force)
        listed.append({
            "url": url,
            "kind": info["kind"],
            "reachable": info["reachable"],
            "models": [m["name"] for m in info["models"]],
            "active": url == active,
        })
    return {"endpoints": listed, "active": active, "server": status(force=force)}


# Ollama's default context is 4096, which the extract prompt plus a real
# transcript overruns -- the reply is then generated with the instructions
# already slid out of the window. OLLAMA_CONTEXT_LENGTH fixes it server-side but
# has to be set before the server starts; this asks for the same thing per
# request, so a session gets the right window without restarting Ollama.
# Offered in the page's picker. A bigger window costs KV-cache memory whether or
# not the document fills it, so this is a real trade rather than "more is better"
# -- hence a choice on the page instead of one value baked in at startup.
NUM_CTX_CHOICES = [4096, 6144, 8192, 12288, 16384, 24576, 32768]
NUM_CTX_MIN, NUM_CTX_MAX = 4096, 32768

# Read after the bounds are defined so a value outside them is clamped at startup
# rather than sitting in `_num_ctx` as something `set_num_ctx` would have refused.
OLLAMA_NUM_CTX = config.env_int("OLLAMA_NUM_CTX", 8192,
                                minimum=NUM_CTX_MIN, maximum=NUM_CTX_MAX)

_num_ctx = OLLAMA_NUM_CTX


def num_ctx() -> int:
    with _lock:
        return _num_ctx


def set_num_ctx(value) -> int:
    """Set the window asked for on every later Ollama request.

    Applies from the next request onwards: a generation already in flight keeps
    the window it was started with.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError("Context size must be a number.")
    if not NUM_CTX_MIN <= value <= NUM_CTX_MAX:
        raise ValueError(
            f"Context size must be between {NUM_CTX_MIN} and {NUM_CTX_MAX}.")
    global _num_ctx
    with _lock:
        _num_ctx = value
    return value


def system_prefix(info: dict = None, enabled: bool = True) -> list:
    """Messages to put before the user message, for the active server.

    Ollama fills an empty system slot from the served model's Modelfile --
    typhoon-ocr1.5-3b ships `SYSTEM You are a helpful assistant.` -- so a request
    with no system message is not a request without a system prompt. This sends
    that text explicitly, which measured byte-identical to letting Ollama inject
    it, so the app no longer depends on a default it does not control: a
    re-`ollama create` with a different Modelfile can no longer move accuracy
    without anything here changing. `settings.OLLAMA_SYSTEM` has the numbers.

    llama-server has no Modelfile and no injected default, and every llama.cpp
    baseline was measured with no system message, so it is still sent none.
    Returned as a list so a call site can splice it in without a conditional.

    `enabled` is the pass-1 profile's veto. Whether a system message helps is a
    property of the served model, not of the backend: it is worth 2.42 points on
    typhoon and fatal on dots.ocr, which answers two tokens and an empty string
    when the slot is filled. The profile that knows which model it is written for
    passes False; the endpoint difference stays here.
    """
    info = info or status()
    if not enabled or info["kind"] != "ollama" or not settings.OLLAMA_SYSTEM:
        return []
    return [{"role": "system", "content": settings.OLLAMA_SYSTEM}]


def request_extras(info: dict = None) -> dict:
    """Fields to merge into a chat request for the active server.

    Ollama routes on the model name and 404s without one; llama-server serves a
    single model and ignores the field, so it is only sent where it is needed.

    The context window is sent to both, under each one's own name: Ollama reads
    `options.num_ctx`, llama-server reads a top-level `n_ctx`. A build that does
    not honour it ignores the field and keeps the window from its own -c, so
    sending it costs nothing where it does not apply.
    """
    info = info or status()
    chosen = num_ctx()
    if info["kind"] == "ollama" and info["model"]:
        extras = {"model": info["model"]}
        if chosen > 0:
            extras["options"] = {"num_ctx": chosen}
        return extras
    return {"n_ctx": chosen} if chosen > 0 else {}


def structured_request(messages: list, schema: dict, max_tokens: int,
                       info: dict = None):
    """URL and body for one non-streaming request whose reply must be JSON.

    Two shapes, and which one you get depends on `schema` rather than on the
    backend:

    * `schema` None -- the plain request, byte for byte what this app has always
      sent: OpenAI-compatible `/v1/chat/completions`, greedy, `response_format:
      {"type": "json_object"}`, DRY where the server takes it. **Every
      measurement in CLAUDE.md was taken on this**, which is why it is still the
      first thing asked.
    * `schema` given -- decoding constrained to those keys. On llama-server that
      is the same endpoint with `response_format.json_schema`. **On Ollama it is
      the native `/api/chat`**, because its `/v1` shim drops everything that
      would hold a small OCR fine-tune to the schema: `json_object` means "valid
      JSON" and nothing about *which* JSON, so the model answers with its own
      {"natural_text": "<the whole page>"} envelope and the shim also drops
      `repeat_penalty`, the one repetition defence Ollama can receive -- `dry_*`
      being a llama.cpp sampler it drops as well.

    So the constrained shape changes two things at once on Ollama, deliberately:
    the grammar stops the envelope, and the penalty stops the degeneration a
    grammar cannot reach, which is repetition *inside* a string value.
    `settings.OLLAMA_REPEAT_PENALTY` has both measurements.

    `schema` is sent as-is, so it must stay inside the JSON Schema subset
    Ollama's grammar runtime accepts: object, array, string, number, properties,
    items, required, enum. No nullable unions, no length or range constraints.

    Returns (url, body). `structured_reply` reads whichever shape comes back.
    """
    info = info or status()
    chosen = num_ctx()
    if schema and info["kind"] == "ollama" and info["model"]:
        options = {
            "num_predict": max_tokens,
            "temperature": 0,
            # Deliberately no top_k. Greedy is already guaranteed by temperature
            # 0, and a single-candidate filter applied before the penalty would
            # make the penalty a no-op -- and the penalty is the load-bearing
            # part. top_p is moot at temperature 0 and is left at the value the
            # measurement was taken with.
            "top_p": 0.5,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
        }
        if chosen > 0:
            options["num_ctx"] = chosen
        return f"{active_url()}/api/chat", {
            "model": info["model"], "messages": messages, "format": schema,
            "options": options, "stream": False,
        }

    return f"{active_url()}/v1/chat/completions", {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        # llama-server takes a schema here; without one this is the plain
        # "return some JSON object" the baselines were measured with.
        "response_format": ({"type": "json_schema",
                             "json_schema": {"name": "fields", "schema": schema}}
                            if schema else {"type": "json_object"}),
        "stream": False,
        **settings.sampler_extras(),
        **request_extras(info),
    }


def structured_reply(body: dict, info: dict = None):
    """(text, truncated, tokens) from either server's reply to the above.

    The two shapes differ in every field: llama-server answers OpenAI-style with
    `choices[0].message.content`, a `finish_reason` and either a `timings` or a
    `usage` block; Ollama's native endpoint answers with `message.content`, a
    `done_reason` and `eval_count`. Read here so the callers stay uniform.
    """
    info = info or status()
    if "choices" in body:
        choice = (body.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        tokens = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
        return text, choice.get("finish_reason") == "length", tokens
    text = ((body.get("message") or {}).get("content") or "").strip()
    return text, body.get("done_reason") == "length", int(body.get("eval_count") or 0)
