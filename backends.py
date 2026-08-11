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
