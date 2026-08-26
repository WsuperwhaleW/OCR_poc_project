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
import prompts  # constants only, no imports of its own beyond the standard library
import settings  # for OLLAMA_SYSTEM and OLLAMA_REASONING_EFFORT; imports just config and jobs, so no cycle

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
# Per endpoint EXTRACTION model, when pass 2 is to run on a different one from
# pass 1. Unset -- the normal case -- means both passes go to `_chosen`.
#
# The two passes ask for different things and the sweep measured how differently:
# pass 1 wants a model that can read Thai off a page, pass 2 wants one that can
# map a transcript onto a form, and the best model in this project at the second
# (`qwen3.5:4b`, 81.0%) cannot do the first at all well while the model chosen
# for the first (typhoon, 42.4% on the form) is near the bottom of the second.
# Splitting the choice is what lets both be picked on their own evidence.
_extract_chosen = {}
_cache = {}


# A model whose name says it is an OCR fine-tune. A heuristic and nothing more --
# it is a substring test on names people chose -- but it is only ever used to
# refuse a configuration, never to select one, so its failure mode is a refusal
# the user can work around by renaming or by picking "same as reading model".
#
# It catches `typhoon-ocr1.5-3b`, `dots.ocr`, and `dots.mocr` (the `m` is part of
# the repo name, and `ocr` still matches inside it).
_OCR_NAME_HINTS = ("ocr",)


def is_ocr_model(name: str) -> bool:
    """True where a model name says it is an OCR fine-tune."""
    lowered = (name or "").lower()
    return any(hint in lowered for hint in _OCR_NAME_HINTS)


def profile_for_model(name: str) -> str:
    """The pass-1 profile a model needs (`prompts.OCR_PROFILES` key).

    The prompt and the system-message veto are properties of the MODEL, not
    preferences: a dots build given the typhoon profile returns an empty
    transcript at HTTP 200, which logs as a clean run and scores 0.0%. Selecting
    a model therefore selects its profile -- see `app.servers_select`.

    A name heuristic, like `is_ocr_model`, and acceptable for the same reason:
    an unrecognised name falls back to the shipped default, which is what every
    pass-1 baseline was measured under, rather than to something exotic.
    """
    lowered = (name or "").lower()
    for hint, profile in prompts.OCR_PROFILE_BY_NAME:
        if hint in lowered:
            return profile
    return prompts.DEFAULT_OCR_PROFILE


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


def known(url: str = None) -> dict:
    """The last probe of `url`, however old, or None if it has never been probed.

    **Never touches the network.** It exists for callers that want to DESCRIBE
    the server rather than use it -- the Summary tab's environment card, which
    is repainted whenever the run-log card refreshes itself, every five seconds
    while the tab is open.

    That is the whole reason it is not `probe(force=False)`: a miss or an expired
    entry makes `probe` go and ask, and on an unreachable endpoint asking costs
    two connect timeouts. Measured, that took the run-log summary from
    milliseconds to **5.1 s**, which is both a slow card and a standing violation
    of this file's own rule -- *never poll the model server*. A display has no
    business making a request.

    Staleness is the caller's to handle: the entry carries no timestamp here
    because everything that uses it says "as last seen" rather than "now".
    """
    with _lock:
        url = clean_url(url) if url else _active
    hit = _cache.get(url)
    return hit[1] if hit else None


def probe(url: str, force: bool = False) -> dict:
    """What is listening at `url`. Cached for CACHE_TTL seconds.

    **This can make a network request**, so nothing on a polling path may call
    it -- see `known`.
    """
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
    # An OCR model first, then any vision model, then whatever is there.
    #
    # **The order matters and this used to be one loop.** `/api/tags` returns
    # most-recently-modified first, so "the first model reporting vision" meant
    # the newest pull won a fresh process -- and once general vision models were
    # pulled for pass 2, a restart could silently resolve pass 1 onto one of
    # them. Every pass-1 baseline in this project was measured on an OCR model,
    # and `qwen3.5`/`phi4-mini` are here to extract, not to read pages.
    #
    # It only ever picks a DEFAULT: an explicit choice above still wins, so a
    # general model can be selected deliberately.
    for wanted in (lambda m: m["vision"] and is_ocr_model(m["name"]),
                   lambda m: m["vision"]):
        for m in info["models"]:
            if wanted(m):
                return m["name"]
    return names[0]


def status(url: str = None, force: bool = False) -> dict:
    """Uniform status for one endpoint, defaulting to the active one.

    Shape is deliberately the same for both server kinds: `available`, `vision`,
    `model`, `url` and `reason` are what the page and the request path consume.

    **Two availabilities, because the two passes need different things.**
    `available` is pass 1's: it wants a model that can read an image, so a model
    Ollama reports as non-vision fails it. `text_available` is pass 2's, which
    sends text and gets text back and does not care. Kept as a second flag rather
    than left for each caller to re-derive from `reachable` and `model`: the
    Fields pane and `POST /api/extract` both said in comments that vision was not
    required here and both then tested `available`, so a text-only model was
    refused by a pane built to measure exactly that.
    """
    with _lock:
        url = clean_url(url) if url else _active
    info = probe(url, force=force)
    model = _resolve_model(url, info)
    # What is wrong for a caller that needs no image. Kept separate from `reason`
    # below, which the vision message overwrites: telling someone running a
    # text-only pass that their model cannot read an image is a true statement
    # about the wrong problem.
    text_reason = info["reason"]
    reason = text_reason
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
        # Pass 2's version of the same question: a server and a model, and
        # nothing about images.
        "text_available": bool(info["reachable"] and model is not None),
        "text_reason": text_reason,
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


# --------------------------------------------------------------------------
# freeing the GPU when the model changes
# --------------------------------------------------------------------------
#
# Only Ollama needs any of this. llama-server holds the one model it was started
# with for its whole life, so there is nothing a switch could release; Ollama
# keeps every model it has served resident for `keep_alive` (5 minutes by
# default) and loads the next one *beside* it, so picking a second model in the
# page puts two sets of weights on one card and the second load spills to CPU or
# fails outright.
#
# `ollama stop <model>` is a request, not a signal: the CLI posts an empty
# generation with keep_alive 0 and the scheduler evicts the weights. That is
# exactly what is sent here, so this is the CLI command and not an imitation of
# it.

# Eviction is quick, but it happens on the same scheduler that is loading the
# model being switched to, so the read timeout is generous rather than PROBE's.
UNLOAD_TIMEOUT = (1.5, 20)


def _same_model(a: str, b: str) -> bool:
    """Model names, comparing an implicit `:latest` with an explicit one.

    `/api/tags` and `/api/ps` both spell the tag out, but a name typed into the
    picker or passed to `compare.py --model` may not, and a mismatch here would
    stop the model that was just selected.
    """
    def norm(name):
        name = (name or "").strip()
        return name if ":" in name else name + ":latest"
    return bool(a) and bool(b) and norm(a) == norm(b)


def loaded_models(url: str) -> list:
    """What Ollama currently holds in memory at `url`, from `/api/ps`.

    Not `/api/tags`: that is everything pulled, which on this machine is most of
    a disk. Only resident models cost VRAM and only they are worth stopping.
    Failure is reported as "nothing loaded" -- an endpoint that cannot answer
    this is one there is no safe way to unload anything on.
    """
    try:
        res = requests.get(f"{url}/api/ps", timeout=PROBE_TIMEOUT)
        if res.status_code != 200:
            return []
        body = res.json()
    except Exception:
        return []
    if not isinstance(body, dict):
        return []
    names = []
    for entry in body.get("models") or []:
        name = entry.get("name") or entry.get("model") or ""
        if name:
            names.append(name)
    return names


def stop_model(url: str, model: str) -> bool:
    """Evict one model from Ollama's memory now. True if it acknowledged.

    A failure is returned, never raised: the switch itself has already happened
    and refusing to complete it because the old model would not let go would be
    worse than leaving the memory occupied for its keep_alive.
    """
    try:
        res = requests.post(f"{url}/api/generate",
                            json={"model": model, "keep_alive": 0},
                            timeout=UNLOAD_TIMEOUT)
        return res.status_code == 200
    except Exception:
        return False


def free_gpu(url: str, keep=None) -> list:
    """Stop every model resident at an Ollama endpoint except `keep`.

    Returns the names actually stopped, which is what the page reports -- a
    switch that silently unloaded something is indistinguishable from one that
    did nothing, and the two have very different consequences for the next run.

    **It stops models this app did not load**, deliberately: the old model is
    not always the one this process last selected (switch A -> B -> C without
    running B and it is A that is still resident), and the point of the feature
    is a free card rather than tidy bookkeeping. On a shared Ollama that is the
    wrong trade -- `OLLAMA_UNLOAD_ON_SWITCH=0` turns the whole thing off.

    A non-Ollama endpoint returns [] without a request: `probe` is cached, and
    llama-server has nothing to unload.

    `keep` takes one name or several. Several is the two-model case: with a
    separate extraction model, evicting everything but the reading model would
    make every pass-2 request pay a fresh load, which is the cost this whole
    function exists to avoid on the other side.
    """
    if probe(url)["kind"] != "ollama":
        return []
    keepers = [keep] if isinstance(keep, str) else list(keep or [])
    keepers = [k for k in keepers if k]
    stopped = []
    for name in loaded_models(url):
        if any(_same_model(name, k) for k in keepers):
            continue
        if stop_model(url, name):
            stopped.append(name)
    return stopped


def select(url: str = None, model: str = None, unload: bool = True) -> dict:
    """Point the app at an endpoint, optionally at one of its models.

    An unknown URL is added to the list rather than rejected, so the page can
    offer a free-text box for a port that was not configured up front.

    Whatever Ollama was holding is then stopped, so the model being switched to
    loads onto a card the model being switched from has let go of -- see
    `free_gpu`. The names stopped ride back on the status dict as `unloaded`.
    Two things about when it runs:

    * The endpoint that was left is unloaded in full, and the one arrived at
      keeps only the model now selected. Switching Ollama -> llama.cpp is the
      case that most needs it and the one a model-only check would miss.
    * `unload=False` is the caller's veto, and `app.py` uses it while the queue
      is working. Eviction is a request to the same scheduler that is serving
      the run in flight, so a switch made mid-batch would be paid for by the
      document being read.
    """
    global _active
    with _lock:
        was = _active
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
    info = status(url, force=True)

    stopped = []
    if unload and settings.OLLAMA_UNLOAD_ON_SWITCH:
        if was != url:
            stopped += free_gpu(was)
        stopped += free_gpu(url, keep=[info["model"], extract_status(url)["model"]])
        for name in stopped:
            config.say(f"[ollama] stopped {name} to free the GPU")
    return {**info, "unloaded": stopped}


def extract_model(url: str = None) -> str:
    """The model pass 2 is set to run on, or None meaning the reading model."""
    with _lock:
        url = clean_url(url) if url else _active
        return _extract_chosen.get(url)


def extract_status(url: str = None, force: bool = False) -> dict:
    """`status`, but describing the model pass 2 will actually run on.

    Same shape as `status` so every request builder downstream keeps working
    unchanged -- `request_extras`, `structured_request` and `system_prefix` all
    read `info["model"]`, and handing them this dict is the whole of what makes
    a second model work.

    Returns the reading model's status untouched when no separate extraction
    model is set, which is the normal case and the one every measurement in this
    project was taken under.
    """
    info = status(url, force=force)
    chosen = extract_model(url)
    if not chosen or _same_model(chosen, info["model"]):
        return info
    # Vision is irrelevant here and deliberately not re-derived: pass 2 sends
    # text. What matters is that the endpoint is up and the model is one it
    # serves, which `text_available` already says.
    served = [m["name"] for m in info["models"]]
    if served and not any(_same_model(chosen, name) for name in served):
        return {**info, "model": chosen, "text_available": False,
                "text_reason": (f"{chosen} is not served by {info['url']}. "
                                "Pick another extraction model.")}
    return {**info, "model": chosen}


def select_extract(model: str, url: str = None, unload: bool = True) -> dict:
    """Choose the model pass 2 runs on. Empty or None means "same as reading".

    **One configuration is refused: a DIFFERENT OCR model.** Reading with one OCR
    fine-tune and extracting with another is the one combination that cannot be
    the right answer -- it pays for a second set of weights to get a second
    model that is bad at pass 2, and the sweep is unambiguous that OCR
    fine-tunes are the wrong tool for the form (typhoon 42.4%, dots.mocr 41.6%,
    dots.ocr 4.1%, against 81.0% for a general model). Extracting with the
    reading model itself is still allowed, because that is the one-model setup
    every baseline in this project was measured under.
    """
    with _lock:
        url = clean_url(url) if url else _active
    chosen = (model or "").strip()
    if chosen:
        reading = status(url)["model"]
        if is_ocr_model(chosen) and not _same_model(chosen, reading):
            raise ValueError(
                f"{chosen} is an OCR model, and {reading or 'the reading model'} "
                "is already reading the page. A second OCR model is the one "
                "combination that cannot help: pass 2 is a text task, and OCR "
                "fine-tunes score worst on it. Pick a general model, or 'same as "
                "reading model'.")
    with _lock:
        if chosen:
            _extract_chosen[url] = chosen
        else:
            _extract_chosen.pop(url, None)
    info = extract_status(url)
    stopped = []
    if unload and settings.OLLAMA_UNLOAD_ON_SWITCH:
        stopped = free_gpu(url, keep=[status(url)["model"], info["model"]])
        for name in stopped:
            config.say(f"[ollama] stopped {name} to free the GPU")
    return {**info, "unloaded": stopped}


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
    server = status(force=force)
    extract = extract_status(active)
    return {
        "endpoints": listed,
        "active": active,
        "server": server,
        # What pass 2 will run on, and which of this endpoint's models may be
        # chosen for it. The page needs the second list because the refusal in
        # `select_extract` should be visible before it is triggered, not after.
        "extract": {
            "model": extract["model"],
            "chosen": extract_model(active) or "",
            "available": extract["text_available"],
            "reason": extract["text_reason"],
            "same_as_reading": not extract_model(active),
            "choices": [m["name"] for m in server["models"]
                        if not is_ocr_model(m["name"])
                        or _same_model(m["name"], server["model"])],
        },
    }


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

    `reasoning_effort` goes to Ollama only, and it is what keeps a reasoning model
    usable here at all: such a model returns its chain of thought in a separate
    field and leaves `content` empty until it stops thinking, so every capped
    request this app makes comes back empty. `settings.OLLAMA_REASONING_EFFORT`
    carries the measurement, including the check that it is a byte-for-byte no-op
    on a model that does not think.
    """
    info = info or status()
    chosen = num_ctx()
    if info["kind"] == "ollama" and info["model"]:
        extras = {"model": info["model"]}
        if settings.OLLAMA_REASONING_EFFORT:
            extras["reasoning_effort"] = settings.OLLAMA_REASONING_EFFORT
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
        body = {
            "model": info["model"], "messages": messages, "format": schema,
            "options": options, "stream": False,
        }
        # The native endpoint spells it `think`, not `reasoning_effort`, and a
        # thinking model here fails exactly as it does on /v1: the whole
        # num_predict goes on the chain of thought and `message.content` comes
        # back empty. Accepted by a model that does not think (verified on
        # typhoon-ocr1.5-3b), so it is sent whenever thinking is switched off
        # rather than only for models known to do it.
        if settings.OLLAMA_REASONING_EFFORT == "none":
            body["think"] = False
        return f"{active_url()}/api/chat", body

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
