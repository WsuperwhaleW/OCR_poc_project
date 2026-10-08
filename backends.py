"""Model server endpoints, and switching between them while the app runs.

Two flavours of server speak the same OpenAI-compatible `/v1/chat/completions`
API but differ everywhere else:

* llama.cpp's `llama-server` -- one model per process, exposes `/props` and
  `/slots`, and ignores the "model" field of a request.
* Ollama -- many models in one process, exposes `/api/tags` and `/api/ps`, has no
  `/props`, and REQUIRES the model name in every request.
* vLLM (2026-09-25) -- the high-end-GPU server the stress test is aimed at. It
  answers `/v1/models` with `owned_by: "vllm"`, requires the model name like
  Ollama, and exposes Prometheus `/metrics`. Any other server that answers
  `/v1/models` is kind `openai` and is spoken to the same way, minus `/metrics`.

Every difference between the two is decided in this module, so `app.py` only ever
asks for "the active endpoint" and gets one uniform status dict back.

Endpoints are probed lazily and cached for a couple of seconds: `status()` is
called once per page, per extraction and per verification, and an uncached probe
would add an HTTP round trip to each of those.
"""

import os
import re
import threading
import time

import requests

import config
import prompts  # constants only, no imports of its own beyond the standard library
import runlog  # for the model ranking a default is taken from; imports config, grounding and settings, so no cycle
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
# The endpoint pass 2 runs on, or None meaning "the reading endpoint" -- the
# one-URL setup every measurement in CLAUDE.md was taken under, and still the
# default (2026-10-01). Set, extraction and every other text request (classify,
# segment, the table agents) go there while page images keep going to `_active`.
#
# **Stored literally, and "separate" is decided against `_active` at the time of
# asking** (`extract_separate`). Picking the extraction server as the reading
# server as well collapses back to one URL without forgetting the choice, so
# moving the reader away again puts extraction back where it was put.
#
# `_extract_chosen` stays keyed by the URL the extraction model LIVES on, so in
# the one-URL setup it means exactly what it always did.
_extract_url = clean_url(settings.EXTRACT_URL) or None
if _extract_url and _extract_url not in _endpoints:
    # Listed so the picker can show it. `overview` probes every listed endpoint,
    # and this one is in use, so the probe is not wasted.
    _endpoints.append(_extract_url)
_cache = {}

# --------------------------------------------------------------------------
# Which engine a server is: pinned, or detected and remembered (2026-10-05)
#
# At the user's request -- *pick the inference server engine so there is no
# 404 request; vLLM has /metrics, so with Ollama or llama.cpp disable it*.
#
# Detection asks /props, then /api/tags, then /v1/models, and the cache lasts
# three seconds -- so until now a vLLM server answered two 404s on EVERY
# re-probe, and Ollama one. Two things stop that:
#
# * **A pinned kind is probed with its own endpoint alone.** Nothing another
#   engine has is ever asked of it, and if it does not answer as that engine it
#   is reported unreachable rather than tried as something else -- the pin is a
#   statement about the server, and silently overruling it would put back the
#   requests it exists to stop.
# * **Auto remembers what it found** and asks that first next time, so a
#   detected server costs its 404s once per process rather than every three
#   seconds. "Re-check" (a forced probe) forgets it and detects from scratch,
#   because that is what it is for after a server is swapped on a port.
#
# The kinds are the ones the request builders already switch on, so pinning
# changes which requests are made and nothing about how they are made.
# --------------------------------------------------------------------------

SERVER_KINDS = ("llama.cpp", "ollama", "vllm", "openai")
SERVER_KIND_LABELS = {
    "llama.cpp": "llama.cpp (llama-server)",
    "ollama": "Ollama",
    "vllm": "vLLM",
    "openai": "OpenAI-compatible (other)",
}
# The one endpoint each kind is probed with -- said in the unreachable reason,
# so a pin that does not match the server names what it asked for.
_KIND_PROBE_PATH = {"llama.cpp": "/props", "ollama": "/api/tags",
                    "vllm": "/v1/models", "openai": "/v1/models"}


def _parse_kinds(raw: str) -> dict:
    """`url=kind,url=kind` from SERVER_KINDS. A bad entry warns and is skipped."""
    out = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        url, _, kind = part.rpartition("=")
        url, kind = clean_url(url), kind.strip().lower()
        if kind == "llamacpp" or kind == "llama":
            kind = "llama.cpp"
        if not url or kind not in SERVER_KINDS:
            config.say(f"[config] SERVER_KINDS: ignoring {part!r} -- expected "
                       f"url=kind with kind one of {', '.join(SERVER_KINDS)}")
            continue
        out[url] = kind
    return out


_kind_pinned = _parse_kinds(settings.SERVER_KINDS)
_kind_seen = {}


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
        "models": [{"name": model, "vision": vision,
                    "quant": quant_of(props.get("model_path"))}],
        "vision": vision,
        "slots": props.get("total_slots"),
        "reason": reason,
        # What the run log calls the engine version. Newer builds only; an older
        # one says nothing and the column stays blank rather than guessed.
        "build": props.get("build_info") or "",
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
            # Ollama states it per model in /api/tags, so it costs no request.
            "quant": ((entry.get("details") or {}).get("quantization_level")
                      or quant_of(entry.get("name") or entry.get("model"))),
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


def _probe_openai(url):
    """vLLM (or another OpenAI-compatible server) if `/v1/models` lists models.

    Asked LAST, after llama.cpp and Ollama, because both of those answer
    `/v1/models` as well -- this is the fallback for a server that has neither
    `/props` nor `/api/tags`, which is what vLLM, SGLang and LM Studio look like
    from here.

    **`kind` is `vllm` only where the server says `owned_by: "vllm"`**, and that
    is the whole of what decides whether the stress test may fire `/metrics` at
    it. A generic OpenAI-compatible server is `openai` and never gets that
    request -- the user's rule, *if I am testing on my machine do not fire it*,
    reduced to a fact the server states about itself rather than a guess from
    its address.

    Vision is reported as unknown (None): `/v1/models` does not say, and the
    status rule already attempts a read on a model that did not say rather than
    refusing one that would have worked.
    """
    try:
        res = http_get(f"{url}/v1/models", {"kind": "openai"},
                       timeout=PROBE_TIMEOUT)
        if res.status_code != 200:
            return None
        body = res.json()
    except Exception:
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return None
    models = [{"name": str(m.get("id") or ""), "vision": None,
               "max_model_len": m.get("max_model_len"),
               # /v1/models does not say; the name often does (-AWQ, -FP8).
               "quant": quant_of(m.get("id"))}
              for m in data if isinstance(m, dict) and m.get("id")]
    if not models:
        return None
    vllm = any(isinstance(m, dict) and m.get("owned_by") == "vllm" for m in data)
    return {"kind": "vllm" if vllm else "openai", "claims_vllm": vllm,
            "reachable": True,
            "model": models[0]["name"], "models": models, "vision": None,
            "slots": None, "reason": None}


def _probe_kind(url, kind):
    """Probe `url` as one engine only -- one request, that engine's endpoint."""
    if kind == "llama.cpp":
        return _probe_llama(url)
    if kind == "ollama":
        return _probe_ollama(url)
    return _probe_openai(url)


def server_kind(url: str = None) -> str:
    """The pinned kind of `url`, or "auto"."""
    with _lock:
        url = clean_url(url) if url else _active
    return _kind_pinned.get(url) or "auto"


def set_server_kind(url: str, kind: str) -> dict:
    """Pin `url` to one engine, or "auto" to detect it. Returns its fresh status.

    The cache entry is dropped and the server re-probed under the new rule at
    once, so the answer the page gets back is what the pin found -- a pin that
    does not match the server is reported unreachable now, not on the next run.
    """
    url = clean_url(url)
    if not url:
        raise ValueError("Empty server URL.")
    kind = (kind or "auto").strip().lower()
    if kind not in SERVER_KINDS and kind != "auto":
        raise ValueError(f"Unknown server type {kind!r}; expected auto or one "
                         f"of {', '.join(SERVER_KINDS)}.")
    with _lock:
        if kind == "auto":
            _kind_pinned.pop(url, None)
        else:
            _kind_pinned[url] = kind
        _kind_seen.pop(url, None)
    _cache.pop(url, None)
    return probe(url, force=True)


def serves_metrics(info: dict) -> bool:
    """Does this server expose Prometheus `/metrics` the stress test may read?

    vLLM only. llama-server has one behind `--metrics`, but it shares the task
    queue with inference -- the reason this project never polls it -- and a
    local server is exactly the case the metrics button must stay off for.
    """
    return bool(info and info.get("kind") == "vllm" and info.get("reachable", True))


def parallel_ok(info: dict) -> bool:
    """May independent requests to this server be sent at once?

    vLLM only, on the server's own statement -- the gate `serves_metrics` uses.
    It batches concurrent requests; llama-server serves one per slot and Ollama
    queues them behind `OLLAMA_NUM_PARALLEL`, so there they would only wait.
    """
    return bool(info and info.get("kind") == "vllm" and info.get("reachable", True))


_POOLED_KINDS = ("vllm", "openai")
_session = None
_session_lock = threading.Lock()


def _live_transport() -> bool:
    """Are `requests.post`/`get` the real ones? A test that patches either is
    stubbing the model server, and must go on seeing every request -- so pooling,
    which sends through a Session the patch cannot reach, is off while it does."""
    return (requests.post, requests.get) == runlog._REAL_HTTP


def _pooled(info: dict) -> bool:
    return bool(settings.HTTP_KEEPALIVE and info
                and info.get("kind") in _POOLED_KINDS and _live_transport())


def _http_session():
    """The one keep-alive Session for vLLM / OpenAI-compatible servers.

    `requests.post` builds a Session per call and closes it, so every request
    paid a TCP (and TLS) handshake -- three per pass-2 request on vLLM, with the
    two /metrics snapshots. Thread-safe for sending; the pool is sized above the
    agentic step count and a stress run's concurrency (`HTTP_POOL_SIZE`).
    A streamed response closed before its body is read closes its connection
    rather than returning it, which is what makes cutting off a looping reply
    still abort it on the server.
    """
    global _session
    with _session_lock:
        if _session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=settings.HTTP_POOL_SIZE)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            _session = session
        return _session


def http_post(url: str, info: dict = None, **kwargs):
    """`requests.post`, on the pooled connection when `info` is vLLM/OpenAI."""
    if _pooled(info):
        return _http_session().post(url, **kwargs)
    return requests.post(url, **kwargs)


def http_get(url: str, info: dict = None, **kwargs):
    """`requests.get`, on the pooled connection when `info` is vLLM/OpenAI."""
    if _pooled(info):
        return _http_session().get(url, **kwargs)
    return requests.get(url, **kwargs)


TOKENIZE_TIMEOUT = (5, 30)


def count_prompt(info: dict, payload: dict):
    """(prompt tokens, model window) for a chat payload, from vLLM's /tokenize.

    vLLM renders the chat template exactly as it will for the real request, so
    the count is the one the server checks `max_tokens` against. The template
    switches the request carries (`enable_thinking`) are passed along because
    they change the rendering. (None, None) where the server cannot say.
    """
    if not (info and info.get("kind") == "vllm" and info.get("url")):
        return None, None
    body = {"messages": payload.get("messages") or [],
            "add_generation_prompt": True}
    if payload.get("model") or info.get("model"):
        body["model"] = payload.get("model") or info.get("model")
    if payload.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = payload["chat_template_kwargs"]
    try:
        res = http_post(f"{info['url']}/tokenize", info, json=body,
                        timeout=TOKENIZE_TIMEOUT)
        if res.status_code != 200:
            return None, None
        reply = res.json()
    except Exception:
        return None, None
    count = reply.get("count")
    window = reply.get("max_model_len") or model_window(info)
    if not isinstance(count, int):
        return None, None
    return count, window if isinstance(window, int) else None


def model_window(info: dict):
    """The served model's `max_model_len`, as `/v1/models` reported it, or None."""
    for model in (info or {}).get("models") or []:
        if model.get("name") == info.get("model"):
            return model.get("max_model_len")
    return None


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


# --------------------------------------------------------------------------
# what a run was made ON: quantisation and engine version, for the run log
# --------------------------------------------------------------------------

# A quantisation written into a model's name or file: GGUF types (Q4_K_M,
# IQ3_XXS, Q8_0, F16, BF16) and the formats vLLM serves (AWQ, GPTQ, FP8, INT4,
# W4A16, MXFP4, NVFP4, BNB). Bounded by non-alphanumerics on both sides, so a
# `q4` inside a word is not one; `_` and `.` are boundaries, `Q4_K_M` is whole.
_QUANT = re.compile(
    r"(?<![A-Za-z0-9])("
    r"I?Q[1-8](?:_[A-Z0-9]{1,3}){0,2}"
    r"|BF16|F16|F32|FP16|FP32|FP8|FP4|MXFP4|NVFP4|INT4|INT8"
    r"|AWQ|GPTQ|BNB|EXL2|W[48]A(?:8|16)"
    r")(?![A-Za-z0-9])", re.IGNORECASE)


def quant_of(name) -> str:
    """The quantisation a model's name or file states, upper-cased, or ``.

    The LAST match wins, because a name ends with its file type
    (`typhoon-ocr-1.5-2b-Q8_0.gguf`). `` is "the name does not say", never a
    guess -- a model served without a quant in its name may be anything.
    """
    found = _QUANT.findall(str(name or ""))
    return found[-1].upper() if found else ""


# url -> (when, version). An engine is not upgraded under a running process
# often, but it is restarted; ten minutes keeps a long sweep to one request.
_versions = {}
VERSION_TTL = 600.0
_VERSION_PATH = {"ollama": "/api/version", "vllm": "/version",
                 "openai": "/version"}


def _engine_version(url, info) -> str:
    """The engine's own version string, or ``. Never raises.

    llama.cpp states it in /props, which the probe already read; Ollama answers
    /api/version and vLLM /version -- both from their HTTP handler, NOT the
    inference queue, so this is not the poll the never-poll rule forbids. It is
    asked at most once per VERSION_TTL per server, and only where a run is
    being logged. A server that does not answer is cached as `` too, so a
    generic OpenAI server is asked once rather than on every row.
    """
    kind = info.get("kind")
    if kind == "llama.cpp":
        return info.get("build") or ""
    path = _VERSION_PATH.get(kind)
    if not path or not url:
        return ""
    hit = _versions.get(url)
    if hit and time.time() - hit[0] < VERSION_TTL:
        return hit[1]
    version = ""
    try:
        res = http_get(f"{url}{path}", info, timeout=PROBE_TIMEOUT)
        if res.status_code == 200:
            body = res.json()
            if isinstance(body, dict):
                version = str(body.get("version") or "")
    except Exception:
        version = ""
    _versions[url] = (time.time(), version)
    return version


def model_meta(url: str = None, model: str = None) -> dict:
    """`{"quant", "engine"}` for `model` on `url`, as the run log records them.

    `engine` is the server kind and its version (`ollama 0.32.14`,
    `vllm 0.11.0`, `llama.cpp b6123-...`); the kind alone where the version is
    not known. `quant` comes from what the server says about the model (Ollama),
    else from its file (llama.cpp's model_path), else from its name. Blank where
    nothing says -- the standing rule, blank is not a value.

    Reads the probe cache (`known`); probes only where the URL was never seen.
    Never raises: a row missing these is better than a run whose log failed.
    """
    try:
        url = clean_url(url) if url else active_url()
        info = known(url) or {}
        if not info and url:
            info = probe(url)
        kind = info.get("kind") or ""
        entry = next((m for m in info.get("models") or []
                      if model and m.get("name") == model), {})
        quant = entry.get("quant") or quant_of(model)
        if not quant and kind == "llama.cpp":
            quant = next((m.get("quant") for m in info.get("models") or []
                          if m.get("quant")), "")
        version = _engine_version(url, info) if info.get("reachable") else ""
        engine = f"{kind} {version}".strip() if kind else ""
        return {"quant": quant or "", "engine": engine}
    except Exception:
        return {"quant": "", "engine": ""}


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

    pinned = _kind_pinned.get(url)
    if pinned:
        order = [pinned]
    else:
        if force:
            _kind_seen.pop(url, None)
        # The remembered kind first, then the detection order. llama.cpp and
        # Ollama both answer /v1/models too, so /v1/models stays LAST among the
        # rest -- asked first only where it is what this server turned out to be.
        seen = _kind_seen.get(url)
        order = ([seen] if seen else []) + [
            k for k in ("llama.cpp", "ollama", "openai") if k != seen
            and not (seen == "vllm" and k == "openai")]
    info = None
    for kind in order:
        info = _probe_kind(url, kind)
        if info:
            break
    if info and pinned in ("vllm", "openai"):
        # /v1/models is the same request for both, and the pin is the user's
        # statement of which it is -- what decides whether /metrics is asked.
        info = {**info, "kind": pinned}
        if pinned == "vllm" and not info.get("claims_vllm"):
            # Kept, because the pin is the user's -- but said, because the one
            # request a vLLM pin licenses (/metrics) is likely to 404 here.
            info["kind_warning"] = (
                f"{url} is pinned to vLLM, but its /v1/models does not say "
                "owned_by vllm, so /metrics may answer 404.")
    if info and not pinned:
        _kind_seen[url] = info["kind"]
    info = info or {
        "kind": None,
        "reachable": False,
        "model": None,
        "models": [],
        "vision": False,
        "slots": None,
        "reason": (
            f"No {SERVER_KIND_LABELS[pinned]} server answered at {url} -- "
            f"{_KIND_PROBE_PATH[pinned]} gave nothing usable. Start it there, or "
            "set the server type to Auto if it is a different engine."
            if pinned else
            f"No model server reachable at {url}. Start llama-server, Ollama or "
            "vLLM there, or pick another endpoint above."
        ),
    }
    info = {**info, "url": url, "kind_pinned": pinned}
    # Stamped on completion, not on entry: a probe that took three seconds to
    # time out would otherwise be stale the moment it was stored, and the very
    # next caller would pay for it again.
    _cache[url] = (time.time(), info)
    return info


# --------------------------------------------------------------------------
# The default model: what the run log ranks first
#
# **Added 2026-09-03 at the user's request** -- *auto default model to the best
# one that is shown*. "Shown" is the Summary tab's headline card and the
# standouts list, so the default is read off `runlog.best_models`, which is that
# same ranking rather than a second opinion computed here.
#
# It is a DEFAULT and only a default: an explicit choice still wins in
# `_resolve_model`, and the name heuristics below it still answer for a model the
# log has never scored. What it replaces is a guess -- a substring test on a
# name, which says nothing about how well the model has done here.
#
# **The log is read behind a cache, because `status()` is on every request
# path.** `runlog.signature()` is one `stat()` call; compiling the ranking over a
# 1300-row log is ~10 ms, which is cheap once and not cheap several times a
# request. The signature moves on an append, an in-place rewrite and a delete, so
# the cache cannot go stale on a change to the file. A log that cannot be read at
# all falls back to no preference rather than raising: a default is a
# convenience, and the app must start without one.

_rank_lock = threading.Lock()
_rank_cache = {"signature": None, "models": None}


def _ranked_models(pass_: str) -> list:
    """Model names this log ranks first for `pass_` (`ocr` or `extract`)."""
    try:
        sig = runlog.signature()
    except Exception:
        return []
    with _rank_lock:
        if _rank_cache["signature"] != sig:
            try:
                _rank_cache["models"] = runlog.best_models()
            except Exception as err:  # a malformed log must not stop a read
                config.say(f"[ocr] note: cannot rank models from the run log ({err})")
                _rank_cache["models"] = {"ocr": [], "extract": []}
            _rank_cache["signature"] = sig
        return list((_rank_cache["models"] or {}).get(pass_) or [])


def best_served(pass_: str, models: list, keep=None) -> str:
    """The highest-ranked model of `pass_` that this endpoint serves, or "".

    `models` is the endpoint's own list, and the SERVED spelling is what comes
    back -- the log records `qwen3.5:4b` where `/api/tags` says
    `qwen3.5:4b:latest`, and a request has to carry the name the server knows.

    `keep` filters the candidates the way each pass needs: pass 1 needs vision,
    pass 2 needs a model `select_extract` would not refuse.
    """
    if not settings.AUTO_BEST_MODEL:
        return ""
    for wanted in _ranked_models(pass_):
        for entry in models or []:
            if _same_model(wanted, entry["name"]) and (keep is None or keep(entry)):
                return entry["name"]
    return ""


def _resolve_model(url, info):
    """Which model an Ollama endpoint should be asked for.

    The explicit choice wins if it is still installed; otherwise the best reader
    the run log knows of, otherwise an OCR model, otherwise the first model that
    reports vision, otherwise the first model at all -- so a freshly added
    endpoint is usable without picking anything.
    """
    if info["kind"] in ("vllm", "openai") and info["models"]:
        # One or several served models and no capability flags to rank on: the
        # explicit choice if it is still served, otherwise what the server
        # lists first.
        names = [m["name"] for m in info["models"]]
        picked = _chosen.get(url)
        return picked if picked in names else info.get("model")
    if info["kind"] != "ollama" or not info["models"]:
        return info.get("model")
    names = [m["name"] for m in info["models"]]
    picked = _chosen.get(url)
    if picked in names:
        return picked
    # The log's best reader that this endpoint serves and that can read an image.
    # `vision is not False` rather than `is True`: None means the server did not
    # say, and refusing a model on a missing capability flag would drop every
    # model an older Ollama serves.
    ranked = best_served("ocr", info["models"], lambda m: m["vision"] is not False)
    if ranked:
        return ranked
    # An OCR model first, then any vision model, then whatever is there.
    #
    # **The order matters and this used to be one loop.** `/api/tags` returns
    # most-recently-modified first, so "the first model reporting vision" meant
    # the newest pull won a fresh process -- and once general vision models were
    # pulled for pass 2, a restart could silently resolve pass 1 onto one of
    # them. Every pass-1 baseline in this project was measured on an OCR model,
    # and `qwen3.5`/`phi4-mini` are here to extract, not to read pages.
    #
    # It still answers for a model the log has never scored, which is what the
    # ranking above cannot do -- a freshly pulled build has no runs behind it.
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
        # The user's pin, or "auto" -- whether `kind` was stated or detected.
        "kind_setting": info.get("kind_pinned") or "auto",
        "kind_warning": info.get("kind_warning"),
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


def candidates() -> list:
    """Endpoints worth trying, best bet first: the constants, then the history.

    The picker's list is a CONSTANT -- `_defaults`, i.e. `OCR_ENDPOINTS` or the
    two ports this machine runs -- and `logs/runs.csv` is the HISTORY. Nothing
    joined them until now, so an endpoint that had served a thousand runs was not
    offered again after a restart unless it happened to be one of the two
    defaults or was typed back in by hand.

    Configured first because that is what this deployment was told to use;
    history after it, most recently used first, which is both what a person means
    by "the server I was on" and the order that finds a live one soonest -- see
    `autoselect`, where every dead candidate costs a connect timeout.

    A read of the log that fails returns the constants alone. A missing history
    is a smaller problem than a server that will not start.
    """
    out = list(_endpoints)
    try:
        history = runlog.servers_seen()
    except Exception:
        history = []
    for url in history:
        url = clean_url(url)
        if url and url not in out:
            out.append(url)
    return out


def autoselect(force: bool = True) -> dict:
    """Make the first endpoint that answers the active one.

    **Added 2026-09-03 at the user's request** -- *auto check and select the
    server that is in the history/constant and is online/ready*. The old default
    was `_endpoints[0]`, which is llama-server on :8080; on the machine this was
    written on that port has 2 runs in the log against Ollama's 1296, so a fresh
    process pointed at a dead port until somebody moved the picker.

    Three rules, and the first two are what keep it cheap:

    * **The active endpoint is tried first and kept if it answers.** Auto-select
      is for a process that has not been told anything, and it must never move a
      choice somebody made.
    * **It stops at the first endpoint that answers**, so the usual case is one
      probe. A dead one costs a full connect timeout twice over (llama, then
      Ollama), which is why the candidate order above matters and why
      `AUTO_SELECT_MAX_CANDIDATES` bounds the list.
    * **Nothing is unloaded.** `select` stops whatever Ollama was holding, on the
      grounds that a switch means a new model needs the card. This is not a
      switch away from anything -- it is deciding where to start -- and evicting
      a model somebody else's process just loaded would be a rude way to begin.
    * **Only the endpoint it lands on joins the picker's list.** The history is
      candidates, not configuration: `overview` probes every listed endpoint on
      every call, so adding eight remembered ports would put eight connect
      timeouts on the page's own status fetch.

    Returns what happened: `selected` (the URL now active), `changed`, `tried`,
    and `reason` when nothing answered. Never raises -- an app with no model
    server has to start anyway and say so, which is `preflight`'s whole rule.
    """
    global _active
    active = active_url()
    # A configured extraction server is not a reading candidate: somebody split
    # the two on purpose, and a dead OCR server must not quietly move page reads
    # onto the box that was set aside for pass 2.
    order = [active] + [u for u in candidates()
                        if u != active and u != _extract_url]
    order = order[:max(1, settings.AUTO_SELECT_MAX_CANDIDATES)]
    tried = []
    for url in order:
        tried.append(url)
        try:
            info = probe(url, force=force)
        except Exception:
            continue
        if info["reachable"]:
            with _lock:
                if url not in _endpoints:
                    _endpoints.append(url)
                changed = url != _active
                _active = url
            return {"selected": url, "changed": changed, "tried": tried,
                    "reason": None}
    return {"selected": active, "changed": False, "tried": tried,
            "reason": ("No model server answered at " + ", ".join(tried) + "."
                       if tried else "No endpoints configured.")}


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
            stopped += free_gpu(was, keep=_in_use(was))
        stopped += free_gpu(url, keep=_in_use(url))
        for name in stopped:
            config.say(f"[ollama] stopped {name} to free the GPU")
    return {**info, "unloaded": stopped}


def _in_use(url: str) -> list:
    """The models this process still needs resident at `url`.

    The reading model where `url` is the reading endpoint, the extraction model
    where it is the extraction endpoint -- both, in the one-URL setup, which is
    exactly the keep list `select` always used. With two endpoints, leaving the
    reading server must not evict the extractor off the other box.
    """
    url = clean_url(url)
    keep = []
    if url == active_url():
        keep.append(status(url)["model"])
    ex = extract_status()
    if ex["url"] == url:
        keep.append(ex["model"])
    return [k for k in keep if k]


def extract_url() -> str:
    """The endpoint pass 2 runs on: the separate one if set, else the reader's."""
    with _lock:
        return _extract_url or _active


def extract_separate() -> bool:
    """True where pass 2 goes to a different URL from the page images."""
    with _lock:
        return bool(_extract_url) and _extract_url != _active


def configured_extract_url() -> str:
    """What was chosen for pass 2, "" meaning "same as the reading server".

    Distinct from `extract_url`, which resolves "" to the reading URL: the
    picker has to show the CHOICE, and "the reading server, which is also this
    URL" is a different choice from "this URL".
    """
    with _lock:
        return _extract_url or ""


def select_extract_url(url: str = None, unload: bool = True) -> dict:
    """Point pass 2 at its own endpoint. Empty or None means the reading one.

    Any server kind this module can probe is accepted -- vLLM, a generic
    OpenAI-compatible server, llama-server or Ollama -- because every request
    builder downstream already reads its kind and URL off the status dict it is
    handed. Nothing about the reading side changes.

    The extraction model is NOT carried across: a name chosen on one server is
    rarely served by another, so the new endpoint starts on its own default
    (see `_extract_model_on`) unless a choice was already recorded for it.
    """
    global _extract_url
    url = clean_url(url) if url else None
    old = extract_url()
    with _lock:
        if url and url not in _endpoints:
            _endpoints.append(url)
        _extract_url = url
    info = extract_status(force=True)
    stopped = []
    if unload and settings.OLLAMA_UNLOAD_ON_SWITCH and old != info["url"]:
        stopped = free_gpu(old, keep=_in_use(old))
        for name in stopped:
            config.say(f"[ollama] stopped {name} to free the GPU")
    return {**info, "unloaded": stopped}


def _extract_model_on(url: str, info: dict):
    """The model a SEPARATE extraction endpoint should be asked for.

    The explicit choice in the server's own spelling where it still serves it,
    otherwise the first model that is not an OCR fine-tune, otherwise whatever
    the server names first. No ranking here: the run log's best extractor is
    written down ONCE, by `autoselect_models` at startup, for the same reason
    the one-URL default is -- a default re-derived per request would move under
    a session as the log grew.

    llama-server serves the one model it was started with, so it is that.
    """
    names = [m["name"] for m in info.get("models") or []]
    chosen = _extract_chosen.get(url)
    if chosen:
        served = next((n for n in names if _same_model(chosen, n)), None)
        return served or chosen
    if info.get("kind") == "llama.cpp":
        return info.get("model")
    general = next((n for n in names if not is_ocr_model(n)), None)
    return general or info.get("model") or (names[0] if names else None)


def _separate_status(url: str, force: bool = False) -> dict:
    """`status`'s shape for an extraction endpoint that reads no page.

    Vision is irrelevant -- pass 2 sends text -- so `available` and
    `text_available` are the same claim here: reachable, and a model to ask.
    """
    info = probe(url, force=force)
    model = _extract_model_on(url, info)
    names = [m["name"] for m in info["models"]]
    reason = info["reason"]
    ok = bool(info["reachable"] and model)
    if info["reachable"] and not model:
        reason = reason or f"The extraction server at {url} serves no model."
    elif ok and names and not any(_same_model(model, n) for n in names):
        ok = False
        reason = (f"{model} is not served by the extraction server {url}. "
                  "Pick another extraction model.")
    return {
        "available": ok,
        "text_available": ok,
        "text_reason": reason,
        "vision": False,
        "vision_known": False,
        "kind": info["kind"],
        # The user's pin, or "auto" -- whether `kind` was stated or detected.
        "kind_setting": info.get("kind_pinned") or "auto",
        "kind_warning": info.get("kind_warning"),
        "model": model,
        "models": info["models"],
        "slots": info["slots"],
        "url": url,
        "reason": reason,
        "num_ctx": num_ctx(),
        "separate": True,
    }


def extract_model(url: str = None) -> str:
    """The model pass 2 is set to run on, or None meaning the default.

    The default is the reading model in the one-URL setup and the extraction
    server's own default where pass 2 has an endpoint of its own. `url` is the
    endpoint the extraction model lives on, defaulting to `extract_url()`.
    """
    url = clean_url(url) if url else extract_url()
    with _lock:
        return _extract_chosen.get(url)


def best_extractor(info: dict) -> str:
    """The log's best EXTRACTOR this endpoint serves, or "" if it is the reader.

    **Used once, by `autoselect_models` at startup, and never on a request
    path.** Pass 2 sends text and gets text, so vision is irrelevant here -- what
    the filter enforces instead is the one combination `select_extract` refuses:
    a second OCR fine-tune beside a reader that is already one. Reading with the
    model itself is always allowed, which is why `_same_model` is the exception.

    "" where the winner IS the reading model, so the one-model setup every
    measurement in this project was taken under is left alone rather than
    written down as a choice that changes nothing.
    """
    reading = info.get("model")
    best = best_served(
        "extract", info.get("models") or [],
        lambda m: not is_ocr_model(m["name"]) or _same_model(m["name"], reading))
    return "" if not best or _same_model(best, reading) else best


def extract_status(url: str = None, force: bool = False) -> dict:
    """`status`, but describing the model pass 2 will actually run on.

    Same shape as `status` so every request builder downstream keeps working
    unchanged -- `request_extras`, `structured_request` and `system_prefix` all
    read `info["model"]`, and handing them this dict is the whole of what makes
    a second model work.

    Returns the reading model's status untouched when no separate extraction
    model is set, which is the normal case and the one every measurement in this
    project was taken under.

    **Where pass 2 has an endpoint of its own** (`select_extract_url`, or
    `EXTRACT_URL`), the status is that endpoint's, and its `url` is what
    `structured_request` sends to -- which is the whole of what moves every text
    request off the reading server. `url` here names the READING endpoint and is
    only honoured in the one-URL setup.
    """
    if (url is None or clean_url(url) == active_url()) and extract_separate():
        return _separate_status(extract_url(), force=force)
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


def autoselect_models(url: str = None) -> dict:
    """Select the models the run log ranks first, once, at startup.

    **Added 2026-09-03 at the user's request** -- *auto default model to the best
    one that is shown ... I am just lazy manual selecting*. It is exactly what
    picking them in the page would do, done for you before the first render, and
    it is **called once from `preflight` and from nowhere else**: after this the
    models are an ordinary choice, so the picker, `select_extract`'s refusal,
    `compare.py` and the random test all keep the meanings they had.

    That is why it is a SELECTION and not a resolution. A default that were
    re-derived on every request would move under a session as the log grew, and
    "same as reading model" in the picker would have had to start meaning two
    different things depending on what had been measured lately.

    Pass 1 needs nothing done: `_resolve_model` already prefers the ranking when
    nobody has chosen, which is where a default has always lived. Pass 2 is the
    one that had no such default -- it fell back to the model reading the page,
    which every sweep in this project says is the wrong tool for the form.

    Returns `{"reading": name, "extract": name or ""}`; `extract` is "" where the
    best extractor IS the reader, which leaves the one-model setup untouched.
    """
    info = status(url)
    picked = {"reading": info["model"], "extract": ""}
    if not settings.AUTO_BEST_MODEL:
        return picked
    if url is None and extract_separate():
        # Pass 2 has its own server: rank ITS models, and write the winner down
        # against that URL so it is a choice from here on, like the one-URL case.
        ex_url = extract_url()
        ex = probe(ex_url)
        best = best_served(
            "extract", ex["models"],
            lambda m: (not is_ocr_model(m["name"])
                       or _same_model(m["name"], info["model"])))
        if best and not _extract_chosen.get(ex_url):
            with _lock:
                _extract_chosen[ex_url] = best
            picked["extract"] = best
        return picked
    if not info["model"]:
        return picked
    best = best_extractor(info)
    if best:
        # Not through `select_extract`: that unloads, and nothing has been
        # loaded yet at startup. The refusal it enforces cannot fire here
        # anyway -- `best_extractor` will not return a second OCR model.
        with _lock:
            _extract_chosen[clean_url(url) if url else _active] = best
        picked["extract"] = best
    return picked


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
    url = clean_url(url) if url else extract_url()
    chosen = (model or "").strip()
    if chosen:
        reading = status()["model"]
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
    info = extract_status()
    stopped = []
    if unload and settings.OLLAMA_UNLOAD_ON_SWITCH:
        stopped = free_gpu(url, keep=_in_use(url))
        for name in stopped:
            config.say(f"[ollama] stopped {name} to free the GPU")
    return {**info, "unloaded": stopped}


def overview(force: bool = False) -> dict:
    """Every configured endpoint plus the active one, for the picker."""
    active = active_url()
    ex_url = extract_url()
    listed = []
    for url in endpoints():
        info = probe(url, force=force)
        listed.append({
            "url": url,
            "kind": info["kind"],
            "kind_setting": info.get("kind_pinned") or "auto",
            "reachable": info["reachable"],
            "models": [m["name"] for m in info["models"]],
            "active": url == active,
            "extract": url == ex_url,
        })
    server = status(force=force)
    extract = extract_status(force=force)
    separate = extract_separate()
    return {
        "endpoints": listed,
        "active": active,
        "server": server,
        # The engines the "Server type" picker offers, besides "auto".
        "kinds": [{"kind": k, "label": SERVER_KIND_LABELS[k]} for k in SERVER_KINDS],
        # What pass 2 will run on, and which of this endpoint's models may be
        # chosen for it. The page needs the second list because the refusal in
        # `select_extract` should be visible before it is triggered, not after.
        "extract": {
            # Where pass 2 goes. `configured` is the CHOICE ("" = same as the
            # reading server); `url` is what that resolves to now.
            "url": extract["url"],
            "configured": configured_extract_url(),
            "separate": separate,
            "kind": extract["kind"],
            "kind_setting": extract.get("kind_setting") or "auto",
            "model": extract["model"],
            "chosen": extract_model() or "",
            "available": extract["text_available"],
            "reason": extract["text_reason"],
            # "Same as the reading model" only exists where both passes share a
            # server -- across two servers there is no reading model to share.
            "same_as_reading": not separate and not extract_model(),
            # Every model the extraction server serves, OCR fine-tunes
            # included: a fields-only random round runs ONE model and any of
            # them may be it, so the random test's lists need the whole set.
            "models": [m["name"] for m in extract["models"]],
            "choices": [m["name"] for m in extract["models"]
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

    `reasoning_effort` goes to Ollama, and `chat_template_kwargs.enable_thinking`
    to llama-server, and it is what keeps a reasoning model
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
    if info["kind"] in ("vllm", "openai"):
        # The model name is required, like Ollama. No window is sent: vLLM fixes
        # it at launch (--max-model-len) and has no per-request equivalent.
        extras = {"model": info["model"]} if info.get("model") else {}
        if settings.OLLAMA_REASONING_EFFORT == "none":
            extras["chat_template_kwargs"] = {"enable_thinking": False}
        # vLLM runs the penalties it is sent; Ollama runs the Modelfile's. This
        # makes the two the same sampler for the same model. Merged last by every
        # caller, so it replaces sampler_extras' repetition_penalty. vLLM only --
        # a generic OpenAI server is not assumed to take these fields.
        if info["kind"] == "vllm":
            extras.update(settings.vllm_sampler(info.get("model")))
        return extras
    extras = {"n_ctx": chosen} if chosen > 0 else {}
    # llama-server serves thinking models too (qwen3.6-35b-a3b, 2026-09-23) and
    # fails the same way: every capped request comes back with its budget spent
    # in `reasoning_content` and `content` empty. Its switch is the chat
    # template's `enable_thinking`, passed through `chat_template_kwargs`; a
    # template that never reads it ignores it, so a non-thinking model is
    # unaffected. Same setting as Ollama's, so one knob turns thinking on for both.
    if settings.OLLAMA_REASONING_EFFORT == "none":
        extras["chat_template_kwargs"] = {"enable_thinking": False}
    return extras


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
        return f"{info.get('url') or active_url()}/api/chat", body

    # The URL off the status dict, not `active_url()`: pass 2 may have an
    # endpoint of its own, and the caller hands over that endpoint's status.
    return f"{info.get('url') or active_url()}/v1/chat/completions", {
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
        **settings.sampler_extras(num_ctx()),
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
