"""What the MODEL SERVER says a request cost, as raw figures, one per phase.

Added 2026-10-05 at the user's request (the app on a server is network hops
from the model server, so its own clocks carry the hops); narrowed 2026-10-07,
also at the user's request: *just report the time for prefill and decode and
server latency separately -- I will add them myself; I want to focus on the raw
GPU speed / compute power*, and then *add prefill and decode tokens per second
too, and be aware of the image sending to the server*.

Nothing here is derived from the app's clock. Each figure is the server's own:

  `server_prefill_seconds`   prompt processing -- the vision encoder and the
                             prompt -- the compute that scales with the image
  `server_decode_seconds`    token generation
  `server_queue_seconds`     time the request waited in the server's queue
  `server_seconds`           the server's end-to-end latency for the request
  `server_prompt_tokens`     every prompt token, IMAGE TOKENS INCLUDED, cached or not
  `server_cached_tokens`     of those, the ones served from the prefix cache
  `server_prefill_tokens`    the ones actually computed in prefill
  `server_generated_tokens`  tokens generated
  `server_decode_tokens`     the ones generated in the decode phase
  `server_prefill_tps`       prefill tokens / prefill seconds
  `server_decode_tps`        decode tokens / decode seconds

**The prefill rate is over the tokens COMPUTED, not the prompt.** A prefix-cache
hit is a token the GPU did not process, so counting it would inflate the rate --
on vLLM with prefix caching on (the v1 default) by a lot on a shared prompt. Where
the cached count cannot be known, the prefill tokens and the rate are left blank
rather than guessed.

**The image is in the prompt tokens and in prefill, and its TRANSFER is in
neither.** A page goes to the server as a base64 PNG in the JSON body (the read
reports its size as `image_bytes`). Uploading it, and the server decoding and
resizing it, happen before the request reaches the engine's queue, so on vLLM
they are inside `server_seconds` (arrival is stamped as the request enters the
input processor) and outside queue, prefill and decode. The vision encoder itself
runs in prefill, and the image's placeholder tokens are counted in
`server_prompt_tokens` -- so the prefill rate on a page read includes the image.

**Sources, and they do not measure the same span** -- `source` says which:

  `vllm /metrics`     the change in each vLLM histogram between a snapshot just
                      before the request and one just after; tokens from the
                      reply's `usage` (the request's own count), falling back to
                      the `request_prompt_tokens` / `request_generation_tokens`
                      histograms. Cached tokens from `usage.prompt_tokens_details`
                      where the server sends it (`--enable-prompt-tokens-details`),
                      else from the change in `vllm:prefix_cache_hits` -- taken
                      only where nothing was running or waiting at either
                      snapshot, since that counter moves for every request.
                      Decode tokens are generated - 1: vLLM's prefill phase
                      ends with the first token sampled.
  `llama.cpp timings` `prompt_ms` / `predicted_ms`, `prompt_n` (computed),
                      `cache_n` (cached), `predicted_n`. No queue figure, and
                      its latency is prefill + decode.
  `ollama`            `prompt_eval_duration` / `eval_duration` /
                      `total_duration`, `prompt_eval_count` / `eval_count` on a
                      native `/api/chat` reply. `/v1` sends no timing at all.

**A /metrics change is attributed only when it covers exactly one request** (or
exactly a batch's requests -- see `Batch`), and is left blank with the reason
otherwise -- the standing rule that blank is not zero.

**Never asked of a server that is not vLLM.** `backends.serves_metrics` is the
gate, the same one the stress test's button uses.

Nothing here raises: a timing figure that breaks the read it is timing is
worse than none.
"""

import threading
import time

import backends
import settings
import stress

E2E = ("vllm:e2e_request_latency_seconds",)
# Phase -> the vLLM histogram whose _sum it is, v1 names first.
PARTS = {
    "queue": ("vllm:request_queue_time_seconds",),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
}
TOKEN_HISTS = {
    "prompt": ("vllm:request_prompt_tokens",),
    "generated": ("vllm:request_generation_tokens",),
}
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
BUSY = ("vllm:num_requests_running", "vllm:num_requests_waiting")

PHASES = ("queue", "prefill", "decode")
TIME_FIELDS = ("server_seconds",) + tuple(f"server_{p}_seconds" for p in PHASES)
TOKEN_FIELDS = ("server_prompt_tokens", "server_cached_tokens",
                "server_prefill_tokens", "server_generated_tokens",
                "server_decode_tokens")
RATE_FIELDS = ("server_prefill_tps", "server_decode_tps")
# What each figure is called on a record, a summary and the run log.
FIELDS = TIME_FIELDS + TOKEN_FIELDS + RATE_FIELDS
# Summed over pages or requests; the rates are recomputed from the sums,
# never averaged.
SUMMED = TIME_FIELDS + TOKEN_FIELDS

METRICS_TIMEOUT = (1.5, 3)
# Waits between after-snapshots while vLLM has not yet recorded the request.
_RETRY_WAITS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


# Per-request /metrics snapshots are switched off while a stress test runs
# (2026-10-14, reported by the user: the app kept firing /metrics at vLLM far
# more often than the stress test's "sample every N s"). Each request's Clock
# took one snapshot before and up to six after, so a run at concurrency 8 sent
# dozens a second -- and under concurrency none of them could be attributed to
# one request anyway. The stress test's own sampler is then the only thing
# asking. A count, not a flag, so two overlapping holders cannot release each
# other; process-wide, because any request sent during a stress run overlaps it.
_suspend_lock = threading.Lock()
_suspended = 0


def suspend_metrics():
    """Stop Clock/Batch from reading /metrics until `resume_metrics`."""
    global _suspended
    with _suspend_lock:
        _suspended += 1


def resume_metrics():
    global _suspended
    with _suspend_lock:
        _suspended = max(0, _suspended - 1)


def metrics_suspended() -> bool:
    return _suspended > 0


def _fetch(url: str):
    """Parsed `/metrics`, or None. Never raises."""
    try:
        res = backends.http_get(f"{url}/metrics", {"kind": "vllm"},
                                timeout=METRICS_TIMEOUT)
        if res.status_code != 200:
            return None
        return stress.parse_prometheus(res.text)
    except Exception:
        return None


def _hist(values: dict, names, suffix: str):
    for name in names:
        if name + suffix in values:
            return values[name + suffix]
    return None


def _idle(snapshot: dict) -> bool:
    """Nothing running or waiting -- so a server-wide counter moved for us alone."""
    if not snapshot:
        return False
    found = [snapshot[name] for name in BUSY if name in snapshot]
    return bool(found) and all(v == 0 for v in found)


def _delta(base: dict, now: dict, names, n: int):
    """A histogram's _sum change, where its _count moved by exactly n."""
    s0, c0 = _hist(base, names, "_sum"), _hist(base, names, "_count")
    s1, c1 = _hist(now, names, "_sum"), _hist(now, names, "_count")
    if None in (s0, c0, s1, c1) or round(c1 - c0) != n:
        return None
    return s1 - s0


def _from_snapshots(base: dict, now: dict, n: int) -> dict:
    """Latency, phases and tokens out of two /metrics snapshots around n requests."""
    out = {"seconds": (_hist(now, E2E, "_sum") or 0.0) - _hist(base, E2E, "_sum"),
           "source": "vllm /metrics"}
    for part, names in PARTS.items():
        value = _delta(base, now, names, n)
        if value is not None:
            out[part] = value
    for key, names in TOKEN_HISTS.items():
        value = _delta(base, now, names, n)
        if value is not None:
            out[f"{key}_tokens"] = int(round(value))
    hits0, hits1 = _hist(base, PREFIX_HITS, ""), _hist(now, PREFIX_HITS, "")
    if None not in (hits0, hits1) and _idle(base) and _idle(now):
        out["cached_tokens"] = int(round(hits1 - hits0))
    return out


def _usage_tokens(usage) -> dict:
    """Token counts off an OpenAI-shaped `usage` block -- the request's own."""
    if not isinstance(usage, dict):
        return {}
    out = {}
    if isinstance(usage.get("prompt_tokens"), (int, float)):
        out["prompt_tokens"] = int(usage["prompt_tokens"])
    if isinstance(usage.get("completion_tokens"), (int, float)):
        out["generated_tokens"] = int(usage["completion_tokens"])
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), (int, float)):
        out["cached_tokens"] = int(details["cached_tokens"])
    return out


def from_body(body) -> dict:
    """Server time and tokens from a reply body that carries its own, or None.

    llama.cpp puts `timings` on a non-streaming reply and on the last chunk of a
    stream; Ollama's native endpoint puts nanosecond durations on the reply.
    """
    if not isinstance(body, dict):
        return None
    timings = body.get("timings")
    if isinstance(timings, dict):
        prompt = timings.get("prompt_ms") or 0
        predicted = timings.get("predicted_ms") or 0
        if prompt or predicted:
            out = {"seconds": (prompt + predicted) / 1000,
                   "prefill": prompt / 1000, "decode": predicted / 1000,
                   "source": "llama.cpp timings"}
            computed, cached = timings.get("prompt_n"), timings.get("cache_n")
            if isinstance(computed, (int, float)):
                out["prefill_tokens"] = int(computed)
                if isinstance(cached, (int, float)):
                    out["cached_tokens"] = int(cached)
                    out["prompt_tokens"] = int(computed) + int(cached)
            if isinstance(timings.get("predicted_n"), (int, float)):
                out["generated_tokens"] = out["decode_tokens"] = int(timings["predicted_n"])
            return out
    total = body.get("total_duration")
    if isinstance(total, (int, float)) and total > 0:
        out = {"seconds": total / 1e9, "source": "ollama"}
        if body.get("prompt_eval_duration"):
            out["prefill"] = body["prompt_eval_duration"] / 1e9
        if body.get("eval_duration"):
            out["decode"] = body["eval_duration"] / 1e9
        if isinstance(body.get("prompt_eval_count"), (int, float)):
            out["prompt_tokens"] = out["prefill_tokens"] = int(body["prompt_eval_count"])
        if isinstance(body.get("eval_count"), (int, float)):
            out["generated_tokens"] = out["decode_tokens"] = int(body["eval_count"])
        return out
    return None


def _rates(out: dict):
    """The two rates, from tokens and seconds already on the record."""
    prefill, decode = out.get("server_prefill_seconds"), out.get("server_decode_seconds")
    computed, decoded = out.get("server_prefill_tokens"), out.get("server_decode_tokens")
    out["server_prefill_tps"] = (round(computed / prefill, 1)
                                 if computed is not None and prefill else None)
    out["server_decode_tps"] = (round(decoded / decode, 1)
                                if decoded is not None and decode else None)


def _record(server: dict, usage: dict = None, requests: int = 1) -> dict:
    """The server's figures, rounded, under the names everything else uses.

    `usage` is the reply's own token count (summed, for a batch of `requests`).
    A server that counts its own tokens alongside its own timings (llama.cpp,
    Ollama native) is taken at its word; otherwise usage wins over a /metrics
    histogram, being the request's own rather than a server-wide difference.
    """
    out = {name: None for name in FIELDS}
    out["source"] = server.get("source", "")
    if server.get("seconds") is None:
        if server.get("why"):
            out["why"] = server["why"]
        return out
    out["server_seconds"] = round(server["seconds"], 3)
    for part in PHASES:
        if server.get(part) is not None:
            out[f"server_{part}_seconds"] = round(server[part], 3)
    tokens = {key: server.get(key) for key in
              ("prompt_tokens", "cached_tokens", "prefill_tokens",
               "generated_tokens", "decode_tokens")}
    if tokens["prefill_tokens"] is None:
        for key, value in (usage or {}).items():
            if value is not None:
                tokens[key] = value
    prompt, cached = tokens["prompt_tokens"], tokens["cached_tokens"]
    computed = tokens["prefill_tokens"]
    if computed is None and prompt is not None and cached is not None:
        computed = max(prompt - cached, 0)
    generated, decoded = tokens["generated_tokens"], tokens["decode_tokens"]
    if decoded is None and generated is not None and out["source"] == "vllm /metrics":
        decoded = max(generated - requests, 0)   # vLLM's prefill ends with token one
    out.update(server_prompt_tokens=prompt, server_cached_tokens=cached,
               server_prefill_tokens=computed, server_generated_tokens=generated,
               server_decode_tokens=decoded)
    _rates(out)
    return out


def io_tokens(record: dict, usage: dict = None) -> tuple:
    """`(in, out)` tokens of one request: what was sent, what came back.

    Added 2026-10-15 for the stress test's per-page and per-file token figures.
    Unlike the `server_*` fields these do NOT need a server time: a vLLM request
    with per-request /metrics off, or an Ollama /v1 request, still counts its
    tokens in `usage`. The server's own count wins where it gave one (llama.cpp
    `timings`, Ollama native), the reply's `usage` otherwise. `in` is the whole
    prompt, cached part included -- a llama.cpp reply without `cache_n` gives
    only the computed part, and that is what is returned for it.
    """
    usage = usage or {}
    record = record or {}
    sent = record.get("server_prompt_tokens")
    if sent is None:
        sent = usage.get("prompt_tokens")
    if sent is None:
        sent = record.get("server_prefill_tokens")
    back = record.get("server_generated_tokens")
    if back is None:
        back = usage.get("generated_tokens")
    return sent, back


class Clock:
    """One request, timed by the server.

    ``clock = Clock(status)`` just before the request (it takes the before-
    snapshot on vLLM), ``clock.body(...)`` / ``clock.timings(...)`` /
    ``clock.usage(...)`` with whatever the reply carried, ``clock.finish()``
    once it is over. `finish` returns the record and adds it to this thread's
    tally if one is open.
    """

    def __init__(self, info: dict):
        self.info = info or {}
        # Inside a `Batch` the requests run concurrently, so a /metrics change
        # around any one of them covers its neighbours too. The batch takes one
        # snapshot each side of the whole lot instead (see `Batch`).
        self.batch = getattr(_local, "batch", None)
        self.metrics = bool(settings.SERVER_TIMING and settings.SERVER_TIMING_METRICS
                            and backends.serves_metrics(self.info)
                            and self.info.get("url")
                            and self.batch is None
                            and not metrics_suspended())
        self.base = _fetch(self.info["url"]) if self.metrics else None
        self.server = None
        self.tokens = {}
        self.done = None

    def body(self, body):
        if isinstance(body, dict):
            self.usage(body.get("usage"))
        if not self.metrics and self.server is None:
            self.server = from_body(body)

    def timings(self, timings):
        if timings:
            self.body({"timings": timings})

    def usage(self, usage):
        self.tokens.update(_usage_tokens(usage))

    def _from_metrics(self) -> dict:
        if self.base is None:
            return {"why": "the before-snapshot of /metrics failed"}
        if _hist(self.base, E2E, "_sum") is None or _hist(self.base, E2E, "_count") is None:
            return {"why": "/metrics has no e2e request latency histogram"}
        base_c = _hist(self.base, E2E, "_count")
        deadline = time.perf_counter() + settings.SERVER_TIMING_WAIT
        for wait in _RETRY_WAITS:
            if wait:
                if time.perf_counter() + wait > deadline:
                    break
                time.sleep(wait)
            now = _fetch(self.info["url"])
            if now is None:
                return {"why": "the after-snapshot of /metrics failed"}
            count = (_hist(now, E2E, "_count") or 0.0) - base_c
            if count < 0.5:
                continue        # not recorded yet -- the stats logger is behind
            if count > 1.5:
                return {"why": f"{int(round(count))} requests finished between "
                               "the snapshots (concurrent traffic), so the change "
                               "cannot be attributed to this one"}
            return _from_snapshots(self.base, now, 1)
        return {"why": "vLLM had not recorded the request within "
                       f"{settings.SERVER_TIMING_WAIT:g}s of it finishing"}

    def finish(self, client_seconds: float = None) -> dict:
        """The record: the server's own figures, each as it came.

        `client_seconds` is accepted for the callers that already time the
        request and is ignored: no figure here is derived from the app's clock.
        """
        if self.done is not None:
            return self.done
        server = self._from_metrics() if self.metrics else (self.server or {})
        out = _record(server, self.tokens)
        out["in_tokens"], out["out_tokens"] = io_tokens(out, self.tokens)
        self.done = out
        if self.batch is not None:
            self.batch.record(out, self.tokens)
        else:
            _add(out)
        return out


class Batch:
    """Requests sent CONCURRENTLY, timed by the server as one span.

    Added with the agentic steps that run at once on vLLM. A per-request
    /metrics difference is unattributable there by construction -- every
    snapshot pair spans the others -- so the batch snapshots once before the
    first request and once after the last, and where the e2e count moved by
    exactly the number of requests it sent, attributes the SUMS to them --
    the latency, and each phase whose count moved by the same number. Token
    counts come from each request's own `usage`, summed. Any other request
    finishing in between makes the counts disagree, and the batch is then left
    unattributed with the reason -- the per-request rule, one level up.

    ``batch = Batch(status)`` in the thread that owns the tally, ``batch.run(fn,
    ...)`` in each worker, ``batch.close()`` back in the owning thread once every
    worker is done. A server that times its own replies (llama.cpp, Ollama's
    native endpoint) needs none of this, and its records are added one by one.
    """

    def __init__(self, info: dict):
        self.info = info or {}
        self.metrics = bool(settings.SERVER_TIMING and settings.SERVER_TIMING_METRICS
                            and backends.serves_metrics(self.info)
                            and self.info.get("url")
                            and not metrics_suspended())
        self.base = _fetch(self.info["url"]) if self.metrics else None
        self.records = []
        self.usages = []
        self.lock = threading.Lock()

    def run(self, fn, *args, **kwargs):
        """Call `fn` in this (worker) thread with its requests counted here."""
        previous = getattr(_local, "batch", None)
        _local.batch = self
        try:
            return fn(*args, **kwargs)
        finally:
            _local.batch = previous

    def record(self, out: dict, usage: dict = None):
        with self.lock:
            self.records.append(out)
            self.usages.append(dict(usage or {}))

    def close(self) -> dict:
        """Add the batch to this thread's tally. Returns the batch-level record."""
        records = list(self.records)
        if not records:
            return None
        if not self.metrics:
            for out in records:
                _add(out)
            return None
        n = len(records)
        usage = {}
        for key in ("prompt_tokens", "cached_tokens", "generated_tokens"):
            values = [u.get(key) for u in self.usages]
            if all(v is not None for v in values):
                usage[key] = sum(values)
        out = _record(self._from_metrics(n), usage, requests=n)
        for key in ("in_tokens", "out_tokens"):
            values = [r.get(key) for r in records]
            out[key] = sum(values) if all(v is not None for v in values) else None
        _add(out, n)
        return {"requests": n, **out}

    def _from_metrics(self, n: int) -> dict:
        if self.base is None:
            return {"why": "the before-snapshot of /metrics failed"}
        if _hist(self.base, E2E, "_sum") is None or _hist(self.base, E2E, "_count") is None:
            return {"why": "/metrics has no e2e request latency histogram"}
        base_c = _hist(self.base, E2E, "_count")
        deadline = time.perf_counter() + settings.SERVER_TIMING_WAIT
        count = 0.0
        for wait in _RETRY_WAITS:
            if wait:
                if time.perf_counter() + wait > deadline:
                    break
                time.sleep(wait)
            now = _fetch(self.info["url"])
            if now is None:
                return {"why": "the after-snapshot of /metrics failed"}
            count = (_hist(now, E2E, "_count") or 0.0) - base_c
            if count < n - 0.5:
                continue        # the stats logger has not caught up yet
            if count > n + 0.5:
                return {"why": f"{int(round(count))} requests finished during a "
                               f"batch of {n} (other traffic), so the change "
                               "cannot be attributed to it"}
            return _from_snapshots(self.base, now, n)
        return {"why": f"vLLM had recorded {int(round(count))} of the batch's {n} "
                       f"requests within {settings.SERVER_TIMING_WAIT:g}s"}


# --------------------------------------------------------------------------
# totals: a read's pages, and the tally of every request one extraction makes
# --------------------------------------------------------------------------

def _finish_totals(out: dict) -> dict:
    """Rates from the summed tokens and seconds; token counts back to ints."""
    _rates(out)
    for name in TOKEN_FIELDS:
        if out.get(name) is not None:
            out[name] = int(round(out[name]))
    return out


def summarise_records(records) -> dict:
    """Every field over several records (a read's pages).

    A figure is summed only where EVERY record reported it -- a total that
    mixed pages with a figure and pages without would read as smaller than it
    is -- and the rates are recomputed from the sums, never averaged.
    """
    out = {}
    for name in SUMMED:
        values = [r.get(name) for r in records]
        out[name] = (round(sum(values), 3)
                     if values and all(v is not None for v in values) else None)
    _finish_totals(out)
    return {name: out.get(name) for name in FIELDS}


_local = threading.local()


def current_batch():
    """Reuse the outer document batch when its Hybrid steps fan out as well."""
    return getattr(_local, "batch", None)


def open_tally():
    """Start counting this thread's requests. Returns the previous tally."""
    previous = getattr(_local, "tally", None)
    _local.tally = {"requests": 0, "attributed": 0, "sources": [], "why": "",
                    "io": {"in_tokens": 0, "out_tokens": 0},
                    "io_have": {"in_tokens": 0, "out_tokens": 0},
                    "sums": {name: 0.0 for name in SUMMED},
                    "have": {name: 0 for name in SUMMED}}
    return previous


def close_tally(previous=None) -> dict:
    """This thread's tally as a result dict, or None where nothing was timed."""
    tally = getattr(_local, "tally", None)
    _local.tally = previous
    return summary(tally)


def _add(record: dict, requests: int = 1):
    tally = getattr(_local, "tally", None)
    if tally is None:
        return
    tally["requests"] += requests
    # In/out tokens are counted whether or not the server gave a time.
    for key in ("in_tokens", "out_tokens"):
        if record.get(key) is not None:
            tally["io"][key] += record[key]
            tally["io_have"][key] += requests
    if record.get("server_seconds") is None:
        if record.get("why") and not tally["why"]:
            tally["why"] = record["why"]
        return
    tally["attributed"] += requests
    for name in SUMMED:
        if record.get(name) is not None:
            tally["sums"][name] += record[name]
            tally["have"][name] += requests
    if record["source"] and record["source"] not in tally["sources"]:
        tally["sources"].append(record["source"])


def summary(tally: dict) -> dict:
    """Server totals over the requests the server timed. None where none were made.

    A figure is summed only where EVERY attributed request reported it (a
    llama.cpp reply has no queue figure, so one in the tally blanks the queue
    total); the rates are recomputed from the sums. A request the server gave
    no figure for is in `requests` and in no sum.
    """
    if not tally or not tally.get("requests"):
        return None
    out = {"requests": tally["requests"], "attributed": tally["attributed"],
           "source": "+".join(tally["sources"])}
    for name in SUMMED:
        whole = tally["attributed"] and tally["have"][name] == tally["attributed"]
        out[name] = round(tally["sums"][name], 3) if whole else None
    _finish_totals(out)
    # Summed only where every request reported it, like every figure here.
    for key in ("in_tokens", "out_tokens"):
        have = tally.get("io_have", {}).get(key, 0)
        out[key] = int(tally["io"][key]) if have and have == tally["requests"] else None
    if tally.get("why"):
        out["why"] = tally["why"]
    return out
