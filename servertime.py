"""How long the MODEL SERVER says a request took, beside how long the app waited.

Added 2026-10-05 at the user's request: *on the server the app takes time to
hop to the model server, so the time tracked in the app is not accurate -- use
an endpoint like vLLM's /metrics to get the time and subtract it from the time
tracked*. Every clock in this app is client-side (`time.perf_counter()` around
the request), so on a deployment where the app and the model server are several
network hops apart every figure carries the hops in it. The server's own figure
does not, and the difference between the two is the hop time.

**Three sources, and they do not measure the same span** -- `source` says which:

  `vllm /metrics`     the change in `vllm:e2e_request_latency_seconds` between a
                      snapshot taken just before the request and one just after:
                      arrival at the server to the last token, QUEUE INCLUDED.
                      With it come the queue, prefill and decode parts.
  `llama.cpp timings` `prompt_ms + predicted_ms`, which llama-server already
                      sends. Compute only: time in its queue is not in it, so
                      under load the "hop" figure includes that queue.
  `ollama`            `total_duration` on a native `/api/chat` reply. Ollama's
                      `/v1` endpoint sends no timing at all, so most Ollama
                      requests here have no server figure.

**A /metrics difference is attributed only when it covers exactly one request.**
vLLM's histograms are server-wide; if another request finished between the two
snapshots the change belongs to both and splitting it would be a guess. It is
left blank and counted as `unattributed` -- the standing rule that blank is not
zero. A stress run above one concurrent will mostly be unattributed, which is
correct: there, the stress report's own `/metrics` view is the server figure.

**Never asked of a server that is not vLLM.** `backends.serves_metrics` is the
gate, the same one the stress test's button uses -- llama-server's `/metrics`
shares the task queue with inference, which is why this project never polls it.

No module here imports Flask or makes a request except `_fetch`, and nothing in
it raises: a timing figure that breaks the read it is timing is worse than none.
"""

import threading
import time

import requests

import backends
import settings
import stress

E2E = ("vllm:e2e_request_latency_seconds",)
# Part name -> the histogram it is the mean of, v1 names first.
PARTS = {
    "queue": ("vllm:request_queue_time_seconds",),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
    "ttft": ("vllm:time_to_first_token_seconds",),
}
METRICS_TIMEOUT = (1.5, 3)
# Waits between after-snapshots while vLLM has not yet recorded the request.
_RETRY_WAITS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


def _fetch(url: str):
    """Parsed `/metrics`, or None. Never raises."""
    try:
        res = requests.get(f"{url}/metrics", timeout=METRICS_TIMEOUT)
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


def from_body(body) -> dict:
    """Server time from a reply body that carries its own, or None.

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
            return {"seconds": (prompt + predicted) / 1000,
                    "prefill": prompt / 1000, "decode": predicted / 1000,
                    "source": "llama.cpp timings"}
    total = body.get("total_duration")
    if isinstance(total, (int, float)) and total > 0:
        out = {"seconds": total / 1e9, "source": "ollama"}
        if body.get("prompt_eval_duration"):
            out["prefill"] = body["prompt_eval_duration"] / 1e9
        if body.get("eval_duration"):
            out["decode"] = body["eval_duration"] / 1e9
        return out
    return None


class Clock:
    """One request, timed by the app and by the server.

    ``clock = Clock(status)`` just before the request (it takes the before-
    snapshot on vLLM), ``clock.body(...)`` / ``clock.timings(...)`` with whatever
    the reply carried, ``clock.finish()`` once it is over. `finish` returns the
    record and adds it to this thread's tally if one is open.
    """

    def __init__(self, info: dict):
        self.info = info or {}
        # Inside a `Batch` the requests run concurrently, so a /metrics change
        # around any one of them covers its neighbours too. The batch takes one
        # snapshot each side of the whole lot instead (see `Batch`).
        self.batch = getattr(_local, "batch", None)
        self.metrics = bool(settings.SERVER_TIMING
                            and backends.serves_metrics(self.info)
                            and self.info.get("url")
                            and self.batch is None)
        self.base = _fetch(self.info["url"]) if self.metrics else None
        self.server = None
        self.done = None
        # The client clock starts AFTER the before-snapshot, so the snapshot's own
        # round trip is not charged to the hops of the request it is measuring.
        self.started = time.perf_counter()

    def body(self, body):
        if not self.metrics and self.server is None:
            self.server = from_body(body)

    def timings(self, timings):
        if timings:
            self.body({"timings": timings})

    def _from_metrics(self) -> dict:
        if self.base is None:
            return {"why": "the before-snapshot of /metrics failed"}
        base_s, base_c = _hist(self.base, E2E, "_sum"), _hist(self.base, E2E, "_count")
        if base_s is None or base_c is None:
            return {"why": "/metrics has no e2e request latency histogram"}
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
            out = {"seconds": (_hist(now, E2E, "_sum") or 0.0) - base_s,
                   "source": "vllm /metrics"}
            for part, names in PARTS.items():
                s0, c0 = _hist(self.base, names, "_sum"), _hist(self.base, names, "_count")
                s1, c1 = _hist(now, names, "_sum"), _hist(now, names, "_count")
                if None not in (s0, c0, s1, c1) and round(c1 - c0) == 1:
                    out[part] = s1 - s0
            return out
        return {"why": "vLLM had not recorded the request within "
                       f"{settings.SERVER_TIMING_WAIT:g}s of it finishing"}

    def finish(self, client_seconds: float = None) -> dict:
        """The record: client and server seconds and the hop time between them."""
        if self.done is not None:
            return self.done
        client = (client_seconds if client_seconds is not None
                  else time.perf_counter() - self.started)
        server = self._from_metrics() if self.metrics else (self.server or {})
        out = {"client_seconds": round(client, 3), "server_seconds": None,
               "network_seconds": None, "source": server.get("source", "")}
        if server.get("seconds") is not None:
            out["server_seconds"] = round(server["seconds"], 3)
            # Not clipped at zero. A negative hop means the two clocks disagree
            # about the span -- a server figure that covers more than the request
            # -- and hiding that would hide the measurement problem.
            out["network_seconds"] = round(client - server["seconds"], 3)
            for part in ("queue", "prefill", "decode", "ttft"):
                if server.get(part) is not None:
                    out[f"server_{part}"] = round(server[part], 3)
        elif server.get("why"):
            out["why"] = server["why"]
        self.done = out
        if self.batch is not None:
            self.batch.record(out)
        else:
            _add(out)
        return out


class Batch:
    """Requests sent CONCURRENTLY, timed by the server as one span.

    Added with the agentic steps that run at once on vLLM. A per-request
    /metrics difference is unattributable there by construction -- every
    snapshot pair spans the others -- so the batch snapshots once before the
    first request and once after the last, and where the e2e count moved by
    exactly the number of requests it sent, attributes the e2e SUM to them.
    That sum is the server's figure for those requests, compared with the SUM
    of their client clocks: like with like, and the hop time is the difference.
    Any other request finishing in between makes the counts disagree, and the
    batch is then left unattributed with the reason -- the per-request rule,
    one level up.

    ``batch = Batch(status)`` in the thread that owns the tally, ``batch.run(fn,
    ...)`` in each worker, ``batch.close()`` back in the owning thread once every
    worker is done. A server that times its own replies (llama.cpp, Ollama's
    native endpoint) needs none of this, and its records are added one by one.
    """

    def __init__(self, info: dict):
        self.info = info or {}
        self.metrics = bool(settings.SERVER_TIMING
                            and backends.serves_metrics(self.info)
                            and self.info.get("url"))
        self.base = _fetch(self.info["url"]) if self.metrics else None
        self.records = []
        self.lock = threading.Lock()

    def run(self, fn, *args, **kwargs):
        """Call `fn` in this (worker) thread with its requests counted here."""
        previous = getattr(_local, "batch", None)
        _local.batch = self
        try:
            return fn(*args, **kwargs)
        finally:
            _local.batch = previous

    def record(self, out: dict):
        with self.lock:
            self.records.append(out)

    def close(self) -> dict:
        """Add the batch to this thread's tally. Returns the batch-level record."""
        records = list(self.records)
        if not records:
            return None
        if not self.metrics:
            for out in records:
                _add(out)
            return None
        out = self._from_metrics(len(records))
        client = sum(r["client_seconds"] for r in records)
        tally = getattr(_local, "tally", None)
        if tally is not None:
            tally["requests"] += len(records)
            if out.get("seconds") is None:
                if out.get("why") and not tally["why"]:
                    tally["why"] = out["why"]
            else:
                tally["attributed"] += len(records)
                tally["client_seconds"] += client
                tally["server_seconds"] += out["seconds"]
                if out["source"] not in tally["sources"]:
                    tally["sources"].append(out["source"])
        return {"requests": len(records), "client_seconds": round(client, 3),
                **out}

    def _from_metrics(self, n: int) -> dict:
        if self.base is None:
            return {"why": "the before-snapshot of /metrics failed"}
        base_s, base_c = _hist(self.base, E2E, "_sum"), _hist(self.base, E2E, "_count")
        if base_s is None or base_c is None:
            return {"why": "/metrics has no e2e request latency histogram"}
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
            return {"seconds": (_hist(now, E2E, "_sum") or 0.0) - base_s,
                    "source": "vllm /metrics"}
        return {"why": f"vLLM had recorded {int(round(count))} of the batch's {n} "
                       f"requests within {settings.SERVER_TIMING_WAIT:g}s"}


# --------------------------------------------------------------------------
# the tally: every request one extraction makes, summed
# --------------------------------------------------------------------------

_local = threading.local()


def open_tally():
    """Start counting this thread's requests. Returns the previous tally."""
    previous = getattr(_local, "tally", None)
    _local.tally = {"requests": 0, "attributed": 0, "client_seconds": 0.0,
                    "server_seconds": 0.0, "sources": [], "why": ""}
    return previous


def close_tally(previous=None) -> dict:
    """This thread's tally as a result dict, or None where nothing was timed."""
    tally = getattr(_local, "tally", None)
    _local.tally = previous
    return summary(tally)


def _add(record: dict):
    tally = getattr(_local, "tally", None)
    if tally is None:
        return
    tally["requests"] += 1
    if record.get("server_seconds") is None:
        if record.get("why") and not tally["why"]:
            tally["why"] = record["why"]
        return
    tally["attributed"] += 1
    tally["client_seconds"] += record["client_seconds"]
    tally["server_seconds"] += record["server_seconds"]
    if record["source"] and record["source"] not in tally["sources"]:
        tally["sources"].append(record["source"])


def summary(tally: dict) -> dict:
    """Totals over the requests the server timed. None where none were made.

    Client and server are both summed over the ATTRIBUTED requests only, so the
    hop figure compares like with like: a request the server gave no figure for
    is in `requests` and in neither sum.
    """
    if not tally or not tally.get("requests"):
        return None
    out = {"requests": tally["requests"], "attributed": tally["attributed"],
           "source": "+".join(tally["sources"])}
    if tally["attributed"]:
        out["client_seconds"] = round(tally["client_seconds"], 3)
        out["server_seconds"] = round(tally["server_seconds"], 3)
        out["network_seconds"] = round(tally["client_seconds"]
                                       - tally["server_seconds"], 3)
    else:
        out["client_seconds"] = out["server_seconds"] = out["network_seconds"] = None
    if tally.get("why"):
        out["why"] = tally["why"]
    return out

