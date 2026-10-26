"""A live view of a vLLM server, read off its Prometheus `/metrics`.

**Built 2026-10-06 at the user's request** -- *a real-time monitor that taps
/metrics, focused on KV cache remaining, VRAM remaining, waiting, concurrent,
token throughput and cache hit*. The stress test samples `/metrics` too, but
only while a test runs and only to report a change over the run; this is the
view of the server as it is NOW, whoever is loading it.

What is here is the part that is not I/O: reading the exposition text with its
labels, turning one scrape into the figures the page shows, turning two scrapes
into rates, and finding the GPU's memory in whatever text carries it. The route
that fetches lives in `app.py`.

**vLLM only, on the server's own statement** -- the `backends.serves_metrics`
gate the stress test uses. llama-server's `/metrics` shares the task queue with
inference, which is why this project never polls it.

Three things worth knowing before reading the numbers:

* **vLLM does not report VRAM.** Its `/metrics` has the KV cache's share of the
  blocks it reserved, not the card's memory. VRAM comes from a GPU exporter
  scraped beside it (DCGM or nvidia_gpu_exporter), from those same series if the
  scrape already carries them, or from this machine's `nvidia-smi` when the
  model server IS this machine. Otherwise it is reported as unknown, with why.
* **KV cache remaining in TOKENS needs `cache_config_info`**, whose labels carry
  the block count and block size. Where a build does not export it, only the
  percentage is shown.
* **Batch size is generated tokens per engine step**, the only reading of it
  `/metrics` gives without its histogram buckets; and **requests that fit** is
  the KV cache divided by what a finished request costs, an estimate.
* **Rates are taken between this process's last two scrapes of that server**,
  whatever asked for them. Two tabs polling halve the interval and leave every
  rate correct, because each is a change over the time it actually covers.
"""

import ipaddress
import math
import re
import threading
import time
from urllib.parse import urlparse

_SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')

# A rate needs two scrapes close enough to mean "now". Further apart than this
# and the figure is an average over minutes nobody was looking at.
RATE_MAX_GAP = 120.0
RATE_MIN_GAP = 0.2

# Requests that must finish in a window before `fit` costs a request off the
# window rather than off the lifetime mean.
FIT_MIN_REQUESTS = 3

_lock = threading.Lock()
_last = {}          # url -> (monotonic time, snapshot)


def parse_samples(text: str) -> list:
    """Prometheus text exposition as `[(name, {label: value}, number)]`.

    Labels are kept, unlike `stress.parse_prometheus`, because two of the
    figures here live in them (`cache_config_info`'s block count and size) and
    because a percentage reported once per engine has to be AVERAGED, not summed.
    Histogram buckets and NaNs are dropped.
    """
    out = []
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m or m.group(1).endswith("_bucket"):
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if math.isnan(value):
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        out.append((m.group(1), labels, value))
    return out


def _series(samples, names):
    """The values of the first name in `names` the scrape carries."""
    for name in names:
        vals = [v for n, _, v in samples if n == name]
        if vals:
            return vals
    return []


def _sum(samples, *names):
    vals = _series(samples, names)
    return sum(vals) if vals else None


def _mean(samples, *names):
    vals = _series(samples, names)
    return sum(vals) / len(vals) if vals else None


def _int_label(labels, key):
    try:
        return int(float(labels.get(key, "")))
    except ValueError:
        return None


def kv_capacity(samples):
    """`(blocks, block_size)` from `vllm:cache_config_info`, or `(None, None)`.

    Summed over engines: with data parallelism each engine reports its own
    block pool, and the server's capacity is all of them.
    """
    blocks = size = None
    for name, labels, _ in samples:
        if name != "vllm:cache_config_info":
            continue
        b = _int_label(labels, "num_gpu_blocks")
        s = _int_label(labels, "block_size")
        if b:
            blocks = (blocks or 0) + b
        if s:
            size = s
    return blocks, size


def kv_max_concurrency(samples):
    """`kv_cache_max_concurrency` off `vllm:cache_config_info`, or None.

    How many requests fit at once if every one used the FULL context window --
    vLLM's own worst case. Summed over engines, like the block count.
    """
    total = None
    for name, labels, _ in samples:
        if name == "vllm:cache_config_info":
            try:
                total = (total or 0.0) + float(labels.get("kv_cache_max_concurrency", ""))
            except ValueError:
                pass
    return round(total, 2) if total is not None else None


def reserved_share(samples):
    """`gpu_memory_utilization` off `vllm:cache_config_info`, or None.

    The share of the card vLLM took at start-up for weights and KV cache. It is
    the only memory figure the vLLM port gives: the card's size is not in it.
    Averaged over engines, which share one setting.
    """
    vals = []
    for name, labels, _ in samples:
        if name == "vllm:cache_config_info":
            try:
                vals.append(float(labels.get("gpu_memory_utilization", "")))
            except ValueError:
                pass
    return round(sum(vals) / len(vals), 4) if vals else None


def finished_by_reason(samples) -> dict:
    """`vllm:request_success_total` per `finished_reason`, summed over engines.

    `stop` is a reply that ended itself, `length` one that ran to its token cap
    (where a looping reply ends up if nothing cuts it off), `abort` one the
    client hung up on -- which is what this app's loop guard does.
    """
    out = {}
    for name, labels, value in samples:
        if name == "vllm:request_success_total":
            reason = labels.get("finished_reason") or "other"
            out[reason] = out.get(reason, 0.0) + value
    return out


def gpu_memory(samples):
    """`{used_mb, total_mb, free_mb, gpus, source}` from exporter series, or None.

    Two exporters are understood, and they report in different units:

    * DCGM (`DCGM_FI_DEV_FB_USED` / `_FB_FREE`, MiB per GPU);
    * nvidia_gpu_exporter (`nvidia_smi_memory_used_bytes` / `_total_bytes`).
    """
    used = _series(samples, ["DCGM_FI_DEV_FB_USED"])
    free = _series(samples, ["DCGM_FI_DEV_FB_FREE"])
    if used and free and len(used) == len(free):
        u, f = sum(used), sum(free)
        return {"used_mb": round(u, 1), "free_mb": round(f, 1),
                "total_mb": round(u + f, 1), "gpus": len(used), "source": "DCGM"}
    used = _series(samples, ["nvidia_smi_memory_used_bytes"])
    total = _series(samples, ["nvidia_smi_memory_total_bytes"])
    if used and total and len(used) == len(total):
        mib = 1024 * 1024
        u, t = sum(used) / mib, sum(total) / mib
        return {"used_mb": round(u, 1), "free_mb": round(t - u, 1),
                "total_mb": round(t, 1), "gpus": len(used),
                "source": "nvidia_gpu_exporter"}
    return None


def parse_smi(text: str):
    """`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`."""
    used = total = 0.0
    n = 0
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used += float(parts[0])
            total += float(parts[1])
        except ValueError:
            continue
        n += 1
    if not n:
        return None
    return {"used_mb": round(used, 1), "free_mb": round(total - used, 1),
            "total_mb": round(total, 1), "gpus": n, "source": "nvidia-smi"}


def is_local(url: str) -> bool:
    """Is this server on this machine? Only then is this machine's GPU its GPU."""
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def snapshot(samples) -> dict:
    """One scrape as the figures the monitor shows. Absent is None, never 0."""
    usage = _mean(samples, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    blocks, size = kv_capacity(samples)
    total_tokens = blocks * size if blocks and size else None
    free_tokens = (round(total_tokens * (1 - usage))
                   if total_tokens is not None and usage is not None else None)
    running = _sum(samples, "vllm:num_requests_running")
    waiting = _sum(samples, "vllm:num_requests_waiting")
    swapped = _sum(samples, "vllm:num_requests_swapped")
    in_system = (None if running is None and waiting is None
                 else (running or 0) + (waiting or 0) + (swapped or 0))
    return {
        "running": running,
        "waiting": waiting,
        "swapped": swapped,
        # Everything the server is holding: what arrived and has not finished.
        "in_system": in_system,
        "reserved_share": reserved_share(samples),
        "kv_usage": round(usage, 4) if usage is not None else None,
        "kv_free": round(1 - usage, 4) if usage is not None else None,
        "kv_blocks": blocks,
        "kv_block_size": size,
        "kv_tokens_total": total_tokens,
        "kv_tokens_free": free_tokens,
        # Lifetime counters; the page wants their rates, which `rates` takes.
        "prompt_tokens": _sum(samples, "vllm:prompt_tokens_total"),
        "generation_tokens": _sum(samples, "vllm:generation_tokens_total"),
        "requests_ok": _sum(samples, "vllm:request_success_total"),
        "finished_by_reason": finished_by_reason(samples),
        "preemptions": _sum(samples, "vllm:num_preemptions_total",
                            "vllm:num_preemptions"),
        "prefix_hits": _sum(samples, "vllm:prefix_cache_hits_total",
                            "vllm:prefix_cache_hits"),
        "prefix_queries": _sum(samples, "vllm:prefix_cache_queries_total",
                               "vllm:prefix_cache_queries"),
        # v0 reports a hit rate directly, as a gauge.
        "prefix_hit_gauge": _mean(samples, "vllm:gpu_prefix_cache_hit_rate"),
        # The multimodal cache: an image whose preprocessing was reused. Counted
        # in ITEMS, not tokens, and separate from the prefix cache -- a request
        # sent with `cache_salt` still hits it.
        "mm_hits": _sum(samples, "vllm:mm_cache_hits_total", "vllm:mm_cache_hits"),
        "mm_queries": _sum(samples, "vllm:mm_cache_queries_total",
                           "vllm:mm_cache_queries"),
        # Engine steps, and the tokens they carried (prefill chunks included).
        "steps": _sum(samples, "vllm:iteration_tokens_total_count"),
        "step_tokens": _sum(samples, "vllm:iteration_tokens_total_sum"),
        # Per finished request: prompt and output tokens, for what one costs.
        "req_count": _sum(samples, "vllm:request_prompt_tokens_count"),
        "req_prompt_tokens": _sum(samples, "vllm:request_prompt_tokens_sum"),
        "req_output_tokens": _sum(samples, "vllm:request_generation_tokens_sum"),
        "kv_max_concurrency": kv_max_concurrency(samples),
    }


def _delta(cur, prev, key):
    a, b = cur.get(key), prev.get(key)
    if a is None or b is None or a < b:      # absent, or the server restarted
        return None
    return a - b


def rates(prev: dict, cur: dict, seconds: float) -> dict:
    """What changed between two snapshots, per second, and the window's hit rate."""
    out = {"seconds": round(seconds, 2)}
    for key, name in (("generation_tokens", "generation_tps"),
                      ("prompt_tokens", "prompt_tps"),
                      ("requests_ok", "requests_per_s"),
                      ("preemptions", "preemptions_per_s")):
        d = _delta(cur, prev, key)
        out[name] = round(d / seconds, 2) if d is not None and seconds > 0 else None
    # Requests RECEIVED per second, which vLLM has no counter for: whatever
    # finished in the window, plus however much the backlog grew. Exact at the
    # two scrape instants, because both terms are read off the same scrapes.
    done = _delta(cur, prev, "requests_ok")
    if (done is not None and cur.get("in_system") is not None
            and prev.get("in_system") is not None and seconds > 0):
        out["arrivals_per_s"] = round(
            max(0.0, done + cur["in_system"] - prev["in_system"]) / seconds, 2)
    else:
        out["arrivals_per_s"] = None
    by_reason = {}
    a, b = cur.get("finished_by_reason") or {}, prev.get("finished_by_reason") or {}
    for reason in sorted(set(a) | set(b)):
        if reason in a and reason in b and a[reason] >= b[reason]:
            by_reason[reason] = int(round(a[reason] - b[reason]))
    out["finished_by_reason"] = by_reason
    hits = _delta(cur, prev, "prefix_hits")
    queries = _delta(cur, prev, "prefix_queries")
    out["prefix_queries"] = queries
    # No queries in the window is "nothing to hit", not a 0% hit rate.
    out["prefix_hit_rate"] = round(hits / queries, 4) if hits is not None and queries else None
    out["preemptions"] = _delta(cur, prev, "preemptions")
    mm_hits, mm_queries = _delta(cur, prev, "mm_hits"), _delta(cur, prev, "mm_queries")
    out["mm_queries"] = mm_queries
    out["mm_hit_rate"] = (round(mm_hits / mm_queries, 4)
                          if mm_hits is not None and mm_queries else None)
    steps = _delta(cur, prev, "steps")
    gen, step_tokens = _delta(cur, prev, "generation_tokens"), _delta(cur, prev, "step_tokens")
    out["steps"] = steps
    out["batch"] = round(gen / steps, 2) if gen is not None and steps else None
    out["tokens_per_step"] = (round(step_tokens / steps, 1)
                              if step_tokens is not None and steps else None)
    count = _delta(cur, prev, "req_count")
    p, o = _delta(cur, prev, "req_prompt_tokens"), _delta(cur, prev, "req_output_tokens")
    out["req_count"] = count
    out["tokens_per_request"] = (round((p + o) / count)
                                 if p is not None and o is not None and count else None)
    return out


def lifetime_hit_rate(snap: dict):
    """Hits over queries since the server started, or v0's own gauge."""
    h, q = snap.get("prefix_hits"), snap.get("prefix_queries")
    if h is not None and q:
        return round(h / q, 4)
    g = snap.get("prefix_hit_gauge")
    return round(g, 4) if g is not None else None


def lifetime_mm_hit_rate(snap: dict):
    """Multimodal cache hits over lookups since the server started, or None."""
    h, q = snap.get("mm_hits"), snap.get("mm_queries")
    return round(h / q, 4) if h is not None and q else None


def lifetime_batch(snap: dict):
    """Generated tokens per engine step since start: the mean decode batch.

    Without speculative decoding a request emits one token per step, so tokens
    generated per step is how many requests a step decoded. Idle time makes no
    steps, so this is the batch while the server was working, not a time average.
    """
    g, n = snap.get("generation_tokens"), snap.get("steps")
    return round(g / n, 2) if g is not None and n else None


def fit(snap: dict, window: dict = None):
    """How many requests the KV cache holds before vLLM has to preempt one.

    `{per_request, source, requests, more_now, at_full_context}`, or None where
    the cache size or a finished request is missing. A request is costed at its
    prompt plus its whole output -- its footprint when it finishes -- using the
    window's finished requests where there were any and the lifetime mean
    otherwise. Prefix sharing is ignored, so the figure leans low.
    """
    total = snap.get("kv_tokens_total")
    window = window or {}
    # One or two requests finishing in a window make the figure jump on every
    # reading; fewer than FIT_MIN_REQUESTS and the lifetime mean answers.
    per = (window.get("tokens_per_request")
           if (window.get("req_count") or 0) >= FIT_MIN_REQUESTS else None)
    source = "window" if per else "lifetime"
    if not per:
        n = snap.get("req_count")
        p, o = snap.get("req_prompt_tokens"), snap.get("req_output_tokens")
        per = round((p + o) / n) if n and p is not None and o is not None else None
    if not total or not per:
        return None
    free = snap.get("kv_tokens_free")
    return {"per_request": per, "source": source,
            "requests": round(total / per, 1),
            "more_now": round(free / per, 1) if free is not None else None,
            "at_full_context": snap.get("kv_max_concurrency")}


def observe(url: str, snap: dict, now: float = None):
    """Remember this scrape and return the rates since the previous one, or None."""
    now = time.monotonic() if now is None else now
    with _lock:
        prev = _last.get(url)
        _last[url] = (now, snap)
    if not prev:
        return None
    gap = now - prev[0]
    if gap < RATE_MIN_GAP or gap > RATE_MAX_GAP:
        return None
    return rates(prev[1], snap, gap)


def _selftest():
    text = (
        '# HELP x\n'
        'vllm:num_requests_running{engine="0"} 3\n'
        'vllm:num_requests_running{engine="1"} 2\n'
        'vllm:num_requests_waiting{engine="0"} 4\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.5\n'
        'vllm:kv_cache_usage_perc{engine="1"} 0.3\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",engine="0"} 1\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",engine="1"} 1\n'
        'vllm:generation_tokens_total 100\n'
        'vllm:request_success_total{finished_reason="stop"} 8\n'
        'vllm:request_success_total{finished_reason="length"} 2\n'
        'vllm:prefix_cache_hits_total 30\n'
        'vllm:prefix_cache_queries_total 60\n'
        'vllm:e2e_request_latency_seconds_bucket{le="1"} 9\n')
    s = snapshot(parse_samples(text))
    assert s["running"] == 5 and s["waiting"] == 4
    assert abs(s["kv_usage"] - 0.4) < 1e-9                  # averaged, not summed
    assert s["kv_tokens_total"] == 32000 and s["kv_tokens_free"] == 19200
    assert lifetime_hit_rate(s) == 0.5
    later = dict(s, generation_tokens=300, prefix_hits=40, prefix_queries=80)
    r = rates(s, later, 2.0)
    assert r["generation_tps"] == 100 and r["prefix_hit_rate"] == 0.5
    assert rates(later, s, 2.0)["generation_tps"] is None   # a restart, not -100
    assert rates(s, dict(s), 2.0)["prefix_hit_rate"] is None
    assert s["in_system"] == 9 and s["finished_by_reason"] == {"stop": 8, "length": 2}
    busier = dict(s, requests_ok=14, in_system=11,
                  finished_by_reason={"stop": 11, "length": 2, "abort": 1})
    r = rates(dict(s, requests_ok=10), busier, 2.0)
    assert r["requests_per_s"] == 2 and r["arrivals_per_s"] == 3   # (4 + 2) / 2
    assert r["finished_by_reason"] == {"length": 0, "stop": 3}      # abort: no base
    g = gpu_memory(parse_samples('DCGM_FI_DEV_FB_USED{gpu="0"} 6000\n'
                                 'DCGM_FI_DEV_FB_FREE{gpu="0"} 18000\n'))
    assert g["free_mb"] == 18000 and g["total_mb"] == 24000
    assert parse_smi("4000, 6144\n")["free_mb"] == 2144
    assert is_local("http://127.0.0.1:8000") and not is_local("http://10.1.2.3:8000")
    assert snapshot([])["kv_usage"] is None
    a = snapshot(parse_samples(
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100",'
        'kv_cache_max_concurrency="2.5"} 1\n'
        'vllm:kv_cache_usage_perc 0.25\n'
        'vllm:generation_tokens_total 1000\n'
        'vllm:iteration_tokens_total_count 400\n'
        'vllm:iteration_tokens_total_sum 4000\n'
        'vllm:mm_cache_queries_total 10\nvllm:mm_cache_hits_total 1\n'
        'vllm:request_prompt_tokens_count 4\n'
        'vllm:request_prompt_tokens_sum 300\n'
        'vllm:request_generation_tokens_sum 100\n'))
    assert lifetime_batch(a) == 2.5 and lifetime_mm_hit_rate(a) == 0.1
    assert fit(a) == {"per_request": 100, "source": "lifetime", "requests": 16.0,
                      "more_now": 12.0, "at_full_context": 2.5}
    b = dict(a, generation_tokens=1400, steps=500, step_tokens=4600,
             mm_hits=4, mm_queries=14, req_count=7, req_prompt_tokens=500,
             req_output_tokens=200)
    r = rates(a, b, 2.0)
    assert r["batch"] == 4 and r["tokens_per_step"] == 6 and r["mm_hit_rate"] == 0.75
    assert r["tokens_per_request"] == 100 and fit(b, r)["source"] == "window"
    assert fit(snapshot([])) is None


_selftest()
