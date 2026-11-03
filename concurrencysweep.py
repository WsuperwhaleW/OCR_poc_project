"""The concurrency sweep: one stress run that steps its concurrency up over time,
and every vLLM /metrics figure against the concurrency it was taken at.

**Built 2026-11-03 at the user's request** -- *a new test that changes the
concurrency over time, working directly with vLLM /metrics: throughput, time
between tokens, time to first token, e2e and latency, every metric against the
concurrency, each with a graph*. It is the curve the stress test's *Not done*
named: concurrency 1, 2, 4, 8, ... at a fixed queue, reading the aggregate
tokens/s against one request's decode rate and the latency tails.

**Every level runs the SAME files, in the same order** (the user's rule: *a
sweep up to 128 OCRs the same 128 files at 1 concurrent as at 128*). The levels
then differ in one thing only, the concurrency; a different draw per level
would put the documents' own spread -- a 1-page receipt against a 13-page
statement -- into the curve.

The runner is the stress test's own (`app._stress_run` runs one `_stress_phase`
per level), so a level's jobs are exactly a stress run's jobs. This module holds
the parts that are not I/O: the levels, the histogram buckets the stress view
drops, and the per-level figures.

**Each level is read off the server as a DELTA**: one full /metrics scrape right
before the level's first job and one right after its last, so a histogram's
percentiles are over that level's requests only. Prometheus histograms are
cumulative since the server started; a percentile of the totals would mix every
level before it into the one being drawn.
"""

import math
import re

import stress

# Powers of two up to the stress test's own ceiling are the usual batching
# curve; a box can ask for any ascending list.
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)
MAX_LEVELS = 16

# The histograms read per level. Each is a list of names, the first one the
# server exposes wins -- vLLM renamed time-per-output-token between versions.
HISTOGRAMS = [
    ("ttft", ["vllm:time_to_first_token_seconds"]),
    ("itl", ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"]),
    ("e2e", ["vllm:e2e_request_latency_seconds"]),
    ("queue", ["vllm:request_queue_time_seconds"]),
    ("prefill", ["vllm:request_prefill_time_seconds"]),
    ("decode", ["vllm:request_decode_time_seconds"]),
    ("inference", ["vllm:request_inference_time_seconds"]),
    ("prompt_len", ["vllm:request_prompt_tokens"]),
    ("output_len", ["vllm:request_generation_tokens"]),
]
COUNTERS = [
    ("generation_tokens", ["vllm:generation_tokens_total"]),
    ("prompt_tokens", ["vllm:prompt_tokens_total"]),
    ("requests", ["vllm:request_success_total"]),
    ("preemptions", ["vllm:num_preemptions_total", "vllm:num_preemptions"]),
    ("prefix_hits", ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"]),
    ("prefix_queries", ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"]),
]
# Read from the stress sampler's ticks during the level: mean and peak.
GAUGES = ("running", "waiting", "kv_cache")
QUANTILES = (("p50", .50), ("p90", .90), ("p99", .99))


def parse_levels(value) -> list:
    """The levels asked for, ascending and distinct, or ValueError.

    Takes a list or a string such as `"1, 2, 4 8"`. Blank is the default.
    """
    if value in (None, "", []):
        return list(DEFAULT_LEVELS)
    items = value if isinstance(value, list) else re.split(r"[\s,;]+", str(value).strip())
    levels = set()
    for item in items:
        if item in ("", None):
            continue
        try:
            level = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"sweep levels must be whole numbers; {item!r} is not.")
        if not 1 <= level <= stress.MAX_CONCURRENCY:
            raise ValueError(f"sweep levels must be 1 to {stress.MAX_CONCURRENCY}; "
                             f"{level} is not.")
        levels.add(level)
    if not levels:
        return list(DEFAULT_LEVELS)
    if len(levels) > MAX_LEVELS:
        raise ValueError(f"a sweep takes at most {MAX_LEVELS} levels.")
    return sorted(levels)


def level_jobs(levels: list, asked=None) -> int:
    """How many jobs every level queues: `asked`, or the highest level.

    The highest level is the default because it is the smallest queue that
    fills that level once; every lower level works through the same jobs in
    more waves.
    """
    try:
        asked = int(asked) if asked not in (None, "") else 0
    except (TypeError, ValueError):
        raise ValueError("sweep_jobs must be a whole number.")
    return max(1, min(asked or max(levels), stress.MAX_JOBS))


# --------------------------------------------------------------------------
# histogram buckets
# --------------------------------------------------------------------------

_BUCKET = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)_bucket\{([^}]*)\}\s+(\S+)')
_LE = re.compile(r'(?:^|,)\s*le="([^"]*)"')


def parse_buckets(text: str) -> dict:
    """Every histogram's buckets: `{name: {le: count}}`, summed over label sets.

    `stress.parse_prometheus` drops the buckets -- its view is means -- and a
    percentile needs them. Summed for the reason that function sums: a vLLM
    server reports one series per model and engine, and the sweep is about the
    server. `le` is a float; `+Inf` is `math.inf`.
    """
    out = {}
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _BUCKET.match(line)
        if not m:
            continue
        le = _LE.search(m.group(2))
        if not le:
            continue
        try:
            bound = math.inf if le.group(1) in ("+Inf", "Inf") else float(le.group(1))
            value = float(m.group(3))
        except ValueError:
            continue
        if math.isnan(value):
            continue
        series = out.setdefault(m.group(1), {})
        series[bound] = series.get(bound, 0.0) + value
    return out


def bucket_delta(now: dict, base: dict):
    """`now - base`, bucket by bucket, or None where a count went DOWN (a restart)."""
    out = {}
    for le, count in (now or {}).items():
        diff = count - (base or {}).get(le, 0.0)
        if diff < -1e-9:
            return None
        out[le] = max(0.0, diff)
    return out


def quantile(q: float, buckets: dict):
    """Prometheus' `histogram_quantile` over cumulative buckets, or None.

    Linear within the bucket the rank lands in, from the bucket below's bound
    (0 under the first). A rank in the `+Inf` bucket is reported as the highest
    finite bound -- the most the histogram can say.
    """
    if not buckets or math.inf not in buckets:
        return None
    ordered = sorted(buckets.items())
    total = buckets[math.inf]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= rank:
            if le == math.inf:
                return prev_le
            if count == prev_count:
                return le
            lo = prev_le if prev_le < le else 0.0
            return lo + (le - lo) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return prev_le


def _pick(values: dict, names, suffix=""):
    for name in names:
        if name + suffix in values:
            return name, values[name + suffix]
    return None, None


# --------------------------------------------------------------------------
# one level
# --------------------------------------------------------------------------

def level_view(before: tuple, after: tuple, seconds: float, ticks=()) -> dict:
    """The server's figures for one level, from the scrapes either side of it.

    `before` and `after` are `(values, buckets)` -- `stress.parse_prometheus`
    and `parse_buckets` of the same scrape. `seconds` is the level's wall clock
    and turns the counters into rates. `ticks` are the stress sampler's views
    during the level, for the gauges.

    Each histogram comes back as `{mean, p50, p90, p99, count}` over the
    level's requests; a series the server does not expose is absent, and a
    level with none of its requests finished has `count: 0` and no figures.
    """
    v0, b0 = before
    v1, b1 = after
    seconds = max(seconds or 0.0, 1e-9)
    out = {"seconds": round(seconds, 2)}
    for key, names in HISTOGRAMS:
        name, s1 = _pick(v1, names, "_sum")
        if name is None:
            continue
        c1 = v1.get(name + "_count")
        if c1 is None:
            continue
        s = s1 - (v0.get(name + "_sum") or 0.0)
        c = c1 - (v0.get(name + "_count") or 0.0)
        if c < 0:           # the server restarted mid-level
            out[key] = {"count": None}
            continue
        entry = {"count": int(round(c))}
        if c > 0:
            entry["mean"] = round(s / c, 4)
            delta = bucket_delta(b1.get(name), b0.get(name))
            for label, q in QUANTILES:
                value = quantile(q, delta) if delta else None
                entry[label] = round(value, 4) if value is not None else None
        out[key] = entry
    for key, names in COUNTERS:
        name, cur = _pick(v1, names)
        if name is None:
            continue
        diff = cur - (v0.get(name) or 0.0)
        out[key] = None if diff < 0 else diff
    for key in ("generation_tokens", "prompt_tokens", "requests"):
        if out.get(key) is not None:
            out[key + "_per_s"] = round(out[key] / seconds, 3)
    hits, queries = out.get("prefix_hits"), out.get("prefix_queries")
    out["prefix_hit_rate"] = round(hits / queries, 4) if hits is not None and queries else None
    itl = (out.get("itl") or {}).get("mean")
    # One request's own decode rate: falls as concurrency rises on a batching
    # server while the aggregate above climbs.
    out["decode_rate"] = round(1 / itl, 2) if itl else None
    for key in GAUGES:
        vals = [t.get(key) for t in ticks if t.get(key) is not None]
        out[key] = ({"mean": round(sum(vals) / len(vals), 4), "max": round(max(vals), 4),
                     "n": len(vals)} if vals else None)
    return out


def client_view(rep: dict) -> dict:
    """What this side measured over the same level, from `stress.report`.

    `latency` is a whole job, end to end, measured here (network and this
    side's queue included); `achieved` is the mean number of jobs in flight
    (busy time over wall clock, Little's law), which a level's tail pulls under
    the level's limit.
    """
    lat = rep.get("latency") or {}
    ocr = rep.get("ocr") or {}
    extract = rep.get("extract") or {}
    tokens = (ocr.get("tokens") or 0) + (extract.get("tokens") or 0)
    wall = rep.get("wall_seconds") or 0
    e2e = lat.get("end_to_end") or {"n": 0}
    return {
        "completed": rep.get("completed"),
        "attempted": rep.get("attempted"),
        "error_rate": rep.get("error_rate"),
        "jobs_per_minute": rep.get("jobs_per_minute"),
        "tokens_per_second": round(tokens / wall, 2) if wall else None,
        "latency": e2e,
        "page_latency": ocr.get("latency_page"),
        "ttft": ocr.get("ttft"),
        "decode_tps": ocr.get("decode_tps"),
        "achieved": (round(e2e["mean"] * e2e["n"] / wall, 2)
                     if wall and e2e.get("n") else None),
    }


def summary(levels: list, role_index: int = 0) -> dict:
    """Where throughput peaked, and the first level that bought under 10% more.

    Over the first server's output tokens/s. The knee is where adding requests
    stops adding throughput and starts adding only latency.
    """
    points = [(row["level"], (row["servers"][role_index]["view"]
                              if len(row.get("servers") or []) > role_index else {})
               .get("generation_tokens_per_s")) for row in levels]
    points = [(lv, tps) for lv, tps in points if tps is not None]
    if not points:
        return {"peak_level": None, "peak_tps": None, "knee_level": None}
    peak = max(points, key=lambda p: p[1])
    knee = next((lv for (_, prev), (lv, tps) in zip(points, points[1:])
                 if prev > 0 and tps < prev * 1.10), None)
    return {"peak_level": peak[0], "peak_tps": peak[1], "knee_level": knee}


def _selftest():
    assert parse_levels("8, 1 2,2") == [1, 2, 8]
    assert parse_levels("") == list(DEFAULT_LEVELS)
    text = ('vllm:e2e_request_latency_seconds_bucket{le="1.0",model_name="a"} 2\n'
            'vllm:e2e_request_latency_seconds_bucket{le="2.0",model_name="a"} 6\n'
            'vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="a"} 8\n'
            'vllm:e2e_request_latency_seconds_bucket{model_name="b",le="+Inf"} 2\n')
    b = parse_buckets(text)["vllm:e2e_request_latency_seconds"]
    assert b == {1.0: 2.0, 2.0: 6.0, math.inf: 10.0}
    assert quantile(.5, {1.0: 2.0, 2.0: 6.0, math.inf: 8.0}) == 1.5
    assert quantile(.99, {1.0: 2.0, 2.0: 6.0, math.inf: 8.0}) == 2.0
    assert bucket_delta({1.0: 1.0}, {1.0: 3.0}) is None


_selftest()
