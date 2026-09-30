"""The stress test: N requests in flight against one model server, and what it did.

**Built 2026-09-25 at the user's request** -- *make it like a benchmark stress
test: I set the number of concurrent and the queue*, aimed at a high-end GPU
server that may be vLLM. Everything else in this project measures ONE run at a
time; this measures what the server does when many arrive at once.

This module holds the parts that are not I/O, so they can be tested without a
model server: how the dataset is drawn, how a 100+ page file is glued together
for the classification mode, how Prometheus `/metrics` is read, and how the
records of every job are rolled into the one report the page shows. The runner
itself lives in `app.py` beside the read and extraction paths it drives.

**Nothing here is written to the run log, and that is the user's rule, not an
economy** -- *do not show the independent values or record them, only the final
metric and score*. A stress run is a measurement of the SERVER under load; its
per-document figures were taken under contention nobody else ran under, and
averaging them into the setting tables would mix two different questions under
one column.
"""

import math
import random
import re

import segment

MODES = ("ocr", "extract", "classify", "full")
DEFAULT_MODE = "ocr"

# Hard ceilings, so a typo in a box cannot start ten thousand requests.
MAX_CONCURRENCY = 256
MAX_JOBS = 5000
MAX_WARMUP = 64
MAX_PACK_PAGES = 1000
DEFAULT_PACK_PAGES = 100

# How many timeline samples the report keeps. One per progress tick (about a
# second) would be thousands on a long run; the chart needs a few hundred.
TIMELINE_POINTS = 400

# Stages a job can be in, in pipeline order. `queued` is waiting for a worker.
STAGES = ("queued", "ocr", "extract", "segment", "classify")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def percentile(ordered, q: float):
    """Linear-interpolated percentile of an already-sorted list, or None."""
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values) -> dict:
    """n, mean, p50/p90/p95/p99, min, max of the non-None values -- or n=0.

    **Every latency on the page is one of these**, because a mean alone hides the
    only thing a stress test is for: the tail. A server that answers most
    requests in 3 s and one in twenty in 40 s has a fine mean and a queue.
    """
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"n": 0}
    r = lambda v: round(v, 3) if v is not None else None       # noqa: E731
    return {"n": len(vals), "mean": r(sum(vals) / len(vals)),
            "p50": r(percentile(vals, .50)), "p90": r(percentile(vals, .90)),
            "p95": r(percentile(vals, .95)), "p99": r(percentile(vals, .99)),
            "min": r(vals[0]), "max": r(vals[-1])}


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# --------------------------------------------------------------------------
# the dataset -- drawn the way the random test draws documents
# --------------------------------------------------------------------------

def draw_cases(count: int, cases: list, strategy: str, rng, history=None,
               lock: str = "") -> list:
    """Which document each job reads, `count` of them.

    **The same two rules the random test offers** (`randomtest.STRATEGIES`),
    because the user asked for the dataset to be set the same way:

    * `balanced` -- the least-read document each time, counting the run log's
      history plus the jobs drawn above it (`randomtest.case_order`);
    * `uniform` -- shuffled passes over the corpus, so every document is used
      once before any is used twice. Models, Detail and shape are fixed for the
      whole stress run, so the document IS the scenario, and this is the random
      test's *no scenario twice* rule with the only axis left.

    A lock pins every job to one document.
    """
    if lock:
        return [lock] * count
    if not cases:
        return []
    if strategy == "uniform":
        out, pool = [], sorted(cases)
        while len(out) < count:
            rng.shuffle(pool)
            out.extend(pool)
        return out[:count]
    import randomtest  # the planner's own rule, not a copy of it
    return [slot["case"] for slot in
            randomtest.case_order(count, sorted(cases), history or {}, rng)]


def case_pages(text: str) -> list:
    """A truth transcript, one string per page."""
    return segment.split_pages(text or "")


def build_pack(cases: list, truths: dict, docs_of: dict, target_pages: int,
               rng) -> dict:
    """One long file glued from ground-truth transcripts, and where its documents are.

    The classification mode's input: *handle 100+ page classification*. It is
    `mockOcr/long_pack.pdf`'s recipe applied to transcripts rather than to PDFs,
    so a job costs segmentation and classification and nothing else -- no page
    is read, and a boundary missed because a heading did not survive a read is
    not a boundary the walk got wrong.

    Documents are appended until the file reaches `target_pages`, drawn at
    random from `cases`, **never the same case twice in a row**: two identical
    documents back to back share a number, a date, parties and totals, so the
    boundary between them is one no reader could find and marking a model
    against it would be marking it against luck.

    `docs_of[case]` is the manifest's per-document split of that case --
    `[{"pages": [...], "doc_types": [...]}]` -- or empty for a case that is one
    document, whose types are then `docs_of`'s `"_types"` fallback.

    Returns the pages, the joined transcript (with `--- page N ---` markers, the
    shape a real multi-page read has) and the expected documents: each a list of
    global page numbers plus the types a person says it is.
    """
    pages, expected, used, previous = [], [], [], None
    pool = sorted(c for c in cases if truths.get(c))
    if not pool:
        return {"pages": [], "text": "", "expected": [], "cases": []}
    while len(pages) < max(2, target_pages):
        choices = [c for c in pool if c != previous] or pool
        case = rng.choice(choices)
        previous = case
        own = case_pages(truths[case])
        offset = len(pages)
        documents = docs_of.get(case) or []
        if documents:
            for doc in documents:
                expected.append({"pages": [offset + p for p in doc.get("pages") or []],
                                 "types": list(doc.get("doc_types") or []),
                                 "case": case})
        else:
            expected.append({"pages": [offset + p for p in range(1, len(own) + 1)],
                             "types": list(docs_of.get("_types", {}).get(case) or []),
                             "case": case})
        pages.extend(own)
        used.append(case)
    text = "\n\n".join(f"--- page {i} ---\n{p}" for i, p in enumerate(pages, 1))
    return {"pages": pages, "text": text, "expected": expected, "cases": used}


def grouping_score(found_groups, expected) -> dict:
    """How many of the expected documents came back as exactly their pages.

    A document counts only where BOTH of its boundaries are right -- the same
    page set, no more and no less -- because a document with one wrong edge
    answers its form out of another document's page or loses one of its own.
    """
    found = {tuple(sorted(g)) for g in found_groups}
    exact = sum(1 for doc in expected if tuple(sorted(doc["pages"])) in found)
    return {"expected": len(expected), "found": len(found_groups), "exact": exact}


# --------------------------------------------------------------------------
# vLLM /metrics
# --------------------------------------------------------------------------

_SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)')

# What the page shows, and how. `gauge` is read as-is; `counter` as the change
# since the test began; `hist` as mean = delta(_sum) / delta(_count).
# vLLM renamed several of these between v0 and v1, so each is a list of names
# and the first one the server exposes wins.
METRICS = [
    ("running", "gauge", ["vllm:num_requests_running"]),
    ("waiting", "gauge", ["vllm:num_requests_waiting"]),
    ("kv_cache", "gauge", ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]),
    ("prompt_tokens", "counter", ["vllm:prompt_tokens_total"]),
    ("generation_tokens", "counter", ["vllm:generation_tokens_total"]),
    ("requests_ok", "counter", ["vllm:request_success_total"]),
    ("preemptions", "counter", ["vllm:num_preemptions_total", "vllm:num_preemptions"]),
    ("prefix_hits", "counter", ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"]),
    ("prefix_queries", "counter", ["vllm:prefix_cache_queries_total",
                                   "vllm:prefix_cache_queries"]),
    ("ttft", "hist", ["vllm:time_to_first_token_seconds"]),
    ("tpot", "hist", ["vllm:inter_token_latency_seconds",
                      "vllm:time_per_output_token_seconds"]),
    ("queue_time", "hist", ["vllm:request_queue_time_seconds"]),
    ("e2e", "hist", ["vllm:e2e_request_latency_seconds"]),
    ("prefill_time", "hist", ["vllm:request_prefill_time_seconds"]),
    ("decode_time", "hist", ["vllm:request_decode_time_seconds"]),
]


def parse_prometheus(text: str) -> dict:
    """Prometheus text exposition, summed over label sets: `{name: value}`.

    Summed because a vLLM server reports one series per model and per engine,
    and what a stress test wants is the server's total. Histogram buckets are
    dropped -- the page shows means from `_sum`/`_count`, and the percentiles it
    cares about it measures itself, client-side, where they include the network.
    """
    out = {}
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
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


def _pick(values: dict, names, suffix=""):
    for name in names:
        if name + suffix in values:
            return values[name + suffix]
    return None


def metrics_view(now: dict, base: dict = None, seconds: float = None) -> dict:
    """The curated metrics: gauges now, counters and histograms since `base`.

    `base` absent means *since the server started* -- what the manual button
    shows. `seconds` turns the token counters into rates, which is the figure a
    stress test on vLLM is usually after: generation tokens per second as the
    SERVER counts them, beside the client-side figure the report computes.
    """
    base = base or {}
    view = {}
    for key, kind, names in METRICS:
        if kind == "gauge":
            view[key] = _pick(now, names)
        elif kind == "counter":
            cur = _pick(now, names)
            view[key] = None if cur is None else cur - (_pick(base, names) or 0.0)
        else:
            s, c = _pick(now, names, "_sum"), _pick(now, names, "_count")
            if s is None or c is None:
                view[key] = None
                continue
            s -= _pick(base, names, "_sum") or 0.0
            c -= _pick(base, names, "_count") or 0.0
            view[key] = round(s / c, 4) if c > 0 else None
            view[key + "_count"] = int(c)
    hits, queries = view.get("prefix_hits"), view.get("prefix_queries")
    view["prefix_hit_rate"] = (round(hits / queries, 4)
                               if hits is not None and queries else None)
    if seconds and seconds > 0:
        for key in ("prompt_tokens", "generation_tokens"):
            if view.get(key) is not None:
                view[key + "_per_s"] = round(view[key] / seconds, 2)
    view["known"] = sum(1 for key, _, _ in METRICS if view.get(key) is not None)
    return view


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

COMPLETED = ("ok", "looped", "truncated")


def thin_timeline(points: list, limit: int = TIMELINE_POINTS) -> list:
    """Every `k`-th sample, keeping the last, so a long run still draws."""
    if len(points) <= limit:
        return list(points)
    step = math.ceil(len(points) / limit)
    kept = points[::step]
    if kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def report(records: list, wall: float, mode: str, concurrency: int) -> dict:
    """Every job's record, rolled into the one report the page shows.

    **Three populations, and each figure says which it is over:**

    * `attempted` -- every job that started. The error rate is over this.
    * `completed` -- the server returned a transcript or a form (`ok`, and a
      read that looped or ran to the cap -- the server did that work, so its
      latency is real). Latencies and throughput are over this.
    * `ok` -- finished cleanly. **Scores are over this and nothing else**: the
      standing rule that a failure is counted and never scored, and a looped
      read's transcript is a fragment whatever it scored.

    Cancelled and skipped jobs are counted and in no figure: a job nobody let
    finish says nothing about the server.
    """
    started = [r for r in records if r.get("status") not in ("skipped",)]
    attempted = [r for r in started if r.get("status") != "cancelled"]
    done = [r for r in attempted if r.get("status") in COMPLETED]
    ok = [r for r in attempted if r.get("status") == "ok"]
    failed = [r for r in attempted if r.get("status") == "error"]
    wall = max(wall or 0.0, 1e-9)

    def total(key, rows=done):
        return sum(r.get(key) or 0 for r in rows)

    counts = {s: sum(1 for r in records if r.get("status") == s)
              for s in ("ok", "looped", "truncated", "error", "cancelled", "skipped")}
    out = {
        "mode": mode,
        "concurrency": concurrency,
        "jobs": len(records),
        "attempted": len(attempted),
        "completed": len(done),
        "counts": counts,
        "error_rate": round(len(failed) / len(attempted), 4) if attempted else None,
        "wall_seconds": round(wall, 2),
        "jobs_per_minute": round(len(done) / wall * 60, 2),
        "latency": {"end_to_end": distribution(r.get("total_s") for r in done),
                    "queue_wait": distribution(r.get("wait_s") for r in attempted)},
        "errors": _top_errors(failed),
    }
    if mode in ("ocr", "full"):
        pages = total("pages")
        tokens = total("ocr_tokens")
        busy = total("ocr_s")
        out["ocr"] = {
            "pages": pages,
            "pages_per_second": round(pages / wall, 3),
            "tokens": tokens,
            # Client-side, over the whole wall clock: what the SERVER produced
            # per second with this many requests in flight.
            "tokens_per_second": round(tokens / wall, 2),
            "latency_job": distribution(r.get("ocr_s") for r in done),
            "latency_page": distribution(p for r in done for p in r.get("page_s") or []),
            "ttft": distribution(p for r in done for p in r.get("ttft_s") or []),
            # One request's own decode rate. Falls as concurrency rises on a
            # batching server while the aggregate above climbs -- the two
            # together are the batching curve.
            "decode_tps": distribution(p for r in done for p in r.get("page_tps") or []),
            "busy_seconds": round(busy, 2),
            "char_accuracy": _mean(r.get("char_accuracy") for r in ok),
            "char_scored": sum(1 for r in ok if r.get("char_accuracy") is not None),
            "looped": counts["looped"], "truncated": counts["truncated"],
        }
    if mode in ("extract", "full"):
        rows = [r for r in done if r.get("extract_s") is not None]
        scored = [r for r in ok if r.get("field_expected") and not r.get("unscored")]
        correct = sum((r.get("field_correct") or 0) + 0.5 * (r.get("field_partial") or 0)
                      for r in scored)
        expected = sum(r.get("field_expected") or 0 for r in scored)
        busy = sum(r.get("extract_s") or 0 for r in rows)
        out["extract"] = {
            "latency_job": distribution(r.get("extract_s") for r in rows),
            "tokens": sum(r.get("extract_tokens") or 0 for r in rows),
            "tokens_per_second": round(sum(r.get("extract_tokens") or 0
                                           for r in rows) / wall, 2),
            "busy_seconds": round(busy, 2),
            # Macro (each job one vote) and pooled (each value one vote), the
            # two readings the Fields tab already names.
            "field_accuracy": _mean(r.get("field_rate") for r in scored),
            "field_pooled": round(correct / expected, 4) if expected else None,
            "field_values": expected,
            "field_scored": len(scored),
            "unscored": sum(1 for r in ok if r.get("unscored")),
            "partial_replies": sum(1 for r in rows if r.get("partial_reply")),
        }
    if mode == "full" and out.get("ocr") and out.get("extract"):
        a, b = out["ocr"]["busy_seconds"], out["extract"]["busy_seconds"]
        out["stage_share"] = {"ocr": round(a / (a + b), 4) if a + b else None,
                              "extract": round(b / (a + b), 4) if a + b else None}
    if mode == "classify":
        exp = total("docs_expected", ok)
        pages = total("pages")
        out["classify"] = {
            "pages": pages,
            "pages_per_second": round(pages / wall, 3),
            "latency_segment": distribution(r.get("segment_s") for r in done),
            "latency_classify": distribution(r.get("classify_s") for r in done),
            "asks": total("asks"),
            "asks_per_100_pages": round(100 * total("asks") / pages, 2) if pages else None,
            "refused": total("refused"),
            "documents_expected": exp,
            "documents_found": total("docs_found", ok),
            "grouping_exact": round(total("docs_exact", ok) / exp, 4) if exp else None,
            # Of the documents grouped exactly -- a type is only comparable on a
            # document whose pages are the right ones.
            "types_exact": (round(total("types_exact", ok) / total("docs_exact", ok), 4)
                            if total("docs_exact", ok) else None),
            "form_match": (round(total("form_match", ok) / total("docs_exact", ok), 4)
                           if total("docs_exact", ok) else None),
            "model_classified": total("model_classified", ok),
        }
    return out


def _top_errors(failed: list, limit: int = 5) -> list:
    """The commonest error messages, counted -- never which document they were on."""
    tally = {}
    for r in failed:
        key = (r.get("error") or "error")[:160]
        tally[key] = tally.get(key, 0) + 1
    return [{"error": k, "count": v}
            for k, v in sorted(tally.items(), key=lambda kv: -kv[1])[:limit]]


def seeded_rng(seed):
    """`(rng, seed)` -- a new seed when none was given, so a run can be repeated."""
    if seed in (None, ""):
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = abs(hash(str(seed))) % (2 ** 31)
    return random.Random(seed), seed


def _selftest():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    d = distribution([3, 1, 2, None])
    assert d["n"] == 3 and d["p50"] == 2 and d["max"] == 3
    assert distribution([])["n"] == 0
    rng = random.Random(1)
    got = draw_cases(7, ["a", "b", "c"], "uniform", rng)
    assert len(got) == 7 and set(got[:3]) == {"a", "b", "c"}
    assert draw_cases(3, ["a"], "uniform", rng, lock="z") == ["z", "z", "z"]
    assert draw_cases(2, ["a", "b"], "balanced", rng, {"a": 3})[0] == "b"
    text = ("# HELP x\nvllm:num_requests_running{model=\"a\"} 2\n"
            "vllm:num_requests_running{model=\"b\"} 1\n"
            "vllm:generation_tokens_total 100\n"
            "vllm:e2e_request_latency_seconds_sum 10\n"
            "vllm:e2e_request_latency_seconds_count 4\n"
            "vllm:e2e_request_latency_seconds_bucket{le=\"1\"} 3\n")
    now = parse_prometheus(text)
    assert now["vllm:num_requests_running"] == 3
    assert "vllm:e2e_request_latency_seconds_bucket" not in now
    view = metrics_view(now, {"vllm:generation_tokens_total": 40.0}, seconds=10)
    assert view["running"] == 3 and view["generation_tokens"] == 60
    assert view["generation_tokens_per_s"] == 6 and view["e2e"] == 2.5
    truths = {"a": "p1", "b": "q1\n--- page 2 ---\nq2"}
    pack = build_pack(["a", "b"], truths, {"b": [{"pages": [1, 2], "doc_types": ["X"]}],
                                            "_types": {"a": ["Y"]}}, 6, random.Random(3))
    assert len(pack["pages"]) >= 6
    assert all(p1 != p2 for p1, p2 in zip(pack["cases"], pack["cases"][1:]))
    assert sum(len(d["pages"]) for d in pack["expected"]) == len(pack["pages"])
    assert grouping_score([[1], [2, 3]], [{"pages": [1]}, {"pages": [2]}])["exact"] == 1
    rep = report([{"status": "ok", "total_s": 2, "wait_s": 0, "pages": 1,
                   "ocr_s": 2, "ocr_tokens": 10, "char_accuracy": .9},
                  {"status": "error", "error": "boom", "wait_s": 1},
                  {"status": "cancelled"}], 4.0, "ocr", 2)
    assert rep["attempted"] == 2 and rep["error_rate"] == .5
    assert rep["ocr"]["char_accuracy"] == .9 and rep["errors"][0]["count"] == 1


_selftest()
