"""Thai/English document OCR web app.

Inference runs entirely on an external model server -- llama.cpp's `llama-server`
or Ollama -- over its OpenAI-compatible HTTP API. No model weights, torch, or
transformers in this process. This app only decodes uploads into page images,
caps their resolution, and streams the server's Markdown back to the browser.
Which server is in use can be switched from the page; see `backends.py`.

The resolution cap is the dominant cost lever, because prefill scales with pixels.

Accepts any raster format Pillow can open, plus PDFs (PyMuPDF), HEIC/HEIF
(pillow-heif) and multi-page TIFF/GIF.
"""

import base64
import copy
import gzip
import zlib
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import queue
import uuid
import weakref
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path

import requests
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

import backends
import settings
import blame
import concurrencysweep
import config
import dryreplay
import doctype_mapping
import fieldscore
import extractionlog
import extractionlimit
import fixscore
import grounding
import machine
import monitor
import normalise
import prompts
import priority
import userprompts
import userpromptsets
import randomtest
import runlog
import runhistory
import scoring
import segment
import segmentparallel
import servertime
import stress
import tables
import validate
import verify
import vllmbatch
import workspace

# Prompt text and tunables live outside this file so either can be read, edited
# and diffed without the request-assembly code around them. `prompts.py` holds
# the prompts; `settings.py` holds every value the app runs with, and the
# comment beside each one says what was measured to arrive at it.
from prompts import (
    EXTRACT_REMINDER,
    extract_prompt,
    rules_for_types,
    type_phrase,
    type_line,
    step_rules,
    CLASSIFY_PROMPT,
    classify_vocabulary,
    TYPE_NAMES,
    extract_schema,
    INCOME_ITEMS_KEY,
    fields_for_types,
    income_items_schema,
    items_for_types,
    mandatory_for_types,
    mandatory_items_for_types,
    steps_for_types as _builtin_steps_for_types,
    EXTRACT_STEP_PREFIX,
    EXTRACT_STEP_RETRY,
    EXTRACT_STEP_TASK,
    HYBRID_STEP_TASK,
    EXTRACT_STEPS,
    LINE_ITEM_SCHEMA,
    OTHER_FIELDS_SCHEMA,
    PROMPT,
    OCR_PROFILES,
    DEFAULT_OCR_PROFILE,
)
from settings import (
    OCR_PROFILE,
    MONITOR_GPU_URL,
    MONITOR_IDLE_MAX,
    MONITOR_MIN_INTERVAL,
    ACCEPTED_SUFFIXES,
    AGENTIC_EXTRACT,
    EXTRACT_OTHER_FIELDS,
    AUTO_SELECT_SERVER,
    AGENTIC_PARALLEL,
    AGENTIC_PARALLEL_MAX,
    EXTRACT_DOCUMENT_PARALLEL_MAX,
    EXTRACT_REQUEST_CONCURRENCY,
    OCR_PARALLEL,
    OCR_PARALLEL_MAX,
    OCR_PARALLEL_MIN_PAGES,
    VLLM_FIT_MAX_TOKENS,
    VLLM_FIT_MIN_TOKENS,
    VLLM_PREFIX_WARMUP,
    VLLM_PREFIX_WARMUP_WAIT,
    AGENTIC_RETRIES,
    DEFAULT_DETAIL,
    DETAIL_PRESETS,
    DRY_MULTIPLIER,
    next_detail,
    resolve_detail,
    EXTRACT,
    EXTRACT_LOOP_MIN_REPEATS,
    EXTRACT_LOOP_TAIL_CHARS,
    CLASSIFY_ESCALATE_MULTI,
    CLASSIFY_MAX_CHARS,
    CLASSIFY_MAX_TOKENS,
    CLASSIFY_MIN_CONFIDENCE,
    CLASSIFY_WITH_MODEL,
    SEGMENT_DOCUMENTS,
    SEGMENT_MAX_CHARS,
    SEGMENT_MAX_PAGES,
    SEGMENT_CHAT,
    SEGMENT_CHAT_WINDOW,
    SEGMENT_TYPE_GATE,
    SEGMENT_CHAT_MAX_ASKS,
    SEGMENT_PARALLEL_MAX,
    SEGMENT_PARALLEL_MIN_PAGES,
    SEGMENT_PARALLEL_OVERLAP,
    SEGMENT_MAX_TOKENS,
    SEGMENT_WITH_MODEL,
    TABLE_EXTRACT,
    TABLE_CONCAT_AGENT,
    TABLE_FIX_AGENT,
    TABLE_CONTEXT_SHARE,
    TABLE_FIX_MAX_ROWS,
    TABLE_IDENTIFY_AGENT,
    TABLE_IDENTIFY_MAX_ROWS,
    TABLE_RECUT_MAX_ROWS,
    TABLE_RECUT_MAX_TOKENS,
    TABLE_RECUT_WITH_MODEL,
    TYPE_FRAMING_AGENTIC,
    TYPE_FRAMING_BULLETS,
    TYPE_FRAMING_SINGLE,
    EXTRACT_MAX_TOKENS,
    EXTRACT_REPEAT_THRESHOLD,
    EXTRACT_LOOP_ABORT,
    EXTRACT_CYCLE_MIN_CHARS,
    EXTRACT_SCHEMA,
    GEN_TIMEOUT,
    LOOP_CHECK_EVERY,
    STREAM_FLUSH_MS,
    LOOP_COUNTER_MAX_UNIT,
    LOOP_COUNTER_MIN_LINE,
    LOOP_COUNTER_MIN_REPEATS,
    LOOP_DEAD_MAX_LETTERS,
    LOOP_DEAD_TAIL_CHARS,
    LOOP_GUARD,
    LOOP_MIN_REPEATS,
    LOOP_RETRY,
    LOOP_TAIL_CHARS,
    STRESS_DRY_RETRY,
    MAX_JOBS,
    MAX_NEW_TOKENS,
    MAX_PAGES,
    MAX_UPLOAD_MB,
    MIN_READ_FOR_FIELDS,
    PDF_DPI,
    PROMPT_FIRST,
    PROMPT_FIRST_OLLAMA,
    SPLIT_PAGE_RATIO,
    SPLIT_STACKED_PAGES,
    SUMMARY_RUNS,
    TRIM_MARGINS,
    TRIM_PAD,
    TRIM_TOLERANCE,
    ocr_sampling,
    ocr_sampling_state,
    sampler_extras,
    set_ocr_sampling,
)

from PIL import Image, ImageChops, ImageOps, ImageSequence, UnidentifiedImageError

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:  # pragma: no cover - optional dependency
    HEIF_OK = False

try:
    import fitz  # PyMuPDF

    PDF_OK = True
except ImportError:  # pragma: no cover - optional dependency
    PDF_OK = False

# The engine (input, read, grouping, extraction, checks) lives in engine.py;
# this module is the benchmark app around it. See engine.py.
import engine
import webcommon


app = Flask(__name__)
# Upload cap, sub-path mount, per-tab workspace and gzip, shared with service_app.
webcommon.install(app, MAX_UPLOAD_MB)


# --------------------------------------------------------------------------
# Sweeps in flight
#
# **What the queue's 409 was actually protecting, kept after the queue went.**
# The rule is that a run is attributed to the server that ran it: model, backend
# and URL are captured per page as it is read, so half a BATCH on one server and
# half on another would be logged and scored as if one server had done it. The
# job queue used to be the only batch this process could run, so the guard
# counted queued jobs.
#
# The random test is a batch too -- many rounds, and it switches models between
# every one of them -- and it was never covered by that count, because it was not
# the queue. So the counter moved to the thing that still needs it rather than
# being deleted with the thing that did.
#
# **A single streaming read is deliberately NOT counted.** That was true before
# and stays true: one read is attributed correctly whatever is selected next, and
# the switch simply takes effect on the following run. It is only a run made of
# SEVERAL reads that can be split across two servers.
# The thread running a sweep, or None. **The THREAD is the sweep** -- Werkzeug's
# threaded server handles one request per thread and does not pool them, so a
# request that ended, however it ended, is a thread that is no longer alive.
#
# **One per WORKSPACE** (2026-10-19): each browser tab has its own pickers, so a
# sweep in one tab switches only that tab's models and a sweep in another is no
# business of its guard. The globals are the default workspace's -- the CLI's.
_sweep_thread = None
_sweep_cancel = None
_sweep_lock = threading.Lock()
workspace.field("sweep", lambda: (None, None), fresh=True)


def _sweep_state() -> tuple:
    """(thread, cancel) of the current workspace's sweep."""
    return workspace.get("sweep", (_sweep_thread, _sweep_cancel))


def _set_sweep(thread, cancel):
    global _sweep_thread, _sweep_cancel
    if not workspace.put("sweep", (thread, cancel)):
        _sweep_thread, _sweep_cancel = thread, cancel


def begin_sweep(cancel=None):
    """Mark this thread as running a sweep, and hand back its cancel flag.

    `cancel` lets the stress test bring a `MultiCancel`; the random test takes
    the one-slot default.
    """
    with _sweep_lock:
        thread, _ = _sweep_state()
        if thread is not threading.current_thread():
            workspace.hold()
        cancel = cancel or engine.Cancel()
        _set_sweep(threading.current_thread(), cancel)
        return cancel


def end_sweep():
    """Release it. Idempotent, and only the thread that took it may release."""
    with _sweep_lock:
        thread, _ = _sweep_state()
        if thread is threading.current_thread():
            _set_sweep(None, None)
            workspace.release()


def cancel_sweep() -> bool:
    """Ask the running sweep to stop. True if there was one to ask.

    **Stop has to be a REQUEST, and that is the whole reason this exists.** The
    page used to stop a sweep by aborting its own `fetch`, which tells the
    server nothing -- and an abandoned streaming response is not stopped here:
    it runs every remaining chunk and only then reaches the generator's
    `finally` (measured; see `sweeps_running`). So every round nobody wanted
    went on reading at 20-80s each, and the next Run was refused with a 409 that
    was perfectly accurate and read as a bug.

    **Setting the flag does not release the guard**, deliberately. The sweep is
    still running until it notices, cuts its round short and writes that row, so
    a 409 in that window is still the truth. What changed is that the window is
    one round rather than the rest of the plan.
    """
    with _sweep_lock:
        thread, cancel = _sweep_state()
    if cancel is not None and thread is not None and thread.is_alive():
        cancel.set()
        return True
    return False


def stop_requested(cancel) -> bool:
    """Has this round been asked to stop? Tolerates `None` for a plain call."""
    return cancel is not None and cancel.is_set()


def sweeps_running() -> bool:
    """Is a multi-read run in flight on some OTHER thread?

    **What a client disconnect actually does, measured rather than assumed, on a
    minimal Flask app against the dev server:** an abandoned streaming response
    is NOT stopped -- it runs every remaining chunk, then its generator's
    `finally` runs and the request thread dies. `call_on_close` never fires at
    all. Both halves of that are the opposite of what was assumed while this was
    built, and both cost time: a guard that stayed at 409 for a minute after a
    killed `curl` reads exactly like a leaked counter, and is not one -- the
    sweep really was still reading.

    So `end_sweep` in the generator's `finally` is sufficient on this server, and
    the thread test is what makes it not DEPEND on that. A dead thread cannot
    still be reading, so it is exact, needs no timeout to guess with, and cannot
    strand the guard if a `finally` is ever skipped -- which is worth having
    because the failure it prevents is a process that refuses every server switch
    until it is restarted.

    **A 409 that outlasts the client is therefore correct**, not stale: the run
    is still going, and it is still the run the guard is protecting.
    """
    with _sweep_lock:
        thread, _ = _sweep_state()
    return bool(thread and thread.is_alive()
                and thread is not threading.current_thread())


def _send_batch(url: str, body: dict) -> dict:
    """`send` for a `Batcher` -- one POST to vLLM's batch route, parsed. Only a
    vLLM server is ever batched, so the request goes on the pooled connection."""
    res = backends.http_post(url, {"kind": "vllm"}, json=body, timeout=engine.GEN_TIMEOUT)
    if res.status_code == 404:
        raise vllmbatch.BatchError(
            f"{url} answered HTTP 404 -- this vLLM build has no batch route. "
            "Turn off Batch requests, or update vLLM.", 404)
    if res.status_code != 200:
        raise vllmbatch.BatchError(
            f"vllm batch HTTP {res.status_code}: {res.text[:300].strip()}",
            res.status_code)
    try:
        return res.json()
    except ValueError:
        raise vllmbatch.BatchError("the batch reply was not JSON", 502)


def _progress_swap(cb):
    """Install `cb` as this thread's progress hook. Returns the one it replaced."""
    previous = getattr(engine._progress, "cb", None)
    engine._progress.cb = cb
    return previous


def loop_aborts() -> dict:
    with engine._loop_aborts_lock:
        return dict(engine._loop_aborts)


def resolve_segments(text: str, pages=None, status: dict = None):
    """`(segments, how)`, without the page-by-page progress. See `_segment_stream`."""
    return engine._drain(engine._segment_stream(text, pages, status))


def _step_message(text: str, step: dict, codes=(), mode="agentic") -> str:
    """The message for one step: prefix, transcript, then this step's question.

    Assembled in that order on purpose. The transcript first is what makes the
    prefix shared across every step and therefore cacheable; the instructions last
    is what keeps an OCR fine-tune from answering with its own transcript envelope,
    which is the same thing EXTRACT_REMINDER buys the single-shot prompt.
    """
    return engine._step_prefix(text) + engine._step_question(step, codes, mode)


def _step_count_label(mode="agentic"):
    """How many agentic steps a run makes, as "7" or "7-17" across the forms.

    Over single types and pairs, because a real document names two about as often
    as one and a pair's union can need more steps than either alone.

    The top of the range is the UNCLASSIFIED form rather than any document: a
    page nothing could place is asked the union of every requirement, which is
    seventeen steps. Every real form is seven or nine.
    """
    codes = list(prompts.DOC_TYPE_FIELDS)
    combos = [[]] + [[c] for c in codes] + [[a, b] for a in codes for b in codes
                                            if a != b]
    counts = {len(engine.steps_for_types(c, mode=mode)) for c in combos}
    return (str(min(counts)) if len(counts) == 1
            else f"{min(counts)}-{max(counts)}")


def set_table_agents(fix=None, identify=None, concat=None) -> dict:
    global TABLE_FIX_AGENT, TABLE_IDENTIFY_AGENT, TABLE_CONCAT_AGENT
    with engine._table_agents_lock:
        mine = dict(workspace.get("table_agents", engine._agents_default()))
        for key, value in (("fix", fix), ("identify", identify), ("concat", concat)):
            if value is not None:
                mine[key] = bool(value)
        if not workspace.put("table_agents", mine):
            TABLE_FIX_AGENT, TABLE_IDENTIFY_AGENT = mine["fix"], mine["identify"]
            TABLE_CONCAT_AGENT = mine["concat"]
    return engine.table_agents()


def set_read_floor(fraction: float) -> float:
    """Clamped to 0-1 rather than refused: every value in range means something.

    0 scores every extraction whatever the read did, which is the shape for
    filling the log and letting the summary's own floor decide what to average.
    """
    pass  # this state lives in engine.py
    try:
        value = max(0.0, min(float(fraction), 1.0))
    except (TypeError, ValueError):
        raise ValueError("min_read_pct must be a number between 0 and 100.")
    with engine._read_floor_lock:
        if not workspace.put("read_floor", value):
            engine._read_floor = value
    return value


def set_extract_mode(mode: str) -> str:
    if mode not in ("single", "agentic", "hybrid"):
        raise ValueError("mode must be 'single', 'agentic' or 'hybrid'.")
    pass  # this state lives in engine.py
    with engine._mode_lock:
        if not workspace.put("extract_mode", mode):
            engine._extract_mode = mode
    return mode


def set_extract_extras(on: bool) -> bool:
    pass  # this state lives in engine.py
    with engine._mode_lock:
        if not workspace.put("extract_extras", bool(on)):
            engine._extract_extras = bool(on)
    return engine.extract_extras()


def extract_fields(text: str, mode: str = None, case_id: str = None,
                   steps=None, doc_type: str = None, pages=None,
                   truth_fed: bool = False, extras: bool = None,
                   each=None, schema_first: bool = False) -> dict:
    """`extract_fields_stream` for callers with nowhere to show progress.
    `each`, where given, sees every event (the stress test reads which document
    is being extracted off them)."""
    return engine._drain(engine.extract_fields_stream(text, mode, case_id, steps, doc_type,
                                        pages, truth_fed, extras, schema_first), each)


def read_all_pages(pages, page_texts: list, all_stats: list, cancel=None,
                   run: dict = None) -> None:
    """Read every page into the caller's lists, at once where that is allowed.

    The lists are the caller's so the pages that finished before an error or a
    Stop are still there to summarise. Sequential exactly as before where
    `_page_parallelism` says 1, stopping between pages on `cancel`.
    """
    parallel = engine._page_parallelism(len(pages))
    if parallel > 1:
        if stop_requested(cancel):
            return
        with closing(engine.read_pages_at_once(pages, parallel, cancel, run)) as reads:
            for _, text, stats in reads:
                page_texts.append(text)
                all_stats.append(stats)
        return
    for page in pages:
        if stop_requested(cancel):
            break
        stats = {}
        page_texts.append(engine.read_page(page, stats, cancel=cancel))
        all_stats.append(stats)


def _page_coverage(job_id, counted):
    """`{"pages_total": n, "pages_truncated": n}` for a read, or {}.

    Read back off the job rather than threaded through every summariser's
    arguments: `prepare_input` put it there and each of these already takes the
    job id. **Empty where the total is unknown** -- a re-extraction has no job,
    and a blank is not a claim that nothing was dropped.
    """
    total = (engine.job_context(job_id) or {}).get("pages_total") if job_id else None
    if not isinstance(total, int) or total <= 0:
        return {}
    return {"pages_total": total, "pages_truncated": max(0, total - counted)}


def summarise(all_stats, detail, started, job_id=None, run: dict = None):
    """Roll per-page stats into the totals both endpoints report.

    `run` is what `read_pages_at_once` left for a read whose pages were in
    flight together: how many at once, and the server's timing of the lot as
    one span, which stands in for the per-page figures those pages cannot have.
    """
    tokens = sum(s.get("new_tokens", 0) for s in all_stats)
    decode = sum(s.get("decode_seconds", 0) for s in all_stats)
    models = [s.get("model") for s in all_stats if s.get("model")]
    # Taken from the pages themselves, not from the active endpoint: a run that
    # finished must be reported against the server that actually ran it, even if
    # the page has since been pointed somewhere else.
    urls = [s.get("url") for s in all_stats if s.get("url")]
    kinds = [s.get("backend") for s in all_stats if s.get("backend")]
    profiles = [s.get("ocr_profile") for s in all_stats if s.get("ocr_profile")]
    # The pages the SERVER gave a time for. Each server figure is summed over
    # these alone, and only where every one of them reported it -- a total that
    # mixed pages with a phase and pages without would read as smaller than it is.
    served = [s for s in all_stats if s.get("server_seconds") is not None]
    images = [s.get("image_bytes") for s in all_stats]
    # Tokens sent and generated over the pages, whether or not the server timed
    # them; blank where any page reported none (servertime.io_tokens).
    page_in = [s.get("in_tokens") for s in all_stats]
    page_out = [s.get("out_tokens") for s in all_stats]

    return {
        "page_count": len(all_stats),
        **_page_coverage(job_id, len(all_stats)),
        "detail": detail,
        # From the pages, like model and backend above: a profile switched during
        # a batch must not relabel the pages that were already read under the old
        # one.
        "ocr_profile": profiles[0] if profiles else engine.ocr_profile(),
        "job": job_id,
        "model": models[0] if models else None,
        "url": urls[0] if urls else backends.active_url(),
        "backend": kinds[0] if kinds else None,
        # Which family read these pages. Stated rather than left to be derived
        # from `backend` downstream: `runlog.row_engine` can derive it and does
        # for every row written before the column, but a run that knows the
        # answer should say so -- a derivation is a fallback, not a design.
        "ocr_engine": "llm",
        "resolutions": [s.get("resolution") for s in all_stats],
        "tokens": tokens,
        "truncated": any(s.get("truncated") for s in all_stats),
        "looped": any(s.get("looped") for s in all_stats),
        # Reads thrown away for looping and redone a Detail up (`read_page`).
        "loop_retries": sum(len(s.get("loop_retries") or ()) for s in all_stats),
        "seconds": round(time.perf_counter() - started, 2),
        "prefill_seconds": round(sum(s.get("prefill_seconds", 0) for s in all_stats), 2),
        "decode_seconds": round(decode, 2),
        "tokens_per_second": round(tokens / decode, 2) if decode and tokens else 0,
        **servertime.summarise_records(served),
        # What was uploaded: the pages' base64 PNG payloads, summed. Blank where
        # a page did not record one.
        "image_bytes": (sum(images) if images and all(isinstance(b, int) for b in images)
                        else None),
        "in_tokens": (sum(page_in) if page_in and all(v is not None for v in page_in)
                      else None),
        "out_tokens": (sum(page_out) if page_out and all(v is not None for v in page_out)
                       else None),
        "server_timing": served[0].get("server_timing", "") if served else "",
        "server_timed_pages": len(served),
        **_concurrent_read(run, served),
        "page_stats": all_stats,
    }


def _concurrent_read(run: dict, served: list) -> dict:
    """`summarise`'s fields for a read whose pages went to the server at once.

    Empty for a sequential read. Otherwise `ocr_parallel`, and -- where no page
    carried a server time of its own and the batch could be attributed -- the
    batch's figures in place of the per-page sums, `server_timed_pages` being
    the requests it covers.
    """
    if not run or (run.get("parallel") or 1) <= 1:
        return {}
    out = {"ocr_parallel": run["parallel"]}
    server = run.get("server") or {}
    if not served and server.get("server_seconds") is not None:
        out.update({name: server.get(name) for name in servertime.FIELDS})
        out.update(server_timing=server.get("source") or "",
                   server_timed_pages=server.get("requests") or 0)
    return out


# Two concurrent runs of the same case would otherwise interleave their writes to
# solution/out/<id>.txt and leave a spliced transcript on disk.
_out_lock = threading.Lock()


def evaluate_if_known(case, text):
    """Score against ground truth when the input is a known benchmark document.

    Also writes the transcript to solution/out/ so `compare.py --no-run` can
    re-score exactly what the page displayed.
    """
    if not case:
        return None
    try:
        with _out_lock:
            scoring.OUT.mkdir(parents=True, exist_ok=True)
            (scoring.OUT / f"{case['id']}.txt").write_text(text, "utf-8")
        return scoring.evaluate(case, text)
    except Exception as err:  # scoring must never break a successful OCR run
        return {"error": f"scoring failed: {err}"}


def case_bytes(case_id: str):
    """Load a benchmark document from disk. Raises ValueError if unusable."""
    index = scoring.cases_index()
    if case_id not in index:
        raise ValueError(f"Unknown case '{case_id}'.")
    case = index[case_id]
    if not case["pdf_path"].exists():
        raise ValueError(f"Missing source PDF for {case_id}.")
    return case["pdf_path"].read_bytes(), case


def describe_source(name: str, data: bytes, origin: str) -> dict:
    """What the run log needs to identify the input: name, size and where from.

    Size is measured on the bytes actually read, so a case PDF and the same file
    uploaded by hand report identically.
    """
    return {"name": name, "size_bytes": len(data or b""), "origin": origin}


def request_input(request_files, form, match: bool = True):
    """The bytes a request is asking about, however the request named them.

    A benchmark case, a realData/mockOcr file or an upload resolves to
    one `(data, case, source)`. Shared by the read endpoints and by the preview,
    so a preview cannot be of a different document from the read that follows it.

    `match=False` skips the ground-truth lookup for callers that do not score. It
    hashes the upload, and the page asks `POST /api/match` for that separately.
    """
    case_id = (form.get("case") or "").strip()
    real_name = (form.get("realdata") or "").strip()
    mock_name = (form.get("file") or "").strip()
    if case_id:
        data, case = case_bytes(case_id)
        return data, case, describe_source(case["pdf"], data, "case")
    if real_name:
        data = resolve_real(real_name)
        return data, None, describe_source(real_name, data, "realdata")
    if mock_name:
        data, case = resolve_mock(mock_name)
        return data, case, describe_source(mock_name, data, "folder")
    upload = request_files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("No file uploaded.")
    data = upload.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        abort(413)
    source = describe_source(upload.filename, data, "upload")
    case = None
    if match:
        # Match on contents as well as name, so a renamed copy still scores.
        case, _how = scoring.case_for_upload(filename=upload.filename, data=data)
    return data, case, source


def prepare(request_files, form):
    """Request-shaped wrapper around `prepare_input`.

    Returns (pages, detail, page_job_id, case, source).
    """
    data, case, source = request_input(request_files, form)
    return (*engine.prepare_input(data, form.get("detail", DEFAULT_DETAIL), case, source),
            source)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------



_BENCHMARK_ASSETS = {
    **{name: config.BASE_DIR / 'static' / name for name in
       ('benchmark.js', 'benchmark.css', 'benchmark-performance.js', 'benchmark-results.js')},
    **{name: config.BASE_DIR / 'templates' / name for name in
       ('_stress_local_script.html', '_doctype_mapping_script.html', '_user_settings_script.html')},
}


@lru_cache(maxsize=32)
def _benchmark_version(path, mtime, size):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def benchmark_asset(name):
    path = _BENCHMARK_ASSETS[name]
    stat = path.stat()
    version = _benchmark_version(path, stat.st_mtime_ns, stat.st_size)
    return f'api/benchmark-assets/{version}/{name}'


@app.context_processor
def benchmark_asset_context():
    return {'benchmark_asset': benchmark_asset}


@app.get('/api/benchmark-assets/<version>/<name>')
def benchmark_asset_file(version, name):
    path = _BENCHMARK_ASSETS.get(name)
    if path is None:
        return jsonify(error='Asset not found.'), 404
    stat = path.stat()
    if version != _benchmark_version(path, stat.st_mtime_ns, stat.st_size):
        return jsonify(error='Asset version not found.'), 404
    response = send_from_directory(path.parent, path.name, max_age=31536000)
    response.mimetype = 'text/css' if name.endswith('.css') else 'text/javascript'
    response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get("/")
def index():
    response = app.make_response(render_template(
        "index.html",
        server=engine.llama_status(),
        details=list(engine.DETAIL_PRESETS),
        default_detail=DEFAULT_DETAIL,
        # `field_truth` says whether pass 2 can be *scored* on this document, as
        # opposed to merely run against it: the transcript truth and the field
        # truth are two different files, and the Fields-only pane is the one
        # place where having the first without the second is a live case.
        # doc_type groups the three case pickers into <optgroup>s, so "test the
        # invoices" is one glance rather than a reading of ten prose `kind`
        # lines. Blank sorts into its own group rather than being guessed at.
        cases=[case_payload(c) for c in scoring.cases_index().values()],
        endpoints=backends.endpoints(),
        # The engines the Server type pickers offer, besides "auto".
        server_kinds=[{"kind": k, "label": backends.SERVER_KIND_LABELS[k]}
                      for k in backends.SERVER_KINDS],
        # What pass 2 will run on, seeded at render so both extraction pickers
        # are right on first paint rather than only after a switch or Re-check.
        # Same block `/api/servers` returns, so the page has one shape to read.
        extract=backends.overview()["extract"],
        ctx_choices=backends.NUM_CTX_CHOICES,
        num_ctx=backends.num_ctx(),
        extract_mode=engine.extract_mode(),
        extract_extras=engine.extract_extras(),
        # A RANGE, not a count: the step table is per document type now, so an
        # invoice runs six of the eight steps and a credit note all eight. The
        # picker is drawn before any document is chosen, so the only honest
        # label is the span -- a flat "8 steps" beside a run that made six reads
        # as a run that lost two.
        extract_steps=_step_count_label(),
        hybrid_steps=_step_count_label("hybrid"),
        ocr_profile=engine.ocr_profile(),
        ocr_profiles=[{"id": pid, **spec} for pid, spec in engine.OCR_PROFILES.items()],
        # Whether a cycling read is cut short, and what it would cost if it were
        # not: the page prints the cap rather than repeating a number, for the
        # same reason the read floor below is sent instead of hardcoded.
        loop_guard=engine.loop_guard(),
        ocr_sampling=ocr_sampling_state(),
        table_agents=engine.table_agents(),
        max_new_tokens=engine.MAX_NEW_TOKENS,
        # When a field score is worth writing, for every path that reads a page.
        # Sent rather than hardcoded in the template: it is a setting, it moves
        # from two panes now, and a page that says 50% while the process runs at
        # 0 is describing a build nobody has. `read_floor()` rather than the
        # constant, so a reload shows what is in force and not what started.
        min_read_for_fields=engine.read_floor(),
        min_read_default=engine.MIN_READ_FOR_FIELDS,
        # The bar the Python classifier has to clear before the model is
        # not asked. Sent rather than repeated in the template: a page
        # quoting a threshold the process is not running describes a build
        # nobody has -- the same rule `min_read_for_fields` follows.
        min_classify_confidence=engine.CLASSIFY_MIN_CONFIDENCE,
        # Below this many characters of a script on a page, no rate is reported
        # for it -- three Latin characters score 0% or 100% and nothing between.
        # Sent for the same reason as the two above: the page explains a blank
        # cell by quoting the threshold, and a page quoting a number the process
        # is not running describes a build nobody has.
        script_min_chars=scoring.SCRIPT_MIN_CHARS,
        # The contest controls were removed from the page on 2026-09-09 at the
        # user's request, so nothing here describes them any more.
        # `randomtest.contest_plan` and `POST /api/randomtest` with
        # `{"contest": true}` are untouched: the planner is a pure function and
        # the CLI still drives it, which is where a rematch of a ranking
        # belongs -- it was never a thing to press mid-session.
    ))
    # Rendered from live state, so always checked, but a reload that changed
    # nothing costs a 304 rather than the page again.
    response.headers["Cache-Control"] = "no-cache"
    response.add_etag(weak=True)
    return response.make_conditional(request)


@app.get("/api/health")
def health():
    return jsonify(server=engine.llama_status(), pdf=PDF_OK, heif=HEIF_OK, status="ok")


@app.post("/api/context")
def context_set():
    """Set the context window sent with every later request.

    Deliberately not gated on a sweep: the window is a per-request field, so
    changing it mid-batch is a legitimate thing to do and the run log records
    what each row was read with. Jobs already in flight keep the window they
    started on.
    """
    body = request.get_json(silent=True) or {}
    try:
        value = backends.set_num_ctx(body.get("num_ctx"))
    except ValueError as err:
        return jsonify(error=str(err)), 400
    say(f"[ocr] context window set to {value} tokens")
    return jsonify(num_ctx=value, server=engine.llama_status())


@app.get("/api/servers")
def servers_list():
    """Configured endpoints, what each one is, and which is active."""
    force = request.args.get("probe") == "1"
    previous = backends.known() or {}
    previous_model = (backends._chosen_map().get(backends.active_url())
                      or previous.get("model"))
    overview = backends.overview(force=force)
    current = overview["server"]
    if (force and current.get("available") and current.get("model")
            and current["model"] != previous_model):
        engine.set_ocr_profile(backends.profile_for_model(current["model"]))
    response = jsonify(**overview, ocr_profile=engine.ocr_profile())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/servers")
def servers_select():
    """Switch the app to another model server, and optionally another model.

    Refused while a SWEEP is working -- a random test, which switches models
    between rounds: a batch half-read on one server and half on another would be
    logged and scored as if one server had done it. A single streaming read is
    not blocked and never was; the switch takes effect on the next run.
    """
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    model = (body.get("model") or "").strip()
    # Present and empty means "same as the reading model", which is a real
    # choice and not a missing field -- hence the sentinel rather than "".
    extract = body.get("extract_model")
    # The extraction SERVER (2026-10-01): absent leaves it alone, "" puts pass 2
    # back on the reading server, a URL gives it a server of its own. Same
    # null-vs-string sentinel as `extract_model`, for the same reason.
    extract_url = body.get("extract_url")
    if not url and not model and extract is None and extract_url is None:
        return jsonify(error="Give a url, a model, or both."), 400

    running = sweeps_running()
    if running and url and backends.clean_url(url) != backends.active_url():
        return jsonify(error=f"A run is in flight on {backends.active_url()}. "
                             "Wait for it or stop it before switching "
                             "server."), 409
    # The same rule for pass 2's server: a sweep half-extracted on one box and
    # half on another would be logged as if one had done it.
    if (running and extract_url is not None
            and (backends.clean_url(extract_url) or "")
            != backends.configured_extract_url()):
        return jsonify(error=f"A run is in flight extracting on "
                             f"{backends.extract_url()}. Wait for it or stop it "
                             "before switching the extraction server."), 409

    try:
        # Unloading is vetoed while a sweep is running. A model-only switch is
        # allowed mid-sweep (it takes effect on the next round), but the eviction
        # is a request to the same scheduler serving the read in flight, so a
        # sweep that is working keeps its weights and the next switch frees them.
        if url or model:
            server = backends.select(url or None, model or None, unload=not running)
        else:
            server = backends.status()
        # The extraction server before the extraction model, so a model named in
        # the same request is looked up on the server it is about to run on.
        if extract_url is not None:
            backends.select_extract_url(extract_url, unload=not running)
        # After the reading model, never before: the refusal in `select_extract`
        # is stated against whatever is reading the page, so it has to be asked
        # about the choice this request is making, not the one it replaced.
        if extract is not None:
            backends.select_extract(extract, unload=not running)
        # **The pass-1 profile follows the reading model.** The prompt and the
        # system-message veto are properties of the model rather than
        # preferences: a dots build given the typhoon profile returns an empty
        # transcript at HTTP 200 -- a run that logs as `ok`, scores 0.0%, and has
        # no other symptom. This project's rule was the opposite until
        # 2026-08-21, on the grounds that coupling them made "this prompt against
        # that model" unaskable; what changed is that a real run hit the failure.
        # The comparison is still askable -- POST /api/ocr/profile still sets
        # whatever you ask for, and it stands until the next model switch.
        if url or model:
            engine.set_ocr_profile(backends.profile_for_model(backends.status()["model"]))
    except ValueError as err:
        return jsonify(error=str(err)), 400

    # The pool is left alone. It used to be clamped to the new server's slot
    # count, which silently undid a concurrency setting chosen on purpose; the
    # page reports the mismatch instead so the choice stays the user's.
    # overview() carries the same fresh status under "server"; select() has just
    # primed the cache, so this does not re-probe.
    #
    # `unloaded` is on the response and not only in the console: a switch that
    # stopped a model and one that found nothing to stop look identical on the
    # page, and they leave the card in very different states.
    return jsonify({**backends.overview(), "unloaded": server.get("unloaded", []),
                    # Which pass-1 shape the new model brought with it, so the
                    # Page reading picker repaints instead of showing the one
                    # that was selected before the switch.
                    "ocr_profile": engine.ocr_profile()})


@app.post("/api/servers/kind")
def servers_kind():
    """Pin a server to one inference engine, or put it back on "auto".

    Body: `{"url": ..., "kind": "auto" | "llama.cpp" | "ollama" | "vllm" |
    "openai"}`, url defaulting to the active one. A pinned server is asked only
    that engine's requests -- see `backends.set_server_kind`.

    Refused with 409 mid-sweep for a server in use, the attribution rule: the
    pin decides how requests to it are built, so half a sweep under one engine
    and half under another would be logged as one.
    """
    body = request.get_json(silent=True) or {}
    url = backends.clean_url(body.get("url") or "") or backends.active_url()
    if sweeps_running() and url in (backends.active_url(), backends.extract_url()):
        return jsonify(error=f"A run is in flight on {url}. Wait for it or stop "
                             "it before changing its server type."), 409
    try:
        backends.set_server_kind(url, body.get("kind") or "auto")
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(backends.overview())


def case_payload(case: dict) -> dict:
    """One benchmark case, in the shape both the page and `/api/cases` read.

    **One function, because there were two copies of this and they had already
    drifted** (2026-09-08): the route grew a `documents` list for a file holding
    several documents and the template's copy did not, so the pickers went on
    describing a seven-document pack as one billing note. That is the same
    failure the template's field maps had on 2026-08-19, one layer out -- a fact
    stated twice is a fact that will disagree with itself.
    """
    codes = list(case.get("doc_types") or [])
    documents = case.get("documents") or []
    return {
        "id": case["id"], "pdf": case["pdf"], "kind": case.get("kind", ""),
        # The types the page names, and the one it is filed under. A document is
        # regularly two of them, so the list is the answer and `doc_type` is a
        # label taken off the front of it. **On a pack these are document 1's**
        # -- what the truth file describes and what the extractor is scored on.
        "doc_types": codes,
        "doc_type": normalise.primary_type(codes),
        # The form those types ask for. Sent per case so a caller -- and the
        # page -- can say what WILL be asked before anything runs, rather than
        # having to re-derive the union of the requirements client-side.
        "fields": list(fields_for_types(codes)),
        # Of that form, what the requirement DEMANDS. Beside `fields` rather
        # than derived from it on the page: which keys are Mandatory is a
        # property of the requirement, and a copy of that in JavaScript is how
        # the template's field maps went stale.
        "mandatory": list(mandatory_for_types(codes)),
        # No requirement covers this case's types, so its field score will be
        # taken over the base field set and nothing on it is Mandatory. Sent
        # rather than inferred from an empty `mandatory`, for the reason
        # `mandatory` itself is sent.
        "unknown_type": not mandatory_for_types(codes),
        # **What else is in the file.** Every field above describes the FIRST
        # document; a pack holds several, of several kinds. Without this a
        # picker offering `pack_billing_note_sol015.pdf` says "billing note, 30
        # fields" about a seven-page file holding seven documents of five kinds,
        # and the reader finds the other six only after paying for the run.
        #
        # Per document rather than a union of the types: the union of seven
        # documents' types is a list nothing on the page is, and the pages are
        # what make it readable as a file. EMPTY on an ordinary case, so the
        # front end draws nothing extra for one -- a "1 document" note over a
        # receipt is a control that says nothing.
        "documents": [{"pages": list(d.get("pages") or []),
                       "doc_types": list(d.get("doc_types") or []),
                       "doc_type": normalise.primary_type(d.get("doc_types") or [])}
                      for d in documents] if len(documents) > 1 else [],
        "pages": case.get("pages", 1),
        "available": case["pdf_path"].exists(),
        "field_truth": fieldscore.has_truth(case["id"]),
    }


@app.get("/api/cases")
def cases():
    """Benchmark documents that have a ground-truth transcript."""
    return jsonify(cases=[case_payload(c)
                      for c in scoring.cases_index().values()])


@app.get("/api/truth/<case_id>")
def truth_text(case_id):
    """The hand-written ground truth for one case, as Markdown.

    The page renders this through the same renderer as a transcript, so a
    pipe table in `solution/<id>.md` and an HTML table from the model are read
    side by side in the same shape -- which is the whole point of showing it.
    The text is served verbatim: normalising here would hide from the reader
    exactly the formatting that `scoring.normalise` forgives.

    `cases_index()` is keyed by id, so an id that is not a case 404s rather than
    reaching the filesystem.
    """
    case = scoring.cases_index().get(case_id)
    if not case:
        return jsonify(error=f"Unknown case '{case_id}'."), 404
    path = case["ground_truth"]
    try:
        text = path.read_text("utf-8")
    except OSError as err:
        return jsonify(error=f"Could not read {path.name}: {err}"), 500
    return jsonify(case=case["id"], pdf=case["pdf"], kind=case.get("kind", ""),
                   pages=case.get("pages", 1), file=path.name, text=text)


def _extract_input(body):
    """What one pass-2 request runs on: the text, the case, and the log context.

    Three shapes arrive here and they differ only in where the transcript comes
    from: a re-extraction of a page this process read (`job`), a script posting
    text of its own, and a fields-only run against a fixture's hand-written
    transcript (`case` + `from_truth`) -- which is `compare.py --from-truth` with
    a button on it, and the only one of the three that produces a field score
    without pass 1 having run at all.

    Feeding the ground truth is what makes pass 2 measurable on its own: every
    pass-2 measurement in CLAUDE.md was taken that way, because a wrong value is
    otherwise never attributable to extraction rather than to the read behind it.

    The case is what the field score is looked up by, so an explicit one wins
    over the job's -- a transcript can be pasted from anywhere, and the caller
    naming the document it belongs to is a stronger statement than a page cache
    that may since have evicted it.
    """
    text = body.get("text", "")
    case_id = (body.get("case") or "").strip()
    context = engine.job_context(body.get("job"))

    if case_id:
        case = scoring.cases_index().get(case_id)
        if not case:
            raise ValueError(f"Unknown case '{case_id}'.")
        if body.get("from_truth"):
            path = case["ground_truth"]
            try:
                text = path.read_text("utf-8")
            except OSError as err:
                raise ValueError(f"Could not read {path.name}: {err}")
            # Nothing about a read belongs on this row: no detail preset, because
            # no image was made, and the input named as the file actually fed in
            # rather than as the PDF nobody opened.
            context = {"case": case_id,
                       "source": describe_source(path.name,
                                                 text.encode("utf-8"), "truth")}
        else:
            context = {**context, "case": case_id}

    if not text.strip():
        raise ValueError("No text supplied.")
    return text, (case_id or context.get("case") or None), context


def _is_truth_fed(context) -> bool:
    """Whether this pass-2 request was handed the ground-truth transcript.

    Read off the log context `_extract_input` already builds rather than off the
    request body a second time: that function is the one place the three input
    shapes are told apart, and a second reading of `from_truth` here could
    disagree with the `source=truth` it writes on the row.

    It decides one thing and nothing else: whether `blame` is asked who lost a
    value. Fed the truth, the answer is `extraction` by construction -- see the
    module docstring -- and a tautology on screen is worse than a blank.
    """
    return ((context or {}).get("source") or {}).get("origin") == "truth"


def _requested_mode(body):
    """The extraction shape for one request: what it asked for, or the current one."""
    mode = (body.get("mode") or "").strip().lower() or None
    if mode and mode not in ("single", "agentic", "hybrid"):
        raise ValueError("mode must be 'single', 'agentic' or 'hybrid'.")
    return mode


def _requested_schema_first(body):
    """A per-request option; omitted and legacy callers use JSON first."""
    value = body.get("schema_first", False)
    if not isinstance(value, bool):
        raise ValueError("schema_first must be true or false.")
    return value


def _requested_steps(body):
    """Which agentic steps one request runs, where it asks for only some.

    A benchmark handle, not a setting: it is what makes a change to one step's
    prompt measurable over several documents and models without paying for the
    steps the change did not touch. Held per request rather than in process
    state, so nothing the page or a later round inherits it.

    A restricted result carries `steps_only`, which turns off the field score and
    fills the run log's `extract_steps` column -- both because the keys the run
    never asked for are not missing, they were not wanted.
    """
    steps = body.get("steps")
    if steps in (None, "", []):
        return None
    if isinstance(steps, str):
        steps = [s for s in steps.replace(",", " ").split() if s]
    if not isinstance(steps, list):
        raise ValueError("steps must be a list of step ids.")
    return steps


@app.post("/api/classify")
def classify_endpoint():
    """What kind of document this is, and the form that follows from it.

    The first stage of the pass-2 workflow, exposed on its own so it can be
    read, tested and swept without paying for an extraction: it is what
    `extract_fields` runs before it builds either prompt, through the same
    `_field_set`, so what comes back here is exactly what the extraction would
    be asked for.

    Takes the same input shapes as `/api/extract` -- a posted `text`, a `job`,
    or a `case` with `from_truth` -- plus the same `doc_type` override, which is
    answered straight back as `caller`. There is nothing to log: no model was
    asked anything unless rule 4 ran, and a row saying only "this is a receipt"
    would be a run in the log that read no page and extracted no field.
    """
    body = request.get_json(silent=True) or {}
    try:
        text, case_id, _ = _extract_input(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    # The extraction model's status, because rule 4 is a request to it. A model
    # server that is down is not an error here: the heading table answers
    # without one, and `_classify_with_model` is given None rather than a status
    # it cannot use.
    status = backends.extract_status()
    form = engine._field_set(text, case_id, body.get("doc_types") or body.get("doc_type"),
                      status if status.get("text_available") else None)
    return jsonify({
        **form,
        # What the answer buys, which is the whole reason to ask: the steps an
        # agentic run would walk, the rules `validate` would then apply, and the
        # framing both prompts would carry.
        "steps": [s["id"] for s in engine.steps_for_types(form["doc_types"],
                                                   form.get("extras", True))],
        "rules": list(rules_for_types(form["doc_types"])),
        "type_phrase": type_phrase(form["doc_types"]),
    })


@app.post("/api/extract")
def extract_endpoint():
    """Re-run field extraction on text, without re-reading the page.

    Blocking. Agentic mode is ~15 requests, so the page uses the streaming
    endpoint below and this stays for scripts, which have nowhere to show a step.

    An optional `job` -- the page id the `page`/`done` events carry -- says which
    document the transcript came from, so the run-log row can name it. Without
    one the row is still written, just anonymous.

    `case` names a benchmark document outright, and with `from_truth` the
    transcript is read from that case's `solution/<id>.md` instead of being
    posted: pass 2 alone, on text pass 1 cannot have spoiled.

    `steps` restricts an agentic run to the named steps -- one prompt measured on
    its own, over several documents and models, without paying for the six steps
    the change did not touch. See `_requested_steps`.
    """
    body = request.get_json(silent=True) or {}
    try:
        # Which document this transcript came from, so a re-extraction of a
        # benchmark case is scored the same way the read that produced it was.
        text, case_id, context = _extract_input(body)
        mode = _requested_mode(body)
        steps = _requested_steps(body)
        schema_first = _requested_schema_first(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    # `doc_type` accepts one code or a list -- a document is often more than
    # one type, and a caller overriding the classification has to be able to say
    # so. `doc_types` is the plural spelling of the same field.
    result = extract_fields(text, mode, case_id, steps,
                            doc_type=body.get("doc_types")
                            or body.get("doc_type"),
                            truth_fed=_is_truth_fed(context), schema_first=schema_first)
    engine.log_extract(result, body.get("job"), context)
    return jsonify(result)


@app.post("/api/extract/stream")
def extract_stream_endpoint():
    """The same, as NDJSON, so agentic mode can say which step it is on.

    Single mode emits no step events and one `fields` event, so the browser reads
    both modes the same way.
    """
    body = request.get_json(silent=True) or {}
    try:
        # The document behind this transcript, for the same reason as above.
        # Resolved out here rather than inside the generator: the job cache can
        # evict the page between the request arriving and the stream being
        # consumed, and a `from_truth` read that fails is a 400, not a stream
        # that opens and then apologises.
        text, case_id, context = _extract_input(body)
        mode = _requested_mode(body)
        steps = _requested_steps(body)
        schema_first = _requested_schema_first(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    job_id = body.get("job")

    def generate():
        try:
            stream = engine.extract_fields_stream(text, mode, case_id, steps,
                                           doc_type=body.get("doc_types")
                                           or body.get("doc_type"),
                                           truth_fed=_is_truth_fed(context),
                                           schema_first=schema_first)
            while True:
                try:
                    yield json.dumps(next(stream)) + "\n"
                except StopIteration as stop:
                    # Logged before the last event, so a page that refreshes the
                    # run log when the stream ends finds the row already there.
                    engine.log_extract(stop.value, job_id, context)
                    yield json.dumps({"event": "fields", **stop.value}) + "\n"
                    return
        except Exception as err:  # surface failures inside the stream body
            engine.log_extract({"error": str(err), "mode": mode or engine.extract_mode(),
                         "schema_first": schema_first and (mode or engine.extract_mode()) in ("agentic", "hybrid")},
                        job_id, context)
            yield json.dumps({"event": "error", "error": str(err)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/api/extract/mode")
def extract_mode_get():
    mode = engine.extract_mode()
    return jsonify(mode=mode, steps=(1 if mode == "single" else
                                    len(engine.steps_for_types([], mode=mode))))


@app.post("/api/extract/mode")
def extract_mode_set():
    """Switch the shape of pass 2 for everything this process extracts next.

    Unlike switching server, this is NOT refused while a sweep is running. Mixing
    the two within a batch costs nothing to interpret, because each result and
    each log row carries the mode that produced it -- what the server-switch 409
    protects against is a row that cannot say which server ran it, and there is no
    equivalent here.
    """
    body = request.get_json(silent=True) or {}
    try:
        mode = set_extract_mode((body.get("mode") or "").strip().lower())
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(mode=mode, steps=(1 if mode == "single" else
                                    len(engine.steps_for_types([], mode=mode))))


@app.get("/api/extract/extras")
def extract_extras_get():
    return jsonify(extras=engine.extract_extras())


@app.post("/api/extract/extras")
def extract_extras_set():
    """Ask for `other_fields` or not, for everything this tab extracts next.

    Not refused during a sweep, for the mode's reason: every result carries
    `extras`, so a batch split across the switch still says which ran how.
    """
    body = request.get_json(silent=True) or {}
    on = body.get("extras")
    if not isinstance(on, bool):
        return jsonify(error="extras must be true or false."), 400
    return jsonify(extras=set_extract_extras(on))


@app.get("/api/ocr/profile")
def ocr_profile_get():
    """The pass-1 shape in force, and every shape on offer."""
    return jsonify(profile=engine.ocr_profile(), profiles=[
        {"id": pid, "label": spec["label"], "note": spec["note"],
         "system": spec["system"], "reply": spec["reply"]}
        for pid, spec in engine.OCR_PROFILES.items()
    ])


@app.post("/api/ocr/profile")
def ocr_profile_set():
    """Switch the pass-1 shape for everything this process reads next.

    Not refused while a sweep is running, for the same reason switching extraction
    mode is not: every page stamps the profile it ran under onto its own stats, so
    a batch split across two profiles still says which read what. Switching
    *server* mid-batch is refused because nothing there could say so.
    """
    body = request.get_json(silent=True) or {}
    try:
        name = engine.set_ocr_profile((body.get("profile") or "").strip().lower())
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(profile=name)


@app.get("/api/ocr/loop-guard")
def loop_guard_get():
    """Whether a cycling read is cut short, and what turning that off costs."""
    return jsonify(loop_guard=engine.loop_guard(), max_tokens=engine.MAX_NEW_TOKENS)


@app.post("/api/ocr/loop-guard")
def loop_guard_set():
    """Turn the read backstop on or off for everything this process reads next.

    Not refused while a sweep is running, for the same reason the profile and the
    extraction mode are not: every page stamps the guard it ran under onto its own
    stats, so a batch split across the switch still says which page ran under
    which rule.
    """
    body = request.get_json(silent=True) or {}
    on = body.get("loop_guard")
    if not isinstance(on, bool):
        return jsonify(error="loop_guard must be true or false."), 400
    return jsonify(loop_guard=engine.set_loop_guard(on), max_tokens=engine.MAX_NEW_TOKENS)


@app.get("/api/ocr/sampling")
def ocr_sampling_get():
    """The page-read sampler in force: temperature, top_p, top_k (0 = not sent)."""
    return jsonify(**ocr_sampling_state())


@app.post("/api/ocr/sampling")
def ocr_sampling_set():
    """Change the page-read sampler for every read this process makes next.

    Pass 1 only; extraction stays greedy. Not refused during a sweep, for the loop
    guard's reason -- except that, unlike the guard, nothing is stamped on each
    run, so a sweep split across the change mixes samplers with no record. Said
    on the page. `reset: true` returns to what the process started at.
    """
    body = request.get_json(silent=True) or {}
    if body.get("reset") is True:
        d = ocr_sampling_state()["defaults"]
        body = {"temperature": d["temperature"], "top_p": d["top_p"], "top_k": d["top_k"]}
    try:
        state = set_ocr_sampling(body.get("temperature"), body.get("top_p"), body.get("top_k"))
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(**state)


@app.get("/api/table-agents")
def table_agents_get():
    """Whether the two table agents run on the next extraction."""
    return jsonify(**engine.table_agents())


@app.post("/api/table-agents")
def table_agents_set():
    """Turn either table agent on or off for every extraction this process runs next.

    Not refused during a sweep, for the loop guard's reason: every table records
    what each agent did on it (`fix_agent`, `identify_agent`, `master_from`), so a
    batch split across the switch still says which table ran under which rule.
    """
    body = request.get_json(silent=True) or {}
    fix, identify, concat = body.get("fix"), body.get("identify"), body.get("concat")
    for name, value in (("fix", fix), ("identify", identify), ("concat", concat)):
        if value is not None and not isinstance(value, bool):
            return jsonify(error=f"{name} must be true or false."), 400
    if fix is None and identify is None and concat is None:
        return jsonify(error="send concat, fix and/or identify."), 400
    return jsonify(**set_table_agents(fix, identify, concat))


@app.get("/api/ocr/read-floor")
def read_floor_get():
    """The transcript score a read must reach for its fields to be scored."""
    return jsonify(min_read_pct=round(engine.read_floor() * 100, 1),
                   default_pct=round(engine.MIN_READ_FOR_FIELDS * 100, 1))


@app.post("/api/ocr/read-floor")
def read_floor_set():
    """Move the floor for everything this process reads next.

    A PERCENTAGE on the wire and a fraction inside, the same units the summary's
    own floor uses -- the column it is compared with is a percentage and so is
    every control that sets it. Getting that backwards scores everything or
    nothing.

    Not refused while a sweep is running, for the same reason the profile and the
    extraction mode are not: the flag it decides is written onto each run as that
    run finishes, so a batch split across the switch still says which rule each
    row was written under. What it cannot do is go back -- a score this floor
    stopped being taken is not in the file to be recovered, which is why the
    summary keeps a floor of its own.
    """
    body = request.get_json(silent=True) or {}
    value = body.get("min_read_pct")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return jsonify(error="min_read_pct must be a number between 0 and 100."), 400
    try:
        floor = set_read_floor(float(value) / 100.0)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(min_read_pct=round(floor * 100, 1),
                   default_pct=round(engine.MIN_READ_FOR_FIELDS * 100, 1))


MOCK_DIR = scoring.MOCK


def mock_files():
    """Everything readable in mockOcr/, with its ground-truth case if it has one."""
    if not MOCK_DIR.exists():
        return []
    listing = []
    for path in sorted(MOCK_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ACCEPTED_SUFFIXES:
            continue
        case, how = scoring.case_for_upload(filename=path.name)
        listing.append({
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1024 ** 2, 2),
            "case": case["id"] if case else None,
            "matched_by": how,
        })
    return listing


def resolve_mock(name: str):
    """Read a file from mockOcr/ by name, refusing anything outside it."""
    if not MOCK_DIR.exists():
        raise ValueError("No document folder is configured on this server.")
    candidate = (MOCK_DIR / name).resolve()
    # Reject traversal: the resolved path must still sit inside mockOcr/.
    if MOCK_DIR.resolve() not in candidate.parents:
        raise ValueError("File is outside the mockOcr folder.")
    if not candidate.is_file():
        raise ValueError(f"No such file: {name}")
    if candidate.suffix.lower() not in ACCEPTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {candidate.suffix}")
    data = candidate.read_bytes()
    case, _how = scoring.case_for_upload(filename=candidate.name, data=data)
    return data, case


@app.get("/api/files")
def files_list():
    return jsonify(folder=str(MOCK_DIR), files=mock_files())


REAL_DIR = config.REAL_DIR


def real_files():
    """Everything readable under realData/, at any depth.

    Real documents, with no ground truth: they are read and extracted, never
    scored, so nothing here looks for a case. `name` is the path under the
    folder, which is what every other realData call takes.
    """
    if not REAL_DIR.exists():
        return []
    listing = []
    for path in sorted(REAL_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ACCEPTED_SUFFIXES:
            continue
        listing.append({
            "name": path.relative_to(REAL_DIR).as_posix(),
            "type": real_type(path.name),
            "size_mb": round(path.stat().st_size / 1024 ** 2, 2),
        })
    return listing


def real_type(name: str) -> str:
    """A realData file's type as the page filters on it: its suffix, `.jpeg` as `jpg`."""
    suffix = Path(name).suffix.lower().lstrip(".")
    return {"jpeg": "jpg", "tif": "tiff", "heif": "heic"}.get(suffix, suffix)


def resolve_real(name: str) -> bytes:
    """Read a file from realData/ by name, refusing anything outside it."""
    if not REAL_DIR.exists():
        raise ValueError("There is no realData folder on this server.")
    candidate = (REAL_DIR / name).resolve()
    if REAL_DIR.resolve() not in candidate.parents:
        raise ValueError("File is outside the realData folder.")
    if not candidate.is_file():
        raise ValueError(f"No such file in realData: {name}")
    if candidate.suffix.lower() not in ACCEPTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {candidate.suffix}")
    return candidate.read_bytes()


@app.get("/api/realdata")
def realdata_list():
    files = real_files()
    return jsonify(folder=str(REAL_DIR), exists=REAL_DIR.exists(), files=files,
                   types=sorted({f["type"] for f in files}))


@app.get("/api/realdata/file")
def realdata_file():
    """One realData file's bytes, so the page can load it like a dropped file."""
    name = request.args.get("name") or ""
    try:
        data = resolve_real(name)
    except ValueError as err:
        return jsonify(error=str(err)), 404
    kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return Response(data, mimetype=kind)


@app.post("/api/match")
def match_case():
    """Which ground-truth case does this file correspond to?

    Takes a filename and/or a sha256 the browser computed locally, so the page can
    show the mapping the moment a file is picked without uploading it first.
    """
    body = request.get_json(silent=True) or {}
    case, how = scoring.case_for_upload(
        filename=body.get("filename", ""), sha256=body.get("sha256", "")
    )
    if not case:
        return jsonify(case=None, matched_by=None)
    return jsonify(
        case=case["id"], pdf=case["pdf"], kind=case.get("kind", ""), matched_by=how
    )


@app.get("/api/page/<job_id>/<int:index>")
def page_image(job_id: str, index: int):
    """The prepared page image, for the side-by-side comparison view."""
    with engine._jobs_lock:
        pages = engine._jobs.get(job_id)
    if pages is None:
        return jsonify(error="That upload is no longer cached."), 404
    if not 0 <= index < len(pages):
        return jsonify(error="No such page."), 404
    return Response(
        pages[index],
        mimetype="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/api/preview")
def preview_prepared():
    """One page, prepared exactly as a read would prepare it. No model involved.

    A capped Detail is a real edit to the document: the page is trimmed and
    scaled down before it is ever sent, and every number a run produces is a
    measurement of THAT picture rather than of the file on disk. At 1 MP this
    project measured the model switching from misreading to *fabricating*, which
    is why that preset was deleted -- so being able to look at what the model is
    about to be given, before paying for the read, is worth a route.

    It goes through `trim_margins` and `fit_pixels`, the same two functions
    `prepare_input` calls, rather than letting the browser resize the file for
    display: a preview the read did not produce would show a page no run ever
    saw, which is the one thing a preview must not do.

    Deliberately does NOT `register_job`. `MAX_JOBS` is a RAM ceiling for the
    compare view (~40 MB for a 10-page document at `medium`), and flicking
    through the Detail picker would evict the pages of reads that have actually
    happened.

    Sizes ride in headers rather than in a JSON envelope so the body is the PNG
    itself and the page can hand it straight to a download link. **Nothing
    derived from a filename goes in a header** -- WSGI headers are latin-1 and
    the documents here have Thai names.
    """
    try:
        data, _case, _source = request_input(request.files, request.form, match=False)
        pages, page_total = engine.load_pages(data)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    detail = resolve_detail(request.form.get("detail", DEFAULT_DETAIL))
    try:
        index = int(request.form.get("page", 0))
    except (TypeError, ValueError):
        index = 0
    if not 0 <= index < len(pages):
        return jsonify(error="No such page."), 404

    page = pages[index]
    # Measured before the trim, so the caption can report the whole reduction --
    # for a PDF this is what PDF_DPI rasterised, not anything on disk.
    was = page.size
    prepared = engine.fit_pixels(engine.trim_margins(page) if engine.TRIM_MARGINS else page,
                          engine.DETAIL_PRESETS[detail])
    buf = io.BytesIO()
    prepared.save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png", headers={
        # The prepared page depends on the Detail asked for, and the picker moves
        # between requests to the same URL.
        "Cache-Control": "no-store",
        "X-Preview-Pages": str(len(pages)),
        # What the file HOLDS, against the pages above, which is what would be
        # read. The preview is the one place this can be said before an hour of
        # OCR is paid for, and it costs a header.
        "X-Preview-Pages-Total": str(page_total),
        "X-Preview-Max-Pages": str(engine.MAX_PAGES),
        "X-Preview-Page": str(index),
        # The RESOLVED name: an unknown preset falls back to the default rather
        # than raising, so the caption has to say what actually ran.
        "X-Preview-Detail": detail,
        "X-Preview-Width": str(prepared.width),
        "X-Preview-Height": str(prepared.height),
        "X-Preview-Source-Width": str(was[0]),
        "X-Preview-Source-Height": str(was[1]),
    })


def _random_pools():
    """What this endpoint can currently randomise over.

    Cases need a *transcript* truth to score pass 1 and a *field* truth to score
    pass 2; both are required here, because a round that can report neither
    number is a round that only proves the request did not crash.
    """
    cases = [c["id"] for c in scoring.cases_index().values()
             if fieldscore.has_truth(c["id"])]
    models = engine.llama_status()["models"]
    # Where pass 2 has a server of its own, the extractors are ITS models -- a
    # name drawn from the reading server would not be served where it is sent.
    extract_models = (backends.extract_status()["models"]
                      if backends.extract_separate() else None)
    return randomtest.pools(models, cases, extract_models=extract_models)


def _random_plan(body: dict) -> dict:
    """Turn a request into a plan, or raise ValueError with something readable.

    Two shapes, and they are opposite questions asked of the same runner:

    - the random test -- every axis drawn, to find what breaks;
    - `contest` -- the standouts ranking re-run, every axis pinned except the
      model, to find out whether that ranking was real.

    `strategy` says how the documents are chosen -- see `randomtest.STRATEGIES`.
    It reaches the random test only: a contest pins every axis but the model on
    purpose, so a sampling rule has nothing to decide there.

    The run log goes in as `history` (which the `balanced` strategy spreads the
    rounds against, and which every round reports whichever strategy chose it --
    see `randomtest.case_order`) and, for a contest, as the ranking itself.
    Both are read here rather than inside the planner so that it stays a pure
    function of its arguments: a plan that read a file could not be reproduced
    from a seed by anyone who did not have that file.
    """
    scope = body.get("scope") or randomtest.DEFAULT_SCOPE
    # Exclusions narrow the pools before anything is planned, so they hold for a
    # contest as well: "do not test that model" is a statement about the run, not
    # about one button on the pane.
    pools = randomtest.apply_exclusions(_random_pools(),
                                        body.get("exclude"), scope)
    # A lock and an exclusion naming the same model is a contradiction, and the
    # refusal it would otherwise get -- "not served here" -- would send someone
    # looking at their model server.
    excluded = {name for names in (body.get("exclude") or {}).values()
                for name in (names or [])}
    for axis, value in (body.get("lock") or {}).items():
        if value and value in excluded:
            raise ValueError(f"{value} is locked and excluded at the same time.")
    if body.get("contest"):
        # The whole standouts block goes in, not one ranking: the subject decides
        # which of the four it is contested on, and a `pipeline` contest needs
        # two of them at once. The scope picker does not apply here -- a contest's
        # scope is implied by its subject, because a reader contest that also
        # extracted would be scored partly on the other pass.
        return randomtest.contest_plan(
            standouts=runlog.standouts(), mode=engine.extract_mode(),
            subject=body.get("subject") or randomtest.DEFAULT_SUBJECT,
            documents=body.get("documents"),
            top=body.get("top", randomtest.CONTEST_TOP),
            bottom=body.get("bottom", randomtest.CONTEST_BOTTOM),
            seed=body.get("seed"), history=runlog.case_counts(), **pools)
    return randomtest.plan(
        rounds=body.get("rounds"), seed=body.get("seed"), scope=scope,
        details=list(engine.DETAIL_PRESETS), modes=["single", "agentic", "hybrid"],
        lock=body.get("lock"), strategy=body.get("strategy"),
        history=runlog.case_counts(), **pools)


@app.get("/api/randomtest")
@app.post("/api/randomtest")
def random_test_plan():
    """The plan a run *would* use, without running it.

    Separate from the stream so the page can show what it is about to do -- and
    so the CLI can learn the seed before it starts, which is what makes a run
    repeatable by someone who was not watching it.

    `scope` is how much of the pipeline each round runs -- `full`, `ocr` or
    `fields` (see `randomtest.SCOPES`). One choice for the whole plan, because
    the three answer different questions and a run that mixed them would report
    a mean over two of them.
    """
    body = request.get_json(silent=True) or {}
    try:
        planned = _random_plan(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    # `history` rides along so the page can say what it is balancing against:
    # a plan that gives one document three rounds and another none is correct
    # when the log already holds the other one, and unreadable without it.
    return jsonify({**planned,
                    # The pools the plan was drawn from, so the pane's chips
                    # describe what could have been chosen.
                    "pools": _random_pools(),
                    "history": runlog.case_counts(),
                    "scopes": list(randomtest.SCOPES),
                    "strategies": list(randomtest.STRATEGIES),
                    "max_rounds": randomtest.MAX_ROUNDS})


@app.post("/api/randomtest/stream")
def random_test_stream():
    """Run a plan, one NDJSON event per round.

    **Every round is a real run and is logged like one.** It switches the server
    the same way the page does, reads a real page and extracts from it, so the
    rows it writes are ordinary rows -- which is the point: a random test whose
    results were kept out of the run log would be measuring a path nobody else
    uses.

    A `fields` round logs the `run_type=extract` row a fields-only run always
    writes, and an `ocr` round logs a read whose pass-2 columns are blank -- both
    correct, and both what the setting tables already know how to read.

    The settings are left where the last round put them rather than restored.
    That is deliberate and is stated on the page: restoring them would hide
    which combination was in force when something broke, and the run log records
    per row what actually ran anyway. A `fields` run can therefore leave a
    text-only model selected, which the Workspace pane will refuse to read with
    until something else is picked -- the same visible-state rule, one pane over.
    """
    body = request.get_json(silent=True) or {}
    try:
        planned = _random_plan(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    # One sweep at a time, for the reason switching server is refused during one:
    # two of these interleaved would each be switching the process's model out
    # from under the other, and both would log rounds against a server the other
    # had just moved. The page disables its own button; this is the same rule for
    # a caller that is not the page.
    if sweeps_running():
        return jsonify(error="A random test or stress test is already running. "
                             "Wait for it or stop it before starting another."), 409

    cancel = begin_sweep()

    def generate():
        try:
            yield from rounds()
        finally:
            # The ordinary path: the rounds ran out, one raised, or Stop cut it
            # short. An abandoned stream never reaches this, and does not need
            # to -- see `sweeps_running`, which asks whether the thread is still
            # alive.
            end_sweep()

    def rounds():
        started = time.perf_counter()
        yield json.dumps({"event": "plan", **planned}) + "\n"
        completed = failed = 0
        stopped = ""
        # The finished rounds' summaries, for the run's pages/s and tokens.
        finished = []
        continuous = planned.get("continuous", False)
        total = None if continuous else len(planned["rounds"])
        for index, round_ in enumerate(randomtest.iter_rounds(planned), 1):
            # Checked before the round as well as inside it, so a stop during a
            # `fields` round -- which has no page to interrupt -- still costs one
            # round rather than the rest of the plan.
            if cancel.is_set():
                stopped = f"before round {index}"
                break
            began = time.perf_counter()
            event = {"event": "round", "index": index,
                     "total": total, "plan": round_}
            if continuous:
                yield json.dumps({**event, "event": "round_start"},
                                 ensure_ascii=False) + "\n"
            try:
                event["result"] = randomtest.summarise_round(
                    _run_round(round_, cancel))
                finished.append(event["result"])
                completed += 1
            except engine.SweepCancelled:
                # NOT a failed round, and the order of these two clauses is what
                # keeps it from becoming one. A round nobody let finish says
                # nothing about the settings it ran, and `failed` on the summary
                # line is read as if it did. Its partial row is already logged
                # `cancelled` by `_read_case`.
                stopped = f"during round {index}"
                break
            except Exception as err:                # noqa: BLE001
                # Reported as a failed round and the run continues. A random
                # test that stopped at the first failure would find one problem
                # per invocation, and finding several is the whole point.
                event["error"] = f"{type(err).__name__}: {err}"[:300]
                failed += 1
            event["seconds"] = round(time.perf_counter() - began, 1)
            yield json.dumps(event, ensure_ascii=False) + "\n"
        if stopped:
            yield json.dumps({"event": "stopped", "where": stopped,
                              "completed": completed,
                              "total": total}) + "\n"
        # `done` is emitted on a stop too, so the page has one event that ends a
        # run however it ended and writes its summary line once.
        yield json.dumps({"event": "done", "completed": completed,
                          "failed": failed, "total": total,
                          "continuous": continuous,
                          "stopped": bool(stopped),
                          "seconds": round(time.perf_counter() - started, 1),
                          "totals": randomtest.run_totals(finished),
                          "seed": planned["seed"]}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.post("/api/randomtest/stop")
def random_test_stop():
    """Ask the running sweep to stop.

    **Hanging up is not stopping**, which is what this route exists to fix: an
    abandoned stream runs every remaining round on this server, so the page's
    old Stop left the sweep reading and the next Run was refused by a 409 that
    was telling the truth. See `cancel_sweep`.

    It answers at once and the sweep ends slightly later -- it has to notice the
    flag, cut its read short and write that row. `stopping` says the request was
    delivered, not that the sweep is over; `running` is what says whether it is.
    Poll that, or wait for the stream's own `stopped` event, which is what the
    page does.

    Idempotent, and a stop with nothing running is a 200 saying so rather than
    an error -- a second click, or a Stop that lost a race with the last round,
    is not a problem anyone needs telling about.
    """
    stopping = cancel_sweep()
    return jsonify(stopping=stopping, running=sweeps_running())


def _run_round(round_: dict, cancel=None) -> dict:
    """One random-test round: put the settings in force, then run its scope.

    Three shapes, and the split is the reason the scopes exist at all:

    - `full`   -- read the page, extract from what came back. Both passes, and a
      field score that is therefore partly a measurement of pass 1.
    - `ocr`    -- read and stop. `extract=False` rather than an extraction whose
      result is thrown away: the run log would otherwise carry pass-2 columns
      for a run that was not testing pass 2.
    - `fields` -- no page, no image, no reader. The drawn model is selected as
      the ONE model in force and extraction is left pointing at it, which is the
      one-model setup `select_extract` never refuses -- so an OCR fine-tune can
      be measured on the form here, exactly as the Fields pane measures it.

    **A round always says which model extracts, and "" says "the reader".** That
    is what clears any choice left behind by an earlier round -- or by the
    startup default, which since `settings.AUTO_BEST_MODEL` may have selected a
    different model for pass 2. A plan that does not name an extractor is
    naming the reader, and it has to say so to the app rather than inherit.

    The pass-1 profile is set only where a page is read. Setting it on a
    fields-only round would leave the process on a profile chosen for a model
    that never looked at anything.

    **`cancel` reaches pass 1 and deliberately not pass 2.** A read is a loop
    over pages and over tokens, so it can be cut short at a known point and the
    partial thrown away. An extraction is one request, or seven in agentic mode,
    and unpicking it would mean a half-filled form that scores as a bad one --
    so a round already extracting runs that out. The bound is one round either
    way, which is the whole of what Stop was asked to buy.
    """
    scope = round_.get("scope", randomtest.DEFAULT_SCOPE)
    if scope == "fields":
        if backends.extract_separate():
            # Pass 2 has its own server, so the drawn model is chosen THERE; the
            # reading server is not touched by a round that reads no page.
            backends.select_extract(round_["extractor"], unload=True)
        else:
            backends.select(None, round_["extractor"], unload=True)
            backends.select_extract("", unload=False)
        set_extract_mode(round_["mode"])
        return _extract_case(round_["case"], round_["mode"])

    backends.select(None, round_["reader"], unload=True)
    backends.select_extract(round_["extractor"], unload=False)
    engine.set_ocr_profile(round_["profile"])
    if scope == "ocr":
        return _read_case(round_["case"], round_["detail"], extract=False,
                          cancel=cancel)
    set_extract_mode(round_["mode"])
    return _read_case(round_["case"], round_["detail"], cancel=cancel)


def _extract_case(case_id: str, mode: str) -> dict:
    """Pass 2 alone, on a document's hand-written transcript.

    The same thing the Fields pane's Run button does -- `_extract_input` with
    `from_truth`, so the transcript is read from `solution/<id>.md` server-side
    and pass 1 cannot have spoiled it. Every pass-2 measurement in CLAUDE.md was
    taken this way, and it is the only shape here whose field score is about the
    extractor and nothing else.

    Wrapped in the same envelope a read produces (`extracted`, `status`) so
    `randomtest.summarise_round` reads one shape whatever the round ran. There
    is deliberately no `truth` key: nothing read a page, and a transcript score
    of 100% against the text that was fed in would be a lie about what ran.
    """
    text, case, context = _extract_input({"case": case_id, "from_truth": True})
    result = extract_fields(text, mode, case, truth_fed=True)
    engine.log_extract(result, None, context)
    return {"extracted": result,
            "status": "error" if result.get("error") else "ok"}


def _read_case(case_id: str, detail: str, extract: bool = True,
               cancel=None) -> dict:
    """One benchmark document, read and extracted, exactly as `/api/ocr` does it.

    Shares `prepare_input`, `summarise`, `evaluate_if_known`, `extract_fields`
    and `log_run` with the route rather than reimplementing them: a test path
    that differs from the real one is a test of the test path.

    `extract=False` is the read-only scope: pass 2 does not run and the row's
    pass-2 columns stay blank, which is the honest record of a run that was
    testing the read.

    **`cancel` raises `SweepCancelled` rather than returning what it had**, and
    that is the decision worth keeping: a transcript cut off part-way scores as
    a terrible read, so returning one would put a made-up number in the mean and
    on the round table. The partial is thrown away and the row is written
    `cancelled` -- the same standing as the streaming route's Stop, and for the
    same reason: a read that died four minutes in is exactly what you want a
    record of, just not one that is averaged.
    """
    data, case = case_bytes(case_id)
    source = describe_source(case["pdf"], data, "case")
    pages, detail, job_id, case = engine.prepare_input(data, detail, case, source)

    started = time.perf_counter()
    page_texts, all_stats, run = [], [], {}
    cancelled = False
    try:
        read_all_pages(pages, page_texts, all_stats, cancel, run)
    except Exception:                               # noqa: BLE001
        # A hung-up request raises wherever it was blocked -- mid-prefill, most
        # of the time. If WE are the ones who hung up, that is the stop landing
        # and not a failure; anything else is a real error and is re-raised for
        # the round loop to report as one.
        if not stop_requested(cancel):
            raise
        cancelled = True

    if cancelled or stop_requested(cancel):
        # Summarised so the row carries the pages that did finish and the clock,
        # then logged and dropped. `evaluate_if_known` is deliberately not run:
        # scoring half a document against the whole ground truth would report a
        # stop as a bad read.
        partial = summarise(all_stats, detail, started, job_id, run)
        engine.log_run(partial, source, status="cancelled")
        raise engine.SweepCancelled(f"{case_id} was stopped part-way through")

    if len(page_texts) > 1:
        text = "\n\n".join(f"--- page {i} ---\n{t}"
                            for i, t in enumerate(page_texts, 1) if t)
    else:
        text = page_texts[0]

    payload = summarise(all_stats, detail, started, job_id, run)
    payload["truth"] = evaluate_if_known(case, text)
    if extract and EXTRACT and text.strip():
        payload["extracted"] = extract_fields(
            text, case_id=case["id"] if case else None, pages=page_texts)
    # The read floor is applied inside `log_run`, for every path rather than only
    # this one -- see `apply_read_floor`. Nothing is done here beyond letting it
    # happen, and the flag it may set is on `payload["extracted"]` by the time
    # this returns.
    engine.log_run(payload, source)
    # The same three-way status the run log derives, so a round that looped says
    # so on the page instead of reading as a clean run with a low score. Set
    # after `log_run`, which derives its own from the same two flags.
    payload["status"] = ("looped" if payload.get("looped")
                         else "truncated" if payload.get("truncated") else "ok")
    return payload


# --------------------------------------------------------------------------
# The stress test (2026-09-25)
#
# N requests in flight against the model server, from a queue of M jobs, and one
# report at the end. `stress.py` holds the dataset draw, the 100+ page pack
# builder, the Prometheus reader and the report; what is here is the part that
# talks to the server -- the same `read_page`, `extract_fields`,
# `_segment_stream` and `resolve_doc_types` every other path uses, so a stress
# run exercises the code a real run exercises.
#
# **What it deliberately does NOT share with the random test:**
#
# * **No per-job accuracy row is logged by default.** No run-log row, no `solution/out/`
#   transcript, no page job in the compare-view cache: every figure here was
#   taken under contention no other run was. **`save_log`** (2026-10-15, at the
#   user's request -- the stress test is where the volume of data is) writes one
#   ordinary row per measured job, tagged `stress_run` / `stress_concurrency`
#   so the `stress` filter can keep or drop them. Warm-up, cancelled and skipped
#   jobs are never written; classification has no row shape and is never
#   written. Still no transcript and no page job.
#   Aggregate settings and metrics are saved separately in History.
# * **No model is switched.** The whole run uses the reader and the extraction
#   model selected now. A stress test compares LOAD, and a model change between
#   jobs would be a second variable nobody could separate from the first.
# * **Pages are prepared once, before the clock.** Decode, trim, resize and the
#   PNG encode are this machine's CPU, not the server's; a thousand-job run
#   would otherwise be measuring Pillow. The time it took is reported beside the
#   report, not inside it.
#
# It holds the sweep guard like the random test, so the server cannot be
# switched and a random test cannot start underneath it, and its Stop is the
# same route: no new job starts, every read in flight is hung up (a
# `MultiCancel`), and an extraction already running is allowed to finish -- an
# extraction is one request, or seven, and a half-filled form scores as a bad
# one.
# --------------------------------------------------------------------------

STRESS_METRICS_TIMEOUT = (1.5, 5)
# A job that has finished no page and no request for this long is flagged as
# stalled, once -- a hung request or a server that stopped answering, which no
# other alert catches because nothing comes back to look at.
STRESS_STALL_SECONDS = 300
# How many realData files a stress run prepares at once. Preparing is a 300 DPI
# render, a resize and a PNG encode -- several times the page's final size in
# memory while it runs -- so fifty jobs each starting a different file at once
# must not all do it together.
STRESS_PREPARE_PARALLEL = config.env_int("STRESS_PREPARE_PARALLEL", 2, minimum=1)
# How many "without the N longest jobs" reports a stress run carries, for the
# page's Exclude picker (`stress.without_longest`). Each is a few KB.
STRESS_OUTLIER_MAX = config.env_int("STRESS_OUTLIER_MAX", 10, minimum=0)
# The concurrency sweep waits this long after a level's last job before its
# closing /metrics read, so a request vLLM records an engine step after it
# answered still lands in the level it ran in. Not on the level's clock.
SWEEP_SETTLE_SECONDS = 1.0
_STAGE_VERB = {"ocr": "reading", "extract": "extracting", "segment": "splitting",
               "classify": "classifying"}


class _StressLive:
    """What is in flight right now, by stage, for the progress events.

    Also the run's **alerts** (2026-10-15, at the user's request: an error or a
    loop is flagged the moment it happens, naming the document, rather than
    waiting for the report): a job that raised, a page or reply that looped or
    hit the token cap, a salvaged partial reply, a job that stalled. Each is
    queued here by the worker that saw it and sent by `_stress_phase` on its
    next tick, so it arrives within a second.

    **An alert is an issue with an `id`, not a one-off line** (2026-10-26, at the
    user's request: *if retry and resolve don't add another line, just mark the
    old one*). `update_flag` sends the same id again with its new state -- a
    looped page's row goes `retrying` -> `resolved` or `retry failed` -- and the
    page and the saved History replace the row by id rather than add one.
    """

    def __init__(self, queued: int, prefix: str = ""):
        self._lock = threading.Lock()
        self.stages = {name: 0 for name in stress.STAGES}
        self.stages["queued"] = queued
        self.done = self.failed = self.pages = self.tokens = 0
        # The running score (`score`), summed as jobs finish.
        self._score = {"char_sum": 0.0, "char_n": 0, "rate_sum": 0.0, "rate_n": 0,
                       "correct": 0.0, "expected": 0, "wrong": 0, "wrong_jobs": 0,
                       "docs": 0, "docs_extracted": 0, "files": 0}
        self._alerts = []
        self._issues = {}        # id -> the alert as it stands now
        self._next_id = 0
        # Alert ids are `<prefix><n>`, the prefix unique to the run and phase, so
        # an id names one page of one run -- what `_STRESS_EVIDENCE` keys on.
        self._prefix = prefix
        self._ocr_flagged = set()  # jobs with an OCR flag that opens a window
        self._active = {}        # job number -> {case, stage, last, flagged}
        # The run's `_StressSlots` where a long file may borrow idle workers
        # for its pages (`split_pages`), else None.
        self.slots = None

    def _new_id(self):
        self._next_id += 1
        return f"{self._prefix}{self._next_id}" if self._prefix else self._next_id

    def queued(self) -> int:
        with self._lock:
            return self.stages["queued"]

    def job_key(self, job: int) -> str:
        """Where a job's extraction is kept when an OCR flag of it has nothing
        else to switch to: `<prefix>job<n>`, which no alert id is. The page
        builds the same key from an alert's id."""
        return f"{self._prefix}job{job}"

    def ocr_flagged(self, job: int) -> bool:
        with self._lock:
            return job in self._ocr_flagged

    def flag(self, kind: str, job: int, case: str, message: str, **extra) -> int:
        """Raise an alert; returns its id, for `update_flag`."""
        with self._lock:
            alert = {"id": self._new_id(), "kind": kind, "job": job, "case": case,
                     "message": (message or kind)[:300],
                     "at": time.perf_counter(), **extra}
            if extra.get("evidence") and extra.get("stage") == "ocr":
                self._ocr_flagged.add(job)
            self._issues[alert["id"]] = alert
            self._alerts.append(dict(alert))
            return alert["id"]

    def update_flag(self, alert_id: int, message: str = None, **changes):
        """Change an alert already raised -- its state, its message -- and send
        it again under the same id. It keeps the time it was first raised."""
        with self._lock:
            alert = self._issues.get(alert_id)
            if alert is None:
                return
            if message is not None:
                alert["message"] = message[:300]
            alert.update(changes, updated_at=time.perf_counter())
            self._alerts.append(dict(alert))

    def take_alerts(self) -> list:
        with self._lock:
            out, self._alerts = self._alerts, []
            return out

    def begin(self, job: int, case: str, stage: str):
        with self._lock:
            self._active[job] = {"case": case, "stage": stage,
                                 "last": time.perf_counter(), "flagged": False,
                                 "document": None, "requests": {}, "reading": {}}

    def reading(self, job: int, number: int, page=None):
        """The page `job` is waiting on the server for (None once it is back),
        so a stalled flag can show the pages it stalled on."""
        with self._lock:
            active = self._active.get(job)
            if active:
                if page is None:
                    active["reading"].pop(number, None)
                else:
                    active["reading"][number] = page

    def touch(self, job: int, stage: str = None):
        with self._lock:
            active = self._active.get(job)
            if active:
                active["last"] = time.perf_counter()
                active["stage"] = stage or active["stage"]

    def end(self, job: int):
        with self._lock:
            self._active.pop(job, None)

    def hook(self, job: int):
        """This job's progress hook (`_progress`): every pass-2 request it makes,
        on whatever thread, resets its clock on each line and when it ends."""
        return lambda event, request, what: self.request(job, event, request, what)

    def request(self, job: int, event: str, request, what: str = None):
        """One request's begin, chunk or end. Beginning is not hearing back, so
        only a chunk or the end resets the clock."""
        with self._lock:
            active = self._active.get(job)
            if not active:
                return
            now = time.perf_counter()
            if event == "begin":
                label = (what if what and what.startswith('document ') else
                         ", ".join(p for p in (active["document"], what) if p))
                active["requests"][request] = {"what": label, "last": now}
                return
            active["last"] = now
            if event == "end":
                active["requests"].pop(request, None)
            elif request in active["requests"]:
                active["requests"][request]["last"] = now

    def extract_event(self, job: int, event: dict):
        """An extraction's progress event: which document it is on, if several."""
        if isinstance(event, dict) and event.get("event") == "document":
            with self._lock:
                active = self._active.get(job)
                if active:
                    active["document"] = (f"document {event.get('document')} of "
                                          f"{event.get('documents')}")

    def flag_stalled(self, limit: float):
        """Flag, once each, the jobs that have been silent for `limit` seconds.

        Silent means no request of the job has sent a line or finished -- not
        that the job as a whole has not finished. The alert names the requests
        still in flight, the quietest first.
        """
        now = time.perf_counter()
        with self._lock:
            for job, active in self._active.items():
                if active["stage"] == "extract_queued":
                    continue        # waiting for an extract worker, not a reply
                if not active["flagged"] and now - active["last"] >= limit:
                    active["flagged"] = True
                    verb = _STAGE_VERB.get(active["stage"], active["stage"])
                    quiet = sorted(active["requests"].values(),
                                   key=lambda r: r["last"])
                    names = [r["what"] or "a request" for r in quiet]
                    waiting = ""
                    if names:
                        more = f"; {len(names) - 3} more" if len(names) > 3 else ""
                        waiting = f" (waiting on {'; '.join(names[:3])}{more})"
                    alert = {
                        "id": self._new_id(),
                        "kind": "stalled", "job": job, "case": active["case"],
                        "stage": active["stage"], "at": now,
                        "message": f"nothing back for {now - active['last']:.0f} s "
                                   f"while {verb}{waiting}"
                                   " -- the request may be hung or the server "
                                   "stopped answering"}
                    if active["stage"] == "ocr":
                        # Every OCR flag opens a window (2026-10-27, at the
                        # user's request): here, the pages still out.
                        reads = [{"number": n, "text": "", "page": p,
                                  "detail": (getattr(p, "_stress_origin", None)
                                             or {}).get("detail"),
                                  "looped": False, "truncated": False, "dry": None,
                                  "note": "nothing back yet"}
                                 for n, p in sorted(active["reading"].items())]
                        if len(reads) == 1:
                            alert["page"] = reads[0]["number"]
                        alert["evidence"] = True
                        self._ocr_flagged.add(job)
                        _STRESS_EVIDENCE.put(
                            alert["id"], active["case"], alert.get("page"),
                            reads or [_note_read(alert["message"])], None)
                    self._alerts.append(alert)

    def move(self, frm, to):
        with self._lock:
            if frm:
                self.stages[frm] -= 1
            if to:
                self.stages[to] += 1

    def page(self, tokens: int):
        with self._lock:
            self.pages += 1
            self.tokens += tokens or 0

    def finish(self, failed: bool):
        with self._lock:
            self.done += 1
            self.failed += int(bool(failed))

    def score(self, rec: dict):
        """Add a finished job to the running score (2026-10-28, at the user's
        request: *if the file can be scored, report the score in real time*).

        The report's own populations, so the live figure lands where the report
        does: transcript and field scores over jobs that finished `ok` and were
        scored (not under the read floor); wrong fields and document sets over
        every job that reached them.
        """
        with self._lock:
            s = self._score
            ok = rec.get("status") == "ok"
            if ok and rec.get("char_accuracy") is not None:
                s["char_sum"] += rec["char_accuracy"]
                s["char_n"] += 1
            if ok and rec.get("field_expected") and not rec.get("unscored"):
                if rec.get("field_rate") is not None:
                    s["rate_sum"] += rec["field_rate"]
                    s["rate_n"] += 1
                s["correct"] += ((rec.get("field_correct") or 0)
                                 + 0.5 * (rec.get("field_partial") or 0))
                s["expected"] += rec["field_expected"]
            if rec.get("wrong_fields"):
                s["wrong"] += rec["wrong_fields"]
                s["wrong_jobs"] += 1
            if rec.get("doc_sets") is not None:
                s["files"] += 1
                s["docs"] += rec["doc_sets"]
                s["docs_extracted"] += rec["doc_sets"] - (rec.get("doc_sets_failed") or 0)

    def snapshot(self) -> dict:
        with self._lock:
            s = self._score
            score = {
                "char_accuracy": round(s["char_sum"] / s["char_n"], 4) if s["char_n"] else None,
                "char_scored": s["char_n"],
                "field_accuracy": round(s["rate_sum"] / s["rate_n"], 4) if s["rate_n"] else None,
                "field_pooled": round(s["correct"] / s["expected"], 4) if s["expected"] else None,
                "field_scored": s["rate_n"], "field_values": s["expected"],
                "wrong_fields": s["wrong"], "wrong_jobs": s["wrong_jobs"],
                "files": s["files"], "doc_sets": s["docs"],
                "doc_sets_extracted": s["docs_extracted"]}
            return {"stages": dict(self.stages), "done": self.done,
                    "failed": self.failed, "pages": self.pages,
                    "tokens": self.tokens, "score": score}


def _vllm_metrics(url: str) -> dict:
    """GET `/metrics` from a vLLM server, parsed. Raises on any failure."""
    return _vllm_scrape(url)[0]


def _vllm_scrape(url: str, buckets: bool = False) -> tuple:
    """GET `/metrics`: `(values, histogram buckets)` -- the buckets only when
    asked for (the concurrency sweep's percentiles), else `{}`. Raises on any failure."""
    res = backends.http_get(f"{url}/metrics", {"kind": "vllm"},
                            timeout=STRESS_METRICS_TIMEOUT)
    if res.status_code != 200:
        raise ValueError(f"/metrics answered HTTP {res.status_code}")
    return (stress.parse_prometheus(res.text),
            concurrencysweep.parse_buckets(res.text) if buckets else {})


def _stress_cases(mode: str) -> list:
    """The documents a stress run draws from -- the random test's pool.

    Every case with both truths for the modes that score fields, as
    `_random_pools` has it; the classification mode needs only the transcript
    truth, because its score is the grouping and the manifest's types.
    """
    index = scoring.cases_index()
    if mode == "classify":
        return sorted(cid for cid, c in index.items() if c["ground_truth"].exists())
    return sorted(cid for cid in index if fieldscore.has_truth(cid))


def _stress_config(body: dict) -> dict:
    """The request, checked and clamped, or ValueError naming what is wrong."""
    def number(key, default, low, high):
        try:
            value = int(body.get(key, default) if body.get(key) not in (None, "")
                        else default)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a whole number.")
        return max(low, min(value, high))

    mode = (body.get("mode") or stress.DEFAULT_MODE).strip().lower()
    if mode not in stress.MODES:
        raise ValueError(f"mode must be one of {', '.join(stress.MODES)}.")
    shape = (body.get("extract_mode") or engine.extract_mode()).strip().lower()
    if shape not in ("single", "agentic", "hybrid"):
        raise ValueError("extract_mode must be 'single', 'agentic' or 'hybrid'.")
    schema_first = (_requested_schema_first(body) and shape in ("agentic", "hybrid")
                    and mode in ("extract", "full", "image"))
    save_details = body.get("save_details", False)
    if not isinstance(save_details, bool):
        raise ValueError("save_details must be true or false.")
    strategy = (body.get("strategy") or randomtest.DEFAULT_STRATEGY).strip().lower()
    if strategy not in ("balanced", "uniform", "sequential"):
        raise ValueError("strategy must be 'balanced', 'uniform' or 'sequential'.")
    # `realdata`: the files in realData/, which have no ground truth -- so only
    # the modes that start from a read, and nothing they produce is scored.
    source = (body.get("source") or "benchmark").strip().lower()
    if source not in ("benchmark", "realdata"):
        raise ValueError("source must be 'benchmark' or 'realdata'.")
    if source == "realdata":
        if mode not in REAL_STRESS_MODES:
            raise ValueError("realData files have no ground truth, so only OCR only, "
                             "Full pipeline and Image to extractor can run on them.")
        cases = [f["name"] for f in real_files()]
        if not cases:
            raise ValueError(f"The realData folder ({REAL_DIR}) has no readable files.")
        if "types" in body:
            types = body["types"]
            if not isinstance(types, list) or any(not isinstance(t, str) for t in types):
                raise ValueError("types must be a list of file types, such as ['pdf', 'jpg'].")
            wanted = {real_type("x." + t.strip().lstrip(".")) for t in types}
            cases = [name for name in cases if real_type(name) in wanted]
            if not cases:
                raise ValueError("No realData file is of the chosen type(s).")
    else:
        cases = _stress_cases(mode)
    if "cases" in body:
        selected = body["cases"]
        if not isinstance(selected, list) or any(
                not isinstance(cid, str) for cid in selected):
            raise ValueError("cases must be a list of benchmark document IDs"
                             " or realData file names.")
        if not selected:
            raise ValueError("Select at least one file to stress test.")
        unknown = sorted(set(selected) - set(cases))
        if unknown:
            raise ValueError(f"{', '.join(unknown)} is not a document this mode can use.")
        cases = sorted(set(selected))
    lock = (body.get("lock_case") or "").strip()
    if lock and mode == "classify":
        lock = ""        # a pack of one case glued to itself has no findable boundaries
    if lock and lock not in cases:
        raise ValueError(f"{lock} is not among the selected documents this mode can use.")
    if not cases:
        raise ValueError("No benchmark case has the ground truth this mode needs.")

    used = []
    if mode in ("ocr", "full"):
        info = engine.llama_status()
        if not info["available"]:
            raise ValueError(info["reason"])
        used.append(info)
    sees = None
    if mode in ("extract", "classify", "full", "image"):
        info = backends.extract_status()
        if not info["text_available"]:
            raise ValueError(info["text_reason"])
        used.append(info)
        if mode == "image":
            # Refused only where the server SAYS the model has no vision. A
            # server that does not say (vLLM) is tried, and a model that cannot
            # take an image fails its first request -- an error the run flags.
            sees = _extractor_vision(info)
            if sees is False:
                raise ValueError(f"The extraction model {info.get('model') or ''} reports no "
                                 "vision, so it cannot be sent page images. Choose a vision "
                                 "model for extraction, or run Full pipeline.")
    # vLLM's /batch route (vllmbatch.py). A server that is not vLLM is sent
    # its requests one by one as always; a run where none of them is, is refused.
    vllm_batch = bool(body.get("vllm_batch"))
    if vllm_batch and not any(backends.serves_metrics(info) for info in used):
        raise ValueError("Batch requests need a vLLM server; this mode uses "
                         + " and ".join(f"{i.get('kind') or 'no server'} at {i.get('url')}"
                                        for i in used) + ".")
    # The concurrency sweep (2026-11-03): its figures ARE vLLM's /metrics, so a
    # mode whose servers are none of them vLLM has nothing to draw.
    sweep = None
    if body.get("sweep") not in (None, False):
        sweep = concurrencysweep.parse_levels(
            None if body["sweep"] is True else body["sweep"])
        if not any(backends.serves_metrics(info) for info in used):
            raise ValueError("The concurrency sweep reads vLLM's /metrics; this mode uses "
                             + " and ".join(f"{i.get('kind') or 'no server'} at {i.get('url')}"
                                            for i in used) + ".")
    # Every level runs the same jobs (the same files, in the same order), so
    # only the concurrency differs between them; blank is the highest level.
    per_level = concurrencysweep.level_jobs(sweep, body.get("sweep_jobs")) if sweep else 0
    if sweep and per_level * len(sweep) > stress.MAX_JOBS:
        raise ValueError(f"The sweep would queue {per_level * len(sweep)} jobs ({per_level} at "
                         f"each of {len(sweep)} levels), over the {stress.MAX_JOBS} ceiling; "
                         "use fewer levels or fewer jobs per level.")

    return {
        "mode": mode,
        # A sweep's pools and batcher are sized for its highest level.
        "concurrency": (max(sweep) if sweep
                        else number("concurrency", 4, 1, stress.MAX_CONCURRENCY)),
        "sweep": sweep,
        "sweep_jobs": per_level,
        # A sweep's queue is per level; `jobs` is their sum.
        "jobs": (per_level * len(sweep) if sweep
                 else number("jobs", 20, 1, stress.MAX_JOBS)),
        "warmup": number("warmup", 0, 0, stress.MAX_WARMUP),
        "max_seconds": number("max_seconds", 0, 0, 7 * 24 * 3600),
        "pack_pages": number("pack_pages", stress.DEFAULT_PACK_PAGES, 2,
                             stress.MAX_PACK_PAGES),
        "metrics_every": number("metrics_every", 5, 1, 600),
        "metrics": bool(body.get("metrics")) or bool(sweep),
        "no_cache": bool(body.get("no_cache")),
        # "Use DRY": on unless the request says otherwise; off sends every
        # request of the run without DRY (`backends.suppress_dry`).
        "dry": body.get("dry", True) is not False,
        # A long file reads its pages on workers left idle once the queue is
        # empty (`_StressSlots`). On unless the request says otherwise.
        "split_pages": body.get("split_pages", True) is not False,
        # Full pipeline only: a Mandatory field left empty is asked of the page
        # images with its Hybrid step (`_stress_fill_missing`). On unless the
        # request says otherwise; the page's toggle sends it.
        "fill_missing": body.get("fill_missing", True) is not False and mode == "full",
        "validation_retry": body.get("validation_retry") is True and mode == "full",
        "vllm_batch": vllm_batch,
        # Classification has no row shape: a glued pack is no benchmark case and
        # its score is a grouping, so nothing is written for it.
        # Nor has an image-fed job: a run-log row is a read and the extraction
        # of its transcript, and its grounded ratio is over that transcript.
        "save_log": bool(body.get("save_log")) and mode not in ("classify", "image"),
        "save_details": save_details and mode in ("extract", "full", "image"),
        "extractor_vision": sees,
        "history_monitor": bool(body.get("history_monitor", True)),
        "gpu_url": str(body.get("gpu_url") or "").strip() or MONITOR_GPU_URL,
        "gpu_extract_url": str(body.get("gpu_extract_url") or "").strip() or MONITOR_GPU_URL,
        "detail": resolve_detail(body.get("detail") or "medium"),
        "extract_mode": shape,
        "schema_first": schema_first,
        # Whether `other_fields` is asked for, fixed for the run like the shape.
        "extras": (body["extras"] if isinstance(body.get("extras"), bool)
                   else engine.extract_extras()),
        "strategy": strategy,
        "seed": body.get("seed"),
        "lock": lock,
        "source": source,
        "cases": cases,
    }


# The stress modes a realData file can run in: they start from the page. Extract
# and classify start from a ground-truth transcript, which realData has none of.
REAL_STRESS_MODES = ("ocr", "full", "image")


def _extractor_vision(info: dict):
    """Whether the extraction model takes images: True, False, or None where its
    server does not say. Read off the model's own entry, not off `info["vision"]`
    -- an extraction status describes the READER's vision, or none at all."""
    model = info.get("model")
    entry = next((m for m in info.get("models") or []
                  if model and backends._same_model(m.get("name") or "", model)), None)
    if entry is None or entry.get("vision") is None:
        return None
    return bool(entry["vision"])


def _stress_pages(case_id: str, detail: str, source: str = "benchmark") -> tuple:
    """A case's pages, prepared exactly as `prepare_input` does it, encoded once.

    Not `prepare_input` itself, because that registers a page job -- and a
    stress run's pages in the compare-view cache would evict the reads somebody
    is looking at, `MAX_JOBS` being a RAM ceiling. Returns `(pages, source)`,
    the source being what a saved row names the input as. With `source`
    `realdata`, `case_id` is a file name under realData/.
    """
    if source == "realdata":
        data = resolve_real(case_id)
        name, origin = case_id, "realdata"
    else:
        data, case = case_bytes(case_id)
        name, origin = case["pdf"], "case"
    pages, _ = engine.load_pages(data)
    budget = engine.DETAIL_PRESETS[detail]
    prepared = [engine.fit_pixels(engine.trim_margins(page) if engine.TRIM_MARGINS else page, budget)
                for page in pages]
    for index, page in enumerate(prepared):
        page._data_uri = engine.image_data_uri(page)
        # Where a looped read climbs from (`_stress_retry_page`). Only the
        # address is kept, not the trimmed page: holding every page of the run
        # at full resolution for the few that loop would multiply its memory.
        page._stress_origin = {"case": case_id, "source": source,
                               "index": index, "detail": detail}
        engine.tag_page(page, f"{case_id} p{index + 1}/{len(prepared)}", detail)
    return prepared, describe_source(name, data, origin)


def _one_page(data: bytes, index: int):
    """Page `index` of a file, rasterised alone where it is a PDF."""
    if engine._looks_like_pdf(data) and PDF_OK:
        zoom = engine.PDF_DPI / 72.0
        with engine._PDF_LOCK, fitz.open(stream=data, filetype="pdf") as doc:
            if index >= doc.page_count:
                return None
            pix = doc[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pages, _ = engine.load_pages(data)
    return pages[index] if index < len(pages) else None


def _stress_retry_page(page):
    """A stress page to read again after it looped, one Detail up -- or None.

    The stress twin of `loop_retry_page`, which needs a page `prepare_input`
    tagged: stress pages are prepared by `_stress_pages`, so the trimmed source
    is rebuilt from the file here, only for the page that looped. None when
    LOOP_RETRY is off, the page is at `original` already, or the next budget
    gives back the same image.
    """
    origin = getattr(page, "_stress_origin", None)
    if not (engine.LOOP_RETRY and origin):
        return None
    higher = next_detail(origin["detail"])
    if higher is None:
        return None
    data = (resolve_real(origin["case"]) if origin["source"] == "realdata"
            else case_bytes(origin["case"])[0])
    source = _one_page(data, origin["index"])
    if source is None:
        return None
    if engine.TRIM_MARGINS:
        source = engine.trim_margins(source)
    bigger = engine.fit_pixels(source, engine.DETAIL_PRESETS[higher])
    if bigger.size == page.size:
        return None
    bigger._data_uri = engine.image_data_uri(bigger)
    bigger._stress_origin = dict(origin, detail=higher)
    if hasattr(page, "_dry_label"):
        engine.tag_page(bigger, page._dry_label, higher)
    return bigger


# --------------------------------------------------------------------------
# Image mode (2026-10-27, at the user's request: *a mode that can send the image
# directly to extractor if they can handle image ... keep all the feature from
# the ocr and extraction like if error allow me to see it or retry if the field
# is missing*). No page is read. Each job sends its prepared pages to the
# extraction model with the run's shape and extras (`_feeding_images`), and its
# flags open the pages beside the form, as a full run's do. Where the form comes
# back with a Mandatory field empty, or the reply loops or hits the cap, the
# pages are rebuilt one Detail up and extracted once more -- the climb a looped
# read takes (`_stress_retry_page`, under LOOP_RETRY) -- and the flag's one row
# goes `retrying` -> `resolved` or `retry failed`.
# --------------------------------------------------------------------------

def _image_reads(pages, note: str) -> list:
    """Evidence entries for pages sent to the extractor: the page, no text."""
    return [{"number": n, "text": "", "page": page,
             "detail": (getattr(page, "_stress_origin", None) or {}).get("detail"),
             "looped": False, "truncated": False, "dry": None, "note": note[:300]}
            for n, page in enumerate(pages, 1)]


def _image_extract_once(pages, cfg: dict, case_id, live: _StressLive, job: int) -> dict:
    """One image-fed extraction of `pages`, with the run's shape and extras.

    The file is one document: segmentation reads page text, and there is none.
    `case_id` still decides the form where the classifier cannot, and scores it.
    """
    text = prompts.IMAGE_TRANSCRIPT.format(pages=len(pages))
    with engine._feeding_images([getattr(p, "_data_uri", None) or engine.image_data_uri(p)
                          for p in pages]):
        return extract_fields(text, cfg["extract_mode"], case_id, pages=[text],
                              extras=cfg.get("extras"),
                              schema_first=cfg.get("schema_first", False),
                              each=lambda event: live.extract_event(job, event))


def _image_cost(result: dict) -> tuple:
    """What makes one image-fed form worse than another: an error, then gaps."""
    return (bool(result.get("error")),
            sum(len(keys) for _, keys in engine._missing_required(result)))


def _stress_image_extract(pages, cfg: dict, case_id, rec: dict, live: _StressLive,
                          job: int, who: str, window, cancel) -> tuple:
    """Image mode's extraction of one job: `(result, reads, alert)`.

    `reads` is what a flag's window shows -- each page sent, both attempts' where
    there was a retry. `alert` is the flag this raised and has kept up to date,
    or None where the first form came back whole; the caller raises no second
    flag for the same problem. `window` is the job's `fields_window`.

    The kept form is the better of the two (no error, then fewest gaps; the
    retry on a tie), and its tokens are both attempts': the server did both.
    """
    detail = cfg["detail"]
    result = _image_extract_once(pages, cfg, case_id, live, job)
    reads = _image_reads(pages, f"sent to the extractor at {detail}")
    kind, what, fields = engine._image_problem(result)
    if kind is None:
        return result, reads, None
    extra = {"fields": fields} if fields else {}
    began = time.perf_counter()
    bigger = ([_stress_retry_page(p) for p in pages]
              if engine.LOOP_RETRY and not cancel.is_set() else [])
    rec["retry_prepare_s"] = round(time.perf_counter() - began, 3)
    higher = next((b._stress_origin["detail"] for b in bigger if b is not None), None)
    if higher is None:
        why = ("the run was stopped" if cancel.is_set() else "LOOP_RETRY is off"
               if not engine.LOOP_RETRY else "no higher Detail gives bigger pages")
        alert = live.flag(kind, job, who, f"{what} at {detail} -- no retry: {why}",
                          stage="extract", state="no retry", evidence=True, **extra)
        window(alert, result, engine._missing_required(result))
        return result, reads, alert
    alert = live.flag(kind, job, who, f"{what} at {detail} -- extracting again at {higher}",
                      stage="extract", state="retrying", evidence=True, **extra)
    window(alert, result, engine._missing_required(result))
    rec["image_retries"] = 1
    live.touch(job)
    retried = [b if b is not None else p for b, p in zip(bigger, pages)]
    second = _image_extract_once(retried, cfg, case_id, live, job)
    kind2, what2, _ = engine._image_problem(second)
    if second.get("error") and kind2 is None:
        what2 = "the request failed: " + str(second["error"])[:160]
    reads = (_image_reads(pages, f"attempt 1 at {detail}: {what}")
             + _image_reads(retried, f"attempt 2 at {higher}: " + (what2 or "nothing missing")))
    took_retry = _image_cost(second) <= _image_cost(result)
    kept = second if took_retry else result
    rec["image_detail"] = higher if took_retry else detail
    if not what2:
        rec["image_recovered"] = 1
        live.update_flag(alert, f"{what} at {detail} -- resolved: extracted again at "
                                f"{higher} with nothing missing", state="resolved")
    else:
        live.update_flag(alert, f"{what} at {detail}; again at {higher}, {what2} -- kept "
                                f"the {rec['image_detail']} form",
                         state="retry failed")
    kept = dict(kept)
    kept["tokens"] = (result.get("tokens") or 0) + (second.get("tokens") or 0)
    served = [r.get("server_time") or {} for r in (result, second)]
    if any(served):
        kept["server_time"] = dict(kept.get("server_time") or {})
        for key in ("in_tokens", "out_tokens"):
            values = [s.get(key) for s in served if s.get(key) is not None]
            kept["server_time"][key] = sum(values) if values else None
    return kept, reads, alert


def _recheck_events(result, images, texts, fill_missing, validation_retry, case_id):
    """`recheck_fields_from_images`, with each flag streamed as it is raised or updated.

    The rechecks run on a thread so the stream can say which of the two is asking
    the images and how it ended, rather than one spinner for both. The callback
    only queues; the generator does the yielding.
    """
    if not engine.image_rechecks_needed(result, fill_missing, validation_retry):
        engine.recheck_fields_from_images(result, images, texts, fill_missing=fill_missing,
                                   validation_retry=validation_retry, case_id=case_id)
        return
    events, done = queue.Queue(), object()

    def work():
        try:
            engine.recheck_fields_from_images(result, images, texts, fill_missing=fill_missing,
                                       validation_retry=validation_retry, case_id=case_id,
                                       each=lambda e: events.put(copy.deepcopy(e)))
        finally:
            events.put(done)

    # Bound to this request's workspace: the model, extraction mode and agents
    # the recheck asks with are per-workspace settings.
    threading.Thread(target=workspace.bind(work), daemon=True).start()
    yield json.dumps({"event": "image_recheck", "state": "starting",
                      "message": "Rechecking fields from the page images"}) + "\n"
    while (event := events.get()) is not done:
        yield json.dumps(event) + "\n"


class _StressPages:
    """realData pages for a stress run, prepared when a job first needs them and
    dropped when the last job drawn for that file has finished.

    The benchmark prepares every case up front, which is fine for 23 small
    fixtures. realData can be hundreds of real scans: preparing all of them
    before the first job held every page of every file -- the image and its PNG
    -- for the whole run, and sent nothing down the stream while it did. At
    concurrency 50 over 300 jobs that ran the process out of memory or let the
    connection idle out, and the page saw a network error. Now at most the files
    in flight are held, and the stream ticks from the start.

    Fills `prepared["pages"]` and `prepared["sources"]`, which the job reads as
    it always has.
    """

    def __init__(self, specs: list, detail: str, prepared: dict):
        self.detail = detail
        self.prepared = prepared
        self.uses = {}
        for spec in specs:
            self.uses[spec["case"]] = self.uses.get(spec["case"], 0) + 1
        self.lock = threading.Lock()
        self.locks = {}
        self.slots = threading.Semaphore(STRESS_PREPARE_PARALLEL)

    def take(self, case_id: str) -> float:
        """Make sure `case_id` is prepared. Returns the seconds this call spent."""
        started = time.perf_counter()
        with self.lock:
            lock = self.locks.setdefault(case_id, threading.Lock())
        # One job prepares a file; the others drawn for it wait and share it.
        with lock:
            if case_id not in self.prepared["pages"]:
                # Named before it is read, so a file that will not open is
                # still saved as the failure it was.
                self.prepared["sources"].setdefault(
                    case_id, describe_source(case_id, b"", "realdata"))
                with self.slots:
                    pages, source = _stress_pages(case_id, self.detail, "realdata")
                self.prepared["pages"][case_id] = pages
                self.prepared["sources"][case_id] = source
        return time.perf_counter() - started

    def release(self, case_id: str) -> None:
        """One job drawn for `case_id` is done; the last one frees its pages."""
        with self.lock:
            self.uses[case_id] = self.uses.get(case_id, 1) - 1
            if self.uses[case_id] <= 0:
                self.prepared["pages"].pop(case_id, None)


class _StressSlots:
    """A stress run's request slots: one per worker of its concurrency.

    A job holds one for as long as it runs, so with every worker busy nothing is
    left to lend. Once the queue is empty, a worker that finishes has no job to
    pick up and its slot sits free -- and a long file still reading its pages
    one at a time is the only thing left (2026-10-26, at the user's request: a
    100-page file read alone for a long time while every other worker was
    idle). `lend` hands such a slot to one of that file's pages, which gives it
    back when the page is read. So the run never has more page requests in
    flight than its concurrency, and the idle capacity is used.
    """

    def __init__(self, n: int):
        self._sem = threading.Semaphore(n)

    def hold(self):
        self._sem.acquire()

    def release(self):
        self._sem.release()

    def lend(self) -> bool:
        return self._sem.acquire(blocking=False)


# How many flagged pages the stress test keeps for the Compare view. Each holds
# every attempt's prepared page (a PNG of up to a few MB) and its text.
STRESS_EVIDENCE_MAX = config.env_int("STRESS_EVIDENCE_MAX", 100, minimum=0)


class _StressEvidence:
    """What a flagged stress page actually was: each attempt's page and text.

    **Built 2026-10-26 at the user's request** -- *on the page that got DRY flag
    or loop let me see the output: I click that log and the compare below shows
    the original of that page and the text that got the error.* Keyed by the
    alert's id; a loop alert and a DRY alert on the same page share one list of
    attempts, which the read keeps appending to while it retries. Held in memory,
    newest `STRESS_EVIDENCE_MAX` pages only, so a saved run reopened after a
    restart, or long after, has its rows but not their pages.
    """

    def __init__(self, limit: int):
        self._lock = threading.Lock()
        self._limit = limit
        self._items = OrderedDict()

    def put(self, alert_id, case: str, page: int, attempts: list, truth_case,
            **extra):
        """`extra` is what a kind of alert adds -- a missing-required flag's
        `fields` (`_fields_evidence`), the form beside the pages it was read off."""
        with self._lock:
            self._items[str(alert_id)] = {"case": case, "page": page,
                                          "attempts": attempts, "truth_case": truth_case,
                                          **extra}
            self._items.move_to_end(str(alert_id))
            while len(self._items) > self._limit:
                self._items.popitem(last=False)

    def get(self, alert_id):
        with self._lock:
            item = self._items.get(str(alert_id))
            return None if item is None else dict(item, attempts=list(item["attempts"]))


_STRESS_EVIDENCE = _StressEvidence(STRESS_EVIDENCE_MAX)


def _note_read(note: str, page=None, detail=None) -> dict:
    """A read with no text, only what happened to it -- for a flag whose page
    failed or never came back. `page` is None where no page was read."""
    return {"detail": detail, "text": "", "page": page, "looped": False,
            "truncated": False, "dry": None, "note": note[:300]}


def _page_png(page) -> bytes:
    """The PNG the model was sent for a stress page."""
    uri = getattr(page, "_data_uri", "") or ""
    if uri.startswith("data:image/png;base64,"):
        return base64.b64decode(uri.split(",", 1)[1])
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    return buf.getvalue()


def _at_details(details) -> str:
    """` at low, medium` -- the Details a page looped at, for its alert."""
    named = [d for d in details if d]
    return f" at {', '.join(named)}" if named else ""


def _stress_read(pages, cancel, rec: dict, live: _StressLive,
                 stats_out: list = None, job: int = 0, case: str = "",
                 kept: list = None) -> tuple:
    """Pass 1 over prepared pages. `(page_texts, looped, truncated)`.

    `stats_out` collects each page's stats as it starts, for a job that is
    saved to the run log -- a list the caller owns, so the pages that finished
    before an error are still there to write. A page that looped or hit the
    token cap is flagged as soon as it comes back, with its page number.

    **Pages are read one at a time, on this job's own slot, until a worker is
    idle** (`live.slots`, set where the run splits pages): once every job of the
    queue has started, a free slot is lent to this file's next page and the
    pages run side by side. They are joined in page order whatever order they
    finished in, so the transcript is the one a sequential read gives -- the
    same split-and-join the Workspace does on vLLM (`read_pages_at_once`).
    `pages_lent` is how many pages were read on a lent slot, and `pages_peak`
    the most this job had in flight at once.

    `kept`, where given, gets each page's read that was kept -- its number, text,
    the page it was sent and how it ended -- for the Fields evidence of a
    missing-required flag (`_fields_evidence`).
    """
    started = time.perf_counter()
    lock = threading.Lock()
    tally = {"retries": 0, "recovered": 0, "prep": 0.0, "dry": 0, "dry_checked": 0,
             "dry_retries": 0, "dry_recovered": 0}
    inner = engine.MultiCancel()                   # a failed page hangs up its siblings
    hangup = engine._Hangup(inner)                 # held here: `attach` keeps it weakly
    cancel.attach(hangup)
    cut = "the reading model repeated itself and the loop guard cut the page off"

    def read_one(number, page, stats):
        """One page, with its loop and DRY retries. Returns `(text, tokens)`.

        A page that loops has ONE alert however many times it is read: raised
        `retrying` (or `no retry`), and marked `resolved` or `retry failed` by
        the read that settles it. DRY firing is one alert per page the same way,
        its message growing by a Detail each attempt it fired on.

        **DRY firing only flags the page by default** (2026-11-05, at the user's
        request). With `STRESS_DRY_RETRY` on, a page DRY fired on is read again
        one Detail up too (2026-10-27), even where it did not loop: a
        repeat the server had to break is a sign the page was going wrong, and
        the text DRY steered out of may be damaged. The climb stops at the first
        read that neither looped nor needed DRY. Should a retry loop where an
        earlier read did not, the latest read that did not loop is the one kept.
        """
        page_tokens, attempts, looped_at = 0, [], []
        loop_alert = dry_alert = None
        dry_seen, dry_checked, dry_message = [], False, ""
        kept = None             # (text, stats, attempt) of the latest read that did not loop
        origin = getattr(page, "_stress_origin", None) or {}
        truth_case = case if origin.get("source", "benchmark") == "benchmark" else None
        # Every read of this page, for the Compare view of whichever alert it
        # raises (`_StressEvidence`). Kept only if it raises one.
        seen = []

        def attempt_of(detail):
            return {"detail": detail, **{key: stats.get(key) for key in
                    ("resolution", "new_tokens", "seconds")}}

        while True:
            was = (getattr(page, "_stress_origin", None) or {}).get("detail")
            live.reading(job, number, page)
            try:
                text = engine.read_page(page, stats, cancel=inner)
            except Exception as err:
                if cancel.is_set() or inner.is_set():
                    raise engine.SweepCancelled("stopped")
                # Every read of the page, the one that failed last, for the
                # error flag's window (`_stress_job`).
                try:
                    err.stress_reads = seen + [_note_read(
                        f"failed: {type(err).__name__}: {err}", page, was)]
                except AttributeError:
                    pass
                raise
            finally:
                live.reading(job, number)
            if cancel.is_set():
                raise engine.SweepCancelled("stopped")
            live.touch(job)
            page_tokens += stats.get("new_tokens") or 0
            dry = stats.get("dry") or {}
            fired, looped = bool(dry.get("fired")), bool(stats.get("looped"))
            seen.append({"detail": was, "text": text, "page": page,
                         "looped": looped,
                         "truncated": bool(stats.get("truncated")),
                         "dry": dry if fired else None,
                         # Where DRY fired and where the loop is, for the window.
                         "marks": engine.read_marks(text, stats)})
            dry_checked = dry_checked or bool(dry.get("checked"))
            if fired:
                dry_seen.append(f"at {was}: {dryreplay.describe(dry)}" if was
                                else dryreplay.describe(dry))
                dry_message = "DRY penalised a repeat " + "; ".join(dry_seen)
                if dry_alert is None:
                    dry_alert = live.flag("dry", job, case, dry_message, stage="ocr",
                                          page=number, evidence=True)
                    _STRESS_EVIDENCE.put(dry_alert, case, number, seen, truth_case)
                else:
                    live.update_flag(dry_alert, dry_message)
            if not looped:
                kept = (text, dict(stats), attempt_of(was))
            redo_dry = fired and STRESS_DRY_RETRY
            if not looped and not redo_dry:
                break
            # Flagged first, then read again one Detail up (2026-10-26, at the
            # user's request). The retry's page is rebuilt from the file, off
            # the clock: that is this machine's work, not the server's.
            prep_began = time.perf_counter()
            try:
                retry = _stress_retry_page(page)
            except Exception:                   # noqa: BLE001
                retry = None
            with lock:
                tally["prep"] += time.perf_counter() - prep_began
            higher = retry._stress_origin["detail"] if retry is not None else None
            if looped:
                looped_at.append(was)
                where = _at_details(looped_at)
                if retry is not None:
                    state = "retrying"
                    message = f"{cut}{where}; reading it again at {higher}"
                elif attempts:
                    state, message = "retry failed", f"{cut}{where}; the retry failed"
                else:
                    state, message = "no retry", cut + where
                if loop_alert is None:
                    loop_alert = live.flag("looped", job, case, message, stage="ocr",
                                           page=number, state=state, evidence=True)
                    _STRESS_EVIDENCE.put(loop_alert, case, number, seen, truth_case)
                else:
                    live.update_flag(loop_alert, message, state=state)
            if redo_dry and retry is not None:
                live.update_flag(dry_alert, f"{dry_message}; reading it again at "
                                 f"{higher}", state="retrying")
            if retry is None:
                break
            attempts.append(attempt_of(was))
            stats.clear()
            stats.update(loop_retries=list(attempts), detail=higher)
            with lock:
                tally["retries" if looped else "dry_retries"] += 1
            page = retry
        # A retry that looped where an earlier read did not: keep the earlier
        # read -- only ever reachable on a climb DRY started.
        restored = None
        if stats.get("looped") and kept is not None:
            final = attempt_of(stats.get("detail"))
            text, saved, restored = kept
            stats.clear()
            stats.update(saved, loop_retries=[a for a in attempts + [final]
                                              if a is not restored],
                         detail=restored["detail"])
        with lock:
            tally["dry"] += int(dry_alert is not None)
            tally["dry_checked"] += int(dry_checked)
        if loop_alert is not None and not stats.get("looped"):
            with lock:
                tally["recovered"] += 1
            settled = (f"kept the read at {stats['detail']}, which did not loop"
                       if restored else f"read cleanly at {stats['detail']}")
            live.update_flag(loop_alert, f"{cut}{_at_details(looped_at)}; {settled}",
                             state="resolved")
        if dry_alert is not None and STRESS_DRY_RETRY:
            now_fired = bool((stats.get("dry") or {}).get("fired"))
            if not now_fired and not stats.get("looped"):
                with lock:
                    tally["dry_recovered"] += 1
                live.update_flag(dry_alert, f"{dry_message}; read again at "
                                 f"{stats.get('detail')} with nothing to penalise",
                                 state="resolved")
            elif attempts:
                live.update_flag(dry_alert, f"{dry_message}; the retry did not clear "
                                 f"it, kept the read at {stats.get('detail')}",
                                 state="retry failed")
            else:
                live.update_flag(dry_alert, f"{dry_message}; no higher Detail to "
                                 "read it at", state="no retry")
        if stats.get("looped"):
            pass                                # flagged above, every attempt
        elif stats.get("truncated"):
            cap_alert = live.flag("truncated", job, case, "the page read hit the "
                                  "token cap before it finished", stage="ocr",
                                  page=number, evidence=True)
            _STRESS_EVIDENCE.put(cap_alert, case, number, seen, truth_case)
        # Every attempt's tokens: the server generated them.
        live.page(page_tokens)
        return text, page_tokens, page

    slots = getattr(live, "slots", None)
    results = [None] * len(pages)
    lent = peak = 0
    if slots is None or len(pages) < 2:
        for number, page in enumerate(pages, 1):
            if cancel.is_set():
                raise engine.SweepCancelled("stopped")
            stats = {}
            if stats_out is not None:
                stats_out.append(stats)
            rec["at_page"] = number             # so an error's alert names the page
            results[number - 1] = (*read_one(number, page, stats), stats)
        peak = 1 if pages else 0
    else:
        pool = ThreadPoolExecutor(max_workers=len(pages),
                                  thread_name_prefix="stress-page")
        running, nxt, own_busy, error = {}, 0, False, None
        try:
            while True:
                # Start the next pages: on this job's own slot when it is free,
                # then on any slot an idle worker left -- only once nothing is
                # queued, so a job waiting to start never waits behind a page.
                while nxt < len(pages) and error is None and not cancel.is_set():
                    if not own_busy:
                        own_busy, slot = True, "own"
                    elif live.queued() == 0 and slots.lend():
                        slot = "lent"
                        lent += 1
                    else:
                        break
                    stats = {}
                    if stats_out is not None:
                        stats_out.append(stats)
                    number = nxt + 1
                    future = pool.submit(workspace.bind(read_one), number,
                                         pages[nxt], stats)
                    running[future] = (slot, number, stats)
                    nxt += 1
                    peak = max(peak, len(running))
                if not running:
                    break
                # A second at most, so a slot another job frees is noticed
                # without one of these pages having to finish first.
                finished, _ = wait(list(running), timeout=1.0,
                                   return_when=engine.FIRST_COMPLETED)
                for future in finished:
                    slot, number, stats = running.pop(future)
                    if slot == "own":
                        own_busy = False
                    else:
                        slots.release()
                    try:
                        results[number - 1] = (*future.result(), stats)
                    except engine.SweepCancelled:
                        pass                    # a Stop, or hung up by a sibling
                    except Exception as exc:    # noqa: BLE001
                        if error is None:
                            error = exc
                            rec["at_page"] = number
                            inner.set()
        finally:
            for slot, _, _ in running.values():
                if slot == "lent":
                    slots.release()
            pool.shutdown(wait=False, cancel_futures=True)
        if error is not None:
            raise error
        if cancel.is_set():
            raise engine.SweepCancelled("stopped")

    texts, looped, truncated = [], False, False
    ttft, page_s, page_tps, tokens = [], [], [], 0
    page_in, page_out = [], []
    for number, (text, page_tokens, page, stats) in enumerate(results, 1):
        texts.append(text)
        if kept is not None:
            kept.append({"number": number, "text": text, "page": page,
                         "detail": (getattr(page, "_stress_origin", None)
                                    or {}).get("detail"),
                         "looped": bool(stats.get("looped")),
                         "truncated": bool(stats.get("truncated")), "dry": None})
        looped = looped or bool(stats.get("looped"))
        truncated = truncated or bool(stats.get("truncated"))
        tokens += page_tokens
        # The per-page timings describe the read that was kept.
        ttft.append(stats.get("ttft_seconds"))
        page_in.append(stats.get("in_tokens"))
        page_out.append(stats.get("out_tokens"))
        page_s.append(stats.get("seconds"))
        if stats.get("tokens_per_second"):
            page_tps.append(stats["tokens_per_second"])
    prep = tally["prep"]
    rec.pop("at_page", None)
    rec.update(loop_retries=tally["retries"], loop_recovered=tally["recovered"],
               retry_prepare_s=round(prep, 3), dry_pages=tally["dry"],
               dry_checked=tally["dry_checked"], dry_retries=tally["dry_retries"],
               dry_recovered=tally["dry_recovered"])
    rec.update(ocr_s=round(time.perf_counter() - started - prep, 3), pages=len(pages),
               ocr_tokens=tokens, ttft_s=ttft, page_s=page_s, page_tps=page_tps,
               page_in=page_in, page_out=page_out, pages_lent=lent,
               pages_peak=peak)
    return texts, looped, truncated


def _stress_classify(pack: dict, rec: dict, live: _StressLive, cancel,
                     job: int = 0) -> None:
    """Group a long glued file into documents, then classify each one.

    Its model requests are counted as the file's EXTRACTION tokens (the user's
    rule: classify counts as extraction) -- a tally round the whole of it.
    """
    previous = servertime.open_tally()
    try:
        _stress_classify_body(pack, rec, live, cancel, job)
    finally:
        tally = servertime.close_tally(previous) or {}
        rec["extract_in"] = tally.get("in_tokens")
        rec["extract_out"] = tally.get("out_tokens")


def _stress_classify_body(pack: dict, rec: dict, live: _StressLive, cancel,
                          job: int = 0) -> None:
    status = backends.extract_status()
    started = time.perf_counter()
    asked = refused = 0
    steps = engine._segment_stream(pack["text"], pack["pages"], status)
    while True:
        if cancel.is_set():
            steps.close()
            raise engine.SweepCancelled("stopped")
        try:
            event = next(steps)
        except StopIteration as done:
            segments, _ = done.value
            break
        live.touch(job)
        if event.get("event") == "segment_page":
            asked = event.get("asked_total", asked)
            refused = event.get("refused_total", refused)
    rec["segment_s"] = round(time.perf_counter() - started, 3)
    live.move("segment", "classify")
    live.touch(job, "classify")

    started = time.perf_counter()
    expected = {tuple(sorted(doc["pages"])): doc for doc in pack["expected"]}
    exact = types = forms = by_model = 0
    for seg in segments:
        if cancel.is_set():
            raise engine.SweepCancelled("stopped")
        codes, how = engine.resolve_doc_types(seg.get("text") or "", status=status,
                                       pages=seg.get("page_texts"))[:2]
        live.touch(job)
        by_model += how == "model"
        doc = expected.get(tuple(sorted(seg.get("pages") or [])))
        if doc is None:
            continue
        exact += 1
        types += set(codes) == set(doc["types"])
        forms += (prompts.fields_for_types(codes)
                  == prompts.fields_for_types(doc["types"]))
    rec.update(classify_s=round(time.perf_counter() - started, 3),
               pages=len(pack["pages"]), asks=asked, refused=refused,
               docs_expected=len(pack["expected"]), docs_found=len(segments),
               docs_exact=exact, types_exact=types, form_match=forms,
               model_classified=by_model, status="ok")


def _field_value(value) -> str:
    """One extracted value as the Fields evidence shows it: text as it is, a
    table or an object as compact JSON, nothing as ''."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# Field-score verdicts that make an extracted value wrong (2026-10-28, at the
# user's request: *if the extraction is wrong in a field, flag it instantly*).
_WRONG_STATUS = {"wrong": "wrong", "partial": "partly right", "missed": "not returned",
                 "spurious": "invented"}


def _wrong_fields(result: dict, missing=()) -> list:
    """Each document's fields that do not match its ground truth:
    `[(document, [{"key", "status", "expected", "actual", "blame"}])]`.

    Read off the document's own `field_score` -- its scalars and its table
    cells -- so nothing is scored a second time. Only the values the score is
    taken over (`required`), plus a value invented where the page states none.
    A `missed` value the missing-required flag already names is left to it, so
    one empty field is not flagged twice. `blame` is who lost it -- `ocr`,
    `extract` or `unknown` -- where `blame.check` attributed it (never on a
    truth-fed run, whose transcript IS the truth).
    """
    named = {n: set(keys) for n, keys in missing}
    documents = [d for d in (result.get("documents") or []) if isinstance(d, dict)]
    out = []
    for doc in documents or [result]:
        score = doc.get("field_score")
        if not isinstance(score, dict) or score.get("error"):
            continue
        number = doc.get("document") or 1
        culprits = ((doc.get("blame") or {}).get("fields") or {})
        rows = list((score.get("scalars") or {}).get("rows") or [])
        for table in ("line_items", "income_items"):
            rows += (score.get(table) or {}).get("rows") or []
        wrong = []
        for row in rows:
            status, path = row.get("status"), row.get("path")
            if status not in _WRONG_STATUS or not path:
                continue
            if row.get("required") is False and status != "spurious":
                continue
            if status == "missed" and path in named.get(number, ()):
                continue
            wrong.append({"key": path, "status": status,
                          "expected": _field_value(row.get("expected")),
                          "actual": _field_value(row.get("actual")),
                          "blame": (culprits.get(path) or {}).get("blame")})
        if wrong:
            out.append((number, wrong))
    return out


def _wrong_message(wrong: list, several: bool, limit: int = 230) -> str:
    """One line naming each wrong field, what was wrong and who lost it."""
    who = {"ocr": "read", "extract": "extraction", "unknown": "unclear"}
    parts = []
    for number, rows in wrong:
        for row in rows:
            tags = [t for t in (_WRONG_STATUS[row["status"]] if row["status"] != "wrong"
                                else "", who.get(row["blame"] or "")) if t]
            parts.append((f"doc {number} " if several else "") + row["key"]
                         + (f" ({', '.join(tags)})" if tags else ""))
    count = sum(len(rows) for _, rows in wrong)
    head = f"{count} field(s) do not match the ground truth: "
    # An alert keeps 300 characters. Cut at a field, never mid-name; the whole
    # list rides on the flag's `fields` and in its window.
    shown = []
    for part in parts:
        rest = len(parts) - len(shown) - 1
        tail = f"; and {rest} more" if rest else ""
        if len(head) + len("; ".join(shown + [part])) + len(tail) > limit:
            break
        shown.append(part)
    more = len(parts) - len(shown)
    return head + "; ".join(shown) + (f"; and {more} more" if more else "")


def _fields_evidence(result: dict, missing: list) -> list:
    """The form of each document, for the Compare view of a missing-required flag.

    **Built 2026-10-26 at the user's request** -- *allow display window on field
    too*: a loop or DRY row opens the page and its text, and a missing-required
    row opens the same window with the fields beside them. One entry per
    document: every key it was asked, its value, whether the requirement demands
    it and whether it is one of the ones flagged.
    """
    flagged = {n: set(keys) for n, keys in missing}
    # Values that do not match the ground truth, marked in the same table
    # (`_wrong_fields`): the verdict, what the truth says, and who lost it.
    wrong = {n: {row["key"]: row for row in rows}
             for n, rows in _wrong_fields(result, missing)}
    documents = [d for d in (result.get("documents") or []) if isinstance(d, dict)]
    out = []
    for doc in documents or [result]:
        number = doc.get("document") or 1
        classified = doc.get("classified") or {}
        fields = doc.get("fields") if isinstance(doc.get("fields"), dict) else {}
        required = set(classified.get("mandatory") or [])
        gaps = flagged.get(number, set())
        bad = wrong.get(number, {})
        asked = list(doc.get("fields_asked") or classified.get("fields") or fields)
        if doc.get("items_asked") and prompts.INCOME_ITEMS_KEY not in asked:
            asked.append(prompts.INCOME_ITEMS_KEY)
        rows = [{"key": key, "value": _field_value(fields.get(key)),
                 "required": key in required
                 or (key == prompts.INCOME_ITEMS_KEY
                     and bool(classified.get("mandatory_items"))),
                 "missing": key in gaps} for key in asked]
        # A blank table cell is flagged by its path, which no asked key names.
        rows += [{"key": key, "value": "", "required": True, "missing": True}
                 for key in sorted(gaps) if key not in asked]
        # So is a wrong table cell.
        rows += [{"key": key, "value": bad[key]["actual"], "required": True,
                  "missing": False} for key in bad
                 if key not in asked and key not in gaps]
        for row in rows:
            if row["key"] in bad:
                row.update(wrong=bad[row["key"]]["status"],
                           expected=bad[row["key"]]["expected"],
                           blame=bad[row["key"]]["blame"])
        out.append({"document": number, "pages": list(doc.get("pages") or []),
                    "doc_types": list(doc.get("doc_types") or []),
                    "error": doc.get("error") or "", "rows": rows,
                    "validation_rechecks": doc.get("validation_image_rechecks") or [],
                    "image_fills": doc.get("image_fills") or []})
    return out


def _stress_job(spec: dict, cfg: dict, prepared: dict, cancel,
                live: _StressLive) -> dict:
    """One job of the queue, start to finish, on this thread. Returns its record;
    never raises. `_stress_phase` drives `_stress_job_steps` instead, so a full
    job's extraction can run on the extract workers."""
    steps = _stress_job_steps(spec, cfg, prepared, cancel, live)
    try:
        while True:
            next(steps)
    except StopIteration as done:
        return done.value


def _stress_job_steps(spec: dict, cfg: dict, prepared: dict, cancel,
                      live: _StressLive):
    """One job of the queue, as a generator. Returns its record; never raises.

    **A full job yields once, between its read and its extraction**
    (2026-10-28, at the user's request: *30 concurrency means per instance, not
    combined -- 30 OCR and 30 extract*). The read is done on one of the run's
    OCR workers; the yield hands the job to an extract worker, which resumes it
    (`_stress_phase`). So the run has up to `concurrency` reads AND up to
    `concurrency` extractions in flight, where before one pool of `concurrency`
    workers held each job through both and the two stages shared it. A
    generator keeps every local of the job across the hand-off; what is
    thread-local (the progress hook) is put back before the yield and set again
    after it. Every other mode never yields.
    """
    rec = {"wait_s": round(time.perf_counter() - spec["submitted"], 3)}
    deadline = spec.get("deadline")
    lazy = prepared.get("lazy")
    if cancel.is_set() or (deadline and time.perf_counter() > deadline):
        live.move("queued", None)
        rec["status"] = "cancelled" if cancel.is_set() else "skipped"
        if lazy is not None:
            lazy.release(spec["case"])
        return rec
    mode = cfg["mode"]
    stage = {"ocr": "ocr", "full": "ocr", "extract": "extract", "image": "extract",
             "classify": "segment"}[mode]
    # The OCR slot this job holds while it reads (`_StressSlots`); given back
    # when its read is done, so an idle worker's slot can be lent to a long
    # file's pages while this job is extracting.
    holding = live.slots is not None
    if holding:
        live.slots.hold()
    live.move("queued", stage)
    began = time.perf_counter()
    # Who the alerts name: the job's place in the queue (warm-up included) and
    # its document -- for classification, the cases glued into its file.
    job = spec["index"] + 1
    if mode == "classify":
        glued = prepared["packs"][spec.get("pack", spec["index"])].get("cases") or []
        who = f"file of {len(glued)} docs: " + ", ".join(dict.fromkeys(glued))
    else:
        who = spec["case"]
    live.begin(job, who, stage)
    # Saving to the run log (`save_log`, measured jobs only): one ordinary row
    # per job, written the way the route writes it -- `summarise` right after the
    # read, then `log_run` with the extraction on it -- and tagged with the run
    # and its concurrency, because its timings were taken under that load.
    log = bool(spec.get("log"))
    tag = {"stress_run": cfg.get("run_id", ""),
           "stress_concurrency": cfg["concurrency"]} if log else {}
    all_stats, row, logged = [], None, False
    read_began = began
    # Where an extraction flag puts its Compare window, once there is a
    # transcript to extract from (set below); None before that.
    fields_window = None
    kept = []                   # each page's kept read, for Fields evidence
    # The first extraction flag with a window, and what the extraction gave:
    # what an OCR flag of this file switches to (see `finally`).
    extract_alert = result = None
    missing = []
    # Every pass-2 request this job makes resets its stall clock as it streams
    # and when it ends -- not only the extraction as a whole (see `_progress`).
    previous_hook = _progress_swap(live.hook(job))
    hooked = True
    try:
        if mode == "classify":
            _stress_classify(prepared["packs"][spec.get("pack", spec["index"])],
                             rec, live, cancel, job)
        else:
            case_id = spec["case"]
            # A realData file has no case: nothing it reads or extracts is scored.
            real = cfg.get("source") == "realdata"
            case = None if real else scoring.cases_index()[case_id]
            if lazy is not None:
                # Prepared now rather than before the run (`_StressPages`). The
                # time is kept out of the job's latency, which is the server's.
                rec["prepare_s"] = round(lazy.take(case_id), 3)
                began += rec["prepare_s"]
                live.touch(job)
            texts, looped, truncated, truth = None, False, False, None
            if mode in ("ocr", "full"):
                read_began = time.perf_counter()
                texts, looped, truncated = _stress_read(
                    prepared["pages"][case_id], cancel, rec, live,
                    stats_out=all_stats if log else None, job=job, case=who,
                    kept=kept)
                # Rebuilding a retried page is this machine's work, kept out of
                # the job's latency like `prepare_s`.
                began += rec.get("retry_prepare_s") or 0
                text = engine.join_page_texts(texts) if len(texts) > 1 else texts[0]
                try:
                    truth = scoring.evaluate(case, text) if case else None
                except Exception:                   # noqa: BLE001
                    truth = None
                rec["char_accuracy"] = (truth or {}).get("char_accuracy")
                if log:
                    row = {**summarise(all_stats, cfg["detail"], read_began),
                           "truth": truth, **tag}
            if mode in ("extract", "full", "image"):
                # A read that looped or hit the cap is still classified and
                # extracted: its OCR flag says what went wrong with the read, and
                # the extraction's own flag (`_missing_required`) what it cost.
                if mode == "full":
                    # Hand the job to an extract worker (the docstring). Its OCR
                    # slot and this thread's progress hook are given back first.
                    if holding:
                        live.slots.release()
                        holding = False
                    _progress_swap(previous_hook)
                    hooked = False
                    live.move("ocr", "extract_queued")
                    stage = "extract_queued"
                    live.touch(job, stage)
                    waited = time.perf_counter()
                    yield "extract"
                    previous_hook = _progress_swap(live.hook(job))
                    hooked = True
                    # Waiting for an extract worker is this side's queue, kept
                    # out of the job's latency like `wait_s`.
                    rec["extract_wait_s"] = round(time.perf_counter() - waited, 3)
                    began += rec["extract_wait_s"]
                    live.move("extract_queued", "extract")
                    stage = "extract"
                    live.touch(job, stage)
                    if cancel.is_set():
                        raise engine.SweepCancelled()
                elif mode == "image":
                    # No read: the pages go to the extractor as they are, and a
                    # flag's window shows each page beside the form.
                    images = prepared["pages"][case_id]
                    text = prompts.IMAGE_TRANSCRIPT.format(pages=len(images))
                    texts = [text]
                    kept = _image_reads(images, f"sent to the extractor at {cfg['detail']}")
                    rec["pages"] = len(images)
                else:
                    text = prepared["truths"][case_id]

                def fields_window(alert, result, missing=()):
                    """Every extraction flag opens the window a loop row does,
                    with the form beside it (2026-10-26, at the user's request:
                    *show image with a field extracted so I can see the missing
                    field*): the pages and their text in full mode, the
                    transcript it was fed in extract mode (no page was read)."""
                    nonlocal extract_alert
                    extract_alert = extract_alert or alert
                    _STRESS_EVIDENCE.put(
                        alert, who, None,
                        list(kept) or [{"number": None, "text": text, "page": None,
                                        "detail": None, "looped": False,
                                        "truncated": False, "dry": None}],
                        None if real else case_id,
                        fields=_fields_evidence(result, list(missing)))

                started = time.perf_counter()
                retry_alert = None
                if mode == "image":
                    result, kept, retry_alert = _stress_image_extract(
                        images, cfg, None if real else case_id, rec, live, job, who,
                        fields_window, cancel)
                    # Rebuilding the pages one Detail up is this machine's work,
                    # kept out of the job's latency like `prepare_s`.
                    began += rec.get("retry_prepare_s") or 0
                    started += rec.get("retry_prepare_s") or 0
                else:
                    result = extract_fields(
                        text, cfg["extract_mode"], None if real else case_id, pages=texts,
                        truth_fed=mode == "extract", extras=cfg.get("extras"),
                        schema_first=cfg.get("schema_first", False),
                        each=lambda event: live.extract_event(job, event))
                    if mode == "full" and cfg.get("fill_missing", True):
                        # A Mandatory field left empty is asked of the page
                        # images with its Hybrid step, before the form is timed,
                        # logged or scored (`_stress_fill_missing`). The page's
                        # toggle turns it off.
                        why = (" -- the read looped, which may be why" if looped else
                               " -- the read hit the token cap, which may be why"
                               if truncated else "")
                        retry_alert = engine._stress_fill_missing(
                            result, prepared["pages"][case_id], texts,
                            None if real else case_id, rec, live, job, who,
                            fields_window, cancel, why)
                    if mode == "full" and cfg.get("validation_retry"):
                        engine._stress_recheck_validation(
                            result, prepared["pages"][case_id], texts,
                            None if real else case_id, rec, live, job, who, fields_window, cancel)
                rec["extract_s"] = round(time.perf_counter() - started, 3)
                if spec.get("detail"):
                    # Hold the existing result; format and send it after timing ends.
                    rec["_extraction_result"] = result
                rec["extract_tokens"] = result.get("tokens") or 0
                # How many document sets the file was split into and extracted,
                # since one file can hold several (2026-10-28, at the user's
                # request). Set before an error is raised, so a file whose every
                # document failed still says how many it held.
                rec.update(_stress_doc_sets(result, case))
                # Every pass-2 request of this file: classify, segment, steps or
                # the single request, table agents (servertime's tally).
                served = result.get("server_time") or {}
                rec["extract_in"] = served.get("in_tokens")
                rec["extract_out"] = served.get("out_tokens")
                chats = list(engine._agentic_chats(result))
                if chats:
                    rec["agentic_chats"] = [{**chat, "job": job, "case": case_id} for chat in chats]
                if log:
                    logged = True
                    if row is not None:
                        row["extracted"] = result
                        engine.log_run(row, prepared["sources"][case_id])
                    else:
                        engine.log_extract(result, None, {
                            "case": case_id,
                            "source": prepared["sources"][case_id]}, extra=tag)
                live.touch(job)
                missing = engine._missing_required(result)
                if result.get("error"):
                    if retry_alert is not None:
                        # Image mode flagged it, retried it and marked the row
                        # with how that went; its window shows the kept result.
                        rec["alerted"] = True
                        fields_window(retry_alert, result, missing)
                    elif result.get("looped") or result.get("truncated"):
                        rec["alerted"] = True
                        alert = live.flag("looped" if result.get("looped") else "truncated",
                                          job, who, result["error"], stage="extract",
                                          evidence=True)
                        fields_window(alert, result, missing)
                    raise ValueError(result["error"])
                if result.get("partial"):
                    alert = live.flag("partial", job, who, result["partial"]
                                      if isinstance(result["partial"], str) else
                                      "the extraction reply was cut off; only the fields "
                                      "it finished were kept", stage="extract",
                                      evidence=True)
                    fields_window(alert, result, missing)
                rec["missing_required"] = sum(len(keys) for _, keys in missing)
                if retry_alert is not None:
                    fields_window(retry_alert, result, missing)
                elif missing:
                    several = len(missing) > 1 or (result.get("documents_found") or 1) > 1
                    listed = "; ".join((f"doc {n}: " if several else "") + ", ".join(keys)
                                       for n, keys in missing)
                    why = (" -- the read looped, which may be why" if looped else
                           " -- the read hit the token cap, which may be why"
                           if truncated else "")
                    alert = live.flag("missing", job, who,
                                      f"{rec['missing_required']} required field(s) "
                                      f"came back empty: {listed}{why}", stage="extract",
                                      fields=[f"doc{n}.{k}" if several else k
                                              for n, keys in missing for k in keys],
                                      evidence=True)
                    fields_window(alert, result, missing)
                payload = {"truth": truth, "looped": looped,
                           "truncated": truncated, "extracted": result}
                if mode == "full":
                    engine.apply_read_floor(payload)
                summary = randomtest.summarise_round(payload)
                rec.update(field_rate=(summary["field"] / 100
                                       if summary["field"] is not None else None),
                           field_correct=summary["correct"],
                           field_partial=summary["partial"],
                           field_expected=summary["expected"],
                           unscored=bool(summary["unscored"]),
                           partial_reply=summary["partial_reply"])
                # A value that does not match the ground truth is flagged the
                # moment the job is scored (2026-10-28, at the user's request),
                # naming the field and who lost it; its window marks it in the
                # form with what the truth says. Scored files only.
                wrong = [] if real else _wrong_fields(result, missing)
                rec["wrong_fields"] = sum(len(rows) for _, rows in wrong)
                if wrong:
                    several = len(wrong) > 1 or (result.get("documents_found") or 1) > 1
                    alert = live.flag(
                        "wrong", job, who,
                        _wrong_message(wrong, several)
                        + (" -- the read scored under the read floor, so this score "
                           "is not counted" if summary["unscored"] else ""),
                        stage="extract", evidence=True,
                        fields=[f"doc{n}.{row['key']}" if several else row["key"]
                                for n, rows in wrong for row in rows])
                    fields_window(alert, result, missing)
            if row is not None and not logged:      # read only
                logged = True
                engine.log_run(row, prepared["sources"][case_id])
            rec["status"] = ("looped" if looped else
                             "truncated" if truncated else "ok")
    except engine.SweepCancelled:
        rec["status"] = "cancelled"
    except Exception as err:                        # noqa: BLE001
        if cancel.is_set():
            rec["status"] = "cancelled"
        else:
            rec["status"] = "error"
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
            if not rec.pop("alerted", False):     # a looped reply is flagged already
                page = rec.get("at_page")
                # An extraction that raised has no form to show, but its window
                # still has the pages it was given and says what went wrong.
                window = fields_window if stage == "extract" else None
                alert = live.flag("error", job, who, rec["error"], stage=stage,
                                  **({"page": page} if page else {}),
                                  **({"evidence": True}
                                     if window or stage == "ocr" else {}))
                if window:
                    window(alert, {"error": rec["error"]})
                elif stage == "ocr":
                    # A read that failed opens on the page it failed on; one
                    # that failed before any page was read, on what went wrong.
                    _STRESS_EVIDENCE.put(
                        alert, who, page,
                        getattr(err, "stress_reads", None)
                        or kept + [_note_read("failed: " + rec["error"])],
                        None if cfg.get("source") == "realdata" else spec["case"])
            # A job that failed is saved as the failure it was. A cancelled or
            # skipped one is not: nobody let it finish, so it measured nothing.
            if log and not logged and mode != "classify":
                _stress_log_error(mode, spec, prepared, all_stats, read_began,
                                  cfg, tag, err)
    finally:
        # A flagged read still goes on to extraction, so an OCR flag's window
        # can switch to this file's extraction (2026-10-27, at the user's
        # request). An extraction flag is that window already; with none, the
        # extraction is kept under the job's key.
        if (fields_window is not None and extract_alert is None
                and result is not None and live.ocr_flagged(job)):
            fields_window(live.job_key(job), result, missing)
        if hooked:
            _progress_swap(previous_hook)
        if holding:
            live.slots.release()
        live.end(job)
        rec.pop("at_page", None)
        rec.pop("alerted", None)
        live.move("classify" if mode == "classify" and "segment_s" in rec
                  else stage, None)
        if lazy is not None:
            lazy.release(spec["case"])
    rec["total_s"] = round(time.perf_counter() - began, 3)
    # When it ended on the phase's clock, and whose it was: what lets the report
    # be taken again without the longest jobs (`stress.without_longest`).
    rec["end_s"] = round(time.perf_counter() - spec["submitted"], 3)
    rec["job"], rec["who"] = job, who
    live.score(rec)
    live.finish(rec["status"] == "error")
    return rec


def _stress_doc_sets(result: dict, case) -> dict:
    """A job's document sets: found in the file, extracted, failed, and expected.

    `found` is how many documents pass 2 split the file into
    (`documents_found`, 1 for a file read as one). `failed` is the documents that
    came back with an error -- `documents_failed` on a split file, the result's
    own error on a single one. `expected` is the manifest's split for a benchmark
    case (1 where it lists none), and None for a realData file, which has no
    manifest entry to compare with.
    """
    found = int(result.get("documents_found") or 1)
    if found > 1:
        failed = int(result.get("documents_failed") or 0)
    else:
        failed = 1 if result.get("error") else 0
    expected = None
    if case:
        expected = len(case.get("documents") or []) or 1
    return {"doc_sets": found, "doc_sets_failed": failed,
            "doc_sets_expected": expected}


def _stress_log_error(mode, spec, prepared, all_stats, read_began, cfg, tag,
                      err) -> None:
    """The run-log row of a saved stress job that raised. Never raises."""
    try:
        case_id = spec["case"]
        if mode in ("ocr", "full"):
            row = {**summarise(all_stats, cfg["detail"], read_began), **tag}
            engine.log_run(row, prepared["sources"][case_id], error=err)
        else:
            engine.log_extract({"error": f"{type(err).__name__}: {err}",
                         "mode": cfg.get("extract_mode"),
                         "schema_first": bool(cfg.get("schema_first"))}, None,
                        {"case": case_id, "source": prepared["sources"][case_id]},
                        extra=tag)
    except Exception:                               # noqa: BLE001
        pass


def _stress_targets(reader: dict, extractor: dict) -> list:
    """The vLLM servers a stress run watches, each once: `[{role, url}]`.

    The reading server and the extraction server are both sampled, whichever
    of them the run's mode loads: a full-pipeline run loads both, and an OCR-
    or extraction-only run shows the other one idle beside it. A server that is not vLLM is never asked (`serves_metrics`), and a
    URL used for both passes is one target, `reader + extractor`.
    """
    out = []
    for role, info in (("reader", reader), ("extractor", extractor)):
        if not info or not backends.serves_metrics(info) or not info.get("url"):
            continue
        same = next((t for t in out if t["url"] == info["url"]), None)
        if same:
            same["role"] += " + " + role
        else:
            out.append({"role": role, "url": info["url"]})
    return out


def _stress_sample(targets: list, bases: dict, elapsed: float) -> tuple:
    """One /metrics read of every target: `(views, snapshots, errors)`."""
    views, snaps, errors = [], {}, []
    for t in targets:
        try:
            snaps[t["url"]] = now = _vllm_metrics(t["url"])
        except Exception as err:                    # noqa: BLE001
            errors.append(f"{t['role']} {t['url']}: {err}"[:200])
            continue
        views.append({**t, "view": stress.metrics_view(now, bases.get(t["url"]), elapsed)})
    return views, snaps, errors


class _StressSampler:
    """Reads /metrics every `every` s on its own thread, whatever the load.

    The sampler used to run inside the progress loop, which woke on every job
    that finished (2026-10-19, reported by the user: /metrics in the stress
    test "blocking the server request" at high concurrency). A read there held
    up every progress event behind vLLM's /metrics, which answers slowest
    exactly when the server is busiest. Here the clock alone decides: one tick
    every `every` s from the start, never two reads at once, and a tick that
    comes due while a slow read is still out is skipped rather than queued. The
    progress loop only picks up the latest result, so it never waits on it.
    """

    def __init__(self, targets: list, bases: dict, every: float, started: float):
        self._targets, self._bases = targets, bases
        self._every, self._started = every, started
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fresh = None              # (views, errors) not yet handed out
        self.last = {}                  # the last snapshot of each target, by URL
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="stress-metrics")

    def start(self):
        if self._targets:
            self._thread.start()
        return self

    def _run(self):
        tick = self._started + self._every
        while not self._stop.wait(max(0.0, tick - time.perf_counter())):
            views, snaps, errors = _stress_sample(
                self._targets, self._bases, time.perf_counter() - self._started)
            with self._lock:
                self.last.update(snaps)
                self._fresh = (views, errors)
            # Ticks missed during a slow read are skipped, not caught up.
            now = time.perf_counter()
            while tick <= now:
                tick += self._every

    def take(self):
        """The newest `(views, errors)` since the last call, or None."""
        with self._lock:
            fresh, self._fresh = self._fresh, None
            return fresh

    def stop(self):
        """Stops ticking. A read in flight finishes on its own; nothing waits."""
        self._stop.set()
        with self._lock:
            return dict(self.last)


def _stress_phase(specs: list, cfg: dict, prepared: dict, cancel, phase: str,
                  targets: list = (), bases: dict = None):
    """Run one queue of jobs at the configured concurrency, yielding progress.

    Every job is queued at once and a pool of `concurrency` workers works
    through it -- which is what makes `queue_wait` a real figure: a job's wait is
    the time it spent behind the jobs ahead of it, on THIS side of the wire.

    **The concurrency is per stage, not per job** (2026-10-28, at the user's
    request). A full run has two pools of `concurrency` workers: the OCR pool
    reads, and a job whose read is done is handed to the extract pool
    (`_stress_job_steps` yields there), so up to `concurrency` reads and up to
    `concurrency` extractions are in flight together. The extract side can sit
    under its limit -- it only has the files the reads have finished. Every
    other mode has one stage and one pool.
    Returns `(records, wall, timeline, last_metrics, draining)`, where
    `last_metrics` is the last snapshot of each target, by URL.

    **On Stop it returns at once**, with every job that had not finished counted
    as cancelled, and hands back the ones still running as `draining`. A request
    still waiting in the SERVER's queue has no response yet to hang up on -- the
    `Cancel` docstring has the measurement -- so waiting for those before
    reporting made a Stop take as long as the server's backlog (two minutes,
    measured through Ollama). The caller reports first and drains after,
    holding the guard until the last one ends.
    """
    # A warm-up's ids and the run's must not collide, nor this run's with the last.
    live = _StressLive(len(specs), prefix=f"{phase}-{uuid.uuid4().hex[:6]}-")
    if cfg.get("split_pages") and cfg["mode"] in ("ocr", "full"):
        live.slots = _StressSlots(cfg["concurrency"])
    pool = ThreadPoolExecutor(max_workers=cfg["concurrency"],
                              thread_name_prefix="stress")
    second = (ThreadPoolExecutor(max_workers=cfg["concurrency"],
                                 thread_name_prefix="stress-extract")
              if cfg.get("mode") == "full" else None)
    started = time.perf_counter()
    deadline = started + cfg["max_seconds"] if cfg["max_seconds"] and phase == "run" else None
    for spec in specs:
        spec["submitted"], spec["deadline"] = started, deadline
        # Warm-up jobs are thrown away from the report, and from the log too.
        spec["log"] = bool(cfg.get("save_log")) and phase == "run"
        spec["detail"] = bool(cfg.get("save_details")) and phase == "run"
    futures = [_stress_submit(pool, second, spec, cfg, prepared, cancel, live)
               for spec in specs]
    pending, timeline = set(futures), []
    sampler = _StressSampler(list(targets), bases or {}, cfg["metrics_every"],
                             started).start()
    tick = started

    def alerts():
        # Sent as they are seen, before the tick's progress event: the user's
        # rule is that a failure is reported when it happens, not at the end.
        for alert in live.take_alerts():
            at = alert.pop("at")
            updated = alert.pop("updated_at", None)
            if updated is not None:
                alert["updated"] = round(updated - started, 1)
            yield json.dumps({"event": "alert", "phase": phase,
                              "elapsed": round(at - started, 1), **alert}) + "\n"

    try:
        while pending and not cancel.is_set():
            # One event a second on the clock, not one per finished job: at
            # high concurrency jobs finish many times a second.
            tick += 1.0
            _, pending = wait(pending, timeout=max(0.0, tick - time.perf_counter()))
            now = time.perf_counter()
            tick = max(tick, now - 1.0)
            elapsed = now - started
            live.flag_stalled(STRESS_STALL_SECONDS)
            yield from alerts()
            snap = live.snapshot()
            stages = snap["stages"]
            timeline.append([round(elapsed, 1)] + [stages[s] for s in stress.STAGES]
                            + [snap["done"]])
            event = {"event": "progress", "phase": phase,
                     "elapsed": round(elapsed, 1), "total": len(specs), **snap,
                     "pages_per_second": round(snap["pages"] / elapsed, 3) if elapsed else 0,
                     "tokens_per_second": round(snap["tokens"] / elapsed, 2) if elapsed else 0}
            fresh = sampler.take()
            if fresh:
                views, errors = fresh
                if views:
                    event["servers_metrics"] = views
                    event["server_metrics"] = views[0]["view"]
                if errors:
                    event["metrics_error"] = "; ".join(errors)
            yield json.dumps(event) + "\n"
        yield from alerts()          # the last jobs' alerts, after the last tick
    finally:
        # Queued jobs are dropped; running ones carry on and are drained by the
        # caller. Nothing here blocks on them.
        pool.shutdown(wait=False, cancel_futures=True)
        if second is not None:
            second.shutdown(wait=False, cancel_futures=True)
        last_metrics = sampler.stop()
    wall = time.perf_counter() - started
    records, draining = [], []
    for future in futures:
        if future.done() and not future.cancelled():
            records.append(future.result())
        else:
            records.append({"status": "cancelled"})
            if not future.cancelled():
                draining.append(future)
    return records, wall, timeline, last_metrics, draining


def _stress_submit(pool, second, spec, cfg, prepared, cancel, live):
    """Start one job on `pool`; returns a future of its record.

    Where the job yields (a full job whose read is done) the rest of it is
    submitted to `second`, the extract pool, and the returned future resolves
    when that finishes. A job whose continuation is dropped -- the extract pool
    shut down on a Stop before an extract worker took it -- is closed, so its
    `finally` runs, and resolves as cancelled. Both steps run in the workspace
    current now, which `workspace.bind` would only give the first.
    """
    out = Future()
    steps = _stress_job_steps(spec, cfg, prepared, cancel, live)

    def advance():
        try:
            next(steps)
        except StopIteration as done:
            return True, done.value
        return False, None

    advance = workspace.bind(advance)

    def settle(inner, last):
        if inner.cancelled():
            steps.close()
            if not out.done():
                out.set_result({"status": "cancelled"})
            return
        try:
            finished, rec = inner.result()
        except BaseException as err:                # noqa: BLE001
            out.set_exception(err)
            return
        if finished:
            out.set_result(rec)
        elif last or second is None:
            # A step yielded where nothing can take the rest of it.
            steps.close()
            out.set_result({"status": "cancelled"})
        else:
            try:
                nxt = second.submit(advance)
            except RuntimeError:
                # The extract pool is shut down (a Stop): end the job here,
                # where it sees the cancel and stops at once.
                try:
                    finished, rec = advance()
                except BaseException as err:        # noqa: BLE001
                    out.set_exception(err)
                    return
                if not finished:
                    steps.close()
                out.set_result(rec if finished else {"status": "cancelled"})
                return
            nxt.add_done_callback(lambda f: settle(f, True))

    pool.submit(advance).add_done_callback(lambda f: settle(f, False))
    return out


def _sweep_scrape(targets: list) -> tuple:
    """One full /metrics read of every target, buckets included:
    `({url: (values, buckets)}, errors)`."""
    snaps, errors = {}, []
    for t in targets:
        try:
            snaps[t["url"]] = _vllm_scrape(t["url"], buckets=True)
        except Exception as err:                    # noqa: BLE001
            errors.append(f"{t['role']} {t['url']}: {err}"[:200])
    return snaps, errors


def _sweep_phases(specs: list, cfg: dict, prepared: dict, cancel, targets: list):
    """The concurrency sweep: one `_stress_phase` per level, lowest first.

    Every level's jobs run to the end before the next level starts, so each
    level's /metrics delta -- a full scrape right before it and one
    `SWEEP_SETTLE_SECONDS` after its last job -- is that level's requests and
    nobody else's. Progress events carry the level; a `sweep_level` event goes
    out as each level ends, so the page draws the curve as it grows.

    Returns `_stress_phase`'s five values over the whole sweep (the timeline
    laid end to end, one level after another) and the sweep's report part. A
    Stop ends it at the level running; the levels never started are cancelled.
    """
    levels, per_level = cfg["sweep"], cfg["sweep_jobs"]
    top = cfg["concurrency"]
    records, timeline, rows, last, draining = [], [], [], {}, []
    wall, at = 0.0, 0
    try:
        for index, level in enumerate(levels):
            if cancel.is_set():
                break
            chunk = specs[at:at + per_level]
            at += len(chunk)
            cfg["concurrency"] = level
            before, errors = _sweep_scrape(targets)
            ticks = {t["url"]: [] for t in targets}
            phase = _stress_phase(chunk, cfg, prepared, cancel, "run", targets,
                                  {url: snap[0] for url, snap in before.items()})
            try:
                while True:
                    try:
                        line = next(phase)
                    except StopIteration as stop:
                        result = stop.value
                        break
                    event = json.loads(line)
                    if event.get("event") == "progress":
                        for srv in event.get("servers_metrics") or []:
                            ticks.setdefault(srv["url"], []).append(srv["view"])
                        event.update(level=level, level_index=index, levels=len(levels))
                        line = json.dumps(event) + "\n"
                    yield line
            finally:
                phase.close()
            level_records, level_wall, level_timeline, level_last, draining = result
            settle = time.perf_counter() + SWEEP_SETTLE_SECONDS
            while not cancel.is_set() and time.perf_counter() < settle:
                time.sleep(0.1)
            after, more = _sweep_scrape(targets)
            errors += more
            servers = []
            for t in targets:
                if t["url"] in before and t["url"] in after:
                    view = concurrencysweep.level_view(before[t["url"]], after[t["url"]],
                                                       level_wall, ticks.get(t["url"]) or ())
                else:
                    view = None
                servers.append({**t, "view": view})
            row = {"level": level, "jobs": len(chunk), "wall_seconds": round(level_wall, 2),
                   "stopped": cancel.is_set(),
                   "client": concurrencysweep.client_view(
                       stress.report(level_records, level_wall, cfg["mode"], level)),
                   "servers": servers, "errors": errors}
            rows.append(row)
            yield json.dumps({"event": "sweep_level", "index": index,
                              "levels": len(levels), **row}) + "\n"
            timeline.extend([round(p[0] + wall, 1)] + p[1:] for p in level_timeline)
            wall += level_wall
            records.extend(level_records)
            last.update(level_last)
    finally:
        cfg["concurrency"] = top
    # The levels a Stop kept from starting: queued, never run.
    records.extend({"status": "cancelled"} for _ in specs[at:])
    sweep = {"levels": rows, "targets": targets, "jobs_per_level": per_level,
             "summary": concurrencysweep.summary(rows)}
    return records, wall, timeline, last, draining, sweep


def _stress_detail_identity(cfg):
    """Hash the code and selected truth files outside the benchmark clock."""
    index = scoring.cases_index() if cfg.get("source", "benchmark") == "benchmark" else {}
    benchmarks = {case: {"transcript": index[case]["ground_truth"],
                         "fields": fieldscore.truth_path(case),
                         "pages": index[case].get("pages"),
                         "doc_types": index[case].get("doc_types") or []}
                  for case in cfg["cases"] if case in index}
    return extractionlog.provenance(prompts.__file__, __file__, {
        "AGENTIC_RETRIES": engine.AGENTIC_RETRIES, "EXTRACT_SCHEMA": engine.EXTRACT_SCHEMA,
        "AGENTIC_PARALLEL": engine.AGENTIC_PARALLEL, "AGENTIC_PARALLEL_MAX": engine.AGENTIC_PARALLEL_MAX,
        "EXTRACT_DOCUMENT_PARALLEL_MAX": engine.EXTRACT_DOCUMENT_PARALLEL_MAX,
        "EXTRACT_REQUEST_CONCURRENCY": engine.EXTRACT_REQUEST_CONCURRENCY,
        "SEGMENT_PARALLEL_MAX": engine.SEGMENT_PARALLEL_MAX,
        "SEGMENT_PARALLEL_MIN_PAGES": engine.SEGMENT_PARALLEL_MIN_PAGES,
        "SEGMENT_PARALLEL_OVERLAP": engine.SEGMENT_PARALLEL_OVERLAP,
        "SEGMENT_MAX_PAGES": engine.SEGMENT_MAX_PAGES,
        "NORMALIZATION_VERSION": normalise.NORMALIZATION_VERSION}, extra_files={
            "normalise": normalise.__file__, "fieldscore": fieldscore.__file__,
            "segment": segment.__file__, "segmentparallel": segmentparallel.__file__,
            "extractionlimit": extractionlimit.__file__, "backends": backends.__file__,
            "grounding": grounding.__file__, "stress": stress.__file__,
            "manifest": config.SOLUTION_DIR / "manifest.json"}, benchmarks=benchmarks)


def _stress_run(cfg: dict, cancel):
    """The whole run: plan, prepare, warm up, measure, report."""
    rng, seed = stress.seeded_rng(cfg["seed"])
    mode = cfg["mode"]
    cfg["run_id"] = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-{seed}"
    total = cfg["warmup"] + cfg["jobs"]
    detail_identity = _stress_detail_identity(cfg) if cfg.get("save_details") else None
    reader = engine.llama_status() if mode in ("ocr", "full") else None
    extractor = backends.extract_status() if mode != "ocr" else None
    info = reader or extractor
    # Both servers are watched whatever the mode (the user's request,
    # 2026-10-19): a run that loads one shows the other beside it.
    targets = _stress_targets(reader or engine.llama_status(),
                              extractor or backends.extract_status())         if cfg["metrics"] else []

    specs = [{"index": i} for i in range(total)]
    # A concurrency sweep draws ONE level's jobs and runs them again at every
    # level, same files in the same order: only the concurrency differs.
    sweep_jobs = cfg["sweep_jobs"] if cfg.get("sweep") else 0
    measured = sweep_jobs or cfg["jobs"]
    if sweep_jobs:
        for k, spec in enumerate(specs[cfg["warmup"]:]):
            spec["pack"] = cfg["warmup"] + k % sweep_jobs     # classify: the same glued file
    if mode != "classify":
        # realData files are never in the run log's case counts, so "balanced"
        # balances over this run's own draws.
        history = (runlog.case_counts() if cfg["strategy"] == "balanced"
                   and cfg.get("source") != "realdata" else {})
        if cfg["strategy"] == "sequential":
            # Measured jobs start at the first selected file even after warm-up.
            drawn = [case_id for count in (cfg["warmup"], measured)
                     for case_id in stress.draw_cases(
                         count, cfg["cases"], cfg["strategy"], rng, history, cfg["lock"])]
        else:
            drawn = stress.draw_cases(
                cfg["warmup"] + measured, cfg["cases"], cfg["strategy"], rng, history,
                cfg["lock"])
        if sweep_jobs:
            level = drawn[cfg["warmup"]:]
            drawn = drawn[:cfg["warmup"]] + [level[k % sweep_jobs] for k in range(cfg["jobs"])]
        for spec, case_id in zip(specs, drawn):
            spec["case"] = case_id

    yield json.dumps({
        "event": "plan", "mode": mode, "seed": seed,
        "concurrency": cfg["concurrency"], "jobs": cfg["jobs"],
        # The concurrency sweep: each level's limit and its queue, in run order.
        "sweep": ([{"level": lv, "jobs": sweep_jobs} for lv in cfg["sweep"]]
                  if sweep_jobs else None),
        # The files every level runs, in the order it runs them.
        "sweep_cases": ([s.get("case") for s in specs[cfg["warmup"]:][:sweep_jobs]]
                        if sweep_jobs and mode != "classify" else None),
        "warmup": cfg["warmup"], "max_seconds": cfg["max_seconds"],
        "strategy": cfg["strategy"], "lock": cfg["lock"],
        "source": cfg.get("source", "benchmark"),
        "cases": cfg["cases"],
        "detail": cfg["detail"] if mode in ("ocr", "full", "image") else "",
        "extract_mode": cfg["extract_mode"] if mode in ("extract", "full", "image") else "",
        "schema_first": bool(cfg.get("schema_first")),
        "extras": (bool(cfg.get("extras", True)) if mode in ("extract", "full", "image")
                   else None),
        # Image mode: whether the extraction model says it takes images (None:
        # its server does not say, and the run is the test of it).
        "extractor_vision": cfg.get("extractor_vision") if mode == "image" else None,
        "pack_pages": cfg["pack_pages"] if mode == "classify" else 0,
        "documents": len({s["case"] for s in specs if "case" in s}),
        "server": {"url": info["url"], "kind": info["kind"]},
        "reader": (reader or {}).get("model") or "",
        "extractor": (extractor or {}).get("model") or "",
        "metrics": bool(targets),
        "metrics_servers": targets,
        # Which server kinds took the bypass and how; a server that has no
        # per-request switch is named rather than counted as uncached.
        "save_log": bool(cfg.get("save_log")),
        "save_details": bool(cfg.get("save_details")),
        "run_id": cfg["run_id"] if cfg.get("save_log") else "",
        "no_cache": bool(cfg.get("no_cache")),
        "dry": cfg.get("dry", True) is not False,
        "split_pages": bool(cfg.get("split_pages")) and cfg["mode"] in ("ocr", "full"),
        "fill_missing": bool(cfg.get("fill_missing")) and cfg["mode"] == "full",
        "validation_retry": bool(cfg.get("validation_retry")) and cfg["mode"] == "full",
        "no_cache_effect": ({s["kind"]: backends.cache_bypass_effect(s["kind"])
                             for s in (reader, extractor) if s}
                            if cfg.get("no_cache") else {}),
        # Requests in flight together sent as one /batch request, up to the
        # concurrency each; the servers named are the ones that take it.
        "vllm_batch": bool(cfg.get("vllm_batch")),
        "batch_size": cfg["concurrency"] if cfg.get("vllm_batch") else 0,
        "batch_servers": ([s["url"] for s in (reader, extractor)
                           if s and backends.serves_metrics(s)]
                          if cfg.get("vllm_batch") else []),
    }) + "\n"

    began = time.perf_counter()
    prepared = {"pages": {}, "truths": {}, "packs": [], "sources": {}}
    index = scoring.cases_index()
    if mode == "classify":
        truths = {cid: index[cid]["ground_truth"].read_text("utf-8")
                  for cid in cfg["cases"]}
        docs_of = {cid: index[cid].get("documents") or [] for cid in cfg["cases"]}
        docs_of["_types"] = {cid: index[cid].get("doc_types") or []
                             for cid in cfg["cases"]}
        for i in range(cfg["warmup"] + measured):
            prepared["packs"].append(stress.build_pack(
                cfg["cases"], truths, docs_of, cfg["pack_pages"], rng))
            yield json.dumps({"event": "preparing", "done": i + 1,
                              "total": cfg["warmup"] + measured}) + "\n"
    elif cfg.get("source") == "realdata":
        # Each file is prepared by the first job that draws it and freed after
        # the last one (`_StressPages`), not all of them here.
        prepared["lazy"] = _StressPages(specs, cfg["detail"], prepared)
    else:
        wanted = sorted({s["case"] for s in specs})
        for i, case_id in enumerate(wanted, 1):
            if cancel.is_set():
                break
            if mode in ("ocr", "full", "image"):
                prepared["pages"][case_id], prepared["sources"][case_id] = \
                    _stress_pages(case_id, cfg["detail"])
            else:
                path = index[case_id]["ground_truth"]
                prepared["truths"][case_id] = path.read_text("utf-8")
                # Named as the file fed in, the way a fields-only run names it.
                prepared["sources"][case_id] = describe_source(
                    path.name, prepared["truths"][case_id].encode("utf-8"), "truth")
            yield json.dumps({"event": "preparing", "done": i,
                              "total": len(wanted)}) + "\n"
    prepare_seconds = round(time.perf_counter() - began, 2)

    draining = []
    if cfg["warmup"] and not cancel.is_set():
        draining = (yield from _stress_phase(specs[:cfg["warmup"]], cfg, prepared,
                                             cancel, "warmup"))[-1]

    bases = {}
    for t in targets:
        try:
            bases[t["url"]] = _vllm_metrics(t["url"])
        except Exception:                           # noqa: BLE001
            pass
    records, wall, timeline, last = [], 0.0, [], {}
    batcher = engine._batcher if cfg.get("vllm_batch") else None
    if batcher is not None:
        batcher.reset_counts()        # the warm-up's batches are not the run's
    sweep = None
    if cfg.get("sweep") and not cancel.is_set():
        records, wall, timeline, last, draining, sweep = yield from _sweep_phases(
            specs[cfg["warmup"]:], cfg, prepared, cancel, targets)
    elif not cancel.is_set():
        records, wall, timeline, last, draining = yield from _stress_phase(
            specs[cfg["warmup"]:], cfg, prepared, cancel, "run", targets, bases)

    report = stress.report(records, wall, mode, cfg["concurrency"])
    report["timeline"] = stress.thin_timeline(timeline)
    report["timeline_columns"] = ["t", *stress.STAGES, "done"]
    report["prepare_seconds"] = prepare_seconds
    # A sweep's slowest jobs are its highest level's by construction, so
    # dropping them would only cut that level; the per-level figures stand.
    report["outliers"] = ([] if sweep is not None else
                          stress.without_longest(records, wall, mode, cfg["concurrency"],
                                                 STRESS_OUTLIER_MAX))
    if sweep is not None:
        report["sweep"] = sweep
    if batcher is not None:
        report["vllm_batch"] = batcher.summary()
    if targets:
        views, snaps, _ = _stress_sample(targets, bases, wall)
        for t in targets:          # a failed last read falls back to the last sample
            if t["url"] not in snaps and t["url"] in last:
                views.append({**t, "view": stress.metrics_view(
                    last[t["url"]], bases.get(t["url"]), wall)})
        views.sort(key=lambda v: [t["url"] for t in targets].index(v["url"]))
        if views:
            report["servers_metrics"] = views
            report["server_metrics"] = views[0]["view"]
    if cfg.get("save_details"):
        for event in extractionlog.events(records, specs[cfg["warmup"]:],
                                           prepared["sources"], cfg["warmup"], detail_identity,
                                           report=report):
            yield json.dumps(event, ensure_ascii=False) + "\n"
        # A stopped run may still drain requests; release exported results now.
        for record in records:
            record.pop("_extraction_result", None)
    yield json.dumps({"event": "done", "seed": seed, "stopped": cancel.is_set(),
                      "draining": len(draining), "report": report}) + "\n"
    # After a Stop: the report is out, and what is left is requests the server
    # is still working on. The guard is held until they end, so nothing else is
    # started on top of them and measured against their load.
    pending = set(draining)
    while pending:
        _, pending = wait(pending, timeout=1.0)
        yield json.dumps({"event": "draining", "in_flight": len(pending)}) + "\n"


class _StressBackground:
    """A stress run in its own thread, and every NDJSON line it has sent.

    **The run does not belong to the request that started it** (2026-10-27, at
    the user's request: *allow the stress to run as background even if the
    frontend is disconnected*). It used to be the response's generator, so a
    page that reloaded, slept or lost the network hung the run up with it --
    History said "The client disconnected before the report finished" and the
    rest of a long run was gone. Now a thread runs it to the end and keeps each
    line here; a response only follows the list, from any line, so a page that
    comes back reattaches (`/api/stress/attach`) and replays what it missed.

    One per workspace, the newest -- finished or not -- kept until the next
    run starts there, so a page that missed the end can still read the report.
    """

    def __init__(self, body: dict, mode: str):
        self.id = uuid.uuid4().hex[:12]
        self.request = body
        self.mode = mode
        self.started = time.time()
        self.finished = None
        self.lines = []
        self._cond = threading.Condition()
        # What `stress_runs_live` reports: the plan, then the latest
        # phase and progress, picked off the events the run sends anyway.
        self.status = {"phase": "starting"}

    # Events whose content is worth keeping for `public`. Every line is
    # `json.dumps({"event": ...})`, so the name sits at a fixed offset.
    _STATUS_EVENTS = ("plan", "preparing", "progress", "draining", "done")

    def emit(self, line: str):
        name = line[11:line.find('"', 11)] if line.startswith('{"event": "') else ""
        update = None
        if name in self._STATUS_EVENTS:
            try:
                update = json.loads(line)
            except ValueError:
                update = None
        with self._cond:
            self.lines.append(line)
            if update is not None:
                self._note(name, update)
            self._cond.notify_all()

    def _note(self, name: str, event: dict):
        if name == "plan":
            self.status.update({
                "concurrency": event.get("concurrency"), "jobs": event.get("jobs"),
                "warmup": event.get("warmup"), "reader": event.get("reader") or "",
                "extractor": event.get("extractor") or "",
                "server": (event.get("server") or {}).get("url") or ""})
        elif name == "preparing":
            self.status["phase"] = "preparing"
        elif name == "progress":
            self.status.update({"phase": event.get("phase") or "run",
                                "done": event.get("done"), "total": event.get("total"),
                                "failed": event.get("failed")})
        elif name == "draining":
            self.status.update({"phase": "draining", "in_flight": event.get("in_flight")})
        elif name == "done":
            self.status["phase"] = "draining"

    def public(self) -> dict:
        """What another workspace may see of this run: no request body."""
        with self._cond:
            return {"id": self.id, "mode": self.mode, "started": self.started,
                    "elapsed": round(time.time() - self.started, 1), **self.status}

    def finish(self):
        with self._cond:
            self.finished = time.time()
            self._cond.notify_all()

    def summary(self) -> dict:
        with self._cond:
            return {"id": self.id, "running": self.finished is None,
                    "started": self.started, "finished": self.finished,
                    "events": len(self.lines), "mode": self.mode,
                    "request": self.request}

    def follow(self, start: int = 0):
        """The lines from `start` on, as they come, until the run has ended.

        A follower that hangs up only stops following: the run never hears.
        """
        at = max(0, start)
        while True:
            with self._cond:
                while at >= len(self.lines) and self.finished is None:
                    self._cond.wait(timeout=30)
                batch, ended = self.lines[at:], self.finished is not None
            at += len(batch)
            if batch:
                yield "".join(batch)
            if ended and at >= len(self.lines):
                return


_stress_background = None
_stress_start_lock = threading.Lock()
# Every stress run still going, in ANY workspace (2026-10-28, for the user
# page: a stress test elsewhere loads the same model servers its jobs use).
# `_stress_job_here` is per workspace and cannot answer that.
_stress_live = []
_stress_live_lock = threading.Lock()


def stress_runs_live() -> list:
    """Every stress run in this process that has not finished, oldest first."""
    with _stress_live_lock:
        jobs = [job for job in _stress_live if job.finished is None]
    return [job.public() for job in jobs]
workspace.field("stress_background", lambda: None, fresh=True)


def _stress_job_here():
    """This workspace's newest background stress run, or None."""
    return workspace.get("stress_background", _stress_background)


def _keep_stress_job(job):
    global _stress_background
    if not workspace.put("stress_background", job):
        _stress_background = job


def _stress_follow_response(job, start=0):
    return Response(job.follow(start), mimetype="application/x-ndjson",
                    headers={"X-Stress-Run": job.id, "Cache-Control": "no-store"})


@app.post("/api/stress/stream")
def stress_stream():
    """Start a stress test and follow it: one NDJSON progress event a second
    and one report.

    The run goes on in the background whether or not anyone is following
    (`_StressBackground`); the `X-Stress-Run` header names it for
    `/api/stress/attach`. The full report is saved to History, separately from
    the accuracy run log. Optional `save_log` also writes a CSV row per
    measured job.
    """
    body = request.get_json(silent=True) or {}
    try:
        cfg = _stress_config(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    with _stress_start_lock:
        if sweeps_running():
            return jsonify(error="A random test or stress test is already running. "
                                 "Wait for it or stop it before starting another."), 409
        job = _StressBackground(body, cfg["mode"])
        started = threading.Event()
        thread = threading.Thread(target=workspace.bind(_stress_background_run),
                                  args=(job, cfg, started), daemon=True,
                                  name=f"stress-run-{job.id}")
        thread.start()
        # The thread takes the sweep guard itself, as the run's owner; the next
        # Run is not let through until it has.
        started.wait()
        _keep_stress_job(job)
    return _stress_follow_response(job)


@app.get("/api/stress/active")
def stress_active():
    """This tab's newest stress run, running or finished, or `{"run": null}`."""
    job = _stress_job_here()
    return jsonify(run=job.summary() if job is not None else None)


@app.get("/api/stress/attach")
def stress_attach():
    """Follow this tab's stress run again, from line `from` (0 replays it all).

    `id` names the run the page was following: if a newer one has replaced it,
    this is refused rather than splicing two runs into one report.
    """
    job = _stress_job_here()
    wanted = request.args.get("id")
    if job is None or (wanted and wanted != job.id):
        return jsonify(error="That stress run is no longer kept here."), 404
    try:
        start = int(request.args.get("from", 0))
    except ValueError:
        return jsonify(error="from must be a whole number."), 400
    return _stress_follow_response(job, start)


def _stress_background_run(job, cfg, started):
    """The thread of a background stress run: take the guard, run, keep each line."""
    try:
        cancel = begin_sweep(engine.MultiCancel())
    finally:
        started.set()
    with _stress_live_lock:
        _stress_live.append(job)
    try:
        for line in _stress_generate(cfg, cancel):
            job.emit(line)
    except Exception as err:                        # noqa: BLE001
        job.emit(json.dumps({"event": "error",
                             "error": f"{type(err).__name__}: {err}"[:300]}) + "\n")
    finally:
        end_sweep()
        job.finish()
        with _stress_live_lock:
            if job in _stress_live:
                _stress_live.remove(job)


def _stress_generate(cfg, cancel):
    """Every NDJSON line of one stress run, in the thread that holds the guard."""
    # The stress sampler and optional History monitor collect /metrics;
    # per-request timing must not add its own scrapes under contention.
    servertime.suspend_metrics()
    # "Bypass prompt cache": every request of the run and its drain is sent
    # uncacheable, so prefill is measured rather than a cache lookup.
    no_cache = bool(cfg.get("no_cache"))
    no_dry = cfg.get("dry", True) is False
    pass  # this state lives in engine.py
    try:
        if no_cache:
            backends.bypass_prefix_cache()
        if no_dry:
            backends.suppress_dry()
        if cfg.get("vllm_batch"):
            # One batcher for every server the run uses; the group key
            # keeps each URL's requests apart.
            engine._batcher = vllmbatch.Batcher(cfg["concurrency"], _send_batch)
        configuration = {**cfg, "ocr_profile": engine.ocr_profile(),
                         "sampling": ocr_sampling_state(), "loop_guard": engine.loop_guard(),
                         "read_floor": engine.read_floor(), "table_agents": engine.table_agents(),
                         "context": backends.num_ctx(),
                         "workspace": workspace.current_id(),
                         "reader": engine.llama_status() if cfg.get("mode") in ("ocr", "full") else None,
                         "extractor": (backends.extract_status() if cfg.get("mode")
                                       in ("extract", "classify", "full", "image") else None)}
        yield from _history_stream(_stress_run(cfg, cancel), "stress", configuration)
    except Exception as err:                        # noqa: BLE001
        yield json.dumps({"event": "error",
                          "error": f"{type(err).__name__}: {err}"[:300]}) + "\n"
    finally:
        if engine._batcher is not None:
            engine._batcher.close()
            engine._batcher = None
        if no_cache:
            backends.restore_prefix_cache()
        if no_dry:
            backends.restore_dry()
        servertime.resume_metrics()


class _HistoryMonitor:
    """Collect History snapshots without pausing the benchmark's generator.

    Start with one reading, then sample on a clock, skipping ticks missed by a
    slow scrape. Stop freezes the completed samples without waiting for network
    I/O or forcing a final scrape. A reading still in flight is discarded, so a
    saved report cannot change after it has been emitted.
    """

    def __init__(self, configuration):
        self._targets = []
        if configuration.get("history_monitor"):
            for role in ("reader", "extractor"):
                info = configuration.get(role)
                if (info and backends.serves_metrics(info)
                        and not any(other["url"] == info["url"] for other in self._targets)):
                    gpu_key = "gpu_extract_url" if role == "extractor" else "gpu_url"
                    self._targets.append({**info, "role": role,
                                          "gpu_url": configuration.get(gpu_key, "")})
        self._every = max(1.0, float(configuration.get("metrics_every", 5)))
        self._samples, self._total = deque(maxlen=300), 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # _monitor_scrape also reads per-tab loop-guard settings.
        self._thread = threading.Thread(target=workspace.bind(self._run), daemon=True,
                                        name="history-monitor")

    def start(self):
        if self._targets:
            self._thread.start()
        return self

    def _sample(self):
        for target in self._targets:
            if self._stop.is_set():
                return
            info = {k: v for k, v in target.items() if k != "gpu_url"}
            try:
                sample, _ = _monitor_scrape(info, target["url"], target["gpu_url"])
            except Exception as err:
                sample = {"error": str(err), "url": target["url"], "at": time.time()}
            with self._lock:
                if self._stop.is_set():
                    return
                self._samples.append({**sample, "role": target["role"]})
                self._total += 1

    def _run(self):
        tick = time.monotonic()
        while not self._stop.is_set():
            self._sample()
            now = time.monotonic()
            while tick <= now:
                tick += self._every
            if self._stop.wait(max(0.0, tick - time.monotonic())):
                return

    def stop(self):
        """Latest completed readings and total count; never join a slow scrape."""
        self._stop.set()
        with self._lock:
            return list(self._samples), self._total


def _history_stream(stream, kind, configuration):
    """Save on the server before emitting completion, including failed/closed streams."""
    run_id, storage_error, progress, finished, plan = None, "", None, False, None
    monitor_sampler, alerts = _HistoryMonitor(configuration), []
    # A concurrency sweep's finished levels, kept so a run that never reaches
    # its report (a restart, a crash) still has its curve so far in History.
    sweep_levels = []

    def saved_monitor():
        samples, total = monitor_sampler.stop()
        saved = {"monitor_samples": samples, "monitor_samples_total": total, "alerts": alerts}
        if sweep_levels:
            saved["sweep_levels"] = sweep_levels
        return saved

    try:
        run_id = runhistory.start(kind, configuration)
    except Exception as err:
        storage_error = f"History could not be saved: {err}"
    try:
        monitor_sampler.start()
        for line in stream:
            event = json.loads(line)
            stage = event.get("event")
            if stage in ("detail_log_start", "detail_log_job", "detail_log_end"):
                # Download-only payloads never enter History, CSV, or progress.
                if stage == "detail_log_start":
                    event["configuration"] = configuration
                    event["plan"] = plan
                yield json.dumps(event, ensure_ascii=False) + "\n"
                continue
            if stage == "alert":
                # An alert sent again under its id is the same issue, updated
                # (`_StressLive.update_flag`): it replaces the row, in place.
                at = next((i for i, a in enumerate(alerts)
                           if event.get("id") and a.get("id") == event["id"]), None)
                if at is None:
                    alerts.append(dict(event))
                else:
                    alerts[at] = dict(event)
            if event.get("event") == "plan":
                plan = dict(event)
                if run_id:
                    try:
                        runhistory.save_plan(run_id, event)
                    except Exception as err:
                        storage_error = f"History could not be saved: {err}"
            if event.get("event") == "progress":
                progress = event
            if stage == "sweep_level":
                sweep_levels.append(dict(event))
            if event.get("event") == "done":
                event.update(saved_monitor())
                if run_id:
                    try:
                        report = event.get("report") or {}
                        state = ("stopped" if event.get("stopped") else
                                 "completed_with_errors" if (report.get("error_rate") or 0) > 0
                                 else "completed")
                        runhistory.finish(run_id, state, event, progress)
                    except Exception as err:
                        storage_error = f"History could not be saved: {err}"
                finished = True
            if run_id:
                event["history_id"] = run_id
            if storage_error:
                event["history_error"] = storage_error
            yield json.dumps(event) + "\n"
        if run_id and not finished:
            runhistory.finish(run_id, "interrupted", result=saved_monitor(),
                              progress=progress, error="The stream ended without a final report.")
    except GeneratorExit:
        if run_id and not finished:
            try:
                runhistory.finish(run_id, "interrupted", result=saved_monitor(), progress=progress,
                                  error="The client disconnected before the report finished.")
            except Exception:
                pass
        raise
    except Exception as err:
        if run_id and not finished:
            try:
                runhistory.finish(run_id, "error", result=saved_monitor(), progress=progress,
                                  error=f"{type(err).__name__}: {err}"[:1000])
            except Exception:
                pass
        raise
    finally:
        monitor_sampler.stop()
        stream.close()


@app.get("/api/history")
def history_list():
    try:
        return jsonify(runhistory.list_runs(request.args.get("limit", 50),
                                            request.args.get("offset", 0)))
    except (ValueError, TypeError):
        return jsonify(error="limit and offset must be whole numbers."), 400
    except Exception as err:
        return jsonify(error=f"Could not read History: {err}"), 503


@app.get("/api/history/<run_id>")
def history_detail(run_id):
    try:
        item = runhistory.get(run_id)
    except Exception as err:
        return jsonify(error=f"Could not read History: {err}"), 503
    if item is None:
        return jsonify(error="History run not found."), 404
    response = jsonify(item)
    if request.args.get("download") == "1":
        response.headers["Content-Disposition"] = f'attachment; filename="run-{run_id}.json"'
    return response


@app.get("/api/local-preview/pdfjs/6.4.299/<path:name>")
def local_preview_pdfjs(name: str):
    """Versioned browser PDF renderer; local documents never enter this route."""
    response = send_from_directory(config.BASE_DIR / "vendor" / "pdfjs" / "6.4.299",
                                   name, max_age=31536000)
    if name.endswith(".mjs"):
        response.mimetype = "text/javascript"
    elif name.endswith(".wasm"):
        response.mimetype = "application/wasm"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/stress/evidence/<alert_id>")
def stress_evidence(alert_id: str):
    """A flagged stress page for the Compare view: each read of it, in order,
    with its text. Read `n`'s page image is `<this URL>/<n>.png`."""
    item = _STRESS_EVIDENCE.get(alert_id)
    if item is None and "job" in alert_id:          # `_StressLive.job_key`
        return jsonify(error="This file's extraction is not kept: it has not "
                             "finished yet, the run did not extract, or it is "
                             f"older than the newest {STRESS_EVIDENCE_MAX} "
                             "flagged pages."), 404
    if item is None:
        return jsonify(error="This page is no longer kept -- only the newest "
                             f"{STRESS_EVIDENCE_MAX} flagged pages since this "
                             "server started are."), 404
    case = scoring.cases_index().get(item["truth_case"] or item["case"], {})
    filename = str(case.get("pdf") or item["case"]).replace("\\", "/").rsplit("/", 1)[-1]
    response = jsonify(case=item["case"], filename=filename,
                   page=item["page"], truth_case=item["truth_case"],
                   fields=item.get("fields"),
                   attempts=[{"detail": a["detail"], "text": a["text"],
                              "looped": a["looped"], "truncated": a["truncated"],
                              "dry": a["dry"], "page": a.get("number"),
                              "note": a.get("note"),
                              "marks": a.get("marks") or [],
                              "image": a.get("page") is not None}
                             for a in item["attempts"]])
    response.headers['Cache-Control'] = 'private, no-cache'
    response.add_etag(weak=True)
    return response.make_conditional(request)


@app.get("/api/stress/evidence/<alert_id>/<int:index>.png")
def stress_evidence_image(alert_id: str, index: int):
    """One read's page image, as the model was sent it."""
    item = _STRESS_EVIDENCE.get(alert_id)
    if (item is None or not 0 <= index < len(item["attempts"])
            or item["attempts"][index].get("page") is None):
        return jsonify(error="This page is no longer kept."), 404
    return Response(_page_png(item["attempts"][index]["page"]), mimetype="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/stress/stop")
def stress_stop():
    """Ask the running stress test to stop. The same guard as the random test."""
    return jsonify(stopping=cancel_sweep(), running=sweeps_running())


@app.get("/api/stress/metrics")
def stress_metrics():
    """One snapshot of the active server's Prometheus `/metrics` -- vLLM only.

    **Refused, not fired, on anything else**, which is the user's rule: the
    button is for the high-end vLLM box, and on a local llama-server or Ollama
    nothing is requested at all. `?server=extract` asks about the extraction
    model's server instead of the reader's, for a text-only stress run.
    """
    info = (backends.extract_status() if request.args.get("server") == "extract"
            else backends.status())
    if not backends.serves_metrics(info):
        return jsonify(error=f"{info.get('kind') or 'no server'} at {info['url']} is "
                             "not vLLM, so /metrics was not requested.",
                       kind=info.get("kind"), url=info["url"]), 409
    try:
        now = _vllm_metrics(info["url"])
    except Exception as err:                        # noqa: BLE001
        return jsonify(error=f"Could not read {info['url']}/metrics: {err}"), 502
    return jsonify(kind=info["kind"], url=info["url"], at=time.time(),
                   view=stress.metrics_view(now),
                   vllm={k: v for k, v in sorted(now.items()) if k.startswith("vllm:")})


# --------------------------------------------------------------------------
# The live /metrics monitor (2026-10-06)
#
# One scrape per call, and the page calls it on a timer while the monitor is
# open. That is a poll, which this project refuses for llama.cpp -- `/slots`
# there shares the inference queue -- and it is allowed here for the reason the
# stress test samples `/metrics` mid-run: vLLM serves it from its own HTTP
# handler. So the gate is the same one: `backends.serves_metrics`, the server's
# own statement that it is vLLM. Anything else is refused and nothing is sent.
#
# The server is looked up with `backends.known`, never `status`: a status call
# re-probes `/v1/models` whenever its 3 s cache has expired, which at a 2 s poll
# would be a second request per tick for nothing the monitor shows.
# --------------------------------------------------------------------------

MONITOR_TIMEOUT = (1.5, 4)
_smi_cache = {"at": 0.0, "value": None}
# The last /monitor answer per (vLLM url, exporter url), and the lock that
# makes two calls landing together share one scrape rather than race to two.
_monitor_cache = {}
_monitor_lock = threading.Lock()


def _local_gpu():
    """This machine's GPU memory from `nvidia-smi`, cached a second, or None."""
    now = time.monotonic()
    if now - _smi_cache["at"] < 1.0:
        return _smi_cache["value"]
    text = machine._run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                         "--format=csv,noheader,nounits"])
    _smi_cache.update(at=now, value=monitor.parse_smi(text))
    return _smi_cache["value"]


def _monitor_target(which: str = ""):
    """The vLLM server to watch: the reading server, else the extraction one.

    `which` names one: "ocr" the reading server, "extract" the extraction
    server (the reader's own URL where none is set apart). The page asks for
    each in turn to watch both side by side.
    """
    if which == "ocr":
        urls = [backends.active_url()]
    elif which == "extract":
        urls = [backends.extract_url()]
    else:
        urls = [backends.active_url()]
        if backends.extract_separate():
            urls.append(backends.extract_url())
    seen = None
    for url in urls:
        info = backends.known(url) or backends.status(url)
        seen = seen or info
        if backends.serves_metrics(info):
            return info, None
    kinds = ", ".join(f"{(backends.known(u) or {}).get('kind') or 'unknown'} at {u}"
                      for u in urls)
    return seen, f"{kinds}: not vLLM, so /metrics is not requested."


def _vram(samples, url, gpu_url):
    """`(gpu dict or None, why not)` -- exporter, the vLLM scrape, or nvidia-smi."""
    if gpu_url:
        try:
            res = requests.get(gpu_url, timeout=MONITOR_TIMEOUT)
            if res.status_code != 200:
                return None, f"{gpu_url} answered HTTP {res.status_code}"
            found = monitor.gpu_memory(monitor.parse_samples(res.text))
            return found, (None if found else
                           f"{gpu_url} carries no DCGM or nvidia_gpu_exporter memory series")
        except Exception as err:                    # noqa: BLE001
            return None, f"could not read {gpu_url}: {err}"[:200]
    found = monitor.gpu_memory(samples)
    if found:
        return found, None
    if monitor.is_local(url):
        found = _local_gpu()
        return found, None if found else "nvidia-smi did not answer on this machine"
    return None, ("Only the vLLM port is reachable, and vLLM's /metrics does not "
                  "report the card's memory. A GPU exporter URL fills this in "
                  "where one can be reached.")


@app.get("/api/monitor")
def monitor_now():
    """One live reading of the vLLM server: KV cache, VRAM, queue, throughput.

    `server=ocr|extract` picks the reading or the extraction server; without it,
    the reading server, else the extraction one. `gpu=` names a GPU exporter's
    /metrics URL for VRAM, overriding `MONITOR_GPU_URL`. Rates are since this
    process last scraped the same server (`monitor.observe`), so the first call
    of a session has none.
    """
    which = (request.args.get("server") or "").strip().lower()
    if which not in ("", "ocr", "extract"):
        return jsonify(error="server must be ocr or extract."), 400
    info, refused = _monitor_target(which)
    if refused:
        return jsonify(error=refused, kind=(info or {}).get("kind"),
                       url=(info or {}).get("url")), 409
    url = info["url"]
    gpu_url = (request.args.get("gpu") or "").strip() or MONITOR_GPU_URL
    key = (url, gpu_url)
    with _monitor_lock:
        hit = _monitor_cache.get(key)
        now = time.monotonic()
        if hit and now - hit[0] < MONITOR_MIN_INTERVAL:
            return jsonify(**hit[1], cached=True, age=round(now - hit[0], 2),
                           min_interval=MONITOR_MIN_INTERVAL, idle_max=MONITOR_IDLE_MAX)
        body, code = _monitor_scrape(info, url, gpu_url)
        if code == 200:
            _monitor_cache[key] = (time.monotonic(), body)
    return jsonify(**body, cached=False, age=0.0, min_interval=MONITOR_MIN_INTERVAL,
                   idle_max=MONITOR_IDLE_MAX), code


def _monitor_scrape(info, url, gpu_url):
    """One real /metrics read: `(body, status)`. Failures are never cached."""
    try:
        res = backends.http_get(f"{url}/metrics", {"kind": "vllm"},
                                timeout=MONITOR_TIMEOUT)
    except Exception as err:                        # noqa: BLE001
        return dict(error=f"Could not read {url}/metrics: {err}"[:300], url=url), 502
    if res.status_code != 200:
        return dict(error=f"{url}/metrics answered HTTP {res.status_code}",
                    url=url), 502
    samples = monitor.parse_samples(res.text)
    snap = monitor.snapshot(samples)
    gpu, gpu_why = _vram(samples, url, gpu_url)
    rates = monitor.observe(url, snap)
    return dict(url=url, kind=info.get("kind"), model=info.get("model"),
                at=time.time(), snapshot=snap, rates=rates,
                lifetime_hit_rate=monitor.lifetime_hit_rate(snap),
                lifetime_mm_hit_rate=monitor.lifetime_mm_hit_rate(snap),
                lifetime_batch=monitor.lifetime_batch(snap),
                fit=monitor.fit(snap, rates),
                gpu=gpu, gpu_reason=gpu_why, gpu_url=gpu_url or "",
                loop_aborts=loop_aborts(),
                loop_abort_on=bool(engine.EXTRACT_LOOP_ABORT and engine.loop_guard())), 200



@app.post("/api/ocr")
def ocr():
    """Blocking read. Convenient for scripts; the page uses /api/ocr/stream."""
    try:
        pages, detail, job_id, case, source = prepare(request.files, request.form)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    started = time.perf_counter()
    page_texts, all_stats, run = [], [], {}
    try:
        read_all_pages(pages, page_texts, all_stats, run=run)
    except ValueError as err:
        engine.log_run(summarise(all_stats, detail, started, job_id, run), source, error=err)
        return jsonify(error=str(err)), 400
    except requests.RequestException as err:
        engine.log_run(summarise(all_stats, detail, started, job_id, run), source, error=err)
        return jsonify(error=f"model server connection failed: {err}"), 502

    text = engine.join_page_texts(page_texts)

    payload = summarise(all_stats, detail, started, job_id, run)
    payload["truth"] = evaluate_if_known(case, text)
    if EXTRACT and request.form.get("extract", "1") != "0" and text.strip():
        payload["extracted"] = extract_fields(
            text, case_id=case["id"] if case else None, pages=page_texts)
        engine.recheck_fields_from_images(payload["extracted"], pages, page_texts,
                                  fill_missing=request.form.get("fill_missing") == "1",
                                  validation_retry=request.form.get("validation_retry") == "1",
                                  case_id=case["id"] if case else None)
    engine.log_run(payload, source)
    return jsonify(text=text, pages=page_texts, **payload)


def batch_tokens(pieces, interval: float):
    """Coalesce streamed pieces into lists sent no more often than `interval` s.

    The first piece goes out alone and at once, so time to first token is still
    visible; the rest wait until `interval` has passed since the last batch, and
    whatever is pending goes out when the stream ends -- or fails, before the
    error propagates, so the text on screen is everything that was read. It can
    only flush when a piece arrives: a generator has no timer, and a stalled
    stream has nothing new to show anyway. interval <= 0 is one piece a batch.
    """
    if interval <= 0:
        for piece in pieces:
            yield [piece]
        return
    pending, last = [], None
    try:
        for piece in pieces:
            pending.append(piece)
            now = time.perf_counter()
            if last is None or now - last >= interval:
                yield pending
                pending, last = [], now
    except Exception:
        if pending:
            yield pending
        raise
    if pending:
        yield pending


@app.post("/api/ocr/stream")
def ocr_stream():
    """NDJSON stream, so a slow read shows partial text instead of hanging."""
    try:
        pages, detail, job_id, case, source = prepare(request.files, request.form)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    want_extract = request.form.get("extract", "1") != "0"
    fill_missing = request.form.get("fill_missing") == "1"
    validation_retry = request.form.get("validation_retry") == "1"

    def page_started(index, page):
        return json.dumps(
            {
                "event": "page",
                "page": index,
                "total": len(pages),
                "resolution": f"{page.width}x{page.height}",
                # Sent per page so the compare view can show the source
                # image while that page is still being transcribed.
                "job": job_id,
            }
        ) + "\n"

    def page_finished(index, stats):
        yield json.dumps({"event": "page_done", "page": index, **stats}) + "\n"
        if stats.get("truncated"):
            yield json.dumps({"event": "truncated", "page": index}) + "\n"
        if stats.get("looped"):
            yield json.dumps({"event": "looped", "page": index}) + "\n"

    def generate():
        started = time.perf_counter()
        collected, all_stats, run = [], [], {}
        summary = {}
        try:
            parallel = engine._page_parallelism(len(pages))
            if parallel > 1:
                # Every page is in flight at once, so there is no one page whose
                # tokens could stream: interleaved, they would splice pages into
                # each other on screen. Each page arrives whole instead, in page
                # order, as the same page/token/page_done events a sequential
                # read sends -- the page needs nothing new to assemble it.
                yield json.dumps({"event": "pages_parallel", "total": len(pages),
                                  "parallel": parallel, "job": job_id}) + "\n"
                with closing(engine.read_pages_at_once(pages, parallel, run=run)) as reads:
                    for index, page_text, stats in reads:
                        yield page_started(index, pages[index - 1])
                        yield json.dumps({"event": "token", "text": page_text,
                                          "n": stats.get("new_tokens") or 1}) + "\n"
                        collected.append(page_text)
                        all_stats.append(stats)
                        yield from page_finished(index, stats)
            else:
                for index, page in enumerate(pages, 1):
                    yield page_started(index, page)
                    stats = {}
                    while True:
                        parts = []
                        for batch in batch_tokens(engine.stream_page(page, stats),
                                                  STREAM_FLUSH_MS / 1000):
                            parts.extend(batch)
                            yield json.dumps({"event": "token", "text": "".join(batch),
                                              "n": len(batch)}) + "\n"
                        # The same climb `read_page` makes, said out loud: the
                        # page drops the looped text and streams the retry in.
                        retry = engine.loop_retry_page(page, stats)
                        if retry is None:
                            break
                        engine.record_retry(stats, page, retry)
                        page = retry
                        yield json.dumps({"event": "loop_retry", "page": index,
                                          "detail": stats["detail"],
                                          "resolution": f"{page.width}x{page.height}"}) + "\n"
                    collected.append(engine.finish_page("".join(parts), stats))
                    all_stats.append(stats)
                    yield from page_finished(index, stats)

            if len(collected) > 1:
                text = "\n\n".join(
                    f"--- page {i} ---\n{t}" for i, t in enumerate(collected, 1) if t
                )
            else:
                text = collected[0] if collected else ""
            summary = summarise(all_stats, detail, started, job_id, run)
            summary["truth"] = evaluate_if_known(case, text)
            yield json.dumps(
                {"event": "done", "text": text, "pages": collected, **summary}
            ) + "\n"

            # Loop the transcript back through the model for structured fields.
            # Emitted after "done" so the transcript is already on screen while
            # this runs.
            if EXTRACT and want_extract and text.strip():
                yield json.dumps({"event": "extracting"}) + "\n"
                # Agentic mode yields a step at a time on the way through; single
                # mode yields nothing and the loop runs once. Either way the
                # result arrives as the same "fields" event.
                stream = engine.extract_fields_stream(
                    text, case_id=case["id"] if case else None, pages=collected)
                while True:
                    try:
                        yield json.dumps(next(stream)) + "\n"
                    except StopIteration as stop:
                        result = stop.value
                        break
                if fill_missing or validation_retry:
                    yield from _recheck_events(result, pages, collected, fill_missing,
                                               validation_retry, case["id"] if case else None)
                summary["extracted"] = result
                # Applied before the event rather than left to `log_run` below:
                # this is the one path that hands the fields to the browser
                # before the row is written, and a page that had already drawn
                # the score would go on showing it. Same call, same rule --
                # `apply_read_floor` is idempotent for exactly this.
                engine.apply_read_floor(summary)
                yield json.dumps({"event": "fields", **result}) + "\n"

            # Logged once everything has run, so the row carries the accuracy and
            # the extraction counts rather than just the OCR timings.
            engine.log_run(summary, source)
            yield json.dumps({"event": "logged"}) + "\n"
        except GeneratorExit:
            # The browser hung up -- Stop, or a closed tab. Still worth a row.
            engine.log_run(summary or summarise(all_stats, detail, started, job_id, run),
                    source, status="cancelled")
            raise
        except Exception as err:  # surface failures inside the stream body
            engine.log_run(summary or summarise(all_stats, detail, started, job_id, run),
                    source, error=err)
            yield json.dumps({"event": "error", "error": str(err)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


def _environment(rows: list, runs: dict = None) -> dict:
    """The Summary tab's environment card: the machine, the server, the runs.

    **Three claims, kept apart on purpose**, because merging any two of them is a
    lie the card would have no way to signal:

    - `machine` -- the box THIS PROCESS is running on, probed locally. It says
      nothing about where the rows were produced: the log outlives the machine.
    - `server` -- the model server in force NOW, and the settings this process
      would send. Also not a property of the rows; switching the endpoint does
      not rewrite them.
    - `runs` -- what the rows in view were actually made under, counted from the
      log itself and narrowing with the card's filters. The hardware and warmth
      marks in there are INFERRED and labelled as such.

    So a card showing an RTX 3060 beside a row of Ollama runs is saying "this
    machine has one" and "those runs used Ollama", never "those runs used this
    card" -- which is a claim nothing in this project can make.

    **The server half never makes a request** -- `backends.known` returns the
    last probe or nothing at all. This route is on a hot path: the run-log
    card re-fetches it on every Refresh and every filter change, and
    `llama_status()` here took the summary from milliseconds to **5.1 s** against
    an unreachable endpoint, two connect timeouts at a time. It is also the rule
    this project has had since the beginning -- *never poll the model server*,
    because llama.cpp serves `/slots` from the same task queue as inference and a
    background poll cancels work in flight. A card that describes the server must
    not be the thing that pesters it.

    So the endpoint is reported **as last seen**, and `probed` is false until
    something that legitimately talks to the server has done so -- the page's own
    status fetch, a read, or startup.
    """
    info = backends.known()
    extract_model = backends.extract_model()
    return {
        "machine": machine.describe(),
        "server": {
            # The endpoint that WOULD be used, which is known without asking
            # anyone -- unlike everything else here, which is only as fresh as
            # the last probe.
            "url": backends.active_url(),
            "probed": info is not None,
            "backend": (info or {}).get("kind"),
            "model": (info or {}).get("model"),
            # Only where pass 2 runs on a different model -- blank reads as "the
            # same as the reading model", the rule the CSV column follows.
            "extract_model": extract_model or None,
            # Likewise only where pass 2 has a server of its own.
            "extract_url": (backends.extract_url()
                            if backends.extract_separate() else None),
            "slots": (info or {}).get("slots"),
            "reachable": (info or {}).get("reachable"),
            # Not from the probe: it is what this process sends, and it is true
            # whether or not anything is listening.
            "num_ctx": backends.num_ctx(),
        },
        # The knobs in force, so a slide can say what the numbers beside it were
        # produced under. Read from `settings` rather than from the environment,
        # for the reason that module exists: a second read of an env var can
        # disagree with the first after a clamp.
        "settings": {
            "detail": DEFAULT_DETAIL,
            "ocr_profile": engine.ocr_profile(),
            "extract_mode": "agentic" if engine.AGENTIC_EXTRACT else "single",
            "dry": DRY_MULTIPLIER,
            "max_new_tokens": engine.MAX_NEW_TOKENS,
            "extract_max_tokens": engine.EXTRACT_MAX_TOKENS,
        },
        # `runs` is passed in by a cached run-log payload, which holds the
        # log's half of this card and asks only for the live half again.
        "runs": runlog.conditions(rows) if runs is None else runs,
    }


# The last few run-log payloads, keyed by the log's version and the query
# (2026-10-21, at the user's request: *keep the same logic but i want to save
# time*). Every figure on the card is a function of the file and the query and
# nothing else -- no table reads the clock or a model server -- so an unchanged
# file asked the same question gets the same answer without the ~3 s of
# compiling a 52k-row log costs. A page reload, a second tab, or a control put
# back where it was is then a lookup. Only the environment's live half (the
# machine and the server as last probed) is rebuilt on every request.
#
# Four entries, because a payload is a few MB of Python objects and the card
# asks for one view at a time.
_runs_cache = OrderedDict()
_runs_cache_lock = threading.Lock()
_RUNS_CACHE_SIZE = 4


def _runs_payload(limit: int, window=None, min_read_pct=None,
                  include: dict = None, exclude: dict = None,
                  drop_single_source: bool = False, bias: bool = False) -> dict:
    """`_compile_runs`, from `_runs_cache` where the log has not moved since."""
    version = runlog._stat_key()
    key = (version, limit, window, min_read_pct,
           json.dumps(include, sort_keys=True, default=str),
           json.dumps(exclude, sort_keys=True, default=str),
           bool(drop_single_source), bool(bias))
    with _runs_cache_lock:
        hit = _runs_cache.get(key) if version is not None else None
        if hit is not None:
            _runs_cache.move_to_end(key)
    if hit is None:
        hit = _compile_runs(limit, window, min_read_pct, include, exclude,
                            drop_single_source, bias)
        if version is not None:
            with _runs_cache_lock:
                _runs_cache[key] = hit
                while len(_runs_cache) > _RUNS_CACHE_SIZE:
                    _runs_cache.popitem(last=False)
    # A shallow copy down to the one block that is rebuilt, so the cached entry
    # is never edited.
    payload = dict(hit)
    payload["totals"] = dict(hit["totals"])
    payload["totals"]["environment"] = _environment(
        None, runs=hit["totals"]["environment"]["runs"])
    return payload


def _compile_runs(limit: int, window=None, min_read_pct=None,
                  include: dict = None, exclude: dict = None,
                  drop_single_source: bool = False, bias: bool = False) -> dict:
    """The run-log card's whole payload: rows, compiled tables, and the pickers.

    **Built 2026-08-24 at the user's request** -- the tables were compiled under
    two process settings (`SUMMARY_RUNS`, `MIN_READ_FOR_FIELDS`) and over every
    row in the file, and neither could be moved without a restart or a hand-edit
    of the CSV.

    Three things are separated on purpose, and the separation is the whole point:

    - **`runs` is the log and is never filtered.** The raw table shows what ran;
      a row hidden there would be a row nobody could find to delete. The user's
      rule in as many words: *raw data will we store no matter what -- e.g. OCR
      gets 20%, passes to extract, but is not computed in the table if the
      threshold is above 20%*.
    - **`totals` is the tables, and is compiled over the FILTERED rows under the
      requested view.** A filtered row is gone from it entirely -- not shown and
      uncounted, but absent -- because *how does my best model do on sol001* is
      not answerable any other way.
    - **`facets` is built from the unfiltered log**, so a value that has just
      been excluded is still in the picker to be put back.

    Nothing here writes: a view is a way of reading the file, and the file is the
    same afterwards. That is what makes any threshold safe to try.
    """
    everything = runlog.read(limit=10 ** 6)
    matched = runlog.filter_rows(everything, include=include, exclude=exclude)
    # A percentage on the wire and a fraction in the setting. Named for its unit
    # because the log's own column is a percentage and the setting is not --
    # `runlog._field_trusted` carries the same warning, and getting it wrong
    # trusts everything or nothing.
    floor = None if min_read_pct is None else max(0.0, min(min_read_pct, 100.0)) / 100.0
    with runlog.view(window=window, min_read=floor):
        if drop_single_source:
            # The toggle's own words: exclude single-source failures from every
            # table BUT the errors one. So the ranking, standouts, reading,
            # extraction and per-document tables are compiled without those runs,
            # while `errors` is computed over the full filtered set -- it is where
            # the single-source rows are pointed AT, so it must still show them.
            # Computed inside the view so the window that decides "single source"
            # is the window the tables are read under.
            reduced = runlog.drop_single_source(matched)
            totals = runlog.totals(reduced, logged=len(everything))
            totals["errors"] = runlog.errors(matched)
            totals["single_source_dropped"] = len(matched) - len(reduced)
            # `matched` is the filter count, not the post-drop one, so the status
            # line keeps saying how many rows the FILTER matched -- the drop is a
            # separate figure reported beside it.
            totals["matched"] = len(matched)
        else:
            totals = runlog.totals(matched, logged=len(everything))
        # The presentation summary, compiled over the same filtered rows under
        # the same view -- so the card's chips reach it like every other table.
        #
        # **And the card-wide single-source toggle reaches it too**, which it did
        # not until 2026-08-25: this read `matched` while every other table read
        # `reduced`, so one card was showing two populations with nothing saying
        # so, and the toggle's own tooltip -- *from every table except the Errors
        # one* -- was false about the Summary tab. Measured on the live log, that
        # was worth 56% -> 35% on one setting's error rate.
        #
        # `single_source=False` because the drop has already happened: the rule
        # is not idempotent, and running it again would find NEW single-source
        # groups among what survived the first pass.
        #
        # Its own bias is applied on top and is its own toggle -- weak models and
        # time outliers -- because that tab is the one view here meant to be
        # shown rather than argued from. `bias=False` returns the same six blocks
        # over everything, which is what makes the bias inspectable rather than
        # load-bearing.
        totals["presentation"] = runlog.presentation(
            reduced if drop_single_source else matched,
            bias=bias, single_source=not drop_single_source)
        # Beside it rather than inside it: the environment describes the runs
        # under every view, biased or not, and a card that changed its machine
        # when a checkbox moved would be describing a different computer.
        totals["environment"] = _environment(matched)
    return {
        "runs": everything[:limit],
        "columns": runlog.COLUMNS,
        "totals": totals,
        "facets": runlog.log_facets(),
        # What the process would have used, so the page can say which of its
        # controls is at the server's own default and which the reader moved.
        "defaults": {"window": SUMMARY_RUNS,
                     "min_read_pct": round(engine.MIN_READ_FOR_FIELDS * 100, 1)},
        "filters": {"include": include or {}, "exclude": exclude or {}},
    }


def _run_limit(value, fallback: int = 50) -> int:
    try:
        return max(1, min(int(value), 1000))
    except (TypeError, ValueError):
        return fallback


@app.get("/api/runs")
def runs_list():
    """Recent rows of the run log, newest first, under the process settings."""
    return jsonify(**_runs_payload(_run_limit(request.args.get("limit", 50))))


@app.post("/api/runs/query")
def runs_query():
    """The same payload, recompiled under a different window, floor and filter.

    Body: `{limit, window, min_read_pct, include: {field: [...]},
    exclude: {field: [...]}, drop_single_source: bool}`. Every key is optional
    and `null` means *use the process setting* -- so a page moving one control does not silently reset the
    other. `runlog.FILTER_FIELDS` names the fields; an unknown one is ignored
    rather than refused, because a stale bookmark should return the table.

    A POST because the body is a nested filter, not because it changes anything:
    **this route writes nothing.**
    """
    body = request.get_json(silent=True) or {}

    def number(name, cast):
        value = body.get(name)
        if value is None or value == "":
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    include = body.get("include") if isinstance(body.get("include"), dict) else None
    exclude = body.get("exclude") if isinstance(body.get("exclude"), dict) else None
    window = number("window", int)
    if window is not None:
        window = max(0, min(window, 100000))
    return jsonify(**_runs_payload(_run_limit(body.get("limit", 50)),
                                   window=window,
                                   min_read_pct=number("min_read_pct", float),
                                   include=include, exclude=exclude,
                                   drop_single_source=bool(body.get("drop_single_source")),
                                   # **Defaults OFF, like every other control
                                   # here** (2026-08-25, at the user's request:
                                   # *show all if i do not toggle to exclude
                                   # them like the other page*). It shipped
                                   # defaulting on earlier the same day and that
                                   # was reversed: a summary that quietly leaves
                                   # models out is one more thing to remember
                                   # about a page whose whole job is to be read
                                   # at a glance.
                                   bias=bool(body.get("bias"))))


@app.get("/api/runs/stat")
def runs_stat():
    """Whether the run log has changed, in one `stat()` call.

    **This is what makes the card refresh itself when rows arrive from somewhere
    else** -- a `curl` against `/api/ocr`, the random test, a second tab, or
    another process entirely. Until now every refresh was tied to something the
    page itself had started, so a log filled from outside sat unread until
    somebody reloaded.

    It is deliberately not `/api/runs` with a small limit: that route compiles
    the whole summary (~90 KB, every table walked) and a poll would build and
    discard all of it on a timer. This reads two numbers off the filesystem, and
    the page fetches the real payload only when they move.

    **Not a violation of the never-poll rule.** That rule is about the MODEL
    SERVER: llama.cpp serves `/slots` from the same task queue as inference, so
    polling it cancels work in flight. This is the app's own CSV on local disk
    and touches no model server.
    """
    return jsonify(**runlog.signature())


@app.post("/api/runs/delete")
def runs_delete():
    """Remove named rows from the log. The one destructive route on this card.

    Body: `{rows: [{_row, timestamp, file}, ...]}` -- the keys the page received
    from `/api/runs`. `_row` is the row's position in the file and is checked
    against the timestamp and file found there before anything is removed, so a
    page holding a stale list deletes nothing rather than deleting its neighbour
    (`runlog.delete`).

    **The rows are archived, not destroyed**: they are appended to
    `logs/runs.deleted.csv` on the way out. A delete is how a reader takes a run
    out of every table and every mean, which the filters cannot do permanently;
    it is not a reason to lose the record that the run happened.
    """
    body = request.get_json(silent=True) or {}
    rows = body.get("rows")
    if not isinstance(rows, list) or not rows:
        return jsonify(error="Name the rows to delete."), 400
    outcome = runlog.delete(rows)
    if outcome is None:
        return jsonify(error="The run log could not be rewritten."), 500
    return jsonify(**outcome)


@app.get("/api/runs.csv")
def runs_csv():
    """The log file itself, for a spreadsheet."""
    if not runlog.LOG_PATH.exists():
        return jsonify(error="Nothing logged yet."), 404
    # Under the log's lock: another instance may be rewriting it -- see
    # `runlog._LogLock`.
    with runlog._lock:
        body = runlog.LOG_PATH.read_bytes()
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="ocr-runs.csv"'},
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File is larger than {MAX_UPLOAD_MB} MB.",
                   code="upload_too_large", limit_mb=MAX_UPLOAD_MB), 413


# Encoding-safe printing lives in config so the packaging script gets it too --
# both print installation paths, and both ran on a machine whose console could
# not encode them.
say = config.say


# A key in a JS object literal: at the start of a line, or after a comma. Both,
# because these maps put two keys on one line where the values are short -- an
# earlier version anchored to the line start only and silently saw 7 of the 11.
_JS_MAP_KEYS = re.compile(r"(?:^\s+|,\s*)(\w+)\s*:", re.M)


def template_field_keys():
    """The keys `static/benchmark.js` knows how to draw, per map.

    Returns {"FIELD_LABELS": {...}, "FIELD_HINTS": {...}, "FIELD_SECTIONS": {...}}.

    Parsed rather than imported because the maps are JavaScript. It is crude on
    purpose -- it reads the key names out of three literals and nothing else --
    and it exists because the failure it catches is silent: **a key absent from
    `FIELD_SECTIONS` is not rendered at all**, and when the schema was widened on
    2026-08-19 a third of it simply did not appear on the page, with no error
    anywhere. Returns empty maps rather than raising if the file has moved on;
    a parser that fails loudly on a cosmetic edit would be worse than the bug.
    """
    try:
        text = (config.BASE_DIR / "static" / "benchmark.js").read_text("utf-8")
    except Exception:                                    # noqa: BLE001
        return {}

    def literal(name, opener, closer):
        try:
            start = text.index(f"const {name} = {opener}")
        except ValueError:
            return ""
        return text[start:text.index(closer, start)]

    labels = set(_JS_MAP_KEYS.findall(literal("FIELD_LABELS", "{", "};")))
    hints = set(_JS_MAP_KEYS.findall(literal("FIELD_HINTS", "{", "};")))
    sections = set(re.findall(r"'(\w+)'",
                              literal("FIELD_SECTIONS", "[", "\n];")))
    # Section TITLES are quoted the same way as the keys; only names the schema
    # knows are keys, and a key the schema does not know is what we are looking
    # for, so compare against the union of both rather than filtering here.
    return {"FIELD_LABELS": labels, "FIELD_HINTS": hints,
            "FIELD_SECTIONS": sections}


def check_template_fields():
    """Lines describing any drift between the schema and the page's own maps.

    Empty when they agree. A key in the schema and not on the page is invisible
    to a reader; a key on the page and not in the schema is a row that can only
    ever be empty. Both are worth a line at startup and neither is worth a
    refusal -- the page still works, it just does not show everything.
    """
    maps = template_field_keys()
    if not maps:
        return ["could not read the field maps out of static/benchmark.js"]
    schema = set(prompts._SCALAR_KEYS)
    problems = []
    for name in ("FIELD_LABELS", "FIELD_HINTS", "FIELD_SECTIONS"):
        drawn = maps[name]
        missing = sorted(schema - drawn)
        if missing:
            problems.append(f"{name} does not know {', '.join(missing)}")
        # Only names that ARE schema keys can be extra; section titles and other
        # words in those literals are not keys and are not a fault.
        extra = sorted((drawn & set(prompts.RETIRED_KEYS)) - schema)
        if extra:
            problems.append(f"{name} still lists retired key(s) "
                            f"{', '.join(extra)}")
    return problems


def preflight():
    """Report what this instance can and cannot do, before it accepts traffic.

    Everything here is a warning, never a refusal. The optional pieces -- PDF
    input, HEIC input, the benchmark fixtures, a reachable model server -- are
    each missing on some legitimate deployment, and a server that exits because
    the model server has not started yet is worse than one that says so and waits
    for it. The point is that a misconfiguration is visible in the first ten lines
    of the log rather than discovered later as a puzzling 500.
    """
    say(f"[ocr] listening on http://{config.HOST}:{config.PORT}")
    # A key the page's maps do not know is a key the page does not draw, and
    # nothing else says so -- see `check_template_fields`.
    for problem in check_template_fields():
        say(f"[ocr] WARNING: {problem}")
    if config.HOST not in ("127.0.0.1", "localhost", "::1"):
        say(f"[ocr] WARNING: bound to {config.HOST}, reachable from other hosts. "
            "This app has no authentication -- put it behind a reverse proxy or "
            "keep it on a trusted network.")

    # The machine before the model server: every timing in the run log is a wall
    # clock on THIS box, and until now nothing recorded which box that was.
    machine.report()

    # Find a server before reporting on one. The endpoint list starts on
    # llama-server's port whether or not anything is listening there, so without
    # this a machine running Ollama starts up pointed at a dead port and stays
    # there until somebody moves the picker -- and every probe of it costs two
    # connect timeouts. Startup is the one place this may be paid: it is a
    # network request, so it must never go on a polling path.
    if AUTO_SELECT_SERVER:
        picked = backends.autoselect()
        if picked["changed"]:
            say(f"[ocr] auto-selected {picked['selected']} "
                f"(tried {', '.join(picked['tried'])})")
        elif picked["reason"]:
            say(f"[ocr] {picked['reason']} Leaving the endpoint on "
                f"{picked['selected']}; use the picker or set OCR_ENDPOINTS.")

    status = engine.llama_status()
    say(f"[ocr] {status['kind'] or 'server'} {status['url']} "
        f"model={status['model']} available={status['available']}")
    # The models, selected once, here -- so the page opens on them and nobody
    # has to work the pickers to get the build this log says is the best one.
    # After this they are an ordinary choice, which is what keeps every other
    # meaning on the page intact; see `backends.autoselect_models`.
    if backends.extract_separate():
        ex = backends.extract_status(force=True)
        say(f"[ocr] extraction server: {ex['kind'] or 'server'} {ex['url']} "
            f"model={ex['model']} available={ex['text_available']}"
            + (f" -- {ex['text_reason']}" if ex["text_reason"] else ""))
    picked = backends.autoselect_models()
    # Named only where pass 2 is not running on the reading model, so the usual
    # one-model line stays as short as it was.
    if picked["extract"]:
        say(f"[ocr] extracting with {picked['extract']} -- the run log's best "
            "extractor. AUTO_BEST_MODEL=0 to use the reading model for both "
            "passes, which is what every baseline was measured under.")
    say(f"[ocr] switchable endpoints: {', '.join(backends.endpoints())}")
    if status["reason"]:
        say(f"[ocr] {status['reason']}")

    if not PDF_OK:
        say("[ocr] WARNING: PyMuPDF missing -- PDF uploads will be rejected. "
            "pip install pymupdf")
    if not HEIF_OK:
        say("[ocr] note: pillow-heif missing -- HEIC/HEIF uploads unsupported.")

    cases = scoring.cases_index()
    say(f"[ocr] documents folder: {config.MOCK_DIR} "
        f"({len(mock_files())} readable)" if config.MOCK_DIR.exists()
        else f"[ocr] documents folder: {config.MOCK_DIR} (absent -- upload only)")
    say(f"[ocr] ground truth: {config.SOLUTION_DIR} ({len(cases)} scored cases)"
        if cases else
        f"[ocr] ground truth: {config.SOLUTION_DIR} (absent -- accuracy scoring off)")
    say(f"[ocr] real data: {REAL_DIR} ({len(real_files())} readable, unscored)"
        if REAL_DIR.exists() else f"[ocr] real data: {REAL_DIR} (absent)")

    # The one directory that must be writable. Checked by actually creating it,
    # because a read-only mount or a wrong owner is the failure this is for, and
    # neither shows up in an existence test.
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.LOG_DIR / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        say(f"[ocr] run log: {runlog.LOG_PATH}")
    except OSError as err:
        say(f"[ocr] WARNING: cannot write to {config.LOG_DIR} ({err}). "
            "Runs will not be logged; set OCR_LOG_DIR to a writable path.")


def port_free(host: str, port: int) -> bool:
    """True when nothing else is listening on `host:port`.

    Bound WITHOUT SO_REUSEADDR on Windows, which is the point: there that flag
    lets a second socket bind a port another process is already listening on,
    and Werkzeug sets it -- so a second copy of the app used to start "fine" on
    the first one's port, and each request went to whichever of the two Windows
    picked. Elsewhere the flag only forgives TIME_WAIT and a listener still
    refuses the bind, so it is set there as the server sets it.
    """
    import socket
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return True   # let the server report an unresolvable host itself
    family, _, _, _, address = infos[0]
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(address)
        except OSError:
            return False
    return True


def choose_port() -> int:
    """The port this instance listens on; see `config.PORT_FIXED`."""
    if config.PORT_FIXED:
        if not port_free(config.HOST, config.PORT):
            say(f"[ocr] port {config.PORT} is already in use -- another copy of "
                "this app? Set PORT to a free port, or unset it to take the next "
                "free one.", stream=sys.stderr)
            sys.exit(1)
        return config.PORT
    for port in range(config.PORT, min(config.PORT + config.PORT_TRIES, 65536)):
        if port_free(config.HOST, port):
            if port != config.PORT:
                say(f"[ocr] port {config.PORT} is in use (another instance?) -- "
                    f"this one takes {port}.")
            return port
    say(f"[ocr] ports {config.PORT}-{config.PORT + config.PORT_TRIES - 1} are all "
        "in use. Set PORT to a free port.", stream=sys.stderr)
    sys.exit(1)


# Register against this module object even when launched as __main__: importing
# app again here would create a second engine and a second set of globals.
import userweb
get_user_service = userweb.register(app, engine)
import serviceapi
serviceapi.register(app, engine, get_user_service)
# Preserve the User API's origin check before parsing or validating its body.
app.before_request(webcommon.validate_api_json_object)


if __name__ == "__main__":
    # Before preflight, which prints the URL. Werkzeug's server sets
    # SO_REUSEADDR, which on Windows would let a racing second instance bind
    # this port anyway -- see `port_free` -- so it is turned off there.
    config.PORT = choose_port()
    if os.name == "nt":
        import werkzeug.serving
        werkzeug.serving.BaseWSGIServer.allow_reuse_address = False
    preflight()
    # What every browser tab starts from: the server and models preflight just
    # selected, and every switch as the environment set it.
    workspace.seal()
    get_user_service().start()
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False,
        threaded=True,
    )


# The User page's Stress chip reports this app's live stress runs.
engine.stress_runs_live = stress_runs_live


def __getattr__(name):
    """`app.X` for what moved to engine.py, for tools and tests that still read it.
    Patching must target `engine` -- the engine reads its own module globals."""
    try:
        return getattr(engine, name)
    except AttributeError:
        raise AttributeError(f"module 'app' has no attribute {name!r}") from None
