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
import gzip
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import weakref
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, render_template, request

import backends
import settings
import blame
import config
import dryreplay
import fieldscore
import extractionlog
import fixscore
import grounding
import machine
import monitor
import normalise
import prompts
import randomtest
import runlog
import runhistory
import scoring
import segment
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
    steps_for_types,
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
    GEN_CONNECT_TIMEOUT,
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

# PyMuPDF is not thread-safe: two threads inside MuPDF at once can crash the
# whole process, which the page sees as a stream cut off mid-run
# (ERR_INCOMPLETE_CHUNKED_ENCODING). Every PDF open and render holds this.
_PDF_LOCK = threading.Lock()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


class _ProxyPrefix:
    """Answer a request made under any path prefix as if it were made at `/`.

    **Worked out per request, never configured** (2026-10-20): the app sits
    behind a reverse proxy at a sub-path -- nginx serving `host:3000/OCR/` --
    and the page builds every request from its own address, so whatever path
    the page was reached at is the prefix. A proxy that strips the prefix
    (`proxy_pass http://host:port/;`) sends plain `/api/...` and nothing here
    happens. One that passes it through sends `/OCR/api/...`, and:

    * `X-Forwarded-Prefix`, where the proxy sends one, is taken first;
    * otherwise everything before `/api/` is the prefix -- every route but the
      page itself lives under `/api/`;
    * otherwise a GET for a path that is no route of this app and ends in `/`
      (or has no dot in its last part) is the page, served at that prefix.

    The prefix moves to SCRIPT_NAME, where Flask expects a mount point.
    """

    def __init__(self, flask_app):
        self.flask_app = flask_app
        self.wsgi = flask_app.wsgi_app

    def _is_route(self, path: str, method: str) -> bool:
        try:
            self.flask_app.url_map.bind("localhost").match(path, method=method)
            return True
        except Exception:                           # NotFound, MethodNotAllowed, redirects
            return False

    def _prefix(self, environ, path: str) -> str:
        forwarded = (environ.get("HTTP_X_FORWARDED_PREFIX") or "").strip().rstrip("/")
        if forwarded and (path == forwarded or path.startswith(forwarded + "/")):
            return forwarded
        method = environ.get("REQUEST_METHOD", "GET")
        if self._is_route(path, method):
            return ""
        # Match an actual nested route before falling back to the legacy mount
        # heuristic: /ocr/user and /ocr/static/user/app.js are not the main page.
        for match in re.finditer('/', path[1:]):
            at = match.start() + 1
            if self._is_route(path[at:], method):
                return path[:at]
        at = path.find("/api/")
        if at > 0:
            return path[:at]
        last = path.rstrip("/").rsplit("/", 1)[-1]
        if (method in ("GET", "HEAD") and path != "/" and not self._is_route(path, method)
                and (path.endswith("/") or "." not in last)):
            return path.rstrip("/")
        return ""

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "") or "/"
        prefix = self._prefix(environ, path)
        if prefix:
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + prefix
            environ["PATH_INFO"] = path[len(prefix):] or "/"
        return self.wsgi(environ, start_response)


app.wsgi_app = _ProxyPrefix(app)


@app.before_request
def _enter_workspace():
    """Put this request in its browser tab's workspace -- see `workspace`.

    The page sends its id as a header on every fetch, and as `?ws=` on the page
    load itself, which no script can add a header to. Neither: the default
    workspace, which is how every caller other than the page has always run.
    """
    workspace.enter(request.headers.get(workspace.HEADER) or request.args.get("ws"))


@app.after_request
def _gzip_json(response):
    """Compress a JSON reply for a client that accepts it.

    Added 2026-10-21, at the user's request: *save time and data transfer of
    log retriving*. The run-log card's payload is ~0.3-0.9 MB of JSON and
    compresses about 10x. Only whole JSON replies: a stream (the ndjson reads)
    has to reach the page line by line, and anything already encoded is left
    alone, so a proxy in front can still compress whatever this does not.
    """
    if (response.status_code != 200 or response.mimetype != "application/json"
            or response.is_streamed or response.direct_passthrough
            or "Content-Encoding" in response.headers
            or "gzip" not in request.headers.get("Accept-Encoding", "").lower()):
        return response
    body = response.get_data()
    if len(body) < 1024:
        return response
    response.set_data(gzip.compress(body, compresslevel=6))
    response.headers["Content-Encoding"] = "gzip"
    response.vary.add("Accept-Encoding")
    return response

# Prepared page images, kept so the browser can show the source beside the text.
# In memory, so MAX_JOBS is a RAM ceiling rather than a history depth -- see
# settings.py.
_jobs = OrderedDict()
_jobs_lock = threading.Lock()

# What each of those jobs was: name, size, origin, detail, page count and case.
# Kept beside the images, and evicted with them, so a re-extraction can be logged
# against the document it belongs to instead of as an anonymous row. Text only --
# a few hundred bytes per job, which is why it is not what MAX_JOBS is sized for.
_job_meta = OrderedDict()
# Which workspace each job belongs to. MAX_JOBS is per workspace, so one user
# reading a stack of files does not push another user's pages out of the compare
# view; JOBS_CEILING bounds the lot, oldest first, whoever they belong to.
_job_owner = {}
JOBS_CEILING = MAX_JOBS * 20


def _evict_job(job_id):
    _jobs.pop(job_id, None)
    _job_meta.pop(job_id, None)
    _job_owner.pop(job_id, None)


def register_job(pages, context=None):
    """Cache PNG-encoded page images and return a job id."""
    job_id = uuid.uuid4().hex[:12]
    encoded = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        encoded.append(buf.getvalue())
    owner = workspace.current_id()
    with _jobs_lock:
        _jobs[job_id] = encoded
        _job_meta[job_id] = context or {}
        _job_owner[job_id] = owner
        mine = [j for j in _jobs if _job_owner.get(j) == owner]
        for evicted in mine[:max(0, len(mine) - MAX_JOBS)]:
            _evict_job(evicted)
        while len(_jobs) > JOBS_CEILING:
            _evict_job(next(iter(_jobs)))
    return job_id


def replace_job_page(job_id: str, index: int, page) -> None:
    """Swap one cached page for the image actually read, while the job is held.

    A loop retry reads a bigger image than the one `register_job` cached, and the
    compare and layout views draw the model's output over the cached one.
    """
    if not job_id:
        return
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    with _jobs_lock:
        pages = _jobs.get(job_id)
        if pages is not None and 0 <= index < len(pages):
            pages[index] = buf.getvalue()


def job_context(job_id: str) -> dict:
    """What the prepared pages under this id were, or {} once they are gone.

    An id whose images have been evicted answers empty rather than raising: a
    re-extraction of a transcript still on screen is worth logging even when the
    page images behind it have fallen out of the cache.
    """
    with _jobs_lock:
        return dict(_job_meta.get(job_id or "", {}))


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


class SweepCancelled(Exception):
    """Raised inside a round when the sweep has been asked to stop.

    Caught by the round loop, which reports a STOP rather than a failure. A
    round nobody let finish says nothing about the settings it ran, and the
    failure count on that summary line is read as if it did.
    """


class Cancel:
    """A stop flag that also hangs up whatever the round is blocked on.

    Duck-types `threading.Event` for `is_set`/`set`, which is all the OCR
    workers ask of the cancel they are handed.

    **What the hang-up buys, and what it measurably does NOT.** Between tokens
    the flag is enough -- `read_page` checks it every chunk. The close is for
    the case the flag cannot reach: a thread blocked in `res.iter_lines()` with
    no token coming, which is a stalled stream, and which this path has no read
    timeout to rescue (deliberately -- a slow first token is normal, see
    `stream_page`).

    **It cannot shorten an Ollama prefill, measured rather than assumed**: on a
    2550x3300 page `requests.post()` itself returned after 221.76s and the first
    SSE line arrived in the same instant, so Ollama withholds the response
    headers until generation begins. During prefill there is no response object
    in existence to hang up on -- the thread is inside `post()`. Two stops
    measured through the app landed 178s and 134s after the button with ZERO
    tokens decoded, which is that, not a flag anyone missed.

    So the honest bound on a Stop is **one prefill**, and the fix for a long one
    is the standing fix for prefill: a lower Detail. What Stop removes is the
    rest of the plan, which was the whole complaint.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._current = None

    def is_set(self) -> bool:
        return self._event.is_set()

    def attach(self, closeable):
        """Hang the stop on this request instead of the one before it.

        **One slot, because a sweep has one read in flight at a time** -- the
        round loop is sequential and a page is read before the next is asked
        for. A list would grow a stale entry per page across a 50-round run for
        no gain, and there is no second live request for it to hold.

        Closing at once when the flag is ALREADY set is the case worth having:
        on a multi-page document the stop can land between the check at the top
        of the page loop and this request going out, and a response attached
        after the stop would otherwise be the one thing nobody hangs up on.
        """
        with self._lock:
            if not self._event.is_set():
                self._current = closeable
                return
        _close_quietly(closeable)

    def set(self):
        with self._lock:
            self._event.set()
            current, self._current = self._current, None
        if current is not None:
            _close_quietly(current)


def _close_quietly(closeable):
    """Hang up, and never let the hang-up itself be the error anyone sees."""
    try:
        closeable.close()
    except Exception:                               # noqa: BLE001
        pass


class MultiCancel(Cancel):
    """A `Cancel` for a run with MANY requests in flight -- the stress test.

    `Cancel.attach` keeps one slot because a sweep has one read in flight at a
    time; a stress run has as many as its concurrency. The live requests are
    held weakly, so a response that has finished and been dropped does not stay
    referenced for the rest of a thousand-job run.
    """

    def __init__(self):
        super().__init__()
        self._live = weakref.WeakSet()

    def attach(self, closeable):
        with self._lock:
            if not self._event.is_set():
                try:
                    self._live.add(closeable)
                except TypeError:                   # not weak-referenceable
                    pass
                return
        _close_quietly(closeable)

    def set(self):
        with self._lock:
            self._event.set()
            live, self._live = list(self._live), weakref.WeakSet()
        for closeable in live:
            _close_quietly(closeable)


def begin_sweep(cancel=None):
    """Mark this thread as running a sweep, and hand back its cancel flag.

    `cancel` lets the stress test bring a `MultiCancel`; the random test takes
    the one-slot default.
    """
    with _sweep_lock:
        thread, _ = _sweep_state()
        if thread is not threading.current_thread():
            workspace.hold()
        cancel = cancel or Cancel()
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


# --------------------------------------------------------------------------
# model server
# --------------------------------------------------------------------------

def llama_status(force: bool = False):
    """Reachability, model and vision capability of the *active* server.

    Which server that is can change at runtime, so this is asked for per request
    rather than resolved once at import. `backends` caches the probe, so the extra
    calls cost nothing.
    """
    return backends.status(force=force)


def chat_url():
    """Chat completions endpoint of the active server."""
    return f"{backends.active_url()}/v1/chat/completions"


_DIGITS = re.compile(r"\d+")


def _counter_loop(text: str) -> bool:
    """Detect an enumerating loop inside one long line, e.g. 'ปี 1 ปี 2 ปี 3 ...'.

    Digits are collapsed so successive iterations compare equal. Restricted to a
    single long line with a short repeating unit, so a table whose rows differ only
    by their amounts -- newline separated, and longer -- is never flagged.

    Only a *word*-bearing unit (`ปี #`) is the enumerating loop this is for. A
    low-information unit -- an empty `<tr><td></td></tr>` row that typhoon emits as
    one un-broken HTML line -- is deliberately NOT flagged here at any count: a
    blank table is legitimate content, and the only thing that separates a bounded
    one from a runaway is total volume, which is `_structural_runaway`'s job, not a
    per-line repeat count. So a blank table of any number of rows passes here.
    """
    for line in text[-LOOP_TAIL_CHARS * 2 :].splitlines():
        if len(line) < LOOP_COUNTER_MIN_LINE:
            continue
        norm = _DIGITS.sub("#", line)
        for unit in range(4, LOOP_COUNTER_MAX_UNIT + 1):
            block = norm[-unit:]
            if not block.strip() or _low_information(block):
                continue
            if norm.endswith(block * LOOP_COUNTER_MIN_REPEATS):
                return True
    return False


# Any Unicode letter, Thai included (\w minus digits and underscore). A repeated
# line with no letter carries no readable word -- it is blank table markup, a rule,
# or a row of identical numbers, all of which large forms produce legitimately.
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# Strip everything that is structure, not content, before looking for a letter:
# complete tags, a backslash-escape (`\n`, `\t` -- the dots layout reply escapes
# the table's newlines into a JSON string, and the `n` of every `\n` would
# otherwise read as a word), and a tag fragment left dangling when a slice cuts
# mid-tag (`...<td` at the end, `td>...` at the start).
_STRUCTURE = re.compile(r"<[^>]*>|\\.|<[^<>]*$|^[^<>]*>")


def _readable(s: str) -> str:
    """`s` with markup, escapes and dangling tag fragments removed."""
    return _STRUCTURE.sub("", s)


def _low_information(s: str) -> bool:
    """A repeated unit that is structure, not content.

    Blank `<tr><td></td></tr>` rows, pipe rules, and rows of identical numbers or
    dashes all carry no word. A stretch of them is what a blank or uniform region
    of a big gridded form looks like, and from the tail alone it is identical to a
    runaway empty-row loop -- so it is held to a much higher repeat count than a
    line containing actual words, which never repeats legitimately. `_readable`
    drops the markup first: a letter inside a `<td>`, or the `n` of an escaped
    `\\n`, is structure, not content, so an empty cell reads as low information.
    """
    return not _LETTER.search(_readable(s))


def _structural_runaway(text: str) -> bool:
    """A long tail carrying no words -- a structural loop whatever its shape.

    The per-line and block checks both key off newlines and a bounded repeat
    count; the dots layout profile defeats them by escaping the whole table onto
    one JSON line (see LOOP_DEAD_TAIL_CHARS). This is the shape-agnostic backstop:
    once this many characters have gone by with almost no letter left after the
    markup is removed, the model is emitting structure, not reading -- a bounded
    blank grid is far shorter than this, so it is a runaway, not a real table.
    """
    tail = text[-LOOP_DEAD_TAIL_CHARS:]
    if len(tail) < LOOP_DEAD_TAIL_CHARS:
        return False
    return len(_LETTER.findall(_readable(tail))) <= LOOP_DEAD_MAX_LETTERS


def looks_repetitive(text: str, tail_chars: int = LOOP_TAIL_CHARS,
                     min_repeats: int = LOOP_MIN_REPEATS) -> bool:
    """True when the tail of the output is cycling on the same block.

    A repeated unit that carries a *word* is a loop at `min_repeats` -- real
    content never repeats verbatim. A **low-information** unit -- an empty table
    row, a rule, a line of identical numbers -- is left to `_structural_runaway`
    and never counted here, so a blank or uniform table region (common on the
    uncapped `original` detail, and legitimate) is not called a loop however many
    rows it runs to. Only a genuinely huge dead run -- far larger than any real
    form's blank grid -- trips, and it trips on volume, not on a row count. See
    `_low_information` and `_structural_runaway`.
    """
    if _counter_loop(text) or _structural_runaway(text):
        return True

    tail = text[-tail_chars:]

    # Whole-line cycling: a consecutive run of identical *substantive* lines -- a
    # loop often trails a stray closing tag. Runs on whatever is available, so a
    # short prose loop is caught as it streams rather than only after 600 chars. A
    # run of blank/structural rows is skipped: it is a table, not a loop.
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    run = 1
    for prev, cur in zip(lines, lines[1:]):
        if cur == prev and len(cur) > 3:
            run += 1
            if run >= min_repeats and not _low_information(cur):
                return True
        else:
            run = 1

    # Block cycling, which also catches loops with no newlines at all -- but it
    # needs a full tail to be meaningful, since a short periodic string is not yet
    # distinguishable from a value that will change on the next token. Anchored at
    # the end, where a loop always is while streaming. Structural blocks are
    # skipped for the same reason as the lines above.
    if len(tail) < tail_chars:
        return False
    for unit in range(4, len(tail) // min_repeats + 1):
        block = tail[-unit:]
        if not block.strip() or _low_information(block):
            continue
        if tail.endswith(block * min_repeats):
            return True
    return False


def image_data_uri(image: Image.Image) -> str:
    # The stress test encodes each prepared page once, before its clock starts,
    # and hangs the result on the image: a page sent a hundred times under load
    # would otherwise pay a PNG encode on this machine a hundred times, and that
    # CPU is not the server being measured. Every other caller sets nothing.
    cached = getattr(image, "_data_uri", None)
    if cached:
        return cached
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Qwen3-VL (typhoon-ocr1.5): 16-px patches merged 2x2, so one visual token per
# 32x32 px. Measured on vLLM: a 1243x1608 page is 2272 prompt tokens, of which
# ~1950 are the image. Used only to keep max_tokens inside a vLLM window.
VLLM_PIXELS_PER_TOKEN = 1024
VLLM_TEXT_MARGIN = 600  # the instruction and the chat template, with room to spare


def tag_page(page: Image.Image, label: str, detail: str) -> None:
    """Name the page for the vLLM DRY processor's log line, e.g. "sol003 p1/2 low".

    The server logs one line per DRY request saying whether DRY fired on it
    (../vllm/podman-ocr-server/dry_logits_processor.py); this tag, sent as
    `vllm_xargs.dry_tag`, is how that line names the document and page.
    """
    page._dry_label = label
    page._dry_tag = f"{label} {detail}"


def vllm_page(image: Image.Image) -> Image.Image:
    """The page as vLLM should get it: enlarged to `VLLM_MIN_PIXELS` if smaller.

    vLLM never enlarges a small page, and the WHT certificates (0.6-1.6 MP) read
    badly as they are -- sol019 looped. See settings.VLLM_MIN_PIXELS.
    """
    floor = settings.VLLM_MIN_PIXELS
    pixels = image.width * image.height
    if not floor or pixels >= floor:
        return image
    scale = (floor / pixels) ** 0.5
    return image.resize((round(image.width * scale), round(image.height * scale)),
                        Image.LANCZOS)


def vllm_page_fields(image: Image.Image, status: dict) -> dict:
    """Request fields that make vLLM read the page the app prepared.

    `mm_processor_kwargs.max_pixels` lifts the server's own image cap to
    `VLLM_MAX_PIXELS`, and `max_tokens` is kept inside the served window: vLLM
    refuses a request whose prompt plus max_tokens exceeds --max-model-len, and a
    4 MP page is ~4,100 prompt tokens against the local container's 8,192.
    """
    fields = {}
    cap = settings.VLLM_MAX_PIXELS
    if cap:
        fields["mm_processor_kwargs"] = {"max_pixels": cap}
    window = backends.model_window(status)
    if window:
        pixels = image.width * image.height
        if cap:
            pixels = min(pixels, cap)
        room = window - -(-pixels // VLLM_PIXELS_PER_TOKEN) - VLLM_TEXT_MARGIN
        fields["max_tokens"] = max(256, min(MAX_NEW_TOKENS, room))
    return fields


def stream_page(image: Image.Image, stats: dict = None,
                profile: str = None, cancel=None):
    """Stream tokens for one page from llama-server.

    `profile` names the pass-1 shape (`prompts.OCR_PROFILES`); omitted, the one
    the process is currently set to. Resolved once here and reported on `stats`,
    so a page reads and logs under the profile it actually ran with even if the
    setting is flipped mid-batch -- the same rule `extract_mode` follows.

    `cancel` is a `Cancel` a random test hands down so its Stop can reach the
    read. It is registered against the response rather than polled, because the
    wait that matters is prefill -- a blocking `iter_lines` with no tokens yet
    to check a flag between. Hanging up raises out of this generator, and the
    caller that asked for the stop is the one that decides that is not an error.
    """
    status = llama_status()
    if not status["available"]:
        raise ValueError(status["reason"])
    spec = profile_spec(profile)
    # Resolved once, like the profile: a read is aborted, or is not, under one
    # rule for the whole of it, whatever the switch does mid-stream.
    guard = loop_guard()
    if stats is not None:
        stats["ocr_profile"] = profile or ocr_profile()
        stats["loop_guard"] = guard

    # Prompt BEFORE image on llama.cpp: it reuses the longest common KV prefix
    # between requests, so with the image first every request differs from token 0
    # and nothing is cacheable. Putting the static ~450-token instruction first
    # lets it be prefilled once and reused for every subsequent page.
    #
    # Ollama gets the conventional image-first ordering instead. There the same
    # trick makes the model ignore the image and recite its own built-in prompt
    # -- see PROMPT_FIRST. Ollama caches prompt prefixes on its own anyway, so
    # nothing is lost.
    prompt_first = PROMPT_FIRST_OLLAMA if status["kind"] == "ollama" else PROMPT_FIRST
    text_part = {"type": "text", "text": spec["prompt"]}
    # The page goes to the server as a base64 PNG inside the JSON body. Its size
    # is reported (`image_bytes`): uploading it, and the server decoding and
    # resizing it, are in no server phase -- see servertime.py.
    vllm = status["kind"] == "vllm"
    dry_tag = getattr(image, "_dry_tag", None)  # vllm_page may return a new image
    if vllm:
        image = vllm_page(image)
    image_uri = image_data_uri(image)
    image_part = {"type": "image_url", "image_url": {"url": image_uri}}
    content = [text_part, image_part] if prompt_first else [image_part, text_part]

    payload = {
        # Ollama gets an explicit system message here; llama.cpp gets none. See
        # backends.system_prefix -- this reproduces what Ollama was injecting from
        # the Modelfile by itself, rather than adding anything new.
        # The profile can veto the system message outright: on dots.ocr an
        # occupied system slot returns two tokens and nothing else, whatever the
        # prompt says. See prompts.OCR_PROFILES.
        "messages": backends.system_prefix(status, spec["system"])
                    + [{"role": "user", "content": content}],
        "max_tokens": MAX_NEW_TOKENS,
        # Greedy by default (see settings.ocr_sampling); OCR_TEMPERATURE=0.1 +
        # OCR_TOP_P=0.6 + OCR_TOP_K=0 is typhoon's recommended setting.
        # min_p stays pinned off so llama-server's default 0.05 cannot apply.
        **ocr_sampling(),
        "min_p": 0.0,
        **sampler_extras(backends.num_ctx()),
        "stream": True,
        # Ollama does not send llama.cpp's `timings`; asking for usage in the
        # final chunk gives an authoritative token count on both servers.
        "stream_options": {"include_usage": True},
        **backends.request_extras(status),
    }
    if vllm:
        payload.update(vllm_page_fields(image, status))

    # The server's own clock, beside ours. Built BEFORE `started` so the vLLM
    # before-snapshot of /metrics is not charged to this page's time. See
    # servertime.py: its prefill, decode, queue and latency, as the server gave them.
    clock = servertime.Clock(status)
    started = time.perf_counter()
    first_at = None
    timings, usage, pieces = {}, {}, 0
    collected, looped = [], False
    # The cap this request was SENT with: on vLLM it is fitted to the window
    # (vllm_page_fields) and can be far below MAX_NEW_TOKENS, so a page that ran
    # to it would never reach MAX_NEW_TOKENS and never be flagged truncated.
    cap = payload.get("max_tokens") or MAX_NEW_TOKENS
    finish = None
    # "Batch requests" on a stress run: this page is one conversation of a
    # vLLM /batch request (vllmbatch.py). Nothing streams, so there is no first
    # token to time and nothing for the loop guard to cut -- the finished text
    # is tested below instead.
    batcher = _batcher_for(chat_url(), status)
    # Names the page in the server's DRY log line (see `tag_page`). Only beside the
    # DRY settings, and only on a request of its own: a /batch body shares one
    # vllm_xargs across its conversations, so a per-page value would stop pages
    # grouping at all (vllmbatch.group_key).
    if vllm and batcher is None and dry_tag and "vllm_xargs" in payload:
        payload["vllm_xargs"] = {**payload["vllm_xargs"], "dry_tag": dry_tag}
    # Where DRY was asked for, the reply's token ids, so whether it fired can be
    # worked out here (dryreplay.py) -- the server can only say so in its own log.
    dry_xargs = (payload.get("vllm_xargs") or {}) if vllm else {}
    dry_ids = None
    if dry_xargs.get("dry_multiplier"):
        payload["return_token_ids"] = True
        dry_ids = []
    if batcher is not None:
        body = _await_batched(batcher, chat_url(), payload, cancel)
        usage = body.get("usage") or {}
        finish = (body.get("choices") or [{}])[0].get("finish_reason")
        returned = (body.get("choices") or [{}])[0].get("token_ids")
        if dry_ids is not None and isinstance(returned, list):
            dry_ids.extend(returned)
        piece = (((body.get("choices") or [{}])[0].get("message") or {})
                 .get("content") or "")
        if piece:
            pieces = 1
            collected.append(piece)
            yield piece
    else:
        with backends.http_post(
            chat_url(), status,
            json=payload,
            stream=True,
            # No read timeout: this is the streaming path, where the gap between
            # tokens is the only thing a timeout could measure and a slow first
            # token is normal. The client disconnecting is what ends it.
            timeout=(GEN_CONNECT_TIMEOUT, None),
        ) as res:
            if cancel is not None:
                cancel.attach(res)
            if res.status_code != 200:
                detail = res.text[:300].strip()
                if res.status_code == 500 and status["kind"] == "llama.cpp":
                    raise ValueError(
                        "llama-server rejected the image (HTTP 500). This normally "
                        "means vision is not enabled -- restart it with "
                        f"--mmproj <mmproj-...gguf>. Server said: {detail}"
                    )
                if res.status_code in (400, 404) and status["kind"] == "ollama":
                    raise ValueError(
                        f"Ollama rejected the request for model '{status['model']}' "
                        f"(HTTP {res.status_code}). Check the model is pulled on "
                        f"{status['url']} and can read images. Server said: {detail}"
                    )
                raise ValueError(
                    f"{status['kind'] or 'model server'} HTTP {res.status_code}: {detail}")

            # Decode the SSE body as UTF-8 explicitly. requests' decode_unicode=True
            # falls back to latin-1 for text/event-stream (no charset in the header),
            # which turns every Thai character into mojibake.
            for raw in res.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    # Read on to the end of the stream rather than break: a
                    # response closed with its last chunk unread closes its
                    # socket, so the keep-alive pool could never reuse it.
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("timings"):
                    timings = obj["timings"]
                if obj.get("usage"):
                    usage = obj["usage"]
                # The usage-only final chunk carries no choices at all.
                choices = obj.get("choices") or [{}]
                finish = choices[0].get("finish_reason") or finish
                if dry_ids is not None and isinstance(choices[0].get("token_ids"), list):
                    dry_ids.extend(choices[0]["token_ids"])
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    if first_at is None:
                        first_at = time.perf_counter()
                    pieces += 1
                    collected.append(piece)
                    yield piece
                    # Check periodically, not per token -- the scan is O(tail^2).
                    if (guard and pieces % LOOP_CHECK_EVERY == 0
                            and looks_repetitive("".join(collected))):
                        looped = True
                        break

    finished = time.perf_counter()
    # With the guard down nothing interrupted the stream, so the tail is tested
    # once, here, on what actually came back. The switch turns off the ABORT and
    # not the detection: a read that cycled the whole way to the token cap is a
    # looped read and has to say so, or it would log `ok` and be averaged as a
    # transcript.
    if (not guard or batcher is not None) and looks_repetitive("".join(collected)):
        looped = True

    clock.timings(timings)
    clock.usage(usage)
    served = clock.finish(finished - started)
    if stats is not None:
        total = finished - started
        # llama.cpp reports authoritative prompt/predict splits; fall back to
        # client-side timing when the build does not send them.
        if timings:
            prefill = timings.get("prompt_ms", 0) / 1000
            decode = timings.get("predicted_ms", 0) / 1000
            new_tokens = int(timings.get("predicted_n", pieces))
        else:
            # Ollama: no prompt/predict split, so time to first token stands in
            # for prefill. Token count comes from usage when the server sent it,
            # since one SSE chunk is not always one token.
            prefill = (first_at - started) if first_at else total
            decode = max(finished - first_at, 1e-9) if first_at else 0
            new_tokens = int(usage.get("completion_tokens") or pieces)
        stats.update(
            # Client-side time to first token, on every backend. `prefill_seconds`
            # is llama.cpp's own prompt_ms where it sends one, which EXCLUDES the
            # time the request waited in the server's queue -- exactly the part a
            # stress test with more requests than slots is measuring.
            ttft_seconds=(None if batcher is not None
                          else round((first_at or finished) - started, 3)),
            new_tokens=new_tokens,
            truncated=finish == "length" or new_tokens >= cap,
            looped=looped,
            resolution=f"{image.width}x{image.height}",
            megapixels=round(image.width * image.height / 1e6, 2),
            seconds=round(total, 2),
            prefill_seconds=round(prefill, 2),
            decode_seconds=round(decode, 2),
            # Decode rate excludes prefill, so it compares cleanly across pages.
            tokens_per_second=round(new_tokens / decode, 2) if decode and new_tokens else 0,
            model=status["model"],
            backend=status["kind"],
            url=status["url"],
            # What the SERVER says this page cost, phase by phase, as it came:
            # latency, queue, prefill, decode. Nothing is derived from the app's
            # clock. None where the server gave no figure (Ollama /v1) or one
            # that cannot be attributed (a /metrics change spanning concurrent
            # requests). See servertime.py.
            **{name: served[name] for name in servertime.FIELDS},
            server_timing=served["source"],
            # What this page sent and got back, counted whether or not the
            # server timed it (servertime.io_tokens). `out` falls back to the
            # app's own count, which every backend has.
            in_tokens=served.get("in_tokens"),
            out_tokens=(served.get("out_tokens") if served.get("out_tokens") is not None
                        else new_tokens),
            image_bytes=len(image_uri),
        )
        if dry_ids is not None:
            # `checked` False: DRY was on, but this server sent no token ids.
            stats["dry"] = (dryreplay.replay(dry_ids, dry_xargs) if dry_ids
                            else {"checked": False})


def _timed_post(url: str, payload: dict, status: dict):
    """`requests.post` for one non-streaming model request, timed by the server
    as well. The record lands in this thread's `servertime` tally, if one is open
    -- which is how an extraction's classify, step, agent and table requests are
    summed without threading a clock through each of them. A request that fails
    or answers non-200 is not timed: the server did no work worth measuring.

    On vLLM the request is streamed instead, so a reply that starts looping can
    be cut off (`_streamed_post`); the caller gets the same shape either way.
    """
    res = _post_once(url, payload, status)
    if res.status_code == 400 and _context_overflow(res.text, status):
        fitted = _fit_max_tokens(payload, status)
        if fitted is not None:
            res = _post_once(url, fitted, status)
    return res


_agentic_chat_context = threading.local()


def _chat_measurement(payload, status, response=None, error=None, seconds=None):
    """One physical chat request. Counts come from its reply, never /metrics.

    No prompts or replies are retained here. A missing usage count is unknown,
    not zero, and a vLLM window is its served max_model_len, not our UI picker.
    """
    body = {}
    if response is not None and response.status_code == 200:
        try:
            body = response.json()
        except (ValueError, TypeError):
            pass
    body = body if isinstance(body, dict) else {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    sent = count(usage.get("prompt_tokens", body.get("prompt_eval_count")))
    output = count(usage.get("completion_tokens", body.get("eval_count")))
    computed, cached = count(timings.get("prompt_n")), count(timings.get("cache_n"))
    if sent is None and computed is not None and cached is not None:
        sent = computed + cached
    if output is None:
        output = count(timings.get("predicted_n"))
    # A loop-aborted stream may contain only a count of text chunks. Those are
    # not tokenizer tokens, so do not present that fallback as measured usage.
    if body.get("loop_aborted") and usage.get("prompt_tokens") is None:
        output = None
    options = payload.get("options") or {}
    if status.get("kind") in ("vllm", "openai"):
        window = backends.model_window(status)
        if window is None:
            known = backends.known(status.get("url")) or {}
            window = backends.model_window({**known, "model": status.get("model")})
        source = "server"
    else:
        window = options.get("num_ctx") or payload.get("n_ctx")
        source = "requested"
    window = count(window) or None
    used = sent + output if sent is not None and output is not None else None
    return {
        "in_tokens": sent, "out_tokens": output, "used_tokens": used,
        "reply_cap": count(payload.get("max_tokens", payload.get("max_completion_tokens",
                                                                 options.get("num_predict")))),
        "context_limit": window, "context_source": source if window else None,
        "used_pct": round(100 * used / window, 2) if used is not None and window else None,
        "seconds": round(seconds, 3) if seconds is not None else None,
        "http_status": response.status_code if response is not None else None,
        "schema": bool(payload.get("format") or
                       (payload.get("response_format") or {}).get("type") == "json_schema"),
        "error": str(error)[:200] if error else "",
    }


def _post_once(url: str, payload: dict, status: dict):
    payload = _with_images(url, payload, status)
    chats = getattr(_agentic_chat_context, "requests", None)
    if chats is None:
        return _post_dispatch(url, payload, status)
    began, response, error = time.perf_counter(), None, None
    try:
        response = _post_dispatch(url, payload, status)
        return response
    except Exception as err:
        error = err
        raise
    finally:
        chats.append({"request": len(chats) + 1, **_chat_measurement(
            payload, status, response, error, time.perf_counter() - began)})


def _post_dispatch(url: str, payload: dict, status: dict):
    with _watched() as progress:
        batcher = _batcher_for(url, status)
        if batcher is not None:
            return _batched_post(batcher, url, payload, status)
        if _guards_extraction(url, payload, status):
            return _streamed_post(url, payload, status, progress)
        return _plain_post(url, payload, status)


def _plain_post(url: str, payload: dict, status: dict):
    clock = servertime.Clock(status)
    res = backends.http_post(url, status, json=payload, timeout=GEN_TIMEOUT)
    # The whole reply is in, so the prefill certainly is: release anything
    # waiting on this request's prefix being cached (see `_prefilled`).
    _prefilled()
    if res.status_code == 200:
        try:
            clock.body(res.json())
        except ValueError:
            pass
        clock.finish()
    return res


# The stress test's batcher, while a run with "Batch requests" on holds the
# sweep guard; None otherwise. Process-wide like the prompt-cache bypass, and for
# the same reason: the guard means the stress run is the only thing sending.
# See vllmbatch.py.
_batcher = None


def _batcher_for(url: str, status: dict):
    """The batcher to send this request through, or None to send it alone.

    vLLM only (`serves_metrics`, the server's own statement), and only the
    chat endpoint, the one shape vLLM's /batch route takes.
    """
    if workspace.get('user_pipeline', False):
        return None
    batcher = _batcher
    if (batcher is None or not backends.serves_metrics(status)
            or not vllmbatch.batchable(url)):
        return None
    return batcher


def _send_batch(url: str, body: dict) -> dict:
    """`send` for a `Batcher` -- one POST to vLLM's batch route, parsed. Only a
    vLLM server is ever batched, so the request goes on the pooled connection."""
    res = backends.http_post(url, {"kind": "vllm"}, json=body, timeout=GEN_TIMEOUT)
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


def _await_batched(batcher, url: str, payload: dict, cancel=None) -> dict:
    """This request's own reply out of a batch. Raises `BatchError` on failure,
    and on Stop -- `cancel` hangs the waiter up like a stream."""
    ticket = batcher.submit(url, payload)
    if cancel is not None:
        cancel.attach(ticket)
    return ticket.result()


def _batched_post(batcher, url: str, payload: dict, status: dict):
    """`_plain_post`, through the batch: the same reply shape, the same tally."""
    clock = servertime.Clock(status)
    try:
        body = _await_batched(batcher, url, payload)
    except vllmbatch.BatchError as err:
        _prefilled()
        return _Reply(err.status_code or 502, text=str(err))
    _prefilled()
    clock.body(body)
    clock.finish()
    return _Reply(200, body)


_OVERFLOW = re.compile(r"maximum context length|max_tokens|max_completion_tokens"
                       r"|context window", re.I)


def _context_overflow(text: str, status: dict) -> bool:
    """Is this vLLM 400 the prompt plus `max_tokens` not fitting the window?"""
    return bool(VLLM_FIT_MAX_TOKENS and status and status.get("kind") == "vllm"
                and _OVERFLOW.search(text or ""))


def _fit_max_tokens(payload: dict, status: dict):
    """The payload with `max_tokens` lowered to what the window has room for.

    vLLM rejects a request whose prompt plus `max_tokens` is longer than
    `--max-model-len` outright, so a long transcript with the usual 4096-token
    cap fails before a token is generated. `/tokenize` counts the prompt with
    the server's own template, and the request is asked once more with the room
    that is left. The reply may then stop at the lower cap, which reads exactly
    as any reply that ran to its cap. None where there is not room for
    `VLLM_FIT_MIN_TOKENS`, or the server cannot count -- the 400 then stands.
    """
    count, window = backends.count_prompt(status, payload)
    if count is None or window is None:
        return None
    room = window - count - 8          # a margin for the template's last tokens
    asked = payload.get("max_tokens") or payload.get("max_completion_tokens") or 0
    if room < VLLM_FIT_MIN_TOKENS or (asked and room >= asked):
        return None
    fitted = dict(payload, max_tokens=room)
    if "max_completion_tokens" in fitted:
        fitted["max_completion_tokens"] = room
    config.say(f"[extract] prompt is {count} tokens of a {window}-token window; "
               f"asking again with max_tokens {room} instead of {asked}")
    return fitted


# A callback the agentic fan-out hangs on its first step: called once, from the
# worker thread asking that step, as soon as the server has prefilled the prompt.
_prefill_hook = threading.local()


def _prefilled():
    hook = getattr(_prefill_hook, "cb", None)
    if hook is not None:
        _prefill_hook.cb = None
        hook()


# Who is told that a pass-2 request is still making progress: the stress test's
# stall alert. Until 2026-10-19 that alert heard only when a job's whole
# extraction returned, so a multi-document file -- classify, split, one
# extraction per document, table agents -- read as "stalled" under load while
# every one of its requests was moving. `cb(event, request, what)`, with `event`
# "begin", "chunk" (a streamed line arrived) or "end"; `request` is an object
# unique to one request and `what` names it ("step parties"), from `_asking`.
# Thread-local like `_prefill_hook`; `_carry_progress` takes it into the agentic
# step threads, which a thread-local alone would never reach.
_progress = threading.local()


def _progress_swap(cb):
    """Install `cb` as this thread's progress hook. Returns the one it replaced."""
    previous = getattr(_progress, "cb", None)
    _progress.cb = cb
    return previous


def _carry_progress(fn):
    """`fn`, set to run with this thread's progress hook -- for handing to a thread.
    The page images of an image-fed extraction (`_feeding_images`) go with it."""
    cb = getattr(_progress, "cb", None)
    what = getattr(_progress, "what", None)
    images = getattr(_progress, "images", None)
    if cb is None and images is None:
        return fn

    def run(*args, **kwargs):
        previous, previous_what = _progress_swap(cb), getattr(_progress, "what", None)
        previous_images = getattr(_progress, "images", None)
        _progress.what, _progress.images = what, images
        try:
            return fn(*args, **kwargs)
        finally:
            _progress_swap(previous)
            _progress.what, _progress.images = previous_what, previous_images
    return run


# An image-fed extraction (2026-10-27, the stress test's image mode): no page is
# read, and every pass-2 request this thread makes -- classify, the single
# request, each agentic step and its retries -- carries the page images at the
# head of its user message (`_with_images`, in `_post_once`). The prompts are
# the text ones unchanged; the transcript they wrap is `prompts.IMAGE_TRANSCRIPT`.
# The checks that need a transcript to check against stand down while it is on:
# grounding re-asks, the classifier's heading test, the item table, blame.


@contextmanager
def _feeding_images(uris):
    """Send these page images (data URIs) with every pass-2 request in the block.

    Never wrapped round a `yield`, for the reason `_asking` gives."""
    previous = getattr(_progress, "images", None)
    _progress.images = list(uris) or None
    try:
        yield
    finally:
        _progress.images = previous


def _image_fed() -> bool:
    """Whether this thread's extraction is reading page images, not a transcript."""
    return bool(getattr(_progress, "images", None))


def _with_images(url: str, payload: dict, status: dict) -> dict:
    """`payload` with this thread's page images in its last user message, or as it is.

    Images first, then the text: every request of one extraction then opens with
    the same images and the same prefix, so a server's prefix cache still covers
    what the agentic steps share. Ollama's native `/api/chat` (a schema retry)
    takes base64 beside a string; everything else takes OpenAI content parts. On
    vLLM the server's own image cap is lifted as a read lifts it
    (`vllm_page_fields`), so the extractor sees the Detail the run chose.
    """
    images = getattr(_progress, "images", None)
    if not images or not isinstance(payload.get("messages"), list):
        return payload
    messages = [dict(m) for m in payload["messages"]]
    user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if user is None or not isinstance(user.get("content"), str):
        return payload
    if url.endswith("/api/chat"):
        user["images"] = [uri.split(",", 1)[-1] for uri in images]
    else:
        user["content"] = ([{"type": "image_url", "image_url": {"url": uri}} for uri in images]
                           + [{"type": "text", "text": user["content"]}])
    out = {**payload, "messages": messages}
    if (status or {}).get("kind") == "vllm" and settings.VLLM_MAX_PIXELS:
        out.setdefault("mm_processor_kwargs", {"max_pixels": settings.VLLM_MAX_PIXELS})
    return out


@contextmanager
def _asking(what: str):
    """Name the requests made inside this block, for the stall alert.

    Never wrapped round a `yield`: a generator can resume on another thread,
    and the name would stay behind on this one.
    """
    previous = getattr(_progress, "what", None)
    _progress.what = what
    try:
        yield
    finally:
        _progress.what = previous


def _no_progress():
    pass


@contextmanager
def _watched():
    """One request, reported to the progress hook. Yields the call to make on
    each streamed line; "end" is sent however the request finishes."""
    cb = getattr(_progress, "cb", None)
    if cb is None:
        yield _no_progress
        return
    request, what = object(), getattr(_progress, "what", None)
    cb("begin", request, what)
    try:
        yield lambda: cb("chunk", request, what)
    finally:
        cb("end", request, what)


# Pass-2 requests this process cut off for looping, since it started. Read by the
# live monitor, which shows the server's own abort count beside it.
_loop_aborts = {"count": 0, "tokens": 0, "last": None}
_loop_aborts_lock = threading.Lock()


def loop_aborts() -> dict:
    with _loop_aborts_lock:
        return dict(_loop_aborts)


def _guards_extraction(url: str, payload: dict, status: dict) -> bool:
    """Is this pass-2 request one to stream and cut off if it loops?

    vLLM only, on the server's own statement (`serves_metrics`): a vLLM request
    whose client hangs up is aborted server-side, which is what makes cutting it
    worth anything, and no measurement in this project was taken there. Only the
    OpenAI-compatible endpoint -- the one shape vLLM is sent.
    """
    return bool(EXTRACT_LOOP_ABORT and loop_guard()
                and backends.serves_metrics(status)
                and url.endswith("/v1/chat/completions")
                and not payload.get("stream"))


_ENTRY = re.compile(r'\{\s*"label"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
                    r'\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def _entries_cycling(text: str, max_unit: int = 12) -> bool:
    """True when the reply is ENDING in a repeated sequence of list entries.

    The last k entries (1 <= k <= 12, label and value both) equal the k before
    them, three times running, over at least `EXTRACT_CYCLE_MIN_CHARS`. Anchored
    at the end because that is where a loop is while it streams.

    Deliberately not `_repeated_list` ("one entry three times, anywhere"). That
    is a fine diagnosis of a reply that has already failed and a bad reason to
    cut a live one: the model writes table rows into `other_fields` under their
    column heading, so a heading -- even a heading with the same value, three
    charge rows sharing one billing period -- repeats once per row.
    """
    found = list(_ENTRY.finditer(text))
    entries = [m.groups() for m in found]
    n = len(entries)
    for k in range(1, max_unit + 1):
        if n < 3 * k:
            break
        unit = entries[n - k:]
        reps = 1
        while n >= (reps + 1) * k and entries[n - (reps + 1) * k:n - reps * k] == unit:
            reps += 1
        if reps >= 3 and found[-1].end() - found[n - reps * k].start() >= EXTRACT_CYCLE_MIN_CHARS:
            return True
    return False


def extraction_looping(text: str) -> bool:
    """The test a streamed pass-2 reply is cut off on.

    Two halves: the wide extraction window of `looks_repetitive` -- the test
    `_why_unparsable` already diagnoses a finished reply with -- and a sequence of
    list entries cycling at the end (`_entries_cycling`), which catches the
    `other_fields` loop the window cannot hold. Not tried before the reply is as
    long as the window: a short reply cannot hold enough of either to be a loop.

    A false cut is not free -- entries after it are lost -- so both halves are
    the strict kind. Measured on 453 real replies, 13 of them long enough to be
    tested: none is cut.
    """
    if len(text) < EXTRACT_LOOP_TAIL_CHARS:
        return False
    return (looks_repetitive(text, tail_chars=EXTRACT_LOOP_TAIL_CHARS,
                             min_repeats=EXTRACT_LOOP_MIN_REPEATS)
            or _entries_cycling(text))


class _Reply:
    """A finished reply assembled from a stream, in the non-streaming shape, so
    every caller of `_timed_post` reads one shape whichever way it was fetched."""

    def __init__(self, status_code: int, body: dict = None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


def _streamed_post(url: str, payload: dict, status: dict, progress=_no_progress):
    """One pass-2 request, streamed, and closed the moment it starts cycling.

    Closing the connection is the whole of the termination: vLLM aborts a request
    whose client has gone and frees its slot and KV cache. What arrived before the
    cut is returned as a reply that stopped at its cap (`finish_reason: length`),
    so the callers' salvage and diagnosis run exactly as on a loop that ran to the
    cap -- `_why_unparsable` names it a loop -- and the body carries
    `loop_aborted` for anything that wants to know it was cut rather than run out.
    `progress` is called on every line that arrives (the stall alert's clock).
    """
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    clock = servertime.Clock(status)
    collected, pieces, usage, finish, looped = [], 0, {}, None, False
    with backends.http_post(url, status, json=body, stream=True,
                            timeout=GEN_TIMEOUT) as res:
        if res.status_code != 200:
            _prefilled()
            return _Reply(res.status_code, text=res.text)
        for raw in res.iter_lines():
            if not raw:
                continue
            # vLLM sends nothing until the first token exists, so the first line
            # of the body means the prompt has been prefilled -- and cached.
            _prefilled()
            progress()
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                # Read on to the end of the stream rather than break: a
                # response closed with its last chunk unread closes its
                # socket, so the keep-alive pool could never reuse it.
                continue
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choice = (obj.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                collected.append(piece)
                pieces += 1
                if (pieces % LOOP_CHECK_EVERY == 0
                        and extraction_looping("".join(collected))):
                    looped = True
                    break
    # Leaving the `with` closed the connection; on a break that is the abort.
    text = "".join(collected)
    reply = {"choices": [{"message": {"content": text},
                          "finish_reason": "length" if looped else finish}],
             "usage": usage or {"completion_tokens": pieces}}
    if looped:
        reply["loop_aborted"] = True
        with _loop_aborts_lock:
            _loop_aborts["count"] += 1
            _loop_aborts["tokens"] += pieces
            _loop_aborts["last"] = time.time()
        config.say(f"[extract] cut off a looping reply from {status.get('model')} after "
            f"~{pieces} tokens; vLLM aborts it and frees the slot")
    clock.usage(usage)
    clock.finish()
    return _Reply(200, reply, text)


def _first_json_object(text: str):
    """Pull the first balanced {...} out of a reply, ignoring braces in strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _why_unparsable(raw: str, err: Exception, truncated: bool, status: dict):
    """(message, flags) for a reply that would not parse.

    Split out so the same diagnosis serves both outcomes: it is the error when
    nothing could be kept, and the warning printed above the fields when
    `_salvage_json` rescued part of the reply. A parse error here is nearly
    always a reply that was cut off rather than malformed syntax, and "invalid
    JSON" sends you looking at the parser instead of at the model.
    """
    # Wider window than the OCR stream uses: an extraction loop repeats a whole
    # clause inside one JSON string, so the repeating unit is long and a
    # 600-character tail cannot hold enough repeats of it to be recognised.
    if looks_repetitive(raw, tail_chars=EXTRACT_LOOP_TAIL_CHARS,
                        min_repeats=EXTRACT_LOOP_MIN_REPEATS):
        return ("The model repeated itself while extracting and never closed the "
                "JSON.", {"looped": True})
    if _is_ocr_envelope(raw):
        return (f"'{status['model']}' ignored the extraction prompt and transcribed "
                'the document instead, returning its OCR {"natural_text": ...} '
                "envelope. The transcript is long enough to drown the instructions. "
                "Re-extract; if it persists, extract one page at a time.",
                {"envelope": True})
    if _repeated_list(raw):
        return ("The model cycled over the same rows until it hit the "
                f"{EXTRACT_MAX_TOKENS}-token cap, so the JSON never closed. Raising "
                "the cap will not help -- it buys more of the loop. Try Re-extract, "
                "or extract one page at a time.", {"looped": True})
    if truncated:
        return (f"The JSON was cut off mid-value at the {EXTRACT_MAX_TOKENS}-token "
                "cap. Raise EXTRACT_MAX_TOKENS if the document really has this many "
                "line items.", {"truncated": True})
    return (f"model did not return valid JSON ({err})", {})


def _salvage_json(raw: str):
    """Parse a reply that was cut off mid-value, keeping what completed.

    A looping extraction is not a wasted request. sol005 fills twenty scalars
    and the first rows of the table correctly and *then* cycles inside
    `other_fields` until the token cap, and the whole reply is currently thrown
    away for want of a closing brace. This rewinds to the last element that
    finished, closes the containers still open there, and parses that.

    What it deliberately does not do is guess: nothing is invented to fill the
    tail, the caller marks the result partial, and the verbatim reply is kept so
    the drop is visible. Same rule as grounding -- say less, never make it up.

    Returns the parsed object, or None when there is nothing whole to keep.
    """
    start = raw.find("{")
    if start < 0:
        return None
    stack, cut, cut_stack = [], None, None
    in_str = escaped = False
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            # A container just closed, so everything up to here is whole. Cut
            # AFTER it, keeping the element that finished.
            cut, cut_stack = i + 1, list(stack)
        elif ch == "," and stack:
            # A comma separates two elements, so everything before it is whole.
            # Cut BEFORE it: what follows is the fragment that never finished.
            cut, cut_stack = i, list(stack)
    if not stack or cut is None or not cut_stack:
        return None
    try:
        parsed = json.loads(raw[start:cut] + "".join(reversed(cut_stack)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repeated_list(raw: str, threshold: int = EXTRACT_REPEAT_THRESHOLD) -> bool:
    """True when the same list entry was emitted over and over.

    `looks_repetitive` misses this: the model cycles over a *set* of entries
    rather than one block, so no single repeating unit spans the tail. Comparing
    the entries catches it, and a genuine document never lists one of them three
    times over.

    Both lists in the schema are checked, not just the table. sol005 cycles a
    seven-entry block through `other_fields` -- with only "description" counted
    nothing fired, and the reply was reported as "cut off at the token cap, raise
    EXTRACT_MAX_TOKENS", which is advice that buys more loop.
    """
    seen = {}
    for text in re.findall(r'"(?:description|label)"\s*:\s*"([^"]{12,})"', raw):
        seen[text] = seen.get(text, 0) + 1
        if seen[text] >= threshold:
            return True
    return False


def _is_ocr_envelope(raw: str) -> bool:
    """True when a reply is an OCR model's built-in {"natural_text": ...} answer.

    Checked on the raw text rather than on parsed fields, because the envelope is
    just as recognisable when it was cut off before it could be parsed at all.
    """
    return raw.lstrip().lstrip("`json \n").startswith('{"natural_text"')


# How many lines from the top of a transcript to look in for the heading. The
# heading is not always near the head of the page -- sol002 prints it eighteen
# lines down, under two letterheads and a logo block -- and this is the same
# claim `prompts._EP_HEAD`'s doc-type rule makes, applied to the search.
# How far apart two lines may be and still be one heading, and how much less of a
# heading the second may be. Both deliberately tight: 1 means literally the next
# printed line, and the slack lets a bilingual pair differ slightly in coverage
# without letting a half-matched caption in.
_TYPE_HEADING_GAP = 1
_TYPE_HEADING_SLACK = 0.15

_TYPE_SCAN_LINES = 40

# A heading is a short line. A paragraph that happens to contain the words
# "tax invoice" is body text, and classifying on it would pick a form off a
# sentence in the small print.
_TYPE_SCAN_MAX_CHARS = 80


def classify_transcript(text: str):
    """(codes, heading, confidence, detail) -- what the page calls itself.

    The heading is returned with the codes because it is the EVIDENCE for them:
    it is reported on the result, and it is what rule 4 below demands of the
    model before it will believe an answer.

    Deterministic and model-free -- it is `normalise.document_types`, a table of
    needles, run line by line over the top of the page. **Nothing is asked of the
    model to decide this**, which matters more here than usual: the answer
    chooses the FORM the model is then asked to fill, so a classification the
    model made would be a schema the model chose for itself.

    A LIST, because a document is regularly more than one type at once -- a page
    headed ใบเสร็จรับเงิน/ใบกำกับภาษี is a receipt and a tax invoice, and both
    forms apply. Returning one and discarding the rest is the bug this signature
    exists to prevent.

    An empty list is a real answer. It gets the default field set, which is wider
    than any typed one -- guessing a type to pick a narrower form would drop
    Mandatory fields on a coin flip.

    `confidence` is `normalise.heading_confidence` of the line the answer came
    off, and it is what decides whether the model is asked at all: above
    `CLASSIFY_MIN_CONFIDENCE` this answer stands on its own and no request is
    made. `detail` is its working, so the number can be taken apart.
    """
    best = ([], "", 0.0, {})
    scored = []                     # (line number, codes, line, confidence, detail)
    for number, line in enumerate((text or "").splitlines()[:_TYPE_SCAN_LINES], 1):
        line = line.strip()
        if not line or len(line) > _TYPE_SCAN_MAX_CHARS:
            continue
        codes = normalise.document_types(line)
        if not codes:
            continue
        confidence, detail = normalise.heading_confidence(line, number)
        scored.append((number, codes, line, confidence, detail))
        # **The BEST-scoring line wins, not the first** (2026-09-02). First-wins
        # was the old rule and it existed to stop a credit note's body text --
        # which regularly names the invoice it credits -- making the page an
        # invoice as well. Scoring answers that better than order does: a line
        # that IS a heading scores 1.00 and a sentence mentioning one scores
        # 0.23-0.30, so the real heading wins even where the mention comes first.
        # Ties keep the earlier line, which is the old behaviour where the score
        # cannot separate two lines.
        if confidence > best[2]:
            best = (codes, line, confidence, detail)

    # **A heading printed on two lines is ONE heading** (2026-09-04). sol001 heads
    # itself `ใบแจ้งหนี้` and `STATEMENT OF ACCOUNT` on consecutive lines, each a
    # complete heading scoring 1.00 and each naming a different type. Taking the
    # better of the two is a coin toss between two right answers, and it threw
    # away the half that mattered: the document resolved to
    # STATEMENT_OF_ACCOUNT alone, which no requirement covers, so it was asked
    # for the 30-key default form and NOTHING it returned could be scored. The
    # manifest has both, and until now the manifest was the only thing that did.
    #
    # So an ADJACENT line that also classifies, and that is as much a heading as
    # the winner, is unioned into it. Adjacency is the whole of the safety: a
    # bilingual heading pair is printed together, while the body text naming
    # another document -- the case best-scoring was introduced for -- is lines
    # away and scores far lower anyway. Both tests have to pass.
    if best[0]:
        winner = next(e for e in scored if e[2] == best[1] and e[1] == best[0])
        merged, lines = list(best[0]), [best[1]]
        for number, codes, line, confidence, _detail in scored:
            if line == winner[2]:
                continue
            if abs(number - winner[0]) > _TYPE_HEADING_GAP:
                continue
            # As much a heading as the winner is, not merely classifiable: a
            # partial line beside a real heading is a caption or a reference,
            # and this must not let one in through the back door.
            if confidence < best[2] - _TYPE_HEADING_SLACK:
                continue
            new_codes = [c for c in codes if c not in merged]
            if new_codes:
                merged.extend(new_codes)
                lines.append(line)
        if len(merged) > len(best[0]):
            # Reported in the type order the schema reads in, and with both lines
            # as the evidence -- the heading is what the model's answer is checked
            # against, and half of a two-line heading would fail a check the page
            # actually passes.
            merged = [c for c in normalise.DOCUMENT_TYPES if c in merged]                      + [c for c in merged if c not in normalise.DOCUMENT_TYPES]
            best = (merged, " / ".join(lines), best[2], best[3])
    return best


def _classify_with_model(text: str, status: dict):
    """(codes, heading) from the model: which kind of document this transcript is.

    **The classifier for a real document since 2026-09-02**, ahead of the printed
    heading table rather than behind it. What that buys is the case the table
    cannot cover: a page heading itself in wording nobody has written a needle
    for, which used to take the DEFAULT field set -- every key of every
    requirement at once, the widest form here and the most fill pressure a small
    model is ever put under.

    **It still does not choose the schema it is then asked to fill**, which is
    the objection `normalise.py`'s header spends three paragraphs on, and the
    difference is the guards below: the answer is a code from a closed list plus
    a heading that has to be ON the page, so it is a reading of printed text that
    Python then checks -- not a judgement taken on trust. An answer that fails
    either test is discarded whole and the heading table answers instead.

    Two guards, and the second is the one that makes this safe to run first:

    * the codes are filtered to the vocabulary it was offered, so an invented
      category cannot reach `DOC_TYPE_FIELDS`;
    * **the heading it quotes must actually be printed on the page**, checked
      with `grounding.squash` the same way every extracted value is checked. An
      answer that cannot point at a heading is thrown away whole -- codes and
      all -- because the codes were supposedly read off that heading.

    Returns ([], "") on anything unusable, which sends `resolve_doc_types` on to
    the heading table -- the pre-2026-09-02 answer, so a refusal costs the run
    nothing but the request. Never raises, for the same reason.
    """
    if not CLASSIFY_WITH_MODEL or not (text or "").strip():
        return [], ""
    message = (CLASSIFY_PROMPT.format(vocabulary=classify_vocabulary())
               + text[:CLASSIFY_MAX_CHARS])
    try:
        url, payload = backends.structured_request(
            backends.system_prefix(status)
            + [{"role": "user", "content": message}],
            None, CLASSIFY_MAX_TOKENS, status)
        with _asking("classify"):
            res = _timed_post(url, payload, status)
        if res.status_code != 200:
            return [], ""
        raw, _, _ = backends.structured_reply(res.json(), status)
        answer = json.loads(_first_json_object(strip_fence(raw)) or raw)
        if not isinstance(answer, dict):
            return [], ""
        # Exactly two keys are read, and everything else in the reply is
        # dropped on the floor. **A number the model volunteered is never taken**
        # -- not a confidence, not a score, not a probability. It would be a
        # token sequence about a token sequence with nothing behind it, and it
        # would sit on the page beside figures that were all computed from
        # something re-derivable. The heading is checked against the page below;
        # that check IS the confidence, and it is Python's.
        heading = str(answer.get("heading") or "").strip()
        codes = [c for c in (answer.get("types") or [])
                 if isinstance(c, str) and c in TYPE_NAMES]
    except Exception:
        # Including a JSON error: this is one small request whose whole purpose
        # is to narrow a form, and there is a correct answer to give without it.
        return [], ""
    if not codes or not heading:
        return [], ""
    squashed = grounding.squash(heading)
    # Read off page images there is no printed text to hold the heading to; the
    # codes are still held to the vocabulary above.
    if not _image_fed() and (not squashed or squashed not in grounding.squash(text)):
        # It named a heading the page does not print, so it read nothing -- the
        # codes are a guess about a document it did not look at. The heading
        # table gets the question instead.
        return [], ""
    # Deduped in the vocabulary's own order, so two spellings of one answer and
    # two orderings of one pair of codes produce the same form.
    return [c for c in TYPE_NAMES if c in set(codes)], heading


def _needs_second_opinion(codes, confidence):
    """Why the table's answer is not good enough on its own, or "".

    Two reasons, and they are different failures of a needle match:

    * **not enough of the line is the heading** -- the coverage gate, which
      catches a document kind mentioned in passing in a sentence;
    * **more than one type** -- which coverage CANNOT catch, because a second
      type costs only a few characters on a line the first type already fills.
      A page that merely cites another document (`ใบลดหนี้ อ้างอิงใบกำกับภาษี
      123`) scores like a heading and comes back as two types, one of them
      wrong -- and a wrong type widens the form, adds rules the document is not
      held to, and frames both prompts with a lie about what it is reading.

    The second rule escalates legitimate slash headings too, and knowingly:
    `ใบเสร็จรับเงิน/ใบกำกับภาษี` really is both, and six of the ten fixtures are
    multi-type. That is the price of the certainty, and on this corpus the model
    returns the same types, so it buys the certainty for a request rather than
    for an answer.
    """
    if not codes:
        return ""
    if confidence < CLASSIFY_MIN_CONFIDENCE:
        return f"the heading scored {confidence:.0%}, under {CLASSIFY_MIN_CONFIDENCE:.0%}"
    if CLASSIFY_ESCALATE_MULTI and len(codes) > 1:
        return ("the heading names more than one type, and one of them may be a "
                "reference to another document")
    return ""


def resolve_doc_types(text: str, case_id: str = None, doc_type=None,
                      status: dict = None, pages=None):
    """(codes, how, heading) -- what this document is, and on whose authority.

    In order of confidence, the same shape as `scoring.case_for_upload`'s rules
    and for the same reason: `how` is reported, so a run never has to be trusted
    about a choice it cannot show its working for.

      1. the caller said so
      2. **the Python heading table, where it is sure** -- above
         `CLASSIFY_MIN_CONFIDENCE` AND naming one type, and then no request is
         made at all
      3. the model, where a status is given
      4. the manifest, for a benchmark case
      5. the Python table's answer below the bar, rather than nothing
      6. nothing -- the default set

    **Rules 2 and 3 are the user's own design** (2026-09-02): *use code to
    classify but if its confidence is more than 90% ... if not, let llm do it*.
    Cheap and certain first, the model where the page is not obvious. The whole
    of what makes it work is that the certainty is MEASURED rather than assumed
    -- `normalise.heading_confidence` scores how much of the matched line is
    actually the heading, so `ใบเสร็จรับเงิน/ใบกำกับภาษี` on its own line scores
    1.00 and a sentence mentioning a tax invoice scores 0.23.

    **A heading the table cannot place scores nothing at all, so it goes to the
    model** -- which is the case the model was promoted for on the same day, and
    the one this ordering preserves. What changed is only that a page whose
    heading is unmistakable no longer pays a request to be told so.

    **The manifest and the low-confidence answer are both still under the model**
    rather than in front of it: a fixture must exercise the path a real document
    takes, which is why the manifest stopped answering first earlier today.

    **The manifest is still checked, and a disagreement is REPORTED rather than
    resolved** -- `_field_set` puts the manifest's own answer on the form as
    `doc_type_expected` whenever the two differ. The manifest is a person's
    reading and the model's is a machine's; when a measured number moves, that
    field is what says which of the two the run was built on.

    Rule 2 is the only one that costs a request -- one short one, on the first
    `CLASSIFY_MAX_CHARS` of the transcript, and about half a second warm.
    `_classify_with_model` has what it refuses to believe, and it refuses by
    falling through rather than by failing: a classification that cannot be
    trusted leaves the run exactly where it would have been without it.

    **The manifest outranks the page**, which is the point of having it: a
    heading can name a type a person disagrees with, and the manifest is where
    that correction lives. sol001 is the standing example -- it prints ใบแจ้งหนี้
    in Thai and STATEMENT OF ACCOUNT in English, so the classifier reports both,
    and the manifest is what settles which reading this project uses.

    `doc_type` accepts a string or a list; a string is wrapped, so a caller with
    one type in hand does not have to know about the plural.

    `heading` is the line the answer was read off, where one was: it is what the
    page says about itself, and a classification nobody can check against the
    page is exactly what the model's rule refuses to accept.

    The fourth return value is that heading's confidence, or **None** where the
    answer came from a person and there is no page evidence to score -- the
    caller and the manifest. None, never 0.0: "nobody measured this" and "this
    measured zero" are different claims, and only the second is a bad heading.
    """
    if doc_type:
        codes = [doc_type] if isinstance(doc_type, str) else list(doc_type)
        return codes, "caller", "", None, ""
    read, heading, confidence, _ = classify_transcript(text)
    why = _needs_second_opinion(read, confidence)
    if read and not why:
        return read, "transcript", heading, confidence, ""
    if status:
        codes, quoted = _classify_with_model(text, status)
        if codes:
            # **The model may ADD a requirement, never take the last one away**
            # (2026-09-04). Both answers are readings of the same printed
            # heading and both have been checked against the page, so where the
            # model's names only types no requirement covers and the table's
            # names one that does, the two are unioned rather than the table's
            # being dropped.
            #
            # It is one-directional on purpose, and the asymmetry is the whole
            # of why it is safe: a type with no requirement contributes no key
            # and no Mandatory field, so adding the table's answer can only ever
            # narrow the form and give the document something to be scored on.
            # Losing it does the opposite, silently -- sol001 resolved to
            # STATEMENT_OF_ACCOUNT alone, was asked for the 30-key default form,
            # and NOTHING it returned could be scored, which is indistinguishable
            # on the page from an extraction that found nothing.
            if read and not mandatory_for_types(codes) and mandatory_for_types(read):
                codes = list(codes) + [c for c in read if c not in codes]
                quoted = " / ".join(x for x in (quoted, heading) if x)
            return (codes, "model", quoted,
                    normalise.heading_confidence(quoted)[0], why)
    if case_id:
        # The manifest's answer for THIS document, where the case holds more
        # than one -- see `case_doc_types`. A pack's entry for the pages in
        # hand, or the file's own list where it holds one document.
        named = case_doc_types(case_id, pages)
        if named:
            return named, "case", "", None, why
    if read:
        # Under the bar and used anyway, because the alternative is the default
        # form -- the union of every requirement, the widest here and the most
        # fill pressure a small model is ever put under. A weak reading of the
        # page beats no reading; the confidence rides on the result and says so.
        return read, "transcript", heading, confidence, why
    return [], "unclassified", "", None, why


def case_doc_types(case_id: str, pages=None) -> list:
    """What a person says this case is -- for the whole file, or for one document.

    **A benchmark case is not always one document** (2026-09-08). sol015 is
    seven, of five different kinds, and one `doc_types` for the file would be
    seven wrong answers or one right one and six wrong. A case that holds more
    than one carries `documents` in the manifest -- pages and types per document
    -- and this reads the entry the given pages fall in.

    `pages` empty asks about the FILE, which is what a caller with no segment in
    hand means, and it gets the top-level list: document 1's types on a pack,
    because that is the document the truth file describes.

    An entry that covers none of the given pages returns [] rather than falling
    back to the file's answer. A pack whose manifest was edited to cover four of
    its five documents is a manifest that has nothing to say about the fifth,
    and saying so lets the classifier answer instead of putting document 1's
    types on it.
    """
    case = scoring.cases_index().get(case_id) or {}
    documents = case.get("documents") or []
    if pages and documents:
        for entry in documents:
            if pages[0] in (entry.get("pages") or []):
                return list(entry.get("doc_types") or [])
        return []
    return list(case.get("doc_types") or [])


def manifest_types(case_id: str, pages=None) -> list:
    """What a person says this benchmark case is, or []. Never the classifier.

    Read separately from `resolve_doc_types` so the manifest can be a CHECK on an
    answer it no longer gets to give: since the model classifies first, the only
    thing that can say a fixture was built on a different form from the one its
    truth file was written against is the manifest, held beside the answer rather
    than in front of it.

    `pages` names the document being classified, on a case that holds more than
    one -- without it the check would hold document 1's types against every
    document of a pack and report six disagreements that are not.
    """
    return case_doc_types(case_id, pages)


# --------------------------------------------------------------------------
# stage 0a: how many DOCUMENTS are in the file (2026-09-08)
# --------------------------------------------------------------------------
#
# Ahead of the type classifier, because it decides what that classifier is
# given: a three-page file holding three documents is three classifications and
# three forms, and asking one form of all three pages puts three issue dates and
# three totals in front of eleven keys. `segment.py` holds the rules and the
# reasoning; what is here is the model escalation and the reporting, which is
# the same shape `resolve_doc_types` already has one stage down.


def _page_digest(pages) -> str:
    """The head of each page, numbered, for the boundary question.

    The top of a page is what answers it -- the letterhead, the heading, the
    document number, a `Page 2 of 2`. Sending whole pages would put a multi-page
    document's every table row into a request that reads none of them, and the
    cap is small for the same reason `CLASSIFY_MAX_CHARS` is.
    """
    return "\n\n".join(
        "--- page %d ---\n%s" % (number, (page or "").strip()[:SEGMENT_MAX_CHARS])
        for number, page in enumerate(pages, 1))


def _segment_with_model(pages, status: dict):
    """The page groups the model reads in this file, or None.

    **Asked only where Python could not tell** -- two pages of the same kind,
    each with its own heading, neither numbered. Every other boundary in
    `segment.py` is a reading of what the page prints, and a request that could
    only ever disagree with printed text can only make the answer worse.

    The answer is checked before it is believed, by `segment.valid_groups`, and
    the check is the whole of why this is safe to run at all: every page appears
    exactly once, every group is consecutive, nothing is out of range. A
    grouping that loses a page would drop it from the run silently, and one that
    repeats a page would extract it twice under two documents' names.

    Nothing else in the reply is read -- no type, no confidence, no reason. The
    types are resolved per document afterwards by the classifier that already
    exists, and a number the model volunteered is never taken here any more than
    it is there.

    Returns None on anything unusable, which leaves Python's own answer standing
    -- so a refusal costs the run the request and nothing else. Never raises.
    """
    if not SEGMENT_WITH_MODEL or len(pages) < 2:
        return None
    message = prompts.SEGMENT_PROMPT.format(count=len(pages)) + _page_digest(pages)
    try:
        url, payload = backends.structured_request(
            backends.system_prefix(status)
            + [{"role": "user", "content": message}],
            None, SEGMENT_MAX_TOKENS, status)
        with _asking("split"):
            res = _timed_post(url, payload, status)
        if res.status_code != 200:
            return None
        raw, _, _ = backends.structured_reply(res.json(), status)
        answer = json.loads(_first_json_object(strip_fence(raw)) or raw)
        if not isinstance(answer, dict):
            return None
        return segment.valid_groups(answer.get("documents"), len(pages))
    except Exception:
        return None


class _SegmentChat:
    """The rolling boundary walk: one short question per unsettled boundary.

    **Why this exists when the one-shot above does not answer it.**
    `_segment_with_model` puts a digest of every page in one request, so its
    prompt grows with the FILE -- which is what `SEGMENT_MAX_PAGES` is, and why
    a longer file used to be read as one document instead. Here the question is
    turned round: the pages arrive one at a time and the document so far is the
    conversation, so the prompt is bounded by the length of a DOCUMENT and a
    file of any length is a walk of short requests.

    **The reset is the design, not an economy.** What is being asked is whether
    this page continues THIS document; the pages of the one before it are not
    context for that, they are contamination -- and without the reset document 7
    would be judged partly against document 1's parties and totals. So `opened`
    throws the memory away and starts again from the page that opened the
    document.

    **A conversation is not memory the server holds.** Both backends are
    stateless: every turn is resent, so what this class calls memory is a list
    rebuilt into each request. That is what `SEGMENT_CHAT_WINDOW` is for -- an
    unbounded one grows with the document until it overruns `num_ctx` on exactly
    the long files this was built for. The opening page is always kept and the
    middle drops out, because what identifies a document is its first page and
    its most recent one.

    **Python's decisions go in as USER turns, never as assistant ones.** A page
    `segment.py` settled still has to be in the memory, or the next question is
    asked against a document with holes in it -- but writing Python's reading as
    something the model said would make the conversation a record of a thing
    that did not happen, and the next answer would be conditioned on it.

    Three guards, the same three the classifier has and for the same reasons:

    * the reply is read for exactly two keys and nothing else -- **no number is
      taken**, here as anywhere;
    * **the evidence it quotes must be printed on the page it was asked about**,
      checked with `grounding.squash` like every extracted value. An answer that
      cannot point at the page is discarded whole and Python's reading stands;
    * a refusal costs the walk that one boundary and nothing else -- it never
      raises and never stops the walk.
    """

    def __init__(self, status: dict):
        self.status = status
        # The current document's conversation. Thrown away by `opened`.
        self._first = None
        self._turns = []
        # Reported: a walk that asked nothing and a walk that was refused
        # everything reach the same answer by different roads.
        self.asked = 0
        self.answered = 0
        self.refused = 0
        self.capped = False

    # -- the memory ------------------------------------------------------

    def _digest(self, page: str) -> str:
        return (page or "").strip()[:SEGMENT_MAX_CHARS]

    def opened(self, number: int, page: str):
        self._first = {
            "role": "user",
            "content": prompts.SEGMENT_CHAT_PROMPT.format(
                number=number, digest=self._digest(page))}
        self._turns = []

    def kept(self, number: int, page: str, why: str):
        self._turns.append({
            "role": "user",
            "content": prompts.SEGMENT_CHAT_KEPT.format(
                number=number, why=why, digest=self._digest(page))})

    def _messages(self, number: int, page: str):
        """The opening page, the tail of the document, and the question.

        The opening turn is never dropped: it is what the document IS, and a
        window that let it fall off would leave the model comparing a page
        against the middle of a table it can no longer identify.
        """
        window = max(SEGMENT_CHAT_WINDOW - 1, 1)
        return (backends.system_prefix(self.status)
                + [self._first]
                + self._turns[-window:]
                + [{"role": "user",
                    "content": prompts.SEGMENT_CHAT_ASK.format(
                        number=number, digest=self._digest(page))}])

    # -- the question ----------------------------------------------------

    def decide(self, seg, number: int, page: str, guess: bool, why: str):
        """(new, why) from one question, or None to leave Python's reading."""
        if self._first is None:
            return None
        if self.asked >= SEGMENT_CHAT_MAX_ASKS:
            self.capped = True
            return None
        self.asked += 1
        answer = self._ask(number, page)
        if answer is None:
            self.refused += 1
            return None
        same, evidence = answer
        self.answered += 1
        # The reply goes back into the memory unedited: this turn is the one
        # thing in the conversation the model actually said, and rewriting it
        # would condition the next answer on words it did not use.
        self._turns.append({
            "role": "assistant",
            "content": json.dumps({"same": same, "evidence": evidence},
                                  ensure_ascii=False)})
        return (not same,
                "the model reads page %d as %s, quoting %s"
                % (number, "the same document" if same else "a new document",
                   json.dumps(evidence, ensure_ascii=False)))

    def _ask(self, number: int, page: str):
        """(same, evidence) from one request, or None. Never raises."""
        try:
            url, payload = backends.structured_request(
                self._messages(number, page), None, SEGMENT_MAX_TOKENS,
                self.status)
            with _asking(f"split, page {number}"):
                res = _timed_post(url, payload, self.status)
            if res.status_code != 200:
                return None
            raw, _, _ = backends.structured_reply(res.json(), self.status)
            reply = json.loads(_first_json_object(strip_fence(raw)) or raw)
            if not isinstance(reply, dict):
                return None
            if not isinstance(reply.get("same"), bool):
                return None
            evidence = str(reply.get("evidence") or "").strip()
            # **The quote has to be on the page it was asked about.** This is
            # the guard that makes a sequential walk safe to act on: an answer
            # that can point at the page is a reading of printed text, and one
            # that cannot is a guess wearing a citation. Checked against the
            # whole page rather than the digest the model was shown -- the
            # digest is a window on the page, and a correct quote from just past
            # its edge is still a correct quote.
            if not evidence:
                return None
            if grounding.squash(evidence) not in grounding.squash(page or ""):
                return None
            return bool(reply["same"]), evidence
        except Exception:
            return None

    def note(self) -> str:
        """What the walk cost, for the reason the split reports anything."""
        if not self.asked:
            return "every boundary was read off the pages"
        parts = ["%d boundary question(s), %d answered"
                 % (self.asked, self.answered)]
        if self.refused:
            parts.append("%d refused" % self.refused)
        if self.capped:
            parts.append("stopped at SEGMENT_CHAT_MAX_ASKS=%d"
                         % SEGMENT_CHAT_MAX_ASKS)
        return ", ".join(parts)


def resolve_segments(text: str, pages=None, status: dict = None):
    """`(segments, how)`, without the page-by-page progress. See `_segment_stream`."""
    return _drain(_segment_stream(text, pages, status))


def _segment_stream(text: str, pages=None, status: dict = None):
    """`(segments, how)` -- the documents in this file, and on whose authority.

    `how` is reported for the reason `doc_type_from` is: a run that came back as
    three documents has to be able to say what split it, and a split nobody can
    check is one nobody should take on trust.

      * `rules` -- `segment.py` read it off the pages, and was sure
      * `model` -- a boundary was a guess and the model was asked, about the
        whole file in one request
      * `chat` -- the file was too long for that, so it was WALKED: one question
        per unsettled boundary, carrying the document so far
      * `guess` -- a boundary was a guess, the model was not asked or would not
        answer, and Python's own reading stands
      * `whole` -- not split at all: one page, the setting off, or too many pages

    `pages` is the read's own page list where the caller still has it. Falling
    back to `segment.split_pages` is not a lesser path -- it is the only one a
    truth-fed run has -- but the list is exact and the parse is a re-derivation,
    so the list wins where there is one.

    **A one-page file returns one segment and never asks anything**, which is
    eleven of the thirteen fixtures and very nearly every real upload.
    """
    pages = [p for p in (list(pages) if pages is not None
                         else segment.split_pages(text))]
    if len(pages) < 2:
        return segment.one(pages or [text], classify_transcript, text,
                           "the file is one page"), "whole"
    if not SEGMENT_DOCUMENTS:
        return segment.one(pages, classify_transcript, text,
                           "splitting is off (SEGMENT_DOCUMENTS=0)"), "whole"
    walking = len(pages) > SEGMENT_MAX_PAGES
    if walking and not (SEGMENT_CHAT and status):
        # No model server, or the walk is off. Read as one rather than split
        # badly, and it says so -- remembering that one document is one form,
        # asked once, filled from however many pages this file has.
        return segment.one(pages, classify_transcript, text,
                           "the file has %d pages, over SEGMENT_MAX_PAGES=%d"
                           % (len(pages), SEGMENT_MAX_PAGES)), "whole"
    # Too long for the one-shot question, whose prompt carries a digest of every
    # page, is **walked instead** (2026-09-16): one short question per unsettled
    # boundary, the document so far as the conversation, and the memory reset at
    # every cut -- so the prompt is bounded by the length of a document rather
    # than of the file.
    chat = _SegmentChat(status) if walking else None
    # Every multi-page file reports its grouping page by page, walked or not
    # (2026-09-16, at the user's request: *see how many pages remain, what the
    # current session is at, and view the grouped pages*). On a short file the
    # steps arrive in a burst and cost nothing; on a long one they are the only
    # thing that says the run is moving between the last page read and the first
    # field extracted.
    yield {"event": "segmenting", "pages": len(pages), "walk": walking}
    # The type-change gate is the classifier's own bar, CLASSIFY_MIN_CONFIDENCE:
    # a change of type the table is not sure of is a guess, and a guess is what
    # gets asked about. See `segment._boundary`.
    #
    # **The walk only.** Measured on qwen3.5:9b: gated on a SHORT file, one
    # unsure type change sends the whole file to the one-shot question, which
    # regroups every page and overruled correct splits -- sol015 and sol022 both
    # came back wrong where the rules alone are right. The walk asks about the
    # one boundary in doubt and nothing else, so the gate is safe there.
    steps = segment.walk(pages, classify_transcript, whole=text, ask=chat,
                         min_confidence=(CLASSIFY_MIN_CONFIDENCE
                                         if walking and SEGMENT_TYPE_GATE else 0.0))
    while True:
        try:
            step = next(steps)
        except StopIteration as done:
            segments = done.value
            break
        # The head of the page, which is what the walk itself was shown -- a
        # reader verifying a group sees the text the decision was taken on. The
        # whole page is already on screen as the transcript; this is bounded at
        # SEGMENT_MAX_CHARS so a hundred-page walk stays a few tens of KB.
        yield {"event": "segment_page", **step,
               "remaining": step["pages"] - step["page"],
               "head": (pages[step["page"] - 1] or "").strip()[:SEGMENT_MAX_CHARS],
               **({"asked_total": chat.asked, "answered_total": chat.answered,
                   "refused_total": chat.refused} if chat else {})}
    if walking:
        # `chat` only where a question actually settled something. A walk
        # nobody answered IS Python's own reading and must not claim otherwise
        # -- the same rule `doc_type_from` follows, and the reason a refusal
        # here is reported rather than swallowed.
        return segments, ("chat" if chat.answered else "guess")
    if not segment.uncertain(segments):
        return segments, "rules"
    if status:
        groups = _segment_with_model(pages, status)
        if groups:
            return segment.apply_groups(pages, groups, classify_transcript), "model"
    return segments, "guess"


def _segment_record(seg: dict, index: int, total: int) -> dict:
    """One document's place in the file, in the shape the page draws it.

    Without the page texts: this travels to the browser on an event and rides on
    the result, and the transcript is already there once. The same rule that
    keeps the transcript out of the run log.
    """
    return {"document": index, "documents": total,
            "pages": list(seg["pages"]),
            "page_range": segment.page_range(seg),
            "doc_types": list(seg["codes"]),
            "heading": seg.get("heading") or "",
            "start_reason": seg.get("start_reason") or "",
            # Whether the boundary that opened this document is a reading of the
            # page or a guess. A guess that the model was then asked about is
            # not one any more -- `apply_groups` opens its segments certain --
            # so this reads false only where nobody could settle it.
            "certain": bool(seg.get("certain", True)),
            "joins": list(seg.get("joins") or [])}


def _field_set(text: str, case_id: str = None, doc_type=None, status=None,
               pages=None, extras=None):
    """Everything downstream needs about one document's form, in one dict.

    **Resolved once per extraction and handed to everything**: the prompt's
    shape, the prompt's TEXT, the JSON grammar, the step table, what grounding
    calls missing, what `validate` checks and what `fieldscore` scores all come
    off this dict, so no two of them can disagree about which document this is.

    `keys` and `mandatory` are the UNION over every type the document names -- a
    page that is both a credit note and a tax invoice must answer for both, and
    an intersection would let each type excuse the other's obligations.
    `doc_type` is the primary one, for a label and nothing else.

    `extras` is whether `other_fields` is asked for; None takes the switch.
    """
    codes, how, heading, confidence, escalated = resolve_doc_types(
        text, case_id, doc_type, status, pages)
    # What the manifest says, where it says anything and the answer above came
    # from somewhere else. Present ONLY on a disagreement, so the field is a
    # finding rather than a row of metadata: a fixture whose form was chosen by
    # the model against a person's reading is the one case where a moved field
    # score has an explanation that is not the extractor's.
    expected = manifest_types(case_id, pages) if case_id and how != "case" else []
    if sorted(expected) == sorted(codes):
        expected = []
    return {"doc_types": codes,
            "doc_type": normalise.primary_type(codes),
            "doc_type_from": how,
            "doc_type_heading": heading,
            "doc_type_expected": expected,
            # How much of the line the answer was read off is actually a heading,
            # 0-1, or None where the answer came from a person (the caller, the
            # manifest) and there is no page evidence to score. **None, never 0**
            # -- the standing rule: "nobody measured this" and "this measured
            # zero" are different claims.
            "doc_type_confidence": confidence,
            # Why the table's own answer was not taken on its own, where it was
            # not: the coverage gate, or more than one type. Empty where it was
            # taken, so the field is a reason rather than a status -- and it is
            # what makes a `model` answer legible, since "the model classified
            # this" does not say what the cheap path thought first.
            "doc_type_escalated": escalated,
            "keys": list(fields_for_types(codes)),
            "mandatory": list(mandatory_for_types(codes)),
            # No requirement covers any type in play, so nothing below is
            # Mandatory and the form is the base field set -- the union of every
            # requirement, which is what `fields_for_types` falls back to. It is
            # a flag rather than a thing every reader infers from an empty
            # `mandatory`, because three of them word themselves differently for
            # it: the Classify box says so, `fieldscore` scores the base set and
            # marks the rate `unknown_type`, and the round table stops calling
            # its denominator "required".
            #
            # **It changes what is SCORED and nothing about compliance.**
            # `mandatory` stays empty, so `validate` demands nothing and no key
            # is marked REQUIRED on the page -- demanding a key of a document no
            # table covers is a claim this project has no authority to make.
            "unknown_type": not mandatory_for_types(codes),
            # The row shape of a TABLE this form asks for, or []. Only the
            # withholding certificate asks for one, and it is here rather than
            # folded into `keys` because a row is not a key: the prompt, the
            # grammar, the step table, grounding and `validate` each need it as
            # its own thing, and it is exactly the sort of second answer that
            # drifts from the first if two callers work it out separately.
            "items": list(items_for_types(codes)),
            # Of that table, the cells the requirement marks Mandatory. Its own
            # entry rather than folded into `mandatory` for the reason `items` is
            # its own entry: a row's obligation is not a key's, and `validate`
            # reports it against a path no key list could match.
            "mandatory_items": list(mandatory_items_for_types(codes)),
            # Whether `other_fields` is asked for: the single prompt and grammar
            # and the agentic `other` step all read it off here.
            "extras": extract_extras() if extras is None else bool(extras)}


def _classified(form: dict, mode: str) -> dict:
    """What stage 0 decided, in the shape the page draws it.

    The run's middle stage made visible: the type, the authority behind it, the
    line it was read off, and -- the half that actually matters to a reader --
    **what the answer then bought**, which is the form, the steps and the
    validation rules that follow from it. A type on its own is trivia; a type
    that explains why there are eleven rows below it and not sixteen is the
    reason the stage exists.

    Sent as an event AND attached to the result. Both, because they answer at
    different times: the event is what the page has while the extraction is
    still running, and the result is what a re-render an hour later has. One
    function so the two cannot drift.
    """
    codes = form["doc_types"]
    return {
        "doc_types": list(codes),
        "doc_type": form["doc_type"],
        "doc_type_from": form["doc_type_from"],
        "doc_type_heading": form.get("doc_type_heading") or "",
        # The manifest's own answer, on a benchmark case where it differs from
        # the one this run used. Empty on every other run, and empty on
        # agreement -- it is a disagreement, not a field.
        "doc_type_expected": list(form.get("doc_type_expected") or []),
        # How much of the line this was read off is a heading, or None where a
        # person gave the answer and there is nothing on the page to score.
        "doc_type_confidence": form.get("doc_type_confidence"),
        # Why the model was asked, where it was.
        "doc_type_escalated": form.get("doc_type_escalated") or "",
        "type_phrase": type_phrase(codes),
        "fields": list(form["keys"]),
        "mandatory": list(form["mandatory"]),
        # Whether any requirement covers this document at all. The page says so
        # beside the field count, because "30 fields" with no required clause
        # otherwise reads as a form nobody can pass rather than as one nobody
        # has written a requirement for yet.
        "unknown_type": bool(form.get("unknown_type")),
        # The cells of the table this form asks for, or []. On the strip beside
        # the field count, because "eleven fields" and "fourteen fields and a
        # table" are different amounts of work and the difference is invisible
        # in a count of scalars.
        "items": list(form["items"]),
        # Which of those cells the requirement demands, so the page can mark a
        # column heading rather than work the requirement out in JavaScript.
        "mandatory_items": list(form["mandatory_items"]),
        # Named only for the shape that walks them. Single mode asks one
        # question, and reporting the steps it did not take would describe a run
        # that did not happen.
        "steps": ([s["id"] for s in steps_for_types(codes, form.get("extras", True), mode=mode)]
                  if mode in ("agentic", "hybrid") else []),
        "extras": form.get("extras", True),
        "rules": list(rules_for_types(codes)),
        # Whether the answer reached the PROMPT or only the field list. It is
        # per mode and a setting can turn it off, so a reader comparing two runs
        # has to be able to see which of them was told.
        "framed": bool(_framing(form, mode)),
    }


def _extract_single(text: str, status: dict, form: dict) -> dict:
    """The whole schema in one request: all 29 scalars and both lists at once.

    The cheaper of the two shapes, and the one a fresh process starts in.
    `_extract_agentic` below asks for the same schema a few fields at a time;
    `extract_fields` picks between them.

    Asked twice at most, and the second time is a different question rather than
    the same one repeated: the retry constrains decoding to `EXTRACT_JSON_SCHEMA`
    instead of to "any valid JSON", which makes the model's own transcript
    envelope unreachable rather than merely discouraged.

    **The retry only runs when the first attempt yielded nothing at all**, and
    that restraint is measured, not caution. Constraining every request costs
    accuracy where the plain one already worked: on sol002 it took grounded
    values from 28 to 19 and invented a table row; on sol003 it filled five rows
    of unit prices the page never printed. Every baseline in CLAUDE.md was taken
    on the unconstrained request, and it stays the first thing asked.
    """
    result = _extract_once(text, status, None, form)
    if "fields" in result or not EXTRACT_SCHEMA:
        return result
    # Constrained to THIS document's form, not to the union of every form: a
    # grammar listing a key the prompt never asked for is an invitation to fill
    # it, which is the note `prompts.extract_schema` carries.
    retry = _extract_once(text, status,
                          extract_schema(form["keys"], form["items"],
                                         form.get("extras", True)), form)
    retry["seconds"] = round(result.get("seconds", 0) + retry.get("seconds", 0), 2)
    if "fields" not in retry:
        # Both failed. The first reply is the one to report -- it is the one the
        # baselines were measured on -- but say the constrained retry failed too,
        # so the next person does not spend an hour re-inventing it.
        result["error"] += (" A retry constrained to the field schema failed as "
                            f"well: {retry['error']}")
        result["seconds"] = retry["seconds"]
        return result
    retry["schema_retry"] = True
    # Both replies are kept, labelled the way the agentic steps label theirs. The
    # first one is why the second was asked -- usually the OCR envelope, or a
    # string that never closed -- and returning only the reply that worked throws
    # away the one worth reading, which is the same defect `_ask_step`'s `collect`
    # list was built to fix.
    retry["raw"] = ("--- " + _reply_label("single", 0, 0, failed=True) + " ---\n"
                    + (result.get("raw") or "") + "\n\n"
                    + "--- " + _reply_label("single", 0, 1) + " ---\n"
                    + (retry.get("raw") or ""))
    return retry


def _framing(form: dict, mode: str):
    """The codes this mode's prompt is framed with -- the form's, or none.

    () is the pre-2026-09-01 prompt exactly, with the form untouched, which is
    what makes an A/B here mean anything.

    **It takes the mode because the two modes measured opposite**, and the
    default is on for single and off for agentic. That is not a hedge: the rules
    this framing carries are about telling one key from another, and agentic
    already asks two or three keys at a time, so it has far less of the confusion
    they fix and pays the tokens anyway. CLAUDE.md has the 210 runs.
    """
    on = TYPE_FRAMING_SINGLE if mode == "single" else TYPE_FRAMING_AGENTIC
    return form["doc_types"] if on else ()


def _extract_once(text: str, status: dict, schema, form: dict) -> dict:
    """One extraction request, parsed, salvaged if it was cut off, and grounded.

    `schema` constrains decoding to those keys where the server supports it and
    is None for the plain "return some JSON object" request. Everything else --
    the prompt, the sampler, the diagnosis of an unusable reply -- is the same
    either way, which is what makes the two attempts comparable.
    """
    # Named rather than inlined so the reply can be shown beside the exact
    # message that produced it. Instructions, transcript, instructions again --
    # the closing repeat is what stops an OCR fine-tune answering with its own
    # transcript envelope, and seeing that on screen is half of why it is there.
    # "single" because this function IS single mode's one request -- agentic
    # never reaches here, it goes through `_step_question`.
    message = (extract_prompt(form["keys"], _framing(form, "single"),
                              TYPE_FRAMING_BULLETS, form["items"], form.get("extras", True))
               + text + EXTRACT_REMINDER)
    url, payload = backends.structured_request(
        backends.system_prefix(status)
        + [{"role": "user", "content": message}],
        schema, EXTRACT_MAX_TOKENS, status)

    started = time.perf_counter()
    try:
        with _asking("extraction"):
            res = _timed_post(url, payload, status)
        if res.status_code != 200:
            return {"error": f"{status['kind'] or 'model server'} HTTP "
                             f"{res.status_code}: {res.text[:200]}"}
        body = res.json()
        raw, truncated, _ = backends.structured_reply(body, status)
    except Exception as err:
        return {"error": f"extraction request failed: {err}"}

    elapsed = round(time.perf_counter() - started, 2)
    candidate = _first_json_object(strip_fence(raw)) or raw
    partial = None
    try:
        fields = json.loads(candidate)
    except json.JSONDecodeError as err:
        why, flags = _why_unparsable(raw, err, truncated, status)
        # A cut-off reply is not a wasted request. The model usually fills the
        # scalars correctly and only *then* starts cycling inside a list, and
        # returning nothing throws away twenty right answers to punish the tail.
        # Not attempted for the OCR envelope: what completed there is a
        # transcript, and salvaging it would put a page of prose into a field.
        salvaged = None if flags.get("envelope") else _salvage_json(raw)
        if not salvaged or all(grounding.is_blank(v) for v in salvaged.values()):
            return {"error": why, **flags, "raw": raw[:2000], "prompt": message,
                    "seconds": elapsed}
        fields = salvaged
        partial = (why + " The fields below are what the reply had finished "
                         "before it stopped; the rest are empty because it never "
                         "got to them, not because the page is silent.")
    if not isinstance(fields, dict):
        return {"error": "model returned JSON that is not an object",
                "raw": raw[:2000], "prompt": message, "seconds": elapsed}
    # Valid JSON, wrong JSON: a short document's envelope closes inside the cap and
    # would otherwise be stored and scored as if it were the extracted fields.
    if _is_ocr_envelope(raw):
        return {"error": f"'{status['model']}' returned a transcript in its OCR "
                         '{"natural_text": ...} envelope, not the requested fields. '
                         "Re-extract; if it persists, extract one page at a time.",
                "envelope": True, "raw": raw[:2000], "prompt": message,
                "seconds": elapsed}
    if not form.get("extras", True):
        # Not asked for, so not kept if the model wrote some anyway: the list
        # is empty, as it is on an agentic run without the `other` step.
        fields["other_fields"] = []

    _, _, tokens = backends.structured_reply(body, status)
    return {
        "fields": fields,
        # Set when the reply was rescued from a cut-off body rather than parsed
        # whole. Carried as its own key, not folded into the fields: every count
        # on the page -- missing, tiers, grounded ratio -- is computed over keys
        # the model never reached, and reading those as "the document does not
        # state this" is exactly the mistake this string exists to prevent.
        "partial": partial,
        # The reply exactly as it arrived, kept on the success path too and not
        # only on the error paths below. A field can be missing because the model
        # never wrote it or because the fence-stripping and first-object salvage
        # above cut it off, and only the verbatim text tells those apart.
        "raw": raw,
        # And the message that produced it. A reply is only readable against the
        # question, and every prompt in this project is a measurement -- the one
        # actually sent is worth being able to check against `prompts.py` without
        # a server log.
        "prompt": message,
        # Every value traced back to the transcript it came from. The prompt asks
        # the model not to invent; this is the part that checks, because a
        # plausible invented value is indistinguishable from a read one on screen.
        "grounding": grounding.check(fields, text, form["keys"],
                                     form["mandatory"], form["items"]),
        # Coverage by delivery tier, the same counts the run log records. Sent so
        # the page can say it in one line instead of leaving you to count filled
        # rows; it is deliberately NOT part of the grounding dict, because how
        # many fields came back is a different question from whether they are real.
        "tiers": grounding.tier_counts(fields, form["keys"]),
        # Which of these figures already carry tax, decided in Python from the
        # figures just extracted. It rides on the extraction result because the
        # Fields tab labels the totals from it: "Amount ex VAT" is a plain lie on
        # a VAT-inclusive page, and a reader taking `subtotal` for a pre-tax
        # figure would be out by the VAT.
        "vat_basis": verify.vat_basis(fields, text),
        # The normalisations the field requirement asks for -- a standard document
        # type, branch codes, tax IDs reduced to digits, the period split, and the
        # references as a list. Worked out in Python from the values just copied,
        # and kept OUT of `fields` on purpose: `grounding.check` walks that dict
        # and would report every one of these as an invention, which is the one
        # thing they provably are not.
        "derived": normalise.derive(fields, text),
        "seconds": elapsed,
        "tokens": tokens,
        "model": status["model"],
        "backend": status["kind"],
        # Which server answered. Pass 2 may have an endpoint of its own
        # (`backends.select_extract_url`), and a run is attributed to the server
        # that ran it, never to the one selected afterwards.
        "url": status.get("url") or "",
        # Which shape produced this. On screen and in the run log, because the two
        # cost very different amounts of wall clock and fill the schema in
        # different ways -- a row without it cannot be compared with one beside it.
        "mode": "single",
    }


# --------------------------------------------------------------------------
# pass 2, agentic: the same schema, one to three fields per request
# --------------------------------------------------------------------------
#
# One request for 31 keys asks a 2B model to hold the whole page and the whole
# form in mind at once, and what it does under that load is fill a key from
# whatever is nearest rather than from the label that names it -- sol005's
# Location Code arriving as buyer_name. A step names two or three labels and asks
# for two or three keys, so there is much less for a value to land in by mistake.
#
# Three things fall out of doing it this way, and they are the reason it exists
# rather than side effects:
#
# * The transcript leads every step's message and only the question at the end
#   changes, so llama.cpp prefills the document once and reuses it. The steps
#   after the first cost their question and their answer.
# * A step that comes back with a value which is not in the transcript can be
#   asked again with that value quoted back at it, for the price of a short
#   question -- see EXTRACT_STEP_RETRY. Grounding stops being a report at the end
#   and becomes something the extraction can act on while it is running.
# * A step that fails to parse costs that step's fields. In single mode the same
#   failure costs the whole extraction.
#
# It is not free: ~15 requests instead of 1, and the schema-wide instructions
# ("never write one value into two of these fields") can only be enforced within a
# step, not across them. Which is why the mode is a switch rather than the default.


def _step_schema(step: dict) -> dict:
    """The JSON schema for one step: its own keys and nothing else.

    Derived from the step's `keys` rather than written out beside its skeleton,
    because a schema that disagreed with the skeleton would be a silent bug and
    there is no way to keep twenty hand-written pairs honest. The step's
    skeleton stays the readable statement of what is asked for; this is the same
    thing in the form the sampler can be held to.
    """
    return {
        "type": "object",
        "properties": {
            step.get("request_aliases", {}).get(key, key): (LINE_ITEM_SCHEMA if key == "line_items"
                  else income_items_schema(prompts.INCOME_ITEM_KEYS)
                  if key == INCOME_ITEMS_KEY
                  else OTHER_FIELDS_SCHEMA if key == "other_fields"
                  else {"type": "string"})
            for key in step["keys"]
        },
        "required": [step.get("request_aliases", {}).get(key, key) for key in step["keys"]],
    }


def _chat(content: str, max_tokens: int, status: dict, schema: dict = None):
    """One non-streaming text request. Returns (raw reply, truncated, tokens).

    Sampling is set exactly as the single-shot pass sets it -- greedy, the reply
    constrained to `schema`, no DRY -- so the two modes differ in the shape of
    the prompt and in nothing else, which is what makes a comparison between them
    mean anything. Raises on transport failure and on a non-200, so a step can
    record why it produced nothing instead of silently contributing empty fields.
    """
    url, payload = backends.structured_request(
        backends.system_prefix(status) + [{"role": "user", "content": content}],
        schema, max_tokens, status)
    res = _timed_post(url, payload, status)
    if res.status_code != 200:
        raise ValueError(f"{status['kind'] or 'model server'} HTTP "
                         f"{res.status_code}: {res.text[:200]}")
    return backends.structured_reply(res.json(), status)


def _step_prefix(text: str) -> str:
    """The part of every step's message that is the same for all of them.

    Split out so it can be *shown* once rather than fifteen times, and so nothing
    can drift: `_step_message` is these two functions joined, which is what the
    model is actually sent.
    """
    return EXTRACT_STEP_PREFIX + text


def _step_question(step: dict, codes=(), mode="agentic") -> str:
    """The part that differs: this step's title, skeleton and rules.

    `codes` names the document's type in the task line and adds the type's rules
    for the keys THIS step owns -- never for the others, which is what keeps a
    rule aimed at one step out of the fourteen answers it was not aimed at. It
    goes in the question rather than in the shared prefix for the same reason
    `EXTRACT_REMINDER` closes the single-shot message: what is said after the
    transcript is what the model is still holding when it answers.
    """
    task = HYBRID_STEP_TASK if mode == "hybrid" else EXTRACT_STEP_TASK
    return task.format(title=step["title"],
                                    context=type_line(codes),
                                    skeleton=step["skeleton"],
                                    rules=step_rules(step, codes,
                                                     TYPE_FRAMING_BULLETS))


def _step_message(text: str, step: dict, codes=(), mode="agentic") -> str:
    """The message for one step: prefix, transcript, then this step's question.

    Assembled in that order on purpose. The transcript first is what makes the
    prefix shared across every step and therefore cacheable; the instructions last
    is what keeps an OCR fine-tune from answering with its own transcript envelope,
    which is the same thing EXTRACT_REMINDER buys the single-shot prompt.
    """
    return _step_prefix(text) + _step_question(step, codes, mode)


def _extras_only(step: dict) -> bool:
    """True where a step owns `other_fields` and nothing else.

    The one thing that decides whether a broken reply from this step is an
    ERROR or merely a short answer, so it is a test on the keys rather than a
    step id: `other_fields` is the overflow valve for everything the fourteen
    priority-1 keys do not cover, and a page has as many extra fields as the
    model decides it has. Nothing downstream rules on them -- `fieldscore`
    keeps them out of the headline because their labels are the model's own
    wording -- so an entry lost off the end of that list costs no measurement.

    Deliberately NOT "every key this step owns is a list". A step owning
    `line_items` would be a claim about the charges table, which the truth
    files do rule on; this is about the one key nobody scores.
    """
    return bool(step["keys"]) and set(step["keys"]) == {"other_fields"}


def _salvage_step(step: dict, raw: str):
    """The entries this step had finished before its reply stopped, or None.

    Ten entries and an eleventh cut off at the token cap is ten entries, not a
    failure: `_salvage_json` rewinds to the last element that closed, and here
    that is exactly the boundary between what the model finished saying and
    what it did not. Nothing is invented to fill the tail -- same rule as the
    single-shot salvage, and as grounding.

    None where nothing whole survived, so the caller still pays for the
    schema-constrained attempt and still raises if that fails too. An empty
    list is None as well: a step that returned no entry at all has not been
    rescued, it has merely been parsed.

    **The half-written entry is dropped, and dropping it is the point.**
    `_salvage_json` rewinds to the last element that closed, and inside a
    truncated `{"label": "...", "value": "half of a v` that is the COMMA after
    the label -- so it hands back an entry carrying a label and no value.
    `OTHER_FIELDS_SCHEMA` requires both, and a label with the value missing is
    not a fact the page states; it is the shape of a sentence the model was
    still writing. Keeping it would put an empty-valued field on screen among
    real ones, which is the same mistake as reading a key the reply never
    reached as "the document does not state this".
    """
    try:
        values = _step_values(step, raw, salvage=True)
    except Exception:
        return None
    whole = {key: ([e for e in value if "label" in e and "value" in e]
                   if key == "other_fields" and isinstance(value, list) else value)
             for key, value in values.items()}
    return whole if any(whole.get(key) for key in step["keys"]) else None


def _step_values(step: dict, raw: str, salvage: bool = False):
    """The keys this step owns, pulled out of its reply. Raises on unusable JSON.

    Only the step's own keys are taken. A step that answers with keys belonging to
    another step has guessed at fields it was not shown the rules for, and taking
    those would put back exactly the cross-contamination the split is here to stop.

    `salvage` accepts a reply that never closed, keeping the elements that did.
    It is passed only for an extras-only step (`_extras_only`): everywhere else
    a reply that stopped mid-value is a failure worth reporting as one, because
    the keys it dropped are keys something scores.
    """
    try:
        parsed = json.loads(_first_json_object(strip_fence(raw)) or raw)
    except json.JSONDecodeError:
        parsed = _salvage_json(raw) if salvage else None
        if parsed is None:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("reply was JSON but not an object")
    values = {}
    for key in step["keys"]:
        requested = step.get("request_aliases", {}).get(key, key)
        if requested not in parsed:
            if requested != key:
                # A reply using the canonical storage key did not answer the
                # actual question. Let the normal schema retry request it again.
                raise ValueError(f"reply is missing requested field: {requested}")
            continue
        value = parsed[requested]
        if key in ("line_items", INCOME_ITEMS_KEY, "other_fields"):
            values[key] = [v for v in value if isinstance(v, dict)] \
                if isinstance(value, list) else []
        elif isinstance(value, (str, int, float)):
            values[key] = value
    return values


def _ask_step(content: str, step: dict, status: dict, collect: list = None):
    """Ask one step and parse its reply.

    Returns (values, replies, truncated, tokens, salvaged).

    By default asked plain first, exactly as the baselines were measured. A reply that will
    not parse is asked once more with decoding constrained to this step's own
    keys, which is a different question rather than a repeat -- under greedy
    decoding, asking again in the same words returns the same answer.

    That second attempt is aimed at one measured failure: on Ollama, steps die by
    transcribing the page into their own first key, and the reply then runs to the
    step's cap. Instructions-last does not prevent it and a bigger cap buys more
    page rather than more answer; a grammar that does not contain the page does.
    Raises the second failure if both attempts fail.

    `collect` is where the raw replies are recorded, and it is a parameter rather
    than a return value because this function raises on total failure. Returning
    them only on the success path is what used to throw away the reply of every
    step that died -- exactly the replies worth reading, since a step that
    answered is already visible in its values. A dead step now leaves its text in
    `raw` beside the others.

    `salvaged` says the values came out of a reply that never closed. Only an
    extras-only step can be salvaged (`_extras_only`), and it is tried BEFORE
    the schema-constrained attempt: a list cut off at the token cap is not the
    failure that attempt was built for -- a grammar does not shorten an answer,
    so the second request would run to the same cap -- and the entries that
    finished are already in hand. Everywhere else the parse failure stands and
    the schema attempt runs exactly as it did.
    """
    # Schema-first runs also use the schema on grounding re-asks. A malformed
    # constrained reply is surfaced rather than repeated with an identical schema.
    replies, tokens = [], 0
    collect = replies if collect is None else collect
    extras = _extras_only(step)

    def keep(text):
        replies.append(text)
        if collect is not replies:
            collect.append(text)

    schema_first = bool(step.get("schema_first"))
    if schema_first:
        raw, truncated, used = _chat(content, step["max_tokens"], status, _step_schema(step))
    else:
        raw, truncated, used = _chat(content, step["max_tokens"], status)
    keep(raw)
    tokens += used
    try:
        return _step_values(step, raw), replies, truncated, tokens, False
    except Exception:
        kept = _salvage_step(step, raw) if extras else None
        if kept:
            return kept, replies, truncated, tokens, True
        if schema_first or not EXTRACT_SCHEMA:
            raise
    raw, truncated, used = _chat(content, step["max_tokens"], status,
                                 _step_schema(step))
    keep(raw)
    tokens += used
    try:
        return _step_values(step, raw), replies, truncated, tokens, False
    except Exception:
        kept = _salvage_step(step, raw) if extras else None
        if kept:
            return kept, replies, truncated, tokens, True
        raise


def _ungrounded_in(values: dict, source) -> list:
    """The step's scalar answers that are not in the transcript, as label = value.

    Blank and nil answers are skipped for the same reason `grounding.check` skips
    them: neither is a claim about the page. Line items are skipped too -- a table
    row is checked cell by cell at the end, and quoting a whole row back at the
    model is a worse question than not asking one.

    Nothing, where the step read page images (`_image_fed`): there is no
    transcript, and every value would be quoted back as invented.
    """
    bad = []
    if _image_fed():
        return bad
    for key, value in values.items():
        if isinstance(value, list):
            continue
        if grounding.is_blank(value) or grounding.is_nil(value):
            continue
        if not source.holds(value):
            bad.append(f"- {key} = {str(value)[:120]}")
    return bad


def _reply_label(step_id: str, attempt: int, n: int, failed: bool = False) -> str:
    """Which request produced one reply: the step, and what was different about it.

    A step can send up to four requests -- plain, schema-constrained, and either
    of those again as a re-ask -- and the four are not interchangeable when
    something has gone wrong: the schema attempt only runs because the plain one
    would not parse, and the re-ask only because the first answer was not on the
    page. A reply with no label attached is a wall of JSON that cannot be tied
    back to the question that produced it.
    """
    return (step_id + (" (retry)" if attempt else "") + (" (schema)" if n else "")
            + (" (failed)" if failed else ""))


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
    counts = {len(steps_for_types(c, mode=mode)) for c in combos}
    return (str(min(counts)) if len(counts) == 1
            else f"{min(counts)}-{max(counts)}")


def _steps_for(wanted, doc_types=(), extras=True, mode="agentic"):
    """The step table, or the named subset of it, in the table's own order.

    A measurement tool and nothing else: it is what lets one step's prompt be
    swept over five documents and four models without paying for the six steps
    the change did not touch. Unknown ids are refused rather than skipped -- a
    sweep that quietly asked for no steps would report a schema-wide failure.

    The order is the document's final step table, never the caller's. A historical
    component id such as total_incl now selects its complete related group.
    """
    table = steps_for_types(doc_types, extras, mode=mode)
    if not wanted:
        return table
    ids = [s.strip() for s in wanted if str(s).strip()]
    # Checked against THIS document's table, so asking a receipt for the
    # `original_invoice` step is refused by name rather than silently running
    # nothing. The message names the steps this type actually has.
    known = {name for s in table for name in (s["id"], *s.get("includes_steps", ()))}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown extraction step(s): {', '.join(unknown)} -- "
                         f"{' + '.join(doc_types) or 'an unclassified document'}"
                         f" has {', '.join(sorted(known))}")
    if not ids:
        raise ValueError("no extraction steps named")
    selected = set(ids)
    return tuple(s for s in table
                 if selected.intersection((s["id"], *s.get("includes_steps", ()))))


def _agentic_parallelism(status: dict, steps: int) -> int:
    """How many agentic steps go to the server at once: 1 means one at a time.

    More than one only on a server that batches concurrent requests
    (`backends.parallel_ok`, vLLM) and with `AGENTIC_PARALLEL` on. Capped by
    `AGENTIC_PARALLEL_MAX` where it is set, and never above the step count.
    """
    if not (AGENTIC_PARALLEL and steps > 1 and backends.parallel_ok(status)):
        return 1
    cap = AGENTIC_PARALLEL_MAX or steps
    return max(1, min(cap, steps))


def _warm_step(warmed, *args):
    """`_run_step` for the step that warms the prefix cache: `warmed` is set as
    soon as its first request is prefilled, and in any case once it is over, so a
    step that fails cannot keep the others waiting."""
    _prefill_hook.cb = warmed.set
    try:
        return _run_step(*args)
    finally:
        _prefill_hook.cb = None
        warmed.set()


def _run_step(step: dict, prefix: str, question: str, status: dict, source):
    """`_run_step_asked`, its requests named after the step for the stall alert."""
    previous = getattr(_agentic_chat_context, "requests", None)
    chats = _agentic_chat_context.requests = []
    try:
        with _asking(f"step {step['id']}"):
            outcome = _run_step_asked(step, prefix, question, status, source)
        if chats:
            outcome[0]["chats"] = chats
        return outcome
    finally:
        _agentic_chat_context.requests = previous


def _run_step_asked(step: dict, prefix: str, question: str, status: dict, source):
    """Ask one agentic step, with its grounding re-ask, and record what happened.

    Returns (record, values, replies, tokens). Self-contained by design: it reads
    the shared prefix and its own question and nothing another step produced,
    which is what lets `_extract_agentic` send the steps at once on vLLM. The
    labelled replies come back rather than being appended to a shared list, so
    a concurrent run can merge them in table order.
    """
    step_replies, total_tokens = [], 0
    # `raw` holds this step's own replies, verbatim, in the order they were
    # asked for. They are also concatenated into the result's single `raw`
    # for the whole-reply pane, and kept per step as well because that pane
    # cannot say which of fifteen questions a given block of JSON answers --
    # and a step that failed is exactly the one whose text is worth reading.
    record = {"id": step["id"], "title": step["title"],
              "keys": list(step["keys"]), "attempts": 0, "retried": False,
              "raw": [],
              # This step's own question, without the shared prefix. Set
              # before anything is asked, so a step whose request never came
              # back can still show what it was asked -- which is the state
              # where "what did we send it?" is the actual question.
              "prompt": question}
    if step.get("request_aliases"):
        record["request_aliases"] = dict(step["request_aliases"])
    values, elapsed = {}, 0.0
    best_bad = None

    for attempt in range(AGENTIC_RETRIES + 1):
        # A re-ask is a different question -- the rejected values quoted back
        # -- so it is kept beside the reply it produced rather than folded
        # into the step's base question.
        asked = question + (EXTRACT_STEP_RETRY.format(
            rejected="\n".join(best_bad)) if attempt else "")
        content = prefix + asked
        record["attempts"] = attempt + 1
        at = time.perf_counter()
        truncated = False
        # Filled by `_ask_step` as each reply arrives, so a step that raises
        # still leaves its text behind. Labelled here afterwards on the
        # success path, and in the handler on the failure path.
        raws = []
        try:
            attempt_values, raws, truncated, tokens, salvaged = _ask_step(
                content, step, status, raws)
            if salvaged:
                # The reply stopped mid-entry and what had finished was
                # kept. Recorded rather than passed over: the step answered
                # with less of the page than it found, and a count of
                # entries with nothing beside it reads as the whole of what
                # the document labels.
                record["salvaged"] = True
            for n, raw in enumerate(raws):
                label = _reply_label(step["id"], attempt, n)
                step_replies.append(f"--- {label} ---\n" + raw)
                record["raw"].append({"label": label, "text": raw,
                                      "prompt": asked})
            record["schema_retry"] = len(raws) > 1
            total_tokens += tokens
        except Exception as err:
            for n, raw in enumerate(raws):
                label = _reply_label(step["id"], attempt, n, failed=True)
                step_replies.append(f"--- {label} ---\n" + raw)
                record["raw"].append({"label": label, "text": raw,
                                      "prompt": asked})
            elapsed += time.perf_counter() - at
            # Recorded against the step and then dropped: the remaining steps
            # do not depend on this one, and the rest of the form is a better
            # answer than none of it. A cut-off reply is named as one, because
            # that step wants a bigger cap rather than another attempt.
            why = (f"the reply hit this step's {step['max_tokens']}-token cap "
                   "and the JSON never closed" if truncated else str(err))
            # A retry that fails leaves the answer it was meant to correct
            # standing, so the step has fields and is not a failed step. Saying
            # it failed would send someone looking for keys that are there.
            where = "error" if best_bad is None else "retry_error"
            # An extras-only step that dies costs no key anything rules on,
            # so it is reported and NOT counted as a failed step: it does
            # not reach `steps_failed`, the page's failed-step banner or the
            # run log's failure rate. A priority-1 step that dies still
            # does, because its keys are the form. See `_extras_only`.
            if where == "error" and _extras_only(step):
                where = "extras_error"
            record[where] = why
            break
        elapsed += time.perf_counter() - at

        # Retry wording names the keys the model was actually asked for, while
        # results, grounding and scoring keep the canonical application fields.
        requested_values = {step.get("request_aliases", {}).get(k, k): v
                            for k, v in attempt_values.items()}
        bad = _ungrounded_in(requested_values, source)
        # Keep whichever attempt invents least. A retry that comes back worse
        # than the answer it was meant to correct is not an improvement, and
        # under greedy decoding that happens whenever the model has nothing
        # better to offer than what it already said.
        if best_bad is None or len(bad) < len(best_bad):
            values, best_bad = attempt_values, bad
        if not bad or attempt >= AGENTIC_RETRIES:
            break
        record["retried"] = True

    record["seconds"] = round(elapsed, 2)
    record["values"] = {k: v for k, v in values.items() if not isinstance(v, list)}
    record["items"] = sum(len(v) for v in values.values() if isinstance(v, list))
    record["ungrounded"] = len(best_bad or [])
    return record, values, step_replies, total_tokens


def _extract_agentic(text: str, status: dict, form: dict, only=None, mode="agentic"):
    """Walk the step table, yielding progress, and return the merged result.

    A generator so that both the browser and the run stream can show which step is
    running without a callback that cannot yield: iterate it for the events, and
    take the finished result from StopIteration.value (`_drain` does this).

    `only` restricts the walk to the named steps. The result then says so in
    `steps_only`, and everything downstream that would otherwise report the keys
    nobody asked for as missing -- the field score, the run-log row -- stands
    down on that key. See `_steps_for`.
    """
    # The steps this document's TYPE asks for, before any restriction the
    # caller applied. `steps_only` is measured against this rather than against
    # the whole step table: an invoice legitimately runs six of the eight steps
    # and is a complete run of its own form, not a partial run of someone else's.
    full_table = steps_for_types(form["doc_types"], form.get("extras", True), mode=mode)
    table = _steps_for(only, form["doc_types"], form.get("extras", True), mode=mode)
    if form.get("schema_first"):
        # Copy shared definitions so concurrent JSON-first runs stay unchanged.
        table = [{**step, "schema_first": True} for step in table]
    source = grounding.Source(text)
    started = time.perf_counter()
    fields, steps, replies = {}, [], []
    total_tokens = 0

    # Sent once because it IS once: every step's message opens with this same
    # block and the same transcript, and only the question at the end differs.
    # That is the whole reason the steps are cheap on llama.cpp, and showing it
    # per step would say the opposite of what the design does.
    prefix = _step_prefix(text)

    # Concurrency, decided once per run. vLLM batches concurrent requests, and
    # no step reads another's answer (see `AGENTIC_PARALLEL`), so on that
    # server the steps are asked at once. Everywhere else, one after another,
    # exactly as every measurement in CLAUDE.md was taken.
    parallel = _agentic_parallelism(status, len(table))
    framing = _framing(form, mode)

    yield {"event": "extract_steps", "total": len(table),
           "steps": [{"id": s["id"], "title": s["title"], "keys": list(s["keys"])}
                     for s in table],
           "prompt_prefix": prefix, "parallel": parallel}

    # Each step's outcome, by its position in the table. Merged in TABLE order
    # whichever finished first, so a concurrent run produces the same `fields`,
    # `steps` and `raw` order as a sequential one -- the page, the run log and a
    # diff between two runs then cannot tell the two apart by layout.
    outcomes = [None] * len(table)

    def running(index, step):
        return {"event": "extract_step", "step": index, "total": len(table),
                "id": step["id"], "title": step["title"], "status": "running"}

    def done(index, step, record):
        return {"event": "extract_step", "step": index, "total": len(table),
                "id": step["id"], "title": step["title"], "status": "done", **record}

    if parallel > 1:
        for index, step in enumerate(table, 1):
            yield running(index, step)
        # One server-timing span for the lot: a /metrics difference around any
        # single concurrent request covers its neighbours. See servertime.Batch.
        batch = servertime.Batch(status)
        with ThreadPoolExecutor(max_workers=parallel,
                                thread_name_prefix="agentic-step") as pool:
            def submit(index):
                step = table[index]
                return pool.submit(_carry_progress(workspace.bind(batch.run)), _run_step, step,
                                   prefix, _step_question(step, framing, mode), status,
                                   source)

            # The warm-up only exists to fill the prefix cache; with the cache
            # bypassed it would just delay the other steps for nothing.
            if VLLM_PREFIX_WARMUP and not backends.prefix_cache_bypassed():
                # Step 1 alone first; the rest once its prefill is done, so they
                # find the shared prefix in vLLM's cache rather than each
                # prefilling it again in the same scheduler step.
                warmed = threading.Event()
                futures = {pool.submit(_carry_progress(workspace.bind(batch.run)), _warm_step,
                                       warmed, table[0],
                                       prefix, _step_question(table[0], framing, mode),
                                       status, source): 0}
                warmed.wait(VLLM_PREFIX_WARMUP_WAIT)
                futures.update({submit(index): index
                                for index in range(1, len(table))})
            else:
                futures = {submit(index): index for index in range(len(table))}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = futures[future]
                    outcomes[index] = future.result()
                    yield done(index + 1, table[index], outcomes[index][0])
        batch.close()
    else:
        for index, step in enumerate(table):
            yield running(index + 1, step)
            outcomes[index] = _run_step(step, prefix, _step_question(step, framing, mode),
                                        status, source)
            yield done(index + 1, step, outcomes[index][0])

    for record, values, step_replies, tokens in outcomes:
        fields.update(values)
        replies.extend(step_replies)
        total_tokens += tokens
        steps.append(record)

    elapsed = round(time.perf_counter() - started, 2)
    failed = [s for s in steps if s.get("error")]
    # Every step that returned nothing, whether or not that counts as a failure.
    # The bail-out below is about there being no answer at all, which an
    # extras-only step is as capable of producing as any other -- it just does
    # not make the run a failure on its own.
    dead = [s for s in steps if s.get("error") or s.get("extras_error")]
    if dead and len(dead) == len(steps):
        return {"error": "Every extraction step failed. First: "
                         + (dead[0].get("error") or dead[0]["extras_error"]),
                "mode": mode, "steps": steps, "seconds": elapsed,
                **({"parallel": parallel} if parallel > 1 else {}),
                "prompt_prefix": prefix,
                **({"steps_only": [s["id"] for s in table]}
                   if len(table) < len(full_table) else {}),
                "raw": "\n\n".join(replies)[:4000]}

    # Ordered the way the schema lists the keys rather than the way the steps
    # filled them, so the JSON on screen and the JSON a script copies are the same
    # shape in both modes.
    ordered = {k: fields[k] for k in grounding.SCALAR_FIELDS if k in fields}
    for key in ("line_items", "other_fields"):
        ordered[key] = fields.get(key) or []
    # Only where this form asked for the table. Unlike the two above, which have
    # been on every result since before the schema was per type, an empty
    # `income_items` on an invoice would be a claim that a table was asked for
    # and came back empty -- and `grounding.check` would then have to be told to
    # ignore the very key it was told to audit.
    if form["items"]:
        ordered[INCOME_ITEMS_KEY] = fields.get(INCOME_ITEMS_KEY) or []

    return {
        "fields": ordered,
        "raw": "\n\n".join(replies),
        # On the result as well as on the event that announced the steps: a
        # result read back from a sweep, or from the blocking endpoint, never
        # saw the event, and a step's question is unreadable without what led it.
        "prompt_prefix": prefix,
        "grounding": grounding.check(ordered, text, form["keys"],
                                     form["mandatory"], form["items"]),
        "tiers": grounding.tier_counts(ordered, form["keys"]),
        "vat_basis": verify.vat_basis(ordered, text),
        # Same derivation as single mode, from the merged answer rather than from
        # any one step -- the references list in particular spans several of them.
        "derived": normalise.derive(ordered, text),
        "seconds": elapsed,
        "tokens": total_tokens,
        "model": status["model"],
        "backend": status["kind"],
        # Which server answered. Pass 2 may have an endpoint of its own
        # (`backends.select_extract_url`), and a run is attributed to the server
        # that ran it, never to the one selected afterwards.
        "url": status.get("url") or "",
        "mode": mode,
        # How many steps were in flight at once, where it was more than one
        # (vLLM). Absent on a sequential run, which is every run on llama.cpp
        # and Ollama: the wall clock of the two shapes is not comparable.
        **({"parallel": parallel} if parallel > 1 else {}),
        "steps": steps,
        # Present only when this run was restricted to part of the step table, so
        # nothing downstream reads its keys as the whole form. Absent on an
        # ordinary run rather than holding every id, for the same reason.
        **({"steps_only": [s["id"] for s in table]} if len(table) < len(full_table)
           else {}),
        # Named separately from the per-step errors because a partial answer is
        # the normal outcome here rather than a failure: the page says which parts
        # of the form were not filled instead of showing them as simply empty.
        "steps_failed": [{"id": s["id"], "title": s["title"], "error": s["error"]}
                         for s in failed],
        # Reported separately and never folded into `steps_failed`: a broken
        # reply from an extras-only step costs entries nothing scores, so it is
        # a note about how much of the page came back rather than a failure of
        # the extraction. Absent when there is none, like `steps_only`.
        **({"steps_extras": [{"id": s["id"], "title": s["title"],
                              "error": s["extras_error"]}
                             for s in steps if s.get("extras_error")]}
           if any(s.get("extras_error") for s in steps) else {}),
    }


# --------------------------------------------------------------------------
# pass 2: which shape runs
# --------------------------------------------------------------------------

# Switched from the page rather than read from the environment per run, because it
# is a thing you flip while looking at a document that came out wrong. Held for the
# process, not per request: a sweep running now and a Re-extract clicked during it
# both extract the same way, and every result carries the mode it actually ran in
# so a log row is still attributable when it is switched mid-batch.
_extract_mode = "agentic" if AGENTIC_EXTRACT else "single"
_mode_lock = threading.Lock()

# The pass-1 shape, held for the process for the same reasons as the mode above,
# and switched from the page for the same reason: it is a thing you flip while
# looking at a document that came back empty.
#
# Clamped here rather than in `settings`, which imports nothing but `config` and
# so cannot see the profile table. An unknown name warns and falls back instead of
# killing startup -- a typo in a service file should cost one setting, not the app.
_ocr_profile = OCR_PROFILE if OCR_PROFILE in OCR_PROFILES else DEFAULT_OCR_PROFILE
if _ocr_profile != OCR_PROFILE:
    config.say(f"OCR_PROFILE={OCR_PROFILE!r} is not a known profile "
               f"({', '.join(OCR_PROFILES)}); using {_ocr_profile}.", sys.stderr)
_profile_lock = threading.Lock()

# These five are the page's switches, so each browser tab has its own copy --
# see `workspace`. The globals are the default workspace's.
workspace.field("extract_mode", lambda: _extract_mode)
workspace.field("ocr_profile", lambda: _ocr_profile)

# Whether pass 2 asks for `other_fields` at all. Held and switched like the mode
# above, and stamped on every result as `extras` so a row says which it ran under.
# See `settings.EXTRACT_OTHER_FIELDS`.
_extract_extras = bool(EXTRACT_OTHER_FIELDS)
workspace.field("extract_extras", lambda: _extract_extras)


def ocr_profile() -> str:
    with _profile_lock:
        return workspace.get("ocr_profile", _ocr_profile)


def set_ocr_profile(name: str) -> str:
    if name not in OCR_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(OCR_PROFILES)}.")
    global _ocr_profile
    with _profile_lock:
        if not workspace.put("ocr_profile", name):
            _ocr_profile = name
    return name


def profile_spec(name: str = None) -> dict:
    """The profile a request should be built from, resolved once per read.

    Taken by name and returned whole so a run cannot be assembled half from one
    profile and half from another -- the prompt, the system-message veto and the
    way the reply is read have to be the same profile's, or the request is not a
    measurement of either.
    """
    return OCR_PROFILES[name or ocr_profile()]


# Whether a cycling read is ABORTED. Held for the process and switched from the
# page for the same reason the two above are: it is a thing you flip while looking
# at a read that was cut short, and the next read should honour it without a
# restart.
#
# It does not turn detection off, only the abort -- see `settings.LOOP_GUARD`. A
# read that loops with the guard down still comes back flagged `looped`, so it
# still reads as a failure and still stays out of every mean; what changes is that
# the transcript is the whole of what the model produced rather than the part
# before the backstop fired.
_loop_guard = LOOP_GUARD
_loop_guard_lock = threading.Lock()
workspace.field("loop_guard", lambda: _loop_guard)


def loop_guard() -> bool:
    with _loop_guard_lock:
        return workspace.get("loop_guard", _loop_guard)


def set_loop_guard(on: bool) -> bool:
    global _loop_guard
    with _loop_guard_lock:
        if not workspace.put("loop_guard", bool(on)):
            _loop_guard = bool(on)
    return loop_guard()


# The two table agents, switchable from the page (2026-10-01). Process state like
# the loop guard, and stored IN the module globals `TABLE_FIX_AGENT` /
# `TABLE_IDENTIFY_AGENT` rather than beside them, so there is one value the code
# reads and the tests patch -- two copies of one switch would disagree after the
# first flip. Each document reads both once, at the start of its table.
#
# Per workspace like the rest: a tab that has flipped one holds its own three,
# and the default workspace reads the globals the tests patch.
_table_agents_lock = threading.Lock()


def _agents_default() -> dict:
    return {"concat": bool(TABLE_CONCAT_AGENT), "fix": bool(TABLE_FIX_AGENT),
            "identify": bool(TABLE_IDENTIFY_AGENT)}


workspace.field("table_agents", _agents_default)


def table_agents() -> dict:
    with _table_agents_lock:
        return {**workspace.get("table_agents", _agents_default()),
                "table": bool(TABLE_EXTRACT)}


def set_table_agents(fix=None, identify=None, concat=None) -> dict:
    global TABLE_FIX_AGENT, TABLE_IDENTIFY_AGENT, TABLE_CONCAT_AGENT
    with _table_agents_lock:
        mine = dict(workspace.get("table_agents", _agents_default()))
        for key, value in (("fix", fix), ("identify", identify), ("concat", concat)):
            if value is not None:
                mine[key] = bool(value)
        if not workspace.put("table_agents", mine):
            TABLE_FIX_AGENT, TABLE_IDENTIFY_AGENT = mine["fix"], mine["identify"]
            TABLE_CONCAT_AGENT = mine["concat"]
    return table_agents()


# The transcript score a read has to reach before the extraction taken from it is
# worth a field score. Held for the process and switched from the page for the
# same reason the three above are: it is a thing you flip while looking at a run
# whose fields came back unscored, and the next run should honour it without a
# restart.
#
# **There are two floors and they are not the same knob.** This one decides
# whether a correctness figure is WRITTEN, at the moment a run happens, and
# nothing can undo it afterwards -- a score never taken is not in the file.
# `runlog.read_floor` decides whether a figure already in the file is AVERAGED,
# and moves freely because it re-derives trust from the row's own
# `char_accuracy` every time a table is compiled. So the summary's floor is the
# reversible one and stays where it was; this is the one to leave low.
_read_floor = MIN_READ_FOR_FIELDS
_read_floor_lock = threading.Lock()
workspace.field("read_floor", lambda: _read_floor)


def read_floor() -> float:
    """The run-time floor as a FRACTION. See `runlog.read_floor` for the other."""
    with _read_floor_lock:
        return workspace.get("read_floor", _read_floor)


def set_read_floor(fraction: float) -> float:
    """Clamped to 0-1 rather than refused: every value in range means something.

    0 scores every extraction whatever the read did, which is the shape for
    filling the log and letting the summary's own floor decide what to average.
    """
    global _read_floor
    try:
        value = max(0.0, min(float(fraction), 1.0))
    except (TypeError, ValueError):
        raise ValueError("min_read_pct must be a number between 0 and 100.")
    with _read_floor_lock:
        if not workspace.put("read_floor", value):
            _read_floor = value
    return value


def _unscorable_read(payload: dict, floor: float = None) -> str:
    """Why this read's fields must not be scored, or "" where they may be.

    **Pass 2 can only map values pass 1 produced.** A field score taken over a
    broken transcript is a measurement of the read wearing the extractor's name:
    it marks down whatever extracted and lets whatever read get away with it.
    That gap is measured -- dots.mocr agentic scored 32.7% over its own reads
    against 46.8% over the ground truth -- and it is why every pass-2 baseline in
    CLAUDE.md was taken from truth.

    Four ways a read disqualifies its own extraction, and the last is the one the
    page can move:

    - it looped, or was cut off at the token cap -- the transcript is a fragment
      whatever it scored;
    - it returned nothing at 0.0%;
    - it scored under the floor in force.

    **None of these makes the run a failure.** A read that finished and scored
    badly is a run, and it is counted and averaged as one -- `runlog._incomplete`
    calls a loop or a crash a failure and nothing else. What is dropped here is
    only the pass-2 figure taken over that transcript.

    **A document with no transcript truth is never suppressed.** Nothing here can
    tell a bad read from an unmeasured one, and refusing to score on a guess
    would silently blank the figure for every document that has no `.md`.
    """
    floor = read_floor() if floor is None else floor
    if payload.get("looped"):
        return "the read looped"
    if payload.get("truncated"):
        return "the read hit the token cap"
    scored = (payload.get("truth") or {}).get("char_accuracy")
    if scored is None:
        return ""
    if scored <= 0:
        return "the read returned nothing"
    if scored < floor:
        return (f"the read scored {scored * 100:.1f}%, under "
                f"{floor * 100:.0f}%")
    return ""


def apply_read_floor(payload: dict, floor: float = None) -> str:
    """Mark this run's extraction unscorable where its read disqualifies it.

    **The one place the rule is applied, so no path can be left out of it.** It
    used to live in `_read_case` alone -- the random test's full rounds -- and a
    document read through the page or `/api/ocr` wrote a field score
    however badly it had been read. Two kinds of row in one file, with nothing in
    the CSV to say which rule each was written under. Called from `log_run`,
    which every read path goes through on its way to the log, and once more in
    the streaming route so the browser is told at the same time the log is.

    **It flags, it does not delete.** The score stays on the result and the
    Fields tab still marks every value against it -- those marks are what explain
    a bad read, and blanking them removes the evidence along with the number.
    What the flag costs is the headline rate on both tabs and the correctness
    columns in the run log (`runlog._extract_cells`), which are the places the
    figure would otherwise be read as the extractor's.

    Idempotent: the second call re-derives the same reason and overwrites it.
    """
    extracted = payload.get("extracted")
    if not isinstance(extracted, dict):
        return ""
    why = _unscorable_read(payload, floor)
    # Set on the file AND on every document in it. A split file's score is one
    # document's (see `_extract_documents`), and the page marks each document's
    # values from that document's own result -- so a flag that stopped at the
    # top level would leave the rate off the file and on the document, which is
    # the same figure said two ways and one of them wrong.
    for target in [extracted] + [d for d in (extracted.get("documents") or [])
                                 if isinstance(d, dict)]:
        if why:
            target["fields_unscored"] = why
        else:
            target.pop("fields_unscored", None)
    return why


def extract_mode() -> str:
    with _mode_lock:
        return workspace.get("extract_mode", _extract_mode)


def set_extract_mode(mode: str) -> str:
    if mode not in ("single", "agentic", "hybrid"):
        raise ValueError("mode must be 'single', 'agentic' or 'hybrid'.")
    global _extract_mode
    with _mode_lock:
        if not workspace.put("extract_mode", mode):
            _extract_mode = mode
    return mode


def extract_extras() -> bool:
    with _mode_lock:
        return bool(workspace.get("extract_extras", _extract_extras))


def set_extract_extras(on: bool) -> bool:
    global _extract_extras
    with _mode_lock:
        if not workspace.put("extract_extras", bool(on)):
            _extract_extras = bool(on)
    return extract_extras()


def _recut_message(table: dict) -> str:
    """The one request the item table can cost: which column is each cell under.

    It carries the TABLE and not the document -- the headings, the rows that fit
    (as the pattern to follow) and the rows that do not, one cell to a line.
    Which two cells of six are one value is a question about that row; the rest
    of the page is context for a different question and would be prefill spent
    on nothing.
    """
    columns = table["columns"]
    bad = list(table["misaligned"])
    good = [i for i in range(len(table["rows"])) if i not in set(bad)][:4]
    examples = "\n".join(
        "%d. %s" % (n, " | ".join("%d: %s" % (c, cell or "(empty)")
                                  for c, cell in enumerate(table["rows"][i], 1)))
        for n, i in enumerate(good, 1))
    cells = [tables.filled_cells(table["rows"][i]) for i in bad]
    rows = "\n".join(
        "Row %d -- %d cells:\n%s" % (
            n, len(row),
            "\n".join("  c%d: %s" % (c, cell) for c, cell in enumerate(row, 1)))
        for n, row in enumerate(cells, 1))
    # Every cell named, so the reply cannot be the right shape without placing
    # each one. `0` is a placeholder and not a column.
    skeleton = "{ \"rows\": [\n%s\n] }" % ",\n".join(
        "  { %s }" % ", ".join('"c%d": 0' % c for c in range(1, len(row) + 1))
        for row in cells)
    return prompts.TABLE_RECUT_PROMPT.format(
        count=len(columns),
        headings="\n".join("%d. %s" % (n, heading or "(no heading printed)")
                           for n, heading in enumerate(columns, 1)),
        examples=(prompts.TABLE_RECUT_EXAMPLES.format(rows=examples)
                  if good else ""),
        rows=rows, skeleton=skeleton)


def _recut_answers(answer, table: dict) -> list:
    """The reply as one list of column numbers per misaligned row.

    The reply names each cell; this puts them back in cell order. A row the
    reply left a cell out of comes back as None, which `tables.recut` refuses --
    a missing answer is not an answer of "no column".
    """
    rows = answer.get("rows") if isinstance(answer, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for position, index in enumerate(table["misaligned"]):
        entry = rows[position] if position < len(rows) else None
        count = len(tables.filled_cells(table["rows"][index]))
        if isinstance(entry, dict):
            picked = [entry.get("c%d" % c) for c in range(1, count + 1)]
            out.append(picked if len(entry) == count else None)
        elif isinstance(entry, list):
            out.append(entry)
        else:
            out.append(None)
    return out


def _recut_with_model(table: dict, status: dict, check: bool = False) -> dict:
    """Ask the extraction model which column each cell of a misfit row is under.

    Asked ONLY where `tables.py` found rows that do not fit their headings, and
    what comes back is column NUMBERS: `tables.recut` does the cutting. No
    character of the reply reaches the table, so this can move a cell boundary
    and can do nothing else -- it cannot correct, drop or invent a value -- and
    an answer that is the wrong shape costs that row its repair and nothing
    more.

    Never raises. The table is a measurement beside the extraction, and a failed
    repair leaves it exactly as Python read it, flagged.
    """
    bad = list(table.get("misaligned") or [])
    if not bad or not TABLE_RECUT_WITH_MODEL:
        return table
    if table.get("headings_lost"):
        # Nothing to cut the rows TO: the read kept no headings.
        table["recut"] = {"asked": 0, "taken": 0, "refused": 0,
                          "skipped": "the table has no headings to fit rows to"}
        return table
    if len(bad) > TABLE_RECUT_MAX_ROWS:
        table["recut"] = {"asked": 0, "taken": 0, "refused": 0,
                          "skipped": "%d rows do not fit, over the %d one request "
                                     "carries" % (len(bad), TABLE_RECUT_MAX_ROWS)}
        return table
    message = _recut_message(table)
    cells = sum(len(tables.filled_cells(table["rows"][i])) for i in bad)
    started = time.time()
    try:
        # A few tokens per cell: the reply is numbers, not the rows again.
        raw, _, tokens = _chat(message,
                               min(TABLE_RECUT_MAX_TOKENS, 120 + 10 * cells),
                               status)
        answer = json.loads(_first_json_object(strip_fence(raw)) or raw)
    except Exception as err:
        table["recut"] = {"asked": len(bad), "taken": 0, "refused": len(bad),
                          "error": str(err)[:200]}
        return table
    # `check` -- the fix agent's path -- lets the code refuse a cut the cells
    # contradict: on sol007 at 2 MP both gemma4:e4b and qwen3.5:9b answered with
    # the shifted layout as read, words under Amount.
    table = tables.recut(table, _recut_answers(answer, table), check=check)
    table["recut"].update(seconds=round(time.time() - started, 2),
                          tokens=tokens or 0, model=status.get("model") or "",
                          prompt=message, raw=raw)
    return table


def _table_headings(columns) -> str:
    return "\n".join("%d. %s" % (n, heading or "(no heading printed)")
                     for n, heading in enumerate(columns, 1))


def _table_rows_text(table: dict, indexes) -> str:
    """Rows as `Row N: 1: cell | 2: cell`, numbered by their place in the table,
    so a reply naming Row 4 names the same row the code means by index 3."""
    return "\n".join(
        "Row %d: %s" % (i + 1, " | ".join(
            "%d: %s" % (c, (cell or "").strip() or "(empty)")
            for c, cell in enumerate(table["rows"][i], 1)))
        for i in indexes)


def _as_bool(value):
    """True/False, or None. `"true"` and `"false"` count: gemma4:e4b answers the
    identify question with the JSON literal on one call and the same word in
    quotes on the next, and the word is not in doubt. Anything else is not an
    answer -- "yes", 1, null."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _as_int(value):
    """A whole number, or None. `"3"` counts, for the reason `_as_bool` gives;
    a bool does not, though Python calls it an int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _table_cells_text(table: dict) -> str:
    """Every filled cell on a line of its own, named by row and column.

    For the fix agent, which QUOTES cells. On the compact `1: cell | 2: cell`
    line gemma4:e4b quoted the column prefixes themselves -- `1:`, `2:` -- on
    six of 32 real tables, and once a paragraph of the prompt (2026-10-01).
    A line per cell leaves nothing on it to quote but the cell.
    """
    columns = table["columns"]
    out = []
    for i, row in enumerate(table["rows"], 1):
        out.append("Row %d" % i)
        for c, cell in enumerate(row, 1):
            if (cell or "").strip():
                heading = columns[c - 1] if c - 1 < len(columns) else ""
                out.append("  row %d, column %d (%s): %s"
                           % (i, c, heading or "no heading", cell.strip()))
    return chr(10).join(out)


def _agent_reply(raw: str):
    """A reply's JSON object, or a ValueError naming what came back instead."""
    answer = json.loads(_first_json_object(strip_fence(raw)) or raw)
    if not isinstance(answer, dict):
        raise ValueError("the reply was not a JSON object")
    return answer


def _frame_note(table: dict) -> str:
    """What the code knows about where the table's frame ends, for the fix agent.

    `TABLE_FRAME_NOTE` where it found the totals line after some other line --
    something was read inside the frame; `TABLE_AFTER_FRAME_NOTE` where the
    table was read with its totals already in it and none follows; else ""."""
    tail = table.get("tail") or []
    at = tables.totals_line(tail)
    if at is None and tail and any(d.get("why") == "totals row"
                                   for d in table.get("dropped") or []):
        return prompts.TABLE_AFTER_FRAME_NOTE
    if not at:
        return ""
    return prompts.TABLE_FRAME_NOTE.format(
        totals=at + 1, inside="L1" if at == 1 else "L1 to L%d" % at,
        were="was" if at == 1 else "were")


def _table_fix_agent(table: dict, status: dict, snap=None) -> dict:
    """Agent 1: take what does not belong in the table out of it, then fix the cuts.

    One request carrying the TABLE and the part of its page around it -- every
    row, cell by cell under its numbered headings, and `TABLE_CONTEXT_SHARE` of
    the page's lines before it and after it (`TABLE_CLEAN_PROMPT`). The reply QUOTES
    text in a cell that is not part of the table's data: a rubber stamp's or a
    seal's words, a QR code's or a logo's description, a handwritten note, stray
    text from elsewhere on the page.

    **The code does the removing, and the code can refuse.** `tables.remove_text`
    looks for the quote in THAT cell and cuts the cell's own characters out, so
    a quote the model reworded is simply not found; `tables.removal_refused`
    keeps an amount under a money heading and a whole cell that is the kind of
    value its heading asks for. What is removed is not dropped -- it is kept as
    plain text, under `outside_text`.

    Then the cell boundaries, exactly as with the agent off: `realign` by what
    each cell is, and the column question (`_recut_with_model`, with
    `check_cut`) for a row that leaves unsettled. Taking a stamp's word out of a
    row is very often what lets the row fit its headings again, which is why
    the cleaning comes first.

    Never raises: a failed request leaves the table to the code, and says so.
    """
    record = {"asked": False, "removed": [], "refusals": [], "rows_emptied": [],
              "model": status.get("model") or "", "prompt": "", "raw": "",
              "seconds": 0.0, "tokens": 0}
    table["fix_agent"] = record

    def then_cuts(table):
        table = tables.reclassify(tables.realign(table))
        if snap:
            snap("realign", "code", table)
        table = _recut_with_model(table, status, check=True)
        if snap:
            snap("recut", "model", table)
        return table

    columns = table["columns"]
    if table.get("headings_lost") or len(columns) < 2 or not table["rows"]:
        record["skipped"] = ("the table has no headings to judge a cell by"
                             if table["rows"] else "the table has no rows")
        return then_cuts(table)
    if len(table["rows"]) > TABLE_FIX_MAX_ROWS:
        record["skipped"] = ("%d rows, over the %d one request carries"
                             % (len(table["rows"]), TABLE_FIX_MAX_ROWS))
        return then_cuts(table)
    message = prompts.TABLE_CLEAN_PROMPT.format(
        count=len(columns), headings=_table_headings(columns),
        rows=_table_cells_text(table),
        before=chr(10).join("B%d %s" % (n, line) for n, line
                            in enumerate(table.get("context_before") or [], 1))
        or "(none)",
        lines=chr(10).join("L%d %s" % (n, line)
                           for n, line in enumerate(table.get("tail") or [], 1))
        or "(none)",
        frame=_frame_note(table))
    record.update(asked=True, prompt=message)
    started = time.time()
    try:
        cells = sum(len(tables.filled_cells(r)) for r in table["rows"])
        raw, _, tokens = _chat(message, min(TABLE_RECUT_MAX_TOKENS,
                                            160 + 12 * cells), status)
        record.update(raw=raw, tokens=tokens or 0)
        answer = _agent_reply(raw)
        entries = answer.get("remove")
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise ValueError("the reply's remove was not a list")
    except Exception as err:
        record["error"] = str(err)[:200]
        record["seconds"] = round(time.time() - started, 2)
        return then_cuts(table)
    # A placeholder entry copied from the instructions -- row 0, or empty text --
    # is not an answer and is not reported as a refusal either.
    entries = [e for e in entries
               if not (isinstance(e, dict) and not str(e.get("text") or "").strip())]
    # The lines after the table first: they are numbered against the tail as
    # it was sent, and nothing below changes it.
    table = tables.sort_tail(table, answer.get("stray_lines"),
                             answer.get("table_lines"))
    record["lines"] = table.pop("tail_sorted", {})
    table = tables.remove_text(table, entries)
    done = table.pop("removal", {})
    record.update(removed=done.get("removed", []), refusals=done.get("refusals", []),
                  rows_emptied=done.get("rows_emptied", []),
                  seconds=round(time.time() - started, 2))
    if snap:
        snap("clean", "agent", table, removed_rows=list(done.get("rows_emptied", [])))
    return then_cuts(table)


def _table_concat_agent(table: dict, status: dict, snap=None) -> dict:
    """Agent 0: which of the read's other tables are pieces of this one.

    Shown the item table's headings and last rows, and every other table the
    read made of the document after it (`tables.concat_parts`). The reply is
    table NUMBERS; `tables.concat` does the joining, placing each cell by what it
    is and filing a totals line as a totals line. A number that names no table
    is ignored, and a table whose rows cannot be placed is refused and named.
    Never raises: a failed request leaves the table as the read cut it.
    """
    parts = tables.concat_parts(table)
    record = {"asked": False, "offered": len(parts), "joined": [], "refusals": [],
              "by_sum": [], "model": status.get("model") or "", "prompt": "",
              "raw": "", "seconds": 0.0, "tokens": 0}
    table["concat"] = record
    if not parts:
        record["skipped"] = "the read made no other table after this one"
        return table
    # The code first: a piece holding one of the table's column totals is its
    # totals block, by arithmetic. Joined without asking; the model is asked
    # only about what that leaves.
    by_sum = tables.sum_parts(table)
    if by_sum:
        table = tables.concat(table, by_sum, totals=by_sum)
        done = table.pop("concat_done", {})
        record.update(by_sum=done.get("joined", []), joined=list(done.get("joined", [])),
                      refusals=list(done.get("refusals", [])),
                      rows_added=done.get("rows_added", 0),
                      dropped_added=done.get("dropped_added", 0))
        parts = [p for p in parts if p["number"] not in by_sum]
        if not parts:
            record["skipped"] = "every other table holds a column total -- joined by the code"
            if snap:
                snap("concat", "code", table)
            return table
    shown = lambda cells: " | ".join(c for c in cells if c) or "(blank)"
    others = "\n".join(
        "T%d (page %s)\n%s" % (p["number"], p.get("page"),
                               "\n".join("  " + shown(r)
                                         for r in p["rows"][:tables.CONCAT_SHOW_ROWS]))
        for p in parts)
    message = prompts.TABLE_CONCAT_PROMPT.format(
        count=len(table["columns"]), headings=_table_headings(table["columns"]),
        last_rows="\n".join(shown(r) for r in table["rows"][-3:]) or "(no rows)",
        others=others)
    record.update(asked=True, prompt=message)
    started = time.time()
    try:
        raw, _, tokens = _chat(message, 120, status)
        record.update(raw=raw, tokens=tokens or 0)
        answer = _agent_reply(raw)
        picks = answer.get("same_table")
        if picks is None:
            picks = []
        if not isinstance(picks, list):
            raise ValueError("the reply's same_table was not a list")
    except Exception as err:
        record["error"] = str(err)[:200]
        record["seconds"] = round(time.time() - started, 2)
        return table
    numbers = []
    for n in picks:
        if isinstance(n, str) and n.strip().upper().lstrip("T").isdigit():
            n = int(n.strip().upper().lstrip("T"))
        if isinstance(n, int) and not isinstance(n, bool):
            numbers.append(n)
    offered = {p["number"] for p in parts}
    table = tables.concat(table, [n for n in numbers if n in offered])
    done = table.pop("concat_done", {})
    record.update(joined=record["joined"] + done.get("joined", []),
                  refusals=record["refusals"] + done.get("refusals", []),
                  rows_added=record.get("rows_added", 0) + done.get("rows_added", 0),
                  dropped_added=record.get("dropped_added", 0) + done.get("dropped_added", 0),
                  seconds=round(time.time() - started, 2))
    if snap:
        snap("concat", "agent", table)
    return table


def _identify_fields(fields) -> str:
    """The few extracted values the identify agent is shown, one per line."""
    if not isinstance(fields, dict):
        return ""
    lines = ["  %s: %s" % (label, str(fields[key]).strip())
             for key, label in prompts.TABLE_IDENTIFY_FIELDS
             if isinstance(fields.get(key), (str, int, float))
             and str(fields[key]).strip() not in ("", "-")]
    return ("Values read from this document:\n%s\n" % "\n".join(lines)
            if lines else "")


def _table_identify_agent(table: dict, form: dict, fields, status: dict) -> dict:
    """Agent 2: master table or item list, from the table and nothing else.

    Sent the table (its headings and up to `TABLE_IDENTIFY_MAX_ROWS` rows), the
    document's type and a few values the extraction read -- never the page.
    The reply is `is_master_table` and the NUMBER of the column holding the
    other documents' numbers, and code checks the claim:

      * a `true` is taken only where `tables.reference_column_ok` passes the
        column it names -- a heading that does not say it is something else, and
        cells filled on most rows. A master table without a filled reference
        column is not one, and a model saying so is not evidence that it is.
      * a `false` is taken as given: there is nothing printed for it to point at.

    The Python rule's answer is kept beside it as `master_rules`, and answers
    instead wherever the agent is off, fails, or is refused -- `master_from`
    says which. Never raises.
    """
    table["master_rules"] = table.get("is_master_table")
    table["master_rules_why"] = table.get("master_why")
    table["master_from"] = "rules"
    if not table_agents()["identify"] or not table.get("rows"):
        return table
    columns = table["columns"]
    bad = set(table.get("misaligned") or [])
    shown = [i for i in range(len(table["rows"])) if i not in bad] \
        or list(range(len(table["rows"])))
    shown = shown[:TABLE_IDENTIFY_MAX_ROWS]
    codes = form.get("doc_types") or []
    message = prompts.TABLE_IDENTIFY_PROMPT.format(
        doc_type=prompts.type_phrase(codes) or "not known",
        fields=_identify_fields(fields), count=len(columns),
        headings=_table_headings(columns), shown=len(shown),
        total=len(table["rows"]), rows=_table_rows_text(table, shown))
    record = {"answer": None, "column": None, "taken": False,
              "model": status.get("model") or "", "prompt": message, "raw": "",
              "seconds": 0.0, "tokens": 0}
    table["identify_agent"] = record
    started = time.time()
    try:
        raw, _, tokens = _chat(message, 120, status)
        record.update(raw=raw, tokens=tokens or 0)
        answer = _agent_reply(raw)
    except Exception as err:
        record["error"] = str(err)[:200]
        record["seconds"] = round(time.time() - started, 2)
        return table
    record["seconds"] = round(time.time() - started, 2)
    verdict = _as_bool(answer.get("is_master_table"))
    column = _as_int(answer.get("reference_column"))
    record.update(answer=verdict if isinstance(verdict, bool) else None,
                  column=column)
    if not isinstance(verdict, bool):
        record["refused"] = "the reply did not say true or false"
        return table
    named = (columns[column - 1] or "column %d" % column) \
        if column and 1 <= column <= len(columns) else ""
    if verdict:
        why = tables.reference_column_ok(table, column - 1 if column else -1)
        if why:
            record["refused"] = ("it said master table, naming column %s, and %s"
                                 % (column if column is not None else "none", why))
            return table
        table["reference_columns"] = [named]
        table["master_why"] = (
            "the model read each row as another document, identified under %s; "
            "the code checked that column is filled on most rows" % named)
    else:
        table["master_why"] = (
            "the model read the rows as goods or services, not as other documents"
            + (" (it named %s as the column citing a document)" % named
               if named else ""))
    table["is_master_table"] = verdict
    table["master_from"] = "model"
    record["taken"] = True
    return table


def _item_table(text: str, form: dict, status: dict, pages=None,
                fields=None):
    """`_item_table_stream` for a caller with nowhere to show progress."""
    return _drain(_item_table_stream(text, form, status, pages, fields))


def _agent_brief(record) -> dict:
    """What one table agent did, small enough for a progress event.

    The page draws a stage box and a progress line from this while the run is
    still going; the full record -- the prompt, the reply, every refusal --
    rides on the finished table, and copying it into every event would put the
    prompt on the wire twice.
    """
    if not isinstance(record, dict):
        return {}
    brief = {"model": record.get("model") or "",
             "seconds": record.get("seconds") or 0.0}
    for key in ("skipped", "error", "refused"):
        if record.get(key):
            brief[key] = str(record[key])[:160]
    if "removed" in record:
        lines = record.get("lines") or {}
        brief.update(asked=bool(record.get("asked")),
                     removed=len(record.get("removed") or []),
                     refused_n=len(record.get("refusals") or [])
                     + len(lines.get("refusals") or []),
                     stray_lines=len(lines.get("stray") or [])
                     + len(lines.get("stray_before") or []),
                     table_lines=len(lines.get("own") or [])
                     + len(lines.get("own_before") or []))
    if "answer" in record:
        brief.update(answer=record.get("answer"), column=record.get("column"),
                     taken=bool(record.get("taken")))
    if "offered" in record:
        brief.update(asked=bool(record.get("asked")), offered=record.get("offered", 0),
                     joined=len(record.get("joined") or []),
                     rows_added=record.get("rows_added", 0),
                     dropped_added=record.get("dropped_added", 0))
    return brief


def _item_table_stream(text: str, form: dict, status: dict, pages=None,
                       fields=None, document=None):
    """The document's item table and whether it lists items or other documents.

    Read out of the transcript in Python -- see `tables.py`, whose docstring is
    where the reasoning lives. No pass-2 prompt is involved and no field is
    touched, so this can be on for every run without moving a field score.

    A generator since 2026-10-01, so the page can say which table agent is
    running: `table_agent` events, one `running` and one `done` (or `off`) per
    agent, plus a `read` event for the table itself. `_item_table` drains it for
    a caller with nowhere to show progress, and returns exactly what this does.

    Returns the table, None where the document rules none, or `...` -- the
    sentinel for "not looked for", which is a different statement from "looked
    and found none" and must not read as it:

      * `TABLE_EXTRACT` is off, or
      * the document's type rules a table of its own. A withholding certificate
        has `income_items`, asked for by its requirement in its own four
        cells; the same rows a second time under another name would be two
        answers to one question.

    `pages` is where this document sits in its FILE. One page of a pack arrives
    here as bare text with no page marker in it, so the table would call it
    page 1 whichever page of the file it was; a document of several pages
    carries its own markers and needs no help.
    """
    if not TABLE_EXTRACT or form.get("items") or _image_fed():
        return ...      # an image-fed extraction has no transcript to cut it from
    tag = {"document": document} if document else {}

    def event(agent, state, **extra):
        return {"event": "table_agent", "agent": agent, "status": state,
                **tag, **extra}

    # Read once: a switch flipped from the page mid-run takes effect on the next
    # document, not between the two halves of this one.
    agents = table_agents()
    fix_on, identify_on = agents["fix"], agents["identify"]
    concat_on = agents["concat"]
    # The table after every step, so the Table tab can say what each one
    # changed and who made the change: the read, the fix agent's cleaning, the
    # code's re-cut, the model's re-cut. Copies -- every step edits in place.
    stages = []

    def snap(stage, by, t, **extra):
        stages.append({"stage": stage, "by": by,
                       "columns": list(t.get("columns") or []),
                       "rows": [list(r) for r in t.get("rows") or []],
                       "misaligned": list(t.get("misaligned") or []),
                       **({"asked": bool((t.get("recut") or {}).get("asked"))}
                          if stage == "recut" else {}),
                       **extra})

    try:
        # The rows as the read cut them, structural repairs only. With the fix
        # agent off, `realign` then runs exactly as `item_table(text)` would
        # have run it -- `reclassify(realign(...))`, the same two calls.
        table = tables.item_table(text, align=False,
                                  context_share=TABLE_CONTEXT_SHARE)
        if table is None:
            yield event("read", "none")
            return None
        snap("read", "read", table)
        yield event("read", "done", rows=len(table["rows"]),
                    columns=len(table["columns"]))
        # Agent 0 first: the fix agent should see the whole table, and the page
        # after the whole table.
        if concat_on and table.get("_others"):
            yield event("concat", "running", offered=len(table["_others"]),
                        model=status.get("model") or "")
            with _asking("table concat agent"):
                table = _table_concat_agent(table, status, snap)
            yield event("concat", "done", **_agent_brief(table.get("concat")))
        else:
            yield event("concat", "off" if not concat_on else "skipped")
        if fix_on:
            yield event("fix", "running", rows=len(table["rows"]),
                        model=status.get("model") or "")
            with _asking("table fix agent"):
                table = _table_fix_agent(table, status, snap)
            yield event("fix", "done", **_agent_brief(table.get("fix_agent")))
        else:
            yield event("fix", "off")
            table = tables.reclassify(tables.realign(table))
            snap("realign", "code", table)
            with _asking("table recut"):
                table = _recut_with_model(table, status)
            snap("recut", "model", table)
        # Agent 2 last, on the table as it now stands.
        asking = identify_on and bool(table.get("rows"))
        if asking:
            yield event("identify", "running", model=status.get("model") or "")
        with _asking("table identify agent"):
            table = _table_identify_agent(table, form, fields, status)
        yield event("identify", "done" if asking else "off",
                    is_master_table=table.get("is_master_table"),
                    master_from=table.get("master_from"),
                    **_agent_brief(table.get("identify_agent")))
        table["stages"] = stages
        table = tables.public(table)
        if pages and len(pages) == 1:
            page = pages[0]
            table["pages"] = [page]
            table["row_pages"] = [page] * len(table["rows"])
            for entry in table.get("dropped") or []:
                entry["page"] = page
        return table
    except Exception as err:  # pragma: no cover - never worth an extraction
        yield event("read", "error", error=str(err)[:200])
        return {"error": f"table extraction failed: {err}"}


def _score_item_table(result: dict, case_id: str, pages=None, text: str = "") -> dict:
    """Attach the table's own score, where the case has ground truth for it.

    Beside `field_score` and never inside it: no requirement makes a row of this
    table Mandatory, so it has no place in a headline that is the requirement's
    Mandatory set, and it measures the read rather than the extraction.
    """
    if not (case_id and isinstance(result, dict) and "item_table" in result):
        return result
    try:
        actual = result["item_table"]
        if isinstance(actual, dict) and actual.get("error"):
            actual = None
        score = fieldscore.evaluate_table(case_id, actual, pages)
        if score:
            result["table_score"] = score
    except Exception as err:  # pragma: no cover - a score is never worth a 500
        result["table_score"] = {"error": f"table scoring failed: {err}"}
    # The fix agent's own score, against `solution/tables/<id>.md` -- a key of
    # its own and never folded into `table_score` or `field_score`: it says
    # whether the agent's decisions were right, which neither of those can.
    try:
        actual = result.get("item_table")
        fixed = fixscore.score(case_id, actual, text) if isinstance(actual, dict) else None
        if fixed:
            result["fix_score"] = fixed
    except Exception as err:  # pragma: no cover - a score is never worth a 500
        result["fix_score"] = {"error": f"fix-agent scoring failed: {err}"}
    return result


def extract_fields_stream(text: str, mode: str = None, case_id: str = None,
                          steps=None, doc_type: str = None, pages=None,
                          truth_fed: bool = False, extras: bool = None,
                          schema_first: bool = False):
    """Turn a finished transcript into structured JSON, yielding progress.

    A separate text-only pass rather than part of the OCR prompt: mixing
    transcription and interpretation in one request measurably degrades the
    transcript, and this way extraction can be re-run without re-reading the page.

    **A file is not a document** (2026-09-08). The transcript is split into one
    document per document first, and each is extracted on its own text with its
    own form, its own grounding and its own validation -- so three documents in
    one file answer three forms rather than one form three times over. `pages` is
    the read's own page list where the caller has it; without it the split is
    read back out of the `--- page N ---` markers, which is the only thing a
    truth-fed run has.

    **One document is the path it always was, byte for byte.** `segment.segment`
    over a file it reads as one hands back the transcript it was given rather
    than a rebuild of it, and `_extract_document` below is the whole of the old
    body -- so eleven of the thirteen fixtures, and every single-page upload,
    take exactly the requests they took before. Only the shape of the RESULT
    grows, and only additively: `segments`, `documents_found` and `split_from`.

    Single mode has no progress to report beyond the two stage events and yields
    nothing else. Both shapes return the same result dict, plus `mode` saying
    which one ran.

    `case_id` is a benchmark document this transcript came from, and it is scored
    here rather than at each of the four call sites: the two modes build their
    result dicts separately, and a score attached in only some of the places a
    result is produced is a score that quietly disappears depending on which
    button was pressed.
    """
    mode = mode or extract_mode()
    schema_first = bool(schema_first and mode in ("agentic", "hybrid"))
    if not text.strip():
        return {"error": "Nothing to extract from.", "schema_first": schema_first}
    # The EXTRACTION model's status, which is the reading model's unless one has
    # been chosen separately. Handing this dict to the request builders is the
    # whole of what makes a second model work: they all read `info["model"]`.
    status = backends.extract_status()
    # `text_available`, not `available`: pass 2 sends the transcript as text and
    # gets JSON back, so a model with no vision can still be measured on the
    # form -- and measuring exactly that is what the Fields pane is for. This
    # read `available` until 2026-08-20, which refused every text-only model
    # with a complaint about images.
    if not status["text_available"]:
        return {"error": status["text_reason"], "schema_first": schema_first}
    # Whether `other_fields` is asked for, decided once for every document of
    # the file so a switch flipped mid-file cannot split it.
    extras = extract_extras() if extras is None else bool(extras)
    # Every model request this extraction makes -- the segment and classify
    # questions, the steps or the single request, the table agents -- is timed
    # by the server too, and summed here into `server_time`. Closed in `finally`
    # so a run that raises does not leave the tally open on this thread.
    previous = servertime.open_tally()
    try:
        result = yield from _extract_file(text, mode, case_id, steps, doc_type,
                                          pages, truth_fed, status, extras, schema_first)
    finally:
        served = servertime.close_tally(previous)
    if isinstance(result, dict) and served:
        result["server_time"] = served
    if isinstance(result, dict):
        result["schema_first"] = schema_first
        if _image_fed():
            _ungrounded_result(result)
    return result


def _ungrounded_result(result: dict) -> None:
    """An image-fed result's grounding, said to be unchecked rather than ~0%.

    `grounding.check` holds each value to the transcript, and an image-fed
    extraction has none -- every value would read as invented, and a grounded
    ratio of zero would be a claim about the model that nothing measured. The
    keys left blank (`missing`) are a fact about the form, and are kept.
    """
    for doc in [result, *[d for d in result.get("documents") or [] if isinstance(d, dict)]]:
        found = doc.get("grounding")
        if isinstance(found, dict):
            doc["grounding"] = {"skipped": "no transcript: the extractor read the page images",
                                "missing": list(found.get("missing") or []),
                                "statuses": {}, "flagged": [], "counts": {}, "checked": 0,
                                "grounded_ratio": None}
    result["image_fed"] = True


def _extract_file(text, mode, case_id, steps, doc_type, pages, truth_fed, status,
                  extras=True, schema_first=False):
    """The body of `extract_fields_stream`, split out so the server-time tally
    can be wrapped round all of it."""
    segments, split_from = yield from _segment_stream(text, pages, status)
    records = [_segment_record(seg, index, len(segments))
               for index, seg in enumerate(segments, 1)]
    # Emitted on every run, one document or several. A page that only heard
    # about the split when there was one would have nothing to say for the
    # ordinary case, and "one document" is a finding about a two-page file quite
    # as much as "two documents" is.
    yield {"event": "segmented", "documents": len(segments),
           "split_from": split_from, "segments": records}
    if len(segments) > 1:
        result = yield from _extract_documents(segments, records, mode, case_id,
                                               steps, doc_type, status,
                                               truth_fed, extras, schema_first)
    else:
        result = yield from _extract_document(segments[0]["text"], mode, case_id,
                                              steps, doc_type, status,
                                              truth_fed=truth_fed, extras=extras,
                                              schema_first=schema_first)
    if isinstance(result, dict):
        # On every result, however many documents: what the file was read as is
        # a fact about the run, and a reader looking at one document's fields
        # needs to know whether that was the whole file.
        result["segments"] = records
        result.setdefault("documents_found", 1)
        result["split_from"] = split_from
        # Whether `other_fields` was asked for. An empty list means "none
        # found" only where it was; the run log reads this to leave it blank.
        result["extras"] = extras
    return result


def _extract_document(text: str, mode: str, case_id: str, steps, doc_type,
                      status: dict, record: dict = None,
                      truth_fed: bool = False, extras: bool = True,
                      schema_first: bool = False):
    """One document: classify it, ask for its form, ground it, validate it, score it.

    The whole of what `extract_fields_stream` used to be, lifted out unchanged so
    that a file holding one document takes the same requests it always took, and
    so that a file holding three takes that same path three times rather than a
    second implementation of it.

    `record` is this document's place in the file, where there is more than one.
    It rides on the `classified` event only: without it a page receiving seven
    step events and three classifications cannot tell which document any of them
    belongs to.
    """
    # Which form to ask for. Resolved ONCE, here, and handed to whichever shape
    # runs -- the prompt, the JSON grammar, the step table, what grounding calls
    # missing, what validate checks and what fieldscore scores all come off this
    # one dict, so the two modes cannot disagree about which document this is.
    form = _field_set(text, case_id, doc_type, status,
                      (record or {}).get("pages"), extras)
    form["schema_first"] = bool(schema_first and mode in ("agentic", "hybrid"))
    # Stage 0, announced before either shape starts. It is the one part of a
    # pass-2 run that is finished before the first request goes out, so a page
    # that shows it immediately is showing something true rather than a
    # placeholder -- and in single mode it is the ONLY progress event there is.
    yield {"event": "classified", "mode": mode, **_classified(form, mode),
           **({"document": record["document"], "documents": record["documents"],
               "pages": list(record["pages"]),
               "page_range": record["page_range"]} if record else {})}
    if mode in ("agentic", "hybrid"):
        result = yield from _extract_agentic(text, status, form, steps, mode=mode)
    else:
        # Refused rather than ignored: single mode is one request for the whole
        # schema, so there is no honest way to ask it for part of one, and a
        # sweep that thought it was measuring one step would be measuring
        # fourteen keys instead.
        if steps:
            return {"error": "steps only apply to agentic or hybrid extraction."}
        result = _extract_single(text, status, form)
    result = _validate_fields(result, form, text, mode)
    if isinstance(result, dict):
        result["schema_first"] = form["schema_first"]
    # The item table, read out of the same transcript. Attached whether or not
    # the extraction itself produced fields -- the table does not depend on it --
    # and skipped for a run restricted to some steps, which asked a narrower
    # question than "what does this document say".
    if isinstance(result, dict) and not steps:
        table = yield from _item_table_stream(
            text, form, status, (record or {}).get("pages"), result.get("fields"),
            (record or {}).get("document"))
        if table is not ...:
            result["item_table"] = table
            result = _score_item_table(result, case_id,
                                       (record or {}).get("pages"), text)
    return _score_fields(result, case_id, form,
                         (record or {}).get("pages"), text, truth_fed)


def _extract_documents(segments, records, mode: str, case_id: str, steps,
                       doc_type, status: dict, truth_fed: bool = False,
                       extras: bool = True, schema_first: bool = False):
    """Every document in the file, one at a time, merged into one result.

    **The fields are independent and are kept that way.** Each document is
    extracted from its own pages only, so a value on page 3 cannot fill a key of
    the document on page 1 -- which is the whole of what splitting buys, and is
    exactly the failure that made a three-document file worth splitting: one
    form asked of three documents' text fills each key from whichever page the
    model saw first, and every check downstream then endorses it, because the
    value really is on the page.

    **Only document 1 is scored against a benchmark case's truth file**, and it
    says so on the others rather than leaving them looking unscored for some
    other reason. `solution/<id>.fields.json` is a person's record of ONE
    document; scoring the second document of a split file against it would mark
    a correct extraction wrong for not being the first document.
    """
    documents, failed = [], 0
    for seg, record in zip(segments, records):
        yield {"event": "document", "document": record["document"],
               "documents": len(records), "pages": list(record["pages"]),
               "page_range": record["page_range"]}
        result = yield from _extract_document(seg["text"], mode, case_id,
                                              steps, doc_type, status, record,
                                              truth_fed, extras, schema_first)
        if not isinstance(result, dict):
            result = {"error": "extraction returned nothing"}
        result["document"] = record["document"]
        result["pages"] = list(record["pages"])
        result["page_range"] = record["page_range"]
        if case_id and not result.get("field_score"):
            # Scored where the truth file has a block for these pages, and
            # silent about it where it has none. **Not an error**: a pack's
            # truth file is filled in a document at a time, and a document
            # nobody has written answers for yet is unscored rather than wrong.
            result["field_truth_scope"] = (
                "the ground truth for %s states nothing for page%s %s"
                % (case_id, "" if len(record["pages"]) == 1 else "s",
                   record["page_range"]))
        if result.get("error"):
            failed += 1
        documents.append(result)
        yield {"event": "document_done", "document": record["document"],
               "documents": len(records),
               "filled": len(result.get("fields") or {}),
               "error": result.get("error") or ""}
    return _merge_documents(documents, records, mode, failed)


def _merge_documents(documents, records, mode: str, failed: int) -> dict:
    """One result for a file holding several documents.

    **`fields` is deliberately absent.** Two documents both state a
    `document_number`, and a merged dict would have to drop one of them or
    rename both -- either way producing a form nobody asked for and nobody can
    check. The documents are the answer and they are in `documents`; anything
    reading a multi-document result reads that list.

    What IS merged is the run-wide arithmetic, because those really are
    file-level facts: the clock, the tokens, the grounding ratio over every value
    the file produced. `grounding` carries document-prefixed paths so a flagged
    value can still be traced to the document it came from -- a bare
    `buyer_name` would name three different fields in a three-document file.

    `field_score` is not merged and is not summed: at most one document is
    scored (see `_extract_documents`), so the file's score is that document's
    and the scope says which. Summing the others in as zeroes would report a
    file as badly extracted for holding documents nobody has written an answer
    sheet for.
    """
    codes, asked, items = [], [], []
    for result in documents:
        for code in result.get("doc_types") or []:
            if code not in codes:
                codes.append(code)
        for key in result.get("fields_asked") or []:
            if key not in asked:
                asked.append(key)
        for key in result.get("items_asked") or []:
            if key not in items:
                items.append(key)
    scored = [result for result in documents if result.get("field_score")]
    merged = {
        "mode": mode,
        # The answer. Every reader of a multi-document result reads this.
        "documents": documents,
        "documents_found": len(documents),
        # How many of them came back with nothing. Counted rather than folded
        # into an error, because a file whose second document failed still
        # extracted its first and that is a result, not a failed run.
        "documents_failed": failed,
        "segments": records,
        "doc_types": codes,
        "doc_type": normalise.primary_type(codes),
        "doc_type_from": (documents[0].get("doc_type_from") or "") if documents else "",
        "fields_asked": asked,
        "items_asked": items,
        "seconds": round(sum(result.get("seconds") or 0 for result in documents), 2),
        "tokens": sum(result.get("tokens") or 0 for result in documents),
        "model": next((result.get("model") for result in documents
                       if result.get("model")), ""),
        "url": next((result.get("url") for result in documents
                     if result.get("url")), ""),
        "partial": any(result.get("partial") for result in documents),
        **({"parallel": max(result.get("parallel") or 0 for result in documents)}
           if any(result.get("parallel") for result in documents) else {}),
        "grounding": _merge_grounding(documents),
        # Every document's attribution as one for the file, paths prefixed with
        # the document they belong to exactly as grounding's are. The counts sum
        # because the score's counts pool the same way, which is what keeps the
        # run-log invariant true of a pack as well as of a page.
        "blame": blame.merge([(result.get("document") or index, result.get("blame"))
                              for index, result in enumerate(documents, 1)]),
        # Stage 0 of the FIRST document, so a page rendering the strip has
        # something true to draw before the reader picks a document. The
        # per-document classification is on each entry of `documents`.
        "classified": documents[0].get("classified") if documents else None,
    }
    if scored:
        merged["field_score"] = _merge_field_scores(scored, len(documents))
    # The item tables stay on their documents, like the fields: seven documents
    # rule seven tables and one merged table would be a table nobody printed.
    # What IS a file-level fact is the score over all of them.
    tabled = fieldscore.pool_tables([result.get("table_score")
                                     for result in documents])
    if tabled:
        merged["table_score"] = tabled
    fixed = fixscore.pool([result.get("fix_score") for result in documents])
    if fixed:
        merged["fix_score"] = fixed
    if failed and failed == len(documents):
        # Every document failed, so the run failed. Reported as an error rather
        # than as a result with nothing in it, which is the same rule the
        # all-steps-failed bail-out follows one level down.
        merged["error"] = documents[0].get("error") or "every document failed"
    return merged


def _merge_field_scores(scored, documents: int) -> dict:
    """Every scored document of one file, as ONE score for the file.

    **At the user's request** (2026-09-08): *if the doc got more we still score
    them as 1 doc but we do more of an average way.* A pack used to be scored on
    its first document alone, which reported a seven-document file on a seventh
    of itself and made a pack look like a small easy case.

    The arithmetic is `fieldscore.pool`, shared with `compare.py` -- pooled
    counts for the headline, with the unweighted mean of the documents' own
    rates beside it as `accuracy_macro`. Its docstring is where the reasoning
    lives; what is added here is only the file's own shape.
    """
    merged = fieldscore.pool([one["field_score"] for one in scored])
    merged["documents"] = documents
    # Which document each rate belongs to. `fieldscore.pool` knows the pages a
    # score was taken over and not the position of that document in the file,
    # which is what the page's strip is keyed on.
    for entry, one in zip(merged.get("per_document") or [], scored):
        entry["document"] = one.get("document")
    return merged


def _merge_grounding(documents) -> dict:
    """Every document's grounding, as one file-level audit.

    Paths are prefixed with the document they belong to, because `buyer_name`
    names a different field in each of them and a flagged value nobody can trace
    back to a document is a finding nobody can act on. The ratio is recomputed
    from the summed counts rather than averaged over the documents -- a
    one-field document and a thirty-field one are not two equal opinions about
    how much of this file was invented.
    """
    statuses, missing, flagged, counts, checked = {}, [], [], {}, 0
    for result in documents:
        found = result.get("grounding") or {}
        prefix = "doc%d." % (result.get("document") or 1)
        for path, status in (found.get("statuses") or {}).items():
            statuses[prefix + path] = status
        missing.extend(prefix + path for path in (found.get("missing") or []))
        for entry in found.get("flagged") or []:
            entry = dict(entry)
            if entry.get("field"):
                entry["field"] = prefix + entry["field"]
            flagged.append(entry)
        for name, count in (found.get("counts") or {}).items():
            counts[name] = counts.get(name, 0) + count
        checked += found.get("checked") or 0
    return {"statuses": statuses, "missing": missing, "flagged": flagged,
            "counts": counts, "checked": checked,
            "grounded_ratio": (round(counts.get("grounded", 0) / checked, 4)
                               if checked else None)}


def _validate_fields(result: dict, form: dict, text: str, mode: str) -> dict:
    """Attach the requirement's validation rules, and say which form they are for.

    Outside `fields`, like `derived` and for the same reason: `grounding.check`
    walks that dict and would report a computed verdict as an invention.

    Never allowed to break an extraction. The fields are the product and this is
    a measurement of them -- and the corpus is the standing reminder that a
    validator must not gate, since seventeen of twenty tax IDs in `solution/`
    fail the checksum for being anonymised mocks.
    """
    if not isinstance(result, dict):
        return result
    # Recorded even where nothing was extracted: which form was asked for is a
    # fact about the run, and a reader looking at an empty result needs it most.
    # Both: `doc_types` is what the page names and what the form was built
    # from, `doc_type` is the primary one to file it under. A single code alone
    # would repeat the bug fixed on 2026-08-31 -- it drops every type after the
    # first, and with them every field only the dropped type required.
    result["doc_types"] = list(form["doc_types"])
    result["doc_type"] = form["doc_type"]
    result["doc_type_from"] = form["doc_type_from"]
    # The line the type was read off, where one was. Reported for the same
    # reason `doc_type_from` is: the form below was chosen by this, and a choice
    # nobody can check is one nobody should have to take on trust.
    result["doc_type_heading"] = form.get("doc_type_heading") or ""
    # A disagreement with the manifest, where there is one. On the result as well
    # as on `classified`, because a result read back later is all a reader has.
    result["doc_type_expected"] = list(form.get("doc_type_expected") or [])
    result["doc_type_confidence"] = form.get("doc_type_confidence")
    result["doc_type_escalated"] = form.get("doc_type_escalated") or ""
    # The form itself, so the page renders the questions that were ASKED rather
    # than every key it knows how to draw. A key the form omits is not an empty
    # answer -- it is a question nobody put -- and an empty row for it would read
    # as a document that states nothing about it.
    result["fields_asked"] = list(form["keys"])
    # The table's cells, or []. Named apart from `fields_asked` for the reason
    # `_field_set` gives: the page draws them as a table and the scalars as rows.
    result["items_asked"] = list(form["items"])
    # The whole of stage 0, for a result being rendered after the fact -- a
    # re-extraction loaded from the Fields pane, or a blocking POST that had no
    # stream to carry the event.
    # The mode is passed in rather than read off the result: a run that failed
    # before either shape reported one still has to say whether ITS prompt was
    # framed, and defaulting that to single would describe the wrong run.
    result["classified"] = _classified(form, mode)
    if not result.get("fields"):
        return result
    result["normalized"] = normalise.normalize_fields(result["fields"])
    result["normalization_version"] = normalise.NORMALIZATION_VERSION
    try:
        result["validation"] = validate.validate(
            result["fields"], keys=form["keys"], doc_types=form["doc_types"],
            mandatory=form["mandatory"], transcript=text,
            items=form["items"])
    except Exception as err:  # pragma: no cover - never worth a 500
        result["validation"] = {"error": f"validation failed: {err}"}
    return result


def _score_fields(result: dict, case_id: str, form: dict = None,
                  pages=None, transcript: str = "",
                  truth_fed: bool = False) -> dict:
    """Attach the field score, when this document has field ground truth.

    Absent rather than an error when there is none: most documents are not
    benchmark cases, and a `field_score` holding only a complaint on every real
    upload would have to be filtered out by everything that reads a result.

    Scoring must never break an extraction that worked -- the fields are the
    product, the score is a measurement of it.
    """
    if not (case_id and isinstance(result, dict) and result.get("fields")):
        return result
    # A run restricted to some of the steps did not ask for the rest of the
    # schema, and scoring it would report every key nobody asked for as missed --
    # a headline accuracy of 3 values out of 43 for a run that got all three
    # right. The caller that restricted the steps knows which keys it wanted and
    # scores those itself.
    if result.get("steps_only"):
        return result
    try:
        if fieldscore.has_truth(case_id):
            # Scored over the form that RAN. A truth file states more than any
            # one form asks for -- it is the human's record of the page, not of
            # the schema -- so scoring every key in it would mark an invoice
            # wrong for not returning a currency nobody requested.
            # The headline is taken over what the requirement DEMANDS of this
            # document's types; an Optional field is still judged and reported,
            # under `optional`, and simply does not decide the rate.
            #
            # `mandatory` is EMPTY where no requirement covers the type, and
            # `fieldscore` reads that as its own state: score the base field
            # set and mark the rate `unknown_type`. Passing None instead would
            # score the same keys and lose the distinction, which is the half a
            # reader needs -- a rate over a requirement and a rate over
            # everything the ground truth states are different claims.
            result["field_score"] = fieldscore.evaluate(
                case_id, result["fields"],
                keys=(form or {}).get("keys"),
                doc_types=(form or {}).get("doc_types") or [],
                mandatory=(form or {}).get("mandatory"),
                items_mandatory=(form or {}).get("mandatory_items"),
                # Which document of the file this is. A pack's truth file holds
                # one block per document, so without this every document would
                # be marked against the first one's answers.
                pages=pages)
            # Which pass lost each value the score just charged for. Attached
            # here rather than anywhere else for the same reason the score is:
            # this is the one place a result has its fields, its truth and the
            # transcript they were read from all in hand at once.
            result["blame"] = _attribute_blame(result["field_score"], case_id,
                                               transcript, truth_fed)
    except Exception as err:  # pragma: no cover - a score is never worth a 500
        result["field_score"] = {"error": f"field scoring failed: {err}"}
    return result


def _page_text(case_id: str) -> str:
    """A case's own ground-truth transcript, or "" where there is none.

    Read whole rather than cut to this document's pages, and that is the
    conservative direction on both of the questions `blame` asks it: a value
    found anywhere in the file's truth is a value the page really prints, so the
    only attributions this can move are the two that would otherwise have
    accused somebody -- never the other way about.
    """
    try:
        case = scoring.cases_index().get(case_id)
        return case["ground_truth"].read_text("utf-8") if case else ""
    except Exception:  # pragma: no cover - a missing page is not a failure
        return ""


def _attribute_blame(score, case_id: str, transcript: str,
                     truth_fed: bool) -> dict:
    """`blame.check`, guarded, because an attribution is never worth a run.

    It moves no score and no verdict -- see `blame.py`, which is emphatic about
    that -- so a failure here costs a note and must not cost the fields.
    """
    if _image_fed():
        return {"skipped": "no transcript: the extractor read the page images"}
    try:
        return blame.check(score, transcript, _page_text(case_id), truth_fed)
    except Exception as err:  # pragma: no cover
        return {"skipped": f"attribution failed: {err}"}


def _drain(generator, each=None):
    """Run a generator that returns a value, discarding what it yields -- or
    handing each event to `each`."""
    while True:
        try:
            event = next(generator)
        except StopIteration as stop:
            return stop.value
        if each is not None:
            each(event)


def extract_fields(text: str, mode: str = None, case_id: str = None,
                   steps=None, doc_type: str = None, pages=None,
                   truth_fed: bool = False, extras: bool = None,
                   each=None, schema_first: bool = False) -> dict:
    """`extract_fields_stream` for callers with nowhere to show progress.
    `each`, where given, sees every event (the stress test reads which document
    is being extracted off them)."""
    return _drain(extract_fields_stream(text, mode, case_id, steps, doc_type,
                                        pages, truth_fed, extras, schema_first), each)


def strip_fence(text: str) -> str:
    """Unwrap a whole-output ```markdown fence.

    A stock Qwen3-VL often wraps the entire page in a fence, which would then
    render as a code block instead of as the table/headings it describes.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first, sep, rest = stripped.partition("\n")
    if not sep or not rest.rstrip().endswith("```"):
        return stripped
    # Only unwrap when the opening fence has no content of its own (```markdown).
    if first[3:].strip().lower() not in ("", "markdown", "md", "html"):
        return stripped
    return rest.rstrip()[:-3].rstrip()


_PAGE_NO = re.compile(r"<page_number>\s*(.*?)\s*</page_number>", re.I | re.S)
_FIGURE = re.compile(r"</?figure>\s*", re.I)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)


def normalise_output(text: str) -> str:
    """Strip structural markup the model emits despite being told not to.

    typhoon-ocr is finetuned to wrap page numbers in <page_number> and to open with
    a Markdown '# ' heading; prompting does not suppress either. The page number is
    real page content so it is kept, just unwrapped -- the tag is not.
    """
    text = _PAGE_NO.sub(r"\1", text)
    text = _FIGURE.sub("", text)
    text = _HEADING.sub("", text)
    # The model HTML-escapes ampersands even in plain prose ("RESORT &amp; VILLAS").
    # Deliberately not unescaping &lt;/&gt;, which would invent tags.
    for entity, char in (("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'),
                         ("&#39;", "'"), ("&apos;", "'")):
        text = text.replace(entity, char)
    return text.strip()


def layout_text(raw: str) -> str:
    """Flatten a dots.ocr layout reply into a transcript, in reading order.

    The profile asks for a JSON array of blocks -- `bbox`, `category`, `text` --
    already sorted the way a person would read the page, so the transcript is
    those texts joined. Everything downstream (scoring, pass 2, the run log) then
    sees the same kind of string it sees from any other profile, which is the
    whole point of doing this here rather than teaching five other modules about
    layout blocks.

    A reply that never closed is salvaged the same way `_salvage_json` salvages
    pass 2: keep the blocks that finished, drop the one it died inside. This is
    the common case rather than an edge -- the model runs to the token cap inside
    a single block on both fixtures it was measured on -- and a partial page is
    worth more than nothing, with `looks_repetitive` and the truncation flag
    already saying the read did not finish.

    `Picture` blocks carry no text by the prompt's own rule, so they contribute
    nothing rather than an empty line.
    """
    return "\n".join(b["text"] for b in layout_blocks(raw) if b["text"])


def layout_blocks(raw: str) -> list:
    """The blocks of a layout reply, kept whole: bbox, category and text.

    `layout_text` throws the geometry away because a transcript has no room for
    it. The page's Layout view is the one thing that wants it, so the parsing
    lives here and both callers share it -- two parsers for one reply would
    disagree about the salvage and put the boxes out of step with the text
    beside them.

    Coordinates are the model's own, in the pixel space of the image it was
    sent, and are passed through unchecked beyond being four numbers: a box in
    the wrong place is a finding, not something to quietly correct. A block
    whose bbox is missing or malformed keeps its text and gets `None`, so it
    still appears in the list the transcript was built from.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        blocks = json.loads(raw)
    except ValueError:
        cut = raw.rfind("},")
        if cut < 0:
            return []
        try:
            blocks = json.loads(raw[:cut + 1] + "]")
        except ValueError:
            return []
    if isinstance(blocks, dict):
        # The prompt says "a single JSON object", and a model that takes that
        # literally wraps the array in one. Take the first list it holds.
        blocks = next((v for v in blocks.values() if isinstance(v, list)), [])
    if not isinstance(blocks, list):
        return []
    out = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        box = block.get("bbox")
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(n, (int, float)) and not isinstance(n, bool)
                        for n in box)):
            box = None
        out.append({
            "bbox": box,
            "category": str(block.get("category") or "")[:40],
            "text": (block.get("text") or "").strip(),
        })
    return out


def finish_page(raw: str, stats: dict = None, profile: str = None) -> str:
    """One page's raw reply -> its transcript, with the boxes kept on `stats`.

    Called wherever a page finishes -- the streaming endpoint, the blocking one
    and the random test -- so all three produce the same thing from the same
    bytes. `layout` rides on the page stats, which already travel to the browser
    in `page_done` and in `summarise`'s `page_stats`, so nothing new had to be
    plumbed for it; `runlog.record` names the columns it writes, so the log is
    unaffected.

    Only a layout profile sets the key at all. On a Markdown profile it is
    absent rather than empty, which is what lets the page tell "this run had no
    geometry to show" from "this page found nothing".
    """
    profile = profile or (stats or {}).get("ocr_profile")
    spec = profile_spec(profile)
    text = read_reply(raw, profile)
    if stats is not None:
        # The reply exactly as it arrived, before the fence, the layout flatten
        # and normalise_output have had it. Kept because every one of those steps
        # throws something away on purpose, and the only way to check what was
        # thrown away is to see what came in -- the coordinates of a layout reply
        # most of all, which the transcript cannot carry.
        stats["raw"] = raw
        if spec["reply"] == "layout_json":
            stats["layout"] = layout_blocks(strip_fence(raw))
    return text


def read_reply(raw: str, profile: str = None) -> str:
    """One profile's raw answer, turned into the transcript everything else uses."""
    spec = profile_spec(profile)
    if spec["reply"] == "layout_json":
        return normalise_output(layout_text(strip_fence(raw)))
    return normalise_output(strip_fence(raw))


def loop_retry_page(page, stats: dict):
    """The page to read again after a looped read, one Detail up -- or None.

    Only a page `prepare_input` tagged can climb: the tag holds the trimmed page
    before it was scaled, so the retry is rescaled from that rather than enlarged
    from the smaller image. None when LOOP_RETRY is off, the read did not loop,
    the page is already at `original`, or the next budget would give back the
    same image (a page small enough to fit the lower one already).
    """
    origin = getattr(page, "_loop_retry", None)
    if not (LOOP_RETRY and origin and stats.get("looped")):
        return None
    higher = next_detail(origin["detail"])
    if higher is None:
        return None
    bigger = fit_pixels(origin["source"], DETAIL_PRESETS[higher])
    if bigger.size == page.size:
        return None
    bigger._loop_retry = dict(origin, detail=higher)
    if hasattr(page, "_dry_label"):
        tag_page(bigger, page._dry_label, higher)
    replace_job_page(origin["job"], origin["index"], bigger)
    return bigger


def record_retry(stats: dict, was, now) -> None:
    """Set the looped attempt aside on `stats`, so the retry fills it afresh.

    The page's figures end up describing the read that was kept; what each
    looped attempt cost is on `loop_retries`, and `detail` says where it ended.
    """
    attempts = stats.get("loop_retries", []) + [{
        "detail": was._loop_retry["detail"],
        **{key: stats.get(key) for key in ("resolution", "new_tokens", "seconds")}}]
    stats.clear()
    stats.update(loop_retries=attempts, detail=now._loop_retry["detail"])


def read_page(image: Image.Image, stats: dict = None,
              profile: str = None, cancel=None) -> str:
    """Blocking full-page read.

    Deliberately drains the streaming generator rather than issuing its own
    request, so both endpoints report timings measured exactly the same way.

    `cancel` is a `Cancel` that stops the drain between tokens, and hangs up the
    request underneath for a stream that has stalled with no token to check
    between. It is how a random test's Stop reaches the page in flight; the
    caller is expected to throw the partial away, because half a page is not a
    measurement. A page still in prefill finishes its prefill first -- see
    `Cancel`. Without one this drains to the end exactly as it always has.

    A read that loops is read again one Detail up -- see `loop_retry_page`.
    """
    stats = {} if stats is None else stats
    while True:
        parts = []
        for chunk in stream_page(image, stats, profile, cancel):
            parts.append(chunk)
            if cancel is not None and cancel.is_set():
                break
        if cancel is not None and cancel.is_set():
            break
        retry = loop_retry_page(image, stats)
        if retry is None:
            break
        record_retry(stats, image, retry)
        image = retry
    return finish_page("".join(parts), stats, profile)


def _page_parallelism(pages: int) -> int:
    """How many pages of one read go to the server at once: 1 means one at a time.

    More than one only for a document of `OCR_PARALLEL_MIN_PAGES` or more, on a
    server that batches concurrent requests (`backends.parallel_ok`, vLLM), with
    `OCR_PARALLEL` on. Capped by `OCR_PARALLEL_MAX` where it is set. The server
    is only asked about once the page count says it could matter, so a short
    read costs no status lookup it did not already make.
    """
    if not (OCR_PARALLEL and pages >= OCR_PARALLEL_MIN_PAGES):
        return 1
    if not backends.parallel_ok(llama_status()):
        return 1
    return max(1, min(OCR_PARALLEL_MAX or pages, pages))


class _Hangup:
    """Something a one-slot `Cancel` can hang up that stops MANY requests.

    A random test hands its read one `Cancel`, which holds one live request; a
    read whose pages are in flight together has several. The read attaches this
    to the round's `Cancel` and gives the pages a `MultiCancel` of their own, so
    the round's Stop reaches every page.
    """

    def __init__(self, cancel):
        self.cancel = cancel

    def close(self):
        self.cancel.set()


def read_pages_at_once(pages, parallel: int, cancel=None, run: dict = None):
    """Read `pages` `parallel` at a time; yield `(number, text, stats)` in PAGE order.

    The split-and-join that lets a 13-page file take about as long as its
    slowest page rather than all thirteen (see `OCR_PARALLEL`). Every page goes
    through `read_page`, exactly as a sequential read sends it, and the results
    are handed back by page number -- page 3 is yielded only once pages 1 and 2
    have been -- so the caller joins them in document order whatever order the
    server finished them in, and the transcript is the one a sequential read
    gives.

    A page that fails stops the rest: the others are hung up and the error is
    raised as soon as it is known, rather than after every page before it in
    the document has finished. So does closing this generator -- the browser's
    Stop on the streaming route.

    `run`, where given, gets `parallel` and `server`: the server's own timing
    of the read as one span (`servertime.Batch`), because a /metrics difference
    around any one page covers its neighbours and the pages themselves carry
    none.
    """
    inner = MultiCancel()
    if cancel is not None:
        cancel.attach(_Hangup(inner))
    batch = servertime.Batch(llama_status())

    def one(page):
        stats = {}
        return read_page(page, stats, cancel=inner), stats

    pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="ocr-page")
    futures = [pool.submit(workspace.bind(batch.run), one, page) for page in pages]
    complete = False
    try:
        pending, ready = set(futures), 0
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()             # a failed page raises here, at once
            while ready < len(futures) and futures[ready].done():
                text, stats = futures[ready].result()
                ready += 1
                yield ready, text, stats
        complete = True
    finally:
        if not complete:
            inner.set()
        pool.shutdown(wait=True, cancel_futures=True)
        server = batch.close()
        if run is not None:
            run.update(parallel=parallel, server=server)


def read_all_pages(pages, page_texts: list, all_stats: list, cancel=None,
                   run: dict = None) -> None:
    """Read every page into the caller's lists, at once where that is allowed.

    The lists are the caller's so the pages that finished before an error or a
    Stop are still there to summarise. Sequential exactly as before where
    `_page_parallelism` says 1, stopping between pages on `cancel`.
    """
    parallel = _page_parallelism(len(pages))
    if parallel > 1:
        if stop_requested(cancel):
            return
        with closing(read_pages_at_once(pages, parallel, cancel, run)) as reads:
            for _, text, stats in reads:
                page_texts.append(text)
                all_stats.append(stats)
        return
    for page in pages:
        if stop_requested(cancel):
            break
        stats = {}
        page_texts.append(read_page(page, stats, cancel=cancel))
        all_stats.append(stats)


# --------------------------------------------------------------------------
# file loading
# --------------------------------------------------------------------------

def _looks_like_pdf(data: bytes) -> bool:
    return data[:1024].lstrip()[:5] == b"%PDF-"


def _capped(items, limit: int):
    """`items`, at most `limit` of them -- and ALL of them when `limit` is 0.

    `zip(items, range(limit))` was the old spelling and it reads as harmless
    until the limit is zero, where it yields nothing rather than everything.
    """
    for index, item in enumerate(items):
        if limit and index >= limit:
            return
        yield item


def load_pages(data: bytes):
    """Decode an upload into `(pages, total)` -- the pages READ and the pages IT HAS.

    **The two differ, silently, and that is why this returns both** (2026-09-16).
    `MAX_PAGES` caps what is rasterised, and until now the cap was applied with
    `min()` and nothing else: a hundred-page file came back as ten pages, the run
    reported `status=ok` and `pages=10`, and a benchmark document was then scored
    against the ground truth of all of it. That is the silent-drop-at-HTTP-200
    class this project is organised around, in the one place that had no symptom
    at all -- and it is live on `sol023`, which has 13 pages against a default cap
    of 10.

    `total` is what the file holds. Every caller is expected to carry it: the
    preview reports it before a run is paid for, and `prepare_input` puts it on
    the job so the result and the run log can say the read was partial.
    """
    if _looks_like_pdf(data):
        if not PDF_OK:
            raise ValueError("PDF support needs PyMuPDF: pip install pymupdf")
        pages = []
        zoom = PDF_DPI / 72.0
        with _PDF_LOCK, fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("That PDF is password-protected.")
            total = doc.page_count
            # MAX_PAGES 0 is no cap. A bare `min(total, MAX_PAGES)` reads
            # NOTHING there rather than everything -- the same trap as the
            # `zip(..., range(limit))` in the frame branch, and both are why
            # `_capped` exists instead of either spelling.
            for page in doc.pages(0, min(total, MAX_PAGES) if MAX_PAGES else total):
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        if not pages:
            raise ValueError("That PDF has no pages.")
        return pages, total

    try:
        image = Image.open(io.BytesIO(data))
        # Multi-frame formats (TIFF scans, GIF) carry a page per frame.
        # A multi-frame image is capped the same way, so its total is counted
        # the same way -- `n_frames` where the format reports one, else the
        # frames actually decoded (a single-frame image has exactly one).
        frames = [
            ImageOps.exif_transpose(frame).convert("RGB")
            for frame in _capped(ImageSequence.Iterator(image), MAX_PAGES)
        ]
        total = int(getattr(image, "n_frames", len(frames)) or len(frames))
        # A frame can itself hold several pages (`split_stacked_pages`). Every
        # piece is a page from here on: the cap counts pieces, and so does the
        # total -- each frame left undecoded still counted as one.
        if SPLIT_STACKED_PAGES:
            pieces = [piece for frame in frames for piece in split_stacked_pages(frame)]
            total += len(pieces) - len(frames)
            frames = list(_capped(pieces, MAX_PAGES))
    except UnidentifiedImageError:
        hint = "" if HEIF_OK else " (HEIC/HEIF needs: pip install pillow-heif)"
        raise ValueError(f"Unsupported or corrupt file format.{hint}") from None
    except Exception:
        raise ValueError("Could not read that file.") from None

    if not frames:
        raise ValueError("That file contains no image data.")
    return frames, total


# A4's height over its width, and the share of a piece either side of an expected
# boundary that the cut may move into to find the gap between two pages.
_PAGE_RATIO = 2 ** 0.5
_SPLIT_WINDOW = 0.15


def split_stacked_pages(image: Image.Image, ratio: float = None) -> list:
    """Cut a frame that holds several pages into one image per page.

    A scanner that stitches a batch into a single TIFF frame produces an image
    two or three A4 pages tall, and it looks like one page in a viewer. Read
    whole, every page of it lands in one request: the reply is two or three
    pages long and runs into `MAX_NEW_TOKENS`, and the pages are downscaled to
    one page's pixel budget on the way, so the text is half the size it should
    be.

    Only a frame longer than `ratio` times its width (default
    `SPLIT_PAGE_RATIO`) is cut, into as many A4-shaped pieces as its length
    holds, and the same across for one wider than `ratio` times its height.
    Each cut is moved to the middle of the longest blank band within
    `_SPLIT_WINDOW` of where an A4 boundary would be, so it lands in the
    margin between two pages -- or, on a long receipt, between two lines --
    rather than through a line of text. Anything else comes back as `[image]`.
    """
    ratio = ratio or SPLIT_PAGE_RATIO
    w, h = image.size
    if h >= w * ratio:
        tall, length, across = True, h, w
    elif w >= h * ratio:
        tall, length, across = False, w, h
    else:
        return [image]
    count = max(2, round(length / across / _PAGE_RATIO))
    # Ink per row (or column), 0-255, from a 1-pixel-wide box average of the
    # thresholded page. A row at the bottom of its window is one no text crosses.
    ink = image.convert("L").point(lambda p: 255 if p < 160 else 0)
    profile = list(ink.resize((1, h) if tall else (w, 1), Image.BOX).getdata())
    step = length / count
    reach = int(step * _SPLIT_WINDOW)
    cuts = [0]
    for k in range(1, count):
        target = int(step * k)
        lo, hi = max(cuts[-1] + 1, target - reach), min(length - 1, target + reach)
        floor = min(profile[lo:hi + 1])
        best, run_start = None, None
        for at in range(lo, hi + 2):
            blank = at <= hi and profile[at] <= floor + 2
            if blank and run_start is None:
                run_start = at
            elif not blank and run_start is not None:
                middle = (run_start + at - 1) // 2
                key = (at - run_start, -abs(middle - target))
                if best is None or key > best[0]:
                    best = (key, middle)
                run_start = None
        cuts.append(best[1])
    cuts.append(length)
    return [image.crop((0, a, w, b) if tall else (a, 0, b, h))
            for a, b in zip(cuts, cuts[1:])]


def trim_margins(image: Image.Image, tolerance: int = TRIM_TOLERANCE,
                 pad: int = TRIM_PAD) -> Image.Image:
    """Crop uniform blank borders.

    Measured on real 300 DPI invoices this removes 0-18% of the page. Note what that
    does and does not buy:

    * At a capped Detail (the default) it saves NO time -- both the trimmed and
      untrimmed page are scaled to the same pixel budget, so the model sees the same
      token count either way. What it buys is quality: the budget is spent on content
      instead of margin, rendering text ~1.10x larger at the same cost.
    * At Detail `original` (uncapped) it is a genuine ~13% pixel, and therefore prefill,
      saving.

    Conservative by design: it only trims a genuinely uniform border, never more
    than 45% of either axis, and leaves a small pad so nothing is clipped.
    """
    gray = image.convert("L")
    # Background colour taken from the corners; a scan border is whatever the
    # corners agree on.
    w, h = gray.size
    corners = [
        gray.getpixel((0, 0)),
        gray.getpixel((w - 1, 0)),
        gray.getpixel((0, h - 1)),
        gray.getpixel((w - 1, h - 1)),
    ]
    background = max(set(corners), key=corners.count)
    if corners.count(background) < 3:
        return image  # not a uniform border, leave it alone

    flat = Image.new("L", gray.size, background)
    diff = ImageChops.difference(gray, flat).point(
        lambda p: 255 if p > tolerance else 0
    )
    box = diff.getbbox()
    if box is None:
        return image  # blank page

    left, top, right, bottom = box
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w, right + pad)
    bottom = min(h, bottom + pad)

    if right - left < w * 0.55 or bottom - top < h * 0.55:
        return image  # suspiciously aggressive, keep the original
    if (right - left) * (bottom - top) >= w * h * 0.98:
        return image  # nothing meaningful to gain
    return image.crop((left, top, right, bottom))


def fit_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Scale down so width*height stays under the budget. 0 means no cap."""
    total = image.width * image.height
    if max_pixels <= 0 or total <= max_pixels:
        return image
    scale = (max_pixels / total) ** 0.5
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def join_page_texts(page_texts) -> str:
    if len(page_texts) > 1:
        return "\n\n".join(f"--- page {i} ---\n{text}"
                           for i, text in enumerate(page_texts, 1) if text)
    return page_texts[0] if page_texts else ""


def _page_coverage(job_id, counted):
    """`{"pages_total": n, "pages_truncated": n}` for a read, or {}.

    Read back off the job rather than threaded through every summariser's
    arguments: `prepare_input` put it there and each of these already takes the
    job id. **Empty where the total is unknown** -- a re-extraction has no job,
    and a blank is not a claim that nothing was dropped.
    """
    total = (job_context(job_id) or {}).get("pages_total") if job_id else None
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
        "ocr_profile": profiles[0] if profiles else ocr_profile(),
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


def prepare_input(data: bytes, detail: str, case=None, source=None, *, cache=True, context_out=None):
    """Decode, trim and resize. Returns (pages, detail, page_job_id, case).

    `source` is carried into the job cache rather than only into the run log, so
    a later re-extraction against this transcript can name the same document.
    The user portal sets cache=False and receives metadata in context_out: it
    keeps the original on disk and needs no encoded preview in the RAM cache.
    """
    # Through the alias table, so an old preset name from a saved setting or a
    # script lands on the nearest preset that still exists rather than silently
    # on the default. See `settings.resolve_detail`.
    detail = resolve_detail(detail)
    pages, total = load_pages(data)
    budget = DETAIL_PRESETS[detail]
    # Trim first, then fit: the pixel budget is spent on content, not margins.
    trimmed = [trim_margins(page) if TRIM_MARGINS else page for page in pages]
    prepared = [fit_pixels(page, budget) for page in trimmed]
    context = {"source": source or {}, "detail": detail, "pages": len(prepared),
               # What the FILE holds, against what was read. Carried on the job
               # rather than threaded through a wider return, because `summarise`
               # already has the job id and `job_context` already exists -- and a
               # fifth element on this tuple would have to be unpicked at every
               # caller, which is how one of them ends up dropping it.
               "pages_total": total,
               "case": case["id"] if case else ""}
    if context_out is not None:
        context_out.update(context)
    job_id = register_job(prepared, context) if cache else None
    label = context["case"] or (source or {}).get("name") or "upload"
    for index, page in enumerate(prepared):
        tag_page(page, f"{label} p{index + 1}/{len(prepared)}", detail)
    # Where a looped read climbs from (`loop_retry_page`). Untagged at the top
    # Detail, which has nowhere to go, so only a lower one holds the trimmed page.
    if LOOP_RETRY and next_detail(detail):
        for index, (page, trim) in enumerate(zip(prepared, trimmed)):
            page._loop_retry = {"source": trim, "detail": detail,
                                "job": job_id, "index": index}
    return prepared, detail, job_id, case


def _stamp_model_meta(summary) -> None:
    """Put the quantisation and engine version on a summary and its extraction.

    Here, in the one funnel every logged run goes through, for the read floor's
    reason: stamped at each call site, it would be stamped at most of them. Each
    pass is described by ITS OWN model and server -- pass 2 may run on a second
    server (`extract_url`) and a second model -- and an existing value is kept,
    so a caller that already knows is never overruled. Never raises.
    """
    if not isinstance(summary, dict):
        return
    try:
        if summary.get("model") and not summary.get("engine"):
            meta = backends.model_meta(summary.get("url"), summary.get("model"))
            summary.setdefault("quant", meta["quant"])
            summary["engine"] = meta["engine"]
        extracted = summary.get("extracted")
        if (isinstance(extracted, dict) and extracted.get("model")
                and not extracted.get("engine")):
            meta = backends.model_meta(extracted.get("url") or summary.get("url"),
                                       extracted.get("model"))
            extracted.setdefault("quant", meta["quant"])
            extracted["engine"] = meta["engine"]
    except Exception:
        pass


def log_run(summary: dict, source: dict, status: str = None, error=None,
            run_type: str = "ocr"):
    """Append one line to the run log. Never raises.

    Measurements only -- no transcript, no fields, no images. See `runlog.py`.

    The row a read writes is remembered against its page job, so a later
    re-extraction of the same transcript can find it again and put better figures
    on it. Timestamp and file name are the whole key -- nothing about the row is
    held open, and losing it costs the update, not the run.

    **The read floor is applied here**, before the row is built, because this is
    the one funnel every read path goes through -- the page, the random test, the
    blocking route and the random test. Applying it at each of those instead is
    how it came to be applied at exactly one of them. See `apply_read_floor`.
    """
    try:
        apply_read_floor(summary if isinstance(summary, dict) else {})
    except Exception:
        pass
    _stamp_model_meta(summary)
    try:
        row = runlog.record(summary, source,
                            {"status": status, "error": str(error) if error else "",
                             "run_type": run_type})
        job_id = summary.get("job") if isinstance(summary, dict) else None
        if row and job_id and run_type == "ocr":
            with _jobs_lock:
                if job_id in _job_meta:
                    _job_meta[job_id]["log"] = {"timestamp": row["timestamp"],
                                                "file": row["file"]}
    except Exception:
        pass


def log_extract(result: dict, job_id: str = None, context: dict = None,
                extra: dict = None):
    """Append a row for a pass-2-only run -- the Re-extract button, or a script.

    Extraction can be re-run any number of times against one transcript, and each
    run has its own mode, its own grounded ratio and its own field counts. Before
    this, none of that reached the log: the row written when the page was read
    kept whichever extraction happened to run with it, and every later one was
    invisible.

    It is a new row rather than an edit to that one. The file is append-only by
    construction (`runlog._migrate` only ever adds columns), a row records what
    one request did at one time, and the first extraction is a measurement in its
    own right -- overwriting it would destroy the before/after that re-extracting
    exists to produce.

    The pass-1 columns stay blank: nothing re-read the page. `job_id` recovers
    which document this was, from the same cache the compare view reads its
    images out of.

    `context` supplied outright is for the extraction that never had a page job:
    a fields-only run reads its transcript out of `solution/<id>.md`, so the
    document is known while nothing in this process ever read a page of it. It
    carries no `log` key, so `update_extract` below finds no read row to improve
    -- correctly: those figures belong to a transcript this run did not use.
    """
    result = result or {}
    context = job_context(job_id) if context is None else context
    error = result.get("error") or ""
    summary = {
        # Pass 1 did not run, so page_count/seconds/tokens are left out entirely
        # rather than passed as zero -- see the blank-is-not-zero rule in runlog.
        "detail": context.get("detail", ""),
        # Falls back to the model pass 2 was ABOUT to run on. An extraction that
        # failed before it began -- a refused model, an unreachable server --
        # carries no model of its own, and a failure attributed to nobody cannot
        # be counted against the setting that produced it. That is how ten
        # straight refusals sat in this file invisible to every failure rate in
        # it.
        "model": result.get("model") or backends.extract_status()["model"] or "",
        "backend": result.get("backend") or "",
        # No page was read, so the row's server is the one that EXTRACTED --
        # which is a different box from the reader's where pass 2 has its own.
        "url": result.get("url") or backends.extract_url(),
        "extracted": result,
        # Enough for the `case` column to name the document; the accuracy scores
        # belong to the read that produced the transcript, not to this row.
        "truth": {"case": context.get("case", "")},
        # Cells the caller knows and this function does not -- a stress job's
        # run id and concurrency.
        **(extra or {}),
    }
    status = ("error" if error
              else "partial" if result.get("partial")
              else "ok")
    log_run(summary, context.get("source") or {}, status=status, error=error,
            run_type="extract")

    # And, when this extraction beat the one logged with the read, put its
    # figures on that row too: the read row is what a per-document view of the
    # log reports, and leaving the worse extraction there means the better one is
    # only ever seen by someone who reads to the bottom of the file. The appended
    # row above keeps the history either way, and a worse re-extraction changes
    # nothing.
    outcome = None
    try:
        outcome = runlog.update_extract(context.get("log"), summary)
    except Exception:
        pass
    if outcome and outcome.get("updated"):
        # stderr, where the request log already goes: `say`'s print to stdout is
        # buffered when stdout is a pipe, and a notice about a row that changed
        # under a fixed timestamp is worth nothing if it surfaces minutes later.
        config.say(f"[ocr] re-extract improved {context.get('case') or 'the read'} "
                   f"{outcome['before']} -> {outcome['after']}; run-log row updated",
                   stream=sys.stderr)
    return outcome


def describe_source(name: str, data: bytes, origin: str) -> dict:
    """What the run log needs to identify the input: name, size and where from.

    Size is measured on the bytes actually read, so a case PDF and the same file
    uploaded by hand report identically.
    """
    return {"name": name, "size_bytes": len(data or b""), "origin": origin}


def request_input(request_files, form, match: bool = True):
    """The bytes a request is asking about, however the request named them.

    Three shapes -- a benchmark case, a file in mockOcr/, an upload -- resolve to
    one `(data, case, source)`. Shared by the read endpoints and by the preview,
    so a preview cannot be of a different document from the read that follows it.

    `match=False` skips the ground-truth lookup for callers that do not score. It
    hashes the upload, and the page asks `POST /api/match` for that separately.
    """
    case_id = (form.get("case") or "").strip()
    mock_name = (form.get("file") or "").strip()
    if case_id:
        data, case = case_bytes(case_id)
        return data, case, describe_source(case["pdf"], data, "case")
    if mock_name:
        data, case = resolve_mock(mock_name)
        return data, case, describe_source(mock_name, data, "folder")
    upload = request_files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("No file uploaded.")
    data = upload.read()
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
    return (*prepare_input(data, form.get("detail", DEFAULT_DETAIL), case, source),
            source)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------



@app.get("/")
def index():
    return render_template(
        "index.html",
        server=llama_status(),
        details=list(DETAIL_PRESETS),
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
        extract_mode=extract_mode(),
        extract_extras=extract_extras(),
        # A RANGE, not a count: the step table is per document type now, so an
        # invoice runs six of the eight steps and a credit note all eight. The
        # picker is drawn before any document is chosen, so the only honest
        # label is the span -- a flat "8 steps" beside a run that made six reads
        # as a run that lost two.
        extract_steps=_step_count_label(),
        hybrid_steps=_step_count_label("hybrid"),
        ocr_profile=ocr_profile(),
        ocr_profiles=[{"id": pid, **spec} for pid, spec in OCR_PROFILES.items()],
        # Whether a cycling read is cut short, and what it would cost if it were
        # not: the page prints the cap rather than repeating a number, for the
        # same reason the read floor below is sent instead of hardcoded.
        loop_guard=loop_guard(),
        ocr_sampling=ocr_sampling_state(),
        table_agents=table_agents(),
        max_new_tokens=MAX_NEW_TOKENS,
        # When a field score is worth writing, for every path that reads a page.
        # Sent rather than hardcoded in the template: it is a setting, it moves
        # from two panes now, and a page that says 50% while the process runs at
        # 0 is describing a build nobody has. `read_floor()` rather than the
        # constant, so a reload shows what is in force and not what started.
        min_read_for_fields=read_floor(),
        min_read_default=MIN_READ_FOR_FIELDS,
        # The bar the Python classifier has to clear before the model is
        # not asked. Sent rather than repeated in the template: a page
        # quoting a threshold the process is not running describes a build
        # nobody has -- the same rule `min_read_for_fields` follows.
        min_classify_confidence=CLASSIFY_MIN_CONFIDENCE,
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
    )


@app.get("/api/health")
def health():
    return jsonify(server=llama_status(), pdf=PDF_OK, heif=HEIF_OK, status="ok")


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
    return jsonify(num_ctx=value, server=llama_status())


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
        set_ocr_profile(backends.profile_for_model(current["model"]))
    response = jsonify(**overview, ocr_profile=ocr_profile())
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
            set_ocr_profile(backends.profile_for_model(backends.status()["model"]))
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
                    "ocr_profile": ocr_profile()})


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


@app.get("/api/schema")
def schema():
    """Every document type this app knows, and the form each one asks for.

    Reference data, not a run: no model server is touched, nothing is logged, and
    the answer is the same on every call. It is what the **Doc types** tab draws.

    **It is assembled here rather than in the template, and that is the whole
    point of it being a route.** Which keys a requirement asks for, which of them
    it makes Mandatory, and which validation rules a type adds are facts stated
    once -- in `prompts.DOC_TYPE_FIELDS`, `prompts.MANDATORY_FIELDS`,
    `prompts.TYPE_RULES` and `validate.RULE_NOTES` -- and a second copy in
    JavaScript is exactly how fifteen keys silently vanished from the Fields tab
    on 2026-08-19. The page owns the wording of a label and nothing else.

    Three groups, and the split IS the answer to "what do we know":

    * a type with a requirement -- four of them -- carries its own field list.
    * a type the classifier recognises but no requirement covers contributes NO
      key (`fields` is empty), which is `DOC_TYPE_FIELDS`' standing rule. It is
      still listed, because being classifiable and being scoreable are different
      things and a reader looking for TAX_INVOICE should find it rather than
      conclude the page is broken.
    * `unknown` is what a document reduces to when no type in play has a
      requirement -- the base field set, `prompts.DEFAULT_FIELDS`, which is the
      union of every requirement and is Mandatory in none of it.
    """
    def form(code):
        own = tuple(prompts.DOC_TYPE_FIELDS.get(code, ()))
        return {
            "code": code,
            # The article form, as a prompt says it -- "an invoice", "a
            # withholding tax certificate (50 thawi)". The page trims the article.
            "name": TYPE_NAMES[code],
            # The printed wordings that classify a page as this. Raw, not the
            # squashed needles that do the matching -- see normalise.TYPE_HEADINGS.
            "headings": list(normalise.TYPE_HEADINGS.get(code, ())),
            # This type's OWN contribution to a form, which is empty for a type
            # no requirement covers. Not `fields_for_types([code])`: that falls
            # back to the base set for such a type, which is a true answer to a
            # different question and would read here as a requirement.
            "fields": list(own),
            "mandatory": list(prompts.MANDATORY_FIELDS.get(code, ())),
            "items": list(prompts.DOC_TYPE_ITEMS.get(code, ())),
            "mandatory_items": list(prompts.MANDATORY_ITEMS.get(code, ())),
            "rules": list(prompts.TYPE_RULES.get(code, ())),
            "has_requirement": bool(own),
        }

    return jsonify(
        # Most specific first, the order everything else ranks these in.
        types=[form(code) for code in prompts.TYPE_SPECIFICITY],
        # What is asked when no type in play has a requirement. `mandatory` is
        # empty on purpose and is not an oversight: nothing is demanded of a
        # document no table covers, and `fieldscore` scores the whole base set
        # instead, marking the rate `unknown_type`.
        unknown={"fields": list(prompts.DEFAULT_FIELDS), "mandatory": [],
                 "items": [], "rules": []},
        # One line per validation rule, and whether it can run here at all --
        # four of the nine need data this process has not got.
        rule_notes=validate.RULE_NOTES,
        # Every key in the schema, in the order the prompt lists them, so the
        # page can draw a matrix in reading order without sorting it itself.
        keys=list(prompts._SCALAR_KEYS),
        item_keys=list(prompts.INCOME_ITEM_KEYS),
        items_key=INCOME_ITEMS_KEY)


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
    context = job_context(body.get("job"))

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
    form = _field_set(text, case_id, body.get("doc_types") or body.get("doc_type"),
                      status if status.get("text_available") else None)
    return jsonify({
        **form,
        # What the answer buys, which is the whole reason to ask: the steps an
        # agentic run would walk, the rules `validate` would then apply, and the
        # framing both prompts would carry.
        "steps": [s["id"] for s in steps_for_types(form["doc_types"],
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
    log_extract(result, body.get("job"), context)
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
            stream = extract_fields_stream(text, mode, case_id, steps,
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
                    log_extract(stop.value, job_id, context)
                    yield json.dumps({"event": "fields", **stop.value}) + "\n"
                    return
        except Exception as err:  # surface failures inside the stream body
            log_extract({"error": str(err), "mode": mode or extract_mode(),
                         "schema_first": schema_first and (mode or extract_mode()) in ("agentic", "hybrid")},
                        job_id, context)
            yield json.dumps({"event": "error", "error": str(err)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/api/extract/mode")
def extract_mode_get():
    mode = extract_mode()
    return jsonify(mode=mode, steps=(1 if mode == "single" else
                                    len(steps_for_types([], mode=mode))))


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
                                    len(steps_for_types([], mode=mode))))


@app.get("/api/extract/extras")
def extract_extras_get():
    return jsonify(extras=extract_extras())


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
    return jsonify(profile=ocr_profile(), profiles=[
        {"id": pid, "label": spec["label"], "note": spec["note"],
         "system": spec["system"], "reply": spec["reply"]}
        for pid, spec in OCR_PROFILES.items()
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
        name = set_ocr_profile((body.get("profile") or "").strip().lower())
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(profile=name)


@app.get("/api/ocr/loop-guard")
def loop_guard_get():
    """Whether a cycling read is cut short, and what turning that off costs."""
    return jsonify(loop_guard=loop_guard(), max_tokens=MAX_NEW_TOKENS)


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
    return jsonify(loop_guard=set_loop_guard(on), max_tokens=MAX_NEW_TOKENS)


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
    return jsonify(**table_agents())


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
    return jsonify(min_read_pct=round(read_floor() * 100, 1),
                   default_pct=round(MIN_READ_FOR_FIELDS * 100, 1))


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
                   default_pct=round(MIN_READ_FOR_FIELDS * 100, 1))


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
    with _jobs_lock:
        pages = _jobs.get(job_id)
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
        pages, page_total = load_pages(data)
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
    prepared = fit_pixels(trim_margins(page) if TRIM_MARGINS else page,
                          DETAIL_PRESETS[detail])
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
        "X-Preview-Max-Pages": str(MAX_PAGES),
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
    models = llama_status()["models"]
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
            standouts=runlog.standouts(), mode=extract_mode(),
            subject=body.get("subject") or randomtest.DEFAULT_SUBJECT,
            documents=body.get("documents"),
            top=body.get("top", randomtest.CONTEST_TOP),
            bottom=body.get("bottom", randomtest.CONTEST_BOTTOM),
            seed=body.get("seed"), history=runlog.case_counts(), **pools)
    return randomtest.plan(
        rounds=body.get("rounds"), seed=body.get("seed"), scope=scope,
        details=list(DETAIL_PRESETS), modes=["single", "agentic", "hybrid"],
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
            except SweepCancelled:
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
    set_ocr_profile(round_["profile"])
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
    log_extract(result, None, context)
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
    pages, detail, job_id, case = prepare_input(data, detail, case, source)

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
        log_run(partial, source, status="cancelled")
        raise SweepCancelled(f"{case_id} was stopped part-way through")

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
    log_run(payload, source)
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
                label = ", ".join(p for p in (active["document"], what) if p)
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

    def snapshot(self) -> dict:
        with self._lock:
            return {"stages": dict(self.stages), "done": self.done,
                    "failed": self.failed, "pages": self.pages,
                    "tokens": self.tokens}


def _vllm_metrics(url: str) -> dict:
    """GET `/metrics` from a vLLM server, parsed. Raises on any failure."""
    res = backends.http_get(f"{url}/metrics", {"kind": "vllm"},
                            timeout=STRESS_METRICS_TIMEOUT)
    if res.status_code != 200:
        raise ValueError(f"/metrics answered HTTP {res.status_code}")
    return stress.parse_prometheus(res.text)


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
    shape = (body.get("extract_mode") or extract_mode()).strip().lower()
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
        info = llama_status()
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

    return {
        "mode": mode,
        "concurrency": number("concurrency", 4, 1, stress.MAX_CONCURRENCY),
        "jobs": number("jobs", 20, 1, stress.MAX_JOBS),
        "warmup": number("warmup", 0, 0, stress.MAX_WARMUP),
        "max_seconds": number("max_seconds", 0, 0, 7 * 24 * 3600),
        "pack_pages": number("pack_pages", stress.DEFAULT_PACK_PAGES, 2,
                             stress.MAX_PACK_PAGES),
        "metrics_every": number("metrics_every", 5, 1, 600),
        "metrics": bool(body.get("metrics")),
        "no_cache": bool(body.get("no_cache")),
        # A long file reads its pages on workers left idle once the queue is
        # empty (`_StressSlots`). On unless the request says otherwise.
        "split_pages": body.get("split_pages", True) is not False,
        # Full pipeline only: a Mandatory field left empty is asked of the page
        # images with its Hybrid step (`_stress_fill_missing`). On unless the
        # request says otherwise; the page's toggle sends it.
        "fill_missing": body.get("fill_missing", True) is not False and mode == "full",
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
                   else extract_extras()),
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
    pages, _ = load_pages(data)
    budget = DETAIL_PRESETS[detail]
    prepared = [fit_pixels(trim_margins(page) if TRIM_MARGINS else page, budget)
                for page in pages]
    for index, page in enumerate(prepared):
        page._data_uri = image_data_uri(page)
        # Where a looped read climbs from (`_stress_retry_page`). Only the
        # address is kept, not the trimmed page: holding every page of the run
        # at full resolution for the few that loop would multiply its memory.
        page._stress_origin = {"case": case_id, "source": source,
                               "index": index, "detail": detail}
        tag_page(page, f"{case_id} p{index + 1}/{len(prepared)}", detail)
    return prepared, describe_source(name, data, origin)


def _one_page(data: bytes, index: int):
    """Page `index` of a file, rasterised alone where it is a PDF."""
    if _looks_like_pdf(data) and PDF_OK:
        zoom = PDF_DPI / 72.0
        with _PDF_LOCK, fitz.open(stream=data, filetype="pdf") as doc:
            if index >= doc.page_count:
                return None
            pix = doc[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pages, _ = load_pages(data)
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
    if not (LOOP_RETRY and origin):
        return None
    higher = next_detail(origin["detail"])
    if higher is None:
        return None
    data = (resolve_real(origin["case"]) if origin["source"] == "realdata"
            else case_bytes(origin["case"])[0])
    source = _one_page(data, origin["index"])
    if source is None:
        return None
    if TRIM_MARGINS:
        source = trim_margins(source)
    bigger = fit_pixels(source, DETAIL_PRESETS[higher])
    if bigger.size == page.size:
        return None
    bigger._data_uri = image_data_uri(bigger)
    bigger._stress_origin = dict(origin, detail=higher)
    if hasattr(page, "_dry_label"):
        tag_page(bigger, page._dry_label, higher)
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
    with _feeding_images([getattr(p, "_data_uri", None) or image_data_uri(p)
                          for p in pages]):
        return extract_fields(text, cfg["extract_mode"], case_id, pages=[text],
                              extras=cfg.get("extras"),
                              schema_first=cfg.get("schema_first", False),
                              each=lambda event: live.extract_event(job, event))


def _image_problem(result: dict) -> tuple:
    """`(kind, what, fields)` of an image-fed form worth extracting again, or
    `(None, "", [])`. An error other than a looped or capped reply is not one:
    it is the server refusing the request, and asking again with bigger images
    would only be refused again."""
    if result.get("error"):
        if result.get("looped"):
            return "looped", "the reply looped", []
        if result.get("truncated"):
            return "truncated", "the reply hit the token cap", []
        return None, "", []
    missing = _missing_required(result)
    if not missing:
        return None, "", []
    several = len(missing) > 1 or (result.get("documents_found") or 1) > 1
    listed = "; ".join((f"doc {n}: " if several else "") + ", ".join(keys)
                       for n, keys in missing)
    count = sum(len(keys) for _, keys in missing)
    return ("missing", f"{count} required field(s) came back empty: {listed}",
            [f"doc{n}.{k}" if several else k for n, keys in missing for k in keys])


def _image_cost(result: dict) -> tuple:
    """What makes one image-fed form worse than another: an error, then gaps."""
    return (bool(result.get("error")),
            sum(len(keys) for _, keys in _missing_required(result)))


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
    kind, what, fields = _image_problem(result)
    if kind is None:
        return result, reads, None
    extra = {"fields": fields} if fields else {}
    began = time.perf_counter()
    bigger = ([_stress_retry_page(p) for p in pages]
              if LOOP_RETRY and not cancel.is_set() else [])
    rec["retry_prepare_s"] = round(time.perf_counter() - began, 3)
    higher = next((b._stress_origin["detail"] for b in bigger if b is not None), None)
    if higher is None:
        why = ("the run was stopped" if cancel.is_set() else "LOOP_RETRY is off"
               if not LOOP_RETRY else "no higher Detail gives bigger pages")
        alert = live.flag(kind, job, who, f"{what} at {detail} -- no retry: {why}",
                          stage="extract", state="no retry", evidence=True, **extra)
        window(alert, result, _missing_required(result))
        return result, reads, alert
    alert = live.flag(kind, job, who, f"{what} at {detail} -- extracting again at {higher}",
                      stage="extract", state="retrying", evidence=True, **extra)
    window(alert, result, _missing_required(result))
    rec["image_retries"] = 1
    live.touch(job)
    retried = [b if b is not None else p for b, p in zip(bigger, pages)]
    second = _image_extract_once(retried, cfg, case_id, live, job)
    kind2, what2, _ = _image_problem(second)
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


# --------------------------------------------------------------------------
# Filling a missing field from the page (2026-10-27, at the user's request: *in
# full pipeline stress test if the field is missing we will send the image
# directly and ask for that specific field by using the hybrid prompt*). Where a
# full run's form comes back with a Mandatory field empty, each document's page
# images go to the extractor with only the Hybrid step(s) that own the empty
# keys -- the transcript is not sent, the pages are what is read -- and what
# comes back fills the blanks. Nothing already filled is overwritten. The form
# is then validated and scored again, so the job's field score and its Missing
# required count are of the form the pipeline ended with. One flag row says how
# it went: `retrying` -> `resolved` or `retry failed`, as image mode's does.
# --------------------------------------------------------------------------

def _fill_steps(codes, keys) -> tuple:
    """`(step ids, keys no step owns)`: the Hybrid steps that ask for `keys`.

    A blank table cell (`income_items[0].amount_paid`) is asked for by the step
    that owns the table. Off the same table the document was asked from, so a
    grouped step (`totals`, `parties`) is asked whole and only its empty keys are
    taken from the reply.
    """
    table = steps_for_types(codes, False, mode="hybrid")
    ids, orphans = [], []
    for key in keys:
        base = key.split("[", 1)[0]
        step = next((s for s in table if base in s["keys"]), None)
        if step is None:
            orphans.append(key)
        elif step["id"] not in ids:
            ids.append(step["id"])
    return ids, orphans


def _fill_form(doc: dict) -> dict:
    """The form `doc` was extracted with, rebuilt off its `classified` record:
    what a fill asks, validates and scores against."""
    classified = doc.get("classified") or {}
    return {"doc_types": list(classified.get("doc_types") or doc.get("doc_types") or []),
            "keys": list(classified.get("fields") or doc.get("fields_asked") or []),
            "mandatory": list(classified.get("mandatory") or []),
            "items": list(classified.get("items") or doc.get("items_asked") or []),
            "mandatory_items": list(classified.get("mandatory_items") or []),
            "extras": False, "schema_first": False}


def _fill_ask(pages, form: dict, ids) -> dict:
    """The Hybrid steps `ids`, asked of `pages` as images and nothing else.

    `_extract_agentic` directly rather than `extract_fields`: the document is
    already split and classified, and a fill must not classify the placeholder
    text again or segment it.
    """
    status = backends.extract_status()
    if not status["text_available"]:
        return {"error": status["text_reason"]}
    text = prompts.IMAGE_TRANSCRIPT.format(pages=len(pages))
    with _feeding_images([getattr(p, "_data_uri", None) or image_data_uri(p)
                          for p in pages]):
        return _drain(_extract_agentic(text, status, form, only=list(ids), mode="hybrid"))


def _gaps(doc: dict) -> int:
    return sum(len(keys) for _, keys in _missing_required(doc))


def _revalidate(doc: dict, form: dict, text: str) -> None:
    """`_validate_fields`'s normalisation and validation of `doc`'s fields, again."""
    fields = doc.get("fields") or {}
    doc["normalized"] = normalise.normalize_fields(fields)
    try:
        doc["validation"] = validate.validate(
            fields, keys=form["keys"], doc_types=form["doc_types"],
            mandatory=form["mandatory"], transcript=text, items=form["items"])
    except Exception as err:  # noqa: BLE001 - never worth the job
        doc["validation"] = {"error": f"validation failed: {err}"}


def _fill_document(doc: dict, keys, answer: dict, form: dict, text: str) -> list:
    """Put `answer`'s values into `doc`'s empty `keys`; the keys it filled.

    A scalar fills only a blank. The table is a whole list, so the new one is
    taken only where it leaves fewer demanded cells empty than the old.
    """
    got = answer.get("fields") or {}
    fields = dict(doc.get("fields") or {})
    filled = []
    for key in keys:
        if "[" in key or key == prompts.INCOME_ITEMS_KEY:
            continue
        if validate._blank(fields.get(key)) and not validate._blank(got.get(key)):
            fields[key] = got[key]
            filled.append(key)
    doc["fields"] = fields
    _revalidate(doc, form, text)
    table = [k for k in keys if k.split("[", 1)[0] == prompts.INCOME_ITEMS_KEY]
    if table and got.get(prompts.INCOME_ITEMS_KEY):
        trial = dict(doc, fields={**fields, prompts.INCOME_ITEMS_KEY: got[prompts.INCOME_ITEMS_KEY]})
        _revalidate(trial, form, text)
        if _gaps(trial) < _gaps(doc):
            doc.update(fields=trial["fields"], normalized=trial["normalized"],
                       validation=trial["validation"])
            filled += [k for k in table if k not in
                       {key for _, ks in _missing_required(doc) for key in ks}]
    return filled


def _stress_fill_missing(result: dict, images, texts, case_id, rec: dict,
                         live: _StressLive, job: int, who: str, window, cancel,
                         why: str = ""):
    """A full run's Mandatory fields left empty, asked of the page images.

    Returns the flag this raised and kept up to date, or None where nothing was
    missing (or the extraction failed, which its own flag says). The caller
    raises no second flag. `result` is filled in place, revalidated and
    rescored; its tokens and server time grow by the fill's.
    """
    if result.get("error") or cancel.is_set():
        return None
    missing = _missing_required(result)
    if not missing:
        return None
    _, what, flagged = _image_problem(result)
    documents = [d for d in (result.get("documents") or []) if isinstance(d, dict)]
    by_number = {(d.get("document") or 1): d for d in documents} if documents else {1: result}
    several = len(by_number) > 1
    plan = []
    orphans = []
    for number, keys in missing:
        doc = by_number.get(number)
        if doc is None:
            continue
        form = _fill_form(doc)
        ids, lost = _fill_steps(form["doc_types"], keys)
        orphans += [f"doc{number}.{k}" if several else k for k in lost]
        if ids:
            plan.append((number, doc, form, [k for k in keys if k not in lost], ids))
    extra = {"fields": flagged} if flagged else {}
    if not plan:
        alert = live.flag("missing", job, who,
                          f"{what}{why} -- no retry: no Hybrid step asks for "
                          + ", ".join(orphans), stage="extract", state="no retry",
                          evidence=True, **extra)
        window(alert, result, missing)
        return alert
    asked = ", ".join(dict.fromkeys(i for *_, ids in plan for i in ids))
    alert = live.flag("missing", job, who,
                      f"{what}{why} -- asking the page images with the Hybrid "
                      f"prompt ({asked})", stage="extract", state="retrying",
                      evidence=True, **extra)
    window(alert, result, missing)
    rec["fill_jobs"] = 1
    rec["fill_asked"] = sum(len(keys) for _, _, _, keys, _ in plan)
    live.touch(job)
    before = _gaps(result) if not documents else sum(_gaps(d) for d in documents)
    filled, errors, tokens = [], [], 0
    previous = servertime.open_tally()
    try:
        for number, doc, form, keys, ids in plan:
            if cancel.is_set():
                break
            numbers = [p for p in (doc.get("pages") or []) if 1 <= p <= len(images)]
            numbers = numbers or list(range(1, len(images) + 1))
            # This document's own transcript, which validation checks against.
            own = [texts[p - 1] for p in numbers if p <= len(texts)]
            text = join_page_texts(own) if len(own) > 1 else "".join(own)
            try:
                answer = _fill_ask([images[p - 1] for p in numbers], form, ids)
            except Exception as err:            # noqa: BLE001
                answer = {"error": f"{type(err).__name__}: {err}"}
            tokens += answer.get("tokens") or 0
            if answer.get("error"):
                errors.append(str(answer["error"])[:160])
                continue
            got = _fill_document(doc, keys, answer, form, text)
            filled += [f"doc{number}.{k}" if several else k for k in got]
            if got:
                doc["image_filled"] = list(doc.get("image_filled") or []) + got
                if case_id:
                    _score_fields(doc, case_id, form, doc.get("pages") if several else None,
                                  text)
                live.touch(job)
    finally:
        served = servertime.close_tally(previous)
    if documents and any(d.get("image_filled") for d in documents):
        scored = [d for d in documents if d.get("field_score")]
        if scored:
            result["field_score"] = _merge_field_scores(scored, len(documents))
    result["tokens"] = (result.get("tokens") or 0) + tokens
    if served:
        result["server_time"] = dict(result.get("server_time") or {})
        for key in ("in_tokens", "out_tokens"):
            values = [v for v in (result["server_time"].get(key), served.get(key))
                      if v is not None]
            result["server_time"][key] = sum(values) if values else None
    after = _gaps(result) if not documents else sum(_gaps(d) for d in documents)
    rec["fill_filled"] = before - after
    note = (" · failed: " + "; ".join(errors)) if errors else ""
    if orphans:
        note += " · no Hybrid step asks for " + ", ".join(orphans)
    if not after:
        live.update_flag(alert, f"{what}{why} -- resolved: the page images filled "
                                + ", ".join(filled) + note, state="resolved")
    else:
        remaining = _missing_required(result)
        listed = "; ".join((f"doc {n}: " if several else "") + ", ".join(keys)
                           for n, keys in remaining)
        live.update_flag(alert, f"{what}{why}; asked the page images, "
                                f"{len(filled) or 'none'} filled"
                                + (f" ({', '.join(filled)})" if filled else "")
                                + f", still empty: {listed}{note}",
                         state="retry failed")
    return alert


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
    inner = MultiCancel()                   # a failed page hangs up its siblings
    hangup = _Hangup(inner)                 # held here: `attach` keeps it weakly
    cancel.attach(hangup)
    cut = "the reading model repeated itself and the loop guard cut the page off"

    def read_one(number, page, stats):
        """One page, with its loop and DRY retries. Returns `(text, tokens)`.

        A page that loops has ONE alert however many times it is read: raised
        `retrying` (or `no retry`), and marked `resolved` or `retry failed` by
        the read that settles it. DRY firing is one alert per page the same way,
        its message growing by a Detail each attempt it fired on.

        **A page DRY fired on is read again one Detail up too** (2026-10-27, at
        the user's request; `STRESS_DRY_RETRY`), even where it did not loop: a
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
                text = read_page(page, stats, cancel=inner)
            except Exception as err:
                if cancel.is_set() or inner.is_set():
                    raise SweepCancelled("stopped")
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
                raise SweepCancelled("stopped")
            live.touch(job)
            page_tokens += stats.get("new_tokens") or 0
            dry = stats.get("dry") or {}
            fired, looped = bool(dry.get("fired")), bool(stats.get("looped"))
            seen.append({"detail": was, "text": text, "page": page,
                         "looped": looped,
                         "truncated": bool(stats.get("truncated")),
                         "dry": dry if fired else None})
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
                raise SweepCancelled("stopped")
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
                                   return_when=FIRST_COMPLETED)
                for future in finished:
                    slot, number, stats = running.pop(future)
                    if slot == "own":
                        own_busy = False
                    else:
                        slots.release()
                    try:
                        results[number - 1] = (*future.result(), stats)
                    except SweepCancelled:
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
            raise SweepCancelled("stopped")

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
    steps = _segment_stream(pack["text"], pack["pages"], status)
    while True:
        if cancel.is_set():
            steps.close()
            raise SweepCancelled("stopped")
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
            raise SweepCancelled("stopped")
        codes, how = resolve_doc_types(seg.get("text") or "", status=status,
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


def _agentic_chats(result):
    """Numeric request records across every document, without prompt/raw text."""
    for document in result.get("documents") or [result]:
        for step in document.get("steps") or []:
            for chat in step.get("chats") or []:
                yield {**chat, "step": step["id"], "title": step.get("title", ""),
                       "keys": step.get("keys") or [], "document": document.get("document", 1)}


def _missing_required(result: dict) -> list:
    """Each document's Mandatory fields that came back empty: `[(document, [keys])]`.

    **The extraction half of a stress run's flags** (2026-10-26, at the user's
    request: *OCR stays the same, but the extraction will flag the missing
    required field*). A looped read still goes on to classify and extract, and
    what it costs shows up here, as the fields the form demanded and did not get,
    rather than as a second loop flag.

    The scalars are read off `fields` against the form's own `mandatory` list,
    not off `validate`'s rows: three Mandatory keys (`amount_incl_vat`,
    `vat_total`, `total_wht_amount`) have no rule, so no `absent` row either. The
    table's cells come from `validate`, which already reports a demanded blank
    cell per row, and a table that never came back as the table key itself. A
    document of a type no requirement covers demands nothing and is never flagged.
    """
    out = []
    documents = [d for d in (result.get("documents") or []) if isinstance(d, dict)]
    for doc in documents or [result]:
        classified = doc.get("classified") or {}
        fields = doc.get("fields") if isinstance(doc.get("fields"), dict) else {}
        missing = [key for key in classified.get("mandatory") or []
                   if validate._blank(fields.get(key))]
        if classified.get("mandatory_items"):
            checks = (doc.get("validation") or {}).get("checks")
            if checks is None:                  # nothing extracted, so no table
                missing.append(prompts.INCOME_ITEMS_KEY)
            else:
                missing.extend(row["field"] for row in checks
                               if row.get("mandatory") and row.get("state") == validate.ABSENT
                               and row["field"].startswith(prompts.INCOME_ITEMS_KEY))
        missing = list(dict.fromkeys(missing))
        if missing:
            out.append((doc.get("document") or 1, missing))
    return out


def _field_value(value) -> str:
    """One extracted value as the Fields evidence shows it: text as it is, a
    table or an object as compact JSON, nothing as ''."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _fields_evidence(result: dict, missing: list) -> list:
    """The form of each document, for the Compare view of a missing-required flag.

    **Built 2026-10-26 at the user's request** -- *allow display window on field
    too*: a loop or DRY row opens the page and its text, and a missing-required
    row opens the same window with the fields beside them. One entry per
    document: every key it was asked, its value, whether the requirement demands
    it and whether it is one of the ones flagged.
    """
    flagged = {n: set(keys) for n, keys in missing}
    documents = [d for d in (result.get("documents") or []) if isinstance(d, dict)]
    out = []
    for doc in documents or [result]:
        number = doc.get("document") or 1
        classified = doc.get("classified") or {}
        fields = doc.get("fields") if isinstance(doc.get("fields"), dict) else {}
        required = set(classified.get("mandatory") or [])
        gaps = flagged.get(number, set())
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
        out.append({"document": number, "pages": list(doc.get("pages") or []),
                    "doc_types": list(doc.get("doc_types") or []),
                    "error": doc.get("error") or "", "rows": rows})
    return out


def _stress_job(spec: dict, cfg: dict, prepared: dict, cancel,
                live: _StressLive) -> dict:
    """One job of the queue, start to finish. Returns its record; never raises."""
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
    if live.slots is not None:
        live.slots.hold()
    live.move("queued", stage)
    began = time.perf_counter()
    # Who the alerts name: the job's place in the queue (warm-up included) and
    # its document -- for classification, the cases glued into its file.
    job = spec["index"] + 1
    if mode == "classify":
        glued = prepared["packs"][spec["index"]].get("cases") or []
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
    try:
        if mode == "classify":
            _stress_classify(prepared["packs"][spec["index"]], rec, live, cancel, job)
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
                text = join_page_texts(texts) if len(texts) > 1 else texts[0]
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
                    live.move("ocr", "extract")
                    stage = "extract"
                    live.touch(job, stage)
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
                        retry_alert = _stress_fill_missing(
                            result, prepared["pages"][case_id], texts,
                            None if real else case_id, rec, live, job, who,
                            fields_window, cancel, why)
                rec["extract_s"] = round(time.perf_counter() - started, 3)
                if spec.get("detail"):
                    # Hold the existing result; format and send it after timing ends.
                    rec["_extraction_result"] = result
                rec["extract_tokens"] = result.get("tokens") or 0
                # Every pass-2 request of this file: classify, segment, steps or
                # the single request, table agents (servertime's tally).
                served = result.get("server_time") or {}
                rec["extract_in"] = served.get("in_tokens")
                rec["extract_out"] = served.get("out_tokens")
                chats = list(_agentic_chats(result))
                if chats:
                    rec["agentic_chats"] = [{**chat, "job": job, "case": case_id} for chat in chats]
                if log:
                    logged = True
                    if row is not None:
                        row["extracted"] = result
                        log_run(row, prepared["sources"][case_id])
                    else:
                        log_extract(result, None, {
                            "case": case_id,
                            "source": prepared["sources"][case_id]}, extra=tag)
                live.touch(job)
                missing = _missing_required(result)
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
                    apply_read_floor(payload)
                summary = randomtest.summarise_round(payload)
                rec.update(field_rate=(summary["field"] / 100
                                       if summary["field"] is not None else None),
                           field_correct=summary["correct"],
                           field_partial=summary["partial"],
                           field_expected=summary["expected"],
                           unscored=bool(summary["unscored"]),
                           partial_reply=summary["partial_reply"])
            if row is not None and not logged:      # read only
                logged = True
                log_run(row, prepared["sources"][case_id])
            rec["status"] = ("looped" if looped else
                             "truncated" if truncated else "ok")
    except SweepCancelled:
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
        _progress_swap(previous_hook)
        if live.slots is not None:
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
    live.finish(rec["status"] == "error")
    return rec


def _stress_log_error(mode, spec, prepared, all_stats, read_began, cfg, tag,
                      err) -> None:
    """The run-log row of a saved stress job that raised. Never raises."""
    try:
        case_id = spec["case"]
        if mode in ("ocr", "full"):
            row = {**summarise(all_stats, cfg["detail"], read_began), **tag}
            log_run(row, prepared["sources"][case_id], error=err)
        else:
            log_extract({"error": f"{type(err).__name__}: {err}",
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
    started = time.perf_counter()
    deadline = started + cfg["max_seconds"] if cfg["max_seconds"] and phase == "run" else None
    for spec in specs:
        spec["submitted"], spec["deadline"] = started, deadline
        # Warm-up jobs are thrown away from the report, and from the log too.
        spec["log"] = bool(cfg.get("save_log")) and phase == "run"
        spec["detail"] = bool(cfg.get("save_details")) and phase == "run"
    job = workspace.bind(_stress_job)
    futures = [pool.submit(job, spec, cfg, prepared, cancel, live)
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


def _stress_detail_identity(cfg):
    """Hash the code and selected truth files outside the benchmark clock."""
    index = scoring.cases_index() if cfg.get("source", "benchmark") == "benchmark" else {}
    benchmarks = {case: {"transcript": index[case]["ground_truth"],
                         "fields": fieldscore.truth_path(case),
                         "pages": index[case].get("pages"),
                         "doc_types": index[case].get("doc_types") or []}
                  for case in cfg["cases"] if case in index}
    return extractionlog.provenance(prompts.__file__, __file__, {
        "AGENTIC_RETRIES": AGENTIC_RETRIES, "EXTRACT_SCHEMA": EXTRACT_SCHEMA,
        "AGENTIC_PARALLEL": AGENTIC_PARALLEL, "AGENTIC_PARALLEL_MAX": AGENTIC_PARALLEL_MAX,
        "SEGMENT_MAX_PAGES": SEGMENT_MAX_PAGES,
        "NORMALIZATION_VERSION": normalise.NORMALIZATION_VERSION}, extra_files={
            "normalise": normalise.__file__, "fieldscore": fieldscore.__file__,
            "segment": segment.__file__, "backends": backends.__file__,
            "grounding": grounding.__file__, "stress": stress.__file__,
            "manifest": config.SOLUTION_DIR / "manifest.json"}, benchmarks=benchmarks)


def _stress_run(cfg: dict, cancel):
    """The whole run: plan, prepare, warm up, measure, report."""
    rng, seed = stress.seeded_rng(cfg["seed"])
    mode = cfg["mode"]
    cfg["run_id"] = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-{seed}"
    total = cfg["warmup"] + cfg["jobs"]
    detail_identity = _stress_detail_identity(cfg) if cfg.get("save_details") else None
    reader = llama_status() if mode in ("ocr", "full") else None
    extractor = backends.extract_status() if mode != "ocr" else None
    info = reader or extractor
    # Both servers are watched whatever the mode (the user's request,
    # 2026-10-19): a run that loads one shows the other beside it.
    targets = _stress_targets(reader or llama_status(),
                              extractor or backends.extract_status())         if cfg["metrics"] else []

    specs = [{"index": i} for i in range(total)]
    if mode != "classify":
        # realData files are never in the run log's case counts, so "balanced"
        # balances over this run's own draws.
        history = (runlog.case_counts() if cfg["strategy"] == "balanced"
                   and cfg.get("source") != "realdata" else {})
        if cfg["strategy"] == "sequential":
            # Measured jobs start at the first selected file even after warm-up.
            drawn = [case_id for count in (cfg["warmup"], cfg["jobs"])
                     for case_id in stress.draw_cases(
                         count, cfg["cases"], cfg["strategy"], rng, history, cfg["lock"])]
        else:
            drawn = stress.draw_cases(
                total, cfg["cases"], cfg["strategy"], rng, history, cfg["lock"])
        for spec, case_id in zip(specs, drawn):
            spec["case"] = case_id

    yield json.dumps({
        "event": "plan", "mode": mode, "seed": seed,
        "concurrency": cfg["concurrency"], "jobs": cfg["jobs"],
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
        "split_pages": bool(cfg.get("split_pages")) and cfg["mode"] in ("ocr", "full"),
        "fill_missing": bool(cfg.get("fill_missing")) and cfg["mode"] == "full",
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
        for i in range(total):
            prepared["packs"].append(stress.build_pack(
                cfg["cases"], truths, docs_of, cfg["pack_pages"], rng))
            yield json.dumps({"event": "preparing", "done": i + 1,
                              "total": total}) + "\n"
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
    batcher = _batcher if cfg.get("vllm_batch") else None
    if batcher is not None:
        batcher.reset_counts()        # the warm-up's batches are not the run's
    if not cancel.is_set():
        records, wall, timeline, last, draining = yield from _stress_phase(
            specs[cfg["warmup"]:], cfg, prepared, cancel, "run", targets, bases)

    report = stress.report(records, wall, mode, cfg["concurrency"])
    report["timeline"] = stress.thin_timeline(timeline)
    report["timeline_columns"] = ["t", *stress.STAGES, "done"]
    report["prepare_seconds"] = prepare_seconds
    report["outliers"] = stress.without_longest(records, wall, mode, cfg["concurrency"],
                                                STRESS_OUTLIER_MAX)
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
        cancel = begin_sweep(MultiCancel())
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
    global _batcher
    try:
        if no_cache:
            backends.bypass_prefix_cache()
        if cfg.get("vllm_batch"):
            # One batcher for every server the run uses; the group key
            # keeps each URL's requests apart.
            _batcher = vllmbatch.Batcher(cfg["concurrency"], _send_batch)
        configuration = {**cfg, "ocr_profile": ocr_profile(),
                         "sampling": ocr_sampling_state(), "loop_guard": loop_guard(),
                         "read_floor": read_floor(), "table_agents": table_agents(),
                         "context": backends.num_ctx(),
                         "workspace": workspace.current_id(),
                         "reader": llama_status() if cfg.get("mode") in ("ocr", "full") else None,
                         "extractor": (backends.extract_status() if cfg.get("mode")
                                       in ("extract", "classify", "full", "image") else None)}
        yield from _history_stream(_stress_run(cfg, cancel), "stress", configuration)
    except Exception as err:                        # noqa: BLE001
        yield json.dumps({"event": "error",
                          "error": f"{type(err).__name__}: {err}"[:300]}) + "\n"
    finally:
        if _batcher is not None:
            _batcher.close()
            _batcher = None
        if no_cache:
            backends.restore_prefix_cache()
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

    def saved_monitor():
        samples, total = monitor_sampler.stop()
        return {"monitor_samples": samples, "monitor_samples_total": total, "alerts": alerts}

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
    return jsonify(case=item["case"], page=item["page"], truth_case=item["truth_case"],
                   fields=item.get("fields"),
                   attempts=[{"detail": a["detail"], "text": a["text"],
                              "looped": a["looped"], "truncated": a["truncated"],
                              "dry": a["dry"], "page": a.get("number"),
                              "note": a.get("note"),
                              "image": a.get("page") is not None}
                             for a in item["attempts"]])


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
                loop_abort_on=bool(EXTRACT_LOOP_ABORT and loop_guard())), 200



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
        log_run(summarise(all_stats, detail, started, job_id, run), source, error=err)
        return jsonify(error=str(err)), 400
    except requests.RequestException as err:
        log_run(summarise(all_stats, detail, started, job_id, run), source, error=err)
        return jsonify(error=f"model server connection failed: {err}"), 502

    text = join_page_texts(page_texts)

    payload = summarise(all_stats, detail, started, job_id, run)
    payload["truth"] = evaluate_if_known(case, text)
    if EXTRACT and request.form.get("extract", "1") != "0" and text.strip():
        payload["extracted"] = extract_fields(
            text, case_id=case["id"] if case else None, pages=page_texts)
    log_run(payload, source)
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
            parallel = _page_parallelism(len(pages))
            if parallel > 1:
                # Every page is in flight at once, so there is no one page whose
                # tokens could stream: interleaved, they would splice pages into
                # each other on screen. Each page arrives whole instead, in page
                # order, as the same page/token/page_done events a sequential
                # read sends -- the page needs nothing new to assemble it.
                yield json.dumps({"event": "pages_parallel", "total": len(pages),
                                  "parallel": parallel, "job": job_id}) + "\n"
                with closing(read_pages_at_once(pages, parallel, run=run)) as reads:
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
                        for batch in batch_tokens(stream_page(page, stats),
                                                  STREAM_FLUSH_MS / 1000):
                            parts.extend(batch)
                            yield json.dumps({"event": "token", "text": "".join(batch),
                                              "n": len(batch)}) + "\n"
                        # The same climb `read_page` makes, said out loud: the
                        # page drops the looped text and streams the retry in.
                        retry = loop_retry_page(page, stats)
                        if retry is None:
                            break
                        record_retry(stats, page, retry)
                        page = retry
                        yield json.dumps({"event": "loop_retry", "page": index,
                                          "detail": stats["detail"],
                                          "resolution": f"{page.width}x{page.height}"}) + "\n"
                    collected.append(finish_page("".join(parts), stats))
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
                stream = extract_fields_stream(
                    text, case_id=case["id"] if case else None, pages=collected)
                while True:
                    try:
                        yield json.dumps(next(stream)) + "\n"
                    except StopIteration as stop:
                        result = stop.value
                        break
                summary["extracted"] = result
                # Applied before the event rather than left to `log_run` below:
                # this is the one path that hands the fields to the browser
                # before the row is written, and a page that had already drawn
                # the score would go on showing it. Same call, same rule --
                # `apply_read_floor` is idempotent for exactly this.
                apply_read_floor(summary)
                yield json.dumps({"event": "fields", **result}) + "\n"

            # Logged once everything has run, so the row carries the accuracy and
            # the extraction counts rather than just the OCR timings.
            log_run(summary, source)
            yield json.dumps({"event": "logged"}) + "\n"
        except GeneratorExit:
            # The browser hung up -- Stop, or a closed tab. Still worth a row.
            log_run(summary or summarise(all_stats, detail, started, job_id, run),
                    source, status="cancelled")
            raise
        except Exception as err:  # surface failures inside the stream body
            log_run(summary or summarise(all_stats, detail, started, job_id, run),
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
            "ocr_profile": ocr_profile(),
            "extract_mode": "agentic" if AGENTIC_EXTRACT else "single",
            "dry": DRY_MULTIPLIER,
            "max_new_tokens": MAX_NEW_TOKENS,
            "extract_max_tokens": EXTRACT_MAX_TOKENS,
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
                     "min_read_pct": round(MIN_READ_FOR_FIELDS * 100, 1)},
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
    return jsonify(error=f"File is larger than {MAX_UPLOAD_MB} MB."), 413


# Encoding-safe printing lives in config so the packaging script gets it too --
# both print installation paths, and both ran on a machine whose console could
# not encode them.
say = config.say


# A key in a JS object literal: at the start of a line, or after a comma. Both,
# because these maps put two keys on one line where the values are short -- an
# earlier version anchored to the line start only and silently saw 7 of the 11.
_JS_MAP_KEYS = re.compile(r"(?:^\s+|,\s*)(\w+)\s*:", re.M)


def template_field_keys():
    """The keys `templates/index.html` knows how to draw, per map.

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
        text = (config.BASE_DIR / "templates" / "index.html").read_text("utf-8")
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
        return ["could not read the field maps out of templates/index.html"]
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

    status = llama_status()
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
get_user_service = userweb.register(app, sys.modules[__name__])


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
