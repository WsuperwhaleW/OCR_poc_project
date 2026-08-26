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
import io
import json
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict

import requests
from flask import Flask, Response, jsonify, render_template, request

import backends
import config
import fieldscore
import grounding
import jobs
import machine
import normalise
import randomtest
import runlog
import scoring
import verify

# Prompt text and tunables live outside this file so either can be read, edited
# and diffed without the request-assembly code around them. `prompts.py` holds
# the prompts; `settings.py` holds every value the app runs with, and the
# comment beside each one says what was measured to arrive at it.
from prompts import (
    EXTRACT_JSON_SCHEMA,
    EXTRACT_PROMPT,
    EXTRACT_REMINDER,
    EXTRACT_STEP_PREFIX,
    EXTRACT_STEP_RETRY,
    EXTRACT_STEP_TASK,
    EXTRACT_STEPS,
    LINE_ITEM_SCHEMA,
    OTHER_FIELDS_SCHEMA,
    PROMPT,
    OCR_PROFILES,
    DEFAULT_OCR_PROFILE,
)
from settings import (
    OCR_PROFILE,
    ACCEPTED_SUFFIXES,
    AGENTIC_EXTRACT,
    AGENTIC_RETRIES,
    DEFAULT_DETAIL,
    DETAIL_PRESETS,
    DRY_MULTIPLIER,
    resolve_detail,
    EXTRACT,
    EXTRACT_LOOP_MIN_REPEATS,
    EXTRACT_LOOP_TAIL_CHARS,
    EXTRACT_MAX_TOKENS,
    EXTRACT_REPEAT_THRESHOLD,
    EXTRACT_SCHEMA,
    GEN_CONNECT_TIMEOUT,
    GEN_TIMEOUT,
    LOOP_CHECK_EVERY,
    LOOP_COUNTER_MAX_UNIT,
    LOOP_COUNTER_MIN_LINE,
    LOOP_COUNTER_MIN_REPEATS,
    LOOP_DEAD_MAX_LETTERS,
    LOOP_DEAD_TAIL_CHARS,
    LOOP_GUARD,
    LOOP_MIN_REPEATS,
    LOOP_TAIL_CHARS,
    MAX_JOBS,
    MAX_NEW_TOKENS,
    MAX_PAGES,
    MAX_UPLOAD_MB,
    MIN_READ_FOR_FIELDS,
    PDF_DPI,
    PROMPT_FIRST,
    PROMPT_FIRST_OLLAMA,
    SUMMARY_RUNS,
    TRIM_MARGINS,
    TRIM_PAD,
    TRIM_TOLERANCE,
    WORKERS_OVERRIDE,
    sampler_extras,
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


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

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


def register_job(pages, context=None):
    """Cache PNG-encoded page images and return a job id."""
    job_id = uuid.uuid4().hex[:12]
    encoded = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        encoded.append(buf.getvalue())
    with _jobs_lock:
        _jobs[job_id] = encoded
        _job_meta[job_id] = context or {}
        while len(_jobs) > MAX_JOBS:
            evicted, _ = _jobs.popitem(last=False)
            _job_meta.pop(evicted, None)
    return job_id


def job_context(job_id: str) -> dict:
    """What the prepared pages under this id were, or {} once they are gone.

    An id whose images have been evicted answers empty rather than raising: a
    re-extraction of a transcript still on screen is worth logging even when the
    page images behind it have fallen out of the cache.
    """
    with _jobs_lock:
        return dict(_job_meta.get(job_id or "", {}))


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
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stream_page(image: Image.Image, stats: dict | None = None,
                profile: str = None):
    """Stream tokens for one page from llama-server.

    `profile` names the pass-1 shape (`prompts.OCR_PROFILES`); omitted, the one
    the process is currently set to. Resolved once here and reported on `stats`,
    so a page reads and logs under the profile it actually ran with even if the
    setting is flipped mid-batch -- the same rule `extract_mode` follows.
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
    image_part = {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
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
        # Fully deterministic decoding. temperature 0 should already force greedy,
        # but llama-server's defaults (top_k 40, top_p 0.95, min_p 0.05) are pinned
        # explicitly so transcription can never drift between runs.
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        **sampler_extras(),
        "stream": True,
        # Ollama does not send llama.cpp's `timings`; asking for usage in the
        # final chunk gives an authoritative token count on both servers.
        "stream_options": {"include_usage": True},
        **backends.request_extras(status),
    }

    started = time.perf_counter()
    first_at = None
    timings, usage, pieces = {}, {}, 0
    collected, looped = [], False
    with requests.post(
        chat_url(),
        json=payload,
        stream=True,
        # No read timeout: this is the streaming path, where the gap between
        # tokens is the only thing a timeout could measure and a slow first
        # token is normal. The client disconnecting is what ends it.
        timeout=(GEN_CONNECT_TIMEOUT, None),
    ) as res:
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
                break
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
    if not guard and looks_repetitive("".join(collected)):
        looped = True

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
            new_tokens=new_tokens,
            truncated=new_tokens >= MAX_NEW_TOKENS,
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
        )


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


def _extract_single(text: str, status: dict) -> dict:
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
    result = _extract_once(text, status, None)
    if "fields" in result or not EXTRACT_SCHEMA:
        return result
    retry = _extract_once(text, status, EXTRACT_JSON_SCHEMA)
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


def _extract_once(text: str, status: dict, schema) -> dict:
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
    message = EXTRACT_PROMPT + text + EXTRACT_REMINDER
    url, payload = backends.structured_request(
        backends.system_prefix(status)
        + [{"role": "user", "content": message}],
        schema, EXTRACT_MAX_TOKENS, status)

    started = time.perf_counter()
    try:
        res = requests.post(url, json=payload, timeout=GEN_TIMEOUT)
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
        "grounding": grounding.check(fields, text),
        # Coverage by delivery tier, the same counts the run log records. Sent so
        # the page can say it in one line instead of leaving you to count filled
        # rows; it is deliberately NOT part of the grounding dict, because how
        # many fields came back is a different question from whether they are real.
        "tiers": grounding.tier_counts(fields),
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
            key: (LINE_ITEM_SCHEMA if key == "line_items"
                  else OTHER_FIELDS_SCHEMA if key == "other_fields"
                  else {"type": "string"})
            for key in step["keys"]
        },
        "required": list(step["keys"]),
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
    res = requests.post(url, json=payload, timeout=GEN_TIMEOUT)
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


def _step_question(step: dict) -> str:
    """The part that differs: this step's title, skeleton and rules."""
    return EXTRACT_STEP_TASK.format(title=step["title"],
                                    skeleton=step["skeleton"],
                                    rules=step["rules"])


def _step_message(text: str, step: dict) -> str:
    """The message for one step: prefix, transcript, then this step's question.

    Assembled in that order on purpose. The transcript first is what makes the
    prefix shared across every step and therefore cacheable; the instructions last
    is what keeps an OCR fine-tune from answering with its own transcript envelope,
    which is the same thing EXTRACT_REMINDER buys the single-shot prompt.
    """
    return _step_prefix(text) + _step_question(step)


def _step_values(step: dict, raw: str):
    """The keys this step owns, pulled out of its reply. Raises on unusable JSON.

    Only the step's own keys are taken. A step that answers with keys belonging to
    another step has guessed at fields it was not shown the rules for, and taking
    those would put back exactly the cross-contamination the split is here to stop.
    """
    parsed = json.loads(_first_json_object(strip_fence(raw)) or raw)
    if not isinstance(parsed, dict):
        raise ValueError("reply was JSON but not an object")
    values = {}
    for key in step["keys"]:
        if key not in parsed:
            continue
        value = parsed[key]
        if key in ("line_items", "other_fields"):
            values[key] = [v for v in value if isinstance(v, dict)] \
                if isinstance(value, list) else []
        elif isinstance(value, (str, int, float)):
            values[key] = value
    return values


def _ask_step(content: str, step: dict, status: dict, collect: list = None):
    """Ask one step and parse its reply. Returns (values, replies, truncated, tokens).

    Asked plain first, exactly as the baselines were measured. A reply that will
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
    """
    replies, tokens = [], 0
    collect = replies if collect is None else collect

    def keep(text):
        replies.append(text)
        if collect is not replies:
            collect.append(text)

    raw, truncated, used = _chat(content, step["max_tokens"], status)
    keep(raw)
    tokens += used
    try:
        return _step_values(step, raw), replies, truncated, tokens
    except Exception:
        if not EXTRACT_SCHEMA:
            raise
    raw, truncated, used = _chat(content, step["max_tokens"], status,
                                 _step_schema(step))
    keep(raw)
    tokens += used
    return _step_values(step, raw), replies, truncated, tokens


def _ungrounded_in(values: dict, source) -> list:
    """The step's scalar answers that are not in the transcript, as label = value.

    Blank and nil answers are skipped for the same reason `grounding.check` skips
    them: neither is a claim about the page. Line items are skipped too -- a table
    row is checked cell by cell at the end, and quoting a whole row back at the
    model is a worse question than not asking one.
    """
    bad = []
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


def _steps_for(wanted):
    """The step table, or the named subset of it, in the table's own order.

    A measurement tool and nothing else: it is what lets one step's prompt be
    swept over five documents and four models without paying for the six steps
    the change did not touch. Unknown ids are refused rather than skipped -- a
    sweep that quietly asked for no steps would report a schema-wide failure.

    The order is always `EXTRACT_STEPS`', never the caller's, so a restricted run
    is the full run with steps removed rather than a differently ordered one.
    """
    if not wanted:
        return EXTRACT_STEPS
    ids = [s.strip() for s in wanted if str(s).strip()]
    known = {s["id"] for s in EXTRACT_STEPS}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown extraction step(s): {', '.join(unknown)} -- "
                         f"have {', '.join(sorted(known))}")
    if not ids:
        raise ValueError("no extraction steps named")
    return tuple(s for s in EXTRACT_STEPS if s["id"] in set(ids))


def _extract_agentic(text: str, status: dict, only=None):
    """Walk the step table, yielding progress, and return the merged result.

    A generator so that both the browser and the run stream can show which step is
    running without a callback that cannot yield: iterate it for the events, and
    take the finished result from StopIteration.value (`_drain` does this).

    `only` restricts the walk to the named steps. The result then says so in
    `steps_only`, and everything downstream that would otherwise report the keys
    nobody asked for as missing -- the field score, the run-log row -- stands
    down on that key. See `_steps_for`.
    """
    table = _steps_for(only)
    source = grounding.Source(text)
    started = time.perf_counter()
    fields, steps, replies = {}, [], []
    total_tokens = 0

    # Sent once because it IS once: every step's message opens with this same
    # block and the same transcript, and only the question at the end differs.
    # That is the whole reason the steps are cheap on llama.cpp, and showing it
    # per step would say the opposite of what the design does.
    prefix = _step_prefix(text)

    yield {"event": "extract_steps", "total": len(table),
           "steps": [{"id": s["id"], "title": s["title"], "keys": list(s["keys"])}
                     for s in table],
           "prompt_prefix": prefix}

    for index, step in enumerate(table, 1):
        yield {"event": "extract_step", "step": index, "total": len(table),
               "id": step["id"], "title": step["title"], "status": "running"}

        question = _step_question(step)
        message = prefix + question
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
                attempt_values, raws, truncated, tokens = _ask_step(
                    content, step, status, raws)
                for n, raw in enumerate(raws):
                    label = _reply_label(step["id"], attempt, n)
                    replies.append(f"--- {label} ---\n" + raw)
                    record["raw"].append({"label": label, "text": raw,
                                          "prompt": asked})
                record["schema_retry"] = len(raws) > 1
                total_tokens += tokens
            except Exception as err:
                for n, raw in enumerate(raws):
                    label = _reply_label(step["id"], attempt, n, failed=True)
                    replies.append(f"--- {label} ---\n" + raw)
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
                record["error" if best_bad is None else "retry_error"] = why
                break
            elapsed += time.perf_counter() - at

            bad = _ungrounded_in(attempt_values, source)
            # Keep whichever attempt invents least. A retry that comes back worse
            # than the answer it was meant to correct is not an improvement, and
            # under greedy decoding that happens whenever the model has nothing
            # better to offer than what it already said.
            if best_bad is None or len(bad) < len(best_bad):
                values, best_bad = attempt_values, bad
            if not bad or attempt >= AGENTIC_RETRIES:
                break
            record["retried"] = True

        fields.update(values)
        record["seconds"] = round(elapsed, 2)
        record["values"] = {k: v for k, v in values.items() if not isinstance(v, list)}
        record["items"] = sum(len(v) for v in values.values() if isinstance(v, list))
        record["ungrounded"] = len(best_bad or [])
        steps.append(record)
        yield {"event": "extract_step", "step": index, "total": len(table),
               "id": step["id"], "title": step["title"], "status": "done", **record}

    elapsed = round(time.perf_counter() - started, 2)
    failed = [s for s in steps if s.get("error")]
    if len(failed) == len(steps):
        return {"error": "Every extraction step failed. First: "
                         + failed[0]["error"],
                "mode": "agentic", "steps": steps, "seconds": elapsed,
                "prompt_prefix": prefix,
                **({"steps_only": [s["id"] for s in table]}
                   if len(table) < len(EXTRACT_STEPS) else {}),
                "raw": "\n\n".join(replies)[:4000]}

    # Ordered the way the schema lists the keys rather than the way the steps
    # filled them, so the JSON on screen and the JSON a script copies are the same
    # shape in both modes.
    ordered = {k: fields[k] for k in grounding.SCALAR_FIELDS if k in fields}
    for key in ("line_items", "other_fields"):
        ordered[key] = fields.get(key) or []

    return {
        "fields": ordered,
        "raw": "\n\n".join(replies),
        # On the result as well as on the event that announced the steps: a
        # result read back from the queue, or from the blocking endpoint, never
        # saw the event, and a step's question is unreadable without what led it.
        "prompt_prefix": prefix,
        "grounding": grounding.check(ordered, text),
        "tiers": grounding.tier_counts(ordered),
        "vat_basis": verify.vat_basis(ordered, text),
        # Same derivation as single mode, from the merged answer rather than from
        # any one step -- the references list in particular spans several of them.
        "derived": normalise.derive(ordered, text),
        "seconds": elapsed,
        "tokens": total_tokens,
        "model": status["model"],
        "backend": status["kind"],
        "mode": "agentic",
        "steps": steps,
        # Present only when this run was restricted to part of the step table, so
        # nothing downstream reads its keys as the whole form. Absent on an
        # ordinary run rather than holding every id, for the same reason.
        **({"steps_only": [s["id"] for s in table]} if len(table) < len(EXTRACT_STEPS)
           else {}),
        # Named separately from the per-step errors because a partial answer is
        # the normal outcome here rather than a failure: the page says which parts
        # of the form were not filled instead of showing them as simply empty.
        "steps_failed": [{"id": s["id"], "title": s["title"], "error": s["error"]}
                         for s in failed],
    }


# --------------------------------------------------------------------------
# pass 2: which shape runs
# --------------------------------------------------------------------------

# Switched from the page rather than read from the environment per run, because it
# is a thing you flip while looking at a document that came out wrong. Held for the
# process, not per request: a batch queued now and a Re-extract clicked during it
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


def ocr_profile() -> str:
    with _profile_lock:
        return _ocr_profile


def set_ocr_profile(name: str) -> str:
    if name not in OCR_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(OCR_PROFILES)}.")
    global _ocr_profile
    with _profile_lock:
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


def loop_guard() -> bool:
    with _loop_guard_lock:
        return _loop_guard


def set_loop_guard(on: bool) -> bool:
    global _loop_guard
    with _loop_guard_lock:
        _loop_guard = bool(on)
    return _loop_guard


def extract_mode() -> str:
    with _mode_lock:
        return _extract_mode


def set_extract_mode(mode: str) -> str:
    if mode not in ("single", "agentic"):
        raise ValueError("mode must be 'single' or 'agentic'.")
    global _extract_mode
    with _mode_lock:
        _extract_mode = mode
    return mode


def extract_fields_stream(text: str, mode: str = None, case_id: str = None,
                          steps=None):
    """Turn a finished transcript into structured JSON, yielding progress.

    A separate text-only pass rather than part of the OCR prompt: mixing
    transcription and interpretation in one request measurably degrades the
    transcript, and this way extraction can be re-run without re-reading the page.

    Single mode has no progress to report and yields nothing. Both shapes return
    the same result dict, plus `mode` saying which one ran.

    `case_id` is a benchmark document this transcript came from, and it is scored
    here rather than at each of the four call sites: the two modes build their
    result dicts separately, and a score attached in only some of the places a
    result is produced is a score that quietly disappears depending on which
    button was pressed.
    """
    if not text.strip():
        return {"error": "Nothing to extract from."}
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
        return {"error": status["text_reason"]}
    if (mode or extract_mode()) == "agentic":
        result = yield from _extract_agentic(text, status, steps)
    else:
        # Refused rather than ignored: single mode is one request for the whole
        # schema, so there is no honest way to ask it for part of one, and a
        # sweep that thought it was measuring one step would be measuring
        # fourteen keys instead.
        if steps:
            return {"error": "steps only apply to agentic extraction."}
        result = _extract_single(text, status)
    return _score_fields(result, case_id)


def _score_fields(result: dict, case_id: str) -> dict:
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
            result["field_score"] = fieldscore.evaluate(case_id, result["fields"])
    except Exception as err:  # pragma: no cover - a score is never worth a 500
        result["field_score"] = {"error": f"field scoring failed: {err}"}
    return result


def _drain(generator):
    """Run a generator that returns a value, discarding what it yields."""
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def extract_fields(text: str, mode: str = None, case_id: str = None,
                   steps=None) -> dict:
    """`extract_fields_stream` for callers with nowhere to show progress."""
    return _drain(extract_fields_stream(text, mode, case_id, steps))


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


def finish_page(raw: str, stats: dict | None = None, profile: str = None) -> str:
    """One page's raw reply -> its transcript, with the boxes kept on `stats`.

    Called wherever a page finishes -- the streaming endpoint, the blocking one
    and the queue worker -- so all three produce the same thing from the same
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


def read_page(image: Image.Image, stats: dict | None = None,
              profile: str = None) -> str:
    """Blocking full-page read.

    Deliberately drains the streaming generator rather than issuing its own
    request, so both endpoints report timings measured exactly the same way.
    """
    return finish_page("".join(stream_page(image, stats, profile)), stats, profile)


# --------------------------------------------------------------------------
# file loading
# --------------------------------------------------------------------------

def _looks_like_pdf(data: bytes) -> bool:
    return data[:1024].lstrip()[:5] == b"%PDF-"


def load_pages(data: bytes):
    """Decode an upload into a list of RGB page images."""
    if _looks_like_pdf(data):
        if not PDF_OK:
            raise ValueError("PDF support needs PyMuPDF: pip install pymupdf")
        pages = []
        zoom = PDF_DPI / 72.0
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("That PDF is password-protected.")
            for page in doc.pages(0, min(doc.page_count, MAX_PAGES)):
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        if not pages:
            raise ValueError("That PDF has no pages.")
        return pages

    try:
        image = Image.open(io.BytesIO(data))
        # Multi-frame formats (TIFF scans, GIF) carry a page per frame.
        frames = [
            ImageOps.exif_transpose(frame).convert("RGB")
            for frame, _ in zip(ImageSequence.Iterator(image), range(MAX_PAGES))
        ]
    except UnidentifiedImageError:
        hint = "" if HEIF_OK else " (HEIC/HEIF needs: pip install pillow-heif)"
        raise ValueError(f"Unsupported or corrupt file format.{hint}") from None
    except Exception:
        raise ValueError("Could not read that file.") from None

    if not frames:
        raise ValueError("That file contains no image data.")
    return frames


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


def summarise(all_stats, detail, started, job_id=None):
    """Roll per-page stats into the totals both endpoints report."""
    tokens = sum(s.get("new_tokens", 0) for s in all_stats)
    decode = sum(s.get("decode_seconds", 0) for s in all_stats)
    models = [s.get("model") for s in all_stats if s.get("model")]
    # Taken from the pages themselves, not from the active endpoint: a run that
    # finished must be reported against the server that actually ran it, even if
    # the page has since been pointed somewhere else.
    urls = [s.get("url") for s in all_stats if s.get("url")]
    kinds = [s.get("backend") for s in all_stats if s.get("backend")]
    profiles = [s.get("ocr_profile") for s in all_stats if s.get("ocr_profile")]
    return {
        "page_count": len(all_stats),
        "detail": detail,
        # From the pages, like model and backend above: a profile switched during
        # a batch must not relabel the pages that were already read under the old
        # one.
        "ocr_profile": profiles[0] if profiles else ocr_profile(),
        "job": job_id,
        "model": models[0] if models else None,
        "url": urls[0] if urls else backends.active_url(),
        "backend": kinds[0] if kinds else None,
        "resolutions": [s.get("resolution") for s in all_stats],
        "tokens": tokens,
        "truncated": any(s.get("truncated") for s in all_stats),
        "looped": any(s.get("looped") for s in all_stats),
        "seconds": round(time.perf_counter() - started, 2),
        "prefill_seconds": round(sum(s.get("prefill_seconds", 0) for s in all_stats), 2),
        "decode_seconds": round(decode, 2),
        "tokens_per_second": round(tokens / decode, 2) if decode and tokens else 0,
        "page_stats": all_stats,
    }


# Two queued jobs for the same case would otherwise interleave their writes to
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


def prepare_input(data: bytes, detail: str, case=None, source=None):
    """Decode, trim and resize. Returns (pages, detail, page_job_id, case).

    `source` is carried into the job cache rather than only into the run log, so
    a later re-extraction against this transcript can name the same document.
    """
    # Through the alias table, so an old preset name from a saved setting or a
    # script lands on the nearest preset that still exists rather than silently
    # on the default. See `settings.resolve_detail`.
    detail = resolve_detail(detail)
    pages = load_pages(data)
    budget = DETAIL_PRESETS[detail]
    # Trim first, then fit: the pixel budget is spent on content, not margins.
    prepared = [fit_pixels(trim_margins(page) if TRIM_MARGINS else page, budget)
                for page in pages]
    context = {"source": source or {}, "detail": detail, "pages": len(prepared),
               "case": case["id"] if case else ""}
    return prepared, detail, register_job(prepared, context), case


def log_run(summary: dict, source: dict, status: str = None, error=None,
            run_type: str = "ocr"):
    """Append one line to the run log. Never raises.

    Measurements only -- no transcript, no fields, no images. See `runlog.py`.

    The row a read writes is remembered against its page job, so a later
    re-extraction of the same transcript can find it again and put better figures
    on it. Timestamp and file name are the whole key -- nothing about the row is
    held open, and losing it costs the update, not the run.
    """
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


def log_extract(result: dict, job_id: str = None, context: dict = None):
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
        "url": backends.active_url(),
        "extracted": result,
        # Enough for the `case` column to name the document; the accuracy scores
        # belong to the read that produced the transcript, not to this row.
        "truth": {"case": context.get("case", "")},
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

def run_job(job):
    """Worker body: OCR every page, then extract the fields.

    Cancellation is checked between pages rather than mid-page -- llama.cpp has no
    way to abandon a generation already in flight, so a finer granularity would be
    a lie about how quickly Cancel takes effect.
    """
    data, case = (None, None)
    if job.kind == "case":
        data, case = case_bytes(job.payload)
    elif job.kind == "file":
        data, case = resolve_mock(job.payload)
    else:
        data = job.payload
        case, _how = scoring.case_for_upload(filename=job.name, data=data)
    source = describe_source(job.name, data, "queue")

    job.stage = "preparing"
    pages, detail, page_job_id, case = prepare_input(data, job.detail, case, source)
    job.pages_total = len(pages)
    job.detail = detail

    started = time.perf_counter()
    collected, all_stats = [], []
    payload = {}
    # Logged from `finally` so a cancelled or failed job still leaves a row --
    # a read that died after four minutes is exactly what you want recorded.
    try:
        for index, page in enumerate(pages, 1):
            job.check_cancelled()
            job.stage = f"reading page {index} of {len(pages)}"
            stats = {}
            text = "".join(stream_page(page, stats))
            # Read back under the profile this page ran with, not the one set
            # now: a layout-JSON reply flattened as Markdown would be logged and
            # scored as a page of JSON.
            collected.append(finish_page(text, stats))
            all_stats.append(stats)
            job.pages_done = index

        job.check_cancelled()
        if len(collected) > 1:
            text = "\n\n".join(f"--- page {i} ---\n{t}"
                               for i, t in enumerate(collected, 1) if t)
        else:
            text = collected[0] if collected else ""

        payload = {"text": text, "pages": collected,
                   **summarise(all_stats, detail, started, page_job_id)}
        payload["truth"] = evaluate_if_known(case, text)

        if EXTRACT and text.strip():
            job.check_cancelled()
            job.stage = "extracting fields"
            # Stepped through rather than called, so agentic mode can name the
            # step in the queue row and can be cancelled between steps. Single
            # mode yields nothing, so this is the plain call it used to be.
            stream = extract_fields_stream(
                text, case_id=case["id"] if case else None)
            while True:
                try:
                    event = next(stream)
                except StopIteration as stop:
                    payload["extracted"] = stop.value
                    break
                if event.get("event") == "extract_step" \
                        and event.get("status") == "running":
                    job.stage = (f"extracting {event['step']}/{event['total']}:"
                                 f" {event['title']}")
                    job.check_cancelled()
        job.stage = "finished"
        log_run(payload, source)
        return payload
    except jobs.Cancelled:
        log_run(payload or summarise(all_stats, detail, started, page_job_id),
                source, status="cancelled")
        raise
    except Exception as err:
        log_run(payload or summarise(all_stats, detail, started, page_job_id),
                source, error=err)
        raise


def server_slots():
    """Parallel slots the active server advertises, or None if it does not.

    llama-server reports `total_slots`; Ollama's parallelism is set by
    OLLAMA_NUM_PARALLEL and is not exposed over the API, so it stays unknown and
    the page simply does not claim a number.
    """
    try:
        return backends.status().get("slots")
    except Exception:
        return None


def default_workers():
    """One worker per server slot: more just hides the wait inside the server."""
    if WORKERS_OVERRIDE:
        return WORKERS_OVERRIDE
    return max(1, int(server_slots() or 1))


job_queue = jobs.JobQueue(run_job, workers=default_workers())


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
        cases=[{"id": c["id"], "pdf": c["pdf"], "kind": c.get("kind", ""),
                "field_truth": fieldscore.has_truth(c["id"])}
               for c in scoring.cases_index().values()],
        mock_files=mock_files(),
        endpoints=backends.endpoints(),
        # What pass 2 will run on, seeded at render so both extraction pickers
        # are right on first paint rather than only after a switch or Re-check.
        # Same block `/api/servers` returns, so the page has one shape to read.
        extract=backends.overview()["extract"],
        ctx_choices=backends.NUM_CTX_CHOICES,
        num_ctx=backends.num_ctx(),
        extract_mode=extract_mode(),
        extract_steps=len(EXTRACT_STEPS),
        ocr_profile=ocr_profile(),
        ocr_profiles=[{"id": pid, **spec} for pid, spec in OCR_PROFILES.items()],
        # Whether a cycling read is cut short, and what it would cost if it were
        # not: the page prints the cap rather than repeating a number, for the
        # same reason the read floor below is sent instead of hardcoded.
        loop_guard=loop_guard(),
        max_new_tokens=MAX_NEW_TOKENS,
        # The random test's rule about when a field score is worth writing. Sent
        # rather than hardcoded in the template: it is a setting, and a page that
        # says 50% while the process runs at 0 is describing a build nobody has.
        min_read_for_fields=MIN_READ_FOR_FIELDS,
        # What a contest pins: how many from each end of the ranking it runs, and
        # the Detail every contender runs at. Sent rather than repeated in the
        # template for the same reason as the threshold above -- the page must
        # describe the build it is talking to.
        contest_top=randomtest.CONTEST_TOP,
        contest_bottom=randomtest.CONTEST_BOTTOM,
        contest_detail=randomtest.CONTEST_DETAIL,
        # What a contest can be about. The page paints its picker from this so a
        # subject added here cannot be missing there -- the same failure the
        # field-label maps had when the schema widened.
        contest_subjects=[{"id": sid, "label": spec["label"], "scope": spec["scope"]}
                          for sid, spec in randomtest.SUBJECTS.items()],
    )


@app.get("/api/health")
def health():
    return jsonify(server=llama_status(), pdf=PDF_OK, heif=HEIF_OK, status="ok")


@app.post("/api/context")
def context_set():
    """Set the context window sent with every later request.

    Deliberately not gated on the queue: the window is a per-request field, so
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
    return jsonify(backends.overview(force=force))


@app.post("/api/servers")
def servers_select():
    """Switch the app to another model server, and optionally another model.

    Refused while the queue is working: a document half-read on one server and
    half on another would be logged and scored as if one server had done it.
    """
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    model = (body.get("model") or "").strip()
    # Present and empty means "same as the reading model", which is a real
    # choice and not a missing field -- hence the sentinel rather than "".
    extract = body.get("extract_model")
    if not url and not model and extract is None:
        return jsonify(error="Give a url, a model, or both."), 400

    running = job_queue.stats()["counts"].get("running", 0)
    if running and url and backends.clean_url(url) != backends.active_url():
        return jsonify(error=f"{running} job(s) still running on "
                             f"{backends.active_url()}. Wait or cancel them "
                             "before switching server."), 409

    try:
        # Unloading is vetoed while anything is running. A model-only switch is
        # allowed mid-queue (it takes effect on the next run), but the eviction
        # is a request to the same scheduler serving the run in flight, so a
        # queue that is working keeps its weights and the next switch frees them.
        if url or model:
            server = backends.select(url or None, model or None, unload=not running)
        else:
            server = backends.status()
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


@app.get("/api/cases")
def cases():
    """Benchmark documents that have a ground-truth transcript."""
    return jsonify(cases=[
        {"id": c["id"], "pdf": c["pdf"], "kind": c.get("kind", ""),
         "pages": c.get("pages", 1), "available": c["pdf_path"].exists(),
         "field_truth": fieldscore.has_truth(c["id"])}
        for c in scoring.cases_index().values()
    ])


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


def _requested_mode(body):
    """The extraction shape for one request: what it asked for, or the current one."""
    mode = (body.get("mode") or "").strip().lower() or None
    if mode and mode not in ("single", "agentic"):
        raise ValueError("mode must be 'single' or 'agentic'.")
    return mode


def _requested_steps(body):
    """Which agentic steps one request runs, where it asks for only some.

    A benchmark handle, not a setting: it is what makes a change to one step's
    prompt measurable over several documents and models without paying for the
    steps the change did not touch. Held per request rather than in process
    state, so nothing the page or the queue does next inherits it.

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
    except ValueError as err:
        return jsonify(error=str(err)), 400
    result = extract_fields(text, mode, case_id, steps)
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
    except ValueError as err:
        return jsonify(error=str(err)), 400

    job_id = body.get("job")

    def generate():
        try:
            stream = extract_fields_stream(text, mode, case_id, steps)
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
            log_extract({"error": str(err)}, job_id, context)
            yield json.dumps({"event": "error", "error": str(err)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/api/extract/mode")
def extract_mode_get():
    return jsonify(mode=extract_mode(), steps=len(EXTRACT_STEPS))


@app.post("/api/extract/mode")
def extract_mode_set():
    """Switch the shape of pass 2 for everything this process extracts next.

    Unlike switching server, this is NOT refused while the queue is busy. Mixing
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
    return jsonify(mode=mode, steps=len(EXTRACT_STEPS))


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

    Not refused while the queue is busy, for the same reason switching extraction
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

    Not refused while the queue is busy, for the same reason the profile and the
    extraction mode are not: every page stamps the guard it ran under onto its own
    stats, so a batch split across the switch still says which page ran under
    which rule.
    """
    body = request.get_json(silent=True) or {}
    on = body.get("loop_guard")
    if not isinstance(on, bool):
        return jsonify(error="loop_guard must be true or false."), 400
    return jsonify(loop_guard=set_loop_guard(on), max_tokens=MAX_NEW_TOKENS)


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


@app.post("/api/queue/mode")
def queue_mode():
    """Switch between running one document at a time and running several.

    Concurrent mode is guided by the server's slot count: beyond that the extra
    workers only queue inside the model server, where the wait is invisible.
    """
    body = request.get_json(silent=True) or {}
    mode = (body.get("mode") or "").strip().lower()
    if mode not in ("sequential", "concurrent"):
        return jsonify(error="mode must be 'sequential' or 'concurrent'."), 400

    slots = server_slots()

    if mode == "sequential":
        job_queue.set_auto_scale(False)
        job_queue.set_workers(1)
    else:
        # Concurrent means "as wide as the batch": queueing five documents
        # starts five requests. An explicit worker count is a floor, not a cap --
        # the slot count is advice printed on the page, not a limit enforced here.
        job_queue.set_auto_scale(True)
        requested = body.get("workers")
        target = int(requested) if requested else max(job_queue.worker_count, 2)
        job_queue.set_workers(target)

    return jsonify(mode=mode, llama_slots=slots, **job_queue.stats())


@app.post("/api/queue/run")
def queue_run():
    """Start the queue, or stop it handing out more work.

    Queueing a document no longer starts it: the queue holds until this is
    called, which is what makes the run mode and the worker count settable
    against a batch you can already see. `{"start": false}` pauses -- a document
    already in flight finishes, because llama.cpp cannot abandon a generation
    it has begun and pretending otherwise would be a lie about what the button
    does. Cancelling is `DELETE /api/queue/<id>`.

    A batch that drains closes the gate behind it, so the next one waits for its
    own start.
    """
    body = request.get_json(silent=True) or {}
    if body.get("start", True):
        job_queue.start()
    else:
        job_queue.pause()
    # stats() already carries "started"; passing it again would be a duplicate
    # keyword argument.
    return jsonify(**job_queue.stats())


@app.get("/api/queue")
def queue_list():
    return jsonify(jobs=[j.to_dict() for j in job_queue.list()], **job_queue.stats())


@app.post("/api/queue")
def queue_add():
    """Enqueue one or more documents. Accepts uploads and/or case ids.

    Everything in the request is validated and read first, then handed to the
    queue in one go. Nothing starts running until the whole batch is queued, so
    a multi-file upload is a fair concurrency test rather than a head start for
    whichever file was parsed first.
    """
    detail = resolve_detail(request.form.get("detail", DEFAULT_DETAIL))

    specs = []
    # Case ids may be repeated: cases=sol001&cases=sol002
    for case_id in request.form.getlist("cases"):
        case_id = case_id.strip()
        if not case_id:
            continue
        try:
            _data, case = case_bytes(case_id)
        except ValueError as err:
            return jsonify(error=str(err)), 400
        specs.append((case_id, "case", detail, case_id))

    for name in request.form.getlist("files"):
        name = name.strip()
        if not name:
            continue
        try:
            resolve_mock(name)          # validate now, re-read in the worker
        except ValueError as err:
            return jsonify(error=str(err)), 400
        specs.append((name, "file", detail, name))

    for upload in request.files.getlist("image"):
        if not upload.filename:
            continue
        # Read now: the request object is gone by the time a worker picks it up.
        specs.append((upload.filename, "upload", detail, upload.read()))

    if not specs:
        return jsonify(error="Nothing to queue."), 400
    added = job_queue.submit_many(specs)
    return jsonify(added=[j.to_dict() for j in added], **job_queue.stats())


@app.get("/api/queue/<job_id>")
def queue_get(job_id):
    job = job_queue.get(job_id)
    if not job:
        return jsonify(error="No such job."), 404
    return jsonify(job.to_dict(include_result=True))


@app.delete("/api/queue/<job_id>")
def queue_cancel(job_id):
    job = job_queue.cancel(job_id)
    if not job:
        return jsonify(error="No such job."), 404
    return jsonify(job.to_dict())


@app.post("/api/queue/clear")
def queue_clear():
    return jsonify(removed=job_queue.clear_finished(), **job_queue.stats())


@app.post("/api/queue/workers")
def queue_workers():
    body = request.get_json(silent=True) or {}
    try:
        count = int(body.get("workers", 1))
    except (TypeError, ValueError):
        return jsonify(error="workers must be a number."), 400
    job_queue.set_workers(count)
    slots = server_slots()
    # stats() already carries "workers"; passing it separately as well would be a
    # duplicate keyword argument.
    return jsonify(llama_slots=slots, **job_queue.stats())


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
        pages = load_pages(data)
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
    return randomtest.pools(llama_status()["models"], cases)


def _random_plan(body: dict) -> dict:
    """Turn a request into a plan, or raise ValueError with something readable.

    Two shapes, and they are opposite questions asked of the same runner:

    - the random test -- every axis drawn, to find what breaks;
    - `contest` -- the standouts ranking re-run, every axis pinned except the
      model, to find out whether that ranking was real.

    The run log goes in as `history` (which spreads the rounds over the documents
    -- see `randomtest.case_order`) and, for a contest, as the ranking itself.
    Both are read here rather than inside the planner so that it stays a pure
    function of its arguments: a plan that read a file could not be reproduced
    from a seed by anyone who did not have that file.
    """
    scope = body.get("scope") or randomtest.DEFAULT_SCOPE
    # Exclusions narrow the pools before anything is planned, so they hold for a
    # contest as well: "do not test that model" is a statement about the run, not
    # about one button on the pane.
    pools = randomtest.apply_exclusions(_random_pools(), body.get("exclude"), scope)
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
        details=list(DETAIL_PRESETS), modes=["single", "agentic"],
        lock=body.get("lock"), history=runlog.case_counts(), **pools)


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
    return jsonify({**planned, "pools": _random_pools(),
                    "history": runlog.case_counts(),
                    "scopes": list(randomtest.SCOPES),
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

    # Refused for the same reason switching server is: a batch half-read on one
    # model and half on another would be logged as if one had done it, and this
    # switches models between every round.
    running = job_queue.stats()["counts"].get("running", 0)
    if running:
        return jsonify(error=f"{running} job(s) still running. Wait or cancel "
                             "them before starting a random test."), 409

    def generate():
        started = time.perf_counter()
        yield json.dumps({"event": "plan", **planned}) + "\n"
        completed = failed = 0
        for index, round_ in enumerate(planned["rounds"], 1):
            began = time.perf_counter()
            event = {"event": "round", "index": index,
                     "total": len(planned["rounds"]), "plan": round_}
            try:
                event["result"] = randomtest.summarise_round(_run_round(round_))
                completed += 1
            except Exception as err:                # noqa: BLE001
                # Reported as a failed round and the run continues. A random
                # test that stopped at the first failure would find one problem
                # per invocation, and finding several is the whole point.
                event["error"] = f"{type(err).__name__}: {err}"[:300]
                failed += 1
            event["seconds"] = round(time.perf_counter() - began, 1)
            yield json.dumps(event, ensure_ascii=False) + "\n"
        yield json.dumps({"event": "done", "completed": completed,
                          "failed": failed, "total": len(planned["rounds"]),
                          "seconds": round(time.perf_counter() - started, 1),
                          "seed": planned["seed"]}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


def _run_round(round_: dict) -> dict:
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

    The pass-1 profile is set only where a page is read. Setting it on a
    fields-only round would leave the process on a profile chosen for a model
    that never looked at anything.
    """
    scope = round_.get("scope", randomtest.DEFAULT_SCOPE)
    if scope == "fields":
        backends.select(None, round_["extractor"], unload=True)
        backends.select_extract("", unload=False)
        set_extract_mode(round_["mode"])
        return _extract_case(round_["case"], round_["mode"])

    backends.select(None, round_["reader"], unload=True)
    backends.select_extract(round_["extractor"], unload=False)
    set_ocr_profile(round_["profile"])
    if scope == "ocr":
        return _read_case(round_["case"], round_["detail"], extract=False)
    set_extract_mode(round_["mode"])
    return _read_case(round_["case"], round_["detail"])


def _unscorable_read(payload: dict) -> str:
    """Why this read's fields must not be scored, or "" where they may be.

    **Pass 2 can only map values pass 1 produced.** A field score taken over a
    broken transcript is a measurement of the read wearing the extractor's name:
    it marks down whatever extracted and lets whatever read get away with it.
    That gap is measured -- dots.mocr agentic scored 32.7% over its own reads
    against 46.8% over the ground truth -- and it is why every pass-2 baseline in
    CLAUDE.md was taken from truth.

    So the score is dropped, not the extraction: the run still exercises the real
    path, still writes its row, and still reports coverage, grounding and
    `other_fields`. What it does not do is put a correctness figure on the board
    that belongs to the other pass.

    Four ways a read disqualifies its own extraction, and the last is the
    threshold the user set (0.75 since 2026-08-24):

    - it looped, or was cut off at the token cap -- the transcript is a fragment
      whatever it scored;
    - it returned nothing at 0.0%;
    - it scored under `MIN_READ_FOR_FIELDS`.

    **None of these makes the run a failure.** A read that finished and scored
    badly is a run, and it is counted and averaged as one -- `runlog._incomplete`
    calls a loop or a crash a failure and nothing else. What is dropped here is
    only the pass-2 figure taken over that transcript.

    **A document with no transcript truth is never suppressed.** Nothing here can
    tell a bad read from an unmeasured one, and refusing to score on a guess
    would silently blank the figure for every document that has no `.md`.
    """
    if payload.get("looped"):
        return "the read looped"
    if payload.get("truncated"):
        return "the read hit the token cap"
    scored = (payload.get("truth") or {}).get("char_accuracy")
    if scored is None:
        return ""
    if scored <= 0:
        return "the read returned nothing"
    if scored < MIN_READ_FOR_FIELDS:
        return (f"the read scored {scored * 100:.1f}%, under "
                f"{MIN_READ_FOR_FIELDS * 100:.0f}%")
    return ""


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
    result = extract_fields(text, mode, case)
    log_extract(result, None, context)
    return {"extracted": result,
            "status": "error" if result.get("error") else "ok"}


def _read_case(case_id: str, detail: str, extract: bool = True) -> dict:
    """One benchmark document, read and extracted, exactly as `/api/ocr` does it.

    Shares `prepare_input`, `summarise`, `evaluate_if_known`, `extract_fields`
    and `log_run` with the route rather than reimplementing them: a test path
    that differs from the real one is a test of the test path.

    `extract=False` is the read-only scope: pass 2 does not run and the row's
    pass-2 columns stay blank, which is the honest record of a run that was
    testing the read.
    """
    data, case = case_bytes(case_id)
    source = describe_source(case["pdf"], data, "case")
    pages, detail, job_id, case = prepare_input(data, detail, case, source)

    started = time.perf_counter()
    page_texts, all_stats = [], []
    for page in pages:
        stats = {}
        page_texts.append(read_page(page, stats))
        all_stats.append(stats)

    if len(page_texts) > 1:
        text = "\n\n".join(f"--- page {i} ---\n{t}"
                            for i, t in enumerate(page_texts, 1) if t)
    else:
        text = page_texts[0]

    payload = summarise(all_stats, detail, started, job_id)
    payload["truth"] = evaluate_if_known(case, text)
    if extract and EXTRACT and text.strip():
        payload["extracted"] = extract_fields(text, case_id=case["id"] if case else None)
        # A field score is only about the extractor when the extractor was given
        # something to work with. Where the read failed or scored under
        # MIN_READ_FOR_FIELDS, the extraction still ran and is still logged --
        # what is dropped is the score, so the row carries coverage, grounding
        # and `other_fields` and no correctness figure. See `_unscorable_read`.
        why = _unscorable_read(payload)
        if why:
            payload["extracted"].pop("field_score", None)
            payload["fields_unscored"] = why
    log_run(payload, source)
    # The same three-way status the run log derives, so a round that looped says
    # so on the page instead of reading as a clean run with a low score. Set
    # after `log_run`, which derives its own from the same two flags.
    payload["status"] = ("looped" if payload.get("looped")
                         else "truncated" if payload.get("truncated") else "ok")
    return payload


@app.post("/api/ocr")
def ocr():
    """Blocking read. Convenient for scripts; the page uses /api/ocr/stream."""
    try:
        pages, detail, job_id, case, source = prepare(request.files, request.form)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    started = time.perf_counter()
    page_texts, all_stats = [], []
    try:
        for page in pages:
            stats = {}
            page_texts.append(read_page(page, stats))
            all_stats.append(stats)
    except ValueError as err:
        log_run(summarise(all_stats, detail, started, job_id), source, error=err)
        return jsonify(error=str(err)), 400
    except requests.RequestException as err:
        log_run(summarise(all_stats, detail, started, job_id), source, error=err)
        return jsonify(error=f"model server connection failed: {err}"), 502

    if len(page_texts) > 1:
        text = "\n\n".join(
            f"--- page {i} ---\n{t}" for i, t in enumerate(page_texts, 1) if t
        )
    else:
        text = page_texts[0]

    payload = summarise(all_stats, detail, started, job_id)
    payload["truth"] = evaluate_if_known(case, text)
    if EXTRACT and request.form.get("extract", "1") != "0" and text.strip():
        payload["extracted"] = extract_fields(
            text, case_id=case["id"] if case else None)
    log_run(payload, source)
    return jsonify(text=text, pages=page_texts, **payload)


@app.post("/api/ocr/stream")
def ocr_stream():
    """NDJSON stream, so a slow read shows partial text instead of hanging."""
    try:
        pages, detail, job_id, case, source = prepare(request.files, request.form)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    want_extract = request.form.get("extract", "1") != "0"

    def generate():
        started = time.perf_counter()
        collected, all_stats = [], []
        summary = {}
        try:
            for index, page in enumerate(pages, 1):
                yield json.dumps(
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
                parts, stats = [], {}
                for chunk in stream_page(page, stats):
                    parts.append(chunk)
                    yield json.dumps({"event": "token", "text": chunk}) + "\n"
                collected.append(finish_page("".join(parts), stats))
                all_stats.append(stats)
                yield json.dumps({"event": "page_done", "page": index, **stats}) + "\n"
                if stats.get("truncated"):
                    yield json.dumps({"event": "truncated", "page": index}) + "\n"
                if stats.get("looped"):
                    yield json.dumps({"event": "looped", "page": index}) + "\n"

            if len(collected) > 1:
                text = "\n\n".join(
                    f"--- page {i} ---\n{t}" for i, t in enumerate(collected, 1) if t
                )
            else:
                text = collected[0] if collected else ""
            summary = summarise(all_stats, detail, started, job_id)
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
                    text, case_id=case["id"] if case else None)
                while True:
                    try:
                        yield json.dumps(next(stream)) + "\n"
                    except StopIteration as stop:
                        result = stop.value
                        break
                summary["extracted"] = result
                yield json.dumps({"event": "fields", **result}) + "\n"

            # Logged once everything has run, so the row carries the accuracy and
            # the extraction counts rather than just the OCR timings.
            log_run(summary, source)
            yield json.dumps({"event": "logged"}) + "\n"
        except GeneratorExit:
            # The browser hung up -- Stop, or a closed tab. Still worth a row.
            log_run(summary or summarise(all_stats, detail, started, job_id),
                    source, status="cancelled")
            raise
        except Exception as err:  # surface failures inside the stream body
            log_run(summary or summarise(all_stats, detail, started, job_id),
                    source, error=err)
            yield json.dumps({"event": "error", "error": str(err)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


def _environment(rows: list) -> dict:
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
    last probe or nothing at all. This route is on a polling path: the run-log
    card re-fetches itself every five seconds while the tab is open, and
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
        "runs": runlog.conditions(rows),
    }


def _runs_payload(limit: int, window=None, min_read_pct=None,
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
        # **Its own bias is applied on top and is its own toggle**: it drops
        # single-source failures, weak models and time outliers whether or not
        # the card-wide one is on, because that tab is the one view here that is
        # meant to be shown rather than argued from. `bias=False` returns the
        # same six blocks over everything, which is what makes the bias
        # inspectable rather than load-bearing.
        totals["presentation"] = runlog.presentation(matched, bias=bias)
        # Beside it rather than inside it: the environment describes the runs
        # under every view, biased or not, and a card that changed its machine
        # when a checkbox moved would be describing a different computer.
        totals["environment"] = _environment(matched)
    return {
        "runs": everything[:limit],
        "columns": runlog.COLUMNS,
        "totals": totals,
        "facets": runlog.facets(everything),
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
    return Response(
        runlog.LOG_PATH.read_bytes(),
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
    if config.HOST not in ("127.0.0.1", "localhost", "::1"):
        say(f"[ocr] WARNING: bound to {config.HOST}, reachable from other hosts. "
            "This app has no authentication -- put it behind a reverse proxy or "
            "keep it on a trusted network.")

    # The machine before the model server: every timing in the run log is a wall
    # clock on THIS box, and until now nothing recorded which box that was.
    machine.report()

    status = llama_status()
    say(f"[ocr] {status['kind'] or 'server'} {status['url']} "
        f"model={status['model']} available={status['available']}")
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


if __name__ == "__main__":
    preflight()
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False,
        threaded=True,
    )
