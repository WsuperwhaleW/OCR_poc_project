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
import grounding
import jobs
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
)
from settings import (
    ACCEPTED_SUFFIXES,
    AGENTIC_EXTRACT,
    AGENTIC_RETRIES,
    DEFAULT_DETAIL,
    DETAIL_PRESETS,
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
    LOOP_MIN_REPEATS,
    LOOP_TAIL_CHARS,
    MAX_JOBS,
    MAX_NEW_TOKENS,
    MAX_PAGES,
    MAX_UPLOAD_MB,
    PDF_DPI,
    PROMPT_FIRST,
    PROMPT_FIRST_OLLAMA,
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


def register_job(pages):
    """Cache PNG-encoded page images and return a job id."""
    job_id = uuid.uuid4().hex[:12]
    encoded = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        encoded.append(buf.getvalue())
    with _jobs_lock:
        _jobs[job_id] = encoded
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)
    return job_id


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
    """
    for line in text[-LOOP_TAIL_CHARS * 2 :].splitlines():
        if len(line) < LOOP_COUNTER_MIN_LINE:
            continue
        norm = _DIGITS.sub("#", line)
        for unit in range(4, LOOP_COUNTER_MAX_UNIT + 1):
            block = norm[-unit:]
            if block.strip() and norm.endswith(block * LOOP_COUNTER_MIN_REPEATS):
                return True
    return False


def looks_repetitive(text: str, tail_chars: int = LOOP_TAIL_CHARS,
                     min_repeats: int = LOOP_MIN_REPEATS) -> bool:
    """True when the tail of the output is cycling on the same block.

    Checks whether the last LOOP_TAIL_CHARS characters are built from a single
    repeated unit. Only long runs trip it, so a document that legitimately repeats
    a short value a few times is left alone.
    """
    if _counter_loop(text):
        return True

    tail = text[-tail_chars:]
    if len(tail) < tail_chars:
        return False

    # Whole-line cycling: look for a consecutive run of identical lines anywhere in
    # the tail, not just at the very end -- a loop often trails a stray closing tag.
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    run = 1
    for prev, cur in zip(lines, lines[1:]):
        if cur == prev and len(cur) > 3:
            run += 1
            if run >= min_repeats:
                return True
        else:
            run = 1

    # Block cycling, which also catches loops with no newlines at all. Anchored at
    # the end, which is where a loop always is while streaming; a periodic string
    # stays periodic from the end even if the check lands mid-block.
    for unit in range(4, len(tail) // min_repeats + 1):
        block = tail[-unit:]
        if block.strip() and tail.endswith(block * min_repeats):
            return True
    return False


def image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stream_page(image: Image.Image, stats: dict | None = None):
    """Stream tokens for one page from llama-server."""
    status = llama_status()
    if not status["available"]:
        raise ValueError(status["reason"])

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
    text_part = {"type": "text", "text": PROMPT}
    image_part = {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
    content = [text_part, image_part] if prompt_first else [image_part, text_part]

    payload = {
        # Ollama gets an explicit system message here; llama.cpp gets none. See
        # backends.system_prefix -- this reproduces what Ollama was injecting from
        # the Modelfile by itself, rather than adding anything new.
        "messages": backends.system_prefix(status)
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
                if (pieces % LOOP_CHECK_EVERY == 0
                        and looks_repetitive("".join(collected))):
                    looped = True
                    break

    finished = time.perf_counter()
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
    return retry


def _extract_once(text: str, status: dict, schema) -> dict:
    """One extraction request, parsed, salvaged if it was cut off, and grounded.

    `schema` constrains decoding to those keys where the server supports it and
    is None for the plain "return some JSON object" request. Everything else --
    the prompt, the sampler, the diagnosis of an unusable reply -- is the same
    either way, which is what makes the two attempts comparable.
    """
    url, payload = backends.structured_request(
        backends.system_prefix(status)
        + [{"role": "user", "content": EXTRACT_PROMPT + text + EXTRACT_REMINDER}],
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
            return {"error": why, **flags, "raw": raw[:2000], "seconds": elapsed}
        fields = salvaged
        partial = (why + " The fields below are what the reply had finished "
                         "before it stopped; the rest are empty because it never "
                         "got to them, not because the page is silent.")
    if not isinstance(fields, dict):
        return {"error": "model returned JSON that is not an object",
                "raw": raw[:2000], "seconds": elapsed}
    # Valid JSON, wrong JSON: a short document's envelope closes inside the cap and
    # would otherwise be stored and scored as if it were the extracted fields.
    if _is_ocr_envelope(raw):
        return {"error": f"'{status['model']}' returned a transcript in its OCR "
                         '{"natural_text": ...} envelope, not the requested fields. '
                         "Re-extract; if it persists, extract one page at a time.",
                "envelope": True, "raw": raw[:2000], "seconds": elapsed}

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
    there is no way to keep fifteen hand-written pairs honest. The step's
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


def _step_message(text: str, step: dict) -> str:
    """The message for one step: prefix, transcript, then this step's question.

    Assembled in that order on purpose. The transcript first is what makes the
    prefix shared across every step and therefore cacheable; the instructions last
    is what keeps an OCR fine-tune from answering with its own transcript envelope,
    which is the same thing EXTRACT_REMINDER buys the single-shot prompt.
    """
    return (EXTRACT_STEP_PREFIX + text
            + EXTRACT_STEP_TASK.format(title=step["title"],
                                       skeleton=step["skeleton"],
                                       rules=step["rules"]))


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


def _ask_step(content: str, step: dict, status: dict):
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
    """
    replies, tokens = [], 0
    raw, truncated, used = _chat(content, step["max_tokens"], status)
    replies.append(raw)
    tokens += used
    try:
        return _step_values(step, raw), replies, truncated, tokens
    except Exception:
        if not EXTRACT_SCHEMA:
            raise
    raw, truncated, used = _chat(content, step["max_tokens"], status,
                                 _step_schema(step))
    replies.append(raw)
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


def _extract_agentic(text: str, status: dict):
    """Walk the step table, yielding progress, and return the merged result.

    A generator so that both the browser and the run stream can show which step is
    running without a callback that cannot yield: iterate it for the events, and
    take the finished result from StopIteration.value (`_drain` does this).
    """
    source = grounding.Source(text)
    started = time.perf_counter()
    fields, steps, replies = {}, [], []
    total_tokens = 0

    yield {"event": "extract_steps", "total": len(EXTRACT_STEPS),
           "steps": [{"id": s["id"], "title": s["title"], "keys": list(s["keys"])}
                     for s in EXTRACT_STEPS]}

    for index, step in enumerate(EXTRACT_STEPS, 1):
        yield {"event": "extract_step", "step": index, "total": len(EXTRACT_STEPS),
               "id": step["id"], "title": step["title"], "status": "running"}

        message = _step_message(text, step)
        record = {"id": step["id"], "title": step["title"],
                  "keys": list(step["keys"]), "attempts": 0, "retried": False}
        values, elapsed = {}, 0.0
        best_bad = None

        for attempt in range(AGENTIC_RETRIES + 1):
            content = message
            if attempt:
                content += EXTRACT_STEP_RETRY.format(rejected="\n".join(best_bad))
            record["attempts"] = attempt + 1
            at = time.perf_counter()
            truncated = False
            try:
                attempt_values, raws, truncated, tokens = _ask_step(
                    content, step, status)
                for n, raw in enumerate(raws):
                    replies.append(f"--- {step['id']}"
                                   + (" (retry)" if attempt else "")
                                   + (" (schema)" if n else "") + " ---\n" + raw)
                record["schema_retry"] = len(raws) > 1
                total_tokens += tokens
            except Exception as err:
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
        yield {"event": "extract_step", "step": index, "total": len(EXTRACT_STEPS),
               "id": step["id"], "title": step["title"], "status": "done", **record}

    elapsed = round(time.perf_counter() - started, 2)
    failed = [s for s in steps if s.get("error")]
    if len(failed) == len(steps):
        return {"error": "Every extraction step failed. First: "
                         + failed[0]["error"],
                "mode": "agentic", "steps": steps, "seconds": elapsed,
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
        "grounding": grounding.check(ordered, text),
        "tiers": grounding.tier_counts(ordered),
        "vat_basis": verify.vat_basis(ordered, text),
        "seconds": elapsed,
        "tokens": total_tokens,
        "model": status["model"],
        "backend": status["kind"],
        "mode": "agentic",
        "steps": steps,
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


def extract_fields_stream(text: str, mode: str = None):
    """Turn a finished transcript into structured JSON, yielding progress.

    A separate text-only pass rather than part of the OCR prompt: mixing
    transcription and interpretation in one request measurably degrades the
    transcript, and this way extraction can be re-run without re-reading the page.

    Single mode has no progress to report and yields nothing. Both shapes return
    the same result dict, plus `mode` saying which one ran.
    """
    if not text.strip():
        return {"error": "Nothing to extract from."}
    status = llama_status()
    if not status["available"]:
        return {"error": status["reason"]}
    if (mode or extract_mode()) == "agentic":
        return (yield from _extract_agentic(text, status))
    return _extract_single(text, status)


def _drain(generator):
    """Run a generator that returns a value, discarding what it yields."""
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def extract_fields(text: str, mode: str = None) -> dict:
    """`extract_fields_stream` for callers with nowhere to show progress."""
    return _drain(extract_fields_stream(text, mode))


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


def read_page(image: Image.Image, stats: dict | None = None) -> str:
    """Blocking full-page read.

    Deliberately drains the streaming generator rather than issuing its own
    request, so both endpoints report timings measured exactly the same way.
    """
    return normalise_output(strip_fence("".join(stream_page(image, stats))))


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
    * At Detail `max` (uncapped) it is a genuine ~13% pixel, and therefore prefill,
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
    return {
        "page_count": len(all_stats),
        "detail": detail,
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


def prepare_input(data: bytes, detail: str, case=None):
    """Decode, trim and resize. Returns (pages, detail, page_job_id, case)."""
    if detail not in DETAIL_PRESETS:
        detail = DEFAULT_DETAIL
    pages = load_pages(data)
    budget = DETAIL_PRESETS[detail]
    # Trim first, then fit: the pixel budget is spent on content, not margins.
    prepared = [fit_pixels(trim_margins(page) if TRIM_MARGINS else page, budget)
                for page in pages]
    return prepared, detail, register_job(prepared), case


def log_run(summary: dict, source: dict, status: str = None, error=None):
    """Append one line to the run log. Never raises.

    Measurements only -- no transcript, no fields, no images. See `runlog.py`.
    """
    try:
        runlog.record(summary, source,
                      {"status": status, "error": str(error) if error else ""})
    except Exception:
        pass


def describe_source(name: str, data: bytes, origin: str) -> dict:
    """What the run log needs to identify the input: name, size and where from.

    Size is measured on the bytes actually read, so a case PDF and the same file
    uploaded by hand report identically.
    """
    return {"name": name, "size_bytes": len(data or b""), "origin": origin}


def prepare(request_files, form):
    """Request-shaped wrapper around `prepare_input`.

    Returns (pages, detail, page_job_id, case, source).
    """
    case = None
    case_id = (form.get("case") or "").strip()
    mock_name = (form.get("file") or "").strip()
    if case_id:
        data, case = case_bytes(case_id)
        source = describe_source(case["pdf"], data, "case")
    elif mock_name:
        data, case = resolve_mock(mock_name)
        source = describe_source(mock_name, data, "folder")
    else:
        upload = request_files.get("image")
        if upload is None or not upload.filename:
            raise ValueError("No file uploaded.")
        data = upload.read()
        source = describe_source(upload.filename, data, "upload")
        # Match on contents as well as name, so a renamed copy still scores.
        case, _how = scoring.case_for_upload(filename=upload.filename, data=data)
    return (*prepare_input(data, form.get("detail", DEFAULT_DETAIL), case), source)


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
    pages, detail, page_job_id, case = prepare_input(data, job.detail, case)
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
            collected.append(normalise_output(strip_fence(text)))
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
            stream = extract_fields_stream(text)
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
        cases=[{"id": c["id"], "pdf": c["pdf"], "kind": c.get("kind", "")}
               for c in scoring.cases_index().values()],
        mock_files=mock_files(),
        endpoints=backends.endpoints(),
        ctx_choices=backends.NUM_CTX_CHOICES,
        num_ctx=backends.num_ctx(),
        extract_mode=extract_mode(),
        extract_steps=len(EXTRACT_STEPS),
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
    if not url and not model:
        return jsonify(error="Give a url, a model, or both."), 400

    running = job_queue.stats()["counts"].get("running", 0)
    if running and url and backends.clean_url(url) != backends.active_url():
        return jsonify(error=f"{running} job(s) still running on "
                             f"{backends.active_url()}. Wait or cancel them "
                             "before switching server."), 409

    try:
        server = backends.select(url or None, model or None)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    # The pool is left alone. It used to be clamped to the new server's slot
    # count, which silently undid a concurrency setting chosen on purpose; the
    # page reports the mismatch instead so the choice stays the user's.
    # overview() carries the same fresh status under "server"; select() has just
    # primed the cache, so this does not re-probe.
    return jsonify(backends.overview())


@app.get("/api/cases")
def cases():
    """Benchmark documents that have a ground-truth transcript."""
    return jsonify(cases=[
        {"id": c["id"], "pdf": c["pdf"], "kind": c.get("kind", ""),
         "pages": c.get("pages", 1), "available": c["pdf_path"].exists()}
        for c in scoring.cases_index().values()
    ])


def _requested_mode(body):
    """The extraction shape for one request: what it asked for, or the current one."""
    mode = (body.get("mode") or "").strip().lower() or None
    if mode and mode not in ("single", "agentic"):
        raise ValueError("mode must be 'single' or 'agentic'.")
    return mode


@app.post("/api/extract")
def extract_endpoint():
    """Re-run field extraction on text, without re-reading the page.

    Blocking. Agentic mode is ~15 requests, so the page uses the streaming
    endpoint below and this stays for scripts, which have nowhere to show a step.
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    if not text.strip():
        return jsonify(error="No text supplied."), 400
    try:
        mode = _requested_mode(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(extract_fields(text, mode))


@app.post("/api/extract/stream")
def extract_stream_endpoint():
    """The same, as NDJSON, so agentic mode can say which step it is on.

    Single mode emits no step events and one `fields` event, so the browser reads
    both modes the same way.
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    if not text.strip():
        return jsonify(error="No text supplied."), 400
    try:
        mode = _requested_mode(body)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    def generate():
        try:
            stream = extract_fields_stream(text, mode)
            while True:
                try:
                    yield json.dumps(next(stream)) + "\n"
                except StopIteration as stop:
                    yield json.dumps({"event": "fields", **stop.value}) + "\n"
                    return
        except Exception as err:  # surface failures inside the stream body
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
    detail = request.form.get("detail", DEFAULT_DETAIL)
    if detail not in DETAIL_PRESETS:
        detail = DEFAULT_DETAIL

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
        payload["extracted"] = extract_fields(text)
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
                collected.append(normalise_output(strip_fence("".join(parts))))
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
                stream = extract_fields_stream(text)
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


@app.get("/api/runs")
def runs_list():
    """Recent rows of the run log, newest first."""
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 1000))
    except ValueError:
        limit = 50
    return jsonify(runs=runlog.read(limit), columns=runlog.COLUMNS,
                   totals=runlog.totals())


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
