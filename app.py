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
# the four prompts; `settings.py` holds every value the app runs with, and the
# comment beside each one says what was measured to arrive at it.
from prompts import EXTRACT_PROMPT, EXTRACT_REMINDER, PROMPT, VERIFY_PROMPT
from settings import (
    ACCEPTED_SUFFIXES,
    DEFAULT_DETAIL,
    DETAIL_PRESETS,
    EXTRACT,
    EXTRACT_LOOP_MIN_REPEATS,
    EXTRACT_LOOP_TAIL_CHARS,
    EXTRACT_MAX_TOKENS,
    EXTRACT_REPEAT_THRESHOLD,
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
    VERIFY_MAX_TOKENS,
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
        "messages": [{"role": "user", "content": content}],
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


def _repeated_line_items(raw: str, threshold: int = EXTRACT_REPEAT_THRESHOLD) -> bool:
    """True when the same line item was emitted over and over.

    `looks_repetitive` misses this: the model cycles over a *set* of items rather
    than one block, so no single repeating unit spans the tail. Comparing the
    descriptions catches it, and a genuine document never lists one description
    three times over.
    """
    seen = {}
    for desc in re.findall(r'"description"\s*:\s*"([^"]{12,})"', raw):
        seen[desc] = seen.get(desc, 0) + 1
        if seen[desc] >= threshold:
            return True
    return False


def _is_ocr_envelope(raw: str) -> bool:
    """True when a reply is an OCR model's built-in {"natural_text": ...} answer.

    Checked on the raw text rather than on parsed fields, because the envelope is
    just as recognisable when it was cut off before it could be parsed at all.
    """
    return raw.lstrip().lstrip("`json \n").startswith('{"natural_text"')


def extract_fields(text: str) -> dict:
    """Ask the model to turn a finished transcript into structured JSON.

    A separate text-only request rather than part of the OCR prompt: mixing
    transcription and interpretation in one pass measurably degrades the
    transcript, and this way the extraction can be re-run without re-reading
    the page.
    """
    if not text.strip():
        return {"error": "Nothing to extract from."}
    status = llama_status()
    if not status["available"]:
        return {"error": status["reason"]}
    payload = {
        "messages": [{"role": "user",
                      "content": EXTRACT_PROMPT + text + EXTRACT_REMINDER}],
        "max_tokens": EXTRACT_MAX_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        # Ask the server to constrain decoding to valid JSON where it supports
        # it; the parser below still copes if it does not.
        "response_format": {"type": "json_object"},
        "stream": False,
        **backends.request_extras(status),
    }

    started = time.perf_counter()
    try:
        res = requests.post(chat_url(), json=payload, timeout=GEN_TIMEOUT)
        if res.status_code != 200:
            return {"error": f"{status['kind'] or 'model server'} HTTP "
                             f"{res.status_code}: {res.text[:200]}"}
        body = res.json()
        choice = body["choices"][0]
        raw = (choice["message"].get("content") or "").strip()
        truncated = choice.get("finish_reason") == "length"
    except Exception as err:
        return {"error": f"extraction request failed: {err}"}

    elapsed = round(time.perf_counter() - started, 2)
    candidate = _first_json_object(strip_fence(raw)) or raw
    try:
        fields = json.loads(candidate)
    except json.JSONDecodeError as err:
        # A looping model produces JSON that is merely *unfinished*; say so,
        # because "invalid JSON" sends you looking in the wrong place.
        # Wider window than the OCR stream uses: an extraction loop repeats a
        # whole clause inside one JSON string, so the repeating unit is long and
        # a 600-character tail cannot hold enough repeats of it to be recognised.
        if looks_repetitive(raw, tail_chars=EXTRACT_LOOP_TAIL_CHARS,
                            min_repeats=EXTRACT_LOOP_MIN_REPEATS):
            return {"error": "The model repeated itself while extracting and never "
                             "closed the JSON. Try Re-extract; if it persists the "
                             "transcript is probably too noisy to parse.",
                    "looped": True, "raw": raw[:2000], "seconds": elapsed}
        # A parse error here is nearly always a cut-off reply rather than
        # malformed syntax, and "invalid JSON" sends you looking at the parser
        # instead of at the token budget or the model.
        if _is_ocr_envelope(raw):
            return {"error": f"'{status['model']}' ignored the extraction prompt and "
                             "transcribed the document instead, returning its OCR "
                             '{"natural_text": ...} envelope. The transcript is long '
                             "enough to drown the instructions. Re-extract; if it "
                             "persists, extract one page at a time.",
                    "envelope": True, "raw": raw[:2000], "seconds": elapsed}
        if _repeated_line_items(raw):
            return {"error": "The model cycled over the same line items until it hit "
                             f"the {EXTRACT_MAX_TOKENS}-token cap, so the JSON never "
                             "closed. Raising the cap will not help. Try Re-extract, "
                             "or extract one page at a time -- a long transcript with "
                             "repeated rows is what triggers this.",
                    "looped": True, "raw": raw[:2000], "seconds": elapsed}
        # A parse error here is nearly always a cut-off reply rather than malformed
        # syntax, and "invalid JSON" sends you looking at the parser instead of at
        # the token budget.
        if truncated:
            return {"error": f"The JSON was cut off mid-value at the "
                             f"{EXTRACT_MAX_TOKENS}-token cap. Raise "
                             "EXTRACT_MAX_TOKENS if the document really has this "
                             "many line items.",
                    "truncated": True, "raw": raw[:2000], "seconds": elapsed}
        return {"error": f"model did not return valid JSON ({err})",
                "raw": raw[:2000], "seconds": elapsed}
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

    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    # llama.cpp reports `timings`, Ollama reports OpenAI-style `usage`.
    tokens = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
    return {
        "fields": fields,
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
        # Which of these figures already carry tax, decided the same way the
        # number checks decide it. It belongs on the extraction result and not
        # only on the verification one, because the Fields tab labels the totals
        # from it: "Amount ex VAT" is a plain lie on a VAT-inclusive page, and a
        # reader taking `subtotal` for a pre-tax figure would be out by the VAT.
        "vat_basis": verify.vat_basis(fields, text),
        "seconds": elapsed,
        "tokens": tokens,
        "model": status["model"],
        "backend": status["kind"],
    }


def verify_with_llm(fields: dict, checks: list, basis: dict = None) -> dict:
    """Ask the model to review fields the calculator cannot judge.

    Given the arithmetic results up front, explicitly told not to redo them --
    a 2B model is poor at addition, so letting it 'check' the sums would add
    noise and occasionally contradict a correct calculation.
    """
    status = llama_status()
    if not status["available"]:
        return {"error": status["reason"]}
    summary = [
        f"- {c['name']}: {c['status'].upper()}"
        + ("" if c["difference"] is None else f" (out by {c['difference']:,.2f})")
        for c in checks
    ]
    # Stated before the results, because it is what makes them read correctly:
    # without it a VAT-inclusive page looks to the reviewer like lines that fail
    # to add up to the goods total, and it reports that as an error.
    basis_note = ""
    if basis:
        basis_note = (
            f"\n\nVAT basis of this document: {basis['summary']}"
            f" — {basis['source']}"
            + (f" ({basis['evidence']})" if basis.get("evidence") else "")
            + ".\n"
            + ("The line amounts printed on this page ALREADY contain VAT. They are"
               " meant to reach the VAT-inclusive total, not the goods total, and the"
               " VAT figure is contained within them rather than added to them."
               if basis["inclusive"] else
               "The line amounts printed on this page are before VAT, which is added"
               " below them.")
        )
    content = (
        VERIFY_PROMPT
        + json.dumps(fields, ensure_ascii=False, indent=1)
        + basis_note
        + "\n\nCalculator results (authoritative):\n"
        + "\n".join(summary)
    )

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": VERIFY_MAX_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        # Same repetition guard as the OCR pass. Without it the model can lock
        # into repeating a clause inside a JSON string until it hits the token
        # cap, which surfaces only as an unterminated-string parse error.
        **sampler_extras(),
        "response_format": {"type": "json_object"},
        "stream": False,
        **backends.request_extras(status),
    }
    started = time.perf_counter()
    try:
        res = requests.post(chat_url(), json=payload, timeout=GEN_TIMEOUT)
        if res.status_code != 200:
            return {"error": f"{status['kind'] or 'model server'} HTTP "
                             f"{res.status_code}: {res.text[:200]}"}
        raw = (res.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as err:
        return {"error": f"verification request failed: {err}"}

    elapsed = round(time.perf_counter() - started, 2)
    parsed = _first_json_object(strip_fence(raw)) or raw
    try:
        review = json.loads(parsed)
    except json.JSONDecodeError as err:
        return {"error": f"model did not return valid JSON ({err})",
                "raw": raw[:1500], "seconds": elapsed}

    issues = review.get("issues") if isinstance(review, dict) else None
    return {
        "issues": [i for i in (issues or []) if isinstance(i, dict)],
        "notes": (review.get("notes") if isinstance(review, dict) else "") or "",
        # Same reason as the extraction pass: an issue silently dropped here --
        # because it was not an object, or sat outside the first JSON object --
        # is invisible unless the verbatim reply is kept alongside.
        "raw": raw,
        "seconds": elapsed,
    }


def verify_fields(fields: dict, transcript: str = None, ask_llm: bool = True) -> dict:
    """Arithmetic checks, then optionally the model's review of the rest.

    The transcript is passed through so a VAT rate the extractor missed can still
    be recovered from the document text.
    """
    # Decided once and passed to both the calculator and the model, so the page
    # cannot be checked on one basis and reviewed on the other.
    basis = verify.vat_basis(fields, transcript)
    checks = verify.run_checks(fields, transcript, basis)
    rate, source = verify.resolve_vat_rate(fields, transcript)
    result = {
        "checks": checks,
        "vat_rate": rate,
        "vat_rate_source": source,
        "vat_basis": basis,
        **verify.summarise_checks(checks),
    }
    if ask_llm:
        result["review"] = verify_with_llm(fields, checks, basis)
    return result


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
    """Worker body: OCR every page, then extract and verify.

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
            payload["extracted"] = extract_fields(text)
            if payload["extracted"].get("fields"):
                job.check_cancelled()
                job.stage = "verifying numbers"
                payload["verified"] = verify_fields(
                    payload["extracted"]["fields"], transcript=text)
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


@app.post("/api/extract")
def extract_endpoint():
    """Re-run field extraction on text, without re-reading the page."""
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    if not text.strip():
        return jsonify(error="No text supplied."), 400
    return jsonify(extract_fields(text))


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


@app.post("/api/verify")
def verify_endpoint():
    """Re-run number verification on a fields object."""
    body = request.get_json(silent=True) or {}
    fields = body.get("fields")
    if not isinstance(fields, dict):
        return jsonify(error="No fields object supplied."), 400
    return jsonify(verify_fields(fields,
                                 transcript=body.get("text"),
                                 ask_llm=body.get("llm", True)))


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
        if payload["extracted"].get("fields"):
            payload["verified"] = verify_fields(
                payload["extracted"]["fields"], transcript=text)
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
                result = extract_fields(text)
                summary["extracted"] = result
                yield json.dumps({"event": "fields", **result}) + "\n"

                # Third pass: check the numbers that were just extracted.
                if result.get("fields"):
                    yield json.dumps({"event": "verifying"}) + "\n"
                    checked = verify_fields(result["fields"], transcript=text)
                    summary["verified"] = checked
                    yield json.dumps({"event": "verified", **checked}) + "\n"

            # Logged once everything has run, so the row carries the accuracy and
            # the number verdict rather than just the OCR timings.
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
