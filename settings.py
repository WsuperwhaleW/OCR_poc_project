"""Every tunable `app.py` reads, in one file.

Split out of `app.py` so the knobs can be found and changed without reading the
code that consumes them. `config.py` still owns the paths and the typed
environment readers; this module owns the *values* the OCR app runs with, and
imports the readers from there.

Three things worth knowing before editing anything here:

* **Read once, at import.** A change takes effect on the next start of the
  process, never mid-run. Values already handed to a job in flight stay as they
  were.
* **Read through `config.env_*`, never `os.environ` directly.** Those clamp,
  warn on stderr and fall back, so a typo in a service file degrades one
  setting instead of killing startup. A new environment-backed setting belongs
  in `.env.example` with its default.
* **Most of these were measured, not chosen.** The comments say what was
  measured. The usual failure mode for a bad value here is a silent accuracy
  drop at HTTP 200, not an exception -- run `python compare.py` after changing
  one.
"""

import config
import jobs  # for MAX_WORKERS only; jobs.py imports nothing but the standard library

# --------------------------------------------------------------------------
# intake limits
# --------------------------------------------------------------------------

# All four are per-request costs paid by the model server, so a deployment
# sharing one server between several users will want them lower than these
# single-user defaults.
MAX_UPLOAD_MB = config.env_int("MAX_UPLOAD_MB", 32, minimum=1, maximum=512)
MAX_PAGES = config.env_int("MAX_PAGES", 10, minimum=1, maximum=200)
# Render PDFs above the pixel cap so the downscale resamples from real detail
# rather than the model reading a coarsely rasterised page.
PDF_DPI = config.env_int("PDF_DPI", 300, minimum=72, maximum=600)
MAX_NEW_TOKENS = config.env_int("MAX_NEW_TOKENS", 4096, minimum=256)

# File types the folder picker will offer and `resolve_mock` will read. PDFs go
# through PyMuPDF, everything else through Pillow; the multi-frame formats
# (TIFF, GIF) are read a page per frame.
ACCEPTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
                     ".tif", ".tiff", ".heic", ".heif"}

# Rendered page images are kept so the browser can show the source side by side
# with the extracted text. These are the *prepared* pages -- post-rasterisation
# and post-downscale -- so the comparison shows exactly what the model saw.
# Held in memory, so this is a RAM ceiling as much as a history depth: a 10-page
# document at `accurate` is ~40 MB of PNG. Lower it on a small server.
MAX_JOBS = config.env_int("MAX_JOBS", 5, minimum=1, maximum=100)

# Queue workers. 0 means "one per slot the model server advertises", which is the
# right answer: more workers than slots adds no throughput, it just moves the wait
# out of jobs.py -- where it is visible and cancellable -- into llama.cpp's own
# queue, where it is neither. Set OCR_WORKERS only to override that.
WORKERS_OVERRIDE = config.env_int("OCR_WORKERS", 0, minimum=0,
                                  maximum=jobs.MAX_WORKERS)

# --------------------------------------------------------------------------
# request timeouts
# --------------------------------------------------------------------------

# (connect, read) for a generation request. The read timeout is the one that
# matters: a CPU-only server reading a dense page at `max` detail can legitimately
# take many minutes, and a timeout that fires mid-generation throws away work the
# server is still doing. Raise it rather than lower it on slow hardware.
GEN_CONNECT_TIMEOUT = config.env_float("GEN_CONNECT_TIMEOUT", 10.0, minimum=1.0)
GEN_READ_TIMEOUT = config.env_float("GEN_READ_TIMEOUT", 1800.0, minimum=30.0)
GEN_TIMEOUT = (GEN_CONNECT_TIMEOUT, GEN_READ_TIMEOUT)

# --------------------------------------------------------------------------
# image preparation
# --------------------------------------------------------------------------

# Qwen3-VL turns roughly every 3136 pixels into one visual token, and prefill --
# the dominant cost -- scales with that. Small Thai glyphs and tone marks are the
# first thing lost when the cap is too low.
DETAIL_PRESETS = {
    "fast": 1_000_000,
    "balanced": 2_000_000,
    "accurate": 4_000_000,
    "max": 0,  # 0 = no downscaling, feed the page at native resolution
}
# Measured, not assumed: on a real 300 DPI receipt, "max" (2550x3300) sent the model
# into a counter loop and hit the 4096-token cap after 636 s, while "accurate"
# (1758x2275) finished clean in 303 s. Past ~4 MP the extra visual tokens degrade this
# model rather than helping, so accuracy-first means "accurate", not "max".
DEFAULT_DETAIL = "accurate"

# Crop blank scan margins before applying the resolution cap. Set TRIM_MARGINS=0 to
# send pages exactly as rasterised.
TRIM_MARGINS = config.env_bool("TRIM_MARGINS", True)
# How far a pixel may differ from the corner background before it counts as
# content, and how much untouched border to leave around what is found. The pad
# is why nothing gets clipped by a crop that lands a pixel tight.
TRIM_TOLERANCE = 12
TRIM_PAD = 10

# --------------------------------------------------------------------------
# sampling and prompt ordering
# --------------------------------------------------------------------------

# Put the static instruction before the image so llama.cpp can cache it across
# requests. Set false to restore the conventional image-first ordering.
#
# llama.cpp ONLY. Measured against Ollama 0.32.6 serving typhoon-ocr1.5-3b, the
# same ordering makes the model ignore the image completely and emit its own
# built-in instruction text instead of a transcript -- 228 tokens of prompt echo,
# 6.7% accuracy, on a page that reads at 90%+ with the image first. So the
# ordering is decided per backend in `stream_page`, and this flag only reaches
# llama.cpp. Set PROMPT_FIRST_OLLAMA=1 to opt Ollama in and re-measure.
PROMPT_FIRST = config.env_bool("PROMPT_FIRST", True)
PROMPT_FIRST_OLLAMA = config.env_bool("PROMPT_FIRST_OLLAMA", False)

# The system message sent to Ollama, and the reason it exists at all: the app was
# already sending one without knowing it.
#
# `app.py` builds every request as a single user message. Ollama fills the empty
# system slot from the served model's own Modelfile, and scb10x/typhoon-ocr1.5-3b
# ships `SYSTEM You are a helpful assistant.`, so that text has been part of every
# OCR request this app has ever sent to Ollama. Measured on sol005 at `balanced`:
# sending it explicitly is byte-identical to sending nothing (2854 prompt tokens
# either way), which is what makes this default a no-op rather than a change.
#
# It is also mildly load-bearing, which is why it is reproduced rather than
# dropped. Suppressing the system block entirely costs 2.42 points of mean
# character accuracy across the five fixtures (76.58 -> 74.16). The words do not
# appear to matter -- an OCR-specific persona scored identically to the generic
# one -- so what is being held here is the slot, not its content.
#
# Ollama only. llama-server has no Modelfile and is currently sent no system
# message at all; every llama.cpp baseline in CLAUDE.md was measured that way, so
# `backends.system_prefix` deliberately does not send this there.
#
# OLLAMA_SYSTEM= (explicitly empty) sends no system message, which is the shape to
# use when comparing the two backends bare -- the same reasoning as DRY_MULTIPLIER=0
# below. It is a measurably worse setting for ordinary use.
OLLAMA_SYSTEM = config.env_str("OLLAMA_SYSTEM", "You are a helpful assistant.",
                               allow_empty=True)

# Repetition control. DRY only penalises a sequence once it repeats for longer than
# DRY_ALLOWED_LENGTH tokens, so identical table cells and repeated amounts survive
# while a runaway loop gets broken. Set DRY_MULTIPLIER to 0 to disable entirely.
#
# DRY is a llama.cpp sampler. Ollama's OpenAI-compatible endpoint silently drops
# dry_*, top_k, min_p and repeat_penalty -- verified by sending repeat_penalty 5.0
# to both: /v1 returned clean output, the native /api/chat returned garbage. So
# leaving DRY on means llama.cpp runs with loop suppression that Ollama cannot
# receive, which flatters llama.cpp on exactly the documents where a small model
# loops. Set DRY_MULTIPLIER=0 to take it off both sides and compare bare.
DRY_MULTIPLIER = config.env_float("DRY_MULTIPLIER", 0.8, minimum=0.0)
DRY_ALLOWED_LENGTH = config.env_int("DRY_ALLOWED_LENGTH", 32, minimum=1)


def sampler_extras():
    """llama.cpp-only sampling controls, omitted entirely when DRY is disabled.

    Omitted rather than sent as no-ops so that with DRY_MULTIPLIER=0 both backends
    receive a byte-identical request body apart from Ollama's "model" field --
    there is then nothing left to explain a difference in the results except the
    server itself.
    """
    if DRY_MULTIPLIER <= 0:
        return {}
    return {
        # repeat_penalty stays OFF: documents legitimately repeat values (0.00,
        # currency codes, identical cells) and a flat penalty corrupts them.
        "repeat_penalty": 1.0,
        # DRY instead. It penalises *sequence* repetition, and allowed_length lets
        # short legitimate repeats through while breaking runaway loops -- which
        # greedy decoding on a 2B model is prone to.
        "dry_multiplier": DRY_MULTIPLIER,
        "dry_base": 1.75,
        "dry_allowed_length": DRY_ALLOWED_LENGTH,
        "dry_penalty_last_n": -1,
    }


# --------------------------------------------------------------------------
# loop detection
# --------------------------------------------------------------------------

# Belt and braces: even with DRY a model can lock into a loop, burning the full
# token budget and minutes of wall clock. Stop the stream when the tail is clearly
# cycling.
LOOP_TAIL_CHARS = 600
LOOP_MIN_REPEATS = 4
# Counter loops ("ปี 1 ปี 2 ปี 3 ...") never repeat exactly, so they are caught by
# normalising digit runs first. That is only applied *within a single line* with a
# short unit -- a table whose rows differ only by their numbers is legitimate, and
# its rows are newline separated, so this cannot mistake one for a loop.
LOOP_COUNTER_MIN_REPEATS = 6
LOOP_COUNTER_MAX_UNIT = 40
LOOP_COUNTER_MIN_LINE = 160
# How often the streaming read tests its own tail. Every 24 pieces rather than
# every token because the scan is O(tail^2).
LOOP_CHECK_EVERY = 24

# The same test applied to a failed extraction reply, with a much wider window:
# an extraction loop repeats a whole clause inside one JSON string, so the
# repeating unit is long and a 600-character tail cannot hold enough repeats of
# it to be recognised.
EXTRACT_LOOP_TAIL_CHARS = 1600
EXTRACT_LOOP_MIN_REPEATS = 3
# How many times one line-item description may appear before the reply is called
# a loop rather than a parse failure. A genuine document never lists the same
# description three times over.
EXTRACT_REPEAT_THRESHOLD = 3

# --------------------------------------------------------------------------
# passes 2 and 3
# --------------------------------------------------------------------------

# Second pass: feed the finished transcript back to the model as text and ask for
# structured fields. Text-only, so there is no image to prefill and it costs a
# fraction of the OCR run.
EXTRACT = config.env_bool("EXTRACT", True)
# A 2-page receipt with a dozen line items needs ~2500 tokens of JSON at eight
# sub-fields per item; 1536 cut those documents off mid-value, which surfaced as
# a JSON parse error rather than as the budget problem it was.
EXTRACT_MAX_TOKENS = config.env_int("EXTRACT_MAX_TOKENS", 4096, minimum=256)

# Which shape the second pass runs in when nothing has switched it at runtime.
# The page's Extraction button switches it per server, so this is only the mode a
# fresh process starts in.
#
# "single" asks for all 29 scalars and both lists in one request. "agentic" walks
# `prompts.EXTRACT_STEPS`, asking for one to three fields per request against the
# same transcript. Agentic costs more requests and more wall clock; what it buys
# is that a field can only be filled from the handful of labels its own step names,
# which is what stops a nearby code landing in buyer_name -- and that one step
# failing to parse costs that step's fields rather than the whole extraction.
AGENTIC_EXTRACT = config.env_bool("AGENTIC_EXTRACT", False)

# How many times a step may be asked again after it returned a value that is not
# in the transcript. The retry quotes the rejected values back, so it is a
# different question rather than the same one repeated; asking again in identical
# words returns the identical answer under greedy decoding. 0 disables the retry.
#
# One is the measured sweet spot: the document is already prefilled by then, so a
# retry costs a short question and a short answer, and a second retry almost never
# changed an answer the first had not.
AGENTIC_RETRIES = config.env_int("AGENTIC_RETRIES", 1, minimum=0, maximum=3)

# Third pass. Much smaller: the reply is a short list of issues, not a document.
VERIFY_MAX_TOKENS = 768
