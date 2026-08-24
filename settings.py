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
# document at `medium` is ~40 MB of PNG. Lower it on a small server.
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
# matters: a CPU-only server reading a dense page at `original` detail can legitimately
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
    "original": 0,        # 0 = no downscaling, feed the page at native resolution
    "medium": 4_000_000,
    "low": 2_000_000,
}
# Measured, not assumed: on a real 300 DPI receipt, native resolution (2550x3300)
# sent the model into a counter loop and hit the 4096-token cap after 636 s, while
# 4 MP (1758x2275) finished clean in 303 s. Past ~4 MP the extra visual tokens
# degrade this model rather than helping, so accuracy-first means `medium`, not
# `original`.
DEFAULT_DETAIL = "medium"

# **Three presets, and the old four-name vocabulary is gone** (renamed
# 2026-08-21, the old names dropped 2026-08-24, both at the user's request:
# *i only need 3 set (original/medium/low)*, then *drop the accurate/fast/max*).
# The renames kept every measured pixel budget:
#
#   original = the old `max`      -- uncapped, unchanged
#   medium   = the old `accurate` -- 4 MP, unchanged, and still the default
#   low      = the old `balanced` -- 2 MP, unchanged
#
# **The old `fast` (1 MP) was deleted rather than renamed, and it is the one
# preset that really went away.** At 1 MP this model stops misreading and starts
# INVENTING: on sol002 it fabricated an address that is not on the page (91.8%
# against 97.6% at 2 MP). A garbled word is a visible failure; a plausible
# invented one is not, and a preset whose failure mode is fabrication is not a
# preset to offer as "quick". Anyone who wants it back adds one line to
# `DETAIL_PRESETS` -- the pixel budget is the whole of what a preset is.
#
# **An alias table stood here for three days and is deliberately not replaced.**
# It accepted the old names everywhere a Detail arrives, which kept saved page
# state and `compare.py --detail accurate` working -- and it also kept three of
# them reachable from every request this app takes, so a script could go on
# asking for a vocabulary the picker no longer offers and nothing would say so.
# `resolve_detail` now falls back to the default for anything it does not know,
# which is what it already did for a typo. The one place the old names still
# have to be understood is the RUN LOG, which is full of them and cannot be
# rewritten -- that reading lives in `runlog.DETAIL_RENAMES`, next to the tables
# that need it and nowhere else.


def resolve_detail(name: str) -> str:
    """A Detail as this build spells it: the preset itself, or the default.

    One function rather than a membership test at each call site, because there
    are four of them and they must not disagree about what an unknown name
    means. Unknown is not an error: a Detail arrives from a form field, a saved
    browser setting and two CLIs, and falling back beats refusing a read over a
    stale dropdown. What it costs is that a request for `fast` silently reads at
    4 MP rather than the 1 MP it asked for -- correct, since 1 MP is gone, and
    the run log records `medium`, so it never claims a budget it did not use.
    """
    name = (name or "").strip().lower()
    return name if name in DETAIL_PRESETS else DEFAULT_DETAIL

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
# OCR request this app has ever sent to Ollama. Measured on sol005 at `low` (2 MP,
# then called `balanced` -- same budget, renamed):
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

# Ollama only. A reasoning model answers in two parts -- a chain of thought and
# then the answer -- and Ollama returns the first in its own field, leaving
# `content` EMPTY until the thinking finishes. Every request this app makes is
# capped (an agentic step at 120-900 tokens, an OCR page at MAX_NEW_TOKENS), so
# such a model spends the whole cap thinking and returns nothing at all. Measured
# 2026-08-19 on `qwen3.5:4b`: all seven agentic steps failed on all five
# fixtures with `Expecting value: line 1 column 1 (char 0)`, which is what an
# empty string looks like to a JSON parser. Raising the caps is not the fix --
# it buys more thinking, the same way a bigger cap buys more page from a model
# that is echoing.
#
# "none" turns thinking off through the OpenAI-compatible endpoint. It is SAFE ON
# A MODEL THAT DOES NOT THINK: sent to typhoon-ocr1.5-3b beside an otherwise
# identical body, the reply was byte-identical (same 361 chars, same md5, same
# 2360 prompt tokens), so no baseline in CLAUDE.md moves because of it.
#
# Set it to "low"/"medium"/"high" to let a reasoning model think -- and then raise
# EXTRACT_MAX_TOKENS and the step caps in `prompts.EXTRACT_STEPS` to pay for it,
# because the budget is shared between the thinking and the answer.
# OLLAMA_REASONING_EFFORT= (explicitly empty) sends nothing, which is the shape
# for a server that rejects the field.
OLLAMA_REASONING_EFFORT = config.env_str("OLLAMA_REASONING_EFFORT", "none",
                                         allow_empty=True)

# Stop the model that was in use when the picker switches to another one, so the
# new model loads onto a card the old one has let go of. Ollama only, and the
# asymmetry is the whole reason this setting exists: llama-server holds exactly
# one model for the life of its process, while Ollama keeps every model it has
# served resident for its keep_alive (5 minutes by default) and loads the next
# one beside it. Two 3B models at F16 is the difference between a run that fits
# on a 6 GB card and one that spills to system RAM -- and a spilled run is not an
# error, it is the same read at a fraction of the speed.
#
# What it sends is what `ollama stop <model>` sends: an empty generation with
# keep_alive 0. See `backends.free_gpu`, which also owns the part worth knowing
# before turning this on for a shared server -- it stops every model resident at
# the endpoint, including ones this app never loaded, because the model still
# holding the card is often not the one this process last selected.
#
# It never runs while the queue has a job going (`app.py` passes the veto):
# eviction goes to the same scheduler that is serving the run in flight, so a
# switch made mid-batch would be paid for by the document being read.
OLLAMA_UNLOAD_ON_SWITCH = config.env_bool("OLLAMA_UNLOAD_ON_SWITCH", True)

# Which pass-1 shape to start in: a key of `prompts.OCR_PROFILES`. Not validated
# here -- this module deliberately imports nothing but `config` and `jobs`, so
# `app.py` checks the name against the table and falls back with a warning, the
# same way the env readers above degrade one setting instead of killing startup.
#
# `typhoon` is the default and every pass-1 baseline in CLAUDE.md was measured on
# it. `dots` exists because a second model needed a different prompt AND no
# system message at once: sending either of typhoon's to dots.ocr returns two
# tokens and an empty string at HTTP 200. A profile is what makes those two move
# together, so a half-switched request cannot be built.
#
# The profile also decides whether OLLAMA_SYSTEM above is sent at all. Where they
# disagree the profile wins, because it is the narrower statement -- OLLAMA_SYSTEM
# says what to send when a system message is wanted, not that one always is.
OCR_PROFILE = config.env_str("OCR_PROFILE", "typhoon")

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
# pass 2
# --------------------------------------------------------------------------

# Second pass: feed the finished transcript back to the model as text and ask for
# structured fields. Text-only, so there is no image to prefill and it costs a
# fraction of the OCR run.
EXTRACT = config.env_bool("EXTRACT", True)
# A 2-page receipt with a dozen line items needs ~2500 tokens of JSON at eight
# sub-fields per item; 1536 cut those documents off mid-value, which surfaced as
# a JSON parse error rather than as the budget problem it was.
EXTRACT_MAX_TOKENS = config.env_int("EXTRACT_MAX_TOKENS", 4096, minimum=256)

# Constrain the reply to the field schema instead of merely to "valid JSON".
# On by default because it is what stops the OCR fine-tune answering pass 2 with
# its own transcript envelope: the key names are in the grammar, so
# {"natural_text": ...} is not a reachable answer. Sent as `format` to Ollama's
# native endpoint and as `response_format.json_schema` to llama-server -- see
# `backends.structured_request`. Set EXTRACT_SCHEMA=0 to go back to
# {"type": "json_object"}, which is how every measurement taken before
# 2026-08-17 was run.
EXTRACT_SCHEMA = config.env_bool("EXTRACT_SCHEMA", True)

# Repetition control for Ollama, and the ONE place this project sets
# repeat_penalty above 1.0. Read it together with DRY_MULTIPLIER above, which
# explains why that is normally the wrong lever.
#
# The short version: DRY is a llama.cpp sampler, and Ollama's /v1 shim drops
# dry_*, repeat_penalty, top_k and min_p alike, so pass 2 on Ollama ran with no
# repetition defence at all. Measured on sol005, pass 2 only, schema-constrained
# on the native endpoint, everything else identical:
#
#   repeat_penalty 1.0  -- payment_reference degenerated into 1111111111... and
#                          the JSON never closed
#   repeat_penalty 1.1  -- parsed, 31 keys, 6 rows, 94.9% of values grounded
#
# A grammar cannot stop repetition *inside* a string value; only the sampler
# can. This is why the schema alone does not fix the loop.
#
# It is not free, and the cost is the one DRY exists to avoid: on the same run a
# quantity came back as 1.731,118.40, a legitimately repeated thousands
# separator turned into a decimal point. Set OLLAMA_REPEAT_PENALTY=1.0 to take
# it off and get the corruption-free, loop-prone behaviour back.
OLLAMA_REPEAT_PENALTY = config.env_float("OLLAMA_REPEAT_PENALTY", 1.1,
                                         minimum=1.0)

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

# The lowest character accuracy a transcript may score and still have the fields
# extracted from it SCORED. Below it -- and on a read that looped, was cut off,
# or returned nothing -- extraction still runs, and its field score is dropped
# instead of written.
#
# **A read below it is not a failed run.** It is a poor one, it counts as a run,
# and it counts in the pass-1 accuracy mean that says so. Failure means a loop or
# a crash -- see `runlog._incomplete`. What this decides is only whether the
# PASS-2 figure taken over that transcript is worth writing down.
#
# **This is about what a number means, not about saving work.** Pass 2 can only
# map values that pass 1 actually produced, so a field score taken over a broken
# transcript is a measurement of the read wearing the extractor's name: it drags
# down the setting that extracted and lets the setting that read get away with
# it. The gap is real and is recorded in CLAUDE.md -- dots.mocr agentic scored
# 32.7% over its own reads against 46.8% over the ground truth, and on the one
# document it read at 48.3% it returned half the fields it returned from truth.
#
# **0.75 since 2026-08-24**, the user's figure both times (0.5 when the rule was
# built three days earlier). Neither is measured; what is measured is the effect
# the rule exists for -- dots.mocr agentic scored 32.7% over its own reads
# against 46.8% over the ground truth, and on the one document it read at 48.3%
# it returned half the fields it returned from truth. Raising the bar to 75%
# suppresses more scores, which is the point: a transcript three-quarters right
# still hands pass 2 wrong values to map, and the score that comes back is the
# read's mistake wearing the extractor's name.
#
# Set MIN_READ_FOR_FIELDS=0 to score every extraction whatever the read did,
# which is how every row written before 2026-08-21 was recorded.
#
# It applies where BOTH passes run against a document whose transcript truth is
# known -- the random test's `full` scope. A read with no ground truth cannot be
# judged this way and is never suppressed on a guess.
MIN_READ_FOR_FIELDS = config.env_float("MIN_READ_FOR_FIELDS", 0.75,
                                       minimum=0.0, maximum=1.0)

# --------------------------------------------------------------------------
# run log
# --------------------------------------------------------------------------

# How many of the most recent runs **of each setting, model and document** every
# COMPILED figure is taken over -- the ranking tables, the per-document bests,
# the means, the standouts. **Not a slice off the end of the file**: it is applied
# per group by `runlog.recent_by`, with the key each table groups on, so a
# setting is described by its own last twenty runs and one busy evening on
# another setting cannot push it out of the table. The raw rows are all still in
# the CSV; what this bounds is what gets averaged.
#
# **Set at the user's request, 2026-08-21**: *only get 20 latest run only since
# some old run maybe on the old system*. The log is append-only across changes to
# the thing being measured, and this project changes it constantly -- a scorer
# that started aligning blocks, a schema that grew fifteen keys, four Detail
# presets that became three. Rows written on either side of one of those are not
# two samples of anything, and a mean over them describes a build nobody is
# running. Two of the three log resets in CLAUDE.md were that problem being fixed
# by hand; this is the standing version of it.
#
# It is rows, not reads: a re-extraction is a run and takes a place in its
# group's window. 0 means the whole log, which is how every figure before this
# date was taken.
#
# **`runlog.case_counts` deliberately ignores it.** The random test's fairness
# rule needs the whole history -- a document read thirty rows ago has been read,
# and windowing that would send every round back to the same few fixtures.
SUMMARY_RUNS = config.env_int("SUMMARY_RUNS", 50, minimum=0)
