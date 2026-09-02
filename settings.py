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
#
# LOOP_GUARD is the off switch, and it disables exactly one thing: ABORTING a read
# that is cycling. Detection is never disabled -- a stream that ran to the token
# cap while looping is still flagged `looped`, so a run that would have been cut
# short is still attributable and still kept out of every mean, and the extraction
# pass's "the model repeated itself and never closed the JSON" diagnosis is
# untouched. Turning it off is for reading the whole of a run the backstop was
# cutting short, and it is not free: a true runaway then costs the full
# MAX_NEW_TOKENS and the wall clock that goes with them (measured at ~8k tokens
# and 75 s on dots.ocr where the backstop stopped it at 864 tokens and 14 s).
# Switchable per process from the page as well, like the pass-1 profile.
LOOP_GUARD = config.env_bool("LOOP_GUARD", True)
LOOP_TAIL_CHARS = 600
LOOP_MIN_REPEATS = 4
# A repeated line/block that carries no actual *word* -- blank table markup
# (<tr><td></td></tr>), a pipe rule, a row of identical numbers or dashes -- is
# "low information". A run of these is what a blank or uniform region of a big
# gridded form looks like, and it is LEGITIMATE content: the count-based checks
# (whole-line, block, counter) never flag a low-information unit at any repeat
# count, because a blank table is a table, and the only thing separating a bounded
# one from a runaway is total volume, not a per-row count. Volume is the density
# backstop below. A word-bearing repeat (real prose, `ปี 1 ปี 2 ...`) is a loop at
# LOOP_MIN_REPEATS as before -- real content never repeats verbatim.
#
# Shape-agnostic backstop for a structural runaway -- a long tail carrying almost
# no word at all. This is the ONLY thing that flags a blank-row region, and it
# does so on volume: once LOOP_DEAD_TAIL_CHARS characters have gone by with
# <= LOOP_DEAD_MAX_LETTERS readable letters left after tags and backslash-escapes
# are stripped, the model is emitting structure, not reading. This catches the two
# real runaways measured here -- dots.mocr + sol003 (~185 escaped empty rows to the
# cap) and typhoon + sol009 cold at `original` (~335 empty rows on one HTML line) --
# while passing a bounded blank table. At ~29 chars per empty row the default
# tolerates ~roughly LOOP_DEAD_TAIL_CHARS/29 rows before it trips; raise it (env
# LOOP_DEAD_TAIL_CHARS) if a real form's blank grid is larger, at the cost of more
# wasted tokens on a true runaway. It is deliberately letter-based, so a single
# 2000-char stretch of pure digits with no label could false-trip; the fixtures
# here all carry Thai labels in their rows, so none does.
LOOP_DEAD_TAIL_CHARS = config.env_int("LOOP_DEAD_TAIL_CHARS", 3000, minimum=400)
LOOP_DEAD_MAX_LETTERS = 5
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

# Pass 0: which KIND of document this is. The answer chooses the field set both
# pass-2 shapes are then asked for, and it frames both prompts -- see
# `prompts.type_block`.
#
# **The MODEL is asked FIRST, on every document, since 2026-09-02** (at the
# user's request: *it need to let llm check first always since sometimes its new
# doc or random doc*). The benchmark manifest and the printed-heading table
# (`normalise.DOCUMENT_TYPES`) are what catch it when its answer cannot be
# believed, rather than what stop it being asked -- so a fixture exercises the
# same path a real upload takes, which it did not before.
#
# `app._classify_with_model` throws the answer away unless the heading it quotes
# is really printed on the page; then the manifest answers, then the table. A
# disagreement with the manifest is REPORTED on the result as
# `doc_type_expected` rather than resolved.
#
# Off sends the pre-2026-09-02 order -- manifest, then the heading table, and the
# default (widest) form for anything neither places. It is a comparison knob like
# `DRY_MULTIPLIER=0`, not a preference, and it is also how to make a sweep stop
# paying a classify request per run.
#
# **Measured on all ten fixtures with `gemma4:e4b`: 10/10 get the same FORM the
# manifest would have given**, so no pass-2 number in CLAUDE.md moves. Only
# sol001's label differs, and the type it drops has no requirement behind it.
CLASSIFY_WITH_MODEL = config.env_bool("CLASSIFY_WITH_MODEL", True)
# How sure the PYTHON classifier has to be before the model is not asked at all
# (2026-09-02, at the user's request: *use code to classify but if its confidence
# is more than 90% ... if not, let llm do it*).
#
# The figure is `normalise.heading_confidence`: how much of the line the answer
# was read off is actually the heading, discounted a little further down the
# page. **It measures the evidence on the page, not anybody's self-belief**,
# which is what lets the same function score the model's quoted heading and put
# the two answers on one scale.
#
# 0.9 is the user's number and it is a threshold rather than a measurement --
# but it is not arbitrary against this corpus: every real heading here scores
# 1.00 (a line that is only the heading, `ใบเสร็จรับเงิน/ใบกำกับภาษี` included,
# whose two needles cover all of it) and a body-text mention of a document kind
# scores 0.23-0.30. There is nothing between 0.30 and 1.00 to be sensitive to.
#
# 0 asks the model never (the table always wins where it matched anything);
# above 1.0 asks it always.
CLASSIFY_MIN_CONFIDENCE = config.env_float("CLASSIFY_MIN_CONFIDENCE", 0.9,
                                           minimum=0.0, maximum=1.01)
# Send a MULTI-TYPE answer from the table to the model, however confident it is
# (2026-09-02, at the user's request: *if keyword detected multiple type let llm
# check/decide, since sometimes the file only mentions another doc type as a
# reference and code may detect that as a doc type, which is wrong*).
#
# **The risk is specific to a needle match and coverage does not see it.** A
# second type costs a handful of characters on a line the first type already
# fills, so `ใบลดหนี้ อ้างอิงใบกำกับภาษี 123` scores like a heading and comes
# back as two types, one of which is a reference to a different document. One
# wrong type widens the form, adds validation rules the document is not held to,
# and frames both prompts with a lie about what it is reading.
#
# The cost is that a LEGITIMATE slash heading escalates too --
# `ใบเสร็จรับเงิน/ใบกำกับภาษี` really is both, and six of the ten fixtures are
# multi-type -- so this trades most of the gate's speed for the certainty. On
# this corpus it changes no answer: measured, the model returns the same types.
CLASSIFY_ESCALATE_MULTI = config.env_bool("CLASSIFY_ESCALATE_MULTI", True)
# Tell the pass-2 prompts what the document is. Off sends the prompt that was
# sent before 2026-09-01 -- byte for byte, which `prompts._selftest` asserts --
# while leaving the FORM exactly as this document's type chose it. That is the
# whole point of the switch and the only honest way to measure the framing: an
# arm that also changed which keys were asked for would be measuring two things.
# It is a comparison knob, like DRY_MULTIPLIER=0, not a preference.
# One short JSON object -- a heading and a code or two. The cap is small on
# purpose: a reply long enough to overrun it is one that started transcribing the
# page, which is this project's oldest pass-2 failure and is not an answer worth
# waiting for.
# Whether the pass-2 prompts are TOLD the type. Off sends the prompt that was
# sent before 2026-09-01 -- byte for byte, which `prompts._selftest` asserts --
# while leaving the FORM exactly as the type chose it. Which is the only honest
# way to measure this: an arm that also changed the keys would move two things.
#
# **Measured per mode, because the two modes measured differently** -- CLAUDE.md,
# 250 runs over ten fixtures and five models. Single mode gains 4.9 points of
# accuracy and 4.7 of precision, and on one model it recovers two whole credit
# notes that had been coming back as an empty skeleton.
#
# Agentic is the close call, and it is ON at the user's decision (2026-09-01)
# rather than by the mean: the five-model mean is -0.54 and every point of that
# is gemma4:e2b's -4.5, while THREE of five improve -- including both of the two
# best models, which gain 1.2 each. The project's best pass-2 number, qwen3.5:9b
# agentic at 92.0, is only reachable with it. A setting that is going to be
# adopted is adopted on one model, not on the mean of five.
#
# **Turn this off when running gemma4:e2b or qwen3.5:4b in agentic mode** -- they
# are the two that lose, by 4.5 and 1.2.
TYPE_FRAMING_SINGLE = config.env_bool("TYPE_FRAMING_SINGLE", True)
TYPE_FRAMING_AGENTIC = config.env_bool("TYPE_FRAMING_AGENTIC", True)
# The framing in two halves: naming the document, and the per-key rules that
# come with it. Off names the type and rules nothing. Kept as a knob because the
# halves measured very differently -- in single mode the rules are the whole of
# the gain and naming alone is WORSE than saying nothing, while in agentic the
# naming is what costs and the rules are nearly free.
TYPE_FRAMING_BULLETS = config.env_bool("TYPE_FRAMING_BULLETS", True)
CLASSIFY_MAX_TOKENS = config.env_int("CLASSIFY_MAX_TOKENS", 160, minimum=32)
# How much of the transcript the question carries. A heading is at the top of the
# page or a few inches down it -- sol002's is eighteen lines in -- and sending a
# ten-page document to ask one question about its first inch is prefill spent on
# nothing.
CLASSIFY_MAX_CHARS = config.env_int("CLASSIFY_MAX_CHARS", 4000, minimum=200)

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

# --------------------------------------------------------------------------
# Classifying a read: where it ran, and whether it was warm
# --------------------------------------------------------------------------
#
# The log records no hardware and no warm/cold flag -- the app talks HTTP to a
# server it did not launch, so it cannot know how many layers are on the GPU or
# whether the model was resident. Both are INFERRED from measurements already in
# the row, which is a heuristic and is labelled as one everywhere it shows.
#
# **GPU vs CPU, from decode tok/s.** A GPU decodes this workload at tens of
# tokens a second (an RTX 3060 laptop runs ~18-150 here depending on the model);
# CPU-only llama.cpp on a 2-4B vision model is single digits. A read whose
# `tokens_per_second` is at or above this is called `gpu`, below it `cpu`. The
# default sits well under every GPU figure this log has ever held and well above
# any plausible CPU one, so it separates the two without a run to tune it on --
# but it is a proxy, not a probe, and a very small model on a slow GPU could dip
# under it.
GPU_MIN_TPS = config.env_float("GPU_MIN_TPS", 15.0, minimum=0.0)

# **Warm vs cold, from prefill.** A warm read reuses state -- llama.cpp's KV
# cache for a repeated prefix, or a model already resident -- and pays almost no
# prefill (a cached page 2 measured 0.07 s against 33 s cold). A read whose
# `prefill_seconds` is below this is `hot`, at or above it `cold`. It is an
# absolute threshold rather than one relative to the setting, so it is a per-row
# fact the filters can use: cold prefill is seconds even at the lowest Detail,
# and a sub-second prefill is a cache hit whatever read it. A read with no
# prefill figure (Ollama sometimes omits it, and re-extractions read no page) is
# neither -- it is left blank, not guessed.
WARM_PREFILL_MAX = config.env_float("WARM_PREFILL_MAX", 1.0, minimum=0.0)

# **Outlier fence for the time analysis.** A cold model load or a runaway loop
# puts a read's time far above the rest of its (model, document) cell -- 1489 s
# against a 30 s median in this log -- and one such point drags a correlation or
# a mean on its own. The analysis tab drops a read whose time is more than this
# many median-absolute-deviations from its cell's median (a robust fence: MAD is
# not itself moved by the outlier it is measuring). Applied only where a cell has
# enough runs to have a shape (`ANALYSIS_OUTLIER_MIN_RUNS`); 0 disables it.
ANALYSIS_OUTLIER_MADS = config.env_float("ANALYSIS_OUTLIER_MADS", 3.5, minimum=0.0)
# **3, not 4, and only because the ratio test above backs it up.** At n=3 the
# median is a real middle value and the fence works well -- it is what catches
# [85, 87, 1028]. The danger of a small cell is that a tight cluster makes the
# MAD tiny and ordinary jitter look extreme, and that is exactly what the ratio
# floor refuses: [17.2, 17.4, 30] is 42 MADs out and only 1.7x, so it stays.
# n=2 cannot be tested at all and is not a threshold choice -- the median sits
# between the two points, so neither is far from it however different they are.
# Those reads are counted and reported as untested rather than silently kept.
ANALYSIS_OUTLIER_MIN_RUNS = config.env_int("ANALYSIS_OUTLIER_MIN_RUNS", 3, minimum=3)

# **And a read must also be this many times off the median before it counts**,
# which is what stops the MAD fence firing on ordinary jitter. MAD is
# scale-free: a cell whose reads cluster tightly (17.2, 17.4, 17.5, 17.6) has a
# MAD of ~0.2 s, so a perfectly ordinary 24 s read is "15 MADs out" and would be
# dropped. Timing data does not work that way -- a few seconds of wall clock is
# noise, not a finding, however reproducible the rest of the cell was.
#
# So both tests must pass: the shape test above, and this ratio. Measured here
# it separates the two populations cleanly -- the genuine runaways are 8x to 16x
# their cell median (1411 s against 141 s, 562 s against 35 s) while every false
# positive was between 0.67x and 1.4x. Two-sided, because a read far FASTER than
# its cell is equally not representative: a cached prefix or a truncated reply.
ANALYSIS_OUTLIER_MIN_RATIO = config.env_float("ANALYSIS_OUTLIER_MIN_RATIO", 2.0,
                                              minimum=1.0)



# --------------------------------------------------------------------------
# The presentation summary's bias (2026-08-25, at the user's request: *try be
# bias by default in this page -- exclude bad product, outlier, single error
# case and some old error case*).
#
# **This is the one view in the project that is allowed to be selective, and it
# is selective by a STATED RULE over the log rather than by a list of model
# names.** A hard-coded "do not show dots.ocr" would be a claim this file makes
# about a model; a threshold is a claim the log makes, and it moves when the
# model does. Everything it drops is named on the page and one toggle puts it
# all back, which is what keeps a biased view honest -- the bias is the
# argument, not a hidden edit.
#
# The figure compared is `runlog._standout_score`: accuracy x (1 - failure
# rate), i.e. what one attempt is worth -- and it is computed over the SAME
# grouping the summary prints, so the number that disqualifies a model is the
# number on the row beside it. Judging it on a differently windowed figure was
# the first shape and it was wrong in the way that is hardest to argue with: a
# model showing 41% failure in the table was being kept by a rule that had
# seen 39%.
#
# Measured on the 510-row log the day this shipped, the two rules leave exactly
# the two OCR builds in the reading tables (typhoon 78, dots.mocr 71 per
# attempt) and exactly the three general models in the extraction ones
# (qwen3.5:4b 55, qwen3.5:2b 47, phi4-mini 43) -- which is the project's own
# conclusion in CLAUDE.md, reached here from the rows instead of asserted.
# **Relative to the best model in that pass, not an absolute cut.** An absolute
# threshold has to be right for both passes at once and there is no such number
# here: a 43-per-attempt reader is a poor product beside an 78, while a
# 43-per-attempt extractor is the third best thing this project owns. A share of
# the leader says the thing that is actually meant -- *this delivers less than
# 60% of what the best available model delivers, so it is not what you would
# ship* -- and it re-scales on its own when a better model arrives, which is
# exactly when an absolute threshold would start hiding the wrong rows.
PRESENT_MIN_SHARE = config.env_float("PRESENT_MIN_SHARE", 0.6,
                                     minimum=0.0, maximum=1.0)
# **A failure rate this high disqualifies whatever the score is**, and it is
# absolute because reliability is not graded on a curve. dots.ocr reads 79.9% on
# the runs that finish and does not finish 41% of them; a summary that ranked it
# on the survivors would recommend a build this project has already dropped.
# Kept separate from the share above rather than folded into it, because the two
# say different things to a reader -- one is "not good enough", the other "not
# reliable enough", and a product can fail either alone.
PRESENT_MAX_FAILURE = config.env_float("PRESENT_MAX_FAILURE", 40.0, minimum=0.0)
# **Thin evidence never disqualifies.** A model with one or two runs is kept
# whatever it scored -- dropping it would be the same smear `STANDOUT_MIN_RUNS`
# refuses one level down, and a model newly pulled would vanish from the summary
# before it had a chance to be measured. So the bias only ever acts on a model
# the log has something to say about.
PRESENT_MIN_RUNS = config.env_int("PRESENT_MIN_RUNS", 3, minimum=1)
