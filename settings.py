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

# --------------------------------------------------------------------------
# intake limits
# --------------------------------------------------------------------------

# All four are per-request costs paid by the model server, so a deployment
# sharing one server between several users will want them lower than these
# single-user defaults.
MAX_UPLOAD_MB = config.env_int("MAX_UPLOAD_MB", 32, minimum=1, maximum=512)
# **0 means no cap: read every page of the file** (2026-09-16, at the user's
# request -- *no cap page*). A positive value caps and the rest of the file is
# DROPPED, which the page and the run log now say out loud rather than silently
# -- see `app.load_pages`.
#
# What bounds a long file now is `MAX_UPLOAD_MB` above and the machine: nothing
# here is streamed, so `load_pages` holds every page at `PDF_DPI` in one list
# (~26 MB per A4 page at 300 DPI, so ~2.4 GB for 100 pages) and `prepare_input`
# builds the downscaled copies beside it -- ~3.6 GB peak at `medium` for 100
# pages, measured. Lower `Detail` and `MAX_JOBS` are the levers, in that order.
MAX_PAGES = config.env_int("MAX_PAGES", 0, minimum=0)
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

# Both backends, despite the name (llama-server gets it as
# `chat_template_kwargs.enable_thinking=false` since 2026-09-23, when a
# qwen3.6-35b-a3b there returned empty content on every capped request -- the
# same failure as below). A reasoning model answers in two parts -- a chain of thought and
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


# --------------------------------------------------------------------------
# Starting on something that works (2026-09-03, at the user's request: *auto
# default model to the best one that is shown, and auto check and select the
# server that is in the history/constant and is online*).
#
# Both of these change a DEFAULT and nothing else. An explicit choice -- the
# page's pickers, `compare.py --server/--model`, a locked random-test round --
# still wins in every case, so nothing that was reproducible before stops being
# reproducible. What changes is what a fresh process does when nobody has said.

# Probe the configured endpoints and the ones the run log has runs against, and
# make the first one that answers active.
#
# The old default was `_endpoints[0]`, which is llama-server on :8080 -- and on
# the machine this was written on that port has served 2 runs of 1298 while
# Ollama on :11434 served the other 1296. So the app started, every render, and
# every request pointed at a dead port until somebody moved the picker, and a
# dead port is not cheap: it costs a full connect timeout, twice (llama, then
# Ollama), on every probe.
#
# The cost is paid once, at startup, in `preflight` -- never on a polling path,
# which is the rule this project already has about the model server. It stops at
# the first endpoint that answers, so the usual case is one probe.
AUTO_SELECT_SERVER = config.env_bool("AUTO_SELECT_SERVER", True)

# A SECOND endpoint for pass 2 (2026-10-01, at the user's request: *connect to 2
# URLs, one for OCR and another for extraction, via the OpenAI-compatible API, and
# keep the old one-URL setup for local / Ollama*). Unset -- the default -- is that
# one-URL setup: both passes go to the active endpoint, exactly as every
# measurement in CLAUDE.md was taken. Set, pass 2 (extraction, the classify and
# segment questions, the table agents) goes here instead, while the page images
# keep going to the reading endpoint. Any server kind this app can probe works on
# either side -- vLLM, llama-server, Ollama or a generic OpenAI-compatible one.
# Switchable at runtime from the page's "Extraction server" picker.
EXTRACT_URL = config.env_str("EXTRACT_URL", "", allow_empty=True)
# Which inference engine each server is, pinned (2026-10-05, at the user's
# request: *pick the server engine so there is no 404 request*). Comma
# separated `url=kind`, kind one of llama.cpp / ollama / vllm / openai, e.g.
#   SERVER_KINDS=http://gpu-box:8000=vllm,http://127.0.0.1:11434=ollama
# A pinned server is spoken to as that engine ONLY: it is probed with that
# engine's endpoint alone (`/props`, `/api/tags` or `/v1/models`), and nothing
# another engine has (`/api/ps`, `/metrics`) is ever requested of it. A server
# not named here is detected ("auto"), which asks /props, then /api/tags, then
# /v1/models -- the 404s a vLLM or Ollama server sees in its log. Also settable
# per server from the page's "Server type" picker.
SERVER_KINDS = config.env_str("SERVER_KINDS", "", allow_empty=True)
# Server-side timing (2026-10-05, at the user's request; narrowed 2026-10-07:
# *report the time for prefill and decode and server latency separately -- I
# will add them myself; I want the raw GPU speed*). Every model request is also
# timed by the SERVER, and its own figures are recorded as they came, one per
# phase -- `server_prefill_seconds`, `server_decode_seconds`,
# `server_queue_seconds` and `server_seconds` (its end-to-end latency). Nothing
# is derived from the app's clock. Sources:
#   vLLM       /metrics, read before and after the request -- the change in
#              `vllm:request_{prefill,decode,queue}_time_seconds` and
#              `vllm:e2e_request_latency_seconds`. Two extra GETs per request,
#              vLLM only.
#   llama.cpp  the `timings` block it already sends (prompt_ms, predicted_ms;
#              no queue figure, and latency = prefill + decode).
#   Ollama     prompt_eval / eval / total_duration on the native /api/chat
#              reply; /v1 sends nothing, so a read on Ollama has no server time.
# Under concurrency a /metrics difference covers more than one request and
# cannot be attributed, so it is left blank rather than guessed. 0 asks nothing
# extra of any server.
SERVER_TIMING = config.env_bool("SERVER_TIMING", True)
# How long to wait for vLLM to record a request that has just finished before
# calling it unattributed. The stream can close a moment before the server's
# stats logger has run, so the after-snapshot is retried within this budget.
SERVER_TIMING_WAIT = config.env_float("SERVER_TIMING_WAIT", 1.0, minimum=0.0)

# The live /metrics monitor (2026-10-06). vLLM does not report VRAM, so the
# monitor reads it from a GPU exporter scraped beside it -- DCGM
# (DCGM_FI_DEV_FB_*) or nvidia_gpu_exporter (nvidia_smi_memory_*_bytes). Blank
# falls back to series the vLLM scrape itself carries, then to this machine's
# nvidia-smi when the model server is on this machine. The page can override it.
MONITOR_GPU_URL = config.env_str("MONITOR_GPU_URL", "", allow_empty=True)
# The shortest gap between two /metrics scrapes the monitor will make of one
# server. A call inside it gets the last reading back (`cached`), so every tab
# and every client watching costs one scrape per gap between them, and the
# page never asks more often than this however its box is set. vLLM's
# counters move per engine step, so a reading more often than this is mostly
# the same figures again.
MONITOR_MIN_INTERVAL = config.env_float("MONITOR_MIN_INTERVAL", 2.0, minimum=0.0, maximum=60.0)
# While the server is idle (nothing running or waiting, no tokens since the
# last reading) the page doubles its gap after each idle reading, up to this
# many seconds, and drops back to the box's value at the first sign of work.
# A failed reading backs off the same way. 0 turns the backoff off.
MONITOR_IDLE_MAX = config.env_float("MONITOR_IDLE_MAX", 30.0, minimum=0.0, maximum=600.0)
# How many candidates a single auto-select is willing to probe. Each dead one is
# ~3 s of connect timeouts, and the list is the constants plus every server in
# the log, which grows without bound on a machine that has moved endpoints
# around. Bounded so a long history cannot turn startup into a minute of waiting
# for ports nobody is running any more.
AUTO_SELECT_MAX_CANDIDATES = config.env_int("AUTO_SELECT_MAX_CANDIDATES", 8,
                                            minimum=1)

# Default each pass's model to the one the run log RANKS FIRST for that pass,
# among the models the active endpoint actually serves.
#
# It replaces a substring test on the model's name as the pass-1 default (an
# `ocr` in the name), and it replaces "the reading model" as the pass-2 one. The
# name test was never a claim about quality -- it was a guard against a fresh
# process resolving pass 1 onto whatever had been pulled most recently -- and
# `runlog.best_models` guards the same thing with evidence instead: the same
# ranking the Summary tab's headline cards print. It is still a fallback below
# an explicit choice, and the name test is still a fallback below IT, for a
# model the log has never scored.
#
# **The two passes rank differently and that is the point.** On the log this was
# written against, pass 1 ranks typhoon first (86.6%, 1 failure in 50) where the
# name test picked dots.mocr (85.7%, 8 failures) -- within a point on the runs
# that finish, separated by the failures, which is what a name cannot see. Pass 2
# ranks qwen3.5:9b first (90.4%) with typhoon seventh of nine at 38.8%; until now
# a one-model setup ran the second pass on the first pass's winner, which this
# project's own sweeps say is the wrong tool for the form.
#
# One thing to know before leaving it on: the ranking is accuracy x (1 - failure
# rate) and says nothing about cost. qwen3.5:9b is 6.6 GB, spills off a 6 GB
# card, and takes 53-81 s an extraction against gemma4:e4b's 14-35 s for about
# two points more. Pick the model by hand where that trade matters; the page's
# picker says which model is auto-selected and why.
AUTO_BEST_MODEL = config.env_bool("AUTO_BEST_MODEL", True)

# Which pass-1 shape to start in: a key of `prompts.OCR_PROFILES`. Not validated
# here -- this module deliberately imports nothing but `config`, so
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


# How many tokens of the tail DRY scans for repeats. 0 means "the whole context
# window", resolved from the window the request itself is being sent with, so the
# two cannot disagree after a Context change -- the same rule as never re-reading
# an env var in a second module.
#
# This was hard-coded -1, llama.cpp's documented "= context size", and every
# llama.cpp baseline in CLAUDE.md was measured under it. A 2026 llama-server
# build (b1-67a17c1) validates the field as 0 <= n <= INT_MAX and rejects -1 with
# HTTP 400 on EVERY request -- the app was unusable against it. Sending the window
# as a number is the same sampler on both builds; verified against that server,
# which accepts 0, 8192 and 32768 and refuses -1.
#
# Do NOT let this fall through to the server's own default: that build defaults
# it to 64, so an omitted field is a different sampler rather than the measured
# one. A positive value here overrides the window, and 0 in llama.cpp's own
# vocabulary (disable DRY) is DRY_MULTIPLIER=0 here.
DRY_PENALTY_LAST_N = config.env_int("DRY_PENALTY_LAST_N", 0, minimum=0)

# Used when the caller does not know the window -- it is always passed in from
# backends.num_ctx() on both request paths, so this is a floor rather than a
# setting anyone is expected to meet.
DRY_PENALTY_FALLBACK = 8192

# The flat repetition penalty, sent on every OpenAI-compatible request (every
# page read, every plain extraction request, every step) under BOTH names it
# goes by: `repeat_penalty` is llama.cpp's, `repetition_penalty` is vLLM's (and
# most other OpenAI-compatible servers'). A server ignores the name it does not
# know. Ollama's /v1 shim drops both -- its one penalty is the native-endpoint
# OLLAMA_REPEAT_PENALTY on the constrained retry.
#
# 1.1 by default, set at the user's request on 2026-10-05. It was 1.0 (off)
# until then, and EVERY llama.cpp figure in CLAUDE.md was measured at 1.0, so a
# number taken under 1.1 is not comparable with them. The known cost: documents
# legitimately repeat values (0.00, currency codes, identical cells) and a flat
# penalty can corrupt them -- on sol005 1.1 turned a repeated thousands separator
# into a decimal point (1.731,118.40). REPETITION_PENALTY=1.0 turns it off,
# leaving DRY above as the only loop defence.
REPETITION_PENALTY = config.env_float("REPETITION_PENALTY", 1.1, minimum=1.0)

# vLLM gets the penalties OLLAMA ACTUALLY APPLIES to the same model, rather than
# the ones this app sends -- set 2026-10-06 at the user's request, after the vLLM
# server looped where the same model on Ollama did not.
#
# The two were never running the same sampler. Ollama's /v1 drops every penalty
# in the request (measured: repeat_penalty 5.0 changed nothing there), so what it
# runs is the served model's Modelfile, or failing that its own default --
# repeat_penalty 1.1 over the last 64 tokens. vLLM honours the request as sent.
# Read off the Modelfiles on this machine (~/.ollama/models, 2026-10-06):
#
#   typhoon-ocr1.5-3b   repeat_penalty 1.2, temperature 0.1
#   qwen3.5:9b          presence_penalty 1.5, temperature 1, top_k 20, top_p 0.95
#   gemma4:e4b          temperature 1, top_k 64, top_p 0.95   (no penalty: default 1.1)
#   dots.mocr           stop strings only                     (no penalty: default 1.1)
#
# Only the PENALTIES are mirrored. temperature / top_k / top_p in a Modelfile are
# overridden by the request's greedy settings on Ollama too, so both servers
# decode greedily already; copying them would make vLLM sample where Ollama does
# not. What cannot be mirrored: Ollama's repeat window (64 tokens) -- vLLM's
# repetition_penalty has no window and counts the prompt as well as the reply.
#
# Matched on a substring of the served model name, lower-cased, first hit wins;
# anything unmatched gets Ollama's default. A VLLM_* override below replaces the
# table's value for every model. VLLM_MATCH_OLLAMA=0 sends vLLM exactly what
# every other backend gets (REPETITION_PENALTY, no presence penalty).
VLLM_MATCH_OLLAMA = config.env_bool("VLLM_MATCH_OLLAMA", True)
OLLAMA_DEFAULT_REPEAT_PENALTY = 1.1
VLLM_SAMPLER_BY_MODEL = (
    ("typhoon", {"repetition_penalty": 1.2}),
    ("qwen", {"repetition_penalty": OLLAMA_DEFAULT_REPEAT_PENALTY,
              "presence_penalty": 1.5}),
)
# Blank = use the table. presence/frequency follow OpenAI's -2..2.
VLLM_REPETITION_PENALTY = config.env_float("VLLM_REPETITION_PENALTY", None,
                                           minimum=1.0)
VLLM_PRESENCE_PENALTY = config.env_float("VLLM_PRESENCE_PENALTY", None,
                                         minimum=-2.0, maximum=2.0)
VLLM_FREQUENCY_PENALTY = config.env_float("VLLM_FREQUENCY_PENALTY", None,
                                          minimum=-2.0, maximum=2.0)


def vllm_sampler(model: str = "") -> dict:
    """The penalty fields a vLLM request carries for `model`.

    Merged after `sampler_extras`, so `repetition_penalty` here replaces the
    shared one. Empty when VLLM_MATCH_OLLAMA is off and nothing is overridden.
    """
    fields = {}
    if VLLM_MATCH_OLLAMA:
        name = (model or "").lower()
        fields = {"repetition_penalty": OLLAMA_DEFAULT_REPEAT_PENALTY}
        for needle, values in VLLM_SAMPLER_BY_MODEL:
            if needle in name:
                fields = dict(values)
                break
    for key, value in (("repetition_penalty", VLLM_REPETITION_PENALTY),
                       ("presence_penalty", VLLM_PRESENCE_PENALTY),
                       ("frequency_penalty", VLLM_FREQUENCY_PENALTY)):
        if value is not None:
            fields[key] = value
    return fields


# --- Pass-1 (OCR) sampling ----------------------------------------------------
# Pass 1 was fully greedy (temperature 0, top_k 1, top_p 1) and every baseline in
# CLAUDE.md was taken that way. Typhoon's own recommended values are temperature
# 0.1 and top_p 0.6, which is what ships now, at the user's request.
#
# Two consequences worth knowing:
#   * top_k 1 would make temperature and top_p moot (one candidate is always the
#     argmax), so OCR_TOP_K defaults to 0 = the field is omitted and the server's
#     own top_k applies. Set OCR_TOP_K=1 with OCR_TEMPERATURE=0 to get greedy back.
#   * A read is no longer byte-for-byte reproducible, so a one-point difference
#     between two reads is no longer proof of anything. Pass 2 is untouched and
#     stays greedy; Ollama's /v1 honours temperature and top_p and drops top_k.
OCR_TEMPERATURE = config.env_float("OCR_TEMPERATURE", 0.1, minimum=0.0, maximum=2.0)
OCR_TOP_P = config.env_float("OCR_TOP_P", 0.6, minimum=0.01, maximum=1.0)
OCR_TOP_K = config.env_int("OCR_TOP_K", 0, minimum=0)


def ocr_sampling() -> dict:
    """The decoding fields of a pass-1 request (min_p is pinned off by the caller)."""
    fields = {"temperature": OCR_TEMPERATURE, "top_p": OCR_TOP_P}
    if OCR_TOP_K > 0:
        fields["top_k"] = OCR_TOP_K
    return fields


def sampler_extras(n_ctx: int = 0):
    """Sampling controls beyond the greedy core.

    The penalty is always sent, under both of its names. The DRY block is
    llama.cpp-only and omitted entirely when DRY is disabled -- omitted rather
    than sent as no-ops so that with DRY_MULTIPLIER=0 both backends receive a
    byte-identical request body apart from Ollama's "model" field, and there is
    then nothing left to explain a difference in the results except the server
    itself. (The penalty fields are the same on both, so that still holds.)

    `n_ctx` is the window this request is being sent with (backends.num_ctx());
    DRY scans that whole window unless DRY_PENALTY_LAST_N says otherwise.
    """
    penalty = {
        "repeat_penalty": REPETITION_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if DRY_MULTIPLIER <= 0:
        return penalty
    try:
        window = int(n_ctx)
    except (TypeError, ValueError):
        window = 0
    last_n = DRY_PENALTY_LAST_N or (window if window > 0 else DRY_PENALTY_FALLBACK)
    return {
        # See REPETITION_PENALTY: off by default, DRY is the loop defence.
        **penalty,
        # DRY instead. It penalises *sequence* repetition, and allowed_length lets
        # short legitimate repeats through while breaking runaway loops -- which
        # greedy decoding on a 2B model is prone to.
        "dry_multiplier": DRY_MULTIPLIER,
        "dry_base": 1.75,
        "dry_allowed_length": DRY_ALLOWED_LENGTH,
        # A number, never -1. See DRY_PENALTY_LAST_N above.
        "dry_penalty_last_n": last_n,
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

# The live read sends its tokens to the browser at most once per this many
# milliseconds, as one batched `token` event carrying the pieces' text and count.
# One event per token is one NDJSON line, one JSON parse and one DOM update per
# token -- 150 a second on a fast GPU -- for text nobody reads faster than a few
# times a second. The first piece of a page is sent at once so time to first
# token still shows; what is left is sent before `page_done`. 0 sends every token.
# Display only: the transcript, the timings and the loop guard never see it.
STREAM_FLUSH_MS = config.env_int("STREAM_FLUSH_MS", 100, minimum=0, maximum=2000)

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
# Cut off a pass-2 request that starts looping, on vLLM. Every pass-2 request
# there is STREAMED, its tail tested as it arrives, and the connection closed the
# moment it cycles -- vLLM aborts a request whose client has gone, so the slot
# and its KV cache are freed instead of being held to the token cap (up to 4096
# tokens per request) on a shared server. What came back before the cut is kept
# and salvaged exactly as a reply that ran to the cap would be. vLLM only
# (`backends.serves_metrics`): every llama.cpp and Ollama measurement in this
# project was taken on the non-streaming request, and that is what they still
# get. Also needs the page's loop guard on -- one switch for "stop a repeat".
EXTRACT_LOOP_ABORT = config.env_bool("EXTRACT_LOOP_ABORT", True)
# The list half of that test: the last few `other_fields` entries repeating as a
# whole sequence, three times running, over at least this many characters. NOT
# "one entry seen three times" -- measured 2026-10-06 on 453 real gemma4:e4b
# replies, that cut 6 of the 13 long enough to test and every one was sound: the
# model writes table rows into `other_fields` under their column heading, so a
# heading repeats once per row. The longest legitimate repeated run seen was
# 3 x 1 entry over 184 characters; a loop repeats until the cap.
EXTRACT_CYCLE_MIN_CHARS = config.env_int("EXTRACT_CYCLE_MIN_CHARS", 800, minimum=200)

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

# --------------------------------------------------------------------------
# the item table (2026-09-30)
# --------------------------------------------------------------------------

# Read each document's item table out of its transcript and say whether it is a
# list of items or a list of other documents (`is_master_table`). See `tables.py`.
#
# **It costs pass 2 nothing and moves none of its numbers**: the table is parsed
# in Python from text pass 1 already produced, no prompt changes, and the field
# score never reads it. Off returns results exactly as they were before it
# existed -- no `item_table` key at all -- which is the shape to use for
# comparing against a build that predates it.
TABLE_EXTRACT = config.env_bool("TABLE_EXTRACT", True)
# Ask the extraction model to re-cut rows that do not fit the table's headings.
# ONE request per document, and only where there is such a row -- a table the
# read produced as a proper grid is never sent anywhere.
#
# Safe to leave on because of what the answer IS, not because of who gives it:
# the model is asked which column each cell belongs under, and `tables.recut`
# does the cutting. No character of the reply reaches the table, so the model
# moves a cell boundary and can do nothing else. Off keeps such rows as read,
# flagged `misaligned`, and `is_master_table` stays undetermined where no row
# lines up at all.
TABLE_RECUT_WITH_MODEL = config.env_bool("TABLE_RECUT_WITH_MODEL", True)
# The most rows one re-cut request carries. A table with more misaligned rows
# than this is not a table with a few bad cuts, it is a read that did not
# produce a grid, and the honest report is the flag rather than a long request.
TABLE_RECUT_MAX_ROWS = config.env_int("TABLE_RECUT_MAX_ROWS", 40, minimum=1)
# The reply is one small number per cell, so it is sized from the cell count
# and this is only the ceiling.
TABLE_RECUT_MAX_TOKENS = config.env_int("TABLE_RECUT_MAX_TOKENS", 1200, minimum=64)

# The two table AGENTS (2026-10-01, at the user's request). Each is sent the
# TABLE -- never the document -- and neither answer reaches the table as text.
#
# The fix agent quotes text in the table's cells that does not belong there --
# a stamp's words, a QR code's description, stray text -- and the code takes it
# out where the quote is found in that cell, keeping it as plain text
# (`outside_text`). `tables.removal_refused` keeps amounts, dates, document
# numbers and anything too short to be stray. The cell boundaries are then
# settled exactly as with it off: `realign`, then the column question.
TABLE_FIX_AGENT = config.env_bool("TABLE_FIX_AGENT", True)
# The concat agent (2026-10-01, at the user's request: *merge the table into one
# if there is one table in the document*). An OCR read often cuts one printed
# table into pieces -- its totals block, or rows continued under headings it
# garbled, come out as tables of their own. The model is shown the item table and
# the other tables after it and names which are parts of it; the code joins them
# (`tables.concat`). It runs before the fix agent, so the fix agent sees the whole
# table and the page after it.
TABLE_CONCAT_AGENT = config.env_bool("TABLE_CONCAT_AGENT", True)
# The identify agent decides `is_master_table` from the table, the document's
# type and a few extracted values. A `true` is taken only where the column it
# names is a filled reference column (`tables.reference_column_ok`); otherwise,
# and when it is off or fails, the Python rule answers. The rule's answer is kept
# beside the agent's as `master_rules`, so a disagreement is visible.
TABLE_IDENTIFY_AGENT = config.env_bool("TABLE_IDENTIFY_AGENT", True)
# The longest table the fix agent reads in one request. Past this the table is
# not sent and Python repairs it alone, and the result says so.
TABLE_FIX_MAX_ROWS = config.env_int("TABLE_FIX_MAX_ROWS", 60, minimum=1)
# What share of the page's lines the fix agent is shown on EACH side of the
# table -- 0.2 is 20% before and 20% after, a page's lines counting every line
# of text and every row of every table on it (`tables.context_window`).
# 2026-10-01, at the user's request: a table the read cut short leaves its
# headings above it or its totals below it, sometimes as a second small table.
TABLE_CONTEXT_SHARE = config.env_float("TABLE_CONTEXT_SHARE", 0.2,
                                       minimum=0.0, maximum=1.0)
# How many rows the identify agent is shown. Telling documents from goods does
# not need every row, and the first dozen are the cheapest evidence there is.
TABLE_IDENTIFY_MAX_ROWS = config.env_int("TABLE_IDENTIFY_MAX_ROWS", 12, minimum=1)

# --------------------------------------------------------------------------
# stage 0b: how many DOCUMENTS are in the file (2026-09-08)
# --------------------------------------------------------------------------

# Split a multi-page read into one document per document and extract each
# separately. Off reads every page of a file as one document, which is what
# every measurement in CLAUDE.md was taken under -- so this is the knob that
# reproduces them, the same role `CLASSIFY_WITH_MODEL=0` plays one stage up.
#
# **It cannot change a single-page read at all**, and that is by construction
# rather than by a guard: `segment.segment` over one page returns one document,
# and one document takes the path it always took. Eleven of the thirteen
# fixtures are one page.
SEGMENT_DOCUMENTS = config.env_bool("SEGMENT_DOCUMENTS", True)
# Ask the model where the boundaries are, but ONLY where Python could not tell.
# The rules in `segment.py` are readings of what the page prints -- a page
# numbering itself, a heading naming another type, a page with no heading at all
# -- and a request that could only ever disagree with printed text can only make
# the answer worse. What is left for the model is the one case no rule reaches:
# two pages of the same type, each with its own heading, neither numbered.
#
# Off keeps Python's own guess, which SPLITS. That direction is deliberate --
# see `segment._boundary`: a continuation page split off wrongly still extracts
# what it prints and says it was a guess, while two documents merged wrongly
# answer one form out of two documents' figures with nothing anywhere saying so.
SEGMENT_WITH_MODEL = config.env_bool("SEGMENT_WITH_MODEL", True)
# How much of each page that question carries. It is a question about which
# pages belong together, and what answers it is the top of each page -- the
# letterhead, the heading, the document number. Sending whole pages would put a
# multi-page document's every table row into a request that reads none of them.
SEGMENT_MAX_CHARS = config.env_int("SEGMENT_MAX_CHARS", 700, minimum=100)
# One short JSON object holding a list of lists of page numbers. Capped small
# for the reason `CLASSIFY_MAX_TOKENS` is: a reply long enough to overrun it is
# one that started transcribing the pages instead of grouping them.
SEGMENT_MAX_TOKENS = config.env_int("SEGMENT_MAX_TOKENS", 200, minimum=32)
# The most pages the ONE-SHOT question is asked over. Its prompt carries a
# digest of every page, so it grows with the file -- which is the whole reason
# for the cap, and the reason the walk below exists rather than the cap simply
# being raised.
#
# Above it the file is not read as one document any more (2026-09-16): it goes
# to `SEGMENT_CHAT`, and only falls back to one document where that is off or
# there is no model server to ask.
SEGMENT_MAX_PAGES = config.env_int("SEGMENT_MAX_PAGES", 20, minimum=2)
# Past that cap, walk the file page by page instead of asking about all of it at
# once: one short question per unsettled boundary, carrying the document so far
# as the conversation, and the memory reset the moment a document ends.
#
# **The prompt is then bounded by the length of a DOCUMENT rather than of the
# file**, which is what makes a hundred-page file answerable at all -- and the
# reset is what stops document 7 being judged against document 1's parties and
# totals.
#
# Off restores the pre-2026-09-16 answer for a long file: read as one document,
# saying so. That is not a small difference -- one document is one form, asked
# once, filled from a hundred pages of candidates, with `grounding.py` endorsing
# every value because it genuinely is on some page.
SEGMENT_CHAT = config.env_bool("SEGMENT_CHAT", True)
# How many turns of the current document the question carries, INCLUDING the
# page the document opened on, which is always kept.
#
# **A conversation is not memory the server holds** -- both backends are
# stateless, so every turn is resent, and an unbounded one grows with the
# document until it overruns `num_ctx` on exactly the long files this was built
# for. What identifies a document is its first page (the heading, the number,
# the parties) and its most recent (the running table, the `3 of 7`); the middle
# contributes least and is what drops out.
SEGMENT_CHAT_WINDOW = config.env_int("SEGMENT_CHAT_WINDOW", 8, minimum=2)
# The most questions one file may cost. Only an UNSETTLED boundary asks, so a
# file whose pages number themselves costs none of these however long it is --
# the cap is for the pathological file where nothing is settled and every page
# is a question. Reached, the rest of the walk is Python's own reading, and the
# result says how many were asked.
SEGMENT_CHAT_MAX_ASKS = config.env_int("SEGMENT_CHAT_MAX_ASKS", 200, minimum=1)

# A change of type the classifier is under CLASSIFY_MIN_CONFIDENCE about is a
# question on the walk rather than a certain cut (2026-09-16). It costs requests
# and it is only as good as the model answering: on the 100-page pack qwen3.5:9b
# stayed 78/78 at 50 questions instead of 13, while gemma4:e4b fell 76 -> 64 of
# 78. Off restores trusting every type change. Never applied to a short file --
# there one unsure boundary sends the WHOLE file to the one-shot regroup.
SEGMENT_TYPE_GATE = config.env_bool("SEGMENT_TYPE_GATE", True)

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

# On a vLLM extraction server, ask the agentic steps AT ONCE rather than one
# after another. Added 2026-10-05 at the user's request: vLLM batches concurrent
# requests, so seven steps sent together cost about one step's wall clock.
#
# Safe because no step reads another's answer: each is sent the shared prefix
# and its own question, and its reply contributes only its own keys -- the
# standing agentic rule. A step's grounding re-ask stays inside that step.
#
# vLLM only (`backends.parallel_ok`). llama-server serves one request per slot
# and the instruction-first shape exists to share ONE prefilled prefix, and
# Ollama queues concurrent requests behind `OLLAMA_NUM_PARALLEL`; on either the
# steps would only wait for each other with nothing gained. 0 turns it off.
AGENTIC_PARALLEL = config.env_bool("AGENTIC_PARALLEL", True)

# The most steps in flight at once when `AGENTIC_PARALLEL` applies. 0 = every
# step of the form at once (7-9 on a typed form, 17 unclassified).
AGENTIC_PARALLEL_MAX = config.env_int("AGENTIC_PARALLEL_MAX", 0, minimum=0, maximum=64)

# Hold the first agentic step back until its prefill is done, THEN send the rest,
# when the steps go out at once (above). Added 2026-10-07 at the user's request
# (*use vLLM's endpoints to optimise the requests*). Every step opens with the
# same block and the same transcript; sent together, vLLM schedules them in one
# step and prefills that prefix once per request, because its prefix cache only
# serves blocks already computed. Sent after the first token of step 1, the other
# steps find the prefix cached. Costs one prefill of wall clock where the cache is
# off. 0 sends every step at once, as on 2026-10-05.
VLLM_PREFIX_WARMUP = config.env_bool("VLLM_PREFIX_WARMUP", True)
# The longest the other steps wait for step 1's first token, in seconds. Step 1
# failing releases them at once; this only bounds a server that never answers.
VLLM_PREFIX_WARMUP_WAIT = config.env_float("VLLM_PREFIX_WARMUP_WAIT", 60.0, minimum=0.0)

# One pooled keep-alive connection set for requests to a vLLM / OpenAI-compatible
# server (2026-10-07). `requests.post` opens a new TCP (and TLS) connection per
# call, and a pass-2 request on vLLM is three calls -- /metrics, the chat, /metrics
# -- so a remote server paid three handshakes per request. 0 restores one
# connection per call. llama.cpp and Ollama are not affected either way.
HTTP_KEEPALIVE = config.env_bool("HTTP_KEEPALIVE", True)
# Connections kept per host. Above the agentic step count and a stress run's
# concurrency, or urllib3 drops the extra connections after each use.
HTTP_POOL_SIZE = config.env_int("HTTP_POOL_SIZE", 64, minimum=1, maximum=1024)

# On a vLLM HTTP 400 saying the prompt plus max_tokens is longer than the model's
# window, count the prompt with vLLM's /tokenize, lower max_tokens to what fits,
# and ask once more (2026-10-07). Text requests only (pass 2). Nothing is sent on
# a request that fits. 0 lets the 400 stand.
VLLM_FIT_MAX_TOKENS = config.env_bool("VLLM_FIT_MAX_TOKENS", True)
# The smallest reply worth retrying for; a window with less room than this left
# after the prompt keeps the server's 400.
VLLM_FIT_MIN_TOKENS = config.env_int("VLLM_FIT_MIN_TOKENS", 128, minimum=1)

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
