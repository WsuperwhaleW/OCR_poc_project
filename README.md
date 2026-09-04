# Thai Document OCR

Upload a Thai/English document, get back a verbatim transcript and the fields extracted
from it, each value traced back to the text it came from.

This app holds no model weights and imports no torch, transformers, accelerate or numpy.
It decodes uploads into page images, caps their resolution, and streams the result back
from an **external** model server over its OpenAI-compatible HTTP API. Which server it
talks to is switchable from the page while it runs.

---

## 1. What you need first

| | |
|---|---|
| Python | 3.11 or newer |
| A model server | llama.cpp's `llama-server`, or Ollama — running separately, with a **vision-capable** model loaded |

The model server's own installation, tuning and choice of model are not documented here.
This app only requires that it accepts an image and speaks `/v1/chat/completions`.

Two things it must have, or reads fail:

- **Vision enabled.** llama.cpp needs a projector (`--mmproj`) or it returns HTTP 500 for
  every image. The app detects this and says so on the page rather than letting the read
  fail.
- **A context window big enough for a page plus the reply.** Too small and field
  extraction comes back as truncated JSON. For Ollama the app can set this per request —
  see [Context window](#context-window).

An example llama.cpp launch:

```bash
./llama-server -m model.gguf --mmproj mmproj-model.gguf --port 8080 -ngl 99 -c 16384 -np 1 -fa on --temp 0
```

## 2. Install

Use a virtual environment — the pins in `requirements.txt` have deliberate upper bounds,
and installing them into the system Python drags every other project along with them.

**Run all three lines in the folder that contains `app.py`, in this order.** A venv belongs
to the folder it was created in: a second copy of the project needs its own, and installing
into one does nothing for the other.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`python -m pip` rather than plain `pip`, deliberately: it installs into the interpreter you
are actually running, so it cannot silently install somewhere else.

### Offline install (no internet on the target)

`wheelhouse/` holds every dependency as a prebuilt wheel, so a machine with no route to
PyPI installs from disk. Same venv steps; only the install line changes:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

`--no-index` stops pip reaching for PyPI at all, so the install either succeeds from
`wheelhouse/` or fails immediately and says which package was missing — rather than
hanging on a network timeout on a box that has no network.

**The bundled wheels are built for Linux x86_64 on CPython 3.11.** Eleven of the sixteen
are pure Python and install anywhere; the other five are compiled and are not portable:

| Wheel | Runs on |
|---|---|
| `pillow`, `pillow_heif`, `markupsafe`, `charset_normalizer` | Linux x86_64, CPython **3.11 only** |
| `pymupdf` | Linux x86_64, CPython 3.9 or newer (`abi3`) |

So the general "Python 3.11 or newer" in [step 1](#1-what-you-need-first) narrows to
**exactly 3.11** on this path — a 3.12 or 3.13 venv rejects the cp311 wheels, as does
Windows or arm64. pip reports `No matching distribution found` and names the package. That
means the wheelhouse is wrong for the machine, not that the requirement is unavailable;
rebuild it for the target.

### Rebuilding the wheelhouse

Run this on a machine that **does** have internet, then copy the folder to the target:

```bash
python -m pip download -r requirements.txt -d wheelhouse --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.11
```

`--platform` and `--python-version` describe the **target** machine, not the one running
the command, so a Windows laptop can build a Linux server's wheelhouse. Both flags require
`--only-binary=:all:`, which is also what guarantees no source distribution sneaks in —
an sdist would need a compiler on the offline box.

Set `--python-version` to the interpreter the target will actually run, and
`--platform` to its architecture (`manylinux2014_aarch64` for arm64, `win_amd64` for
64-bit Windows). Delete the old wheels first; `pip download` adds to the folder rather
than replacing it, and two versions of the same package leave pip to pick.

### Check the install before starting

```powershell
python -c "import flask, requests, PIL; print('deps ok')"
```

Anything other than `deps ok` means the install did not land where this shell is looking,
and `python app.py` will fail on the first import.

| What you see | What it means |
|---|---|
| `(.venv)` missing from the prompt | the venv is not active — run the activate line again, in this folder |
| `ModuleNotFoundError: No module named 'requests'` (or `flask`, `PIL`) | the venv is active but nothing was installed into it — the `pip install` line was skipped, or was run in a different folder or before activating. Re-run it |
| `Activate.ps1 cannot be loaded ... execution policy` | run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate again |
| `python` not found | Python is not on PATH; use the full path to `python.exe`, or reinstall Python with "Add to PATH" ticked |

To see exactly which interpreter and which pip a shell is using:

```powershell
python -c "import sys; print(sys.executable)"
```

The path it prints must end in `.venv\Scripts\python.exe` inside **this** project folder.
If it points at a system Python or at another project's `.venv`, that is the whole problem.

`deactivate` leaves the venv. **Every command in this README assumes it is active**, and a
new terminal starts without it.

A note on OneDrive: the project works fine inside a synced folder, but OneDrive can move,
lock or de-hydrate a `.venv` in the background, which breaks it in ways that look exactly
like the errors above. If the venv keeps going bad, delete `.venv` and rebuild it — it is
disposable — or keep the project outside the synced tree.

`pymupdf` (PDF input) and `pillow-heif` (iPhone photos) are optional — without them the
app starts anyway and refuses those formats with a clear message.

## 3. Configure (optional)

Every setting has a working default, so you can skip this and come back when you need to
move a port or point at a different model server. See [Configuration](#configuration) for
what is worth changing.

`.env.example` is the full list with defaults. Copy it and edit:

```bash
cp .env.example .env
```

**The app does not read `.env` itself** — there is no `python-dotenv` dependency, on
purpose, so nothing loads a file the deployment did not ask for. Load it yourself before
starting, or just set the one or two variables you need on the command line.

**Windows (PowerShell)**

```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^\s*([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim()) } }
```

**Linux / macOS**

```bash
set -a; . ./.env; set +a
```

`.env` is gitignored and never ends up in the deployment zip.

## 4. Start

```bash
python app.py
```

Open <http://localhost:5000>. Startup takes about a second — nothing is loaded into this
process.

Startup prints what it found. Read these lines before assuming it is healthy:

```
[ocr] listening on http://127.0.0.1:5000
[ocr] llama.cpp http://127.0.0.1:8080 model=... available=True
[ocr] documents folder: /srv/ocr/mockOcr (5 readable)
[ocr] ground truth: /srv/ocr/solution (5 scored cases)
[ocr] run log: /srv/ocr/logs/runs.csv
```

A missing model server, a missing PDF library, an absent fixtures directory and an
unwritable log directory each print a warning and the app **still starts**. None of them
is fatal.

To point at a model server elsewhere, or move the app's own port, without a `.env` at all:

```powershell
$env:LLAMA_URL="http://127.0.0.1:11434"; $env:PORT="8000"; python app.py
```

```bash
LLAMA_URL=http://127.0.0.1:11434 PORT=8000 python app.py
```

Device placement, threads and quantization belong to the model server. This app has no
CPU/GPU control.

---

## Configuration

Every setting is an environment variable, listed with its default in
[`.env.example`](.env.example) — see [step 3](#3-configure-optional) for how to load a
`.env`, or set them in the service unit.

A malformed value prints a warning and falls back to the default rather than stopping
startup. Settings are read **once at import**, so changing one needs a restart of
`app.py` — including after re-loading a `.env`.

Every default lives in [`settings.py`](settings.py), which is also where the handful of
tunables with no environment variable of their own are kept; the prompts are in
[`prompts.py`](prompts.py). Edit either without going near the request code in `app.py`.

The ones that matter for a deployment:

| Variable | Default | Why you would change it |
|---|---|---|
| `OCR_HOST` | `127.0.0.1` | Loopback. Leave it — see [Security](#security). |
| `PORT` | `5000` | Port to bind. |
| `LLAMA_URL` | `http://127.0.0.1:8080` | Where the model server is listening. |
| `OCR_ENDPOINTS` | *(unset)* | Comma-separated list offered in the server picker. |
| `AUTO_SELECT_SERVER` | `1` | Probe the configured endpoints and the ones the run log has runs against at startup, and select the first that answers. `0` starts on the first in the list whether or not anything is listening there. |
| `AUTO_SELECT_MAX_CANDIDATES` | `8` | How many endpoints one auto-select will probe. A dead port costs a pair of connect timeouts. |
| `AUTO_BEST_MODEL` | `1` | Select each pass's model at startup: the one the run log ranks first for that pass. Both stay changeable in the pickers. `0` starts both passes on the reading model — the one-model setup every figure in `CLAUDE.md` was measured under. |
| `OLLAMA_SYSTEM` | `You are a helpful assistant.` | System message sent with every Ollama request; it reproduces what Ollama otherwise injects from the served model's Modelfile. Ollama only — llama-server is sent none. Set it empty to send no system message. |
| `EXTRACT_SCHEMA` | `1` | Whether a field-extraction reply that cannot be parsed is asked again with decoding constrained to the field schema. The first request is unconstrained either way. `0` turns the retry off, so an unusable reply is reported instead. |
| `OLLAMA_REPEAT_PENALTY` | `1.1` | Repetition penalty on the constrained retry above. Ollama only, and the only place this app sets one above `1.0`. `1.0` turns it off. |
| `OLLAMA_UNLOAD_ON_SWITCH` | `1` | Stop the models Ollama is holding when you switch model or server, freeing the GPU for the new one. Ollama only. `0` leaves them loaded until their `keep_alive` expires. |
| `OCR_LOG_DIR` | `./logs` | The only directory written to. Point it elsewhere to mount the app directory read-only. |
| `OCR_MOCK_DIR` | `./mockOcr` | Source documents for the folder picker. Absent ⇒ upload-only. |
| `OCR_SOLUTION_DIR` | `./solution` | Ground truth. Absent ⇒ accuracy scoring switches off and the page hides its controls. |
| `MAX_UPLOAD_MB` | `32` | Per-upload size cap. |
| `MAX_PAGES` | `10` | Pages read per document — a direct cap on the worst case cost of one request. |
| `MAX_JOBS` | `5` | Rendered documents held in memory for the compare view. A 10-page document at `medium` is ~40 MB, so this is a RAM ceiling. |
| `GEN_READ_TIMEOUT` | `1800` | Raise on slow hardware; a timeout firing mid-generation throws away work the model server is still doing. |
| `EXTRACT` | `1` | Set `0` to run the OCR pass only. |
| `AGENTIC_EXTRACT` | `0` | Start with field extraction in agentic mode. Switchable from the page at any time; this only sets what a fresh process starts in. |
| `AGENTIC_RETRIES` | `1` | How many times an agentic step may be re-asked after returning a value that is not in the transcript. `0` turns the retry off. |

## Security

**No authentication, no authorisation, no rate limiting, no CSRF protection.** This is a
single-user tool meant to be reached over loopback, which is why `OCR_HOST` defaults to
`127.0.0.1` and why startup warns whenever it does not. That default is the security
control; changing it removes the only one there is.

Anyone who can reach the port can upload documents, read every document in `mockOcr/`, and
download the full run log. The server picker also accepts an arbitrary URL, so the port is
a request-forgery vector into whatever the host can reach.

`python app.py` runs Flask's development server, which prints a warning saying so. It is
threaded and fine for local traffic. Anything beyond loopback needs a reverse proxy in
front of it, with authentication.

---

## Using the page

### Input

| Input | Handled by | Notes |
|---|---|---|
| JPG, PNG, WEBP, BMP, PPM, ICO, … | Pillow | anything Pillow can decode |
| GIF, multi-page TIFF | Pillow | each frame is treated as a page |
| HEIC / HEIF (iPhone photos) | `pillow-heif` | optional dependency |
| PDF | `pymupdf` | rendered at 300 DPI, optional dependency |

Drop a file on the **Workspace** pane, pick one from the folder list, or pick a benchmark
case. Multi-page input is capped at 10 pages, uploads at 32 MB (`MAX_PAGES`,
`MAX_UPLOAD_MB`). PDF and HEIC can't render in an `<img>`, so the page shows a filename
card instead of a thumbnail.

**The Workspace holds one file, and a new one replaces it** — a read is one measurement, and
documents are tested one at a time. Dropping several keeps the first and says which, and how
many it ignored, rather than silently discarding them. For batches use the **Queue** tab,
whose drop zone takes as many as you like.

### Detail (input resolution)

The **Detail** selector caps how many pixels are sent. It is the main fidelity/latency
dial, because the cost of a read scales with the pixel count.

| Preset | Cap |
|---|---|
| `original` | none — the page exactly as rasterised |
| `medium` **(default)** | 4.0 MP |
| `low` | 2.0 MP |

Three presets, and three names. The earlier `max` / `accurate` / `balanced` / `fast` were
renamed on 2026-08-21 and dropped altogether on 2026-08-24 — **they are no longer accepted
anywhere a Detail arrives**, and a request naming one now reads at the default (`medium`)
and is logged as `medium`, so the run log never claims a budget it did not use. Update
saved scripts: `compare.py --detail accurate` means `--detail medium`, `max` means
`original`, `balanced` means `low`.

The old 1 MP `fast` was deleted rather than renamed, and it is the one preset that really
went away: at 1 MP this model stops misreading and starts **inventing** — on one bundled
case it produced an address that is not on the page — and a preset whose failure mode is
fabrication is not one to offer as the quick option.

The run log keeps whatever it recorded, old names included. The setting and per-document
tables read a renamed preset as the one that replaced it, and leave out runs at a deleted
one; the card says how many, because a setting nobody can select is not one to recommend.

PDFs are rasterised at 300 DPI (`PDF_DPI`) so the downscale resamples from real detail.
Uniform blank margins are cropped before the cap is applied; disable with `TRIM_MARGINS=0`.

#### Seeing the page the model will get

At `medium` and `low` the workspace preview shows the **prepared** page rather than the file
you picked: trimmed, scaled to the cap, and produced by the same code the read uses, so it is
the picture the model is about to be given. Under it, the preset, the prepared size and the
size it came down from, and a **Download** button that saves that exact PNG.

It follows the Detail picker, so switching between the presets shows what each one costs the
page. It also works for PDFs and for a selected benchmark case, which have no browser preview
of their own — page 1, with the page count beside it. At `original` there is nothing to
prepare, so the preview goes back to the file itself and the bar disappears.

No model server is needed for this, and nothing is read: `POST /api/preview` takes the same
`image` / `case` / `file` field the read takes, plus `detail` and an optional `page`, and
answers with the PNG (sizes in `X-Preview-*` headers). Unlike a real read it is not cached, so
previewing does not evict prepared pages the Compare card is still showing.

**Neither end of the range is safe by default.** Too few pixels and fine detail is lost;
too many is not reliably better either. `medium` is the default because it scored best on
the bundled cases — score a case yourself before moving off it.

### Page reading (which prompt goes with the page)

**Page reading** picks the shape of pass 1. A profile is one prompt *plus* whether a system
message is sent with it, because on some models those two cannot be chosen independently.

| Profile | |
|---|---|
| **Typhoon OCR (Markdown transcript)** | The default. Written for `typhoon-ocr1.5`; returns the page as Markdown with tables as `<table>` HTML. Every accuracy figure quoted for this project was measured on it |
| **dots.ocr (layout JSON)** | dots.ocr's own prompt, sent verbatim with no system message. Returns a JSON array of layout blocks — `bbox`, `category`, `text` — which the app flattens back into a transcript in reading order before scoring, extraction or the run log see it |

Two things to know:

- **Both halves of a profile matter.** Send dots.ocr the typhoon prompt, or the right prompt
  with a system message attached, and it answers with two tokens and an empty transcript at
  HTTP 200 — no error anywhere. That is why this is one control instead of two.
- **The profile follows the model.** Picking a reading model selects the shape that model needs
  — the prompt and the system-message veto are properties of the model, and the wrong pairing
  returns an empty transcript at HTTP 200 rather than an error. You can still override it to
  compare a prompt against a model; the override stands until the next model switch. The run log
  records both per row either way.

Set the starting profile with `OCR_PROFILE` (`typhoon` or `dots`). Switching it applies to
this process — the queue and any other browser tab included — and takes effect on the next
page read; a page already streaming finishes under the profile it started with.

### Stop a read that starts repeating (the loop backstop)

A small model decoding greedily can lock onto one line or one table row and repeat it until it
burns the whole token budget — minutes of wall clock for a transcript that stopped saying
anything new near the top. The checkbox under **Page reading** is the backstop: ticked, a read
whose tail is cycling is stopped where it starts repeating and reported as `looped`.

Untick it to let such a read run to `MAX_NEW_TOKENS` (4096). **Only the abort is turned off.**
The transcript is still tested when it arrives and is still flagged `looped`, so a run that
cycled the whole way still reads as a failure, is still kept out of every mean, and still has
its field score suppressed — what you get is all of what the model produced instead of the part
before the backstop fired. Extraction's own loop reporting is not affected either way.

Turn it off to read a page the backstop is cutting short; turn it back on for anything timed or
scored in bulk, because a true runaway then costs the full budget on every page of a batch. If
a document loops repeatedly, **lower Detail first** — a huge image is the usual trigger — and
raising the token cap does not help.

Set the starting state with `LOOP_GUARD` (`1`/`0`). Like the profile, switching it applies to
the whole process and takes effect on the next page read; a page already streaming finishes
under the rule it started with, and the run log's `status` says which.

### Score fields only above (the read floor)

Pass 2 can only map values pass 1 produced, so a field score taken over a broken transcript
marks down whatever extracted and lets whatever read get away with it. **Score fields only
above** is the transcript accuracy a read has to reach before the extraction taken from it is
worth a field score. It sits on the **Workspace** pane, under the loop backstop, and again on
the **Random test** pane: one setting behind two boxes, so moving either moves both.

Under the floor the extraction still runs and the row is still written — keys filled, grounded
ratio, extra fields, timings, and the transcript score itself. What is dropped is the
correctness figure: `field_acc`, `field_expected`, `p1_correct`, `p1_partial` and `p1_scored`
are left blank, the Accuracy tab prints the reason instead of a bar, and the Fields tab prints
it instead of the rate. **The per-value marks stay** — those are what explain a bad read, and
blanking them would remove the evidence along with the number. A random-test round says
**unscored** with the reason in place of its score.

It applies to **every path that reads a page** — this pane, the queue, `POST /api/ocr`,
`POST /api/ocr/stream` and the random test — and it is applied at the moment a run is logged.
A read that looped, was cut off at the token cap, or came back empty is left unscored whatever
the floor says: that transcript is a fragment however it scored. A document with no
`solution/<id>.md` is never suppressed, because nothing there can tell a bad read from an
unmeasured one. **A low score is not a failure** — it is a run, and it counts in the pass-1
mean.

**There are two read floors and only one of them can be undone.** This one decides what is
**written**, and a score never taken is not in the log to recover. The run-log card's own
**Read floor** decides what is **averaged**, and moves freely because every table re-derives
trust from each row's transcript score. So the useful shape is to leave this one at `0` — score
everything, keep the figures — and raise the card's floor to decide what the tables mean by
them.

Set the starting value with `MIN_READ_FOR_FIELDS` (a fraction, default `0.75`). Switching it
applies to the whole process and takes effect on the next run; the page sends what is on screen
before each run, so a run is judged by the floor its operator is looking at.

### Raw output (the reply before anything touched it)

A **Raw output** toggle sits at the top right of the Result card, next to Layout. It shows the
model's answer for one page exactly as it arrived — before the code fence is stripped, before a
layout reply is flattened into a transcript, and before `normalise_output` removes the markup the
model emits regardless of the prompt. Page arrows for multi-page documents, and a Copy button.

It is the only place the discarded material is visible: `<page_number>` tags and leading `# `
headings on a Markdown profile, and the `bbox` coordinates on a layout one, which a transcript has
no way to carry. Enabled as soon as a run returns anything, on every profile.

Like Layout, it is a toggle rather than a tab — press it again to go back to where you were.

### Layout (where the model says it found each block)

A **Layout** toggle sits at the top right of the Result card. It shows the page exactly as the
model received it with the model's own bounding boxes drawn on top, numbered in the order they
were returned. Click a box to read the text that came back for it, with its category and
coordinates; click it again to clear. Multi-page documents get page arrows.

It is a toggle, not a fifth tab — pressing it again returns you to the tab you were on.

**It is enabled only when the run actually returned boxes**, which today means a profile whose
reply carries geometry (**dots.ocr**). A Markdown profile has no coordinates to draw, so the
button stays disabled rather than showing an empty page, and a fresh run on such a profile
returns you to the transcript rather than leaving you on a blank view.

Boxes are drawn at the model's own coordinates against the image's own pixel size, unscaled and
uncorrected. A box in the wrong place is the model putting it there — which is the point of
looking.

### What a run does

Each run is up to two passes against the model server:

1. **OCR** — the page image in, a verbatim transcript out, streamed to the browser as it
   arrives. Shown in **Markdown** (raw) and **Rendered**.
2. **Extract** — the transcript back in as *text*, structured JSON out. Every returned
   value is then traced back to the transcript **in Python**; a value that is not in the
   text is flagged rather than displayed as data. Shown in **Fields**.

The grounding check in pass 2 is deterministic on purpose — it is done by this app, not
asked of the model. Turn extraction off with `EXTRACT=0`, or send `extract=0` with the
request. It can be re-run on its own from the page without re-reading the image, or via
`POST /api/extract`.

Nothing flagged is deleted or rewritten. A flagged value stays visible with its flag.

### Fields

Twenty-nine scalar fields, plus `line_items` and `other_fields`, grouped the way the page
is laid out:

| Group | Keys |
|---|---|
| Document | `document_type`, `document_number`, `issue_date`, `due_date`, `service_period`, `currency`, `vat_rate` |
| References | `reference_document`, `po_number`, `original_invoice_number`, `contract_number`, `customer_code`, `location_code` |
| Seller | `seller_name`, `seller_tax_id`, `seller_branch`, `seller_address` |
| Buyer | `buyer_name`, `buyer_tax_id`, `buyer_branch`, `buyer_address` |
| Totals | `subtotal`, `vat_total`, `amount_incl_vat`, `withholding_tax_total`, `net_payable`, `amount_in_words` |
| Payment | `payment_method`, `payment_reference` |

Each value carries a state: **grounded** (found in the transcript), **missing** (came back
empty), **nil** (the page printed `-`), **ungrounded** (not on the page — the model
produced it), **computed** (a total that is not printed but equals the sum of the extracted
figures).

**The totals are four separate figures, not one grand total.** `subtotal` is before VAT,
`amount_incl_vat` is after VAT and before withholding, `net_payable` is what is left to pay
once withholding comes off. Only the lines the page actually prints get filled; the rest
stay empty rather than being derived.

Whether the printed figures already carry VAT is decided in Python from the figures
themselves, and the tab labels itself from it — the money column reads *Amount (incl. VAT)*
or *Amount (ex VAT)*, with a note saying which basis the page is on and what said so.

The **References** group hides itself when it comes back empty; a cash receipt legitimately
has none of those fields.

### Field extraction: one request, or field by field

Pass 2 runs in one of two shapes, chosen by **Field extraction** in the sidebar or by the
**Agentic** button beside Re-extract. The two controls are the same switch.

| | one request | agentic |
|---|---|---|
| Requests per document | 1 | 15 |
| Fields asked for at once | all 31 | 1–3 |
| A step that returns something not on the page | — | asked again once |
| A reply that will not parse | costs the whole extraction | costs that step's fields |
| Speed | fast | slower |

Agentic mode walks seven steps — the heading, the reference, the seller, the buyer, the
currency, the totals, and whatever is left over — asking for one to three fields each against
the same transcript. Use it when a value keeps landing in the wrong field; the single request
is faster and is what a fresh process starts in.

While it runs, the Fields tab lists the steps and marks each one done, re-asked or failed.
The list stays after the run, under the fields, so a value can be traced back to the step
that produced it. A step that failed is called out above the fields: its keys are empty
because the question was never answered, which is not the same as the document being silent.

**The step that lists `other_fields` is the exception, and it is never a failure.** Its reply
is a list, so a reply cut off at the step's token cap has still finished most of what it was
saying: the entries that closed are kept and the half-written one at the end is dropped — ten
whole entries and an eleventh cut in half is recorded as ten. The row says *cut off — kept
what finished*, in amber rather than red, and a note above the fields says the list is short
for that reason rather than because the page has little on it. Nothing counts it as an error:
not the failed-step banner, not `steps_failed`, not the run log's failure rate. The reason is
what `other_fields` is — everything the document type's own field set does not cover, under
the document's own labels, which nothing scores. A step that owns one of those keys is a
failure when it breaks, exactly as before.

The switch is server-side and takes effect on the next extraction — including queued
documents and the run log's `extract_mode` column, which records the shape each row actually
ran in. It is **not** refused while the queue is busy, unlike switching model server.

**The fields on screen say which shape produced them**, first thing on the Fields tab status
line: `single prompt` or `agentic · 7 steps`, then the seconds, tokens, model and grounded
share. That is the mode this result ran in, not the current setting — the picker may have been
switched since, and a result loaded from the queue can be older still.

`AGENTIC_EXTRACT=1` starts in agentic mode; `AGENTIC_RETRIES` sets how many times a step may
be re-asked (default 1, `0` to turn the retry off). Over HTTP, send `{"mode": "agentic"}` to
`POST /api/extract`, or `POST /api/extract/mode` to change the setting.

**A step row can show what was sent and what came back**, through two buttons. Neither panel
opens until you press for it, and neither button is on every row: each appears where it
answers a question you would actually be asking.

| Button | On which steps |
|---|---|
| **prompt** | any step that has answered — the question is for reading against the answer it produced, so there is nothing to check while the step is still running |
| **raw** | only a step with something to explain: it failed, its reply was cut off part-way through the extra fields, it was re-asked, its first reply would not parse, it answered with a value that is not on the page, or it left one of its own keys empty. **Hover the button and it says which of those it is.** |

So a clean step carries one button and a troubled one carries two, and the rows worth opening
are the rows with more on them. A clean step's reply is not lost — the **Raw output** checkbox
at the top of the tab still holds every reply of the run, concatenated and labelled.

**The reply panel is one block per request**, because a step can send more than one:

| Label | The request |
|---|---|
| `seller` | the plain ask |
| `seller (schema)` | decoding constrained to that step's keys, asked because the plain reply would not parse |
| `seller (retry)` | asked again with the rejected values quoted back, because the first answer was not on the page |
| `… (failed)` | the step raised: neither attempt produced usable JSON |

Two of those are worth knowing by shape. **A step showing two blocks is a step whose first
reply would not parse** — the schema attempt only ever runs for that reason. And **a failed
step keeps its replies**, which is the text most worth reading: its keys come back empty, and
empty says nothing about why.

**The prompt panel is laid out the way the message is.** Every step sends the same
instructions and the same transcript, then its own question — that order is what lets
llama.cpp prefill the document once for the whole run. So the panel shows one **shared
prefix** line for the whole list, with its own show/hide, and then this step's question in
full. A step that was re-asked shows the second question too, so what changed between the two
attempts is on screen rather than inferred.

Panels stay open while the rest of the run finishes, and across re-extractions of the same
document.

In single mode there are no steps. The message is one string (instructions, transcript,
instructions again) and it sits collapsed at the top of the **Raw output** pane as *Prompt
sent — N characters*; when the schema rescue runs, that pane shows **both** replies — the one
that failed and the one that worked — instead of only the one that worked.

### Fields only (pass 2 without a read)

The **Fields only** pane runs extraction on its own, against a case's hand-written
transcript. `solution/<id>.md` goes straight into pass 2 — the page is never read, no image
is made, and nothing a read got wrong can reach the fields.

Pick a document, a model server and model, and the extraction shape, then press **Extract
fields**. The results land on the **Fields** tab exactly as they do after a read, scored
against `solution/<id>.fields.json` where there is one. The pane says which file it is about
to feed in, how long it is, and whether that document has a field truth file to be scored
against.

Use it to compare two models, or the two extraction shapes, on the same input: with the
transcript fixed, the difference between two runs is the extractor's.

Four things to know:

- **The model picker here is the same setting as the Workspace pane's**, and so are the server
  and the extraction shape — one process, one choice. Changing one here changes it there.
- **Vision is not required.** Pass 2 sends text and gets text back, so a model with no image
  support can still be measured on the form. The Workspace **Read document** button still
  needs one.
- **There is no transcript accuracy for these runs.** The Accuracy tab is switched off and the
  ground truth is shown on the Markdown and Rendered tabs, because that is the input, not a
  read. Scoring the ground truth against itself would be 100% and would mean nothing.
- **Each run appends its own `extract` row to the run log**, with the file named `<id>.md` and
  the source `truth`. It never overwrites the pass-2 columns of an earlier read: those figures
  belong to a transcript this run did not use.

`python compare.py --fields-only --from-truth` is the same thing from the command line,
across every case at once.

### Compare

The card below the result puts two of three panels side by side, for proofreading.
**Show** opens it; the buttons on the left of its toolbar choose the pair:

| View | |
|---|---|
| **Source › OCR** *(default)* | the prepared page beside the transcript |
| **Source › Truth** | the prepared page beside `solution/<case>.md` |
| **OCR › Truth** | the transcript beside the ground truth it is scored against |

A view is offered only when both of its halves exist. The source image and the transcript
come from a run; ground truth comes from the fixture, so it can be read on its own — pick a
benchmark case, or drop a file that matches one, and open the card before running anything.

The image shown is the **prepared** page — after PDF rasterisation and after the Detail
downscale — so it is literally what was sent. Ground truth is rendered through the same
renderer as a transcript, so a pipe table in the fixture and an HTML table from the model
appear as the same shape of table.

- Page arrows for multi-page documents; image and transcript stay on the same page. Ground
  truth is one file for the whole document and its caption says so.
- **Markdown** checkbox swaps the text panels between rendered output and raw text.
- **Sync scroll** links whichever two panels are showing, proportionally.
- **Open image** opens the prepared page full size in a new tab, and **Download** saves it —
  the same picture the workspace preview offers, for the run that actually happened.
- Available while a run is still streaming.

Prepared pages are cached in memory per upload (`MAX_JOBS = 5`, oldest evicted) and served
from `GET /api/page/<job>/<n>`. Ground truth is served verbatim from
`GET /api/truth/<case>` and fetched once per case. Nothing is written to disk.

### Queue tab

One document at a time streams in the Workspace pane. Several go to the **Queue** tab, which
is the only place that fills the queue: its drop zone takes any number of files, and **Queue
all cases** and **Queue whole folder** sit beside it. The Workspace has no queue buttons — it
holds the one document being tested, and what it does with it is read it.

**Queuing does not start anything: the queue holds until you press Run queue.** That is what
makes the run mode and the worker count settable against a batch you can already see, and it
lets a batch be assembled over several drops. The button counts what is waiting — *Run queue
(3)* — and the line under it says `holding — press Run queue` so a queue that is deliberately
not working cannot be mistaken for a stalled one.

While it runs the same button reads **Pause after current**. Pausing stops the queue handing
out further documents; one already reading is left to finish, because llama.cpp cannot abandon
a generation it has started and a button that claimed otherwise would be lying. To stop that
one, **Cancel** it. A batch that drains closes the gate behind it, so the next thing you queue
waits for its own Run.

| Run mode | |
|---|---|
| **Sequential** *(default)* | one worker, documents run one after another |
| **Concurrent** | as wide as the batch; set **Workers** explicitly if you want a different number |

Match the worker count to the model server's slot count (`-np` on llama-server). More
workers than slots does not add throughput — it moves the waiting inside the model server,
where it is invisible and cannot be cancelled. The page shows the server's slot count as
advice.

A batch is queued in full before any worker picks up the first item, so a multi-file drop
is a fair concurrency test. Jobs do not survive a restart.

### Model server picker

The **Model server** dropdown at the top of the left card switches which server the app
talks to, without restarting it. Offered by default:

| Endpoint | Usually |
|---|---|
| `http://127.0.0.1:8080` | llama-server (the `LLAMA_URL` default) |
| `http://127.0.0.1:11434` | Ollama's standard port |

Set the list with `OCR_ENDPOINTS` if your servers listen elsewhere.

**The app picks one that is online at startup.** The list above is a constant and the run
log is a history, and neither says which port has something listening on it today. So
before it reports on a server, the app probes the configured endpoints and then every
server `logs/runs.csv` has runs against — most recently used first — and makes the first
one that answers active. It stops at the first, so the usual case is one probe, and it
never moves off an endpoint that answers. Startup prints the endpoint it landed on and
what it tried. `AUTO_SELECT_SERVER=0` turns it off; `AUTO_SELECT_MAX_CANDIDATES` bounds
how many it will try, since a dead port costs a pair of connect timeouts.

It happens **at startup only**. Once the app is running the server picker is the way to
move, and **Re-check** re-probes the endpoint already selected — after starting a model
server the app did not find at launch, press Re-check, or restart the app.

### The models are selected for you at startup

Startup also picks the model for each pass: the one the run log **ranks first for that
pass**, among the models the active endpoint serves. It is the same ranking the Summary
tab's headline cards and the Full rank panel show (`runlog.best_models`, accuracy ×
(1 − failure rate), windowed per model like every other figure on that card), and a model
with a track record outranks one with a run or two behind it.

**They are ordinary selections, exactly as if you had used the pickers** — so the page
opens on them, and changing either is the picker as usual. Nothing is re-decided later: a
default that moved under a session as the log grew would be worse than one set once.

- **The two passes rank differently, and that is the point**: reading lands on the OCR
  model, while extraction lands on whichever general model has scored best here rather
  than on whatever is reading the page. Pick **same as reading model** to go back.
- A model the log has never scored is still reachable as a reading default: below the
  ranking, the old rules answer (an OCR model first, then anything with vision).
- The ranking knows nothing about cost. Where a slower model wins by a point or two, pick
  the cheaper one by hand.

Start the app with `AUTO_BEST_MODEL=0` to run both passes on the reading model, which is
the one-model setup every figure in `CLAUDE.md` was measured under and the setting for a
sweep whose numbers are to be compared with those. `compare.py` prints the extraction model
in its header whenever it differs from the reader.

The two server kinds are detected, not configured — `backends.py` probes `/props` first
(llama.cpp) then `/api/tags` (Ollama) — and the app adapts:

| | llama-server | Ollama |
|---|---|---|
| Models | exactly one, chosen at launch | many; a **Model** picker appears when there is more than one |
| Vision | `modalities.vision` from `/props` | `capabilities` per model from `/api/tags` |
| Concurrency | `total_slots` from `/props` | not exposed, so the page claims no number |
| Timings | `timings` block: real prompt/predict split | OpenAI `usage`; time-to-first-token stands in for prefill |

### A separate model for extraction

The two passes ask for different things — reading Thai off an image, and mapping a transcript
onto a form — and a model can be good at one and poor at the other. **Model · extracts fields**,
beside the reading model in both the Workspace and Fields panes, chooses what pass 2 runs on:

- **same as reading model** — one model does both, which is what every figure in the run log
  predating this choice was measured under, and what `AUTO_BEST_MODEL=0` starts on. It is no
  longer what a fresh start selects: since 2026-09-03 startup picks the extractor the run log
  ranks first (see above), which is usually not the model reading the page.
- **any general model the endpoint serves** — pass 2 sends text and gets JSON back, so it does
  not need vision. A text-only model is a perfectly good extractor and is offered here.

**One combination is refused** (HTTP 400): reading with one OCR model and extracting with a
*different* OCR model. It costs a second set of weights to get a second model at the task OCR
fine-tunes are worst at. Extracting with the reading model itself is always allowed — that is
the one-model case, not a second OCR model. The picker leaves the refused models out rather than
offering them and failing, but the server enforces the rule regardless.

On Ollama, **both** chosen models are kept resident when the app frees the GPU, so pass 2 does
not pay a fresh model load on every request.

The run log records `extract_model` only where the two passes differed, and the **Pass 2** table
groups rows by the model that *extracted*, naming the reading model as `reading: …` underneath —
a field score taken over a real transcript is partly a measurement of pass 1.

Switching is **refused while the queue is running** (HTTP 409) — half a document read on
one server and half on another would be logged and scored as if one server had done it.
Finish or cancel those jobs first. A single streaming run is not blocked, but the switch
only affects the *next* run: the model, backend and URL are captured per page as it is
read.

Switching a **model** on Ollama stops the models it was holding, so the new one loads onto
a card the old one has let go of. Ollama keeps every model it has served resident for its
`keep_alive` (5 minutes by default) and loads the next one beside it; what the app sends is
what `ollama stop <model>` sends. The picker's status line names what it stopped.

Three things to know about it:

* **It stops every model resident at that endpoint**, not only the one this app selected —
  the model still holding the card is often not the one last picked here. Set
  `OLLAMA_UNLOAD_ON_SWITCH=0` if the Ollama server is shared with anything else.
* **llama-server is untouched.** It serves the one model it was started with for the life
  of the process, so there is nothing a switch could release.
* **It is skipped while the queue is working**, even though a model-only switch is allowed
  there. Eviction goes to the same scheduler that is serving the run in flight.

The app never polls the model server. It asks for status when the page renders, when you
press **Re-check**, and once at startup, and that is all.

### Server status

The status bar shows a green/red dot, the model name, the server kind, and a **Re-check**
button. When the server is unreachable or cannot see images, the reason is shown and **Read
document stays disabled**:

| Condition | Message |
|---|---|
| Nothing on the port | `No model server reachable at <url>.` |
| llama-server, `modalities.vision=false` | `...running '<model>' with vision disabled ... Restart it with --mmproj` |
| Ollama, model has no vision capability | `...has no vision capability ... Choose a vision model` |
| Ollama with nothing pulled | `...has no models pulled.` |

A model whose capabilities the server does not report is shown as *vision not reported* and
allowed through.

### Context window

Extraction sends the whole transcript, so the server's context has to hold the prompt, the
document and the reply. Too small and the JSON comes back truncated.

The **Context** dropdown sets it per request (4096 – 32768) without restarting anything; it
starts at `OLLAMA_NUM_CTX` (default 8192). A change is refused with 409 while a job is
running.

It applies to **Ollama only**, whose OpenAI-compatible endpoint has no other way to set it.
The dropdown is disabled while llama-server is active, because its window is fixed at
launch by `-c`. Setting it on the Ollama server instead also works:

```
set OLLAMA_CONTEXT_LENGTH=16384
```

### When a read fails

| Reported as | Means |
|---|---|
| `truncated` | the reply hit `MAX_NEW_TOKENS` (4096) and was cut off; a warning banner shows on the page |
| `looped` | the output was cycling. With the backstop on the read was aborted there; with it off the read ran to the cap and is flagged all the same, rather than letting a repeating transcript pass as a complete one |
| `error` | the request itself failed — see the message and [Server status](#server-status) |

Extraction has the same two failure modes and reports them the same way, so an unterminated
JSON reply is labelled as a loop rather than a cryptic parse error. **Re-extract** retries
just that pass. If a document loops repeatedly: lower **Detail** first, and extract one page
at a time. Raising the token cap does not help.

---

## Checking accuracy against ground truth

`solution/` holds a hand-transcribed expected transcript for each PDF in `mockOcr/` —
`sol001.md` … `sol013.md`, in Markdown so the tables render when you open them. Edit these
directly; they are the files the scores are computed from. Scoring lives in `scoring.py` and
is shared by the web page and the CLI, so the browser and the terminal can never report
different numbers for the same run.

`solution/<id>.fields.json` is the second ground truth: which **value** belongs in which
**extracted field** for the same document. It scores pass 2, the way the `.md` scores pass 1,
and it is filled in by hand — see *Field extraction accuracy* below.

`solution/out/*.txt` is the transcript of the last run and `solution/out/*.fields.json` the
last extraction, both written automatically. Deleting them is harmless.
`solution/manifest.json` maps each id to its source PDF.

### Picking a set by kind of document

Each case in `manifest.json` carries `doc_types` — a **list**, because a document is often
more than one type. The codes are `normalise.document_types`' vocabulary. The source PDFs are
named for them, so a set can be picked from the shell or from the page:

```bash
ls mockOcr/invoice_*.pdf
```

| filed under | cases | also |
|---|---|---|
| `STATEMENT_OF_ACCOUNT` | sol001 | invoice |
| `INVOICE` | sol002 | — |
| `RECEIPT` | sol003, sol004, sol005, sol007, sol011, sol012 | tax invoice |
| `CREDIT_NOTE` | sol006, sol008, sol009, sol010 | sol006 is also a tax invoice |
| `TAX_INVOICE` | sol013 | the only case that is a tax invoice and nothing else. No requirement covers that type on its own, so it is asked the widest form (30 keys) and nothing is Mandatory — it is marked **unknown type** and its headline field score is taken over the 13 values its truth file states |
| `WHT_CERTIFICATE` | — | the withholding tax certificate (มาตรา 50 ทวิ) is a recognised type with a form and validation rules of its own, and no fixture yet |

A file is named for **every** type its heading names, in that order — so
`receipt_tax_invoice_sol003.pdf`, `credit_note_tax_invoice_sol006.pdf`,
`statement_of_account_invoice_sol001.pdf`. Globbing finds a family
(`ls mockOcr/credit_note_*`); `GET /api/cases` is what to read for the exact list.

The three case dropdowns on the page (**Benchmark case**, **Lock document**, **Ground-truth
document**) group their options under the type each case is filed under, and `GET /api/cases`
reports `doc_types` per case. The PDFs were renamed to this scheme on 2026-08-31 (sol011-sol013 arrived under it on 2026-09-03); the names they
shipped under before that are listed as `aliases`, so an upload of an original still matches
by name, and any copy of one still matches by contents whatever it is called.

### From the web page

Pick a **Benchmark case** from the dropdown and hit **Read document** — the server loads
that PDF itself, so there is nothing to upload.

Or just drop a file in: it is **mapped to its solution automatically**, and a badge under
the file says which one and how it was matched before you run anything. The browser hashes
the file locally and asks `POST /api/match`; nothing is uploaded for the lookup. Matching is
tried in order of confidence:

| Order | Rule | Survives |
|---|---|---|
| 1 | exact filename | — |
| 2 | filename ignoring case and punctuation | `Receipt CDS ... mock1.PDF` |
| 3 | `aliases` listed in `manifest.json` | any rename you declare |
| 4 | **sha256 of the file contents** | any rename at all |
| 5 | embedded document number (`BS…`, `F…TINV`) | re-exported files |

A file that matches more than one case is reported as unmatched rather than scored against
a guess.

When the input has a known ground truth, an **Accuracy** tab appears with three
colour-coded bars (red < 80%, amber < 95%, green above) and a colourised diff against the
expected text — showing differences in content only, with spacing and line breaks
folded out. The headline character accuracy is also appended to the footer line. Every
scored run is written to `solution/out/<id>.txt`.

The tab also says how this run compares with **the best that document has ever scored** —
the highest character accuracy in the run log for that case across every run and every
setting, with the model, backend and detail that reached it, or *the best this document has
scored* when the current run is it. A percentage on its own does not say whether it is good:
93.1% is the ceiling on the handwritten scan and a poor result on the printed pages.

### From the CLI

The app must be running — each case is sent through it over HTTP at
`http://127.0.0.1:$PORT`, so the whole real pipeline is exercised rather than a library
call.

```bash
python compare.py
```

```bash
python compare.py sol005
```

```bash
python compare.py --no-run
```

| Flag | |
|---|---|
| `--no-run` | re-score the saved `solution/out/<id>.txt` without paying for OCR again |
| `--detail original\|medium\|low` | override the resolution preset |
| `--app URL` | score a deployed instance instead of localhost |
| `--keep-tables` | compare table markup literally instead of normalising it. Fill rules are kept too, so this is also how to reproduce a score taken before they were dropped |
| `--fields` | also score the extracted fields — see below |
| `--fields-only` | score the fields and skip the transcript diff and accuracy |
| `--from-truth` | extract from `solution/<id>.md` instead of from a transcript, so pass 1 contributes nothing to the pass-2 score |
| `--no-extract` | re-score the saved `solution/out/<id>.fields.json` without extracting again |
| `--mode single\|agentic` | the extraction shape for this run, instead of whatever the app is set to |
| `--profile typhoon\|dots` | the pass-1 shape for this run: which OCR prompt is sent, and whether a system message goes with it |
| `--model NAME` | switch the app to this model before the sweep. Any unique substring of a served name is enough (`--model dots.ocr`); an ambiguous or unknown one is refused, with the served names listed. Ollama only — `llama-server` serves the one model it was started with |
| `--server URL` | switch the app to this model server before the sweep, e.g. `http://127.0.0.1:11434` |

The server and model in force are printed above the first case whether or not either flag
was given, and the same pair is written to every run-log row. Neither flag has an effect on
a run that makes no model call (`--no-run` on its own, `--no-extract`); that is said rather
than done silently.

Each case reports:

| Metric | Meaning |
|---|---|
| `char accuracy` | **how much of the ground truth the transcript actually produced** — content characters of the page that came back, as contiguous runs, over the content characters the page prints. Missing and misread text costs; **text the transcript has and the page does not costs nothing** (it is counted separately as `invented`, and printed beside the score). Until 2026-09-04 this was 1 − character edit distance / length, which charged extra text the same as lost text; the rule changed because pass 2 is what the number is for — a value the read dropped cannot then be extracted, while a line the page does not print leaves every real value where it was. `word accuracy` has always been computed this way, so the two are now consistent. **It scores content only** — every invisible character is removed from both sides first, so line breaks, blank lines, indentation, cell padding, tabs, non-breaking and other Unicode spaces, and zero-width marks cannot change it. So is every piece of table markup: `<table>`, `<tr>`, `<td>`, `<th>` and their `colspan`/`rowspan` attributes, `<br>`, pipes and the `\|---\|` separator row all come out, leaving the cell text. A table scores on what it says, not on how it was marked up. The `--- page 2 ---` separator this app puts between the pages of one document goes too, and so does a printed blank waiting to be filled in — a run of four or more dots, underscores or dashes, which is how a model spells the ruled line after วันที่/Date or เช็ค หมายเลข on a form. A rule carries no content, and a hand transcription writes none. Only the characters a reader would get information from are compared. |
| `word accuracy` | longest-common-subsequence word overlap. Whitespace does not count here either, but re-ordering does: a value read correctly in the wrong place lowers this and not `char accuracy`. |
| `char acc. w/o marks` | `char accuracy` with Thai tone marks and above/below vowels stripped. If this is much higher than `char accuracy`, the right letters are being read and the marks lost — a different problem from wrong words. |
| `numbers`, `Thai`, `English` | the same measure over one script at a time, and each prints the count it is out of. `numbers` is digits in Arabic or Thai numerals — amounts, dates, tax IDs, document numbers; `Thai` is Thai letters with their marks and vowels; `English` is the Latin alphabet, which is what the letters are and not a claim that the words are English. **They do not add up to `char accuracy`**: punctuation, symbols and currency signs belong to no script and are in the headline only. A page printing fewer than 20 characters of a script reports no rate for it and says how few there were — three characters score 0% or 100% and nothing in between. In the SUMMARY table the three means cover only the documents that print enough of that script, and a line under the table says which. |

…followed by a unified diff of expected vs actual, listing **differences in content**
only. A group of lines whose two sides hold the same text differently wrapped is
folded out of it, because where the model put the line breaks and the spaces does not
move the score either. An empty diff therefore means the content matches, not that the
two files are byte for byte the same.

On Windows, Thai in the diff prints as `?` — the console defaults to cp1252 and cannot
encode it. The scores are unaffected. For readable diffs:

```bash
set PYTHONIOENCODING=utf-8
```

**The scores are only as good as `solution/*.md`.** Those files are one person's reading of
the rendered pages — correct them where you disagree. `sol003` is a handwritten scan and its
handwritten values in particular deserve a second look.

### Flagging what is wrong: value, number, line, character

`compare.py` says how well each document was read. `ocrflag.py` says *what* is wrong with
it, leading with the part that decides whether the document is usable — **did the values
survive the read**. The app must be running, the same as above.

```bash
python ocrflag.py
```

```bash
python ocrflag.py sol003
```

```bash
python ocrflag.py --no-run
```

It reads each case with pass 1 only — no fields are extracted — saves the transcript to
`solution/out/<id>.txt` like `compare.py` does, and prints:

| Section | |
|---|---|
| `FILES` | one row per document, **most lost first**: character accuracy, values lost, numbers lost and invented, and `LOOPED`/`TRUNCATED` where the read did not finish. A row is marked `FLAG` when it lost a value or a figure, scored under 90%, or did not finish |
| `VALUES` | every key/value `solution/<id>.fields.json` states — names and numbers included — and whether the transcript still holds it, with what the read has in its place: `buyer_tax_id '0155737222723' read has '015573722723'`. The test is the one `grounding.py` uses, so a value reported lost is exactly a value pass 2 could not have grounded, however good the extraction model is |
| `NUMBERS` | every figure the page prints, compared **by value** — so a figure printed one way and read another is not flagged — listing the ones the read lost and the ones it invented. Figures under three digits are not counted; a `1` occurs on every page |
| `--text` | the line and character differences: `missing` / `invented` / `misread` / `moved` lines with the file and line to open (`sol003.md:31`), the differing character spans under each, and a corpus-wide split into `digits`, `marks` (Thai tone marks and above/below vowels) and `text`. Off by default — most of it is words in prose, which cost a point of character accuracy and no field |

| Flag | |
|---|---|
| `--no-run` | flag the saved `solution/out/<id>.txt` instead of reading again |
| `--text` | also list the line and character differences |
| `--all` | list every value, not only the ones that were lost |
| `--model NAME` | model to read with, any unique substring. Defaults to typhoon |
| `--profile typhoon\|dots`, `--server URL`, `--detail`, `--app URL` | as `compare.py` |
| `--keep-tables` | flag table markup and fill rules literally too |
| `--top N` | findings printed per case (default 25) |
| `--json PATH` | write every value, figure and character span as JSON |

Nothing that costs the score nothing is flagged: whitespace, line wrapping, cell padding,
table markup and printed fill rules are removed from both sides first, a line the model
split or ran together is folded away, and content the model emitted in a different order is
reported as `moved` and left out of the totals.

**`VALUES` and character accuracy rank the documents differently, and that is the point of
having both.** A page can read at 93% and have lost the one figure somebody needs, and a
page can read at 64% because the model paraphrased a paragraph of payment terms while every
value on it survived intact.

### Field extraction accuracy

The transcript score says how much of the page was read. It says nothing about whether the
right value ended up in the right field, and neither does the grounding check — that only
asks whether a value appears somewhere on the page, so a customer code returned as the
buyer's name passes it.

`solution/<id>.fields.json` is what answers that. One file per case, hand-written, holding
the value that belongs in each of the 14 extracted keys. **All five ship filled in** — every
key is stated, so an empty one means *the page does not print this* and is scored as such.

These files are maintained and verified by hand, and they hold **values and nothing else** —
no notes, no rationale, no metadata. The notation is: `""` the page does not print this,
`"text"` it prints this exactly, `[ … ]` it prints this in more than one way and any of them
is correct, `null` not checked yet. Create a file for a new case with:

```bash
python fieldscore.py init
```

That writes an empty file for any case that has none, and **never overwrites one that
exists** (`--force` does, and there is no undo — it would discard the hand-written files that
ship). A new file starts every key at `null`; the states are the whole format:

| Value | Means |
|---|---|
| `null` | not checked yet — left out of every count, so a half-filled file scores its filled half |
| `""` | the document does not print this. A value here is scored as **spurious** |
| `"text"` | the document prints this. Copy it exactly as printed — same digits, same separators, same language |
| `"-"` | the document prints a dash where a figure would go. Either a dash or an empty answer counts as correct, which is what the extraction prompt asks for |
| `["…", "…"]` | the document prints this value in more than one way and a reply matching **any** of them is correct |

A list is for a value the page states twice — a name printed in Thai on the letterhead and
again in English below it, a heading printed in both languages on adjacent lines. Keep it to
one or two readings and make sure every one of them is printed on the page: a list long
enough to catch anything is a key that is no longer being scored. Where a key has several
readings, the score reports the one the answer came closest to, so *expected* beside the
answer is always the reading it was judged against. The Fields tab names the others after it —
under a wrong value as *Ground truth: X — or "Y"*, and under a **correct** one as *Also
accepted: "Y"*, because a green mark on a key with two readings otherwise looks exactly like a
green mark on a key with one, and the reader cannot tell whether the other language would have
been taken.

**The line-item table is not in this file.** It is read out of the Markdown table in
`solution/<id>.md`, which already transcribes those rows — each column is matched to one of
the eight line-item cells by its heading, totals rows printed inside the table are dropped,
and a cell no column maps to is expected empty on every row. Every run prints what it mapped:

```
rows from sol005.md: description ← รายการ Description, period ← ประจำงวด Period, …; 2 row(s) dropped (totals row)
columns not scored, no key matches their heading: ลำดับ, เลขที่ใบแจ้งหนี้
```

If a heading is not recognised, name it yourself — `"table_columns": {"จำนวนเงินรับ": "amount"}`
— or set `"score_table": false` to leave the table out of the score. `other_fields` is still
`null` until you fill it; a list means it is scored. The file carries a `_readme` block
restating all of this, and keys starting with `_` are ignored by the scorer, so notes to
yourself are safe there.

**Each file opens with a `_mandatory` note** saying which types the case is, which of its keys
the requirement demands (`required`) and which it asks for without demanding (`found_only`) —
so whoever is filling one in can see at a glance which keys have to be answered. It is a note
and never values: nothing is scored from it. The rule itself lives in `prompts.MANDATORY_FIELDS`,
and a note that has drifted from it is reported as a warning on every run that loads the file —
fix it there, not here.

Comparison is loose about presentation and strict about content: punctuation, spacing and
Thai digits are normalised away and figures are compared by value, so `1,200` and `1,200.00`
score equal. You do not have to guess how the model will write a number.

Then:

```bash
python compare.py --fields
```

```bash
python compare.py sol005 --fields --from-truth
```

`--from-truth` feeds `solution/<id>.md` into extraction instead of reading the page, so pass 1
contributes nothing to the score. The **Fields only** pane does the same for one document at a
time, and lets you change model between runs.

**The headline is taken over the fields the requirement marks Mandatory.** A field it marks
Yes is scored; a field it marks No is asked for, returned, grounded, judged against the truth
file and shown on the page with its verdict — it is *found*, and it does not move the rate.
Nobody is held to an Optional field, so a document that leaves one out is compliant and must
not read as incomplete. What that covers follows the type: eleven keys on an invoice, ten on a
credit note, seven of a receipt's eleven. The Optional ones are reported beside the headline
as their own rate, and every run prints what it was scored over — `scored over: the
requirement's Mandatory fields` — because 100% of nothing and 100% of eleven are the same
number and not the same claim.

**A document no requirement covers yet is scored over the base field set instead, and marked
`unknown type`.** A bare tax invoice, or a page nothing could classify, has nothing Mandatory
— so its headline is taken over every key its truth file states, and everything that prints
that rate says what it is over: the run prints `scored over: every key asked for -- no
requirement covers this document type yet`, the Classify box carries an **unknown type**
badge, and the bar on the Accuracy tab counts *stated* values rather than *required* ones. It
is not a compliance figure: nothing on such a document is marked `REQUIRED`, no extra
validation rule runs, and `validate` demands nothing of it. It is what the extractor found.

| Metric | Meaning |
|---|---|
| `field accuracy` | correct values ÷ **Mandatory** values the truth file says the page prints — or, on a type no requirement covers, ÷ every value it states. **The headline.** |
| loose | the same, counting a partial match — one value contains the other, so the right thing was found and too much or too little of it was taken |
| `field precision` | correct ÷ everything the extractor filled in. Falls when it invents values the page does not state |
| `optional fields` | the same accuracy over the fields the requirement marks No. Reported, never in the headline; a row of the miss table that belongs to one is marked `(optional)` |
| `by tier` | the same accuracy over each delivery tier. Pass 2 extracts priority 1 only, so tier 1 is the whole schema and tiers 2 and 3 are empty |
| `line items` | how many rows were matched, returned and invented, and the accuracy of the cells inside the matched ones — against the table read out of `solution/<id>.md`. Rows are matched by content, not by position, so one dropped row does not mis-score every row after it |
| `other fields` | reported, and deliberately **not** part of the headline: those labels are the model's own wording |

Under the numbers is a table of every value that was wrong, missed or spurious, with what was
expected beside what came back. The summary at the end totals by value rather than by case,
so a file covering three keys does not weigh as much as one covering two hundred.

The same score appears on the page whenever the document read is a case with a field truth
file, in three places:

- the **Accuracy** tab — a fourth bar, *Field extraction*, beside the three transcript bars,
  with one line under it saying how many values were correct, wrong, missed and spurious;
- the **Fields** tab — the same headline, the list of every value that missed, and which
  column of the `.md` table was taken for which line-item key;
- the **Run log** card — the *Grounded · fields* cell carries `56% of 39 fields` under the
  grounded percentage. **That number does not read down the column**: its denominator is
  whatever that document's truth file rules on, so the count of values is always printed with
  it, and a row for a document with no truth file shows nothing there rather than 0%. Both are absent while the truth file is
still all `null` — a bar reading 0.0% because nobody has filled it in would be a score, and
there is no score. It reaches the run log as `field_acc` and `field_expected`.

---

## Random test

The **Random test** pane runs the whole pipeline on settings nobody chose: a ground-truth
document, a random model to read it, a random model to extract from it, a random Detail and a
random extraction shape. Set how many runs you want and press **Run random test**.

**It is the only thing here that tests the whole path.** `compare.py` and the field sweeps feed
pass 2 a ground-truth transcript on purpose, so a field score measures the extractor and nothing
else — which leaves decoding, trimming, resizing, the pass-1 profile, the case match and
extraction-from-a-real-transcript untested by anything automatic.

| | |
|---|---|
| **What each round runs** | how much of the pipeline to exercise &mdash; the three scopes below |
| **Runs** | 1 to 50. A round is a real read plus a real extraction, so tens of seconds each |
| **Seed** | leave blank for a new one; the seed used is put in the box when the run starts. It fixes the models, Details and shapes, and their order; the documents are chosen from the run log as it stands, so a seed replays the same plan exactly only while the log has not moved |

**Exclusions.** Every served model appears as a chip under **Models that may read** and
**Models that may extract**; unticking one takes it out of the draw. The two lists are
separate because a model can be poor at one pass and good at the other. Exclusions apply to
contests as well, `""` (same as reading model) is never excluded, and emptying a pool the
run actually needs is refused with a reason. On the CLI: `--exclude-reader MODEL`,
`--exclude-extractor MODEL`, repeatable.

**Locks.** Any of the document, the OCR model, the extraction model and the extraction
shape can be pinned instead of drawn, from the four pickers above the Run button. A locked axis is that axis
taken out of the experiment — locking the document is how a model comparison stops being a
document comparison as well. A lock this endpoint cannot honour (an unserved model, a model
pass 2 may not run on here, a document with no ground truth, a shape that is not `single`
or `agentic`) is **refused with a reason** rather than quietly ignored, and the locks are
printed beside the seed when the run starts. A read-only run has no extraction, so its
extraction-model and shape pickers are hidden.

**Three scopes, one per run:**

| | |
|---|---|
| **Full** | read the page, then extract from what came back. Both passes, and the field score is therefore partly a measurement of the read: a value pass 1 got wrong cannot then be extracted right |
| | **A read that failed &mdash; looped, cut off, or empty &mdash; or that scored under the read floor does not get its fields scored.** The extraction still runs and the row is still written with what came back (keys filled, grounded ratio, extra fields, timings); the correctness columns are left blank, and the round says **unscored** with the reason. A low score is **not** a failed run — it is a run, and it counts in the pass-1 mean. **Score fields only above** on this pane and on the Workspace pane are one setting; set it to `0` to score every extraction whatever the read did |
| **Read only** | pass 1 and stop. No extraction, no field score, and the row it logs has blank pass-2 columns |
| **Fields only** | pass 2 alone, on `solution/<id>.md`. No page is read and no image is made, so what comes back wrong is the extractor's &mdash; the same thing the Fields pane does, and the shape every pass-2 measurement was taken under. Any served model can be drawn, vision or not, and it runs as the one model in force |

They are separate runs rather than a fourth thing to randomise: the three answer
different questions, and a run that mixed them would report a mean over two of them.
Every round says which it ran, and a pass that did not run reads **not run** rather than
zero.

A fields-only run can leave a text-only model selected when it stops. The Workspace pane
will refuse to read a page with it until something else is picked &mdash; the settings are
left where the last round put them, as below.

**The rounds appear on the main page, not in this pane** — the controls decide a run, and
the rounds take the **Result** card's place while they are on screen (Compare goes with it,
since it holds that read's page images). A round is five columns wide and a run is up to
fifty of them, which a 380px pane cannot show without clipping the numbers the feature
exists to produce.

The two never share the screen, and the Read button is disabled while a test is running.
**Clear** on the card — or simply reading a document by hand — puts the Result card back;
nothing is lost, because every round is a real run and is in the run log.

### Contest — re-run the top 5 and bottom 5

The **Run contest** button in the same pane takes the ranking from
[Standouts](#standouts) and runs the top 5 and bottom 5 models again, **on the same
documents, at one fixed Detail** (`medium`), with the extraction shape the app is
currently set to. The only thing that varies is the model.

That is the opposite of what the random test does, and deliberately so: a random plan gives
every round its own Detail and its own page, which is what makes it good at finding failures
and useless for settling an argument about which model is better. A contest is the rematch.

| | |
|---|---|
| **Contest between** | what the contest is about: **OCR model**, **extraction model**, **document**, or **reader + extractor** pairs. The subject decides the scope too — a reader contest that also extracted would be scored partly on the other pass — so the scope picker above does not apply to a contest |
| **Top** | how many from each end of the ranking |
| **and the bottom** | off runs the leaders alone. The two halves answer different questions: the top is *which should I use*, the bottom is *is this really as bad as the log says* |
| **Documents** | how many documents each contender runs, 1 to 10 — the least-run ones, so a contest also spreads its rounds. Hidden for a **document** contest, where the documents are the contenders |
| who enters | only entries that have a **score**, and only those this endpoint still serves. Anything ranked but unserved, and anything unranked, is reported and skipped |

A **document** contest reads every contending document with the model the log currently
ranks first. A **reader + extractor** contest runs every top reader against every top
extractor, and is trimmed from the bottom of each ranking if the pairing count would not
fit in 50 rounds.

Where fewer than ten models are ranked, every one of them runs and the run says so — a model
that is in both the top five and the bottom five is tagged as neither.

The whole plan appears as soon as the run starts — every round `queued`, the current one
`running`, then filled in with the transcript score, the field score (a partial counting half),
how many extra fields came back, and the clock. A round that fails is marked and the run
continues: stopping at the first failure would find one problem per invocation.

**The field score is the same one everything else reports** — the fields the requirement marks
Mandatory for that document's type. A round prints `5+0p/7 req, 1/2 opt`: five of the seven
required values correct, and one of the two the truth file rules on that the requirement only
asks for. The optional half appears only where there is one, so an invoice — every key of
which is Mandatory — shows just `9+0p/10 req`. A type no requirement covers reads `12+0p/13
base` instead: the denominator is the base field set, not a requirement.

Two things worth knowing:

- **Every round is a real run and appends its own row to the run log**, so its results feed the
  same tables as everything else. A random test whose rows were kept out would be measuring a
  path nobody else uses.
- **The settings are left where the last round put them**, not restored. A combination that
  broke something is still selected when the run stops, which is what you want to look at it.

**The document is the one thing that is not random.** Each round goes to whichever case the run
log has read fewest times, ties broken at random, counting the rounds planned above it as well —
so the runs spread over the fixtures instead of piling onto whichever one keeps being drawn, and
a document nobody has run yet is picked first. The number beside each case in the table is how
many reads it already had, which is why it was chosen. Re-extraction rows are not counted (they
read no page) and failed reads are (the document has had its turn).

Only documents with **both** a transcript truth and a field truth are used — a round that can
report neither number only proves the request did not crash.

**Which models each pass may draw on:**

- **a reader is any model that reports vision.** Pass 1 sends an image, so a text-only model is
  not a candidate; nothing else is excluded. A fields-only round has no reader at all.
- **an extractor is the reading model itself, or a model that is not an OCR fine-tune** — the
  same rule the server enforces, so a plan never contains a round it would refuse.

A general vision model can therefore be planned as a reader. That is deliberate: it is how the
run that read a fixture at 98.5% was found. What stays narrow is the **default** — with nothing
chosen, the app resolves an OCR model, so a general model reads a page because something picked
it and never because it was the newest pull.

The same thing runs headless:

```bash
python randomtest.py http://localhost:5000 --rounds 10
```

```bash
python randomtest.py http://localhost:5000 --seed 1787218747
```

```bash
python randomtest.py http://localhost:5000 --scope fields --rounds 10
```

```bash
python randomtest.py http://localhost:5000 --contest --documents 2
```

## Run log

Every document read appends one row to `logs/runs.csv` — from the page, from the queue, from
`/api/ocr`, and from `compare.py`. The **Run log** card at the bottom of the page shows the
recent rows and links the CSV.

**Re-extracting appends a row too.** Every press of **Re-extract**, and every call to
`/api/extract`, writes its own row with `run_type` = `extract`, and the card refreshes itself
when the extraction finishes. The row carries the extraction's own figures — mode, seconds,
tokens, grounding, tier coverage — against the document the transcript came from. Its pass-1
columns (`pages`, `seconds`, `tokens`, and the accuracy scores) are **blank**: nothing re-read
the page, and copying the earlier row's numbers forward would count one read twice in every
total.

**A better re-extraction is also written back onto the read's own row.** If it scores higher —
more priority-1 values **correct**, then more priority-1 keys filled, then a higher
`grounded_pct` — its figures replace the pass-2 columns of the row for the read it came from,
and `extract_updated` records when that happened. 9/14 correct becomes 11/14; 11/14 is never
pulled back down to 9/14, and an extraction that failed can never displace one that ran.
Correctness ranks first and coverage second, so a run that filled every key with whatever was
nearest does not outrank one that filled fewer and got them right; a document with no field
truth file has no correctness figure on either side and is ranked on coverage, as it always
was. The read's own columns are never
touched, and the re-extraction keeps its own row regardless, so the history stays complete and
the read row reports the best extraction of that transcript rather than whichever one happened
to run first. The **Fields** cell marks an updated row.

**It holds measurements only.** No transcript, no extracted fields, no page images:

| Column | |
|---|---|
| `timestamp` | local time, seconds resolution |
| `file`, `file_size_mb`, `pages`, `detail`, `source` | what was read, and how it got in (`upload`/`folder`/`case`/`queue`) |
| `server`, `backend`, `model` | which endpoint and model actually ran it |
| `seconds`, `prefill_seconds`, `decode_seconds` | runtime, split into prompt processing and generation. The card shows the split under the total, because the two move for different reasons — prefill scales with pixels, decode with output length |
| `tokens`, `tokens_per_second` | OCR output tokens; the rate is decode-only |
| `extract_seconds`, `extract_tokens` | pass 2 |
| `extract_mode` | `single` or `agentic` — the shape pass 2 ran in, taken from the result, so a mode switched mid-batch still labels each row correctly. Blank on a run that never extracted |
| `extract_steps` | the agentic steps this row's extraction ran, where it ran only some of them (`POST /api/extract` with `steps`). Blank on every ordinary run — a full walk names no steps here. A row that names steps was measuring one step: its tier counts are out of the keys that step owns, and `field_acc` is blank because a part of the form is not scored against the whole of it |
| `ocr_profile` | `typhoon` or `dots` — the pass-1 shape that read the page, taken from the pages themselves for the same reason. Blank on rows written before profiles existed, and on `run_type=extract` rows, which read no page |
| `grounded_pct`, `ungrounded`, `fields_missing` | share of extracted values found in the transcript, how many were not, and how many fields the document does not state |
| `field_acc`, `field_expected` | pass 2 scored against `solution/<id>.fields.json`: the share of the values that came back correct, and how many values that was. Blank on every document without a field truth file, so it does not read down the column like `grounded_pct` — read the accuracy beside its own `field_expected`, because 100% of three keys and 100% of thirteen are the same cell and not the same claim |
| `other_fields`, `other_distinct` | how many entries came back under `other_fields` — everything the page states that the document type's own field set does not cover — and how many of them were **distinct**. A count only; the labels are the model's own wording and stay out of the file, like the transcript. The pair is the point: 104 against 5 is a loop, 12 against 12 is a document. Blank where nothing was extracted, `0` where the extraction ran and named none |
| `extract_model` | the model pass 2 ran on, where it is **not** the one that read the page. Blank on the one-model setup, which reads as "the same as `model`" rather than as unknown |
| `extract_looped` | `1` when the same entry came back three or more times — a reply that cycled rather than a document with a lot on it. Such a run is dropped from the extras ranking rather than allowed to win it. It counts as a **failure** only where the cycle cost the form: `other_fields` holds what the type's field set does not cover and nothing scores it, so a run that filled the form and then cycled through the extras is a complete extraction with a short list. Where `p1_present` is `0` the reply cycled away the form itself, and that is a failure. Blank where nothing was extracted, `0` where the extraction ran cleanly |
| `p1_correct`, `p1_partial`, `p1_scored` | priority-1 values that are the value the field truth file says belongs in that key, values where one contains the other, and how many values that file rules on. **Correctness, not coverage** — `p1_present` counts a key filled with anything at all. The setting table scores a partial as **half a value**; `p1_correct` keeps the strict count, so both readings survive here. Blank on every document without a field truth file, for the same reason `field_acc` is: a `0` would read as an extraction that got everything wrong rather than as one with no answer sheet |
| `p1_present`, `p1_absent`, `p2_present`, `p2_absent`, `p3_present`, `p3_absent` | field coverage by delivery tier — how many of each tier's keys came back filled. Pass 2 extracts priority 1 only, so `p2`/`p3` read `0/0`: this build asked for none of them, which is not the same as a run that extracted nothing and leaves them blank. Present and absent are both written, so a row always says what its counts were out of — and since the form is per document type, that sum is 11 on a commercial form, 14 on a withholding certificate, and 30 where nothing classified the page |
| `doc_types`, `doc_type_from` | every type pass 2 built the form from, `+`-joined, and on whose authority: `case` (the benchmark manifest), `model` (read by the model and checked against the page), `transcript` (the fallback — matched in Python from the printed heading), `caller`, or blank. Read it beside `p1_present`/`p1_absent`, which are counted against that form — two rows reading 11/13 and 13/15 are both complete runs of different forms |
| `case`, `char_accuracy`, `word_accuracy`, `char_accuracy_no_marks`, `invented_chars` | percentages, blank when the input has no ground truth. **`char_accuracy` is the score** — content only, and order-blind: the transcript's blocks are matched to the ground truth's by content first, so a page read correctly but walked in a different order is not charged for it. Missing and misread content cost; **extra content does not, since 2026-09-04** — `invented_chars` counts it instead, and is blank on every row written before that date, which is every row whose `char_accuracy` was an edit distance — an exact test for which era a row belongs to. **The log was not reset for the change**: every table covers each setting's own most recent runs, so the older rows leave on their own as new ones arrive, and the pass-1 table says how many are left until they are gone. Until then, a mean mixing the two is a mean over two questions. The other two stay for diagnosis and are not shown on the page: `word_accuracy` is order-*sensitive*, so the gap between the two is what a reordering looks like |
| `thai_accuracy`, `latin_accuracy`, `digit_accuracy`, `thai_chars`, `latin_chars`, `digit_chars` | the same transcript recall taken over one script at a time, and how many characters of each the ground truth holds. `digit_accuracy` is the one to read first: every Mandatory field in the requirement is a figure, a date or an ID, and a page can score 95% overall while losing the digit that makes an amount wrong. `latin` is the Latin **alphabet** — a romanised Thai company name is Latin script and is not English. **The three do not add up to `char_accuracy`**; punctuation and symbols belong to no script and are in the headline only. A rate is blank where the page prints fewer than 20 characters of that script — the count beside it says why — and on every row written before 2026-09-04 |
| `field_verdicts` | which **field** the extraction got right, as `key=letter` pairs joined by `;`. The letters are the six field-score verdicts: `c` correct, `p` partial, `w` wrong, `m` missed (the page states it, the extractor returned nothing), `s` spurious (the page states nothing, the extractor filled it anyway), `a` absent (both empty — agreement, scored neither way). `m` and `s` are opposite mistakes and neither means "empty". Field names only; no value ever reaches this file. Blank wherever `field_acc` is blank, and on rows written before 2026-09-04. This is what the **Weak spots** panel is compiled from |
| `status`, `error` | `ok` / `partial` / `truncated` / `looped` / `cancelled` / `error` |
| `run_type` | `ocr` for a document read, `extract` for a re-extraction of a transcript already read. Blank on rows written before the column existed |
| `extract_updated` | set when a later, better re-extraction replaced this row's pass-2 columns, so `timestamp` no longer says when they were measured. Blank on the normal case |

Coverage is not correctness. `p1_present` counts what came back filled, not what came back
right — read it beside `p1_correct` where the document has a field truth file, and beside
`grounded_pct` where it does not. Each row's **Fields** cell also carries the shape that
filled it, `single` or `agentic`: one request and seven fill the schema in different ways,
so two rows of counts are not comparable without it.

The card header reports the **mean** accuracy over every scored row and the **best** single
score in the file, and above the table is the best each document has ever reached — over
every run and every setting, with the model, backend, detail and mode that reached it on
hover. The mean moves with whatever was being tried lately; the best says what the document
is known to be capable of, which is the number to beat.

### Light or dark

**Auto / Light / Dark**, top right of the page. `Auto` follows your desktop and is the
default; the other two override it in either direction and are remembered across reloads.
Pick **Light** for projecting or screenshotting — it is what the Summary panel is meant to be
shown in. Light is a soft grey ground with an off-white card rather than white-on-white, so it
does not glare on a projector.

### Reading the card: nine panels, sortable, filterable

The card is nine panels, and only the last of them is the log:

| Panel | |
|---|---|
| **Summary** | the six tables worth showing, at presentation size, with two banner cards above them. Opens by default |
| **Full rank** | best and worst per model and per document, for each pass. The chips above it are each document's best score |
| **Best reading** | pass 1 per setting — which model, backend, Detail and profile to read a page with |
| **Best extraction** | pass 2 per setting — which model and shape to extract fields with |
| **Per document** | one row per ground-truth document: the best transcript, the best fields, the quickest complete run |
| **Weak spots** | not how good, but at **what**: the transcript score split into Thai, English and numerals per model and per document, and the field score split into the individual keys — a model × field grid and a document × field grid. Every other panel ranks on a mean and therefore averages exactly this away |
| **Time × Doc × Accuracy** | pass 1 only: read time against the document against the transcript score, with the Detail tables and the outlier list |
| **Errors** | what is failing, ranked on the failures rather than folded into an accuracy |
| **Raw data** | the rows of `logs/runs.csv` themselves — unfiltered, and the only place a run can be deleted |

**Every column sorts**, including the one each table is ranked by. Click a header to sort
descending, again for ascending. The highlighted row stays the best by the table's own
ranking however you sort it, so re-ordering never re-labels the winner. Rows with nothing to
sort — *not scored*, *no complete run* — go last in both directions: no score is not a low
score.

#### The Summary panel

Built for showing — projected, or read from across a room — so everything on it is a size up
from the rest of the card. Two banner cards, then six numbered tables.

Block 1, *which model reads a page best*, carries **Numbers**, **Thai** and **English** beside
the transcript accuracy: the same score over one script at a time, numbers first because every
Mandatory field in the requirement is a figure, a date or an ID. They are recall over three
subsets of the column to their left and do not add up to it. **Weak spots** has the same split
per document, and the per-field one.

**The headline cards** answer the two questions the tab exists for, at a size that needs no
leaning in: **best at reading a page** and **best extractor**. The extraction card names a
model *and* a request shape — `agentic` or `single`, on a pill — because the two are not two
samples of one setting and a figure pooled across them describes neither. Only the winning
shape is shown; block 5 has the full grid for the other. Each card carries the model, the
percentage, the clock, its error rate and what it was measured over, and says *nothing scored
in this view* if a filter has emptied it rather than disappearing. They name the *ranking's*
winner and never the sorted table's top row, so re-sorting a table by time cannot re-label the
fastest model "best".

**The environment card** is three columns, and the split is the point — none of them implies
another:

| Column | |
|---|---|
| **This machine** | GPU, CPU, memory and OS, detected on the box this app is running on now. On NVIDIA hardware the card, its memory and the driver come from `nvidia-smi`; anything else reads *not detected*, and the probe is named so you can tell "no GPU" from "could not ask" |
| **Server now** | the endpoint, models, context and defaults this process *would* use for the next run |
| **These runs** | what the rows in view were actually made under — dates, backend, models, Detail, and the inferred hardware and warm/cold split. **Only this column describes the numbers below**, and it narrows with the chips like every table |

The log outlives the other two: a row read on a different machine, or against a server since
switched, is still in it and is still counted. So the card never says *these runs used this
GPU* — nothing in this project can. Hardware and warm start are **inferred** from the decode
rate and the prefill, and are labelled as such wherever they appear.

**The card never contacts the model server.** It reports the endpoint *as last seen* and says
*not probed yet* if nothing has asked — the panel refreshes itself every few seconds, and
polling a llama.cpp server cancels work in flight. Press **Re-check** beside the server picker
to refresh it.

**Colour carries the comparison**, not just the numbers: every figure has a magnitude bar
under it, comparable **down its own column** — accuracy against 100%, the clocks against the
slowest in that column. The error column is the one place a longer bar is worse, which is why
it is the only one drawn in red. The Detail grids tint each cell too: by the score in the
accuracy grid, and by how many times that row's cheapest preset it cost in the time grid. Rank
badges beside each model are the **ranking's** order and do not renumber when you sort a
column.

Then the six tables:

| | |
|---|---|
| **1** | which model reads a page best — transcript accuracy and error rate |
| **2** | what each reader costs — time per document, decode rate, and average tokens generated |
| **3** | what raising **Detail** buys and what it costs, as **two** model × preset grids — accuracy in one, time in the other |
| **4** | which model extracts fields best — field accuracy and error rate |
| **5** | **single against agentic**, per model, with the extraction clock |
| **6** | what each extractor costs |

Three things are deliberately different here from every other panel:

- **One error column, and no failure flags anywhere else.** The other tables print a red
  incomplete count beside the setting, because there it qualifies every figure on the row.
  Here the error rate is a column and nothing else.
- **Its own colour scale**: red under 70%, amber to 90%, green at 90% and above. Error rates
  are coloured on the same scale read from the other end, so 10% error is green and 30% is
  red. The rest of the page keeps its own scale.
- **The shared Drop single-source failures toggle applies here too**, like every other panel
  — so does every chip. What narrows the headline cards is: that toggle, the chips, and this
  panel's own **Exclude weak models**. None of them changes an accuracy mean on its own, since
  a failed run is never scored anyway; what they move is the error rate and the run count.
- **It shows everything by default**, like every other panel. Exclude models with the chips
  above, exactly as elsewhere — or tick **Exclude weak models** to do it by rule instead of by
  hand. A strip of chips then names every model that was dropped and why on hover; unticking it
  puts all of them back and the strip disappears.
- **No explanatory text.** Every caveat is on the heading's or the column header's tooltip
  instead, so the panel is figures, colour and headings — and the spread and range behind each
  mean are one hover away rather than printed under it.

Three rules make up that toggle, and all three are computed from the log rather than from a
list of model names, so the result changes when a model does:

| Rule | |
|---|---|
| **single-source failures** | failures that came entirely from one pairing — a document all of whose failures are one model, or a model all of whose failures are one document. Only the failing runs go; that model's clean runs stay |
| **weak models** | a model that delivers less than `PRESENT_MIN_SHARE` (**60%**) of what the best model *in that pass* delivers, or fails more than `PRESENT_MAX_FAILURE` (**40%**) of its runs. Judged per pass, on the same figures the tables print, and never on fewer than `PRESENT_MIN_RUNS` (**3**) runs |
| **time outliers** | a run whose clock is a robust outlier within its own cell — a cold model load or a runaway. **Only its time is discounted**; the run still counts and still scores |

Because the weak-model bar is relative, dropping the leading model with a chip can let a
previously excluded one back in: the bar moves with the field. That is the rule working, and
the banner will say so.

**Reset clears it**, like every other narrowing on the card.

#### Filtering

Under the controls is a row of chips per **document**, **reading model**, **extraction model**
and **extraction shape**. Click one to keep **only** that value, again to **drop** it, again
to clear. So *how does my best model do on sol001* is: keep `sol001`, drop the models you are
not asking about.

**A dropped row is absent, not hidden.** It is out of the run counts, the failure rates, the
means and the window, exactly as if it had not been run — which is the only reading under
which the question above has an answer. The chip counts are taken over the **whole** log
rather than over what survived, so a value you have just dropped is still there to put back.

**The Raw data panel is never filtered.** It is the log, and a row nobody can see is a row
nobody can delete. If a filter matches nothing, a banner says so and names what is in force —
the tables go empty, the log does not.

#### Half a pipeline

Some runs are half a pipeline on purpose: a read-only round (`--scope ocr`) never extracts, and
a fields-only round (`--scope fields`) is fed the ground truth and never reads. **Pipeline**
isolates them:

| | |
|---|---|
| **Any** | every run (default) |
| **Both passes** | read a page *and* extracted from it |
| **Read only** | read a page and never extracted |
| **Extract only** | extracted from a transcript it did not read |

The model chips reach the same place: turning off **every** reading model leaves the runs that
read no page, and turning off every extraction model leaves the runs that extracted nothing.
That works because **a model column is blank on a pass the run did not perform** — a fields-only
row carries the model that *extracted* in `model`, and it is not offered or matched as a reading
model, because it read nothing. So those pickers only ever list models that actually did the
job the picker is about, and the counts beside them are counts of that job.

Whichever route you take, a panel with nothing to show says why rather than going blank. One
caveat is printed where it applies: under **Read only** the pass-2 table is not empty, because a
read-only run that *looped* is counted as an extraction that never got started — a run that
never asked for pass 2 and one that died before reaching it are identical in the log, both
leaving the pass-2 columns and `extract_mode` blank. Those rows are pass-1 failures; compare
extraction under **Both passes** or **Extract only**.

#### The two knobs

| | |
|---|---|
| **Read floor** | transcript accuracy under which a field score already in the log is not **averaged**. Pass 2 can only map the values pass 1 gave it, so a field score over a broken transcript is the read's mistake wearing the extractor's name. Defaults to `MIN_READ_FOR_FIELDS` (0.75 → **75%**); `0` counts every field score whatever the read did. **This is the reversible floor of the two** — it re-derives trust from each row's own transcript score every time a table is built, so it can be moved back at any time. The floor on the Workspace and Random test panes decides what gets **written**, and a score never taken is not here to recover |
| **Runs averaged** | how many recent runs **of each setting, model and document** each table covers. Defaults to `SUMMARY_RUNS` (**50**); `0` is the whole log |

**Neither knob writes anything, and neither hides a run from the log.** Every run is stored
whatever they say: a read that scored 20% is written, is passed to pass 2, and is in the Raw
data panel — what a floor above 20% decides is that the field score it produced describes the
read rather than the extractor, so it stays out of the correctness figures. Raising the floor
is how you take propagated OCR error off the extractor's bill; the note beside the controls
says which values are in force and whether they are the server's own.

Both apply to **every row already in the file**, not only to runs made after they were
changed, so the tables always report the rule you are looking at.

#### Deleting a run

Tick rows in **Raw data** and press **Delete selected**. That is the one control on this card
that changes the file: the rows leave `runs.csv` and therefore every table, every mean and
every count.

**They are archived, not destroyed** — appended to `logs/runs.deleted.csv`, which is the only
way back. A row is identified by its position in the file, checked against its timestamp and
file name before anything is removed, so a page holding a stale list (a run finished while
rows were ticked) deletes nothing and says so rather than deleting a neighbour.

#### Nothing is logged while the model server is stubbed

A run whose reply came from a fixture is not a measurement, and the log has no column that
could say so afterwards. `runlog.record` therefore writes nothing while `requests.post` or
`requests.get` has been replaced — every model-server call in this app goes through those two
— and prints one line on stderr naming the fix. This is not hypothetical: a verification run
once wrote 52 rows under a model called `stub` which, because the fixture answered with the
ground truth, scored 100% and sat at the top of the pass-2 setting table.

To keep the rows from such a run, point `OCR_LOG_DIR` at a scratch directory, which is the
right thing to do for any experiment. To log anyway — for a wrapper around `requests` that is
not a fake — set `runlog.ALLOW_PATCHED_TRANSPORT = True`.

### Best setting per document

Under those chips, one row per ground-truth document answers *which setting should I use for
this page* three ways, because the log answers it three ways and they are rarely the same run:

| Column | |
|---|---|
| **Best fields** | the highest `field_acc` this document has scored, with the number of values it was scored over. This is pass 2's correctness — whether a value landed in the key it belongs in. Reads *not scored yet* where there is no `solution/<id>.fields.json`, or no run has been scored against one: blank is not zero |
| **Fastest full run** | the quickest run that finished, scored something, and ran **both** passes, split into read and extract. A read that skipped extraction is left out — it took less time because it did less work. Its accuracy is printed beside the clock, because fast is a claim about cost and nothing else |
| **Best transcript** | the highest `char_accuracy`, with word accuracy under it — the same figure as the chip above, in the same row as the other two |

Every figure carries the setting that produced it — model, backend, detail, pass-1 profile,
extraction mode — with the full model name and the timestamp on hover. A figure from a
re-extraction row is marked `re-extract only`: it scored pass 2 against a transcript some
earlier run read, so the detail and profile beside it are not what produced the number.

Two things to keep in mind reading it. **Fields and transcript are separate scores** — a run
can read the page almost perfectly and still put the values in the wrong keys, which is why
both are shown and why the fields column prints the transcript score under it. And **llama.cpp
caches prompts**, so re-reading a page it has already seen is faster than the first time under
any setting; a fastest time set on a repeat read is not a setting you can expect cold.

The same figures are in `GET /api/runs` under `totals.by_case`.

### Which setting to run at all

Under that table are two more, answering the question the per-document table scatters across
one row each: *which setting should I run*. **Two tables, not one** — pass 1 is judged on the
transcript and paid for in prefill and decode, pass 2 on whether a value reached the key it
belongs in and paid for in neither. One row averaging both answers neither question.

Both follow the same rule for repeats: **the runs of one document are averaged before the
documents are.** Two runs on `sol001` and one on `sol005` is not three samples of a setting —
pooling them weights `sol001` twice, and two settings being compared have rarely been run the
same number of times on the same pages.

Each figure carries two spreads, and they are different claims:

| | |
|---|---|
| **± repeat** | how far the setting moved when the same document was run twice, averaged over the documents that were repeated. Reproducibility. Under greedy decoding this should be near zero — a figure that is not is worth chasing |
| **± across docs** | the spread of the per-document means. How much the documents differ, which is a property of the fixtures as much as of the setting. **Not an error bar on the mean** |

Both read blank where there is nothing to take them over: one document, or no document run twice.

#### Pass 1 — best setting to read a page

| Column | |
|---|---|
| **Setting** | model, backend, Detail and pass-1 profile *together*. The same model at a different Detail is a different thing to run, and the profile decides the prompt and the system slot |
| **Transcript** | character accuracy against `solution/<id>.md`, with word accuracy under it. Word accuracy counts ordering and character accuracy does not, so a value read correctly in the wrong place shows in one and not the other |
| **Numbers**, **Thai**, **English** | the same score over one script at a time — digits, Thai letters, the Latin alphabet. Numbers first because every Mandatory field in the requirement is a figure, a date or an ID. They are recall over three subsets of the Transcript column and do **not** add up to it. Blank where the documents printed too little of a script to measure it |
| **Prefill** | vision tower plus prompt encode, paid once per page before the first token. This is the half that scales with Detail |
| **Decode** | generating the transcript, with the rate under it. The rate excludes prefill, so it compares across pages of different resolutions |
| **Total** | wall clock for the document |

A **failure rate** appears beside the setting when any of its reads did not finish. Two shapes
count, and they are named apart because they are found in different places:

- the run **said so** — `looped`, `truncated`, `cancelled`, `error`;
- the run said `ok` and returned an **empty transcript** — scored 0.0%. That failure has no
  other symptom: it is fast, it is clean, and only the accuracy column gives it away.

**A failed run is counted and never scored.** It raises the failure rate and stays out of every
mean — accuracy, word accuracy, prefill, decode and total — because a run that did not finish
did not measure anything. Read the two figures together: a high score beside a high failure rate
is one good run among several bad ones, which is precisely what a mean with the failures folded
into it could not tell you. `char` counts only the documents that produced a number, and a
setting whose every run failed shows no score at all.

The same rule applies to the per-document table above and to the **Pass 2** table: a run that
did not finish cannot hold a record or enter the extras ranking either.

**llama.cpp caches prompts**, so a prefill figure that includes a re-read of a page it had
already seen is lower than anyone gets cold. The log has no column saying whether a read was
warm, so this cannot be filtered — only known about.

#### Pass 2 — best setting to extract fields

Ranked on **field accuracy**: priority-1 values that are the value the ground truth says
belongs in that key, as a share of the values that file rules on, meaned over documents. A
**`partial` scores half a value** — one string contains the other, so the model found the right
thing and took too much or too little of it, which is neither a hit nor a miss. The truth files
state most of their type's field set, so the denominator is 11 to 15 and it is **per document**,
which is why this is a mean of per-document rates and never one big fraction.

The strict count (`correct` only) and the generous one (`partial` counting full) are both still
in the CSV and in the CLI report, because the gap between them is the diagnosis: a wide gap is
over-capture and truncation, a low pair with no gap is misreading.

| Column | |
|---|---|
| **Setting** | the model that **extracted**, backend and extraction shape. Where a different model read the page, it is named as `reading: …` — a field score taken over a real transcript is partly a measurement of pass 1. A **failure rate** appears here when any of its runs cycled away the form, or never replied. A reply that filled them and then cycled through `other_fields` is not a failure — see `extract_looped` |
| **Field accuracy** | the headline, with the raw counts totalled over the scored runs beneath it. Reads *not scored yet* where nothing this setting ran on has a `solution/<id>.fields.json` |
| **Extra fields, ranked** | `other_fields` scored against the other settings run on the **same document**, as points **per document** — see below |
| **Extra fields** | how many **distinct** entries came back under `other_fields`, meaned over documents, with the raw total under it where the two differ |
| **Filled** | how many of the 14 priority-1 keys came back filled with anything at all. Coverage, not correctness, and deliberately not what the table is sorted on: read it against the score. Many filled and few correct is a model inventing; few filled is one giving up, and they share an accuracy |

**Single and agentic are separate rows.** Some models do well in one shape and badly in the
other — the measured gap on one model is 2.5× — and it is not a gap a model carries with it, so
averaging the two reports a number neither shape produces.

**Everything else about a setting is pooled**, including what pass 2 was fed. A run fed
`solution/<id>.md` measures pass 2 alone; a run fed a transcript pass 1 produced measures *both*
passes, because a value the OCR misread cannot then be extracted correctly. They share a row, so
a setting whose runs are mostly real reads is being marked on its OCR as well as its extraction
— `source` in the CSV is what tells the two apart.

Restricted agentic runs (rows naming `extract_steps`) are left out: they answer part of the
form, so their counts are not a measurement of the setting on it.

##### How `other_fields` is ranked

**There is no ground truth for `other_fields` and there cannot be one** — the labels are the
model's own wording, and a page has as many extra fields as a reader decides it has. So it is
scored the only way an unscorable output can be: **against the other settings run on the same
document.** Three settings on `sol001` ranking B, C, A gives B +2, C +1, A +0, and the points
are summed over every document.

Points are *how many settings you strictly beat*, which is what makes ties behave: two settings
that returned the same count get the same points. A document only one setting has been run on
is skipped entirely and does not count towards the *of N* either — awarding 0 out of 0 there
would make a setting look weak for having had no opponent.

**The points are shown per document**, because the settings being compared have not entered the
same number of contests: +10 from one document and +26 from five are not the same achievement,
and the raw totals rank whichever setting was run most.

**A run whose reply cycled does not enter the contest at all.** Without that rule the column
rewards exactly the failure it should penalise — a reply that repeats one invented line a
hundred times returns 104 "extra fields" and beats every honest reply on the page. What is
compared for the runs that do enter is `other_distinct`, so repetition below the flag's
threshold cannot inflate a total either.

Being kept out of the contest is **not** the same as being counted as a failure, and a cycle
that cost only the extras is not one: the run filled the fourteen keys and is scored on them
like any other. It is barred from a ranking of *how much of the page came back* because its
count is inflated, which is a fairness rule about that one column.

Even so, **this column measures volume and volume is not quality.** Nothing here checks that
the extra fields are real; `grounded_pct` is the separate check for that, and it cannot catch a
loop, because repeated page text is still page text.

The same figures are in `GET /api/runs` under `totals.by_ocr` and `totals.by_extract`, best
first.

Failed and cancelled runs are logged too. Move the file with `OCR_LOG_DIR`. It is written as
UTF-8 with a BOM so Excel opens Thai filenames correctly; new columns are only ever
appended, so old rows stay readable.

**Every compiled figure covers the most recent runs of its own row** — up to
`SUMMARY_RUNS` (default **50**) of *that setting*, *that model* or *that document*, not the
last 20 lines of the file. Each table says which it is per — *recent runs per setting*, *per
document*, *per row* — and deliberately **does not print the number**: the window is a
ceiling, and a setting with three runs is summarised over three. What each row actually
covers is the count on the row itself (`N run(s)`, `2 of 7`); the ceiling is in the tooltip.
The chips read *best recently* rather than *best ever*.

Two reasons, and the second is why it is per row rather than a slice off the end. The log is
append-only across changes to the thing being measured — a scorer that starts counting
differently, a schema that grows, presets that are renamed — so a mean over rows from both
sides of one of those describes a build nobody is running. And a slice would let one busy
evening on one setting push every other setting out of the tables entirely.

The CSV keeps everything; `SUMMARY_RUNS=0` summarises all of it. The row table below still
lists the most recent 50 rows — it is the log, not a summary of it. The random test's
document-fairness rule ignores the window on purpose: a document read thirty rows ago has
still been read.

**To start the tables from scratch**, rename `logs/runs.csv` — the next run writes a fresh one
with the current header. Rows written before a column existed read blank in it, which is
correct but leaves them out of anything ranked on that column.

Because each row carries the server, the model and the accuracy together, this is the
straightforward way to answer "is 11434 actually better than 8080 on my documents" — run the
same cases on each and compare the columns.

#### Weak spots — not how good, but at what

Every other panel on this card ranks on a mean, and a mean is the wrong shape for *what is it
bad at*: a model reading 92% overall can be reading 71% of the digits, and a field score of
8 of 11 says nothing about **which** three. This panel is four tables, two per pass, and each
pass asks the same question of a model and of a document.

**Pass 1 — which script is lost.** One row per model, and one per document, each with the
transcript score and then the same score over numerals, Thai and the Latin alphabet, with the
count each rate is out of. The last column names the **weakest** of the three and how far below
the model's own best it is — a two-point gap is noise, a twenty-point one is a finding. Models
are keyed on the model **alone**, without Detail or profile: this asks what a model is bad at
across everything it has read, which is a different question from *which setting to run* (that
is **Best reading**). Documents are ordered **worst first** — the table is a work list, not a
ranking.

**Pass 2 — which field goes wrong.** A table of every key, weakest first, then a **model ×
field** grid and a **document × field** grid in that same column order, so a column means the
same thing in both. A cell is the share of that key the extractor got right, a partial counting
half — the same arithmetic the field score uses, so a cell here and the field accuracy of the
same run cannot disagree. Under a cell below 100% is the commonest way it fails: `wrong`,
`missed` (the page states it, the extractor returned nothing) or `spurious` (the page states
nothing, the extractor filled it anyway). **`missed` and `spurious` are opposite mistakes** and
want opposite fixes.

A cell is blank where nothing measured it — the key was not asked of that document's type, or
every verdict on it was `absent` (both sides empty, which is agreement) or `spurious` (nothing
on the page to be right about). The second is called out under the cell as *N invented*,
because on the keys this corpus leaves empty by construction that is the whole story.

Both halves need rows written on or after 2026-09-04, when the per-script and per-field columns
were added; an older run has a score and no breakdown, which is not a low score and is left out
rather than counted as zero. The pass-2 half also needs a `solution/<id>.fields.json` and a read
good enough to judge the extraction by — a field score taken over a broken transcript is the
read's mistake wearing the extractor's name, and the read floor above the card is what decides
that.

### Standouts

The top of the run-log card answers four questions the tables under it cannot put
into words: **which model and which document are carrying this, and which are
failing it** &mdash; once for reading a page and once for extracting fields.

Each list is ranked on

```
score = accuracy x (1 - failure rate)
```

which is what one attempt is worth: a run that failed counts as zero, so failures
outweigh a few points of accuracy. The accuracy printed beside it is taken over
the runs that **finished**, which is why the failures are always printed with it.

| | |
|---|---|
| **0.0** | runs failed and not one survived to be measured &mdash; a real result, and the worst there is |
| **not scored** | nothing failed and nothing was scored either: no ground truth, or a field score left off because the read was too poor to judge it by. Ranked nowhere |

Models here are grouped **without** their Detail, profile or extraction shape, unlike
the two setting tables lower down: those answer *which setting to run*, this answers
*which model is carrying its weight* across everything it has been asked to do. A
model can be worst here and still hold the best single row there, at one Detail.

A group with only one run is marked and cannot be the headline best or worst while a
better-evidenced one exists. And **a document's row is not a verdict on the
document** &mdash; a fixture every model reads badly may be the hardest page here, or
its hand-written ground truth may be wrong.

---

## The extracted fields

**Each document type has its own field list, straight from its requirement.** A document is
asked for the union of the lists of the types it names — six of the ten fixtures name two.

| field | key | invoice | credit note | receipt |
|---|---|---|---|---|
| Document Type | `document_type` | Yes | Yes | — |
| INV / CN No. | `document_number` | Yes | Yes | — |
| INV Date | `issue_date` | Yes | Yes | — |
| PO / GR / RTV No. | `po_gr_rtv_number` | **Yes** | optional | — |
| INV / RTV / CNR No. | `inv_rtv_cnr_number` | — | — | **Yes** |
| Supplier Tax ID | `seller_tax_id` | Yes | Yes | Yes |
| Supplier Tax Branch Code | `seller_branch` | Yes | Yes | Yes |
| Buyer Tax ID | `buyer_tax_id` | Yes | Yes | Yes |
| Buyer Tax Branch Code | `buyer_branch` | Yes | Yes | Yes |
| Amount Exclude VAT | `subtotal` | Yes | Yes | Yes |
| VAT Amount | `vat_total` | Yes | Yes | Yes |
| Amount Include VAT | `amount_incl_vat` | Yes | Yes | — |
| Remaining Amount | `remaining_amount` | — | — | optional |
| Remaining VAT Amount | `remaining_vat_amount` | — | — | optional |
| Payment Date | `payment_date` | — | — | optional |
| Cheque No. | `cheque_number` | — | — | optional |

Each of these three forms is eleven keys. Everything else the page states comes back under
its own printed label in `other_fields`, which nothing scores.

**The withholding tax certificate (มาตรา 50 ทวิ) is a fourth form and shares nothing with
them** — no document type, no document number, no VAT totals, and two parties who are not a
buyer and a seller. It is fourteen keys **and a table**:

| field | key | |
|---|---|---|
| Book No. (เล่มที่) | `book_no` | optional |
| Certificate No. (เลขที่) | `certificate_no` | optional |
| Sequence No. in P.N.D. Form (ลำดับที่ในแบบ) | `sequence_no` | optional |
| Payer Tax ID (เลขประจำตัวผู้เสียภาษีอากร) | `payer_tax_id` | **Yes** |
| Payer Name (ชื่อผู้จ่ายเงิน) | `payer_name` | **Yes** |
| Payer Branch (สำนักงานใหญ่/สาขาที่) | `payer_branch` | optional |
| Payer Address (ที่อยู่ผู้จ่ายเงิน) | `payer_address` | optional |
| Payee Tax ID | `payee_tax_id` | **Yes** |
| Payee Name | `payee_name` | **Yes** |
| Payee Branch | `payee_branch` | optional |
| Payee Address | `payee_address` | optional |
| Dividend Tax Rate Option (กรณี 40(4)(ข)) | `dividend_rate_option` | optional |
| Total Amount Paid (รวมเงินที่จ่าย) | `total_amount_paid` | **Yes** |
| Total WHT Amount (รวมภาษีที่หักนำส่ง) | `total_wht_amount` | **Yes** |

The **payer** is the party that paid the income and withheld the tax
(ผู้มีหน้าที่หักภาษี ณ ที่จ่าย); the **payee** is the party it was withheld from
(ผู้ถูกหักภาษี ณ ที่จ่าย). The requirement prints the word *Payer* over both blocks; they are
named apart here because the certificate carries the same four things twice and only the
label above each block says whose they are.

**It is the one form that asks for a table**, under `income_items` — one object per row of
income the certificate covers, all four cells Mandatory:

| cell | | |
|---|---|---|
| `income_type` | Income Type Description (รายละเอียดประเภทเงินได้) | the row's own wording, including what was written against a free-text option |
| `payment_date` | Payment Date (วัน เดือน ปี ที่จ่าย) | copied as printed; the Buddhist-era year is converted when the row is validated, never when it is read |
| `amount_paid` | Line Amount Paid (จำนวนเงินที่จ่าย) | added up and reconciled against `total_amount_paid` |
| `wht_amount` | Line WHT Amount (ภาษีที่หักและนำส่งไว้) | may not exceed that row's `amount_paid`; added up against `total_wht_amount` |

`Derived WHT Rate %` (อัตราภาษีที่หัก) is **not** extracted. Its own name says it is derived,
so it is worked out in Python from the two figures on the row and arrives under
`derived.wht_rates`, one entry per row, `null` where either figure could not be read. The
Fields tab draws it as a muted last column so it cannot be mistaken for something the page
printed.

No fixture in `solution/` is a withholding certificate, so nothing here has been scored
against ground truth — the rows are extracted, grounded against the transcript and validated,
but not field-scored.

**A receipt is not asked for its own number, date or heading.** Its requirement asks which
document is being *settled*, not what the receipt itself is — so `document_type`,
`document_number`, `issue_date` and `amount_incl_vat` are not on its form, and its own number
and date arrive in `other_fields`. Classification is unaffected: the type is read off the
transcript in Python, never out of an extracted field.

**A type with no requirement contributes no fields.** `TAX_INVOICE`, `STATEMENT_OF_ACCOUNT`
and `DEBIT_NOTE` have no list of their own, so `receipt + tax invoice` asks the receipt's
eleven and nothing more. A document naming *only* such a type — or none at all — is asked the
union of all four requirements (30 keys), because a narrower form chosen on a guess drops
Mandatory fields while a wider one costs tokens. That union is the widest thing this app ever
asks for — an unclassified document runs 17 agentic steps against a typed document's 7 or 9 —
so it is worth classifying a page rather than relying on it.

The certificate's **table** is the one thing the default does *not* inherit: a scalar nobody
asked for comes back empty, whereas a table a document does not rule comes back filled from
whatever is nearest.

### The rules differ by type as well

| | invoice | credit note | receipt |
|---|---|---|---|
| INV / CN No. | pattern + **duplicate check** | pattern + **original document matching** | — |
| INV / RTV / CNR No. | — | — | pattern + **reference matching** |
| INV Date | date format + **not in the future** | date format only | — |
| Remaining Amount | — | — | numeric + **outstanding balance** |
| Remaining VAT Amount | — | — | **VAT balance** |
| Payment Date | — | — | date format |

And the certificate's own, which no other type is held to:

| | withholding tax certificate |
|---|---|
| Payer / Payee Tax ID | 13 digits + mod-11 check digit |
| Payer / Payee Branch | 5-digit code |
| Payer / Payee Name | **fuzzy match against a Company Master at 90%** — reported as *not run*, there being no Company Master here |
| Line Amount Paid, Line WHT Amount | numeric, not negative, and the tax not above the amount paid |
| Payment Date (per row) | date format, Buddhist era to Western |
| Total Amount Paid, Total WHT Amount | numeric + **sum reconciliation** against the table's own column |
| Dividend Tax Rate Option | **checkbox detection** — required only where a row states dividend income |

The sum reconciliation is skipped rather than failed where a row's figure could not be read: a
sum missing one addend is not evidence that the total is wrong.

A credit note legitimately post-dates the invoice it corrects, which is why the future-date
rule does not run on one. `Cheque No.` has no validation rule in the requirement and gets
none: a field with nothing to check is not a field that passes, it simply is not checked.

### How the type is decided

In order of confidence, and the answer is reported on every result as `doc_types` (a list)
with `doc_type_from` saying which rule produced it:

| order | rule | `doc_type_from` |
|---|---|---|
| 1 | the request said so — `doc_types` (or `doc_type`) on `POST /api/extract` | `caller` |
| 2 | **the printed heading, matched in Python — where it is at least 90% sure** | `transcript` |
| 3 | **the model**, given the transcript and the list of codes | `model` |
| 4 | the benchmark manifest's `doc_types`, for one of the ten cases | `case` |
| 5 | the Python answer that was under the bar, rather than nothing | `transcript` |
| 6 | nothing recognised — the default form is asked | `unclassified` |

**Cheap and certain first, the model where the page is not obvious.** Rule 2 is
`normalise.document_types` run line by line over the top of the transcript — a table of
needles, no model involved — and every candidate line is scored: **how much of it the matched
needles actually account for**, discounted a little further down the page. A line that *is* a
heading scores 1.00; a sentence mentioning a document kind scores 0.2–0.3. Above
`CLASSIFY_MIN_CONFIDENCE` (0.9) that answer stands on its own and **no request is made at
all**.

A line that opens with a reference phrase — อ้างถึง, ตามที่, *with reference to*, *as per* —
is disqualified outright rather than scored. A credit note naming the invoice it credits is
the commonest way a page mentions a type it is not, and there is no score at which that
should win.

**A multi-type answer always goes to the model, however confident it is.** The confidence
measures how much of the *line* is heading, and a second type costs only a few characters on a
line the first already fills — so `ใบลดหนี้ ใบกำกับสินค้า 123` (a credit note citing an
invoice) and `ใบเสร็จรับเงิน/ใบกำกับภาษี` (genuinely both) look alike to it, and one wrong
type widens the form and adds rules the document is not held to. `CLASSIFY_ESCALATE_MULTI=0`
turns this off; with it on, six of the ten fixtures escalate, because a Thai slash heading is
the normal way to be two things at once.

**Rule 3 is the model**, asked whenever the table is unsure or silent — a page heading itself
in wording nobody wrote a needle for, which otherwise gets the default form, the union of
every requirement at 30 keys. It is offered the closed list of codes and asked for the ones
that fit, together with the heading it read them off, and **the answer is thrown away whole
unless that heading is really printed on the page**, checked with the same test every
extracted value gets. Codes outside the list are dropped.

**The confidence is reported, and it measures the evidence rather than anyone's self-belief.**
The model's answer is scored by the same function, off the heading it quoted, so the two are
on one scale — `doc_type_confidence` on the result, and `100% sure` / `43% sure` beside the
answer on the **Classify** box. It is absent, never 0%, where a person gave the answer (the
request, or the manifest): there is no heading on the page to score.

Where the model was asked, **why** it was asked rides with the answer as `doc_type_escalated`
and appears in the Classify box's tooltip — *the heading scored 43%, under 90%*, or *the
heading names more than one type*. "The model classified this" does not say what the cheap
path thought first, which is the question worth answering when the two disagree.

**Where the model's answer differs from the manifest's, the model's is used and the
disagreement is reported** — `doc_type_expected` on the result, and an amber note on the
Classify box saying *manifest says …*. The manifest is a person's reading and the model's is a
machine's; when a measured number moves, that field is what says which the run was built on.

Rule 2 is the only one that costs a request: one short one, on the first part of the
transcript, about half a second on a warm model. `CLASSIFY_WITH_MODEL=0` restores the old
order — manifest, then heading table — which is also how to stop a sweep paying for it. An
unrecognised document gets the default form, which is wider than any typed one; guessing a
type to pick a narrower form would drop Mandatory fields on a coin flip.

**The prompt is told the answer, and by default only in single mode.** The type names the
form, and it also frames the request: a *What this document is* block after the general rules,
carrying whatever that type has to say about a key on the form — a credit note's cited invoice
is not a PO number, a receipt's own number is not the document it settles, a receipt's own date
is not the payment date. In agentic mode the same framing goes in each step's task line, with
each step carrying only the rules for the keys **it** owns.

Measured over ten fixtures and five models, it gains single mode about five points of both
accuracy and precision. Agentic is closer — three of the five models gain and two lose — and
it is on because the two strongest models are among the three that gain.

| setting | default | |
|---|---|---|
| `TYPE_FRAMING_SINGLE` | `1` | frame the single-request prompt |
| `TYPE_FRAMING_AGENTIC` | `1` | frame each agentic step. Set `0` for `gemma4:e2b` or `qwen3.5:4b`, the two models it costs |
| `TYPE_FRAMING_BULLETS` | `1` | include the per-key rules, not just the type's name |

Nothing is emitted for a document rule 5 could not place, so an unclassified run is asked
exactly what it was asked before any of this existed — byte for byte, which a self-test at
import asserts.

**The manifest outranks the page.** A heading can name a type a person disagrees with, and
`doc_types` in `solution/manifest.json` is where that correction lives. sol001 is the standing
example: it prints its Thai heading as an invoice and its English one as a statement of
account, so the classifier reports both.

A document is filed under the **first** of its types — the most distinguishing one, since
being a tax invoice is the least distinguishing thing about a document here. That is what
names its PDF and what groups it in the three case dropdowns; the option says *also …* where
there are more types. It is never what chooses the field set.

**The page follows the form.** On the Fields tab only the keys that form asked for are
drawn, and a whole section drops when the form asks for none of it — an invoice shows
*Document, References, Supplier, Buyer, Totals*; a receipt shows *References, Supplier,
Buyer, Totals, Outstanding, Payment* and no Document section at all, because its requirement
asks nothing about the receipt itself. A row for a key the form omits would read as a
document that states nothing about it, when in fact nobody asked.

Every type is on the status line beside the check count, with the rule that produced them and
the type-specific validation rules that ran in the tooltip. Picking a case in the **Benchmark
case** or **Ground-truth document** picker says which form it will be asked — *receipt + tax
invoice · 11 fields* — before anything runs.

**Fields the requirement marks Mandatory carry a `REQUIRED` chip on their label**, and which
they are follows the type: an invoice demands all eleven of its keys, a receipt seven of its
eleven, and a document no requirement covers demands none, so nothing on it is marked — the
Classify box says **unknown type** instead, and the field score below is over the base field
set. The status line lists them in its tooltip.

A Mandatory field that came back empty reads **required, not returned**, in red, instead of
the plain *missing* an optional one gets — the page is not allowed to be silent about it, so
the empty row is a finding rather than an absence. The status line counts them beside the
check count: `12/13 checks pass · 2 required missing`. It is a separate figure and never
folded into the check count, because an absent field is neither a pass nor a failure of a
rule — the rule never ran on it.

On the withholding certificate's income table the requirement marks the cells Mandatory of
every row, so the chip goes on the **column heading**, and a row that leaves one of those
cells empty prints `required` in it.

The same distinction is carried through the rest of the page, so the score's scope is never
something you have to remember:

| where | what it says |
|---|---|
| the three case pickers | `receipt + tax invoice · 11 fields (7 required)`, before anything runs |
| the pipeline's **Classify** box | `11 fields, 7 required · 3 extra checks` |
| a field row's label | the `REQUIRED` chip |
| a found-only value's verdict | drawn muted and reading `· not scored` — it was judged and it is outside the rate |
| the **Accuracy** tab's *Field extraction* bar | `N required value(s) correct`, with the found-only count named after it |
| the **Run log** cell | `90% of 10 required` |

Where no requirement covers the type, nothing is marked and no required count is printed —
`0 required` would read as a form nobody can pass, when in fact nobody is holding the document
to anything.

## Validation

`validate.py` runs the requirement's validation rules over the values pass 2 returned. It is
deterministic Python, like `normalise.py` and the grounding check, and nothing about it is
asked of the model.

| field | rules | on |
|---|---|---|
| `document_type` | the heading normalises to a known type, and to one this document was taken to be | invoice, credit note |
| `document_number` | pattern; **duplicate check**; **original document matching** | invoice, credit note |
| `issue_date` | reads as a date; **not in the future** | invoice, credit note |
| `po_gr_rtv_number` | pattern; reference matching needs the PO/GR/RTV system | invoice, credit note |
| `inv_rtv_cnr_number` | pattern; reference matching needs the document being settled | receipt |
| `seller_tax_id`, `buyer_tax_id` | thirteen digits, and the mod-11 check digit | every form |
| `seller_branch`, `buyer_branch` | normalises to a five-digit branch code | every form |
| `subtotal`, `vat_total`, `amount_incl_vat` | each reads as an amount; subtotal + VAT = the total; VAT is the standard rate of the figure it is charged on | every form that asks the key |
| `remaining_amount` | numeric; outstanding balance needs what was owed before this payment | receipt |
| `remaining_vat_amount` | numeric; **VAT balance** — the tax part is the standard rate of the balance | receipt |
| `payment_date` | reads as a date | receipt |

Only the rules a form's own requirement names are run. `validation.rules` on every result says
which of them did, so a rule that passed can be told from one this document was never held to
— and **`vat_balance` is the only one of the receipt's three external-looking rules that can
actually run here**, because it compares two figures the page prints against each other.

The rules in the *on* column come from the requirement's own table for that type, and the
union applies where a document is more than one. `validation.rules` on every result says which
of them ran, so a rule that passed can be told from one this document was never held to.

Each check comes back in one of five states, and the three that are not a pass or a failure
are the ones worth understanding:

| state | means |
|---|---|
| `ok` | the rule ran and the value satisfies it |
| `failed` | the rule ran and the value does not |
| `warning` | worth a look, but a correct document can fail it. Only the VAT-rate rule uses it: a page that mixes taxed and untaxed lines legitimately blends below the standard rate |
| `unchecked` | **the rule could not run** — it needs data this process has not got, or the value it depends on is not printed. Never counted as a pass |
| `absent` | the field came back empty, so there is nothing to check. Marked `mandatory` where the requirement demands that field of this type |

It arrives on every extraction result as **`validation`**, beside `derived` and outside
`fields` — a computed verdict has nothing on the page to be grounded against, so merging it
in would have the grounding check report it as an invention.

**It reports and never gates.** No value is rewritten, no run fails, and nothing here reaches
an accuracy denominator. On the Fields tab a `failed` or `warning` rule prints under the value
it is about; a rule that passed or could not run prints nothing, and the status line carries
the counts. The pass rate is taken over the rules that actually ran.

## Normalised values

Some of the values the field requirement asks for are *classifications* rather than
readings: a standard document type separate from the printed heading, a branch **code**
with head office written `00000`, a tax ID reduced to digits, a service period split into
its two ends, and the references collected as a list rather than packed into one text
field.

None of those are asked of the model. Pass 2 copies the heading and the branch line as
printed, the grounding check confirms each against the transcript, and `normalise.py`
derives the rest afterwards in Python. They arrive on every extraction result as
**`derived`**, alongside `fields` rather than inside it:

```json
"derived": {
  "document_type_code": "RECEIPT_TAX_INVOICE",
  "seller_branch_code": "00000",
  "buyer_branch_code": "00068",
  "seller_tax_id_digits": "0101111011111",
  "buyer_tax_id_digits": "0101111001110",
  "seller_tax_id_valid": true,
  "buyer_tax_id_valid": true,
  "service_period_from": "01/01/2026",
  "service_period_to": "31/01/2026",
  "electronic_tax": true,
  "references": [
    { "type": "PO", "number": "12121212121", "date": "", "source": "header" },
    { "type": "INVOICE", "number": "510210009577", "date": "2026-01-31",
      "source": "line_item" }
  ]
}
```

| Key | |
|---|---|
| `document_type_code` | `INVOICE`, `CREDIT_NOTE`, `DEBIT_NOTE`, `RECEIPT`, `TAX_INVOICE` or `RECEIPT_TAX_INVOICE`, from the printed heading. `""` where the heading is not one this build recognises — a document whose type cannot be read belongs in review, not in a default |
| `seller_branch_code`, `buyer_branch_code` | the branch line reduced to a five-digit code, head office written `00000`. `""` for a branch the page names but does not number |
| `*_tax_id_digits` | the tax ID with its separators removed, so `0-5454-54545-54-5` and `0545454545545` are one value |
| `*_tax_id_valid` | whether that is thirteen digits. Reported, never enforced — it is the cheapest signal that a tax-ID key holds something that is not a tax ID |
| `service_period_from`, `service_period_to` | the two ends of the period, where it has two. Both `""` otherwise |
| `electronic_tax` | whether the page says it was filed with the Revenue Department electronically |
| `references` | every document this one cites, deduplicated, each with its `type`, `number`, `date` and whether it came from a header key or a table row |

Two things follow from `derived` sitting outside `fields`, and both are deliberate:

- **Grounding never sees it.** `grounding.check` walks `fields` and reports anything it
  cannot find on the page as invented. A computed value has nothing to be grounded
  against, so merging these in would flag the one class of value that provably is not an
  invention.
- **The field score never sees it either.** `solution/*.fields.json` scores what the model
  returned. These are checked by the self-tests in `normalise.py`'s own tables instead,
  because their correctness is a property of the code rather than of the run.


---

## API

`GET /api/ocr/profile` — the pass-1 profile in force and the ones on offer:
`{"profile":"typhoon","profiles":[{"id","label","note","system","reply"}, ...]}`.

`POST /api/ocr/profile` — `{"profile":"dots"}` switches it, and answers with what was
accepted. `400` for an unknown name. Not refused while the queue is busy: every page records
the profile it ran under, so a batch split across two is still readable afterwards.

`GET /api/ocr/loop-guard` — whether a cycling read is cut short, and the cap it would run to
without it: `{"loop_guard":true,"max_tokens":4096}`.

`POST /api/ocr/loop-guard` — `{"loop_guard":false}` switches it, and answers with what was
accepted. `400` for anything that is not `true` or `false`. Not refused while the queue is
busy, for the same reason as the profile. Turning it off stops reads being **aborted**; it does
not stop them being **detected**, so a run that cycled still comes back `looped`.

`GET /api/ocr/read-floor` — the transcript accuracy a read must reach for its fields to be
scored, and what the process started at: `{"min_read_pct":75.0,"default_pct":75.0}`.

`POST /api/ocr/read-floor` — `{"min_read_pct":0}` moves it, and answers with what was accepted.
A **percentage**, clamped to 0–100; `400` for anything that is not a number. Not refused while
the queue is busy, for the same reason as the profile — each run is flagged as it finishes, so
a batch split across the switch still says which rule wrote each row. What it cannot do is go
back: a score this floor stopped being taken is not in the log, which is why the run-log card
keeps a floor of its own.

`POST /api/ocr/stream` — multipart form, fields `image` and `detail`
(`original`/`medium`/`low`; the old four names are accepted and mapped). Returns NDJSON:

```
{"event":"page","page":1,"total":2,"resolution":"1500x1050"}
{"event":"token","text":"..."}
{"event":"page_done","page":1,"new_tokens":517,"resolution":"1500x1050","megapixels":1.57,
 "seconds":24.9,"prefill_seconds":13.0,"decode_seconds":11.9,
 "tokens_per_second":18.55,"truncated":false,"model":"...","backend":"llama.cpp",
 "url":"http://127.0.0.1:8080"}
{"event":"truncated","page":1}
{"event":"done","text":"...","pages":[...],"page_count":2,"tokens":517,"page_stats":[...]}
{"event":"logged"}
```

`logged` is emitted last, once the row for the run is on disk.

Every extraction result — on `/api/ocr`, on the `fields` event, and from `/api/extract` —
carries `fields`, `grounding`, `tiers`, `vat_basis`, `derived` and `mode`. It also carries
**`field_score`** when the document is a benchmark case with a `solution/<id>.fields.json`
to score against: `overall`, `scalars`, `line_items`, `other_fields`, `coverage`, and a row
per checked key saying what was expected and what came back. The key is simply absent for
everything else, which is most documents.

| Route | |
|---|---|
| `POST /api/ocr` | same fields as the stream, blocking, whole JSON at once. Drains the same generator internally, so timings are measured identically |
| `POST /api/classify` | what kind of document a transcript is, and the form that follows: `doc_types`, `doc_type`, `doc_type_from`, `doc_type_heading`, the `keys` and `mandatory` fields of that form, the agentic `steps` it would walk, the validation `rules` it adds, and the `type_phrase` both prompts would carry. Takes the same input as `/api/extract` — `text`, or a `job`, or a `case` with `from_truth` — and the same `doc_type`/`doc_types` override. Writes no run-log row: nothing was read and no field was extracted |
| `POST /api/extract` | re-run pass 2 on a transcript the app already has. Takes an optional `mode` (`single`/`agentic`) for that one call, and an optional `job` — the id from the `page`/`done` events — which names the document in the run-log row it writes, and is also what lets the reply carry `field_score`. `{"case": "sol003"}` names that document outright instead, and with `{"from_truth": true}` the transcript is read from `solution/sol003.md` and no `text` is needed: pass 2 alone, on text pass 1 cannot have spoiled. `400` for an id that is not a case. `{"steps": ["document"]}` runs only those agentic steps, for measuring one step's prompt on its own: the reply then carries `steps_only`, no `field_score` (the keys nobody asked for are not missing), and its run-log row names the steps in `extract_steps`. `400` in single mode, which has no part of the schema to ask for |
| `POST /api/extract/stream` | the same as `/api/extract`, as NDJSON: `classified` first — what the document was decided to be and the form that follows, before any request goes to the model — then `extract_steps` once, then an `extract_step` per step as it starts and finishes, then `fields`. Agentic mode only emits the step events; single mode emits `fields` alone |
| `GET` · `POST /api/extract/mode` | read or set the extraction shape for everything this process extracts next. Body `{"mode": "single"}` or `{"mode": "agentic"}` |
| `GET /api/page/<job>/<n>` | PNG of prepared page `n` (0-based). `job` is returned on the `page` and `done` events. 404s once the upload falls out of the cache |
| `POST /api/preview` | the prepared page **before** any read, so the Detail can be seen rather than guessed at. Multipart, taking the same `image` / `case` / `file` field as `/api/ocr` plus `detail` and an optional `page` (0-based). Answers with the PNG; `X-Preview-Pages`, `X-Preview-Detail` (as resolved), `X-Preview-Width`/`-Height` and `X-Preview-Source-Width`/`-Height` carry the numbers. No model server needed, and nothing is cached or logged |
| `GET /api/health` | active server status (reachable, kind, model, vision) and whether the PDF/HEIF decoders are available |
| `GET /api/servers` | every configured endpoint, what each one is, and which is active. `?probe=1` bypasses the status cache |
| `POST /api/servers` | `{"url": "...", "model": "..."}`, either field optional. 409 while the queue has a job running |
| `POST /api/context` | set the Ollama context window for subsequent requests |
| `GET /api/cases` · `GET /api/files` | benchmark cases, and readable documents in `mockOcr/`. Each case says `doc_types` (a list — a document is often more than one) and `field_truth`: whether it has a `solution/<id>.fields.json`, and so whether an extraction of it can be scored |
| `GET /api/truth/<case>` | the hand-written ground truth for one case, verbatim, plus the case's pdf, kind and page count. 404 for an id that is not a case |
| `GET`/`POST /api/queue` | list or enqueue. `GET /api/queue/<id>`, `DELETE /api/queue/<id>`, `POST /api/queue/clear`, `POST /api/queue/mode`, `POST /api/queue/workers` |
| `POST /api/queue/run` | release the queue so workers pick up what is in it; `{"start": false}` stops it handing out more. Queueing alone never starts a read. `started` in the queue's stats says which state it is in |
| `POST /api/match` | look up the ground-truth case for a file by name or sha256 |
| `GET /api/runs?limit=50` | recent run-log rows, newest first, plus the compiled tables under the process settings |
| `POST /api/runs/query` | the same payload recompiled under a different window, read floor and filter. Body `{limit, window, min_read_pct, include: {field: [...]}, exclude: {field: [...]}}`, every key optional and `null` meaning *use the process setting*. Filterable fields: `case`, `model`, `extract_model`, `extract_mode`, `backend`, `detail`, `ocr_profile`, `status`, `run_type`, `source`, `pipeline` (`both`/`read`/`extract`). **Writes nothing** — `runs` is always the unfiltered log, `totals` is compiled over what matched |
| `POST /api/runs/delete` | remove rows from the log, archiving them to `logs/runs.deleted.csv`. Body `{rows: [{_row, timestamp, file}]}` as they came back from `/api/runs`; a row whose timestamp and file no longer match the position is reported as `stale` and left alone |
| `GET /api/runs.csv` | the log file itself |

---

## Deploying

### Build the archive

```bash
python package.py
```

Writes `dist/thai-ocr-<date>.zip` — everything needed to run, and nothing else.

```bash
python package.py --no-fixtures
```

Omits `mockOcr/` and `solution/` for an upload-only deployment (~0.1 MB instead of
~2.2 MB).

The file list in `package.py` is an **allow-list**: a new source file has to be added to
`FILES` there or it will not ship. Never in the archive: `__pycache__/`, `logs/*.csv`,
`solution/out/`, `.env`, `.claude/`.

### Install on the server

```bash
unzip thai-ocr-<date>.zip && cd thai-ocr-<date>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

**On a server with no internet, copy `wheelhouse/` in beside `app.py` yourself** — it is
not in the zip, because it is built for one platform and one Python version while the zip
runs anywhere. Then install from it instead:

```bash
unzip thai-ocr-<date>.zip && cd thai-ocr-<date>
cp -r /path/to/wheelhouse .
python3.11 -m venv .venv && source .venv/bin/activate
pip install --no-index --find-links wheelhouse -r requirements.txt
python app.py
```

See [Offline install](#offline-install-no-internet-on-the-target) for what the bundled
wheels require, and how to rebuild them for a different target.

Nothing is compiled and no paths are baked in — every path is resolved from `app.py`'s own
directory, so the archive runs wherever it is unpacked, under any account.

The zip contains `.env.example` but **never a `.env`** — configure the deployment in its
service unit, or drop a `.env` beside `app.py` and load it in the unit's `ExecStartPre`
/ launch script. A systemd unit wants the venv's interpreter by absolute path:

```ini
WorkingDirectory=/srv/ocr
EnvironmentFile=/srv/ocr/.env
ExecStart=/srv/ocr/.venv/bin/python app.py
```

`EnvironmentFile` is systemd reading the file and handing the app the variables — the app
still never parses `.env` itself.

---

## Project layout

| File | What it owns |
|---|---|
| `app.py` | Flask routes, image preparation, both passes, streaming |
| `settings.py` | Every tunable the app runs with — limits, timeouts, detail presets, sampling, loop thresholds |
| `prompts.py` | The prompts, and nothing else — the ones the passes send, plus the step table agentic extraction walks |
| `config.py` | Every path and the typed environment readers. The only place that knows where anything lives |
| `backends.py` | Endpoint probing and switching; every llama.cpp-vs-Ollama difference |
| `grounding.py` | Checking extracted fields against the transcript they came from |
| `fieldscore.py` | Scoring extracted fields against the hand-written field ground truth, and `init` to create it |
| `verify.py` | Reads document amounts, and decides whether the extracted figures already carry VAT |
| `normalise.py` | Derives the normalised values from what pass 2 copied — standard document type, branch codes, tax-ID digits, the reference list |
| `scoring.py` | Ground-truth lookup and accuracy scoring, shared by page and CLI |
| `runlog.py` | The CSV run log |
| `jobs.py` | The in-process queue and its worker pool |
| `compare.py` | CLI benchmark runner |
| `package.py` | Builds the deployable zip |
| `templates/index.html` | The whole UI, in one file |

## Notes for whoever edits this

- **`templates/index.html` edits need a server restart.** `app.py` runs with
  `debug=False`, so Jinja caches compiled templates for the process lifetime; a browser
  reload shows the old page.
- **Settings are read once at import**, so a changed environment variable needs a restart
  too. Read them through `config.env_*` rather than `os.environ` directly, and add new ones
  to `.env.example` with their default.
- **A new source file must be added to `FILES` in `package.py`** or it will not ship.
- **There is no unit test suite.** `python compare.py` is the regression test, and it is
  worth running after any change to a prompt, a sampler setting, or how a request is
  assembled — the failure mode there is a silent accuracy drop at HTTP 200, not an
  exception.
