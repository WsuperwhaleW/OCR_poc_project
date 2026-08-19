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
| `OLLAMA_SYSTEM` | `You are a helpful assistant.` | System message sent with every Ollama request; it reproduces what Ollama otherwise injects from the served model's Modelfile. Ollama only — llama-server is sent none. Set it empty to send no system message. |
| `EXTRACT_SCHEMA` | `1` | Whether a field-extraction reply that cannot be parsed is asked again with decoding constrained to the field schema. The first request is unconstrained either way. `0` turns the retry off, so an unusable reply is reported instead. |
| `OLLAMA_REPEAT_PENALTY` | `1.1` | Repetition penalty on the constrained retry above. Ollama only, and the only place this app sets one above `1.0`. `1.0` turns it off. |
| `OLLAMA_UNLOAD_ON_SWITCH` | `1` | Stop the models Ollama is holding when you switch model or server, freeing the GPU for the new one. Ollama only. `0` leaves them loaded until their `keep_alive` expires. |
| `OCR_LOG_DIR` | `./logs` | The only directory written to. Point it elsewhere to mount the app directory read-only. |
| `OCR_MOCK_DIR` | `./mockOcr` | Source documents for the folder picker. Absent ⇒ upload-only. |
| `OCR_SOLUTION_DIR` | `./solution` | Ground truth. Absent ⇒ accuracy scoring switches off and the page hides its controls. |
| `MAX_UPLOAD_MB` | `32` | Per-upload size cap. |
| `MAX_PAGES` | `10` | Pages read per document — a direct cap on the worst case cost of one request. |
| `MAX_JOBS` | `5` | Rendered documents held in memory for the compare view. A 10-page document at `accurate` is ~40 MB, so this is a RAM ceiling. |
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
| `fast` | 1.0 MP |
| `balanced` | 2.0 MP |
| `accurate` **(default)** | 4.0 MP |
| `max` | none — native resolution |

PDFs are rasterised at 300 DPI (`PDF_DPI`) so the downscale resamples from real detail.
Uniform blank margins are cropped before the cap is applied; disable with `TRIM_MARGINS=0`.

**Neither end of the range is safe by default.** Too few pixels and fine detail is lost;
too many is not reliably better either. `accurate` is the default because it scored best on
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
- **The profile does not follow the model.** Selecting a model in the server picker does not
  change the profile, and nothing checks that they match; a mismatch is a legitimate thing to
  measure. The run log records both per row.

Set the starting profile with `OCR_PROFILE` (`typhoon` or `dots`). Switching it applies to
this process — the queue and any other browser tab included — and takes effect on the next
page read; a page already streaming finishes under the profile it started with.

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
- **Open image** opens the prepared page full size in a new tab.
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

The two server kinds are detected, not configured — `backends.py` probes `/props` first
(llama.cpp) then `/api/tags` (Ollama) — and the app adapts:

| | llama-server | Ollama |
|---|---|---|
| Models | exactly one, chosen at launch | many; a **Model** picker appears when there is more than one |
| Vision | `modalities.vision` from `/props` | `capabilities` per model from `/api/tags` |
| Concurrency | `total_slots` from `/props` | not exposed, so the page claims no number |
| Timings | `timings` block: real prompt/predict split | OpenAI `usage`; time-to-first-token stands in for prefill |

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

The app never polls the model server. It asks for status when the page renders and when you
press **Re-check**, and that is all.

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
| `looped` | the output was cycling and the app aborted it, rather than letting a repeating transcript pass as a complete one |
| `error` | the request itself failed — see the message and [Server status](#server-status) |

Extraction has the same two failure modes and reports them the same way, so an unterminated
JSON reply is labelled as a loop rather than a cryptic parse error. **Re-extract** retries
just that pass. If a document loops repeatedly: lower **Detail** first, and extract one page
at a time. Raising the token cap does not help.

---

## Checking accuracy against ground truth

`solution/` holds a hand-transcribed expected transcript for each PDF in `mockOcr/` —
`sol001.md` … `sol005.md`, in Markdown so the tables render when you open them. Edit these
directly; they are the files the scores are computed from. Scoring lives in `scoring.py` and
is shared by the web page and the CLI, so the browser and the terminal can never report
different numbers for the same run.

`solution/<id>.fields.json` is the second ground truth: which **value** belongs in which
**extracted field** for the same document. It scores pass 2, the way the `.md` scores pass 1,
and it is filled in by hand — see *Field extraction accuracy* below.

`solution/out/*.txt` is the transcript of the last run and `solution/out/*.fields.json` the
last extraction, both written automatically. Deleting them is harmless.
`solution/manifest.json` maps each id to its source PDF.

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
expected text. The headline character accuracy is also appended to the footer line. Every
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
| `--detail fast\|balanced\|accurate\|max` | override the resolution preset |
| `--app URL` | score a deployed instance instead of localhost |
| `--keep-tables` | compare table markup literally instead of normalising it |
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
| `char accuracy` | 1 − character edit distance / length. **The headline number, and it scores content only** — every invisible character is removed from both sides first, so line breaks, blank lines, indentation, cell padding, tabs, non-breaking and other Unicode spaces, and zero-width marks cannot change it. So is every piece of table markup: `<table>`, `<tr>`, `<td>`, `<th>` and their `colspan`/`rowspan` attributes, `<br>`, pipes and the `\|---\|` separator row all come out, leaving the cell text. A table scores on what it says, not on how it was marked up. The `--- page 2 ---` separator this app puts between the pages of one document goes too. Only the characters a reader would see on the paper are compared. |
| `word accuracy` | longest-common-subsequence word overlap. Whitespace does not count here either, but re-ordering does: a value read correctly in the wrong place lowers this and not `char accuracy`. |
| `char acc. w/o marks` | `char accuracy` with Thai tone marks and above/below vowels stripped. If this is much higher than `char accuracy`, the right letters are being read and the marks lost — a different problem from wrong words. |

…followed by a unified diff of expected vs actual.

On Windows, Thai in the diff prints as `?` — the console defaults to cp1252 and cannot
encode it. The scores are unaffected. For readable diffs:

```bash
set PYTHONIOENCODING=utf-8
```

**The scores are only as good as `solution/*.md`.** Those files are one person's reading of
the rendered pages — correct them where you disagree. `sol003` is a handwritten scan and its
handwritten values in particular deserve a second look.

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

| Metric | Meaning |
|---|---|
| `field accuracy` | correct values ÷ values the truth file says the page prints. **The headline.** |
| loose | the same, counting a partial match — one value contains the other, so the right thing was found and too much or too little of it was taken |
| `field precision` | correct ÷ everything the extractor filled in. Falls when it invents values the page does not state |
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
more priority-1 keys filled, then more priority-2, then a higher `grounded_pct` — its figures
replace the pass-2 columns of the row for the read it came from, and `extract_updated` records
when that happened. 12/14 becomes 14/14; 14/14 is never pulled back down to 12/14, and an
extraction that failed can never displace one that ran. The read's own columns are never
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
| `ocr_profile` | `typhoon` or `dots` — the pass-1 shape that read the page, taken from the pages themselves for the same reason. Blank on rows written before profiles existed, and on `run_type=extract` rows, which read no page |
| `grounded_pct`, `ungrounded`, `fields_missing` | share of extracted values found in the transcript, how many were not, and how many fields the document does not state |
| `field_acc`, `field_expected` | pass 2 scored against `solution/<id>.fields.json`: the share of the values that came back correct, and how many values that was. Blank on every document without a field truth file, so it does not read down the column like `grounded_pct` — read the accuracy beside its own `field_expected`, because 100% of three keys and 100% of fourteen are the same cell and not the same claim |
| `p1_present`, `p1_absent`, `p2_present`, `p2_absent`, `p3_present`, `p3_absent` | field coverage by delivery tier — how many of each tier's keys came back filled. Pass 2 extracts priority 1 only, so `p2`/`p3` read `0/0`: this build asked for none of them, which is not the same as a run that extracted nothing and leaves them blank. Present and absent are both written, so a row always says what its counts were out of |
| `case`, `char_accuracy`, `word_accuracy`, `char_accuracy_no_marks` | percentages, blank when the input has no ground truth |
| `status`, `error` | `ok` / `partial` / `truncated` / `looped` / `cancelled` / `error` |
| `run_type` | `ocr` for a document read, `extract` for a re-extraction of a transcript already read. Blank on rows written before the column existed |
| `extract_updated` | set when a later, better re-extraction replaced this row's pass-2 columns, so `timestamp` no longer says when they were measured. Blank on the normal case |

Coverage is not correctness. `p1_present` counts what came back filled, not what came back
right — read it beside `grounded_pct`. Each row's **Fields** cell also carries the shape that
filled it, `single` or `agentic`: one request and seven fill the schema in different ways,
so two rows of counts are not comparable without it.

The card header reports the **mean** accuracy over every scored row and the **best** single
score in the file, and above the table is the best each document has ever reached — over
every run and every setting, with the model, backend, detail and mode that reached it on
hover. The mean moves with whatever was being tried lately; the best says what the document
is known to be capable of, which is the number to beat.

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

Failed and cancelled runs are logged too. Move the file with `OCR_LOG_DIR`. It is written as
UTF-8 with a BOM so Excel opens Thai filenames correctly; new columns are only ever
appended, so old rows stay readable.

Because each row carries the server, the model and the accuracy together, this is the
straightforward way to answer "is 11434 actually better than 8080 on my documents" — run the
same cases on each and compare the columns.

---

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

`POST /api/ocr/stream` — multipart form, fields `image` and `detail`
(`fast`/`balanced`/`accurate`/`max`). Returns NDJSON:

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
| `POST /api/extract` | re-run pass 2 on a transcript the app already has. Takes an optional `mode` (`single`/`agentic`) for that one call, and an optional `job` — the id from the `page`/`done` events — which names the document in the run-log row it writes, and is also what lets the reply carry `field_score` |
| `POST /api/extract/stream` | the same as `/api/extract`, as NDJSON: `extract_steps` once, then an `extract_step` per step as it starts and finishes, then `fields`. Agentic mode only emits the step events; single mode emits `fields` alone |
| `GET` · `POST /api/extract/mode` | read or set the extraction shape for everything this process extracts next. Body `{"mode": "single"}` or `{"mode": "agentic"}` |
| `GET /api/page/<job>/<n>` | PNG of prepared page `n` (0-based). `job` is returned on the `page` and `done` events. 404s once the upload falls out of the cache |
| `GET /api/health` | active server status (reachable, kind, model, vision) and whether the PDF/HEIF decoders are available |
| `GET /api/servers` | every configured endpoint, what each one is, and which is active. `?probe=1` bypasses the status cache |
| `POST /api/servers` | `{"url": "...", "model": "..."}`, either field optional. 409 while the queue has a job running |
| `POST /api/context` | set the Ollama context window for subsequent requests |
| `GET /api/cases` · `GET /api/files` | benchmark cases, and readable documents in `mockOcr/` |
| `GET /api/truth/<case>` | the hand-written ground truth for one case, verbatim, plus the case's pdf, kind and page count. 404 for an id that is not a case |
| `GET`/`POST /api/queue` | list or enqueue. `GET /api/queue/<id>`, `DELETE /api/queue/<id>`, `POST /api/queue/clear`, `POST /api/queue/mode`, `POST /api/queue/workers` |
| `POST /api/queue/run` | release the queue so workers pick up what is in it; `{"start": false}` stops it handing out more. Queueing alone never starts a read. `started` in the queue's stats says which state it is in |
| `POST /api/match` | look up the ground-truth case for a file by name or sha256 |
| `GET /api/runs?limit=50` | recent run-log rows, newest first, plus totals |
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
