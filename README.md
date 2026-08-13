# Thai Document OCR

Upload a Thai/English document, get back a verbatim transcript, the fields extracted from
it, and an arithmetic check on its numbers.

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
| `OCR_LOG_DIR` | `./logs` | The only directory written to. Point it elsewhere to mount the app directory read-only. |
| `OCR_MOCK_DIR` | `./mockOcr` | Source documents for the folder picker. Absent ⇒ upload-only. |
| `OCR_SOLUTION_DIR` | `./solution` | Ground truth. Absent ⇒ accuracy scoring switches off and the page hides its controls. |
| `MAX_UPLOAD_MB` | `32` | Per-upload size cap. |
| `MAX_PAGES` | `10` | Pages read per document — a direct cap on the worst case cost of one request. |
| `MAX_JOBS` | `5` | Rendered documents held in memory for the compare view. A 10-page document at `accurate` is ~40 MB, so this is a RAM ceiling. |
| `GEN_READ_TIMEOUT` | `1800` | Raise on slow hardware; a timeout firing mid-generation throws away work the model server is still doing. |
| `EXTRACT` | `1` | Set `0` to run the OCR pass only. |

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

### What a run does

Each run is up to three passes against the model server:

1. **OCR** — the page image in, a verbatim transcript out, streamed to the browser as it
   arrives. Shown in **Markdown** (raw) and **Rendered**.
2. **Extract** — the transcript back in as *text*, structured JSON out. Every returned
   value is then traced back to the transcript **in Python**; a value that is not in the
   text is flagged rather than displayed as data. Shown in **Fields**.
3. **Verify** — the numbers are checked, again **in Python**: column sums, per-line
   arithmetic, VAT rate, and the Thai amount-in-words against the numeral. Shown in
   **Numbers**.

Passes 2 and 3 are deterministic on purpose — the arithmetic and the grounding check are
done by this app, not asked of the model. Turn them off with `EXTRACT=0`, or send
`extract=0` with the request. Either can be re-run on its own from the page without
re-reading the image, or via `POST /api/extract` and `POST /api/verify`.

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

Whether the printed figures already carry VAT is decided before anything is checked, and
the tab labels itself from it — the money column reads *Amount (incl. VAT)* or *Amount (ex
VAT)*, with a note saying which basis the page is on and what said so.

The **References** group hides itself when it comes back empty; a cash receipt legitimately
has none of those fields.

### Compare with source

The card below the result puts the source page and the extracted text side by side, for
proofreading against the original. **Show** opens it.

The image shown is the **prepared** page — after PDF rasterisation and after the Detail
downscale — so it is literally what was sent.

- Page arrows for multi-page documents; image and text stay on the same page.
- **Markdown** checkbox swaps the right pane between rendered output and raw text.
- **Sync scroll** links the panes proportionally.
- **Open image** opens the prepared page full size in a new tab.
- Available while a run is still streaming.

Prepared pages are cached in memory per upload (`MAX_JOBS = 5`, oldest evicted) and served
from `GET /api/page/<job>/<n>`. Nothing is written to disk.

### Queue tab

One document at a time streams in the Workspace pane. Drop in more than one — or use **Add
to queue**, **Queue all cases**, **Queue whole folder** — and they go to the **Queue** tab.

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

`solution/out/*.txt` is the transcript of the last run, written automatically. Deleting it
is harmless. `solution/manifest.json` maps each id to its source PDF.

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

Each case reports:

| Metric | Meaning |
|---|---|
| `char accuracy` | 1 − character edit distance / length |
| `word accuracy` | longest-common-subsequence word overlap |
| `char acc. w/o marks` | same, with Thai tone marks and above/below vowels stripped. If this is much higher than `char accuracy`, the right letters are being read and the marks lost — a different problem from wrong words. |

…followed by a unified diff of expected vs actual.

On Windows, Thai in the diff prints as `?` — the console defaults to cp1252 and cannot
encode it. The scores are unaffected. For readable diffs:

```bash
set PYTHONIOENCODING=utf-8
```

**The scores are only as good as `solution/*.md`.** Those files are one person's reading of
the rendered pages — correct them where you disagree. `sol003` is a handwritten scan and its
handwritten values in particular deserve a second look.

---

## Run log

Every document read appends one row to `logs/runs.csv` — from the page, from the queue, from
`/api/ocr`, and from `compare.py`. The **Run log** card at the bottom of the page shows the
recent rows and links the CSV.

**It holds measurements only.** No transcript, no extracted fields, no page images:

| Column | |
|---|---|
| `timestamp` | local time, seconds resolution |
| `file`, `file_size_mb`, `pages`, `detail`, `source` | what was read, and how it got in (`upload`/`folder`/`case`/`queue`) |
| `server`, `backend`, `model` | which endpoint and model actually ran it |
| `seconds`, `prefill_seconds`, `decode_seconds` | runtime, split into prompt processing and generation |
| `tokens`, `tokens_per_second` | OCR output tokens; the rate is decode-only |
| `extract_seconds`, `extract_tokens`, `verify_seconds` | passes 2 and 3 |
| `grounded_pct`, `ungrounded`, `fields_missing` | share of extracted values found in the transcript, how many were not, and how many fields the document does not state |
| `p1_present`, `p1_absent`, `p2_present`, `p2_absent` | field coverage by delivery tier — how many of the 14 priority-1 and 14 priority-2 keys came back filled. Blank on a run that extracted nothing, which is not the same as zero |
| `case`, `char_accuracy`, `word_accuracy`, `char_accuracy_no_marks` | percentages, blank when the input has no ground truth |
| `verdict` | the number-check result |
| `status`, `error` | `ok` / `truncated` / `looped` / `cancelled` / `error` |

Coverage is not correctness. `p1_present` counts what came back filled, not what came back
right — read it beside `grounded_pct`.

Failed and cancelled runs are logged too. Move the file with `OCR_LOG_DIR`. It is written as
UTF-8 with a BOM so Excel opens Thai filenames correctly; new columns are only ever
appended, so old rows stay readable.

Because each row carries the server, the model and the accuracy together, this is the
straightforward way to answer "is 11434 actually better than 8080 on my documents" — run the
same cases on each and compare the columns.

---

## API

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

| Route | |
|---|---|
| `POST /api/ocr` | same fields as the stream, blocking, whole JSON at once. Drains the same generator internally, so timings are measured identically |
| `POST /api/extract` · `POST /api/verify` | re-run pass 2 or pass 3 on a transcript the app already has |
| `GET /api/page/<job>/<n>` | PNG of prepared page `n` (0-based). `job` is returned on the `page` and `done` events. 404s once the upload falls out of the cache |
| `GET /api/health` | active server status (reachable, kind, model, vision) and whether the PDF/HEIF decoders are available |
| `GET /api/servers` | every configured endpoint, what each one is, and which is active. `?probe=1` bypasses the status cache |
| `POST /api/servers` | `{"url": "...", "model": "..."}`, either field optional. 409 while the queue has a job running |
| `POST /api/context` | set the Ollama context window for subsequent requests |
| `GET /api/cases` · `GET /api/files` | benchmark cases, and readable documents in `mockOcr/` |
| `GET`/`POST /api/queue` | list or enqueue. `GET /api/queue/<id>`, `DELETE /api/queue/<id>`, `POST /api/queue/clear`, `POST /api/queue/mode`, `POST /api/queue/workers` |
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
| `app.py` | Flask routes, image preparation, the three passes, streaming |
| `settings.py` | Every tunable the app runs with — limits, timeouts, detail presets, sampling, loop thresholds |
| `prompts.py` | The four prompts, and nothing else |
| `config.py` | Every path and the typed environment readers. The only place that knows where anything lives |
| `backends.py` | Endpoint probing and switching; every llama.cpp-vs-Ollama difference |
| `grounding.py` | Checking extracted fields against the transcript they came from |
| `verify.py` | Arithmetic and VAT checks on the extracted numbers |
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
