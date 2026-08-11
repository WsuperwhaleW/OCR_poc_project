# Thai Document OCR

Upload-a-file web page that reads a whole page and returns structured Markdown.

This app holds no model weights and imports no torch, transformers, accelerate or numpy.
It decodes uploads into page images, caps their resolution, and streams the result back
from an **external** model server over its OpenAI-compatible HTTP API. Which server it
talks to is switchable from the page while it runs; see
[Switching model server](#switching-model-server).

## Run

You need a vision-capable model server listening somewhere — llama.cpp's `llama-server`
or Ollama. Its configuration, tuning and choice of model are its own business and are not
documented here; the app only requires that it can accept an image and speak
`/v1/chat/completions`.

Two integration requirements:

- **The server must have vision enabled.** llama.cpp needs a projector (`--mmproj`) or it
  returns HTTP 500 for every image. The app detects this and says so rather than letting
  the read fail — see [Server status](#server-status).
- **Its context window must fit a page plus the reply.** Too small and extraction returns
  truncated JSON; for Ollama the app can set this per request, see
  [Context window](#context-window).

Then:

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Startup is ~1 s — nothing is loaded.

Point at a model server with `LLAMA_URL` (default `http://127.0.0.1:8080`), and change the
app's own port with `PORT`.

Device placement, threads and quantization are the model server's settings, not the app's.
This app has no CPU/GPU control.

## Deploying

### Building the archive

```bash
python package.py
```

Writes `dist/thai-ocr-<date>.zip` — everything needed to run, and nothing else.
`package.py --no-fixtures` omits `mockOcr/` and `solution/` for an upload-only
deployment (~0.1 MB instead of ~2.2 MB).

The file list in `package.py` is an **allow-list**. A new source file has to be added
there or it will not ship — which is the point: a deny-list quietly ships whatever was
dropped in the directory since anyone last thought about it. Never in the archive:
`__pycache__/`, `logs/*.csv`, `solution/out/`, `.env`, `.claude/`.

### Installing on the server

```bash
unzip thai-ocr-<date>.zip && cd thai-ocr-<date>
pip install -r requirements.txt
python app.py
```

Nothing is compiled and no paths are baked in — every path is resolved from
`app.py`'s own directory, so the archive runs wherever it is unpacked, under any
account. Startup prints what it found; read those lines before assuming it is
healthy:

```
[ocr] listening on http://127.0.0.1:5000
[ocr] llama.cpp http://127.0.0.1:8080 model=... available=True
[ocr] documents folder: /srv/ocr/mockOcr (5 readable)
[ocr] ground truth: /srv/ocr/solution (5 scored cases)
[ocr] run log: /srv/ocr/logs/runs.csv
```

A missing model server, a missing PDF library, an absent fixtures directory and an
unwritable log directory each print a warning and the app **still starts**. None of
them is fatal, and a server that refuses to boot because the model server has not
come up yet is worse than one that says so and waits.

### Configuration

Every setting is an environment variable, listed with its default in
[`.env.example`](.env.example). The app does not read `.env` itself — there is no
`python-dotenv` dependency — so export the values in the shell or set them in the
service unit. A malformed value prints a warning and falls back to the default
rather than stopping startup, so a typo in a service file cannot leave the app dead
with no explanation.

The ones that matter for a deployment:

| Variable | Default | Why you would change it |
|---|---|---|
| `OCR_HOST` | `127.0.0.1` | Loopback. Leave it — see [Local only](#local-only). |
| `PORT` | `5000` | Port to bind. |
| `LLAMA_URL` | `http://127.0.0.1:8080` | Where the model server is listening. |
| `OCR_LOG_DIR` | `./logs` | The only directory written to. Point it elsewhere to mount the app directory read-only. |
| `OCR_MOCK_DIR` | `./mockOcr` | Source documents for the folder picker. Absent ⇒ upload-only. |
| `OCR_SOLUTION_DIR` | `./solution` | Ground truth. Absent ⇒ accuracy scoring switches off and the page hides its controls. |
| `MAX_UPLOAD_MB` | `32` | Per-upload size cap. |
| `MAX_PAGES` | `10` | Pages read per document — a direct cap on the worst-case cost of one request. |
| `MAX_JOBS` | `5` | Rendered documents held in memory for the compare view. A 10-page document at `accurate` is ~40 MB, so this is a RAM ceiling. |
| `GEN_READ_TIMEOUT` | `1800` | Raise on slow hardware; a timeout firing mid-generation throws away work the model server is still doing. |

### Local only

**This app has no authentication, no authorisation, no rate limiting and no CSRF
protection.** It is a single-user tool meant to be reached over loopback, which is
why `OCR_HOST` defaults to `127.0.0.1` and why startup warns whenever it does not.
That default is the security control; changing it removes the only one there is.

Anyone who can reach the port can upload documents, read every document in
`mockOcr/`, and download the full run log. Keep it on loopback.

`python app.py` also runs Flask's development server, which prints a warning saying
so. It is threaded and it is fine for the local traffic this is sized for; it is not
hardened for hostile traffic, which is the same conclusion as the paragraph above.

## Switching model server

The **Model server** dropdown at the top of the left card switches which server the app
talks to, without restarting it. Offered by default:

| Endpoint | Usually |
|---|---|
| `http://127.0.0.1:8080` | llama-server (the `LLAMA_URL` default) |
| `http://127.0.0.1:11434` | Ollama's standard port |

Set the list with `OCR_ENDPOINTS` (comma separated) if your servers listen on different
ports.

The two server kinds are detected, not configured — `backends.py` probes `/props` first
(llama.cpp) then `/api/tags` (Ollama) — and the app adapts from there:

| | llama-server | Ollama |
|---|---|---|
| Models | exactly one, chosen at launch | many; the app sends `model` on every request and shows a **Model** picker when there is more than one |
| Vision | `modalities.vision` from `/props` | `capabilities` per model from `/api/tags` |
| Concurrency | `total_slots` from `/props` | not exposed, so the page claims no number |
| Timings | `timings` block: real prompt/predict split | OpenAI `usage`; time-to-first-token stands in for prefill |

**The app never polls the model server.** It asks for status when the page renders and
when you press Re-check, and that is all. llama.cpp serves `GET /slots` from the *same
task queue* as inference, so a background poll issued mid-read queues behind the page,
times out on our end, and shows up in the server log as `srv stop: cancel task` — churn
aimed at the process doing the actual work. Timings are recorded per run in the run log
instead, where they can be compared after the fact rather than watched.

Switching is **refused while the queue is running** (HTTP 409) — half a document read on
one server and half on another would be logged and scored as if one server had done it.
Finish or cancel those jobs first. A single streaming run from the page is not blocked,
but the switch only affects the *next* run: the model, backend and URL are captured
per page as it is read, so a finished run is always reported against the server that
actually ran it.

## Server status

The page shows a status bar with a green/red dot, the model name, the server kind, and a
**Re-check** button (which bypasses the 3-second status cache and re-reads the model
list). When the server is unreachable, or cannot see images, the reason is shown and
**Read document stays disabled** — you can't start an upload that is going to fail:

| Condition | Message |
|---|---|
| Nothing on the port | `No model server reachable at <url>.` |
| llama-server, `modalities.vision=false` | `...running '<model>' with vision disabled ... Restart it with --mmproj` |
| Ollama, model has no vision capability | `...has no vision capability ... Choose a vision model` |
| Ollama with nothing pulled | `...has no models pulled.` |

A model whose capabilities the server does not report is shown as *vision not reported*
and allowed through — attempting the read and failing is more useful than refusing a
model that would have worked.

## Accepted files

| Input | Handled by | Notes |
|---|---|---|
| JPG, PNG, WEBP, BMP, PPM, ICO, … | Pillow | anything Pillow can decode |
| GIF, multi-page TIFF | Pillow | each frame is treated as a page |
| HEIC / HEIF (iPhone photos) | `pillow-heif` | optional dependency |
| PDF | `pymupdf` | rendered at 300 DPI, optional dependency |

Multi-page input is capped at 10 pages, upload limit 32 MB (`MAX_PAGES`,
`MAX_UPLOAD_MB`). PDF and HEIC can't render in an `<img>`, so the page shows a filename
card instead of a thumbnail.

## Detail (input resolution)

The **Detail** selector caps how many pixels the app sends. It is the main
fidelity/latency dial, because the cost of a read scales with the pixel count.

| Preset | Cap |
|---|---|
| `fast` | 1.0 MP |
| `balanced` | 2.0 MP |
| `accurate` **(default)** | 4.0 MP |
| `max` | none — native resolution |

PDFs are rasterised at 300 DPI (`PDF_DPI`) so the downscale resamples from real detail
rather than from a coarsely rendered page. Uniform blank margins are cropped before the
cap is applied, so the pixel budget goes to content; disable with `TRIM_MARGINS=0`.

**Neither end of the range is safe by default.** Too few pixels and fine detail is lost;
too many is not reliably better either. `accurate` is the default because it scored best
on the bundled cases — score a case yourself before moving off it, see
[Checking accuracy against ground truth](#checking-accuracy-against-ground-truth).

## Compare view

The **Compare** tab puts the source page and the extracted text side by side, so you can
proofread the transcription against the original.

The image shown is the **prepared** page — after PDF rasterisation and after the Detail
downscale — so it is literally what was sent, not the original file. If a word came back
wrong because the downscale destroyed a tone mark, you are looking at the same pixels.

- Page arrows for multi-page documents; image and text stay on the same page.
- **Markdown** checkbox swaps the right pane between rendered output and raw text.
- **Sync scroll** links the panes proportionally — they have different natural heights,
  so it matches by fraction of scrollable distance rather than pixel offset.
- **Open image** opens the prepared page full size in a new tab.
- Available while a run is still streaming: the source image appears as soon as a page
  starts, with text filling in beside it.

Prepared pages are cached in memory per upload (`MAX_JOBS = 5`, oldest evicted) and
served from `GET /api/page/<job>/<n>`. Nothing is written to disk.

## Pipeline

Each run is up to three passes against the model server:

1. **OCR** — the page image in, a verbatim transcript out, streamed to the browser as it
   arrives.
2. **Extract** — the transcript back in as *text*, structured JSON out. Text-only, so it
   costs a fraction of pass 1. Every returned value is then traced back to the transcript
   **in Python**: a value that is not in the text was not read, it was written, and it is
   flagged rather than displayed as data. Shown in **Fields**.
3. **Verify** — the numbers are checked, again **in Python**: column sums, per-line
   arithmetic, VAT rate, and the Thai amount-in-words against the numeral. Shown in
   **Numbers**.

Passes 2 and 3 are deterministic on purpose — the arithmetic and the grounding check are
done by this app, not asked of the model. Disable them with `EXTRACT=0`, or send
`extract=0` with the request. Both can be re-run on their own from the page without
re-reading the page, or via `POST /api/extract` and `POST /api/verify`.

Nothing flagged is deleted or rewritten. A flagged value stays visible with its flag,
because the one thing worse than a wrong value is a wrong value the tool quietly removed
before anyone compared it with the page.

### Extracted fields

Twenty-nine scalar fields, plus `line_items` and `other_fields`. The **Fields** tab groups
them the way the page is laid out:

| Group | Keys |
|---|---|
| Document | `document_type`, `document_number`, `issue_date`, `due_date`, `service_period`, `currency`, `vat_rate` |
| References | `reference_document`, `po_number`, `original_invoice_number`, `contract_number`, `customer_code`, `location_code` |
| Seller | `seller_name`, `seller_tax_id`, `seller_branch`, `seller_address` |
| Buyer | `buyer_name`, `buyer_tax_id`, `buyer_branch`, `buyer_address` |
| Totals | `subtotal`, `vat_total`, `amount_incl_vat`, `withholding_tax_total`, `net_payable`, `amount_in_words` |
| Payment | `payment_method`, `payment_reference` |

**The totals are four separate figures, not one grand total.** `subtotal` is before VAT,
`amount_incl_vat` is after VAT and before withholding, `net_payable` is what is left to
pay once withholding comes off. Only the lines the page actually prints get filled; the
rest stay empty rather than being derived.

Whether the printed figures already carry VAT is decided before anything is checked, and
the **Fields** tab labels itself from it — the money column reads *Amount (incl. VAT)* or
*Amount (ex VAT)*, with a note above the fields saying which basis the page is on and what
said so.

The **References** group hides itself entirely when it comes back empty, and a cash
receipt legitimately has none of those fields.

### Context window

Extraction sends the whole transcript, so the server's context has to hold the prompt, the
document and the reply. Too small and the JSON comes back truncated.

The **Context** dropdown sets it per request (4096 – 32768) without restarting anything; it
starts at `OLLAMA_NUM_CTX` (default 8192). A change is refused with 409 while a job is
running, so a batch cannot be measured half on one window and half on another.

It applies to Ollama only, whose OpenAI-compatible endpoint has no other way to set it.
The dropdown is disabled while llama-server is active, because its window is fixed at
launch and there is no per-request equivalent. Setting it on the Ollama server instead
also works:

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
JSON reply is labelled as a loop rather than surfacing as a cryptic parse error.
**Re-extract** retries just that pass. If a document loops repeatedly, extracting one page
at a time is the reliable workaround; raising the token cap is not.

## Run log

Every document read appends one row to `logs/runs.csv` — from the page, from the queue,
from `/api/ocr`, and from `compare.py`. The **Run log** card at the bottom of the page
shows the recent rows and links the CSV for a spreadsheet.

**It holds measurements only.** No transcript, no extracted fields, no page images — so
it stays small and is safe to keep:

| Column | |
|---|---|
| `timestamp` | local time, seconds resolution |
| `file`, `file_size_mb`, `pages`, `detail`, `source` | what was read, and how it got in (`upload`/`folder`/`case`/`queue`) |
| `server`, `backend`, `model` | which endpoint and model actually ran it |
| `seconds`, `prefill_seconds`, `decode_seconds` | runtime, split into prompt processing and generation |
| `tokens`, `tokens_per_second` | OCR output tokens; the rate is decode-only |
| `extract_seconds`, `extract_tokens`, `verify_seconds` | passes 2 and 3 |
| `grounded_pct`, `ungrounded`, `fields_missing` | share of extracted values found in the transcript, how many were not, and how many fields the document does not state |
| `p1_present`, `p1_absent`, `p2_present`, `p2_absent` | field coverage by delivery tier — how many of the 14 priority-1 and 14 priority-2 keys came back filled. Counts only; the values stay out of the file. Blank on a run that extracted nothing, which is not the same as zero |
| `case`, `char_accuracy`, `word_accuracy`, `char_accuracy_no_marks` | percentages, blank when the input has no ground truth |
| `verdict` | the number-check result |
| `status`, `error` | `ok` / `truncated` / `looped` / `cancelled` / `error` |

Coverage is not correctness. `p1_present` counts what came back filled, not what came back
right — read it beside `grounded_pct`, because a run can fill all 28 keys and get half of
them wrong.

Failed and cancelled runs are logged too — a read that died after four minutes is exactly
what you want a record of. Logging never raises: a log that breaks the run it is logging
would be worse than a missing row.

Adding a column is safe on an existing log: the next run widens the header and leaves the
older rows blank in the new columns rather than re-labelling what is already there.

Move the file with `OCR_LOG_DIR`. Written as UTF-8 with a BOM so Excel opens Thai
filenames correctly; the header is written once, and new columns are only ever appended,
so old rows stay readable.

Because the row carries the server, the model and the accuracy together, this is the
straightforward way to answer "is 11434 actually better than 8080 on my documents" —
run the same cases on each and compare the columns.

## Checking accuracy against ground truth

`solution/` holds a hand-transcribed expected transcript for each PDF in `mockOcr/`.
Scoring lives in `scoring.py` and is shared by the web page and the CLI, so the browser
and the terminal can never report different numbers for the same run.

Ground truth is **Markdown** — `solution/sol001.md` … `sol005.md` — so the tables render
when you open them. Edit these directly; they are the file the scores are computed from.

Table markup is normalised away before comparison, so the pipe tables here and the HTML
`<table>` that usually comes back compare equal. `<br>` inside a Markdown cell counts as
the line break it stands for, matching a real newline inside an HTML cell.

Two things to know when editing them:

- A Markdown table needs a uniform column count, so a ragged row is padded with empty
  trailing cells. `sol003` has a 4-column header over 5-column data rows (the satang
  column) and its header carries a padding cell. Padding is ignored by scoring.
- A literal `|` inside a cell must be written `\|`; the scorer unescapes it.

Do not confuse `solution/*.md` (expected text, hand written) with `solution/out/*.txt`
(the transcript of the last run, written automatically). Deleting `out/` is harmless.

### From the web page

Pick a **Benchmark case** from the dropdown and hit **Read document** — the server loads
that PDF itself, so there is nothing to upload.

Or just drop a file in: it is **mapped to its solution automatically**, and a badge under
the file says which one and how it was matched, before you run anything. The browser
hashes the file locally and asks `POST /api/match`; nothing is uploaded for the lookup.
Matching is tried in order of confidence:

| Order | Rule | Survives |
|---|---|---|
| 1 | exact filename | — |
| 2 | filename ignoring case and punctuation | `Receipt CDS ... mock1.PDF` |
| 3 | `aliases` listed in `manifest.json` | any rename you declare |
| 4 | **sha256 of the file contents** | any rename at all |
| 5 | embedded document number (`BS…`, `F…TINV`) | re-exported files |

Rule 5 requires the number to identify exactly one case. `BS2603030054` appears on both
`sol001` (the invoice) and `sol005` (the receipt for the same transaction), so a file
named only that is reported as unmatched rather than scored against a coin-flip — a
confidently wrong accuracy number is worse than none.

Selecting a file clears any benchmark selection and vice versa, so there is never a
question of which input is about to run.

When the input has a known ground truth, an **Accuracy** tab appears with three colour-
coded bars (red < 80%, amber < 95%, green above) and a colourised diff against the
expected text. The headline character accuracy is also appended to the footer line, so
you can see it without switching tabs. Every scored run is written to
`solution/out/<id>.txt`, so `python compare.py <id> --no-run` re-scores exactly what the
page displayed.

### From the CLI

```bash
python compare.py                  # every case
python compare.py sol005           # one case
python compare.py --detail fast    # override Detail
python compare.py --no-run         # re-score saved output without re-running OCR
```

The app must be running: each case is sent through it over HTTP at
`http://127.0.0.1:$PORT`, so the whole real pipeline is exercised rather than a library
call.

Thai in the diff output prints as `?` on a Windows console, which defaults to cp1252
and cannot encode it. The scores are unaffected — only the printing degrades, deliberately,
because a `UnicodeEncodeError` after the OCR has already been paid for loses the whole
run. For readable diffs:

```bash
set PYTHONIOENCODING=utf-8
```

Each case reports:

| Metric | Meaning |
|---|---|
| `char accuracy` | 1 − character edit distance / length |
| `word accuracy` | longest-common-subsequence word overlap |
| `char acc. w/o marks` | same, with Thai tone marks and above/below vowels stripped. If this is much higher than `char accuracy`, the right letters are being read and the marks lost — a different problem from wrong words. |

…followed by a unified diff of expected vs actual. Fresh output is saved to
`solution/out/<id>.txt` so a bad run can be inspected, or re-scored with `--no-run`.

Table markup is normalised before comparison, so an HTML `<table>` and a Markdown pipe
table with the same cells score identically — that is formatting, not recognition. Pass
`--keep-tables` to compare markup literally.

**The scores are only as good as `solution/*.md`.** Those files were transcribed by
reading the rendered pages, not copied from a model's output, but they are one person's
reading — correct them where you disagree. `sol003` is a handwritten scan and its
handwritten values in particular deserve a second look. `solution/manifest.json` maps each
id to its source PDF.

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

`POST /api/ocr` — same fields, blocking, returns the whole JSON at once (including
`page_stats`). It drains the same streaming generator internally, so its timings are
measured identically.

`GET /api/page/<job>/<n>` — PNG of prepared page `n` (0-based) for that upload. `job` is
returned on the `page` and `done` events, and in `/api/ocr`'s response. 404s once the
upload falls out of the cache.

`GET /api/health` — active server status (reachable, kind, model, vision) and whether the
PDF/HEIF decoders are available.

`GET /api/servers` — every configured endpoint, what each one is, and which is active.
`?probe=1` bypasses the status cache.

`POST /api/servers` — `{"url": "...", "model": "..."}`, either field optional. Switches
the app to one of the configured endpoints. 409 while the queue has a job running.

`POST /api/extract` · `POST /api/verify` — re-run pass 2 or pass 3 on a transcript the app
already has, without re-reading the page.

`POST /api/match` — look up the ground-truth case for a file by name or sha256.

`GET /api/runs?limit=50` — recent run-log rows, newest first, plus totals.
`GET /api/runs.csv` — the log file itself.

## Layout

| File | What it owns |
|---|---|
| `app.py` | Flask routes, image preparation, the three passes, streaming. |
| `config.py` | Every path and the typed environment readers. The only place that knows where anything lives. |
| `backends.py` | Endpoint probing and switching; every llama.cpp-vs-Ollama difference. |
| `grounding.py` | Checking extracted fields against the transcript they came from. |
| `verify.py` | Arithmetic and VAT checks on the extracted numbers. |
| `scoring.py` | Ground-truth lookup and accuracy scoring, shared by page and CLI. |
| `runlog.py` | The CSV run log. |
| `jobs.py` | The in-process queue and its worker pool. |
| `compare.py` | CLI benchmark runner. |
| `package.py` | Builds the deployable zip. |

Adding a new source file means adding it to `FILES` in `package.py`, or it will not be
in the archive.

## Note when editing

`app.py` runs with `debug=False`, so **Jinja caches compiled templates for the process
lifetime** — edits to `templates/index.html` need a server restart, not just a browser
reload.

Settings are read **once at import**, so a changed environment variable needs a restart
too. Read them through `config.env_*` rather than `os.environ` directly: those clamp to
a range, warn on stderr and fall back to the default, so a bad value in a service file
degrades one setting instead of killing startup with a traceback.

There is no unit test suite. `python compare.py` is the regression test, and it is worth
running after any change to how a request is assembled — the failure mode there is a
silent accuracy drop at HTTP 200, not an exception.
