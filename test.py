"""Run OCR, field extraction, or both end to end -- from the command line, no UI.

Runs the app IN THIS PROCESS through Flask's test client, so nothing has to be
started first: no `python app.py`, no browser. It calls the same routes the page
calls (/api/ocr/stream, /api/extract/stream), so what you see here is what the
page would show, and every run is logged to logs/runs.csv like a page run.

The model server still has to be running -- this app infers nothing itself.

    # OCR only: read a file and print the transcript as it streams
    python test.py ocr mockOcr/invoice_sol002.pdf --server http://127.0.0.1:8000 --model typhoon-ocr

    # OCR a benchmark case and score it against solution/<id>.md
    python test.py ocr --case sol002 --server http://127.0.0.1:8000 --model typhoon-ocr

    # Extraction only, from a transcript file (or from a case's ground truth)
    python test.py extract --text-file page.txt --server http://127.0.0.1:8000 --model typhoon-ocr
    python test.py extract --case sol002 --server http://127.0.0.1:8000 --model typhoon-ocr

    # End to end: read, then extract fields, optionally with a second model/server
    python test.py e2e --case sol002 --server http://127.0.0.1:8000 --model typhoon-ocr \
        --extract-server http://127.0.0.1:8001 --extract-model qwen

Everything the run returned is also saved as JSON (--save, default test_output/).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Thai in a transcript must not crash a cp1252 console after a five-minute read.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def parse_args():
    ap = argparse.ArgumentParser(
        description="OCR / extract / end-to-end test without the web UI.")
    ap.add_argument("what", choices=["ocr", "extract", "e2e"],
                    help="ocr = pass 1 only; extract = pass 2 only; e2e = both")
    ap.add_argument("file", nargs="?",
                    help="document to read (PDF or image); for 'extract', use "
                         "--text-file or --case instead")
    ap.add_argument("--case", help="benchmark case id, e.g. sol002. For ocr/e2e "
                    "it reads that case's file from mockOcr/ and scores the "
                    "transcript; for extract it uses solution/<id>.md as input")
    ap.add_argument("--text-file", help="extract: transcript to extract fields from")
    ap.add_argument("--server", help="model server URL for reading, e.g. "
                    "http://127.0.0.1:8000 (default: auto-detect a live one)")
    ap.add_argument("--model", help="model to read with; a unique part of the "
                    "name is enough (e.g. typhoon-ocr)")
    ap.add_argument("--extract-server", help="separate server for extraction "
                    "(default: the reading server)")
    ap.add_argument("--extract-model", help="model to extract with (default: "
                    "the reading model)")
    ap.add_argument("--detail", choices=["low", "medium", "original"],
                    help="page resolution for OCR (default: the app's, medium)")
    ap.add_argument("--mode", choices=["single", "agentic"],
                    help="extraction shape (default: the app's)")
    ap.add_argument("--save", default="test_output",
                    help="folder for the full JSON result ('' to skip)")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write to logs/runs.csv (logs to a temp dir)")
    ap.add_argument("--quiet", action="store_true",
                    help="do not stream tokens; print the result at the end")
    args = ap.parse_args()

    if args.what in ("ocr", "e2e") and not (args.file or args.case):
        ap.error(f"'{args.what}' needs a file to read, or --case")
    if args.what == "extract" and not (args.text_file or args.case):
        ap.error("'extract' needs --text-file or --case")
    if args.file and not Path(args.file).is_file():
        ap.error(f"no such file: {args.file}")
    if args.text_file and not Path(args.text_file).is_file():
        ap.error(f"no such file: {args.text_file}")
    return args


ARGS = parse_args()
if ARGS.no_log:
    # Settings are read once at import, so this has to be set before app loads.
    import tempfile
    os.environ["OCR_LOG_DIR"] = tempfile.mkdtemp(prefix="ocr-test-log-")

import app as ocr_app  # noqa: E402  (after the env above)
import backends  # noqa: E402

CLIENT = ocr_app.app.test_client()


# --------------------------------------------------------------------- helpers

def rule(title=""):
    print(f"\n{'=' * 8} {title} {'=' * max(4, 60 - len(title))}" if title else "=" * 72,
          flush=True)


def fail(message):
    print(f"\nERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def call(method, url, **kw):
    res = getattr(CLIENT, method)(url, **kw)
    body = res.get_json(silent=True) or {}
    if res.status_code >= 400 or body.get("error"):
        fail(f"{url}: {body.get('error') or f'HTTP {res.status_code}'}")
    return body


def resolve_model(server, wanted):
    """A unique substring of a served model name is enough (as in compare.py)."""
    names = [m["name"] if isinstance(m, dict) else m
             for m in (server.get("models") or [])]
    if wanted in names:
        return wanted
    hits = [n for n in names if wanted.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        fail(f"no model matching '{wanted}' at {server.get('url')} -- served: "
             f"{', '.join(names) or 'none (is the server up?)'}")
    fail(f"'{wanted}' matches {len(hits)} models: {', '.join(hits)}")


def select_servers():
    """Point the app at the requested server(s) and model(s), and say which."""
    if ARGS.server:
        call("post", "/api/servers", json={"url": ARGS.server})
    else:
        picked = backends.autoselect()
        if picked.get("reason"):
            print(f"[test] {picked['reason']}", flush=True)
    server = call("get", "/api/servers")["server"]
    if ARGS.model:
        call("post", "/api/servers",
             json={"model": resolve_model(server, ARGS.model)})
    if ARGS.extract_server:
        call("post", "/api/servers", json={"extract_url": ARGS.extract_server})
    if ARGS.extract_model:
        body = call("get", "/api/servers")
        # `extract` lists its own models only when pass 2 is on another server.
        target = (body.get("extract") if (body.get("extract") or {}).get("models")
                  else body["server"])
        call("post", "/api/servers",
             json={"extract_model": resolve_model(target, ARGS.extract_model)})
    if ARGS.mode:
        call("post", "/api/extract/mode", json={"mode": ARGS.mode})

    body = call("get", "/api/servers")
    server, extract = body["server"], body.get("extract") or {}
    print(f"[test] reading:    {server.get('kind') or 'server'} {server.get('url')} "
          f"model={server.get('model')} available={server.get('available')}")
    if server.get("reason"):
        print(f"[test]             {server['reason']}")
    if ARGS.what != "ocr":
        mode = call("get", "/api/extract/mode").get("mode")
        print(f"[test] extracting: {extract.get('url') or server.get('url')} "
              f"model={extract.get('model') or server.get('model')} mode={mode}")
    print(flush=True)
    return server


def stream(url, **kw):
    """Yield the NDJSON events of a streaming route, one dict per line."""
    res = CLIENT.post(url, buffered=False, **kw)
    if res.status_code >= 400:
        body = res.get_json(silent=True) or {}
        fail(f"{url}: {body.get('error') or f'HTTP {res.status_code}'}")
    buf = ""
    for chunk in res.response:
        buf += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if line.strip():
                yield json.loads(line)
    if buf.strip():
        yield json.loads(buf)


def pct(value):
    return "-" if value is None else f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"


# ------------------------------------------------------------- event printing

def on_event(ev):
    """Print one stream event as it arrives. Returns the final result, if any."""
    kind = ev.get("event")
    if kind == "page":
        rule(f"page {ev['page']} of {ev['total']}  ({ev.get('resolution')})")
    elif kind == "token":
        if not ARGS.quiet:
            sys.stdout.write(ev.get("text", ""))
            sys.stdout.flush()
    elif kind == "page_done":
        bits = [f"{ev.get('new_tokens', ev.get('tokens', '?'))} tokens"]
        if ev.get("seconds") is not None:
            bits.append(f"{ev['seconds']} s")
        if ev.get("tokens_per_second"):
            bits.append(f"{ev['tokens_per_second']} tok/s")
        print(f"\n[page {ev['page']} done: {', '.join(bits)}]", flush=True)
    elif kind in ("looped", "truncated"):
        print(f"[page {ev['page']}: {kind.upper()} -- transcript is incomplete]",
              flush=True)
    elif kind == "extracting":
        rule("extracting fields")
    elif kind == "segmented":
        docs = ev.get("documents") or ev.get("segments") or []
        if isinstance(docs, list) and len(docs) > 1:
            print(f"[file split into {len(docs)} documents]", flush=True)
    elif kind == "classified":
        types = " + ".join(ev.get("doc_types") or []) or "unknown type"
        doc = f" (document {ev['document']} of {ev['documents']})" if ev.get("documents") else ""
        print(f"[classified{doc}: {types}, from {ev.get('doc_type_from') or '?'}]",
              flush=True)
    elif kind == "extract_step" and ev.get("status") == "done":
        state = "FAILED" if ev.get("error") else "ok"
        print(f"  step {ev['step']}/{ev['total']} {ev.get('title') or ev.get('id')}: "
              f"{state}", flush=True)
    elif kind == "table_agent" and ev.get("status") == "done":
        print(f"  table {ev.get('agent')} agent done", flush=True)
    elif kind == "error":
        print(f"\n[error] {ev.get('error')}", flush=True)
    elif kind == "progress" and ev.get("message"):
        print(f"[{ev['message']}]", flush=True)


# -------------------------------------------------------------- result printing

def print_ocr(done):
    rule("OCR result")
    if ARGS.quiet:
        print(done.get("text", ""))
        rule()
    print(f"pages:      {done.get('page_count') or len(done.get('pages') or [])}")
    for key, label in (("seconds", "seconds"), ("tokens", "tokens"),
                       ("tokens_per_second", "tok/s"), ("model", "model"),
                       ("status", "status")):
        if done.get(key) not in (None, ""):
            print(f"{label + ':':<11} {done[key]}")
    truth = done.get("truth") or {}
    if truth.get("char_accuracy") is not None:
        print(f"accuracy:   {pct(truth['char_accuracy'])} characters vs "
              f"solution/{truth.get('case', '?')}.md"
              + (f"  (word {pct(truth['word_accuracy'])})"
                 if truth.get("word_accuracy") is not None else ""))
        split = [f"{name} {pct(truth[key])}" for key, name in
                 (("thai_accuracy", "Thai"), ("latin_accuracy", "English"),
                  ("digit_accuracy", "digits")) if truth.get(key) is not None]
        if split:
            print(f"            {', '.join(split)}")
        print(f"            {truth.get('expected_chars', 0) - truth.get('matched_chars', 0)} "
              f"characters missed, {truth.get('invented_chars', 0)} extra (not charged)")
    elif done.get("truth") is None:
        print("accuracy:   not scored (no ground truth for this file)")


def print_fields(result, title="Extracted fields"):
    if result.get("error"):
        rule(title)
        print(f"ERROR: {result['error']}")
        return
    for i, doc in enumerate(result.get("documents") or [], 1):
        print_fields(doc, f"Document {i}: {' + '.join(doc.get('doc_types') or [])} "
                          f"(pages {doc.get('page_range') or doc.get('pages')})")
    if result.get("documents"):
        print_score(result.get("field_score"), "Whole file")
        return

    rule(title)
    types = " + ".join(result.get("doc_types") or []) or "unknown type"
    print(f"type: {types}   mode: {result.get('extract_mode') or result.get('mode')}"
          f"   model: {result.get('model')}   {result.get('seconds', '?')} s")
    fields = result.get("fields") or {}
    statuses = (result.get("grounding") or {}).get("statuses") or {}
    rows = {r["path"]: r for r in ((result.get("field_score") or {})
                                   .get("scalars", {}).get("rows") or [])}
    width = max([len(k) for k in fields] + [10])
    for key, value in fields.items():
        if key in ("other_fields", "line_items", "income_items"):
            continue
        tags = []
        if statuses.get(key) and statuses[key] != "grounded":
            tags.append(statuses[key])
        row = rows.get(key)
        if row:
            tags.append(row["status"] + ("" if row.get("required", True) else ", optional"))
            if row["status"] not in ("correct", "absent") and row.get("expected"):
                tags.append(f"expected: {row['expected']}")
        shown = value if value not in (None, "") else "-"
        print(f"  {key:<{width}}  {shown}" + (f"   [{'; '.join(tags)}]" if tags else ""))

    for name in ("income_items", "line_items"):
        if fields.get(name):
            print(f"\n  {name}:")
            for n, item in enumerate(fields[name], 1):
                print(f"    {n}. " + " | ".join(f"{k}={v}" for k, v in item.items() if v))
    extra = fields.get("other_fields") or []
    if extra:
        print(f"\n  other_fields ({len(extra)}):")
        for e in extra:
            if isinstance(e, dict):
                print(f"    {e.get('label')}: {e.get('value')}")
            else:
                print(f"    {e}")

    table = result.get("item_table")
    if table and table.get("rows"):
        print(f"\n  item table ({len(table['rows'])} rows, "
              f"is_master_table={table.get('is_master_table')}):")
        print("    " + " | ".join(table.get("columns") or []))
        for r in table["rows"]:
            cells = r.get("cells", r) if isinstance(r, dict) else r
            print("    " + " | ".join(str(c) for c in cells))

    grounding = result.get("grounding") or {}
    if grounding.get("grounded_ratio") is not None:
        print(f"\n  grounded: {pct(grounding['grounded_ratio'])} of values found "
              "in the transcript")
    print_score(result.get("field_score"))


def print_score(score, label="Field score"):
    if not score:
        return
    overall = score.get("overall") or {}
    if overall.get("expected"):
        print(f"  {label}: {pct(overall.get('accuracy'))} -- "
              f"{(overall.get('counts') or {}).get('correct', 0)} of {overall['expected']} correct "
              f"({score.get('scored_scope') or 'scored values'})")


def save(name, data):
    if not ARGS.save:
        return
    folder = Path(ARGS.save)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{ARGS.what}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[test] full result saved to {path}", flush=True)


# ------------------------------------------------------------------- the runs

def run_read(extract):
    form = {"extract": "1" if extract else "0"}
    if ARGS.detail:
        form["detail"] = ARGS.detail
    if ARGS.case and not ARGS.file:
        form["case"] = ARGS.case
        name = ARGS.case
    else:
        name = Path(ARGS.file).stem
        if ARGS.case:
            form["case"] = ARGS.case
    handle = open(ARGS.file, "rb") if ARGS.file else None
    try:
        if handle:
            form["image"] = (handle, Path(ARGS.file).name)
        done, fields = None, None
        for ev in stream("/api/ocr/stream", data=form,
                         content_type="multipart/form-data"):
            on_event(ev)
            if ev.get("event") == "done":
                done = ev
                print_ocr(done)
            elif ev.get("event") == "fields":
                fields = ev
    finally:
        if handle:
            handle.close()
    if done is None:
        fail("the read ended without a result")
    if extract:
        if fields:
            print_fields(fields)
        else:
            print("\n[test] no extraction ran (empty transcript, or EXTRACT=0)")
    save(name, {"ocr": done, "extracted": fields})


def run_extract():
    if ARGS.text_file:
        body = {"text": Path(ARGS.text_file).read_text(encoding="utf-8")}
        if ARGS.case:
            body["case"] = ARGS.case   # score it against that case's truth
        name = Path(ARGS.text_file).stem
    else:
        body = {"case": ARGS.case, "from_truth": True}
        name = ARGS.case
        print(f"[test] extracting from the ground-truth transcript "
              f"solution/{ARGS.case}.md", flush=True)
    rule("extracting fields")
    result = None
    for ev in stream("/api/extract/stream", json=body):
        on_event(ev)
        if ev.get("event") == "fields":
            result = ev
    if result is None:
        fail("the extraction ended without a result")
    print_fields(result)
    save(name, result)


def main():
    select_servers()
    started = time.perf_counter()
    if ARGS.what == "extract":
        run_extract()
    else:
        run_read(extract=ARGS.what == "e2e")
    print(f"[test] done in {time.perf_counter() - started:.1f} s"
          + ("" if ARGS.no_log else " -- logged to logs/runs.csv"), flush=True)


if __name__ == "__main__":
    main()
