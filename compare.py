"""Score fresh OCR output against the ground truth in solution/.

    python compare.py                  # run every case
    python compare.py sol005           # one case
    python compare.py --detail low     # override Detail
    python compare.py --no-run         # re-score saved output, no OCR
    python compare.py --app http://ocr.internal:5000    # score a deployed app
    python compare.py --model dots.ocr --profile dots     # a different model entirely

`--model` takes any unique substring of a served model's name and switches the
running app to it before the sweep, the same way the page's picker does (add
`--server URL` to move endpoint too). The server and model in force are printed
above the first case whether or not anything was switched: two scores in this
file are not comparable unless both say what produced them.

And pass 2, against the hand-written field ground truth in solution/*.fields.json:

    python compare.py --fields                 # read the page, then score the fields
    python compare.py --fields --no-run        # extract from the saved transcript
    python compare.py --fields --from-truth    # extract from the ground-truth text
    python compare.py --fields --no-extract    # re-score the saved extraction

`--from-truth` is the one worth knowing about: it feeds `solution/<id>.md` into
pass 2 instead of a transcript, so pass 1 contributes no noise and a change to the
extraction prompt is measured on its own. Every pass-2 measurement in CLAUDE.md
was taken that way.

Sends each PDF through the running web app, so the whole real pipeline is
exercised. Scoring lives in scoring.py and fieldscore.py, shared with the web
page, so the terminal and the browser can never disagree.
"""

import argparse
import json
import os
import sys

import requests

import config
import fieldscore
import scoring

# Every line below can carry Thai -- case filenames, and the diff bodies most of
# all. A cp1252 console cannot encode those, and a benchmark run that dies with a
# UnicodeEncodeError *after* the OCR has been paid for is the worst possible time
# to lose the output. See config.say.
say = config.say

# Where the running app is. Defaults to a local instance on its own default port;
# override to score against a deployed one (`--app`, or OCR_APP_URL in the
# environment) without editing this file.
DEFAULT_APP = os.environ.get("OCR_APP_URL") or f"http://127.0.0.1:{config.PORT}"


def run_ocr(app, pdf, detail):
    data = {} if detail is None else {"detail": detail}
    with pdf.open("rb") as fh:
        res = requests.post(
            f"{app}/api/ocr", files={"image": (pdf.name, fh)}, data=data, timeout=3600
        )
    body = res.json()
    if not res.ok or body.get("error"):
        raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
    return body


def run_extract(app, text, mode=None, job=None):
    """Pass 2 on its own: a transcript in, the extraction result out.

    The app appends its own run-log row for this, the same as the Re-extract
    button does -- a benchmark sweep is exactly the history worth having.
    """
    payload = {"text": text}
    if mode:
        payload["mode"] = mode
    if job:
        payload["job"] = job
    res = requests.post(f"{app}/api/extract", json=payload, timeout=3600)
    body = res.json()
    if not res.ok or body.get("error"):
        raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
    return body


def resolve_model(server, wanted):
    """Match a typed model name against what the endpoint actually serves.

    An Ollama name carries a registry path and a tag --
    `hf.co/mradermacher/dots.ocr-GGUF:latest` -- which nobody wants to retype on
    every benchmark command, so a unique substring is accepted. An ambiguous one
    is refused rather than guessed at: the same rule scoring.case_for_upload
    follows, for the same reason. A sweep attributed to the wrong model is worse
    than a sweep that did not start.
    """
    names = [m["name"] if isinstance(m, dict) else m
             for m in (server.get("models") or [])]
    if wanted in names:
        return wanted
    hits = [n for n in names if wanted.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise RuntimeError(f"no model matching '{wanted}' at {server.get('url')}"
                           f" -- served: {', '.join(names) or 'none'}")
    raise RuntimeError(f"'{wanted}' matches {len(hits)}: {', '.join(hits)}")


def select_profile(app, name):
    """Set the pass-1 shape for the sweep, and say what it is.

    Separate from select_server because it is a different axis: the profile
    decides the prompt and whether a system message is sent, the model decides
    who reads it, and a score is only comparable with another when both match.
    """
    res = requests.post(f"{app}/api/ocr/profile", json={"profile": name}, timeout=30)
    body = res.json()
    if not res.ok or body.get("error"):
        raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
    return body["profile"]


def current_profile(app):
    try:
        return requests.get(f"{app}/api/ocr/profile", timeout=30).json()["profile"]
    except Exception:
        # An older deployment has no such route. Not worth failing a sweep over:
        # the run log still records what it used.
        return None


def select_loop_guard(app, on):
    """Turn the read backstop on or off for the sweep, and say which it is.

    A third axis again: it decides whether a cycling read is CUT SHORT, not what
    is read or who reads it. A sweep taken with it off is not comparable with one
    taken with it on -- a runaway costs the full token budget and comes back
    longer -- so it is stated on the status line beside the model and the profile.
    """
    res = requests.post(f"{app}/api/ocr/loop-guard",
                        json={"loop_guard": on}, timeout=30)
    body = res.json()
    if not res.ok or body.get("error"):
        raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
    return body["loop_guard"]


def current_loop_guard(app):
    try:
        return requests.get(f"{app}/api/ocr/loop-guard",
                            timeout=30).json()["loop_guard"]
    except Exception:
        # Same as current_profile: an older deployment simply has no such route.
        return None


def select_server(app, url, model):
    """Point the running app at a server, and at one of its models.

    Uses the same endpoint the page's picker uses, so the CLI can never select
    something the browser could not -- including the app's refusal to switch
    while the queue has a job running. Two requests when both are given: the
    model name is resolved against the *new* endpoint's list, which is not known
    until the switch has happened.

    Returns the app's own status dict, which is what gets printed above the
    sweep. Every run of this file is a measurement, and a measurement with no
    model name attached is not comparable with the next one.
    """
    res = requests.get(f"{app}/api/servers", timeout=30)
    if not res.ok:
        raise RuntimeError(f"HTTP {res.status_code} from {app}/api/servers")
    server = res.json()["server"]
    for payload in ([{"url": url}] if url else []) + ([{"model": model}] if model else []):
        if "model" in payload:
            payload = {"model": resolve_model(server, payload["model"])}
        res = requests.post(f"{app}/api/servers", json=payload, timeout=60)
        body = res.json()
        if not res.ok or body.get("error"):
            raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
        server = body["server"]
    return server


def bar(value, width=28):
    return "#" * int(round(value * width)) + "." * (width - int(round(value * width)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="case ids, e.g. sol001 (default: all)")
    ap.add_argument("--detail", default=None, help="original|medium|low")
    ap.add_argument("--no-run", action="store_true", help="score saved output only")
    ap.add_argument("--keep-tables", action="store_true",
                    help="compare table markup literally")
    ap.add_argument("--app", default=DEFAULT_APP,
                    help=f"base URL of the running app (default: {DEFAULT_APP})")
    ap.add_argument("--fields", action="store_true",
                    help="also score pass 2 against solution/<id>.fields.json")
    ap.add_argument("--fields-only", action="store_true",
                    help="score pass 2 only -- no transcript diff or accuracy")
    ap.add_argument("--from-truth", action="store_true",
                    help="extract from solution/<id>.md instead of from a "
                         "transcript, so pass 1 contributes no noise")
    ap.add_argument("--no-extract", action="store_true",
                    help="re-score the saved extraction, no model call")
    ap.add_argument("--mode", default=None, choices=["single", "agentic"],
                    help="extraction shape for this run (default: the app's)")
    ap.add_argument("--server", default=None,
                    help="switch the app to this model server first, e.g. "
                         "http://127.0.0.1:11434")
    ap.add_argument("--profile", default=None,
                    help="pass-1 shape for this run: typhoon|dots "
                         "(default: the app's). Sets the OCR prompt and whether "
                         "a system message is sent")
    ap.add_argument("--loop-guard", default=None, choices=["on", "off"],
                    help="whether a cycling read is aborted (default: the "
                         "app's). off lets it run to the token cap; it is still "
                         "detected and still logged as looped")
    ap.add_argument("--model", default=None,
                    help="switch to this model first; a unique substring of the "
                         "name is enough (e.g. dots.ocr). Ollama only -- "
                         "llama-server serves whatever it was started with")
    args = ap.parse_args()
    app = args.app.rstrip("/")

    # --from-truth and --fields-only are both about pass 2, so neither needs
    # --fields spelled out beside it.
    fields = args.fields or args.fields_only or args.from_truth or args.no_extract
    # The transcript is the ground truth itself under --from-truth, so scoring it
    # would report 100% on every case and mean nothing.
    score_text = not (args.fields_only or args.from_truth)

    # Nothing below calls the model when both passes are read from disk, and a
    # switch made for such a run would silently outlive it -- the app keeps the
    # selection.
    calls_model = (not (args.no_run or args.from_truth or args.no_extract)
                   or (fields and not args.no_extract))
    if calls_model:
        try:
            server = select_server(app, args.server, args.model)
            profile = (select_profile(app, args.profile) if args.profile
                       else current_profile(app))
            guard = (select_loop_guard(app, args.loop_guard == "on")
                     if args.loop_guard else current_loop_guard(app))
        except Exception as err:
            say(f"server: {err}", sys.stderr)
            return 2
        say(f"server: {server.get('kind') or 'no server'} at {server.get('url')}"
            + (f"  {server['model']}" if server.get("model") else "")
            + (f"  profile {profile}" if profile else "")
            + ("" if guard is None else
               "  loop guard on" if guard else "  loop guard OFF"))
        if not server.get("available"):
            say(f"  warning: {server.get('reason') or 'not available'}", sys.stderr)
    elif args.server or args.model or args.profile or args.loop_guard:
        say("--server/--model/--profile/--loop-guard ignored: this run makes no "
            "model call.", sys.stderr)

    index = scoring.cases_index()
    if not index:
        say(f"no ground-truth cases found in {scoring.SOLUTION}", sys.stderr)
        return 2
    ids = args.ids or list(index)
    unknown = [i for i in ids if i not in index]
    if unknown:
        say(f"unknown case(s): {', '.join(unknown)}", sys.stderr)
        return 2

    scoring.OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for cid in ids:
        case = index[cid]
        out_path = scoring.OUT / f"{cid}.txt"
        fields_path = scoring.OUT / f"{cid}{fieldscore.TRUTH_SUFFIX}"
        meta = ""
        body = {}

        if args.from_truth:
            actual = case["ground_truth"].read_text("utf-8")
            meta = "  [from ground truth, pass 1 not run]"
        elif args.no_run or args.no_extract:
            if not out_path.exists():
                say(f"{cid}: no saved output; run without --no-run first")
                continue
            actual = out_path.read_text("utf-8")
        else:
            if not case["pdf_path"].exists():
                say(f"{cid}: missing {case['pdf_path']}", sys.stderr)
                continue
            say(f"{cid}: running OCR on {case['pdf']} ...")
            try:
                body = run_ocr(app, case["pdf_path"], args.detail)
            except Exception as err:
                say(f"{cid}: FAILED - {err}", sys.stderr)
                continue
            actual = body["text"]
            out_path.write_text(actual, "utf-8")
            flags = [f for f, on in (("LOOPED", body.get("looped")),
                                     ("TRUNCATED", body.get("truncated"))) if on]
            meta = (f"  [{body['seconds']}s, {body['tokens']} tok, {body['detail']}"
                    + (", " + " ".join(flags) if flags else "") + "]")

        r = (scoring.evaluate(case, actual, ignore_tables=not args.keep_tables)
             if score_text else {"case": cid, "pdf": case["pdf"], "diff": []})
        results.append(r)

        # ---- pass 2 -------------------------------------------------------
        if fields:
            extracted, note = None, ""
            if args.no_extract:
                if fields_path.exists():
                    extracted = json.loads(fields_path.read_text("utf-8"))
                    note = "  [saved extraction]"
                else:
                    note = "  [no saved extraction; drop --no-extract]"
            elif body.get("extracted") and not args.mode:
                # The read already extracted, in the app's current mode. Asking
                # again would pay for a second pass to measure the same thing.
                extracted = body["extracted"]
            else:
                say(f"{cid}: extracting fields ...")
                try:
                    extracted = run_extract(app, actual, args.mode, body.get("job"))
                except Exception as err:
                    note = f"  [extraction FAILED: {err}]"
            if extracted:
                fields_path.write_text(
                    json.dumps(extracted, ensure_ascii=False, indent=2), "utf-8")
                r["extract_mode"] = extracted.get("mode", "")
                r["extract_seconds"] = extracted.get("seconds", "")
                if extracted.get("partial"):
                    note += "  [partial: the reply was salvaged, not parsed whole]"
                if extracted.get("steps_failed"):
                    note += ("  [steps failed: "
                             + ", ".join(s["id"] for s in extracted["steps_failed"])
                             + "]")
                # Scored here rather than read off the app's own `field_score`, so
                # a deployed instance is scored against the truth files in *this*
                # working copy -- the same rule pass 1 already follows.
                r["fields"] = fieldscore.evaluate(cid, extracted.get("fields"))
            r["fields_note"] = note

        # ---- report -------------------------------------------------------
        say(f"\n{'=' * 78}\n{cid}  ({case['pdf']}){meta}\n{'=' * 78}")
        if score_text:
            say(f"  char accuracy        {bar(r['char_accuracy'])} "
                f"{r['char_accuracy']:6.1%}")
            say(f"  word accuracy        {bar(r['word_accuracy'])} "
                f"{r['word_accuracy']:6.1%}")
            say(f"  char acc. w/o marks  {bar(r['char_accuracy_no_marks'])} "
                f"{r['char_accuracy_no_marks']:6.1%}")
            say(f"  length: expected {r['expected_words']} words, "
                f"got {r['actual_words']}")
        if fields:
            if r.get("fields_note"):
                say(f" {r['fields_note'].strip()}")
            if r.get("extract_mode"):
                say(f"  extraction: {r['extract_mode']} mode, "
                    f"{r.get('extract_seconds', '?')}s")
            if r.get("fields"):
                for line in fieldscore.format_report(r["fields"]):
                    say(line)
        if score_text:
            if r["diff"]:
                say("\n  --- differences ---")
                for line in r["diff"]:
                    say("  " + line)
            else:
                say("\n  exact match")

    if len(results) > 1:
        say(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
        if score_text:
            say(f"  {'case':10} {'char':>8} {'word':>8} {'no-marks':>10}")
            for r in results:
                say(f"  {r['case']:10} {r['char_accuracy']:7.1%} "
                    f"{r['word_accuracy']:7.1%} {r['char_accuracy_no_marks']:9.1%}")
            n = len(results)
            say(f"  {'MEAN':10} {sum(x['char_accuracy'] for x in results)/n:7.1%} "
                f"{sum(x['word_accuracy'] for x in results)/n:7.1%} "
                f"{sum(x['char_accuracy_no_marks'] for x in results)/n:9.1%}")
        if fields:
            scored = [r for r in results
                      if r.get("fields") and not r["fields"].get("error")]
            say("")
            say(f"  {'case':10} {'fields':>8} {'loose':>8} {'precision':>11} "
                f"{'values':>8}")
            for r in results:
                got = r.get("fields") or {}
                if not got or got.get("error"):
                    # No truth file, or no extraction to score. Said as a dash and
                    # a reason rather than as 0%, which would read as an
                    # extraction that got everything wrong.
                    why = got.get("error") or (r.get("fields_note") or "").strip()                         or "nothing extracted"
                    say(f"  {r['case']:10} {'--':>8}   {why}")
                    continue
                overall = got["overall"]
                say(f"  {r['case']:10} {fieldscore.pct(overall['accuracy']):>8} "
                    f"{fieldscore.pct(overall['accuracy_loose']):>8} "
                    f"{fieldscore.pct(overall['precision']):>11} "
                    f"{overall['expected']:>8}")
            if len(scored) > 1:
                # Weighted by values rather than by case: the mean of five
                # percentages taken over three keys and over two hundred says the
                # sol003 truth file matters as much as the sol005 one.
                correct = sum(r["fields"]["overall"]["counts"]["correct"]
                              for r in scored)
                expected = sum(r["fields"]["overall"]["expected"] for r in scored)
                say(f"  {'TOTAL':10} "
                    f"{fieldscore.pct(correct / expected if expected else None):>8} "
                    f"{'':>8} {'':>11} {expected:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
