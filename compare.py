"""Score fresh OCR output against the ground truth in solution/.

    python compare.py                  # run every case
    python compare.py sol005           # one case
    python compare.py --detail fast    # override Detail
    python compare.py --no-run         # re-score saved output, no OCR
    python compare.py --app http://ocr.internal:5000    # score a deployed app

Sends each PDF through the running web app, so the whole real pipeline is
exercised. Scoring lives in scoring.py, shared with the web page, so the terminal
and the browser can never disagree.
"""

import argparse
import os
import sys

import requests

import config
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


def bar(value, width=28):
    return "#" * int(round(value * width)) + "." * (width - int(round(value * width)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="case ids, e.g. sol001 (default: all)")
    ap.add_argument("--detail", default=None, help="fast|balanced|accurate|max")
    ap.add_argument("--no-run", action="store_true", help="score saved output only")
    ap.add_argument("--keep-tables", action="store_true",
                    help="compare table markup literally")
    ap.add_argument("--app", default=DEFAULT_APP,
                    help=f"base URL of the running app (default: {DEFAULT_APP})")
    args = ap.parse_args()
    app = args.app.rstrip("/")

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
        meta = ""

        if args.no_run:
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

        r = scoring.evaluate(case, actual, ignore_tables=not args.keep_tables)
        results.append(r)

        say(f"\n{'=' * 78}\n{cid}  ({case['pdf']}){meta}\n{'=' * 78}")
        say(f"  char accuracy        {bar(r['char_accuracy'])} {r['char_accuracy']:6.1%}")
        say(f"  word accuracy        {bar(r['word_accuracy'])} {r['word_accuracy']:6.1%}")
        say(f"  char acc. w/o marks  {bar(r['char_accuracy_no_marks'])} "
            f"{r['char_accuracy_no_marks']:6.1%}")
        say(f"  length: expected {r['expected_words']} words, got {r['actual_words']}")
        if r["diff"]:
            say("\n  --- differences ---")
            for line in r["diff"]:
                say("  " + line)
        else:
            say("\n  exact match")

    if len(results) > 1:
        say(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
        say(f"  {'case':10} {'char':>8} {'word':>8} {'no-marks':>10}")
        for r in results:
            say(f"  {r['case']:10} {r['char_accuracy']:7.1%} {r['word_accuracy']:7.1%} "
                f"{r['char_accuracy_no_marks']:9.1%}")
        n = len(results)
        say(f"  {'MEAN':10} {sum(x['char_accuracy'] for x in results)/n:7.1%} "
            f"{sum(x['word_accuracy'] for x in results)/n:7.1%} "
            f"{sum(x['char_accuracy_no_marks'] for x in results)/n:9.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
