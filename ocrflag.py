"""Which value, which figure, which file -- what is wrong with a pass-1 read.

`compare.py` answers *how well was this document read*, in one number per case.
This answers the question that comes straight after it, and it answers it in the
order that decides whether a document is usable:

  **value**  every key/value the truth file states -- names and numbers -- and
             whether the transcript still holds it, with what the read has in
             its place
  **number** every figure the page prints, compared BY VALUE, listing the ones
             the read lost and the ones it invented
  **file**   which document to open first, ranked by what it lost rather than by
             what it scored
  **line and char** (`--text`) the differences underneath all that: a dropped
             line, an invented one, the exact spans inside a garbled one, and
             whether the damage is digits, Thai marks or words

**The first two sections are the report and the last is the appendix, which is
the ordering the user asked for and the one the evidence supports.** A Thai word
garbled in a paragraph of payment terms costs a point of `char_accuracy` and
nothing else; a digit lost from a tax ID, or a name read one character short, is
a field that cannot be extracted correctly however good pass 2 is. Measured on
the ten fixtures, the two orderings disagree flatly: the worst-scoring document
in the corpus loses no value at all, and a document reading 90% loses five.

Everything here is derived from `scoring.py` and nothing is measured twice: the
same `normalise`, the same `fold_spacing`, the same content-only comparison. That
is the point rather than an economy -- **a flag has to mean the same thing the
score charges for**, or the report sends someone to fix text that costs nothing.
Two consequences worth knowing:

* **Layout is never flagged.** Whitespace, line wrapping, cell padding, table
  markup and a printed fill rule come out of both sides first (see
  `scoring.INVISIBLE`, `TABLE_CELL`, `FILL_RULE`), and `fold_spacing` then folds
  away a line the model split or ran together. None of that moves
  `char_accuracy`, so none of it is a problem, and reporting it as one is the
  defect the diff itself was fixed for on 2026-08-27.
* **Reading order is never flagged either**, on the same terms `score` forgives
  it: the flags are computed twice, once against the transcript as printed and
  once against `align_blocks`' reordering of it, and the variant that flags less
  content is the one reported (`order` says which). Without that, a document the
  model walked in a different order flags every line as missing AND invented
  while its accuracy figure says it was read correctly.

Pass 1 only, deliberately -- it reads with `extract=0`. A value that landed in
the wrong key is `fieldscore`'s question, against a different ground truth.

    python ocrflag.py                     # read every case, then flag
    python ocrflag.py sol003 sol006       # two cases
    python ocrflag.py --no-run            # flag the saved solution/out/*.txt
    python ocrflag.py --all               # every value, not only the lost ones
    python ocrflag.py --text              # the line and character appendix too
    python ocrflag.py --detail low
    python ocrflag.py --model dots.mocr --profile dots
    python ocrflag.py --json flags.json   # the whole thing, machine-readable

The model defaults to typhoon because every pass-1 baseline in CLAUDE.md is a
typhoon measurement; `--model` takes any unique substring, the same as
`compare.py`, and the model and profile in force are printed above the report
whether or not either was switched.
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter

import requests

import compare
import config
import fieldscore
import grounding
import scoring
import verify

say = config.say

# The model this is expected to be pointed at. Not a lock -- `--model` overrides
# it -- but a default worth having: a flag report is a list of things to go and
# look at, and one taken on whichever model the app was last left on is a list
# about a configuration nobody chose.
DEFAULT_MODEL = "typhoon"

# How alike two lines have to be before one is called a misreading of the other
# rather than a pair of unrelated lines. Same figure and the same reasoning as
# `scoring.BLOCK_MATCH_MIN`: below it two lines share a few incidental
# characters and nothing else, and pairing them would report a dropped line and
# an invented one as a single garbled line -- which are different failures with
# different fixes.
LINE_PAIR_MIN = 0.45

# A document under this is flagged whatever else it did. A review threshold, not
# a measurement: `settings.MIN_READ_FOR_FIELDS` is 0.75 and is where this project
# stops trusting a transcript enough to score fields from it, so a read above
# that can still be worth a look and one below it certainly is.
FILE_FLOOR = 0.90

# How much content has to be present on the other side before a finding is called
# MOVED rather than missing or invented. Four, the same figure and the same
# reasoning as `fieldscore.PARTIAL_MIN_CHARS` and `grounding`'s note about short
# values: `1` and `.` occur on every page, so a shorter span would report itself
# as found somewhere no matter what happened to it.
MOVED_MIN_CHARS = 4

DIGIT = re.compile(r"\d")
LETTER_OR_DIGIT = re.compile(r"[^\W_]", re.UNICODE)


def read_case(app, case, detail):
    """One document read through the running app, pass 1 only.

    `extract=0` because nothing here looks at a field, and pass 2 on the app's
    current extraction model costs several times the read it would be riding on.
    """
    data = {"extract": "0"}
    if detail:
        data["detail"] = detail
    with case["pdf_path"].open("rb") as fh:
        res = requests.post(f"{app}/api/ocr",
                            files={"image": (case["pdf_path"].name, fh)},
                            data=data, timeout=3600)
    body = res.json()
    if not res.ok or body.get("error"):
        raise RuntimeError(body.get("error", f"HTTP {res.status_code}"))
    return body


def _source_lines(raw: str):
    """content -> the line numbers of the raw file that hold it.

    Best effort, and it is what makes a finding addressable: the flags are
    computed on the NORMALISED text, whose numbering is its own (markup becomes
    newlines, blank lines go), so a line number taken there points at nothing
    anybody can open. Matching back on content finds the line in the file on
    disk, which is where a reader has to look.
    """
    index = {}
    for number, line in enumerate(raw.splitlines(), 1):
        key = scoring.content_only(line)
        if key:
            index.setdefault(key, []).append(number)
    return index


def _locate(index, text):
    """The raw line number for a normalised line, or None.

    Exact content first, then containment -- normalising a table row splits it
    into several lines, so a flagged line is often a fragment of a raw one.
    Ambiguous containment is refused rather than guessed at, the same rule
    `scoring.case_for_upload` follows: a wrong line number sends a reader
    somewhere else with no way to tell that it did.
    """
    key = scoring.content_only(text)
    if not key:
        return None
    hit = index.get(key)
    if hit:
        return hit[0]
    holders = [n for k, ns in index.items() if key in k for n in ns]
    return holders[0] if len(holders) == 1 else None


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _pair(exp, act):
    """Pair the two sides of a replaced block, best pair first.

    By content, never by position -- the rule `fieldscore._pair_rows` follows for
    line items and `align_blocks` for whole pages, and for the same reason: one
    dropped line otherwise shifts every line after it and reports a whole block
    as garbled when one line of it is simply missing.
    """
    scored = sorted(
        ((_ratio(scoring.content_only(e), scoring.content_only(a)), i, j)
         for i, e in enumerate(exp) for j, a in enumerate(act)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    pairs, used_e, used_a = [], set(), set()
    for ratio, i, j in scored:
        if ratio < LINE_PAIR_MIN or i in used_e or j in used_a:
            continue
        pairs.append((i, j))
        used_e.add(i)
        used_a.add(j)
    return (sorted(pairs),
            [i for i in range(len(exp)) if i not in used_e],
            [j for j in range(len(act)) if j not in used_a])


def _classify(expected: str, actual: str, in_number: bool = False) -> str:
    """What KIND of damage a character span is.

    `marks` is separated from `text` for the reason `char_accuracy_no_marks`
    exists: losing a Thai tone mark or an upper vowel is a different failure from
    reading the wrong word, and it is the first thing to go when a page is
    downscaled too far. `digits` is separated because on these documents a wrong
    digit is a wrong amount or a wrong tax ID -- an error that grounds happily
    and reaches a field.

    **A span of pure punctuation counts as digits when it sits BETWEEN two
    figures**, which is the one part of this that is not obvious. `1.234`
    against `1,234` differs by no digit at all and is out by a factor of a
    thousand; filing that under `text` would put the most expensive error on
    these documents in the bucket nobody reads first.

    `in_number` is what keeps that rule narrow, and it was narrowed after
    measuring: testing whether the LINE anywhere carries a figure swept in every
    `), ` read as `). ` on sol004's charge table -- thirteen of them, real
    misreadings and not one of them an amount. A separator is only an amount when
    there is an amount on both sides of it.
    """
    span = expected + actual
    if expected and actual and (scoring.THAI_MARKS.sub("", expected)
                                == scoring.THAI_MARKS.sub("", actual)):
        return "marks"
    # A mark DROPPED or INVENTED outright, which the test above cannot see
    # because one side of it is empty. A Thai tone mark is a combining
    # character and so is neither a letter nor a digit to `\w`, which is what
    # sent it to the punctuation rule below and out as `digits`.
    if span and not scoring.THAI_MARKS.sub("", span):
        return "marks"
    if DIGIT.search(span):
        return "digits"
    if in_number and not LETTER_OR_DIGIT.search(scoring.THAI_MARKS.sub("", span)):
        return "digits"
    return "text"


def _between_digits(text: str, start: int, end: int) -> bool:
    """Whether the span at [start:end) has a figure on both sides of it."""
    return (text[start - 1:start].isdigit() if start else False) \
        and text[end:end + 1].isdigit()


def _char_ops(expected: str, actual: str):
    """The spans inside one line that differ, on content only."""
    exp, act = scoring.content_only(expected), scoring.content_only(actual)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, exp, act, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        # Either side is enough: a thousands separator eaten from the truth and
        # one invented in the reply are the same size of error, and only one of
        # them has digits around it in its own string.
        in_number = (_between_digits(exp, i1, i2)
                     or _between_digits(act, j1, j2))
        ops.append({
            "tag": {"replace": "misread", "delete": "dropped",
                    "insert": "invented"}[tag],
            "at": i1,
            "expected": exp[i1:i2],
            "actual": act[j1:j2],
            "class": _classify(exp[i1:i2], act[j1:j2], in_number),
        })
    return ops


def _moved(findings, exp_lines, act_lines):
    """Reclassify anything that is not missing at all, only somewhere else.

    A truth line the model ran onto the end of another line, or emitted before
    the block it belongs to, is reported by the diff as a line missing HERE and
    text invented THERE -- two findings, twice the characters, for content that
    is present and correct. `char_accuracy` charges next to nothing for it (see
    `scoring.align_blocks`), so neither does this: `moved` is counted and shown,
    and kept out of `flagged_chars`.

    The test is containment of the whole content on the other side, which is why
    it cannot fire on a line the model actually garbled -- that content is not
    there to be found. `MOVED_MIN_CHARS` keeps a two-character label from
    matching half the page.
    """
    exp_all = scoring.content_only("\n".join(exp_lines))
    act_all = scoring.content_only("\n".join(act_lines))
    for f in findings:
        if f["kind"] in ("missing", "invented"):
            key = scoring.content_only(f["expected"] or f["actual"])
            other = act_all if f["kind"] == "missing" else exp_all
            if len(key) >= MOVED_MIN_CHARS and key in other:
                f["kind"] = "moved"
        for op in f["ops"]:
            if op["tag"] == "misread":
                continue
            span = op["expected"] or op["actual"]
            other = act_all if op["tag"] == "dropped" else exp_all
            if len(span) >= MOVED_MIN_CHARS and span in other:
                op["class"] = "moved"
    return findings


def _findings(exp_lines, act_lines):
    """Every line-level problem between one pair of texts, and their char spans."""
    folded = scoring.fold_spacing(exp_lines, act_lines)
    matcher = difflib.SequenceMatcher(
        None,
        [scoring.content_only(ln) for ln in exp_lines],
        [scoring.content_only(ln) for ln in folded],
        autojunk=False,
    )
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        exp, act = exp_lines[i1:i2], folded[j1:j2]
        pairs, lone_e, lone_a = (_pair(exp, act) if tag == "replace"
                                 else ([], list(range(len(exp))),
                                       list(range(len(act)))))
        for i, j in pairs:
            out.append({"kind": "misread", "expected": exp[i], "actual": act[j],
                        "ops": _char_ops(exp[i], act[j])})
        for i in lone_e:
            out.append({"kind": "missing", "expected": exp[i], "actual": "",
                        "ops": []})
        for j in lone_a:
            out.append({"kind": "invented", "expected": "", "actual": act[j],
                        "ops": []})
    return _moved(out, exp_lines, folded)


def _cost(findings) -> int:
    """Content characters the findings account for, moved content excluded.

    The tie-break below, and the only figure that can compare two orderings of
    one transcript: a count of lines would call a document that dropped one long
    table row better read than one that dropped three short labels.
    """
    total = 0
    for f in findings:
        if f["kind"] == "moved":
            continue
        if f["kind"] == "misread":
            total += sum(max(len(o["expected"]), len(o["actual"]))
                         for o in f["ops"] if o["class"] != "moved")
        else:
            total += len(scoring.content_only(f["expected"] or f["actual"]))
    return total


def flag(case, actual_text: str, ignore_tables: bool = True) -> dict:
    """Every file-, line- and character-level problem in one read.

    Both orderings are tried and the cheaper one is reported, mirroring
    `scoring.score`, which keeps the better of the aligned and positional edit
    distances. Reporting the positional flags for a transcript whose accuracy
    came from the aligned pass would describe a document the score never saw.
    """
    raw_truth = case["ground_truth"].read_text("utf-8")
    expected = scoring.normalise(raw_truth, ignore_tables)
    actual = scoring.normalise(actual_text, ignore_tables)
    exp_lines = expected.splitlines()

    variants = {
        "as printed": _findings(exp_lines, actual.splitlines()),
        "realigned": _findings(
            exp_lines,
            scoring.normalise(scoring.align_blocks(expected, actual),
                              ignore_tables).splitlines()),
    }
    order, findings = min(variants.items(), key=lambda kv: _cost(kv[1]))

    truth_at = _source_lines(raw_truth)
    read_at = _source_lines(actual_text)
    for f in findings:
        f["truth_line"] = _locate(truth_at, f["expected"]) if f["expected"] else None
        f["read_line"] = _locate(read_at, f["actual"]) if f["actual"] else None

    result = scoring.score(expected, actual)
    counts = Counter(f["kind"] for f in findings)
    ops = [o for f in findings for o in f["ops"]]
    return {
        "values": check_values(case["id"], actual_text),
        "numbers": check_numbers(raw_truth, actual_text),
        "case": case["id"],
        "truth_file": case["ground_truth"].name,
        "order": order,
        "char_accuracy": result["char_accuracy"],
        "expected_chars": result["expected_chars"],
        "flagged_chars": _cost(findings),
        "lines": {"missing": counts["missing"], "invented": counts["invented"],
                  "misread": counts["misread"], "moved": counts["moved"],
                  "total": len(exp_lines)},
        "chars": Counter(o["class"] for o in ops),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Values and numbers -- what the report leads with.
#
# The character sections above answer "what did the model get wrong". These
# answer the question that actually decides whether a document is usable: **did
# the values survive the read**. A Thai word garbled in a paragraph of terms and
# conditions costs a point of `char_accuracy` and nothing else; a digit lost from
# a tax ID, or a name read one character short, is a field that cannot then be
# extracted correctly however good pass 2 is. Those are not the same finding and
# should not be read out of the same list.
# ---------------------------------------------------------------------------

# A number as it is printed: digits (Thai digits included, which `squash`
# normalises) with separators inside the run. Deliberately greedy about `,` `.`
# `/` and `-` so a date, a document number and an amount all come out whole --
# a rule that split `17/3/69` into three numbers would report two of them found
# in a page full of small figures.
_PRINTED_NUMBER = re.compile(r"[\d๐-๙][\d๐-๙.,/-]*")

# Below this many digits a "number" is a list marker, a row index or a page
# number, and it occurs on every page. Same reasoning as `MOVED_MIN_CHARS` and
# `grounding`'s note that a one- or two-character value grounds in any page of
# text: a rule that flags every `1` reports noise in the section that must not
# have any.
NUMBER_MIN_DIGITS = 3


def _value_kind(key: str, value: str) -> str:
    """What sort of value a key holds, from the key and the truth's own value.

    Derived rather than tabulated, so a key added to the schema is classified
    without this file being edited -- and so the classification describes what
    the page actually prints for THIS case rather than what the key is called.
    """
    if key.endswith("_name"):
        return "name"
    if key.endswith(("_date",)) or re.fullmatch(r"[\d๐-๙/.\- ]+", value or ""):
        return "date" if key.endswith("_date") else "number"
    digits = sum(c.isdigit() for c in value or "")
    return "number" if digits and digits >= len(value.replace(" ", "")) / 2 else "text"


def _stated_values(case_id: str):
    """The keys this case's truth file states, straight from the JSON.

    **Read from the file rather than through `fieldscore.load_truth`, and that is
    the point.** The loader drops `prompts.RETIRED_KEYS` -- which is where
    `seller_name` and `buyer_name` went when the schema narrowed to the
    requirement's eleven. The values are still in the files, they are still what
    the page prints, and a report about whether the names survived the read has
    to see them. Nothing here scores a schema, so nothing here needs the schema.

    A key left `""` states that the page does not print it, and `null` that
    nobody has checked; neither can be looked for in a transcript, so both are
    skipped. `-` is a printed dash and is skipped for the same reason.
    """
    path = fieldscore.truth_path(case_id)
    if not path.exists():
        return []
    raw = json.loads(path.read_text("utf-8"))
    out = []
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        readings = [r for r in fieldscore.accepted(value)
                    if isinstance(r, str) and r.strip() and r.strip() != "-"]
        if readings:
            out.append((key, readings))
    return out


def _nearest(expected: str, lines, numbers):
    """The closest thing the read does hold, so a loss says what it became.

    A value reported only as missing sends someone back to the page to find out
    what happened to it; `0155737222723 -> 015573722723` says it in one line.
    Numbers are matched against the read's printed figures and everything else
    against its lines, because a name's neighbourhood is its line while a
    figure's is itself.
    """
    want = scoring.content_only(expected)
    best, ratio = "", 0.0
    if _PRINTED_NUMBER.fullmatch(expected.replace(" ", "")):
        for candidate in numbers:
            r = _ratio(want, scoring.content_only(candidate))
            if r > ratio:
                best, ratio = candidate, r
        return (best.strip(), round(ratio, 2)) if ratio >= 0.5 else ("", 0.0)

    # A short value sits INSIDE a line rather than being one -- a branch code
    # after a tax ID, a name in a letterhead run together with an address -- so
    # comparing it with whole lines scores it against everything else on the
    # line and reports a value that is plainly there as nowhere to be found.
    # The window is taken around the longest run the two have in common.
    for line in lines:
        match = difflib.SequenceMatcher(None, want, scoring.content_only(line),
                                        autojunk=False).find_longest_match()
        if not match.size:
            continue
        pad = max(4, len(want) // 2)
        start = max(0, match.b - match.a - pad)
        window = line[start:start + len(want) + 2 * pad]
        r = _ratio(want, scoring.content_only(window))
        if r > ratio:
            best, ratio = window, r
    # Below this the "nearest" is a different value that happens to share a few
    # characters, and printing it would invent a misreading that did not happen.
    return (best.strip(), round(ratio, 2)) if ratio >= 0.5 else ("", 0.0)


def check_values(case_id: str, transcript: str):
    """Every value the truth file states, and whether the read still holds it.

    `grounding.Source` is the test, which is deliberate: it is the same matcher
    that decides whether an EXTRACTED value is on the page, so a value this
    reports as lost is exactly a value pass 2 could not have grounded. Loose
    about presentation and strict about content -- `1,200` finds `1,200.00`, a
    baht|satang cell finds its decimal -- so nothing here flags a difference in
    how a figure is printed.
    """
    source = grounding.Source(transcript)
    lines = [ln for ln in transcript.splitlines() if ln.strip()]
    numbers = _PRINTED_NUMBER.findall(transcript)
    out = []
    for key, readings in _stated_values(case_id):
        held = next((r for r in readings if source.holds(r)), None)
        expected = readings[0]
        near, ratio = ("", 0.0) if held else _nearest(expected, lines, numbers)
        out.append({
            "key": key,
            "kind": _value_kind(key, expected),
            "expected": expected,
            "readings": readings,
            "status": "read" if held else "lost",
            "nearest": near,
            "nearest_ratio": ratio,
        })
    return out


def check_numbers(expected_text: str, actual_text: str):
    """Every figure the page prints, against every figure the read produced.

    Compared BY VALUE through `grounding`, not as text, so `1,200` and `1,200.00`
    are one number and a baht|satang cell is the decimal it stands for. What is
    left is a figure that changed or vanished -- which on these documents is an
    amount, a tax ID, a date or a document number, and is the one class of error
    that survives every check downstream: an invented figure grounds, sums
    correctly if the model is consistent, and reads exactly like a real one.
    """
    def printed(text):
        seen = {}
        for token in _PRINTED_NUMBER.findall(text):
            token = token.rstrip(".,/-")
            if sum(c.isdigit() for c in token) < NUMBER_MIN_DIGITS:
                continue
            value = verify.parse_amount(token)
            if value is not None:
                seen.setdefault(round(abs(value), 2), token)
        return seen

    want, got = printed(expected_text), printed(actual_text)
    return {
        "lost": [t for v, t in sorted(want.items()) if v not in got],
        "invented": [t for v, t in sorted(got.items()) if v not in want],
        "expected": len(want),
        "read": len(got),
    }


def _lost_values(report):
    return [v for v in report["values"] if v["status"] == "lost"]


def _flagged(report) -> bool:
    """Whether the document itself is a problem, not merely imperfect.

    **A lost value flags the document however well it scored**, and that is the
    ordering this report is built on: a page can read at 93% and still have lost
    the one figure somebody needs. A read that did not finish -- looped,
    truncated -- is flagged whatever it scored too, because the transcript is a
    fragment and the score describes only the part that arrived.
    """
    return (bool(report.get("status")) or bool(_lost_values(report))
            or bool(report["numbers"]["lost"])
            or report["char_accuracy"] < FILE_FLOOR)


def _preview(f, width):
    text = f["expected"] or f["actual"]
    return text if len(text) <= width else text[:width - 1] + "…"


def report(reports, top, width, show_text=False, show_all=False):
    # Ordered by what was lost, not by what was scored: a document that read at
    # 93% and dropped a tax ID is a worse problem than one that read at 78% and
    # kept every value, and the whole point of this report is to say so.
    worst = sorted(reports, key=lambda r: (-len(_lost_values(r)),
                                           -len(r["numbers"]["lost"]),
                                           r["char_accuracy"]))
    say("")
    say("FILES  most lost first")
    for r in worst:
        n, lost = r["numbers"], _lost_values(r)
        say(f"  {'FLAG' if _flagged(r) else '    '} {r['case']}  "
            f"{r['char_accuracy'] * 100:5.1f}%  "
            f"values {len(lost)} lost of {len(r['values']):2d}   "
            f"numbers {len(n['lost'])} lost of {n['expected']:3d}, "
            f"{len(n['invented'])} invented"
            + (f"  {r['status'].upper()}" if r.get("status") else ""))

    say("")
    say("VALUES  the key/value the truth file states, and whether the read still "
        "holds it")
    say("        (`lost` is the test grounding uses, so a lost value is one pass "
        "2 could not have grounded;")
    # grounding's own standing caveat, repeated here because this section prints
    # `ok` beside values short enough for it to apply to -- บาท, a branch word.
    say("         `ok` on a very short value is weak evidence -- two or three "
        "characters are found in any page)")
    for r in worst:
        lost = _lost_values(r)
        shown = r["values"] if show_all else lost
        if not shown:
            say(f"  {r['case']}  all {len(r['values'])} values read")
            continue
        say(f"  {r['case']}  {len(lost)} lost of {len(r['values'])}")
        for v in sorted(shown, key=lambda v: (v["status"] == "read", v["kind"])):
            mark = "LOST" if v["status"] == "lost" else "ok  "
            got = (f"   read has  {v['nearest']!r}" if v["nearest"]
                   else "   not in the read at all" if v["status"] == "lost" else "")
            say(f"    {mark} {v['kind']:6} {v['key']:22} "
                f"{v['expected'][:width]!r}{got}")

    say("")
    say("NUMBERS  every figure the page prints, compared BY VALUE -- so a figure "
        "printed one way")
    say(f"         and read another is not flagged, and one under "
        f"{NUMBER_MIN_DIGITS} digits is not counted")
    for r in worst:
        n = r["numbers"]
        if not n["lost"] and not n["invented"]:
            say(f"  {r['case']}  all {n['expected']} figures read")
            continue
        say(f"  {r['case']}  {len(n['lost'])} lost of {n['expected']}, "
            f"{len(n['invented'])} invented")
        if n["lost"]:
            say(f"    lost      {', '.join(n['lost'][:top])}"
                + (f"  ... +{len(n['lost']) - top}" if len(n["lost"]) > top else ""))
        if n["invented"]:
            say(f"    invented  {', '.join(n['invented'][:top])}"
                + (f"  ... +{len(n['invented']) - top}"
                   if len(n["invented"]) > top else ""))

    if not show_text:
        spans = sum(len(f["ops"]) for r in reports for f in r["findings"])
        lines = sum(r["lines"]["missing"] + r["lines"]["invented"]
                    + r["lines"]["misread"] for r in reports)
        say("")
        say(f"(also {lines} line and {spans} character differences, most of them "
            "words in prose -- `--text` to list them)")
        return

    for r in worst:
        if not r["findings"]:
            continue
        say("")
        say(f"{r['case']}  {r['truth_file']}  --  "
            f"{r['lines']['missing']} missing, {r['lines']['invented']} invented, "
            f"{r['lines']['misread']} misread"
            + (f"  (+{r['lines']['moved']} moved, which cost the score nothing)"
               if r["lines"]["moved"] else ""))
        # Moved content is printed last: it is shown because a reader comparing
        # the two texts by eye will see it and wonder, not because it is a
        # problem, and putting it above the real findings buries them.
        ordered = sorted(r["findings"],
                         key=lambda f: (f["kind"] == "moved",
                                        f["truth_line"] is None,
                                        f["truth_line"] or 0))
        for f in ordered[:top]:
            where = (f"{r['truth_file']}:{f['truth_line']}" if f["truth_line"]
                     else (f"read:{f['read_line']}" if f["read_line"] else "?"))
            say(f"  {f['kind']:8} {where:22} {_preview(f, width)}")
            for op in f["ops"]:
                say(f"      {op['tag']:8} {op['class']:6} "
                    f"expected {op['expected']!r} -> got {op['actual']!r}")
        if len(ordered) > top:
            say(f"  ... {len(ordered) - top} more (raise --top, or --json)")

    ops = [o for r in reports for f in r["findings"] for o in f["ops"]]
    if not ops:
        return
    kinds = Counter(o["class"] for o in ops)
    say("")
    say("CHARS  " + "  ".join(f"{k} {v}" for k, v in kinds.most_common())
        + f"   ({sum(kinds.values())} spans in all)")
    common = Counter((o["expected"], o["actual"]) for o in ops
                     if o["tag"] == "misread")
    for (exp, act), n in common.most_common(10):
        if n < 2:
            break
        say(f"  x{n:<3} {exp!r} -> {act!r}")

    digits = [(r["case"], f, o) for r in reports for f in r["findings"]
              for o in f["ops"] if o["class"] == "digits"]
    if digits:
        say("")
        say(f"DIGITS  {len(digits)} spans -- a wrong digit is a wrong amount or a "
            "wrong tax ID, and grounding cannot see it")
        for cid, f, o in digits[:top]:
            say(f"  {cid}  {o['tag']:8} expected {o['expected']!r} -> "
                f"got {o['actual']!r}   in: {_preview(f, width)}")
        if len(digits) > top:
            say(f"  ... {len(digits) - top} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="case ids, e.g. sol003 (default: all)")
    ap.add_argument("--detail", default=None, help="original|medium|low")
    ap.add_argument("--no-run", action="store_true",
                    help="flag the saved solution/out/<id>.txt, no OCR")
    ap.add_argument("--keep-tables", action="store_true",
                    help="flag table markup literally too")
    ap.add_argument("--app", default=compare.DEFAULT_APP,
                    help=f"base URL of the running app "
                         f"(default: {compare.DEFAULT_APP})")
    ap.add_argument("--server", default=None, help="switch endpoint before reading")
    ap.add_argument("--model", default=None,
                    help="model to read with, any unique substring "
                         f"(default: {DEFAULT_MODEL})")
    ap.add_argument("--profile", default=None, help="pass-1 profile")
    ap.add_argument("--text", action="store_true",
                    help="also list the line and character differences -- the "
                         "words in prose, which the value sections leave out")
    ap.add_argument("--all", action="store_true",
                    help="list every value, not only the ones that were lost")
    ap.add_argument("--top", type=int, default=25,
                    help="findings printed per case (default: 25)")
    ap.add_argument("--width", type=int, default=90, help="line preview width")
    ap.add_argument("--json", default=None, help="write the whole report here")
    args = ap.parse_args()
    app = args.app.rstrip("/")

    if not args.no_run:
        try:
            server = compare.select_server(app, args.server,
                                           args.model or DEFAULT_MODEL)
            profile = (compare.select_profile(app, args.profile) if args.profile
                       else compare.current_profile(app))
        except Exception as err:
            say(f"server: {err}", sys.stderr)
            return 2
        say(f"server: {server.get('kind') or 'no server'} at {server.get('url')}"
            + (f"  {server['model']}" if server.get("model") else "")
            + (f"  profile {profile}" if profile else ""))
        if not server.get("available"):
            say(f"  warning: {server.get('reason') or 'not available'}", sys.stderr)
    elif args.model or args.server or args.profile:
        say("--model/--server/--profile ignored: --no-run makes no model call.",
            sys.stderr)

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
    reports = []
    for cid in ids:
        case = index[cid]
        out_path = scoring.OUT / f"{cid}.txt"
        status, meta = "", ""
        if args.no_run:
            if not out_path.exists():
                say(f"{cid}: no saved output; run without --no-run first")
                continue
            text = out_path.read_text("utf-8")
        else:
            if not case["pdf_path"].exists():
                say(f"{cid}: missing {case['pdf_path']}", sys.stderr)
                continue
            say(f"{cid}: reading {case['pdf']} ...")
            try:
                body = read_case(app, case, args.detail)
            except Exception as err:
                say(f"{cid}: FAILED - {err}", sys.stderr)
                continue
            text = body["text"]
            out_path.write_text(text, "utf-8")
            status = " ".join(f for f, on in (("looped", body.get("looped")),
                                              ("truncated", body.get("truncated")))
                              if on)
            meta = (f"  [{body['seconds']}s, {body['tokens']} tok, "
                    f"{body['detail']}" + (", " + status if status else "") + "]")
        r = flag(case, text, ignore_tables=not args.keep_tables)
        r["status"] = status
        reports.append(r)
        say(f"  {cid}: {r['char_accuracy'] * 100:.1f}%{meta}")

    if not reports:
        return 1
    report(reports, args.top, args.width, args.text, args.all)
    if args.json:
        payload = [{**r, "chars": dict(r["chars"])} for r in reports]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        say(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
