"""A CSV log of every document read, for comparing servers, models and settings.

Deliberately holds no transcript, no extracted fields and no page images -- only
the measurements: when it ran, which model on which server, how long it took, how
many tokens it produced, which file it read, how big that file was, and how
accurate the result was where a ground truth exists.

That makes the log safe to keep, cheap to append to, and small enough to open in
a spreadsheet after a few hundred runs.

One row per document (not per page); per-page detail stays in the Monitor card,
which is live rather than historical.
"""

import contextlib
import csv
import os
import sys
import threading
from datetime import datetime

import requests

import config
import grounding
import prompts
import scoring
import settings
from settings import SUMMARY_RUNS

# The HTTP functions every model-server call in this project goes through, taken
# at import BEFORE anything can replace them. `backends` and `app` both call
# `requests.post`/`requests.get` by module attribute, so a test that fakes the
# model server does it here, and this pair stops matching.
_REAL_HTTP = (requests.post, requests.get)
_transport_warned = False


def live_transport() -> bool:
    """False when something has replaced the HTTP this app reaches a model with.

    **A stubbed model server is an experiment, and an experiment does not belong
    in the run log.** On 2026-09-03 a fixture verification stubbed
    `requests.post`, ran 26 cells through the real routes in a process pointed at
    the real log, and wrote 52 rows under a model called `stub`. Because the stub
    answered with the ground truth, 24 of them scored 100% -- so a model that does
    not exist sat at the TOP of `by_extract`, above the best real one, and seventh
    in the extract ranking. Nothing on a row says whether the reply came from a
    model or from a fixture, so nothing downstream could have caught it.

    The rule that should have prevented it was already written (*experiments run
    on a second instance with its own `OCR_LOG_DIR`*) and was simply forgotten,
    which is what makes a guard worth having instead of another sentence.

    **Its reach is the mechanism, not the intent.** It catches a replacement of
    `requests.post`/`requests.get`, which is how every stub of this app's model
    server has been written; it cannot see one that swaps the whole `requests`
    module inside another module's namespace, or a fake server on a real socket.
    Neither is a reason to skip the cheap case.

    Set `runlog.ALLOW_PATCHED_TRANSPORT = True` to log anyway -- for a wrapper
    that is not a fake, such as a retry adapter.
    """
    return ALLOW_PATCHED_TRANSPORT or (requests.post, requests.get) == _REAL_HTTP


# Deliberate opt-out for a process that wraps `requests` without faking it.
ALLOW_PATCHED_TRANSPORT = False

ROOT = config.BASE_DIR
LOG_DIR = config.LOG_DIR
LOG_PATH = LOG_DIR / "runs.csv"

# Column order is the file format. Appending a new one is safe because `_migrate`
# rewrites an older file's header on the next run; reordering or renaming is not,
# because it would silently re-label the values in rows already written.
COLUMNS = [
    "timestamp",        # local ISO time, seconds resolution
    "file",             # document name as it was submitted
    "file_size_mb",
    "pages",
    "detail",           # resolution preset
    "source",           # upload | folder | case | queue
    "server",           # endpoint URL
    "backend",          # llama.cpp | ollama
    "model",
    "status",           # ok | truncated | looped | error | cancelled
    "seconds",          # wall clock for the whole document
    "prefill_seconds",
    "decode_seconds",
    "tokens",
    "tokens_per_second",
    "extract_seconds",
    "extract_tokens",
    # single | agentic. Taken from the result rather than from the current
    # setting, so a mode switched during a batch still labels each row with the
    # shape that actually produced it. Blank when the run never extracted.
    "extract_mode",
    # Grounding of the extracted fields against the transcript: the share of
    # values found on the page, and how many were not. Logged because an
    # invented field costs nothing in time or tokens and so shows up in no
    # other column -- a run can look fast, clean and wrong.
    "grounded_pct",
    "ungrounded",
    "fields_missing",
    # Field coverage by delivery tier: how many of the 14 priority-1 and 14
    # priority-2 keys came back filled, and how many did not. Counts only -- the
    # values themselves stay out of this file, same as the transcript does.
    # Coverage is not correctness: read these beside grounded_pct, because a run
    # can fill all 28 and invent half of them.
    "p1_present",
    "p1_absent",
    "p2_present",
    "p2_absent",
    # Pass 2 scored against hand-written field ground truth -- the share of the
    # values the truth file says the document prints that came back correct, and
    # how many values that was. Blank on every document that has no
    # solution/<id>.fields.json, which is most of them, so it is NOT comparable
    # down the column the way grounded_pct is: read a row's accuracy beside its
    # own `field_expected`, because 100% of three keys and 100% of twenty-nine
    # are the same cell and not the same claim.
    "field_acc",
    "field_expected",
    "case",             # ground-truth case id, blank when unscored
    "char_accuracy",
    "word_accuracy",
    "char_accuracy_no_marks",
    "error",
    # Which passes this row is a record of. `ocr` is a document read (pass 1, plus
    # pass 2 where it ran); `extract` is a re-extraction of a transcript already
    # read, so every pass-1 column on it -- pages, seconds, tokens, and the
    # accuracy scores -- is deliberately blank: nothing re-read the page, and
    # copying the earlier row's numbers forward would double-count one read in
    # every total taken over this file. Blank means the row predates the column.
    "run_type",
    # When a re-extraction replaced this row's pass-2 columns, and the row's own
    # `timestamp` therefore no longer says when those figures were measured. Only
    # ever set on a read row, and only by a re-extraction that scored better --
    # see `update_extract`. Blank means the extraction is the one that ran with
    # the read, which is the normal case.
    "extract_updated",
    # Which pass-1 shape read the document: typhoon | dots (prompts.OCR_PROFILES).
    # Appended at the end, like every column before it, because reordering
    # silently re-labels data already written. Taken from the summary rather than
    # from the current setting, so a profile switched during a batch still labels
    # each row with the shape that produced it -- the same rule as extract_mode.
    # Blank on rows written before profiles existed, and on re-extract rows, which
    # did not read a page.
    "ocr_profile",
    # DRY multiplier in force for the run. Recorded because it is the one setting
    # that does NOT reach both backends: llama.cpp applies it, Ollama's
    # OpenAI-compatible endpoint drops it. Rows compared across backends are only
    # meaningful when this reads 0 on both.
    "dry",
    # The third delivery tier -- the requirement's "advanced extraction" set.
    # Appended here rather than beside p2_present for the reason every note above
    # gives: inserting a column mid-file re-labels every value written to the
    # right of it. A reader wanting the three tiers together reads them by name.
    #
    # Tier sizes changed when the schema grew to cover the field requirement, so
    # p1/p2 counts are comparable across the change only as a RATE: present and
    # absent are both written, and their sum is what the tier was on that row.
    "p3_present",
    "p3_absent",
    # Which agentic steps this row's extraction ran, when it ran only some of
    # them. Blank on every ordinary run -- a full walk of the step table names no
    # steps here rather than listing all of them, so the column reads as an
    # exception and not as a setting.
    #
    # It exists because a restricted run is a measurement of ONE step and its
    # coverage columns are not: p1_present reads 3 of 14 on a run that asked for
    # three keys and got all three right, and field_acc is blank because a
    # partial answer is not scored against the whole form. Anything reading those
    # columns has to skip a row that names steps here.
    "extract_steps",
    # How many entries came back under `other_fields`. A count, like the tier
    # columns and for the same reason: those labels are the model's own wording,
    # and wording belongs in this file no more than the transcript does.
    #
    # It is worth a column because the narrowed schema leans on `other_fields`
    # for everything the fourteen keys do not cover -- addresses, due dates, PO
    # and contract numbers, the charges table -- so a run that found a great deal
    # of the page and a run that found almost none of it were, until now,
    # indistinguishable here. Blank where nothing was extracted; 0 where the
    # extraction ran and named nothing.
    "other_fields",
    # Priority-1 values scored CORRECT against solution/<id>.fields.json, and how
    # many values that file rules on. Correctness, not coverage: `p1_present`
    # counts a key that came back filled with anything at all, and the entire
    # reason `fieldscore` exists is that a filled key can be filled with the
    # wrong value -- sol005's `buyer_name` = `10-00-1-Springroll` is present,
    # grounded, and not the buyer's name.
    #
    # Blank on every document with no field truth file, exactly like `field_acc`
    # and for the same reason: a 0 here would read as an extraction that got
    # everything wrong rather than as one nobody has written an answer sheet for.
    # Read `p1_correct` beside `p1_scored` and never down the column on its own --
    # the denominator is whatever that document's truth file states.
    "p1_correct",
    "p1_scored",
    # Priority-1 values scored `partial` -- the model found the right thing and
    # took too much or too little of it. Logged because the comparison tables
    # award it HALF a value: a partial is neither a hit nor a miss, and counting
    # it as either is a claim the score cannot support. The strict count stays
    # in `p1_correct`, so both readings survive in the file.
    "p1_partial",
    # `other_fields` again, with the repetition taken out: how many DISTINCT
    # entries came back (`grounding.list_repetition`). The pair is the whole
    # point -- 104 against 5 is a loop, 12 against 12 is a document.
    #
    # This is the count a comparison between models should use. Ranking on the
    # raw total rewards the exact failure it should penalise: a reply that
    # repeats one invented line a hundred times returns 104 "extra fields" and
    # beats every honest reply on the page.
    "other_distinct",
    # The model pass 2 ran on, where it is not the one that read the page. Blank
    # on the one-model setup every measurement before 2026-08-20 was taken under,
    # so blank reads as "the same as `model`" rather than as unknown.
    #
    # It exists because `model` alone stopped identifying a run the moment the
    # two passes could differ: a row reading typhoon + qwen3.5:4b and a row
    # reading typhoon alone are different propositions with the same `model`.
    "extract_model",
    # Set when that repetition check fired: the same entry three or more times,
    # the same threshold `app._repeated_list` uses. A run flagged here is counted
    # as a FAILURE by the setting tables and is dropped from the extras contest
    # entirely, rather than being allowed to win it.
    #
    # Blank on a run that never extracted, `0` on one that extracted cleanly --
    # the blank-is-not-zero rule the tier columns follow.
    "extract_looped",
    # Every document type pass 2 asked the form of, `+`-joined, and on whose
    # authority -- "case" (the benchmark manifest), "transcript" (classified from
    # the printed heading, in Python) or "caller". Appended at the END of this
    # list, like every column before it, because inserting mid-file re-labels
    # every value to the right of it.
    #
    # **A LIST, because a document is regularly more than one type**: six of the
    # ten fixtures head themselves with a slash, and the form is the union of the
    # types named. A single code here would drop every type after the first,
    # which is the bug this whole column was added alongside fixing.
    #
    # It is worth a column because the FORM changed with it: `p1_present` and
    # `p1_absent` are counted against the keys that document's types asked for,
    # so two rows can read 11/13 and 13/15 and both be complete runs. The
    # denominator is on the row, and this says which form produced it. Blank on
    # rows written before 2026-08-31 and on any run that never extracted.
    "doc_types",
    "doc_type_from",
    # Transcript characters that answer to nothing on the page -- text the model
    # produced and the document does not print. **Appended 2026-09-04, when
    # `char_accuracy` stopped charging for it**: the score is recall of the
    # ground truth now, so a read that transcribes the whole page and adds a
    # paragraph of its own scores like one that added nothing. That is the right
    # rule for pass 2 -- extra text leaves every real value where it was, a
    # dropped value cannot be extracted at all -- and it makes the one failure
    # this project was built to catch invisible in the headline. So it is
    # counted here instead. Read it against `expected_chars`, which is not in
    # this file: 181 invented characters means one thing on a 3400-character
    # page and another on a 634-character one.
    #
    # Blank on a row that read no page, and on every row written before the
    # column existed -- which is every row whose `char_accuracy` was an edit
    # distance, and therefore every row that must not be averaged with a newer
    # one anyway.
    "invented_chars",
    # The same transcript recall taken over the three scripts these pages are
    # written in, and how many characters of each the ground truth holds
    # (`scoring.SCRIPTS`). **Appended 2026-09-04 at the user's request** -- one
    # headline cannot say which of the three a model is bad at, and they are not
    # worth the same: every Mandatory field in the requirement is a figure, a
    # date or an ID, so `digit_accuracy` is the column to read first and a page
    # can score 95% overall while losing the one digit that makes an amount
    # wrong.
    #
    # `latin` is the Latin ALPHABET, which is what "English" means on these
    # pages -- a romanised Thai company name is Latin script and is not English.
    #
    # **The three rates do not add up to `char_accuracy`**: punctuation, symbols
    # and currency signs belong to no script and are in the headline only.
    #
    # The `_chars` cells are the denominators, and they are the reason a rate
    # here can be blank on a row that has a `char_accuracy`: a document printing
    # fewer than `scoring.SCRIPT_MIN_CHARS` of a script scores 0% or 100% and
    # nothing between, so no rate is written and the count says why. Blank is
    # never zero, here as everywhere else in this file.
    "thai_accuracy",
    "latin_accuracy",
    "digit_accuracy",
    "thai_chars",
    "latin_chars",
    "digit_chars",
    # Which FIELD each extraction got right, as `key=letter` pairs joined by
    # `;` -- the one thing this file has never held that a per-document
    # weakness table needs. Appended 2026-09-04 at the user's request: *each doc
    # to see in each doc the model weakness is in what field*.
    #
    # The letters are `fieldscore`'s six verdicts, first letter each:
    #   c correct   p partial   w wrong
    #   m missed (the page states it, the extractor returned nothing)
    #   s spurious (the page does not state it, the extractor filled it anyway)
    #   a absent (both empty -- agreement, and scored neither way)
    # `m` and `s` are OPPOSITE mistakes and neither means "empty"; see
    # `fieldscore.STATUS_MEANING`, which this is the compressed form of.
    #
    # **Scalars only, and the field NAMES only.** No value ever reaches this
    # file -- the same rule that keeps the transcript out of it -- and a table
    # row's cells are left out because `income_items[0].amount_paid` is a path,
    # not a field, and one row of a table is not a weakness of a key.
    #
    # Blank where the document has no field truth file, where the read was too
    # poor to judge the extraction by (`app.apply_read_floor`), and on every row
    # written before the column: the same rows on which `field_acc` is blank, and
    # for the same reason. Requiredness is NOT encoded here -- `doc_types` on the
    # same row says which requirement was in force, and `prompts` says what it
    # demands, so encoding it twice would let the two disagree.
    "field_verdicts",
]

# The value the run was actually made with, taken from `settings` rather than
# re-read here: two independent reads of one environment variable can disagree
# after a clamp or a typo, and this column would then describe a run that never
# happened. `settings` imports nothing but `config`, so there is no import cycle
# (app imports runlog).
_DRY = settings.DRY_MULTIPLIER

_lock = threading.Lock()


# --------------------------------------------------------------------------
# The view a summary is compiled under
#
# **The raw rows are stored whatever the settings say; only what is AVERAGED
# moves** (2026-08-24, at the user's request: *allow me to move ocr threshold /
# x last run and recompute the table with these setting*). A read that scored
# 20% is still written, still shown in the raw table, and still passed to pass 2
# -- what a threshold decides is whether the figure it produced is allowed to
# describe a setting.
#
# Two knobs, both of which already existed as process settings and neither of
# which could be moved without restarting:
#
#   window   -- `settings.SUMMARY_RUNS`, how many recent runs OF EACH GROUP a
#               table covers. 0 is the whole log.
#   min_read -- `settings.MIN_READ_FOR_FIELDS` as a FRACTION, the transcript
#               score under which a field score is not a measurement of the
#               extractor. 0 scores everything.
#
# A thread-local override rather than a parameter threaded through fifteen
# functions: `_field_trusted` is four call levels below `totals` and is reached
# from `by_case`, `by_extract` and `standouts` independently, so a parameter
# would have to exist on every one of them and could be forgotten on exactly one.
# Thread-local because two browsers asking for two different windows at once must
# not see each other's.
_view = threading.local()


def _view_opts() -> dict:
    return getattr(_view, "opts", None) or {}


@contextlib.contextmanager
def view(window=None, min_read=None):
    """Compile the tables under a different window and read threshold.

    Either may be None, which means *use the process setting* -- so a caller
    overriding one does not silently reset the other. Restores whatever was in
    force on the way out, including a nested view's, so this composes.
    """
    previous = getattr(_view, "opts", None)
    _view.opts = {"window": window, "min_read": min_read}
    try:
        yield
    finally:
        _view.opts = previous


def window_size() -> int:
    """Runs per group the current view averages over. See `recent_by`."""
    value = _view_opts().get("window")
    return SUMMARY_RUNS if value is None else value


def read_floor() -> float:
    """Transcript accuracy, as a FRACTION, under which a field score is not the
    extractor's. See `_field_trusted`."""
    value = _view_opts().get("min_read")
    return settings.MIN_READ_FOR_FIELDS if value is None else value


def _row_status(summary, error):
    if error:
        return "error"
    if summary.get("looped"):
        return "looped"
    if summary.get("truncated"):
        return "truncated"
    return "ok"


def _migrate():
    """Widen an existing log written before a column was added.

    Without this, appending a row with more fields than the file's header writes
    the extras unnamed, and `csv.DictReader` drops them into its restkey -- so the
    new column reads back empty for exactly the rows that have it. Caller holds
    the lock.

    Only ever adds: if the file has a column this build does not know about, it is
    left alone rather than dropped, because the log outliving one build's schema is
    the normal case and losing a column is worse than a wide file.

    Returns the field names the caller must write with, which is not always
    COLUMNS: a file carrying an unknown column keeps it, and rows must line up
    with the header actually on disk.
    """
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return COLUMNS
    with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if header == COLUMNS:
            return COLUMNS
        rows = list(reader)

    merged = header + [c for c in COLUMNS if c not in header]
    if merged == header:  # nothing new, only ordering this build does not expect
        return header
    # Written beside the log and swapped in, so an interrupted migration leaves the
    # original intact rather than a half-rewritten log.
    temp = LOG_PATH.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row.pop(None, None)  # restkey leftovers from an earlier bad append
            writer.writerow({c: row.get(c, "") for c in merged})
    os.replace(temp, LOG_PATH)
    return merged


def _pct(value):
    """Accuracy as a percentage with one decimal, blank when not scored."""
    if value is None:
        return ""
    return round(float(value) * 100, 2)


# Everything one extraction measured. Built in one place because `update_extract`
# writes these cells onto a row `record` wrote: two spellings of the same figures
# would drift, and a row half-described by one extraction and half by another is
# not a measurement of either.
EXTRACT_COLUMNS = ("extract_seconds", "extract_tokens", "extract_mode",
                   "extract_steps", "other_fields",
                   "grounded_pct", "ungrounded", "fields_missing",
                   "p1_present", "p1_absent", "p2_present", "p2_absent",
                   "p3_present", "p3_absent",
                   "field_acc", "field_expected", "p1_correct", "p1_scored",
                   "p1_partial", "other_distinct", "extract_looped",
                   "extract_model", "doc_types", "doc_type_from",
                   "field_verdicts")

_TIERS = ("p1_present", "p1_absent", "p2_present", "p2_absent",
          "p3_present", "p3_absent")


def _extract_cells(summary: dict) -> dict:
    """The pass-2 half of a row: what was extracted, how, and how real it is."""
    extracted = (summary or {}).get("extracted") or {}
    fields = extracted.get("fields")
    grounded = extracted.get("grounding") or {}
    repetition = grounding.list_repetition(fields if isinstance(fields, dict) else {})
    return {
        # Which form pass 2 asked for, and how that was decided. Both blank
        # rather than guessed where the run never extracted -- blank is not a
        # type, and this file's standing rule is that blank never means zero.
        # Joined with "+" rather than a comma: this file is CSV, and a comma
        # inside a value is a quoting problem waiting for the one reader that
        # splits by hand.
        "doc_types": "+".join(extracted.get("doc_types") or []),
        "doc_type_from": extracted.get("doc_type_from", ""),
        "extract_seconds": extracted.get("seconds", ""),
        "extract_tokens": extracted.get("tokens", ""),
        "extract_mode": extracted.get("mode", ""),
        # Written only where the two passes differed, so the column reads as an
        # exception. `summary["model"]` is the reading model; on a re-extraction
        # there is no read and nothing to differ from, so it is written outright.
        "extract_model": (extracted.get("model") or "")
                         if (extracted.get("model")
                             and extracted.get("model") != (summary or {}).get("model"))
                         else "",
        "extract_steps": ",".join(extracted.get("steps_only") or []),
        # Blank rather than 0 where nothing was extracted at all, the same rule
        # the tiers follow below: an extraction that never ran did not name zero
        # extra fields, it named none because it never answered.
        **({"other_fields": repetition["entries"],
            "other_distinct": repetition["distinct"],
            "extract_looped": 1 if repetition["looped"] else 0}
           if isinstance(fields, dict) else
           {"other_fields": "", "other_distinct": "", "extract_looped": ""}),
        "grounded_pct": _pct(grounded.get("grounded_ratio")),
        "ungrounded": len(grounded.get("flagged") or []) if grounded else "",
        "fields_missing": len(grounded.get("missing") or []) if grounded else "",
        # Blank rather than 0 where nothing was extracted: a failed run left the
        # tiers unmeasured, which is not the same as measuring them at zero.
        #
        # Counted against `fields_asked` -- the form this document's TYPE asked
        # for -- so `p1_present + p1_absent` is the size of the form that ran.
        # Against the union instead it would report an invoice as missing the
        # keys only a credit note is asked for, and the pair would stop being
        # readable as a denominator. A result from before per-type forms carries
        # no `fields_asked` and falls back to the whole schema, which is what it
        # was counted against when it was written.
        **(grounding.tier_counts(fields, extracted.get("fields_asked")) if fields
           else {k: "" for k in _TIERS}),
        # **No correctness figure where the read was too poor to judge one by**
        # (`app.apply_read_floor`): a field score taken over a broken transcript
        # marks down whatever extracted and lets whatever read get away with it.
        # The score itself is still on the result and the page still marks each
        # value with it -- what must not reach this file is the NUMBER, because
        # every table over this file reads it as the extractor's.
        #
        # Blank here means the same thing it means everywhere else in this row:
        # not measured. `_field_trusted` refuses a written figure from the same
        # kind of row at view time, so the two rules agree; the difference is
        # that this one cannot be undone and that one can.
        **_field_cells(None if extracted.get("fields_unscored")
                       else extracted.get("field_score")),
    }


def _field_cells(score: dict) -> dict:
    """The field score, where the document has field ground truth to score against.

    Blank otherwise, and blank on the error shape too: a cell saying 0% because
    nobody has written a truth file would read as an extraction that got
    everything wrong.
    """
    overall = (score or {}).get("overall") or {}
    if not overall.get("expected"):
        return {k: "" for k in ("field_acc", "field_expected", "p1_correct",
                                "p1_scored", "p1_partial", "field_verdicts")}
    # **The counts come from the headline, which since 2026-09-02 is the
    # requirement's Mandatory set rather than the priority-1 tier.** The column
    # names are a file format and are not renamed, so read a `p1_*` cell as
    # "of what the requirement demands of this document" from that date and as
    # "of the priority-1 tier" before it -- the two are the same keys on an
    # invoice and differ by four on a receipt. Written as a pair or not at all:
    # a count of correct values with no denominator beside it is not a figure
    # anyone can read, and the denominator is per document.
    scored = overall.get("expected") or 0
    counts = overall.get("counts") or {}
    return {"field_acc": _pct(overall.get("accuracy")),
            "field_expected": overall["expected"],
            "p1_correct": counts.get("correct", "") if scored else "",
            "p1_partial": counts.get("partial", "") if scored else "",
            "p1_scored": scored if scored else "",
            "field_verdicts": field_verdicts(score)}


# The six verdicts as one letter each, and back. `fieldscore.STATUS_MEANING` is
# the authority on what they claim; this is only how they are spelled in a CSV
# cell that has to hold eleven of them.
VERDICT_LETTERS = {"correct": "c", "partial": "p", "wrong": "w",
                   "missed": "m", "spurious": "s", "absent": "a"}
VERDICT_OF_LETTER = {v: k for k, v in VERDICT_LETTERS.items()}


def field_verdicts(score: dict) -> str:
    """`key=letter;key=letter` for the scalars one extraction was judged on.

    **Optional fields are in it too.** The headline is the requirement's
    Mandatory set, but *which field is this model weak on* is a question about
    every key it was asked for -- and a key that is Optional on a receipt is
    Mandatory on an invoice, so dropping the Optional half here would make the
    weakness table blind on exactly the documents where the field is easiest to
    fix. Requiredness is read back from `doc_types` and `prompts`, never stored.

    Table cells are deliberately left out: `income_items[0].amount_paid` is a
    path into one row, and one row going wrong is not a weakness of a key.
    """
    rows = ((score or {}).get("scalars") or {}).get("rows") or []
    return ";".join(f"{r['path']}={VERDICT_LETTERS[r['status']]}"
                    for r in rows if r.get("status") in VERDICT_LETTERS)


def parse_verdicts(cell: str) -> dict:
    """One `field_verdicts` cell back as {field: verdict}. {} when unwritten."""
    out = {}
    for part in str(cell or "").split(";"):
        key, _, letter = part.partition("=")
        if key and letter in VERDICT_OF_LETTER:
            out[key] = VERDICT_OF_LETTER[letter]
    return out


def _num(value, default=-1.0) -> float:
    """A cell as a number; blank and unparsable both sort below any real value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_score(cells: dict) -> tuple:
    """How good one extraction was, as a tuple that sorts.

    Three progressively weaker claims about the same pass, asked strongest
    first, so a run only ever falls through to a weaker one where the stronger
    is unavailable on both sides:

    - the priority-1 field score -- values that are the value the field truth
      file says belongs in that key, a `partial` counting half. The only figure
      in this file that says a value landed in the right place. It ranks first
      because the whole reason `fieldscore` was built is that the two figures
      below cannot tell a right value from a merely present one.
    - `p1_present` -- keys that came back filled with anything at all. What
      actually ranks the great majority of rows, because most documents have no
      `solution/<id>.fields.json` and every one of them ties at blank above.
    - `grounded_pct` -- the share of what came back that is printed somewhere on
      the page. A tie-break now rather than a rank of its own.

    **Priority 1 is the only tier here.** `p2_*` and `p3_*` used to rank second
    and are gone from this tuple: since pass 2 narrowed to the requirement's
    priority-1 set they are written from empty lists, so they read 0 on every
    row and separate nothing. Widen the schema again and they belong back in,
    under `p1_correct`.

    Blank sorts below zero, so an extraction that never ran can never displace
    one that did -- and a document nobody has written a truth file for ties at
    blank on `p1_correct` and is ranked on coverage, rather than being judged
    against a correctness figure that cannot be computed for it. That is the
    same blank-means-two-things trap this tuple used to avoid by leaving
    correctness out altogether; what makes it safe to rank on now is that the
    weaker claims are still underneath it rather than replaced by it.

    **One asymmetry worth expecting, and it is transitional.** A row that has
    been scored beats one that has not, whatever the coverage: 1 of 14 values
    correct outranks a row that filled all 14 but predates this column or ran
    before its document had a truth file. That is the rule the user asked for --
    coverage is not the measure -- and it only ever compares a scored row with an
    unscored one. Two rows measured under the same build both carry the column
    and are compared on it.
    """
    scored = _num(cells.get("p1_scored"), 0.0)
    # A rate would not be comparable across two documents' denominators; the
    # half-credited COUNT is, which is the same reason `p1_correct` is in this
    # tuple and `field_acc` is not.
    field = (_num(cells.get("p1_correct")) if scored <= 0 else
             _num(cells.get("p1_correct"), 0.0)
             + 0.5 * max(_num(cells.get("p1_partial"), 0.0), 0.0))
    return (field, _num(cells.get("p1_present")), _num(cells.get("grounded_pct")))


def record(summary: dict, source: dict = None, extras: dict = None) -> dict:
    """Append one run. Returns the row written, or None if logging failed.

    `summary` is the payload both OCR endpoints already build (`summarise()` plus
    truth/extracted). Nothing here is required: a failed run logs what it
    knows and leaves the rest blank, because a row saying a read failed after 40 s
    is the row you most want later.

    **Nothing is written while the model server is being faked** -- see
    `live_transport`. It is refused HERE, at the one place this file is appended
    to, rather than at each of the six callers: the caller that forgets is
    exactly the one that will be written next, and the one that did forget was an
    ad-hoc script that no reviewer of this repo would ever have seen.
    """
    if not live_transport():
        global _transport_warned
        if not _transport_warned:
            _transport_warned = True
            config.say("[runlog] the model server's HTTP is stubbed in this "
                       "process, so runs are NOT being logged -- a fixture's "
                       "answer is not a measurement. Point OCR_LOG_DIR at a "
                       "scratch directory to keep the rows, or set "
                       "runlog.ALLOW_PATCHED_TRANSPORT to log anyway.",
                       stream=sys.stderr)
        return None

    summary = summary or {}
    source = source or {}
    extras = extras or {}

    truth = summary.get("truth") or {}
    if truth.get("error"):
        truth = {}
    error = extras.get("error") or ""

    size = source.get("size_bytes")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": source.get("name", ""),
        "file_size_mb": round(size / 1024 ** 2, 3) if size else "",
        "pages": summary.get("page_count", ""),
        "detail": summary.get("detail", ""),
        "source": source.get("origin", ""),
        "server": summary.get("url", ""),
        "backend": summary.get("backend", ""),
        "model": summary.get("model") or "",
        "status": extras.get("status") or _row_status(summary, error),
        "seconds": summary.get("seconds", ""),
        "prefill_seconds": summary.get("prefill_seconds", ""),
        "decode_seconds": summary.get("decode_seconds", ""),
        "tokens": summary.get("tokens", ""),
        "tokens_per_second": summary.get("tokens_per_second", ""),
        **_extract_cells(summary),
        "case": truth.get("case", ""),
        "char_accuracy": _pct(truth.get("char_accuracy")),
        "word_accuracy": _pct(truth.get("word_accuracy")),
        "char_accuracy_no_marks": _pct(truth.get("char_accuracy_no_marks")),
        # Thai, Latin and numerals apart. `_pct` writes blank for a rate the
        # page had too little of that script to measure, which is exactly what
        # blank has to mean here -- the count beside it says why.
        **{f"{name}_accuracy": _pct(truth.get(f"{name}_accuracy"))
           for name in scoring.SCRIPTS},
        **{f"{name}_chars": truth.get(f"{name}_chars", "")
           for name in scoring.SCRIPTS},
        # Blank rather than 0 where nothing scored the page: no read, or no
        # ground truth to be extra to. The standing rule -- blank is not zero.
        "invented_chars": truth.get("invented_chars", ""),
        "error": str(error)[:300],
        "run_type": extras.get("run_type") or "ocr",
        "ocr_profile": summary.get("ocr_profile", ""),
        "dry": _DRY,
    }

    try:
        with _lock:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            new = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
            fields = _migrate()
            # newline="" per the csv module; utf-8-sig so Excel opens Thai
            # filenames correctly instead of as mojibake.
            with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields,
                                        extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerow(row)
    except Exception:
        # A log that breaks the run it is logging is worse than a missing row.
        return None
    return row


def update_extract(key: dict, summary: dict) -> dict:
    """Put a better re-extraction's figures onto the row of the read it came from.

    The only write in this module that is not an append, and it is deliberately
    narrow: it touches `EXTRACT_COLUMNS` and nothing else, only on the one row
    identified by `key` (the timestamp and file of the read), and only when the
    new extraction scores strictly higher by `extract_score` -- 9/14 priority-1
    values correct replaced by 11/14, never the other way round. The read's own columns --
    pages, timings, tokens, the accuracy scores -- are never touched, because
    nothing re-read the page.

    `extract_updated` is stamped with the re-extraction's own timestamp, so the
    row still says that its pass-2 figures were measured later than its
    `timestamp` claims. The re-extraction keeps its own appended row either way,
    so nothing is lost by this: what changes is which extraction the *read* row
    reports, which is the one every per-document figure is read from.

    Returns {"updated": bool, "before": tuple, "after": tuple} or None when the
    row could not be found or the log could not be written. Never raises.
    """
    key = key or {}
    if not key.get("timestamp"):
        return None
    cells = _extract_cells(summary)
    try:
        with _lock:
            if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
                return None
            fields = _migrate()
            with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))

            # Read rows only, and the last one that matches. The re-extraction
            # appends its own row for the same file, and at second resolution it
            # can carry the same timestamp as the read it came from -- without the
            # run_type test the update lands on that row instead, compares it with
            # itself and reports nothing to do.
            target = None
            for row in reversed(rows):
                if ((row.get("run_type") or "ocr") != "extract"
                        and row.get("timestamp") == key["timestamp"]
                        and row.get("file", "") == key.get("file", "")):
                    target = row
                    break
            if target is None:
                return None

            before, after = extract_score(target), extract_score(cells)
            if after <= before:
                return {"updated": False, "before": before, "after": after}

            target.update(cells)
            target["extract_updated"] = datetime.now().isoformat(timespec="seconds")

            temp = LOG_PATH.with_suffix(".csv.tmp")
            with temp.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    row.pop(None, None)
                    writer.writerow({c: row.get(c, "") for c in fields})
            os.replace(temp, LOG_PATH)
    except Exception:
        return None
    return {"updated": True, "before": before, "after": after}


# The key `read` stamps on every row: its position in the file, counted from the
# top and stable for the life of that row, because the log is append-only
# everywhere except `update_extract` (which rewrites values in place, never the
# order) and `delete`.
#
# It is not a column and is never written back -- every writer here builds its
# row from an explicit field list, so a synthetic key cannot leak into the CSV.
# It exists because the log has no other unique key: two rows of one batch share
# a timestamp to the second and a file name, and `delete` has to name ONE of them.
ROW_INDEX = "_row"


def _read_all() -> list:
    """Every row in file order, oldest first, each stamped with `ROW_INDEX`."""
    if not LOG_PATH.exists():
        return []
    try:
        with _lock:
            with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
    except Exception:
        return []
    for i, row in enumerate(rows):
        row.pop(None, None)   # restkey leftovers from an earlier bad append
        row[ROW_INDEX] = i
    return rows


def read(limit: int = 100) -> list:
    """The most recent rows, newest first."""
    return _read_all()[-limit:][::-1]


def signature() -> dict:
    """A cheap "has the log changed" token: the file's size and mtime.

    **Two stat fields, not a parse.** The summary payload is ~90 KB and compiling
    it walks every table; a page that wanted to notice new rows would otherwise
    have to build all of that on a timer and throw it away unchanged nearly every
    time. This is one `stat()` call.

    It catches an append and also an in-place rewrite -- `update_extract` and
    `delete` both move mtime -- so "changed" means every way this file changes.
    Size alone would miss a rewrite that happened to preserve the length.

    Zeroes for a log that does not exist yet, which is a real state (nothing has
    been run) and not an error: the page shows an empty card either way, and the
    signature starts moving as soon as the first row lands.
    """
    try:
        st = LOG_PATH.stat()
        return {"size": st.st_size, "mtime": round(st.st_mtime, 3)}
    except OSError:
        return {"size": 0, "mtime": 0}


# --------------------------------------------------------------------------
# Asking the log a narrower question
#
# **Built 2026-08-24 at the user's request**, in their terms: *i want to exclude
# weak model in this sol1 run to see how my best model do in sol1*, and *see best
# transcript on sol9 without some weak ocr model*. Both are the same shape -- a
# table over a subset of the rows -- and neither was answerable without editing
# the CSV by hand.
#
# **A filter drops the ROW, not the figure.** A row excluded here is not in
# `runs`, not in the failure rate and not in the window, exactly as if it had
# never been run. That is what makes *how does my best model do on sol001*
# answerable at all; softening it to "shown but not counted" would answer a
# different question and look identical on screen.
#
# Filtering happens BEFORE windowing, so "the last 20 runs of this setting
# without dots.ocr" means twenty rows that survived the filter, and not twenty
# rows of which some were then thrown away.

# What a filter may name, and how a row answers it. The two model fields are
# separate because they are separate questions -- a model can read badly and
# extract well, and this project has two that do -- and `extract_model` falls
# back to `model` for the one-model setup every row before 2026-08-20 was written
# under, which is the same reading `by_extract` takes.
# Which passes a row is a record of. **Three real kinds, not two**, and the log
# is full of all three -- the random test's scopes produce them deliberately
# (`full`, `ocr`, `fields`), and on the log this was written against they were
# 170 / 41 / 80.
#
# It exists because half a pipeline is a thing to look at ON ITS OWN (2026-08-24,
# at the user's request: *since some test is only half, allow to close all
# reading model or all extraction model to see only half pipeline*). Filtering
# that by model cannot work -- see the note on `model` below -- because "no
# reading model" is a property of the run, not a value a model picker can hold.
PIPELINES = ("both", "read", "extract")


def _pipeline(row: dict) -> str:
    """`both`, `read` (pass 1 only) or `extract` (pass 2 only, no page read)."""
    if (row.get("run_type") or "ocr") == "extract":
        return "extract"
    # The same test `by_extract` uses to decide a row has a pass-2 figure at all.
    ran_pass_2 = not ((row.get("p1_present") in (None, ""))
                      and (row.get("other_fields") in (None, "")))
    return "both" if ran_pass_2 else "read"


# --------------------------------------------------------------------------
# Inferred marks: where a read ran, and whether it was warm
#
# **Neither is in the log, and neither can be** -- the app talks HTTP to a server
# it did not launch, so it never knows the GPU layer count or whether the model
# was resident. Both are read off measurements the row already carries, which is
# a heuristic; the page labels them "inferred" wherever they show, and the
# thresholds live in `settings` beside the reasoning. Both are blank -- not
# guessed -- where the figure they need is missing, so they drop out of the
# facets rather than offering an empty bucket to filter on.

def run_hardware(row: dict) -> str:
    """`gpu` or `cpu` from the decode rate, or `` where there is no rate.

    A GPU decodes this workload at tens of tokens a second and a CPU at single
    digits, so `tokens_per_second` separates them cleanly -- see
    `settings.GPU_MIN_TPS`. A proxy, not a probe.
    """
    tps = _num(row.get("tokens_per_second"), None)
    if tps is None or tps <= 0:
        return ""
    return "gpu" if tps >= settings.GPU_MIN_TPS else "cpu"


def run_start(row: dict) -> str:
    """`hot` or `cold` from prefill, or `` where there is no prefill.

    A warm read reuses cached state and pays almost no prefill; a cold one pays
    it in full. Absolute threshold, so it is a per-row fact -- see
    `settings.WARM_PREFILL_MAX`. A re-extraction read no page and has no prefill,
    so it is neither.
    """
    if (row.get("run_type") or "ocr") == "extract":
        return ""
    prefill = _num(row.get("prefill_seconds"), None)
    if prefill is None:
        return ""
    return "hot" if prefill < settings.WARM_PREFILL_MAX else "cold"


FILTER_FIELDS = {
    "case": lambda r: r.get("case") or r.get("file") or "",
    # **The READING model, and blank where nothing was read.** A
    # `run_type=extract` row carries the model that EXTRACTED in this column --
    # it has no reading model, because it read no page -- so returning it here
    # made every fields-only row answer a question about pass 1 that it has no
    # answer to. Excluding a reading model then dropped the extract-only rows
    # that model had extracted, and excluding every reading model emptied the
    # card: 80 rows that never read anything were being matched on a pass they
    # did not run.
    #
    # Blank is dropped from `facets`, so those rows are simply not in the reading
    # picker; and an INCLUDE on a reading model still excludes them, which is
    # correct -- "only what typhoon read" is not a set that contains a run
    # typhoon never read for.
    "model": lambda r: "" if (r.get("run_type") or "ocr") == "extract"
                       else (r.get("model") or ""),
    # **The EXTRACTING model, and blank where nothing was extracted** -- the
    # mirror of the rule above, and it exists for the same reason: turning off
    # every extraction model to see the reading half must leave the reading half,
    # not an empty card. This column is blank on the one-model setup, so it falls
    # back to `model`, which meant a run that never extracted still answered with
    # a model that never extracted for it.
    #
    # "Never extracted" is `_pipeline` == read: no pass-2 figures. That covers a
    # read-only round AND a full round that died before it reached pass 2, which
    # the CSV cannot tell apart -- both leave the pass-2 columns and
    # `extract_mode` blank. The cost is that excluding a model here no longer
    # drops the runs where it failed before returning anything; the benefit is
    # that the picker describes what a row IS, which is what made the reading
    # side wrong.
    "extract_model": lambda r: "" if _pipeline(r) == "read"
                     else (r.get("extract_model") or r.get("model") or ""),
    "pipeline": _pipeline,
    "extract_mode": lambda r: r.get("extract_mode") or "",
    "backend": lambda r: r.get("backend") or "",
    # A lambda, not the function itself: `detail_of` is defined below this
    # block and the dict is built at import. Deferring the lookup to call
    # time is what lets the filter live beside `read`, where it is used.
    "detail": lambda r: detail_of(r),
    "ocr_profile": lambda r: r.get("ocr_profile") or "",
    "status": lambda r: r.get("status") or "",
    "run_type": lambda r: r.get("run_type") or "ocr",
    "source": lambda r: r.get("source") or "",
    # Inferred, not recorded -- see `run_hardware` / `run_start`. Filterable like
    # any other field, so "GPU cold reads only" is one pair of chips, and blank
    # (no rate / no prefill) drops out of the facets rather than offering an
    # empty bucket.
    "hardware": run_hardware,
    "start": run_start,
}


def _values(spec) -> set:
    """One filter clause as a set, tolerant of what a query string can carry."""
    if spec is None:
        return set()
    if isinstance(spec, str):
        spec = [spec]
    return {str(v) for v in spec if str(v) != ""}


def filter_rows(rows: list, include: dict = None, exclude: dict = None) -> list:
    """Rows matching every `include` clause and no `exclude` one.

    `include` is per field: a field naming values keeps only rows whose value is
    one of them, and a field naming none is not a constraint at all -- **an empty
    picker means all of them, never none of them**, which is the difference
    between a filter nobody has touched and one that hides the whole table.

    `exclude` wins where the two name the same value, because it is the narrower
    statement and because that is the only reading under which adding an
    exclusion can never widen a result.

    An unknown field name is ignored rather than raising: the caller is a query
    string, and a stale bookmark should return the table rather than an error.
    """
    include = {k: _values(v) for k, v in (include or {}).items() if k in FILTER_FIELDS}
    exclude = {k: _values(v) for k, v in (exclude or {}).items() if k in FILTER_FIELDS}
    include = {k: v for k, v in include.items() if v}
    exclude = {k: v for k, v in exclude.items() if v}
    if not include and not exclude:
        return rows
    kept = []
    for row in rows:
        values = {field: FILTER_FIELDS[field](row)
                  for field in set(include) | set(exclude)}
        if any(values[f] in v for f, v in exclude.items()):
            continue
        if all(values[f] in v for f, v in include.items()):
            kept.append(row)
    return kept


def facets(rows: list = None) -> dict:
    """Every value each filterable field actually takes, with how many rows hold it.

    **Taken over the whole log, never over the filtered rows**, and that is the
    point of it: a picker built from what survived the filter loses the option
    that would put a row back the moment it is excluded, so an exclusion could
    never be undone from the page. The counts are the unfiltered counts for the
    same reason.

    Blank values are dropped -- a run that failed before it resolved a model has
    no model, and offering `""` as something to filter on names nothing anyone is
    looking for.
    """
    rows = _read_all() if rows is None else rows
    out = {}
    for field, value_of in FILTER_FIELDS.items():
        counts = {}
        for row in rows:
            value = value_of(row)
            if value:
                counts[value] = counts.get(value, 0) + 1
        out[field] = [{"value": v, "runs": counts[v]}
                      for v in sorted(counts, key=lambda k: (-counts[k], k))]
    return out


# Rows removed from the log are appended here rather than dropped on the floor.
# The user asked for a delete and this is one -- the row leaves `runs.csv`, every
# table and every count -- but this project's standing habit with rows it can no
# longer average is to ARCHIVE them (three resets in CLAUDE.md, every one of them
# a move rather than a truncation), and a mis-click is otherwise permanent.
DELETED_PATH = LOG_DIR / "runs.deleted.csv"


def delete(keys: list) -> dict:
    """Remove named rows from the log, keeping a copy in `DELETED_PATH`.

    `keys` are `{"_row": int, "timestamp": str, "file": str}` as the page
    received them from `read`. **The index alone is not trusted**: it is a
    position in a file another request may have appended to or deleted from
    since, so the timestamp and file are checked against the row found there and
    one that does not agree is reported as `stale` rather than deleted. Deleting
    the wrong row is silent and permanent, which is the one mistake this module
    exists not to make.

    Returns {"deleted", "stale", "remaining", "archive"}, or None where the log
    could not be written -- never raises, the same rule `record` follows.
    """
    wanted = {}
    for key in keys or []:
        try:
            index = int(key.get(ROW_INDEX) if isinstance(key, dict) else key)
        except (TypeError, ValueError, AttributeError):
            continue
        wanted[index] = key if isinstance(key, dict) else {}
    if not wanted:
        return {"deleted": 0, "stale": 0, "remaining": None}
    try:
        with _lock:
            if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
                return {"deleted": 0, "stale": len(wanted), "remaining": 0}
            fields = _migrate()
            with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))

            keep, removed, stale = [], [], 0
            for i, row in enumerate(rows):
                row.pop(None, None)
                key = wanted.get(i)
                if key is None:
                    keep.append(row)
                    continue
                # The guard: the row at that position must still be the row the
                # page was looking at. A key that names neither field cannot be
                # checked and is taken at its word.
                agrees = all(row.get(name, "") == key.get(name)
                             for name in ("timestamp", "file") if key.get(name) is not None)
                if agrees:
                    removed.append(row)
                else:
                    stale += 1
                    keep.append(row)
            if not removed:
                return {"deleted": 0, "stale": stale, "remaining": len(keep)}

            new_archive = (not DELETED_PATH.exists()
                           or DELETED_PATH.stat().st_size == 0)
            with DELETED_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields,
                                        extrasaction="ignore")
                if new_archive:
                    writer.writeheader()
                for row in removed:
                    writer.writerow({c: row.get(c, "") for c in fields})

            # Written beside the log and swapped in, so an interrupted delete
            # leaves the original intact -- the shape `_migrate` already uses.
            temp = LOG_PATH.with_suffix(".csv.tmp")
            with temp.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in keep:
                    writer.writerow({c: row.get(c, "") for c in fields})
            os.replace(temp, LOG_PATH)
    except Exception:
        return None
    return {"deleted": len(removed), "stale": stale, "remaining": len(keep),
            "archive": str(DELETED_PATH)}


def recent_by(rows: list, key_of, limit: int = None) -> list:
    """The newest `SUMMARY_RUNS` rows OF EACH GROUP, not of the file.

    **The window is per setting, per model and per document -- not a slice off
    the end of the log** (2026-08-21, at the user's request: *20 means 20 of that
    setting / that model / that doc name*). A global slice was the first shape
    and it is wrong in both directions: twenty rows of one evening's contest wipe
    every other setting out of the tables entirely, while a setting run twice a
    month keeps rows from a build that no longer exists. Per group, every setting
    is described by its own last twenty runs and nothing else -- which is the
    question the ranking tables ask.

    So each table trims with the key it groups on: `by_ocr` by its four-part
    setting, `by_extract` by its three, `by_case` by the document, and each of
    `standouts`' four lists by its own. It is done inside those functions rather
    than by the caller for that reason -- the key is theirs, and `totals` hands
    all four the same untrimmed rows.

    `rows` must be newest first, which is what `read` returns. `limit` 0 (or
    `SUMMARY_RUNS` 0) keeps everything, which is how every figure taken before
    this date was taken.

    The default comes from `window_size`, not from `SUMMARY_RUNS` directly, so a
    request compiling the tables under a different window (`view`) moves every
    table at once rather than the ones somebody remembered to pass it to.
    """
    limit = window_size() if limit is None else limit
    if not limit:
        return rows
    kept, seen = [], {}
    for row in rows:
        key = key_of(row)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= limit:
            kept.append(row)
    return kept


def _setting_key(columns):
    """A row reduced to one of the setting tuples, for `recent_by`."""
    return lambda row: tuple(row.get(column, "") or "" for column in columns)


def _document_key(row) -> str:
    """The document a row is about: the ground-truth case, else the file name.

    The same key `_group` buckets by, so "that doc name" means one thing
    throughout -- an upload nobody has ground truth for still has its own window
    rather than sharing one with every other unscored file.
    """
    return row.get("case") or row.get("file") or ""


def case_counts(rows: list = None) -> dict:
    """How many times each ground-truth document has been READ, over the log.

    Written for the random test, whose fairness rule needs one number per case:
    without it the planner drew a case uniformly at random every round, which
    over any run anyone actually sits through leaves some documents at five runs
    and others at none. See `randomtest.plan`.

    **Re-extraction rows are not counted, and that is the point of counting
    reads rather than rows.** A `run_type=extract` row scored pass 2 against a
    transcript some earlier read produced, so counting it would charge the
    document twice for one reading -- and a read is what the random test plans.

    A failed read still counts. It is a round the document has already been
    given, and skipping it here would keep handing rounds to whichever fixture
    breaks the most -- which is the opposite of spreading them.

    Documents with no row are simply absent; the caller knows which cases exist
    and a zero it invents here would claim this file had said something about a
    document it has never seen.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    counts = {}
    for row in rows:
        case = row.get("case") or ""
        if case and (row.get("run_type") or "ocr") != "extract":
            counts[case] = counts.get(case, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Detail presets the log names and this build no longer has
#
# **The log outlives a rename, and it is full of one.** The presets were renamed
# on 2026-08-21 (`max` -> `original`, `accurate` -> `medium`, `balanced` ->
# `low`) and the old names were dropped from `settings.py` on 2026-08-24 at the
# user's request. The CSV keeps what it recorded -- 110 of the 205 rows in it
# name a preset the picker no longer offers -- and rewriting it is not an option
# this project takes: those rows are true records of what ran, and the same rule
# already governs the two earlier scorer changes.
#
# So the reading lives here, where the tables are, and applies to the tables
# ONLY. `read()` is untouched: the row list on the page is the LOG, and a value
# rewritten there would misreport what ran.

# A rename is folded, because the two names are one pixel budget and therefore
# one measurement. Without this, `by_ocr` reports the same setting twice under
# two names and splits its runs -- which is exactly what it was doing, over more
# rows than it was grouping correctly.
DETAIL_RENAMES = {"max": "original", "accurate": "medium", "balanced": "low"}

# **`fast` is NOT folded, and that is the whole reason this is two tables rather
# than one.** It was 1 MP; the nearest surviving preset is 2 MP. Pooling those
# rows into `low` would average two budgets under one name -- and 1 MP is the
# budget this project deleted for making the model FABRICATE rather than merely
# misread, so the runs least worth pooling into anything are exactly these.
#
# What a retired row may still do is everything that is about a model or a
# document: it was a real read of a real page, and `standouts`, `case_counts`
# and the header means all keep it. What it may not do is name a setting --
# neither in `by_ocr` nor as a `by_case` record -- because the setting does not
# exist and cannot be run again. Naming one is the confidently-wrong label this
# project refuses everywhere else.
RETIRED_DETAILS = frozenset({"fast"})

# The presets in PIXEL-BUDGET order, cheapest first -- `low` (2 MP), `medium`
# (4 MP), `original` (uncapped). Derived from `settings.DETAIL_PRESETS` rather
# than written out, so a preset added or re-budgeted there cannot leave a
# hard-coded list here disagreeing with it.
#
# **The order is what makes a Detail sweep readable**: Detail is the one
# category on this card that is ORDERED, so a table of it read left to right is
# "more pixels ->" and the trend in the row is the finding. Alphabetical would
# put `original` first and `medium` last, which reads as noise. `0` means
# uncapped and is therefore the LARGEST budget, not the smallest -- sorted with
# infinity, or `original` would sort to the cheap end and invert every trend.
def _detail_budget(name: str) -> float:
    budget = settings.DETAIL_PRESETS.get(name)
    if budget is None:
        return float("inf")
    return float("inf") if budget == 0 else float(budget)


DETAIL_ORDER = sorted(settings.DETAIL_PRESETS, key=_detail_budget)


def detail_of(row: dict) -> str:
    """A row's Detail as this build spells it."""
    return DETAIL_RENAMES.get((row.get("detail") or "").strip().lower(),
                              (row.get("detail") or "").strip().lower())


def _current_detail(row: dict) -> bool:
    """False where a row ran at a preset this build cannot produce."""
    return detail_of(row) not in RETIRED_DETAILS


def _for_summary(rows: list) -> list:
    """Rows as the summary tables read them, with renamed Details folded.

    Copies rather than edits in place, so the same list can be handed to
    `read`'s caller and to a table without one changing what the other sees.
    Idempotent, which is what lets `totals` apply it once and each table apply
    it again on rows it was handed.
    """
    folded = []
    for row in rows:
        name = detail_of(row)
        folded.append(row if name == (row.get("detail") or "")
                      else {**row, "detail": name})
    return folded


def retired_detail(rows: list = None) -> dict:
    """How many rows ran at each Detail this build no longer offers.

    Reported rather than left implicit: a table that silently drops a quarter of
    the log reads as a table nobody has run much. Counted over the WHOLE file,
    not a window, because that is the unambiguous statement -- these rows exist
    and are not in the setting tables.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    counts = {}
    for row in rows:
        name = detail_of(row)
        if name in RETIRED_DETAILS:
            counts[name] = counts.get(name, 0) + 1
    return counts


def legacy_char_rows(rows: list = None) -> dict:
    """Reads whose `char_accuracy` was the OLD metric, and how many there are.

    **The scorer changed on 2026-09-04** -- `char_accuracy` was an edit distance
    and is now recall of the ground truth, which stopped charging for content the
    page does not print. Measured on the ten saved outputs, that is worth about
    +7.6 points of mean, so a table averaging both eras under one column name is
    averaging two different questions.

    **The log was deliberately NOT reset for it** (the user's call, and the third
    time this has come up): the rows are real records of real runs, `SUMMARY_RUNS`
    windows every table to each setting's own recent runs, and the old rows
    therefore leave the tables on their own as new ones arrive. This function is
    how far along that is -- reported while it is non-zero and silent afterwards,
    so the mechanism removes its own notice.

    **The discriminator is exact and cost nothing to get.** `invented_chars` was
    appended in the same change, so on any row that has a `char_accuracy` at all,
    a blank one means the edit distance and a filled one means recall. A row with
    no `char_accuracy` scored no page and is in no accuracy mean either way, so it
    is counted in neither figure here.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    scored = [r for r in rows if not _blank(r.get("char_accuracy"))]
    legacy = [r for r in scored if _blank(r.get("invented_chars"))]
    return {"legacy": len(legacy), "scored": len(scored)}


# The knobs one run can be repeated with. Every "best" in `by_case` carries the
# whole set, for the reason `totals` already gives for `best_by_case`: a winning
# score with nothing attached is not a setting anyone can adopt. Kept as one
# tuple because three spellings of "what this run ran under" would drift apart.
SETTING_COLUMNS = ("model", "backend", "detail", "ocr_profile", "extract_mode")


def _setting(row: dict) -> dict:
    """One row reduced to what it ran under, when it ran, and which passes it is.

    `run_type` rides along because a best taken from a re-extraction row is a
    measurement of pass 2 alone -- the transcript it scored was read by an
    earlier run under settings this dict does not describe.
    """
    cells = {c: row.get(c, "") for c in SETTING_COLUMNS}
    cells["timestamp"] = row.get("timestamp", "")
    cells["run_type"] = row.get("run_type") or "ocr"
    return cells


def _elapsed(row: dict) -> dict:
    """The two passes' wall clock, and their sum where the row ran both.

    `total` is None when either half is missing rather than the half that is
    there: a read that never extracted took less time because it did less work,
    and calling that the faster setting is how a summary recommends doing half
    the job.
    """
    read_s, extract_s = _num(row.get("seconds"), None), _num(row.get("extract_seconds"), None)
    total = None if read_s is None or extract_s is None else round(read_s + extract_s, 2)
    return {"seconds": read_s, "extract_seconds": extract_s, "total_seconds": total}


def by_case(rows: list = None) -> dict:
    """For each ground-truth document: the settings that read it best and quickest.

    Three separate answers, because they are three separate questions and the
    same run rarely wins all of them:

    - `best_char`  -- the highest transcript accuracy the document has reached.
    - `best_field` -- the highest field score, which is pass 2's correctness and
      the only figure here that says a value landed in the right key. Blank for a
      document with no `solution/<id>.fields.json`, and blank for one that has a
      truth file no run has been scored against yet, which is not the same as 0%.
    - `fastest`    -- the quickest run that did the whole job.

    Two rules make the answers comparable, and both narrow the field on purpose:

    `field_acc` only reads down a column within one document -- its denominator
    is whatever that document's truth file rules on -- which is exactly why the
    ranking is per case and why `field_expected` is carried beside every figure:
    a truth file filled in further between two runs moves the denominator, and a
    percentage against 15 values is not a percentage against 18.

    `fastest` is taken over reads that finished (`status` ok), scored something
    at all, and ran both passes. A read that returned an empty transcript at
    HTTP 200 is the failure this project is organised around; letting it win the
    speed column would hand a rosette to the one failure that has no other
    symptom. Its accuracy is carried with it regardless, because fast is a claim
    about cost and nothing else.

    Ties keep the earliest run, so a setting that reached a score first is not
    displaced by a later repeat of it; a tie on `best_field` is broken towards
    the faster run, which is the whole of what "and time" can mean once accuracy
    is equal.

    **A run that did not finish cannot hold a record either.** `best_char` and
    `best_field` skip the same runs the setting tables refuse to score, so a
    looped read cannot be reported as what a document is capable of. It was
    already true of `fastest` for the same reason and is now true of all three.
    A document every run of which failed reads *not scored* rather than naming
    the least-bad failure.
    """
    rows = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    # Each document's own last runs, so a fixture read once a month is not
    # described by a build that has since been replaced -- and is not pushed out
    # of the table by another fixture's busy evening either. See `recent_by`.
    rows = recent_by(rows, _document_key)
    out = {}
    for row in reversed(rows):        # oldest first, so an equal score keeps the first to reach it
        case = row.get("case") or ""
        if not case:
            continue
        entry = out.setdefault(case, {"case": case, "runs": 0, "extracts": 0,
                                      "best_char": None, "best_field": None,
                                      "fastest": None})
        is_extract = (row.get("run_type") or "ocr") == "extract"
        entry["extracts" if is_extract else "runs"] += 1
        setting, elapsed = _setting(row), _elapsed(row)
        char = _num(row.get("char_accuracy"), None)
        # Not "what did this row record" but "what does it measure": a field
        # score over a transcript the read got wrong belongs to the read.
        field = _num(row.get("field_acc"), None) if _field_trusted(row) else None

        # A failed run is counted in `runs`/`extracts` above and is eligible for
        # no record below. So is a run at a Detail this build no longer offers:
        # every column here names a setting to adopt, and a record set at 1 MP
        # names one nobody can select. Same shape as the failure rule -- counted
        # as a run, eligible for nothing. See `RETIRED_DETAILS`.
        if _incomplete(row) or not _current_detail(row):
            continue
        if char is not None:
            best = entry["best_char"]
            if best is None or char > best["char_accuracy"]:
                entry["best_char"] = {**setting, **elapsed, "char_accuracy": char,
                                      "word_accuracy": row.get("word_accuracy", ""),
                                      "field_acc": field}
        if field is not None:
            best = entry["best_field"]
            # Faster wins an exact tie. `total_seconds` is None on a re-extraction
            # (it read no page) and on a read that never extracted, and an unknown
            # time can never displace a known one.
            better = (best is None or field > best["field_acc"]
                      or (field == best["field_acc"]
                          and elapsed["total_seconds"] is not None
                          and (best["total_seconds"] is None
                               or elapsed["total_seconds"] < best["total_seconds"])))
            if better:
                entry["best_field"] = {**setting, **elapsed, "field_acc": field,
                                       "field_expected": row.get("field_expected", ""),
                                       "char_accuracy": char}
        if (not is_extract and row.get("status") == "ok" and char
                and elapsed["total_seconds"] is not None):
            best = entry["fastest"]
            if best is None or elapsed["total_seconds"] < best["total_seconds"]:
                entry["fastest"] = {**setting, **elapsed, "char_accuracy": char,
                                    "field_acc": field,
                                    "field_expected": row.get("field_expected", "")}
    return out


# Statuses that say a run did not finish what it started. `partial` is NOT one
# of them: a salvaged reply is a real result that says so, and the extraction
# table counts its own failure -- a cycled reply -- separately and by content.
_INCOMPLETE = ("error", "cancelled", "truncated", "looped")


def _incomplete(row: dict) -> bool:
    """True where a run did not FINISH: it looped, was cut off, errored, or was
    cancelled.

    **A low score is not a failure** (2026-08-24, at the user's request: *those
    low ocr score does not count as fail, fail is loop or crash*). This also
    counted a run that finished and scored 0.0% until then, on the argument that
    an empty transcript at HTTP 200 has no other symptom. The user's reading is
    the one that ships and it is the better one: a run that came back is a
    measurement, and 0% is what that measurement says. Calling it a failure
    removed it from the accuracy mean, which is exactly where a model that
    returns nothing should be visible -- it flattered the mean and inflated the
    failure rate with the same row.

    So a 0% read now lands in the mean as 0%, and the failure rate counts only
    runs that did not produce a result at all. `by_ocr` still reports how many
    finished empty, as `empty`, beside the mean rather than inside the failure
    count -- it is a diagnosis, not a failure.

    Two things deliberately did NOT change with it. `by_case`'s speed column
    still refuses a 0% run, because a record set by returning nothing is not a
    speed anyone can use; and `app._unscorable_read` still refuses to score
    fields extracted from an empty transcript, because that figure would belong
    to the read.
    """
    return (row.get("status") or "") in _INCOMPLETE


def _extras_loop(row: dict) -> bool:
    """True where a cycled reply cost `other_fields` entries and nothing else.

    **A failure in the extras is not a failure of the extraction** (2026-08-27,
    at the user's request: *that error on other extraction field will not count
    as error ... but if main field error we will count that as error*). The
    reason it is safe is what `other_fields` is: the overflow valve for
    everything the fourteen priority-1 keys do not cover, whose labels are the
    model's own wording and which `fieldscore` therefore keeps out of every
    headline. A reply that filled the form and then cycled through the extras
    measured the setting exactly as well as one that stopped there politely.

    The test is that the main fields survived -- `p1_present` above zero. A
    cycle that ate the whole reply leaves it at 0 and is a failure like any
    other, which is the half of the rule that keeps this honest: it excuses the
    loops that cost nothing, never the ones that cost the form.

    `extract_looped` itself is unchanged and still written, so the extras
    contest still refuses to let a cycled reply win on volume -- that is a
    fairness rule about ranking, not a count of errors.
    """
    return (_num(row.get("extract_looped"), 0.0) > 0
            and _num(row.get("p1_present"), 0.0) > 0)


def _extract_incomplete(row: dict) -> bool:
    """True where PASS 2 did not finish: it cycled away the form, or never replied.

    Separate from `_incomplete` because the two passes fail differently and a
    row can be one without the other -- a page that read cleanly and then died
    inside pass 2 is a good read and a failed extraction, and the two tables
    must be able to say so independently.

    A cycle confined to `other_fields` is NOT one of these; see `_extras_loop`.
    """
    return ((row.get("status") or "") in _INCOMPLETE
            or (_num(row.get("extract_looped"), 0.0) > 0
                and not _extras_loop(row)))


def _complete_only(cases: dict, failed) -> dict:
    """The same {document: [runs]} grouping with the failed runs taken out.

    **A failed run is counted, never scored** (2026-08-21, at the user's
    request). It reverses the rule this file held until then -- that a setting
    which loops should carry its loops into its mean, because dropping them
    reports the accuracy of the runs that happened to survive. Both readings are
    defensible and the user's is the one that ships: a run that did not finish
    did not measure anything, and averaging a truncated transcript's score with
    whole ones describes neither.

    What makes it safe is that **the failure rate is reported beside every mean**
    -- 60% at a 100% failure rate is legible as one good run out of five, and it
    was not legible when the four bad ones were inside the 60%.

    A document whose every run failed drops out entirely rather than
    contributing an empty entry, so `documents` beside the mean counts documents
    that actually produced a number.
    """
    out = {}
    for case, rows in cases.items():
        keep = [row for row in rows if not failed(row)]
        if keep:
            out[case] = keep
    return out


def _blank(value) -> bool:
    """A cell nobody wrote. Distinct from a cell holding 0, which is a measurement."""
    return value is None or value == ""


def _stats(values):
    """Mean and sample standard deviation, or (None, None) over nothing.

    Sample (n-1), not population: these are repeated measurements of a setting
    that could have been repeated again, not a complete enumeration of anything.
    The SD is None at n=1 rather than 0 -- one run has no spread, which is not
    the same claim as a run that was repeated and did not move.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return round(mean, 2), None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(mean, 2), round(var ** 0.5, 2)


def _per_case(cases: dict, pick) -> dict:
    """One figure aggregated the only way that compares two settings honestly.

    **A repeat is averaged before a document is, and a document is never averaged
    with another document's runs.** Two runs of one setting on sol001 and one on
    sol005 is not three samples of that setting -- pooling them weights sol001
    twice, and the settings being compared rarely ran the same number of times on
    the same documents. So: mean over the runs of each document, then mean over
    the documents.

    The two spreads answer different questions and both are reported:

    - `sd_repeat` -- the spread WITHIN a document, averaged over the documents
      that were run more than once. How reproducible the setting is. Under greedy
      decoding this should be ~0, and a figure that is not is worth knowing
      about: Ollama's first request after a model loads differs from every later
      one, and single-mode extraction has a run-to-run spread worth about five
      points on this denominator.
    - `sd_case` -- the spread of the per-document means. How consistent the
      setting is across documents, which is a property of the fixtures as much as
      of the setting. Never read it as an error bar on `mean`.

    Both are None where there is nothing to take them over: one document, or no
    document run twice.
    """
    case_means, within = [], []
    for rows in cases.values():
        mean, sd = _stats([pick(row) for row in rows])
        if mean is None:
            continue
        case_means.append(mean)
        if sd is not None:
            within.append(sd)
    if not case_means:
        return {"mean": None, "sd_case": None, "sd_repeat": None, "documents": 0}
    mean, sd_case = _stats(case_means)
    return {"mean": mean, "sd_case": sd_case,
            "sd_repeat": round(sum(within) / len(within), 2) if within else None,
            "documents": len(case_means)}


def _group(rows: list, keys: tuple, keep) -> dict:
    """Rows as {setting -> document -> [runs]}, for `_per_case` to reduce.

    The document is the ground-truth case where there is one and the file name
    otherwise, so an unscored document still groups its own repeats together
    instead of counting once per run.
    """
    groups = {}
    for row in rows:
        if not keep(row):
            continue
        key = tuple(row.get(column, "") or "" for column in keys)
        document = row.get("case") or row.get("file") or ""
        groups.setdefault(key, {}).setdefault(document, []).append(row)
    return groups


# The pass-1 knobs. A setting is these four together: the same model at a
# different detail is a different thing to run, and `ocr_profile` decides the
# prompt and the system slot, which on dots.ocr is the difference between a
# transcript and an empty string at HTTP 200.
OCR_SETTING = ("model", "backend", "detail", "ocr_profile")

# The pass-2 knobs. `extract_mode` is in here rather than averaged over, because
# single and agentic are not two samples of one setting -- they are two shapes
# with different failure modes, and the measured gap between them is 2.5x.
EXTRACT_SETTING = ("extract_on", "backend", "extract_mode")




def by_ocr(rows: list = None) -> list:
    """Pass 1 per setting: how well it reads a page, and where the time goes.

    Reads only -- a `run_type=extract` row never touched pass 1, and its pass-1
    columns are blank by construction.

    **Every read is counted, including the ones with no accuracy at all.** This
    filtered on `char_accuracy` until 2026-08-20, which quietly meant *the
    failure count could only see failures that still produced a scored
    transcript*: in one archived log, all four `error` and `cancelled` reads were
    invisible to it and only the eight `looped` ones showed. A failure rate blind
    to outright failure is worse than none. Runs with no accuracy contribute to
    `runs` and `failed` and are skipped by the averages, which `_per_case` does
    already -- `char["documents"]` says how many documents actually fed a number.

    **A run that did not finish is counted and not scored.** It contributes to
    `runs`, `failed` and `failure_rate`, and to none of the accuracy or timing
    means -- a truncated transcript scored what it scored, but what it scored is
    not a measurement of the setting. Read the two together: a high mean beside a
    high failure rate is one good run among several bad ones, which is exactly
    what a mean with the failures folded in could not say.

    Prefill and decode are separate columns rather than one total because they
    move for different reasons -- prefill with pixels, decode with output length
    -- and the whole of the detail-preset trade lives in the first of them.

    Returns settings best first by mean character accuracy.
    """
    rows = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    # Twenty runs of THIS setting, not the log's last twenty rows: a setting is
    # ranked on how it has been doing lately, and two settings compared here have
    # rarely been run in the same week.
    #
    # Windowed AFTER the fold, so `medium` and the `accurate` that is the same
    # 4 MP share one window instead of holding twenty rows each.
    rows = recent_by(rows, _setting_key(OCR_SETTING))
    # A row at a retired preset names no setting anyone can run, so it is not
    # a row in this table at all -- see `RETIRED_DETAILS`. It is still a read,
    # and `standouts` and the header means still count it.
    groups = _group(rows, OCR_SETTING,
                    lambda r: (r.get("run_type") or "ocr") != "extract"
                    and _current_detail(r))
    out = []
    for key, cases in groups.items():
        runs = [row for case_rows in cases.values() for row in case_rows]
        failed = [r for r in runs if _incomplete(r)]
        # Only the runs that finished reach a mean. See `_complete_only`.
        scored = _complete_only(cases, _incomplete)
        out.append({
            **dict(zip(OCR_SETTING, key)),
            "runs": len(runs),
            "documents": len(cases),
            "failed": len(failed),
            "failure_rate": round(100.0 * len(failed) / len(runs), 1) if runs else None,
            # The two shapes apart, because they are found in different ways and
            # fixed in different places: one is a run that reported a problem,
            # the other is a run that reported success and returned nothing.
            # Runs that finished and returned nothing. **Not a failure**
            # since 2026-08-24 -- they are inside `char` as the 0% they scored,
            # which is where a model that returns nothing belongs. Reported
            # because the two are found and fixed in different places, and a
            # mean of 40% made of half good reads and half empty ones is a
            # different problem from a mean of 40% made of mediocre ones.
            "empty": sum(1 for r in runs if not _incomplete(r)
                         and (_num(r.get("char_accuracy"), None) or 0) <= 0.0
                         and not _blank(r.get("char_accuracy"))),
            "char": _per_case(scored, lambda r: _num(r.get("char_accuracy"), None)),
            # The same recall over Thai, Latin and numerals, so a setting can be
            # read for WHICH of the three it loses rather than only for how much
            # it loses -- a setting at 92% overall can be at 71% on the digits,
            # and the digits are every Mandatory field in the requirement. Blank
            # on rows written before 2026-09-04 and on documents that print too
            # little of a script to measure; `_per_case` skips both, so a setting
            # with no such rows reports no rate rather than a zero.
            **{name: _per_case(scored,
                               lambda r, n=name: _num(r.get(f"{n}_accuracy"), None))
               for name in scoring.SCRIPTS},
            "word": _per_case(scored, lambda r: _num(r.get("word_accuracy"), None)),
            "prefill": _per_case(scored, lambda r: _num(r.get("prefill_seconds"), None)),
            "decode": _per_case(scored, lambda r: _num(r.get("decode_seconds"), None)),
            "seconds": _per_case(scored, lambda r: _num(r.get("seconds"), None)),
            "tps": _per_case(scored, lambda r: _num(r.get("tokens_per_second"), None)),
        })
    out.sort(key=lambda e: (e["char"]["mean"] if e["char"]["mean"] is not None else -1.0,
                            e["documents"]), reverse=True)
    return out


def _other_points(groups: dict) -> dict:
    """`other_fields` scored by rank within each document, summed.

    **There is no ground truth for `other_fields` and there cannot be one** --
    the labels are the model's own wording, and a page has as many extra fields
    as a reader decides it has. So it is scored the only way an unscorable output
    can be: against the other settings that read the SAME document. Three
    settings on sol001 ranking B, C, A gives B +2, C +1, A +0.

    Points are *how many settings you strictly beat*, which is what makes ties
    behave: two settings that returned the same count get the same points, and
    neither is placed above the other by the order they happen to be iterated in.

    **A document only one setting has been run on is skipped entirely**, and does
    not count towards `possible` either. Awarding 0 out of 0 there would let a
    setting look under-performing for having had no opponent, and a denominator
    that included it would flatter whichever setting ran alone.

    Two things keep a loop from winning it, and both are needed:

    - **A run flagged `extract_looped` does not enter the contest at all.** It is
      not a smaller result, it is a failed one, and it is counted as such in the
      failure rate instead. Ignoring it here is the user's instruction in as many
      words.
    - **The value compared is `other_distinct`, not `other_fields`.** Repetition
      below the flag's threshold still inflates a total, and distinct entries are
      the honest count of what a reply contributed. Rows written before that
      column existed fall back to the raw total, which is all they can say.
    """
    per_case = {}
    for key, cases in groups.items():
        for case, rows in cases.items():
            values = []
            for row in rows:
                if _num(row.get("extract_looped"), 0.0) > 0:
                    continue
                value = _num(row.get("other_distinct"), None)
                if value is None:
                    value = _num(row.get("other_fields"), None)
                if value is not None:
                    values.append(value)
            if values:
                per_case.setdefault(case, {})[key] = sum(values) / len(values)

    scores = {key: {"points": 0, "possible": 0, "contests": 0} for key in groups}
    for entries in per_case.values():
        if len(entries) < 2:
            continue
        for key, value in entries.items():
            scores[key]["points"] += sum(1 for other in entries.values() if other < value)
            scores[key]["possible"] += len(entries) - 1
            scores[key]["contests"] += 1
    return scores


def _field_trusted(row: dict) -> bool:
    """Whether this row's field score is a measurement of the EXTRACTOR.

    **The read-quality rule applied at scoring time, not only at logging time**
    (2026-08-24). `app._unscorable_read` keeps the figure off new rows written by
    the random test; this keeps it out of every table for rows that already have
    one -- runs logged before the rule existed, runs logged under a lower
    threshold, and every page upload and queue job, none of which suppress
    anything. Without it, raising `MIN_READ_FOR_FIELDS` would change what gets
    written and leave the tables reporting the old rule for months.

    A field score taken over a transcript pass 1 got wrong is the read's mistake
    wearing the extractor's name: pass 2 can only map the values it was given.
    So the row still counts as a run, still reports coverage, grounding and
    `other_fields` -- and contributes no correctness figure.

    `char_accuracy` is blank on a `run_type=extract` row (it read no page, and
    was usually fed the ground truth) and on any document with no transcript
    truth. Both are trusted: there is no bad read to propagate, and refusing to
    score on a guess would blank every unscored document's fields as well.
    """
    floor = read_floor()
    if not floor:
        return True
    char = _num(row.get("char_accuracy"), None)
    # The column is a percentage; the setting is a fraction.
    return char is None or char >= floor * 100.0


def _p1_rate(row: dict):
    """One run's priority-1 field score, or None where it was never scored.

    **A `partial` is worth half a value** (2026-08-20, at the user's request).
    One value contains the other, so the model found the right thing and took too
    much or too little of it: scoring that as a miss is as wrong as scoring it as
    a hit. `fieldscore` reports all three readings -- strict, half and loose --
    and this is the one the setting comparison ranks on.

    A rate per run, so a repeat can be averaged and its spread taken. The truth
    files state 11 to 13 of the fourteen keys, so this denominator is about 12
    and is per document -- which is why the table means these rather than
    dividing one sum of correct values by one sum of expected ones.
    """
    correct = _num(row.get("p1_correct"), None)
    expected = _num(row.get("p1_scored"), None)
    if correct is None or not expected or not _field_trusted(row):
        return None
    # Blank on every row written before the column existed, which is a row that
    # scored no partials as far as this file can say. Zero is the safe reading:
    # it degrades to the strict rate rather than inventing half-credit.
    partial = _num(row.get("p1_partial"), 0.0)
    return 100.0 * (correct + 0.5 * max(partial, 0.0)) / expected


def by_extract(rows: list = None) -> list:
    """Pass 2 per setting: how many priority-1 values land in the right key.

    **Single and agentic are separate rows**, at the user's request and on the
    evidence: the measured gap between them on typhoon is 2.5x, and it is not a
    gap a model carries with it -- a model that answers the whole form in one
    request well is not thereby good at twenty small questions, or the reverse.
    Averaging them would report a number neither shape produces.

    Two figures, because pass 2 has two outputs and only one can be marked
    against an answer sheet:

    - `field` -- priority-1 values scored CORRECT against
      `solution/<id>.fields.json`, as a rate, meaned over documents. Only
      `correct` counts; a `partial` is not half a value in the right key.
    - `other` -- `other_fields`, scored by rank within each document, because
      there is nothing there to be right against, and reported as points PER
      DOCUMENT so settings run on different numbers of documents compare. Runs
      whose reply cycled are excluded from it and counted in `failure_rate`
      instead. See `_other_points`.

    **Read-fed and truth-fed runs are pooled into one row** (2026-08-20, at the
    user's request, to keep the table readable as models are added). They went
    through two earlier shapes -- separate rows, then truth-fed only -- and both
    multiplied rows nobody was comparing.

    The caveat that motivated the split is still true and is now the reader's to
    hold: a run fed `solution/<id>.md` measures pass 2 alone, while one fed a
    transcript pass 1 produced measures BOTH passes, because a value the OCR
    misread cannot then be extracted correctly. Measured on the one setting that
    has both, dots.mocr agentic scored 32.7% read-fed against 46.8% truth-fed. So
    a setting whose runs are mostly real reads is being marked on its OCR as well
    as its extraction, and `source` in the CSV is what says which.

    Restricted agentic runs (`extract_steps`) are excluded: they answer part of
    the form, so their coverage counts and their absent field score are not a
    measurement of the setting on it.

    Returns settings best first by mean field accuracy.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    # The model that did pass 2, which is `extract_model` where the two passes
    # differed and `model` otherwise. Derived here rather than stored twice: a
    # column repeating the reading model on every one-model row would be blank's
    # opposite problem, and two columns saying the same thing eventually
    # disagree.
    rows = [{**r, "extract_on": (r.get("extract_model") or r.get("model") or "")}
            for r in rows]
    # After `extract_on` exists, because it is part of the key this windows by.
    rows = recent_by(rows, _setting_key(EXTRACT_SETTING))
    groups = _group(rows, EXTRACT_SETTING,
                    lambda r: (not r.get("extract_steps")
                               and ((r.get("status") or "") in _INCOMPLETE
                                    or not (_blank(r.get("p1_present"))
                                            and _blank(r.get("other_fields"))))))
    points = _other_points(groups)
    out = []
    for key, cases in groups.items():
        runs = [row for case_rows in cases.values() for row in case_rows]
        scored = [r for r in runs if not _blank(r.get("p1_correct"))
                  and not _extract_incomplete(r) and _field_trusted(r)]
        # A run whose reply cycled, or one that never produced a reply at all.
        # Counted rather than quietly dropped: a setting that fails on two
        # documents in five is not the same proposition as one that never does,
        # and the extras contest those runs are excluded from would otherwise
        # make them look merely absent.
        looped = sum(1 for r in runs if _extract_incomplete(r))
        measured = sum(1 for r in runs if not _blank(r.get("extract_looped"))
                       or (r.get("status") or "") in _INCOMPLETE)
        # Only the runs whose extraction finished reach a mean, the same rule
        # `by_ocr` follows. A cycled reply filled keys and some of them are even
        # right, but a run that ran to its token cap did not measure the setting.
        ok = _complete_only(cases, _extract_incomplete)
        out.append({
            **dict(zip(EXTRACT_SETTING, key)),
            # Kept under its old name too, so callers and the page do not have to
            # know which of the two columns the name came out of.
            "model": key[0],
            # The reading model(s) these extractions sat behind, where that is a
            # different model. Named rather than counted: a pass-2 figure taken
            # over a real transcript is partly a measurement of pass 1, and the
            # only thing that says whose is this.
            "read_by": sorted({r.get("model", "") for r in runs
                               if r.get("extract_model")
                               and r.get("model") and r.get("model") != key[0]}),
            "runs": len(runs),
            "documents": len(cases),
            "scored_runs": len(scored),
            "looped": looped,
            # Share of the runs this file can actually say went either way.
            # None where no run of this setting carries the column, which is not
            # the same claim as a failure rate of zero.
            "failure_rate": round(100.0 * looped / measured, 1) if measured else None,
            "field": _per_case(ok, _p1_rate),
            # Raw counts over the scored runs, so a rate is never the only thing
            # on the row: 50% of 12 values and 50% of 2 are the same percentage.
            "correct": sum(int(_num(r.get("p1_correct"), 0)) for r in scored),
            "expected": sum(int(_num(r.get("p1_scored"), 0)) for r in scored),
            "filled": _per_case(ok, lambda r: _num(r.get("p1_present"), None)),
            "other_count": _per_case(ok, lambda r: _num(r.get("other_fields"), None)),
            "other_distinct": _per_case(ok, lambda r: _num(r.get("other_distinct"), None)),
            **{f"other_{k}": v for k, v in points[key].items()},
        })
        # Points per document rather than points in total, because the settings
        # being compared have not entered the same number of contests: +10 from
        # one document and +26 from five are not the same achievement, and the
        # raw totals rank the setting that was run most. Asked for in exactly
        # those terms. None where the setting never had an opponent.
        entry = out[-1]
        entry["other_rate"] = (round(entry["other_points"] / entry["other_contests"], 2)
                               if entry["other_contests"] else None)
    out.sort(key=lambda e: (e["field"]["mean"] if e["field"]["mean"] is not None else -1.0,
                            e["other_rate"] if e["other_rate"] is not None else -1.0,
                            e["documents"]), reverse=True)
    return out


# A group ranked on fewer runs than this is marked `thin`: it is still listed
# and still ranked, but it may not be the headline best or worst while a
# better-evidenced group is available. One run at 99% is not a track record, and
# a "worst" that is one unlucky read is a smear rather than a finding.
STANDOUT_MIN_RUNS = 2


def _standout_score(accuracy, failure_rate):
    """One number per group, so best and worst are answerable at all.

    **A failed run counts as zero and a finished one counts what it scored**,
    which is the user's ranking rule -- failures first, accuracy second --
    expressed as a product rather than as a sort with two keys:

        score = accuracy x (1 - failure rate)

    It reads as *what this is worth per attempt*. A model that reads at 95% but
    fails half the time is worth 47.5, below one that reads at 70% and always
    finishes -- the ordering a two-key sort would also give, except that this
    degrades smoothly instead of letting one failure in twenty outrank twenty
    points of accuracy.

    **Where nothing was scored, the failure rate decides whether that is a
    result or an absence**, and the two must not be conflated:

    - **0.0** -- runs failed and not one of them survived to be measured. A real
      measurement and the worst there is: every attempt was worth nothing.
    - **None** -- nothing failed and nothing was scored either. That is a group
      nobody has ground truth for, or one whose field score was suppressed
      because the read was too poor to judge it by. Ranked nowhere, and shown as
      *not scored* rather than as a zero.
    """
    if accuracy is not None:
        return round(accuracy * (1.0 - (failure_rate or 0.0) / 100.0), 1)
    return 0.0 if failure_rate else None


def _bucket(rows: list, key_of, inner_of) -> dict:
    """Rows as {group -> repeat-set -> [runs]}, for `_per_case` to reduce.

    The inner key is what a mean is taken over BEFORE the group's own mean, and
    it is deliberately different for the two questions this asks:

    - ranking models, the inner key is the document -- two runs on sol001 are one
      document's worth of evidence, not two, or a model that happened to be run
      repeatedly on one fixture would be ranked on that fixture.
    - ranking documents, it is the model -- otherwise the document is described
      by whichever model was pointed at it most often.

    Rows with no group key are dropped: an upload with no `case` cannot be a
    document standout, and a run with no model recorded cannot be a model one.
    """
    out = {}
    for row in rows:
        key, inner = key_of(row), inner_of(row)
        if not key:
            continue
        out.setdefault(key, {}).setdefault(inner or "", []).append(row)
    return out


def _standouts(groups: dict, failed, pick, scored_test) -> list:
    """One ranked list: every group, best first, unranked ones last.

    `failed` decides which runs did not finish, `pick` reads the accuracy off a
    run, and `scored_test` says whether a run carries an accuracy at all --
    three arguments because the two passes answer all three differently, and the
    alternative was two near-identical copies of this.
    """
    out = []
    for key, inner in groups.items():
        runs = [row for rows in inner.values() for row in rows]
        bad = [row for row in runs if failed(row)]
        rate = round(100.0 * len(bad) / len(runs), 1) if runs else None
        # Only the runs that finished reach the mean -- the same rule the setting
        # tables follow, and the reason the failure rate is printed beside it.
        accuracy = _per_case(_complete_only(inner, failed), pick)
        out.append({
            "key": key,
            "runs": len(runs),
            "groups": len(inner),
            "failed": len(bad),
            "failure_rate": rate,
            "accuracy": accuracy["mean"],
            "spread": accuracy["sd_case"],
            "scored_runs": sum(1 for row in runs
                               if scored_test(row) and not failed(row)),
            "score": _standout_score(accuracy["mean"], rate),
            # Listed and ranked, but not eligible to be the headline while
            # anything better-evidenced exists. See STANDOUT_MIN_RUNS.
            "thin": len(runs) < STANDOUT_MIN_RUNS,
        })
    # Unranked last: a group nothing scored has no place in an ordering by score,
    # and putting it at either end would read as a verdict.
    out.sort(key=lambda e: (e["score"] is not None,
                            e["score"] if e["score"] is not None else 0.0,
                            e["runs"]), reverse=True)
    return out


def _headline(ranked: list) -> dict:
    """The best and worst of one ranked list, and the rule for picking them.

    Thin groups are skipped while a substantial one is available at that end and
    fallen back to when it is not -- a list of three one-run groups still has a
    best and a worst, and saying so with `thin` on the entry is more useful than
    saying nothing at all.

    `best` and `worst` are never the same entry: one group is a list with a best
    and no worst, because "worst" over one thing is a slur rather than a ranking.
    """
    ranked = [e for e in ranked if e["score"] is not None]
    if not ranked:
        return {"best": None, "worst": None}
    solid = [e for e in ranked if not e["thin"]]
    best = (solid or ranked)[0]
    # The two ends are chosen separately, and one solid group does not make a
    # ranking: if it is the only one, the other end comes from the full list and
    # carries its own `thin` mark. Under a 20-row window most groups have a
    # single run, and "best X, worst nothing" would leave the list unread rather
    # than unranked.
    worst = (solid if len(solid) > 1 else ranked)[-1]
    return {"best": best, "worst": worst if worst is not best else None}


def standouts(rows: list = None) -> dict:
    """Best and worst, four ways: model and document, for each pass.

    Built 2026-08-21 at the user's request, in their own terms -- *dots.ocr is
    worse at extracting fields, 10/11 fail and 0.5% accuracy*; *sol007 is worst
    at OCR, every model fails it*. The setting tables already held every figure
    in here; what they could not do is say which is the worst out loud, and a
    table sorted on accuracy alone puts a setting that never finishes nowhere
    near the bottom.

    **Models are grouped WITHOUT their Detail, profile or mode**, unlike `by_ocr`
    and `by_extract`. That is the point of it: those tables answer *which setting
    to run*, and this answers *which model is carrying its weight* across
    everything it has been asked to do. Read them together -- a model that is
    worst here can still hold the best single row over there, at one Detail.

    Four lists, each best first, each entry carrying `runs`, `failed`,
    `failure_rate`, `accuracy` and `score`:

    | | grouped by | failed when | scored on |
    |---|---|---|---|
    | `ocr.models` | the model that read | `_incomplete` -- a loop or a crash, never a low score | `char_accuracy` |
    | `ocr.cases` | the document | `_incomplete` | `char_accuracy` |
    | `extract.models` | the model that extracted | `_extract_incomplete` | priority-1 fields, a partial counting half |
    | `extract.cases` | the document | `_extract_incomplete` | the same |

    **A document's row is not a verdict on the document.** A fixture that reads
    worst across every model may be the hardest page here, or its hand-written
    ground truth may be wrong -- this file already flags sol003 as the likeliest
    place for that. The list says where to look, not what is there.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    reads = [r for r in rows if (r.get("run_type") or "ocr") != "extract"]
    # The same rows `by_extract` counts: pass 2 left a figure, or the run failed
    # in a way that says pass 2 never got to. Restricted step runs are excluded
    # there and here -- they answered part of the form on purpose.
    extracts = [{**r, "extract_on": (r.get("extract_model") or r.get("model") or "")}
                for r in rows
                if not r.get("extract_steps")
                and ((r.get("status") or "") in _INCOMPLETE
                     or not (_blank(r.get("p1_present"))
                             and _blank(r.get("other_fields"))))]

    def char(row):
        return _num(row.get("char_accuracy"), None)

    def has_char(row):
        return not _blank(row.get("char_accuracy"))

    def has_field(row):
        # The same test the rate uses, or a group reads "1 scored" beside an
        # accuracy of None -- the count and the mean disagreeing about whether
        # anything was measured.
        return not _blank(row.get("p1_correct")) and _field_trusted(row)

    # **Each list windows by the thing it ranks**, which is the whole of what
    # "20 of that model / that doc name" means: a model's row is its own last
    # twenty reads, a document's row is that document's last twenty, and neither
    # is affected by how busy anything else has been.
    by_model = lambda r: r.get("model")                          # noqa: E731
    by_extract_on = lambda r: r.get("extract_on")                # noqa: E731
    by_case = lambda r: r.get("case")                            # noqa: E731
    lists = {
        "ocr": {
            "models": _standouts(
                _bucket(recent_by(reads, by_model), by_model, _document_key),
                _incomplete, char, has_char),
            "cases": _standouts(
                _bucket(recent_by(reads, by_case), by_case, by_model),
                _incomplete, char, has_char),
        },
        "extract": {
            "models": _standouts(
                _bucket(recent_by(extracts, by_extract_on), by_extract_on,
                        _document_key),
                _extract_incomplete, _p1_rate, has_field),
            "cases": _standouts(
                _bucket(recent_by(extracts, by_case), by_case, by_extract_on),
                _extract_incomplete, _p1_rate, has_field),
        },
    }
    return {pass_: {kind: {"ranked": ranked, **_headline(ranked)}
                    for kind, ranked in kinds.items()}
            for pass_, kinds in lists.items()}


# --------------------------------------------------------------------------
# Where a model is weak, rather than how good it is
#
# Two compilations, both added 2026-09-04 at the user's request -- *add to
# summary to check each model strength and weakness, also each doc to see in
# each doc the model weakness is in what field*. Neither is a new measurement:
# every figure in them comes from cells `record` already writes. What they add
# is the axis the ranking tables deliberately average over.
#
# `by_ocr`, `by_extract` and `standouts` all answer *which is best*, and a mean
# is exactly the wrong shape for *what is it bad at*: a model reading 92% overall
# can be reading 71% of the digits, and the digits are every Mandatory field in
# the requirement. Same one level down in pass 2 -- a field score of 8/11 says
# nothing about WHICH three, and the three are the same three on every document
# until somebody looks.
# --------------------------------------------------------------------------

# What each script column is called on screen. `latin` is the Latin ALPHABET,
# which is what "English" means on these pages -- a romanised Thai company name
# is Latin script and is not English.
SCRIPT_LABELS = {"thai": "Thai", "latin": "English", "digit": "Numbers"}


def _script_reads(rows: list) -> list:
    """Reads that carry at least one per-script rate.

    A row written before 2026-09-04 has none, which is not a low score and must
    not read as one -- so it is left out of this table entirely rather than
    counted as a model that cannot read Thai. Incomplete runs are excluded on the
    standing rule: a run that did not finish did not measure anything.
    """
    return [r for r in rows
            if (r.get("run_type") or "ocr") != "extract"
            and not _incomplete(r)
            and any(not _blank(r.get(f"{name}_accuracy"))
                    for name in scoring.SCRIPTS)]


def _script_entry(key: str, cases: dict) -> dict:
    """One row of a script table: the three rates, the headline, and the gap.

    `cases` is {inner -> [runs]} and every figure is `_per_case`-shaped, so the
    per-document-first rule holds here as everywhere else on this card: a model
    run five times on one fixture is one document's worth of evidence.

    `weakest` and `gap` are the answer said out loud. A reader can find the
    lowest of three numbers, but the whole point of the table is that the lowest
    one is not the same for every model -- and a gap of two points is noise while
    a gap of twenty is a finding, which the three rates side by side do not say.
    """
    runs = [row for rows_ in cases.values() for row in rows_]
    entry = {
        "key": key,
        "runs": len(runs),
        "documents": len(cases),
        "char": _per_case(cases, lambda r: _num(r.get("char_accuracy"), None)),
    }
    for name in scoring.SCRIPTS:
        entry[name] = _per_case(
            cases, lambda r, n=name: _num(r.get(f"{n}_accuracy"), None))
        # The denominator, meaned the same way: "68% of ~1900 Thai characters"
        # is a finding and "68% of six" is not, and only the count separates
        # them. A rate is blank where the page had too little of that script to
        # measure (`scoring.SCRIPT_MIN_CHARS`); the count is written anyway, so
        # blank says *not enough of it* rather than *nobody looked*.
        entry[f"{name}_chars"] = _per_case(
            cases, lambda r, n=name: _num(r.get(f"{n}_chars"), None))["mean"]
    rated = {name: entry[name]["mean"] for name in scoring.SCRIPTS
             if entry[name]["mean"] is not None}
    if len(rated) > 1:
        worst = min(rated, key=rated.get)
        entry["weakest"] = worst
        entry["gap"] = round(max(rated.values()) - rated[worst], 1)
    else:
        entry["weakest"], entry["gap"] = None, None
    return entry


def script_accuracy(rows: list = None) -> dict:
    """Transcript accuracy split into Thai, English and numerals.

    Pass 1 only, and every rate is recall of the ground truth's own characters of
    that script -- so a model that answers a Thai word with Latin letters loses
    the Thai and does not gain the Latin.

    **The three do not add up to the headline.** Punctuation, symbols and
    currency signs belong to no script and are in `char_accuracy` only.

    Two tables because there are two questions and one grouping cannot answer
    both:

    | | grouped by | ordered |
    |---|---|---|
    | `by_model` | the model that read, WITHOUT its Detail or profile | best first -- it is a ranking |
    | `by_case` | the document | **worst first** -- it is a work list |

    The model table is keyed on the model alone, like `standouts` and unlike
    `by_ocr`: this asks what a model is bad at across everything it has been
    given, not which setting to run.
    """
    rows = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    reads = _script_reads(rows)
    by_model = lambda r: r.get("model")                          # noqa: E731
    by_doc = lambda r: r.get("case")                             # noqa: E731
    models = [_script_entry(key, cases) for key, cases in
              _bucket(recent_by(reads, by_model), by_model, _document_key).items()]
    cases = [_script_entry(key, cases) for key, cases in
             _bucket(recent_by(reads, by_doc), by_doc, by_model).items()]
    models.sort(key=lambda e: (e["char"]["mean"] if e["char"]["mean"] is not None
                               else -1.0), reverse=True)
    # Worst first: a document every model reads badly is where the work is, and
    # a table of documents sorted best first buries it under the easy ones.
    cases.sort(key=lambda e: (e["char"]["mean"] if e["char"]["mean"] is not None
                              else 10 ** 6))
    return {"scripts": list(scoring.SCRIPTS), "labels": dict(SCRIPT_LABELS),
            "by_model": models, "by_case": cases,
            "reads": len(reads), "min_chars": scoring.SCRIPT_MIN_CHARS}


# What one verdict is worth when a field's rate is taken. The same arithmetic
# `fieldscore` uses for the headline, so a field's rate here and `field_acc` on
# the same row cannot disagree: a partial is half -- the model found the right
# thing and took too much or too little of it -- and `absent` is agreement
# rather than a score, so it is in no denominator.
#
# **`spurious` is not in the denominator either, and it is counted separately.**
# The page does not state the value and the model filled it in anyway, so there
# is nothing to be right about -- but on the keys this corpus leaves empty by
# construction (`po_gr_rtv_number`) it is the entire story, and a table that
# folded it into a rate would report those keys as unmeasured.
_VERDICT_POINTS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "missed": 0.0}


def _verdict_reads(rows: list) -> list:
    """(row, {field: verdict}) for extractions whose field score is trustworthy.

    The same three exclusions `by_extract` and `standouts` apply, for the same
    reasons: a restricted step run answered part of the form on purpose, an
    incomplete extraction measured nothing, and a field score taken over a
    transcript pass 1 got wrong is the read's mistake wearing the extractor's
    name (`_field_trusted`).
    """
    out = []
    for row in rows:
        if row.get("extract_steps") or _extract_incomplete(row):
            continue
        if not _field_trusted(row):
            continue
        verdicts = parse_verdicts(row.get("field_verdicts"))
        if verdicts:
            out.append((row, verdicts))
    return out


def _required_of(row: dict) -> set:
    """The keys the requirement demands of the document this row read.

    Taken from the row's own `doc_types` through `prompts`, never stored beside
    the verdicts: the same key is Mandatory on an invoice and Optional on a
    credit note, and two copies of that fact would eventually disagree. A row
    written before `doc_types` existed says nothing, which is why the answer can
    be "unknown" and is reported as such rather than guessed at.
    """
    codes = [c for c in str(row.get("doc_types") or "").split("+") if c]
    if not codes:
        return set()
    return set(prompts.mandatory_for_types(codes))


def _verdict_cell(buckets: dict) -> dict:
    """One (group, field) cell: the pooled counts, and the inner-first rate.

    `buckets` is {inner -> [verdicts]}. The rate is the mean of each inner
    group's own rate -- documents for a model's row, models for a document's row
    -- so a model run five times on one fixture does not describe itself with
    that fixture. The COUNTS are pooled, because a count is what happened and
    averaging one would be arithmetic about nothing.
    """
    counts = {k: 0 for k in VERDICT_LETTERS}
    rates = []
    for verdicts in buckets.values():
        points, scored = 0.0, 0
        for verdict in verdicts:
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict in _VERDICT_POINTS:
                points += _VERDICT_POINTS[verdict]
                scored += 1
        if scored:
            rates.append(100.0 * points / scored)
    scored_total = sum(counts[k] for k in _VERDICT_POINTS)
    return {
        "rate": round(sum(rates) / len(rates), 1) if rates else None,
        "counts": counts,
        "scored": scored_total,
        "spurious": counts["spurious"],
        "runs": sum(counts.values()),
        # The commonest way this field goes wrong, which is the thing a rate
        # cannot say: `missed` and `spurious` are OPPOSITE mistakes and want
        # opposite fixes -- one is the page saying something the extractor did
        # not, the other the extractor saying something the page does not.
        "worst": max(("wrong", "missed", "spurious"), key=lambda k: counts[k])
                 if any(counts[k] for k in ("wrong", "missed", "spurious"))
                 else None,
    }


def _verdict_group(pairs: list, key_of, inner_of) -> list:
    """Ranked rows of one field-weakness grid.

    Each row is a group (a model, or a document) and holds one cell per field it
    was ever judged on. Ordered by the group's own mean over its fields, worst
    first: this table is a work list, not a ranking.
    """
    groups = {}
    for row, verdicts in pairs:
        key, inner = key_of(row), inner_of(row)
        if not key:
            continue
        entry = groups.setdefault(key, {"key": key, "rows": [], "fields": {}})
        entry["rows"].append(row)
        for field, verdict in verdicts.items():
            entry["fields"].setdefault(field, {}).setdefault(
                inner or "", []).append(verdict)

    out = []
    for entry in groups.values():
        cells = {field: _verdict_cell(buckets)
                 for field, buckets in entry["fields"].items()}
        rated = [c["rate"] for c in cells.values() if c["rate"] is not None]
        required = set()
        for row in entry["rows"]:
            required |= _required_of(row)
        out.append({
            "key": entry["key"],
            "runs": len(entry["rows"]),
            "inner": len({inner_of(r) or "" for r in entry["rows"]}),
            "cells": cells,
            "mean": round(sum(rated) / len(rated), 1) if rated else None,
            # Named on the row so the answer does not have to be read off a
            # grid of eleven numbers. Three at most: a list of everything below
            # par is the grid again.
            "worst_fields": [f for f, _ in sorted(
                ((f, c["rate"]) for f, c in cells.items() if c["rate"] is not None),
                key=lambda pair: pair[1])[:3]],
            "required": sorted(required),
        })
    out.sort(key=lambda e: (e["mean"] if e["mean"] is not None else 10 ** 6))
    return out


def field_weakness(rows: list = None) -> dict:
    """Which FIELD each model gets wrong, and which field each document loses.

    Reads `field_verdicts`, the per-field column appended 2026-09-04. Rows
    written before it have none and are simply absent -- blank is not a low
    score, here as everywhere else in this file.

    Three views over the same verdicts:

    | | says |
    |---|---|
    | `fields` | every key, weakest first, pooled over everything -- *what is hard* |
    | `by_model` | a model x field grid -- *what is THIS model bad at* |
    | `by_case` | a document x field grid -- *what does this page lose* |

    **The field order is the ranking**, weakest first, and it is the same order
    in all three grids so a column means the same thing wherever it is read.

    Two things this deliberately does not do. It does not score table cells: a
    path like `income_items[0].amount_paid` is one row of a table going wrong,
    which is not a weakness of a key. And it does not rank a model on this --
    `by_extract` and `standouts` do that, over the same runs, and a second
    ranking computed a second way would eventually disagree with them.
    """
    rows = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    rows = [{**r, "extract_on": (r.get("extract_model") or r.get("model") or "")}
            for r in rows]
    by_model = lambda r: r.get("extract_on")                     # noqa: E731
    by_doc = lambda r: r.get("case")                             # noqa: E731
    # Windowed by the thing each grid groups on, the standing rule: a model's
    # row is its own recent runs and a document's row is that document's.
    models = _verdict_group(_verdict_reads(recent_by(rows, by_model)),
                            by_model, _document_key)
    pairs = _verdict_reads(recent_by(rows, by_doc))
    cases = _verdict_group(pairs, by_doc, by_model)

    pooled, required_on, asked_on = {}, {}, {}
    for row, verdicts in pairs:
        required = _required_of(row)
        for field, verdict in verdicts.items():
            pooled.setdefault(field, {}).setdefault(
                _document_key(row), []).append(verdict)
            asked_on[field] = asked_on.get(field, 0) + 1
            if field in required:
                required_on[field] = required_on.get(field, 0) + 1
    fields = []
    for field, buckets in pooled.items():
        cell = _verdict_cell(buckets)
        demanded = required_on.get(field, 0)
        fields.append({
            "field": field,
            **cell,
            # How often the requirement demanded this key, rather than a flag:
            # the same key is Mandatory on an invoice and Optional on a credit
            # note, so "always", "sometimes" and "never" are three different
            # things and only a count can say which.
            "required_runs": demanded,
            "asked_runs": asked_on.get(field, 0),
            "required": ("always" if demanded == asked_on.get(field, 0) and demanded
                         else "never" if not demanded else "sometimes"),
        })
    fields.sort(key=lambda e: (e["rate"] if e["rate"] is not None else 10 ** 6,
                               e["field"]))
    return {
        "fields": fields,
        "order": [e["field"] for e in fields],
        "by_model": models,
        "by_case": cases,
        "runs": len(pairs),
        "verdicts": dict(VERDICT_LETTERS),
    }


def best_models(rows: list = None) -> dict:
    """Which models this log ranks first, per pass, best first.

    **Built 2026-09-03 at the user's request** -- *auto default model to the best
    one that is shown* -- and it is deliberately the SAME figure the Summary
    tab's headline card and `standouts` print, not a second opinion computed
    beside them. `backends._resolve_model` used to default pass 1 by a substring
    test on the model's NAME; this defaults it by what the model has actually
    scored here, and a default the page cannot explain is worse than no default.

    Returns `{"ocr": [name, ...], "extract": [name, ...]}` -- a preference order
    rather than one winner, because the caller is choosing among the models one
    endpoint happens to serve and the leader is regularly not one of them.

    Two rules, both taken from `_headline` rather than invented here:

    * **A group nothing scored is left out.** `score is None` means the log has
      no measurement of that model, and a default has to be a measurement.
    * **Thin evidence never outranks evidenced.** Groups at or above
      `STANDOUT_MIN_RUNS` come first, in score order, and the thin ones follow in
      their own score order -- so one lucky read cannot become the default while
      a model with a track record is served, and a freshly pulled model is still
      reachable as a default when nothing else is.

    Windowed per model like everything else on that card (`standouts` calls
    `recent_by`), so this says *best lately* and not *best ever*.
    """
    lists = standouts(rows)

    def order(ranked):
        scored = [e for e in ranked if e["score"] is not None]
        return [e["key"] for e in scored if not e["thin"]] +                [e["key"] for e in scored if e["thin"]]

    return {pass_: order(lists[pass_]["models"]["ranked"])
            for pass_ in ("ocr", "extract")}


def servers_seen(rows: list = None) -> list:
    """Every endpoint URL this log has a run against, most recently used first.

    The picker's list is a CONSTANT (`backends._defaults`) and the log is the
    HISTORY, and until now nothing joined them: an endpoint that had served a
    hundred runs was not offered again after a restart unless it happened to be
    one of the two defaults or was typed in by hand.

    Most recent first because that is the order a person means by "the server I
    was using", and because `backends.autoselect` probes in order and a dead port
    costs a full connect timeout -- the ones most likely to answer should be
    asked first.
    """
    rows = _read_all() if rows is None else rows
    out = []
    for row in reversed(rows):
        url = (row.get("server") or "").strip()
        if url and url not in out:
            out.append(url)
    return out


# --------------------------------------------------------------------------
# The error table
#
# **Built 2026-08-24 at the user's request** -- *add another table that ranks
# error, find the most error model and error doc, and the most error set/pair*.
# Every figure in it was already derivable from the log and none of it was
# anywhere on the card: `by_ocr` and `by_extract` print a failure rate per
# SETTING, and `standouts` folds failures into a score with accuracy. Neither
# answers *what is failing, and is it one model, one document, or one pairing of
# the two* -- and the third of those is the question a ranking of either axis
# alone cannot ask, because a document that fails ten times under one model is
# not a difficult document.

# Every distinct way this log says a run went wrong. The first four are `status`
# values (`_INCOMPLETE`); `extract_loop` is not a status at all -- it is
# `extract_looped`, set when a reply cycled -- and it is in here because a cycled
# extraction is a failure the status column never mentions.
ERROR_KINDS = ("error", "cancelled", "truncated", "looped", "extract_loop")

ERROR_KIND_LABEL = {
    "error": "crashed",
    "cancelled": "cancelled",
    "truncated": "hit the token cap",
    "looped": "read looped",
    "extract_loop": "extraction cycled",
}


def _error_kind(row: dict, pass_: str):
    """How this row failed the named pass (`read` or `extract`), or None.

    **This attributes a failure to the pass that failed, and it is deliberately
    NOT `_extract_incomplete`.** That test asks a different question -- *did pass
    2 produce a figure this table can average* -- and answers yes-it-failed for a
    run whose READ looped, because a field score over a truncated transcript is
    not a measurement of the extractor. That is right for a mean and wrong for
    blame: on the log this was written against, 54 of the 69 incomplete rows
    carried pass-2 figures, so calling all 69 extraction failures would charge
    the extracting model for something the reader did.

    So the two counts differ on purpose, and the page says so. What cannot be
    told apart is the one case the CSV genuinely cannot: a full run that crashed
    somewhere leaves no pass-2 columns, and *died before pass 2* and *died inside
    pass 2* look identical. Those are read failures here -- the pass that is
    known to have started.

    At most one kind per row per pass, so `kinds` counts rows and sums to
    `failed`.
    """
    status = (row.get("status") or "")
    if pass_ == "read":
        return status if status in _INCOMPLETE else None
    # A cycle that cost only `other_fields` is not charged to anyone: the form
    # came back. Same rule as `_extract_incomplete`, applied to blame rather
    # than to averaging, and it has to be the same or the Errors tab and the
    # failure rate would disagree about what happened on one row.
    if _num(row.get("extract_looped"), 0.0) > 0 and not _extras_loop(row):
        return "extract_loop"
    # A status failure is pass 2's own only where there was no pass 1 to blame it
    # on -- a fields-only run was fed a transcript and read nothing.
    return status if status in _INCOMPLETE and _pipeline(row) == "extract" else None


def _concentration(rows: list, blame_of) -> dict:
    """Which single value on the OTHER axis produced most of these failures.

    **The whole reason the toggle exists** (the user's own words: *some case got
    dominated by 1 model or 1 doc*). A document with nine failures is a finding
    if six models produced them and an artefact if one model produced all nine,
    and the two are the same number in a ranking. So every row names its biggest
    single contributor, how much of the row it is, and how many contributors
    there were at all.

    `sources` is the count that makes `top` readable: 8 of 8 from one model over
    one source is not a concentration, it is the only thing that ran.
    """
    counts = {}
    for row in rows:
        label = blame_of(row) or ""
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return {"top": None, "sources": 0}
    key = max(counts, key=lambda k: (counts[k], k))
    total = sum(counts.values())
    return {"top": {"key": key, "failed": counts[key],
                    "share": round(100.0 * counts[key] / total, 1)},
            "sources": len(counts)}


def _error_entries(rows: list, pass_: str, key_of, blame_of=None,
                   empty=False) -> list:
    """One ranked list of error counts, most errors first.

    Ranked on the COUNT and not on the rate, because *most error* is what was
    asked for and a rate says something else: one run that failed is 100% and is
    not the thing anybody needs to look at first. The rate is printed beside it
    and the column sorts, so either reading is one click away.

    `share` is of this list's own failures, so the shares of a list sum to 100%
    and say how concentrated the pass is. It is not comparable across two lists
    -- each is windowed by its own key.

    `empty` adds the count of runs that FINISHED and scored 0.0% -- the empty
    transcript at HTTP 200 this project is organised around. It is reported
    beside the failures and never added to them: a run that came back is a
    measurement, and 0% is what it measured (see `_incomplete`). Reads only;
    there is no equivalent for pass 2 that is not already the field score.
    """
    rows = recent_by(rows, key_of)
    groups = {}
    for row in rows:
        key = key_of(row)
        if not key or (isinstance(key, tuple) and not all(key)):
            continue
        groups.setdefault(key, []).append(row)
    out = []
    for key, runs in groups.items():
        bad = [r for r in runs if _error_kind(r, pass_)]
        kinds = {}
        for row in bad:
            kind = _error_kind(row, pass_)
            kinds[kind] = kinds.get(kind, 0) + 1
        entry = {
            "key": " · ".join(key) if isinstance(key, tuple) else key,
            # Both halves of a pair, so the page can name and drop them
            # separately. A single-axis list carries its own key here too, so
            # nothing downstream has to know which shape it is reading.
            "parts": list(key) if isinstance(key, tuple) else [key],
            "runs": len(runs),
            "failed": len(bad),
            "failure_rate": round(100.0 * len(bad) / len(runs), 1) if runs else None,
            "kinds": kinds,
            **(_concentration(bad, blame_of) if blame_of
               else {"top": None, "sources": 0}),
        }
        if empty:
            # Finished, and returned nothing. Not a failure -- see `_incomplete`
            # -- and worth its own column: a document every model reads empty is
            # a different problem from one every model loops on, and the two are
            # fixed in different places.
            entry["empty"] = sum(1 for r in runs if not _error_kind(r, "read")
                                 and not _blank(r.get("char_accuracy"))
                                 and _num(r.get("char_accuracy"), 0.0) <= 0.0)
        out.append(entry)
    total = sum(e["failed"] for e in out)
    for entry in out:
        entry["share"] = (round(100.0 * entry["failed"] / total, 1)
                          if total else None)
    out.sort(key=lambda e: (e["failed"],
                            e["failure_rate"] if e["failure_rate"] is not None else -1.0,
                            e["runs"]), reverse=True)
    return out


def errors(rows: list = None) -> dict:
    """What is failing, ranked three ways, for each pass.

    | list | grouped by | a row says |
    |---|---|---|
    | `models` | the model that did that pass | this model fails N of its runs |
    | `cases` | the document | this document fails under N runs |
    | `pairs` | the model AND the document | this PAIRING fails N times |

    **The pairs list is the one that cannot be derived by eye from the other
    two**, and it is what the request was for: a document at the top of `cases`
    and a model at the top of `models` may be one pairing failing repeatedly or
    may be two independent problems, and only the pair says which.

    **A run is in the list for every pass it actually performed.** The two passes
    are keyed on the same derived fields the filter chips use
    (`FILTER_FIELDS["model"]` / `["extract_model"]`), which are blank on a pass
    the run did not run -- so a fields-only round is in the extraction lists and
    in no reading list, and a read-only round the reverse. That is also what
    makes the page's drop toggles honest: the value a row is keyed on is exactly
    the value the chip for it filters.

    Two-model runs are therefore counted once in each pass, under a different
    model each time. `runs` summed over the models list can exceed the run count,
    which is correct: a run that read on typhoon and extracted on qwen is a run
    of both.
    """
    rows = read(limit=10 ** 6) if rows is None else rows
    out = {}
    for pass_, model_of in (("read", FILTER_FIELDS["model"]),
                            ("extract", FILTER_FIELDS["extract_model"])):
        mine = [r for r in rows if model_of(r)]
        # The headline counts are windowed per DOCUMENT, the rule `totals` uses
        # for its own header: the three lists below each window by their own key,
        # so no one of them can be the count of the pass.
        headline = recent_by(mine, _document_key)
        kinds = {}
        for row in headline:
            kind = _error_kind(row, pass_)
            if kind:
                kinds[kind] = kinds.get(kind, 0) + 1
        out[pass_] = {
            "runs": len(headline),
            "failed": sum(kinds.values()),
            "kinds": kinds,
            "models": _error_entries(mine, pass_, model_of, _document_key,
                                     empty=pass_ == "read"),
            "cases": _error_entries(mine, pass_, _document_key, model_of,
                                    empty=pass_ == "read"),
            "pairs": _error_entries(
                mine, pass_, lambda r, m=model_of: (m(r), _document_key(r))),
        }
    return out


def single_source_failures(rows: list) -> set:
    """`_row` ids of the failing runs a single-source error row is made of.

    **The set the toggle removes from every table but `errors` itself** (built
    2026-08-24 at the user's request: *the single-source toggle excludes from the
    other tables too, and the error table highlights the single source instead*).

    A failing run is single-source when, among a model's windowed failures, every
    one came from the same document -- or, among a document's windowed failures,
    every one came from the same model. That is exactly the `sources == 1` an
    `errors` row carries, computed the same way over the same window
    (`recent_by` reads the view context), so the runs named here are precisely
    the rows that table highlights. Whole runs, keyed by `_row`: a run has one
    identity in the file, and dropping it from a mean is the honest reading of
    "exclude this error".

    Both passes and both axes are unioned. A run that is single-source for its
    read is dropped whichever table would have counted it -- a looped read has no
    honest extraction figure to keep either.
    """
    ids = set()
    for pass_, model_of in (("read", FILTER_FIELDS["model"]),
                            ("extract", FILTER_FIELDS["extract_model"])):
        mine = [r for r in rows if model_of(r)]
        for key_of, blame_of in ((model_of, _document_key),
                                 (_document_key, model_of)):
            windowed = recent_by(mine, key_of)
            groups = {}
            for row in windowed:
                key = key_of(row)
                if key:
                    groups.setdefault(key, []).append(row)
            for runs in groups.values():
                bad = [r for r in runs if _error_kind(r, pass_)]
                if bad and len({blame_of(r) or "" for r in bad}) == 1:
                    ids.update(r.get(ROW_INDEX) for r in bad
                               if r.get(ROW_INDEX) is not None)
    return ids


def drop_single_source(rows: list) -> list:
    """`rows` without the failing runs a single-source error row is made of.

    See `single_source_failures`. Only the FAILING runs of a single-source group
    are removed -- a model's clean runs stay, so a model that loops on sol004
    five times and reads everything else is still ranked on everything else,
    minus the one pairing that was dragging it. That is the whole point of the
    toggle: a document dominated by one bad model is not a hard document, and its
    other runs should still speak for it.
    """
    ids = single_source_failures(rows)
    return [r for r in rows if r.get(ROW_INDEX) not in ids] if ids else rows


# --------------------------------------------------------------------------
# Do time, document and accuracy depend on each other? (2026-08-24, at the
# user's request: *a tab for time vs file type vs accuracy analysis -- compare
# whether these 3 vars depend on each other or not; it depends on the model and
# doc*).
#
# **Pass 1 only.** Every figure here is a read: `char_accuracy` is the transcript
# score, `seconds` is the wall clock for the read, and the document is the
# ground-truth case. Extraction has its own tables; folding it in would mix two
# quantities scored on different things, the same reason `by_ocr` and
# `by_extract` are separate.
#
# **The correlation and share-of-spread figures are gone** (2026-08-25, at the
# user's request: the two bar blocks at the top of the tab were removed as
# nonsense, and `_pearson`/`_eta_squared` went with them). What ships is grouped
# figures over the same reads -- the model x document grid, the Detail tables,
# time per model, the per-document and per-model spreads -- which is the material
# an r or an eta-squared was computed from, read directly. Do not reinstate them
# without being asked.
#
# Nothing here is a claim of cause. A document that takes longer is mostly a
# longer document -- page count and file size, whatever reads it -- and the page
# says so.

# Reads only, scored, timed, and finished -- the three variables all have to be
# present for a row to be a point in any of these, and a looped read's accuracy
# is not a measurement (the same rule the tables apply). A 0% read that FINISHED
# is kept: it is a real measurement and its point is a real one.
def _analysis_reads(rows: list) -> list:
    return [r for r in rows
            if (r.get("run_type") or "ocr") != "extract"
            and (r.get("case") or "")
            and not _blank(r.get("char_accuracy"))
            and not _blank(r.get("seconds"))
            and not _incomplete(r)
            and _current_detail(r)]


def _median(values: list):
    """The middle value, or None over nothing. Robust where a mean is not."""
    s = sorted(v for v in values if v is not None)
    n = len(s)
    if not n:
        return None
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _flag_time_outliers(points: list, cell=None) -> int:
    """Mark reads whose time is a robust outlier within their (model, case) cell.

    Sets `p["outlier"]` on every point and returns how many were flagged. A cold
    model load or a runaway loop puts one read's time far above the rest of its
    cell -- 1489 s against a 30 s median -- and a single such point moves a
    correlation or a mean by itself. The fence is median +/- k x MAD (scaled by
    1.4826 so k is in standard deviations for normal data): MAD is not itself
    dragged by the outlier it is measuring, which a standard deviation would be.

    **The cell is (model, document, DETAIL) and the Detail is not optional.**
    Pooling the presets was the first shape and it is wrong in a way that biases
    the very table it feeds: raising Detail legitimately costs x1.5 to x2.6 of
    the wall clock, so a cell mixing presets has a bimodal time distribution, and
    a median dragged towards whichever preset was run most flags the others as
    outliers. Measured here, `dots.mocr x sol008 x original` is [114.2, 136.8] --
    two reads agreeing with each other -- and pooling put them against a 24.8 s
    median made mostly of `low` reads, so both were dropped. Removing the
    expensive `original` reads would then have made `original` look cheaper than
    it is, in the Detail-cost table, silently.

    Only within a cell, and only where the cell has enough runs to have a shape
    (`ANALYSIS_OUTLIER_MIN_RUNS`) -- a two-run cell has no middle to be far from,
    and an inherently long document is not an outlier for taking long. The finer
    cell means fewer reads are tested at all, which is the right way to be wrong:
    an untested read stays in, and this fence should never drop a legitimate one
    to catch a stray. `MADS` 0 disables it, which is how to see the analysis with
    every point in.
    """
    for p in points:
        p["outlier"] = False
        # Set on every point that was actually TESTED, not only on the ones that
        # failed the test: a read sitting comfortably inside its cell should be
        # able to say so, and a cell too small to test says that instead.
        p["cell_median"] = None
        p["cell_mads"] = None
        p["cell_ratio"] = None
    mads = settings.ANALYSIS_OUTLIER_MADS
    ratio = settings.ANALYSIS_OUTLIER_MIN_RATIO
    if not mads:
        return 0
    # The cell a read is judged against. Defaults to the pass-1 one; the
    # presentation summary passes its own for pass 2, where the shape of the
    # request (single or agentic) plays the part Detail plays here -- an agentic
    # extraction is fifteen requests and legitimately costs several times what a
    # single one does, so pooling the two would flag the dearer shape wholesale
    # for exactly the reason pooling the Detail presets did.
    cell = cell or (lambda p: (p["model"], p["case"], p["detail"]))
    groups = {}
    for p in points:
        if p["seconds"] is not None:
            groups.setdefault(cell(p), []).append(p)
    flagged = 0
    for pts in groups.values():
        if len(pts) < settings.ANALYSIS_OUTLIER_MIN_RUNS:
            continue
        med = _median([p["seconds"] for p in pts])
        mad = _median([abs(p["seconds"] - med) for p in pts])
        if not mad or mad <= 0:      # more than half the cell share one time
            continue
        for p in pts:
            # How far out, in the fence's own units, so an excluded read can be
            # shown with the evidence against it rather than merely named. A
            # reader has to be able to see that 1519 s against a 97 s median is
            # not a borderline call -- and to spot it when one is.
            p["cell_median"] = round(med, 2)
            p["cell_mads"] = round(abs(p["seconds"] - med) / (1.4826 * mad), 1)
            # The ratio the second test uses, kept so an excluded read can show
            # both reasons it was dropped and not just the scale-free one.
            p["cell_ratio"] = round(p["seconds"] / med, 2) if med else None
            # BOTH tests, never either: the MAD fence finds the shape and the
            # ratio insists the difference is big enough to be worth calling an
            # outlier at all. See `ANALYSIS_OUTLIER_MIN_RATIO` -- a tightly
            # clustered cell makes the first test fire on a few seconds of
            # ordinary wall-clock jitter.
            far = abs(p["seconds"] - med) > mads * 1.4826 * mad
            big = (ratio and med > 0
                   and (p["seconds"] >= med * ratio or p["seconds"] <= med / ratio))
            if far and big:
                p["outlier"] = True
                flagged += 1
    return flagged


def ocr_analysis(rows: list = None) -> dict:
    """Whether read time, document and transcript accuracy depend on each other.

    Pass 1 only. See the block above for what each figure means and why a ratio
    is not a cause. Windowed per (model, document) cell, the same
    recent-runs-of-that-thing rule the other tables use, so the analysis
    describes how the current builds behave rather than the whole history.

    **Time outliers are excluded before anything is computed** (2026-08-24, at
    the user's request), per cell and robustly -- see `_flag_time_outliers`. A
    cold-load or looped read's time is not representative of the setting, and one
    of them drags a correlation on its own. The count removed is reported, and
    the excluded reads are still in the raw table and the CSV.

    **Each read is marked GPU or CPU and hot or cold** (`run_hardware` /
    `run_start`), both inferred and both filterable card-wide. The breakdown
    counts and a per-mark split of the accuracy-vs-time correlation are returned,
    because the whole reason to mark them is the question *does the relationship
    hold on a GPU / when cold as well* -- a link that only appears cold is the
    cache talking, not the model.

    The grid of cells is the raw material the correlations summarise: one cell
    per model x document, with the mean time and mean accuracy in it. Reading
    across a row (one model, every document) is accuracy-vs-document by eye;
    reading down a column (one document, every model) is accuracy-vs-model.
    """
    rows = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    reads = _analysis_reads(rows)
    reads = recent_by(reads, lambda r: (r.get("model") or "", r.get("case") or ""))

    # One point per read: the two numbers, the two categories, and the two marks.
    points = []
    for row in reads:
        points.append({
            "model": row.get("model") or "",
            "case": row.get("case") or "",
            "seconds": _num(row.get("seconds"), None),
            "char": _num(row.get("char_accuracy"), None),
            "hardware": run_hardware(row),
            "start": run_start(row),
            # Already folded to this build's spelling by `_for_summary`, and
            # `_analysis_reads` has dropped the retired presets -- so this is one
            # of the three current names, and the ORDER of them is meaningful
            # (see `DETAIL_ORDER`), which is what makes a Detail sweep readable.
            "detail": detail_of(row),
        })

    excluded = _flag_time_outliers(points)
    kept = [p for p in points if not p["outlier"]]
    # Reads in a cell too small for the fence to test -- `cell_mads` is None on
    # exactly those. **Reported rather than quietly kept**: a cell of two reads
    # cannot be tested at all (the median sits between them, so neither is far
    # from it however different they are), and this log has a 25-minute read
    # surviving for that reason. A fence that says how much it could not check is
    # the only kind anyone should trust a mean behind.
    # Zero when the fence is switched off entirely: nothing was tested then, and
    # reporting every read as "not tested" would raise an alarm about a rule the
    # reader has deliberately disabled. The count means *the fence ran and could
    # not check these*, which is only a claim worth making while it is running.
    untested = (sum(1 for p in kept if p["cell_mads"] is None)
                if settings.ANALYSIS_OUTLIER_MADS else 0)
    # **The excluded reads are LISTED, not just counted** (2026-08-24, at the
    # user's request: *shows them too*). An exclusion nobody can inspect is
    # indistinguishable from a bug, and this fence is the one thing on the tab
    # that silently changes every figure -- so each dropped read comes back with
    # the evidence against it: its time, its cell's median, and how many MADs out
    # it was. Worst first, because the question is always "what got dropped and
    # was that right".
    dropped = sorted((p for p in points if p["outlier"]),
                     key=lambda p: -(p["cell_mads"] or 0))

    models = sorted({p["model"] for p in kept if p["model"]})
    cases = sorted({p["case"] for p in kept if p["case"]})

    # model x document cells, each its own mean and spread, over the kept reads.
    cell_groups = {}
    for p in kept:
        cell_groups.setdefault((p["model"], p["case"]), []).append(p)
    cells = []
    for (model, case), pts in cell_groups.items():
        char_mean, char_sd = _stats([p["char"] for p in pts])
        time_mean, time_sd = _stats([p["seconds"] for p in pts])
        # The cell's own dominant hardware and warmth, for the grid marks. Where
        # a cell mixes the two -- some reads cold, some warm -- the mark is the
        # majority and the tooltip carries the split.
        cells.append({"model": model, "case": case, "runs": len(pts),
                      "char_mean": char_mean, "char_sd": char_sd,
                      "time_mean": time_mean, "time_sd": time_sd,
                      "hardware": _dominant(pts, "hardware"),
                      "start": _dominant(pts, "start")})

    # Counts by mark, blank folded into `unknown` so they still sum to the total.
    def marks(field):
        out = {}
        for p in kept:
            out[p[field] or "unknown"] = out.get(p[field] or "unknown", 0) + 1
        return out

    return {
        "points": len(kept),
        "runs": len(kept),
        "excluded": excluded,           # time outliers dropped before computing
        "untested": untested,           # kept, but in a cell too small to test
        "total": len(points),           # kept + excluded
        "documents": len(cases),
        "models": models,
        "cases": cases,
        "cells": cells,
        "hardware_counts": marks("hardware"),
        "start_counts": marks("start"),
        "outliers": [{"model": p["model"], "case": p["case"], "detail": p["detail"],
                      "seconds": p["seconds"], "char": p["char"],
                      "hardware": p["hardware"], "start": p["start"],
                      "cell_median": p["cell_median"], "cell_mads": p["cell_mads"],
                      "cell_ratio": p["cell_ratio"]}
                     for p in dropped],
        # What each mark is worth, so the two inferred marks are SHOWN rather
        # than only used to split a correlation. A degenerate split -- one bucket
        # holding everything -- is legible here as a single row, which is the
        # honest picture of a log measured on one machine.
        "mark_stats": _mark_stats(kept),
        # Per document and per model, both numbers, with the range beside the
        # spread. See `_figure` for why min/max are there at all.
        "case_stats": _entity_stats(kept, "case"),
        "model_stats": _entity_stats(kept, "model"),
        # The thresholds these marks and the fence were drawn at. Returned rather
        # than repeated in the template for the reason the read floor already is:
        # a page quoting a figure the process is not running describes a build
        # nobody has, and all three are env-tunable.
        "gpu_min_tps": settings.GPU_MIN_TPS,
        "warm_prefill_max": settings.WARM_PREFILL_MAX,
        "outlier_mads": settings.ANALYSIS_OUTLIER_MADS,
        "outlier_ratio": settings.ANALYSIS_OUTLIER_MIN_RATIO,
        **_detail_tables(kept),
    }


def _figure(points: list, pick) -> dict:
    """One number over a set of reads: mean, spread, range, and what fed it.

    `runs` is beside every mean for the reason the whole card follows -- a mean
    with no count is not a figure anyone can read, and these cells are often one
    or two runs deep.

    **`min` and `max` are here because a mean and an SD do not describe a small
    sample** (2026-08-24, at the user's request: *show average sd min max in each
    case/model*). An SD over three reads is barely a statistic, and the range is
    the honest way to say how far apart they actually were -- it is a fact about
    the reads rather than an estimate from them. The two disagree usefully: a
    tight SD with a wide range is one stray read, which is exactly the shape the
    outlier fence is looking for one level down.

    `runs` and `measured` differ where a read carries no value for this
    particular number, so a mean is never read against a count that did not feed
    it.
    """
    values = [v for v in (pick(p) for p in points) if v is not None]
    mean, sd = _stats(values)
    return {"mean": mean, "sd": sd, "runs": len(points), "measured": len(values),
            "min": round(min(values), 2) if values else None,
            "max": round(max(values), 2) if values else None}


def _detail_tables(points: list) -> dict:
    """The three Detail and model questions, asked as tables rather than ratios.

    **Built 2026-08-24 at the user's request** -- *time vs detail for each doc*,
    *time in each model*, *detail vs accuracy*. All three are grouped means and
    none is a correlation, because the thing being varied is a preset with three
    values: an r over three points says nothing, while the three means read
    against each other say the whole of it.

    | key | rows x columns | the question |
    |---|---|---|
    | `time_detail_by_case` | document x Detail | what does raising Detail COST on this page |
    | `acc_detail` | Detail | what does raising Detail BUY -- and it is not monotone |
    | `model_time` | model | which model is expensive, at what accuracy |

    **The first two are the two halves of one trade and are deliberately not one
    table.** Detail is the project's headline accuracy/cost knob and the standing
    finding is that it is NOT monotone -- past ~4 MP the extra visual tokens
    degrade this model, so `original` can cost more and score less than `medium`.
    A single table of "Detail vs both" hides that by inviting a diagonal reading;
    two tables, cost and benefit, make the non-monotone case visible as a row
    that rises in one and falls in the other.

    **Per document for the cost, pooled for the benefit.** Time scales with the
    page -- a long document takes longer at every preset -- so a pooled time
    column would mostly rank the fixtures by length and say nothing about the
    preset. Accuracy has no such offset, so pooling it is what makes the trend
    legible; `by_case` under it keeps the per-document reading available.

    `model_time` carries accuracy beside the clock for the reason `by_case`'s
    speed column does: fast is a claim about cost alone, and the cheapest model
    here is not the one to run.
    """
    def group(pts, field):
        out = {}
        for p in pts:
            if p.get(field):
                out.setdefault(p[field], []).append(p)
        return out

    def secs(p):
        return p["seconds"]

    def char(p):
        return p["char"]

    # Every Detail actually present, in pixel-budget order -- the columns of the
    # two Detail tables, so both read left to right as "more pixels ->" and a
    # preset nothing ran at is simply not a column.
    present = {p["detail"] for p in points if p.get("detail")}
    details = [d for d in DETAIL_ORDER if d in present]

    # 1. Time vs Detail, per document. One row per document, one cell per preset.
    time_detail = []
    for case, pts in sorted(group(points, "case").items()):
        cells = {d: _figure(g, secs) for d, g in group(pts, "detail").items()}
        # The cheapest and dearest preset this document actually ran at, so the
        # row can say what raising Detail cost it without the reader diffing two
        # cells by eye. None where it only ran at one preset -- there is no
        # "raising" to price.
        ran = [d for d in details if d in cells and cells[d]["mean"] is not None]
        factor = None
        if len(ran) > 1:
            lo, hi = cells[ran[0]]["mean"], cells[ran[-1]]["mean"]
            factor = round(hi / lo, 2) if lo else None
        time_detail.append({"case": case, "runs": len(pts), "cells": cells,
                            "from": ran[0] if len(ran) > 1 else None,
                            "to": ran[-1] if len(ran) > 1 else None,
                            "factor": factor})

    # 2. Detail vs accuracy, pooled, with the per-document split under it.
    acc_detail = []
    for d in details:
        pts = [p for p in points if p.get("detail") == d]
        acc_detail.append({
            "detail": d,
            "accuracy": _figure(pts, char),
            "seconds": _figure(pts, secs),
            "documents": len({p["case"] for p in pts}),
            # Per document, so a preset that looks better only because it was
            # run on the easy fixtures is visible as such. This is the same
            # confound `_per_case` exists for, and the same fix.
            "by_case": {c: _figure(g, char)
                        for c, g in sorted(group(pts, "case").items())},
            # **And per model, which is where the confound actually lives.**
            # The models have not been run equally at every preset, so a pooled
            # Detail trend is partly a ranking of whichever models were run most
            # at each -- measured here, the pooled figures fall steadily with
            # more pixels while the main OCR model is FLAT across all three and
            # two others collapse only at `original`. That is the same mixing
            # effect the accuracy-vs-time split exists for, and it is why the
            # page draws this as a model x Detail grid rather than three numbers.
            "by_model": {m: _figure(g, char)
                         for m, g in sorted(group(pts, "model").items())},
        })

    # 3. Time per model, with accuracy beside it.
    model_time = []
    for model, pts in sorted(group(points, "model").items()):
        model_time.append({
            "model": model,
            "seconds": _figure(pts, secs),
            "accuracy": _figure(pts, char),
            "documents": len({p["case"] for p in pts}),
            # Per document again: models have not all been run on the same
            # fixtures, and a model that only ever read the short ones would
            # otherwise look fast.
            "by_case": {c: _figure(g, secs)
                        for c, g in sorted(group(pts, "case").items())},
        })

    return {"details": details,
            "time_detail_by_case": time_detail,
            "acc_detail": acc_detail,
            "model_time": model_time}


def _mark_counts(points: list, field: str) -> dict:
    """How a set of reads splits across one mark, blanks folded into `unknown`."""
    out = {}
    for p in points:
        key = p.get(field) or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


def _entity_stats(points: list, field: str) -> list:
    """Per document or per model: both numbers, with mean, SD and range.

    **Built 2026-08-24 at the user's request** -- *show average sd min max in
    each case/model*. Everything here could be read off the model x document
    grid by eye; what a row adds is the POOLED figure for one document across
    every model that read it, or for one model across every document it read,
    which the grid deliberately does not compute.

    The marks ride along so the two inferred ones are visible per row rather than
    only in aggregate -- a document every model read cold is a different thing
    from one with a warm read in the mean, and the composition strip cannot say
    which document.
    """
    groups = {}
    for p in points:
        if p.get(field):
            groups.setdefault(p[field], []).append(p)
    out = []
    for key, pts in sorted(groups.items()):
        out.append({
            "key": key,
            "runs": len(pts),
            "accuracy": _figure(pts, lambda p: p["char"]),
            "seconds": _figure(pts, lambda p: p["seconds"]),
            # How many of the OTHER axis this row is made of -- documents for a
            # model, models for a document. A mean over one of them is not a
            # general claim, and this is the only thing on the row that says so.
            "others": len({p["model" if field == "case" else "case"] for p in pts}),
            "details": sorted({p["detail"] for p in pts if p.get("detail")},
                              key=lambda d: DETAIL_ORDER.index(d)
                              if d in DETAIL_ORDER else 99),
            "hardware": _mark_counts(pts, "hardware"),
            "start": _mark_counts(pts, "start"),
        })
    return out


def _mark_stats(points: list) -> list:
    """What each inferred mark is worth: accuracy and time, meaned per bucket.

    One row per value of each mark (`GPU`, `CPU`, `cold`, `hot`), so the marks
    are shown as figures rather than only used to split a correlation.

    **A one-row mark is the expected case on a single-machine log and is still
    worth printing**: it says outright that this mark explains nothing here,
    which is a finding a reader would otherwise have to infer from a missing
    table. Blanks are excluded -- `unknown` is not a bucket anything can be
    concluded about.
    """
    out = []
    for kind in ("hardware", "start"):
        groups = {}
        for p in points:
            if p.get(kind):
                groups.setdefault(p[kind], []).append(p)
        for value, pts in sorted(groups.items()):
            out.append({
                "kind": kind,
                "key": value,
                "runs": len(pts),
                "accuracy": _figure(pts, lambda p: p["char"]),
                "seconds": _figure(pts, lambda p: p["seconds"]),
                "models": len({p["model"] for p in pts}),
                "documents": len({p["case"] for p in pts}),
            })
    return out


def _dominant(points: list, field: str) -> str:
    """The most common non-blank value of `field` in `points`, or `` if none."""
    counts = {}
    for p in points:
        if p.get(field):
            counts[p[field]] = counts.get(p[field], 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda k: (counts[k], k))


# --------------------------------------------------------------------------
# The presentation summary (2026-08-25, at the user's request: *1 page tab on
# summarization for presentation ... try be bias by default in this page --
# exclude bad product, outlier, single error case and some old error case*).
#
# **Every other table on this card is built to be defensible; this one is built
# to be SHOWN.** The difference is not the numbers -- they come from the same
# rows through the same helpers -- it is that this one is allowed to leave
# things out, and says which. Six blocks, in the order they were asked for:
# reading ranked, reading timed, Detail priced, extraction ranked, the two
# extraction shapes, extraction timed.
#
# Three things it does NOT do, each of which the other tabs do:
#
# - **No failure flags in the setting cell.** `by_ocr` prints a red incomplete
#   count beside the model because it qualifies every other figure on the row;
#   here the failure rate is one column and nothing else, as asked. A slide with
#   a warning under every row reads as a system that does not work.
# - **No pooled cross-model Detail row.** That number was deleted from the
#   analysis tab as the most misleading on the page -- the models have not been
#   run equally at each preset, so a pooled trend is partly a ranking of
#   whichever model was run most at each. The Detail block is a model x preset
#   grid for exactly that reason, and the bias makes it small enough to read.
# - **No claim that a rate is comparable across denominators.** Every mean here
#   carries the runs and documents behind it, the rule the whole card follows.


def _pres_weak(grouped: list) -> dict:
    """Models one pass's own figures disqualify: name -> why.

    **A rule over the rows, never a list of model names.** The project's own
    conclusions -- dots.ocr is dead, typhoon is the wrong tool for pass 2, qwen
    and phi were pulled for extraction and are not the readers -- are all
    reachable from `_standout_score` and a failure rate, and reaching them that
    way means the summary updates when a model does instead of when this file
    does.

    **It judges the rows the summary PRINTS.** `grouped` is the same
    `_pres_group` output the table is drawn from, so the score that disqualifies
    a model is the score beside its name. Reading it off a differently windowed
    figure was the first shape, and it kept a model showing 41% failure in the
    table because the rule had seen 39% somewhere else.

    Two tests, disqualifying different products. `PRESENT_MIN_SHARE` is relative
    to the best model in this pass -- *delivers less than 60% of what the leader
    delivers* -- because no absolute cut is right for both passes at once.
    `PRESENT_MAX_FAILURE` is absolute, because reliability is not graded on a
    curve: a model that reads well on the attempts it completes and does not
    complete enough of them is out whatever the leader manages.

    A model with fewer than `PRESENT_MIN_RUNS` runs is never dropped -- thin
    evidence is not a verdict, the same rule `STANDOUT_MIN_RUNS` applies one
    level down. It is also never the leader the share is taken against, or one
    lucky run would set the bar for everything else.
    """
    scores = {}
    for entry in grouped:
        if not entry["key"]:
            continue
        scores[entry["key"]] = _standout_score(entry["value"]["mean"],
                                               entry["failure_rate"])
    solid = [e for e in grouped if e["runs"] >= settings.PRESENT_MIN_RUNS
             and scores.get(e["key"]) is not None]
    best = max((scores[e["key"]] for e in solid), default=None)
    out = {}
    for entry in grouped:
        key = entry["key"]
        if not key or entry["runs"] < settings.PRESENT_MIN_RUNS:
            continue
        score, rate = scores.get(key), entry["failure_rate"]
        why = []
        if best and score is not None and score < best * settings.PRESENT_MIN_SHARE:
            why.append("delivers %.0f%% of the leader's %.1f per attempt"
                       % (100.0 * score / best, best))
        if rate is not None and rate > settings.PRESENT_MAX_FAILURE:
            why.append("fails %.0f%% of its runs, over %g%%"
                       % (rate, settings.PRESENT_MAX_FAILURE))
        if why:
            out[key] = {"model": key, "reason": " and ".join(why),
                        "score": score, "accuracy": entry["value"]["mean"],
                        "failure_rate": rate, "runs": entry["runs"]}
    return out


def _pres_points(rows: list, pass_: str) -> list:
    """One point per run of `pass_`, with the number, the clock and the knobs.

    Pass 1 is a read: `char_accuracy` and `seconds`. Pass 2 is an extraction:
    the priority-1 field rate and `extract_seconds`. Neither is scored where the
    run did not finish -- the standing rule, and the reason the failure rate is
    a column rather than a footnote -- but the run is still a point, so the
    failure rate has a denominator.
    """
    points = []
    for row in rows:
        if pass_ == "ocr":
            if (row.get("run_type") or "ocr") == "extract":
                continue
            if not _current_detail(row):
                continue
            model, case = row.get("model") or "", _document_key(row)
            if not model or not case:
                continue
            failed = _incomplete(row)
            points.append({
                "model": model, "case": case, "detail": detail_of(row),
                "mode": "", "failed": failed,
                "seconds": None if failed else _num(row.get("seconds"), None),
                "value": None if failed else _num(row.get("char_accuracy"), None),
                "tps": None if failed else _num(row.get("tokens_per_second"), None),
                # Tokens GENERATED -- `predicted_n` / `completion_tokens`, summed
                # over the document's pages. **Not the prompt**: the image's
                # visual tokens and the instruction are prefill and are counted
                # nowhere in this column, so it is the length of the transcript
                # and not the cost of the request. Named for that on the page.
                "tokens": None if failed else _num(row.get("tokens"), None),
                # The same recall over the three scripts, so the tab that is
                # meant to be SHOWN can say which of them a model loses rather
                # than only how much it loses overall. Blank on a run written
                # before 2026-09-04 and on a document that prints too little of
                # a script to measure it; `_pres_group` skips a blank, so a
                # model with none reports no rate rather than a zero.
                **{name: (None if failed
                          else _num(row.get(f"{name}_accuracy"), None))
                   for name in scoring.SCRIPTS},
            })
        else:
            # The same test `by_extract` uses to decide a row has a pass-2
            # figure at all, plus the shape: a run that never reached pass 2
            # carries no `extract_mode`, and grouping it under a blank shape
            # would put a column on the slide for a request nobody made.
            if _pipeline(row) == "read":
                continue
            mode = row.get("extract_mode") or ""
            model = row.get("extract_model") or row.get("model") or ""
            case = _document_key(row)
            if not model or not case or not mode:
                continue
            failed = _extract_incomplete(row)
            points.append({
                "model": model, "case": case, "detail": "", "mode": mode,
                "failed": failed,
                "seconds": None if failed else _num(row.get("extract_seconds"), None),
                "value": None if failed else _p1_rate(row),
                "tps": None,
                # Pass 2 has `extract_tokens`, and it is deliberately not read
                # here: nothing on the tab shows it, and a populated field no
                # column draws is a field the next reader has to work out the
                # status of. Add it when a column wants it.
                "tokens": None,
                # Pass 2 reads no page, so it has no script rates. Written as
                # None rather than left out, so every point has the same shape
                # and `_pres_group` can take the same figures of both passes.
                **{name: None for name in scoring.SCRIPTS},
            })
    return points


def _pres_group(points: list, key_of) -> list:
    """Points grouped, then meaned per document and only then across documents.

    `_per_case`'s rule applied to a flat point list: a model run five times on
    one fixture is not five samples of it. The mean is over the documents, the
    counts are over the runs, and both are on the row.

    **The window is applied here, with the key this grouping uses** (2026-08-28,
    at the user's request: *the run-count filter works on the other tabs but not
    the summary*). It was the one compilation on the card that never called
    `recent_by`, so the card's window box moved every table except the tab most
    likely to be read -- and the payload reported a `window` the six blocks were
    not taken under, which is worse than not offering the control at all.

    It is done in here rather than once in `presentation` because the blocks
    group on four different keys -- model, model x preset, model x shape -- and
    a window taken on one of them describes the other three wrongly. That is the
    card's standing rule (`recent_by`): *N means N of that setting, that model,
    that document*, not a slice off the end of the log, and each table trims
    with the key it groups on.
    """
    points = recent_by(points, key_of)
    groups = {}
    for point in points:
        key = key_of(point)
        groups.setdefault(key, {}).setdefault(point["case"], []).append(point)
    out = []
    for key, cases in groups.items():
        runs = [p for pts in cases.values() for p in pts]
        failed = [p for p in runs if p["failed"]]
        ok = {c: [p for p in pts if not p["failed"]] for c, pts in cases.items()}
        ok = {c: pts for c, pts in ok.items() if pts}

        def per_case(pick, ok=ok, runs=runs):
            # Mean of the per-document means, with the spread of the documents
            # beside it -- `_figure` over one value per document rather than
            # over every run, so a repeat cannot weight its fixture twice.
            means = []
            for pts in ok.values():
                values = [v for v in (pick(p) for p in pts) if v is not None]
                if values:
                    means.append(sum(values) / len(values))
            figure = _figure([{"v": m} for m in means], lambda d: d["v"])
            figure["runs"] = len(runs)
            figure["documents"] = len(means)
            return figure

        out.append({
            "key": key,
            "runs": len(runs),
            "documents": len(cases),
            "failed": len(failed),
            # One column, as asked. A run that did not finish is counted here
            # and scored nowhere -- so a high mean beside a high rate is one
            # good run among several bad ones, which is the only reading of
            # this pair.
            "failure_rate": round(100.0 * len(failed) / len(runs), 1) if runs else None,
            "value": per_case(lambda p: p["value"]),
            **{name: per_case(lambda p, n=name: p.get(n))
               for name in scoring.SCRIPTS},
            "seconds": per_case(lambda p: p["seconds"]),
            "tps": per_case(lambda p: p["tps"]),
            "tokens": per_case(lambda p: p["tokens"]),
        })
    return out


def _pres_cost(entry: dict):
    """Seconds per point of accuracy: a ratio of two means, not a measured rate.

    It prices accuracy roughly and ranks value, and it is not a per-run figure
    -- the same caveat the analysis tab's own `s/point` column carries. None
    where either half is missing or the accuracy is zero, because dividing by
    nothing read is not a cost anyone pays.
    """
    secs, acc = entry["seconds"]["mean"], entry["value"]["mean"]
    if secs is None or not acc or acc <= 0:
        return None
    return round(secs / acc, 2)


def presentation(rows: list = None, bias: bool = True,
                 single_source: bool = True) -> dict:
    """The six-block summary, biased by default and saying so.

    `bias=False` returns the same six blocks over every row that survived the
    card's own filters, which is what makes the bias arguable rather than
    load-bearing: the toggle is one click and the difference is the argument.

    Applied in this order, and the order matters -- the weak-model rule is
    computed over the rows a single-source drop has already left, or a model
    whose whole failure record is one bad pairing would be disqualified by
    failures the summary is about to stop counting.

    **The bias runs over the rows in view, and the per-group window is applied
    under it, per block** (`_pres_group`). The rules are not per block -- a weak
    model is weak for a whole pass, and a time outlier is judged against its own
    cell's median, which is neither of the keys a block groups on -- so they see
    the pool, and each block then takes its own last N of it. One consequence to
    read the banner with: `bias` counts what the rules removed from that pool,
    which can include runs a window would have dropped anyway.

    `single_source=False` says the CALLER has already dropped them, which the
    card-wide toggle does for every other table on the card. **The rule must not
    run twice**: it is not idempotent, because a model whose remaining failures
    all fall on one document becomes single-source on a second pass and loses
    those too. So the caller does it once, and this skips its own step 1 rather
    than repeating it.
    """
    everything = _for_summary(read(limit=10 ** 6) if rows is None else rows)
    steps = []
    rows = everything
    if bias and single_source:
        # 1. Single-source failures. A document dominated by one bad model is
        # not a hard document, and a model that loops on one fixture and reads
        # everything else is not an unreliable model. Only the FAILING runs of
        # such a group go; the clean ones stay and still speak for it.
        reduced = drop_single_source(rows)
        if len(reduced) != len(rows):
            steps.append({"rule": "single_source", "pass": "",
                          "dropped": len(rows) - len(reduced), "models": []})
        rows = reduced
    weak = {"ocr": {}, "extract": {}}

    blocks = {}
    for pass_ in ("ocr", "extract"):
        points = _pres_points(rows, pass_)
        if bias:
            # Grouped once unbiased to decide, then again below over what is
            # left. The first grouping is thrown away -- it exists only so the
            # rule sees the same figures the reader would have.
            weak[pass_] = _pres_weak(_pres_group(points, lambda p: p["model"]))
        if bias and weak[pass_]:
            # 2. Models the log disqualifies for THIS pass. Per pass, because
            # this project has a model that reads best and extracts second
            # worst, and one with no vision at all that ranks near the top of
            # the form -- a single list of "good models" would be wrong about
            # both.
            kept = [p for p in points if p["model"] not in weak[pass_]]
            steps.append({"rule": "weak_model", "pass": pass_,
                          "dropped": len(points) - len(kept),
                          "models": sorted(weak[pass_])})
            points = kept
        if bias:
            # 3. Time outliers, per cell and robustly -- a cold model load or a
            # runaway puts one run's clock far above the rest of its cell and
            # moves a mean on its own. The cell carries the knob that
            # legitimately changes the clock: Detail for a read, the request
            # shape for an extraction. **Only the TIME is discounted**; the run
            # still counts and still scores, because nothing about its accuracy
            # was in question.
            cell = ((lambda p: (p["model"], p["case"], p["detail"]))
                    if pass_ == "ocr" else
                    (lambda p: (p["model"], p["case"], p["mode"])))
            marked = [dict(p) for p in points]
            flagged = _flag_time_outliers(marked, cell=cell)
            if flagged:
                steps.append({"rule": "time_outlier", "pass": pass_,
                              "dropped": flagged, "models": []})
            for point, mark in zip(points, marked):
                if mark["outlier"]:
                    point["seconds"] = None
        blocks[pass_] = points

    ocr, ext = blocks["ocr"], blocks["extract"]

    def ranked(points, key_of):
        out = _pres_group(points, key_of)
        out.sort(key=lambda e: (e["value"]["mean"] if e["value"]["mean"] is not None
                                else -1.0, e["documents"]), reverse=True)
        return out

    models = ranked(ocr, lambda p: p["model"])
    ex_models = ranked(ext, lambda p: p["model"])
    shapes = [{**e, "model": e["key"][0], "mode": e["key"][1]}
              for e in ranked(ext, lambda p: (p["model"], p["mode"]))]
    for entry in models + ex_models:
        entry["model"] = entry["key"]
    for entry in models + ex_models + shapes:
        entry["cost"] = _pres_cost(entry)

    # The Detail grid: one row per model, one cell per preset, and what raising
    # the preset bought and cost THAT MODEL. Never pooled across models -- see
    # the block comment above.
    present = {p["detail"] for p in ocr if p["detail"]}
    details = [d for d in DETAIL_ORDER if d in present]
    by_model = {}
    for cell in _pres_group(ocr, lambda p: (p["model"], p["detail"])):
        by_model.setdefault(cell["key"][0], {})[cell["key"][1]] = cell
    order = [e["model"] for e in models]
    detail_rows = []
    for model in sorted(by_model, key=lambda m: order.index(m) if m in order else 99):
        cells = by_model[model]
        ran = [d for d in details
               if d in cells and cells[d]["value"]["mean"] is not None]
        timed = [d for d in details
                 if d in cells and cells[d]["seconds"]["mean"] is not None]
        row = {"model": model, "cells": cells,
               "from": ran[0] if len(ran) > 1 else None,
               "to": ran[-1] if len(ran) > 1 else None,
               "acc_delta": None, "time_factor": None,
               "time_from": None, "time_to": None,
               "runs": sum(c["runs"] for c in cells.values()),
               "documents": max((c["documents"] for c in cells.values()), default=0)}
        if len(ran) > 1:
            row["acc_delta"] = round(cells[ran[-1]]["value"]["mean"]
                                     - cells[ran[0]]["value"]["mean"], 1)
        if len(timed) > 1:
            low = cells[timed[0]]["seconds"]["mean"]
            row["time_factor"] = (round(cells[timed[-1]]["seconds"]["mean"] / low, 2)
                                  if low else None)
            row["time_from"], row["time_to"] = timed[0], timed[-1]
        detail_rows.append(row)

    return {
        "biased": bool(bias),
        # What the bias did, rule by rule, so the page can print it and a reader
        # can put any of it back. Empty on an unbiased view, and empty on a
        # biased one that found nothing to drop -- which is a real and different
        # statement about the log.
        "bias": steps,
        "weak": {k: sorted(v.values(), key=lambda e: e["score"] or 0.0)
                 for k, v in weak.items()},
        "thresholds": {"min_share": settings.PRESENT_MIN_SHARE,
                       "max_failure": settings.PRESENT_MAX_FAILURE,
                       "min_runs": settings.PRESENT_MIN_RUNS},
        # 1 + 2: ranked on the transcript, and timed. One aggregation, two
        # tables -- the page sorts and columns them differently, because *which
        # reader is best* and *what does each reader cost* are two questions and
        # a reader looking for the second should not have to find it inside the
        # first.
        "ocr_models": models,
        # 3: what raising Detail bought and cost, per model.
        "ocr_details": details,
        "ocr_detail": detail_rows,
        # 4 + 6: the same pair for pass 2.
        "extract_models": ex_models,
        # 5: single against agentic, per model, with the clock.
        "extract_shapes": shapes,
        # Per group, with the key each block groups on -- see `_pres_group`.
        # Reported here since 2026-08-25 and only APPLIED since 2026-08-28: the
        # six blocks were the one compilation on the card that never called
        # `recent_by`, so this key named a window they were not taken under.
        "window": window_size() or None,
        # The rows the bias rules ran over -- BEFORE the per-group window, which
        # is applied per block below them. So this is what the card's filters
        # matched, not the number of runs any one table averages, and every row
        # prints its own `runs` for that.
        "runs": len(rows),
    }


def conditions(rows: list = None) -> dict:
    """What the runs in view were made UNDER, as opposed to what they scored.

    **Built 2026-08-25 for the Summary tab's environment card** (*have a banner
    card for my hardware and this env*). The card's other half is
    `machine.describe`, which reports the box this PROCESS is running on. These
    two must never be merged, and the card keeps them apart on purpose: the log
    outlives the machine, so a row read on another box would otherwise be
    described by this one's CPU.

    Everything here is counted from the rows, so it narrows with the card's
    filters like every other table. The hardware and warmth marks are the
    INFERRED ones (`run_hardware` / `run_start`) and are labelled as such
    wherever they are shown -- the app talks HTTP to a server it did not launch
    and knows neither the GPU layer count nor whether the model was resident.
    """
    rows = _for_summary(_read_all() if rows is None else rows)

    def tally(value_of, keep=None):
        counts = {}
        for row in rows:
            if keep and not keep(row):
                continue
            value = value_of(row)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return [{"value": v, "runs": counts[v]}
                for v in sorted(counts, key=lambda k: (-counts[k], k))]

    stamps = sorted(r.get("timestamp") or "" for r in rows)
    stamps = [s for s in stamps if s]
    return {
        "runs": len(rows),
        # The window these runs were read under, so the card cannot claim to
        # describe more of the log than the tables beside it do.
        "first": stamps[0] if stamps else None,
        "last": stamps[-1] if stamps else None,
        "backends": tally(lambda r: r.get("backend") or ""),
        "servers": tally(lambda r: r.get("server") or ""),
        # The two model questions kept apart, the same rule `FILTER_FIELDS`
        # follows: a fields-only row has no reading model because it read no
        # page, and counting it as one is what emptied the card once already.
        "read_models": tally(FILTER_FIELDS["model"]),
        "extract_models": tally(FILTER_FIELDS["extract_model"]),
        "details": tally(lambda r: detail_of(r),
                         lambda r: (r.get("run_type") or "ocr") != "extract"),
        "profiles": tally(lambda r: r.get("ocr_profile") or ""),
        "modes": tally(lambda r: r.get("extract_mode") or ""),
        # Inferred, never recorded. Named `_inferred` in the payload so nothing
        # downstream can print them as fact by accident.
        "hardware_inferred": tally(run_hardware),
        "start_inferred": tally(run_start),
    }


def totals(rows: list = None, logged: int = None) -> dict:
    """Headline counts and every compiled table, over the most recent rows.

    `rows` is the log, newest first, and defaults to all of it. A caller passes
    a narrower set to ask the same questions of a subset -- one document, or
    every model but one -- and passes `logged` beside it so the card can still
    say what the file holds: **a filtered summary that reported its own row count
    as the size of the log would read as a log somebody had deleted from.**

    **Not over the whole log** (2026-08-21, at the user's request: *only get 20
    latest run only since some old run maybe on the old system*). Every figure
    below -- the means, the per-document bests, `by_ocr`, `by_extract`,
    `by_case`, `standouts` -- is taken over the last `settings.SUMMARY_RUNS`
    rows. See that setting for why: this project changes the thing being measured
    about as often as it measures it, and a row from before a scorer change is
    not a second sample of the same quantity.

    The window is rows, not reads, and it is returned as `window` beside
    `logged`, the true size of the file. A card that says "20 runs" over a log of
    two hundred is describing the wrong thing unless it also says which twenty.

    Re-extractions are counted apart from reads. They are rows like any other,
    but a row that never touched pass 1 is not another document read, and adding
    it to `runs` would inflate the count that every other figure here is read
    against. Its pass-1 columns are blank, so the sums below already skip it.
    """
    everything = read(limit=10 ** 6) if rows is None else rows
    # The header counts are per DOCUMENT-windowed, the same rule the tables use
    # and the only one that means anything here: a mean over "the log's last
    # twenty rows" is a mean over whatever was run last night. The four tables
    # below are handed the whole file and window it by their own keys.
    rows = recent_by(everything, _document_key)
    extracts = sum(1 for r in rows if (r.get("run_type") or "ocr") == "extract")
    accuracies = []
    seconds = 0.0
    tokens = 0
    for row in rows:
        try:
            seconds += float(row.get("seconds") or 0)
            tokens += int(float(row.get("tokens") or 0))
        except ValueError:
            pass
        try:
            if row.get("char_accuracy"):
                accuracies.append(float(row["char_accuracy"]))
        except ValueError:
            pass
    # The best score each document has ever reached, and what reached it.
    # Deliberately taken over *every* row whatever it ran under -- model,
    # backend, detail, prompt, extraction mode -- because that is the question
    # the mean cannot answer: the mean says what a typical run gets, and this
    # says what the document is known to be capable of. The condition is
    # carried with it, since a best score with nothing attached is not
    # reproducible and therefore not much use.
    #
    # Derived from `by_case` rather than counted again here, so the chips on the
    # run-log card and the per-document summary under them cannot name two
    # different runs as one document's best.
    cases = by_case(everything)
    best = {c: s["best_char"] for c, s in cases.items() if s["best_char"]}
    top = max(best.items(), key=lambda kv: kv[1]["char_accuracy"], default=None)
    return {
        "runs": len(rows) - extracts,
        "extracts": extracts,
        "scored": len(accuracies),
        "mean_accuracy": round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
        # The single highest score in the file, and the document that holds it.
        # Read beside `best_by_case`: one easy fixture can carry this number well
        # above what the harder ones have ever managed.
        "best_accuracy": round(top[1]["char_accuracy"], 2) if top else None,
        "best_case": top[0] if top else None,
        "best_by_case": best,
        # The same documents again, with the settings that scored the fields best
        # and finished quickest beside the transcript score -- three questions the
        # single "best ever" chip cannot answer at once. See `by_case`.
        "by_case": cases,
        # And the same runs by setting rather than by document -- which model to
        # run at all, which the per-document table scatters across one row each.
        # Two tables, not one: pass 1 is scored on the transcript and paid for in
        # prefill and decode, pass 2 on whether a value reached the right key,
        # and a single row averaging both answers neither. See `by_ocr` and
        # `by_extract`.
        "by_ocr": by_ocr(everything),
        "by_extract": by_extract(everything),
        # And the same runs asked the one question those two tables cannot put
        # into words: which model and which document are carrying this, and
        # which are failing it. See `standouts`.
        "standouts": standouts(everything),
        # And the same runs counted rather than scored: what is FAILING, per
        # model, per document, and per pairing of the two. The tables above rank
        # on accuracy and fold failures into it; this one ranks on the failures
        # themselves and says whether a row's errors came from one contributor
        # or from many. See `errors`.
        "errors": errors(everything),
        # And the same reads asked whether their three variables move together:
        # does accuracy depend on time, on the document, or on the model? Pass 1
        # only, because the question is about reading a page. See `ocr_analysis`.
        "ocr_analysis": ocr_analysis(everything),
        # And the same runs asked what each model and each document is WEAK at,
        # which every table above averages over by construction: the transcript
        # score split into Thai, English and numerals, and the field score split
        # into the individual keys. See `script_accuracy` and `field_weakness`.
        "script_accuracy": script_accuracy(everything),
        "field_weakness": field_weakness(everything),
        "seconds": round(seconds, 1),
        "tokens": tokens,
        # What this is a summary OF. `logged` is the file; `window` is how many
        # of it were used, or None for all of it -- and every figure above is
        # read against the second, not the first.
        "logged": len(everything) if logged is None else logged,
        # How many rows this summary was compiled over, before any windowing.
        # Equal to `logged` on an unfiltered view and smaller on a filtered one,
        # which is the only thing on the card that says a filter is in force
        # besides the pickers themselves.
        "matched": len(everything),
        # Per group, not a slice of the file: every table below covers the last
        # `window` runs OF EACH setting, model or document. The page says so,
        # because "last 20" reads as "the last 20 rows" otherwise.
        "window": window_size() or None,
        # The read-quality floor these figures were compiled under, as a
        # PERCENTAGE, because that is the unit of the column it is compared with
        # and of the control on the page. A field score from a transcript below
        # it belongs to the read, not to the extractor, and is left out of every
        # correctness figure here -- see `_field_trusted`.
        "min_read_pct": round(read_floor() * 100, 1),
        # Rows that ran at a Detail this build no longer offers, by name. Said
        # out loud because `by_ocr` and `by_case` drop them, and a table quietly
        # missing a quarter of the log reads as a table nobody has run much.
        # Empty on a log with none, which is what a fresh one looks like.
        "retired_detail": retired_detail(everything),
        # How many of the reads in view still carry the pre-2026-09-04
        # `char_accuracy` -- an edit distance rather than recall of the ground
        # truth. Said out loud for the same reason `retired_detail` is: while
        # both eras are in one window, the pass-1 means are over two different
        # questions. It empties itself as the window rolls over, and the note
        # disappears with it.
        "legacy_char": legacy_char_rows(everything),
        "path": str(LOG_PATH),
    }
