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
import threading
from datetime import datetime

import config
import grounding
import settings
from settings import SUMMARY_RUNS

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
                   "extract_model")

_TIERS = ("p1_present", "p1_absent", "p2_present", "p2_absent",
          "p3_present", "p3_absent")


def _extract_cells(summary: dict) -> dict:
    """The pass-2 half of a row: what was extracted, how, and how real it is."""
    extracted = (summary or {}).get("extracted") or {}
    fields = extracted.get("fields")
    grounded = extracted.get("grounding") or {}
    repetition = grounding.list_repetition(fields if isinstance(fields, dict) else {})
    return {
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
        **(grounding.tier_counts(fields) if fields
           else {k: "" for k in _TIERS}),
        **_field_cells(extracted.get("field_score")),
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
                                "p1_scored", "p1_partial")}
    # The priority-1 tier on its own, which since the schema narrowed is very
    # nearly the whole form -- but not by definition, so it is taken from the
    # tier tally rather than assumed to equal `overall`. Written as a pair or not
    # at all: a count of correct values with no denominator beside it is not a
    # figure anyone can read, and `p1_scored` is per document.
    p1 = ((score or {}).get("scalars") or {}).get("p1") or {}
    scored = p1.get("expected") or 0
    counts = p1.get("counts") or {}
    return {"field_acc": _pct(overall.get("accuracy")),
            "field_expected": overall["expected"],
            "p1_correct": counts.get("correct", "") if scored else "",
            "p1_partial": counts.get("partial", "") if scored else "",
            "p1_scored": scored if scored else ""}


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
    """
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


def _extract_incomplete(row: dict) -> bool:
    """True where PASS 2 did not finish: it cycled, or it never replied.

    Separate from `_incomplete` because the two passes fail differently and a
    row can be one without the other -- a page that read cleanly and then looped
    inside `other_fields` is a good read and a failed extraction, and the two
    tables must be able to say so independently.
    """
    return (_num(row.get("extract_looped"), 0.0) > 0
            or (row.get("status") or "") in _INCOMPLETE)


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
    if _num(row.get("extract_looped"), 0.0) > 0:
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
        "path": str(LOG_PATH),
    }
