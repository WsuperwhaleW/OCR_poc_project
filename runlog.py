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

import csv
import os
import threading
from datetime import datetime

import config
import grounding
import settings

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
]

# The value the run was actually made with, taken from `settings` rather than
# re-read here: two independent reads of one environment variable can disagree
# after a clamp or a typo, and this column would then describe a run that never
# happened. `settings` imports nothing but `config`, so there is no import cycle
# (app imports runlog).
_DRY = settings.DRY_MULTIPLIER

_lock = threading.Lock()


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
                   "grounded_pct", "ungrounded", "fields_missing",
                   "p1_present", "p1_absent", "p2_present", "p2_absent",
                   "field_acc", "field_expected")

_TIERS = ("p1_present", "p1_absent", "p2_present", "p2_absent")


def _extract_cells(summary: dict) -> dict:
    """The pass-2 half of a row: what was extracted, how, and how real it is."""
    extracted = (summary or {}).get("extracted") or {}
    grounded = extracted.get("grounding") or {}
    return {
        "extract_seconds": extracted.get("seconds", ""),
        "extract_tokens": extracted.get("tokens", ""),
        "extract_mode": extracted.get("mode", ""),
        "grounded_pct": _pct(grounded.get("grounded_ratio")),
        "ungrounded": len(grounded.get("flagged") or []) if grounded else "",
        "fields_missing": len(grounded.get("missing") or []) if grounded else "",
        # Blank rather than 0 where nothing was extracted: a failed run left the
        # tiers unmeasured, which is not the same as measuring them at zero.
        **(grounding.tier_counts(extracted["fields"]) if extracted.get("fields")
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
        return {"field_acc": "", "field_expected": ""}
    return {"field_acc": _pct(overall.get("accuracy")),
            "field_expected": overall["expected"]}


def _num(value, default=-1.0) -> float:
    """A cell as a number; blank and unparsable both sort below any real value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_score(cells: dict) -> tuple:
    """How good one extraction was, as a tuple that sorts.

    Coverage first and grounding last, which is the order the request is usually
    made in: the complaint is nearly always a field that came back empty. The two
    tiers rank before the ratio because a run that filled two more priority-1
    keys is better even at a slightly lower grounded percentage -- and a run that
    filled the same keys is separated by how many of them are real.

    Blank sorts below zero, so an extraction that never ran can never displace one
    that did.

    `field_acc` is deliberately NOT part of this, tempting though it is as the one
    column that measures correctness rather than coverage: it exists only for the
    handful of documents someone has written a field truth file for. Ranking by it
    would make a blank mean two different things in the same tuple -- "no truth
    file for this document" and "this extraction found nothing" -- and the second
    is the only one the blank-sorts-below-zero rule above is safe for.
    """
    return (_num(cells.get("p1_present")), _num(cells.get("p2_present")),
            _num(cells.get("grounded_pct")))


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
    new extraction scores strictly higher by `extract_score` -- 12/14 priority-1
    keys replaced by 14/14, never the other way round. The read's own columns --
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


def read(limit: int = 100) -> list:
    """The most recent rows, newest first."""
    if not LOG_PATH.exists():
        return []
    try:
        with _lock:
            with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
    except Exception:
        return []
    return rows[-limit:][::-1]


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
    """
    rows = read(limit=10 ** 6) if rows is None else rows
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
        field = _num(row.get("field_acc"), None)

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


def totals() -> dict:
    """Headline counts over the whole log, for the card header.

    Re-extractions are counted apart from reads. They are rows like any other,
    but a row that never touched pass 1 is not another document read, and adding
    it to `runs` would inflate the count that every other figure here is read
    against. Its pass-1 columns are blank, so the sums below already skip it.
    """
    rows = read(limit=10 ** 6)
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
    cases = by_case(rows)
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
        "seconds": round(seconds, 1),
        "tokens": tokens,
        "path": str(LOG_PATH),
    }
