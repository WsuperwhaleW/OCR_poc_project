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
                   "p1_present", "p1_absent", "p2_present", "p2_absent")

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
    }


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
    best = {}
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
        case = row.get("case") or ""
        try:
            value = float(row.get("char_accuracy") or "")
        except ValueError:
            continue
        if case and (case not in best or value > best[case]["char_accuracy"]):
            best[case] = {
                "char_accuracy": value,
                "word_accuracy": row.get("word_accuracy", ""),
                "timestamp": row.get("timestamp", ""),
                "model": row.get("model", ""),
                "backend": row.get("backend", ""),
                "detail": row.get("detail", ""),
                "extract_mode": row.get("extract_mode", ""),
            }
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
        "seconds": round(seconds, 1),
        "tokens": tokens,
        "path": str(LOG_PATH),
    }
