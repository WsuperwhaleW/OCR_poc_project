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
    "verify_seconds",
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
    "verdict",          # number-check verdict
    "error",
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


def record(summary: dict, source: dict = None, extras: dict = None) -> dict:
    """Append one run. Returns the row written, or None if logging failed.

    `summary` is the payload both OCR endpoints already build (`summarise()` plus
    truth/extracted/verified). Nothing here is required: a failed run logs what it
    knows and leaves the rest blank, because a row saying a read failed after 40 s
    is the row you most want later.
    """
    summary = summary or {}
    source = source or {}
    extras = extras or {}

    truth = summary.get("truth") or {}
    if truth.get("error"):
        truth = {}
    extracted = summary.get("extracted") or {}
    grounded = extracted.get("grounding") or {}
    verified = summary.get("verified") or {}
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
        "extract_seconds": extracted.get("seconds", ""),
        "extract_tokens": extracted.get("tokens", ""),
        "verify_seconds": (verified.get("review") or {}).get("seconds", ""),
        "grounded_pct": _pct(grounded.get("grounded_ratio")),
        "ungrounded": len(grounded.get("flagged") or []) if grounded else "",
        "fields_missing": len(grounded.get("missing") or []) if grounded else "",
        # Blank rather than 0 where nothing was extracted: a failed run left the
        # tiers unmeasured, which is not the same as measuring them at zero.
        **(grounding.tier_counts(extracted["fields"])
           if extracted.get("fields") else
           {k: "" for k in ("p1_present", "p1_absent", "p2_present", "p2_absent")}),
        "case": truth.get("case", ""),
        "char_accuracy": _pct(truth.get("char_accuracy")),
        "word_accuracy": _pct(truth.get("word_accuracy")),
        "char_accuracy_no_marks": _pct(truth.get("char_accuracy_no_marks")),
        "verdict": verified.get("verdict", ""),
        "error": str(error)[:300],
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
    """Headline counts over the whole log, for the card header."""
    rows = read(limit=10 ** 6)
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
    return {
        "runs": len(rows),
        "scored": len(accuracies),
        "mean_accuracy": round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
        "seconds": round(seconds, 1),
        "tokens": tokens,
        "path": str(LOG_PATH),
    }
