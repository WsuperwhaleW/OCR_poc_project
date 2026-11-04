"""The stress test's "Save each job to the run log" switch.

Off -- the default -- nothing is written (test_stress.py pins that). On, every
MEASURED job is one ordinary run-log row tagged with the stress run and its
concurrency; warm-up, cancelled and skipped jobs, and classification, write none.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import engine
import runlog

TEXT_STATUS = {"text_available": True, "text_reason": None, "kind": "ollama",
               "url": "http://l", "model": "m"}
READ_STATUS = {"available": True, "reason": None, "kind": "ollama",
               "url": "http://l", "model": "reader"}
RESULT = {"tokens": 10, "model": "m", "field_score": {"overall": {
    "counts": {"correct": 1, "partial": 0}, "expected": 2}}}


def _read_page(page, stats, cancel=None):
    stats.update(new_tokens=5, model="reader", url="http://l", backend="ollama")
    return "a transcript"


def _run(body):
    cfg = app._stress_config(body)
    return [json.loads(x) for x in app._stress_run(cfg, app.MultiCancel())]


class SaveLogTests(unittest.TestCase):
    def setUp(self):
        self.status = patch.object(app.backends, "extract_status",
                                   return_value=TEXT_STATUS)
        self.status.start()
        self.addCleanup(self.status.stop)

    def test_extract_jobs_are_saved_and_warmup_is_not(self):
        with patch.object(app, "extract_fields", return_value=RESULT), \
             patch.object(engine, "log_extract") as logged, \
             patch.object(engine, "log_run",
                          side_effect=AssertionError("an extract job wrote a read row")):
            events = _run({"mode": "extract", "concurrency": 2, "jobs": 3, "warmup": 1,
                           "seed": 1, "save_log": True})
        plan = events[0]
        self.assertTrue(plan["save_log"])
        self.assertTrue(plan["run_id"].endswith("-1"))
        self.assertEqual(logged.call_count, 3)            # the warm-up job is not
        for call in logged.call_args_list:
            result, job, context = call.args
            self.assertIs(result, RESULT)
            self.assertEqual(context["source"]["origin"], "truth")
            self.assertTrue(context["case"])
            self.assertEqual(call.kwargs["extra"], {"stress_run": plan["run_id"],
                                                    "stress_concurrency": 2})

    def test_a_failed_job_is_saved_as_a_failure(self):
        with patch.object(app, "extract_fields", side_effect=ValueError("boom")), \
             patch.object(engine, "log_extract") as logged:
            _run({"mode": "extract", "concurrency": 1, "jobs": 2, "save_log": True})
        self.assertEqual(logged.call_count, 2)
        self.assertIn("boom", logged.call_args.args[0]["error"])

    def test_off_writes_nothing(self):
        with patch.object(app, "extract_fields", return_value=RESULT), \
             patch.object(engine, "log_extract",
                          side_effect=AssertionError("logged with the switch off")):
            events = _run({"mode": "extract", "concurrency": 1, "jobs": 2})
        self.assertFalse(events[0]["save_log"])
        self.assertEqual(events[0]["run_id"], "")

    def test_classification_is_never_saved(self):
        self.assertFalse(app._stress_config({"mode": "classify", "save_log": True})["save_log"])


class SaveLogReadTests(unittest.TestCase):
    """OCR and full: one `log_run` row per job, read and extraction on it."""

    def setUp(self):
        for target, value in ((engine, "llama_status"), (app.backends, "extract_status")):
            p = patch.object(target, value,
                             return_value=READ_STATUS if value == "llama_status"
                             else TEXT_STATUS)
            p.start()
            self.addCleanup(p.stop)
        src = {"name": "x.pdf", "size_bytes": 3, "origin": "case"}
        for target, attr, kw in ((app, "_stress_pages", {"return_value": (["page"], src)}),
                                 (engine, "read_page", {"side_effect": _read_page})):
            p = patch.object(target, attr, **kw)
            p.start()
            self.addCleanup(p.stop)

    def test_ocr_jobs_write_a_read_row(self):
        with patch.object(engine, "log_run") as logged, \
             patch.object(app, "extract_fields",
                          side_effect=AssertionError("an ocr job extracted")):
            events = _run({"mode": "ocr", "concurrency": 2, "jobs": 2, "lock_case": "sol002",
                           "save_log": True})
        self.assertEqual(logged.call_count, 2)
        row, source = logged.call_args.args
        self.assertEqual(source["origin"], "case")
        self.assertEqual(row["stress_run"], events[0]["run_id"])
        self.assertEqual(row["stress_concurrency"], 2)
        self.assertEqual(row["truth"]["case"], "sol002")
        self.assertEqual(row["page_count"], 1)
        self.assertNotIn("extracted", row)

    def test_full_jobs_write_one_row_with_both_passes(self):
        with patch.object(engine, "log_run") as logged, \
             patch.object(app, "extract_fields", return_value=dict(RESULT)), \
             patch.object(engine, "log_extract",
                          side_effect=AssertionError("a full job wrote an extract row")):
            _run({"mode": "full", "concurrency": 1, "jobs": 2, "lock_case": "sol002",
                  "save_log": True})
        self.assertEqual(logged.call_count, 2)
        row = logged.call_args.args[0]
        self.assertEqual(row["extracted"]["tokens"], 10)
        self.assertEqual(row["truth"]["case"], "sol002")


class RunLogColumnTests(unittest.TestCase):
    def test_the_row_carries_the_tag_and_the_filter_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runlog, "LOG_DIR", Path(tmp)), \
             patch.object(runlog, "LOG_PATH", Path(tmp) / "runs.csv"):
            row = runlog.record({"model": "m", "stress_run": "2026-10-15T10:00:00-7",
                                 "stress_concurrency": 8},
                                {"name": "x.pdf", "origin": "case"})
            plain = runlog.record({"model": "m"}, {"name": "y.pdf", "origin": "case"})
            rows = runlog.read(10)
        self.assertEqual(row["stress_run"], "2026-10-15T10:00:00-7")
        self.assertEqual(row["stress_concurrency"], 8)
        self.assertEqual(plain["stress_run"], "")
        self.assertEqual(plain["stress_concurrency"], "")
        kinds = sorted(runlog.FILTER_FIELDS["stress"](r) for r in rows)
        self.assertEqual(kinds, ["normal", "stress"])
        kept = runlog.filter_rows(rows, exclude={"stress": ["stress"]})
        self.assertEqual([r["file"] for r in kept], ["y.pdf"])


if __name__ == "__main__":
    unittest.main()
