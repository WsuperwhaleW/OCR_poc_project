"""The stress report counts the document sets each extracted file held."""

import time
import unittest
from unittest.mock import patch

import app
import scoring
import stress


def _run(case_id, result):
    live = app._StressLive(1)
    cfg = {"mode": "extract", "concurrency": 1, "extract_mode": "single",
           "detail": "low"}
    spec = {"index": 0, "case": case_id, "submitted": time.perf_counter()}
    prepared = {"truths": {case_id: "text"}, "sources": {case_id: "x"}}
    with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)), \
            patch.object(app, "extract_fields", return_value=result):
        return app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)


class DocSetTests(unittest.TestCase):

    def test_a_single_document_counts_one(self):
        got = app._stress_doc_sets({"fields": {}}, {"documents": []})
        self.assertEqual(got, {"doc_sets": 1, "doc_sets_failed": 0,
                               "doc_sets_expected": 1})

    def test_a_split_file_counts_its_documents_and_failures(self):
        got = app._stress_doc_sets({"documents_found": 7, "documents_failed": 2},
                                   {"documents": [{}] * 7})
        self.assertEqual(got, {"doc_sets": 7, "doc_sets_failed": 2,
                               "doc_sets_expected": 7})

    def test_a_realdata_file_has_nothing_to_compare_with(self):
        got = app._stress_doc_sets({"error": "boom"}, None)
        self.assertEqual(got, {"doc_sets": 1, "doc_sets_failed": 1,
                               "doc_sets_expected": None})

    def test_a_pack_job_reaches_the_report(self):
        expected = len(scoring.cases_index()["sol015"].get("documents") or [])
        self.assertGreater(expected, 1)
        result = {"documents": [], "documents_found": expected,
                  "documents_failed": 1, "tokens": 3}
        rec = _run("sol015", result)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual((rec["doc_sets"], rec["doc_sets_failed"],
                          rec["doc_sets_expected"]), (expected, 1, expected))
        one = _run("sol002", {"fields": {}, "tokens": 1})
        docs = stress.report([rec, one], 1.0, "extract", 1)["extract"]["documents"]
        self.assertEqual(docs["files"], 2)
        self.assertEqual(docs["found"], expected + 1)
        self.assertEqual(docs["extracted"], expected)
        self.assertEqual(docs["multi_files"], 1)
        self.assertEqual((docs["split_matched"], docs["split_checked"]), (2, 2))

    def test_a_file_whose_every_document_failed_still_counts(self):
        rec = _run("sol015", {"documents": [], "documents_found": 7,
                              "documents_failed": 7, "error": "every document failed"})
        self.assertEqual(rec["status"], "error")
        docs = stress.report([rec], 1.0, "extract", 1)["extract"]["documents"]
        self.assertEqual((docs["found"], docs["extracted"], docs["failed"]), (7, 0, 7))


if __name__ == "__main__":
    unittest.main()
