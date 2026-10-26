"""A stress run's flags in two parts: the read's, and the extraction's missing
required fields -- which a looped read still goes on to."""
import time
import unittest
from unittest.mock import patch

import app
import prompts
import stress


def _doc(fields, mandatory, document=None, mandatory_items=(), checks=None):
    doc = {"fields": fields,
           "classified": {"mandatory": list(mandatory),
                          "mandatory_items": list(mandatory_items)}}
    if document is not None:
        doc["document"] = document
    if checks is not None:
        doc["validation"] = {"checks": checks}
    return doc


class MissingRequiredTests(unittest.TestCase):

    def test_blank_mandatory_scalars_are_named(self):
        result = _doc({"issue_date": "2026-01-01", "vat_total": "", "buyer_name": None},
                      ["issue_date", "vat_total", "buyer_name"])
        self.assertEqual(app._missing_required(result),
                         [(1, ["vat_total", "buyer_name"])])

    def test_a_type_no_requirement_covers_demands_nothing(self):
        self.assertEqual(app._missing_required(_doc({}, [])), [])

    def test_each_document_of_a_split_file_is_its_own(self):
        result = {"documents": [_doc({"a": "x"}, ["a"], document=1),
                                _doc({"a": ""}, ["a"], document=2)]}
        self.assertEqual(app._missing_required(result), [(2, ["a"])])

    def test_table_cells_come_from_validation(self):
        key = prompts.INCOME_ITEMS_KEY
        checks = [{"field": f"{key}[0].amount_paid", "state": "absent", "mandatory": True},
                  {"field": f"{key}[0].wht_amount", "state": "absent"},
                  {"field": "issue_date", "state": "absent", "mandatory": True}]
        result = _doc({"issue_date": "x"}, ["issue_date"],
                      mandatory_items=["amount_paid"], checks=checks)
        self.assertEqual(app._missing_required(result),
                         [(1, [f"{key}[0].amount_paid"])])
        # Nothing extracted at all: no validation, so the table itself is missing.
        empty = _doc({}, [], mandatory_items=["amount_paid"])
        self.assertEqual(app._missing_required(empty), [(1, [key])])


class FullPipelineLoopTests(unittest.TestCase):

    def test_a_looped_read_is_still_extracted_and_flagged_in_both_parts(self):
        def read(pages, cancel, rec, live, stats_out=None, job=0, case=""):
            live.flag("looped", job, case, "looped", stage="ocr", page=1,
                      state="retry failed")
            rec.update(ocr_s=1.0, pages=1, ocr_tokens=10)
            return ["text"], True, False

        result = _doc({"issue_date": "2026-01-01", "vat_total": ""},
                      ["issue_date", "vat_total"])
        result["tokens"] = 5
        extract = patch.object(app, "extract_fields", return_value=result)
        live = app._StressLive(1)
        cfg = {"mode": "full", "concurrency": 1, "extract_mode": "single",
               "detail": "low"}
        spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
        prepared = {"pages": {"sol002": ["page"]}, "sources": {"sol002": "x"}}
        with patch.object(app, "_stress_read", side_effect=read), extract as called:
            rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
        called.assert_called_once()                  # the loop did not stop it
        self.assertEqual(rec["status"], "looped")
        self.assertEqual(rec["missing_required"], 1)
        alerts = live.take_alerts()
        self.assertEqual([(a["kind"], a["stage"]) for a in alerts],
                         [("looped", "ocr"), ("missing", "extract")])
        missing = alerts[1]
        self.assertEqual(missing["fields"], ["vat_total"])
        self.assertIn("vat_total", missing["message"])
        self.assertIn("the read looped", missing["message"])

        report = stress.report([rec], 1.0, "full", 1)
        self.assertEqual(report["extract"]["missing_required_jobs"], 1)
        self.assertEqual(report["extract"]["missing_required_fields"], 1)


if __name__ == "__main__":
    unittest.main()
