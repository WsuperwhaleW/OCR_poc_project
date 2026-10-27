"""A stress run's flags in two parts: the read's, and the extraction's missing
required fields -- which a looped read still goes on to."""
import time
import unittest
from unittest.mock import patch

from PIL import Image

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


class FieldsEvidenceTests(unittest.TestCase):

    def test_each_asked_key_with_its_value_and_marks(self):
        key = prompts.INCOME_ITEMS_KEY
        result = {"fields": {"issue_date": "2026-01-01", key: [{"amount_paid": ""}]},
                  "fields_asked": ["issue_date", "buyer_name"], "items_asked": ["amount_paid"],
                  "classified": {"mandatory": ["issue_date"],
                                 "mandatory_items": ["amount_paid"]}}
        docs = app._fields_evidence(result, [(1, [f"{key}[0].amount_paid"])])
        self.assertEqual([(r["key"], r["required"], r["missing"]) for r in docs[0]["rows"]],
                         [("issue_date", True, False), ("buyer_name", False, False),
                          (key, True, False), (f"{key}[0].amount_paid", True, True)])
        self.assertEqual(docs[0]["rows"][2]["value"], '[{"amount_paid": ""}]')

    def test_an_extract_only_flag_has_text_and_no_image(self):
        live = app._StressLive(1)
        result = _doc({"a": ""}, ["a"])
        spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
        cfg = {"mode": "extract", "concurrency": 1, "extract_mode": "single"}
        prepared = {"truths": {"sol002": "the transcript"}, "sources": {"sol002": "x"}}
        with patch.object(app, "extract_fields", return_value=result):
            app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
        alert = next(a for a in live.take_alerts() if a["kind"] == "missing")
        with app.app.test_client() as client:
            body = client.get(f"/api/stress/evidence/{alert['id']}").get_json()
            image = client.get(f"/api/stress/evidence/{alert['id']}/0.png")
        self.assertEqual([(a["text"], a["image"]) for a in body["attempts"]],
                         [("the transcript", False)])
        self.assertEqual(image.status_code, 404)


class FullPipelineLoopTests(unittest.TestCase):

    def test_a_looped_read_is_still_extracted_and_flagged_in_both_parts(self):
        def read(pages, cancel, rec, live, stats_out=None, job=0, case="", kept=None):
            live.flag("looped", job, case, "looped", stage="ocr", page=1,
                      state="retry failed")
            rec.update(ocr_s=1.0, pages=1, ocr_tokens=10)
            kept.append({"number": 1, "text": "text", "page": "png", "detail": "low",
                         "looped": True, "truncated": False, "dry": None})
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

        # The row opens the Compare window with the form beside the page.
        with app.app.test_client() as client:
            body = client.get(f"/api/stress/evidence/{missing['id']}").get_json()
        self.assertEqual([(a["page"], a["image"], a["looped"]) for a in body["attempts"]],
                         [(1, True, True)])
        rows = {r["key"]: r for r in body["fields"][0]["rows"]}
        self.assertEqual(rows["vat_total"], {"key": "vat_total", "value": "",
                                             "required": True, "missing": True})
        self.assertFalse(rows["issue_date"]["missing"])

        report = stress.report([rec], 1.0, "full", 1)
        self.assertEqual(report["extract"]["missing_required_jobs"], 1)
        self.assertEqual(report["extract"]["missing_required_fields"], 1)


class ExtractionFlagWindowTests(unittest.TestCase):
    """Every extraction flag opens the page image beside the form, not only
    missing required."""

    def _run(self, **extract):
        def read(pages, cancel, rec, live, stats_out=None, job=0, case="", kept=None):
            rec.update(ocr_s=1.0, pages=1, ocr_tokens=10)
            kept.append({"number": 1, "text": "text", "page": Image.new("RGB", (4, 4)),
                         "detail": "low", "looped": False, "truncated": False,
                         "dry": None})
            return ["text"], False, False

        live = app._StressLive(1)
        cfg = {"mode": "full", "concurrency": 1, "extract_mode": "single",
               "detail": "low"}
        spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
        prepared = {"pages": {"sol002": ["page"]}, "sources": {"sol002": "x"}}
        with patch.object(app, "_stress_read", side_effect=read), \
                patch.object(app, "extract_fields", **extract):
            rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
        alerts = [a for a in live.take_alerts() if a["stage"] == "extract"]
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0]["evidence"])
        with app.app.test_client() as client:
            body = client.get(f"/api/stress/evidence/{alerts[0]['id']}").get_json()
            image = client.get(f"/api/stress/evidence/{alerts[0]['id']}/0.png")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/png")
        return rec, alerts[0], body

    def test_a_looped_extraction_reply_shows_its_empty_form(self):
        result = _doc({"vat_total": ""}, ["vat_total"])
        result.update(error="The model repeated itself while extracting.", looped=True)
        rec, alert, body = self._run(return_value=result)
        self.assertEqual(rec["status"], "error")
        self.assertEqual(alert["kind"], "looped")
        self.assertEqual([(r["key"], r["missing"]) for r in body["fields"][0]["rows"]],
                         [("vat_total", True)])

    def test_an_extraction_that_raised_still_shows_the_page(self):
        rec, alert, body = self._run(side_effect=ValueError("server said no"))
        self.assertEqual(alert["kind"], "error")
        self.assertEqual(body["fields"][0]["rows"], [])
        self.assertIn("server said no", body["fields"][0]["error"])
        self.assertEqual(body["attempts"][0]["text"], "text")


if __name__ == "__main__":
    unittest.main()
