"""A stress job's wrong fields are flagged when it is scored, and the score runs live."""

import time
import unittest
from unittest.mock import patch

import app
import fieldscore
import prompts
import stress

TYPES = ["INVOICE"]


def _result(case_id="sol002", change=None):
    """A form scored against the truth: every key right, except `change`."""
    truth = fieldscore.load_truth(case_id)
    keys = list(prompts.fields_for_types(TYPES))
    fields = {k: fieldscore.accepted(v)[0] for k, v in truth["scalars"].items()
              if k in keys}
    fields.update(change or {})
    mandatory = list(prompts.mandatory_for_types(TYPES))
    score = fieldscore.evaluate(case_id, fields, keys=keys, doc_types=TYPES,
                                mandatory=mandatory)
    return {"fields": fields, "field_score": score, "doc_types": TYPES,
            "classified": {"mandatory": mandatory}, "tokens": 1}


def _run(result, live=None):
    live = live or app._StressLive(1)
    cfg = {"mode": "extract", "concurrency": 1, "extract_mode": "single",
           "detail": "low"}
    spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
    prepared = {"truths": {"sol002": "text"}, "sources": {"sol002": "x"}}
    with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)) as evidence, \
            patch.object(app, "extract_fields", return_value=result):
        rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
        alerts = live.take_alerts()
        with app.app.test_client() as client:
            windows = {a["id"]: client.get(f"/api/stress/evidence/{a['id']}").get_json()
                       for a in alerts if a.get("evidence")}
    del evidence
    return rec, alerts, windows, live


class WrongFieldTests(unittest.TestCase):

    def test_a_right_form_flags_nothing(self):
        rec, alerts, _, live = _run(_result())
        self.assertEqual(rec["wrong_fields"], 0)
        self.assertFalse([a for a in alerts if a["kind"] == "wrong"])
        self.assertEqual(live.snapshot()["score"]["field_pooled"], 1.0)

    def test_a_wrong_value_is_flagged_with_its_truth(self):
        result = _result(change={"document_number": "WRONG-123"})
        rec, alerts, windows, live = _run(result)
        wrong = [a for a in alerts if a["kind"] == "wrong"]
        self.assertEqual(len(wrong), 1)
        self.assertEqual(rec["wrong_fields"], 1)
        self.assertIn("document_number", wrong[0]["message"])
        self.assertEqual(wrong[0]["fields"], ["document_number"])
        self.assertEqual(wrong[0]["stage"], "extract")
        rows = {r["key"]: r for r in windows[wrong[0]["id"]]["fields"][0]["rows"]}
        self.assertEqual(rows["document_number"]["wrong"], "wrong")
        self.assertEqual(rows["document_number"]["value"], "WRONG-123")
        self.assertTrue(rows["document_number"]["expected"])
        score = live.snapshot()["score"]
        self.assertEqual((score["wrong_fields"], score["wrong_jobs"]), (1, 1))
        self.assertLess(score["field_pooled"], 1.0)
        report = stress.report([rec], 1.0, "extract", 1)["extract"]
        self.assertEqual((report["wrong_fields"], report["wrong_field_jobs"]), (1, 1))
        self.assertEqual(report["field_pooled"], score["field_pooled"])

    def test_an_empty_required_field_is_flagged_once(self):
        result = _result(change={"document_number": ""})
        _, alerts, _, _ = _run(result)
        kinds = [a["kind"] for a in alerts]
        self.assertEqual(kinds.count("missing"), 1)
        self.assertNotIn("wrong", kinds)

    def test_each_document_of_a_file_is_checked_on_its_own(self):
        one = dict(_result(), document=1)
        two = dict(_result(change={"subtotal": "1.00"}), document=2)
        found = app._wrong_fields({"documents": [one, two], "documents_found": 2})
        self.assertEqual([(n, [r["key"] for r in rows]) for n, rows in found],
                         [(2, ["subtotal"])])
        self.assertIn("doc 2 subtotal", app._wrong_message(found, True))

    def test_who_lost_it_is_named(self):
        result = _result(change={"subtotal": "1.00"})
        result["blame"] = {"fields": {"subtotal": {"blame": "ocr"}}}
        found = app._wrong_fields(result)
        self.assertEqual(found[0][1][0]["blame"], "ocr")
        self.assertIn("subtotal (read)", app._wrong_message(found, False))

    def test_a_long_list_is_cut_at_a_field(self):
        rows = [{"key": f"field_number_{i}", "status": "spurious", "blame": None}
                for i in range(40)]
        message = app._wrong_message([(1, rows)], False)
        self.assertLessEqual(len(message), 230)
        self.assertTrue(message.startswith("40 field(s)"))
        self.assertRegex(message, r"\(invented\); and \d+ more$")

    def test_the_live_score_is_the_reports(self):
        live = app._StressLive(3)
        recs = [_run(_result(), live)[0],
                _run(_result(change={"subtotal": "1.00"}), live)[0]]
        score = live.snapshot()["score"]
        report = stress.report(recs, 1.0, "extract", 1)["extract"]
        self.assertEqual(score["field_accuracy"], report["field_accuracy"])
        self.assertEqual(score["field_pooled"], report["field_pooled"])
        self.assertEqual(score["field_values"], report["field_values"])
        self.assertEqual(score["doc_sets"], 2)


if __name__ == "__main__":
    unittest.main()
