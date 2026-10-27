"""A full stress run asks the page images for a Mandatory field the form left
empty, with only the Hybrid step that owns it, and fills the blank."""
import time
import unittest
from unittest.mock import patch

from PIL import Image

import app
import prompts
import stress


def _doc(fields, mandatory, **extra):
    return {"fields": fields, "classified": {"doc_types": ["INVOICE"],
                                             "mandatory": list(mandatory),
                                             "mandatory_items": []}, **extra}


def _page(colour=0):
    page = Image.new("RGB", (8, 8), (colour, colour, colour))
    page._data_uri = app.image_data_uri(page)
    return page


class FillStepsTests(unittest.TestCase):

    def test_each_key_is_asked_by_the_hybrid_step_that_owns_it(self):
        ids, orphans = app._fill_steps(["INVOICE"], ["vat_total", "subtotal", "buyer_tax_id"])
        self.assertEqual(ids, ["totals", "parties"])     # grouped Hybrid steps, once each
        self.assertEqual(orphans, [])

    def test_a_table_cell_is_asked_by_the_table_step(self):
        key = prompts.INCOME_ITEMS_KEY
        table = prompts.steps_for_types(["WHT_CERTIFICATE"], False, mode="hybrid")
        owner = next(s["id"] for s in table if key in s["keys"])
        ids, _ = app._fill_steps(["WHT_CERTIFICATE"], [f"{key}[0].amount_paid"])
        self.assertEqual(ids, [owner])

    def test_a_key_no_step_owns_is_named(self):
        self.assertEqual(app._fill_steps(["INVOICE"], ["made_up"]), ([], ["made_up"]))


class FullFillTests(unittest.TestCase):

    def _run(self, result, answers, pages=None, texts=("text",), fill_missing=True):
        def read(pages, cancel, rec, live, stats_out=None, job=0, case="", kept=None):
            rec.update(ocr_s=1.0, pages=len(pages), ocr_tokens=10)
            for n, page in enumerate(pages, 1):
                kept.append({"number": n, "text": texts[n - 1], "page": page,
                             "detail": "low", "looped": False, "truncated": False,
                             "dry": None})
            return list(texts), False, False

        asked = []

        def fill(pages, form, ids):
            asked.append((pages, form["doc_types"], list(ids), app._image_fed()))
            return answers.pop(0)

        live = app._StressLive(1)
        cfg = {"mode": "full", "concurrency": 1, "extract_mode": "single", "detail": "low",
               "fill_missing": fill_missing}
        spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
        prepared = {"pages": {"sol002": pages or [_page()]}, "sources": {"sol002": "x"}}
        with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)), \
                patch.object(app, "_stress_read", side_effect=read), \
                patch.object(app, "extract_fields", return_value=result) as extract, \
                patch.object(app, "_fill_ask", side_effect=fill):
            rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
            alerts = live.take_alerts()
            with app.app.test_client() as client:
                body = (client.get(f"/api/stress/evidence/{alerts[0]['id']}").get_json()
                        if alerts else None)
        extract.assert_called_once()
        return rec, alerts, body, asked

    def test_an_empty_field_is_filled_from_the_page_images(self):
        result = _doc({"subtotal": "100.00", "vat_total": ""}, ["subtotal", "vat_total"],
                      tokens=5)
        page = _page()
        rec, alerts, body, asked = self._run(
            result, [{"fields": {"subtotal": "999", "vat_total": "7.00"}, "tokens": 4}],
            pages=[page])
        self.assertEqual(asked, [([page], ["INVOICE"], ["totals"], False)])
        self.assertEqual(result["fields"], {"subtotal": "100.00", "vat_total": "7.00"})
        self.assertEqual(result["image_filled"], ["vat_total"])
        self.assertEqual(rec["missing_required"], 0)
        self.assertEqual((rec["fill_jobs"], rec["fill_asked"], rec["fill_filled"]), (1, 1, 1))
        self.assertEqual(rec["extract_tokens"], 9)     # the fill's work counted
        self.assertEqual({a["id"] for a in alerts}, {alerts[0]["id"]})
        self.assertEqual([(a["kind"], a["state"]) for a in alerts],
                         [("missing", "retrying"), ("missing", "resolved")])
        self.assertIn("Hybrid", alerts[0]["message"])
        self.assertIn("vat_total", alerts[1]["message"])
        rows = {r["key"]: r for r in body["fields"][0]["rows"]}
        self.assertEqual(rows["vat_total"]["value"], "7.00")
        self.assertFalse(rows["vat_total"]["missing"])

        report = stress.report([rec], 1.0, "full", 1)
        self.assertEqual((report["extract"]["fill_jobs"], report["extract"]["fill_asked"],
                          report["extract"]["fill_filled"]), (1, 1, 1))
        self.assertEqual(report["extract"]["missing_required_jobs"], 0)

    def test_a_fill_that_finds_nothing_leaves_the_field_flagged(self):
        result = _doc({"vat_total": ""}, ["vat_total"])
        rec, alerts, _, _ = self._run(result, [{"fields": {"vat_total": ""}}])
        self.assertEqual(rec["missing_required"], 1)
        self.assertEqual(rec["fill_filled"], 0)
        self.assertEqual(alerts[-1]["state"], "retry failed")
        self.assertIn("still empty: vat_total", alerts[-1]["message"])

    def test_a_fill_that_fails_says_why_and_keeps_the_form(self):
        result = _doc({"vat_total": ""}, ["vat_total"])
        rec, alerts, _, _ = self._run(result, [{"error": "image input is not supported"}])
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["missing_required"], 1)
        self.assertIn("not supported", alerts[-1]["message"])

    def test_each_document_is_asked_with_its_own_pages(self):
        first, second = _page(10), _page(20)
        result = {"documents": [dict(_doc({"vat_total": "1"}, ["vat_total"]),
                                     document=1, pages=[1]),
                                dict(_doc({"vat_total": ""}, ["vat_total"]),
                                     document=2, pages=[2])],
                  "documents_found": 2}
        rec, alerts, _, asked = self._run(result, [{"fields": {"vat_total": "2"}}],
                                          pages=[first, second], texts=("one", "two"))
        self.assertEqual([pages for pages, *_ in asked], [[second]])
        self.assertEqual(result["documents"][1]["fields"]["vat_total"], "2")
        self.assertEqual(result["documents"][0]["fields"]["vat_total"], "1")
        self.assertEqual(rec["missing_required"], 0)
        self.assertEqual(alerts[-1]["state"], "resolved")
        self.assertIn("doc2.vat_total", alerts[-1]["message"])

    def test_the_toggle_off_flags_the_field_and_asks_nothing(self):
        rec, alerts, _, asked = self._run(_doc({"vat_total": ""}, ["vat_total"]), [],
                                          fill_missing=False)
        self.assertEqual(asked, [])
        self.assertEqual([(a["kind"], a.get("state")) for a in alerts], [("missing", None)])
        self.assertEqual(rec["missing_required"], 1)
        self.assertNotIn("fill_jobs", rec)

    def test_a_whole_form_asks_nothing(self):
        rec, alerts, _, asked = self._run(_doc({"vat_total": "1"}, ["vat_total"]), [])
        self.assertEqual((asked, alerts), ([], []))
        self.assertNotIn("fill_jobs", rec)

    def test_the_real_ask_feeds_the_images_to_the_hybrid_steps(self):
        seen = []

        def agentic(text, status, form, only=None, mode="agentic"):
            seen.append((text, only, mode, app._image_fed()))
            return {"fields": {}}
            yield                                       # a generator, as the real one

        status = {"text_available": True}
        with patch.object(app.backends, "extract_status", return_value=status), \
                patch.object(app, "_extract_agentic", side_effect=agentic):
            app._fill_ask([_page()], {"doc_types": []}, ["totals"])
        text, only, mode, fed = seen[0]
        self.assertEqual((only, mode, fed), (["totals"], "hybrid", True))
        self.assertIn("1 page image(s)", text)


class FillToggleConfigTests(unittest.TestCase):

    def _config(self, **body):
        status = {"available": True, "text_available": True, "model": "m",
                  "url": "http://x", "kind": "ollama"}
        with patch.object(app, "llama_status", return_value=status), \
                patch.object(app.backends, "extract_status", return_value=status):
            return app._stress_config(body)

    def test_on_unless_the_request_turns_it_off(self):
        self.assertTrue(self._config(mode="full")["fill_missing"])
        self.assertTrue(self._config(mode="full", fill_missing=True)["fill_missing"])
        self.assertFalse(self._config(mode="full", fill_missing=False)["fill_missing"])

    def test_only_full_pipeline_fills(self):
        self.assertFalse(self._config(mode="extract", fill_missing=True)["fill_missing"])


if __name__ == "__main__":
    unittest.main()
