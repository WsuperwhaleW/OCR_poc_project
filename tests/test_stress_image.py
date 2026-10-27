"""The stress test's image mode: no read, the extractor is sent the page images,
and a form with a required field missing is extracted again one Detail up."""
import time
import unittest
from unittest.mock import patch

from PIL import Image

import app
import stress


def _doc(fields, mandatory, **extra):
    return {"fields": fields, "classified": {"mandatory": list(mandatory),
                                             "mandatory_items": []}, **extra}


def _page(detail="low", size=8):
    page = Image.new("RGB", (size, size))
    page._data_uri = app.image_data_uri(page)
    page._stress_origin = {"case": "sol002", "source": "benchmark", "index": 0,
                           "detail": detail}
    return page


class AttachImagesTests(unittest.TestCase):

    def _payload(self):
        return {"messages": [{"role": "system", "content": "sys"},
                             {"role": "user", "content": "the question"}]}

    def test_nothing_changes_outside_an_image_fed_block(self):
        payload = self._payload()
        self.assertIs(app._with_images("http://x/v1/chat/completions", payload, {}), payload)

    def test_openai_shape_gets_image_parts_first(self):
        payload = self._payload()
        with app._feeding_images(["data:image/png;base64,AAA", "data:image/png;base64,BBB"]):
            out = app._with_images("http://x/v1/chat/completions", payload,
                                   {"kind": "llama.cpp"})
        content = out["messages"][1]["content"]
        self.assertEqual([p["type"] for p in content], ["image_url", "image_url", "text"])
        self.assertEqual(content[2]["text"], "the question")
        self.assertEqual(payload["messages"][1]["content"], "the question")  # untouched
        self.assertNotIn("mm_processor_kwargs", out)

    def test_ollama_native_chat_gets_base64_beside_the_text(self):
        with app._feeding_images(["data:image/png;base64,AAA"]):
            out = app._with_images("http://x/api/chat", self._payload(), {"kind": "ollama"})
        self.assertEqual(out["messages"][1]["images"], ["AAA"])
        self.assertEqual(out["messages"][1]["content"], "the question")

    def test_vllm_lifts_its_image_cap(self):
        with app._feeding_images(["data:image/png;base64,AAA"]):
            out = app._with_images("http://x/v1/chat/completions", self._payload(),
                                   {"kind": "vllm"})
        self.assertIn("max_pixels", out["mm_processor_kwargs"])

    def test_the_images_follow_an_agentic_step_onto_its_thread(self):
        seen = []
        with app._feeding_images(["data:image/png;base64,AAA"]):
            run = app._carry_progress(lambda: seen.append(app._image_fed()))
        import threading
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(seen, [True])
        self.assertFalse(app._image_fed())

    def test_no_value_is_quoted_back_as_ungrounded(self):
        source = app.grounding.Source("nothing here")
        self.assertTrue(app._ungrounded_in({"buyer_name": "ACME"}, source))
        with app._feeding_images(["data:image/png;base64,AAA"]):
            self.assertEqual(app._ungrounded_in({"buyer_name": "ACME"}, source), [])

    def test_grounding_is_said_to_be_unchecked(self):
        result = {"grounding": {"grounded_ratio": 0.0, "checked": 5, "missing": ["a"]},
                  "documents": [{"grounding": {"grounded_ratio": 0.0, "checked": 2}}]}
        app._ungrounded_result(result)
        self.assertIsNone(result["grounding"]["grounded_ratio"])
        self.assertEqual(result["grounding"]["missing"], ["a"])
        self.assertIn("skipped", result["documents"][0]["grounding"])
        self.assertTrue(result["image_fed"])


class ImageJobTests(unittest.TestCase):

    def _run(self, results, pages=None, retry_page=None, loop_retry=True):
        live = app._StressLive(1)
        cfg = {"mode": "image", "concurrency": 1, "extract_mode": "single",
               "detail": "low", "extras": False}
        spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter()}
        prepared = {"pages": {"sol002": pages or [_page()]}, "sources": {"sol002": "x"}}
        sent = []

        def extract(text, mode, case_id, **kwargs):
            sent.append((text, app._image_fed(), kwargs.get("pages"), kwargs.get("extras")))
            return results.pop(0)

        with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)), \
                patch.object(app, "extract_fields", side_effect=extract), \
                patch.object(app, "LOOP_RETRY", loop_retry), \
                patch.object(app, "_stress_retry_page",
                             side_effect=lambda p: retry_page):
            rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
            alerts = live.take_alerts()
            bodies = {}
            with app.app.test_client() as client:
                for alert in alerts:
                    bodies[alert["id"]] = client.get(
                        f"/api/stress/evidence/{alert['id']}").get_json()
        return rec, alerts, bodies, sent

    def test_a_whole_form_raises_no_flag_and_is_scored(self):
        rec, alerts, _, sent = self._run([_doc({"issue_date": "x"}, ["issue_date"],
                                                tokens=7)])
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(alerts, [])
        self.assertEqual(rec["pages"], 1)
        self.assertEqual(rec["extract_tokens"], 7)
        text, fed, pages, extras = sent[0]
        self.assertTrue(fed)
        self.assertIn("1 page image(s)", text)
        self.assertEqual(pages, [text])           # one document: nothing to split on
        self.assertFalse(extras)

    def test_a_missing_field_is_extracted_again_one_detail_up(self):
        bigger = _page("medium", 16)
        rec, alerts, bodies, sent = self._run(
            [_doc({"vat_total": ""}, ["vat_total"], tokens=5),
             _doc({"vat_total": "7.00"}, ["vat_total"], tokens=6)], retry_page=bigger)
        self.assertEqual(len(sent), 2)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual((rec["image_retries"], rec["image_recovered"]), (1, 1))
        self.assertEqual(rec["missing_required"], 0)
        self.assertEqual(rec["extract_tokens"], 11)   # both attempts' work
        # One row, raised retrying and marked resolved -- no second flag.
        self.assertEqual({a["id"] for a in alerts}, {alerts[0]["id"]})
        self.assertEqual([(a["kind"], a["state"]) for a in alerts],
                         [("missing", "retrying"), ("missing", "resolved")])
        self.assertEqual(alerts[0]["fields"], ["vat_total"])
        body = bodies[alerts[0]["id"]]
        self.assertEqual([(a["detail"], a["image"]) for a in body["attempts"]],
                         [("low", True), ("medium", True)])
        self.assertEqual(body["fields"][0]["rows"][0]["value"], "7.00")

        report = stress.report([rec], 1.0, "image", 1)
        self.assertEqual(report["image"], {"pages": 1, "pages_per_second": 1.0,
                                           "retries": 1, "recovered": 1})
        self.assertEqual(report["extract"]["missing_required_jobs"], 0)

    def test_a_retry_that_does_no_better_keeps_the_first_and_says_so(self):
        rec, alerts, _, _ = self._run(
            [_doc({"a": "x", "b": ""}, ["a", "b"]), _doc({"a": "", "b": ""}, ["a", "b"])],
            retry_page=_page("medium", 16))
        self.assertEqual(alerts[-1]["state"], "retry failed")
        self.assertIn("kept the low form", alerts[-1]["message"])
        self.assertEqual(rec["missing_required"], 1)
        self.assertEqual(rec["image_detail"], "low")

    def test_at_the_top_detail_it_is_flagged_without_a_retry(self):
        rec, alerts, _, sent = self._run([_doc({"a": ""}, ["a"])], retry_page=None)
        self.assertEqual(len(sent), 1)
        self.assertEqual([(a["kind"], a["state"]) for a in alerts],
                         [("missing", "no retry")])
        self.assertEqual(rec["missing_required"], 1)

    def test_a_server_error_is_flagged_with_the_page_and_not_retried(self):
        rec, alerts, bodies, sent = self._run(
            [{"error": "llama.cpp HTTP 500: image input is not supported"}],
            retry_page=_page("medium", 16))
        self.assertEqual(len(sent), 1)
        self.assertEqual(rec["status"], "error")
        self.assertEqual([(a["kind"], a["stage"]) for a in alerts], [("error", "extract")])
        body = bodies[alerts[0]["id"]]
        self.assertTrue(body["attempts"][0]["image"])
        self.assertIn("not supported", body["fields"][0]["error"])

    def test_a_looped_reply_is_retried_and_one_row_says_how_it_went(self):
        looped = _doc({}, ["a"], error="The model repeated itself.", looped=True)
        rec, alerts, _, _ = self._run([looped, dict(looped)],
                                      retry_page=_page("medium", 16))
        self.assertEqual(rec["status"], "error")
        self.assertEqual({a["id"] for a in alerts}, {alerts[0]["id"]})
        self.assertEqual([a["state"] for a in alerts], ["retrying", "retry failed"])


class ImageConfigTests(unittest.TestCase):

    def _status(self, vision):
        return {"text_available": True, "model": "m", "url": "http://x", "kind": "ollama",
                "models": [{"name": "m", "vision": vision}]}

    def test_a_model_that_says_it_has_no_vision_is_refused(self):
        with patch.object(app.backends, "extract_status", return_value=self._status(False)):
            with self.assertRaisesRegex(ValueError, "no vision"):
                app._stress_config({"mode": "image"})

    def test_a_model_that_does_not_say_is_tried(self):
        with patch.object(app.backends, "extract_status", return_value=self._status(None)):
            cfg = app._stress_config({"mode": "image", "extract_mode": "agentic",
                                      "extras": False, "detail": "low",
                                      "save_log": True, "save_details": True})
        self.assertEqual((cfg["mode"], cfg["extract_mode"], cfg["extras"], cfg["detail"]),
                         ("image", "agentic", False, "low"))
        self.assertIsNone(cfg["extractor_vision"])
        self.assertFalse(cfg["save_log"])
        self.assertTrue(cfg["save_details"])


if __name__ == "__main__":
    unittest.main()
