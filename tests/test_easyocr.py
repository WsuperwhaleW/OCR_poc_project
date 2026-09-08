import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app
import easy_runtime
import easy_worker
import jobs
import runlog
import scoring


class EasyDecodeTests(unittest.TestCase):
    def test_probe_waits_past_heartbeat_for_ready_metadata(self):
        worker = easy_runtime.EasyWorker()
        events = iter([
            {"event": "heartbeat"},
            {"event": "ready", "modelInfo": {"easyOcrVersion": "1.7.2"}},
        ])
        with patch.object(worker, "_events", return_value=events):
            self.assertEqual(worker.probe()["easyOcrVersion"], "1.7.2")

    def test_unicode_confidence_and_normalized_polygon(self):
        lines = easy_worker.decode_result([
            ([[10, 20], [90, 20], [90, 40], [10, 40]],
             "ภาษาไทย ๒๕๖๘", 0.91),
            ([[0, 50], [50, 50], [50, 70], [0, 70]], "ABC 123", 0.75),
        ], 100, 200)
        self.assertEqual([line["text"] for line in lines],
                         ["ภาษาไทย ๒๕๖๘", "ABC 123"])
        self.assertEqual(lines[0]["confidence"], 0.91)
        self.assertEqual(lines[0]["rect"], [0.1, 0.1, 0.9, 0.2])
        self.assertEqual(lines[1]["bbox"], [0.0, 50.0, 50.0, 70.0])

    def test_invalid_confidence_or_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            easy_worker.decode_result([
                ([[0, 0], [1, 0], [1, 1]], "x", 1.2)
            ], 100, 100)

    def test_line_and_page_order_are_preserved(self):
        image = Image.new("RGB", (200, 100))
        event = {"seconds": 1.25, "lines": [
            {"text": "บรรทัดหนึ่ง", "confidence": .8,
             "bbox": [1, 2, 3, 4], "polygon": [], "rect": []},
            {"text": "second", "confidence": .6,
             "bbox": [5, 6, 7, 8], "polygon": [], "rect": []},
        ]}
        text, stats = app.easy_page_result(event, image)
        self.assertEqual(text, "บรรทัดหนึ่ง\nsecond")
        self.assertEqual([box["text"] for box in stats["layout"]],
                         ["บรรทัดหนึ่ง", "second"])
        self.assertAlmostEqual(stats["mean_confidence"], .7)
        self.assertEqual(app.join_page_texts(["first", "second"]),
                         "--- page 1 ---\nfirst\n\n--- page 2 ---\nsecond")


class EasyAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.addCleanup(app.set_reader, "server")
        self.page = Image.new("RGB", (200, 100), "white")
        self.source = {"name": "sample.pdf", "origin": "upload",
                       "size_bytes": 12}
        self.stats = [{
            "resolution": "200x100", "seconds": .5,
            "model": "th+en", "backend": "easyocr", "url": "local",
            "line_count": 1, "mean_confidence": .93,
            "layout": [], "raw": "[]", "ocr_profile": "easyocr",
        }]
        self.model_info = {
            "recognizer": "th+en", "detector": "CRAFT", "device": "cpu",
            "languages": ["th", "en"], "easyOcrVersion": "1.7.2",
            "torchVersion": "2.14.0+cpu",
        }

    def test_blocking_endpoint_uses_easyocr_and_existing_scorer(self):
        case = next(iter(scoring.cases_index().values()))
        truth_text = case["ground_truth"].read_text("utf-8")
        direct = scoring.evaluate(case, truth_text)
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job1", case, self.source)), \
             patch("app.consume_easy_pages", return_value=(
                [truth_text], self.stats, self.model_info)), \
             patch("app.log_run") as logged, patch.object(app, "EXTRACT", False):
            response = self.client.post("/api/ocr", data={"reader": "easyocr"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["backend"], "easyocr")
        for metric in ("char_accuracy", "thai_accuracy", "latin_accuracy",
                       "digit_accuracy", "word_accuracy",
                       "char_accuracy_no_marks"):
            self.assertEqual(body["truth"][metric], direct[metric])
        logged.assert_called_once()

    def test_easyocr_transcript_can_feed_existing_field_extractor(self):
        extracted = {"fields": {"document_number": "INV-1"}, "seconds": 1.0}
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job2", None, self.source)), \
             patch("app.consume_easy_pages", return_value=(
                ["Invoice INV-1"], self.stats, self.model_info)), \
             patch("app.extract_fields", return_value=extracted) as extract, \
             patch("app.log_run"):
            response = self.client.post("/api/ocr", data={"reader": "easyocr"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["extracted"], extracted)
        extract.assert_called_once_with("Invoice INV-1", case_id=None)

    def test_worker_failure_is_a_clear_503(self):
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job3", None, self.source)), \
             patch("app.consume_easy_pages",
                   side_effect=easy_runtime.EasyError("model load failed")), \
             patch("app.log_run"):
            response = self.client.post("/api/ocr", data={"reader": "easyocr"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("model load failed", response.get_json()["error"])

    def test_stream_reuses_page_and_layout_contract(self):
        worker_events = iter([
            {"event": "loading", "message": "Loading EasyOCR models"},
            {"event": "page_start", "page": 1, "total": 1},
            {"event": "page_result", "page": 1, "seconds": .4, "lines": [{
                "text": "ไทย", "confidence": .9, "bbox": [1, 2, 20, 12],
                "polygon": [[.01, .02], [.1, .02], [.1, .12], [.01, .12]],
                "rect": [.01, .02, .1, .12],
            }]},
            {"event": "done", "modelInfo": self.model_info},
        ])
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job4", None, self.source)), \
             patch("app.easy_events", return_value=worker_events), \
             patch("app.log_run"), patch.object(app, "EXTRACT", False):
            response = self.client.post("/api/ocr/stream",
                                        data={"reader": "easyocr"})
            events = [json.loads(line)
                      for line in response.get_data(as_text=True).splitlines()]
        self.assertEqual([event["event"] for event in events],
                         ["progress", "page", "token", "page_done", "done", "logged"])
        self.assertEqual(events[3]["layout"][0]["text"], "ไทย")
        self.assertEqual(events[4]["reader"], "easyocr")

    def test_reader_api_probes_only_selected_local_reader(self):
        fake = {"id": "easyocr", "label": "Local EasyOCR", "available": True,
                "recognizer": "th+en", "device": "cpu", "reason": ""}
        with patch("app.easy_runtime.status", return_value=fake) as easy_status, \
             patch("app.paddle_runtime.status") as paddle_status, \
             patch("app.llama_status", return_value={"available": True}):
            paddle_status.return_value = {"id": "paddle", "available": True}
            response = self.client.post("/api/ocr/reader",
                                        json={"reader": "easyocr", "probe": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["selected"], "easyocr")
        easy_status.assert_called_once_with(probe=True)
        paddle_status.assert_called_once_with(probe=False)

    def test_queue_worker_uses_captured_easyocr_reader(self):
        job = jobs.Job("sample.pdf", "upload", "medium", b"pdf", "easyocr")
        with patch("app.prepare_input", return_value=(
                [self.page], "medium", "queue-job", None)), \
             patch("app.consume_easy_pages", return_value=(
                ["queue text"], self.stats, self.model_info)) as consumed, \
             patch("app.evaluate_if_known", return_value=None), \
             patch("app.log_run"), patch.object(app, "EXTRACT", False):
            result = app.run_job(job)
        self.assertEqual(result["reader"], "easyocr")
        self.assertEqual(result["text"], "queue text")
        self.assertEqual(job.stage, "finished")
        consumed.assert_called_once()


class EasyPersistenceTests(unittest.TestCase):
    def test_queue_captures_reader(self):
        job = jobs.Job("x.pdf", "upload", "medium", b"x", "easyocr")
        self.assertEqual(job.to_dict()["reader"], "easyocr")

    def test_easyocr_metadata_is_written_to_shared_csv(self):
        path = Path("logs") / "test-easy-runs.csv"
        path.unlink(missing_ok=True)
        self.addCleanup(path.unlink, missing_ok=True)
        with patch.object(runlog, "LOG_DIR", path.parent), \
             patch.object(runlog, "LOG_PATH", path), \
             patch.object(runlog, "ALLOW_PATCHED_TRANSPORT", True):
            row = runlog.record({
                "backend": "easyocr", "model": "th+en", "seconds": 2.5,
                "ocr_lines": 7, "ocr_confidence": .88, "ocr_device": "cpu",
                "easyocr_version": "1.7.2", "torch_version": "2.14.0+cpu",
                "ocr_detector": "CRAFT", "ocr_languages": "th,en",
            }, {"name": "x.pdf", "origin": "upload"})
            self.assertEqual(row["backend"], "easyocr")
            with path.open(encoding="utf-8-sig") as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved["easyocr_version"], "1.7.2")
            self.assertEqual(saved["torch_version"], "2.14.0+cpu")
            self.assertEqual(saved["ocr_languages"], "th,en")


if __name__ == "__main__":
    unittest.main()
