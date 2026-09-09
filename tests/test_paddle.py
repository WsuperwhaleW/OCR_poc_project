import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app
import paddle_runtime
import paddle_worker
import runlog
import scoring


class PaddleDecodeTests(unittest.TestCase):
    def test_probe_waits_past_heartbeat_for_ready_metadata(self):
        worker = paddle_runtime.PaddleWorker()
        events = iter([
            {"event": "heartbeat"},
            {"event": "ready", "modelInfo": {"paddleOcrVersion": "3.7.0"}},
        ])
        with patch.object(worker, "_events", return_value=events):
            self.assertEqual(worker.probe()["paddleOcrVersion"], "3.7.0")

    def test_unicode_confidence_and_normalized_polygon(self):
        lines = paddle_worker.decode_result({
            "rec_texts": ["ภาษาไทย ๒๕๖๘", "ABC 123"],
            "rec_scores": [0.91, 0.75],
            "rec_polys": [
                [[10, 20], [90, 20], [90, 40], [10, 40]],
                [[0, 50], [50, 50], [50, 70], [0, 70]],
            ],
        }, 100, 200)
        self.assertEqual([line["text"] for line in lines],
                         ["ภาษาไทย ๒๕๖๘", "ABC 123"])
        self.assertEqual(lines[0]["confidence"], 0.91)
        self.assertEqual(lines[0]["rect"], [0.1, 0.1, 0.9, 0.2])
        self.assertEqual(lines[1]["bbox"], [0.0, 50.0, 50.0, 70.0])

    def test_malformed_parallel_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            paddle_worker.decode_result({
                "rec_texts": ["x"], "rec_scores": [], "rec_polys": []
            }, 100, 100)

    def test_line_and_page_order_are_preserved(self):
        image = Image.new("RGB", (200, 100))
        event = {"seconds": 1.25, "lines": [
            {"text": "บรรทัดหนึ่ง", "confidence": .8,
             "bbox": [1, 2, 3, 4], "polygon": [], "rect": []},
            {"text": "second", "confidence": .6,
             "bbox": [5, 6, 7, 8], "polygon": [], "rect": []},
        ]}
        text, stats = app.paddle_page_result(event, image)
        self.assertEqual(text, "บรรทัดหนึ่ง\nsecond")
        self.assertEqual([b["text"] for b in stats["layout"]],
                         ["บรรทัดหนึ่ง", "second"])
        self.assertAlmostEqual(stats["mean_confidence"], .7)
        self.assertEqual(app.join_page_texts(["first", "second"]),
                         "--- page 1 ---\nfirst\n\n--- page 2 ---\nsecond")


class PaddleAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.page = Image.new("RGB", (200, 100), "white")
        self.source = {"name": "sample.pdf", "origin": "upload", "size_bytes": 12}
        self.stats = [{
            "resolution": "200x100", "seconds": .5,
            "model": "th_PP-OCRv5_mobile_rec", "backend": "paddleocr",
            "url": "local", "line_count": 1, "mean_confidence": .93,
            "layout": [], "raw": "[]", "ocr_profile": "paddle",
        }]
        self.model_info = {
            "recognizer": "th_PP-OCRv5_mobile_rec",
            "detector": "PP-OCRv5_mobile_det", "device": "cpu",
            "paddleOcrVersion": "3.7.0", "paddlePaddleVersion": "3.3.1",
        }

    def test_blocking_endpoint_uses_paddle_and_existing_scorer(self):
        case = next(iter(scoring.cases_index().values()))
        truth_text = case["ground_truth"].read_text("utf-8")
        direct = scoring.evaluate(case, truth_text)
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job1", case, self.source)), \
             patch("app.consume_paddle_pages", return_value=(
                [truth_text], self.stats, self.model_info)), \
             patch("app.log_run") as logged, patch.object(app, "EXTRACT", False):
            response = self.client.post("/api/ocr", data={"reader": "paddle"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["backend"], "paddleocr")
        self.assertEqual(body["truth"]["char_accuracy"], direct["char_accuracy"])
        self.assertEqual(body["truth"]["thai_accuracy"], direct["thai_accuracy"])
        self.assertEqual(body["truth"]["latin_accuracy"], direct["latin_accuracy"])
        self.assertEqual(body["truth"]["digit_accuracy"], direct["digit_accuracy"])
        logged.assert_called_once()

    def test_server_reader_path_remains_the_default(self):
        stats = {"new_tokens": 2, "decode_seconds": .1,
                 "prefill_seconds": .2, "seconds": .3,
                 "tokens_per_second": 20, "backend": "llama.cpp",
                 "model": "test-reader", "url": "local", "resolution": "200x100"}
        with patch("app.prepare", return_value=(
                [self.page], "medium", "server-job", None, self.source)), \
             patch("app.read_page", side_effect=lambda _page, target:
                   (target.update(stats) or "server transcript")) as read, \
             patch("app.consume_paddle_pages") as paddle, \
             patch("app.log_run"), patch.object(app, "EXTRACT", False):
            response = self.client.post("/api/ocr", data={"reader": "server"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "server transcript")
        read.assert_called_once()
        paddle.assert_not_called()

    def test_paddle_transcript_can_feed_existing_field_extractor(self):
        extracted = {"fields": {"document_number": "INV-1"}, "seconds": 1.0}
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job2", None, self.source)), \
             patch("app.consume_paddle_pages", return_value=(
                ["Invoice INV-1"], self.stats, self.model_info)), \
             patch("app.extract_fields", return_value=extracted) as extract, \
             patch("app.log_run"):
            response = self.client.post("/api/ocr", data={"reader": "paddle"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["extracted"], extracted)
        # `pages` arrived with document segmentation (2026-09-08): pass 2 is
        # given the page list so a multi-document file can be split into one
        # form per document. A one-page read is a one-element list.
        extract.assert_called_once_with("Invoice INV-1", case_id=None,
                                        pages=["Invoice INV-1"])

    def test_worker_failure_is_a_clear_503(self):
        with patch("app.prepare", return_value=(
                [self.page], "medium", "job3", None, self.source)), \
             patch("app.consume_paddle_pages",
                   side_effect=paddle_runtime.PaddleError("model load failed")), \
             patch("app.log_run") as logged:
            response = self.client.post("/api/ocr", data={"reader": "paddle"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("model load failed", response.get_json()["error"])
        logged.assert_called_once()

    def test_stream_reuses_existing_page_and_layout_contract(self):
        worker_events = iter([
            {"event": "loading", "message": "Loading PaddleOCR models"},
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
             patch("app.paddle_events", return_value=worker_events), \
             patch("app.log_run"), patch.object(app, "EXTRACT", False):
            response = self.client.post("/api/ocr/stream",
                                        data={"reader": "paddle"})
            events = [json.loads(line)
                      for line in response.get_data(as_text=True).splitlines()]
        self.assertEqual([event["event"] for event in events],
                         ["progress", "page", "token", "page_done", "done", "logged"])
        self.assertEqual(events[3]["layout"][0]["text"], "ไทย")
        self.assertEqual(events[4]["text"], "ไทย")

    def test_reader_api_reports_probe_result(self):
        fake = {"id": "paddle", "label": "Local PaddleOCR", "available": True,
                "recognizer": "thai", "device": "cpu", "reason": ""}
        with patch("app.paddle_runtime.status", return_value=fake), \
             patch("app.llama_status", return_value={"available": True}):
            response = self.client.post("/api/ocr/reader",
                                        json={"reader": "paddle", "probe": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["selected"], "paddle")
        app.set_reader("server")


class PersistenceTests(unittest.TestCase):
    def test_paddle_metadata_is_written_to_shared_csv(self):
        path = Path("logs") / "test-paddle-runs.csv"
        path.unlink(missing_ok=True)
        self.addCleanup(path.unlink, missing_ok=True)
        with patch.object(runlog, "LOG_DIR", path.parent), \
             patch.object(runlog, "LOG_PATH", path), \
             patch.object(runlog, "ALLOW_PATCHED_TRANSPORT", True):
            row = runlog.record({
                "backend": "paddleocr", "model": "th_PP-OCRv5_mobile_rec",
                "seconds": 2.5, "ocr_lines": 7, "ocr_confidence": .88,
                "ocr_device": "cpu", "ocr_version": "3.7.0",
                "paddle_version": "3.3.1",
                "ocr_detector": "PP-OCRv5_mobile_det",
            }, {"name": "x.pdf", "origin": "upload"})
            self.assertEqual(row["backend"], "paddleocr")
            with path.open(encoding="utf-8-sig") as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved["ocr_lines"], "7")
            self.assertEqual(saved["ocr_confidence"], "0.88")
            self.assertEqual(saved["paddle_version"], "3.3.1")
            self.assertEqual(saved["ocr_detector"], "PP-OCRv5_mobile_det")


if __name__ == "__main__":
    unittest.main()
