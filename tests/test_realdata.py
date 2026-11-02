"""realData/: real files with no solution, listed, served and stress-tested."""
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app
import workspace


STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock",
          "model": "mock-model"}


class RealDataTests(unittest.TestCase):
    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "realData"
        (self.root / "batch1").mkdir(parents=True)
        (self.root / "a.pdf").write_bytes(b"%PDF-1.4 a")
        (self.root / "b.JPEG").write_bytes(b"jpeg b")
        (self.root / "batch1" / "c.png").write_bytes(b"png c")
        (self.root / "notes.txt").write_text("not a document", encoding="utf-8")
        for target, name, kwargs in (
            (app, "REAL_DIR", {"new": self.root}),
            (app, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.app.test_client()

    def cfg(self, **kwargs):
        return app._stress_config({"mode": "ocr", "source": "realdata", **kwargs})

    def test_listing_walks_subfolders_and_skips_unreadable_types(self):
        data = self.client.get("/api/realdata").get_json()
        self.assertEqual([f["name"] for f in data["files"]],
                         ["a.pdf", "b.JPEG", "batch1/c.png"])
        self.assertEqual(data["types"], ["jpg", "pdf", "png"])

    def test_file_is_served_and_traversal_refused(self):
        res = self.client.get("/api/realdata/file?name=batch1/c.png")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data, b"png c")
        for bad in ("../x.pdf", "notes.txt", "missing.pdf", ""):
            with self.subTest(name=bad):
                self.assertEqual(
                    self.client.get("/api/realdata/file?name=" + bad).status_code, 404)

    def test_workspace_preview_reads_large_realdata_without_an_upload_body(self):
        data = b'%PDF-1.4\n' + b'x' * (4 * 1024 * 1024)
        (self.root / 'batch1' / 'large.pdf').write_bytes(data)
        with patch.dict(app.app.config, MAX_CONTENT_LENGTH=1024), \
                patch.object(app, 'load_pages', return_value=([Image.new('RGB', (8, 8))], 1)) as load, \
                patch.object(app.scoring, 'case_for_upload') as match:
            for prefix in ('', '/ocr'):
                with self.subTest(prefix=prefix):
                    with self.client.post(prefix + '/api/preview', data={
                            'realdata':'batch1/large.pdf', 'detail':'low'}) as response:
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.mimetype, 'image/png')
                        self.assertLess(response.request.content_length, 1024)
                    self.assertEqual(load.call_args.args[0], data)
            match.assert_not_called()

    def test_workspace_read_stream_uses_the_same_server_file_and_records_realdata_origin(self):
        data = b'%PDF-1.4\n' + b'x' * (4 * 1024 * 1024)
        (self.root / 'batch1' / 'large.pdf').write_bytes(data)
        with patch.dict(app.app.config, MAX_CONTENT_LENGTH=1024), \
                patch.object(app, 'prepare_input', return_value=([], 'low', None, None)) as prepare, \
                patch.object(app, '_page_parallelism', return_value=1), \
                patch.object(app, 'log_run') as log, \
                patch.object(app.scoring, 'case_for_upload') as match:
            with self.client.post('/api/ocr/stream', data={
                    'realdata':'batch1/large.pdf', 'detail':'low', 'extract':'0'}) as response:
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual(response.status_code, 200)
                self.assertFalse([e for e in events if e['event'] == 'error'], events)
                self.assertIn('done', [e['event'] for e in events])
            self.assertEqual(prepare.call_args.args[0], data)
            self.assertIsNone(prepare.call_args.args[2])
            self.assertEqual(log.call_args.args[1], {
                'name':'batch1/large.pdf', 'size_bytes':len(data), 'origin':'realdata'})
            match.assert_not_called()

    def test_workspace_realdata_input_rejects_traversal_and_missing_or_unsupported_files(self):
        for name in ('../x.pdf', 'notes.txt', 'missing.pdf'):
            for route in ('/api/preview', '/api/ocr', '/api/ocr/stream'):
                with self.subTest(name=name, route=route):
                    with self.client.post(route, data={'realdata': name}) as response:
                        self.assertEqual(response.status_code, 400)

    def test_random_draws_from_every_file_of_the_chosen_types(self):
        self.assertEqual(self.cfg()["cases"], ["a.pdf", "b.JPEG", "batch1/c.png"])
        self.assertEqual(self.cfg(types=["jpeg", ".png"])["cases"],
                         ["b.JPEG", "batch1/c.png"])
        with self.assertRaises(ValueError):
            self.cfg(types=["tiff"])

    def test_chosen_files_must_be_in_the_folder_and_of_a_chosen_type(self):
        self.assertEqual(self.cfg(cases=["a.pdf"])["cases"], ["a.pdf"])
        with self.assertRaises(ValueError):
            self.cfg(cases=["a.pdf"], types=["png"])
        with self.assertRaises(ValueError):
            self.cfg(cases=["sol001"])

    def test_only_modes_that_start_from_a_read(self):
        self.assertEqual(self.cfg(mode="full")["source"], "realdata")
        for mode in ("extract", "classify"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.cfg(mode=mode)

    def test_empty_folder_is_refused(self):
        with patch.object(app, "REAL_DIR", self.root / "batch1" / "none"):
            with self.assertRaises(ValueError):
                self.cfg()

    def test_run_reads_and_extracts_without_scoring(self):
        cfg = self.cfg(mode="full", cases=["a.pdf"], jobs=2, warmup=1, seed=2,
                       history_monitor=False)
        with patch.object(app, "_stress_pages", return_value=([], {})) as pages, \
                patch.object(app, "_stress_read", return_value=(["a transcript"], False, False)), \
                patch.object(app, "extract_fields", return_value={"tokens": 1}) as extract, \
                patch.object(app.scoring, "evaluate") as evaluate, \
                patch.object(app.runlog, "case_counts") as counts:
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        self.assertTrue(pages.called)
        for call in pages.call_args_list:
            self.assertEqual(call.args, ("a.pdf", cfg["detail"], "realdata"))
        self.assertFalse([e for e in events if e["event"] == "preparing"])
        evaluate.assert_not_called()
        counts.assert_not_called()
        self.assertEqual({call.args[2] for call in extract.call_args_list}, {None})
        self.assertEqual(events[0]["source"], "realdata")
        report = events[-1]["report"]
        self.assertEqual(report["completed"], 2)
        self.assertIsNone(report["ocr"]["char_accuracy"])
        self.assertIsNone(report["extract"]["field_accuracy"])

    def test_pages_are_prepared_once_and_freed_after_the_last_job(self):
        prepared = {"pages": {}, "sources": {}}
        specs = [{"case": "a.pdf"}, {"case": "a.pdf"}, {"case": "b.JPEG"}]
        lazy = app._StressPages(specs, "low", prepared)
        with patch.object(app, "_stress_pages",
                          side_effect=lambda cid, *_: (["page of " + cid], {"name": cid})) as pages:
            threads = [threading.Thread(target=lazy.take, args=("a.pdf",)) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(pages.call_count, 1)
            self.assertEqual(prepared["pages"]["a.pdf"], ["page of a.pdf"])
            lazy.release("a.pdf")
            self.assertIn("a.pdf", prepared["pages"])     # one job still to come
            lazy.release("a.pdf")
            self.assertNotIn("a.pdf", prepared["pages"])
            self.assertEqual(prepared["sources"]["a.pdf"], {"name": "a.pdf"})
            lazy.take("b.JPEG")
            lazy.release("b.JPEG")
        self.assertEqual(prepared["pages"], {})

    def test_preparation_runs_a_bounded_number_at_once(self):
        prepared = {"pages": {}, "sources": {}}
        names = [f"f{i}.pdf" for i in range(8)]
        lazy = app._StressPages([{"case": n} for n in names], "low", prepared)
        inside, peak, lock = [0], [0], threading.Lock()

        def pages(cid, *_):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            with lock:
                inside[0] -= 1
            return [], {}
        with patch.object(app, "_stress_pages", side_effect=pages):
            threads = [threading.Thread(target=lazy.take, args=(n,)) for n in names]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(prepared["pages"]), 8)
        self.assertLessEqual(peak[0], app.STRESS_PREPARE_PARALLEL)

    def test_pdf_rendering_is_never_run_by_two_threads_at_once(self):
        inside, peak, lock = [0], [0], threading.Lock()

        class Pix:
            width = height = 1
            samples = b"\xff\xff\xff"

        class Page:
            def get_pixmap(self, matrix):
                return Pix()

        class Doc:
            needs_pass, page_count = False, 1

            def __enter__(self):
                with lock:
                    inside[0] += 1
                    peak[0] = max(peak[0], inside[0])
                time.sleep(0.05)
                return self

            def __exit__(self, *exc):
                with lock:
                    inside[0] -= 1

            def pages(self, start, stop):
                return [Page()]
        with patch.object(app.fitz, "open", side_effect=lambda **_: Doc()):
            threads = [threading.Thread(target=app.load_pages, args=(b"%PDF-1.4",))
                       for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(peak[0], 1)

    def test_a_file_that_will_not_open_fails_its_jobs_only(self):
        cfg = self.cfg(cases=["a.pdf", "b.JPEG"], jobs=6, concurrency=3,
                       strategy="uniform", seed=1, history_monitor=False)

        def pages(cid, *_):
            if cid == "b.JPEG":
                raise ValueError("cannot open")
            return [], {}
        with patch.object(app, "_stress_pages", side_effect=pages), \
                patch.object(app, "_stress_read", return_value=(["text"], False, False)):
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        counts = events[-1]["report"]["counts"]
        self.assertEqual((counts["ok"], counts["error"]), (3, 3))

    def test_pages_are_named_as_realdata(self):
        page = types.SimpleNamespace()
        with patch.object(app, "load_pages", return_value=([page], None)), \
                patch.object(app, "fit_pixels", side_effect=lambda p, b: p), \
                patch.object(app, "trim_margins", side_effect=lambda p: p), \
                patch.object(app, "image_data_uri", return_value="data:"):
            pages, source = app._stress_pages("batch1/c.png", "medium", "realdata")
        self.assertEqual(source["origin"], "realdata")
        self.assertEqual(source["name"], "batch1/c.png")
        self.assertEqual(len(pages), 1)


if __name__ == "__main__":
    unittest.main()
