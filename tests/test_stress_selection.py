"""Stress jobs and History must honor the chosen benchmark files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import engine
import config
import runhistory
import workspace


STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock",
          "model": "mock-model"}


class StressSelectionTests(unittest.TestCase):
    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.index = {}
        for cid in ("a", "b", "c"):
            truth = self.root / (cid + ".txt")
            truth.write_text(cid + " transcript", encoding="utf-8")
            self.index[cid] = {"id": cid, "ground_truth": truth,
                               "pdf": cid + ".pdf", "doc_types": ["INVOICE"]}
        for target, name, kwargs in (
            (app.scoring, "cases_index", {"return_value": self.index}),
            (app.fieldscore, "has_truth", {"side_effect": lambda cid: cid in ("a", "b")}),
            (engine, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
            (config, "LOG_DIR", {"new": self.root / "logs"}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.app.test_client()

    def cfg(self, **kwargs):
        return app._stress_config({"mode": "extract", **kwargs})

    def test_omitted_selection_keeps_existing_all_files_default(self):
        self.assertEqual(self.cfg()["cases"], ["a", "b"])

    def test_selection_is_validated_and_deduplicated(self):
        self.assertEqual(self.cfg(cases=["b", "a", "b"])["cases"], ["a", "b"])
        for invalid in ([], None, "a", {}, [None], [["a"]], ["missing"], ["c"]):
            with self.subTest(cases=invalid), self.assertRaises(ValueError):
                self.cfg(cases=invalid)

    def test_classification_can_select_transcripts_without_field_truth(self):
        self.assertEqual(self.cfg(mode="classify", cases=["c"])["cases"], ["c"])

    def test_lock_must_be_in_the_selection(self):
        self.assertEqual(self.cfg(cases=["b"], lock_case="b")["lock"], "b")
        with self.assertRaisesRegex(ValueError, "selected documents"):
            self.cfg(cases=["a"], lock_case="b")

    def test_invalid_selection_is_rejected_before_starting_a_run(self):
        with patch.object(app, "_stress_run") as run:
            for selected in ([], ["missing"], ["c"]):
                response = self.client.post("/api/stress/stream", json={
                    "mode": "extract", "cases": selected})
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
            run.assert_not_called()

    def test_warmup_measured_jobs_plan_and_history_use_only_selected_file(self):
        with patch.object(app, "extract_fields", return_value={"tokens": 1}) as extract:
            response = self.client.post("/api/stress/stream", json={
                "mode": "extract", "cases": ["b"], "jobs": 5, "warmup": 2,
                "concurrency": 2, "strategy": "uniform", "seed": 3,
                "history_monitor": False})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line) for line in response.data.splitlines()]
        self.assertEqual(extract.call_count, 7)
        self.assertEqual({call.args[2] for call in extract.call_args_list}, {"b"})
        self.assertEqual({call.args[0] for call in extract.call_args_list}, {"b transcript"})
        self.assertEqual(events[0]["cases"], ["b"])
        self.assertEqual(events[-1]["report"]["completed"], 5)
        saved = runhistory.get(events[-1]["history_id"])
        self.assertEqual(saved["configuration"]["cases"], ["b"])
        self.assertEqual(saved["plan"]["cases"], ["b"])

    def test_classification_packs_use_only_selected_transcripts(self):
        cfg = self.cfg(mode="classify", cases=["a", "c"], pack_pages=6,
                       warmup=1, jobs=2, seed=9)
        with patch.object(app, "_stress_classify", side_effect=lambda pack, rec, *_:
                          rec.update(status="ok")) as classify:
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        self.assertEqual(classify.call_count, 3)
        for call in classify.call_args_list:
            self.assertEqual(set(call.args[0]["cases"]), {"a", "c"})
        self.assertEqual(events[0]["cases"], ["a", "c"])

    def test_sequential_restarts_after_warmup_and_is_saved_in_history(self):
        with patch.object(app, "extract_fields", return_value={"tokens": 1}) as extract, \
             patch.object(app.runlog, "case_counts") as counts:
            response = self.client.post("/api/stress/stream", json={
                "mode": "extract", "cases": ["b", "a"], "jobs": 5, "warmup": 3,
                "concurrency": 1, "strategy": "sequential", "seed": 123,
                "history_monitor": False})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line) for line in response.data.splitlines()]
        self.assertEqual([call.args[2] for call in extract.call_args_list],
                         ["a", "b", "a", "a", "b", "a", "b", "a"])
        counts.assert_not_called()
        self.assertEqual(events[0]["strategy"], "sequential")
        self.assertEqual(events[-1]["report"]["completed"], 5)
        saved = runhistory.get(events[-1]["history_id"])
        self.assertEqual(saved["configuration"]["strategy"], "sequential")
        self.assertEqual(saved["plan"]["strategy"], "sequential")

    def test_ocr_and_full_only_prepare_selected_files(self):
        for mode in ("ocr", "full"):
            with self.subTest(mode=mode):
                cfg = self.cfg(mode=mode, cases=["a"], jobs=2, warmup=1, seed=2)
                with patch.object(app, "_stress_pages", return_value=([], {})) as pages, \
                     patch.object(app, "_stress_read", return_value=(["a transcript"], False, False)), \
                     patch.object(app, "extract_fields", return_value={"tokens": 1}), \
                     patch.object(app.scoring, "evaluate", return_value={"char_accuracy": 1}):
                    events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
                pages.assert_called_once_with("a", cfg["detail"])
                self.assertEqual(events[-1]["report"]["completed"], 2)


if __name__ == "__main__":
    unittest.main()
