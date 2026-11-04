"""Schema-first is opt-in per run, isolated across workers and recorded for comparison."""
import csv
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import app
import engine
import config
import grounding
import prompts
import runhistory
import runlog
from tests.test_summary_extras import row

STEP = prompts.EXTRACT_STEPS[0]
STATUS = {"kind": "vllm", "url": "http://mock", "model": "gemma",
          "available": True, "text_available": True, "reason": "", "text_reason": ""}
FORM = {"doc_types": ["INVOICE"], "items": (),
        "keys": list(STEP["keys"]), "mandatory": []}


class StepTests(unittest.TestCase):
    def test_default_uses_json_then_schema_only_when_parse_fails(self):
        with patch.object(engine, "_chat", side_effect=[("bad JSON", False, 1),
                                                    ('{"document_number":"A"}', False, 2)]) as chat:
            values, _, _, tokens, _ = app._ask_step("same prompt", STEP, STATUS)
        self.assertEqual(values["document_number"], "A")
        self.assertEqual(tokens, 3)
        self.assertEqual(len(chat.call_args_list[0].args), 3)
        self.assertEqual(chat.call_args_list[1].args[3], app._step_schema(STEP))

    def test_enabled_uses_step_schema_on_first_request(self):
        with patch.object(engine, "_chat", return_value=('{"document_number":"A"}', False, 2)) as chat:
            app._ask_step("same prompt", {**STEP, "schema_first": True}, STATUS)
        chat.assert_called_once_with("same prompt", STEP["max_tokens"], STATUS, app._step_schema(STEP))

    def test_first_wire_request_has_schema_and_preserves_prompt(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"document_number":"A"}'},
                                                    "finish_reason": "stop"}],
                                      "usage": {"completion_tokens": 2}}
        sent = []
        for on in (False, True):
            with patch.object(engine, "_timed_post", return_value=response) as post:
                app._ask_step("identical prompt", {**STEP, "schema_first": on}, STATUS)
            sent.append(post.call_args.args[1])
        self.assertEqual(sent[0]["messages"], sent[1]["messages"])
        self.assertEqual(sent[0]["response_format"], {"type": "json_object"})
        self.assertEqual(sent[1]["response_format"]["type"], "json_schema")
        self.assertEqual(sent[1]["response_format"]["json_schema"]["schema"], app._step_schema(STEP))

    def test_schema_failure_is_surfaced_without_silent_plain_fallback(self):
        for reply in ("invalid JSON",):
            with patch.object(engine, "_chat", return_value=(reply, False, 2)) as chat:
                with self.assertRaises(ValueError):
                    app._ask_step("x", {**STEP, "schema_first": True}, STATUS)
                self.assertEqual(chat.call_count, 1)
        with patch.object(engine, "_chat", side_effect=ValueError("HTTP 400 schema unsupported")) as chat:
            with self.assertRaisesRegex(ValueError, "schema unsupported"):
                app._ask_step("x", {**STEP, "schema_first": True}, STATUS)
            self.assertEqual(chat.call_count, 1)

    def test_grounding_reask_keeps_the_schema_and_corrects_the_value(self):
        with patch.object(engine, "AGENTIC_RETRIES", 1), patch.object(engine, "_chat", side_effect=[
                ('{"document_number":"invented"}', False, 2),
                ('{"document_number":"INV-123"}', False, 2)]) as chat:
            record, values, _, _ = app._run_step(
                {**STEP, "schema_first": True}, "source INV-123", "question", STATUS,
                grounding.Source("INV-123"))
        self.assertTrue(record["retried"])
        self.assertEqual(values["document_number"], "INV-123")
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(all(c.args[3] == app._step_schema(STEP) for c in chat.call_args_list))

    def test_concurrent_runs_do_not_mutate_shared_step_definitions(self):
        barrier = threading.Barrier(2)
        seen = {}

        def chat(content, cap, status, schema=None):
            seen[threading.current_thread().name] = schema is not None
            barrier.wait(2)
            return ('{"document_number":"A"}', False, 2)

        def run(on):
            return app._drain(app._extract_agentic("A", STATUS, {**FORM, "schema_first": on}))

        with patch.object(engine, "steps_for_types", return_value=(STEP,)), \
             patch.object(engine, "_steps_for", return_value=(STEP,)), \
             patch.object(engine, "_chat", side_effect=chat), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, [False, True]))
        self.assertEqual(len(results), 2)
        self.assertEqual(set(seen.values()), {False, True})
        self.assertNotIn("schema_first", STEP)


class RequestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        p = patch.object(config, "LOG_DIR", self.root)
        p.start()
        self.addCleanup(p.stop)
        self.client = app.app.test_client()

    def test_both_extract_endpoints_pass_the_request_option(self):
        def stream(*args, **kwargs):
            yield {"event": "extract_steps"}
            return {"mode": "agentic", "fields": {}, "schema_first": kwargs["schema_first"]}

        for url in ("/api/extract", "/api/extract/stream"):
            with self.subTest(url=url), patch.object(engine, "log_extract"), \
                 patch.object(app, "extract_fields", return_value={"fields": {}}) as blocking, \
                 patch.object(engine, "extract_fields_stream", side_effect=stream) as streaming:
                response = self.client.post(url, json={"text": "A", "mode": "agentic", "schema_first": True})
                self.assertEqual(response.status_code, 200)
                response.get_data()
                called = streaming if url.endswith("stream") else blocking
                self.assertTrue(called.call_args.kwargs["schema_first"])

    def test_invalid_boolean_is_a_400_before_inference(self):
        for url in ("/api/extract", "/api/extract/stream", "/api/stress/stream"):
            response = self.client.post(url, json={"text": "A", "schema_first": "yes"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("schema_first", response.get_json()["error"])

    def test_stress_history_and_plan_keep_effective_choice(self):
        def stream(cfg, cancel):
            yield json.dumps({"event": "plan", "schema_first": cfg["schema_first"]})
            yield json.dumps({"event": "done", "report": {}})

        with patch.object(app.backends, "extract_status", return_value=STATUS), \
             patch.object(app, "_stress_run", side_effect=stream):
            response = self.client.post("/api/stress/stream", json={
                "mode": "extract", "extract_mode": "agentic", "schema_first": True,
                "history_monitor": False})
            events = [json.loads(line) for line in response.data.splitlines()]
        item = runhistory.get(events[-1]["history_id"])
        self.assertTrue(item["configuration"]["schema_first"])
        self.assertTrue(item["plan"]["schema_first"])

    def test_disabled_and_single_stress_keep_option_off(self):
        with patch.object(app.backends, "extract_status", return_value=STATUS):
            self.assertFalse(app._stress_config({"mode": "extract", "extract_mode": "agentic"})["schema_first"])
            self.assertFalse(app._stress_config({"mode": "extract", "extract_mode": "single",
                                                "schema_first": True})["schema_first"])

    def test_multiple_documents_receive_the_same_option_and_log_it(self):
        def segment(*_):
            if False:
                yield
            return ([{"text": "A", "pages": [i], "codes": ["INVOICE"]} for i in (1, 2)], "test")

        with patch.object(app.backends, "extract_status", return_value=STATUS), \
             patch.object(engine, "_segment_stream", side_effect=segment), \
             patch.object(engine, "steps_for_types", return_value=(STEP,)), \
             patch.object(engine, "_steps_for", return_value=(STEP,)), \
             patch.object(engine, "TABLE_EXTRACT", False), \
             patch.object(engine, "_chat", return_value=('{"document_number":"A"}', False, 2)) as chat:
            result = app.extract_fields("A", "agentic", doc_type=["INVOICE"], schema_first=True)
        self.assertTrue(result["schema_first"])
        self.assertTrue(all(d["schema_first"] for d in result["documents"]))
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(runlog._extract_cells({"extracted": result})["extract_schema_first"], 1)

    def test_stress_jobs_forward_the_schema_choice(self):
        with patch.object(app.backends, "extract_status", return_value=STATUS), \
             patch.object(app, "extract_fields", return_value={
                 "mode": "agentic", "fields": {}, "schema_first": True, "tokens": 1}) as extract:
            cfg = app._stress_config({"mode": "extract", "extract_mode": "agentic",
                                     "schema_first": True, "cases": ["sol001"],
                                     "jobs": 2, "warmup": 1, "concurrency": 2})
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        self.assertEqual(extract.call_count, 3)
        self.assertTrue(all(c.kwargs["schema_first"] for c in extract.call_args_list))
        self.assertTrue(events[0]["schema_first"])


class LogTests(unittest.TestCase):
    def test_summaries_and_recent_windows_separate_schema_first_from_legacy(self):
        legacy = row(mode="agentic", seconds=100)
        off = row(mode="agentic", seconds=200, extract_schema_first="0")
        on = row(mode="agentic", seconds=10, extract_schema_first="1")
        with runlog.view(window=1):
            result = runlog.presentation([legacy, off, on], bias=False)
            detailed = runlog.by_extract([legacy, off, on])
        shapes = {r["schema_first"]: r for r in result["extract_shapes"]}
        self.assertEqual(set(shapes), {False, True})
        self.assertEqual(shapes[False]["seconds"]["mean"], 100)
        self.assertEqual(shapes[True]["seconds"]["mean"], 10)
        self.assertEqual(len(detailed), 2)
        self.assertEqual({r["extract_schema_first"] for r in detailed}, {"0", "1"})
        self.assertNotIn("extract_schema_first", legacy)

    def test_filter_and_single_mode_default(self):
        rows = [row(mode="agentic"), row(mode="agentic", extract_schema_first="1"),
                row(mode="single", extract_schema_first="1")]
        self.assertEqual(runlog.filter_rows(rows, include={"extract_schema_first": ["on"]}), [rows[1]])
        self.assertFalse(runlog.extract_schema_first_of(rows[2]))

    def test_old_csv_migrates_with_legacy_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = Path(directory), Path(directory) / "runs.csv"
            columns = [c for c in runlog.COLUMNS if c != "extract_schema_first"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow({"file": "old.pdf", "extract_mode": "agentic", "extract_seconds": 123})
            with patch.multiple(runlog, LOG_DIR=root, LOG_PATH=path):
                runlog.record({"extracted": {"mode": "agentic", "schema_first": True,
                                            "fields": {}, "seconds": 10}}, {"name": "new.pdf"})
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["extract_seconds"], "123")
            self.assertEqual(rows[0]["extract_schema_first"], "")
            self.assertEqual(rows[1]["extract_schema_first"], "1")
