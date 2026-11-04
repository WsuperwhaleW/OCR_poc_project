"""Per-request agentic context, distinct from summed per-file usage."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app
import engine
import config
import prompts
import runhistory
import stress


VLLM = {"kind": "vllm", "url": "http://context-test:8001", "model": "m",
        "models": [{"name": "m", "max_model_len": 32768}]}


def reply(sent=1000, output=50, text="{}"):
    return app._Reply(200, {"usage": {"prompt_tokens": sent, "completion_tokens": output,
                                     "prompt_tokens_details": {"cached_tokens": 700}},
                            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


class MeasurementTests(unittest.TestCase):
    def test_actual_usage_includes_cache_and_uses_server_window(self):
        result = app._chat_measurement({"max_tokens": 512, "n_ctx": 8192}, VLLM, reply())
        self.assertEqual((result["in_tokens"], result["out_tokens"], result["used_tokens"]),
                         (1000, 50, 1050))
        self.assertEqual(result["context_limit"], 32768)
        self.assertEqual(result["reply_cap"], 512)
        self.assertEqual(result["used_pct"], 3.2)

    def test_no_usage_and_no_metadata_stay_unknown_without_network_requests(self):
        with patch.object(app.backends, "known", return_value=None), \
             patch.object(app.backends, "http_get", side_effect=AssertionError("extra probe")):
            result = app._chat_measurement({"max_tokens": 100}, {**VLLM, "models": []},
                                          app._Reply(200, {}))
        self.assertIsNone(result["in_tokens"])
        self.assertIsNone(result["used_tokens"])
        self.assertIsNone(result["context_limit"])
        self.assertIsNone(result["used_pct"])

    def test_native_ollama_context_is_labelled_requested(self):
        result = app._chat_measurement({"options": {"num_ctx": 8192, "num_predict": 256}},
            {"kind": "ollama"}, app._Reply(200, {"prompt_eval_count": 100, "eval_count": 0}))
        self.assertEqual((result["context_limit"], result["context_source"]), (8192, "requested"))
        self.assertEqual((result["used_tokens"], result["out_tokens"]), (100, 0))

    def test_llama_computed_prompt_alone_is_not_full_context_usage(self):
        response = app._Reply(200, {"timings": {"prompt_n": 50, "predicted_n": 20}})
        info = {"kind": "llama.cpp"}
        result = app._chat_measurement({"n_ctx": 4096}, info, response)
        self.assertIsNone(result["in_tokens"])
        response.json()["timings"]["cache_n"] = 1000
        result = app._chat_measurement({"n_ctx": 4096}, info, response)
        self.assertEqual(result["used_tokens"], 1070)

    def test_loop_stream_chunk_estimate_is_not_shown_as_exact_tokens(self):
        result = app._chat_measurement({}, VLLM, app._Reply(200, {
            "loop_aborted": True, "usage": {"completion_tokens": 10}}))
        self.assertIsNone(result["out_tokens"])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.step = prompts.steps_for_types(["INVOICE"], extras=False)[0]

    def run_step(self):
        return app._run_step(self.step, "transcript ", "question", VLLM, "transcript")

    def test_schema_retry_is_two_separate_requests(self):
        content = json.dumps({key: "" for key in self.step["keys"]})
        with patch.object(engine, "_post_dispatch", side_effect=[reply(text="invalid"),
                                                             reply(sent=1100, text=content)]):
            outcome = self.run_step()
        chats = outcome[0]["chats"]
        self.assertEqual(len(chats), 2)
        self.assertEqual([c["in_tokens"] for c in chats], [1000, 1100])
        self.assertEqual([c["schema"] for c in chats], [False, True])
        self.assertIsNone(getattr(app._agentic_chat_context, "requests", None))

    def test_context_overflow_retry_records_the_actual_fitted_reply_cap(self):
        content = json.dumps({key: "" for key in self.step["keys"]})
        with patch.object(engine, "_post_dispatch", side_effect=[
                app._Reply(400, text="maximum context length"), reply(text=content)]), \
             patch.object(engine, "VLLM_FIT_MAX_TOKENS", True), \
             patch.object(engine, "_fit_max_tokens", side_effect=lambda payload, _: {
                 **payload, "max_tokens": 20}):
            outcome = self.run_step()
        chats = outcome[0]["chats"]
        self.assertEqual([c["http_status"] for c in chats], [400, 200])
        self.assertEqual(chats[1]["reply_cap"], 20)
        self.assertIsNone(chats[0]["in_tokens"])

    def test_transport_failure_is_recorded_without_guessing_usage(self):
        with patch.object(engine, "_post_dispatch", side_effect=ValueError("offline")):
            outcome = self.run_step()
        self.assertIn("error", outcome[0])
        self.assertEqual(outcome[0]["chats"][0]["error"], "offline")
        self.assertIsNone(outcome[0]["chats"][0]["used_tokens"])

    def test_parallel_steps_do_not_share_chat_lists(self):
        barrier = threading.Barrier(2)
        def dispatch(url, payload, status):
            barrier.wait(timeout=3)
            return reply(sent=payload["sent"])
        def asked(step, *args):
            app._post_once("http://test", {"sent": int(step["id"])}, VLLM)
            return ({"id": step["id"]}, {}, [], 0)
        with patch.object(engine, "_post_dispatch", side_effect=dispatch), \
             patch.object(engine, "_run_step_asked", side_effect=asked), \
             ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda n: app._run_step({"id": str(n)}, "", "", VLLM, ""),
                                     [111, 222]))
        self.assertEqual([o[0]["chats"][0]["in_tokens"] for o in outcomes], [111, 222])
        self.assertTrue(all(len(o[0]["chats"]) == 1 for o in outcomes))


class ReportTests(unittest.TestCase):
    def test_report_and_history_keep_each_chat_without_prompts_and_skip_cancelled_jobs(self):
        base = {**app._chat_measurement({"max_tokens": 128}, VLLM, reply()),
                "step": "document", "title": "Document identity", "keys": ["document_number"],
                "document": 1, "request": 1, "case": "sol002"}
        report = stress.report([
            {"status": "ok", "agentic_chats": [{**base, "job": 1}]},
            {"status": "error", "agentic_chats": [{**base, "job": 2}]},
            {"status": "cancelled", "agentic_chats": [{**base, "job": 3}]}], 1, "extract", 2)
        context = report["agentic_context"]
        self.assertEqual([r["job"] for r in context["requests"]], [1, 2])
        self.assertEqual(context["steps"][0]["requests"], 2)
        self.assertEqual(context["steps"][0]["used"]["max"], 1050)
        self.assertNotIn("prompt", json.dumps(context))
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "LOG_DIR", Path(directory)):
            events = list(app._history_stream((line for line in [json.dumps({"event": "done", "report": report}) + "\n"]),
                                               "stress", {"history_monitor": False}))
            saved = runhistory.get(json.loads(events[-1])["history_id"])
        self.assertEqual(saved["result"]["report"]["agentic_context"], context)

    def test_multiple_documents_and_retries_keep_distinct_identities(self):
        result = {"documents": [{"document": doc, "steps": [
            {"id": "seller", "keys": ["seller_name"], "chats": [{"request": 1}, {"request": 2}]}]}
            for doc in (1, 2)]}
        chats = list(app._agentic_chats(result))
        self.assertEqual([(c["document"], c["request"]) for c in chats], [(1, 1), (1, 2), (2, 1), (2, 2)])

    def test_old_report_does_not_gain_invented_chat_measurements(self):
        self.assertNotIn("agentic_context", stress.report([{"status": "ok"}], 1, "extract", 1))


if __name__ == "__main__":
    unittest.main()
