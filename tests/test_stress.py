"""The stress test: its arithmetic, its vLLM detection, and that it logs nothing.

The model server is stubbed at the functions the runner calls, and the run log
is patched to FAIL the test if anything is recorded -- the user's rule for this
mode is that only the final report is shown and no job is written anywhere.
"""
import json
import random
import unittest
from unittest.mock import MagicMock, patch

import app
import backends
import stress


class StressMathTests(unittest.TestCase):
    def test_distribution_reports_the_tail(self):
        d = stress.distribution([1] * 19 + [40])
        self.assertEqual(d["n"], 20)
        self.assertEqual(d["p50"], 1)
        self.assertGreater(d["p99"], 30)
        self.assertEqual(stress.distribution([None])["n"], 0)

    def test_uniform_uses_every_document_before_any_twice(self):
        got = stress.draw_cases(10, list("abcde"), "uniform", random.Random(2))
        self.assertEqual(sorted(got[:5]), list("abcde"))
        self.assertEqual(sorted(got[5:]), list("abcde"))

    def test_balanced_is_the_random_tests_rule(self):
        got = stress.draw_cases(3, ["a", "b", "c"], "balanced", random.Random(1),
                                history={"a": 5, "b": 5})
        self.assertEqual(got[0], "c")

    def test_pack_never_glues_a_case_to_itself_and_covers_every_page(self):
        truths = {"x": "one", "y": "a\n--- page 2 ---\nb", "z": "three"}
        docs = {"y": [{"pages": [1], "doc_types": ["A"]}, {"pages": [2], "doc_types": ["B"]}],
                "_types": {"x": ["X"], "z": ["Z"]}}
        pack = stress.build_pack(list(truths), truths, docs, 40, random.Random(9))
        self.assertGreaterEqual(len(pack["pages"]), 40)
        self.assertTrue(all(a != b for a, b in zip(pack["cases"], pack["cases"][1:])))
        pages = sorted(p for d in pack["expected"] for p in d["pages"])
        self.assertEqual(pages, list(range(1, len(pack["pages"]) + 1)))
        self.assertIn("--- page 40 ---", pack["text"])

    def test_metrics_are_deltas_since_the_base(self):
        base = stress.parse_prometheus("vllm:generation_tokens_total 100\n"
                                       "vllm:time_to_first_token_seconds_sum 1\n"
                                       "vllm:time_to_first_token_seconds_count 1\n")
        now = stress.parse_prometheus("vllm:generation_tokens_total 400\n"
                                      "vllm:gpu_cache_usage_perc 0.5\n"
                                      "vllm:time_to_first_token_seconds_sum 7\n"
                                      "vllm:time_to_first_token_seconds_count 4\n")
        view = stress.metrics_view(now, base, seconds=10)
        self.assertEqual(view["generation_tokens"], 300)
        self.assertEqual(view["generation_tokens_per_s"], 30)
        self.assertEqual(view["kv_cache"], 0.5)          # the v0 name is read too
        self.assertEqual(view["ttft"], 2.0)              # (7-1)/(4-1)

    def test_scores_are_over_clean_jobs_only(self):
        rep = stress.report([
            {"status": "ok", "total_s": 1, "wait_s": 0, "char_accuracy": 0.9,
             "pages": 1, "ocr_s": 1},
            {"status": "looped", "total_s": 5, "wait_s": 0, "char_accuracy": 0.2,
             "pages": 1, "ocr_s": 5},
            {"status": "cancelled"}], 5.0, "ocr", 2)
        self.assertEqual(rep["completed"], 2)
        self.assertEqual(rep["ocr"]["char_accuracy"], 0.9)
        self.assertEqual(rep["latency"]["end_to_end"]["n"], 2)
        self.assertEqual(rep["counts"]["cancelled"], 1)


class VllmTests(unittest.TestCase):
    def _get(self, body):
        res = MagicMock(status_code=200)
        res.json.return_value = body
        return res

    def test_vllm_is_named_by_owned_by(self):
        with patch.object(backends.requests, "get",
                          return_value=self._get({"data": [{"id": "m", "owned_by": "vllm"}]})):
            info = backends._probe_openai("http://x")
        self.assertEqual(info["kind"], "vllm")
        self.assertTrue(backends.serves_metrics(info))

    def test_another_openai_server_gets_no_metrics(self):
        with patch.object(backends.requests, "get",
                          return_value=self._get({"data": [{"id": "m", "owned_by": "sglang"}]})):
            info = backends._probe_openai("http://x")
        self.assertEqual(info["kind"], "openai")
        self.assertFalse(backends.serves_metrics(info))
        self.assertFalse(backends.serves_metrics({"kind": "ollama"}))
        self.assertFalse(backends.serves_metrics({"kind": "llama.cpp"}))

    def test_vllm_requests_name_the_model_and_send_no_window(self):
        extras = backends.request_extras({"kind": "vllm", "model": "m"})
        self.assertEqual(extras["model"], "m")
        self.assertNotIn("n_ctx", extras)
        self.assertNotIn("options", extras)

    def test_metrics_route_refuses_a_local_server(self):
        client = app.app.test_client()
        with patch.object(app.backends, "status",
                          return_value={"kind": "ollama", "url": "http://l"}), \
             patch.object(app, "_vllm_metrics") as fired:
            res = client.get("/api/stress/metrics")
        self.assertEqual(res.status_code, 409)
        fired.assert_not_called()


class MultiCancelTests(unittest.TestCase):
    class Closeable:
        closed = False

        def close(self):
            self.closed = True

    def test_every_request_in_flight_is_hung_up(self):
        cancel = app.MultiCancel()
        a, b = self.Closeable(), self.Closeable()
        cancel.attach(a)
        cancel.attach(b)
        cancel.set()
        self.assertTrue(a.closed and b.closed)
        late = self.Closeable()
        cancel.attach(late)
        self.assertTrue(late.closed)


class RunTests(unittest.TestCase):
    def test_config_refuses_nonsense_and_clamps(self):
        with self.assertRaises(ValueError):
            app._stress_config({"mode": "bogus"})
        with patch.object(app.backends, "extract_status",
                          return_value={"text_available": True, "text_reason": None}):
            cfg = app._stress_config({"mode": "extract", "concurrency": 99999, "jobs": 0})
            self.assertEqual(cfg["concurrency"], stress.MAX_CONCURRENCY)
            self.assertEqual(cfg["jobs"], 1)
            cls = app._stress_config({"mode": "classify", "lock_case": "sol002"})
            self.assertEqual(cls["lock"], "")      # a pack of one case has no boundaries

    def test_extract_run_reports_and_writes_nothing(self):
        result = {"tokens": 100, "field_score": {"overall": {
            "counts": {"correct": 3, "partial": 0}, "expected": 4}}}
        status = {"text_available": True, "text_reason": None, "kind": "ollama",
                  "url": "http://l", "model": "m"}
        with patch.object(app.backends, "extract_status", return_value=status), \
             patch.object(app, "extract_fields", return_value=result), \
             patch.object(app.runlog, "record",
                          side_effect=AssertionError("the stress test logged a row")), \
             patch.object(app, "log_run",
                          side_effect=AssertionError("the stress test logged a run")):
            cfg = app._stress_config({"mode": "extract", "concurrency": 3, "jobs": 5,
                                      "strategy": "uniform", "seed": 4})
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        done = events[-1]
        self.assertEqual(done["event"], "done")
        rep = done["report"]
        self.assertEqual(rep["completed"], 5)
        self.assertEqual(rep["extract"]["field_pooled"], 0.75)
        self.assertEqual(rep["extract"]["tokens"], 500)
        self.assertEqual(events[0]["event"], "plan")
        self.assertEqual(events[0]["seed"], 4)
        # No per-job event: the report is the whole output.
        self.assertFalse(any(e["event"] not in ("plan", "preparing", "progress", "done")
                             for e in events))

    def test_a_failing_job_is_counted_not_fatal(self):
        status = {"text_available": True, "text_reason": None, "kind": "ollama",
                  "url": "http://l", "model": "m"}
        with patch.object(app.backends, "extract_status", return_value=status), \
             patch.object(app, "extract_fields", side_effect=ValueError("boom")):
            cfg = app._stress_config({"mode": "extract", "concurrency": 2, "jobs": 2})
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        rep = events[-1]["report"]
        self.assertEqual(rep["error_rate"], 1.0)
        self.assertEqual(rep["errors"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
