"""The concurrency sweep: levels run one after another, and each level's vLLM
/metrics figures are a delta over that level alone (`concurrencysweep`)."""
import json
import math
import threading
import time
import unittest
from unittest import mock

import app
import concurrencysweep as cs

VLLM = {"kind": "vllm", "url": "http://vllm:8000", "model": "qwen"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "model": "gemma"}

E2E = "vllm:e2e_request_latency_seconds"
BOUNDS = (0.01, 0.1, 1.0, math.inf)


class LevelTests(unittest.TestCase):
    def test_levels_are_sorted_distinct_and_bounded(self):
        self.assertEqual(cs.parse_levels("16, 1 4;4"), [1, 4, 16])
        self.assertEqual(cs.parse_levels([2, "8"]), [2, 8])
        self.assertEqual(cs.parse_levels(None), list(cs.DEFAULT_LEVELS))
        for bad in ("0", "1, x", "999", ",".join(str(i) for i in range(1, 30))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cs.parse_levels(bad)

    def test_every_level_queues_the_highest_level_unless_asked(self):
        self.assertEqual(cs.level_jobs([1, 8, 128]), 128)
        self.assertEqual(cs.level_jobs([1, 8, 128], "20"), 20)
        self.assertEqual(cs.level_jobs([1, 2], ""), 2)
        with self.assertRaises(ValueError):
            cs.level_jobs([1], "many")


class HistogramTests(unittest.TestCase):
    TEXT = ('# TYPE vllm:e2e_request_latency_seconds histogram\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.5"} 1\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0"} 3\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf"} 4\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="1",le="0.5"} 1\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="1",le="1.0"} 1\n'
            'vllm:e2e_request_latency_seconds_bucket{engine="1",le="+Inf"} 2\n'
            'vllm:e2e_request_latency_seconds_sum 9\n'
            'vllm:e2e_request_latency_seconds_count 6\n')

    def test_buckets_are_summed_over_engines(self):
        self.assertEqual(cs.parse_buckets(self.TEXT),
                         {E2E: {0.5: 2.0, 1.0: 4.0, math.inf: 6.0}})

    def test_quantile_matches_prometheus(self):
        b = {0.5: 2.0, 1.0: 4.0, math.inf: 6.0}
        self.assertAlmostEqual(cs.quantile(.5, b), 0.75)    # rank 3: halfway into (0.5, 1]
        self.assertAlmostEqual(cs.quantile(.25, b), 0.375)  # rank 1.5: inside the first bucket
        self.assertEqual(cs.quantile(.99, b), 1.0)          # in +Inf: the highest finite bound
        self.assertIsNone(cs.quantile(.5, {0.5: 0.0, math.inf: 0.0}))
        self.assertIsNone(cs.quantile(.5, {}))

    def test_a_count_that_went_down_is_a_restart(self):
        self.assertIsNone(cs.bucket_delta({1.0: 2.0}, {1.0: 5.0}))


def _snap(e2e_values, gen_tokens, requests):
    """`(values, buckets)` as one scrape of a server that has seen these requests."""
    buckets = {b: float(sum(1 for v in e2e_values if v <= b)) for b in BOUNDS}
    return ({E2E + "_sum": float(sum(e2e_values)), E2E + "_count": float(len(e2e_values)),
             "vllm:generation_tokens_total": float(gen_tokens),
             "vllm:request_success_total": float(requests)},
            {E2E: buckets})


class LevelViewTests(unittest.TestCase):
    def test_a_level_is_its_own_requests_only(self):
        # Ten slow requests before the level, four fast ones in it.
        before = _snap([0.5] * 10, 1000, 10)
        after = _snap([0.5] * 10 + [0.05] * 4, 1400, 14)
        view = cs.level_view(before, after, 2.0, ticks=[
            {"running": 3, "waiting": 1, "kv_cache": .2},
            {"running": 4, "waiting": 0, "kv_cache": .4}])
        self.assertEqual(view["e2e"]["count"], 4)
        self.assertAlmostEqual(view["e2e"]["mean"], 0.05)
        self.assertLessEqual(view["e2e"]["p99"], 0.1)       # not pulled up by the 0.5 s ones
        self.assertEqual(view["generation_tokens_per_s"], 200)
        self.assertEqual(view["requests_per_s"], 2)
        self.assertEqual(view["running"], {"mean": 3.5, "max": 4, "n": 2})
        self.assertEqual(view["kv_cache"]["max"], .4)

    def test_a_series_the_server_lacks_is_absent_and_a_restart_is_blank(self):
        before = _snap([0.5] * 10, 1000, 10)
        after = _snap([0.05], 50, 1)                         # the server restarted
        view = cs.level_view(before, after, 1.0)
        self.assertNotIn("ttft", view)
        self.assertEqual(view["e2e"], {"count": None})
        self.assertIsNone(view["generation_tokens"])
        self.assertIsNone(view["running"])

    def test_summary_finds_the_peak_and_the_knee(self):
        rows = [{"level": lv, "servers": [{"view": {"generation_tokens_per_s": tps}}]}
                for lv, tps in ((1, 100), (2, 190), (4, 360), (8, 380), (16, 370))]
        self.assertEqual(cs.summary(rows), {"peak_level": 8, "peak_tps": 380,
                                            "knee_level": 8})

    def test_achieved_concurrency_is_busy_time_over_wall(self):
        rep = {"wall_seconds": 10.0, "latency": {"end_to_end": {"n": 8, "mean": 5.0}}}
        self.assertEqual(cs.client_view(rep)["achieved"], 4.0)


class _FakeVllm:
    """A vLLM whose /metrics counts the extraction requests sent to it."""

    def __init__(self, hold=0.05):
        self.lock = threading.Lock()
        self.e2e, self.gen, self.cases = [], 0, []
        self.now = self.peak = 0
        self.hold = hold

    def extract(self, _text, _mode=None, case_id=None, **_k):
        began = time.perf_counter()
        with self.lock:
            self.cases.append(case_id)
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.hold)
        with self.lock:
            self.now -= 1
            self.e2e.append(time.perf_counter() - began)
            self.gen += 10
        return {"tokens": 10, "field_score": {"overall": {
            "counts": {"correct": 1, "partial": 0}, "expected": 1}}}

    def http_get(self, url, info, timeout=None):
        with self.lock:
            lines = [f'vllm:num_requests_running{{engine="0"}} {self.now}',
                     f"vllm:generation_tokens_total {self.gen}",
                     f"vllm:request_success_total {len(self.e2e)}",
                     f"{E2E}_sum {sum(self.e2e)}", f"{E2E}_count {len(self.e2e)}"]
            for b in BOUNDS:
                le = "+Inf" if b == math.inf else b
                lines.append(f'{E2E}_bucket{{le="{le}"}} {sum(1 for v in self.e2e if v <= b)}')
        return mock.Mock(status_code=200, text="\n".join(lines) + "\n")


class SweepRunTests(unittest.TestCase):
    def _patches(self, fake, server=VLLM):
        return [mock.patch.object(app, "llama_status", return_value=server),
                mock.patch.object(app.backends, "extract_status",
                                  return_value=dict(server, text_available=True,
                                                    text_reason=None)),
                mock.patch.object(app, "extract_fields", side_effect=fake.extract),
                mock.patch.object(app.backends, "http_get", side_effect=fake.http_get),
                mock.patch.object(app, "SWEEP_SETTLE_SECONDS", 0.0)]

    def _run(self, body, stop_after_level=None):
        fake = _FakeVllm()
        patches = self._patches(fake)
        for p in patches:
            p.start()
        try:
            cfg = app._stress_config({"mode": "extract", "seed": 1, "metrics_every": 1, **body})
            cancel, events, peaks = app.MultiCancel(), [], []
            self.level_cases = []
            for line in app._stress_run(cfg, cancel):
                event = json.loads(line)
                events.append(event)
                if event["event"] == "sweep_level":
                    with fake.lock:
                        peaks.append(fake.peak)
                        fake.peak = 0
                        self.level_cases.append(fake.cases)
                        fake.cases = []
                    if stop_after_level is not None and event["index"] == stop_after_level:
                        cancel.set()
        finally:
            for p in patches:
                p.stop()
        return cfg, events, peaks

    def test_config_sizes_the_run_from_the_levels(self):
        fake = _FakeVllm()
        patches = self._patches(fake)
        for p in patches:
            p.start()
        try:
            cfg = app._stress_config({"mode": "extract", "sweep": "4, 1, 2"})
            self.assertEqual(cfg["sweep"], [1, 2, 4])
            self.assertEqual(cfg["concurrency"], 4)
            self.assertEqual(cfg["sweep_jobs"], 4)            # the highest level, at every level
            self.assertEqual(cfg["jobs"], 4 * 3)
            self.assertTrue(cfg["metrics"])
            cfg = app._stress_config({"mode": "extract", "sweep": "1, 128", "sweep_jobs": 10})
            self.assertEqual((cfg["sweep_jobs"], cfg["jobs"]), (10, 20))
            with self.assertRaises(ValueError):
                app._stress_config({"mode": "extract", "sweep": "1, 2, 4",
                                    "sweep_jobs": 2000})
        finally:
            for p in patches:
                p.stop()

    def test_a_sweep_is_refused_without_vllm(self):
        fake = _FakeVllm()
        patches = self._patches(fake, server=OLLAMA)
        for p in patches:
            p.start()
        try:
            with self.assertRaisesRegex(ValueError, "vLLM"):
                app._stress_config({"mode": "extract", "sweep": "1,2"})
            # A plain stress run on the same server is still fine.
            self.assertIsNone(app._stress_config({"mode": "extract"})["sweep"])
        finally:
            for p in patches:
                p.stop()

    def test_each_level_runs_at_its_own_limit_and_reads_its_own_delta(self):
        cfg, events, peaks = self._run({"sweep": "1,2,4", "sweep_jobs": 8})
        plan = events[0]
        self.assertEqual(plan["sweep"], [{"level": 1, "jobs": 8}, {"level": 2, "jobs": 8},
                                         {"level": 4, "jobs": 8}])
        levels = [e for e in events if e["event"] == "sweep_level"]
        self.assertEqual([e["level"] for e in levels], [1, 2, 4])
        self.assertEqual(peaks, [1, 2, 4])                   # never past the level's limit
        for e in levels:
            view = e["servers"][0]["view"]
            self.assertEqual(view["requests"], e["jobs"])     # this level's requests only
            self.assertEqual(view["e2e"]["count"], e["jobs"])
            self.assertEqual(view["generation_tokens"], 10 * e["jobs"])
            self.assertIsNotNone(view["e2e"]["p50"])
            self.assertEqual(e["client"]["completed"], e["jobs"])
        progress = [e for e in events if e["event"] == "progress" and e.get("level")]
        self.assertEqual(sorted({e["level"] for e in progress}), [1, 2, 4])
        report = events[-1]["report"]
        self.assertEqual(report["completed"], 24)
        self.assertEqual([r["level"] for r in report["sweep"]["levels"]], [1, 2, 4])
        self.assertEqual(report["concurrency"], 4)           # put back after the levels
        self.assertEqual(report["outliers"], [])
        self.assertEqual(cfg["concurrency"], 4)

    def test_stop_ends_the_sweep_at_the_level_running(self):
        _, events, _ = self._run({"sweep": "1,2,4", "sweep_jobs": 2},
                                 stop_after_level=0)
        done = events[-1] if events[-1]["event"] == "done" else next(
            e for e in events if e["event"] == "done")
        self.assertTrue(done["stopped"])
        self.assertEqual([r["level"] for r in done["report"]["sweep"]["levels"]], [1])
        self.assertEqual(done["report"]["counts"]["cancelled"], 4)

    def test_every_level_reads_the_same_files(self):
        _, events, _ = self._run({"sweep": "1,2,4", "sweep_jobs": 6,
                                  "strategy": "uniform", "warmup": 2})
        plan = events[0]
        self.assertEqual(len(plan["sweep_cases"]), 6)
        self.assertGreater(len(set(plan["sweep_cases"])), 1)  # a real draw, not one file
        # One level's requests are the plan's files; the warm-up's are its own.
        self.assertEqual(len(self.level_cases), 3)
        for got in self.level_cases[1:]:
            self.assertEqual(sorted(got), sorted(plan["sweep_cases"]))
        self.assertEqual(sorted(self.level_cases[0][-6:]), sorted(plan["sweep_cases"]))
        # At one concurrent the files are read in the plan's order.
        self.assertEqual(self.level_cases[0][-6:], plan["sweep_cases"])

    def test_classify_reuses_the_same_glued_files(self):
        fake, seen = _FakeVllm(), []

        def classify(pack, rec, live, cancel, job):
            fake.extract("", case_id=None)
            with fake.lock:
                seen.append(id(pack))
            rec["status"] = "ok"

        patches = self._patches(fake) + [
            mock.patch.object(app, "_stress_classify", side_effect=classify)]
        for p in patches:
            p.start()
        try:
            cfg = app._stress_config({"mode": "classify", "sweep": "1,2", "sweep_jobs": 3,
                                      "pack_pages": 4, "seed": 1})
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        finally:
            for p in patches:
                p.stop()
        self.assertEqual([e["event"] for e in events].count("preparing"), 3)   # 3 packs, not 6
        self.assertEqual(len(seen), 6)
        self.assertEqual(sorted(seen[:3]), sorted(seen[3:]))
        self.assertEqual(len(set(seen)), 3)

    def test_saved_rows_carry_the_level_they_ran_at(self):
        fake = _FakeVllm()
        patches = self._patches(fake)
        for p in patches:
            p.start()
        try:
            with mock.patch.object(app, "log_extract") as logged:
                cfg = app._stress_config({"mode": "extract", "sweep": "1,2,4", "sweep_jobs": 3,
                                          "warmup": 1, "seed": 1, "save_log": True})
                events = [json.loads(x) for x in app._stress_run(cfg, app.MultiCancel())]
        finally:
            for p in patches:
                p.stop()
        tags = [call.kwargs["extra"] for call in logged.call_args_list]
        self.assertEqual(len(tags), 9)                        # 3 per level; no warm-up row
        self.assertEqual({t["stress_run"] for t in tags}, {events[0]["run_id"]})
        self.assertEqual(sorted(t["stress_concurrency"] for t in tags), [1] * 3 + [2] * 3 + [4] * 3)


class SweepHistoryTests(unittest.TestCase):
    def test_a_run_cut_off_before_its_report_keeps_its_finished_levels(self):
        level = {"event": "sweep_level", "index": 0, "levels": 2, "level": 1, "jobs": 2,
                 "client": {}, "servers": [], "errors": []}

        def stream():
            yield json.dumps({"event": "plan", "sweep": [{"level": 1, "jobs": 2}]}) + "\n"
            yield json.dumps(level) + "\n"
            # ...and the server goes away before level 2 ends.

        with mock.patch.object(app.runhistory, "start", return_value="r1"),                 mock.patch.object(app.runhistory, "save_plan"),                 mock.patch.object(app.runhistory, "finish") as finish:
            lines = list(app._history_stream(stream(), "stress", {}))
        self.assertEqual(len(lines), 2)
        status, = finish.call_args.args[1:2]
        self.assertEqual(status, "interrupted")
        self.assertEqual(finish.call_args.kwargs["result"]["sweep_levels"][0]["level"], 1)


if __name__ == "__main__":
    unittest.main()
