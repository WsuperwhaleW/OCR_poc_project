"""History persists exact snapshots and terminal states independently of the CSV."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app
import config
import runhistory


def _ok_job(*_args):
    """A stress job (`_stress_job_steps` is a generator) that finishes at once."""
    return {"status": "ok"}
    yield                                   # makes it a generator


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patch = patch.object(config, "LOG_DIR", Path(self.directory.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = app.app.test_client()

    def stream(self, events, configuration=None):
        return app._history_stream((json.dumps(e) + "\n" for e in events),
                                   "stress", configuration or {"concurrency": 4, "jobs": 20})

    def monitored_stream(self, events, configuration):
        """Let one background sampling cycle finish before ending a short run."""
        sampled = threading.Event()
        collect = app._HistoryMonitor._sample

        def sample(sampler):
            collect(sampler)
            sampled.set()

        def source():
            yield json.dumps(events[0])
            self.assertTrue(sampled.wait(2), "History monitor did not complete a reading")
            for event in events[1:]:
                yield json.dumps(event)

        with patch.object(app._HistoryMonitor, "_sample", sample):
            return [json.loads(line) for line in app._history_stream(source(), "stress", configuration)]

    def test_round_trip_and_list_does_not_include_large_report(self):
        configuration = {"concurrency": 4, "jobs": 20, "label": "ทดสอบ"}
        run_id = runhistory.start("stress", configuration)
        configuration["jobs"] = 999
        plan = {"mode": "full", "seed": 42}
        result = {"report": {"latency": {"p99": 1.234}, "missing": None}}
        runhistory.save_plan(run_id, plan)
        runhistory.finish(run_id, "completed", result)
        item = runhistory.get(run_id)
        self.assertEqual(item["configuration"]["jobs"], 20)
        self.assertEqual(item["configuration"]["label"], "ทดสอบ")
        self.assertEqual(item["plan"], plan)
        self.assertEqual(item["result"], result)
        self.assertIsNotNone(item["finished_at"])
        self.assertNotIn("result", runhistory.list_runs()["runs"][0])

    def test_pagination_and_concurrent_writes_keep_every_run(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda i: runhistory.start("stress", {"jobs": i}), range(8)))
        first, second = runhistory.list_runs(4, 0), runhistory.list_runs(4, 4)
        self.assertEqual(first["total"], 8)
        self.assertEqual({r["id"] for r in first["runs"] + second["runs"]}, set(ids))

    def test_unfinished_runs_from_other_instances_are_not_declared_failed(self):
        pending = runhistory.start("stress", {})
        done = runhistory.start("stress", {})
        runhistory.finish(done, "completed", {"report": {}})
        with patch.object(runhistory, "PROCESS_ID", "new-process"):
            self.assertEqual(runhistory.get(pending)["status"], "unfinished")
            self.assertEqual(runhistory.get(done)["status"], "completed")

    def test_completion_saved_before_client_receives_done_and_not_overwritten(self):
        stream = self.stream([{"event": "plan", "seed": 42},
                              {"event": "done", "report": {"error_rate": 0}}])
        plan = json.loads(next(stream))
        done = json.loads(next(stream))
        self.assertEqual(plan["history_id"], done["history_id"])
        self.assertEqual(runhistory.get(done["history_id"])["status"], "completed")
        stream.close()
        runhistory.finish(done["history_id"], "error", error="late failure")
        self.assertEqual(runhistory.get(done["history_id"])["status"], "completed")

    def test_stopped_and_error_reports_remain_available(self):
        for stopped, rate, expected in [(True, 0, "stopped"),
                                         (False, 0.5, "completed_with_errors")]:
            events = [json.loads(line) for line in self.stream([
                {"event": "plan"}, {"event": "done", "stopped": stopped,
                                     "report": {"error_rate": rate, "errors": ["failure"]}}])]
            item = runhistory.get(events[-1]["history_id"])
            self.assertEqual(item["status"], expected)
            self.assertEqual(item["result"]["report"]["errors"], ["failure"])

    def test_disconnect_and_missing_completion_are_not_success(self):
        stream = self.stream([{"event": "plan"}, {"event": "progress", "done": 2}])
        run_id = json.loads(next(stream))["history_id"]
        stream.close()
        self.assertEqual(runhistory.get(run_id)["status"], "interrupted")
        events = [json.loads(line) for line in self.stream([{"event": "plan"}])]
        self.assertEqual(runhistory.get(events[0]["history_id"])["status"], "interrupted")

    def test_failed_stream_keeps_progress_and_error(self):
        def broken():
            yield json.dumps({"event": "progress", "done": 3})
            raise ValueError("engine unavailable")
        stream = app._history_stream(broken(), "stress", {})
        run_id = json.loads(next(stream))["history_id"]
        with self.assertRaises(ValueError):
            next(stream)
        item = runhistory.get(run_id)
        self.assertEqual(item["status"], "error")
        self.assertEqual(item["progress"]["done"], 3)
        self.assertIn("engine unavailable", item["error"])

    def test_storage_failure_warns_without_discarding_report(self):
        with patch.object(runhistory, "start", side_effect=OSError("read-only disk")):
            events = [json.loads(line) for line in self.stream([
                {"event": "done", "report": {"completed": 5}}])]
        self.assertEqual(events[-1]["report"]["completed"], 5)
        self.assertIn("read-only disk", events[-1]["history_error"])

    def test_saved_monitor_snapshots_include_latest_completed_reading_and_gpu(self):
        cfg = {"history_monitor": True, "metrics_every": 5, "gpu_url": "http://gpu",
               "reader": {"kind": "vllm", "url": "http://vllm"}}
        sample = {"at": 123, "snapshot": {"kv_free": 0.5, "running": 4},
                  "gpu": {"free_mb": 3584}, "rates": {"generation_tps": 100}}
        with patch.object(app, "_monitor_scrape", return_value=(sample, 200)) as scrape:
            events = self.monitored_stream([
                {"event": "plan"}, {"event": "progress"},
                {"event": "done", "report": {}}], cfg)
        item = runhistory.get(events[-1]["history_id"])
        self.assertEqual(len(item["result"]["monitor_samples"]), 1)
        self.assertEqual(item["result"]["monitor_samples"][-1], {**sample, "role": "reader"})
        scrape.assert_called_with({**cfg["reader"], "role": "reader"}, "http://vllm", "http://gpu")

    def test_monitor_never_probes_non_vllm_or_disabled_sampling(self):
        for kind, enabled in [("ollama", True), ("vllm", False)]:
            with patch.object(app, "_monitor_scrape") as scrape:
                list(self.stream([{"event": "done", "report": {}}], {
                    "history_monitor": enabled, "reader": {"kind": kind, "url": "http://x"}}))
            scrape.assert_not_called()

    def test_monitor_retains_latest_300_with_total_sample_count(self):
        cfg = {"history_monitor": True, "metrics_every": 0,
               "reader": {"kind": "vllm", "url": "http://vllm"}}
        samples = iter(({"at": i, "snapshot": {}}, 200) for i in range(306))
        sampler = app._HistoryMonitor(cfg)
        with patch.object(app, "_monitor_scrape", side_effect=lambda *_: next(samples)):
            for _ in range(306):
                sampler._sample()
        retained, total = sampler.stop()
        self.assertEqual(len(retained), 300)
        self.assertEqual(retained[0]["at"], 6)
        self.assertEqual(retained[-1]["at"], 305)
        self.assertEqual(total, 306)

    def test_both_monitor_servers_and_live_alerts_are_retained(self):
        cfg = {"history_monitor": True, "metrics_every": 5,
               "reader": {"kind": "vllm", "url": "http://reader"},
               "extractor": {"kind": "vllm", "url": "http://extractor"},
               "gpu_url": "http://reader-gpu", "gpu_extract_url": "http://extractor-gpu"}
        alert = {"event": "alert", "job": 2, "case": "sol002", "message": "looped"}
        with patch.object(app, "_monitor_scrape", side_effect=lambda info, url, gpu: (
                {"url": url, "gpu_url": gpu, "snapshot": {}, "at": 1}, 200)):
            events = self.monitored_stream([
                {"event": "plan"}, alert, {"event": "done", "report": {}}], cfg)
        result = runhistory.get(events[-1]["history_id"])["result"]
        self.assertEqual(result["alerts"], [alert])
        self.assertEqual({s["url"] for s in result["monitor_samples"]},
                         {"http://reader", "http://extractor"})
        self.assertEqual(result["monitor_samples"][-1]["gpu_url"], "http://extractor-gpu")

    def test_slow_monitor_does_not_block_completion_stop_error_or_disconnect(self):
        for terminal in ("completed", "stopped", "error", "disconnect"):
            with self.subTest(terminal=terminal):
                entered, release = threading.Event(), threading.Event()
                samplers = []
                monitor_class = app._HistoryMonitor
                cfg = {"history_monitor": True, "metrics_every": 5,
                       "reader": {"kind": "vllm", "url": "http://slow-metrics"},
                       "concurrency": 1, "max_seconds": 0, "save_log": False}

                def make_sampler(configuration):
                    sampler = monitor_class(configuration)
                    samplers.append(sampler)
                    return sampler

                def slow_scrape(*_):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("mock metrics timed out")
                    return {"snapshot": {"running": 0}}, 200

                def source():
                    yield json.dumps({"event": "plan"})
                    if not entered.wait(2):
                        raise AssertionError("monitor never started")
                    yield json.dumps({"event": "progress", "done": 1})
                    if terminal == "error":
                        raise ValueError("mock job failed")
                    _, wall, _, _, _ = yield from app._stress_phase(
                        [{"index": 0}], cfg, {}, app.MultiCancel(), "run")
                    yield json.dumps({"event": "done", "stopped": terminal == "stopped",
                                      "report": {"wall_seconds": wall}})

                def consume():
                    stream, events = app._history_stream(source(), "stress", cfg), []
                    try:
                        for line in stream:
                            events.append(json.loads(line))
                            if terminal == "disconnect" and events[-1]["event"] == "progress":
                                stream.close()
                                break
                    except ValueError as err:
                        if terminal != "error":
                            raise
                        self.assertEqual(str(err), "mock job failed")
                    return events

                with patch.object(app, "_HistoryMonitor", side_effect=make_sampler), \
                     patch.object(app, "_monitor_scrape", side_effect=slow_scrape), \
                     patch.object(app, "_stress_job_steps", _ok_job), \
                     ThreadPoolExecutor(max_workers=1) as pool:
                    consumer = pool.submit(consume)
                    try:
                        self.assertTrue(entered.wait(2))
                        # The scrape stays blocked throughout progress, timing,
                        # History persistence and client completion/closure.
                        events = consumer.result(timeout=1)
                        item = runhistory.get(events[0]["history_id"])
                        self.assertEqual(item["status"],
                                         "interrupted" if terminal == "disconnect" else terminal)
                        self.assertEqual(item["result"]["monitor_samples"], [])
                        if terminal in ("completed", "stopped"):
                            self.assertEqual(events[-1]["event"], "done")
                            self.assertLess(events[-1]["report"]["wall_seconds"], 0.5)
                    finally:
                        release.set()
                        for sampler in samplers:
                            sampler._thread.join(2)
                            self.assertFalse(sampler._thread.is_alive())
                    # Late metrics cannot mutate the saved report.
                    self.assertEqual(runhistory.get(events[0]["history_id"]), item)
                    self.assertEqual(samplers[0].stop(), ([], 0))

    def test_monitor_samples_on_its_clock_and_keeps_workspace_settings(self):
        workspace = app.workspace
        ws = workspace.Workspace("history-monitor-tab")
        ws.put("loop_guard", False)
        second = threading.Event()
        records = []
        cfg = {"history_monitor": True, "metrics_every": 1,
               "reader": {"kind": "vllm", "url": "http://clock-metrics"}}
        sampler = None

        def scrape(*_):
            records.append((workspace.current_id(), app.loop_guard()))
            if len(records) == 2:
                second.set()
            return {"snapshot": {}}, 200

        with workspace.use(ws), patch.object(app, "_monitor_scrape", side_effect=scrape):
            sampler = app._HistoryMonitor(cfg).start()
            try:
                # No source/progress events drive this sampler.
                self.assertTrue(second.wait(3))
            finally:
                sampler.stop()
                sampler._thread.join(2)
        self.assertEqual(records[:2], [(ws.id, False), (ws.id, False)])

    def test_monitor_deduplicates_servers_and_saves_scrape_errors(self):
        info = {"kind": "vllm", "url": "http://same-server"}
        sampler = app._HistoryMonitor({"history_monitor": True,
                                       "reader": info, "extractor": info})
        with patch.object(app, "_monitor_scrape", side_effect=TimeoutError("slow metrics")) as scrape:
            sampler._sample()
        samples, total = sampler.stop()
        self.assertEqual(scrape.call_count, 1)
        self.assertEqual(total, 1)
        self.assertEqual(samples[0]["role"], "reader")
        self.assertEqual(samples[0]["url"], info["url"])
        self.assertIn("slow metrics", samples[0]["error"])

    def test_slow_monitor_skips_missed_ticks_instead_of_catching_up(self):
        sampler = app._HistoryMonitor({"history_monitor": True, "metrics_every": 1})
        with patch.object(app.time, "monotonic", side_effect=[0.0, 3.4, 3.4]), \
             patch.object(sampler, "_sample") as scrape, \
             patch.object(sampler._stop, "wait", return_value=True) as pause:
            sampler._run()
        self.assertEqual(scrape.call_count, 1)
        pause.assert_called_once()
        self.assertAlmostEqual(pause.call_args.args[0], 0.6)

    def test_monitor_stop_does_not_start_another_server_scrape(self):
        sampler = app._HistoryMonitor({
            "history_monitor": True,
            "reader": {"kind": "vllm", "url": "http://reader"},
            "extractor": {"kind": "vllm", "url": "http://extractor"}})

        def stopped_during_read(*_):
            sampler.stop()
            return {"snapshot": {}}, 200

        with patch.object(app, "_monitor_scrape", side_effect=stopped_during_read) as scrape:
            sampler._sample()
        self.assertEqual(scrape.call_count, 1)
        self.assertEqual(sampler.stop(), ([], 0))

    def test_routes_return_exact_saved_data_and_json_download(self):
        run_id = runhistory.start("stress", {"concurrency": 8, "jobs": 50})
        runhistory.finish(run_id, "completed", {"report": {"completed": 50}})
        listed = self.client.get('/api/history?limit=1').get_json()
        self.assertEqual(listed["runs"][0]["id"], run_id)
        result = self.client.get('/api/history/' + run_id + '?download=1')
        self.assertEqual(result.get_json()["result"]["report"]["completed"], 50)
        self.assertIn('attachment;', result.headers["Content-Disposition"])
        self.assertEqual(self.client.get('/api/history/invalid').status_code, 404)
        self.assertEqual(self.client.get('/api/history?limit=oops').status_code, 400)

    def test_stress_endpoint_persists_effective_settings_without_csv_rows(self):
        info = {"kind": "ollama", "url": "http://extract", "model": "model",
                "text_available": True, "text_reason": ""}
        source = lambda *_: (json.dumps(e) + '\n' for e in [
            {"event": "plan", "seed": 42}, {"event": "done", "report": {"error_rate": 0}}])
        with patch.object(app.backends, "extract_status", return_value=info), \
             patch.object(app, "_stress_run", side_effect=source), \
             patch.object(app.runlog, "record", side_effect=AssertionError("CSV changed")):
            result = self.client.post('/api/stress/stream', json={
                "mode": "extract", "concurrency": 3, "jobs": 5, "warmup": 2})
            events = [json.loads(line) for line in result.data.splitlines()]
        item = runhistory.get(events[-1]["history_id"])
        self.assertEqual(item["configuration"]["concurrency"], 3)
        self.assertEqual(item["configuration"]["jobs"], 5)
        self.assertEqual(item["configuration"]["warmup"], 2)
        self.assertEqual(item["configuration"]["extractor"]["url"], "http://extract")


if __name__ == '__main__':
    unittest.main()
