"""A stress run samples /metrics on BOTH servers, the OCR and the extraction one."""
import json
import threading
import time
import unittest
from unittest import mock

import app
import engine
READ = {"kind": "vllm", "url": "http://read:8000", "model": "typhoon"}
EXTRACT = {"kind": "vllm", "url": "http://extract:8000", "model": "qwen"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "model": "gemma"}


class TargetTests(unittest.TestCase):
    def test_two_vllm_servers_are_two_targets(self):
        self.assertEqual(app._stress_targets(READ, EXTRACT), [
            {"role": "reader", "url": "http://read:8000"},
            {"role": "extractor", "url": "http://extract:8000"}])

    def test_one_server_for_both_passes_is_one_target(self):
        self.assertEqual(app._stress_targets(READ, dict(EXTRACT, url=READ["url"])),
                         [{"role": "reader + extractor", "url": "http://read:8000"}])

    def test_a_server_that_is_not_vllm_is_never_asked(self):
        self.assertEqual(app._stress_targets(OLLAMA, EXTRACT),
                         [{"role": "extractor", "url": "http://extract:8000"}])
        self.assertEqual(app._stress_targets(READ, None),
                         [{"role": "reader", "url": "http://read:8000"}])
        self.assertEqual(app._stress_targets(OLLAMA, None), [])

    def test_a_sample_reads_every_target_and_keeps_going_past_a_failure(self):
        asked = []

        def fake(url):
            asked.append(url)
            if "extract" in url:
                raise OSError("down")
            return {"vllm:num_requests_running": 2.0}

        targets = app._stress_targets(READ, EXTRACT)
        with mock.patch.object(app, "_vllm_metrics", side_effect=fake):
            views, snaps, errors = app._stress_sample(targets, {}, 1.0)
        self.assertEqual(asked, ["http://read:8000", "http://extract:8000"])
        self.assertEqual([v["role"] for v in views], ["reader"])
        self.assertIn("http://read:8000", snaps)
        self.assertEqual(len(errors), 1)
        self.assertIn("extractor", errors[0])


class RunWatchesBothTests(unittest.TestCase):
    """An extraction-only run still watches the OCR server, idle beside it."""

    def test_an_extract_run_samples_the_reader_too(self):
        result = {"tokens": 10, "field_score": {"overall": {
            "counts": {"correct": 1, "partial": 0}, "expected": 1}}}
        with mock.patch.object(engine, "llama_status", return_value=READ),              mock.patch.object(app.backends, "extract_status",
                               return_value=dict(EXTRACT, text_available=True,
                                                 text_reason=None)),              mock.patch.object(app, "extract_fields", return_value=result),              mock.patch.object(app, "_vllm_metrics",
                               return_value={"vllm:num_requests_running": 0.0}):
            cfg = app._stress_config({"mode": "extract", "jobs": 1, "seed": 1,
                                      "metrics": True})
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        self.assertEqual([t["role"] for t in events[0]["metrics_servers"]],
                         ["reader", "extractor"])
        self.assertEqual([v["role"] for v in events[-1]["report"]["servers_metrics"]],
                         ["reader", "extractor"])


class SamplerTests(unittest.TestCase):
    """/metrics is read on the clock, on its own thread -- not per finished job."""

    def _run(self, every, seconds, read_time=0.0):
        reads, live, peak, lock = [], [0], [0], threading.Lock()

        def fake(url):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(read_time)
            with lock:
                live[0] -= 1
                reads.append(time.perf_counter())
            return {"vllm:num_requests_running": 1.0}

        targets = app._stress_targets(READ, None)
        with mock.patch.object(app, "_vllm_metrics", side_effect=fake):
            sampler = app._StressSampler(targets, {}, every,
                                         time.perf_counter()).start()
            time.sleep(seconds)
            last = sampler.stop()
            time.sleep(read_time + 0.05)
        return reads, peak[0], sampler, last

    def test_reads_follow_the_interval(self):
        reads, peak, _, last = self._run(every=0.1, seconds=0.55)
        self.assertIn(len(reads), (4, 5, 6))
        self.assertEqual(peak, 1)
        self.assertIn("http://read:8000", last)

    def test_a_slow_read_skips_ticks_instead_of_piling_up(self):
        reads, peak, _, _ = self._run(every=0.05, seconds=0.5, read_time=0.2)
        self.assertEqual(peak, 1)
        self.assertLessEqual(len(reads), 3)

    def test_take_hands_each_sample_out_once(self):
        _, _, sampler, _ = self._run(every=0.05, seconds=0.12)
        self.assertIsNotNone(sampler.take())
        self.assertIsNone(sampler.take())

    def test_no_targets_reads_nothing(self):
        with mock.patch.object(app, "_vllm_metrics") as fake:
            sampler = app._StressSampler([], {}, 0.01, time.perf_counter()).start()
            time.sleep(0.05)
            sampler.stop()
        fake.assert_not_called()


if __name__ == "__main__":
    unittest.main()
