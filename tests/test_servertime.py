"""The model server's own time beside the app's, and the hop time between them.

What matters is the attribution rule: a vLLM /metrics change is taken only when
it covers exactly one request, and a server that is not vLLM is never asked
/metrics at all.
"""

import unittest
from unittest import mock

import app
import runlog
import servertime

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m"}
LLAMA = {"kind": "llama.cpp", "url": "http://local:8080", "reachable": True, "model": "m"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "reachable": True, "model": "m"}


def _snap(count, total, queue=None):
    out = {"vllm:e2e_request_latency_seconds_count": float(count),
           "vllm:e2e_request_latency_seconds_sum": float(total)}
    if queue is not None:
        out["vllm:request_queue_time_seconds_count"] = float(count)
        out["vllm:request_queue_time_seconds_sum"] = float(queue)
    return out


class FromBodyTests(unittest.TestCase):
    def test_llama_cpp_timings(self):
        got = servertime.from_body({"timings": {"prompt_ms": 1500, "predicted_ms": 500}})
        self.assertAlmostEqual(got["seconds"], 2.0)
        self.assertEqual(got["source"], "llama.cpp timings")

    def test_ollama_native_durations(self):
        got = servertime.from_body({"total_duration": 3_000_000_000,
                                    "eval_duration": 1_000_000_000})
        self.assertAlmostEqual(got["seconds"], 3.0)
        self.assertAlmostEqual(got["decode"], 1.0)
        self.assertEqual(got["source"], "ollama")

    def test_openai_reply_carries_nothing(self):
        self.assertIsNone(servertime.from_body({"choices": [], "usage": {}}))
        self.assertIsNone(servertime.from_body(None))


class VllmClockTests(unittest.TestCase):
    def _clock(self, snaps, client=5.0):
        with mock.patch.object(servertime, "_fetch", side_effect=snaps) as fetch, \
                mock.patch.object(servertime.time, "sleep"):
            clock = servertime.Clock(VLLM)
            rec = clock.finish(client)
        return rec, fetch

    def test_one_request_is_attributed(self):
        rec, _ = self._clock([_snap(10, 100.0, 4.0), _snap(11, 103.0, 4.5)])
        self.assertAlmostEqual(rec["server_seconds"], 3.0)
        self.assertAlmostEqual(rec["network_seconds"], 2.0)
        self.assertAlmostEqual(rec["server_queue"], 0.5)
        self.assertEqual(rec["source"], "vllm /metrics")

    def test_concurrent_requests_are_not_attributed(self):
        rec, _ = self._clock([_snap(10, 100.0), _snap(12, 107.0)])
        self.assertIsNone(rec["server_seconds"])
        self.assertIsNone(rec["network_seconds"])
        self.assertIn("concurrent", rec["why"])

    def test_waits_for_the_stats_logger(self):
        rec, fetch = self._clock([_snap(10, 100.0), _snap(10, 100.0),
                                  _snap(11, 102.5)])
        self.assertAlmostEqual(rec["server_seconds"], 2.5)
        self.assertEqual(fetch.call_count, 3)

    def test_failed_snapshot_is_blank_not_zero(self):
        rec, _ = self._clock([None, None])
        self.assertIsNone(rec["server_seconds"])


class GateTests(unittest.TestCase):
    def test_metrics_never_asked_of_a_non_vllm_server(self):
        for info in (LLAMA, OLLAMA, {"kind": "openai", "url": "http://x"}):
            with mock.patch.object(servertime, "_fetch") as fetch:
                clock = servertime.Clock(info)
                clock.finish(1.0)
            fetch.assert_not_called()

    def test_switched_off_asks_nothing(self):
        with mock.patch.object(servertime.settings, "SERVER_TIMING", False), \
                mock.patch.object(servertime, "_fetch") as fetch:
            rec = servertime.Clock(VLLM).finish(1.0)
        fetch.assert_not_called()
        self.assertIsNone(rec["server_seconds"])

    def test_llama_body_is_used_when_not_vllm(self):
        clock = servertime.Clock(LLAMA)
        clock.timings({"prompt_ms": 800, "predicted_ms": 200})
        rec = clock.finish(1.3)
        self.assertAlmostEqual(rec["server_seconds"], 1.0)
        self.assertAlmostEqual(rec["network_seconds"], 0.3)


class TallyTests(unittest.TestCase):
    def test_sums_only_the_attributed_requests(self):
        previous = servertime.open_tally()
        try:
            a = servertime.Clock(LLAMA)
            a.timings({"prompt_ms": 1000, "predicted_ms": 0})
            a.finish(1.5)
            servertime.Clock(OLLAMA).finish(9.0)      # /v1: no server figure
        finally:
            got = servertime.close_tally(previous)
        self.assertEqual(got["requests"], 2)
        self.assertEqual(got["attributed"], 1)
        self.assertAlmostEqual(got["client_seconds"], 1.5)
        self.assertAlmostEqual(got["server_seconds"], 1.0)
        self.assertAlmostEqual(got["network_seconds"], 0.5)

    def test_no_tally_open_records_nowhere(self):
        servertime.Clock(LLAMA).finish(1.0)       # must not raise
        self.assertIsNone(servertime.close_tally(None))

    def test_timed_post_lands_in_the_tally(self):
        class Res:
            status_code = 200

            def json(self):
                return {"choices": [], "timings": {"prompt_ms": 400, "predicted_ms": 100}}

        previous = servertime.open_tally()
        try:
            with mock.patch.object(app.requests, "post", return_value=Res()):
                app._timed_post("http://local:8080/v1/chat/completions", {}, LLAMA)
        finally:
            got = servertime.close_tally(previous)
        self.assertEqual(got["attributed"], 1)
        self.assertAlmostEqual(got["server_seconds"], 0.5)
        self.assertEqual(got["source"], "llama.cpp timings")


class RunLogTests(unittest.TestCase):
    def test_columns_appended_at_the_end(self):
        self.assertEqual(runlog.COLUMNS[-7:-1],
                         ["server_seconds", "network_seconds", "server_timing",
                          "extract_server_seconds", "extract_network_seconds",
                          "extract_server_timing"])

    def test_extract_cells_blank_when_untimed(self):
        cells = runlog._server_time_cells({"requests": 3, "attributed": 0,
                                           "server_seconds": None,
                                           "network_seconds": None, "source": ""})
        self.assertEqual(set(cells.values()), {""})

    def test_extract_cells_written_when_timed(self):
        cells = runlog._server_time_cells({"requests": 3, "attributed": 3,
                                           "server_seconds": 4.2,
                                           "network_seconds": 0.6,
                                           "source": "vllm /metrics"})
        self.assertEqual(cells["extract_server_seconds"], 4.2)
        self.assertEqual(cells["extract_network_seconds"], 0.6)
        self.assertEqual(cells["extract_server_timing"], "vllm /metrics")

    def test_summarise_sums_only_timed_pages(self):
        got = app.summarise([{"server_seconds": 2.0, "network_seconds": 0.5,
                              "server_timing": "vllm /metrics"},
                             {"server_seconds": None, "network_seconds": None}],
                            "low", 0.0)
        self.assertEqual(got["server_seconds"], 2.0)
        self.assertEqual(got["network_seconds"], 0.5)
        self.assertEqual(got["server_timed_pages"], 1)


if __name__ == "__main__":
    unittest.main()
