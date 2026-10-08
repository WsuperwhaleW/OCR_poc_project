"""The model server's own figures, phase by phase: prefill, decode, queue, latency.

What matters is the attribution rule -- a vLLM /metrics change is taken only when
it covers exactly one request, and a server that is not vLLM is never asked
/metrics at all -- and that no figure is derived from the app's own clock.
"""

import unittest
from unittest import mock

import app
import runlog
import servertime

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m"}
LLAMA = {"kind": "llama.cpp", "url": "http://local:8080", "reachable": True, "model": "m"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "reachable": True, "model": "m"}


def _snap(count, total, queue=None, prefill=None, decode=None):
    out = {"vllm:e2e_request_latency_seconds_count": float(count),
           "vllm:e2e_request_latency_seconds_sum": float(total)}
    for name, value in (("queue", queue), ("prefill", prefill), ("decode", decode)):
        if value is not None:
            out[f"vllm:request_{name}_time_seconds_count"] = float(count)
            out[f"vllm:request_{name}_time_seconds_sum"] = float(value)
    return out


class FromBodyTests(unittest.TestCase):
    def test_llama_cpp_timings(self):
        got = servertime.from_body({"timings": {"prompt_ms": 1500, "predicted_ms": 500}})
        self.assertAlmostEqual(got["seconds"], 2.0)
        self.assertAlmostEqual(got["prefill"], 1.5)
        self.assertAlmostEqual(got["decode"], 0.5)
        self.assertEqual(got["source"], "llama.cpp timings")

    def test_ollama_native_durations(self):
        got = servertime.from_body({"total_duration": 3_000_000_000,
                                    "prompt_eval_duration": 500_000_000,
                                    "eval_duration": 1_000_000_000})
        self.assertAlmostEqual(got["seconds"], 3.0)
        self.assertAlmostEqual(got["prefill"], 0.5)
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

    def test_one_request_reports_each_phase(self):
        rec, _ = self._clock([_snap(10, 100.0, 4.0, 20.0, 70.0),
                              _snap(11, 103.0, 4.5, 21.0, 71.25)])
        self.assertAlmostEqual(rec["server_seconds"], 3.0)
        self.assertAlmostEqual(rec["server_queue_seconds"], 0.5)
        self.assertAlmostEqual(rec["server_prefill_seconds"], 1.0)
        self.assertAlmostEqual(rec["server_decode_seconds"], 1.25)
        self.assertEqual(rec["source"], "vllm /metrics")

    def test_nothing_is_derived_from_the_app_clock(self):
        rec, _ = self._clock([_snap(10, 100.0), _snap(11, 103.0)], client=99.0)
        self.assertNotIn("network_seconds", rec)
        self.assertNotIn(99.0, rec.values())

    def test_a_phase_the_server_did_not_report_is_blank(self):
        rec, _ = self._clock([_snap(10, 100.0), _snap(11, 103.0)])
        self.assertAlmostEqual(rec["server_seconds"], 3.0)
        self.assertIsNone(rec["server_prefill_seconds"])

    def test_concurrent_requests_are_not_attributed(self):
        rec, _ = self._clock([_snap(10, 100.0, prefill=1.0), _snap(12, 107.0, prefill=3.0)])
        self.assertIsNone(rec["server_seconds"])
        self.assertIsNone(rec["server_prefill_seconds"])
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
        self.assertAlmostEqual(rec["server_prefill_seconds"], 0.8)
        self.assertAlmostEqual(rec["server_decode_seconds"], 0.2)
        self.assertIsNone(rec["server_queue_seconds"])      # llama.cpp has none


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
        self.assertAlmostEqual(got["server_seconds"], 1.0)
        self.assertAlmostEqual(got["server_prefill_seconds"], 1.0)
        self.assertNotIn("network_seconds", got)
        self.assertNotIn("client_seconds", got)

    def test_a_phase_some_requests_lack_is_blank(self):
        previous = servertime.open_tally()
        try:
            a = servertime.Clock(LLAMA)
            a.timings({"prompt_ms": 400, "predicted_ms": 600})
            a.finish()
            with mock.patch.object(servertime, "_fetch",
                                   side_effect=[_snap(1, 1.0, 0.0, 0.0, 0.0),
                                                _snap(2, 3.0, 0.5, 0.5, 1.0)]):
                servertime.Clock(VLLM).finish()
        finally:
            got = servertime.close_tally(previous)
        self.assertAlmostEqual(got["server_seconds"], 3.0)
        self.assertAlmostEqual(got["server_prefill_seconds"], 0.9)
        self.assertAlmostEqual(got["server_decode_seconds"], 1.6)
        self.assertIsNone(got["server_queue_seconds"])     # llama.cpp had none

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
        self.assertAlmostEqual(got["server_prefill_seconds"], 0.4)
        self.assertEqual(got["source"], "llama.cpp timings")


class RunLogTests(unittest.TestCase):
    def test_runlog_names_the_same_fields(self):
        # runlog cannot import servertime (import cycle), so it spells them out.
        self.assertEqual(runlog._SERVER_FIELDS, servertime.FIELDS)

    def test_phase_columns_appended_at_the_end(self):
        tail = runlog.COLUMNS[runlog.COLUMNS.index("extract_parallel") + 1:]
        self.assertEqual(tail,
                         ["server_queue_seconds", "server_prefill_seconds",
                          "server_decode_seconds", "extract_server_queue_seconds",
                          "extract_server_prefill_seconds",
                          "extract_server_decode_seconds", "server_prompt_tokens", "server_cached_tokens",
                          "server_prefill_tokens", "server_generated_tokens",
                          "server_decode_tokens", "server_prefill_tps",
                          "server_decode_tps", "image_bytes",
                          "extract_server_prompt_tokens",
                          "extract_server_cached_tokens",
                          "extract_server_prefill_tokens",
                          "extract_server_generated_tokens",
                          "extract_server_decode_tokens",
                          "extract_server_prefill_tps",
                          "extract_server_decode_tps",
                          "model_quant", "inference_engine",
                          "extract_quant", "extract_inference_engine"])
        # The retired hop columns stay in the file format, written blank.
        self.assertIn("network_seconds", runlog.COLUMNS)
        self.assertIn("extract_network_seconds", runlog.COLUMNS)

    def test_extract_cells_blank_when_untimed(self):
        cells = runlog._server_time_cells({"requests": 3, "attributed": 0,
                                           "server_seconds": None, "source": ""})
        self.assertEqual(set(cells.values()), {""})

    def test_extract_cells_written_when_timed(self):
        cells = runlog._server_time_cells({"requests": 3, "attributed": 3,
                                           "server_seconds": 4.2,
                                           "server_queue_seconds": 0.3,
                                           "server_prefill_seconds": 1.1,
                                           "server_decode_seconds": 2.7,
                                           "source": "vllm /metrics"})
        self.assertEqual(cells["extract_server_seconds"], 4.2)
        self.assertEqual(cells["extract_server_queue_seconds"], 0.3)
        self.assertEqual(cells["extract_server_prefill_seconds"], 1.1)
        self.assertEqual(cells["extract_server_decode_seconds"], 2.7)
        self.assertEqual(cells["extract_network_seconds"], "")
        self.assertEqual(cells["extract_server_timing"], "vllm /metrics")

    def test_summarise_sums_only_timed_pages(self):
        got = app.summarise([{"server_seconds": 2.0, "server_queue_seconds": 0.1,
                              "server_prefill_seconds": 0.6,
                              "server_decode_seconds": 1.3,
                              "server_timing": "vllm /metrics"},
                             {"server_seconds": 1.0, "server_queue_seconds": None,
                              "server_prefill_seconds": 0.2,
                              "server_decode_seconds": 0.8},
                             {"server_seconds": None}],
                            "low", 0.0)
        self.assertEqual(got["server_seconds"], 3.0)
        self.assertEqual(got["server_prefill_seconds"], 0.8)
        self.assertEqual(got["server_decode_seconds"], 2.1)
        self.assertIsNone(got["server_queue_seconds"])     # one page lacked it
        self.assertNotIn("network_seconds", got)
        self.assertEqual(got["server_timed_pages"], 2)

    def test_the_row_carries_the_phases_and_no_hops(self):
        import csv
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "runs.csv"
            with mock.patch.object(runlog, "LOG_PATH", path):
                runlog.record({"server_seconds": 3.0, "server_queue_seconds": None,
                               "server_prefill_seconds": 0.8,
                               "server_decode_seconds": 2.1,
                               "server_timing": "vllm /metrics",
                               "network_seconds": 9.9},
                              {"name": "x.pdf"})
            with path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["server_seconds"], "3.0")
        self.assertEqual(row["server_prefill_seconds"], "0.8")
        self.assertEqual(row["server_decode_seconds"], "2.1")
        self.assertEqual(row["server_queue_seconds"], "")
        self.assertEqual(row["network_seconds"], "")      # retired: never written


class TokenTests(unittest.TestCase):
    """Tokens and the two rates: prefill counts what was COMPUTED."""

    def _vllm(self, snaps, usage=None):
        with mock.patch.object(servertime, "_fetch", side_effect=snaps), \
                mock.patch.object(servertime.time, "sleep"):
            clock = servertime.Clock(VLLM)
            if usage is not None:
                clock.usage(usage)
            return clock.finish()

    def test_vllm_usage_with_cached_detail(self):
        rec = self._vllm([_snap(1, 0.0, 0.0, 0.0, 0.0), _snap(2, 3.0, 0.0, 1.0, 2.0)],
                         {"prompt_tokens": 2600, "completion_tokens": 201,
                          "prompt_tokens_details": {"cached_tokens": 600}})
        self.assertEqual(rec["server_prompt_tokens"], 2600)       # image included
        self.assertEqual(rec["server_cached_tokens"], 600)
        self.assertEqual(rec["server_prefill_tokens"], 2000)
        self.assertEqual(rec["server_decode_tokens"], 200)        # first token is prefill's
        self.assertEqual(rec["server_prefill_tps"], 2000.0)
        self.assertEqual(rec["server_decode_tps"], 100.0)

    def test_vllm_cached_from_prefix_counter_only_when_idle(self):
        def snap(count, hits, running):
            out = _snap(count, 3.0 * count, 0.0, 1.0 * count, 2.0 * count)
            out.update({"vllm:prefix_cache_hits_total": float(hits),
                        "vllm:num_requests_running": float(running),
                        "vllm:num_requests_waiting": 0.0})
            return out
        usage = {"prompt_tokens": 1000, "completion_tokens": 51}
        rec = self._vllm([snap(1, 100, 0), snap(2, 400, 0)], usage)
        self.assertEqual(rec["server_cached_tokens"], 300)
        self.assertEqual(rec["server_prefill_tps"], 700.0)
        # Something else running: the counter moved for it too, so no guess.
        rec = self._vllm([snap(1, 100, 1), snap(2, 400, 0)], usage)
        self.assertIsNone(rec["server_cached_tokens"])
        self.assertIsNone(rec["server_prefill_tokens"])
        self.assertIsNone(rec["server_prefill_tps"])
        self.assertEqual(rec["server_decode_tps"], 25.0)          # decode still known

    def test_vllm_tokens_fall_back_to_histograms(self):
        before = _snap(1, 0.0, 0.0, 0.0, 0.0)
        after = _snap(2, 3.0, 0.0, 1.0, 2.0)
        before.update({"vllm:request_prompt_tokens_count": 1.0,
                       "vllm:request_prompt_tokens_sum": 10.0,
                       "vllm:request_generation_tokens_count": 1.0,
                       "vllm:request_generation_tokens_sum": 5.0})
        after.update({"vllm:request_prompt_tokens_count": 2.0,
                      "vllm:request_prompt_tokens_sum": 1510.0,
                      "vllm:request_generation_tokens_count": 2.0,
                      "vllm:request_generation_tokens_sum": 106.0})
        rec = self._vllm([before, after])
        self.assertEqual(rec["server_prompt_tokens"], 1500)
        self.assertEqual(rec["server_generated_tokens"], 101)
        self.assertEqual(rec["server_decode_tps"], 50.0)

    def test_llama_cpp_counts_its_own(self):
        clock = servertime.Clock(LLAMA)
        clock.timings({"prompt_ms": 500, "predicted_ms": 2000, "prompt_n": 1500,
                       "cache_n": 450, "predicted_n": 100})
        clock.usage({"prompt_tokens": 9999, "completion_tokens": 9999})
        rec = clock.finish()
        self.assertEqual(rec["server_prefill_tokens"], 1500)      # timings win
        self.assertEqual(rec["server_cached_tokens"], 450)
        self.assertEqual(rec["server_prompt_tokens"], 1950)
        self.assertEqual(rec["server_prefill_tps"], 3000.0)
        self.assertEqual(rec["server_decode_tps"], 50.0)

    def test_ollama_native_counts(self):
        rec = servertime._record(servertime.from_body(
            {"total_duration": 4_000_000_000, "prompt_eval_duration": 1_000_000_000,
             "eval_duration": 2_000_000_000, "prompt_eval_count": 800,
             "eval_count": 120}))
        self.assertEqual(rec["server_prefill_tps"], 800.0)
        self.assertEqual(rec["server_decode_tps"], 60.0)

    def test_rates_over_pages_are_from_sums_not_averaged(self):
        pages = [{"server_seconds": 2.0, "server_prefill_seconds": 1.0,
                  "server_decode_seconds": 1.0, "server_prefill_tokens": 1000,
                  "server_decode_tokens": 100},
                 {"server_seconds": 4.0, "server_prefill_seconds": 3.0,
                  "server_decode_seconds": 1.0, "server_prefill_tokens": 1000,
                  "server_decode_tokens": 50}]
        got = servertime.summarise_records(pages)
        self.assertEqual(got["server_prefill_tps"], 500.0)        # 2000 / 4, not mean(1000, 333)
        self.assertEqual(got["server_decode_tps"], 75.0)
        self.assertEqual(got["server_prefill_tokens"], 2000)

    def test_batch_decode_tokens_take_one_per_request(self):
        snaps = [_snap(10, 0.0, 0.0, 0.0, 0.0), _snap(13, 9.0, 0.0, 3.0, 6.0)]
        fetched = []

        def fetch(url):
            fetched.append(url)
            return snaps[min(len(fetched) - 1, 1)]

        with mock.patch.object(servertime, "_fetch", fetch), \
                mock.patch.object(servertime.settings, "SERVER_TIMING_WAIT", 0.0):
            previous = servertime.open_tally()
            batch = servertime.Batch(VLLM)

            def one():
                clock = servertime.Clock(VLLM)
                clock.usage({"prompt_tokens": 1000, "completion_tokens": 21,
                             "prompt_tokens_details": {"cached_tokens": 0}})
                clock.finish()

            for _ in range(3):
                batch.run(one)
            out = batch.close()
            tally = servertime.close_tally(previous)
        self.assertEqual(out["server_decode_tokens"], 60)        # 3 x (21 - 1)
        self.assertEqual(tally["server_decode_tps"], 10.0)
        self.assertEqual(tally["server_prefill_tps"], 1000.0)

    def test_image_bytes_on_the_read_summary(self):
        got = app.summarise([{"server_seconds": None, "image_bytes": 1000},
                             {"server_seconds": None, "image_bytes": 2500}],
                            "low", 0.0)
        self.assertEqual(got["image_bytes"], 3500)
        got = app.summarise([{"server_seconds": None, "image_bytes": 1000},
                             {"server_seconds": None}], "low", 0.0)
        self.assertIsNone(got["image_bytes"])


if __name__ == "__main__":
    unittest.main()
