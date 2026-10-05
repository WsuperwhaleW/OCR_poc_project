"""Agentic steps asked at once on vLLM, one at a time everywhere else.

Two things are pinned: the gate (vLLM only, `AGENTIC_PARALLEL` on), and that a
concurrent run produces exactly what a sequential one does -- the same fields,
the same step records, the same replies in the same order. Plus the server-time
rule for a concurrent batch: one /metrics span for the lot, attributed only where
the count moved by exactly the batch.
"""

import threading
import unittest
from unittest import mock

import app
import prompts
import runlog
import servertime

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "reachable": True, "model": "m"}
LLAMA = {"kind": "llama.cpp", "url": "http://local:8080", "reachable": True, "model": "m"}

CODES = ["INVOICE"]
FORM = {"doc_types": CODES, "items": (),
        "keys": list(prompts.fields_for_types(CODES)),
        "mandatory": list(prompts.mandatory_for_types(CODES))}


def _value(key):
    return f"VAL{abs(hash(key)) % 100000:05d}{key.upper()}"


# Every scalar answer is printed in the transcript, so no step is re-asked.
TEXT = "header\n" + "\n".join(_value(k) for k in FORM["keys"]) + "\n"


def _fake_ask(barrier=None, calls=None):
    def ask(content, step, status, collect=None):
        if calls is not None:
            calls.append((step["id"], threading.current_thread().name))
        if barrier is not None:
            barrier.wait(timeout=5)     # BrokenBarrierError unless all are in flight
        values = {k: ([] if k == "other_fields" else _value(k)) for k in step["keys"]}
        raw = f'{{"step": "{step["id"]}"}}'
        if collect is not None:
            collect.append(raw)
        return values, [raw], False, 3, False
    return ask


def _run(status, ask):
    with mock.patch.object(app, "_ask_step", ask):
        events, gen = [], app._extract_agentic(TEXT, status, FORM)
        while True:
            try:
                events.append(next(gen))
            except StopIteration as stop:
                return stop.value, events


class GateTests(unittest.TestCase):
    def test_vllm_runs_every_step_at_once(self):
        self.assertEqual(app._agentic_parallelism(VLLM, 7), 7)

    def test_local_servers_stay_sequential(self):
        self.assertEqual(app._agentic_parallelism(OLLAMA, 7), 1)
        self.assertEqual(app._agentic_parallelism(LLAMA, 7), 1)
        self.assertEqual(app._agentic_parallelism(
            {**VLLM, "kind": "openai"}, 7), 1)

    def test_switch_and_cap(self):
        with mock.patch.object(app, "AGENTIC_PARALLEL", False):
            self.assertEqual(app._agentic_parallelism(VLLM, 7), 1)
        with mock.patch.object(app, "AGENTIC_PARALLEL_MAX", 3):
            self.assertEqual(app._agentic_parallelism(VLLM, 7), 3)
        self.assertEqual(app._agentic_parallelism(VLLM, 1), 1)

    def test_unreachable_vllm_is_not_parallel(self):
        self.assertEqual(app._agentic_parallelism({**VLLM, "reachable": False}, 7), 1)


class ConcurrentRunTests(unittest.TestCase):
    def setUp(self):
        self.table = app.steps_for_types(CODES)
        self.assertGreater(len(self.table), 1)

    def test_steps_are_in_flight_together_on_vllm(self):
        # A barrier every step must reach before any can return: a sequential
        # walk would break it on the first step.
        barrier = threading.Barrier(len(self.table))
        result, events = _run(VLLM, _fake_ask(barrier))
        self.assertNotIn("error", result)
        self.assertEqual(result["parallel"], len(self.table))
        self.assertEqual(events[0]["parallel"], len(self.table))
        self.assertTrue(all(not s.get("error") for s in result["steps"]))

    def test_ollama_walks_one_at_a_time(self):
        calls = []
        result, events = _run(OLLAMA, _fake_ask(calls=calls))
        self.assertNotIn("parallel", result)
        self.assertEqual([c[0] for c in calls], [s["id"] for s in self.table])
        # running, done, running, done ... -- never two running at once.
        statuses = [e["status"] for e in events if e["event"] == "extract_step"]
        self.assertEqual(statuses, ["running", "done"] * len(self.table))

    def test_concurrent_result_matches_sequential(self):
        seq, _ = _run(OLLAMA, _fake_ask())
        par, _ = _run(VLLM, _fake_ask())
        self.assertEqual(seq["fields"], par["fields"])
        self.assertEqual(seq["raw"], par["raw"])
        strip = lambda steps: [{k: v for k, v in s.items() if k != "seconds"}
                               for s in steps]
        self.assertEqual(strip(seq["steps"]), strip(par["steps"]))
        self.assertEqual(seq["tokens"], par["tokens"])

    def test_every_step_reports_done_once(self):
        _, events = _run(VLLM, _fake_ask())
        done = [e["id"] for e in events
                if e["event"] == "extract_step" and e["status"] == "done"]
        self.assertEqual(sorted(done), sorted(s["id"] for s in self.table))


def _snap(count, total):
    return {"vllm:e2e_request_latency_seconds_count": float(count),
            "vllm:e2e_request_latency_seconds_sum": float(total)}


class BatchTimingTests(unittest.TestCase):
    def _batch(self, snaps, n=3, client=1.5):
        fetched = []

        def fetch(url):
            fetched.append(url)
            return snaps[min(len(fetched) - 1, len(snaps) - 1)]

        with mock.patch.object(servertime, "_fetch", fetch), \
                mock.patch.object(servertime.settings, "SERVER_TIMING_WAIT", 0.0):
            previous = servertime.open_tally()
            batch = servertime.Batch(VLLM)

            def one():
                servertime.Clock(VLLM).finish(client_seconds=client)

            threads = [threading.Thread(target=batch.run, args=(one,))
                       for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            batch.close()
            return servertime.close_tally(previous), fetched

    def test_one_span_for_the_whole_batch(self):
        tally, fetched = self._batch([_snap(10, 5.0), _snap(13, 8.0)])
        self.assertEqual(len(fetched), 2)      # no per-request snapshots
        self.assertEqual(tally["requests"], 3)
        self.assertEqual(tally["attributed"], 3)
        self.assertAlmostEqual(tally["server_seconds"], 3.0)
        self.assertAlmostEqual(tally["client_seconds"], 4.5)
        self.assertAlmostEqual(tally["network_seconds"], 1.5)
        self.assertEqual(tally["source"], "vllm /metrics")

    def test_other_traffic_leaves_it_unattributed(self):
        tally, _ = self._batch([_snap(10, 5.0), _snap(14, 9.0)])
        self.assertEqual(tally["requests"], 3)
        self.assertEqual(tally["attributed"], 0)
        self.assertIsNone(tally["server_seconds"])
        self.assertIn("batch of 3", tally["why"])

    def test_outside_a_batch_nothing_changed(self):
        with mock.patch.object(servertime, "_fetch",
                               side_effect=[_snap(1, 1.0), _snap(2, 1.4)]):
            previous = servertime.open_tally()
            servertime.Clock(VLLM).finish(client_seconds=0.5)
            tally = servertime.close_tally(previous)
        self.assertEqual(tally["attributed"], 1)
        self.assertAlmostEqual(tally["server_seconds"], 0.4)


class RunLogTests(unittest.TestCase):
    def test_column_written_only_when_parallel(self):
        self.assertIn("extract_parallel", runlog.COLUMNS)
        self.assertIn("extract_parallel", runlog.EXTRACT_COLUMNS)
        cells = runlog._extract_cells({"extracted": {"mode": "agentic", "parallel": 7}})
        self.assertEqual(cells["extract_parallel"], 7)
        cells = runlog._extract_cells({"extracted": {"mode": "agentic"}})
        self.assertEqual(cells["extract_parallel"], "")


if __name__ == "__main__":
    unittest.main()
