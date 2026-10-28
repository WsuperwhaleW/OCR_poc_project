"""The stress test's concurrency is per stage: a full run reads N files and
extracts N files at once, not N jobs across both (`_stress_submit`)."""
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import app
import stress


class _Gauge:
    """How many calls are in flight at once, and the most there were."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *_):
        with self.lock:
            self.now -= 1


def _page():
    page = Image.new("RGB", (8, 8), (255, 255, 255))
    page._data_uri = app.image_data_uri(page)
    return page


class StageConcurrencyTests(unittest.TestCase):

    def _run(self, mode, concurrency, jobs, read_s=0.15, extract_s=0.15, stop_after=None):
        reads, extracts = _Gauge(), _Gauge()
        both = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def tally(delta):
            with lock:
                both["now"] += delta
                both["peak"] = max(both["peak"], both["now"])

        def read(pages, cancel, rec, live, stats_out=None, job=0, case="", kept=None):
            with reads:
                tally(1)
                time.sleep(read_s)
                tally(-1)
            rec.update(ocr_s=read_s, pages=1, ocr_tokens=1)
            return ["text"], False, False

        def extract(*_a, **_k):
            with extracts:
                tally(1)
                time.sleep(extract_s)
                tally(-1)
            return {"fields": {}, "tokens": 1}

        cfg = {"mode": mode, "concurrency": concurrency, "extract_mode": "single",
               "detail": "low", "fill_missing": False, "max_seconds": 0,
               "metrics_every": 5, "save_log": False}
        prepared = {"pages": {"sol002": [_page()]}, "truths": {"sol002": "text"},
                    "sources": {"sol002": "x"}}
        specs = [{"index": i, "case": "sol002"} for i in range(jobs)]
        cancel = app.MultiCancel()
        events = []
        with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)), \
                patch.object(app, "_stress_read", side_effect=read), \
                patch.object(app, "extract_fields", side_effect=extract):
            phase = app._stress_phase(specs, cfg, prepared, cancel, "run")
            try:
                while True:
                    events.append(next(phase))
                    if stop_after and len(events) >= stop_after:
                        cancel.set()
            except StopIteration as done:
                records, _, timeline, _, draining = done.value
            for future in draining:
                future.result(timeout=10)
        return records, reads.peak, extracts.peak, both["peak"], timeline

    def test_full_reads_n_and_extracts_n_at_once(self):
        records, reads, extracts, both, _ = self._run("full", 3, 9, read_s=0.1, extract_s=0.4)
        self.assertEqual([r["status"] for r in records], ["ok"] * 9)
        self.assertEqual(reads, 3)                  # never past the limit, per stage
        self.assertEqual(extracts, 3)
        self.assertGreater(both, 3)                 # the stages overlap: more than N in all
        self.assertLessEqual(both, 6)
        # Waiting for an extract worker is reported and kept out of the latency.
        waited = [r.get("extract_wait_s") for r in records]
        self.assertTrue(all(w is not None for w in waited))
        self.assertGreater(max(waited), 0.1)
        report = stress.report(records, 2.0, "full", 3)
        self.assertTrue(report["concurrency_per_stage"])
        self.assertEqual(report["latency"]["extract_wait"]["n"], 9)

    def test_ocr_and_extract_only_runs_keep_one_pool(self):
        for mode in ("ocr", "extract"):
            with self.subTest(mode=mode):
                records, reads, extracts, both, _ = self._run(mode, 2, 6)
                self.assertEqual([r["status"] for r in records], ["ok"] * 6)
                self.assertEqual(both, 2)
                self.assertNotIn("extract_wait_s", records[0])
                self.assertFalse(stress.report(records, 1.0, mode, 2)["concurrency_per_stage"])

    def test_a_job_waiting_to_extract_is_its_own_stage(self):
        _, _, _, _, timeline = self._run("full", 2, 6, read_s=0.05, extract_s=0.5)
        col = 1 + stress.STAGES.index("extract_queued")
        self.assertGreater(max(point[col] for point in timeline), 0)
        self.assertEqual(timeline[-1][col], 0)

    def test_stop_drops_the_jobs_waiting_to_extract(self):
        records, _, _, _, _ = self._run("full", 2, 10, read_s=0.05, extract_s=1.5,
                                        stop_after=1)
        statuses = [r["status"] for r in records]
        self.assertIn("cancelled", statuses)
        self.assertNotIn("error", statuses)


if __name__ == "__main__":
    unittest.main()
