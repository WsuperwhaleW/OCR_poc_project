"""A long file's pages read on workers left idle (`_StressSlots`, `_stress_read`)."""

import json
import threading
import time
import unittest
from unittest.mock import patch

import app
import workspace


class _Pages:
    """A fake `read_page`: each page sleeps, and the most in flight is kept."""

    def __init__(self, seconds=0.05, fail=None):
        self.seconds, self.fail = seconds, fail
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def __call__(self, page, stats, cancel=None):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            end = time.perf_counter() + self.seconds
            while time.perf_counter() < end:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("hung up")
                time.sleep(0.005)
            if page == self.fail:
                raise RuntimeError(f"server error on {page}")
            stats.update(new_tokens=3, seconds=self.seconds)
            return f"text {page}"
        finally:
            with self.lock:
                self.now -= 1


def _live(concurrency, queued=0, held=1):
    live = app._StressLive(queued)
    live.slots = app._StressSlots(concurrency)
    for _ in range(held):          # the jobs running now, this one included
        live.slots.hold()
    return live


class SplitPagesTests(unittest.TestCase):
    def read(self, pages, live, fake):
        rec = {}
        with patch.object(app, "read_page", fake):
            out = app._stress_read(pages, app.MultiCancel(), rec, live, job=1, case="x")
        return out, rec

    def test_idle_workers_read_the_pages_together_in_page_order(self):
        fake, live = _Pages(), _live(4)
        (texts, looped, _), rec = self.read([f"p{i}" for i in range(1, 9)], live, fake)
        self.assertEqual(texts, [f"text p{i}" for i in range(1, 9)])
        self.assertFalse(looped)
        self.assertEqual(fake.peak, 4)              # never past the concurrency
        self.assertEqual(rec["pages_peak"], 4)
        self.assertGreater(rec["pages_lent"], 0)
        self.assertEqual(rec["ocr_tokens"], 24)
        # every lent slot is back: three free, this job's own still held
        self.assertTrue(all(live.slots.lend() for _ in range(3)))
        self.assertFalse(live.slots.lend())

    def test_nothing_is_lent_while_jobs_wait_in_the_queue(self):
        fake, live = _Pages(seconds=0.01), _live(4, queued=2)
        (texts, _, _), rec = self.read(["a", "b", "c"], live, fake)
        self.assertEqual(texts, ["text a", "text b", "text c"])
        self.assertEqual(fake.peak, 1)
        self.assertEqual(rec["pages_lent"], 0)

    def test_busy_workers_lend_nothing(self):
        fake, live = _Pages(seconds=0.01), _live(3, held=3)
        _, rec = self.read(["a", "b", "c"], live, fake)
        self.assertEqual(fake.peak, 1)
        self.assertEqual(rec["pages_lent"], 0)

    def test_without_slots_pages_are_read_one_at_a_time(self):
        fake, live = _Pages(seconds=0.01), app._StressLive(0)
        (texts, _, _), rec = self.read(["a", "b", "c"], live, fake)
        self.assertEqual(texts, ["text a", "text b", "text c"])
        self.assertEqual((fake.peak, rec["pages_lent"]), (1, 0))

    def test_a_failed_page_fails_the_job_names_it_and_returns_the_slots(self):
        fake, live = _Pages(fail="p3"), _live(4)
        rec = {}
        with patch.object(app, "read_page", fake):
            with self.assertRaisesRegex(RuntimeError, "p3"):
                app._stress_read([f"p{i}" for i in range(1, 7)], app.MultiCancel(),
                                 rec, live, job=1, case="x")
        self.assertEqual(rec["at_page"], 3)
        self.assertTrue(all(live.slots.lend() for _ in range(3)))

    def test_a_stop_cancels_the_read(self):
        fake, live, cancel = _Pages(seconds=0.5), _live(4), app.MultiCancel()
        threading.Timer(0.1, cancel.set).start()
        with patch.object(app, "read_page", fake):
            with self.assertRaises(app.SweepCancelled):
                app._stress_read([f"p{i}" for i in range(1, 9)], cancel, {}, live)
        self.assertTrue(all(live.slots.lend() for _ in range(3)))


STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock",
          "model": "mock-model"}


class RunTests(unittest.TestCase):
    """The whole OCR-only runner: two one-page files and one long one."""

    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        for target, name, kwargs in (
            (app, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
            (app.runlog, "case_counts", {"return_value": {}}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_stress(self, **body):
        cfg = app._stress_config({"mode": "ocr", "cases": ["sol001", "sol002", "sol004"],
                                  "concurrency": 4, "jobs": 3, "strategy": "sequential",
                                  "history_monitor": False, **body})
        sizes = {"sol001": 1, "sol002": 1, "sol004": 8}
        fake = _Pages(seconds=0.05)
        with patch.object(app, "_stress_pages",
                          side_effect=lambda case, detail, *a: (
                              [f"{case}-{i}" for i in range(sizes[case])], {})), \
                patch.object(app, "read_page", fake):
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        return cfg, events[-1]["report"], fake, events[0]

    def test_on_by_default_and_the_long_file_uses_the_idle_workers(self):
        cfg, report, fake, plan = self.run_stress()
        self.assertTrue(cfg["split_pages"])
        self.assertTrue(plan["split_pages"])
        self.assertEqual(report["completed"], 3)
        self.assertGreater(report["ocr"]["pages_lent"], 0)
        self.assertGreater(report["ocr"]["pages_peak"], 1)
        self.assertLessEqual(fake.peak, 4)

    def test_off_reads_each_file_page_by_page(self):
        cfg, report, fake, plan = self.run_stress(split_pages=False)
        self.assertFalse(plan["split_pages"])
        self.assertEqual((report["ocr"]["pages_lent"], report["ocr"]["pages_peak"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
