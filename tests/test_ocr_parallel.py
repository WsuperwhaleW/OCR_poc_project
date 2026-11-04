"""A long document's pages read at once on vLLM, and joined back in page order.

Added 2026-10-20 with `OCR_PARALLEL`: a 13-page file (sol023) took thirteen
pages of wall clock where a one-page file took one. Pinned here: the gate (3+
pages, vLLM only, the switch), that the joined transcript is exactly the
sequential one whatever order the pages finished in, that every page really is
in flight together, and that a failed page or a Stop reaches the others.
"""

import json
import threading
import time
import unittest
from unittest import mock

from PIL import Image

import app
import engine
VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True,
        "available": True, "model": "m"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "reachable": True,
          "available": True, "model": "m"}


def _pages(n):
    """n blank pages, told apart by their width: page k is k pixels wide."""
    return [Image.new("RGB", (k, 10), "white") for k in range(1, n + 1)]


def _fake_stream(barrier=None, fail_on=None, stall_on=None):
    """A `stream_page` that reads "page k" off a page k pixels wide.

    Later pages answer FIRST (page 1 sleeps longest), so a reader that joined
    them in finishing order would be caught. `barrier` holds every page until
    all of them are in flight; `fail_on` raises on that page; `stall_on` blocks
    that page until its cancel is set, as a hung-up stream would.
    """
    def stream(image, stats=None, profile=None, cancel=None):
        k = image.width
        if barrier is not None:
            barrier.wait(timeout=5)     # BrokenBarrierError unless all are in flight
        if k == fail_on:
            raise ValueError(f"page {k} failed")
        if k == stall_on:
            while not cancel.is_set():
                time.sleep(0.01)
            raise ValueError("hung up")
        time.sleep(0.02 * (10 - k) if k < 10 else 0)
        if stats is not None:
            stats.update(new_tokens=2, decode_seconds=0.1, seconds=0.1,
                         ocr_profile=app.ocr_profile())
        yield f"page {k} "
        yield "text"
    return stream


class GateTests(unittest.TestCase):
    def parallelism(self, pages, status=VLLM):
        with mock.patch.object(engine, "llama_status", return_value=status):
            return app._page_parallelism(pages)

    def test_three_pages_or_more_on_vllm_go_at_once(self):
        self.assertEqual(self.parallelism(3), 3)
        self.assertEqual(self.parallelism(13), 13)

    def test_short_documents_stay_sequential(self):
        self.assertEqual(self.parallelism(1), 1)
        self.assertEqual(self.parallelism(2), 1)

    def test_only_on_a_server_that_batches(self):
        self.assertEqual(self.parallelism(13, OLLAMA), 1)
        self.assertEqual(self.parallelism(13, {**VLLM, "reachable": False}), 1)

    def test_the_switch_and_the_caps(self):
        with mock.patch.object(engine, "OCR_PARALLEL", False):
            self.assertEqual(self.parallelism(13), 1)
        with mock.patch.object(engine, "OCR_PARALLEL_MAX", 4):
            self.assertEqual(self.parallelism(13), 4)
        with mock.patch.object(engine, "OCR_PARALLEL_MIN_PAGES", 5):
            self.assertEqual(self.parallelism(4), 1)
            self.assertEqual(self.parallelism(5), 5)

    def test_a_short_read_never_asks_the_server(self):
        with mock.patch.object(engine, "llama_status") as status:
            app._page_parallelism(2)
        status.assert_not_called()


class ReadTests(unittest.TestCase):
    def read(self, n, stream, status=VLLM, cancel=None):
        texts, stats, run = [], [], {}
        with mock.patch.object(engine, "llama_status", return_value=status), \
                mock.patch.object(engine, "stream_page", stream):
            app.read_all_pages(_pages(n), texts, stats, cancel, run)
        return texts, stats, run

    def test_pages_come_back_in_page_order(self):
        texts, stats, run = self.read(6, _fake_stream())
        self.assertEqual(texts, [f"page {k} text" for k in range(1, 7)])
        self.assertEqual(len(stats), 6)
        self.assertEqual(run["parallel"], 6)

    def test_the_join_is_the_sequential_one(self):
        together, _, _ = self.read(5, _fake_stream())
        alone, _, run = self.read(5, _fake_stream(), status=OLLAMA)
        self.assertEqual(run, {})
        self.assertEqual(app.join_page_texts(together), app.join_page_texts(alone))

    def test_every_page_is_in_flight_together(self):
        texts, _, _ = self.read(5, _fake_stream(barrier=threading.Barrier(5)))
        self.assertEqual(len(texts), 5)

    def test_a_failed_page_stops_the_read_and_hangs_up_the_rest(self):
        with self.assertRaisesRegex(ValueError, "page 4 failed"):
            self.read(5, _fake_stream(fail_on=4, stall_on=2))

    def test_a_stop_reaches_every_page(self):
        cancel = app.Cancel()
        stream = _fake_stream(stall_on=3)
        threading.Timer(0.3, cancel.set).start()
        with self.assertRaises(ValueError):
            self.read(4, stream, cancel=cancel)
        self.assertTrue(cancel.is_set())

    def test_a_stop_before_the_read_sends_nothing(self):
        cancel = app.Cancel()
        cancel.set()
        stream = mock.Mock(side_effect=AssertionError("nothing should be sent"))
        texts, _, _ = self.read(4, stream, cancel=cancel)
        self.assertEqual(texts, [])


class SummaryTests(unittest.TestCase):
    def test_a_concurrent_read_says_so(self):
        stats = [{"new_tokens": 1, "decode_seconds": 0.1}] * 3
        self.assertEqual(app.summarise(stats, "low", 0.0, None,
                                       {"parallel": 3})["ocr_parallel"], 3)
        self.assertNotIn("ocr_parallel", app.summarise(stats, "low", 0.0, None))

    def test_the_batch_timing_stands_in_for_the_pages(self):
        stats = [{"new_tokens": 1, "decode_seconds": 0.1}] * 3
        server = {name: None for name in app.servertime.FIELDS}
        server.update(server_seconds=4.5, source="metrics", requests=3)
        s = app.summarise(stats, "low", 0.0, None, {"parallel": 3, "server": server})
        self.assertEqual(s["server_seconds"], 4.5)
        self.assertEqual(s["server_timing"], "metrics")
        self.assertEqual(s["server_timed_pages"], 3)


class StreamTests(unittest.TestCase):
    def test_the_stream_sends_whole_pages_in_order(self):
        pages = _pages(4)
        client = app.app.test_client()
        with mock.patch.object(engine, "llama_status", return_value=VLLM), \
                mock.patch.object(engine, "stream_page", _fake_stream()), \
                mock.patch.object(app, "prepare",
                                  return_value=(pages, "low", None, None, {})), \
                mock.patch.object(engine, "log_run"):
            res = client.post("/api/ocr/stream", data={"extract": "0"})
            events = [json.loads(line) for line in res.get_data(as_text=True).splitlines()
                      if line.strip()]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "pages_parallel")
        self.assertEqual(events[0]["parallel"], 4)
        self.assertEqual([e["page"] for e in events if e["event"] == "page"], [1, 2, 3, 4])
        done = next(e for e in events if e["event"] == "done")
        self.assertEqual(done["pages"], [f"page {k} text" for k in range(1, 5)])
        self.assertEqual(done["text"], app.join_page_texts(done["pages"]))
        self.assertEqual(done["ocr_parallel"], 4)


if __name__ == "__main__":
    unittest.main()
