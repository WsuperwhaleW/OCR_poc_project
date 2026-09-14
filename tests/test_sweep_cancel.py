"""Stopping a random test.

The defect these cover: Stop used to abort the page's own `fetch` and tell the
server nothing, and an abandoned stream is not stopped on this server -- so
every remaining round went on reading and the next Run was refused by a 409
that was telling the truth.
"""
import json
import threading
import unittest
from unittest.mock import patch

import app


class CancelTests(unittest.TestCase):
    class Closeable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def test_set_closes_the_request_in_flight(self):
        cancel = app.Cancel()
        res = self.Closeable()
        cancel.attach(res)
        self.assertFalse(cancel.is_set())
        self.assertFalse(res.closed)
        cancel.set()
        self.assertTrue(cancel.is_set())
        self.assertTrue(res.closed)

    def test_attaching_after_the_stop_closes_at_once(self):
        # The window between the page loop's check and the request going out.
        # A response attached after the stop is the one thing nobody would
        # otherwise hang up on.
        cancel = app.Cancel()
        cancel.set()
        res = self.Closeable()
        cancel.attach(res)
        self.assertTrue(res.closed)

    def test_one_slot_and_a_failing_close_is_swallowed(self):
        class Angry(self.Closeable):
            def close(self):
                raise OSError("already gone")

        cancel = app.Cancel()
        first = self.Closeable()
        cancel.attach(first)
        cancel.attach(Angry())          # replaces: one read in flight at a time
        cancel.set()                    # must not raise
        self.assertFalse(first.closed)  # it was already finished with

    def test_stop_requested_tolerates_no_cancel(self):
        self.assertFalse(app.stop_requested(None))


class SweepGuardTests(unittest.TestCase):
    def tearDown(self):
        app.end_sweep()

    def test_cancel_with_nothing_running_is_not_an_error(self):
        app.end_sweep()
        self.assertFalse(app.cancel_sweep())

    def test_begin_hands_back_the_flag_the_stop_will_set(self):
        cancel = app.begin_sweep()
        self.assertTrue(app.cancel_sweep())
        self.assertTrue(cancel.is_set())

    def test_only_the_owning_thread_releases(self):
        app.begin_sweep()
        other = threading.Thread(target=app.end_sweep)
        other.start()
        other.join()
        # Still held: releasing from elsewhere would let a second sweep start
        # beside one that is still reading.
        self.assertTrue(app.cancel_sweep())
        app.end_sweep()
        self.assertFalse(app.cancel_sweep())


class ReadPageCancelTests(unittest.TestCase):
    def test_drain_stops_between_tokens(self):
        cancel = app.Cancel()
        seen = []

        def tokens(image, stats=None, profile=None, cancel_=None):
            for piece in "abcdef":
                if piece == "c":
                    cancel.set()        # the Stop lands while c is generated
                seen.append(piece)
                yield piece

        with patch.object(app, "stream_page",
                          lambda i, s=None, p=None, c=None: tokens(i, s, p, c)):
            text = app.read_page(None, {}, cancel=cancel)
        # The token the flag arrived during is kept and the generator is never
        # asked for another: d, e and f are the rest of the page, not fetched.
        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(text, "abc")

    def test_no_cancel_drains_to_the_end(self):
        with patch.object(app, "stream_page",
                          lambda i, s=None, p=None, c=None: iter("abcdef")):
            self.assertEqual(app.read_page(None, {}), "abcdef")


class ReadCaseCancelTests(unittest.TestCase):
    """`_read_case` throws the partial away and logs the row as cancelled."""

    CASE = "sol019"                     # the smallest fixture: 0.57 MP, 1 page

    def _read(self, cancel, page_reader):
        rows = []
        with patch.object(app, "read_page", page_reader), \
             patch.object(app, "log_run",
                          lambda payload, source, **kw: rows.append((payload, kw))):
            try:
                result = app._read_case(self.CASE, "low", extract=False,
                                        cancel=cancel)
            except app.SweepCancelled as err:
                return rows, err
            return rows, result

    def test_a_stopped_read_is_logged_cancelled_and_raises(self):
        cancel = app.Cancel()

        def reader(image, stats=None, profile=None, cancel_=None, **kw):
            cancel.set()
            return "half a page"

        rows, outcome = self._read(cancel, reader)
        self.assertIsInstance(outcome, app.SweepCancelled)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1].get("status"), "cancelled")
        # No transcript score: half a document against the whole ground truth
        # would report a stop as a bad read.
        self.assertNotIn("truth", rows[0][0])

    def test_a_hung_up_request_is_the_stop_landing_not_a_failure(self):
        cancel = app.Cancel()

        def reader(image, stats=None, profile=None, cancel_=None, **kw):
            cancel.set()
            raise OSError("connection closed by the stopping thread")

        rows, outcome = self._read(cancel, reader)
        self.assertIsInstance(outcome, app.SweepCancelled)
        self.assertEqual(rows[0][1].get("status"), "cancelled")

    def test_a_real_error_is_still_an_error(self):
        def reader(image, stats=None, profile=None, cancel_=None, **kw):
            raise OSError("the model server fell over")

        with self.assertRaises(OSError):
            self._read(app.Cancel(), reader)


class StopRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def tearDown(self):
        app.end_sweep()

    def test_stop_with_nothing_running_answers_rather_than_erroring(self):
        app.end_sweep()
        body = self.client.post("/api/randomtest/stop").get_json()
        self.assertEqual(body, {"stopping": False, "running": False})

    def test_a_stop_ends_the_run_and_it_is_not_counted_as_a_failure(self):
        plan = {"seed": 1, "rounds": [{"case": "a"}, {"case": "b"}],
                "lock": {}, "scenarios": None, "repeats": 0}
        started = threading.Event()

        def run_round(round_, cancel=None):
            started.set()
            # The second round never runs: the loop sees the flag first.
            raise app.SweepCancelled("stopped")

        def stop_soon():
            started.wait(10)
            self.client.post("/api/randomtest/stop")

        threading.Thread(target=stop_soon, daemon=True).start()
        with patch.object(app, "_random_plan", lambda body: plan), \
             patch.object(app, "_run_round", run_round):
            res = self.client.post("/api/randomtest/stream", json={})
            events = [json.loads(line) for line in
                      res.get_data(as_text=True).strip().splitlines()]

        kinds = [e["event"] for e in events]
        self.assertIn("stopped", kinds)
        done = events[-1]
        self.assertTrue(done["stopped"])
        self.assertEqual(done["failed"], 0)
        self.assertEqual(done["completed"], 0)


if __name__ == "__main__":
    unittest.main()
