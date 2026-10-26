"""A stress-test page that loops is flagged, then read again one Detail up."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
import stress


class _Page(SimpleNamespace):
    pass


def _page(detail):
    page = _Page()
    page._stress_origin = {"case": "sol003", "source": "benchmark",
                           "index": 0, "detail": detail}
    return page


class StressLoopRetryTests(unittest.TestCase):

    def _read(self, pages, outcomes, retry):
        """`outcomes` is the looped flag of each read, in order."""
        calls = []

        def read(page, stats, cancel=None):
            calls.append(page._stress_origin["detail"])
            stats.update(looped=outcomes[len(calls) - 1], new_tokens=10,
                         seconds=1.0)
            return "text " + page._stress_origin["detail"]

        live, rec = app._StressLive(1), {}
        with patch.object(app, "read_page", side_effect=read), \
                patch.object(app, "_stress_retry_page", side_effect=retry):
            texts, looped, _ = app._stress_read(pages, app.MultiCancel(), rec, live,
                                                job=1, case="sol003")
        return calls, texts, looped, rec, live.take_alerts()

    def test_a_looped_page_is_flagged_then_read_again_and_recovers(self):
        def up(page):
            nxt = app.next_detail(page._stress_origin["detail"])
            return _page(nxt) if nxt else None
        calls, texts, looped, rec, alerts = self._read([_page("low")],
                                                       [True, False], up)
        self.assertEqual(calls, ["low", "medium"])
        self.assertEqual(texts, ["text medium"])
        self.assertFalse(looped)
        # One issue, sent twice: raised retrying, then marked resolved in place.
        self.assertEqual([(a["kind"], a["state"]) for a in alerts],
                         [("looped", "retrying"), ("looped", "resolved")])
        self.assertEqual(alerts[0]["id"], alerts[1]["id"])
        self.assertIn("reading it again at medium", alerts[0]["message"])
        self.assertIn("at low; read cleanly at medium", alerts[1]["message"])
        self.assertEqual((rec["loop_retries"], rec["loop_recovered"]), (1, 1))
        self.assertEqual(rec["ocr_tokens"], 20)        # both attempts' tokens

    def test_a_page_that_loops_at_every_detail_stays_looped(self):
        def up(page):
            nxt = app.next_detail(page._stress_origin["detail"])
            return _page(nxt) if nxt else None
        calls, _, looped, rec, alerts = self._read([_page("low")],
                                                   [True, True, True], up)
        self.assertEqual(calls, ["low", "medium", "original"])
        self.assertTrue(looped)
        self.assertEqual(len({a["id"] for a in alerts}), 1)   # one row, updated
        self.assertEqual([a["state"] for a in alerts],
                         ["retrying", "retrying", "retry failed"])
        self.assertIn("at low, medium, original; the retry failed", alerts[-1]["message"])
        self.assertEqual((rec["loop_retries"], rec["loop_recovered"]), (2, 0))

    def test_no_retry_left_reads_once(self):
        calls, _, looped, rec, alerts = self._read([_page("original")], [True],
                                                   lambda page: None)
        self.assertEqual(calls, ["original"])
        self.assertTrue(looped)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["state"], "no retry")
        self.assertNotIn("reading it again", alerts[0]["message"])
        self.assertEqual(rec["loop_retries"], 0)

    def test_retry_page_is_rebuilt_one_detail_up(self):
        pages, _ = app._stress_pages("sol003", "low")
        bigger = app._stress_retry_page(pages[0])
        self.assertIsNotNone(bigger)
        self.assertEqual(bigger._stress_origin["detail"], "medium")
        self.assertGreater(bigger.size[0] * bigger.size[1],
                           pages[0].size[0] * pages[0].size[1])
        self.assertTrue(bigger._data_uri.startswith("data:image/png;base64,"))
        with patch.object(app, "LOOP_RETRY", False):
            self.assertIsNone(app._stress_retry_page(pages[0]))

    def test_the_report_counts_retries(self):
        records = [{"status": "ok", "pages": 1, "loop_retries": 1,
                    "loop_recovered": 1, "total_s": 1.0, "ocr_s": 1.0},
                   {"status": "looped", "pages": 1, "loop_retries": 2,
                    "loop_recovered": 0, "total_s": 1.0, "ocr_s": 1.0}]
        out = stress.report(records, 2.0, "ocr", 1)
        self.assertEqual((out["ocr"]["loop_retries"], out["ocr"]["loop_recovered"]),
                         (3, 1))


if __name__ == "__main__":
    unittest.main()
