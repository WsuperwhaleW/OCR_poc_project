"""A stress-test page DRY fired on is read again one Detail up, even unlooped."""
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


def _up(page):
    nxt = app.next_detail(page._stress_origin["detail"])
    return _page(nxt) if nxt else None


FIRED = {"checked": True, "fired": 3, "first": 40, "longest": 300,
         "allowed": 256, "steered": 2}
CLEAN = {"checked": True, "fired": 0, "longest": 20, "allowed": 256}


class StressDryRetryTests(unittest.TestCase):

    def _read(self, outcomes, start="low", retry=_up):
        """`outcomes` is `(looped, dry)` for each read, in order."""
        calls = []

        def read(page, stats, cancel=None):
            calls.append(page._stress_origin["detail"])
            looped, dry = outcomes[len(calls) - 1]
            stats.update(looped=looped, dry=dry, new_tokens=10, seconds=1.0)
            return "text " + page._stress_origin["detail"]

        live, rec = app._StressLive(1), {}
        with patch.object(app, "read_page", side_effect=read), \
                patch.object(app, "_stress_retry_page", side_effect=retry):
            texts, looped, _ = app._stress_read([_page(start)], app.MultiCancel(),
                                                rec, live, job=1, case="sol003")
        return calls, texts, looped, rec, live.take_alerts()

    def test_dry_without_a_loop_climbs_and_clears(self):
        calls, texts, looped, rec, alerts = self._read([(False, FIRED),
                                                        (False, CLEAN)])
        self.assertEqual(calls, ["low", "medium"])
        self.assertEqual(texts, ["text medium"])
        self.assertFalse(looped)
        dry = [a for a in alerts if a["kind"] == "dry"]
        self.assertEqual(len({a["id"] for a in dry}), 1)
        self.assertEqual(dry[-1]["state"], "resolved")
        self.assertIn("reading it again at medium", dry[1]["message"])
        self.assertNotIn("looped", [a["kind"] for a in alerts])
        self.assertEqual((rec["dry_retries"], rec["dry_recovered"]), (1, 1))
        self.assertEqual((rec["loop_retries"], rec["dry_pages"]), (0, 1))

    def test_dry_at_every_detail_keeps_the_top_read(self):
        calls, texts, _, rec, alerts = self._read([(False, FIRED)] * 3)
        self.assertEqual(calls, ["low", "medium", "original"])
        self.assertEqual(texts, ["text original"])
        self.assertEqual(alerts[-1]["state"], "retry failed")
        self.assertEqual((rec["dry_retries"], rec["dry_recovered"]), (2, 0))

    def test_dry_at_original_has_no_retry(self):
        calls, _, _, _, alerts = self._read([(False, FIRED)], start="original")
        self.assertEqual(calls, ["original"])
        self.assertEqual(alerts[-1]["state"], "no retry")

    def test_a_retry_that_loops_falls_back_to_the_unlooped_read(self):
        calls, texts, looped, rec, alerts = self._read(
            [(False, FIRED), (True, CLEAN), (True, CLEAN)])
        self.assertEqual(calls, ["low", "medium", "original"])
        self.assertEqual(texts, ["text low"])
        self.assertFalse(looped)
        loop = [a for a in alerts if a["kind"] == "looped"]
        self.assertEqual(loop[-1]["state"], "resolved")
        self.assertIn("kept the read at low", loop[-1]["message"])
        self.assertEqual([a for a in alerts if a["kind"] == "dry"][-1]["state"],
                         "retry failed")

    def test_switched_off_it_only_flags(self):
        with patch.object(app, "STRESS_DRY_RETRY", False):
            calls, texts, _, rec, alerts = self._read([(False, FIRED)])
        self.assertEqual(calls, ["low"])
        self.assertEqual(texts, ["text low"])
        self.assertEqual([a.get("state") for a in alerts], [None])
        self.assertEqual(rec["dry_retries"], 0)

    def test_the_report_counts_dry_retries(self):
        records = [{"status": "ok", "pages": 1, "dry_retries": 2,
                    "dry_recovered": 1, "total_s": 1.0, "ocr_s": 1.0}]
        out = stress.report(records, 1.0, "ocr", 1)
        self.assertEqual((out["ocr"]["dry_retries"], out["ocr"]["dry_recovered"]),
                         (2, 1))


if __name__ == "__main__":
    unittest.main()
