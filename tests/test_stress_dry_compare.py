"""A stress page DRY fired on is read once more without DRY, for its window."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
import backends
import engine

FIRED = {"checked": True, "fired": 3, "first": 40, "longest": 300,
         "allowed": 256, "steered": 2}


def _page():
    page = SimpleNamespace()
    page._stress_origin = {"case": "sol003", "source": "benchmark",
                           "index": 0, "detail": "low"}
    return page


class DryCompareTests(unittest.TestCase):
    def _read(self, compare=True, fired=True, fail=False):
        calls = []

        def read(page, stats, cancel=None):
            off = backends.dry_suppressed()
            calls.append(off)
            if off:
                if fail:
                    raise RuntimeError("server went away")
                stats.update(looped=True, new_tokens=99, seconds=5.0)
                return "row\nrow\nrow\nrow\nrow"
            stats.update(looped=False, dry=FIRED if fired else None,
                         new_tokens=10, seconds=1.0)
            return "row with dry"

        live, rec = app._StressLive(1), {}
        live.dry_compare = compare
        with patch.object(engine, "read_page", side_effect=read), \
                patch.object(app, "STRESS_DRY_RETRY", False):
            texts, looped, _ = app._stress_read([_page()], app.MultiCancel(),
                                                rec, live, job=1, case="sol003")
        alerts = live.take_alerts()
        return calls, texts, looped, rec, alerts

    def test_the_page_is_read_again_without_dry_and_shown_beside_it(self):
        calls, texts, looped, rec, alerts = self._read()
        self.assertEqual(calls, [False, True])     # the second read had no DRY
        self.assertFalse(backends.dry_suppressed())
        self.assertEqual(texts, ["row with dry"])  # the read with DRY is kept
        self.assertFalse(looped)                   # its loop flags nothing
        self.assertEqual(rec["ocr_tokens"], 10)    # and counts in nothing
        self.assertEqual([a["kind"] for a in alerts], ["dry"])
        reads = app._STRESS_EVIDENCE.get(alerts[0]["id"])["attempts"]
        self.assertEqual(len(reads), 2)
        plain = reads[1]
        self.assertTrue(plain["nodry"])
        self.assertEqual(plain["compare_of"], 0)
        self.assertIsNone(plain["page"])           # the pair's image is shown
        self.assertTrue(plain["looped"])
        self.assertEqual([m["kind"] for m in plain["marks"]], ["loop"])

    def test_off_reads_the_page_once(self):
        calls, *_ = self._read(compare=False)
        self.assertEqual(calls, [False])

    def test_a_clean_read_is_not_compared(self):
        calls, *_ = self._read(fired=False)
        self.assertEqual(calls, [False])

    def test_a_failed_comparison_is_noted_not_raised(self):
        calls, texts, _, _, alerts = self._read(fail=True)
        self.assertEqual(texts, ["row with dry"])
        plain = app._STRESS_EVIDENCE.get(alerts[0]["id"])["attempts"][1]
        self.assertIn("server went away", plain["note"])

    def test_the_switch_is_this_thread_only(self):
        import threading
        seen = []
        with backends.dry_off_here():
            t = threading.Thread(target=lambda: seen.append(backends.dry_suppressed()))
            t.start()
            t.join()
            self.assertTrue(backends.dry_suppressed())
        self.assertEqual(seen, [False])
        self.assertFalse(backends.dry_suppressed())

    def test_config_follows_use_dry(self):
        status = {"kind": "vllm", "model": "m", "url": "http://s", "available": True,
                  "reason": None, "text_available": True, "text_reason": None}
        with patch.object(app.engine, "llama_status", return_value=status), \
                patch.object(app.backends, "extract_status", return_value=status):
            self.assertTrue(app._stress_config({"mode": "extract"})["dry_compare"])
            self.assertFalse(app._stress_config({"mode": "extract",
                                                 "dry_compare": False})["dry_compare"])


if __name__ == "__main__":
    unittest.main()
