"""The stress test's "Use DRY" switch: off sends the run without DRY."""
import unittest
from unittest.mock import patch

import app
import backends
import settings


def _info(kind):
    return {"kind": kind, "model": "m", "url": "http://s"}


class DryOffTests(unittest.TestCase):
    def tearDown(self):
        while backends.dry_suppressed():
            backends.restore_dry()

    def test_vllm_carries_dry_while_on(self):
        with patch.object(settings, "VLLM_DRY", True), \
                patch.object(settings, "DRY_MULTIPLIER", 0.8):
            self.assertIn("dry_multiplier",
                          backends.request_extras(_info("vllm"))["vllm_xargs"])

    def test_vllm_sends_no_xargs_while_off(self):
        backends.suppress_dry()
        with patch.object(settings, "VLLM_DRY", True), \
                patch.object(settings, "DRY_MULTIPLIER", 0.8):
            self.assertNotIn("vllm_xargs", backends.request_extras(_info("vllm")))

    def test_llama_cpp_gets_its_own_off(self):
        with patch.object(backends, "num_ctx", return_value=0):
            self.assertNotIn("dry_multiplier", backends.request_extras(_info("llama.cpp")))
            backends.suppress_dry()
            self.assertEqual(backends.request_extras(_info("llama.cpp"))["dry_multiplier"], 0)

    def test_restore_counts_down(self):
        backends.suppress_dry()
        backends.suppress_dry()
        backends.restore_dry()
        self.assertTrue(backends.dry_suppressed())
        backends.restore_dry()
        self.assertFalse(backends.dry_suppressed())
        backends.restore_dry()                      # never below zero
        self.assertFalse(backends.dry_suppressed())

    def test_config_defaults_on_and_reads_false(self):
        status = dict(_info("vllm"), available=True, reason=None,
                      text_available=True, text_reason=None)
        with patch.object(app.engine, "llama_status", return_value=status), \
                patch.object(app.backends, "extract_status", return_value=status):
            self.assertTrue(app._stress_config({"mode": "extract"})["dry"])
            self.assertFalse(app._stress_config({"mode": "extract", "dry": False})["dry"])

    def test_the_run_holds_the_switch_and_lets_it_go(self):
        seen = []

        def run(cfg, cancel):
            seen.append(backends.dry_suppressed())
            yield "x\n"

        with patch.object(app, "_stress_run", side_effect=run), \
                patch.object(app, "_history_stream", side_effect=lambda g, *a, **k: g):
            list(app._stress_generate({"mode": "extract", "dry": False}, None))
            list(app._stress_generate({"mode": "extract"}, None))
        self.assertEqual(seen, [True, False])
        self.assertFalse(backends.dry_suppressed())


if __name__ == "__main__":
    unittest.main()
