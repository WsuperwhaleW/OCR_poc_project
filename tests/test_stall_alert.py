"""The stress test's stall alert measures silence, not how long a job takes.

A multi-document file is classify + split + one extraction per document + table
agents, and under load the whole of it can run far past the stall limit while
every one of its requests keeps answering. The alert's clock is reset by each
streamed line and each finished request of the job -- on the agentic step
threads too -- so only a request that has genuinely gone quiet is flagged, and
the alert says which one.

Driven through `_stress_job` with a fake three-document agentic extraction whose
requests stream through the real `_timed_post` / `_streamed_post`, and the stall
limit shrunk to half a second.
"""
import json
import threading
import time
import unittest
from unittest.mock import patch

import app
import engine
import prompts

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True,
        "model": "m", "available": True, "text_available": True,
        "text_reason": None}
URL = "http://gpu:8000/v1/chat/completions"
LIMIT = 0.5          # the stall limit, in seconds, for these tests
GAP = 0.1            # between two streamed lines of a healthy request
LINES = 4            # streamed lines per healthy request
DOCUMENTS = 3

CODES = ["INVOICE"]
FORM = {"doc_types": CODES, "items": (),
        "keys": list(prompts.fields_for_types(CODES)),
        "mandatory": list(prompts.mandatory_for_types(CODES))}


def _value(key):
    return f"VAL{abs(hash(key)) % 100000:05d}{key.upper()}"


# Every scalar answer is printed in the transcript, so no step is re-asked.
TEXT = "header\n" + "\n".join(_value(k) for k in FORM["keys"]) + "\n"


class _SlowStream:
    """A streamed reply: `silent` seconds before the first line, `GAP` between."""

    def __init__(self, silent=0.0):
        self.status_code = 200
        self.text = ""
        self.silent = silent
        lines = [{"choices": [{"delta": {"content": "x"}}]}] * LINES
        lines.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        lines.append({"choices": [], "usage": {"completion_tokens": LINES}})
        self.lines = [("data: " + json.dumps(o)).encode() for o in lines]

    def iter_lines(self):
        time.sleep(self.silent)
        for index, line in enumerate(self.lines):
            if 0 < index <= LINES:
                time.sleep(GAP)
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _post(url, json=None, **kwargs):
    return _SlowStream(silent=(json or {}).get("silent", 0.0))


class StallAlertTests(unittest.TestCase):
    def setUp(self):
        self.table = app.steps_for_types(CODES)
        self.assertGreater(len(self.table), 1)
        self.patches = [
            patch.object(engine, "EXTRACT_LOOP_ABORT", True),
            patch.object(engine, "loop_guard", return_value=True),
            patch.object(engine, "VLLM_PREFIX_WARMUP", False),
            patch.object(app.servertime, "_fetch", return_value=None),
            patch.object(app.backends, "extract_status", return_value=VLLM),
            patch.object(app.requests, "post", side_effect=_post),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _run_job(self, quiet=None, quiet_for=0.0):
        """One extract-only stress job over a three-document file, every
        document extracted by the agentic steps at once. `quiet` is the
        (document, step id) whose request says nothing for `quiet_for` seconds
        before answering. Returns (record, alerts)."""
        state = {"document": 0}

        def ask(content, step, status, collect=None):
            silent = (quiet_for if (state["document"], step["id"]) == quiet
                      else 0.0)
            app._timed_post(URL, {"messages": [], "silent": silent}, status)
            values = {k: ([] if k == "other_fields" else _value(k))
                      for k in step["keys"]}
            return values, ["{}"], False, LINES, False

        def extract(text, mode=None, case_id=None, pages=None,
                    truth_fed=False, extras=None, each=None, schema_first=False):
            result = None
            for number in range(1, DOCUMENTS + 1):
                state["document"] = number
                each({"event": "document", "document": number,
                      "documents": DOCUMENTS})
                result = app._drain(app._extract_agentic(TEXT, VLLM, FORM), each)
            return result

        cfg = app._stress_config({"mode": "extract", "concurrency": 1,
                                  "jobs": 1, "lock_case": "sol002",
                                  "extract_mode": "agentic"})
        prepared = {"truths": {"sol002": TEXT}, "sources": {"sol002": "x"},
                    "pages": {}, "packs": []}
        spec = {"index": 0, "case": "sol002",
                "submitted": time.perf_counter(), "deadline": None, "log": False}
        live = app._StressLive(1)
        out = {}
        with patch.object(engine, "_ask_step", ask), \
             patch.object(app, "extract_fields", side_effect=extract):
            worker = threading.Thread(target=lambda: out.update(
                rec=app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)))
            began = time.perf_counter()
            worker.start()
            # The stress phase checks once a second against 300 s; this checks
            # fifty times a second against half a second.
            while worker.is_alive():
                live.flag_stalled(LIMIT)
                time.sleep(0.02)
            worker.join()
        out["took"] = time.perf_counter() - began
        return out["rec"], live.take_alerts(), out["took"]

    def test_a_long_multi_request_file_that_keeps_answering_is_not_stalled(self):
        rec, alerts, took = self._run_job()
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        # The job as a whole ran several stall limits long ...
        self.assertGreater(took, 2 * LIMIT)
        # ... and was never flagged, because no request went quiet for one.
        self.assertEqual([a for a in alerts if a["kind"] == "stalled"], [])

    def test_a_request_that_goes_quiet_is_flagged_and_named(self):
        step = self.table[1]["id"]
        rec, alerts, _ = self._run_job(quiet=(2, step), quiet_for=3 * LIMIT)
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        stalled = [a for a in alerts if a["kind"] == "stalled"]
        self.assertEqual(len(stalled), 1, alerts)
        self.assertEqual((stalled[0]["job"], stalled[0]["case"]), (1, "sol002"))
        message = stalled[0]["message"]
        self.assertIn("extracting", message)
        self.assertIn(f"document 2 of {DOCUMENTS}, step {step}", message)
        # The other steps of that document had answered; only the quiet one
        # is named.
        self.assertEqual(message.count("step "), 1, message)

if __name__ == "__main__":
    unittest.main()
