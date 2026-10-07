"""Cutting off a looping pass-2 request on vLLM, and leaving everything else alone."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import monitor

VLLM = {"kind": "vllm", "url": "http://gpu-box:8000", "model": "m", "available": True}
LLAMA = {"kind": "llama.cpp", "url": "http://local:8080", "model": "m", "available": True}
URL = "http://gpu-box:8000/v1/chat/completions"
SOLUTION = Path(__file__).resolve().parent.parent / "solution"


class _Stream:
    """A streamed chat completion, one SSE line per piece, counting what was read."""

    def __init__(self, pieces, finish="stop"):
        self.status_code = 200
        self.text = ""
        self.read = 0
        self.closed = False
        lines = [{"choices": [{"delta": {"content": p}}]} for p in pieces]
        lines.append({"choices": [{"delta": {}, "finish_reason": finish}]})
        lines.append({"choices": [], "usage": {"completion_tokens": len(pieces)}})
        self.lines = [("data: " + json.dumps(o, ensure_ascii=False)).encode()
                      for o in lines] + [b"data: [DONE]"]

    def iter_lines(self):
        for line in self.lines:
            self.read += 1
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def _loop_pieces():
    head = '{"document_number": "INV-1", "other_fields": ['
    entry = '{"label": "หมายเหตุการชำระเงิน", "value": "โอนเงินเข้าบัญชีบริษัท"}, '
    return [head] + [c for c in entry * 200]          # one piece per character


class GateTests(unittest.TestCase):
    def test_vllm_only_and_only_with_the_guard_on(self):
        with patch.object(app, "EXTRACT_LOOP_ABORT", True), \
             patch.object(app, "loop_guard", return_value=True):
            self.assertTrue(app._guards_extraction(URL, {}, VLLM))
            self.assertFalse(app._guards_extraction(
                "http://local:8080/v1/chat/completions", {}, LLAMA))
            self.assertFalse(app._guards_extraction(
                "http://x:11434/api/chat", {}, dict(VLLM, kind="ollama")))
        with patch.object(app, "EXTRACT_LOOP_ABORT", True), \
             patch.object(app, "loop_guard", return_value=False):
            self.assertFalse(app._guards_extraction(URL, {}, VLLM))
        with patch.object(app, "EXTRACT_LOOP_ABORT", False), \
             patch.object(app, "loop_guard", return_value=True):
            self.assertFalse(app._guards_extraction(URL, {}, VLLM))

    def test_llama_cpp_is_still_sent_the_non_streaming_request(self):
        class Plain:
            status_code = 200
            text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": "{}"},
                                     "finish_reason": "stop"}]}
        with patch.object(app.requests, "post", return_value=Plain()) as post:
            app._timed_post("http://local:8080/v1/chat/completions",
                            {"stream": False}, LLAMA)
        self.assertNotIn("stream", post.call_args.kwargs)
        self.assertFalse(post.call_args.kwargs["json"]["stream"])


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.guard = [patch.object(app, "EXTRACT_LOOP_ABORT", True),
                      patch.object(app, "loop_guard", return_value=True),
                      patch.object(app.servertime, "_fetch", return_value=None)]
        for g in self.guard:
            g.start()

    def tearDown(self):
        for g in self.guard:
            g.stop()

    def test_a_looping_reply_is_cut_off_and_reads_as_a_loop(self):
        stream = _Stream(_loop_pieces(), finish="length")
        before = app.loop_aborts()["count"]
        with patch.object(app.requests, "post", return_value=stream) as post:
            res = app._timed_post(URL, {"messages": [], "stream": False}, VLLM)
        sent = post.call_args.kwargs["json"]
        self.assertTrue(sent["stream"])
        self.assertTrue(sent["stream_options"]["include_usage"])
        self.assertTrue(stream.closed)
        self.assertLess(stream.read, len(stream.lines) // 2)       # stopped early
        body = res.json()
        self.assertTrue(body["loop_aborted"])
        self.assertEqual(app.loop_aborts()["count"], before + 1)
        text, truncated, _ = app.backends.structured_reply(body, VLLM)
        self.assertTrue(truncated)
        message, flags = app._why_unparsable(text, ValueError("x"), truncated, VLLM)
        self.assertTrue(flags.get("looped"), message)

    def test_a_clean_reply_comes_back_whole_and_unchanged(self):
        reply = json.dumps({"document_number": "INV-1", "subtotal": "1,200.00",
                            "other_fields": [{"label": "ผู้รับเงิน", "value": "ก"}]},
                           ensure_ascii=False)
        stream = _Stream(list(reply))
        before = app.loop_aborts()["count"]
        with patch.object(app.requests, "post", return_value=stream):
            body = app._timed_post(URL, {"messages": []}, VLLM).json()
        self.assertEqual(body["choices"][0]["message"]["content"], reply)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("loop_aborted", body)
        self.assertEqual(app.loop_aborts()["count"], before)

    def test_a_refusal_is_passed_through(self):
        stream = _Stream([])
        stream.status_code, stream.text = 400, "bad request"
        with patch.object(app.requests, "post", return_value=stream):
            res = app._timed_post(URL, {"messages": []}, VLLM)
        self.assertEqual((res.status_code, res.text), (400, "bad request"))


class NoFalseCutTests(unittest.TestCase):
    def test_no_ground_truth_reply_would_be_cut(self):
        """Every truth file, dumped as the reply a perfect extractor would send,
        tested at every point the stream would test it: none may trip."""
        checked = 0
        for path in sorted(SOLUTION.glob("*.fields.json")):
            truth = json.loads(path.read_text(encoding="utf-8"))
            blocks = truth.get("documents") or [truth]
            for block in blocks:
                values = {k: (v[0] if isinstance(v, list) and v
                              and isinstance(v[0], str) else v)
                          for k, v in block.items() if not k.startswith("_")}
                for indent in (None, 2):
                    reply = json.dumps(values, ensure_ascii=False, indent=indent)
                    for end in range(app.LOOP_CHECK_EVERY, len(reply) + 1,
                                     app.LOOP_CHECK_EVERY):
                        self.assertFalse(app.extraction_looping(reply[:end]),
                                         f"{path.name} cut at {end}")
                    checked += 1
        self.assertGreater(checked, 40)


class RealShapeTests(unittest.TestCase):
    """The shapes that tripped the first build on real gemma4:e4b replies."""

    @staticmethod
    def _reply(entries):
        return json.dumps({"other_fields": [{"label": l, "value": v} for l, v in entries]},
                          ensure_ascii=False)

    def _never_cut(self, text):
        self.assertGreaterEqual(len(text), app.EXTRACT_LOOP_TAIL_CHARS)
        for end in range(app.LOOP_CHECK_EVERY, len(text) + 1, app.LOOP_CHECK_EVERY):
            self.assertFalse(app.extraction_looping(text[:end]), end)

    def test_table_rows_written_under_their_column_headings(self):
        self._never_cut(self._reply([e for i in range(20) for e in [
            ("Product Code", f"88500{i:05d}"), ("รายการ<br>Description", f"สินค้าที่ {i}"),
            ("อ้างถึงใบกำกับภาษีเลขที่", "IV0005906"), ("Amount (Baht)", f"{i * 37}.00")]]))

    def test_three_rows_sharing_one_period(self):
        lead = [(f"หมายเหตุข้อ {i}", "รายละเอียดการชำระเงินตามสัญญาเช่าพื้นที่") for i in range(25)]
        same = [("ประจำงวด Period", "01/01/2026 - 31/01/2026")] * 3
        self._never_cut(self._reply(lead + same))

    def test_a_seven_entry_cycle_is_cut(self):
        loop = self._reply([(f"หมายเหตุการชำระเงินรอบที่ {i % 7}", "โอนเข้าบัญชี")
                            for i in range(40)])
        cut = next((e for e in range(app.LOOP_CHECK_EVERY, len(loop) + 1,
                                     app.LOOP_CHECK_EVERY)
                    if app.extraction_looping(loop[:e])), None)
        self.assertIsNotNone(cut)
        self.assertLess(cut, len(loop))


class MonitorTests(unittest.TestCase):
    def test_reserved_share_and_request_rates(self):
        samples = monitor.parse_samples(
            'vllm:cache_config_info{block_size="16",gpu_memory_utilization="0.9",'
            'num_gpu_blocks="100"} 1\n'
            'vllm:num_requests_running 2\nvllm:num_requests_waiting 1\n'
            'vllm:request_success_total{finished_reason="stop"} 5\n'
            'vllm:request_success_total{finished_reason="abort"} 1\n')
        s = monitor.snapshot(samples)
        self.assertEqual(s["reserved_share"], 0.9)
        self.assertEqual(s["requests_ok"], 6)
        later = dict(s, requests_ok=10, in_system=1,
                     finished_by_reason={"stop": 8, "abort": 2})
        r = monitor.rates(s, later, 2.0)
        self.assertEqual(r["requests_per_s"], 2)
        self.assertEqual(r["arrivals_per_s"], 1)       # (4 + 1 - 3) / 2
        self.assertEqual(r["finished_by_reason"], {"abort": 1, "stop": 3})

    def test_route_reports_the_loop_cutoff(self):
        info = {"kind": "vllm", "url": "http://gpu-box:8000"}

        class Res:
            status_code = 200
            text = 'vllm:num_requests_running 1\n'
        with patch.object(app.backends, "known", return_value=info), \
             patch.object(app.backends, "active_url", return_value=info["url"]), \
             patch.object(app.backends, "extract_separate", return_value=False), \
             patch.object(app, "MONITOR_GPU_URL", ""), \
             patch.object(app.requests, "get", return_value=Res()):
            d = app.app.test_client().get("/api/monitor").get_json()
        self.assertIn("count", d["loop_aborts"])
        self.assertIn("loop_abort_on", d)
        self.assertIn("vLLM port", d["gpu_reason"])


if __name__ == "__main__":
    unittest.main()
