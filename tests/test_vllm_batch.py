"""The stress test's "Batch requests" toggle: vLLM's /v1/chat/completions/batch.

The batcher is driven with a fake `send`, so what is checked is the grouping,
the split back into single replies and the failure path -- and then that the
two request paths (a page read, a pass-2 request) go through it on vLLM and
only on vLLM.
"""
import threading
import unittest
from unittest.mock import patch

from PIL import Image

import app
import servertime
import vllmbatch

CHAT = "http://v/v1/chat/completions"
VLLM = {"available": True, "reachable": True, "kind": "vllm", "url": "http://v",
        "model": "m"}


def _text(content):
    if isinstance(content, list):          # a page: [text part, image part]
        return "".join(p.get("text", "") for p in content)
    return content


def _echo(calls):
    """A fake batch route: each conversation's reply is its last message's text."""
    def send(url, body):
        calls.append((url, body))
        return {"model": "m", "usage": {"prompt_tokens": 10 * len(body["messages"]),
                                        "completion_tokens": 2 * len(body["messages"])},
                "choices": [{"index": i, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": _text(conv[-1]["content"])},
                             "token_ids": [0, 0]}
                            for i, conv in enumerate(body["messages"])]}
    return send


def _payload(text, **extra):
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 8,
            "stream": True, **extra}


class BatcherTests(unittest.TestCase):
    def test_a_full_group_is_one_batch_and_each_gets_its_own_reply(self):
        calls = []
        batcher = vllmbatch.Batcher(3, _echo(calls), wait=5)
        tickets = [batcher.submit(CHAT, _payload(t)) for t in "abc"]
        got = [t.result(2)["choices"][0]["message"]["content"] for t in tickets]
        self.assertEqual(got, ["a", "b", "c"])
        self.assertEqual(len(calls), 1)
        url, body = calls[0]
        self.assertEqual(url, CHAT + "/batch")
        self.assertEqual(len(body["messages"]), 3)
        self.assertNotIn("stream", body)
        self.assertEqual(tickets[0].result()["usage"]["completion_tokens"], 2)
        self.assertEqual(batcher.summary()["max_size"], 3)

    def test_a_short_group_is_sent_after_the_wait(self):
        calls = []
        batcher = vllmbatch.Batcher(8, _echo(calls), wait=0.05)
        ticket = batcher.submit(CHAT, _payload("x"))
        self.assertEqual(ticket.result(2)["choices"][0]["message"]["content"], "x")
        self.assertEqual(len(calls), 1)

    def test_requests_that_differ_beyond_the_conversation_are_not_mixed(self):
        calls = []
        batcher = vllmbatch.Batcher(2, _echo(calls), wait=0.05)
        a = batcher.submit(CHAT, _payload("a"))
        b = batcher.submit(CHAT, _payload("b", max_tokens=99))
        a.result(2), b.result(2)
        self.assertEqual(sorted(len(body["messages"]) for _, body in calls), [1, 1])

    def test_a_failed_batch_fails_every_request_in_it(self):
        def send(url, body):
            raise vllmbatch.BatchError("no route", 404)
        batcher = vllmbatch.Batcher(2, send, wait=5)
        tickets = [batcher.submit(CHAT, _payload(t)) for t in "ab"]
        for ticket in tickets:
            with self.assertRaises(vllmbatch.BatchError) as caught:
                ticket.result(2)
            self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(batcher.summary()["failed"], 1)

    def test_close_sends_what_is_waiting(self):
        calls = []
        batcher = vllmbatch.Batcher(8, _echo(calls), wait=60)
        ticket = batcher.submit(CHAT, _payload("z"))
        batcher.close()
        self.assertEqual(ticket.result(2)["choices"][0]["message"]["content"], "z")
        with self.assertRaises(vllmbatch.BatchError):
            batcher.submit(CHAT, _payload("late")).result(1)

    def test_usage_without_token_ids_is_shared_and_says_so(self):
        reply = {"choices": [{"index": 0, "message": {"content": "a"}},
                             {"index": 1, "message": {"content": "bbb"}}],
                 "usage": {"prompt_tokens": 9, "completion_tokens": 8}}
        one, two = vllmbatch.split_reply(reply, 2)
        self.assertEqual((one["usage"]["completion_tokens"],
                          two["usage"]["completion_tokens"]), (2, 6))
        self.assertTrue(one["usage_estimated"])
        with self.assertRaises(vllmbatch.BatchError):
            vllmbatch.split_reply(reply, 3)


class AppPathTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.batcher = vllmbatch.Batcher(2, _echo(self.calls), wait=0.05)
        patcher = patch.object(app, "_batcher", self.batcher)
        patcher.start()
        self.addCleanup(patcher.stop)
        timing = patch.object(servertime.settings, "SERVER_TIMING_METRICS", False)
        timing.start()
        self.addCleanup(timing.stop)

    def test_pass_two_goes_through_the_batch_on_vllm(self):
        payloads = [_payload("p", stream=False), _payload("q", stream=False)]
        replies = [None, None]

        def post(i):
            replies[i] = app._post_once(CHAT, payloads[i], VLLM)
        threads = [threading.Thread(target=post, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual([r.status_code for r in replies], [200, 200])
        self.assertEqual(replies[1].json()["choices"][0]["message"]["content"], "q")
        self.assertEqual(len(self.calls), 1)

    def test_a_server_that_is_not_vllm_is_never_batched(self):
        self.assertIsNone(app._batcher_for(CHAT, dict(VLLM, kind="llama.cpp")))
        self.assertIsNone(app._batcher_for("http://v/api/chat", VLLM))
        self.assertIs(app._batcher_for(CHAT, VLLM), self.batcher)

    def test_a_failed_batch_is_a_failed_reply_not_a_crash(self):
        def send(url, body):
            raise vllmbatch.BatchError("this vLLM build has no batch route", 404)
        with patch.object(app, "_batcher", vllmbatch.Batcher(1, send)):
            reply = app._post_once(CHAT, _payload("p", stream=False), VLLM)
        self.assertEqual(reply.status_code, 404)
        self.assertIn("no batch route", reply.text)

    def test_a_page_read_is_one_conversation_of_a_batch(self):
        page = Image.new("RGB", (64, 64), "white")
        app.tag_page(page, "sol003 p1/2", "low")
        stats = {}
        with patch.object(app, "llama_status", return_value=VLLM), \
             patch.object(app, "chat_url", return_value=CHAT), \
             patch.object(app.backends, "model_window", return_value=None), \
             patch.object(app.backends.settings, "vllm_xargs",
                          return_value={"dry_multiplier": 0.8}):
            text = "".join(app.stream_page(page, stats))
        self.assertTrue(text)                  # the prompt, echoed back
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(stats["new_tokens"], 2)
        self.assertIsNone(stats["ttft_seconds"])   # nothing streamed to time
        # A batch shares one vllm_xargs, so a per-page tag would split every group.
        self.assertEqual(self.calls[0][1]["vllm_xargs"], {"dry_multiplier": 0.8})

    def test_a_page_read_on_its_own_names_its_page_for_the_dry_log(self):
        page = Image.new("RGB", (64, 64), "white")
        app.tag_page(page, "sol003 p1/2", "low")
        sent = {}

        def post(url, status, json=None, **kwargs):
            sent.update(json)
            raise RuntimeError("sent")
        with patch.object(app, "llama_status", return_value=VLLM), \
             patch.object(app, "chat_url", return_value=CHAT), \
             patch.object(app.backends, "model_window", return_value=None), \
             patch.object(app, "_batcher_for", return_value=None), \
             patch.object(app.backends.settings, "vllm_xargs",
                          return_value={"dry_multiplier": 0.8}), \
             patch.object(app.backends, "http_post", post):
            with self.assertRaises(Exception):
                "".join(app.stream_page(page, {}))
        self.assertEqual(sent["vllm_xargs"],
                         {"dry_multiplier": 0.8, "dry_tag": "sol003 p1/2 low"})

    def test_config_refuses_batching_where_no_server_is_vllm(self):
        local = {"text_available": True, "text_reason": None, "kind": "ollama",
                 "url": "http://l", "model": "m"}
        with patch.object(app.backends, "extract_status", return_value=local):
            with self.assertRaises(ValueError):
                app._stress_config({"mode": "extract", "vllm_batch": True})
            self.assertFalse(app._stress_config({"mode": "extract"})["vllm_batch"])
        with patch.object(app.backends, "extract_status",
                          return_value={**local, "kind": "vllm", "reachable": True}):
            self.assertTrue(app._stress_config({"mode": "extract",
                                                "vllm_batch": True})["vllm_batch"])


if __name__ == "__main__":
    unittest.main()
