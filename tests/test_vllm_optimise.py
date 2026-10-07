"""Three request optimisations for a vLLM server (2026-10-07).

* keep-alive: requests to vLLM / OpenAI servers go through one pooled Session,
  and nothing else does -- nor does anything while a test stubs the transport;
* /tokenize: a "prompt + max_tokens too long" 400 is retried once with the cap
  lowered to the room the window has;
* prefix warm-up: the agentic fan-out sends step 1 alone and the rest only after
  its prefill, so they hit vLLM's prefix cache.
"""

import threading
import time
import unittest
from unittest import mock

import requests

import app
import backends
import prompts
import servertime

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m",
        "models": [{"name": "m", "max_model_len": 8192}]}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "reachable": True, "model": "m"}


_no_metrics = mock.patch.object(servertime, "_fetch", return_value=None)


def setUpModule():
    # The server-time snapshots would otherwise go to the network for /metrics
    # on the made-up host -- not what any test here is about.
    _no_metrics.start()


def tearDownModule():
    _no_metrics.stop()


class _Res:
    def __init__(self, code, body=None, text=""):
        self.status_code = code
        self._body = body
        self.text = text

    def json(self):
        return self._body


class PoolTests(unittest.TestCase):
    def test_vllm_goes_through_the_session_when_transport_is_live(self):
        session = mock.Mock()
        with mock.patch.object(backends, "_http_session", return_value=session):
            backends.http_post("http://gpu:8000/v1/chat/completions", VLLM, json={})
            backends.http_get("http://gpu:8000/metrics", {"kind": "vllm"})
        session.post.assert_called_once()
        session.get.assert_called_once()

    def test_local_servers_keep_plain_requests(self):
        session = mock.Mock()
        with mock.patch.object(backends, "_http_session", return_value=session), \
                mock.patch.object(backends.requests, "post") as post:
            backends.http_post("http://local:11434/v1/chat/completions", OLLAMA, json={})
            backends.http_post("http://x/v1/chat/completions", None, json={})
        session.post.assert_not_called()
        self.assertEqual(post.call_count, 2)

    def test_a_stubbed_transport_is_never_bypassed(self):
        # The patch itself makes the transport "not live", so the stub sees it.
        session = mock.Mock()
        with mock.patch.object(backends, "_http_session", return_value=session), \
                mock.patch.object(backends.requests, "post") as post:
            backends.http_post("http://gpu:8000/v1/chat/completions", VLLM, json={})
        session.post.assert_not_called()
        post.assert_called_once()

    def test_switch_off(self):
        session = mock.Mock()
        with mock.patch.object(backends, "_http_session", return_value=session), \
                mock.patch.object(backends.settings, "HTTP_KEEPALIVE", False), \
                mock.patch.object(backends.requests, "get") as get:
            backends.http_get("http://gpu:8000/metrics", {"kind": "vllm"})
        session.get.assert_not_called()
        get.assert_called_once()

    def test_the_session_pool_is_sized(self):
        session = backends._http_session()
        adapter = session.get_adapter("http://gpu:8000/")
        self.assertEqual(adapter._pool_maxsize, backends.settings.HTTP_POOL_SIZE)


OVERFLOW = ("This model's maximum context length is 8192 tokens. However, you "
            "requested 9000 tokens (4904 in the messages, 4096 in the completion).")


class FitTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"model": "m", "messages": [{"role": "user", "content": "x"}],
                        "max_tokens": 4096,
                        "chat_template_kwargs": {"enable_thinking": False}}
        # Not streamed: the loop guard path is not what is under test here.
        patcher = mock.patch.object(app, "_guards_extraction", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _route(self, sent, tokenize=None):
        def post(url, json=None, **_):
            sent.append((url, json))
            if url.endswith("/tokenize"):
                return _Res(200, tokenize or {"count": 4904, "max_model_len": 8192})
            if json["max_tokens"] == 4096:
                return _Res(400, text=OVERFLOW)
            return _Res(200, {"choices": [{"message": {"content": "{}"},
                                           "finish_reason": "stop"}]})
        return post

    def test_overflow_is_retried_with_the_room_left(self):
        sent = []
        with mock.patch.object(requests, "post", self._route(sent)):
            res = app._timed_post("http://gpu:8000/v1/chat/completions",
                                  self.payload, VLLM)
        self.assertEqual(res.status_code, 200)
        urls = [u for u, _ in sent]
        self.assertEqual(urls, ["http://gpu:8000/v1/chat/completions",
                                "http://gpu:8000/tokenize",
                                "http://gpu:8000/v1/chat/completions"])
        tokenize = sent[1][1]
        self.assertEqual(tokenize["messages"], self.payload["messages"])
        self.assertTrue(tokenize["add_generation_prompt"])
        self.assertEqual(tokenize["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent[2][1]["max_tokens"], 8192 - 4904 - 8)
        self.assertEqual(self.payload["max_tokens"], 4096)    # not mutated

    def test_no_room_keeps_the_400(self):
        sent = []
        route = self._route(sent, {"count": 8150, "max_model_len": 8192})
        with mock.patch.object(requests, "post", route):
            res = app._timed_post("http://gpu:8000/v1/chat/completions",
                                  self.payload, VLLM)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(len(sent), 2)          # asked, counted, not asked again

    def test_window_from_v1_models_when_tokenize_omits_it(self):
        sent = []
        with mock.patch.object(requests, "post", self._route(sent, {"count": 5000})):
            app._timed_post("http://gpu:8000/v1/chat/completions", self.payload, VLLM)
        self.assertEqual(sent[-1][1]["max_tokens"], 8192 - 5000 - 8)

    def test_other_400s_and_other_servers_are_left_alone(self):
        sent = []

        def post(url, json=None, **_):
            sent.append(url)
            return _Res(400, text="bad request: unknown field")
        with mock.patch.object(requests, "post", post):
            app._timed_post("http://gpu:8000/v1/chat/completions", self.payload, VLLM)
        self.assertEqual(sent, ["http://gpu:8000/v1/chat/completions"])
        self.assertFalse(app._context_overflow(OVERFLOW, OLLAMA))
        with mock.patch.object(app, "VLLM_FIT_MAX_TOKENS", False):
            self.assertFalse(app._context_overflow(OVERFLOW, VLLM))


CODES = ["INVOICE"]
FORM = {"doc_types": CODES, "items": (),
        "keys": list(prompts.fields_for_types(CODES)),
        "mandatory": list(prompts.mandatory_for_types(CODES))}
TEXT = "header\n"


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.table = app.steps_for_types(CODES)
        self.log, self.lock = [], threading.Lock()

    def _ask(self, fail_first=False):
        def ask(content, step, status, collect=None):
            with self.lock:
                self.log.append(("start", step["id"], time.perf_counter()))
            if step["id"] == self.table[0]["id"]:
                time.sleep(0.15)               # "prefilling"
                with self.lock:
                    self.log.append(("prefilled", step["id"], time.perf_counter()))
                if fail_first:
                    raise RuntimeError("server gone")
                app._prefilled()               # what the first SSE line does
                time.sleep(0.15)               # still decoding when the rest start
            values = {k: ([] if k == "other_fields" else "") for k in step["keys"]}
            return values, ["{}"], False, 1, False
        return ask

    def _run(self, ask):
        with mock.patch.object(app, "_ask_step", ask):
            gen = app._extract_agentic(TEXT, VLLM, FORM)
            while True:
                try:
                    next(gen)
                except StopIteration as stop:
                    return stop.value

    def test_rest_start_after_step_one_is_prefilled(self):
        result = self._run(self._ask())
        self.assertEqual(result["parallel"], len(self.table))
        prefilled = next(t for kind, _, t in self.log if kind == "prefilled")
        others = [t for kind, sid, t in self.log
                  if kind == "start" and sid != self.table[0]["id"]]
        self.assertEqual(len(others), len(self.table) - 1)
        self.assertTrue(all(t >= prefilled for t in others))
        # ...and while step 1 is still decoding, not after it has finished.
        first_done = prefilled + 0.15
        self.assertTrue(all(t < first_done for t in others))

    def test_a_failed_first_step_still_releases_the_rest(self):
        start = time.perf_counter()
        result = self._run(self._ask(fail_first=True))
        self.assertLess(time.perf_counter() - start, 5)
        self.assertEqual(len(result["steps"]), len(self.table))

    def test_switch_off_sends_all_at_once(self):
        with mock.patch.object(app, "VLLM_PREFIX_WARMUP", False):
            self._run(self._ask())
        prefilled = next(t for kind, _, t in self.log if kind == "prefilled")
        others = [t for kind, sid, t in self.log
                  if kind == "start" and sid != self.table[0]["id"]]
        self.assertTrue(any(t < prefilled for t in others))


if __name__ == "__main__":
    unittest.main()
