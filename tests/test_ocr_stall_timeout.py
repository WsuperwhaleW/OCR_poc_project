"""An OCR read fails on a server gone silent, and only on silence.

A stalled read used to wait forever: no read timeout on the stream, so one page
held its job, its worker slot and a Stop's drain for as long as the server stayed
quiet. `GEN_READ_TIMEOUT` now bounds the silence -- before the first token and
between tokens -- and a reply that keeps coming may take longer than the limit.

Driven through the real `stream_page` and `requests` against a real local socket,
because what is being tested is what the timeout actually does to a blocked read.
"""
import http.server
import json
import threading
import time
import unittest
from unittest.mock import patch

import requests

import app

LIMIT = 0.5             # the read timeout, in seconds, for these tests
LLAMA = {"kind": "llama.cpp", "available": True, "model": "m", "url": "x"}


def _line(text):
    body = {"choices": [{"delta": {"content": text}}]}
    return ("data: " + json.dumps(body) + "\n\n").encode()


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

    def _token(self, text):
        self.wfile.write(_line(text))
        self.wfile.flush()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        release = self.server.release
        if self.path == "/before-headers":
            release.wait()
        elif self.path == "/mid-stream":
            self._start()
            self._token("a")
            release.wait()
        else:           # /steady: well past the limit in all, never silent for it
            self._start()
            for _ in range(8):
                self._token("a")
                time.sleep(LIMIT / 4)
            self.wfile.write(b"data: [DONE]\n\n")


class StallTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.release = threading.Event()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.release.set)

    def read(self, path):
        url = f"http://127.0.0.1:{self.server.server_address[1]}{path}"
        page = app.Image.new("RGB", (64, 64), "white")
        with patch.object(app, "GEN_TIMEOUT", (1.0, LIMIT)), \
             patch.object(app, "llama_status", return_value=LLAMA), \
             patch.object(app, "chat_url", return_value=url), \
             patch.object(app, "loop_guard", return_value=False), \
             patch.object(app, "_batcher_for", return_value=None), \
             patch.object(app.servertime, "_fetch", return_value=None):
            began = time.perf_counter()
            try:
                return "".join(app.stream_page(page, {})), time.perf_counter() - began
            except requests.RequestException as err:
                return err, time.perf_counter() - began

    def assert_timed_out(self, result, took):
        self.assertIsInstance(result, requests.RequestException)
        self.assertIn("timed out", str(result).lower())
        self.assertLess(took, 4 * LIMIT)

    def test_a_server_that_never_answers_fails_the_read(self):
        self.assert_timed_out(*self.read("/before-headers"))

    def test_a_stream_that_goes_silent_fails_the_read(self):
        self.assert_timed_out(*self.read("/mid-stream"))

    def test_a_read_longer_than_the_limit_that_keeps_coming_finishes(self):
        text, took = self.read("/steady")
        self.assertEqual(text, "a" * 8)
        self.assertGreater(took, LIMIT)


if __name__ == "__main__":
    unittest.main()
