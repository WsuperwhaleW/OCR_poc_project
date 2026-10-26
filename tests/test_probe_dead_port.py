import unittest
from unittest.mock import MagicMock, patch

import requests

import backends


class DeadPortTests(unittest.TestCase):
    def setUp(self):
        backends._cache.clear()
        backends._kind_seen.clear()

    def test_a_refused_connect_stops_after_one_probe(self):
        calls = []

        def refuse(url, *a, **k):
            calls.append(url)
            raise requests.exceptions.ConnectionError("refused")

        with patch.object(backends.requests, "get", side_effect=refuse), \
             patch.object(backends, "http_get", side_effect=refuse):
            info = backends.probe("http://127.0.0.1:59999", force=True)
        self.assertFalse(info["reachable"])
        self.assertEqual(len(calls), 1)

    def test_a_404_still_asks_the_next_engine(self):
        calls = []

        def answer(url, *a, **k):
            calls.append(url)
            res = MagicMock(status_code=200)
            if url.endswith("/v1/models"):
                res.json.return_value = {"data": [{"id": "m", "owned_by": "vllm"}]}
            else:
                res.status_code = 404
            return res

        with patch.object(backends.requests, "get", side_effect=answer), \
             patch.object(backends, "http_get", side_effect=answer):
            info = backends.probe("http://127.0.0.1:59998", force=True)
        self.assertEqual(info["kind"], "vllm")
        self.assertEqual(len(calls), 3)

    def test_loopback_gets_the_short_connect_timeout(self):
        self.assertEqual(backends._probe_timeout("http://127.0.0.1:8001"),
                         backends.LOOPBACK_PROBE_TIMEOUT)
        self.assertEqual(backends._probe_timeout("http://localhost:11434"),
                         backends.LOOPBACK_PROBE_TIMEOUT)
        self.assertEqual(backends._probe_timeout("http://10.0.0.5:8000"),
                         backends.PROBE_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
