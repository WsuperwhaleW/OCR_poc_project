"""Pinning a server to one inference engine, and Auto remembering what it found.

What is asserted is the list of paths each server is asked, because the point
of the feature is the requests that are NOT made: a vLLM server sent /props and
/api/tags answers 404 to both, and only vLLM may ever be asked /metrics.
"""

import unittest
from unittest import mock

import backends

URL = "http://kind-test:9000"


class _Res:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body

    def json(self):
        return self._body


def _fake_server(engine):
    """A requests.get that answers like `engine` and records every path."""
    asked = []

    def get(url, timeout=None):
        path = url[len(URL):]
        asked.append(path)
        if engine == "llama.cpp" and path == "/props":
            return _Res(200, {"default_generation_settings": {},
                              "model_alias": "m", "modalities": {"vision": True}})
        if engine == "ollama" and path == "/api/tags":
            return _Res(200, {"models": [{"name": "m:latest",
                                          "capabilities": ["vision"]}]})
        if path == "/v1/models" and engine in ("vllm", "openai", "llama.cpp", "ollama"):
            owner = "vllm" if engine == "vllm" else "someone"
            return _Res(200, {"data": [{"id": "m", "owned_by": owner}]})
        return _Res(404)

    return get, asked


class ServerKindTests(unittest.TestCase):
    def setUp(self):
        self._saved = (dict(backends._kind_pinned), dict(backends._kind_seen))
        backends._kind_pinned.pop(URL, None)
        backends._kind_seen.pop(URL, None)
        backends._cache.pop(URL, None)

    def tearDown(self):
        backends._kind_pinned.clear()
        backends._kind_pinned.update(self._saved[0])
        backends._kind_seen.clear()
        backends._kind_seen.update(self._saved[1])
        backends._cache.pop(URL, None)

    def _probe(self, engine, force=True):
        get, asked = _fake_server(engine)
        with mock.patch.object(backends.requests, "get", get):
            info = backends.probe(URL, force=force)
        return info, asked

    def test_auto_detects_vllm_once_then_asks_it_first(self):
        info, asked = self._probe("vllm")
        self.assertEqual(info["kind"], "vllm")
        self.assertEqual(asked, ["/props", "/api/tags", "/v1/models"])
        backends._cache.pop(URL, None)  # an expired cache entry, not a Re-check
        info, asked = self._probe("vllm", force=False)
        self.assertEqual(info["kind"], "vllm")
        self.assertEqual(asked, ["/v1/models"])

    def test_auto_remembers_ollama(self):
        self._probe("ollama")
        backends._cache.pop(URL, None)
        info, asked = self._probe("ollama", force=False)
        self.assertEqual(info["kind"], "ollama")
        self.assertEqual(asked, ["/api/tags"])

    def test_recheck_forgets_and_detects_from_scratch(self):
        self._probe("vllm")
        # The port now runs llama-server; a forced probe must find it rather
        # than read it as a generic /v1/models server.
        info, asked = self._probe("llama.cpp", force=True)
        self.assertEqual(info["kind"], "llama.cpp")
        self.assertEqual(asked, ["/props"])

    def test_pinned_vllm_asks_only_v1_models(self):
        get, asked = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            info = backends.set_server_kind(URL, "vllm")
        self.assertEqual(info["kind"], "vllm")
        self.assertEqual(info["kind_pinned"], "vllm")
        self.assertEqual(asked, ["/v1/models"])
        self.assertTrue(backends.serves_metrics(info))

    def test_pinned_ollama_and_llama(self):
        for engine, path in (("ollama", "/api/tags"), ("llama.cpp", "/props")):
            get, asked = _fake_server(engine)
            with mock.patch.object(backends.requests, "get", get):
                info = backends.set_server_kind(URL, engine)
            self.assertEqual(info["kind"], engine)
            self.assertEqual(asked, [path])
            self.assertFalse(backends.serves_metrics(info))

    def test_pin_that_does_not_match_is_unreachable_not_retried(self):
        get, asked = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            info = backends.set_server_kind(URL, "ollama")
        self.assertFalse(info["reachable"])
        self.assertIsNone(info["kind"])
        self.assertEqual(asked, ["/api/tags"])
        self.assertIn("Auto", info["reason"])

    def test_pinned_vllm_on_a_generic_server_is_vllm(self):
        # The pin is the user's statement; /v1/models alone cannot tell them apart.
        get, _ = _fake_server("openai")
        with mock.patch.object(backends.requests, "get", get):
            info = backends.set_server_kind(URL, "vllm")
        self.assertEqual(info["kind"], "vllm")
        self.assertIn("owned_by vllm", info["kind_warning"])

    def test_pinned_vllm_on_vllm_has_no_warning(self):
        get, _ = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            info = backends.set_server_kind(URL, "vllm")
        self.assertNotIn("kind_warning", info)

    def test_pinned_openai_never_gets_metrics(self):
        get, _ = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            info = backends.set_server_kind(URL, "openai")
        self.assertEqual(info["kind"], "openai")
        self.assertFalse(backends.serves_metrics(info))

    def test_auto_unpins(self):
        get, _ = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            backends.set_server_kind(URL, "ollama")
            info = backends.set_server_kind(URL, "auto")
        self.assertEqual(info["kind"], "vllm")
        self.assertEqual(backends.server_kind(URL), "auto")

    def test_unknown_kind_refused(self):
        with self.assertRaises(ValueError):
            backends.set_server_kind(URL, "tgi")

    def test_env_parsing(self):
        parsed = backends._parse_kinds(
            "http://a:8000=vllm, b:11434=Ollama ,c:8080=llamacpp,d:1=nonsense,junk")
        self.assertEqual(parsed, {"http://a:8000": "vllm",
                                  "http://b:11434": "ollama",
                                  "http://c:8080": "llama.cpp"})

    def test_free_gpu_never_asks_a_pinned_vllm_for_api_ps(self):
        get, asked = _fake_server("vllm")
        with mock.patch.object(backends.requests, "get", get):
            backends.set_server_kind(URL, "vllm")
            asked.clear()
            self.assertEqual(backends.free_gpu(URL), [])
        self.assertNotIn("/api/ps", asked)


if __name__ == "__main__":
    unittest.main()
