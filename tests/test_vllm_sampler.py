"""vLLM is sent the penalties Ollama actually applies to the same model."""
import unittest
from unittest.mock import patch

import backends
import settings


def _vllm(model):
    return {"kind": "vllm", "model": model, "url": "http://v"}


class VllmSamplerTests(unittest.TestCase):
    def test_qwen_gets_the_modelfile_presence_penalty(self):
        extras = backends.request_extras(_vllm("Qwen/Qwen3.5-9B"))
        self.assertEqual(extras["presence_penalty"], 1.5)
        self.assertEqual(extras["repetition_penalty"], 1.1)

    def test_typhoon_gets_its_modelfile_repeat_penalty(self):
        extras = backends.request_extras(_vllm("scb10x/typhoon-ocr1.5-3b"))
        self.assertEqual(extras["repetition_penalty"], 1.2)
        self.assertNotIn("presence_penalty", extras)

    def test_unlisted_model_gets_ollamas_default(self):
        extras = backends.request_extras(_vllm("google/gemma-4-e4b-it"))
        self.assertEqual(extras["repetition_penalty"], 1.1)
        self.assertNotIn("presence_penalty", extras)

    def test_other_backends_are_untouched(self):
        for kind in ("ollama", "llama.cpp", "openai"):
            extras = backends.request_extras({"kind": kind, "model": "qwen3.5:9b"})
            self.assertNotIn("presence_penalty", extras, kind)
            self.assertNotIn("repetition_penalty", extras, kind)

    def test_it_replaces_the_shared_penalty_in_the_sent_body(self):
        with patch.object(settings, "REPETITION_PENALTY", 1.0), \
             patch.object(backends, "num_ctx", return_value=8192):
            url, body = backends.structured_request(
                [{"role": "user", "content": "x"}], None, 64, _vllm("qwen3.5-9b"))
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(body["repetition_penalty"], 1.1)
        self.assertEqual(body["presence_penalty"], 1.5)
        self.assertEqual(body["temperature"], 0)

    def test_off_switch_and_overrides(self):
        with patch.object(settings, "VLLM_MATCH_OLLAMA", False):
            self.assertEqual(settings.vllm_sampler("qwen"), {})
        with patch.object(settings, "VLLM_PRESENCE_PENALTY", 0.5), \
             patch.object(settings, "VLLM_FREQUENCY_PENALTY", 0.2):
            fields = settings.vllm_sampler("gemma")
        self.assertEqual(fields["presence_penalty"], 0.5)
        self.assertEqual(fields["frequency_penalty"], 0.2)
        self.assertEqual(fields["repetition_penalty"], 1.1)


if __name__ == "__main__":
    unittest.main()
