"""A separate endpoint for pass 2, and the one-URL default it must not disturb."""

import unittest
from unittest import mock

import backends
import runlog

READER = "http://reader:11434"
EXTRACTOR = "http://extractor:8000"

PROBES = {
    READER: {"kind": "ollama", "reachable": True, "model": None,
             "models": [{"name": "typhoon-ocr", "vision": True},
                        {"name": "gemma4:e4b", "vision": True}],
             "vision": None, "slots": None, "reason": None, "url": READER},
    EXTRACTOR: {"kind": "vllm", "reachable": True, "model": "qwen-big",
                "models": [{"name": "qwen-big", "vision": None}],
                "vision": None, "slots": None, "reason": None, "url": EXTRACTOR},
}


class ExtractEndpointTests(unittest.TestCase):
    def setUp(self):
        self.saved = (backends._active, list(backends._endpoints),
                      dict(backends._chosen), dict(backends._extract_chosen),
                      backends._extract_url)
        backends._active = READER
        backends._chosen.clear()
        backends._chosen[READER] = "typhoon-ocr"
        backends._extract_chosen.clear()
        backends._extract_url = None
        patcher = mock.patch.object(backends, "probe",
                                    side_effect=lambda url, force=False: PROBES[url])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        (backends._active, backends._endpoints, chosen, extract_chosen,
         backends._extract_url) = self.saved
        backends._chosen.clear(); backends._chosen.update(chosen)
        backends._extract_chosen.clear(); backends._extract_chosen.update(extract_chosen)

    def test_one_url_is_the_default_and_unchanged(self):
        info = backends.extract_status()
        self.assertFalse(backends.extract_separate())
        self.assertEqual(info["url"], READER)
        self.assertEqual(info["model"], "typhoon-ocr")
        url, body = backends.structured_request([], None, 10, info)
        self.assertEqual(url, READER + "/v1/chat/completions")

    def test_a_separate_server_takes_every_text_request(self):
        backends.select_extract_url(EXTRACTOR, unload=False)
        self.assertTrue(backends.extract_separate())
        info = backends.extract_status()
        self.assertEqual((info["url"], info["kind"], info["model"]),
                         (EXTRACTOR, "vllm", "qwen-big"))
        self.assertTrue(info["text_available"])
        url, body = backends.structured_request([], None, 10, info)
        self.assertEqual(url, EXTRACTOR + "/v1/chat/completions")
        self.assertEqual(body["model"], "qwen-big")
        # The reader is untouched.
        self.assertEqual(backends.status()["url"], READER)
        self.assertEqual(backends.status()["model"], "typhoon-ocr")

    def test_a_model_not_served_there_is_unavailable(self):
        backends.select_extract_url(EXTRACTOR, unload=False)
        backends._extract_chosen[EXTRACTOR] = "gemma4:e4b"
        info = backends.extract_status()
        self.assertFalse(info["text_available"])
        self.assertIn("not served", info["text_reason"])

    def test_the_reading_server_as_both_collapses_to_one_url(self):
        backends.select_extract_url(READER, unload=False)
        self.assertFalse(backends.extract_separate())
        self.assertEqual(backends.configured_extract_url(), READER)
        self.assertEqual(backends.extract_status()["url"], READER)

    def test_empty_puts_it_back_on_the_reader(self):
        backends.select_extract_url(EXTRACTOR, unload=False)
        backends.select_extract_url("", unload=False)
        self.assertEqual(backends.configured_extract_url(), "")
        self.assertEqual(backends.extract_status()["url"], READER)

    def test_run_log_names_the_extraction_server_only_where_it_differs(self):
        split = runlog._extract_cells({"url": READER, "model": "typhoon-ocr",
                                       "extracted": {"model": "qwen-big",
                                                     "url": EXTRACTOR}})
        same = runlog._extract_cells({"url": READER, "model": "typhoon-ocr",
                                      "extracted": {"model": "typhoon-ocr",
                                                    "url": READER}})
        self.assertEqual(split["extract_server"], EXTRACTOR)
        self.assertEqual(same["extract_server"], "")


if __name__ == "__main__":
    unittest.main()
