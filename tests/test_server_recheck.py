"""Re-check follows a model replaced at the same URL, including pass 2."""

import unittest
from unittest import mock

import app
import engine
import backends
import workspace


READER = "http://recheck-reader:8000"
EXTRACTOR = "http://recheck-extractor:8001"


class RecheckTests(unittest.TestCase):
    def setUp(self):
        self.context = workspace.use(None)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        for name, value in {
            "_active": READER, "_extract_url": EXTRACTOR,
            "_endpoints": [READER, EXTRACTOR],
            "_chosen": {READER: "reader-old"},
            "_extract_chosen": {EXTRACTOR: "extract-old"},
            "_kind_pinned": {READER: "vllm", EXTRACTOR: "vllm"},
            "_kind_seen": {}, "_cache": {},
        }.items():
            patcher = mock.patch.object(backends, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.models = {READER: ["reader-old"], EXTRACTOR: ["extract-old"]}
        self.asked = []
        patcher = mock.patch.object(backends.requests, "get", side_effect=self.get)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = app.app.test_client()
        backends.overview(force=True)
        self.asked.clear()

    def get(self, url, **kwargs):
        self.asked.append(url)
        endpoint = url.removesuffix("/v1/models")
        names = self.models.get(endpoint)
        if names is None:
            return mock.Mock(status_code=503)
        return mock.Mock(status_code=200, json=lambda: {
            "data": [{"id": name, "owned_by": "vllm"} for name in names]})

    def recheck(self, **kwargs):
        response = self.client.get("/api/servers?probe=1", **kwargs)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_replaced_extractor_is_used_by_status_and_inference(self):
        self.models[EXTRACTOR] = ["extract-new"]
        result = self.recheck()
        self.assertEqual(result["extract"]["model"], "extract-new")
        self.assertEqual(result["extract"]["chosen"], "")
        self.assertTrue(result["extract"]["available"])
        self.assertEqual(result["extract"]["models"], ["extract-new"])
        self.assertEqual(result["server"]["model"], "reader-old")
        self.assertEqual(self.asked.count(EXTRACTOR + "/v1/models"), 1)
        info = backends.extract_status()
        url, body = backends.structured_request([], None, 10, info)
        self.assertEqual(url, EXTRACTOR + "/v1/chat/completions")
        self.assertEqual(body["model"], "extract-new")

    def test_valid_extraction_choice_is_preserved(self):
        self.models[EXTRACTOR] = ["extract-new", "extract-old"]
        result = self.recheck()
        self.assertEqual(result["extract"]["model"], "extract-old")
        self.assertEqual(result["extract"]["chosen"], "extract-old")

    def test_offline_extractor_keeps_choice_until_successful_recheck(self):
        self.models[EXTRACTOR] = None
        result = self.recheck()
        self.assertFalse(result["extract"]["available"])
        self.assertEqual(result["extract"]["chosen"], "extract-old")
        self.models[EXTRACTOR] = ["extract-new"]
        self.assertEqual(self.recheck()["extract"]["model"], "extract-new")

    def test_normal_status_does_not_silently_change_an_explicit_choice(self):
        self.models[EXTRACTOR] = ["extract-new"]
        backends._cache.pop(EXTRACTOR)
        info = backends.extract_status()
        self.assertEqual(info["model"], "extract-old")
        self.assertFalse(info["text_available"])

    def test_shared_endpoint_drops_missing_extraction_choice(self):
        backends._extract_url = None
        backends._extract_chosen[READER] = "extract-old"
        result = self.recheck()
        self.assertEqual(result["extract"]["model"], "reader-old")
        self.assertTrue(result["extract"]["same_as_reading"])
        self.assertTrue(result["extract"]["available"])

    def test_reader_replacement_drops_missing_reader_choice(self):
        self.models[READER] = ["dots.ocr-new"]
        with mock.patch.object(engine, '_ocr_profile', 'typhoon'):
            result = self.recheck()
            self.assertEqual(result['ocr_profile'], 'dots')
            self.assertEqual(app.ocr_profile(), 'dots')
        self.assertEqual(result["server"]["model"], "dots.ocr-new")
        self.assertNotIn(READER, backends._chosen)

    def test_rechecking_same_reader_preserves_manual_profile_override(self):
        with mock.patch.object(engine, '_ocr_profile', 'dots'):
            self.assertEqual(self.recheck()['ocr_profile'], 'dots')

    def test_recheck_only_changes_current_workspace_choices(self):
        ws = workspace.Workspace("recheck-test-tab")
        ws.put("active", READER)
        ws.put("extract_url", EXTRACTOR)
        ws.put("chosen", {READER: "reader-old"})
        ws.put("extract_chosen", {EXTRACTOR: "extract-old"})
        self.models[EXTRACTOR] = ["extract-new"]
        with workspace.use(ws):
            self.assertEqual(backends.overview(force=True)["extract"]["model"],
                             "extract-new")
            self.assertNotIn(EXTRACTOR, backends._extract_chosen_map())
        self.assertEqual(backends._extract_chosen[EXTRACTOR], "extract-old")

    def test_recheck_response_cannot_be_reused_from_http_cache(self):
        response = self.client.get("/api/servers?probe=1")
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))


if __name__ == "__main__":
    unittest.main()
