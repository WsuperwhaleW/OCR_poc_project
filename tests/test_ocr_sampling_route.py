import unittest

import app as app_module
import settings


class SamplingRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.saved = (settings.OCR_TEMPERATURE, settings.OCR_TOP_P, settings.OCR_TOP_K)

    def tearDown(self):
        settings.OCR_TEMPERATURE, settings.OCR_TOP_P, settings.OCR_TOP_K = self.saved

    def test_set_and_reset(self):
        r = self.client.post("/api/ocr/sampling", json={"temperature": 0.3, "top_p": 0.8, "top_k": 40})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(settings.ocr_sampling(), {"temperature": 0.3, "top_p": 0.8, "top_k": 40})
        r = self.client.post("/api/ocr/sampling", json={"reset": True})
        self.assertEqual(r.get_json()["temperature"], settings.OCR_SAMPLING_DEFAULT["temperature"])

    def test_partial_update_keeps_the_rest(self):
        self.client.post("/api/ocr/sampling", json={"temperature": 0.0})
        self.assertEqual(settings.OCR_TOP_P, self.saved[1])

    def test_out_of_range_is_refused_and_changes_nothing(self):
        for bad in ({"temperature": 7}, {"top_p": 0}, {"top_k": 1.5}, {"top_k": -1}, {"temperature": "x"}):
            r = self.client.post("/api/ocr/sampling", json=bad)
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual((settings.OCR_TEMPERATURE, settings.OCR_TOP_P, settings.OCR_TOP_K), self.saved)

    def test_pass1_payload_follows_the_setting(self):
        settings.set_ocr_sampling(0.0, 1.0, 1)
        self.assertEqual(settings.ocr_sampling(), {"temperature": 0.0, "top_p": 1.0, "top_k": 1})


if __name__ == "__main__":
    unittest.main()
