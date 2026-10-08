import unittest
from unittest import mock

import settings


class OcrSamplingTests(unittest.TestCase):
    def test_default_is_greedy(self):
        self.assertEqual(settings.ocr_sampling(),
                         {"temperature": 0.0, "top_p": 1.0, "top_k": 1})

    def test_top_k_is_omitted_when_zero(self):
        with mock.patch.object(settings, "OCR_TOP_K", 0):
            self.assertNotIn("top_k", settings.ocr_sampling())

    def test_typhoons_recommended_values_can_be_set(self):
        with mock.patch.object(settings, "OCR_TEMPERATURE", 0.1),                 mock.patch.object(settings, "OCR_TOP_P", 0.6),                 mock.patch.object(settings, "OCR_TOP_K", 0):
            self.assertEqual(settings.ocr_sampling(),
                             {"temperature": 0.1, "top_p": 0.6})


if __name__ == "__main__":
    unittest.main()
