import unittest
from unittest import mock

import settings


class OcrSamplingTests(unittest.TestCase):
    def test_defaults_are_typhoons_recommended_values(self):
        fields = settings.ocr_sampling()
        self.assertEqual(fields["temperature"], 0.1)
        self.assertEqual(fields["top_p"], 0.6)

    def test_top_k_is_omitted_unless_set(self):
        self.assertNotIn("top_k", settings.ocr_sampling())
        with mock.patch.object(settings, "OCR_TOP_K", 1):
            self.assertEqual(settings.ocr_sampling()["top_k"], 1)

    def test_greedy_can_be_restored(self):
        with mock.patch.object(settings, "OCR_TEMPERATURE", 0.0), \
                mock.patch.object(settings, "OCR_TOP_P", 1.0), \
                mock.patch.object(settings, "OCR_TOP_K", 1):
            self.assertEqual(settings.ocr_sampling(),
                             {"temperature": 0.0, "top_p": 1.0, "top_k": 1})


if __name__ == "__main__":
    unittest.main()
