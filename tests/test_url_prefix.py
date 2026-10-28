"""Served under a sub-path: nginx at host:3000/OCR/ in front of the app.

Nothing is configured for the path. The page builds every request from its own
address (`API` in the template), and the app takes whatever prefix a request
arrives under off it (`_ProxyPrefix`), for a proxy that passes the path through
unchanged. A proxy that strips it sends plain paths, which must still work.
"""
import re
import unittest

import app


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_any_prefix_reaches_the_api(self):
        for prefix in ("/OCR", "/tools/ocr"):
            res = self.client.get(prefix + "/api/runs/stat")
            self.assertEqual(res.status_code, 200, prefix)
            self.assertIn("size", res.get_json())

    def test_the_page_is_served_at_its_prefix(self):
        for path in ("/OCR/", "/OCR"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)
            self.assertIn("window.API", res.get_data(as_text=True))

    def test_an_unprefixed_request_still_works(self):
        self.assertEqual(self.client.get("/api/runs/stat").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_a_forwarded_prefix_is_taken_first(self):
        res = self.client.get("/OCR/", headers={"X-Forwarded-Prefix": "/OCR"})
        self.assertEqual(res.status_code, 200)

    def test_a_file_is_not_mistaken_for_the_page(self):
        self.assertEqual(self.client.get("/favicon.ico").status_code, 404)

    def test_a_post_to_an_unknown_path_is_not_the_page(self):
        self.assertEqual(self.client.post("/OCR/").status_code, 404)

    def test_no_request_on_the_page_names_a_fixed_path(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('window.OCR_BASE = location.pathname.replace(/\\/+$/, "")', page)
        self.assertFalse(re.findall(r"fetch\(\s*['\"`]/api/", page))
        for prefix in ('/OCR', '/tools/ocr'):
            mounted = self.client.get(prefix + '/').text
            assets = re.findall(r'(?:href|src)="([^"]*/benchmark-assets/[^"]+)"', mounted)
            self.assertTrue(assets)
            self.assertTrue(all(url.startswith('api/benchmark-assets/') for url in assets))
            self.assertIn("benchmarkBase.href = location.origin + window.OCR_BASE + '/'", mounted)
        self.assertNotIn("appBase", page)       # the App path control is gone


if __name__ == "__main__":
    unittest.main()
