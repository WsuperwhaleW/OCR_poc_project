import unittest
from unittest.mock import patch
import app


class BenchmarkAssetsTests(unittest.TestCase):
    def test_versioned_assets_are_cacheable_and_mount_safe(self):
        client = app.app.test_client()
        with app.app.test_request_context('/'):
            urls = {name: app.benchmark_asset(name) for name in app._BENCHMARK_ASSETS}
        for name, url in urls.items():
            for prefix in ('', '/OCR'):
                with self.subTest(name=name, prefix=prefix):
                    response = client.get(prefix + '/' + url)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('immutable', response.headers['Cache-Control'])
                    self.assertEqual(response.mimetype, 'text/css' if name.endswith('.css') else 'text/javascript')
                    self.assertNotIn(b'{{', response.data)
                    response.close()
        self.assertEqual(client.get('/api/benchmark-assets/wrong/benchmark.js').status_code, 404)
        self.assertEqual(client.get('/api/benchmark-assets/wrong/app.py').status_code, 404)

    def test_evidence_revalidation_and_updates(self):
        client = app.app.test_client()
        item = dict(case='demo.pdf', page=1, truth_case=None, attempts=[], fields=[{'value': 1}])
        with patch.object(app._STRESS_EVIDENCE, 'get', return_value=item):
            first = client.get('/api/stress/evidence/demo')
            tag = first.headers['ETag']
            self.assertEqual(client.get('/api/stress/evidence/demo', headers={'If-None-Match': tag}).status_code, 304)
            item['fields'] = [{'value': 2}]
            changed = client.get('/api/stress/evidence/demo', headers={'If-None-Match': tag})
            self.assertEqual(changed.status_code, 200)
            self.assertNotEqual(changed.headers['ETag'], tag)
