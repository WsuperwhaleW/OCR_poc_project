"""Local previews expose filenames and renderer assets, never local documents."""
import unittest
from unittest.mock import patch

from PIL import Image

import app


class StressLocalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.evidence = app._StressEvidence(5)
        self.client = app.app.test_client()
        self.patch = patch.object(app, '_STRESS_EVIDENCE', self.evidence)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def add(self, case, truth=None):
        self.evidence.put('local-preview-test', case, 2, [{
            'page': Image.new('RGB', (20, 30)), 'number': 2,
            'detail': 'medium', 'text': 'OCR', 'looped': False,
            'truncated': False, 'dry': None,
        }], truth)

    def test_benchmark_uses_exact_manifest_basename_without_reading_source_bytes(self):
        self.add('sol014', 'sol014')
        with patch.object(app.scoring, 'cases_index', return_value={
                'sol014': {'pdf': 'fixtures/credit-note.pdf'}}), \
                patch.object(app, 'case_bytes', side_effect=AssertionError('No document read')):
            data = self.client.get('/api/stress/evidence/local-preview-test').get_json()
        self.assertEqual(data['filename'], 'credit-note.pdf')
        self.assertEqual(data['attempts'][0]['page'], 2)
        self.assertEqual(data['attempts'][0]['text'], 'OCR')

    def test_realdata_nested_paths_return_only_filename(self):
        for case in ('rtv/2026/return.pdf', r'rtv\2026\return.pdf'):
            self.add(case)
            with patch.object(app.scoring, 'cases_index', return_value={}):
                data = self.client.get('/api/stress/evidence/local-preview-test').get_json()
            self.assertEqual(data['filename'], 'return.pdf')

    def test_benchmark_without_truth_still_resolves_source_name(self):
        self.add('sol016')
        with patch.object(app.scoring, 'cases_index', return_value={'sol016': {'pdf': 'sol016.png'}}):
            data = self.client.get('/api/stress/evidence/local-preview-test').get_json()
        self.assertEqual(data['filename'], 'sol016.png')

    def test_model_image_endpoint_still_returns_prepared_png(self):
        self.add('file.pdf')
        response = self.client.get('/api/stress/evidence/local-preview-test/0.png')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/png')
        self.assertTrue(response.data.startswith(b'\x89PNG'))

    def test_pdf_runtime_is_available_under_root_and_deployment_prefix(self):
        for prefix in ('', '/ocr'):
            for name, mime in (('legacy/build/pdf.min.mjs', 'text/javascript'),
                               ('legacy/build/pdf.worker.min.mjs', 'text/javascript'),
                               ('wasm/openjpeg.wasm', 'application/wasm')):
                with self.subTest(prefix=prefix, name=name):
                    response = self.client.get(prefix + '/api/local-preview/pdfjs/6.4.299/' + name)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.mimetype, mime)
                    self.assertIn('max-age=31536000', response.headers['Cache-Control'])
                    response.close()

    def test_runtime_route_cannot_escape_vendor_root(self):
        base = '/api/local-preview/pdfjs/6.4.299/'
        for name in ('../../../../app.py', '%2e%2e/%2e%2e/app.py', 'absent.mjs'):
            self.assertEqual(self.client.get(base + name).status_code, 404)


if __name__ == '__main__':
    unittest.main()
