"""File caps exclude multipart overhead; small uploads reach the app directly."""
import io
import unittest
from unittest.mock import patch

from PIL import Image
import app
import engine
class WorkspaceUploadLimitTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def preview(self, data, prefix=''):
        response = self.client.post(prefix + '/api/preview', data={
            'image': (io.BytesIO(data), 'sample.pdf'), 'detail': 'original'})
        response.request.close()
        response.request.environ['wsgi.input'].close()
        return response

    def test_one_two_and_four_mib_uploads_reach_workspace_preview_without_ocr(self):
        page = Image.new('RGB', (8, 8), 'white')
        with patch.object(engine, 'load_pages', return_value=([page], 1)) as load, \
                patch('requests.sessions.Session.request', side_effect=AssertionError('No OCR/network')):
            for prefix in ('', '/ocr'):
                for size in (1, 2, 4):
                    with self.subTest(prefix=prefix, size=size):
                        data = b'x' * (size * 1024 * 1024)
                        with self.preview(data, prefix) as response:
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.mimetype, 'image/png')
                        self.assertEqual(len(load.call_args.args[0]), len(data))

    def test_workspace_allows_exact_file_cap_with_envelope_but_rejects_one_extra_byte(self):
        page = Image.new('RGB', (8, 8), 'white')
        with patch.object(app, 'MAX_UPLOAD_MB', 1), \
                patch.dict(app.app.config, MAX_CONTENT_LENGTH=2 * 1024 * 1024), \
                patch.object(engine, 'load_pages', return_value=([page], 1)) as load:
            with self.preview(b'x' * (1024 * 1024)) as response:
                self.assertEqual(response.status_code, 200)
            self.assertEqual(load.call_count, 1)
            for size in (1024 * 1024 + 1, 2 * 1024 * 1024 + 1):
                with self.subTest(size=size), self.preview(b'x' * size) as response:
                    self.assertEqual(response.status_code, 413)
                    self.assertEqual(response.json['code'], 'upload_too_large')
                    self.assertEqual(response.json['limit_mb'], 1)
            self.assertEqual(load.call_count, 1)


if __name__ == '__main__':
    unittest.main()
