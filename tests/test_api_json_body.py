"""Bad JSON must fail before model calls, settings changes or log queries."""
import unittest
from unittest.mock import patch

import app


class JsonBodyTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_non_object_and_malformed_json_returns_400_json(self):
        paths = ('/api/classify', '/api/extract', '/api/extract/stream',
                 '/api/extract/mode', '/api/runs/query', '/OCR/api/extract')
        bodies = ('["invalid-object"]', '[]', '"text"', '42', '0',
                  'true', 'false', 'null', '{broken', '')
        with patch.object(app, 'extract_fields') as extract:
            for path in paths:
                for body in bodies:
                    with self.subTest(path=path, body=body):
                        response = self.client.post(path, data=body, content_type='application/json')
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(response.mimetype, 'application/json')
                        self.assertIn('JSON object', response.json['error'])
            extract.assert_not_called()

    def test_valid_object_and_omitted_body_keep_query_defaults(self):
        with patch.object(app, '_runs_payload', return_value={'rows': []}) as payload:
            for options in ({}, {'json': {}}, {'json': {'limit': 7}}):
                response = self.client.post('/api/runs/query', **options)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json, {'rows': []})
            self.assertEqual(payload.call_count, 3)
            self.assertEqual(payload.call_args.args[0], 7)

    def test_json_media_type_variants_and_user_put_are_validated(self):
        response = self.client.put('/api/user/settings', data='[]',
                                   content_type='application/problem+json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.mimetype, 'application/json')

    def test_multipart_upload_still_reaches_its_handler(self):
        response = self.client.post('/api/user/previews', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('JSON object', response.json['error'])

    def test_cross_origin_user_request_is_rejected_before_body_validation(self):
        response = self.client.put('/api/user/settings', json=[],
                                   headers={'Origin': 'http://elsewhere'})
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
