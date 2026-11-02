"""What the server sends the browser: compression, revalidation and preview size."""
import gzip
import io
import json
import tempfile
import unittest
import zlib
from unittest.mock import patch

from flask import Response, request
from PIL import Image

import app
from userstore import UserStore


GZIP = {'Accept-Encoding': 'gzip'}


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_page_scripts_and_styles_are_compressed(self):
        for name in ('benchmark.js', 'benchmark.css'):
            url = '/' + app.benchmark_asset(name)
            plain = self.client.get(url)
            plain.direct_passthrough = False
            self.addCleanup(plain.close)
            packed = self.client.get(url, headers=GZIP)
            with self.subTest(name=name):
                self.assertEqual(packed.headers['Content-Encoding'], 'gzip')
                self.assertEqual(gzip.decompress(packed.data), plain.data)
                self.assertLess(len(packed.data), len(plain.data) / 2)
                self.assertIn('immutable', packed.headers['Cache-Control'])
                self.assertIn('Accept-Encoding', packed.headers['Vary'])

    def test_pdf_runtime_is_compressed(self):
        response = self.client.get('/api/local-preview/pdfjs/6.4.299/legacy/build/pdf.min.mjs', headers=GZIP)
        self.assertEqual(response.headers['Content-Encoding'], 'gzip')
        self.assertTrue(gzip.decompress(response.data))
        self.assertNotIn('Accept-Ranges', response.headers)

    def test_workspace_page_is_compressed_and_revalidated(self):
        first = self.client.get('/', headers=GZIP)
        self.assertEqual(first.headers['Content-Encoding'], 'gzip')
        self.assertIn(b'<html', gzip.decompress(first.data).lower())
        self.assertEqual(first.headers['Cache-Control'], 'no-cache')
        again = self.client.get('/', headers={**GZIP, 'If-None-Match': first.headers['ETag']})
        self.assertEqual(again.status_code, 304)
        self.assertEqual(again.data, b'')

    def test_images_are_left_alone(self):
        with app.app.test_request_context('/', headers=GZIP):
            image = Response(b'x' * 4096, mimetype='image/png')
            self.assertNotIn('Content-Encoding', app._gzip_response(image).headers)

    def test_a_stream_is_decodable_after_every_chunk(self):
        closed = []

        def lines():
            try:
                yield json.dumps({'event': 'page', 'text': 'หน้าแรก'}) + '\n'
                yield json.dumps({'event': 'page_done'}) + '\n'
            finally:
                closed.append(True)

        with app.app.test_request_context('/', headers=GZIP):
            response = app._gzip_response(Response(lines(), mimetype='application/x-ndjson'))
            self.assertEqual(response.headers['Content-Encoding'], 'gzip')
            reader = zlib.decompressobj(31)
            first = reader.decompress(next(response.response))
            self.assertEqual(json.loads(first)['text'], 'หน้าแรก')
            second = reader.decompress(next(response.response))
            self.assertEqual(json.loads(second)['event'], 'page_done')
            response.close()
        self.assertEqual(closed, [True])

    def test_a_stream_is_untouched_for_a_client_without_gzip(self):
        with app.app.test_request_context('/'):
            response = app._gzip_response(Response(iter(['a\n']), mimetype='application/x-ndjson'))
            self.assertNotIn('Content-Encoding', response.headers)
            self.assertEqual(response.get_data(), b'a\n')


class PortalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = UserStore(temp.name, 32)
        service = type('Service', (), {'store': self.store, 'start': lambda self: None})()
        patcher = patch.dict(app.app.extensions, {'user_service': service})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = app.app.test_client()
        self.batch = self.store.create_batch(5, self.store.settings())

    def test_polling_an_unchanged_job_costs_a_304(self):
        job = self.store.add_job(self.batch, 'upload_01', 'one.pdf', b'x')
        url = '/api/user/jobs/' + job['id']
        first = self.client.get(url, headers=GZIP)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers['Cache-Control'], 'private, no-cache')
        same = self.client.get(url, headers={**GZIP, 'If-None-Match': first.headers['ETag']})
        self.assertEqual(same.status_code, 304)
        self.assertEqual(same.data, b'')
        self.store.update(job['id'], progress='moved on')
        moved = self.client.get(url, headers={**GZIP, 'If-None-Match': first.headers['ETag']})
        self.assertEqual(moved.status_code, 200)

    def test_job_status_carries_no_ocr_text(self):
        job = self.store.add_job(self.batch, 'upload_01', 'one.pdf', b'x')
        self.store.update(job['id'], result={'text': 'OCR TEXT', 'pages': ['OCR TEXT']})
        response = self.client.post('/api/user/job-status',
                                    json={'ids': [job['id'], 'not-a-job', 7]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row['id'] for row in response.json['jobs']], [job['id']])
        self.assertEqual(set(response.json['jobs'][0]), {'id', 'status', 'progress', 'updated_at'})
        self.assertEqual(self.client.post('/api/user/job-status', json={'ids': ['a'] * 501}).status_code, 400)
        self.assertEqual(self.client.post('/api/user/job-status', json={'ids': 'a'}).status_code, 400)

    def test_a_mutating_or_failed_reply_is_not_kept(self):
        self.assertEqual(self.client.get('/api/user/jobs/' + 'a' * 32).headers['Cache-Control'], 'no-store')
        self.assertEqual(self.client.post('/api/user/job-status', json={'ids': []}).headers['Cache-Control'], 'no-store')

    def test_a_document_preview_is_kept_by_the_browser(self):
        with app.fitz.open() as document:
            document.new_page().insert_text((50, 50), 'Page 1')
            content = document.tobytes()
        job = self.store.add_job(self.batch, 'upload_01', 'one.pdf', content)
        response = self.client.get('/api/user/jobs/' + job['id'] + '/preview/pages/1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Cache-Control'], 'private, max-age=86400')

    def test_a_scanned_page_is_sent_as_jpeg(self):
        noisy = io.BytesIO()
        Image.effect_noise((1200, 1600), 40).convert('RGB').save(noisy, 'PNG')
        with app.fitz.open() as document:
            document.new_page().insert_image(app.fitz.Rect(0, 0, 595, 842), stream=noisy.getvalue())
            content = document.tobytes()
        job = self.store.add_job(self.batch, 'upload_01', 'scan.pdf', content)
        response = self.client.get('/api/user/jobs/' + job['id'] + '/preview/pages/1')
        self.assertEqual(response.mimetype, 'image/jpeg')
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertLessEqual(max(image.size), 1800)

    def test_a_large_photo_is_fitted_only_when_the_preview_asks(self):
        photo = io.BytesIO()
        Image.effect_noise((3000, 4000), 60).convert('RGB').save(photo, 'PNG')
        self.assertGreater(photo.tell(), 1024 * 1024)
        job = self.store.add_job(self.batch, 'upload_01', 'photo.png', photo.getvalue())
        url = '/api/user/jobs/' + job['id'] + '/preview'
        fitted = self.client.get(url + '?fit=1')
        self.assertEqual(fitted.mimetype, 'image/jpeg')
        self.assertLess(len(fitted.data), photo.tell())
        with Image.open(io.BytesIO(fitted.data)) as image:
            self.assertEqual(max(image.size), 1800)
        original = self.client.get(url)
        self.assertEqual(original.mimetype, 'image/png')
        original.direct_passthrough = False
        self.assertEqual(original.data, photo.getvalue())
        original.close()

    def test_a_small_or_unreadable_image_is_sent_as_it_is(self):
        job = self.store.add_job(self.batch, 'upload_01', 'small.png', b'not really a png')
        response = self.client.get('/api/user/jobs/' + job['id'] + '/preview?fit=1')
        self.assertEqual(response.mimetype, 'image/png')
        response.close()
        job = self.store.add_job(self.batch, 'upload_02', 'big.png', b'x' * (2 * 1024 * 1024))
        response = self.client.get('/api/user/jobs/' + job['id'] + '/preview?fit=1')
        self.assertEqual(response.mimetype, 'image/png')
        response.close()


if __name__ == '__main__':
    unittest.main()
