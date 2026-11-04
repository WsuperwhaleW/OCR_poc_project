"""The Worker API end to end: API key → submit → status → result → ack.

Built 2026-11-04 with /api/v1 (worker-service plan, steps 3 + 4, API keys only).
The real engine runs; only the model servers are fake (`tests/fakemodel.py`).
"""
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

import engine
import serviceapi
import servicestore
import userweb
import webcommon
from tests.fakemodel import OCR_MODEL, TEXT_MODEL, FakeModelServer

ROOT = Path(__file__).resolve().parent.parent
PDF = (ROOT / 'mockOcr' / 'invoice_sol002.pdf').read_bytes()


class WorkerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'OCR_USER_DATA_DIR': cls.directory.name})
        cls.env.start()
        cls.ocr = FakeModelServer(OCR_MODEL).__enter__()
        cls.text = FakeModelServer(TEXT_MODEL).__enter__()
        app = Flask('worker_api_test')
        webcommon.install(app, engine.MAX_UPLOAD_MB)
        service = userweb.register(app, engine)
        serviceapi.register(app, engine, service)
        cls.service = staticmethod(service)
        app.before_request(webcommon.validate_api_json_object)
        cls.app, cls.client = app, app.test_client()
        store = cls.service().store
        store.save_settings({**store.settings(), 'ocr_url': cls.ocr.url, 'extraction_url': cls.text.url})
        cls.key_a = cls.new_key('ERP หลัก')
        cls.key_b = cls.new_key('ทดสอบ')

    @classmethod
    def tearDownClass(cls):
        cls.service().close()
        cls.ocr.__exit__(None, None, None)
        cls.text.__exit__(None, None, None)
        cls.env.stop()
        cls.directory.cleanup()

    @classmethod
    def new_key(cls, name):
        response = cls.client.post('/api/user/api-keys', json={'name': name})
        assert response.status_code == 201, response.get_json()
        return response.get_json()['key']

    def auth(self, key=None, **headers):
        return {'Authorization': 'Bearer ' + (key or self.key_a), **headers}

    def submit(self, key=None, content=PDF, name='invoice.pdf', headers=None, **form):
        return self.client.post('/api/v1/jobs', headers=self.auth(key, **(headers or {})),
                                data={'file': (io.BytesIO(content), name), **form})

    def wait(self, job_id, key=None, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = self.client.get('/api/v1/jobs/' + job_id, headers=self.auth(key)).get_json()
            if body['status'] not in ('queued', 'processing'):
                return body
            time.sleep(0.2)
        self.fail('job did not finish')

    def test_full_flow_submit_status_result_ack(self):
        response = self.submit(client_ref='ERP-000123')
        self.assertEqual(response.status_code, 202, response.get_json())
        body = response.get_json()
        job_id = body['job_id']
        self.assertEqual(body['status'], 'queued')
        self.assertEqual(body['status_url'], '/api/v1/jobs/' + job_id)
        self.assertEqual(response.headers['Location'], body['status_url'])

        status = self.wait(job_id)
        self.assertIn(status['status'], ('succeeded', 'partial'))
        self.assertEqual((status['client_ref'], status['stage'], status['result_version']), ('ERP-000123', 'done', 1))
        self.assertIsNone(status['acknowledged_at'])
        self.assertEqual(set(status['summary']), {'documents', 'schema_valid', 'warnings', 'missing_fields'})
        self.assertIsNotNone(status['timing']['ocr_seconds'])

        result = self.client.get(f'/api/v1/jobs/{job_id}/result', headers=self.auth())
        self.assertEqual(result.status_code, 200)
        data = result.get_json()
        self.assertEqual(data['user_json']['schemaVersion'], '7.0')
        self.assertEqual((data['job_id'], data['client_ref'], data['result_version']), (job_id, 'ERP-000123', 1))
        self.assertEqual(data['processing']['ocr_model'], OCR_MODEL)
        self.assertEqual(data['processing']['extraction_model'], TEXT_MODEL)
        self.assertNotIn('report', data)
        again = self.client.get(f'/api/v1/jobs/{job_id}/result?report=1', headers=self.auth()).get_json()
        self.assertEqual(again['event_id'], data['event_id'], 'same attempt, same event id')
        self.assertIn('missing_fields', again['report'])

        self.assertEqual(self.client.post(f'/api/v1/jobs/{job_id}/ack', headers=self.auth()).status_code, 204)
        first = self.client.get('/api/v1/jobs/' + job_id, headers=self.auth()).get_json()['acknowledged_at']
        self.assertTrue(first)
        self.client.post(f'/api/v1/jobs/{job_id}/ack', headers=self.auth())
        self.assertEqual(self.client.get('/api/v1/jobs/' + job_id, headers=self.auth()).get_json()['acknowledged_at'], first)
        # Ack deletes nothing.
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}/result', headers=self.auth()).status_code, 200)

        # /user History says which client sent it.
        listed = self.client.get('/api/user/jobs').get_json()['jobs']
        row = next(job for job in listed if job['id'] == job_id)
        self.assertEqual((row['client_name'], row['client_ref']), ('ERP หลัก', 'ERP-000123'))

    def test_keys_are_required_and_scope_jobs(self):
        self.assertEqual(self.client.post('/api/v1/jobs', data={}).status_code, 401)
        self.assertEqual(self.client.get('/api/v1/jobs/' + 'a' * 32,
                                         headers={'Authorization': 'Bearer ocr_wrong'}).status_code, 401)
        response = self.submit()
        job_id = response.get_json()['job_id']
        for path in ('', '/result'):
            self.assertEqual(self.client.get(f'/api/v1/jobs/{job_id}{path}', headers=self.auth(self.key_b)).status_code, 404)
        self.assertEqual(self.client.post(f'/api/v1/jobs/{job_id}/ack', headers=self.auth(self.key_b)).status_code, 404)
        self.wait(job_id)

    def test_revoked_key_stops_working_at_once(self):
        key = self.new_key('ชั่วคราว')
        self.assertEqual(self.client.get('/api/v1/jobs/' + 'a' * 32, headers=self.auth(key)).status_code, 404)
        record = next(k for k in self.client.get('/api/user/api-keys').get_json()['keys'] if k['name'] == 'ชั่วคราว')
        self.assertNotIn('hash', record)
        self.assertTrue(record['hint'].startswith('ocr_'))
        self.assertTrue(record['last_used_at'])
        revoked = self.client.post(f"/api/user/api-keys/{record['id']}/revoke").get_json()['record']
        self.assertTrue(revoked['revoked_at'])
        self.assertEqual(self.client.get('/api/v1/jobs/' + 'a' * 32, headers=self.auth(key)).status_code, 401)

    def test_key_is_stored_only_as_a_hash(self):
        with self.service().store.connect() as db:
            stored = [row['hash'] for row in db.execute('SELECT hash FROM api_keys')]
        self.assertNotIn(self.key_a, stored)
        self.assertIn(servicestore.hash_key(self.key_a), stored)

    def test_idempotency_key_replays_or_conflicts(self):
        headers = {'Idempotency-Key': 'upload-7781'}
        first = self.submit(headers=headers, client_ref='R1')
        self.assertEqual(first.status_code, 202)
        replay = self.submit(headers=headers, client_ref='R1')
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.get_json(), first.get_json())
        self.assertEqual(replay.headers.get('Idempotent-Replayed'), 'true')
        conflict = self.submit(headers=headers, client_ref='R2')
        self.assertEqual((conflict.status_code, conflict.get_json()['error']), (409, 'idempotency_conflict'))
        # Keys are per client: the other client may use the same Idempotency-Key.
        other = self.submit(key=self.key_b, headers=headers, client_ref='R1')
        self.assertEqual(other.status_code, 202)
        self.assertNotEqual(other.get_json()['job_id'], first.get_json()['job_id'])
        for job in (first, other):
            self.wait(job.get_json()['job_id'], key=self.key_a if job is first else self.key_b)

    def test_bad_requests(self):
        cases = [
            (self.submit(name='notes.txt'), 400, 'invalid_file'),
            (self.submit(content=b''), 400, 'invalid_file'),
            (self.client.post('/api/v1/jobs', headers=self.auth(), data={}), 400, 'invalid_file'),
            (self.submit(callback_url='https://main.example/cb'), 400, 'callback_not_available'),
            (self.submit(client_ref='x' * 201), 400, 'invalid_client_ref'),
            (self.submit(headers={'Idempotency-Key': 'มีภาษาไทย'}), 400, 'invalid_idempotency_key'),
        ]
        for response, status, code in cases:
            with self.subTest(code):
                self.assertEqual((response.status_code, response.get_json()['error']), (status, code))

    def test_file_over_the_cap_is_a_json_413(self):
        response = self.submit(content=b'x' * ((engine.MAX_UPLOAD_MB + 2) * 1024 * 1024))
        self.assertEqual((response.status_code, response.get_json()['error']), (413, 'file_too_large'))

    def test_result_before_finish_and_after_failure(self):
        broken = self.submit(content=b'not a pdf', name='broken.pdf').get_json()['job_id']
        status = self.wait(broken)
        self.assertEqual(status['status'], 'failed')
        self.assertTrue(status['error'])
        response = self.client.get(f'/api/v1/jobs/{broken}/result', headers=self.auth())
        self.assertEqual((response.status_code, response.get_json()['error']), (422, 'job_failed'))
        with patch.object(self.service(), 'start'):
            queued = self.submit().get_json()['job_id']
            response = self.client.get(f'/api/v1/jobs/{queued}/result', headers=self.auth())
            self.assertEqual((response.status_code, response.get_json()['error']), (409, 'not_finished'))
        self.service().start()
        self.wait(queued)

    def test_model_unavailable_is_503_and_frees_the_idempotency_key(self):
        store = self.service().store
        good = store.settings()
        store.save_settings({**good, 'ocr_url': 'http://127.0.0.1:9'})
        try:
            response = self.submit(headers={'Idempotency-Key': 'retry-me'})
            self.assertEqual((response.status_code, response.get_json()['error']), (503, 'model_unavailable'))
        finally:
            store.save_settings(good)
        retried = self.submit(headers={'Idempotency-Key': 'retry-me'})
        self.assertEqual(retried.status_code, 202)
        self.wait(retried.get_json()['job_id'])

    def test_health_needs_no_key(self):
        response = self.client.get('/api/v1/health')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual((body['status'], body['models']), ('ok', {'ocr': 'online', 'extraction': 'online'}))
        self.assertIn('open_jobs', body['queue'])

    def test_user_retry_bumps_result_version(self):
        job_id = self.submit().get_json()['job_id']
        self.wait(job_id)
        first = self.client.get(f'/api/v1/jobs/{job_id}/result', headers=self.auth()).get_json()
        store = self.service().store
        parent = store.get(job_id)
        cfg = {**parent['configuration']}
        store.retry_extraction(job_id, 'retry_test_0001', cfg)
        self.service().start()
        status = self.wait(job_id)
        deadline = time.time() + 120
        while status['result_version'] < 2 and time.time() < deadline:
            time.sleep(0.2)
            status = self.client.get('/api/v1/jobs/' + job_id, headers=self.auth()).get_json()
        self.assertEqual(status['result_version'], 2)
        second = self.client.get(f'/api/v1/jobs/{job_id}/result', headers=self.auth()).get_json()
        self.assertEqual(second['result_version'], 2)
        self.assertNotEqual(second['event_id'], first['event_id'])
        # The retry attempt itself is not a v1 job id.
        retry_id = next(row['id'] for row in store.retry_history(parent) if row['id'] != job_id)
        self.assertEqual(self.client.get('/api/v1/jobs/' + retry_id, headers=self.auth()).status_code, 404)


if __name__ == '__main__':
    unittest.main()
