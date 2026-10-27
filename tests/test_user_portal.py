"""User queue, real engine adapter and HTTP contracts; no model inference."""
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app
import userpipeline
import workspace
from userpipeline import EngineAdapter, UserService
from userstore import UserStore


def until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for user queue')


class FakeAdapter:
    def models(self, cfg, force=True):
        return {'ocr': {'model': 'reader', 'url': cfg['ocr_url'], 'available': True},
                'extraction': {'model': 'extractor', 'url': cfg['extraction_url'], 'text_available': True}}

    def snapshot(self, cfg):
        return {**cfg, 'models': self.models(cfg)}

    def ocr(self, job, store):
        return {'text': 'Page one', 'pages': ['Page one'], 'warnings': []}

    def extraction(self, job, store):
        return {**job['result'], 'extracted': {'fields': {'buyer_tax_id': '1234567890123'}}}


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, 32)


class StoreTests(StoreCase):
    def test_settings_persist_and_limits_can_be_raised_back_to_deployment_ceiling(self):
        self.assertEqual(self.store.settings()['ocr_concurrency'], 50)
        self.assertEqual(self.store.settings()['extraction_concurrency'], 64)
        self.assertEqual(self.store.settings()['user_file_max'], 0)
        self.store.save_settings({'max_upload_mb': 2, 'shape': 'agentic'})
        reopened = UserStore(self.temp.name, 32)
        self.assertEqual(reopened.settings()['shape'], 'agentic')
        self.assertEqual(reopened.save_settings({'max_upload_mb': 32})['max_upload_mb'], 32)
        for value in ({'max_upload_mb': 33}, {'ocr_concurrency': 0}, {'extraction_concurrency': True},
                      {'shape': 'unknown'}, {'extras': 'false'}, {'queue': 10},
                      {'ocr_url': 'file:///tmp/file'}, {'extraction_url': 'http://x:bad'}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.save_settings(value)

    def test_batch_limit_snapshot_idempotency_and_disk_original(self):
        cfg = self.store.save_settings({'user_file_max': 2, 'shape': 'hybrid'})
        with self.assertRaises(ValueError):
            self.store.create_batch(3, cfg)
        batch = self.store.create_batch(2, cfg)
        self.store.save_settings({'shape': 'single'})
        first = self.store.add_job(batch, 'upload_01', 'one.pdf', b'one')
        again = self.store.add_job(batch, 'upload_01', 'one.pdf', b'one')
        self.assertEqual(first['id'], again['id'])
        self.assertEqual(first['configuration']['shape'], 'hybrid')
        self.assertEqual((self.store.uploads / first['id']).read_bytes(), b'one')
        self.store.add_job(batch, 'upload_02', 'two.pdf', b'two')
        with self.assertRaises(ValueError):
            self.store.add_job(batch, 'upload_03', 'three.pdf', b'three')
        self.assertEqual(len(list(self.store.uploads.iterdir())), 2)

    def test_recovery_keeps_queues_and_ocr_but_never_retries_running_stages(self):
        batch = self.store.create_batch(3, self.store.settings())
        jobs = [self.store.add_job(batch, 'upload_0' + str(n), f'{n}.pdf', b'x') for n in range(3)]
        self.store.update(jobs[0]['id'], status='ocr', result={'text': 'saved partial'})
        self.store.update(jobs[1]['id'], status='queued_extraction', result={'text': 'done OCR'})
        self.store.recover()
        self.assertEqual(self.store.get(jobs[0]['id'])['status'], 'interrupted')
        self.assertEqual(self.store.get(jobs[0]['id'])['result']['text'], 'saved partial')
        self.assertEqual(self.store.get(jobs[1]['id'])['status'], 'queued_extraction')
        self.assertEqual(self.store.get(jobs[2]['id'])['status'], 'queued_ocr')


class SchedulerTests(StoreCase):
    def service(self, adapter):
        value = UserService(self.store, adapter)
        self.addCleanup(value.close)
        return value

    def add(self, count):
        batch = self.store.create_batch(count, self.store.settings())
        return [self.store.add_job(batch, 'upload_' + str(n).zfill(3), f'{n}.pdf', b'x') for n in range(count)]

    def test_stage_limits_are_global_and_ocr_overlaps_extraction(self):
        self.store.save_settings({'ocr_concurrency': 2, 'extraction_concurrency': 1})
        jobs = self.add(4)
        release = threading.Event()
        self.addCleanup(release.set)
        adapter = FakeAdapter()
        mutex = threading.Lock()
        counts, peaks = {'ocr': 0, 'extraction': 0}, {'ocr': 0, 'extraction': 0}
        reached = threading.Event()
        normal_ocr, normal_extraction = adapter.ocr, adapter.extraction

        def run(stage, fn, job, store):
            with mutex:
                counts[stage] += 1
                peaks[stage] = max(peaks[stage], counts[stage])
                if counts['ocr'] and counts['extraction']:
                    reached.set()
            if stage == 'ocr':
                time.sleep(.04)
            else:
                release.wait(4)
            result = fn(job, store)
            with mutex:
                counts[stage] -= 1
            return result

        adapter.ocr = lambda j, s: run('ocr', normal_ocr, j, s)
        adapter.extraction = lambda j, s: run('extraction', normal_extraction, j, s)
        service = self.service(adapter)
        service.start()
        self.assertTrue(reached.wait(4), 'OCR and extraction should overlap')
        until(lambda: sum(self.store.get(j['id'])['status'] == 'queued_extraction' for j in jobs) >= 2)
        self.assertEqual(peaks, {'ocr': 2, 'extraction': 1})
        release.set(); service.wake.set()
        until(lambda: all(self.store.get(j['id'])['status'] == 'completed' for j in jobs))
        self.assertEqual(peaks, {'ocr': 2, 'extraction': 1})

    def test_saved_concurrency_changes_dispatch_without_cancelling_active_work(self):
        self.store.save_settings({'ocr_concurrency': 1})
        jobs = self.add(3)
        release = {j['id']: threading.Event() for j in jobs}
        self.addCleanup(lambda: [event.set() for event in release.values()])
        adapter = FakeAdapter()
        normal = adapter.ocr
        def slow(job, store):
            release[job['id']].wait(5)
            return normal(job, store)
        adapter.ocr = slow
        service = self.service(adapter); service.start()
        until(lambda: self.store.get(jobs[0]['id'])['status'] == 'ocr')
        self.assertEqual(self.store.get(jobs[1]['id'])['status'], 'queued_ocr')
        self.store.save_settings({'ocr_concurrency': 2}); service.wake.set()
        until(lambda: self.store.get(jobs[1]['id'])['status'] == 'ocr')
        self.store.save_settings({'ocr_concurrency': 1}); service.wake.set()
        release[jobs[0]['id']].set()
        until(lambda: self.store.get(jobs[0]['id'])['status'] == 'completed')
        self.assertEqual(self.store.get(jobs[2]['id'])['status'], 'queued_ocr')
        for event in release.values(): event.set()
        service.wake.set()
        until(lambda: all(self.store.get(j['id'])['status'] == 'completed' for j in jobs))

    def test_extraction_failure_keeps_ocr_and_a_restart_resumes_only_waiting_jobs(self):
        jobs = self.add(2)
        self.store.update(jobs[0]['id'], status='queued_extraction',
                          result={'text': 'already read', 'pages': ['already read'], 'warnings': []})
        self.store.update(jobs[1]['id'], status='extraction', result={'text': 'interrupted text'})
        adapter = FakeAdapter()
        def fail(job, store): raise ValueError('model unavailable')
        adapter.extraction = fail
        service = self.service(adapter); service.start()
        until(lambda: self.store.get(jobs[0]['id'])['status'] == 'failed')
        self.assertEqual(self.store.get(jobs[0]['id'])['result']['text'], 'already read')
        self.assertEqual(self.store.get(jobs[1]['id'])['status'], 'interrupted')


class UserWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, app.MAX_UPLOAD_MB)
        self.service = UserService(self.store, FakeAdapter()); self.addCleanup(self.service.close)
        previous = app.app.extensions.get('user_service')
        app.app.extensions['user_service'] = self.service
        def restore():
            if previous is None: app.app.extensions.pop('user_service', None)
            else: app.app.extensions['user_service'] = previous
        self.addCleanup(restore)
        self.client = app.app.test_client()

    def batch(self, count=1, prefix=''):
        response = self.client.post(prefix + '/api/user/batches', json={'file_count': count})
        self.assertEqual(response.status_code, 201, response.data)
        return response.json['batch_id']

    def upload(self, batch, name='one.pdf', content=b'file', upload_id='upload_01', prefix=''):
        response = self.client.post(prefix + '/api/user/jobs', data={
            'batch_id': batch, 'upload_id': upload_id, 'file': (io.BytesIO(content), name)})
        response.request.close()
        response.request.environ['wsgi.input'].close()
        return response

    def test_shared_history_pipeline_and_download_survive_a_new_browser(self):
        response = self.upload(self.batch())
        self.assertEqual(response.status_code, 202)
        job_id = response.json['job']['id']
        until(lambda: self.store.get(job_id)['status'] == 'completed')
        other = app.app.test_client()
        self.assertEqual(other.get('/api/user/jobs').json['jobs'][0]['id'], job_id)
        detail = other.get('/api/user/jobs/' + job_id).json['job']
        self.assertEqual(detail['result']['text'], 'Page one')
        self.assertEqual(detail['configuration']['models']['ocr']['model'], 'reader')
        original = other.get('/api/user/jobs/' + job_id + '/file')
        self.assertEqual(original.data, b'file'); original.close()
        exported = other.get('/api/user/jobs/' + job_id + '/export')
        self.assertEqual(json.loads(exported.data)['id'], job_id); exported.close()

    def test_settings_endpoints_validate_without_changing_benchmark_settings(self):
        old = (app.extract_mode(), app.extract_extras(), app.backends.active_url())
        response = self.client.put('/api/user/settings', json={'shape': 'hybrid', 'ocr_concurrency': 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/user/settings').json['settings']['ocr_concurrency'], 3)
        self.assertEqual(old, (app.extract_mode(), app.extract_extras(), app.backends.active_url()))
        self.assertEqual(self.client.put('/api/user/settings', json={'queue': 4}).status_code, 400)
        self.assertEqual(self.client.post('/api/user/models', json=self.store.settings()).status_code, 200)
        self.assertEqual(self.client.put('/api/user/settings', json=[], headers={'Origin': 'http://elsewhere'}).status_code, 403)

    def test_distinct_user_sessions_still_share_jobs_settings_and_history(self):
        first = {'X-OCR-Workspace': 'user-first-tab'}
        second = {'X-OCR-Workspace': 'user-second-tab'}
        self.addCleanup(lambda: workspace.enter(None))
        batch = self.client.post('/ocr/api/user/batches', json={'file_count': 1}, headers=first).json['batch_id']
        response = self.client.post('/ocr/api/user/jobs', headers=first, data={
            'batch_id': batch, 'upload_id': 'upload_01', 'file': (io.BytesIO(b'file'), 'one.pdf')})
        self.assertEqual(response.status_code, 202)
        job_id = response.json['job']['id']
        until(lambda: self.store.get(job_id)['status'] == 'completed')
        other = app.app.test_client()
        history = other.get('/ocr/api/user/jobs', headers=second)
        self.assertEqual(workspace.current_id(), second['X-OCR-Workspace'])
        self.assertEqual(history.json['jobs'][0]['id'], job_id)
        detail = other.get('/ocr/api/user/jobs/' + job_id, headers=second).json['job']
        stored = json.dumps(detail)
        self.assertNotIn(first['X-OCR-Workspace'], stored)
        self.assertNotIn(second['X-OCR-Workspace'], stored)
        a = self.client.get('/ocr/api/user/settings', headers=first).json
        b = other.get('/ocr/api/user/settings', headers=second).json
        self.assertEqual(a, b)
        page = other.get('/ocr/user?ws=user-second-tab')
        self.assertEqual(page.status_code, 200)
        self.assertIn('var isUser = true;', page.text)
        self.assertEqual(workspace.current_id(), 'user-second-tab')

    def test_file_validation_and_each_upload_uses_its_own_size_limit(self):
        self.store.save_settings({'max_upload_mb': 1, 'user_file_max': 2})
        self.assertEqual(self.client.post('/api/user/batches', json={'file_count': 3}).status_code, 400)
        batch = self.batch(2)
        self.assertEqual(self.upload(batch, content=b'').status_code, 400)
        self.assertEqual(self.upload(batch, name='evil.exe').status_code, 400)
        self.assertEqual(self.upload(batch, content=b'x' * (1024*1024+1)).status_code, 413)
        for index in range(2):
            self.assertEqual(self.upload(batch, content=b'x' * 700000, upload_id=f'upload_0{index}').status_code, 202)
        self.assertEqual(self.upload(batch, upload_id='upload_03').status_code, 400)
        self.assertEqual(self.client.get('/api/user/jobs/not-an-id').status_code, 404)

    def test_portal_embeds_assets_and_api_work_under_a_mount_with_or_without_stripping(self):
        for prefix in ('', '/ocr', '/tools/ocr'):
            page = self.client.get(prefix + '/user')
            self.assertEqual(page.status_code, 200)
            self.assertIn('DOCUMENT WORKSPACE', page.text)
            self.assertNotIn('id="sideTabs"', page.text)
            self.assertIn('.workspace{display:grid', page.text)
            self.assertIn('async function loadSettings()', page.text)
            self.assertNotIn('/static/user/', page.text)
            self.assertNotIn('Whale', page.text)
            self.assertIn('<title>OCR · เอกสารของทีม</title>', page.text)
            self.assertEqual(self.client.get(prefix + '/api/user/settings').status_code, 200)
        response = self.client.get('/user', headers={'X-Forwarded-Prefix': '/ocr'})
        self.assertEqual(response.status_code, 200)
        root = self.client.get('/').text
        self.assertIn('data-pane="settings"', root)
        self.assertIn('id="settingsCard"', root)
        self.assertIn('window.loadUserSettings = async', root)
        self.assertNotIn('/static/user/', root)
        self.assertEqual(self.client.post('/ocr/api/user/batches', json={'file_count': 1}).status_code, 201)


class EngineAdapterTests(StoreCase):
    def test_snapshot_follows_the_ocr_model_profile_without_changing_benchmark(self):
        adapter = EngineAdapter(app)
        models = FakeAdapter().models(self.store.settings())
        models['ocr']['model'] = 'dots.ocr'
        old = app.ocr_profile()
        with patch.object(adapter, 'models', return_value=models):
            cfg = adapter.snapshot(self.store.settings())
        self.assertEqual(cfg['ocr_profile'], 'dots')
        with adapter.context('profile_test', cfg):
            self.assertEqual(app.ocr_profile(), 'dots')
        self.assertEqual(app.ocr_profile(), old)

    def test_real_image_preparation_keeps_benchmark_cache_untouched_and_preserves_page_order(self):
        picture = Image.new('RGB', (20, 20), 'white'); content = io.BytesIO()
        picture.save(content, format='TIFF', save_all=True, append_images=[picture])
        cfg = self.store.settings()
        cfg['models'] = {'ocr': {'model': 'reader'}, 'extraction': {'model': 'extractor'}}
        batch = self.store.create_batch(1, cfg)
        job = self.store.add_job(batch, 'upload_01', 'two.tiff', content.getvalue())
        info = {'available': True, 'text_available': True, 'model': 'reader', 'models': [{'name': 'reader'}]}
        before = set(app._jobs)
        adapter = EngineAdapter(app)
        with patch.object(app.backends, 'status', return_value=info), \
             patch.object(app, '_page_parallelism', return_value=1), \
             patch.object(app, 'read_page', side_effect=['first page', 'second page']):
            result = adapter.ocr(job, self.store)
        self.assertEqual(result['pages'], ['first page', 'second page'])
        self.assertEqual(set(app._jobs), before)
        self.assertEqual(result['pages_total'], 2)
        self.assertIn('--- page 1 ---', result['text'])

    def test_extraction_options_progress_and_compact_result_use_the_current_engine(self):
        cfg = self.store.settings(); batch = self.store.create_batch(1, cfg)
        job = self.store.add_job(batch, 'upload_01', 'one.pdf', b'file')
        self.store.update(job['id'], result={'text': 'page text', 'pages': ['page text'], 'warnings': []})
        job = self.store.get(job['id'])
        def stream(*args, **kwargs):
            yield {'event': 'segmented', 'documents': 1}
            yield {'event': 'extract_steps', 'total': 4}
            yield {'event': 'extract_step', 'status': 'done'}
            return {'fields': {'buyer_tax_id': '1234'}, 'mode': 'hybrid',
                    'raw': 'large reply', 'steps': [{'prompt': 'large prompt'}], 'field_score': {'fake': 'truth'}}
        with patch.object(app.backends, 'extract_status', return_value={'text_available': True}), \
             patch.object(app, 'extract_fields_stream', side_effect=stream) as extract:
            result = EngineAdapter(app).extraction(job, self.store)
        self.assertEqual(extract.call_args.args[0], 'page text')
        self.assertEqual(extract.call_args.kwargs, dict(pages=['page text'], mode='hybrid', extras=False, schema_first=False))
        self.assertEqual(result['extracted'], {'fields': {'buyer_tax_id': '1234'}, 'mode': 'hybrid'})
        self.assertIn('1/4 steps', self.store.get(job['id'])['progress'])

    def test_workspace_cache_and_batcher_are_isolated_from_benchmark(self):
        previous = (workspace.current_id(), app.loop_guard(), app.backends.active_url())
        cfg = {**self.store.settings(), 'loop_guard': False, 'no_cache': True}
        with EngineAdapter(app).context('isolation', cfg), patch.object(app, '_batcher', object()):
            self.assertFalse(app.loop_guard())
            self.assertTrue(app.backends.prefix_cache_bypassed())
            self.assertIsNone(app._batcher_for('http://x/v1/chat/completions', {'kind': 'vllm'}))
        self.assertEqual(previous, (workspace.current_id(), app.loop_guard(), app.backends.active_url()))
        self.assertFalse(app.backends.prefix_cache_bypassed())

    def test_failed_hybrid_step_marks_partial_and_table_diagnostics_are_not_saved(self):
        batch = self.store.create_batch(1, self.store.settings())
        job = self.store.add_job(batch, 'upload_01', 'one.pdf', b'file')
        job['result'] = {'text': 'page text', 'pages': ['page text'], 'warnings': []}
        def stream(*args, **kwargs):
            yield {'event': 'extract_steps', 'total': 4}
            return {'fields': {'subtotal': '10.00'}, 'steps_failed': [{'id': 'parties', 'error': 'timeout'}],
                    'item_table': {'columns': ['Amount'], 'rows': [['10.00']],
                                   'fix_agent': {'prompt': 'large prompt'}, 'stages': [{'raw': 'large raw'}]}}
        with patch.object(app.backends, 'extract_status', return_value={'text_available': True}), \
             patch.object(app, 'extract_fields_stream', side_effect=stream):
            result = EngineAdapter(app).extraction(job, self.store)
        self.assertTrue(result['warnings'])
        self.assertEqual(result['extracted']['steps_failed'][0]['id'], 'parties')
        self.assertEqual(result['extracted']['item_table'], {'columns': ['Amount'], 'rows': [['10.00']]})

    def test_replaced_model_is_not_silently_substituted_for_the_snapshot(self):
        job = {'configuration': {'models': {'extraction': {'model': 'old-model'}}}}
        with patch.object(app.backends, 'extract_status', return_value={
                'text_available': True, 'model': 'old-model', 'models': [{'name': 'new-model'}]}), \
             self.assertRaisesRegex(ValueError, 'โมเดลเปลี่ยน'):
            EngineAdapter(app).check_model(job, 'extraction')


if __name__ == '__main__':
    unittest.main()
