"""Extraction retries reuse OCR, preserve source history and snapshot local edits."""
import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import app
import userprompts
from userpipeline import EngineAdapter, UserService
from userstore import UserStore
from tests.test_user_portal import FakeAdapter, until


class CountingAdapter(FakeAdapter):
    def __init__(self):
        self.reads, self.extracts = 0, 0

    def snapshot(self, cfg):
        raise AssertionError('Retry must not probe OCR')

    def ocr(self, job, store):
        self.reads += 1
        raise AssertionError('Retry must not run OCR')

    def extraction(self, job, store):
        self.extracts += 1
        return {**job['result'], 'extracted': {'doc_types': ['INVOICE'],
                'fields': {'document_number': 'NEW', 'subtotal': '100'}}}


class UserExtractionRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, 32)
        self.adapter = CountingAdapter()
        self.service = UserService(self.store, self.adapter); self.addCleanup(self.service.close)
        old = app.app.extensions.get('user_service')
        app.app.extensions['user_service'] = self.service
        self.addCleanup(lambda: app.app.extensions.pop('user_service', None) if old is None
                        else app.app.extensions.update(user_service=old))
        self.client = app.app.test_client()
        self.store.save_prompt_profile('INVOICE', {'instructions': {'document_number': 'old global'}, 'extra_groups': []}, 0)
        cfg = {**self.store.settings(), 'fill_missing': False,
               'models': FakeAdapter().models(self.store.settings()), 'prompt_profiles': self.store.prompt_profiles()}
        batch = self.store.create_batch(1, cfg)
        with app.fitz.open() as pdf:
            pdf.new_page().insert_text((50, 50), 'Cached OCR source')
            content = pdf.tobytes()
        self.parent = self.store.add_job(batch, 'upload_01', 'source.pdf', content)
        self.store.update(self.parent['id'], status='completed', result={
            'text': 'INVOICE\nCached OCR', 'pages': ['INVOICE\nCached OCR'], 'page_count': 1, 'pages_total': 1,
            'warnings': [], 'ocr_warnings': [], 'timing': {'ocr': 12, 'extraction': 5},
            'extracted': {'doc_types': ['INVOICE'], 'fields': {'document_number': 'OLD'}}})
        self.parent = self.store.get(self.parent['id'])

    def retry(self, parent=None, request_id='retry_00000001', edits=None, prefix=''):
        return self.client.post(prefix + '/api/user/jobs/' + (parent or self.parent['id']) + '/retry-extraction',
            json={'request_id': request_id, 'prompt_profiles': edits or {}})

    def test_retry_runs_only_extraction_preserves_original_and_supports_retry_chains(self):
        before = copy.deepcopy(self.parent)
        edits = {'INVOICE': {'instructions': {'document_number': 'read new location'}, 'extra_groups': []}}
        response = self.retry(edits=edits, prefix='/ocr')
        self.assertEqual(response.status_code, 202, response.data)
        child_id = response.json['job']['id']
        until(lambda: self.store.get(child_id)['status'] == 'completed')
        child = self.store.get(child_id)
        self.assertEqual(self.adapter.reads, 0); self.assertEqual(self.adapter.extracts, 1)
        self.assertEqual(self.store.get(self.parent['id']), before)
        self.assertEqual(child['result']['text'], before['result']['text'])
        self.assertEqual(child['result']['pages'], before['result']['pages'])
        self.assertNotIn('ocr', child['timeline']); self.assertNotIn('ocr', child['result']['timing'])
        self.assertEqual(child['configuration']['prompt_profiles']['INVOICE']['value'], edits['INVOICE'])
        self.assertEqual(child['configuration']['prompt_profiles']['INVOICE']['scope'], 'job')
        self.assertEqual(self.store.prompt_profiles()['INVOICE']['value']['instructions']['document_number'], 'old global')
        displayed = self.client.get('/ocr/api/user/jobs/' + child_id + '/prompts').json
        self.assertEqual(displayed['editor']['value'], edits['INVOICE'])
        self.assertEqual(displayed['available_types'], ['INVOICE'])
        self.assertEqual(self.retry(edits=edits).json['job']['id'], child_id)
        self.assertEqual(self.adapter.extracts, 1)
        next_id = self.retry(child_id, request_id='retry_00000002').json['job']['id']
        until(lambda: self.store.get(next_id)['status'] == 'completed')
        descendant = self.store.get(next_id)
        self.assertEqual(descendant['retry_of'], child_id)
        self.assertEqual(self.store.original(descendant), self.store.original(before))
        self.assertEqual(self.adapter.extracts, 2); self.assertEqual(self.adapter.reads, 0)
        for suffix in ('file', 'preview', 'preview/info', 'preview/pages/1'):
            reply = self.client.get('/ocr/api/user/jobs/' + next_id + '/' + suffix)
            self.assertEqual(reply.status_code, 200, suffix); reply.close()

    def test_actual_engine_adapter_probes_only_extraction_and_receives_cached_text(self):
        self.service.adapter = EngineAdapter(app)
        calls = []
        status = {'model': 'extractor', 'text_available': True, 'available': True,
                  'kind': 'openai', 'url': 'http://127.0.0.1:8001', 'models': [{'name': 'extractor'}]}
        def extract(text, **kwargs):
            calls.append((text, kwargs))
            yield {'event': 'classified', 'doc_types': ['INVOICE']}
            return {'doc_types': ['INVOICE'], 'fields': {'document_number': 'NEW'}}
        with patch.object(app.backends, 'status', side_effect=AssertionError('No OCR probe')), \
             patch.object(app.backends, 'extract_status', return_value=status), \
             patch.object(app, 'prepare_input', side_effect=AssertionError('No OCR or image re-read')), \
             patch.object(app, 'extract_fields_stream', side_effect=extract), \
             patch.object(app, 'image_rechecks_needed', return_value=False):
            response = self.retry()
            self.assertEqual(response.status_code, 202, response.data)
            child_id = response.json['job']['id']
            until(lambda: self.store.get(child_id)['status'] == 'completed')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.parent['result']['text'])
        self.assertEqual(calls[0][1]['pages'], self.parent['result']['pages'])
        self.assertEqual(calls[0][1]['mode'], 'hybrid')

    def test_concurrent_duplicate_requests_enqueue_one_attempt(self):
        def submit(_):
            client = app.app.test_client()
            return client.post('/api/user/jobs/' + self.parent['id'] + '/retry-extraction',
                               json={'request_id': 'retry_same_0001', 'prompt_profiles': {}})
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(submit, range(2)))
        self.assertEqual([r.status_code for r in responses], [202, 202])
        ids = {r.json['job']['id'] for r in responses}; self.assertEqual(len(ids), 1)
        until(lambda: self.store.get(next(iter(ids)))['status'] == 'completed')
        self.assertEqual(self.adapter.extracts, 1)

    def test_missing_ocr_active_jobs_unknown_types_and_invalid_fields_are_rejected(self):
        invalid = self.retry(edits={'INVOICE': {'instructions': {'bad_key': 'x'}}})
        self.assertEqual(invalid.status_code, 400)
        invalid = self.retry(edits={'RECEIPT': {'instructions': {}, 'extra_groups': []}})
        self.assertEqual(invalid.status_code, 400)
        self.store.update(self.parent['id'], status='ocr')
        self.assertEqual(self.retry().status_code, 409)
        self.store.update(self.parent['id'], status='failed', result={'text': '', 'pages': []})
        self.assertEqual(self.retry().status_code, 400)
        self.assertEqual(self.adapter.extracts, 0)

    def test_origin_check_guards_retry_and_ocr_warnings_survive_without_old_extraction_warnings(self):
        url = '/api/user/jobs/' + self.parent['id'] + '/retry-extraction'
        self.assertEqual(self.client.post(url, json={}, headers={'Origin': 'http://elsewhere'}).status_code, 403)
        previous = copy.deepcopy(self.parent['result'])
        previous.pop('ocr_warnings')
        previous['warnings'] = ['ข้อความหน้า 1 อาจอ่านได้ไม่ครบ', 'ดึงข้อมูลบางส่วนได้ไม่ครบ กรุณาตรวจสอบกับต้นฉบับ']
        self.store.update(self.parent['id'], status='partial', result=previous)
        response = self.retry(); child_id = response.json['job']['id']
        until(lambda: self.store.get(child_id)['status'] == 'partial')
        self.assertEqual(self.store.get(child_id)['result']['warnings'], ['ข้อความหน้า 1 อาจอ่านได้ไม่ครบ'])

    def test_audit_keeps_exact_changes_additions_targets_and_each_attempts_parent(self):
        first_value = {'instructions': {'document_number': 'new location'}, 'extra_groups': [
            {'id': 'extra_party', 'title': 'Party name', 'fields': [
                {'key': 'party_name', 'instruction': 'Read supplier name', 'target': 'seller.name'}]}]}
        response = self.retry(edits={'INVOICE': first_value})
        first = response.json['job']; until(lambda: self.store.get(first['id'])['status'] == 'completed')
        audit = first['prompt_changes']
        self.assertEqual(audit['scope'], 'job'); self.assertEqual(audit['parent_job_id'], self.parent['id'])
        self.assertEqual(audit['created_at'], first['created_at'])
        note = next(c for c in audit['changes'] if c['entity'] == 'instruction')
        self.assertEqual((note['before'], note['after']), ('old global', 'new location'))
        added = next(c for c in audit['changes'] if c['entity'] == 'group')
        self.assertEqual(added['action'], 'added'); self.assertIsNone(added['before'])
        self.assertEqual(added['after'], first_value['extra_groups'][0])
        second_value = copy.deepcopy(first_value)
        second_value['extra_groups'][0]['fields'][0]['target'] = 'buyer.name'
        second_value['extra_groups'][0]['fields'].append(
            {'key': 'bill_discount', 'instruction': 'Read printed discount', 'target': 'discount'})
        second = self.retry(first['id'], request_id='retry_audit_02', edits={'INVOICE': second_value}).json['job']
        until(lambda: self.store.get(second['id'])['status'] == 'completed')
        changes = second['prompt_changes']['changes']
        self.assertEqual({c['entity'] for c in changes}, {'destination', 'field'})
        destination = next(c for c in changes if c['entity'] == 'destination')
        self.assertEqual((destination['before'], destination['after']), ('seller.name', 'buyer.name'))
        new_field = next(c for c in changes if c['entity'] == 'field')
        self.assertEqual(new_field['action'], 'added'); self.assertEqual(new_field['after']['key'], 'bill_discount')
        unchanged = self.retry(second['id'], request_id='retry_audit_03').json['job']
        self.assertEqual(unchanged['prompt_changes']['changes'], [])
        until(lambda: self.store.get(unchanged['id'])['status'] == 'completed')
        # A separate branch is compared with its selected source, not the last retry.
        branch = self.retry(request_id='retry_branch_01', edits={'INVOICE': {
            'instructions': {'document_number': 'branch location'}, 'extra_groups': []}}).json['job']
        until(lambda: self.store.get(branch['id'])['status'] == 'completed')
        self.assertEqual(branch['prompt_changes']['changes'][0]['before'], 'old global')
        history = self.client.get('/ocr/api/user/jobs/' + second['id'] + '/retry-history').json['attempts']
        self.assertEqual([row['id'] for row in history], [self.parent['id'], first['id'], second['id'], unchanged['id'], branch['id']])
        self.assertEqual(UserStore(self.temp.name, 32).get(first['id'])['prompt_changes'], audit)
        exported = self.client.get('/api/user/jobs/' + first['id'] + '/export')
        self.assertEqual(exported.json['prompt_changes'], audit); exported.close()
        self.assertEqual(self.store.prompt_profiles()['INVOICE']['value']['instructions']['document_number'], 'old global')

    def test_audit_records_removals_resets_and_reordering_without_inventing_default_edits(self):
        old = {'instructions': {'document_number': 'Custom guide'}, 'extra_groups': [
            {'id': 'extra_a', 'title': 'A', 'fields': [{'key': 'a', 'instruction': 'Read A', 'target': 'seller.name'}]},
            {'id': 'extra_b', 'title': 'B', 'fields': [{'key': 'b', 'instruction': 'Read B', 'target': 'buyer.name'}]}]}
        before = {'INVOICE': {'revision': 1, 'value': old}}
        new = copy.deepcopy(old); new['extra_groups'].reverse()
        reordered = userprompts.changes_between(before, {'INVOICE': {'revision': 2, 'value': new}})
        self.assertEqual([c['entity'] for c in reordered], ['group_order'])
        reset = userprompts.changes_between(before, {'INVOICE': {'value': {'instructions': {}, 'extra_groups': []}}})
        self.assertEqual(len(reset), 3); self.assertTrue(all(c['action'] == 'removed' for c in reset))
        self.assertTrue(all(c['after'] is None for c in reset))
        self.assertEqual(userprompts.changes_between({}, {'INVOICE': {'revision': 1, 'value': {'instructions': {}, 'extra_groups': []}}}), [])


if __name__ == '__main__':
    unittest.main()
