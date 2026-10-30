"""Editable User prompts persist, affect real requests and map by job snapshot."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import app
import prompts
import userjson
import userprompts
import workspace
from userpipeline import EngineAdapter, UserService, compact_extraction
from userstore import UserStore
from tests.test_user_portal import FakeAdapter
from tests.test_userjson import job


def value():
    return {'instructions': {'document_number': 'อ่านจากหัวข้อ Invoice No. เท่านั้น'}, 'extra_groups': [
        {'id': 'extra_supplier', 'title': 'ผู้ขายและส่วนลด', 'fields': [
            {'key': 'supplier_name', 'instruction': 'อ่านชื่อผู้ขายจากหัวเอกสาร', 'target': 'seller.name'},
            {'key': 'bill_discount', 'instruction': 'อ่านส่วนลดท้ายบิล', 'target': 'discount'},
            {'key': 'notes', 'instruction': 'อ่านหมายเหตุเป็นรายการข้อความ', 'target': 'remarks'}]}]}


class UserPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, 32)
        self.service = UserService(self.store, FakeAdapter()); self.addCleanup(self.service.close)
        old = app.app.extensions.get('user_service')
        app.app.extensions['user_service'] = self.service
        self.addCleanup(lambda: app.app.extensions.pop('user_service', None) if old is None
                        else app.app.extensions.update(user_service=old))
        self.client = app.app.test_client()

    def save(self, data=None, revision=0, prefix=''):
        return self.client.put(prefix + '/api/user/hybrid-prompts', json={
            'doc_type': 'INVOICE', 'revision': revision, 'value': value() if data is None else data})

    def test_api_round_trip_is_prefix_safe_durable_and_returns_actual_edited_questions(self):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No inference')):
            self.assertEqual(self.save(prefix='/ocr').status_code, 200)
            data = self.client.get('/ocr/api/user/hybrid-prompts?doc_type=INVOICE').json
            self.assertEqual(data['editor']['revision'], 1)
            self.assertEqual(data['editor']['value'], value())
            self.assertEqual(len(data['groups']), 5)
            self.assertIn('Invoice No. เท่านั้น', data['groups'][0]['question'])
            self.assertIn('supplier_name', data['groups'][-1]['question'])
            self.assertEqual(UserStore(self.temp.name, 32).prompt_profiles()['INVOICE']['value'], value())
            self.assertEqual(self.save().status_code, 409)
            self.assertEqual(self.save(revision=1).json['profile']['revision'], 2)

    def test_origin_protection_applies_before_editing(self):
        response = self.client.put('/api/user/hybrid-prompts', json={}, headers={'Origin': 'http://elsewhere'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.store.prompt_profiles())

    def test_only_field_guidance_and_supported_unique_destinations_are_writable(self):
        mutations = [
            lambda d: d.update(system_prompt='override'),
            lambda d: d['instructions'].update(made_up='bad'),
            lambda d: d['instructions'].update(issue_date=42),
            lambda d: d['instructions'].update(issue_date='a' * 4001),
            lambda d: d['extra_groups'][0]['fields'][0].update(key='document_number'),
            lambda d: d['extra_groups'][0]['fields'][0].update(key='__proto__'),
            lambda d: d['extra_groups'][0]['fields'][0].update(target='file.fileName'),
            lambda d: d['extra_groups'][0]['fields'][0].update(target='documentType.code'),
            lambda d: d['extra_groups'][0]['fields'][0].update(target='documentNo'),
            lambda d: d['extra_groups'][0]['fields'][1].update(target='seller.name'),
            lambda d: d['extra_groups'][0]['fields'][1].update(key='supplier_name'),
            lambda d: d['extra_groups'][0].update(fields=[]),
        ]
        for mutate in mutations:
            data = value(); mutate(data)
            with self.subTest(data=data):
                self.assertEqual(self.save(data).status_code, 400)
        self.assertFalse(self.store.prompt_profiles())

    def test_upload_batches_capture_their_own_prompt_versions(self):
        self.save()
        first = self.client.post('/api/user/batches', json={'file_count': 1}).json
        changed = value(); changed['extra_groups'][0]['fields'][0]['target'] = 'buyer.name'
        self.save(changed, revision=1)
        second = self.client.post('/api/user/batches', json={'file_count': 1}).json
        old = self.store.add_job(first['batch_id'], 'upload_01', 'old.pdf', b'pdf')
        new = self.store.add_job(second['batch_id'], 'upload_02', 'new.pdf', b'pdf')
        self.assertEqual(old['configuration']['prompt_profiles']['INVOICE']['revision'], 1)
        self.assertEqual(new['configuration']['prompt_profiles']['INVOICE']['revision'], 2)
        self.assertEqual(old['configuration']['prompt_profiles']['INVOICE']['value'], value())

    def test_runtime_requests_use_user_edits_and_keep_benchmark_and_agentic_unchanged(self):
        self.save()
        builtin = copy.deepcopy(prompts.steps_for_types(['INVOICE'], False, mode='hybrid'))
        cfg = {**self.store.settings(), 'prompt_profiles': self.store.prompt_profiles()}
        calls = []
        def ask(content, step, status, collect=None):
            calls.append((content, step))
            result = {key: '' for key in step['keys']}
            if step.get('custom_targets'):
                result.update(supplier_name='Supplier Ltd', bill_discount='1,234.50', notes=['Printed note'])
                # Exercise the actual JSON parser and locked schema builder.
                schema = app._step_schema(step)
                self.assertEqual(schema['properties']['notes']['type'], 'array')
                result = app._step_values(step, json.dumps(result))
            return result, [json.dumps(result)], False, 1, False
        form = {'doc_types': ['INVOICE'], 'keys': list(prompts.fields_for_types(['INVOICE'])),
                'mandatory': [], 'items': [], 'extras': False}
        with EngineAdapter(app).context('prompt_execution', cfg), patch.object(app, '_ask_step', ask), \
                patch.object(app, 'AGENTIC_RETRIES', 0):
            table = app.steps_for_types(['INVOICE'], False, mode='hybrid')
            self.assertEqual(len(table), 5)
            self.assertEqual(app.steps_for_types(['INVOICE'], False, mode='agentic'),
                             prompts.steps_for_types(['INVOICE'], False, mode='agentic'))
            extracted = app._drain(app._extract_agentic('Supplier Ltd\n1,234.50\nPrinted note',
                {'kind': 'ollama', 'model': 'fake', 'url': 'http://local'}, form, mode='hybrid'))
        self.assertIn('Invoice No. เท่านั้น', calls[0][0])
        self.assertEqual(extracted['custom_fields']['notes'], ['Printed note'])
        self.assertEqual(extracted['user_prompt_mappings']['supplier_name'], 'seller.name')
        self.assertEqual(extracted['user_prompt_revisions'], {'INVOICE': 1})
        self.assertEqual(extracted['fields']['other_fields'][0]['label'], 'supplier_name')
        self.assertEqual(app.steps_for_types(['INVOICE'], False, mode='hybrid'), builtin)
        self.assertEqual(prompts.steps_for_types(['INVOICE'], False, mode='hybrid'), builtin)
        compact = compact_extraction(extracted)
        compact['doc_types'] = ['INVOICE']
        source = job(compact); source['configuration'] = cfg
        exported = userjson.convert(source)
        self.assertEqual(exported['data']['documents'][0]['seller']['name'], 'Supplier Ltd')
        self.assertEqual(exported['data']['documents'][0]['discount'], 1234.5)
        self.assertEqual(exported['data']['documents'][0]['remarks'], ['Printed note'])
        self.assertTrue(any(row['source'] == 'custom_fields.supplier_name' for row in exported['mappings']))
        # Changing today's stored mapping cannot alter a completed job's export.
        updated = value(); updated['extra_groups'][0]['fields'][0]['target'] = 'buyer.name'
        self.save(updated, revision=1)
        self.assertEqual(userjson.convert(source)['data'], exported['data'])

    def test_invalid_custom_values_are_reported_and_never_defaulted_or_written_outside_schema(self):
        source = job()
        doc = source['result']['extracted']
        doc.update(custom_fields={'x': 'not a number', 'y': 'hacked', 'z': ['plain text']},
                   user_prompt_mappings={'x': 'discount', 'y': 'documentType.code', 'z': 'items'})
        report = userjson.convert(source)
        self.assertNotIn('discount', report['data']['documents'][0])
        self.assertEqual(report['data']['documents'][0]['documentType']['code'], '380')
        self.assertGreaterEqual(len(report['warnings']), 3)

    def test_reset_restores_builtin_questions_and_new_kinds_stay_uneditable(self):
        self.save()
        response = self.save({'instructions': {}, 'extra_groups': []}, revision=1)
        self.assertEqual(response.status_code, 200)
        data = self.client.get('/api/user/hybrid-prompts?doc_type=INVOICE').json
        self.assertEqual(len(data['groups']), 4)
        self.assertNotIn('User field reading guidance', data['groups'][0]['question'])
        response = self.client.put('/api/user/hybrid-prompts', json={
            'doc_type': 'INVOICE_TAX_INVOICE', 'revision': 0, 'value': value()})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
