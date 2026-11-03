"""Shared/detached User forms, immutable job versions and active-field execution."""
import copy
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app
import prompts
import userjson
import userpromptsets
from userpipeline import EngineAdapter, UserService, compact_extraction
from userstore import UserStore
from tests.test_user_portal import FakeAdapter
from tests.test_userjson import job


class UserPromptSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, 32)
        self.service = UserService(self.store, FakeAdapter()); self.addCleanup(self.service.close)
        old = app.app.extensions.get('user_service')
        app.app.extensions['user_service'] = self.service
        self.addCleanup(lambda: app.app.extensions.pop('user_service', None) if old is None else
                        app.app.extensions.update(user_service=old))
        self.client = app.app.test_client()

    def reference(self, code):
        return self.client.get('/api/user/hybrid-prompts', query_string={'doc_type':code}).get_json()

    def change(self, code='INVOICE', action='shared', source=None, value=None, **kwargs):
        state = self.store.prompt_set_snapshot()
        binding = state['bindings'].get(code, {'revision':0})
        record = state['sets'][source or binding['set_id']]
        body = {'doc_type':code, 'binding_revision':binding['revision'], 'set_id':record['id'],
                'revision':record['revision'], 'action':action, 'name':record['name'],
                'value':copy.deepcopy(record['value']) if value is None else value, **kwargs}
        return self.client.post('/ocr/api/user/prompt-sets', json=body), body

    def cfg(self):
        return {**self.store.settings(), 'prompt_sets':self.store.prompt_set_snapshot()}

    def test_catalog_includes_all_user_types_and_extras_without_claiming_untuned_is_configured(self):
        response = self.client.get('/api/user/prompt-sets')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len([r for r in response.json['types'] if r['section'] == 'user']), 11)
        self.assertEqual({r['code'] for r in response.json['types']}, set(prompts.TYPE_NAMES))
        for code in prompts.TYPE_NAMES:
            data = self.reference(code)
            self.assertIn('groups', data)
            self.assertEqual(data['editable_type'], code)
        data = self.reference('DEBIT_NOTE')
        self.assertIsNone(data['editor'])
        self.assertIsNone(data['library']['record'])
        self.assertTrue(any(g['id'].startswith('wht') for g in data['groups']))

    def test_migration_keeps_existing_edits_and_exact_constituent_sharing(self):
        directory = Path(self.temp.name) / 'legacy'; directory.mkdir()
        with sqlite3.connect(directory / 'history.sqlite3') as db:
            db.execute('CREATE TABLE prompt_profiles (code TEXT PRIMARY KEY,revision INTEGER,value TEXT)')
            db.execute('INSERT INTO prompt_profiles VALUES (?,?,?)', ('INVOICE',3,json.dumps({
                'instructions':{'document_number':'Legacy number rule'}, 'extra_groups':[]})))
        db.close()
        migrated = UserStore(directory,32)
        state = migrated.prompt_set_snapshot()
        first = state['bindings']['INVOICE']['set_id']
        self.assertEqual(state['bindings']['INVOICE_TAX_INVOICE']['set_id'], first)
        self.assertEqual(state['sets'][first]['revision'],3)
        self.assertIn('Legacy number rule',state['sets'][first]['steps'][0]['rules'])
        self.assertEqual(state['bindings']['DELIVERY_TAX_INVOICE']['set_id'],first)
        self.assertEqual(UserStore(directory,32).prompt_set_snapshot(), state)

    def test_user_doc_types_reference_follows_current_bindings_and_enabled_runtime_fields(self):
        source = self.reference('INVOICE')['library']['record']['id']
        value = {'disabled_fields':['vat_total'], 'extra_groups':[{
            'id':'extra_tax', 'title':'Tax', 'fields':[
                {'key':'custom_tax', 'instruction':'Read printed tax', 'target':'vatAmount'}]}]}
        response, _ = self.change('DELIVERY_TAX_INVOICE', 'detach', source=source,
                                   name='T04 test', value=value)
        self.assertEqual(response.status_code, 200, response.json)
        before = self.store.prompt_set_snapshot()
        response = self.client.get('/ocr/api/user/doc-types')
        self.assertEqual(response.status_code, 200)
        rows = {row['user_code']: row for row in response.json['types']}
        self.assertEqual(set(rows), {'80','81','380','388','T01','T02','T03','T04','T05','T06','T07'})
        t04 = rows['T04']
        fields = {field['key']:field for group in t04['groups'] for field in group['fields']}
        self.assertNotIn('vat_total', fields)
        self.assertEqual(fields['custom_tax']['target'], 'vatAmount')
        self.assertEqual(t04['field_count'], len(fields))
        self.assertEqual(t04['sets'][0]['name'], 'T04 test')
        self.assertEqual(t04['sets'][0]['shared_types'], ['T04'])
        self.assertNotEqual(t04['sets'][0]['id'], rows['380']['sets'][0]['id'])
        self.assertIn('T02', rows['380']['sets'][0]['shared_types'])
        self.assertEqual(rows['80']['sets'], [])
        self.assertEqual(self.store.prompt_set_snapshot(), before)

    def test_shared_save_reaches_all_bound_types_and_retains_old_versions_and_snapshots(self):
        before = self.cfg()
        value = {'instructions':{'document_number':'Read only OUR NUMBER'}, 'extra_groups':[]}
        response, body = self.change(value=value)
        self.assertEqual(response.status_code,200,response.json)
        for code in ('INVOICE','INVOICE_TAX_INVOICE'):
            self.assertIn('Read only OUR NUMBER', self.reference(code)['groups'][0]['question'])
        old = userpromptsets.select(before['prompt_sets'],['INVOICE'])[0][1]
        self.assertNotIn('Read only OUR NUMBER',old['steps'][0]['rules'])
        versions = self.client.get('/api/user/prompt-sets/'+body['set_id']+'/history').json['versions']
        self.assertEqual([v['revision'] for v in versions],[0,1])
        self.assertEqual(self.client.post('/api/user/prompt-sets',json=body).status_code,409)

    def save_fields(self, code, keys, revision=None):
        revision = revision or self.client.get('/api/user/doc-types').json['revision']
        return self.client.put('/ocr/api/user/doc-types/fields', json={
            'doc_type':code, 'revision':revision, 'enabled_keys':keys})

    def test_field_matrix_can_import_wht_rules_and_detaches_only_the_selected_user_type(self):
        self.change('WHT_CERTIFICATE', value={'instructions':{'payer_name':'Copy the payer legal name only.'}})
        before = self.store.prompt_set_snapshot()
        data = self.client.get('/api/user/doc-types').json
        invoice = next(row for row in data['types'] if row['code'] == 'INVOICE')
        self.assertIn('payer_name', {f['key'] for f in invoice['choices']})
        response = self.save_fields('INVOICE', ['document_number','payer_name'], data['revision'])
        self.assertEqual(response.status_code, 200, response.json)
        after = self.store.prompt_set_snapshot()
        invoice_set = after['sets'][after['bindings']['INVOICE']['set_id']]
        self.assertEqual({k for s in invoice_set['steps'] for k in s['keys']}, {'document_number','payer_name'})
        self.assertTrue(any('Copy the payer legal name only.' in s['rules'] for s in invoice_set['steps']))
        for code in ('INVOICE_TAX_INVOICE','DELIVERY_TAX_INVOICE'):
            self.assertEqual(after['bindings'][code], before['bindings'][code])
        old = before['sets'][before['bindings']['INVOICE']['set_id']]
        self.assertNotIn('payer_name', {k for s in old['steps'] for k in s['keys']})
        # The editor and runtime both understand captured foreign/free-form rules.
        self.assertEqual(self.reference('INVOICE')['editor']['revision'], invoice_set['revision'])
        with EngineAdapter(app).context('field_picker', self.cfg()):
            steps = app.steps_for_types(['INVOICE'], False, mode='hybrid')
            self.assertEqual({k for s in steps for k in s['keys']}, {'document_number','payer_name'})

    def save_matrix(self, changes, revision=None):
        revision = revision or self.client.get('/api/user/doc-types').json['revision']
        return self.client.put('/ocr/api/user/doc-types/fields', json={'revision':revision, 'changes':changes})

    def test_matrix_saves_many_types_at_once_with_the_users_required_fields(self):
        before = self.store.prompt_set_snapshot()
        response = self.save_matrix({
            'INVOICE':{'enabled_keys':['document_number','issue_date','payer_name'], 'required_keys':['issue_date','payer_name']},
            'DELIVERY_TAX_INVOICE':{'enabled_keys':['document_number','seller_tax_id'], 'required_keys':['seller_tax_id']}})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(len(response.json['saved']), 2)
        after = self.store.prompt_set_snapshot()
        # Each saved type got its own copy; types that were not edited keep the shared set.
        self.assertNotEqual(after['bindings']['INVOICE']['set_id'], after['bindings']['DELIVERY_TAX_INVOICE']['set_id'])
        self.assertEqual(after['bindings']['INVOICE_TAX_INVOICE'], before['bindings']['INVOICE_TAX_INVOICE'])
        rows = {row['user_code']:row for row in self.client.get('/api/user/doc-types').json['types']}
        self.assertEqual(rows['380']['mandatory'], ['issue_date','payer_name'])
        self.assertEqual(rows['T04']['mandatory'], ['seller_tax_id'])
        self.assertIn('document_number', rows['T02']['mandatory'])
        self.assertIn('seller_tax_id', self.client.get('/api/user/doc-types').json['requirable'])
        result = self.extract(['INVOICE'])
        self.assertEqual(result['classified']['mandatory'], ['issue_date','payer_name'])
        self.assertNotIn('document_number', str(app._missing_required(result)))
        self.assertIn('payer_name', str(app._missing_required(result)))
        # The older one-type body keeps the saved Required list (minus fields switched off).
        response = self.save_fields('INVOICE', ['document_number','payer_name'])
        self.assertEqual(response.status_code, 200, response.json)
        record = userpromptsets.select(self.store.prompt_set_snapshot(), ['INVOICE'])[0][1]
        self.assertEqual(record['value']['required_fields'], ['payer_name'])

    def board(self):
        data = self.client.get('/api/user/prompt-groups').json
        where = {t['code']:g['id'] for g in data['groups'] for t in g['types']}
        return data, where

    def group_codes(self, data, group_id):
        return [t['code'] for g in data['groups'] if g['id'] == group_id for t in g['types']]

    def test_prompt_groups_are_types_asked_exactly_the_same(self):
        data, where = self.board()
        self.assertEqual({t['code'] for g in data['groups'] for t in g['types']}, set(prompts.TYPE_NAMES))
        self.assertEqual(where['INVOICE'], where['INVOICE_TAX_INVOICE'])
        self.assertEqual(where['DEBIT_NOTE'], where['CANCELLATION_NOTE'])
        # Same set, different built-in Required: not the same group.
        self.assertNotEqual(where['INVOICE'], where['TAX_INVOICE'])
        self.assertEqual(where['TAX_INVOICE'], where['DELIVERY_TAX_INVOICE'])
        # Equal content on two different sets is still one group.
        self.save_matrix({'TAX_INVOICE':{'enabled_keys':['document_number'], 'required_keys':['document_number']},
                          'DELIVERY_TAX_INVOICE':{'enabled_keys':['document_number'], 'required_keys':['document_number']}})
        state = self.store.prompt_set_snapshot()
        self.assertNotEqual(state['bindings']['TAX_INVOICE']['set_id'], state['bindings']['DELIVERY_TAX_INVOICE']['set_id'])
        data, where = self.board()
        self.assertEqual(where['TAX_INVOICE'], where['DELIVERY_TAX_INVOICE'])
        self.assertNotEqual(where['TAX_INVOICE'], where['ABBREVIATED_TAX_INVOICE'])

    def test_moving_a_type_into_a_group_asks_it_exactly_what_the_group_is_asked(self):
        data, where = self.board()
        before = self.store.prompt_set_snapshot()
        target = where['INVOICE']
        response = self.client.post('/ocr/api/user/prompt-groups/moves', json={'revision':data['revision'],
            'moves':{'CREDIT_NOTE':target, 'DEBIT_NOTE':target}})
        self.assertEqual(response.status_code, 200, response.json)
        after, where = self.board()
        self.assertEqual({where[c] for c in ('INVOICE','INVOICE_TAX_INVOICE','CREDIT_NOTE','DEBIT_NOTE')}, {target})
        self.assertEqual(next(g for g in after['groups'] if g['id'] == target)['required'],
                         next(g for g in data['groups'] if g['id'] == target)['required'])
        # Types left in their old groups, and types sharing the old set, keep what they had.
        state = self.store.prompt_set_snapshot()
        for code in ('TAX_INVOICE','DELIVERY_TAX_INVOICE','CANCELLATION_NOTE'):
            self.assertEqual(userpromptsets.fingerprint(state, code), userpromptsets.fingerprint(before, code))
        result = self.extract(['CREDIT_NOTE'])
        self.assertEqual(result['classified']['mandatory'], next(g for g in after['groups'] if g['id'] == target)['required'])
        for body, status in (({'revision':'stale', 'moves':{'RECEIPT':target}}, 409),
                             ({'revision':after['revision'], 'moves':{'RECEIPT':'nope'}}, 400),
                             ({'revision':after['revision'], 'moves':{}}, 400)):
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/user/prompt-groups/moves', json=body).status_code, status)
        self.assertEqual(self.board()[0]['revision'], after['revision'])

    def test_saving_a_group_reaches_every_type_in_it_and_nothing_else_and_one_type_can_split_out(self):
        data, where = self.board()
        codes = self.group_codes(data, where['INVOICE'])
        reference = self.client.get('/api/user/hybrid-prompts', query_string={'doc_type':codes[0], 'template':'1'}).json
        value = copy.deepcopy(reference['editor']['value'])
        value['instructions']['document_number'] = 'GROUP RULE'
        response = self.client.post('/ocr/api/user/prompt-groups/save', json={'revision':data['revision'], 'doc_types':codes, 'value':value})
        self.assertEqual(response.status_code, 200, response.json)
        state = self.store.prompt_set_snapshot()
        for code in codes:
            self.assertIn('GROUP RULE', json.dumps(userpromptsets.effective(state, code)[0]))
        # 388 shared the same set but is another group (different Required): unchanged.
        self.assertNotIn('GROUP RULE', json.dumps(userpromptsets.effective(state, 'TAX_INVOICE')[0]))
        data, where = self.board()
        self.assertEqual(self.group_codes(data, where['INVOICE']), codes)
        # Part of a group is rejected; one type alone splits out.
        self.assertEqual(self.client.post('/api/user/prompt-groups/save', json={'revision':data['revision'],
            'doc_types':['TAX_INVOICE','DELIVERY_TAX_INVOICE'], 'value':value}).status_code, 400)
        value['instructions']['document_number'] = 'T02 ONLY'
        response = self.client.post('/api/user/prompt-groups/save', json={'revision':data['revision'],
            'doc_types':['INVOICE_TAX_INVOICE'], 'value':value})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json['name'].startswith('T02'))
        data, where = self.board()
        self.assertNotEqual(where['INVOICE_TAX_INVOICE'], where['INVOICE'])
        state = self.store.prompt_set_snapshot()
        self.assertIn('GROUP RULE', json.dumps(userpromptsets.effective(state, 'INVOICE')[0]))

    def test_every_type_opens_in_the_group_editor_including_combined_rules(self):
        # The billing-note totals rule is one combined instruction (it talks about
        # billing_total), so it has no per-key bullet for subtotal; it once broke the page.
        for code in prompts.TYPE_NAMES:
            with self.subTest(code=code):
                response = self.client.get('/api/user/hybrid-prompts', query_string={'doc_type':code, 'template':'1'})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertIsNotNone(response.json['editor'])
        totals = next(g for g in self.reference_template('BILLING_NOTE')['editor']['groups'] if g['id'] == 'totals')
        self.assertEqual({f['edit_key'] for f in totals['fields']}, {'subtotal'})
        self.assertEqual(totals['fields'][0]['linked_keys'], ['subtotal','vat_total','amount_incl_vat'])
        record = userpromptsets.template(self.store.prompt_set_snapshot(), 'BILLING_NOTE')
        self.assertEqual([s['rules'] for s in record['steps']],
                         [s['rules'] for s in prompts.steps_for_types(['BILLING_NOTE'], False, mode='hybrid')])

    def reference_template(self, code):
        return self.client.get('/api/user/hybrid-prompts', query_string={'doc_type':code, 'template':'1'}).json

    def test_an_unconfigured_group_is_editable_from_its_built_in_form(self):
        data, where = self.board()
        codes = self.group_codes(data, where['DEBIT_NOTE'])
        reference = self.client.get('/api/user/hybrid-prompts', query_string={'doc_type':'DEBIT_NOTE', 'template':'1'}).json
        self.assertEqual(reference['library']['record']['id'], userpromptsets.TEMPLATE_ID)
        preview = self.client.post('/api/user/hybrid-prompts/preview', json={'doc_type':'DEBIT_NOTE',
            'set_id':userpromptsets.TEMPLATE_ID, 'value':reference['editor']['value']})
        self.assertEqual(preview.status_code, 200, preview.json)
        before = self.store.prompt_set_snapshot()
        response = self.client.post('/api/user/prompt-groups/save', json={'revision':data['revision'], 'doc_types':codes,
                                                                       'value':reference['editor']['value']})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['name'], 'ชุดพื้นฐาน')
        state = self.store.prompt_set_snapshot()
        for code in codes:
            self.assertEqual(userpromptsets.fingerprint(state, code), userpromptsets.fingerprint(before, code))
        self.assertEqual(sorted(self.group_codes(*[self.board()[0], self.board()[1]['DEBIT_NOTE']])), sorted(codes))

    def test_matrix_rejects_bad_required_fields_and_saves_nothing(self):
        self.change('RECEIPT', value={'extra_groups':[{'id':'extra_sales','title':'Sales','fields':[
            {'key':'receipt_sales','instruction':'Read the sales name.','target':'salesman'}]}]})
        before = self.store.prompt_set_snapshot()
        good = {'enabled_keys':['document_number'], 'required_keys':[]}
        for bad in ({'enabled_keys':['document_number'], 'required_keys':['issue_date']},
                    {'enabled_keys':['document_number','receipt_sales'], 'required_keys':['receipt_sales']},
                    {'enabled_keys':['document_number'], 'required_keys':'document_number'},
                    {'enabled_keys':[], 'required_keys':[]}):
            with self.subTest(bad=bad):
                response = self.save_matrix({'CREDIT_NOTE':good, 'INVOICE':bad})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertIn('380', response.json['error'])
                self.assertEqual(self.store.prompt_set_snapshot(), before)
        response = self.save_matrix({'INVOICE':good}, revision='stale')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.store.prompt_set_snapshot(), before)

    def test_field_selection_configures_an_unbound_type_and_can_include_the_income_table(self):
        response = self.save_fields('DEBIT_NOTE', ['document_number','income_items'])
        self.assertEqual(response.status_code, 200, response.json)
        state = self.store.prompt_set_snapshot()
        record = state['sets'][state['bindings']['DEBIT_NOTE']['set_id']]
        self.assertEqual({k for s in record['steps'] for k in s['keys']}, {'document_number','income_items'})
        self.assertEqual(record['base_code'], 'DEBIT_NOTE')
        self.assertEqual(self.reference('DEBIT_NOTE')['editor']['revision'], 1)
        result = self.extract(['DEBIT_NOTE'])
        self.assertEqual(result['items_asked'], list(prompts.INCOME_ITEM_KEYS))

    def test_field_selection_imports_custom_fields_with_their_instruction_and_target(self):
        response, _ = self.change('RECEIPT', value={'extra_groups':[{
            'id':'extra_sales', 'title':'Sales', 'fields':[
                {'key':'receipt_sales', 'instruction':'Read the named sales representative.', 'target':'salesman'}]}]})
        self.assertEqual(response.status_code, 200, response.json)
        response = self.save_fields('INVOICE', ['document_number','receipt_sales'])
        self.assertEqual(response.status_code, 200, response.json)
        record = userpromptsets.select(self.store.prompt_set_snapshot(), ['INVOICE'])[0][1]
        step = next(s for s in record['steps'] if 'receipt_sales' in s['keys'])
        self.assertEqual(step['custom_targets'], {'receipt_sales':'salesman'})
        self.assertIn('Read the named sales representative.', step['rules'])

    def test_field_selection_rejects_conflicts_partial_linked_rules_and_stale_edits_atomically(self):
        response, _ = self.change('RECEIPT', value={'extra_groups':[{
            'id':'extra_tax', 'title':'Tax', 'fields':[
                {'key':'receipt_tax', 'instruction':'Read tax.', 'target':'vatAmount'}]}],
            'disabled_fields':['vat_total']})
        self.assertEqual(response.status_code, 200, response.json)
        before = self.store.prompt_set_snapshot()
        for keys in ([], ['not_a_field'], ['book_no'], ['vat_total','receipt_tax']):
            response = self.save_fields('INVOICE', keys)
            self.assertEqual(response.status_code, 400, response.json)
            self.assertEqual(self.store.prompt_set_snapshot(), before)
        revision = self.client.get('/api/user/doc-types').json['revision']
        self.change('INVOICE', value={'instructions':{'document_number':'Changed elsewhere.'}})
        changed = self.store.prompt_set_snapshot()
        response = self.save_fields('INVOICE', ['document_number'], revision)
        self.assertEqual(response.status_code, 409, response.json)
        self.assertEqual(self.store.prompt_set_snapshot(), changed)

    def test_detach_copies_entire_set_and_literal_binding_blocks_component_override(self):
        value = {'instructions':{'document_number':'T02 ONLY'}, 'extra_groups':[]}
        response,_ = self.change('INVOICE_TAX_INVOICE','detach',value=value,name='T02 custom')
        self.assertEqual(response.status_code,200,response.json)
        cfg = self.cfg()
        with EngineAdapter(app).context('isolated_t02',cfg):
            table = app.steps_for_types(['INVOICE_TAX_INVOICE','INVOICE','TAX_INVOICE'],False,mode='hybrid')
            self.assertEqual(len(table),4)
            self.assertIn('T02 ONLY',table[0]['rules'])
            self.assertNotIn('T02 ONLY',app.steps_for_types(['INVOICE'],False,mode='hybrid')[0]['rules'])
        self.assertNotIn('INVOICE_TAX_INVOICE', {t['code'] for t in self.reference('INVOICE')['library']['shared_types']})

    def test_choose_a_source_for_t04_does_not_change_routing_until_saved(self):
        source = self.reference('INVOICE')['library']['record']['id']
        initial = self.store.prompt_set_snapshot()
        response = self.client.get('/api/user/hybrid-prompts',query_string={
            'doc_type':'DELIVERY_TAX_INVOICE','source_set_id':source})
        self.assertEqual(response.status_code,200)
        self.assertEqual(len(response.json['groups']),4)
        self.assertEqual(self.store.prompt_set_snapshot(),initial)
        response,_ = self.change('DELIVERY_TAX_INVOICE','bind',source=source)
        self.assertEqual(response.status_code,200,response.json)
        self.assertFalse(any(g['id'].startswith('wht') for g in self.reference('DELIVERY_TAX_INVOICE')['groups']))

    def test_field_and_group_disabling_changes_actual_schema_preview_and_does_not_mutate_builtin(self):
        builtin = copy.deepcopy(prompts.steps_for_types(['INVOICE'],False,mode='hybrid'))
        value = {'instructions':{'document_number':'OWN NUMBER'},'extra_groups':[],
                 'disabled_fields':['vat_total'], 'disabled_groups':['po_gr_rtv']}
        response,_ = self.change(value=value)
        self.assertEqual(response.status_code,200,response.json)
        reference = self.reference('INVOICE')
        self.assertFalse(next(g for g in reference['groups'] if g['id']=='po_gr_rtv')['enabled'])
        cfg = self.cfg()
        with EngineAdapter(app).context('disabled_fields',cfg):
            table = app.steps_for_types(['INVOICE'],False,mode='hybrid')
            self.assertFalse(any(s['id']=='po_gr_rtv' for s in table))
            totals = next(s for s in table if s['id']=='totals')
            self.assertNotIn('vat_total',app._step_schema(totals)['properties'])
            self.assertNotIn('vat_total',app._step_question(totals,['INVOICE'],mode='hybrid'))
            question = next(g for g in reference['groups'] if g['id']=='totals')['question']
            self.assertEqual(question,app._step_question(totals,app._framing({'doc_types':['INVOICE']},'hybrid'),mode='hybrid'))
        self.assertEqual(prompts.steps_for_types(['INVOICE'],False,mode='hybrid'),builtin)

    def test_preview_has_no_writes_and_invalid_keys_destinations_and_partial_linked_toggles_fail(self):
        before = self.store.prompt_set_snapshot()
        record = self.reference('INVOICE')['library']['record']
        value = {'instructions':{},'extra_groups':[], 'disabled_groups':['totals']}
        preview = self.client.post('/api/user/hybrid-prompts/preview',json={
            'doc_type':'INVOICE','set_id':record['id'],'value':value})
        self.assertEqual(preview.status_code,200,preview.json)
        self.assertFalse(next(g for g in preview.json['groups'] if g['id']=='totals')['enabled'])
        self.assertEqual(self.store.prompt_set_snapshot(),before)
        for value in ({'disabled_fields':['not_a_key']}, {'disabled_groups':['missing']},
                      {'instructions':{'made_up':'bad'}}, {'disabled_groups':[s['id'] for s in record['base_steps']]}):
            response,_ = self.change(value=value)
            self.assertEqual(response.status_code,400,response.json)
        response,_ = self.change('WHT_CERTIFICATE',value={'disabled_fields':['book_no']})
        self.assertEqual(response.status_code,400,response.json)

    def test_runtime_field_set_rechecks_and_t04_export_remain_consistent(self):
        source = self.reference('INVOICE')['library']['record']['id']
        value = {'instructions':{}, 'extra_groups':[], 'disabled_fields':['vat_total']}
        self.assertEqual(self.change('DELIVERY_TAX_INVOICE','detach',source=source,value=value)[0].status_code,200)
        cfg = self.cfg()
        calls = []
        def ask(content, step, status, collect=None):
            calls.append(step)
            values = {key:'' for key in step['keys']}
            values.update({k:v for k,v in {'document_number':'IV-42','subtotal':'100','amount_incl_vat':'107'}.items() if k in values})
            return values,[json.dumps(values)],False,1,False
        with EngineAdapter(app).context('execute_t04',cfg), patch.object(app,'_ask_step',side_effect=ask), \
                patch.object(app,'AGENTIC_RETRIES',0),patch.object(app,'AGENTIC_PARALLEL',False):
            extracted = app._drain(app._extract_document('ใบส่งของ/ใบกำกับภาษี\nIV-42\n100\n107', 'hybrid', None, None,
                ['DELIVERY_TAX_INVOICE','TAX_INVOICE'], {'kind':'ollama','url':'http://mock','model':'mock'}))
            self.assertNotIn('error',extracted,extracted)
            self.assertNotIn('vat_total',extracted['fields_asked'])
            self.assertFalse(any('vat_total' in step['keys'] for step in calls))
            form = app._fill_form(extracted)
            self.assertNotIn('vat_total',form['keys'])
            self.assertNotIn('vat_total',app._validation_retry_keys(extracted,form))
            ids,orphans = app._fill_steps(extracted['doc_types'],form['mandatory'])
            self.assertFalse(orphans)
        source_job = job(compact_extraction(extracted)); source_job['configuration'] = cfg
        exported = userjson.convert(source_job)
        self.assertEqual(exported['data']['documents'][0]['documentType']['code'],'T04')
        self.assertNotIn('vat_total',source_job['result']['extracted']['classified']['fields'])

    def test_two_concurrent_saves_of_same_version_only_commit_once(self):
        _,body = self.change()
        body['revision'] = 1
        def save():
            try:
                self.store.change_prompt_set(body)
                return 'saved'
            except userpromptsets.Conflict:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(lambda _:save(),range(2))),['saved','conflict'])

    def test_job_local_edit_is_snapshot_only_and_audits_field_toggles(self):
        before = self.store.prompt_set_snapshot()
        local = userpromptsets.retry(before,{'INVOICE':{'instructions':{'document_number':'LOCAL'},
            'extra_groups':[], 'disabled_groups':['totals']}})
        self.assertEqual(self.store.prompt_set_snapshot(),before)
        self.assertNotEqual(local['bindings']['INVOICE']['set_id'],local['bindings']['INVOICE_TAX_INVOICE']['set_id'])
        audit = userpromptsets.changes(before,local)
        self.assertTrue(any(c['entity']=='instruction' and c['key']=='document_number' for c in audit))
        self.assertTrue(any(c['entity']=='group_enabled' and c['key']=='totals' for c in audit))
        self.assertEqual(userpromptsets.changes(before,before),[])

    def test_job_api_reads_its_snapshot_selects_a_source_and_retries_only_that_literal_type(self):
        batch = self.client.post('/api/user/batches',json={'file_count':1}).json
        parent = self.store.add_job(batch['batch_id'],'upload_01','old.pdf',b'pdf')
        self.store.update(parent['id'],status='completed',result={'text':'ใบส่งของ/ใบกำกับภาษี',
            'pages':['ใบส่งของ/ใบกำกับภาษี'], 'extracted':{'doc_types':['DELIVERY_TAX_INVOICE','TAX_INVOICE'],'fields':{}}})
        source_id = batch['configuration']['prompt_sets']['bindings']['INVOICE']['set_id']
        self.change(value={'instructions':{'document_number':'NEW TEAM RULE'},'extra_groups':[]})
        before = self.store.prompt_set_snapshot()
        url = '/api/user/jobs/'+parent['id']+'/prompts'
        data = self.client.get(url).json
        self.assertEqual(data['available_types'],['DELIVERY_TAX_INVOICE'])
        self.assertIsNotNone(data['editor'])
        data = self.client.get(url,query_string={'source_set_id':source_id}).json
        self.assertEqual(data['library']['record']['revision'],0)
        self.assertNotIn('NEW TEAM RULE',data['groups'][0]['question'])
        body = {'request_id':'retry_sets_001','prompt_sets':{'DELIVERY_TAX_INVOICE':{'set_id':source_id,
            'value':{'instructions':{'document_number':'LOCAL T04'},'extra_groups':[],'disabled_fields':['vat_total']}}}}
        with patch.object(self.service,'start'):
            response = self.client.post('/api/user/jobs/'+parent['id']+'/retry-extraction',json=body)
            self.assertEqual(response.status_code,202,response.json)
            child = response.json['job']
            self.assertEqual(self.client.post('/api/user/jobs/'+parent['id']+'/retry-extraction',json=body).json['job']['id'],child['id'])
        snapshot = child['configuration']['prompt_sets']
        record = userpromptsets.select(snapshot,['DELIVERY_TAX_INVOICE','TAX_INVOICE'])[0][1]
        self.assertIn('LOCAL T04',record['steps'][0]['rules'])
        self.assertEqual(self.store.prompt_set_snapshot(),before)
        self.assertTrue(any(c['entity']=='field_enabled' for c in child['prompt_changes']['changes']))
        self.assertEqual(self.store.get(parent['id'])['configuration'],batch['configuration'])

    def test_custom_mapping_is_validated_against_active_fields_and_keeps_typed_arrays(self):
        extra = {'id':'extra_amount','title':'Additional','fields':[
            {'key':'custom_tax','instruction':'Read the printed tax amount','target':'vatAmount'}]}
        value = {'instructions':{},'extra_groups':[extra]}
        self.assertEqual(self.change(value=value)[0].status_code,400)
        value['disabled_fields'] = ['vat_total']
        response,_ = self.change(value=value)
        self.assertEqual(response.status_code,200,response.json)
        step = response.json['record']['steps'][-1]
        self.assertEqual(step['custom_targets'],{'custom_tax':'vatAmount'})
        bad = copy.deepcopy(value);bad['extra_groups'][0]['fields'].append({
            'key':'again','instruction':'Read another printed tax','target':'vatAmount'})
        self.assertEqual(self.change(value=bad)[0].status_code,400)
        bad = copy.deepcopy(value);bad['extra_groups'][0]['fields'][0]['target'] = 'documentType.code'
        self.assertEqual(self.change(value=bad)[0].status_code,400)

    def test_bind_and_shared_save_cannot_silently_discard_edits_or_stale_binding(self):
        source = self.reference('INVOICE')['library']['record']['id']
        response,_ = self.change('DELIVERY_TAX_INVOICE','bind',source=source,value={
            'instructions':{'document_number':'should not disappear'},'extra_groups':[]})
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.store.prompt_set_snapshot()['bindings']['DELIVERY_TAX_INVOICE']['set_id'],source)
        response,body = self.change('INVOICE_TAX_INVOICE','detach')
        self.assertEqual(response.status_code,200)
        body['revision'] = 0;body['action'] = 'shared'
        self.assertEqual(self.client.post('/api/user/prompt-sets',json=body).status_code,409)

    def test_captured_base_survives_engine_rule_changes_in_preview_save_and_retry_audit(self):
        before = self.store.prompt_set_snapshot()
        with patch.object(app.userprompts,'base_steps',side_effect=AssertionError('Must use captured base')):
            data = self.reference('INVOICE')
            self.assertIn('document_number',data['groups'][0]['keys'])
            response,_ = self.change(value={'instructions':{'document_number':'CAPTURED OVERRIDE'},'extra_groups':[]})
            self.assertEqual(response.status_code,200,response.json)
            audit = userpromptsets.changes(before,self.store.prompt_set_snapshot())
            self.assertTrue(any(row['entity']=='instruction' for row in audit))

    def test_disabled_required_field_is_not_extraction_missing_but_user_json_still_reports_it(self):
        self.change(value={'disabled_fields':['document_number']})
        def ask(content,step,status,collect=None):
            values = {key:'' for key in step['keys']}
            return values,[json.dumps(values)],False,1,False
        with EngineAdapter(app).context('disabled_required',self.cfg()), \
                patch.object(app,'_ask_step',side_effect=ask),patch.object(app,'AGENTIC_PARALLEL',False), \
                patch.object(app,'AGENTIC_RETRIES',0):
            result = app._drain(app._extract_document('INVOICE', 'hybrid',None,None,['INVOICE'],
                {'kind':'ollama','model':'fake','url':'http://mock'}))
            self.assertNotIn('document_number',result['classified']['mandatory'])
            self.assertNotIn('document_number',result['fields'])
            self.assertNotIn('document_number',str(app._missing_required(result)))
        report = userjson.convert(job(compact_extraction(result)))
        self.assertTrue(any(row['path'].endswith('.documentNo') for row in report['warnings']))

    def test_linked_wht_toggles_and_custom_array_schema_survive_set_serialization(self):
        value = {'disabled_fields':['book_no','certificate_no'], 'extra_groups':[{
            'id':'extra_notes','title':'Notes','fields':[{'key':'printed_notes','instruction':'Read printed notes as an array','target':'remarks'}]}]}
        response,_ = self.change('WHT_CERTIFICATE',value=value)
        self.assertEqual(response.status_code,200,response.json)
        record = self.reference('WHT_CERTIFICATE')['library']['record']
        numbering = next(s for s in record['steps'] if s['id']=='wht_certificate')
        self.assertEqual(numbering['keys'],['sequence_no'])
        extra = record['steps'][-1]
        self.assertEqual(app._step_schema(extra)['properties']['printed_notes']['type'],'array')
        self.assertEqual(app._step_values(extra,'{"printed_notes":["Printed"]}'),{'printed_notes':['Printed']})

    def test_extra_groups_and_fields_can_be_disabled_without_losing_their_saved_definitions(self):
        value = {'instructions':{}, 'extra_groups':[{'id':'extra_notes','title':'Extra','fields':[
            {'key':'note','instruction':'Read printed notes','target':'remarks'},
            {'key':'seller_name','instruction':'Read the seller name','target':'seller.name'}]}],
            'disabled_fields':['seller_name']}
        response,_ = self.change(value=value)
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['record']['steps'][-1]['keys'],['note'])
        self.assertEqual(len(response.json['record']['value']['extra_groups'][0]['fields']),2)
        value['disabled_groups'] = ['extra_notes']
        response,_ = self.change(value=value)
        self.assertEqual(response.status_code,200,response.json)
        self.assertFalse(any(s['id'].startswith('user_') for s in response.json['record']['steps']))

    def test_partial_groups_on_a_mixed_document_keep_the_more_specific_fields(self):
        self.change(value={'instructions':{'seller_tax_id':'Only the supplier taxpayer block.'}})
        response,_ = self.change('CREDIT_NOTE',value={'disabled_fields':['buyer_tax_id']})
        self.assertEqual(response.status_code,200,response.json)
        with EngineAdapter(app).context('mixed_conflict',self.cfg()):
            steps = app.steps_for_types(['CREDIT_NOTE','INVOICE'],False,mode='hybrid')
            keys = [key for step in steps for key in step['keys']]
            self.assertEqual(len(keys),len(set(keys)))
            self.assertEqual(set(keys),set(prompts.fields_for_types(['CREDIT_NOTE','INVOICE'])))
            owner = next(step for step in steps if 'buyer_tax_id' in step['keys'])
            self.assertEqual(owner['user_prompt_doc_type'],'INVOICE')
            self.assertEqual(owner['keys'],('buyer_tax_id',))
            self.assertEqual(set(app._step_schema(owner)['properties']),{'buyer_tax_id'})
            self.assertNotIn('Only the supplier taxpayer block.',owner['rules'])

    def test_shared_field_guidance_edits_do_not_break_mixed_documents_and_ownership_is_recorded(self):
        self.assertEqual(self.change(value={'instructions':{'seller_tax_id':'Use only the supplier taxpayer block.'},
                                            'extra_groups':[]})[0].status_code,200)
        cfg = self.cfg()
        pairs = (['INVOICE','RECEIPT'],['INVOICE','CREDIT_NOTE'],
                 ['INVOICE_TAX_INVOICE','RECEIPT'],['RECEIPT_TAX_INVOICE','INVOICE'])
        def ask(content,step,status,collect=None):
            return {key:'' for key in step['keys']},[],False,0,False
        with EngineAdapter(app).context('mixed_edits',cfg),patch.object(app,'_ask_step',side_effect=ask), \
                patch.object(app,'AGENTIC_RETRIES',0),patch.object(app,'AGENTIC_PARALLEL',False):
            for pair in pairs:
                with self.subTest(pair=pair):
                    codes = list(dict.fromkeys(c for kind in pair for c in userpromptsets.codes_for(kind)))
                    table = app.steps_for_types(codes,False,mode='hybrid')
                    keys = [key for step in table for key in step['keys']]
                    self.assertEqual(len(keys),len(set(keys)))
                    self.assertEqual(set(keys),set(prompts.fields_for_types(codes)))
                    result = app._drain(app._extract_document('printed evidence','hybrid',None,None,codes,
                        {'kind':'ollama','url':'http://mock','model':'fake'},extras=False))
                    self.assertNotIn('error',result,result)
                    ownership = [key for source in result['user_prompt_sets'] for key in source['keys']]
                    self.assertCountEqual(ownership,keys)
                    first = userpromptsets.select(cfg['prompt_sets'],codes)[0][0]
                    self.assertEqual(next(step for step in table if 'seller_tax_id' in step['keys'])['user_prompt_doc_type'],first)

    def extract(self, codes, cfg=None):
        def ask(content,step,status,collect=None):
            return {key:'' for key in step['keys']},[],False,0,False
        with EngineAdapter(app).context('extract',cfg or self.cfg()),patch.object(app,'_ask_step',side_effect=ask), \
                patch.object(app,'AGENTIC_RETRIES',0),patch.object(app,'AGENTIC_PARALLEL',False):
            return app._drain(app._extract_document('printed evidence','hybrid',None,None,codes,
                {'kind':'ollama','url':'http://mock','model':'fake'},extras=False))

    def test_two_sets_mapping_one_user_json_destination_keep_the_first_on_a_mixed_document(self):
        def note(key):
            return {'instructions':{}, 'extra_groups':[{'id':'extra_sales','title':'Sales','fields':[
                {'key':key,'instruction':'The salesperson name.','target':'salesman'}]}]}
        self.assertEqual(self.change('INVOICE',value=note('inv_sales'))[0].status_code,200)
        self.assertEqual(self.change('RECEIPT',value=note('rcp_sales'))[0].status_code,200)
        for pair in (['INVOICE','RECEIPT'],['RECEIPT_TAX_INVOICE','INVOICE']):
            with self.subTest(pair=pair):
                codes = list(dict.fromkeys(c for kind in pair for c in userpromptsets.codes_for(kind)))
                with EngineAdapter(app).context('targets',self.cfg()):
                    table = app.steps_for_types(codes,False,mode='hybrid')
                keys = [key for step in table for key in step['keys']]
                first = userpromptsets.select(self.cfg()['prompt_sets'],codes)[0][1]['base_code']
                kept, dropped = ('inv_sales','rcp_sales') if first == 'INVOICE' else ('rcp_sales','inv_sales')
                self.assertIn(kept,keys)
                self.assertNotIn(dropped,keys)
                result = self.extract(codes)
                self.assertNotIn('error',result,result)
                skipped = [k for source in result['user_prompt_sets'] for k in source['skipped_keys']]
                self.assertIn(dropped,skipped)

    def test_legacy_profile_save_keeps_the_sets_disabled_fields(self):
        self.change(value={'disabled_fields':['buyer_branch_location']})
        revision = self.store.prompt_set_snapshot()['sets']['builtin_invoice']['revision']
        response = self.client.put('/api/user/hybrid-prompts',json={'doc_type':'INVOICE','revision':revision,
            'value':{'instructions':{'document_number':'Read only OUR NUMBER'},'extra_groups':[]}})
        self.assertEqual(response.status_code,200,response.json)
        record = self.store.prompt_set_snapshot()['sets']['builtin_invoice']
        self.assertEqual(record['value']['disabled_fields'],['buyer_branch_location'])
        self.assertIn('Read only OUR NUMBER',json.dumps(record['steps']))
        self.assertNotIn('buyer_branch_location',[k for s in record['steps'] for k in s['keys']])

    REPLACED = {'instructions':{}, 'disabled_fields':['seller_branch'], 'extra_groups':[{'id':'extra_branch',
        'title':'Branch','fields':[{'key':'my_branch','instruction':'Supplier branch code.','target':'seller.branchCode'}]}]}

    def test_legacy_profile_save_accepts_a_custom_field_replacing_a_disabled_one(self):
        self.assertEqual(self.change(value=copy.deepcopy(self.REPLACED))[0].status_code,200)
        record = self.store.prompt_set_snapshot()['sets']['builtin_invoice']
        response = self.client.put('/api/user/hybrid-prompts',json={'doc_type':'INVOICE','revision':record['revision'],
            'value':{'instructions':{'document_number':'Read only OUR NUMBER'},'extra_groups':self.REPLACED['extra_groups']}})
        self.assertEqual(response.status_code,200,response.json)
        self.assertNotIn('disabled_fields',response.json['profile']['value'])
        record = self.store.prompt_set_snapshot()['sets']['builtin_invoice']
        self.assertEqual(record['value']['disabled_fields'],['seller_branch'])
        keys = [k for s in record['steps'] for k in s['keys']]
        self.assertIn('my_branch',keys)
        self.assertNotIn('seller_branch',keys)

    def test_legacy_retry_keeps_the_jobs_disabled_fields(self):
        self.assertEqual(self.change(value=copy.deepcopy(self.REPLACED))[0].status_code,200)
        batch = self.client.post('/api/user/batches',json={'file_count':1}).json
        parent = self.store.add_job(batch['batch_id'],'upload_02','old.pdf',b'pdf')
        self.store.update(parent['id'],status='completed',result={'text':'Invoice','pages':['Invoice'],
            'extracted':{'doc_types':['INVOICE'],'fields':{}}})
        for request_id, extra in (('retry_legacy_01',[]),('retry_legacy_02',self.REPLACED['extra_groups'])):
            with self.subTest(extra_groups=bool(extra)):
                body = {'request_id':request_id,'prompt_profiles':{'INVOICE':{
                    'instructions':{'document_number':'LOCAL'},'extra_groups':extra}}}
                with patch.object(self.service,'start'):
                    response = self.client.post('/api/user/jobs/'+parent['id']+'/retry-extraction',json=body)
                self.assertEqual(response.status_code,202,response.json)
                snapshot = response.json['job']['configuration']['prompt_sets']
                record = snapshot['sets'][snapshot['bindings']['INVOICE']['set_id']]
                self.assertEqual(record['value']['disabled_fields'],['seller_branch'])
                keys = [k for s in record['steps'] for k in s['keys']]
                self.assertNotIn('seller_branch',keys)
                self.assertEqual('my_branch' in keys,bool(extra))
                self.assertIn('LOCAL',json.dumps(record['steps']))

    def test_user_disabled_fields_lists_only_what_the_user_switched_off(self):
        self.change(value={'disabled_fields':['buyer_branch_location']})
        result = self.extract(userpromptsets.codes_for('DELIVERY_TAX_INVOICE'))
        self.assertNotIn('error',result,result)
        self.assertEqual(result['user_disabled_fields'],['buyer_branch_location'])
        self.assertEqual(result['user_prompt_sets'][0]['skipped_keys'],[])

    def test_sup_pay_defaults_names_and_thai_extra_labels(self):
        state = self.store.prompt_set_snapshot()
        invoice = 'builtin_invoice'
        for code in ('INVOICE','TAX_INVOICE','INVOICE_TAX_INVOICE','DELIVERY_TAX_INVOICE','ABBREVIATED_TAX_INVOICE'):
            self.assertEqual(state['bindings'][code]['set_id'],invoice)
            self.assertFalse(any(g['id'].startswith('wht') for g in self.reference(code)['groups']))
        self.assertEqual(state['bindings']['CREDIT_NOTE']['set_id'],'builtin_credit_note')
        for code in ('RECEIPT','RECEIPT_TAX_INVOICE','RECEIPT_ABBREVIATED_TAX_INVOICE'):
            self.assertEqual(state['bindings'][code]['set_id'],'builtin_receipt')
        for code in ('DEBIT_NOTE','CANCELLATION_NOTE'):
            self.assertNotIn(code,state['bindings'])
        self.assertEqual({r['base_code']:r['name'] for r in state['sets'].values()},userpromptsets.DEFAULT_NAMES)
        extra = {r['code']:r['label'] for r in userpromptsets.catalog() if r['section']=='extra'}
        self.assertEqual(extra['WHT_CERTIFICATE'],'WHT · หนังสือรับรองการหักภาษี ณ ที่จ่าย (50 ทวิ)')
        self.assertEqual(extra['RTV_SHIP_CREDIT'],'RTV · ใบส่งคืนสินค้า')
        self.assertEqual(UserStore(self.temp.name,32).prompt_set_snapshot(),state)

    def test_migration_two_preserves_user_bindings_and_names_and_records_default_renames(self):
        directory = Path(self.temp.name) / 'migration-one'
        legacy = UserStore(directory,32)
        with legacy.connect() as db:
            db.execute('DELETE FROM prompt_set_migrations WHERE id=2')
            db.execute("DELETE FROM prompt_bindings WHERE code IN ('TAX_INVOICE','DELIVERY_TAX_INVOICE','ABBREVIATED_TAX_INVOICE')")
            db.execute("DELETE FROM prompt_binding_events WHERE action='migration_2'")
            for row in db.execute('SELECT * FROM prompt_sets').fetchall():
                record = json.loads(row['payload']); record['name'] = record['base_code']
                payload = json.dumps(record)
                db.execute('UPDATE prompt_sets SET name=?,payload=? WHERE id=?',(record['name'],payload,record['id']))
                db.execute('UPDATE prompt_set_versions SET payload=? WHERE set_id=? AND revision=?',
                           (payload,record['id'],record['revision']))
        state = legacy.prompt_set_snapshot()
        invoice = state['sets']['builtin_invoice']
        legacy.change_prompt_set({'doc_type':'INVOICE','set_id':invoice['id'],'revision':invoice['revision'],
            'binding_revision':0,'action':'shared','name':'My custom invoice','value':invoice['value']})
        receipt = state['sets']['builtin_receipt']
        legacy.change_prompt_set({'doc_type':'DELIVERY_TAX_INVOICE','set_id':receipt['id'],'revision':receipt['revision'],
            'binding_revision':0,'action':'bind'})
        before = legacy.prompt_set_snapshot()
        migrated = UserStore(directory,32)
        after = migrated.prompt_set_snapshot()
        self.assertEqual(after['bindings']['DELIVERY_TAX_INVOICE'],before['bindings']['DELIVERY_TAX_INVOICE'])
        self.assertEqual(after['sets']['builtin_invoice']['name'],'My custom invoice')
        self.assertEqual(after['bindings']['TAX_INVOICE']['set_id'],'builtin_invoice')
        self.assertEqual(after['bindings']['ABBREVIATED_TAX_INVOICE']['set_id'],'builtin_invoice')
        self.assertEqual(after['bindings']['RECEIPT_TAX_INVOICE'],before['bindings']['RECEIPT_TAX_INVOICE'])
        self.assertEqual(after['sets']['builtin_receipt']['name'],'Receipt (Sup&Pay)')
        self.assertEqual(after['sets']['builtin_receipt']['steps'],before['sets']['builtin_receipt']['steps'])
        self.assertEqual([v['payload']['name'] for v in migrated.prompt_set_history('builtin_receipt')],['RECEIPT','Receipt (Sup&Pay)'])
        self.assertEqual(UserStore(directory,32).prompt_set_snapshot(),after)


if __name__ == '__main__':
    unittest.main()
