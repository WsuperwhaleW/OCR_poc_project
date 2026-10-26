"""Grouped requests, aliases and evidence rules; no model accuracy claims."""
import copy
from functools import partial
import itertools
import json
import unittest
from unittest.mock import patch

import app
import extractionlog
import prompts

# These tests cover the tuned variant, now exposed explicitly as Hybrid.
hybrid_steps = partial(prompts.steps_for_types, mode="hybrid")
hybrid_question = partial(app._step_question, mode="hybrid")
hybrid_selected = partial(app._steps_for, mode="hybrid")


STATUS = {"kind": "vllm", "url": "http://mock:8001", "model": "mock", "reachable": True}
PAYMENT = {"inv_rtv_cnr_number": "INV-731", "payment_date": "15/10/2026",
           "cheque_number": "CHQ-29"}
TOTALS = {"subtotal": "100.00", "vat_total": "7.00", "amount_incl_vat": "107.00"}
TEXT = """ใบเสร็จรับเงิน / ใบกำกับภาษี
เลขที่ RC-480 วันที่ 16/10/2026
ใบกำกับภาษีอ้างอิง INV-731 วันที่ชำระ 15/10/2026 เลขเช็ค CHQ-29
ยอดก่อนภาษี 100.00 ภาษีมูลค่าเพิ่ม 7.00 ยอดรวมภาษี 107.00
"""


class GroupSelectionTests(unittest.TestCase):
    def test_every_form_combination_keeps_exact_coverage_without_mutating_definitions(self):
        saved = copy.deepcopy(prompts.EXTRACT_STEPS)
        codes = list(prompts.DOC_TYPE_FIELDS)
        combos = [()] + [combo for n in (1, 2, 3)
                         for combo in itertools.combinations(codes, n)]
        for combo in combos:
            for extras in (False, True):
                table = hybrid_steps(combo, extras)
                keys = [k for s in table for k in s['keys']]
                expected = set(prompts.fields_for_types(combo))
                if prompts.items_for_types(combo):
                    expected.add(prompts.INCOME_ITEMS_KEY)
                if extras:
                    expected.add('other_fields')
                self.assertEqual(set(keys), expected, combo)
                self.assertEqual(len(keys), len(set(keys)), combo)
                self.assertTrue(all(len(s['keys']) <= (6 if s['id'] == 'parties' else 3)
                                    for s in table), combo)
                for step in table:
                    requested = {step.get('request_aliases', {}).get(k, k) for k in step['keys']}
                    self.assertEqual(set(json.loads(step['skeleton'])), requested)
                    self.assertEqual(set(app._step_schema(step)['properties']), requested)
        self.assertEqual(prompts.EXTRACT_STEPS, saved)

    def test_receipt_does_not_gain_a_vat_inclusive_total(self):
        table = {s['id']: s for s in hybrid_steps(['RECEIPT'], extras=False)}
        self.assertEqual(set(table['totals']['keys']), {'subtotal', 'vat_total'})
        self.assertEqual(set(table['payment']['keys']), set(PAYMENT))
        self.assertNotIn('total_incl', table)
        self.assertNotIn('inv_rtv_cnr', table)
        with self.assertRaisesRegex(ValueError, 'total_incl'):
            hybrid_selected(['total_incl'], ['RECEIPT'], extras=False)

    def test_legacy_component_ids_select_a_group_once(self):
        table = hybrid_selected(['total_incl', 'totals'], ['INVOICE'], extras=False)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['id'], 'totals')
        self.assertEqual(set(table[0]['keys']), set(TOTALS))
        table = hybrid_selected(['inv_rtv_cnr', 'payment'], ['RECEIPT'], extras=False)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['id'], 'payment')
        self.assertEqual(set(table[0]['keys']), set(PAYMENT))
        with self.assertRaisesRegex(ValueError, 'made_up'):
            hybrid_selected(['made_up'], ['RECEIPT'], extras=False)

    def test_rtv_keeps_return_roles_in_its_joint_party_request(self):
        table = {s['id']: s for s in hybrid_steps(['RTV_SHIP_CREDIT'], extras=False)}
        self.assertEqual(set(table['totals']['keys']), set(TOTALS))
        for party in ('seller', 'buyer'):
            question = hybrid_question(table['parties'], ['RTV_SHIP_CREDIT'])
            self.assertIn(party + '_tax_id', question)
            self.assertIn('ORIGINAL SALE', question)
            self.assertIn('ลำดับสาขา', question)
            self.assertIn('not the store/location code', question)
            self.assertIn('printed all-zero branch number', question)
            self.assertNotIn('00803', question)
            self.assertNotIn('05656', question)


class RequestPipelineTests(unittest.TestCase):
    def test_grouped_keys_share_one_chat_and_keep_individual_log_attribution(self):
        # Stub only model replies. Real request assembly, parsing, grounding,
        # schema selection, partial-step selection, merging and export still run.
        for codes, keys, old_id in [(['INVOICE'], TOTALS, 'total_incl'),
                                    (['RECEIPT'], PAYMENT, 'inv_rtv_cnr')]:
            for schema_first in (False, True):
                with self.subTest(codes=codes, schema_first=schema_first):
                    form = app._field_set(TEXT, doc_type=codes, extras=False)
                    form['schema_first'] = schema_first
                    requests = []

                    def chat(message, cap, status, schema=None):
                        skeleton = json.loads(app._first_json_object(message))
                        requests.append((message, skeleton, schema, cap))
                        return json.dumps(keys), False, 20

                    with patch.object(app, '_chat', side_effect=chat), \
                         patch.object(app, 'AGENTIC_PARALLEL', False):
                        result = app._drain(app._extract_agentic(TEXT, STATUS, form, only=[old_id], mode="hybrid"))
                    self.assertNotIn('error', result)
                    self.assertEqual(len(requests), 1)
                    message, skeleton, schema, cap = requests[0]
                    self.assertEqual(set(skeleton), set(keys))
                    self.assertEqual(result['fields'], {**keys, 'line_items': [], 'other_fields': []})
                    self.assertFalse(result['steps'][0].get('retried'))
                    self.assertEqual(set(result['steps'][0]['keys']), set(keys))
                    if schema_first:
                        self.assertEqual(set(schema['properties']), set(keys))
                    else:
                        self.assertIsNone(schema)
                    detail = extractionlog.job_detail({'_extraction_result': result},
                                                       {'index': 0}, 'benchmark')
                    rows = {r['path']: r for r in detail['field_rows']}
                    self.assertTrue(set(keys) <= set(rows))
                    self.assertTrue(all(rows[k]['actual'] == value for k, value in keys.items()))
                    self.assertTrue(all(rows[k]['step'] == result['steps'][0]['id'] for k in keys))

    def test_targeted_payment_with_missing_cheque_keeps_reference_and_date(self):
        values = {**PAYMENT, 'cheque_number': ''}
        form = app._field_set(TEXT, doc_type=['RECEIPT'], extras=False)
        with patch.object(app, '_chat', return_value=(json.dumps(values), False, 20)):
            result = app._drain(app._extract_agentic(TEXT, STATUS, form, only=['payment'], mode='hybrid'))
        self.assertEqual(result['fields'], {**values, 'line_items': [], 'other_fields': []})
        self.assertEqual(len(result['steps']), 1)


if __name__ == '__main__':
    unittest.main()
