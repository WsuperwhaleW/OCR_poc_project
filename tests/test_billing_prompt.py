"""Billing prompt routing and transport; replies are stubs, not model evaluations."""
import copy
from functools import partial
import json
import unittest
from unittest.mock import patch

import app
import prompts
import extractionlog

# These tests cover the tuned variant, now exposed explicitly as Hybrid.
hybrid_steps = partial(prompts.steps_for_types, mode="hybrid")
hybrid_question = partial(app._step_question, mode="hybrid")
hybrid_selected = partial(app._steps_for, mode="hybrid")


class BillingPromptTests(unittest.TestCase):
    def test_billing_rules_are_isolated_from_mixed_invoice_and_return_forms(self):
        saved = copy.deepcopy(prompts.EXTRACT_STEPS)
        table = {s['id']: s for s in hybrid_steps(['BILLING_NOTE'], extras=False)}
        self.assertIn(prompts._BILLING_PARTIES, table['parties']['rules'])
        self.assertEqual(table['parties']['includes_steps'], ('seller', 'buyer'))
        self.assertIn(prompts._BILLING_TOTALS, table['totals']['rules'])
        # The covering bill replaces the generic before-VAT question instead
        # of appending an exception after conflicting subtotal instructions.
        self.assertEqual(table['totals']['rules'], prompts._BILLING_TOTALS)
        self.assertIn(prompts._AGENTIC_VAT_RULES.replace('subtotal', 'billing_total'), table['totals']['rules'])
        for codes in (['INVOICE'], ['RECEIPT'], ['WHT_CERTIFICATE'], [],
                      ['BILLING_NOTE', 'INVOICE'], ['BILLING_NOTE', 'RTV_SHIP_CREDIT']):
            for step in hybrid_steps(codes, extras=False):
                self.assertNotIn(prompts._BILLING_PARTIES, step['rules'])
                self.assertNotIn(prompts._BILLING_TOTALS, step['rules'])
        self.assertEqual(prompts.EXTRACT_STEPS, saved)

    def test_billing_form_keeps_exact_schema_coverage_and_small_requests(self):
        for codes in (['BILLING_NOTE'], ['BILLING_NOTE', 'INVOICE'],
                      ['BILLING_NOTE', 'RTV_SHIP_CREDIT'], ['BILLING_NOTE', 'RECEIPT']):
            for extras in (False, True):
                table = hybrid_steps(codes, extras)
                actual = [k for s in table for k in s['keys']]
                expected = set(prompts.fields_for_types(codes))
                if extras:
                    expected.add('other_fields')
                if prompts.items_for_types(codes):
                    expected.add(prompts.INCOME_ITEMS_KEY)
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(set(actual)))
                self.assertTrue(all(len(s['keys']) <= (6 if s['id'] == 'parties' else 3)
                                    for s in table))
                for step in table:
                    requested = {step.get('request_aliases', {}).get(k, k) for k in step['keys']}
                    self.assertEqual(set(json.loads(step['skeleton'])), requested)
                    self.assertEqual(set(app._step_schema(step)['properties']), requested)

    def test_actual_buyer_and_totals_requests_receive_billing_rules(self):
        text = """BILLING NOTE
Supplier: Paper River Ltd.
Customer: Cedar Retail Ltd. (Head Office)
Bill no: BN-051 Date: 15/10/2026
Customer Tax ID: 1234567890123
Invoice INV-X Amount 240.00
Total amount billed 240.00
"""
        status = {'kind': 'vllm', 'model': 'mock', 'url': 'http://mock:8001'}
        answers = {'seller_tax_id': '', 'seller_branch': '', 'seller_branch_location': '',
                   'buyer_tax_id': '1234567890123', 'buyer_branch': 'Head Office',
                   'buyer_branch_location': 'Head Office', 'billing_total': '240.00',
                   'vat_total': '', 'amount_incl_vat': ''}
        for schema_first in (False, True):
            with self.subTest(schema_first=schema_first):
                form = app._field_set(text, doc_type=['BILLING_NOTE'], extras=False)
                form['schema_first'] = schema_first
                requests = []

                def chat(message, cap, server, schema=None):
                    keys = set(json.loads(app._first_json_object(message)))
                    requests.append((message, keys, schema))
                    return json.dumps({k: answers[k] for k in keys}), False, 30

                with patch.object(app, '_chat', side_effect=chat), \
                     patch.object(app, 'AGENTIC_PARALLEL', False):
                    result = app._drain(app._extract_agentic(text, status, form, only=['buyer', 'totals'], mode='hybrid'))
                self.assertNotIn('error', result)
                self.assertEqual(len(requests), 2)
                self.assertIn(prompts._BILLING_PARTIES, requests[0][0])
                self.assertIn(prompts._BILLING_TOTALS, requests[1][0])
                for message, keys, schema in requests:
                    self.assertIn(text, message)
                    if schema_first:
                        self.assertEqual(set(schema['properties']), keys)
                    else:
                        self.assertIsNone(schema)
                for key, value in answers.items():
                    self.assertEqual(result['fields']['subtotal' if key == 'billing_total' else key], value)
                self.assertNotIn('billing_total', result['fields'])
                total = next(s for s in result['steps'] if s['id'] == 'totals')
                self.assertEqual(total['request_aliases'], {'subtotal': 'billing_total'})
                self.assertIn('billing_total', total['raw'][0]['text'])
                detail = extractionlog.job_detail({'_extraction_result': result}, {'index': 0}, 'benchmark')
                stats = next(s for s in detail['documents'][0]['step_stats'] if s['id'] == 'totals')
                self.assertEqual(stats['request_aliases'], {'subtotal': 'billing_total'})
                row = next(r for r in detail['field_rows'] if r['path'] == 'subtotal')
                self.assertEqual(row['actual'], '240.00')
                self.assertTrue(all(not s.get('retried') for s in result['steps']))

    def test_canonical_reply_key_is_reasked_with_the_actual_billing_schema(self):
        step = next(s for s in hybrid_steps(['BILLING_NOTE'], extras=False) if s['id'] == 'totals')
        replies = [(' {"subtotal":"240.00","vat_total":"","amount_incl_vat":""}', False, 10),
                   (' {"billing_total":"240.00","vat_total":"","amount_incl_vat":""}', False, 10)]
        with patch.object(app, '_chat', side_effect=replies) as chat:
            values, raw, _, _, _ = app._ask_step('question', step, {'kind': 'vllm'})
        self.assertEqual(values['subtotal'], '240.00')
        self.assertEqual(len(raw), 2)
        self.assertIn('billing_total', chat.call_args.args[3]['properties'])
        self.assertNotIn('subtotal', chat.call_args.args[3]['properties'])

    def test_grounding_retry_names_billing_total_and_preserves_raw_problem_evidence(self):
        step = next(s for s in hybrid_steps(['BILLING_NOTE'], extras=False) if s['id'] == 'totals')
        replies = [(' {"billing_total":"invented","vat_total":"","amount_incl_vat":""}', False, 10),
                   (' {"billing_total":"240.00","vat_total":"","amount_incl_vat":""}', False, 10)]
        with patch.object(app, '_chat', side_effect=replies) as chat:
            record, values, _, _ = app._run_step(step, 'printed total 240.00',
                hybrid_question(step, ['BILLING_NOTE']), {'kind': 'vllm'}, app.grounding.Source('240.00'))
        self.assertTrue(record['retried'])
        self.assertEqual(values['subtotal'], '240.00')
        self.assertIn('- billing_total = invented', chat.call_args.args[0])
        self.assertNotIn('- subtotal = invented', chat.call_args.args[0])
        detail = extractionlog.job_detail({'_extraction_result': {'fields': values, 'steps': [record]}},
                                           {'index': 0}, 'benchmark')
        evidence = detail['documents'][0]['steps'][0]
        self.assertEqual(evidence['request_aliases'], {'subtotal': 'billing_total'})
        self.assertIn('invented', evidence['raw'][0]['text'])


if __name__ == '__main__':
    unittest.main()
