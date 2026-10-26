"""Billing prompt routing and transport; replies are stubs, not model evaluations."""
import copy
from functools import partial
import json
import unittest
from unittest.mock import patch

import app
import prompts

# These tests cover the tuned variant, now exposed explicitly as Hybrid.
hybrid_steps = partial(prompts.steps_for_types, mode="hybrid")
hybrid_question = partial(app._step_question, mode="hybrid")
hybrid_selected = partial(app._steps_for, mode="hybrid")


class BillingPromptTests(unittest.TestCase):
    def test_billing_rules_are_isolated_from_mixed_invoice_and_return_forms(self):
        saved = copy.deepcopy(prompts.EXTRACT_STEPS)
        table = {s['id']: s for s in hybrid_steps(['BILLING_NOTE'], extras=False)}
        self.assertIn('customer being billed', table['buyer']['title'])
        self.assertIn(prompts._BILLING_PARTIES, table['buyer']['rules'])
        self.assertIn(prompts._BILLING_PARTIES, table['seller']['rules'])
        self.assertIn(prompts._BILLING_TOTALS, table['totals']['rules'])
        # The covering bill replaces the generic before-VAT question instead
        # of appending an exception after conflicting subtotal instructions.
        self.assertEqual(table['totals']['rules'], prompts._BILLING_TOTALS)
        self.assertIn(prompts._AGENTIC_VAT_RULES, table['totals']['rules'])
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
                self.assertTrue(all(len(s['keys']) <= 3 for s in table))
                for step in table:
                    self.assertEqual(set(json.loads(step['skeleton'])), set(step['keys']))
                    self.assertEqual(set(app._step_schema(step)['properties']), set(step['keys']))

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
        answers = {'buyer_tax_id': '1234567890123', 'buyer_branch': 'Head Office',
                   'buyer_branch_location': 'Head Office', 'subtotal': '240.00',
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
                    self.assertEqual(result['fields'][key], value)
                self.assertTrue(all(not s.get('retried') for s in result['steps']))


if __name__ == '__main__':
    unittest.main()
