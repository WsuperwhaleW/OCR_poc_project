"""Paired identity transport and diagnostics; model accuracy needs an AWS run."""
import json
import unittest
from unittest.mock import patch

import app
import extractionlog
import prompts


STATUS = {'kind': 'vllm', 'url': 'http://mock:8001', 'model': 'mock', 'reachable': True}
VALUES = {'seller_tax_id': '1234567890123', 'seller_branch': '00007',
          'seller_branch_location': 'River', 'buyer_tax_id': '9876543210987',
          'buyer_branch': 'Head Office', 'buyer_branch_location': 'Head Office'}
TEXT = """Invoice
Supplier: Paper River Ltd. Tax ID: 1234567890123 Branch No. 00007 Branch name River
Customer: Cedar Retail Ltd. Tax ID: 9876543210987 Head Office
"""


class HybridPartyTests(unittest.TestCase):
    def test_historical_ids_and_new_id_select_the_joint_request_once(self):
        for ids in (['seller'], ['buyer'], ['parties'], ['seller', 'buyer', 'parties']):
            table = app._steps_for(ids, ['INVOICE'], extras=False, mode='hybrid')
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]['id'], 'parties')
            self.assertEqual(table[0]['includes_steps'], ('seller', 'buyer'))
            self.assertEqual(set(table[0]['keys']), set(VALUES))
            self.assertEqual(table[0]['max_tokens'], 600)
        for party in ('seller', 'buyer'):
            old = app._steps_for([party], ['INVOICE'], extras=False, mode='agentic')
            self.assertEqual(old[0]['id'], party)
            self.assertEqual(len(old[0]['keys']), 3)
        with self.assertRaisesRegex(ValueError, 'parties'):
            app._steps_for(['parties'], ['WHT_CERTIFICATE'], extras=False, mode='hybrid')

    def test_paired_request_parses_both_sides_and_exports_one_step(self):
        for schema_first in (False, True):
            for absent in (None, 'seller', 'buyer'):
                with self.subTest(schema_first=schema_first, absent=absent):
                    answers = {k: '' if absent and k.startswith(absent + '_') else v
                               for k, v in VALUES.items()}
                    form = app._field_set(TEXT, doc_type=['INVOICE'], extras=False)
                    form['schema_first'] = schema_first
                    calls = []

                    def chat(message, cap, status, schema=None):
                        calls.append((message, cap, schema))
                        return json.dumps(answers), False, 75

                    with patch.object(app, '_chat', side_effect=chat), \
                         patch.object(app, 'AGENTIC_PARALLEL', False):
                        result = app._drain(app._extract_agentic(
                            TEXT, STATUS, form, only=['seller', 'buyer'], mode='hybrid'))
                    self.assertNotIn('error', result)
                    self.assertEqual(len(calls), 1)
                    message, cap, schema = calls[0]
                    self.assertEqual(cap, 600)
                    self.assertIn(TEXT, message)
                    self.assertEqual(set(json.loads(app._first_json_object(message))), set(VALUES))
                    if schema_first:
                        self.assertEqual(set(schema['properties']), set(VALUES))
                        self.assertEqual(set(schema['required']), set(VALUES))
                    else:
                        self.assertIsNone(schema)
                    self.assertEqual(result['fields'], {**answers, 'line_items': [], 'other_fields': []})
                    self.assertEqual(len(result['steps']), 1)
                    self.assertEqual(result['steps'][0]['id'], 'parties')
                    self.assertFalse(result['steps'][0].get('retried'))

                    catalog = {'contexts': {}, 'questions': {}}
                    detail = extractionlog.job_detail({'_extraction_result': result},
                                                       {'index': 0}, 'benchmark', catalog=catalog)
                    rows = {r['path']: r for r in detail['field_rows']}
                    for key, value in answers.items():
                        self.assertEqual(rows[key]['actual'], value)
                        self.assertEqual(rows[key]['step'], 'parties')
                    doc = detail['documents'][0]
                    self.assertEqual(len(doc['step_stats']), 1)
                    self.assertEqual(set(doc['step_stats'][0]['keys']), set(VALUES))
                    evidence = doc['steps'][0]  # Unscored fields retain response evidence.
                    self.assertEqual(evidence['values'], answers)
                    self.assertEqual(json.loads(evidence['raw'][0]['text']), answers)
                    question = catalog['questions'][evidence['question_id']]
                    self.assertIn('buyer_tax_id', question)
                    self.assertIn('seller_tax_id', question)

    def test_schema_fallback_reasks_the_complete_party_group(self):
        step = app._steps_for(['buyer'], ['INVOICE'], False, mode='hybrid')[0]
        replies = [('not JSON', False, 3), (json.dumps(VALUES), False, 75)]
        with patch.object(app, '_chat', side_effect=replies) as chat:
            values, raw, _, _, _ = app._ask_step('shared transcript and question', step, STATUS)
        self.assertEqual(values, VALUES)
        self.assertEqual(len(raw), 2)
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(all(c.args[1] == 600 for c in chat.call_args_list))
        self.assertEqual(set(chat.call_args.args[3]['properties']), set(VALUES))


if __name__ == '__main__':
    unittest.main()
