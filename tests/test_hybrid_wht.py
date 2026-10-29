"""Grouped WHT transport; inference quality must be checked on the remote model."""
import copy
import json
import unittest
from collections import Counter
from unittest.mock import patch

import app
import extractionlog
import prompts


STATUS = {'kind': 'vllm', 'url': 'http://mock:8001', 'model': 'mock', 'reachable': True}
TEXT = ('Withholding tax certificate\n'
        'Payer: River Ltd. Tax ID: 1234567890123 Head Office Address: 11 River Road\n'
        'Payee: Cedar Ltd. Tax ID: 9876543210987 Branch No. 00007 Branch name Hill\n')
VALUES = {'payer_tax_id': '1234567890123', 'payer_name': 'River Ltd.',
          'payer_branch': 'Head Office', 'payer_branch_location': 'Head Office',
          'payer_address': '11 River Road', 'payee_tax_id': '9876543210987',
          'payee_name': 'Cedar Ltd.', 'payee_branch': '00007',
          'payee_branch_location': 'Hill', 'payee_address': ''}


class HybridWhtTests(unittest.TestCase):
    def test_wht_has_six_groups_with_exact_coverage_and_other_forms_retain_their_groups(self):
        baseline = copy.deepcopy(prompts.EXTRACT_STEPS)
        for extras in (False, True):
            old = prompts.steps_for_types(['WHT_CERTIFICATE'], extras, mode='agentic')
            new = prompts.steps_for_types(['WHT_CERTIFICATE'], extras, mode='hybrid')
            self.assertEqual((len(old), len(new)), (10 + extras, 6 + extras))
            self.assertEqual(Counter(k for s in old for k in s['keys']),
                             Counter(k for s in new for k in s['keys']))
            for step in new:
                self.assertEqual(set(json.loads(step['skeleton'])), set(step['keys']) or {'other_fields'})
        for codes in ([], ['PAYMENT_SCHEDULE'], ['BILLING_NOTE'], ['CHEQUE_DELIVERY']):
            table = prompts.steps_for_types(codes, False, mode='hybrid')
            self.assertIn('wht_payer_branch', {s['id'] for s in table})
            self.assertNotIn('wht_payer_party', {s['id'] for s in table})
        mixed = prompts.steps_for_types(['WHT_CERTIFICATE', 'INVOICE'], False, mode='hybrid')
        keys = [k for s in mixed for k in s['keys']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(prompts.EXTRACT_STEPS, baseline)

    def test_historical_step_ids_and_missing_field_rechecks_resolve_to_one_party(self):
        for party in ('payer', 'payee'):
            primary = 'wht_' + party
            group = primary + '_party'
            for selected in ([primary], [primary + '_branch'], [primary + '_address'],
                             [group], [primary, primary + '_branch', primary + '_address']):
                steps = app._steps_for(selected, ['WHT_CERTIFICATE'], False, mode='hybrid')
                self.assertEqual([s['id'] for s in steps], [group])
                self.assertEqual(set(steps[0]['keys']), {k for k in VALUES if k.startswith(party + '_')})
            ids, orphans = app._fill_steps(['WHT_CERTIFICATE'], [party + '_branch', party + '_address'])
            self.assertEqual(ids, [group]); self.assertEqual(orphans, [])

    def test_grouped_requests_preserve_blank_fields_and_detail_log_ownership(self):
        for schema_first in (False, True):
            form = app._field_set(TEXT, doc_type=['WHT_CERTIFICATE'], extras=False)
            form['schema_first'] = schema_first
            calls = []
            def chat(message, cap, status, schema=None):
                keys = set(json.loads(app._first_json_object(message)))
                self.assertEqual(len(keys), 5)
                self.assertEqual(cap, 600)
                self.assertTrue(all(k.startswith('payer_') for k in keys) or
                                all(k.startswith('payee_') for k in keys))
                if schema_first:
                    self.assertEqual(set(schema['required']), keys)
                else:
                    self.assertIsNone(schema)
                calls.append(keys)
                return json.dumps({k: VALUES[k] for k in keys}), False, 80
            with patch.object(app, '_chat', side_effect=chat), patch.object(app, 'AGENTIC_PARALLEL', False):
                result = app._drain(app._extract_agentic(TEXT, STATUS, form,
                    only=['wht_payer', 'wht_payee_address'], mode='hybrid'))
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['fields'], {**VALUES, 'line_items': [], 'other_fields': [], 'income_items': []})
            catalog = {'contexts': {}, 'questions': {}}
            detail = extractionlog.job_detail({'_extraction_result': result}, {'index': 0},
                                             'benchmark', catalog=catalog)
            rows = {r['path']: r for r in detail['field_rows']}
            for key, value in VALUES.items():
                self.assertEqual(rows[key]['actual'], value)
                party = key.split('_', 1)[0]
                self.assertEqual(rows[key]['step'], 'wht_' + party + '_party')
            self.assertEqual(len(detail['documents'][0]['step_stats']), 2)


if __name__ == '__main__':
    unittest.main()
