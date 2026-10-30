"""Comparison is descriptive; it must not expand or replace the live prompts."""
import copy
import json
import unittest
from unittest.mock import patch

import app
import doctype_mapping as mapping
import prompts


class DocTypeMappingTests(unittest.TestCase):
    def setUp(self):
        self.data = mapping.comparison()
        self.types = {row['user_code']: row for row in self.data['types'] if row['user_code']}
        self.fields = {row['user_field']: row for row in self.data['fields'] if row['user_field']}

    def test_every_user_type_and_leaf_has_exactly_one_comparison_row(self):
        catalog = json.loads((mapping.ROOT / 'user_document_type_catalog.json').read_text(encoding='utf-8'))
        contract = json.loads((mapping.ROOT / 'user_document_bundle_v7.schema.json').read_text(encoding='utf-8'))
        self.assertEqual(set(self.types), {t['code'] for t in catalog['documentTypes']})
        paths = [path for path, _ in mapping._leaves(contract)]
        self.assertEqual(set(paths), set(self.fields))
        self.assertEqual(len(paths), len(self.fields))
        for section in ('types', 'fields'):
            self.assertEqual(sum(self.data['counts'][section].values()), len(self.data[section]))

    def test_combinations_are_explicit_and_scope_false_does_not_filter_t03(self):
        self.assertEqual(self.types['T03']['ours'], ['RECEIPT_TAX_INVOICE'])
        self.assertFalse(self.types['T03']['user_scope'])
        for code in ('T02', 'T03', 'T04', 'T05', 'T06', 'T07'):
            self.assertEqual(self.types[code]['status'], 'mapped')
            self.assertEqual(len(self.types[code]['ours']), 1)
            self.assertFalse(prompts.DOC_TYPE_FIELDS.get(self.types[code]['ours'][0]))

    def test_rtv_and_every_internal_type_stay_visible_without_being_forced_to_credit_note(self):
        represented = {code for row in self.data['types'] for code in row['ours']}
        self.assertEqual(represented, set(prompts.TYPE_NAMES))
        rtv = next(row for row in self.data['types'] if row['ours'] == ['RTV_SHIP_CREDIT'])
        self.assertEqual(rtv['status'], 'extra')
        self.assertFalse(rtv['user_code'])
        self.assertEqual(self.types['81']['ours'], ['CREDIT_NOTE'])

    def test_branch_locations_are_additions_and_retired_names_are_missing(self):
        for party in ('seller', 'buyer'):
            location = next(row for row in self.data['fields'] if row['ours'] == [party + '_branch_location'])
            self.assertEqual(location['status'], 'extra')
            self.assertIn('branchLocation', location['note'])
            self.assertEqual(self.fields[f'documents[].{party}.name']['status'], 'missing')
            self.assertEqual(self.fields[f'documents[].{party}.branchCode']['status'], 'partial')

    def test_table_references_and_financial_meanings_are_not_reported_as_complete(self):
        for path in ('references[].type', 'references[].code', 'items[].itemCode', 'itemType'):
            self.assertEqual(self.fields['documents[].' + path]['status'], 'partial')
        for path in ('netAmount', 'taxBasisTotalAmount'):
            self.assertEqual(self.fields['documents[].' + path]['status'], 'missing')
        self.assertEqual(self.fields['documents[].amountIncludeVat']['ours'], ['amount_incl_vat'])
        self.assertEqual(self.fields['file.fileName']['status'], 'system')

    def test_no_live_field_is_omitted_or_live_configuration_mutated(self):
        before = copy.deepcopy((prompts.DOC_TYPE_FIELDS, prompts.EXTRACT_STEPS, prompts.TYPE_NAMES))
        data = mapping.comparison()
        represented = {source for row in data['fields'] for source in row['ours']}
        self.assertTrue(set(prompts._SCALAR_KEYS).issubset(represented))
        self.assertEqual(before, (prompts.DOC_TYPE_FIELDS, prompts.EXTRACT_STEPS, prompts.TYPE_NAMES))
        with patch.object(prompts, '_SCALAR_KEYS', tuple(k for k in prompts._SCALAR_KEYS if k != 'seller_tax_id')):
            fields = {row['user_field']: row for row in mapping.comparison()['fields']}
            self.assertEqual(fields['documents[].seller.taxId']['status'], 'missing')

    def test_shared_schema_endpoint_includes_mapping_without_touching_a_model(self):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No model request')):
            client = app.app.test_client()
            for prefix in ('', '/ocr'):
                response = client.get(prefix + '/api/schema')
                self.assertEqual(response.status_code, 200)
                body = response.get_json()
                self.assertEqual(body['keys'], list(prompts._SCALAR_KEYS))
                self.assertEqual(body['user_mapping']['version'], '7.0')
                self.assertEqual(len(body['types']), len(prompts.TYPE_NAMES))


if __name__ == '__main__':
    unittest.main()
