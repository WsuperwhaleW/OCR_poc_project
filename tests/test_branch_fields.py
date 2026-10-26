"""Party branch contracts and deterministic scoring, without local model inference."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import extractionlog
import fieldscore
import normalise
import prompts
import runlog


ROOT = Path(__file__).resolve().parents[1]


class BranchNormalizationTests(unittest.TestCase):
    def test_only_numbers_or_explicit_code_captions_become_branch_codes(self):
        for raw, want in [('226', '00226'), ('สาขาที่ ๒๒๖', '00226'),
                          ('Branch No.00012', '00012'), ('สาขา: 01065', '01065'),
                          ('(ศูนย์เลนกลาง) ลำดับสาขา 05656', '05656'),
                          ('สำนักงานใหญ่ สาขาที่ 00068', '00068'),
                          ('HEAD OFFICE', '00000'), ('ลำดับที่ 00000', '00000')]:
            with self.subTest(raw=raw):
                self.assertEqual(normalise.branch_code(raw), want)

    def test_place_digits_and_tax_ids_cannot_turn_into_codes(self):
        for raw in ('สาขาพระราม 9', 'ลาดพร้าว 101', 'สำนักงานใหญ่พระราม 9',
                    '0105543041887', 'สาขาที่ 0105543041887', 'Room 226', 'MyBranch00012', '-12'):
            with self.subTest(raw=raw):
                # The explicit head-office word still has its established meaning.
                want = '00000' if raw.startswith('สำนักงานใหญ่') else ''
                self.assertEqual(normalise.branch_code(raw), want)

    def test_location_removes_caption_and_wrapper_only(self):
        for raw, want in [('(สาขาพระราม 9)', 'พระราม 9'),
                          ('ชื่อ สาขา: อำพัน แพร่', 'อำพัน แพร่'),
                          ('Branch name: Central Rama 9', 'Central Rama 9'),
                          ('(HEAD OFFICE)', 'สำนักงานใหญ่'),
                          ('Branchwood Park', 'Branchwood Park'),
                          ('ศูนย์เลนกลาง', 'ศูนย์เลนกลาง')]:
            self.assertEqual(normalise.field_value('buyer_branch_location', raw), want)

    def test_legal_names_keep_internal_words_and_geographic_qualifiers(self):
        for raw, want in [('ชื่อ กรุ๊ป จำกัด', 'กรุ๊ป จำกัด'),
                          ('ชื่อผู้จ่ายเงิน: บริษัท สาขาไทย จำกัด (ประเทศไทย)',
                           'บริษัท สาขาไทย จำกัด (ประเทศไทย)'),
                          ('บริษัท ชื่อดี จำกัด', 'บริษัท ชื่อดี จำกัด'),
                          ('Name: Example (Thailand) Ltd.', 'Example (Thailand) Ltd.')]:
            self.assertEqual(normalise.field_value('payer_name', raw), want)

    def test_normalized_view_never_mutates_or_infers_missing_fields(self):
        fields = {'buyer_branch': 'สาขาที่ 226', 'buyer_branch_location': '(สาขาพระราม 9)',
                  'seller_address': 'พระราม 9', 'income_items': [{'amount_paid': '1,000'}]}
        saved = copy.deepcopy(fields)
        result = normalise.normalize_fields(fields)
        self.assertEqual(fields, saved)
        self.assertEqual(result['buyer_branch'], '00226')
        self.assertEqual(result['buyer_branch_location'], 'พระราม 9')
        self.assertNotIn('seller_branch_location', result)
        self.assertEqual(normalise.normalize_fields({'buyer_branch': '00000'}),
                         {'buyer_branch': '00000'})
        # A misplaced branch name remains visible and scoreable as an error.
        self.assertEqual(normalise.field_value('buyer_branch', 'พระราม 9'), 'พระราม 9')


class BranchContractTests(unittest.TestCase):
    def test_every_profile_has_exact_step_and_schema_coverage_with_small_requests(self):
        for codes in ([], ['INVOICE'], ['RECEIPT'], ['CREDIT_NOTE'], ['RTV_SHIP_CREDIT'],
                      ['RTV_SHIP_CREDIT', 'CREDIT_NOTE'], ['WHT_CERTIFICATE']):
            keys = prompts.fields_for_types(codes)
            steps = prompts.steps_for_types(codes, extras=False)
            owned = [k for s in steps for k in s['keys'] if k != 'income_items']
            self.assertEqual(set(owned), set(keys))
            self.assertEqual(len(owned), len(set(owned)))
            self.assertTrue(all(len(s['keys']) <= 3 for s in steps))
            for step in steps:
                self.assertEqual(set(json.loads(step['skeleton'])), set(step['keys']))
            schema = prompts.extract_schema(keys, prompts.items_for_types(codes), extras=False)
            self.assertTrue(set(keys) <= set(schema['properties']))

    def test_locations_are_optional_without_changing_existing_requirement(self):
        for codes in (['INVOICE'], ['RECEIPT'], ['CREDIT_NOTE'], ['WHT_CERTIFICATE']):
            self.assertTrue(any(k.endswith('_branch_location') for k in prompts.fields_for_types(codes)))
            self.assertFalse(any(k.endswith('_branch_location') for k in prompts.mandatory_for_types(codes)))

    def test_prompts_require_same_party_evidence_without_fixture_answers(self):
        for codes in (['INVOICE'], ['WHT_CERTIFICATE'], ['RTV_SHIP_CREDIT']):
            text = prompts.extract_prompt(prompts.fields_for_types(codes), codes, extras=False)
            self.assertIn('A place containing digits is still a place', text)
            self.assertIn('ordinary address', text)
            self.assertIn('SAME party block', text)
            self.assertNotIn('ศูนย์เลนกลาง', text)
            self.assertNotIn('05656', text)
            self.assertNotIn('A branch name, a branch number', text)

    def test_validation_adds_normalized_view_but_keeps_raw_fields(self):
        form = app._field_set('INVOICE', doc_type=['INVOICE'], extras=False)
        raw = {'buyer_branch': 'สาขาที่ 226', 'buyer_branch_location': 'สาขาพระราม 9'}
        result = app._validate_fields({'fields': copy.deepcopy(raw)}, form, 'INVOICE', 'agentic')
        self.assertEqual(result['fields'], raw)
        self.assertEqual(result['normalized']['buyer_branch'], '00226')
        self.assertEqual(result['normalized']['buyer_branch_location'], 'พระราม 9')
        self.assertEqual(result['normalization_version'], normalise.NORMALIZATION_VERSION)
        self.assertEqual(app.check_template_fields(), [])

    def test_wht_pipeline_sends_separate_branch_schemas_and_merges_the_answers(self):
        expected = json.loads((ROOT / 'solution/sol016.fields.json').read_text(encoding='utf-8'))
        text = (ROOT / 'solution/sol016.md').read_text(encoding='utf-8')
        form = app._field_set(text, doc_type=['WHT_CERTIFICATE'], extras=False)
        form['schema_first'] = True
        requests = []

        def chat(message, cap, status, schema=None):
            keys = schema['properties']
            requests.append(set(keys))
            values = {k: expected.get(k, '') for k in keys}
            return json.dumps(values, ensure_ascii=False), False, 20

        with patch.object(app, '_chat', side_effect=chat), patch.object(app, 'AGENTIC_PARALLEL', False):
            result = app._drain(app._extract_agentic(text,
                {'kind': 'vllm', 'url': 'http://mock:8001', 'model': 'mock'}, form))
        self.assertNotIn('error', result)
        self.assertEqual(len(requests), len(prompts.steps_for_types(['WHT_CERTIFICATE'], extras=False)))
        self.assertIn({'payer_branch', 'payer_branch_location'}, requests)
        self.assertIn({'payee_branch', 'payee_branch_location'}, requests)
        self.assertTrue(all(len(keys) <= 3 for keys in requests))
        self.assertEqual(result['fields']['payer_branch_location'], 'สำนักงานใหญ่')


class BranchScoringTests(unittest.TestCase):
    def evaluate(self, case, fields, pages=None):
        return fieldscore.evaluate(case, fields, keys=list(fields), pages=pages)

    def test_code_caption_padding_and_location_caption_compare_canonically(self):
        fields = {'buyer_branch': '05656', 'buyer_branch_location': '(ชื่อสาขา: ศูนย์เลนกลาง)'}
        result = self.evaluate('sol009', fields)
        rows = {r['path']: r for r in result['scalars']['rows']}
        self.assertTrue(all(r['status'] == 'correct' for r in rows.values()))
        self.assertEqual(rows['buyer_branch_location']['actual'], fields['buyer_branch_location'])
        self.assertEqual(rows['buyer_branch_location']['normalized_actual'], 'ศูนย์เลนกลาง')
        self.assertEqual(result['normalization_version'], normalise.NORMALIZATION_VERSION)

    def test_place_cannot_pass_as_numeric_branch_and_unknown_truth_is_unscored(self):
        result = self.evaluate('sol012', {'buyer_branch': 'พระราม 9',
                                         'buyer_branch_location': 'สำนักงานใหญ่'}, [1])
        rows = {r['path']: r for r in result['scalars']['rows']}
        self.assertEqual(rows['buyer_branch']['status'], 'wrong')
        self.assertNotIn('buyer_branch_location', rows)

    def test_caption_free_name_matches_but_original_value_remains_in_log(self):
        result = self.evaluate('sol016', {'payer_name': 'ชื่อ กรุ๊ป จำกัด'})
        row = result['scalars']['rows'][0]
        self.assertEqual(row['status'], 'correct')
        self.assertEqual(row['actual'], 'ชื่อ กรุ๊ป จำกัด')
        self.assertEqual(row['normalized_actual'], 'กรุ๊ป จำกัด')

    def test_detail_download_preserves_raw_and_normalized_values(self):
        fields = {'buyer_branch': '05656', 'buyer_branch_location': '(สาขาศูนย์เลนกลาง)'}
        score = self.evaluate('sol009', fields)
        document = {'fields': fields, 'field_score': score,
                    'normalization_version': normalise.NORMALIZATION_VERSION}
        record = {'status': 'ok', '_extraction_result': document}
        log = extractionlog.job_detail(record, {'index': 0, 'case': 'sol009'}, 'benchmark')
        self.assertEqual(log['documents'][0]['normalization_version'], normalise.NORMALIZATION_VERSION)
        row = next(r for r in log['field_rows'] if r['path'] == 'buyer_branch_location')
        self.assertEqual(row['actual'], '(สาขาศูนย์เลนกลาง)')
        self.assertEqual(row['normalized_actual'], 'ศูนย์เลนกลาง')

    def test_original_log_without_new_metadata_remains_readable(self):
        result = extractionlog.job_detail({'_extraction_result': {'fields': {'buyer_branch': '05656'}}},
                                           {'index': 0}, 'benchmark')
        self.assertEqual(result['field_rows'][0]['actual'], '05656')

    def test_csv_records_contract_version_and_does_not_relabel_legacy_rows(self):
        for value, expected in [({'normalization_version': 1}, 1), ({}, ''),
                                ({'documents': [{'normalization_version': 1}, {'normalization_version': 1}]}, 1),
                                ({'documents': [{'normalization_version': 1}, {}]}, '')]:
            self.assertEqual(runlog._extract_cells({'extracted': value})['extract_normalization_version'], expected)

    def test_reviewed_truth_includes_sources_and_conservative_unknowns(self):
        count = 0
        for path in (ROOT / 'solution').glob('*.fields.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            for doc in data.get('documents', [data]):
                review = doc['_branch_review']
                self.assertTrue((ROOT / review['source']).is_file())
                self.assertTrue(review['reviewed_pages'])
                count += 1
        self.assertEqual(count, 39)
        self.assertNotIn('buyer_branch_location', fieldscore.load_truth('sol003')['scalars'])
        self.assertEqual(fieldscore.load_truth('sol023', pages=[6, 7, 8])['scalars']['buyer_branch_location'], 'สุรินทร์')
        self.assertEqual(fieldscore.load_truth('sol015', pages=[1])['scalars']['seller_branch'], 'HEAD OFFICE')


if __name__ == '__main__':
    unittest.main()
