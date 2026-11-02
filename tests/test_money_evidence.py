"""Currency normalization needs document evidence, not just two trailing digits."""
import copy
from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import patch

import userjson
import verify
from tests.test_userjson import job


FLAT = ('จำนวนเงิน (ไม่รวมภาษี) Amount (Exclucive of tax) 25,757\n01\n'
        'ภาษีมูลค่าเพิ่ม V.A.T. 7% 1,802 99\n'
        'รวมเงินทั้งสิ้น Total Amount 27,560 00\n')
WORDS = '(สองหมื่นเจ็ดพันห้าร้อยหกสิบบาทถ้วน)'


def source(text, **amounts):
    saved = job()
    saved['result']['extracted']['pages'] = [1]
    saved['result']['extracted']['fields'].update(
        subtotal='25,757 01', vat_total='1,802 99', amount_incl_vat='27,560 00')
    saved['result']['extracted']['fields'].update(amounts)
    saved['result']['pages'] = [text]
    return saved


class MoneyEvidenceTests(unittest.TestCase):
    def test_candidate_is_strict_and_does_not_claim_evidence(self):
        for raw, expected in [('25,757 01', '25757.01'), ('1,802\n99', '1802.99'),
                              ('27,560 00', '27560'), ('๒๕,๗๕๗ ๐๑', '25757.01')]:
            with self.subTest(raw=raw):
                self.assertEqual(verify.split_amount(raw), Decimal(expected))
        for raw in ['25,757.01', '25,757 1', '25,757', '25,757 -', '1,23 45',
                    '25,757.01 1,802.99', '1 000 00', '123 456', 'INV 25 01',
                    '-25 01', '(25 01)', '25 01 extra', None, True, 2501]:
            with self.subTest(raw=raw):
                self.assertIsNone(verify.split_amount(raw))

    def test_thai_words_only_accept_complete_standalone_currency_lines(self):
        cases = {WORDS: '27560', 'จำนวนเงินตัวอักษร: (หนึ่งร้อยยี่สิบเอ็ดบาทห้าสตางค์)': '121.05',
                 'ตัวอักษร: ศูนย์บาทห้าสิบสตางค์': '.50',
                 'หนึ่งล้านสองแสนสามหมื่นสี่พันห้าร้อยหกสิบเจ็ดบาทแปดสิบเก้าสตางค์': '1234567.89'}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(verify.thai_money_words(raw)[0]['value'], Decimal(expected))
        for raw in ['หมายเหตุ จ่ายหนึ่งร้อยบาทถ้วนแล้ว', 'สองหมื่นบาท',
                    'หนึ่งบาทหนึ่งร้อยสตางค์', 'หนึ่งร้อยสิบสิบบาทถ้วน',
                    'ไม่ทราบบาทถ้วน', 'สิบสองบาทถ้วน extra']:
            with self.subTest(raw=raw):
                self.assertEqual(verify.thai_money_words(raw), [])

    def test_flat_totals_with_words_normalize_saved_fields_and_audit_without_inference(self):
        saved = source(FLAT + WORDS, subtotal='25,757\n01')
        before = copy.deepcopy(saved)
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
            report = userjson.convert(saved)
        doc = report['data']['documents'][0]
        self.assertEqual([doc[k] for k in ('amountExcludeVat', 'vatAmount', 'amountIncludeVat')],
                         [25757.01, 1802.99, 27560.0])
        audit = [m for m in report['mappings'] if m.get('normalization')]
        self.assertEqual(len(audit), 3)
        self.assertEqual(audit[0]['original'], '25,757\n01')
        self.assertEqual(audit[0]['normalization']['evidence']['amount_in_words'], WORDS)
        self.assertEqual(saved, before)
        self.assertTrue(report['schema_valid'])

    def test_actual_sol012_transcript_confirms_receipt_totals_without_using_page_two(self):
        text = (Path(__file__).resolve().parents[1] / 'solution/sol012.md').read_text(encoding='utf-8')
        text = text.split('--- page 2 ---')[0]
        report = userjson.convert(source(text))
        doc = report['data']['documents'][0]
        self.assertEqual(doc['amountExcludeVat'], 25757.01)
        self.assertEqual(doc['vatAmount'], 1802.99)
        self.assertEqual(doc['amountIncludeVat'], 27560.0)

    def test_repetition_arithmetic_and_words_mentioning_a_different_amount_are_not_enough(self):
        for text in [FLAT, FLAT + '(หนึ่งร้อยบาทถ้วน)',
                     FLAT + WORDS + '\n(หนึ่งร้อยบาทถ้วน)',
                     FLAT.replace('1,802 99', '1,802 98') + WORDS,
                     'Account No. 25,757 01\n' + WORDS]:
            with self.subTest(text=text):
                report = userjson.convert(source(text))
                self.assertNotIn('amountExcludeVat', report['data']['documents'][0])
                self.assertTrue(any('บาท–สตางค์' in w['message'] for w in report['warnings']))
                self.assertEqual(report['review_status'], 'needs_review')
                self.assertEqual(report['unmapped'][0]['source']['fields']['subtotal'], '25,757 01')

    def test_words_confirm_only_the_total_when_component_sum_disagrees(self):
        report = userjson.convert(source(FLAT.replace('1,802 99', '1,802 98') + WORDS))
        doc = report['data']['documents'][0]
        self.assertEqual(doc['amountIncludeVat'], 27560.0)
        self.assertNotIn('vatAmount', doc)

    def test_labelled_markdown_and_html_currency_columns_are_sufficient(self):
        for text in [
            '| รายการ | บาท | สตางค์ |\n| --- | --- | --- |\n'
            '| จำนวนเงินก่อนภาษี | 25,757 | 01 |\n| ภาษีมูลค่าเพิ่ม | 1,802 | 99 |\n'
            '| รวมเงินทั้งสิ้น | 27,560 | 00 |',
            '<table><tr><th>รายการ</th><th>Baht บาท</th><th>Satang สตางค์</th></tr>'
            '<tr><td>Subtotal</td><td>25,757</td><td>01</td></tr>'
            '<tr><td>VAT</td><td>1,802</td><td>99</td></tr>'
            '<tr><td>Total Amount</td><td>27,560</td><td>00</td></tr></table>',
        ]:
            with self.subTest(text=text):
                report = userjson.convert(source(text))
                doc = report['data']['documents'][0]
                self.assertEqual([doc[k] for k in ('amountExcludeVat', 'vatAmount', 'amountIncludeVat')],
                                 [25757.01, 1802.99, 27560.0])
                self.assertTrue(all(m['normalization']['method'] == 'baht_satang_columns'
                                    for m in report['mappings'] if m.get('normalization')))

    def test_unheaded_columns_or_matching_money_in_the_wrong_row_are_not_evidence(self):
        for text in [
            '| รายการ | จำนวนเงิน | |\n| --- | --- | --- |\n| จำนวนเงินก่อนภาษี | 25,757 | 01 |',
            '| รายการ | บาท | สตางค์ |\n| --- | --- | --- |\n| ยอดรวมทั้งสิ้น | 25,757 | 01 |',
            '| รายการ | บาท | สตางค์ |\n| --- | --- | --- |\n| จำนวนเงินก่อนภาษี | 25,757 | |',
            '| รายการ | บาท | สตางค์ |\n| --- | --- | --- |\n| จำนวนเงินก่อนภาษี | 25,757 | 1 |',
        ]:
            with self.subTest(text=text):
                self.assertNotIn('amountExcludeVat', userjson.convert(source(text))['data']['documents'][0])

    def test_conflicting_labelled_totals_and_different_extracted_value_are_rejected(self):
        for text, amounts in [(FLAT + WORDS + '\nรวมเงินทั้งสิ้น Total Amount 100 00', {}),
                              (FLAT + WORDS, {'subtotal': '25,758 01'})]:
            with self.subTest(text=text):
                self.assertNotIn('amountExcludeVat', userjson.convert(source(text, **amounts))['data']['documents'][0])

    def test_other_document_pages_and_overlapping_documents_cannot_supply_evidence(self):
        saved = source('ไม่มีหลักฐานในหน้านี้')
        saved['result']['pages'].append(FLAT + WORDS)
        self.assertNotIn('amountExcludeVat', userjson.convert(saved)['data']['documents'][0])
        first = source(FLAT + WORDS)['result']['extracted']
        second = copy.deepcopy(first)
        for pages in ([1], [2]):
            second['pages'] = pages
            saved = source(FLAT + WORDS)
            saved['result']['pages'].append('ไม่มีหลักฐาน')
            saved['result']['extracted'] = {'documents': [first, second]}
            docs = userjson.convert(saved)['data']['documents']
            self.assertNotIn('amountExcludeVat', docs[1])
            self.assertEqual('amountExcludeVat' in docs[0], pages == [2])

    def test_generic_numbers_identifiers_and_existing_decimal_values_do_not_change(self):
        saved = source(FLAT + WORDS, subtotal='25757.01', vat_total='1802.99', amount_incl_vat='27560.00',
                       document_number='25,757 01', buyer_code='1,802 99')
        saved['result']['extracted']['item_table']['rows'][0][3] = '2 01'
        report = userjson.convert(saved)
        doc = report['data']['documents'][0]
        self.assertEqual(doc['documentNo'], '25,757 01')
        self.assertEqual(doc['buyer']['code'], '1,802 99')
        self.assertIsNone(doc['items'][0]['qty'])
        self.assertEqual(doc['amountExcludeVat'], 25757.01)
        self.assertFalse(any(m.get('normalization') for m in report['mappings']))


if __name__ == '__main__':
    unittest.main()
