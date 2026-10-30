"""Literal User kinds are classification additions, not new extraction profiles."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

import app
import normalise
import prompts
import userjson
from tests.test_userjson import job


TITLES = {
    'INVOICE_TAX_INVOICE': ('T02', ('ใบแจ้งหนี้/ใบกำกับภาษี', 'INVOICE / TAX INVOICE',
                                  'ใบกำกับภาษี / ใบแจ้งหนี้', 'ใบแจ้งหนี้\nใบกำกับภาษี')),
    'RECEIPT_TAX_INVOICE': ('T03', ('ใบเสร็จรับเงิน/ใบกำกับภาษี', 'RECEIPT / TAX INVOICE',
                                  'ใบกำกับภาษี\nใบเสร็จรับเงิน')),
    'DELIVERY_TAX_INVOICE': ('T04', ('ใบส่งของ/ใบกำกับภาษี', 'ใบกำกับภาษี/ใบส่งของ',
                                  'INVOICE/TAX INVOICE/DELIVERY', 'DELIVERY ORDER / TAX INVOICE',
                                  'ใบส่งของ\nใบกำกับภาษี')),
    'ABBREVIATED_TAX_INVOICE': ('T05', ('ใบกำกับภาษีอย่างย่อ', 'ABBREVIATED TAX INVOICE')),
    'RECEIPT_ABBREVIATED_TAX_INVOICE': ('T06', ('ใบเสร็จรับเงิน/ใบกำกับภาษีอย่างย่อ',
        'RECEIPT / ABBREVIATED TAX INVOICE', 'ใบเสร็จรับเงิน\nใบกำกับภาษีอย่างย่อ')),
    'CANCELLATION_NOTE': ('T07', ('ใบแจ้งยกเลิก', 'ใบแจ้งการยกเลิก', 'CANCELLATION NOTICE')),
}


class LiteralDocumentTypesTests(unittest.TestCase):
    def test_titles_have_specific_kinds_and_export_exact_user_codes_without_inference(self):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No inference')):
            for kind, (user_code, titles) in TITLES.items():
                for title in titles:
                    with self.subTest(kind=kind, title=title):
                        codes, heading, confidence, _ = app.classify_transcript(title)
                        self.assertIn(kind, codes)
                        self.assertEqual(set(codes) & set(TITLES), {kind})
                        self.assertGreaterEqual(confidence, .9)
                        source = job()
                        source['result']['extracted'].update(doc_types=codes, doc_type_heading=heading)
                        before = copy.deepcopy(source)
                        target = userjson.convert(source)['data']['documents'][0]['documentType']
                        self.assertEqual(target, {'code': user_code, 'name': kind})
                        self.assertEqual(source, before)

    def test_plain_nearby_and_negated_titles_do_not_become_new_kinds(self):
        for title in ('TAX INVOICE', 'ใบกำกับภาษี', 'INVOICE', 'RECEIPT', 'ใบส่งของ',
                      'ใบเสร็จรับเงิน (ไม่ใช่ใบกำกับภาษี)', 'ไม่ใช่ใบแจ้งยกเลิก',
                      'CREDIT NOTE\nอ้างอิงใบกำกับภาษีอย่างย่อ 123',
                      'ใบกำกับภาษี\nออกแทนฉบับเดิมที่ยกเลิก',
                      'TAX INVOICE\nInvoice No: 123',
                      'ใบแจ้งหนี้\nยกเลิกได้ภายใน 7 วัน'):
            with self.subTest(title=title):
                self.assertFalse(set(app.classify_transcript(title)[0]) & set(TITLES))

    def test_classification_does_not_add_tuned_fields_or_mandatory_rules(self):
        tuned = {code for code, fields in prompts.DOC_TYPE_FIELDS.items() if fields}
        self.assertEqual(tuned, {'INVOICE', 'RECEIPT', 'CREDIT_NOTE', 'RTV_SHIP_CREDIT', 'WHT_CERTIFICATE'})
        for kind, (_, titles) in TITLES.items():
            self.assertFalse(prompts.DOC_TYPE_FIELDS.get(kind))
            self.assertFalse(prompts.MANDATORY_FIELDS.get(kind))
            self.assertFalse(prompts.TYPE_RULES.get(kind))
            self.assertFalse(prompts.DOC_TYPE_ITEMS.get(kind))
            codes = app.classify_transcript(titles[0])[0]
            legacy = list(prompts.TYPE_COMPONENTS[kind])
            self.assertEqual(prompts.type_phrase(codes), prompts.TYPE_NAMES[kind])
            self.assertEqual(prompts.fields_for_types(codes), prompts.fields_for_types(legacy))
            self.assertEqual(prompts.mandatory_for_types(codes), prompts.mandatory_for_types(legacy))
            for mode in ('agentic', 'hybrid'):
                self.assertEqual([(s['id'], s['keys']) for s in prompts.steps_for_types(codes, mode=mode)],
                                 [(s['id'], s['keys']) for s in prompts.steps_for_types(legacy, mode=mode)])

    def test_model_specific_kind_requires_matching_printed_title(self):
        def classify(codes, heading, text=None):
            answer = json.dumps({'types': codes, 'heading': heading})
            reply = Mock(status_code=200)
            with patch.object(app, 'CLASSIFY_WITH_MODEL', True), \
                 patch.object(app.backends, 'structured_request', return_value=('local', {})), \
                 patch.object(app.backends, 'structured_reply', return_value=(answer, None, None)), \
                 patch.object(app, '_timed_post', return_value=reply):
                return app._classify_with_model(text or heading, {})[0]
        self.assertEqual(classify(['DELIVERY_TAX_INVOICE'], 'TAX INVOICE'), [])
        self.assertEqual(classify(['CANCELLATION_NOTE'], 'ใบแจ้งหนี้'), [])
        self.assertEqual(classify(['CANCELLATION_NOTE'], 'ใบแจ้งยกเลิก', 'ใบแจ้งหนี้'), [])
        self.assertIn('INVOICE_TAX_INVOICE', classify(['INVOICE', 'TAX_INVOICE'], 'INVOICE/TAX INVOICE'))
        self.assertIn('DELIVERY_TAX_INVOICE', classify(['DELIVERY_TAX_INVOICE'], 'ใบส่งของ/ใบกำกับภาษี'))

    def test_conflicting_specific_kinds_do_not_export_an_arbitrary_user_type(self):
        source = job()
        source['result']['extracted']['doc_types'] = ['INVOICE_TAX_INVOICE', 'RECEIPT_TAX_INVOICE']
        self.assertEqual(userjson.convert(source)['data']['documents'][0]['documentType']['code'], '')


if __name__ == '__main__':
    unittest.main()
