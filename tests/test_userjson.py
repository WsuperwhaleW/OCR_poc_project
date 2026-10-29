"""User JSON is an auditable projection, never a second extraction run."""
import builtins
import copy
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

import userjson


def job(doc=None):
    return {'id': 'demo', 'filename': 'mixed.pdf', 'status': 'completed', 'result': {
        'pages_total': 3, 'page_count': 2, 'extracted': doc or {
            'doc_types': ['INVOICE'], 'pages': [1, 2], 'fields': {
                'document_number': 'INV-001', 'issue_date': '01/10/2569',
                'seller_tax_id': '0-1055-48000-01-1', 'seller_branch': 'สำนักงานใหญ่',
                'seller_branch_location': 'สำนักงานใหญ่', 'buyer_branch': '00016',
                'buyer_branch_location': 'มหาชัย', 'buyer_tax_id': '0105559000001',
                'subtotal': '1,000.00', 'vat_total': '0', 'amount_incl_vat': '1,000.00'},
            'item_table': {'columns': ['ลำดับ', 'รหัสสินค้า', 'รายการ', 'จำนวน', 'ราคาต่อหน่วย', 'จำนวนเงิน'],
                           'rows': [['01', 'SKU-1', 'เก้าอี้', '2', '500.00', '1,000.00']],
                           'is_master_table': False}}}}


class UserJsonTests(unittest.TestCase):
    def test_maps_to_v7_validates_and_preserves_zero_leading_digits_and_source(self):
        source = job(); before = copy.deepcopy(source)
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No model/network')):
            report = userjson.convert(source)
        data = report['data']; doc = data['documents'][0]
        self.assertTrue(report['schema_valid'], report['schema_errors'])
        Draft202012Validator(userjson.schema()).validate(data)
        self.assertEqual(data['file']['pageCount'], 3)
        self.assertEqual(doc['documentType'], {'code': '380', 'name': 'INVOICE'})
        self.assertEqual(doc['documentDate'], '2026-10-01')
        self.assertEqual(doc['seller']['taxId'], '0105548000011')
        self.assertEqual(doc['seller']['branchCode'], '00000')
        self.assertEqual(doc['buyer']['branchCode'], '00016')
        self.assertEqual(doc['vatAmount'], 0)
        self.assertEqual(doc['items'][0]['itemNo'], '01')
        self.assertEqual(doc['items'][0]['qty'], 2)
        self.assertEqual(doc['items'][0]['itemName'], 'เก้าอี้')
        self.assertNotIn('branchLocation', doc['buyer'])
        self.assertEqual(report['unmapped'][0]['source']['fields']['buyer_branch_location'], 'มหาชัย')
        self.assertNotIn('netAmount', doc)
        self.assertIn('documents[0].seller.name', report['missing_fields'])
        self.assertEqual(source, before)
        json.dumps(data, allow_nan=False)
        source['result']['extracted']['fields']['issue_date']='2569-10-01'
        self.assertEqual(userjson.convert(source)['data']['documents'][0]['documentDate'],'2026-10-01')

    def test_bilingual_columns_export_goods_without_footer_rows_or_invented_item_numbers(self):
        source = job()
        table = {
            'columns': ['รหัสสินค้า Material Code', 'รายละเอียด Material Description', 'หน่วย Unit',
                        'จำนวน Quantity', 'ราคาต่อหน่วย Unit Price', 'ส่วนลด Discount', 'จำนวนเงิน Amount'],
            'rows': [['210017913', 'PTCG Booster Pack MA4 (55)', 'BOX', '360', '824.30', '0.00', '296,748.00'],
                     ['ส่วนลดท้ายบิล Total Discount', '', '', '', '', '', '0.00'],
                     ['จำนวนเงินก่อนภาษีมูลค่าเพิ่ม', '', '', '', '', '', '296,748.00']],
            'is_master_table': False}
        source['result']['extracted']['item_table'] = table
        before = copy.deepcopy(source)
        report = userjson.convert(source)
        doc = report['data']['documents'][0]
        self.assertEqual(doc['itemType'], 'ITEM')
        self.assertEqual(doc['items'], [{'itemNo': '', 'itemCode': '210017913',
                         'itemName': 'PTCG Booster Pack MA4 (55)', 'unit': 'BOX',
                         'qty': 360.0, 'price': 824.3, 'amount': 296748.0}])
        self.assertTrue(report['schema_valid'], report['schema_errors'])
        self.assertTrue(any(w['path'].endswith('itemNo') for w in report['warnings']))
        self.assertEqual(report['unmapped'][0]['source']['summary_rows'], table['rows'][1:])
        self.assertEqual(source, before)

    def test_bilingual_aliases_require_whole_labels_and_unambiguous_columns(self):
        self.assertEqual(userjson._columns(['Material Code / รหัสสินค้า', 'Quantity\nจำนวน',
                                           'ราคาต่อหน่วย (Unit Price)', 'จำนวนเงิน Amount']),
                         {'itemCode': 0, 'qty': 1, 'price': 2, 'amount': 3})
        self.assertEqual(userjson._columns(['รหัสสินค้า Quantity', 'Amount including VAT',
                                           'item code', 'product code']), {})

    def test_type_combinations_and_scope_false_are_resolved_without_filtering_t03(self):
        for types, code in ((['RECEIPT','TAX_INVOICE'], 'T03'), (['INVOICE','TAX_INVOICE'], 'T02'),
                            (['CREDIT_NOTE','TAX_INVOICE'], '81'), (['DEBIT_NOTE'], '80')):
            with self.subTest(types=types):
                source=job();source['result']['extracted']['doc_types']=types
                self.assertEqual(userjson.convert(source)['data']['documents'][0]['documentType']['code'],code)
        source=job();source['result']['extracted'].update(doc_types=['TAX_INVOICE'],doc_type_heading='ใบกำกับภาษีอย่างย่อ')
        self.assertEqual(userjson.convert(source)['data']['documents'][0]['documentType']['code'],'T05')

    def test_rtv_is_not_relabelled_as_credit_note_and_is_explicitly_a_nonconforming_preview(self):
        source=job();source['result']['extracted']['doc_types']=['RTV_SHIP_CREDIT','TAX_INVOICE']
        report=userjson.convert(source)
        self.assertFalse(report['schema_valid'])
        self.assertIn('RTV_SHIP_CREDIT',report['data']['documents'][0]['documentType']['name'])
        self.assertNotEqual(report['data']['documents'][0]['documentType']['code'],'81')
        self.assertTrue(report['warnings'])

    def test_bad_money_dates_and_place_only_branch_are_not_filled_with_defaults(self):
        source=job();source['result']['extracted']['fields'].update(
            subtotal='-',vat_total='NaN',amount_incl_vat='1,23',seller_branch='มหาชัย',issue_date='2026-13-31')
        report=userjson.convert(source);doc=report['data']['documents'][0]
        self.assertNotIn('amountExcludeVat',doc);self.assertNotIn('vatAmount',doc)
        self.assertNotIn('amountIncludeVat',doc);self.assertNotIn('branchCode',doc['seller'])
        self.assertEqual(doc['documentDate'],'2026-13-31')
        self.assertTrue(any(w['path'].endswith('documentDate') for w in report['warnings']))
        self.assertEqual(report['unmapped'][0]['source']['fields']['subtotal'],'-')

    def test_references_keep_unknown_types_as_other_and_do_not_invent_dates_or_relations(self):
        source=job();source['result']['extracted']['fields'].update(
            po_gr_rtv_number='PO-001; GR-002; UNKNOWN-3',inv_rtv_cnr_number='INV-004; PO-001')
        report=userjson.convert(source);refs=report['data']['documents'][0]['references']
        self.assertEqual([r['type'] for r in refs],['PO','GR','OTHER','INVOICE'])
        self.assertEqual(refs[2]['code'],'UNKNOWN-3')
        self.assertTrue(all(set(r)=={'type','code'} for r in refs))

    def test_an_untyped_reference_table_is_not_assumed_to_be_invoice_items(self):
        source=job();source['result']['extracted']['item_table']={
            'columns':['เลขที่เอกสาร','จำนวนเงิน'],'rows':[['GR-1','50']], 'is_master_table':True}
        report=userjson.convert(source);doc=report['data']['documents'][0]
        self.assertEqual(doc['itemType'],'');self.assertEqual(doc['items'],[])
        self.assertFalse(report['schema_valid'])
        self.assertEqual(report['unmapped'][0]['source']['item_table']['rows'],[['GR-1','50']])

    def test_explicit_ocr_labels_supplement_user_fields_and_resolve_alphanumeric_po(self):
        source = job(); source['result']['extracted']['pages'] = [1]
        source['result']['extracted']['fields']['po_gr_rtv_number'] = 'POB260300015'
        text = ('<table><tr><td>รหัสลูกค้า Cust Code</td><td>พนักงานขาย Salesman</td><td>เลขที่ใบสั่งซื้อ PO. No.</td></tr>'
                '<tr><td>00189</td><td>สมชาย</td><td>POB260300015</td></tr></table>'
                '<table><tr><td>ชื่อลูกค้า Customer name</td><td>บริษัท ลูกค้า จำกัด</td></tr>'
                '<tr><td>ส่วนลดท้ายบิล Total Discount</td><td>0.00</td></tr></table>')
        source['result']['pages'] = [text]
        before = copy.deepcopy(source)
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No inference')):
            report = userjson.convert(source)
        doc = report['data']['documents'][0]
        self.assertEqual(doc['references'], [{'type': 'PO', 'code': 'POB260300015'}])
        self.assertEqual(doc['buyer']['code'], '00189')
        self.assertEqual(doc['buyer']['name'], 'บริษัท ลูกค้า จำกัด')
        self.assertEqual(doc['salesman'], 'สมชาย'); self.assertEqual(doc['discount'], 0)
        self.assertNotIn('name', doc['seller'])
        self.assertTrue(any(m['path'].endswith('buyer.name') and m['source'].startswith('ocr.label:')
                            for m in report['mappings']))
        self.assertEqual(source, before)

    def test_unknown_po_and_conflicting_labels_are_not_guessed_or_allowed_to_overwrite_fields(self):
        source = job(); source['result']['extracted']['pages'] = [1]
        fields = source['result']['extracted']['fields']
        fields.update(po_gr_rtv_number='POB123', seller_name='Saved seller')
        source['result']['pages'] = ['Seller name: OCR seller\nBuyer name: One\nBuyer name: Two\nCompany unlabelled']
        report = userjson.convert(source); doc = report['data']['documents'][0]
        self.assertEqual(doc['seller']['name'], 'Saved seller')
        self.assertNotIn('name', doc['buyer'])
        self.assertEqual(doc['references'][0]['type'], 'OTHER')
        self.assertTrue(any('หลายค่า' in w['message'] for w in report['warnings']))

    def test_ocr_supplements_are_scoped_to_disjoint_document_pages(self):
        first = job()['result']['extracted']; first['pages'] = [1]
        second = copy.deepcopy(first); second['pages'] = [2]
        source = job({'documents': [first, second]})
        source['result']['pages'] = ['Buyer name: First', 'Buyer name: Second']
        docs = userjson.convert(source)['data']['documents']
        self.assertEqual([d['buyer']['name'] for d in docs], ['First', 'Second'])
        second['pages'] = [1]
        docs = userjson.convert(source)['data']['documents']
        self.assertTrue(all('name' not in d['buyer'] for d in docs))
        second.pop('pages')
        docs = userjson.convert(source)['data']['documents']
        self.assertTrue(all('name' not in d['buyer'] for d in docs))

    def test_replacement_reference_requires_explicit_relation_and_deduplicates_bilingual_text(self):
        source = job(); source['result']['extracted']['pages'] = [1]
        source['result']['pages'] = ['เลขที่อ้างอิง IV999\n'
            'เป็นการยกเลิกและออกฉบับใหม่แทนฉบับเดิม เลขที่ IV2600000001363 วันที่ 17.03.2026\n'
            'This is to cancel and issue a new Invoice/Tax Invoice to replace the cancelled '
            'electronic Invoice/Tax Invoice no. IV2600000001363 dated 17.03.2026']
        doc = userjson.convert(source)['data']['documents'][0]
        self.assertEqual(doc['references'], [{'type': 'INVOICE', 'code': 'IV2600000001363', 'relationType': 'REPLACES'}])

    def test_review_readiness_is_separate_from_schema_and_optional_gaps(self):
        source = job(); report = userjson.convert(source)
        self.assertTrue(report['schema_valid']); self.assertTrue(report['missing_fields'])
        self.assertEqual(report['review_status'], 'ready_for_review')
        source['result']['extracted']['item_table']['rows'][0][0] = ''
        report = userjson.convert(source)
        self.assertTrue(report['schema_valid']); self.assertEqual(report['review_status'], 'needs_review')
        self.assertEqual(report['counts']['schema_errors'], 0)
        self.assertEqual(report['counts']['warnings'], 1)
        source['result']['extracted']['validation'] = {'checks': [
            {'field': 'subtotal', 'rule': 'total', 'state': 'failed', 'detail': 'mismatch'}]}
        source['result']['warnings'] = ['Page one may be incomplete']
        report = userjson.convert(source)
        self.assertEqual(len(report['data_issues']), 2)
        self.assertEqual(report['review_status'], 'needs_review')
        self.assertEqual(report['counts']['warnings'], 3)

    def test_invoice_table_has_explicit_header_and_misaligned_rows_are_preserved_for_review(self):
        source=job();source['result']['extracted']['item_table']={
            'columns':['No.','Invoice no.','Amount'], 'rows':[['1','INV-1','100'],['2','INV-2']], 'is_master_table':True}
        report=userjson.convert(source);doc=report['data']['documents'][0]
        self.assertEqual(doc['itemType'],'INVOICE');self.assertEqual(doc['items'][0]['itemCode'],'INV-1')
        self.assertEqual(len(doc['items']),1)
        self.assertIn(['2','INV-2'],report['unmapped'][0]['source']['table_rows'])

    def test_unparseable_table_number_and_missing_printed_item_no_are_not_discarded_or_invented(self):
        source=job();source['result']['extracted']['item_table']={
            'columns':['Item code','Qty'], 'rows':[['SKU-1','many']], 'is_master_table':False}
        report=userjson.convert(source);item=report['data']['documents'][0]['items'][0]
        self.assertEqual(item['itemNo'],'');self.assertIsNone(item['qty'])
        self.assertIn(['SKU-1','many'],report['unmapped'][0]['source']['table_rows'])
        self.assertTrue(any(w['path'].endswith('itemNo') for w in report['warnings']))

    def test_multiple_documents_keep_their_own_fields_pages_and_no_huge_page_range(self):
        first=job()['result']['extracted'];second=copy.deepcopy(first)
        second.update(doc_types=['CREDIT_NOTE'],pages=[2,10**12]);second['fields']['document_number']='CN-2'
        source=job({'documents':[first,second]});report=userjson.convert(source)
        self.assertEqual(len(report['data']['documents']),2)
        self.assertEqual(report['data']['documents'][1]['documentNo'],'CN-2')
        self.assertNotIn('pageTo',report['data']['documents'][1])
        self.assertEqual(report['data']['documents'][0]['pageTo'],2)

    def test_pending_job_and_validator_not_installed_are_reported_clearly(self):
        pending=job();pending['result'].pop('extracted')
        self.assertFalse(userjson.convert(pending)['schema_valid'])
        original=builtins.__import__
        def no_validator(name,*args,**kwargs):
            if name=='jsonschema':raise ImportError('not installed')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=no_validator):
            report=userjson.convert(job())
        self.assertIsNone(report['schema_valid'])
        self.assertTrue(any('requirements' in w['message'] for w in report['warnings']))


if __name__=='__main__':
    unittest.main()
