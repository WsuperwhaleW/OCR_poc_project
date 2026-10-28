"""Read-only comparison with the User's v7 contract; never used for extraction.

The two schemas/ snapshots came from RunLog d3afa0b. Keep document classification,
prompt requirements and exported results independent of this reference mapping.
"""
import json
from collections import Counter
from pathlib import Path

import prompts

ROOT = Path(__file__).resolve().parent / 'schemas'

# Candidate correspondences, not automatic classification/export rules.
TYPE_MAP = {
    '80': ('mapped', ('DEBIT_NOTE',), 'มีประเภทเดิม แต่ยังไม่มีข้อกำหนดฟิลด์เฉพาะประเภท'),
    '81': ('mapped', ('CREDIT_NOTE',), 'ตรงกับใบลดหนี้เดิม'),
    '380': ('mapped', ('INVOICE',), 'ตรงกับใบแจ้งหนี้เดิม'),
    '388': ('mapped', ('TAX_INVOICE',), 'มีประเภทเดิม แต่ยังไม่มีข้อกำหนดฟิลด์เฉพาะประเภท'),
    'T01': ('mapped', ('RECEIPT',), 'ตรงกับใบเสร็จรับเงินเดิม'),
    'T02': ('combined', ('INVOICE', 'TAX_INVOICE'), 'ระบบเดิมเก็บหลายประเภทในเอกสารเดียว ยังไม่มีรหัส T02 ในผลลัพธ์'),
    'T03': ('combined', ('RECEIPT', 'TAX_INVOICE'), 'ระบบเดิมเก็บหลายประเภทในเอกสารเดียว คง userScope: false ตาม catalog; ไม่ปิดกั้นการใช้งาน'),
    'T04': ('partial', ('TAX_INVOICE',), 'ขาดประเภทใบส่งของที่แยกชัดเจน จึงยังยืนยัน T04 จาก TAX_INVOICE อย่างเดียวไม่ได้'),
    'T05': ('partial', ('TAX_INVOICE',), 'ยังไม่แยกใบกำกับภาษีอย่างย่อออกจากใบกำกับภาษีทั่วไป'),
    'T06': ('partial', ('RECEIPT', 'TAX_INVOICE'), 'ยังไม่แยกใบกำกับภาษีอย่างย่อ จึงยังยืนยัน T06 ไม่ได้'),
    'T07': ('missing', (), 'ยังไม่มีประเภทใบแจ้งยกเลิก'),
}

# Paths are from the contract; source names are from today's actual engine.
FIELD_MAP = {
    'schemaVersion': ('system', (), 'ค่าคงที่ 7.0 ของ User JSON ไม่ต้องถามโมเดล'),
    'file.fileName': ('system', ('job.filename',), 'มีชื่อไฟล์จากงานที่อัปโหลด'),
    'file.fileType': ('system', ('job.filename',), 'สร้างชนิดไฟล์จากข้อมูลไฟล์ ไม่ต้องถามโมเดล'),
    'file.pageCount': ('system', ('result.pages_total', 'result.page_count'), 'มีจำนวนหน้าต้นฉบับและจำนวนหน้าที่ประมวลผลจากขั้น OCR; ใช้ pages_total สำหรับทั้งไฟล์'),
    'documents[].pageFrom': ('partial', ('pages',), 'มีรายการเลขหน้า ต้องแปลงเป็นช่วงและตรวจว่าหน้าต่อเนื่อง'),
    'documents[].pageTo': ('partial', ('pages',), 'มีรายการเลขหน้า ต้องแปลงเป็นช่วงและตรวจว่าหน้าต่อเนื่อง'),
    'documents[].documentType.code': ('partial', ('doc_types',), 'ต้องใช้ตารางประเภทด้านบน; ประเภทรวม/อย่างย่อยังมีข้อจำกัด'),
    'documents[].documentType.name': ('partial', ('doc_types',), 'User JSON ทดลองแปลงเป็น enum ของ User; ประเภทที่ยังไม่รองรับแสดงเป็น Draft'),
    'documents[].documentNo': ('mapped', ('document_number',), 'เลขที่เอกสารของฉบับนี้'),
    'documents[].documentDate': ('partial', ('issue_date',), 'มีวันที่ แต่ยังต้องแปลงรูปแบบวันที่เป็นค่าตามข้อตกลง User'),
    'documents[].seller.taxId': ('mapped', ('seller_tax_id',), 'เลขผู้เสียภาษีผู้ขาย'),
    'documents[].buyer.taxId': ('mapped', ('buyer_tax_id',), 'เลขผู้เสียภาษีผู้ซื้อ'),
    'documents[].seller.branchCode': ('partial', ('seller_branch',), 'มีสาขาเดิม ต้องทำให้เป็นรหัสตัวเลขตามปลายทาง; ไม่ย้ายชื่อสถานที่มาปน'),
    'documents[].buyer.branchCode': ('partial', ('buyer_branch',), 'มีสาขาเดิม ต้องทำให้เป็นรหัสตัวเลขตามปลายทาง; ไม่ย้ายชื่อสถานที่มาปน'),
    'documents[].references[].type': ('partial', ('po_gr_rtv_number', 'inv_rtv_cnr_number'), 'ยังไม่ได้แยกชนิด GR / RTV / PO / INVOICE ต่อ reference'),
    'documents[].references[].code': ('partial', ('po_gr_rtv_number', 'inv_rtv_cnr_number'), 'มีเลขอ้างอิงแบบรวม ต้องแยกเป็นแต่ละรายการ'),
    'documents[].amountExcludeVat': ('mapped', ('subtotal',), 'ยอดก่อน VAT; ต้องแปลงข้อความเป็น number ก่อนส่งออก'),
    'documents[].vatAmount': ('mapped', ('vat_total',), 'VAT; ต้องแปลงข้อความเป็น number ก่อนส่งออก'),
    'documents[].amountIncludeVat': ('mapped', ('amount_incl_vat',), 'ยอดรวม VAT; ต้องแปลงข้อความเป็น number ก่อนส่งออก'),
    'documents[].itemType': ('partial', ('item_table.is_master_table',), 'มีการวิเคราะห์ตาราง แต่ยังไม่ใช่ enum ITEM / INVOICE ที่ยืนยันได้โดยตรง'),
}
MISSING_NOTES = {
    'seller.code': 'ยังไม่มีรหัสผู้ขายเป็นฟิลด์เฉพาะ',
    'buyer.code': 'ยังไม่มีรหัสผู้ซื้อเป็นฟิลด์เฉพาะ',
    'seller.name': 'seller_name ถูกถอดจากชุดฟิลด์ที่ถามแล้ว; other_fields ไม่รับประกันว่าจะได้ชื่อนี้',
    'buyer.name': 'buyer_name ถูกถอดจากชุดฟิลด์ที่ถามแล้ว; other_fields ไม่รับประกันว่าจะได้ชื่อนี้',
    'netAmount': 'ยังไม่มียอดสุทธิแยกต่างหาก; ไม่ถือว่าเท่ากับยอดรวม VAT อัตโนมัติ',
    'taxBasisTotalAmount': 'ยังไม่มีฐานภาษีแยกเฉพาะ; ไม่ถือว่าเท่ากับ subtotal อัตโนมัติ',
    'references[].date': 'ยังไม่มีวันที่แยกต่อ reference',
    'references[].relationType': 'ยังไม่มีความสัมพันธ์แยกต่อ reference',
}


def _leaves(schema, node=None, prefix=''):
    """Expand contract refs so every declared business leaf gets a row."""
    node = schema if node is None else node
    if '$ref' in node:
        target = schema
        for part in node['$ref'].split('/')[1:]:
            target = target[part]
        yield from _leaves(schema, target, prefix)
    elif node.get('type') == 'object':
        for name, child in node.get('properties', {}).items():
            yield from _leaves(schema, child, prefix + ('.' if prefix else '') + name)
    elif node.get('type') == 'array':
        yield from _leaves(schema, node['items'], prefix + '[]')
    else:
        kind = node.get('type', 'string')
        yield prefix, ' / '.join(kind) if isinstance(kind, list) else kind


def comparison():
    catalog = json.loads((ROOT / 'user_document_type_catalog.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'user_document_bundle_v7.schema.json').read_text(encoding='utf-8'))
    types, fields, used_types, used_fields = [], [], set(), set()
    for item in catalog['documentTypes']:
        status, ours, note = TYPE_MAP.get(item['code'], ('missing', (), 'ยังไม่มี mapping'))
        ours = tuple(code for code in ours if code in prompts.TYPE_NAMES)
        if not ours:
            status = 'missing'
        used_types.update(ours)
        types.append({'user_code': item['code'], 'user_name': item['nameTh'],
                      'user_scope': item['userScope'], 'ours': list(ours), 'status': status, 'note': note})
    for code, name in prompts.TYPE_NAMES.items():
        if code not in used_types:
            types.append({'user_code': '', 'user_name': '', 'ours': [code], 'status': 'extra',
                          'note': ('คง RTV แบบเดิมตามที่ใช้อยู่; User มี RTV เฉพาะใน references'
                                   if code == 'RTV_SHIP_CREDIT' else 'ระบบเรามีเพิ่มจาก catalog ของ User: ' + name)})

    for path, kind in _leaves(contract):
        status, ours, note = FIELD_MAP.get(path, ('missing', (), 'ยังไม่มีฟิลด์เฉพาะที่ตรงกับ User'))
        short = path.removeprefix('documents[].')
        if short.startswith('items[].') and short != 'items[].remarks[]':
            status, ours = 'partial', ('item_table.columns', 'item_table.rows')
            note = 'มีตารางดิบตามหัวคอลัมน์ แต่ยังไม่แยก/ตรวจชนิดค่าเป็น ' + short + ' โดยเฉพาะ'
        if status == 'missing':
            note = MISSING_NOTES.get(short, note)
        scalar_sources = tuple(source for source in ours if '.' not in source and source not in ('pages', 'doc_types'))
        if any(source not in prompts._SCALAR_KEYS for source in scalar_sources):
            status, note = 'missing', 'ฟิลด์ที่เคยเทียบไว้ไม่ได้อยู่ในชุดที่ถามปัจจุบันแล้ว'
            ours = ()
        else:
            used_fields.update(scalar_sources)
        fields.append({'user_field': path, 'value_type': kind, 'ours': list(ours),
                       'status': status, 'note': note})
    for key in prompts._SCALAR_KEYS:
        if key in used_fields:
            continue
        note = ('เรามี branchLocation เพิ่ม: ชื่อ/ตำแหน่งสาขา แยกจากรหัสสาขา'
                if key in ('seller_branch_location', 'buyer_branch_location')
                else 'ข้อความประเภทที่อ่านได้; User ขอรหัส/ชื่อประเภทจาก doc_types แทน'
                if key == 'document_type' else 'เรามีฟิลด์นี้เพิ่มจากโครงสร้าง User v7')
        fields.append({'user_field': '', 'value_type': 'string', 'ours': [key], 'status': 'extra', 'note': note})
    fields.append({'user_field': '', 'value_type': 'array', 'ours': ['other_fields'], 'status': 'extra',
                   'note': 'ข้อมูลเพิ่มเติมตามชื่อบนเอกสาร เมื่อเปิด Extra fields; ไม่ใช่การรับประกันฟิลด์ที่ User ขอ'})
    for key in prompts.INCOME_ITEM_KEYS:
        fields.append({'user_field': '', 'value_type': 'string', 'ours': [prompts.INCOME_ITEMS_KEY + '[].' + key],
                       'status': 'extra', 'note': 'ตารางเงินได้ของ 50 ทวิ คงไว้ตามเดิม; ไม่ใช่ตาราง items ของ User'})
    return {'version': '7.0', 'source': 'RunLog jsonschema · d3afa0b',
            'types': types, 'fields': fields,
            'counts': {name: dict(Counter(row['status'] for row in rows))
                       for name, rows in (('types', types), ('fields', fields))}}
