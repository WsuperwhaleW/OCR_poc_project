"""Preview adapter from saved extraction to the User v7 contract. No inference.

Unknown/ambiguous values stay absent (or empty when the contract requires them).
The report keeps unmapped data separately and validates the candidate, so a
plausible-looking draft never quietly becomes a claim of schema conformance.
"""
import copy
import datetime
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import normalise
import validate

_DIGITS = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
_TYPES = {'DEBIT_NOTE': '80', 'CREDIT_NOTE': '81', 'INVOICE': '380', 'TAX_INVOICE': '388',
          'RECEIPT': 'T01', 'INVOICE_TAX_INVOICE': 'T02', 'RECEIPT_TAX_INVOICE': 'T03',
          'DELIVERY_TAX_INVOICE': 'T04', 'ABBREVIATED_TAX_INVOICE': 'T05',
          'RECEIPT_ABBREVIATED_TAX_INVOICE': 'T06', 'CANCELLATION_NOTE': 'T07'}
_MONEY = {'subtotal': 'amountExcludeVat', 'vat_total': 'vatAmount', 'amount_incl_vat': 'amountIncludeVat'}


@lru_cache(maxsize=1)
def schema():
    path = Path(__file__).resolve().parent / 'schemas' / 'user_document_bundle_v7.schema.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _text(value):
    return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ''


def _number(value):
    if isinstance(value, bool):
        return None
    text = _text(value).translate(_DIGITS)
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1].strip()
    if not re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', text):
        return None
    result = float(text.replace(',', ''))
    return result if math.isfinite(result) else None


def _date(value):
    text = _text(value).translate(_DIGITS)
    try:
        parsed = datetime.date.fromisoformat(text)
        if parsed.year > 2400:
            parsed = parsed.replace(year=parsed.year - 543)
        return parsed.isoformat()
    except ValueError:
        parsed = validate.parse_date(text)
        return parsed.isoformat() if parsed else None


def _heading(value):
    return re.sub(r'[\W_]+', '', str(value or '').lower())


# Exact header aliases, not positions or guesses from cell values. A generic
# "code"/"amount" in an unclassified table cannot make it an invoice table.
_COLUMNS = {
    'itemNo': ('no', 'item no', 'item_no', 'ลำดับ', 'ลำดับที่'),
    'itemCode': ('item code', 'item_code', 'product code', 'sku', 'รหัสสินค้า'),
    'invoiceCode': ('invoice no', 'invoice number', 'inv no', 'เลขที่ใบแจ้งหนี้', 'เลขที่ใบกำกับภาษี'),
    'itemName': ('item name', 'item_name', 'description', 'รายละเอียด', 'รายการ', 'ชื่อสินค้า'),
    'qty': ('qty', 'quantity', 'จำนวน', 'จำนวนสินค้า'),
    'unit': ('unit', 'หน่วย'), 'price': ('price', 'unit price', 'unit_price', 'ราคาต่อหน่วย', 'หน่วยละ'),
    'amount': ('amount', 'line total', 'จำนวนเงิน', 'มูลค่า', 'ราคารวม'),
    'vatRate': ('vat rate', 'vat_rate', 'อัตราภาษี', 'อัตราภาษีมูลค่าเพิ่ม'),
    'termDescription': ('term description', 'term_description', 'payment terms', 'เงื่อนไขการชำระเงิน'),
    'remarks': ('remarks', 'remark', 'หมายเหตุ'),
}


def _columns(headers):
    found = {}
    for key, aliases in _COLUMNS.items():
        matches = [i for i, header in enumerate(headers) if _heading(header) in {_heading(a) for a in aliases}]
        if len(matches) == 1:
            found[key] = matches[0]
    return found


def _type(doc, fields, issue, path):
    codes = doc.get('doc_types') or ([doc['doc_type']] if doc.get('doc_type') else [])
    if isinstance(codes, str):
        codes = [codes]
    codes = set(codes or normalise.document_types(fields.get('document_type', '')))
    title = _text(doc.get('doc_type_heading') or fields.get('document_type')).lower()
    # A non-contract type must not disappear just because it co-occurs with a
    # generic tax-invoice heading. In particular, never relabel RTV as CREDIT_NOTE.
    unsupported = codes - set(_TYPES)
    if unsupported:
        name = ' + '.join(sorted(codes))
        issue(path, 'ประเภทเดิมอยู่นอก schema User: ' + name)
        return {'code': '', 'name': name}
    if 'CREDIT_NOTE' in codes:
        name = 'CREDIT_NOTE'
    elif 'DEBIT_NOTE' in codes:
        name = 'DEBIT_NOTE'
    elif 'TAX_INVOICE' in codes and ('อย่างย่อ' in title or 'abbreviated' in title):
        name = 'RECEIPT_ABBREVIATED_TAX_INVOICE' if 'RECEIPT' in codes else 'ABBREVIATED_TAX_INVOICE'
    elif 'TAX_INVOICE' in codes and ('ใบส่งของ' in title or 'delivery' in title):
        name = 'DELIVERY_TAX_INVOICE'
    elif {'RECEIPT', 'TAX_INVOICE'} <= codes:
        name = 'RECEIPT_TAX_INVOICE'
    elif {'INVOICE', 'TAX_INVOICE'} <= codes:
        name = 'INVOICE_TAX_INVOICE'
    elif len(codes) == 1:
        name = next(iter(codes))
    else:
        name = 'UNKNOWN'
        issue(path, 'ยังระบุประเภทเอกสารปลายทางไม่ได้อย่างชัดเจน')
    return {'code': _TYPES.get(name, ''), 'name': name}


def _items(doc, fields, issue, path, unmapped):
    lines = fields.get('line_items')
    table = doc.get('item_table') or {}
    if lines and not (isinstance(lines, list) and all(isinstance(row, dict) for row in lines)):
        unmapped['line_items'] = copy.deepcopy(lines)
        issue(path + '.items', 'line_items ไม่ใช่รายการ object ที่แปลงได้ จึงเก็บต้นฉบับแยกไว้')
    if isinstance(lines, list) and lines and all(isinstance(row, dict) for row in lines):
        headers = list(dict.fromkeys(key for row in lines for key in row))
        rows = [[row.get(key) for key in headers] for row in lines]
        kind = 'ITEM'
    elif isinstance(table, dict) and (table.get('columns') or table.get('rows')):
        headers, rows = table.get('columns') or [], table.get('rows') or []
        cols = _columns(headers)
        if 'itemCode' in cols or ('itemName' in cols and ('qty' in cols or 'price' in cols)):
            kind = 'ITEM'
        elif 'invoiceCode' in cols and table.get('is_master_table') is True:
            kind = 'INVOICE'
        else:
            unmapped['item_table'] = copy.deepcopy(table)
            issue(path + '.itemType', 'มีตารางดิบ แต่ยังยืนยัน ITEM / INVOICE จากหัวคอลัมน์ไม่ได้')
            return '', []
    else:
        issue(path + '.itemType', 'ยังไม่มีตารางที่ระบุชนิดรายการได้; ไม่กำหนด ITEM แทนข้อมูลที่ขาด')
        return '', []
    cols = _columns(headers)
    code_key = 'invoiceCode' if kind == 'INVOICE' else 'itemCode'
    output = []
    for i, row in enumerate(rows):
        prefix = path + f'.items[{i}]'
        if not isinstance(row, list) or len(row) != len(headers):
            issue(prefix, 'แถวไม่ตรงกับจำนวนหัวคอลัมน์ จึงไม่เลื่อนค่าไปจับฟิลด์เอง')
            unmapped.setdefault('table_rows', []).append(copy.deepcopy(row))
            continue
        def cell(key):
            return row[cols[key]] if key in cols else None
        item = {'itemNo': _text(cell('itemNo')), 'itemCode': _text(cell(code_key))}
        for key in ('itemNo', 'itemCode'):
            if not item[key]:
                issue(prefix + '.' + key, 'ไม่พบค่าที่ระบุฟิลด์นี้; ไม่สร้างเลขขึ้นเอง')
        for key in ('itemName', 'unit', 'termDescription'):
            if key in cols:
                item[key] = _text(cell(key)) or None
        invalid_number = False
        for key in ('qty', 'price', 'amount', 'vatRate'):
            if key in cols:
                raw = cell(key)
                item[key] = _number(raw)
                if _text(raw) and item[key] is None:
                    invalid_number = True
                    issue(prefix + '.' + key, 'ค่าต้นฉบับแปลงเป็น number ไม่ได้')
        if invalid_number:
            unmapped.setdefault('table_rows', []).append(copy.deepcopy(row))
        if 'remarks' in cols and _text(cell('remarks')):
            item['remarks'] = [_text(cell('remarks'))]
        output.append(item)
    used = {index for key, index in cols.items() if key != ('itemCode' if kind == 'INVOICE' else 'invoiceCode')}
    if used != set(range(len(headers))) or any(row for row in output if not row['itemCode'] or not row['itemNo']):
        unmapped['item_table' if not lines else 'line_items'] = copy.deepcopy(table if not lines else lines)
    return kind, output


def convert(job):
    """Return candidate, conversion warnings, unmapped originals and schema errors."""
    result = job.get('result') or {}
    extracted = result.get('extracted') or {}
    warnings, unmapped, missing = [], [], []
    def issue(path, message):
        warnings.append({'path': path, 'message': message})
    file = {'fileName': job.get('filename', ''), 'fileType': Path(job.get('filename', '')).suffix.lstrip('.').upper()}
    count = result.get('pages_total') or result.get('page_count') or len(result.get('pages') or [])
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        file['pageCount'] = count
    data = {'schemaVersion': '7.0', 'file': file, 'documents': []}
    docs = extracted.get('documents', [extracted]) if isinstance(extracted, dict) and extracted else []
    if not isinstance(docs, list):
        issue('documents', 'รายการเอกสารต้นฉบับไม่ใช่ array')
        docs = []
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            issue(f'documents[{index}]', 'ผลเอกสารไม่ใช่ object')
            continue
        path = f'documents[{index}]'
        fields = doc.get('fields') or {}
        if not isinstance(fields, dict):
            issue(path, 'fields ต้นฉบับไม่ใช่ object')
            fields = {}
        used, extra = set(), {}
        item = {'documentType': _type(doc, fields, issue, path + '.documentType'),
                'documentNo': _text(fields.get('document_number'))}
        used.update(('document_number', 'document_type'))
        if not item['documentNo']:
            issue(path + '.documentNo', 'ยังไม่มีเลขที่เอกสาร')
        pages = doc.get('pages') or []
        if pages and all(isinstance(page, int) and not isinstance(page, bool) and page > 0 for page in pages):
            ordered = sorted(set(pages))
            if len(ordered) == ordered[-1] - ordered[0] + 1 and (not count or ordered[-1] <= count):
                item.update(pageFrom=ordered[0], pageTo=ordered[-1])
            else:
                issue(path + '.pageFrom', 'เลขหน้าไม่ต่อเนื่องหรือเกินจำนวนหน้าต้นฉบับ จึงไม่สร้างช่วงหน้า')
                extra['pages'] = copy.deepcopy(pages)
        if _text(fields.get('issue_date')):
            parsed = _date(fields['issue_date'])
            item['documentDate'] = parsed or _text(fields['issue_date']); used.add('issue_date')
            if not parsed:
                issue(path + '.documentDate', 'วันที่แปลงเป็น ISO ไม่ได้ จึงคงข้อความเดิม')
        for party in ('seller', 'buyer'):
            values = {}
            for source, target in (('name', 'name'), ('tax_id', 'taxId'), ('branch', 'branchCode')):
                key = party + '_' + source
                raw = _text(fields.get(key))
                if not raw:
                    continue
                if source == 'tax_id':
                    digits = normalise.tax_id_digits(raw)
                    values[target] = digits if len(digits) == 13 else raw
                    if len(digits) != 13:
                        issue(path + '.' + party + '.taxId', 'เลขผู้เสียภาษีไม่ครบ 13 หลัก จึงไม่เติมเลขเอง')
                elif source == 'branch':
                    code = normalise.branch_code(raw)
                    if not code:
                        issue(path + '.' + party + '.branchCode', 'มีข้อความสาขา แต่ยังแปลงเป็นรหัสสาขาไม่ได้')
                        continue
                    values[target] = code
                else:
                    values[target] = raw
                used.add(key)
            if values:
                item[party] = values
        for source, target in _MONEY.items():
            raw = fields.get(source)
            if not _text(raw):
                continue
            amount = _number(raw)
            if amount is None:
                issue(path + '.' + target, 'จำนวนเงินแปลงเป็น number ไม่ได้ จึงไม่ใส่ 0 แทน')
            else:
                item[target] = amount; used.add(source)
        refs = []
        for key in ('po_gr_rtv_number', 'inv_rtv_cnr_number'):
            raw = _text(fields.get(key))
            if not raw:
                continue
            for code in re.split(r'[\n;,]+', raw):
                code = code.strip()
                if not code:
                    continue
                match = re.match(r'^(INVOICE|INV|RTV|PO|GR|CN)(?=[\d\s:._/-])', code, re.I)
                kind = {'INV': 'INVOICE', 'CN': 'CREDIT_NOTE'}.get(match[1].upper(), match[1].upper()) if match else 'OTHER'
                if kind == 'OTHER':
                    issue(path + '.references', 'ไม่ทราบชนิด reference: ' + code + '; เก็บเป็น OTHER')
                ref = {'type': kind, 'code': code}
                if ref not in refs:
                    refs.append(ref)
            used.add(key)
        if refs:
            item['references'] = refs
        item['itemType'], item['items'] = _items(doc, fields, issue, path, extra)
        if fields.get('line_items'):
            used.add('line_items')
        extra['fields'] = {key: copy.deepcopy(value) for key, value in fields.items()
                           if key not in used and value not in ('', None, [], {})}
        if not extra['fields']:
            del extra['fields']
        for key in ('doc_types', 'doc_type', 'error', 'steps_failed', 'steps_extras'):
            if doc.get(key):
                extra[key] = copy.deepcopy(doc[key])
        if extra:
            unmapped.append({'document': index + 1, 'source': extra})
        # Optional does not mean available: list gaps without inventing zeros,
        # names or references merely to make the preview look filled in.
        for key in ('seller.code', 'seller.name', 'buyer.code', 'buyer.name', 'salesman',
                    'discount', 'extraCharge', 'netAmount', 'taxBasisTotalAmount'):
            value = item
            for part in key.split('.'):
                value = value.get(part, {}) if isinstance(value, dict) else {}
            if value == {}:
                missing.append(path + '.' + key)
        data['documents'].append(item)
    if not docs:
        issue('documents', 'ยังไม่มีผล Extraction ให้แปลง')
    schema_errors, valid = [], None
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        issue('', 'ยังตรวจ schema ไม่ได้: โปรดติดตั้ง requirements.txt ที่อัปเดตแล้ว (jsonschema)')
    else:
        for error in Draft202012Validator(schema()).iter_errors(data):
            schema_errors.append({'path': '/'.join(map(str, error.absolute_path)), 'message': error.message})
        valid = not schema_errors
    return {'data': data, 'schema_valid': valid, 'schema_errors': schema_errors,
            'warnings': warnings, 'missing_fields': missing, 'unmapped': unmapped}
