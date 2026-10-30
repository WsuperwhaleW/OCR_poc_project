"""User-only Hybrid field guidance and explicit User JSON destinations.

The built-in engine definitions are immutable. Saved profiles are snapshotted
with uploads, then applied only inside the User worker's workspace.
"""
import copy
import json
import re
from functools import lru_cache
from pathlib import Path

import prompts

CORE_TARGETS = {
    'document_number': 'documentNo', 'issue_date': 'documentDate',
    'seller_tax_id': 'seller.taxId', 'seller_branch': 'seller.branchCode',
    'buyer_tax_id': 'buyer.taxId', 'buyer_branch': 'buyer.branchCode',
    'subtotal': 'amountExcludeVat', 'vat_total': 'vatAmount', 'amount_incl_vat': 'amountIncludeVat',
}
GUIDANCE = {
    'document_type': 'อ่านชื่อประเภทที่พิมพ์เป็นหัวเอกสาร คงข้อความและภาษาเดิม',
    'document_number': 'อ่านเลขที่ของเอกสารฉบับนี้จากหัวข้อเลขที่ ไม่ใช้เลขเอกสารอ้างอิง',
    'issue_date': 'อ่านวันที่ออกเอกสารฉบับนี้ ไม่ใช้วันครบกำหนดหรือวันที่ชำระ',
    'po_gr_rtv_number': 'อ่านเลข PO / GR / RTV ที่เอกสารอ้างอิงตามหัวข้อที่พิมพ์',
    'subtotal': 'อ่านยอดรวมก่อน VAT ของทั้งเอกสาร ไม่คำนวณยอดที่ไม่มีพิมพ์ไว้',
    'vat_total': 'อ่านจำนวนภาษีมูลค่าเพิ่ม ไม่ใช่อัตราร้อยละหรือภาษีหัก ณ ที่จ่าย',
    'amount_incl_vat': 'อ่านยอดรวม VAT ที่พิมพ์ไว้ ไม่คำนวณแทน',
    'income_items': 'อ่านตารางเงินได้ตามคอลัมน์เดิม โดยรักษาการจับคู่ค่าในแต่ละแถว',
}
for party, label in (('seller', 'ผู้ขาย'), ('buyer', 'ผู้ซื้อ'), ('payer', 'ผู้จ่ายเงิน'), ('payee', 'ผู้รับเงิน')):
    for suffix, hint in (('tax_id', 'เลขผู้เสียภาษี'), ('branch', 'รหัสสาขาภาษี'),
                         ('branch_location', 'ชื่อหรือตำแหน่งสาขา'), ('name', 'ชื่อ'), ('address', 'ที่อยู่')):
        GUIDANCE[f'{party}_{suffix}'] = f'อ่าน{hint}ของ{label}จากบล็อกของฝ่ายนั้น ไม่ยืมค่าจากอีกฝ่าย'


@lru_cache(maxsize=1)
def targets():
    contract = json.loads((Path(__file__).parent / 'schemas/user_document_bundle_v7.schema.json').read_text('utf-8'))
    def expand(node):
        if '$ref' in node:
            return expand(contract['$defs'][node['$ref'].rsplit('/', 1)[-1]])
        return {k: expand(v) if isinstance(v, dict) else v for k, v in node.items()}
    properties = contract['$defs']['document']['properties']
    out = {}
    for key, schema in properties.items():
        if key in ('documentType', 'pageFrom', 'pageTo'):
            continue
        if key in ('seller', 'buyer'):
            for leaf, value in contract['$defs']['party']['properties'].items():
                out[key + '.' + leaf] = expand(value)
        else:
            out[key] = expand(schema)
    return out


def base_steps(code):
    if not prompts.DOC_TYPE_FIELDS.get(code):
        raise ValueError('ประเภทนี้ยังไม่มีชุดคำถามเฉพาะให้แก้ไข')
    return prompts.steps_for_types([code], False, mode='hybrid')


def leaf_values(values):
    """Audit/render array answers by cell without treating a whole array as text."""
    out = {}
    def visit(path, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(path + '.' + str(key), child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(path + f'[{index}]', child)
        else:
            out[path] = value
    for key, value in values.items():
        visit(key, value)
    return out


def validate(code, body):
    steps = base_steps(code)
    if not isinstance(body, dict) or set(body) - {'instructions', 'extra_groups'}:
        raise ValueError('แก้ได้เฉพาะคำอธิบายตัวแปรและก้อนคำถามเพิ่มเติม')
    instructions = body.get('instructions', {})
    keys = {key for step in steps for key in step['keys']}
    if not isinstance(instructions, dict) or set(instructions) - keys:
        raise ValueError('มีตัวแปรมาตรฐานที่ไม่อยู่ในประเภทนี้')
    clean = {}
    for key, value in instructions.items():
        if not isinstance(value, str) or len(value) > 4000:
            raise ValueError('คำอธิบายแต่ละตัวแปรต้องเป็นข้อความไม่เกิน 4,000 ตัวอักษร')
        if value.strip():
            clean[key] = value.strip()
    groups = body.get('extra_groups', [])
    if not isinstance(groups, list) or len(groups) > 20:
        raise ValueError('เพิ่มได้ไม่เกิน 20 ก้อนคำถาม')
    reserved = set(prompts._SCALAR_KEYS) | {'income_items', 'line_items', 'other_fields', '__proto__', 'constructor', 'prototype'}
    seen, destinations, ids, normalized = set(), {CORE_TARGETS[k] for k in keys if k in CORE_TARGETS}, set(), []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {'id', 'title', 'fields'}:
            raise ValueError('ก้อนคำถามต้องมี id, title และ fields')
        gid, title, fields = group['id'], group['title'], group['fields']
        if not isinstance(gid, str) or not re.fullmatch(r'extra_[a-zA-Z0-9_]{1,40}', gid) or gid in ids:
            raise ValueError('รหัสก้อนคำถามไม่ถูกต้องหรือซ้ำ')
        ids.add(gid)
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
            raise ValueError('ตั้งชื่อก้อนคำถาม 1–120 ตัวอักษร')
        if not isinstance(fields, list) or not 1 <= len(fields) <= 8:
            raise ValueError('แต่ละก้อนต้องมี 1–8 ตัวแปร')
        normalized_fields = []
        for field in fields:
            if not isinstance(field, dict) or set(field) != {'key', 'instruction', 'target'}:
                raise ValueError('แต่ละตัวแปรต้องมี key, instruction และ target')
            key, instruction, target = field['key'], field['instruction'], field['target']
            if (not isinstance(key, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', key)
                    or key in reserved or key in seen):
                raise ValueError('ชื่อตัวแปรต้องไม่ซ้ำหรือชนชื่อระบบ ใช้ a-z, 0-9, _ และเริ่มด้วยตัวอักษร')
            if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 4000:
                raise ValueError('ระบุวิธีอ่านค่าของตัวแปร 1–4,000 ตัวอักษร')
            if not isinstance(target, str) or target not in targets() or target in destinations:
                raise ValueError('เลือกช่อง User JSON ที่รองรับและไม่ซ้ำกับตัวแปรอื่น')
            seen.add(key); destinations.add(target)
            normalized_fields.append(dict(key=key, instruction=instruction.strip(), target=target))
        normalized.append(dict(id=gid, title=title.strip(), fields=normalized_fields))
    return {'instructions': clean, 'extra_groups': normalized}


def compile_steps(table, codes, profiles):
    """Apply field-level additions, preserving all protected extraction rules."""
    selected = [(code, profiles[code]) for code in prompts.TYPE_SPECIFICITY if code in (codes or ()) and code in profiles]
    if not selected:
        return table
    result = []
    for step in table:
        notes = []
        for code, profile in selected:
            for key in step['keys']:
                note = profile['value']['instructions'].get(key)
                if note:
                    notes.append(f'- {key} ({code}): {note}')
        result.append({**step, 'rules': step['rules'] + '\n\nUser field reading guidance (keep the JSON shape and copy-only rules):\n' + '\n'.join(notes)} if notes else step)
    seen_keys = set()
    seen_targets = {CORE_TARGETS[k] for step in table for k in step['keys'] if k in CORE_TARGETS}
    for code, profile in selected:
        for group in profile['value']['extra_groups']:
            fields = group['fields']
            if any(f['key'] in seen_keys or f['target'] in seen_targets for f in fields):
                raise ValueError('ตัวแปรหรือช่อง User JSON ของประเภทร่วมซ้ำกัน กรุณาแก้ prompt ของประเภทที่เกี่ยวข้อง')
            seen_keys.update(f['key'] for f in fields); seen_targets.update(f['target'] for f in fields)
            schema = {f['key']: copy.deepcopy(targets()[f['target']]) if targets()[f['target']].get('type') == 'array'
                      else {'type': 'string'} for f in fields}
            result.append({'id': f'user_{code}_{group["id"]}', 'title': group['title'],
                'keys': tuple(f['key'] for f in fields), 'custom_targets': {f['key']: f['target'] for f in fields},
                'custom_schema': schema, 'max_tokens': 2000,
                'skeleton': json.dumps({f['key']: [] if schema[f['key']].get('type') == 'array' else '' for f in fields}),
                'rules': 'Copy only printed evidence. Do not calculate or invent missing values. Return empty strings or empty arrays when absent.\n'
                         + '\n'.join(f'- {f["key"]}: {f["instruction"]}' for f in fields)
                         + '\nLocked schemas for array values (use JSON numbers for numeric cells):\n'
                         + json.dumps({k: s for k, s in schema.items() if s.get('type') == 'array'}, ensure_ascii=False)})
    return tuple(result)


def view(code, profile):
    value = profile['value']
    return [{'id': s['id'], 'title': s['title'], 'fields': [
        {'key': k, 'instruction': value['instructions'].get(k, ''),
         'default_instruction': GUIDANCE.get(k, 'อ่านค่าที่พิมพ์ตรงกับตัวแปรนี้ตามกฎระบบ ไม่เดาหรือคำนวณแทน'),
         'target': CORE_TARGETS.get(k, '')} for k in s['keys']]} for s in base_steps(code)]


def changes_between(before, after):
    """Compare captured values, never today's team settings or default prose."""
    changes = []
    def changed(entity, path, old, new, **context):
        if old != new:
            changes.append({'entity': entity, 'path': path,
                            'action': 'added' if old is None else 'removed' if new is None else 'changed',
                            'before': copy.deepcopy(old), 'after': copy.deepcopy(new), **context})
    for code in prompts.TYPE_SPECIFICITY:
        old = (before.get(code) or {}).get('value') or {}
        new = (after.get(code) or {}).get('value') or {}
        instructions_before, instructions_after = old.get('instructions', {}), new.get('instructions', {})
        for key in sorted(set(instructions_before) | set(instructions_after)):
            changed('instruction', 'instructions.' + key, instructions_before.get(key), instructions_after.get(key),
                    doc_type=code, key=key)
        old_list, new_list = old.get('extra_groups', []), new.get('extra_groups', [])
        old_groups, new_groups = {g['id']: g for g in old_list}, {g['id']: g for g in new_list}
        for gid in dict.fromkeys([g['id'] for g in old_list + new_list]):
            a, b = old_groups.get(gid), new_groups.get(gid)
            context = {'doc_type': code, 'group_id': gid, 'group_title': (b or a)['title']}
            if a is None or b is None:
                changed('group', 'extra_groups.' + gid, a, b, **context)
                continue
            changed('group_title', 'extra_groups.' + gid + '.title', a['title'], b['title'], **context)
            aa, bb = {f['key']: f for f in a['fields']}, {f['key']: f for f in b['fields']}
            for key in dict.fromkeys([f['key'] for f in a['fields'] + b['fields']]):
                first, last = aa.get(key), bb.get(key)
                path = 'extra_groups.' + gid + '.fields.' + key
                if first is None or last is None:
                    changed('field', path, first, last, key=key, **context)
                else:
                    changed('field_instruction', path + '.instruction', first['instruction'], last['instruction'], key=key, **context)
                    changed('destination', path + '.target', first['target'], last['target'], key=key, **context)
            first_order, last_order = list(aa), list(bb)
            if set(first_order) == set(last_order):
                changed('field_order', 'extra_groups.' + gid + '.fields', first_order, last_order, **context)
        first_order, last_order = list(old_groups), list(new_groups)
        if set(first_order) == set(last_order):
            changed('group_order', 'extra_groups', first_order, last_order, doc_type=code)
    return changes
