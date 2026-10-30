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
def rule_units(step):
    """Lossless editable slices of the actual English rules, never summaries.

    Unclaimed text remains shared/read-only. A rule explicitly defining two
    variables is one unit; both fields point to that same editable slice.
    """
    rules, keys = step['rules'], tuple(step['keys'])
    if len(keys) == 1:
        return [{'keys': keys, 'start': 0, 'end': len(rules), 'text': rules}]
    pattern = re.compile(r'(?m)^[-*][ \t]+(' + '|'.join(re.escape(k) for k in keys) + r')\b')
    matches = list(pattern.finditer(rules))
    units = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rules)
        paragraph_end = rules.find('\n\n', match.start(), end)
        if paragraph_end >= 0:
            end = paragraph_end
        text = rules[match.start():end].rstrip()
        units.append({'keys': (match.group(1),), 'start': match.start(), 'end': match.start() + len(text), 'text': text})
    for key in keys:
        if any(key in unit['keys'] for unit in units):
            continue
        owners = [u for u in units if re.search(r'\b' + re.escape(key) + r'\b', u['text'])]
        if len(owners) != 1:
            raise ValueError('ยังไม่สามารถแยกข้อกำหนดจริงของตัวแปร ' + key)
        owners[0]['keys'] += (key,)
    if {k for unit in units for k in unit['keys']} != set(keys):
        raise ValueError('ข้อกำหนดตัวแปรไม่ตรงกับก้อนคำถาม')
    return units


def unit_override(unit, instructions):
    entries = [(key, instructions[key]) for key in unit['keys'] if instructions.get(key)]
    distinct = list(dict.fromkeys(text for _, text in entries))
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0]
    # Older profiles could store different notes for variables sharing a rule.
    # Preserve all of them explicitly rather than silently discarding a note.
    return '\n\n'.join(key + ': ' + text for key, text in entries)


def editor_value(code, value):
    instructions = {}
    for step in base_steps(code):
        for unit in rule_units(step):
            override = unit_override(unit, value.get('instructions', {}))
            if override and override.strip() != unit['text'].strip():
                instructions[unit['keys'][0]] = override
    return {'instructions': instructions, 'extra_groups': copy.deepcopy(value.get('extra_groups', []))}


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
    value = editor_value(code, {'instructions': clean, 'extra_groups': normalized})
    if any(len(text) > 4000 for text in value['instructions'].values()):
        raise ValueError('คำอธิบายร่วมของตัวแปรต้องไม่เกิน 4,000 ตัวอักษร')
    return value


def compile_steps(table, codes, profiles):
    """Replace exact field rule slices; shared rules and built-in tables stay intact."""
    selected = [(code, profiles[code]) for code in prompts.TYPE_SPECIFICITY if code in (codes or ()) and code in profiles]
    if not selected:
        return table
    result = []
    for step in table:
        edits = [(code, profile['value'].get('instructions', {})) for code, profile in selected
                 if set(step['keys']) & set(profile['value'].get('instructions', {}))]
        if not edits:
            result.append(step)
            continue
        rules = step['rules']
        for unit in reversed(rule_units(step)):
            replacements = list(dict.fromkeys(unit_override(unit, notes) for _, notes in edits))
            replacements = [text for text in replacements if text is not None]
            if not replacements:
                continue
            if len(replacements) != 1:
                raise ValueError('คำอธิบายตัวแปรของประเภทร่วมขัดกัน: ' + ', '.join(unit['keys']))
            rules = rules[:unit['start']] + replacements[0] + rules[unit['end']:]
        result.append({**step, 'rules': rules})
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
    value = editor_value(code, profile['value'])
    groups = []
    for step in base_steps(code):
        units = rule_units(step)
        fields = []
        for key in step['keys']:
            unit = next(unit for unit in units if key in unit['keys'])
            anchor = unit['keys'][0]
            fields.append({'key': key, 'edit_key': anchor, 'linked_keys': list(unit['keys']),
                           'instruction': value['instructions'].get(anchor, ''),
                           'default_instruction': unit['text'], 'target': CORE_TARGETS.get(key, '')})
        pieces, position = [], 0
        for unit in units:
            pieces.append(step['rules'][position:unit['start']]);position = unit['end']
        pieces.append(step['rules'][position:])
        groups.append({'id': step['id'], 'title': step['title'], 'fields': fields,
                       'shared_rules': ''.join(pieces).strip()})
    return groups


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
        if old or new:
            old, new = editor_value(code, old), editor_value(code, new)
        instructions_before, instructions_after = old.get('instructions', {}), new.get('instructions', {})
        for key in sorted(set(instructions_before) | set(instructions_after)):
            length = len(changes)
            changed('instruction', 'instructions.' + key, instructions_before.get(key), instructions_after.get(key),
                    doc_type=code, key=key)
            if len(changes) > length:
                unit = next(unit for step in base_steps(code) for unit in rule_units(step) if key in unit['keys'])
                changes[-1].update(linked_keys=list(unit['keys']),
                    effective_before=instructions_before.get(key, unit['text']),
                    effective_after=instructions_after.get(key, unit['text']))
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
