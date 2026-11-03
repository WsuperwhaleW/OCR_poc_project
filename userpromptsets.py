"""Versioned, whole-form User Hybrid sets. Never changes built-in extraction."""
import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import prompts
import userprompts


class Conflict(ValueError):
    pass


DEFAULT_NAMES = {
    'INVOICE':'Invoice (Sup&Pay)', 'CREDIT_NOTE':'Credit Note (Sup&Pay)',
    'RECEIPT':'Receipt (Sup&Pay)', 'WHT_CERTIFICATE':'WHT 50 ทวิ (PCP)', 'RTV_SHIP_CREDIT':'RTV',
}
EXTRA_NAMES = {
    'WHT_CERTIFICATE':('WHT','หนังสือรับรองการหักภาษี ณ ที่จ่าย (50 ทวิ)'),
    'RTV_SHIP_CREDIT':('RTV','ใบส่งคืนสินค้า'),
    'CHEQUE_DELIVERY':('CHEQUE_DELIVERY','ใบรับส่งเช็ค'),
    'PAYMENT_ADVICE':('PAYMENT_ADVICE','ใบแจ้งการชำระเงิน'),
    'PAYMENT_SCHEDULE':('PAYMENT_SCHEDULE','ใบนัดชำระเงิน'),
    'PURCHASE_ORDER':('PURCHASE_ORDER','ใบสั่งซื้อ'),
    'STOCK_TRANSFER':('STOCK_TRANSFER','ใบขอโอนสินค้าภายใน'),
    'BILLING_NOTE':('BILLING_NOTE','ใบวางบิล'),
    'STATEMENT_OF_ACCOUNT':('STATEMENT_OF_ACCOUNT','ใบแจ้งยอดบัญชี'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def catalog():
    items = json.loads((Path(__file__).parent / 'schemas/user_document_type_catalog.json').read_text('utf-8'))['documentTypes']
    literal = {'80':'DEBIT_NOTE', '81':'CREDIT_NOTE', '380':'INVOICE', '388':'TAX_INVOICE',
               'T01':'RECEIPT', 'T02':'INVOICE_TAX_INVOICE', 'T03':'RECEIPT_TAX_INVOICE',
               'T04':'DELIVERY_TAX_INVOICE', 'T05':'ABBREVIATED_TAX_INVOICE',
               'T06':'RECEIPT_ABBREVIATED_TAX_INVOICE', 'T07':'CANCELLATION_NOTE'}
    rows = [dict(code=literal[item['code']], user_code=item['code'], name=item['nameTh'],
                 label=item['code'] + ' · ' + item['nameTh'], section='user') for item in items]
    used = {row['code'] for row in rows}
    rows.extend(dict(code=code, user_code='', name=EXTRA_NAMES[code][1],
                     label=' · '.join(EXTRA_NAMES[code]), section='extra')
                for code in prompts.TYPE_NAMES if code not in used)
    return rows


def codes_for(code):
    if code not in prompts.TYPE_NAMES:
        raise ValueError('ไม่รู้จักประเภทเอกสารนี้')
    return [code, *prompts.TYPE_COMPONENTS.get(code, ())]


def select(snapshot, codes):
    """Literal bindings replace their component profiles, never union them back."""
    if not snapshot:
        return []
    bindings = snapshot.get('bindings', {})
    candidates = set(codes or ())
    for code in tuple(candidates):
        if code in bindings:
            candidates.difference_update(prompts.TYPE_COMPONENTS.get(code, ()))
    # A credit/debit note mentioning returns retains its existing note roles.
    if candidates & {'CREDIT_NOTE', 'DEBIT_NOTE'}:
        candidates.discard('RTV_SHIP_CREDIT')
    result, seen = [], set()
    for code in prompts.TYPE_SPECIFICITY:
        binding = bindings.get(code) if code in candidates else None
        if binding and binding['set_id'] not in seen:
            record = snapshot['sets'][binding['set_id']]
            seen.add(binding['set_id'])
            result.append((code, record))
    return result


def validate(base_code, value, base=None):
    if not isinstance(value, dict) or set(value) - {'instructions', 'extra_groups', 'disabled_fields', 'disabled_groups'}:
        raise ValueError('แก้ได้เฉพาะตัวแปร ก้อนคำถาม และสถานะเปิดใช้งาน')
    base = base or userprompts.base_steps(base_code)
    keys = {key for step in base for key in step['keys']}
    raw_groups = value.get('extra_groups', [])
    custom_groups = [g for g in raw_groups if isinstance(g, dict)] if isinstance(raw_groups, list) else []
    custom_ids = {g['id'] for g in custom_groups if isinstance(g.get('id'), str)}
    custom_keys = {f['key'] for g in custom_groups if isinstance(g.get('fields'), list)
                   for f in g['fields'] if isinstance(f, dict) and isinstance(f.get('key'), str)}
    disabled = value.get('disabled_fields', [])
    groups = value.get('disabled_groups', [])
    if (not isinstance(disabled, list) or any(not isinstance(k, str) or k not in keys | custom_keys for k in disabled)
            or not isinstance(groups, list) or any(not isinstance(k, str) or k not in {s['id'] for s in base} | custom_ids for k in groups)):
        raise ValueError('ฟิลด์หรือก้อนที่ปิดไม่อยู่ในชุดต้นแบบ')
    for step in base:
        for unit in userprompts.rule_units(step):
            off = set(disabled) & set(unit['keys'])
            if off and off != set(unit['keys']):
                raise ValueError('ฟิลด์ที่ใช้คำอธิบายร่วมกันต้องเปิดหรือปิดพร้อมกัน: ' + ', '.join(unit['keys']))
    # Existing validation owns key names, typed destinations and array schemas.
    active = {k for s in base if s['id'] not in groups for k in s['keys'] if k not in disabled}
    clean = userprompts.validate(base_code, {k:value.get(k, {} if k == 'instructions' else [])
                                           for k in ('instructions', 'extra_groups')}, base, active)
    if disabled:
        clean['disabled_fields'] = sorted(set(disabled))
    if groups:
        clean['disabled_groups'] = sorted(set(groups))
    return clean


def restrict_step(original, kept, notes=None):
    """Keep only owned fields, using captured rule spans even for free-form edits."""
    step = copy.deepcopy(original)
    wanted = set(kept)
    kept = [key for key in step['keys'] if key in wanted]
    if not kept:
        return None
    if 'user_rule_units' in step:
        units = [{**unit, 'text':step['rules'][unit['start']:unit['end']]}
                 for unit in step['user_rule_units']]
    else:
        units = userprompts.rule_units(step)
    pieces, spans, position, length = [], [], 0, 0
    for unit in units:
        between = step['rules'][position:unit['start']]
        pieces.append(between); length += len(between)
        keys = [key for key in unit['keys'] if key in kept]
        if keys:
            replacement = userprompts.unit_override(unit, notes or {})
            text = unit['text'] if replacement is None else replacement
            spans.append({'keys':keys, 'start':length, 'end':length + len(text)})
            pieces.append(text); length += len(text)
        position = unit['end']
    pieces.append(step['rules'][position:])
    step['rules'] = ''.join(pieces)
    if len(kept) != len(step['keys']):
        # Rewrite the fixed group output instruction after slicing, then fix
        # spans if that instruction precedes a field rule.
        old = 'Return just the six requested keys, without explanations or extra fields.'
        new = 'Return only the requested keys, without explanations or extra fields.'
        at = step['rules'].find(old)
        if at >= 0:
            step['rules'] = step['rules'][:at] + new + step['rules'][at+len(old):]
            for unit in spans:
                if unit['start'] >= at + len(old):
                    unit['start'] += len(new)-len(old); unit['end'] += len(new)-len(old)
                elif unit['end'] > at:
                    unit['end'] += len(new)-len(old)
        aliases = {step.get('request_aliases', {}).get(key,key) for key in kept}
        step['skeleton'] = json.dumps({key:value for key,value in json.loads(step['skeleton']).items()
                                       if key in kept or key in aliases})
        for name in ('request_aliases','custom_schema','custom_targets'):
            if name in step:
                step[name] = {key:value for key,value in step[name].items() if key in kept}
        marker = '\nLocked schemas for array values (use JSON numbers for numeric cells):\n'
        if step.get('custom_schema') is not None and marker in step['rules']:
            step['rules'] = step['rules'].rsplit(marker,1)[0] + marker + json.dumps({
                key:schema for key,schema in step['custom_schema'].items() if schema.get('type') == 'array'}, ensure_ascii=False)
    step['keys'] = tuple(kept)
    step['user_rule_units'] = spans
    return step


def compile_record(record):
    base, value = record['base_steps'], record['value']
    disabled, off_groups = set(value.get('disabled_fields', [])), set(value.get('disabled_groups', []))
    result = []
    for original in base:
        if original['id'] in off_groups:
            continue
        kept = [k for k in original['keys'] if k not in disabled]
        if not kept:
            continue
        result.append(restrict_step(original, kept, value.get('instructions', {})))
    extras = []
    for group in value.get('extra_groups', []):
        if group['id'] in off_groups:
            continue
        group = copy.deepcopy(group)
        group['fields'] = [f for f in group['fields'] if f['key'] not in disabled]
        if group['fields']:
            extras.append(group)
    result = userprompts.compile_steps(result, [record['base_code']], {record['base_code']:{'value':{
        'instructions':{}, 'extra_groups':extras}}})
    for step in result:
        if step.get('custom_targets'):
            group = next(g for g in extras if step['id'] == f'user_{record["base_code"]}_{g["id"]}')
            position = step['rules'].index('- ' + group['fields'][0]['key'] + ': ')
            spans = []
            for field in group['fields']:
                text = '- ' + field['key'] + ': ' + field['instruction']
                spans.append({'keys':[field['key']], 'start':position, 'end':position+len(text)})
                position += len(text)+1
            step['user_rule_units'] = spans
    if not result:
        raise ValueError('เปิดใช้อย่างน้อยหนึ่งฟิลด์หรือก้อนคำถาม')
    return tuple(result)


def _step_targets(step):
    return step.get('custom_targets') or {k:userprompts.CORE_TARGETS[k] for k in step['keys']
                                          if k in userprompts.CORE_TARGETS}


def runtime(table, codes, snapshot, extras=False):
    selected = select(snapshot, codes)
    if not selected:
        return table
    merged, owners, taken = [], set(), set()
    for code, record in selected:
        # Older snapshots predate captured spans. Rebuild only from their own
        # frozen base/value, never from today's team settings or engine defaults.
        compiled = record['steps'] if all('user_rule_units' in s for s in record['steps']) else compile_record(record)
        for original in compiled:
            # The first set in TYPE_SPECIFICITY owns a field and its User JSON
            # destination. Each set is validated alone, so two sets may map
            # different custom fields to one destination; on a mixed document
            # that raised and the document got no fields at all.
            targets = _step_targets(original)
            kept = [key for key in original['keys'] if key not in owners and targets.get(key) not in taken]
            step = restrict_step(original, kept)
            if step is None:
                continue
            owners.update(kept)
            taken.update(targets[key] for key in kept if key in targets)
            step.update(user_prompt_set_id=record['id'], user_prompt_doc_type=code)
            merged.append(step)
    targets = {}
    for index, step in enumerate(merged):
        mapping = step.get('custom_targets') or {k:userprompts.CORE_TARGETS[k] for k in step['keys'] if k in userprompts.CORE_TARGETS}
        for key, target in mapping.items():
            if target in targets and targets[target] != (index,key):
                raise ValueError('ช่อง User JSON ของประเภทร่วมซ้ำกัน: ' + target)
            targets[target] = (index,key)
    # The independently configured overflow question remains controlled by extras.
    if extras:
        merged.extend(copy.deepcopy(s) for s in table if s['id'] == 'other')
    return tuple(merged)


def view(record):
    groups = userprompts.view(record['base_code'], record, record['base_steps'])
    off = set(record['value'].get('disabled_fields', []))
    disabled_groups = set(record['value'].get('disabled_groups', []))
    for group in groups:
        group['enabled'] = group['id'] not in disabled_groups
        for field in group['fields']:
            field['enabled'] = field['key'] not in off
    return groups


def initialize(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS prompt_sets (id TEXT PRIMARY KEY, name TEXT NOT NULL,
            base_code TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS prompt_set_versions (set_id TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(set_id,revision));
        CREATE TABLE IF NOT EXISTS prompt_bindings (code TEXT PRIMARY KEY, set_id TEXT NOT NULL,
            revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS prompt_binding_events (id INTEGER PRIMARY KEY, code TEXT NOT NULL,
            created_at TEXT NOT NULL, action TEXT NOT NULL, before_value TEXT, after_value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS prompt_set_migrations (id INTEGER PRIMARY KEY);
    ''')
    db.execute('BEGIN IMMEDIATE')
    if db.execute('SELECT 1 FROM prompt_set_migrations WHERE id=1').fetchone():
        migrate_sup_pay(db)
        return
    old = {row['code']:dict(row) for row in db.execute('SELECT * FROM prompt_profiles')}
    for code in prompts.DOC_TYPE_FIELDS:
        if not prompts.DOC_TYPE_FIELDS[code]:
            continue
        previous = old.get(code)
        record = dict(id='builtin_' + code.lower(), name=DEFAULT_NAMES[code], base_code=code,
                      revision=previous['revision'] if previous else 0,
                      value=json.loads(previous['value']) if previous else {'instructions':{}, 'extra_groups':[]},
                      base_steps=copy.deepcopy(userprompts.base_steps(code)), created_at=now())
        record['steps'] = compile_record(record)
        _write_version(db, record)
        db.execute('INSERT INTO prompt_bindings VALUES (?,?,0)', (code, record['id']))
        for alias, components in prompts.TYPE_COMPONENTS.items():
            if code in components and code in ('INVOICE', 'RECEIPT'):
                db.execute('INSERT INTO prompt_bindings VALUES (?,?,0)', (alias, record['id']))
    db.execute('INSERT INTO prompt_set_migrations VALUES (1)')
    migrate_sup_pay(db)


def migrate_sup_pay(db):
    """Migration 2: explicit Sup&Pay defaults; existing User bindings win."""
    if db.execute('SELECT 1 FROM prompt_set_migrations WHERE id=2').fetchone():
        return
    defaults = {
        'INVOICE':'builtin_invoice', 'TAX_INVOICE':'builtin_invoice',
        'INVOICE_TAX_INVOICE':'builtin_invoice', 'DELIVERY_TAX_INVOICE':'builtin_invoice',
        'ABBREVIATED_TAX_INVOICE':'builtin_invoice', 'CREDIT_NOTE':'builtin_credit_note',
        'RECEIPT':'builtin_receipt', 'RECEIPT_TAX_INVOICE':'builtin_receipt',
        'RECEIPT_ABBREVIATED_TAX_INVOICE':'builtin_receipt',
    }
    for code, set_id in defaults.items():
        if (db.execute('SELECT 1 FROM prompt_bindings WHERE code=?',(code,)).fetchone()
                or db.execute('SELECT 1 FROM prompt_binding_events WHERE code=?',(code,)).fetchone()):
            continue
        db.execute('INSERT INTO prompt_bindings(code,set_id,revision) SELECT ?,id,0 FROM prompt_sets WHERE id=?', (code,set_id))
        if db.execute('SELECT 1 FROM prompt_bindings WHERE code=?',(code,)).fetchone():
            db.execute('INSERT INTO prompt_binding_events(code,created_at,action,before_value,after_value) VALUES (?,?,?,?,?)',
                       (code,now(),'migration_2',json.dumps(None),json.dumps({'set_id':set_id,'revision':0})))
    for code, name in DEFAULT_NAMES.items():
        row = db.execute('SELECT payload FROM prompt_sets WHERE id=? AND name=?',('builtin_'+code.lower(),code)).fetchone()
        if row:
            record = json.loads(row['payload'])
            record.update(name=name, revision=record['revision']+1)
            _write_version(db,record)
    db.execute('INSERT INTO prompt_set_migrations VALUES (2)')


def _write_version(db, record):
    payload = json.dumps(record, ensure_ascii=False)
    db.execute('INSERT OR REPLACE INTO prompt_sets VALUES (?,?,?,?,?)',
               (record['id'], record['name'], record['base_code'], record['revision'], payload))
    db.execute('INSERT INTO prompt_set_versions VALUES (?,?,?,?)',
               (record['id'], record['revision'], now(), payload))


def snapshot(db):
    return {'version':1,
            'sets':{r['id']:json.loads(r['payload']) for r in db.execute('SELECT * FROM prompt_sets')},
            'bindings':{r['code']:{'set_id':r['set_id'], 'revision':r['revision']}
                        for r in db.execute('SELECT * FROM prompt_bindings')}}


def field_revision(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def field_selection(state, code):
    """Build a selectable superset, retaining each source's frozen field rules."""
    row = next((r for r in catalog() if r['section'] == 'user' and r['code'] == code), None)
    if row is None:
        raise ValueError('เลือกประเภทเอกสารใน catalog ของ User')
    selected = select(state, codes_for(code))
    if selected:
        record = copy.deepcopy(selected[0][1])
    else:
        record = {'base_code':code, 'name':row['label'], 'value':{'instructions':{}, 'extra_groups':[]},
                  'base_steps':copy.deepcopy(prompts.steps_for_types(codes_for(code), False, mode='hybrid'))}
    table = runtime(prompts.steps_for_types(codes_for(code), False, mode='hybrid'), codes_for(code), state)
    enabled = {k for step in table for k in step['keys']}
    base = list(record['base_steps'])
    options, custom = {}, {}

    def add_options(step, source, origins=None):
        for unit in userprompts.rule_units(step):
            for key in unit['keys']:
                options[key] = {'key':key, 'linked_keys':list(unit['keys']),
                                'target':userprompts.CORE_TARGETS.get(key, ''), 'source':(origins or {}).get(key, source)}

    for step in base:
        add_options(step, record['name'], record.get('field_sources'))
    for group in record['value'].get('extra_groups', []):
        for field in group['fields']:
            options[field['key']] = {'key':field['key'], 'linked_keys':[field['key']],
                                     'target':field['target'], 'source':record.get('field_sources', {}).get(field['key'], record['name'])}
    sources = []
    for item in catalog():
        binding = state['bindings'].get(item['code'])
        if binding and binding['set_id'] not in sources:
            sources.append(binding['set_id'])
    for source_id in sources:
        source = state['sets'][source_id]
        for step in source['base_steps']:
            kept = [key for key in step['keys'] if key not in options]
            if not kept:
                continue
            added = restrict_step(step, kept, source['value'].get('instructions', {}))
            added['id'] = 'field_' + str(len(base)) + '_' + step['id']
            base.append(added)
            add_options(added, source['name'], source.get('field_sources'))
        for group in source['value'].get('extra_groups', []):
            for field in group['fields']:
                if field['key'] not in options:
                    options[field['key']] = {'key':field['key'], 'linked_keys':[field['key']],
                                             'target':field['target'], 'source':source.get('field_sources', {}).get(field['key'], source['name'])}
                    custom[field['key']] = copy.deepcopy(field)
    record['base_steps'] = base
    return record, list(options.values()), enabled, custom


def save_fields(db, body):
    """Save one type atomically; shared sets are copied, never edited in place."""
    if not isinstance(body, dict) or set(body) != {'doc_type','revision','enabled_keys'}:
        raise ValueError('ส่งประเภทเอกสาร เวอร์ชันตาราง และรายการฟิลด์ให้ครบ')
    keys = body['enabled_keys']
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('เลือกอย่างน้อยหนึ่งฟิลด์และไม่ส่งชื่อฟิลด์ซ้ำ')
    db.execute('BEGIN IMMEDIATE')
    state = snapshot(db)
    if body['revision'] != field_revision(state):
        raise Conflict('ชุดฟิลด์เปลี่ยนไปแล้ว กรุณาปิด Edit และรีเฟรชตารางก่อนแก้ไขอีกครั้ง')
    code = body['doc_type']
    record, options, _, custom = field_selection(state, code)
    available = {f['key'] for f in options}
    if set(keys) - available:
        raise ValueError('มีฟิลด์ที่ไม่อยู่ในรายการให้เลือก')
    value = copy.deepcopy(record['value'])
    groups = value.setdefault('extra_groups', [])
    group_ids = {g['id'] for g in groups}
    for key in keys:
        if key not in custom:
            continue
        gid = 'extra_pick_' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]
        while gid in group_ids:
            gid += '_'
        group_ids.add(gid)
        groups.append({'id':gid, 'title':custom[key]['key'], 'fields':[custom[key]]})
    owned = {k for s in record['base_steps'] for k in s['keys']} | {f['key'] for g in groups for f in g['fields']}
    value['disabled_fields'] = sorted(owned - set(keys))
    value['disabled_groups'] = []
    record['value'] = validate(record['base_code'], value, record['base_steps'])
    record['field_sources'] = {f['key']:f['source'] for f in options if f['key'] in owned}
    binding = state['bindings'].get(code)
    shared = binding and sum(b['set_id'] == binding['set_id'] for b in state['bindings'].values()) > 1
    if shared or not binding:
        if binding:
            record.update(parent_set_id=record['id'], parent_revision=record['revision'])
        label = next(r['label'] for r in catalog() if r['code'] == code)
        record.update(id=uuid.uuid4().hex, name=label, revision=1, created_at=now())
    else:
        record['revision'] += 1
    record['steps'] = compile_record(record)
    _write_version(db, record)
    if shared or not binding:
        after = {'set_id':record['id'], 'revision':(binding['revision'] if binding else 0) + 1}
        db.execute('INSERT OR REPLACE INTO prompt_bindings VALUES (?,?,?)', (code, after['set_id'], after['revision']))
        db.execute('INSERT INTO prompt_binding_events(code,created_at,action,before_value,after_value) VALUES (?,?,?,?,?)',
                   (code, now(), 'field_selection', json.dumps(binding), json.dumps(after)))
    return record


def change(db, body):
    allowed = {'doc_type','binding_revision','set_id','revision','action','name','value'}
    if not isinstance(body, dict) or set(body) - allowed or not {'doc_type','binding_revision','set_id','revision','action'} <= set(body):
        raise ValueError('ส่งประเภท ชุด เวอร์ชัน และการกระทำให้ครบ')
    code, action = body['doc_type'], body['action']
    codes_for(code)
    if action not in ('shared', 'detach', 'bind'):
        raise ValueError('เลือกแก้ชุดร่วม แยกชุด หรือใช้ชุดร่วม')
    if any(isinstance(body[k], bool) or not isinstance(body[k], int) or body[k] < 0
           for k in ('revision','binding_revision')):
        raise ValueError('เวอร์ชันไม่ถูกต้อง')
    db.execute('BEGIN IMMEDIATE')
    state = snapshot(db)
    binding = state['bindings'].get(code)
    record = state['sets'].get(body['set_id'])
    if (not record or record['revision'] != body['revision']
            or (binding['revision'] if binding else 0) != body['binding_revision']):
        raise Conflict('มีผู้แก้ชุดหรือเปลี่ยนการผูกประเภทแล้ว กรุณาโหลดข้อมูลล่าสุด')
    if action == 'shared' and (not binding or binding['set_id'] != record['id']):
        raise Conflict('ประเภทนี้ไม่ได้ใช้ชุดที่กำลังแก้แล้ว')
    record = copy.deepcopy(record)
    if action == 'bind' and 'value' in body and validate(record['base_code'],body['value'],record['base_steps']) != record['value']:
        raise ValueError('ใช้ชุดร่วมโดยไม่แก้เนื้อหา หากต้องการแก้ให้เลือกแยกชุดหรือแก้ชุดร่วม')
    if action != 'bind':
        record['value'] = validate(record['base_code'], body.get('value', record['value']), record['base_steps'])
        record['name'] = body.get('name', record['name'])
        if not isinstance(record['name'], str) or not 1 <= len(record['name'].strip()) <= 120:
            raise ValueError('ชื่อชุดต้องยาว 1–120 ตัวอักษร')
        record['name'] = record['name'].strip()
        if action == 'detach':
            record.update(id=uuid.uuid4().hex, parent_set_id=record['id'],
                          parent_revision=record['revision'], revision=1, created_at=now())
        else:
            record['revision'] += 1
        record['steps'] = compile_record(record)
        _write_version(db, record)
    if action in ('bind','detach'):
        after = {'set_id':record['id'], 'revision':(binding['revision'] if binding else 0) + 1}
        db.execute('INSERT OR REPLACE INTO prompt_bindings VALUES (?,?,?)', (code, after['set_id'], after['revision']))
        db.execute('INSERT INTO prompt_binding_events(code,created_at,action,before_value,after_value) VALUES (?,?,?,?,?)',
                   (code, now(), action, json.dumps(binding), json.dumps(after)))
    return record


def carry_disabled(record, value):
    """Add a set's on/off state to a value from an older route that never sends it.

    Without this, a legacy save or retry asked disabled fields again, and a
    custom field that replaced a disabled one failed as a duplicate destination.
    Only fields and groups that still exist in the new value are carried.
    """
    if not isinstance(value, dict) or {'disabled_fields', 'disabled_groups'} & set(value):
        return value
    merged = dict(value)
    groups = value.get('extra_groups')
    groups = [g for g in groups if isinstance(g, dict)] if isinstance(groups, list) else []
    fields = ({k for s in record['base_steps'] for k in s['keys']}
              | {f.get('key') for g in groups if isinstance(g.get('fields'), list)
                 for f in g['fields'] if isinstance(f, dict)})
    group_ids = {s['id'] for s in record['base_steps']} | {g.get('id') for g in groups}
    for name, present in (('disabled_fields', fields), ('disabled_groups', group_ids)):
        kept = [k for k in record['value'].get(name, []) if k in present]
        if kept:
            merged[name] = kept
    return merged


def legacy_value(value):
    """The instructions/extra_groups part that the older profile format stores."""
    return {'instructions':value.get('instructions', {}), 'extra_groups':value.get('extra_groups', [])}


def retry(snapshot_value, edits, legacy=False):
    result = copy.deepcopy(snapshot_value)
    for code, value in edits.items():
        binding = result['bindings'].get(code)
        if not binding:
            raise ValueError('เลือกชุดต้นแบบของประเภทนี้ก่อน Retry')
        record = copy.deepcopy(result['sets'][binding['set_id']])
        if legacy:
            value = carry_disabled(record, value)
        record['value'] = validate(record['base_code'], value, record['base_steps'])
        record.update(id='job_' + uuid.uuid4().hex, revision=record['revision'] + 1, scope='job')
        record['steps'] = compile_record(record)
        result['sets'][record['id']] = record
        result['bindings'][code] = {'set_id':record['id'], 'revision':binding['revision']}
    return result


def changes(before, after):
    result = []
    for code in prompts.TYPE_SPECIFICITY:
        a = select(before, [code]); b = select(after, [code])
        a = a[0][1] if a else None; b = b[0][1] if b else None
        if a == b:
            continue
        summary = lambda r: {k:r[k] for k in ('id','name','revision','base_code')} if r else None
        result.append({'entity':'prompt_set', 'action':'changed', 'doc_type':code,
                       'before':summary(a), 'after':summary(b)})
        av, bv = (a or {}).get('value', {}), (b or {}).get('value', {})
        if a and b and a['base_code'] == b['base_code']:
            legacy = lambda v: {k:v.get(k, {} if k == 'instructions' else []) for k in ('instructions','extra_groups')}
            details = userprompts.changes_between({a['base_code']:{'value':legacy(av)}},
                                                  {b['base_code']:{'value':legacy(bv)}},
                                                  {b['base_code']:b['base_steps']})
            result.extend({**row, 'doc_type':code} for row in details)
        for name, entity in (('disabled_fields','field_enabled'), ('disabled_groups','group_enabled')):
            first, last = set(av.get(name, [])), set(bv.get(name, []))
            for key in sorted(first ^ last):
                result.append({'entity':entity, 'action':'changed', 'doc_type':code, 'key':key,
                               'before':key not in first, 'after':key not in last})
    return result
