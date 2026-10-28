import {benchmarkSource} from './benchmark_source.mjs';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
function source(name) {
  const start = page.indexOf('function ' + name + '(');
  assert.notEqual(start, -1, name);
  const end = page.indexOf('\n}', start);
  // The mode predicate is intentionally a single-line function.
  return name === 'isStepMode' ? page.slice(start, page.indexOf('\n', start))
    : page.slice(start, end + 2);
}
function setup() {
  const fields = {};
  const el = id => fields[id] ||= {value: '', checked: false, disabled: false,
    classList: {toggle() {}}};
  const ctx = vm.createContext({extractMode: 'single', EXTRACT_STEP_COUNT: '6-20',
    HYBRID_STEP_COUNT: '5-18', fldMode: el('fldMode'), extractSel: el('extractSel'),
    fxMode: el('fxMode'), fxSchemaFirst: el('fxSchemaFirst'), extractNote: el('extractNote'),
    esc: x => x, stMode: el('stMode'), stEl: el, stSource: 'benchmark', stBatchOn: false, stFillOn: true,
    stSelectedCases: () => ['sol001']});
  const modes = page.slice(page.indexOf('const ST_EXTRACT_MODES'));
  vm.runInContext(modes.slice(0, modes.indexOf('\n')).replace('const ', 'var ') + '\n'
    + ['isStepMode', 'paintMode', 'modeLabel', 'stBody'].map(source).join('\n'), ctx);
  return {ctx, fields, el};
}

test('every extraction picker offers Hybrid, including Random and Stress tests', () => {
  for (const id of ['extractSel', 'fxMode', 'stShape', 'fldMode']) {
    const select = page.match(new RegExp('<select[^>]*id="' + id + '"[^>]*>([\\s\\S]*?)</select>'));
    assert.ok(select, id);
    for (const mode of ['single', 'agentic', 'hybrid']) assert.match(select[1], new RegExp('value="' + mode + '"'));
  }
  const selects = [...page.matchAll(/<select[^>]*>([\s\S]*?)<\/select>/g)].filter(m => m[1].includes('value="agentic"'));
  assert.equal(selects.length, 5);
  assert.ok(selects.every(m => m[1].includes('value="hybrid"')));
});

test('mode controls and result labels distinguish all three shapes', () => {
  const {ctx, fields} = setup();
  for (const mode of ['single', 'agentic', 'hybrid']) {
    ctx.extractMode = mode; ctx.paintMode();
    for (const id of ['fldMode', 'extractSel', 'fxMode']) assert.equal(fields[id].value, mode);
    assert.equal(fields.fxSchemaFirst.disabled, mode === 'single');
    const label = ctx.modeLabel({mode, steps: [{}, {}], schema_first: true});
    assert.ok(label.startsWith(mode));
    if (mode !== 'single') assert.match(label, /2 steps.*Schema first/);
  }
});

test('Stress payload retains Hybrid and enables schema-first only for step-based extraction', () => {
  const {ctx, el} = setup();
  el('stMode').value = 'extract'; el('stShape').value = 'hybrid';
  el('stSchemaFirst').checked = true;
  const body = ctx.stBody();
  assert.equal(body.extract_mode, 'hybrid'); assert.equal(body.schema_first, true);
  el('stShape').value = 'agentic'; assert.equal(ctx.stBody().schema_first, true);
  el('stShape').value = 'single'; assert.equal(ctx.stBody().schema_first, false);
  el('stShape').value = 'hybrid'; el('stMode').value = 'ocr';
  assert.equal(ctx.stBody().schema_first, false);
  // Image to extractor takes the shape and its schema option like Full pipeline.
  el('stMode').value = 'image';
  assert.equal(ctx.stBody().extract_mode, 'hybrid'); assert.equal(ctx.stBody().schema_first, true);
});

test('validation image recheck is opt-in, full-pipeline-only and disabled during a run', () => {
  const {ctx, el} = setup();
  ctx.stRunning = false;
  ctx.document = {getElementById: el};
  el('stFillImages').setAttribute = () => {};
  vm.runInContext(source('stPaintFill'), ctx);
  assert.match(page, /#stValidationField\[hidden\]\s*\{\s*display:\s*none/);
  const input = page.match(/<input id="stValidationImages"[^>]*>/)[0];
  assert.doesNotMatch(input, /checked/);
  for (const mode of ['full', 'ocr', 'extract', 'classify', 'image']) {
    el('stMode').value = mode;
    el('stValidationImages').checked = true;
    ctx.stPaintFill();
    assert.equal(el('stValidationField').hidden, mode !== 'full');
    assert.equal(el('stValidationImages').disabled, mode !== 'full');
    assert.equal(ctx.stBody().validation_retry, mode === 'full');
  }
  el('stMode').value = 'full'; ctx.stRunning = true; ctx.stPaintFill();
  assert.equal(el('stValidationImages').disabled, true);
  el('stValidationImages').checked = false;
  assert.equal(ctx.stBody().validation_retry, false);
});

test('Compare shows the before/answer/after audit and the decision, escaped, without the chats', () => {
  const ctx = vm.createContext({ST_WRONG_TEXT: {}, FIELD_LABELS: {subtotal: 'Subtotal'}, esc: value => String(value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')});
  const NL = String.fromCharCode(10);
  const constant = name => {
    const at = page.slice(page.indexOf('const ' + name));
    return at.slice(0, at.indexOf('};') + 2).replace('const ', 'var ');
  };
  vm.runInContext(['IMGRC_RULE', 'IMGRC_STATE'].map(constant).join(NL) + NL
    + ['cmpFieldsHtml', 'imgrcFieldName', 'imgrcVal', 'imgrcSame', 'imgrcChip', 'imgrcFindings',
       'imgrcState', 'imgrcAev', 'imgrcRulesTable', 'imgrcFillTable', 'imgrcValidationTable',
       'imgrcMeta', 'imageRecheckHtml'].map(source).join(NL), ctx);
  const retry = {fields: ['subtotal'], accepted: false, before: {subtotal: 'bad'},
    candidate: {subtotal: '<script>bad</script>'}, after: {subtotal: 'bad'},
    checks_before: [{field: 'subtotal', rule: 'vat_calculation', state: 'failed'}],
    reason: 'Validation did not improve', chats: [{hidden: 'request statistics'}]};
  const output = ctx.cmpFieldsHtml([{rows: [{key: 'subtotal', value: 'bad'}], validation_rechecks: [retry]}]);
  assert.match(output, /Validation recheck from images/);
  assert.match(output, /Not applied: Validation did not improve/);
  assert.match(output, /Triggered by: Subtotal vat_calculation \(failed\)/);
  assert.match(output, /original kept/);
  assert.ok(output.includes('&lt;script&gt;bad&lt;/script&gt;'));
  assert.ok(!output.includes('<script>'));
  assert.ok(!output.includes('request statistics'));
  retry.accepted = true; retry.changed = ['subtotal']; retry.after = {subtotal: 'good'};
  const applied = ctx.cmpFieldsHtml([{rows: [{key: 'subtotal', value: 'good'}], validation_rechecks: [retry]}]);
  assert.match(applied, /Applied: /);
  assert.match(applied, /edited/);
});

test('both rechecks say what they added, edited and validated', () => {
  const ctx = vm.createContext({ST_WRONG_TEXT: {}, FIELD_LABELS: {subtotal: 'Subtotal', vat_total: 'VAT'},
    esc: value => String(value).replaceAll('<', '&lt;')});
  const NL = String.fromCharCode(10);
  const constant = name => {
    const at = page.slice(page.indexOf('const ' + name));
    return at.slice(0, at.indexOf('};') + 2).replace('const ', 'var ');
  };
  vm.runInContext(['IMGRC_RULE', 'IMGRC_STATE'].map(constant).join(NL) + NL
    + ['imgrcFieldName', 'imgrcVal', 'imgrcSame', 'imgrcChip', 'imgrcFindings', 'imgrcState',
       'imgrcAev', 'imgrcRulesTable', 'imgrcFillTable', 'imgrcValidationTable', 'imgrcMeta',
       'imageRecheckHtml'].map(source).join(NL), ctx);
  const fill = {fields: ['vat_total'], filled: ['vat_total'], accepted: true,
    candidate: {vat_total: '7.00'}, added: {vat_total: '7.00'}, edited: {},
    validated: [{field: 'vat_total', rule: 'numeric', state: 'ok'}]};
  const check = {fields: ['subtotal'], accepted: true, changed: ['subtotal'],
    before: {subtotal: '1'}, candidate: {subtotal: '100'}, after: {subtotal: '100'},
    added: {}, edited: {subtotal: {from: '1', to: '100'}},
    checks_before: [{field: 'subtotal', rule: 'sum', state: 'failed'}],
    rules_before: [{field: 'subtotal', rule: 'sum', state: 'failed'}],
    rules_candidate: [{field: 'subtotal', rule: 'sum', state: 'ok'}],
    rules_after: [{field: 'subtotal', rule: 'sum', state: 'ok'}]};
  const html = ctx.imageRecheckHtml({image_fills: [fill], validation_image_rechecks: [check]},
                                    {fill_missing: true, validation_retry: true, flags: []});
  assert.match(html, /<b>1<\/b> added · <b>1<\/b> edited · <b>2<\/b> validation rule\(s\) checked, <b>1<\/b> now passing/);
  assert.match(html, /<b>Added<\/b> VAT = 7\.00/);
  assert.match(html, /a fill never overwrites a value/);
  assert.match(html, /<b>Edited<\/b> Subtotal: <s>1<\/s> → 100/);
  assert.match(html, /<b>Validated<\/b> 1 rule\(s\): 0 pass, 1 fail or warn/);
  assert.match(html, /<th>Before<\/th><th>Images answer<\/th><th>Now<\/th>/);
  // An attempt that was not applied added and edited nothing.
  const rejected = ctx.imgrcValidationTable({...check, accepted: false, changed: []});
  assert.match(rejected, /<b>Added<\/b> <span class="blank">nothing/);
  assert.match(rejected, /<b>Edited<\/b> <span class="blank">nothing/);
});
