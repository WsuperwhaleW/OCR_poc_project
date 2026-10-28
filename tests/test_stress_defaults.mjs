import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = readFileSync(new URL('../templates/index.html', import.meta.url), 'utf8');
const source = page.slice(page.indexOf('const ST_DEFAULTS_KEY'), page.indexOf('\nfunction stRealTypes()'));
const clean = value => JSON.parse(JSON.stringify(value));
const modes = ['ocr', 'extract', 'classify', 'full'];

function setup(storage = new Map(), failStorage = false) {
  const fields = {};
  function control(id, value, type = 'text', options = []) {
    const el = {id, value: String(value), type, checked: !!value, disabled: false,
      tagName: options.length ? 'SELECT' : 'INPUT', min: '', max: '',
      options: options.map(value => ({value, textContent: value})),
      checkValidity: () => true, reportValidity() {this.reported = true;}};
    Object.defineProperty(el, 'selectedIndex', {get() {return options.indexOf(el.value);}});
    fields[id] = el;
    return el;
  }
  control('stMode', 'ocr', 'select', modes);
  for (const [id, value] of Object.entries({stConc: 50, stJobs: 100, stWarm: 0, stMax: 0, stPages: 100})) {
    Object.assign(control(id, value, 'number'), {min: '0', max: '5000'});
  }
  fields.stConc.max = '256';
  control('stSeed', '');
  control('stShape', 'single', 'select', ['single', 'agentic', 'hybrid']);
  control('stStrategy', 'balanced', 'select', ['balanced', 'uniform', 'sequential']);
  control('stDetail', 'original', 'select', ['original', 'medium', 'low']);
  for (const id of ['stExtras', 'stSchemaFirst', 'stNoCache', 'stSaveLog', 'stSaveDetails', 'stHistoryMonitor', 'stValidationImages']) {
    control(id, id === 'stExtras' || id === 'stHistoryMonitor', 'checkbox');
  }
  const controls = Object.values(fields);
  const list = values => ({boxes: values.map(([value, checked]) => ({value, checked})),
    querySelectorAll(selector) {return this.boxes.filter(b => selector !== 'input:checked' || b.checked);}});
  fields.stFiles = list([['sol001', true], ['sol014', true], ['sol021', false]]);
  fields.stRealTypes = list([['pdf', true], ['png', false]]);
  fields.stRealFiles = list([['a.pdf', true], ['b.png', false]]);
  fields.stSetup = {querySelectorAll: () => controls};
  const classes = new Set();
  fields.stDefaultsNote = {hidden: true, textContent: '', classList: {
    add: v => classes.add(v), remove: v => classes.delete(v)}};
  // An ID outside the Stress setup must never be restored from storage.
  fields.globalModel = {value: 'current-model'};
  const ctx = vm.createContext({API: '/ocr/api', REAL_MODES: ['ocr', 'full'],
    stMode: fields.stMode, stSource: 'benchmark', stBatchOn: false, stFillOn: true,
    stRealPick: 'random', stReal: null, stRunning: false,
    stEl: id => fields[id], stFileBoxes: () => fields.stFiles.boxes,
    stSetSource: value => {ctx.stSource = value;},
    localStorage: {getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => {if (failStorage) throw new Error('quota'); storage.set(key, value);}}});
  vm.runInContext(source + `
    stDefaultBase = stCaptureDefaults();
    api = {load: stLoadModeDefaults, save: stSaveModeDefaults,
           read: stReadDefaultStore, capture: stCaptureDefaults,
           applyReal: stApplyRealDefaults};`, ctx);
  return {ctx, api: ctx.api, fields, storage, classes};
}

test('built-in numbers are mode-specific before any defaults are saved', () => {
  const {api, fields} = setup();
  for (const [mode, concurrent, jobs] of [['ocr', 50, 100], ['extract', 64, 40], ['classify', 4, 20], ['full', 4, 20]]) {
    api.load(mode);
    assert.equal(fields.stConc.value, String(concurrent));
    assert.equal(fields.stJobs.value, String(jobs));
  }
});

test('explicit Save restores every setting and file selection after a fresh page', () => {
  const first = setup();
  first.api.load('extract');
  for (const [id, value] of Object.entries({stConc: '32', stJobs: '120', stWarm: '2', stMax: '300',
    stSeed: '123', stShape: 'agentic', stStrategy: 'sequential', stDetail: 'low', stPages: '25'})) {
    first.fields[id].value = value;
  }
  for (const id of ['stSchemaFirst', 'stNoCache', 'stSaveLog', 'stSaveDetails', 'stValidationImages']) first.fields[id].checked = true;
  first.fields.stExtras.checked = first.fields.stHistoryMonitor.checked = false;
  first.fields.stFiles.boxes[0].checked = false;
  first.ctx.stBatchOn = true;
  const expected = clean(first.api.capture());
  first.api.save();
  assert.match(first.fields.stDefaultsNote.textContent, /Saved defaults/);
  const next = setup(first.storage);
  const saved = next.api.read();
  assert.equal(saved.lastMode, 'extract');
  next.api.load(saved.lastMode);
  assert.deepEqual(clean(next.api.capture()), expected);
});

test('saving and switching modes keeps independent defaults, including shapes and checkboxes', () => {
  const {api, fields} = setup();
  api.load('extract');
  fields.stShape.value = 'agentic'; fields.stStrategy.value = 'sequential';
  fields.stSaveDetails.checked = true; fields.stExtras.checked = false;
  api.save();
  api.load('ocr');
  assert.equal(fields.stSaveDetails.checked, false);
  fields.stStrategy.value = 'uniform'; fields.stConc.value = '48'; api.save();
  api.load('extract');
  assert.equal(fields.stConc.value, '64');
  assert.equal(fields.stStrategy.value, 'sequential');
  assert.equal(fields.stShape.value, 'agentic');
  assert.equal(fields.stSaveDetails.checked, true);
  assert.equal(fields.stExtras.checked, false);
  api.load('ocr');
  assert.equal(fields.stConc.value, '48');
  assert.equal(fields.stStrategy.value, 'uniform');
});

test('Hybrid and its schema option survive Save as default and a fresh page', () => {
  const first = setup(); first.api.load('extract');
  first.fields.stShape.value = 'hybrid'; first.fields.stSchemaFirst.checked = true;
  first.api.save();
  const next = setup(first.storage); next.api.load('extract');
  assert.equal(next.fields.stShape.value, 'hybrid');
  assert.equal(next.fields.stSchemaFirst.checked, true);
});

test('validation image recheck is remembered separately for Full pipeline', () => {
  const first = setup(); first.api.load('full');
  assert.equal(first.fields.stValidationImages.checked, false);
  first.fields.stValidationImages.checked = true; first.api.save();
  const next = setup(first.storage); next.api.load('full');
  assert.equal(next.fields.stValidationImages.checked, true);
  next.api.load('ocr'); assert.equal(next.fields.stValidationImages.checked, false);
  next.api.load('full'); assert.equal(next.fields.stValidationImages.checked, true);
});

test('realData selection waits for asynchronous listing and survives a failed listing', () => {
  const first = setup();
  first.ctx.stReal = {files: []}; first.ctx.stSource = 'realdata'; first.ctx.stRealPick = 'choose';
  first.api.save();
  const next = setup(first.storage);
  next.api.load('ocr');
  assert.equal(next.ctx.stSource, 'realdata');
  assert.equal(next.ctx.stRealPick, 'choose');
  next.ctx.stReal = {error: 'temporary failure'}; next.api.applyReal();
  next.fields.stRealTypes.boxes.forEach(b => b.checked = true);
  next.fields.stRealFiles.boxes.push({value: 'new.pdf', checked: true});
  next.ctx.stReal = {files: []}; next.api.applyReal();
  assert.deepEqual(next.fields.stRealTypes.boxes.map(b => b.checked), [true, false]);
  assert.deepEqual(next.fields.stRealFiles.boxes.map(b => b.checked), [true, false, false]);
});

test('corrupt storage and invalid saved controls fall back without affecting global controls', () => {
  const {api, storage, fields} = setup();
  const key = 'ocrStressDefaults:v1:/ocr/api';
  for (const invalid of ['not JSON', 'null', '{"version":1,"modes":[]}']) {
    storage.set(key, invalid);
    assert.doesNotThrow(() => api.load('extract'));
    assert.equal(fields.stConc.value, '64');
  }
  storage.set(key, JSON.stringify({version: 1, modes: {extract: {controls: {
    stConc: '-9', stShape: 'obsolete-shape', globalModel: 'wrong-model'}}}}));
  api.load('extract');
  assert.equal(fields.stConc.value, '64'); assert.equal(fields.stShape.value, 'single');
  assert.equal(fields.globalModel.value, 'current-model');
});

test('a storage failure reports failure, and a running test cannot save or restore', () => {
  const {api, ctx, fields} = setup(new Map(), true);
  api.save();
  assert.match(fields.stDefaultsNote.textContent, /could not be saved/);
  fields.stDefaultsNote.textContent = 'unchanged'; ctx.stRunning = true;
  api.save(); api.load('extract');
  assert.equal(fields.stDefaultsNote.textContent, 'unchanged');
  assert.equal(fields.stMode.value, 'ocr');
});

test('saving from two tabs preserves the other modes and deployments have separate keys', () => {
  const shared = new Map(); const a = setup(shared), b = setup(shared);
  a.api.load('extract'); a.api.save(); b.api.load('ocr'); b.api.save();
  assert.deepEqual(Object.keys(a.api.read().modes).sort(), ['extract', 'ocr']);
  assert.equal(shared.has('ocrStressDefaults:v1:/ocr/api'), true);
  assert.equal(shared.has('ocrStressDefaults:v1:/api'), false);
});
