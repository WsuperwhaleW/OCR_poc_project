import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = readFileSync(new URL('../templates/index.html', import.meta.url), 'utf8');
function source(name) {
  const start = page.indexOf('function ' + name + '(');
  assert.notEqual(start, -1, name);
  return page.slice(start, page.indexOf('\n}', start) + 2);
}
function setup() {
  const fields = {};
  const el = id => fields[id] ||= {value: '', checked: false, disabled: false, style: {},
    className: '', listeners: {}, addEventListener(kind, fn) {this.listeners[kind] = fn;}};
  class FormData {
    constructor() {this.values = {};}
    append(key, value) {this.values[key] = value;}
  }
  const ctx = vm.createContext({current: null, workspaceCase: '', workspaceSourceSeq: 0,
    controller: null, matchedCase: null, FormData, detailSel: {value: 'medium'},
    document: {getElementById: el}, chip: {style: {}}, mapChip: {}, runNote: {},
    dropPreview() {}, hidePrepared() {}, paintRun() {}, paintServer() {}, localPreview() {},
    refreshPrepared() {}, enableCompare() {}});
  vm.runInContext(['prepBody', 'clearWorkspaceFile'].map(source).join('\n'), ctx);
  const start = page.indexOf("document.getElementById('workspaceCase').addEventListener");
  vm.runInContext(page.slice(start, page.indexOf("document.getElementById('realPick').addEventListener", start)), ctx);
  return {ctx, el};
}

test('Workspace shows Test files before Real files and retains local upload', () => {
  const pane = page.slice(page.indexOf('id="paneSource"'), page.indexOf('<!-- /paneSource -->'));
  assert.ok(pane.indexOf('id="workspaceCase"') < pane.indexOf('id="realPick"'));
  assert.ok(pane.indexOf('id="realPick"') < pane.indexOf('id="drop"'));
  assert.match(pane, /Test files/); assert.match(pane, /Real files/);
  assert.match(pane, /id="workspaceFillImages"/); assert.match(pane, /id="workspaceValidationImages"/);
});

test('selecting a case replaces an uploaded file and preview and run use that same case', () => {
  const {ctx, el} = setup(); ctx.current = {name: 'upload.pdf'};
  el('workspaceCase').value = 'sol014';
  el('workspaceCase').listeners.change({target: el('workspaceCase')});
  assert.equal(ctx.current, null); assert.equal(ctx.workspaceCase, 'sol014');
  assert.equal(ctx.matchedCase, 'sol014');
  assert.equal(el('workspaceCase').value, 'sol014');
  assert.equal(ctx.prepBody().values.case, 'sol014');
  const run = page.slice(page.indexOf("runBtn.addEventListener('click'"));
  let options; ctx.runBtn = {addEventListener(_name, callback) {callback();}};
  ctx.startRead = value => {options = value;};
  vm.runInContext(run.slice(0, run.indexOf('\n});') + 4), ctx);
  assert.equal(options.case, 'sol014'); assert.equal(options.image, null);
  ctx.clearWorkspaceFile(); assert.equal(ctx.prepBody(), null);
  assert.equal(el('workspaceCase').value, '');
});

test('Workspace sends its own two flags and never sends them for OCR only', () => {
  const start = page.indexOf('  const body = new FormData();', page.indexOf('async function startRead('));
  const block = page.slice(start, page.indexOf('  controller = new AbortController();', start));
  const {ctx, el} = setup();
  el('workspaceFillImages').checked = false; el('workspaceValidationImages').checked = true;
  ctx.opts = {case: 'sol014', detail: 'medium', extract: true};
  vm.runInContext('(function(){' + block + 'return body;})()', ctx);
  const execute = () => vm.runInContext('(function(){' + block + 'return body;})()', ctx).values;
  assert.equal(execute().fill_missing, '0'); assert.equal(execute().validation_retry, '1');
  ctx.opts.extract = false;
  assert.equal(execute().extract, '0'); assert.ok(!('validation_retry' in execute()));
});

test('Workspace preferences survive reload independently of Stress and User settings', () => {
  const start = page.indexOf("const workspaceImageDefaultsKey =");
  const block = page.slice(start, page.indexOf('/* ---------- Fields only:', start));
  const storage = new Map();
  function load() {
    const {ctx, el} = setup(); ctx.API = '/ocr/api';
    ctx.localStorage = {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)};
    vm.runInContext(block, ctx); return {ctx, el};
  }
  const first = load(); first.el('workspaceValidationImages').checked = true;
  first.el('workspaceValidationImages').listeners.change();
  const next = load(); assert.equal(next.el('workspaceValidationImages').checked, true);
  assert.deepEqual([...storage.keys()], ['ocrWorkspaceImageRechecks:v1:/ocr/api']);
});

test('real-file selection stays visible, and a slow old load cannot replace a newer case', async () => {
  const {ctx, el} = setup(); ctx.API = '/ocr/api';
  ctx.File = class {constructor(_parts, name, opts) {this.name = name; this.type = opts.type;}};
  ctx.addFiles = files => {ctx.current = files[0]; ctx.workspaceSourceSeq++;};
  let finish;
  ctx.fetch = () => new Promise(resolve => {finish = () => resolve({ok:true, blob:async()=>({type:'application/pdf'})});});
  vm.runInContext('async ' + source('realPickLoad'), ctx);
  let pending = ctx.realPickLoad('invoice/one.pdf'); finish(); await pending;
  assert.equal(ctx.current.name, 'one.pdf'); assert.equal(el('realPick').value, 'invoice/one.pdf');
  assert.equal(el('realPick').disabled, false);
  pending = ctx.realPickLoad('invoice/slow.pdf');
  el('workspaceCase').value = 'sol014'; el('workspaceCase').listeners.change({target:el('workspaceCase')});
  finish(); await pending;
  assert.equal(ctx.workspaceCase, 'sol014'); assert.equal(ctx.current, null);
  assert.equal(el('realPick').value, '');
});
