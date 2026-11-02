import {benchmarkSource} from './benchmark_source.mjs';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
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
  const ctx = vm.createContext({current: null, workspaceCase: '', workspaceReal: '',
    controller: null, matchedCase: null, FormData, detailSel: {value: 'medium'},
    document: {getElementById: el}, chip: {style: {}}, mapChip: {}, runNote: {},
    meta: {classList: {remove() {}}},
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

function setupSearch(files) {
  const {ctx, el} = setup();
  ctx.stReal = {types: [...new Set(files.map(f => f.type))], files};
  ctx.Option = class {constructor(text, value) {this.text = text; this.value = value;}};
  ctx.document.createElement = () => ({childElementCount: 0, options: [],
    append(option) {this.options.push(option); this.childElementCount++;}});
  el('realPick').replaceChildren = (...items) => {el('realPick').items = items;};
  vm.runInContext(source('realPickMatches') + source('realPickPaint'), ctx);
  const listed = () => el('realPick').items.slice(1).flatMap(group => group.options.map(o => o.value));
  return {ctx, el, listed};
}
const realFiles = [
  {name: 'invoice/INV-2024-001.pdf', type: 'pdf', size_mb: 1},
  {name: 'invoice/ใบแจ้งหนี้ มกราคม.pdf', type: 'pdf', size_mb: 1},
  {name: 'scan/photo_001.jpg', type: 'jpg', size_mb: 1},
];

test('search narrows the real files by any case, by folder, in Thai, and word by word', () => {
  const {ctx, el, listed} = setupSearch(realFiles);
  const search = query => {el('realSearch').value = query; ctx.realPickPaint(); return listed();};
  assert.equal(search('').length, 3);
  assert.equal(el('realSearchCount').textContent, '3 files');
  assert.deepEqual(search('inv-2024'), ['invoice/INV-2024-001.pdf']);
  assert.equal(el('realSearchCount').textContent, '1 of 3 files');
  assert.deepEqual(search('มกราคม'), ['invoice/ใบแจ้งหนี้ มกราคม.pdf']);
  assert.deepEqual(search('scan/'), ['scan/photo_001.jpg']);
  assert.deepEqual(search('001 photo'), ['scan/photo_001.jpg']);
  assert.deepEqual(search('nothing-like-this'), []);
  assert.equal(el('realPick').items[0].text, '— no file name matches —');
  assert.equal(el('realPick').items.length, 1);
});

test('a search that excludes the chosen file keeps it listed and selected', () => {
  const {ctx, el, listed} = setupSearch(realFiles);
  el('realPick').value = 'scan/photo_001.jpg';
  el('realSearch').value = 'inv-2024';
  ctx.realPickPaint();
  assert.deepEqual(listed(), ['invoice/INV-2024-001.pdf', 'scan/photo_001.jpg']);
  assert.equal(el('realPick').value, 'scan/photo_001.jpg');
  assert.equal(el('realSearchCount').textContent, '1 of 3 files');
});

test('real-file selection sends only its server name for preview and read, never downloads and uploads it', async () => {
  const {ctx, el} = setup(); ctx.API = '/ocr/api';
  ctx.fetch = () => {throw Error('must not download the file');};
  vm.runInContext('async ' + source('realPickLoad'), ctx);
  await ctx.realPickLoad('invoice/one.pdf');
  assert.equal(ctx.current, null); assert.equal(ctx.workspaceReal, 'invoice/one.pdf');
  assert.equal(el('realPick').value, 'invoice/one.pdf');
  assert.equal(el('realPick').disabled, false);
  assert.deepEqual({...ctx.prepBody().values}, {realdata: 'invoice/one.pdf', detail:'medium'});
  const run = page.slice(page.indexOf("runBtn.addEventListener('click'"));
  let options; ctx.runBtn = {addEventListener(_name, callback) {callback();}};
  ctx.startRead = value => {options = value;};
  vm.runInContext(run.slice(0, run.indexOf('\n});') + 4), ctx);
  assert.equal(options.realdata, 'invoice/one.pdf'); assert.equal(options.image, null);
  ctx.opts = options;
  const start = page.indexOf('  const body = new FormData();', page.indexOf('async function startRead('));
  const block = page.slice(start, page.indexOf('  controller = new AbortController();', start));
  const sent = vm.runInContext('(function(){' + block + 'return body;})()', ctx).values;
  assert.equal(sent.realdata, 'invoice/one.pdf'); assert.ok(!('image' in sent));
  el('workspaceCase').value = 'sol014'; el('workspaceCase').listeners.change({target:el('workspaceCase')});
  assert.equal(ctx.workspaceCase, 'sol014'); assert.equal(ctx.current, null);
  assert.equal(ctx.workspaceReal, ''); assert.equal(ctx.prepBody().values.case, 'sol014');
  assert.equal(el('realPick').value, '');
});

test('dropping an upload clears the realData reference for both preview and read', async () => {
  const {ctx} = setup();
  vm.runInContext('async ' + source('realPickLoad'), ctx);
  await ctx.realPickLoad('large.pdf');
  ctx.previewOne = () => {};
  vm.runInContext(source('addFiles'), ctx);
  const file = {name:'local.pdf'};
  ctx.addFiles([file]);
  assert.equal(ctx.workspaceReal, ''); assert.equal(ctx.current, file);
  assert.equal(ctx.prepBody().values.image, file);
  assert.ok(!('realdata' in ctx.prepBody().values));
});
