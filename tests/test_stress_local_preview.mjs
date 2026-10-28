import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const source = readFileSync(new URL('../templates/_stress_local_script.html', import.meta.url), 'utf8');
const factory = source.split('// Page integration.')[0];
const page = readFileSync(new URL('../templates/index.html', import.meta.url), 'utf8');
function file(name, size=100) {
  return {name, size, lastModified: 0, reads: 0,
    get webkitRelativePath() { throw new Error('Paths must not be inspected'); },
    async arrayBuffer() { this.reads++; return new ArrayBuffer(1); }};
}
function setup(overrides={}) {
  const states=[], made=[], revoked=[], opened=[], rendered=[], destroyed=[];
  const ctx = vm.createContext({});
  vm.runInContext(factory, ctx);
  const api = ctx.createStressLocalPreview({
    createURL(blob) { made.push(blob); return 'blob:local-' + made.length; },
    revokeURL(url) { revoked.push(url); },
    async openPDF(file) { await file.arrayBuffer(); opened.push(file);
      return {numPages: 3, destroy: async () => { destroyed.push(file); }}; },
    async renderPDF(doc, number) { rendered.push(number); return {page: number}; },
    update(state) { states.push(state); }, ...overrides,
  });
  const show = (filename, options={}) => api.show({filename, page: 2, remote: '/ocr/api/model.png',
    detail: 'medium', caseName: 'sol014', ...options});
  return {api, show, states, made, revoked, opened, rendered, destroyed, last: () => states.at(-1)};
}
test('folder selection keeps references only; filenames from nested folders match without path access', async () => {
  const s=setup(), pdf=file('Invoice.PDF'), image=file('receipt.png');
  s.api.setFiles([pdf, image]);
  assert.equal(pdf.reads + image.reads, 0);
  assert.equal(s.api.hasCandidate('invoice.pdf'), true);
  await s.show('invoice.pdf');
  assert.equal(pdf.reads, 1); assert.equal(image.reads, 0);
  assert.equal(s.last().source, 'pc'); assert.match(s.last().url, /^blob:/);
  assert.equal(s.states.some(state => state.url === '/ocr/api/model.png'), false);
});
test('local raster preview uses original file without PDF parsing or reading its contents', async () => {
  const s=setup(), image=file('page.jpg'); s.api.setFiles([image]);
  await s.show('page.jpg');
  assert.equal(image.reads, 0); assert.equal(s.made[0], image);
  assert.match(s.last().caption, /whole original image/);
});
test('unmatched and unsupported originals fall back explicitly to the model image', async () => {
  const s=setup(); s.api.setFiles([file('other.pdf'), file('scan.tiff')]);
  await s.show('missing.pdf'); assert.equal(s.last().url, '/ocr/api/model.png');
  assert.match(s.last().status, /No filename match/);
  await s.show('scan.tiff'); assert.equal(s.last().source, 'server'); assert.match(s.last().status, /format/);
  assert.equal(s.opened.length, 0);
});
test('duplicate names require choice and never silently preview the first file or download the server image', async () => {
  const s=setup(), a=file('same.pdf',100), b=file('same.pdf',200); s.api.setFiles([a,b]);
  await s.show('same.pdf');
  assert.equal(s.last().url, ''); assert.equal(s.opened.length,0); assert.match(s.last().status,/duplicated/);
  await s.api.choose('same.pdf','1'); assert.equal(s.opened[0],b);
  assert.equal(a.reads,0); assert.equal(b.reads,1);
});
test('manual server selection preserves the exact prepared image; selecting a new folder prefers PC again', async () => {
  const s=setup(); s.api.setFiles([file('a.pdf')]); await s.show('a.pdf');
  await s.api.setSource('server'); assert.equal(s.last().url,'/ocr/api/model.png');
  assert.match(s.last().caption,/medium.*server model image/);
  await s.api.setSource('pc'); assert.match(s.last().url,/^blob:/); assert.equal(s.opened.length,1);
  s.api.setFiles([file('a.pdf')]); await s.show('a.pdf'); assert.equal(s.last().source,'pc');
});
test('PDF page number comes from evidence, not retry index, and repeated paints reuse the page', async () => {
  const s=setup(); s.api.setFiles([file('a.pdf')]);
  await s.show('a.pdf',{page:3}); await s.show('a.pdf',{page:3});
  assert.deepEqual(s.rendered,[3]);
  await s.show('a.pdf',{page:2}); assert.deepEqual(s.rendered,[3,2]);
  await s.show('a.pdf',{page:3}); assert.deepEqual(s.rendered,[3,2]);
  assert.match(s.last().download,/-p3-local.png$/);
});
test('failed or out-of-range PDF falls back visibly; no fabricated page', async () => {
  const s=setup(); s.api.setFiles([file('a.pdf')]);
  await s.show('a.pdf',{page:4}); assert.match(s.last().status,/outside/);
  assert.equal(s.last().source,'server'); assert.deepEqual(s.rendered,[]);
  const failed=setup({openPDF:async()=>{throw new Error('encrypted');}});
  failed.api.setFiles([file('locked.pdf')]); await failed.show('locked.pdf');
  assert.match(failed.last().status,/encrypted/);
});
test('a slow PDF cannot replace a newer image selection', async () => {
  let release; const gate=new Promise(resolve=>{release=resolve;});
  const s=setup({renderPDF:async()=>{await gate; return {};}});
  s.api.setFiles([file('slow.pdf'),file('new.png')]);
  const first=s.show('slow.pdf');
  await new Promise(resolve=>setImmediate(resolve));
  s.show('new.png'); release(); await first;
  assert.match(s.last().caption,/new.png/); assert.equal(s.made.length,1);
});
test('preview URLs are bounded, old PDFs are destroyed and clearing releases all references', async () => {
  const s=setup(); s.api.setFiles(['a.png','b.png','c.png','d.png','e.pdf','f.pdf'].map(name=>file(name)));
  for (const name of ['a.png','b.png','c.png','d.png']) await s.show(name);
  assert.equal(s.revoked.length,1);
  await s.show('e.pdf'); await s.show('f.pdf'); assert.equal(s.destroyed.length,1);
  await s.api.dispose(); assert.equal(s.revoked.length,s.made.length);
  assert.equal(s.destroyed.length,2); assert.equal(s.api.hasCandidate('e.pdf'),false);
});
test('changing folders during a pending load disposes the old PDF and cannot expose stale content', async () => {
  let release; const gate=new Promise(resolve=>{release=resolve;}); let destroyed=0;
  const s=setup({openPDF:async()=>{await gate;return {numPages:3,destroy:async()=>{destroyed++;}};}});
  s.api.setFiles([file('old.pdf')]); const pending=s.show('old.pdf');
  await new Promise(resolve=>setImmediate(resolve));
  s.api.setFiles([file('new.png')]); s.show('new.png'); release(); await pending;
  assert.equal(destroyed,1); assert.match(s.last().caption,/new.png/); assert.equal(s.made.length,1);
});
test('integration uses real page metadata and versioned same-origin assets with no persistence or upload', () => {
  assert.match(source,/page: read.page \|\| o.page \|\| 1/);
  assert.doesNotMatch(source,/localStorage|sessionStorage|indexedDB|webkitRelativePath|fetch\(/);
  assert.match(source,/import\(base \+ 'legacy\/build\/pdf.min.mjs'\)/);
  assert.match(source,/isEvalSupported: false/);
  assert.match(page,/webkitdirectory multiple hidden/);
  assert.ok(page.indexOf('id="stLocalFolder"') < page.indexOf('id="stSetup"'));
});
