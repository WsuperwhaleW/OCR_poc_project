import {benchmarkSource} from './benchmark_source.mjs';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const source = readFileSync(new URL('../templates/_stress_local_script.html', import.meta.url), 'utf8');
const factory = source.split('// Page integration.')[0];
const page = benchmarkSource();
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

function evidenceSetup(local=true) {
  const pending=[],opened=[],notes=[];
  const rows=['a','b'].map((id,i)=>({dataset:{alertId:id,alertCase:'folder/'+id+'.pdf',alertPage:String(i+2)}}));
  const host={querySelectorAll:()=>rows};
  const ctx=vm.createContext({API:'/ocr/api',stEvidenceRequest:0,cmpOverride:null,cmpView:'',FX_CASES:[],
    window:{StressLocalPreview:{hasCandidate:()=>local}},
    stNote:{textContent:'',classList:{add(){}}},
    document:{getElementById:()=>({hidden:false})},
    loadStressEvidence:url=>new Promise((resolve,reject)=>pending.push({url,resolve,reject})),
    stOpenEvidenceCompare(id,scroll=true){opened.push({id,scroll,source:ctx.cmpOverride});},
    enableCompare(){},stPaintFlagLinks:message=>notes.push(message)});
  const start=page.indexOf('async function stShowEvidence(');
  const end=page.indexOf('\nfunction stOpenEvidenceCompare(',start);
  vm.runInContext(page.slice(start,end),ctx);
  const finish=(index,id)=>pending[index].resolve({case:'folder/'+id+'.pdf',filename:id+'.pdf',
    page:2,truth_case:null,fields:[{document:1,rows:[]}],attempts:[{page:2,text:'OCR '+id,image:true}]});
  return {ctx,host,pending,opened,notes,finish};
}

test('PC original opens before the slow evidence request completes, then fields arrive without scrolling again',async()=>{
  const s=evidenceSetup();const request=s.ctx.stShowEvidence('a','missing',s.host,'1');
  assert.equal(s.opened.length,1,'must open before awaiting any server response');
  assert.equal(s.ctx.cmpOverride.filename,'folder/a.pdf');
  assert.equal(s.ctx.cmpOverride.reads[0].page,2);
  assert.equal(s.ctx.cmpOverride.reads[0].src,null);
  assert.equal(s.ctx.cmpOverride.loading,true);
  s.finish(0,'a');await request;
  assert.equal(s.ctx.cmpOverride.reads[0].text,'OCR a');
  assert.equal(s.ctx.cmpView,'source-fields');
  assert.equal(s.opened[1].scroll,false);
});

test('out-of-order evidence replies and errors cannot replace the newer PC selection',async()=>{
  for(const fail of [false,true]){
    const s=evidenceSetup();const first=s.ctx.stShowEvidence('a','loop',s.host,'1');
    const second=s.ctx.stShowEvidence('b','loop',s.host,'2');s.finish(1,'b');await second;
    if(fail)s.pending[0].reject(Error('old failed'));else s.finish(0,'a');await first;
    assert.equal(s.ctx.cmpOverride.alert,'b');assert.equal(s.ctx.cmpOverride.filename,'b.pdf');
    assert.equal(s.opened.length,3);assert.equal(s.notes.length,0);
  }
});

test('an unavailable evidence record leaves the local original open with an explicit results error',async()=>{
  const s=evidenceSetup();const request=s.ctx.stShowEvidence('a','loop',s.host,'1');
  s.pending[0].reject(Error('Server unavailable'));await request;
  assert.equal(s.ctx.cmpOverride.filename,'folder/a.pdf');
  assert.equal(s.ctx.cmpOverride.loading,false);
  assert.match(s.ctx.cmpOverride.loadError,/Server unavailable/);
  assert.equal(s.opened.length,1);
});

test('without a selected local file the prepared server image still uses its evidence URL',async()=>{
  const s=evidenceSetup(false);const request=s.ctx.stShowEvidence('a','loop',s.host,'1');
  assert.equal(s.opened.length,0);s.finish(0,'a');await request;
  assert.equal(s.ctx.cmpOverride.reads[0].src,'/ocr/api/stress/evidence/a/0.png');
  assert.equal(s.opened[0].scroll,true);
});

test('benchmark case names resolve to the manifest filename before requesting evidence',async()=>{
  const s=evidenceSetup();s.ctx.FX_CASES=[{id:'sol014',pdf:'credit-note.pdf'}];
  s.host.querySelectorAll=()=>[{dataset:{alertId:'a',alertCase:'sol014',alertPage:'3'}}];
  const request=s.ctx.stShowEvidence('a','loop',s.host,'1');
  assert.equal(s.ctx.cmpOverride.filename,'credit-note.pdf');assert.equal(s.ctx.cmpOverride.page,3);
  s.finish(0,'a');await request;
});

test('selecting another folder rebuilds the filename index without retaining old candidates',()=>{
  const s=setup();s.api.setFiles([file('old.pdf')]);s.api.setFiles([file('new.pdf')]);
  assert.equal(s.api.hasCandidate('old.pdf'),false);assert.equal(s.api.hasCandidate('folder/new.pdf'),true);
});

test('PDF pages can be chosen independently of the recorded read and survive comparison refreshes',async()=>{
  const s=setup();s.api.setFiles([file('a.pdf')]);
  await s.show('a.pdf',{selection:'alert:0'});
  assert.equal(s.last().pdfPages,3);assert.equal(s.last().pdfPage,2);
  await s.api.setPage(1);assert.equal(s.last().pdfPage,1);
  await s.api.setPage(3);assert.equal(s.last().pdfPage,3);
  await s.show('a.pdf',{selection:'alert:0',remote:'/results-arrived.png'});
  assert.equal(s.last().pdfPage,3);assert.match(s.last().caption,/page 3/);
  assert.match(s.last().download,/-p3-local.png$/);
  assert.deepEqual(s.rendered,[2,1,3]);assert.equal(s.opened.length,1);
});

test('switching recorded reads or files resets the manual PDF page but server image remains exact',async()=>{
  const s=setup();s.api.setFiles([file('a.pdf'),file('b.pdf')]);
  await s.show('a.pdf',{selection:'alert:0'});await s.api.setPage(3);
  await s.api.setSource('server');assert.equal(s.last().url,'/ocr/api/model.png');
  assert.match(s.last().caption,/page 2/);assert.equal(s.last().pdfPages,undefined);
  await s.api.setSource('pc');assert.equal(s.last().pdfPage,3);
  await s.show('a.pdf',{selection:'alert:1',page:1});assert.equal(s.last().pdfPage,1);
  await s.api.setPage(3);await s.show('b.pdf',{selection:'other:0',page:2});
  assert.equal(s.last().pdfPage,2);
});

test('invalid page numbers do not render outside the PDF and restore the selected page',async()=>{
  const s=setup();s.api.setFiles([file('a.pdf')]);await s.show('a.pdf');
  for(const value of ['',0,-1,4,1.5,'text',Infinity]){
    await s.api.setPage(value);assert.equal(s.last().pdfPage,2);
  }
  assert.deepEqual(s.rendered,[2]);
  await s.api.setPage('3');assert.equal(s.last().pdfPage,3);
});

test('a failed manual PDF page leaves page controls usable instead of showing a different server page',async()=>{
  const s=setup({renderPDF:async(_doc,page)=>{if(page===3)throw Error('broken page');return {page};}});
  s.api.setFiles([file('a.pdf')]);await s.show('a.pdf');await s.api.setPage(3);
  assert.equal(s.last().source,'pc');assert.equal(s.last().url,'');assert.equal(s.last().pdfPages,3);
  assert.match(s.last().status,/Choose another page/);
  await s.api.setPage(2);assert.equal(s.last().pdfPage,2);assert.match(s.last().url,/^blob:/);
});

test('rapid PDF page changes only publish the newest selected page',async()=>{
  const s=setup();s.api.setFiles([file('a.pdf')]);await s.show('a.pdf');
  const first=s.api.setPage(1),last=s.api.setPage(3);await Promise.all([first,last]);
  assert.equal(s.last().pdfPage,3);assert.deepEqual(s.rendered,[2,3]);
});
