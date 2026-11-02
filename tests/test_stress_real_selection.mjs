import {benchmarkSource} from './benchmark_source.mjs';
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
  const elements = {};
  function element() {
    return {value:'', checked:false, disabled:false, hidden:false, dataset:{}, style:{}, children:[],
      classList:{toggle() {}}, setAttribute() {},
      append(...children) {this.children.push(...children); for (const child of children) child.parentElement = this;},
      replaceChildren(...children) {this.children = []; this.append(...children);},
      querySelectorAll(selector) {
        const children = this.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
        if (selector === '*') return children;
        return children.filter(child => selector === 'button' ? child.tag === 'button'
          : child.tag === 'input' && (selector === 'input:checked' ? child.checked
            : selector === 'input:not(:checked)' ? !child.checked : true));
      }};
  }
  const el = id => elements[id] ||= element();
  const files = Array.from({length:200}, (_, i) => ({name:`invoice/inv-${String(i).padStart(3,'0')}.pdf`, type:'pdf', size_mb:4}));
  files.push({name:'scan/photo.png', type:'png', size_mb:1});
  const ctx = vm.createContext({API:'/api', stSource:'realdata', stRealPick:'choose', stReal:null,
    stMode:{value:'full'}, stRun:{}, stAbort:null, rtRunning:false, stRunning:false,
    REAL_MODES:['ocr','full','image'], ST_EXTRACT_MODES:['extract','full','image'],
    stBatchOn:false, stFillOn:false, stEl:el, stSelectedCases:() => [],
    isStepMode:value => value === 'hybrid',
    document:{createElement(tag) {return {...element(), tag};}},
    fetch:async () => ({ok:true, json:async () => ({files:[...files], types:['pdf','png']})}),
    stApplyRealDefaults() {}, realPickPaint() {}});
  vm.runInContext(['realPickMatches','stRealTypes','stRealPool','stRealSelected',
    'stPaintReal','stRealSelectFiles','stBody'].map(source).join('\n')
    + '\nasync ' + source('stLoadReal'), ctx);
  ctx.stPaintFiles = () => ctx.stPaintReal(false);
  return {ctx, el, files, boxes:() => el('stRealFiles').querySelectorAll('input'),
    body:() => JSON.parse(JSON.stringify(ctx.stBody()))};
}

test('choose five PDFs out of 200 without clearing 195 preselected files', async () => {
  const {ctx, el, boxes, body} = setup();
  await ctx.stLoadReal();
  assert.equal(boxes().filter(b => b.checked).length, 0);
  assert.equal(ctx.stRun.disabled, true);
  el('stRealTypes').querySelectorAll('input').find(b => b.value === 'png').checked = false;
  boxes().slice(0,5).forEach(box => {box.checked = true;});
  ctx.stPaintFiles();
  assert.deepEqual(body().cases, boxes().slice(0,5).map(b => b.value));
  assert.deepEqual(body().types, ['pdf']);
  assert.equal(ctx.stRun.disabled, false);
  assert.equal(el('stRealChoose').hidden, false);
  assert.equal(el('stRealFiles').hidden, false);
});

test('Random hides the picker and sends the entire type pool independently of manual selections', async () => {
  const {ctx, el, boxes, body} = setup();
  await ctx.stLoadReal();
  boxes()[0].checked = true;
  el('stRealTypes').querySelectorAll('input').find(b => b.value === 'png').checked = false;
  ctx.stRealPick = 'random'; ctx.stPaintFiles();
  assert.equal(el('stRealChoose').hidden, true);
  assert.equal(el('stRealFiles').hidden, true);
  assert.equal(ctx.stRealSelected().length, 200);
  assert.equal(ctx.stRun.disabled, false);
  assert.ok(!('cases' in body()));
  assert.match(page, /\.rt-files\[hidden\], #stRealChoose\[hidden\]\s*\{\s*display:\s*none;/);
  ctx.stRealPick = 'choose'; ctx.stPaintFiles();
  assert.equal(ctx.stRealSelected().length, 1);
});

test('search and Select visible respect types; Clear selection also clears hidden files', async () => {
  const {ctx, el, boxes, body} = setup();
  await ctx.stLoadReal();
  el('stRealSearch').value = 'INV-00';
  ctx.stPaintFiles(); ctx.stRealSelectFiles(true);
  assert.equal(body().cases.length, 10);
  assert.equal(boxes().filter(b => b.parentElement.style.display !== 'none').length, 10);
  el('stRealSearch').value = 'photo';
  el('stRealTypes').querySelectorAll('input').find(b => b.value === 'png').checked = false;
  ctx.stPaintFiles(); ctx.stRealSelectFiles(true);
  assert.equal(el('stRealSelectVisible').disabled, true);
  assert.equal(body().cases.length, 10);
  ctx.stRealSelectFiles(false);
  assert.equal(boxes().filter(b => b.checked).length, 0);
  assert.equal(ctx.stRun.disabled, true);
});

test('Refresh preserves selected files while new files remain unselected, and busy runs cannot change selection', async () => {
  const {ctx, files, boxes} = setup();
  await ctx.stLoadReal(); boxes()[0].checked = true;
  files.push({name:'new.pdf', type:'pdf', size_mb:2});
  await ctx.stLoadReal();
  assert.deepEqual(boxes().filter(b => b.checked).map(b => b.value), ['invoice/inv-000.pdf']);
  assert.equal(boxes().find(b => b.value === 'new.pdf').checked, false);
  ctx.stRunning = true; ctx.stRealSelectFiles(false);
  assert.equal(boxes()[0].checked, true);
});
