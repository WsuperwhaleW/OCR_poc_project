import {benchmarkSource} from './benchmark_source.mjs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
const start = page.indexOf('let documentLiveStates =');
const end = page.indexOf('\n}', page.indexOf('function stepEvent(', start)) + 2;

test('interleaved document events restore their own classification, steps and prompt',()=>{
  const ctx = vm.createContext({Map, classified:{doc_types:['OLD']}, extractSteps:null,
    extractPrefix:'old', tableLive:null, runningDoc:{document:1}, curMode:'none',
    tabFields:{classList:{contains:()=>false}}, fileSplit:null, segWalk:null,
    pagebar:{dataset:{},textContent:''}, enableGrouping(){},
    paintTableTab(){}, paintFields(){}, scheduleGrouping(){}});
  vm.runInContext(page.slice(start,end) + '\nthis.snapshot=()=>({classified,extractSteps,extractPrefix,tableLive,runningDoc});',ctx);
  const event = (document, event, rest={})=>ctx.stepEvent({event,document,documents:2,pages:[document],...rest});
  ctx.stepEvent({event:'documents_parallel',document_parallel:2});
  event(1,'document');event(1,'classified',{doc_types:['INVOICE']});
  event(1,'extract_steps',{steps:[{id:'a'},{id:'b'}],prompt_prefix:'document-one'});
  event(2,'document');event(2,'classified',{doc_types:['RECEIPT']});
  event(2,'extract_steps',{steps:[{id:'c'}],prompt_prefix:'document-two'});
  event(1,'extract_step',{step:1,id:'a',status:'done'});
  let state=ctx.snapshot();
  assert.deepEqual([...state.classified.doc_types],['INVOICE']);
  assert.equal(state.extractPrefix,'document-one');
  assert.deepEqual(state.extractSteps.map(s=>s.id),['a','b']);
  assert.equal(state.extractSteps[0].status,'done');
  assert.equal(state.extractSteps[1].status,'waiting');
  event(2,'extract_step',{step:1,id:'c',status:'done'});
  state=ctx.snapshot();
  assert.deepEqual([...state.classified.doc_types],['RECEIPT']);
  assert.equal(state.extractPrefix,'document-two');
  assert.deepEqual(state.extractSteps.map(s=>s.id),['c']);
  ctx.stepEvent({event:'documents_parallel',document_parallel:2});
  event(1,'document');
  assert.equal(ctx.snapshot().classified,null);
  assert.equal(ctx.snapshot().extractPrefix,'');
  ctx.stepEvent({event:'segment_progress',pages:100,pages_checked:67,parallel:4,starts_found:3});
  assert.match(ctx.pagebar.textContent,/67 \/ 100/);
  assert.match(ctx.pagebar.textContent,/4 workers/);
});
