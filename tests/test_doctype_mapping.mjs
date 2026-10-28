import {benchmarkSource} from './benchmark_source.mjs';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const script = readFileSync(new URL('../templates/_doctype_mapping_script.html', import.meta.url), 'utf8');
function setup() {
  const listeners = {}, window = {};
  vm.runInNewContext(script, {window, document:{addEventListener(name, fn){listeners[name]=fn;}}});
  return {render:window.renderDocTypeMapping, listeners};
}
const mapping = {version:'7.0',source:'RunLog',counts:{types:{extra:1},fields:{missing:1}},
  types:[{user_code:'',user_name:'',ours:['RTV_SHIP_CREDIT'],status:'extra',note:'คง RTV'}],
  fields:[{user_field:'documents[].seller.name',value_type:'string',ours:[],status:'missing',note:'ยังขาด'}]};
test('shared renderer explains mapping limits and escapes all source-provided labels', () => {
  const {render}=setup(), data=structuredClone(mapping);
  data.types[0].note='<img src=x onerror=alert(1)>';
  const html=render(data);
  assert.match(html,/RTV_SHIP_CREDIT/);assert.match(html,/documents\[\]\.seller.name/);
  assert.match(html,/เรามีเพิ่ม/);assert.match(html,/ยังขาด/);assert.match(html,/ไม่เปลี่ยน prompt/);
  assert.match(html,/&lt;img/);assert.doesNotMatch(html,/<img/);
  assert.match(html,/data-dm-view="fields" hidden/);
  assert.equal(render(null),'');
});
test('status filter hides non-matching rows and reports an empty result for each section independently', () => {
  const {listeners}=setup();
  function group(statuses){const rows=statuses.map(dmStatus=>({dataset:{dmStatus},hidden:false})),empty={hidden:true};
    return {rows,empty,querySelectorAll:()=>rows,querySelector:()=>empty};}
  const types=group(['mapped','extra']),fields=group(['missing','partial']);
  const root={querySelectorAll:()=>[types,fields]},target={matches:()=>true,closest:()=>root,value:'extra'};
  listeners.change({target});assert.deepEqual(types.rows.map(row=>row.hidden),[true,false]);
  assert.equal(fields.empty.hidden,false);assert.equal(types.empty.hidden,true);
  target.value='all';listeners.change({target});assert.equal(fields.rows.some(row=>row.hidden),false);
});
test('switching mapping tables stays inside the clicked panel', () => {
  const {listeners}=setup();
  const tabs=['types','fields'].map(dmTab=>({dataset:{dmTab},setAttribute(name,value){this[name]=value;}}));
  const groups=['types','fields'].map(dmView=>({dataset:{dmView}}));
  const root={querySelectorAll:selector=>selector==='[data-dm-tab]'?tabs:groups};
  tabs[1].closest=()=>root;listeners.click({target:{closest:()=>tabs[1]}});
  assert.equal(groups[0].hidden,true);assert.equal(groups[1].hidden,false);
  assert.equal(tabs[1]['aria-pressed'],'true');
});
test('both Doc types views reuse the reference renderer; it has no model or storage calls', () => {
  const index=benchmarkSource();
  const user=readFileSync(new URL('../templates/user.html',import.meta.url),'utf8');
  const app=readFileSync(new URL('../templates/_user_app_script.html',import.meta.url),'utf8');
  assert.match(index,/_doctype_mapping_script.html/);assert.match(user,/_doctype_mapping_script.html/);
  assert.match(index,/renderDocTypeMapping\?\.\(d.user_mapping\)/);assert.match(app,/renderDocTypeMapping\(schema.user_mapping\)/);
  assert.doesNotMatch(script,/fetch\(|localStorage|sessionStorage/);
});
