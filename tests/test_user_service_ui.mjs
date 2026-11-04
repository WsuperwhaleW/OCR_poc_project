// The /user Service and API tabs (2026-11-04, worker-service plan step 2b).
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const read = name => readFileSync(new URL('templates/' + name, root), 'utf8');
const appSource = read('_user_app_script.html');
const serviceSource = read('_user_service_script.html');
const apiSource = read('_user_api_script.html');
const config = {ocr_concurrency:50,extraction_concurrency:64,detail:'medium',shape:'hybrid',extras:false,schema_first:false,
  fill_missing:true,validation_retry:false,loop_guard:true,no_cache:false,user_file_max:0,max_upload_mb:32,
  ocr_url:'http://127.0.0.1:8000',extraction_url:'http://127.0.0.1:8001'};

class Element {
  constructor(tag='div'){this.tag=tag;this.hidden=false;this.listeners={};this.children=[];this.value='';this.style={};
    this.dataset={};this.textContent='';const classes=new Set();this.classes=classes;
    this.classList={toggle:(name,on)=>{if(on??!classes.has(name))classes.add(name);else classes.delete(name);},
      add:name=>classes.add(name),remove:name=>classes.delete(name),contains:name=>classes.has(name)};}
  addEventListener(name,fn){this.listeners[name]=fn;}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=items;}
  setAttribute(name,value){this[name]=value;}
  removeAttribute(name){delete this[name];}
  get firstChild(){return this.children[0];}
  all(){return this.children.flatMap(child=>child instanceof Element?[child,...child.all()]:[]);}
  querySelectorAll(selector){
    const [tag,attr]=selector.split('[');
    return this.all().filter(e=>e.tag===tag&&(!attr||e.dataset[attr.replace(/^data-|\]$/g,'')]!==undefined));
  }
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  text(){return [this.textContent,...this.children.map(child=>child instanceof Element?child.text():child.textContent||'')].join(' ');}
}
const settle = async () => {for(let i=0;i<30;i++)await Promise.resolve();};
const json = body => ({ok:true,status:200,headers:{get:()=>null},json:async()=>body});

const overview = {
  generated_at:'2026-11-04T08:30:00+00:00',
  alerts:[{level:'error',text:'งานล้มเหลว 1 จาก 6 งานในชั่วโมงล่าสุด'}],
  health:{app_version:'dev-038fe4b',started_at:'2026-11-04T08:00:00+00:00',uptime_seconds:1800,data_dir:'/data/user',
    data_bytes:5_500_000,free_bytes:25e9,total_bytes:456e9,
    models:{ocr:{url:'http://ocr',kind:'vllm',model:'typhoon-ocr',available:true,reason:'',probe_age_seconds:4},
            extraction:{url:'http://ext',kind:null,model:null,available:null,reason:'',probe_age_seconds:null}}},
  queue:{open_jobs:3,oldest_wait_seconds:42,stages:{
    ocr:{waiting:2,running:1,running_recorded:1,concurrency:2,oldest_wait_seconds:42},
    extraction:{waiting:0,running:0,running_recorded:0,concurrency:2,oldest_wait_seconds:null}}},
  requests:{extraction:[{url:'http://ext:80',limit:128,active:3,peak:14,waiting_user:0,waiting_background:1}],
    extraction_limit:128,ocr:null,ocr_url:'http://ocr'},
  throughput:{hours:Array.from({length:24},(_,i)=>({hour:new Date(Date.UTC(2026,10,4,i)).toISOString(),completed:i%3,partial:i===23?2:0,failed:i===23?1:0})),
    last_hour:{completed:2,partial:2,failed:1,total:5},last_24h:{completed:20,partial:2,failed:1,total:23},
    last_7d:{completed:90,partial:5,failed:2,total:97},retries_24h:1},
  speed:{queue:{count:6,p50:0.1,p95:0.3,max:0.3},ocr:{count:6,p50:0.8,p95:1.9,max:1.9},extraction:null,total:{count:5,p50:2.5,p95:5.2,max:5.2}},
  quality:{finished:23,completed_rate:.87,partial_rate:.09,failed_rate:.04,unknown_type_rate:0,
    top_doc_types:[{code:'INVOICE',count:4},{code:'RECEIPT',count:1}],
    user_json:{checked:22,schema_valid_rate:.95,avg_warnings:.5,avg_missing_fields:2.1}},
  failures:[{id:'f'.repeat(32),filename:'broken_scan.pdf',status:'failed',finished_at:'2026-11-04T08:18:03+00:00',stage:'ocr',error:'Unsupported file'}],
  planned:[{id:'capacity',step:5,title:'Capacity & admission',shows:'…'},{id:'callbacks',step:6,title:'Callbacks',shows:'…'},
    {id:'retention',step:7,title:'Retention & storage',shows:'…'}],
};
const reference = {
  groups:[{id:'v1',title:'Worker API v1',note:'แผน'},{id:'user',title:'User portal API',note:'ใช้งานได้'}],
  callback:{status:'planned',title:'Callback เมื่องานเสร็จ',summary:'…',headers:[{name:'X-OCR-Event-Id',value:'uuid'}],
    retries:['30 วินาที'],body:{job_id:'x'},verify:{python:'def verify(): ...',javascript:'function verify() {}'}},
  endpoints:[
    {id:'v1-submit',group:'v1',status:'planned',method:'POST',path:'/api/v1/jobs',title:'ส่งไฟล์เข้าคิว',summary:'รับไฟล์',auth:'Bearer',
     headers:[{name:'Authorization',value:'Bearer <API token>',required:true}],params:[{name:'file',in:'form',type:'file',required:true,desc:'ไฟล์'}],
     multipart:true,responses:[{code:202,body:{job_id:'x'},note:'รับงาน',kind:'json'},{code:404,body:null,note:'ไม่พบ',kind:'empty'}],
     snippets:{curl:"curl -X POST '{BASE}/api/v1/jobs'",python:'import requests',javascript:"await fetch('{BASE}/api/v1/jobs')"}},
    {id:'settings-put',group:'user',status:'live',method:'PUT',path:'/api/user/settings',title:'บันทึก Settings',summary:'…',auth:'…',
     headers:[],params:[],body:{detail:'medium'},responses:[{code:200,body:{settings:{}},note:'',kind:'json'}],
     snippets:{curl:"curl -X PUT '{BASE}/api/user/settings' -H 'X-OCR-User-Token: {USER_TOKEN}'",python:'x',javascript:'y'}},
  ],
};

function page({hiddenDocument=false}={}){
  const items={},listeners={},timers=new Map(),calls=[];let serial=0;
  const el=id=>items[id] ||= new Element();
  for(const view of ['history','types','prompts','service','api'])el(view+'View').hidden=true;
  const document={hidden:hiddenDocument,listeners,addEventListener(name,fn){listeners[name]=fn;},getElementById:el,
    createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>({textContent:text}),querySelectorAll:()=>[]};
  const fetch=async url=>{
    calls.push(url);
    if(url.endsWith('/service/overview'))return json(overview);
    if(url.endsWith('/api-reference'))return json(reference);
    if(url.endsWith('/settings'))return json({settings:config});
    if(url.endsWith('/models'))return json({models:{ocr:{},extraction:{}}});
    if(url.endsWith('/stress'))return json({runs:[]});
    if(url.includes('/jobs?'))return json({jobs:[],counts:{},more:false});
    return json({});
  };
  const window={USER_BASE:'/ocr',USER_API_TOKEN:'page-token',location:{origin:'https://ocr.example'},
    addEventListener:(name,fn)=>listeners['window:'+name]=fn,renderUserOcr(){}};
  const ctx=vm.createContext({document,fetch,window,Date,console,Node:Element,
    setTimeout:(fn,ms)=>{timers.set(++serial,{fn,ms});return serial;},clearTimeout:id=>timers.delete(id)});
  for(const source of [serviceSource,apiSource,appSource])vm.runInContext(source,ctx);
  const polls=()=>[...timers.values()].filter(t=>t.ms===10000&&String(t.fn).includes('load')).length;
  return {el,document,listeners,timers,calls,window,polls};
}

test('the Service tab polls every 10 s only while it is showing',async()=>{
  const {el,calls,polls}=page();await settle();
  assert.equal(calls.filter(url=>url.endsWith('/service/overview')).length,0,'not fetched before the tab is opened');
  el('navService').listeners.click();await settle();
  assert.equal(el('serviceView').hidden,false);
  assert.deepEqual(calls.filter(url=>url.includes('/service/overview')),['/ocr/api/user/service/overview']);
  assert.equal(polls(),1);
  el('navUpload').listeners.click();
  assert.equal(polls(),0,'leaving the tab stops the poll');
});

test('a hidden browser tab stops the Service poll and a visible one resumes it',async()=>{
  const {el,document,polls,calls}=page();await settle();
  el('navService').listeners.click();await settle();
  document.hidden=true;document.listeners.visibilitychange();
  assert.equal(polls(),0);
  const before=calls.length;
  document.hidden=false;document.listeners.visibilitychange();await settle();
  assert.ok(calls.slice(before).includes('/ocr/api/user/service/overview'));
  assert.equal(polls(),1);
});

test('the Service tab draws every card, the alerts and the planned placeholders',async()=>{
  const {el}=page();await settle();
  el('navService').listeners.click();await settle();
  assert.equal(el('serviceError').hidden,true,'no error shown');
  const cards=el('serviceGrid').children;
  assert.equal(cards.length,10);
  const titles=cards.map(card=>card.children[0].textContent);
  assert.deepEqual(titles.slice(0,7),['สุขภาพระบบ','คิวตอนนี้','คำขอที่ส่งถึง model','ปริมาณงาน','ความเร็ว (24 ชั่วโมง)',
    'คุณภาพ (24 ชั่วโมง)','งานที่ล้มเหลวล่าสุด']);
  for(const card of cards.slice(7))assert.match(card.text(),/ยังไม่เปิดใช้ · ขั้นที่ [567]/);
  assert.equal(el('serviceAlerts').children.length,1);
  const text=el('serviceGrid').text();
  assert.match(text,/typhoon-ocr/);assert.match(text,/ยังไม่ได้ตรวจ/);
  assert.match(text,/กำลังส่ง 3 \/ 128/);assert.match(text,/ยังไม่มีเพดานคำขอ/);
  assert.match(text,/broken_scan\.pdf/);
  // 24 hourly groups in the chart.
  const chart=cards[3].children.find(child=>child.tag==='svg');
  assert.equal(chart.children.filter(child=>child.tag==='g').length,24);
});

test('a failure row opens that job',async()=>{
  const {el,window}=page();await settle();
  const opened=[];window.openUserJob=id=>opened.push(id);
  el('navService').listeners.click();await settle();
  const failures=el('serviceGrid').children[6];
  const button=failures.all().find(child=>child.tag==='button');
  button.listeners.click();
  assert.deepEqual(opened,['f'.repeat(32)]);
});

test('the API tab lists v1 before the User API, shows the first endpoint and fills in the base URL',async()=>{
  const {el,calls}=page();await settle();
  el('navApi').listeners.click();await settle();
  assert.equal(calls.filter(url=>url.endsWith('/api-reference')).length,1);
  const groups=el('apiList').children;
  assert.deepEqual(groups.map(group=>group.children[0].textContent),['Worker API v1','User portal API']);
  assert.equal(groups[0].children.filter(child=>child.tag==='button').length,2,'v1 endpoint + the callback');
  const detail=el('apiDetail').text();
  assert.match(detail,/ส่งไฟล์เข้าคิว/);assert.match(detail,/แผน · ยังไม่เปิดใช้/);
  assert.match(detail,/curl -X POST 'https:\/\/ocr\.example\/ocr\/api\/v1\/jobs'/);
  assert.equal(el('apiBase').textContent,'https://ocr.example/ocr');
  // Opening the tab again does not fetch the reference again.
  el('navUpload').listeners.click();el('navApi').listeners.click();await settle();
  assert.equal(calls.filter(url=>url.endsWith('/api-reference')).length,1);
});

test('the API filter narrows the list and a write example carries the page token',async()=>{
  const {el}=page();await settle();
  el('navApi').listeners.click();await settle();
  el('apiFilter').value='settings';el('apiFilter').listeners.input();
  const groups=el('apiList').children;
  assert.deepEqual(groups.map(group=>group.children[0].textContent),['User portal API']);
  groups[0].children.find(child=>child.tag==='button').listeners.click();
  assert.match(el('apiDetail').text(),/X-OCR-User-Token: page-token/);
  el('apiFilter').value='no such endpoint';el('apiFilter').listeners.input();
  assert.match(el('apiList').text(),/ไม่พบ endpoint/);
});

test('neither new tab hard-codes an /api/ path',()=>{
  for(const source of [serviceSource,apiSource]){
    assert.ok(!source.includes("fetch('/api/"));
    for(const match of source.matchAll(/['"]\/api\//g))
      assert.ok(source.slice(Math.max(0,match.index-20),match.index).endsWith('window.USER_BASE + '),'paths go through window.USER_BASE');
  }
});
