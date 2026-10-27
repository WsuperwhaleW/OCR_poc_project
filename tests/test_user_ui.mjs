import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const page = readFileSync(new URL('templates/index.html', root), 'utf8');
const settingsSource = readFileSync(new URL('templates/_user_settings_script.html', root), 'utf8');
const appSource = readFileSync(new URL('templates/_user_app_script.html', root), 'utf8');
const renderSource = readFileSync(new URL('templates/_user_render_script.html', root), 'utf8');
const config = {ocr_concurrency:50,extraction_concurrency:64,detail:'medium',shape:'hybrid',
  extras:false,schema_first:false,loop_guard:true,no_cache:false,user_file_max:0,max_upload_mb:32,
  ocr_url:'http://127.0.0.1:8000',extraction_url:'http://127.0.0.1:8001'};
const clean = value => JSON.parse(JSON.stringify(value));

class Element {
  constructor(tag='div'){this.tag=tag;this.hidden=false;this.listeners={};this.children=[];this.value='';this.disabled=false;
    this.style={};
    this.classList={toggle(){},add(){},remove(){}};}
  addEventListener(name,fn){this.listeners[name]=fn;}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  setAttribute(name,value){this[name]=value;}
  removeAttribute(name){delete this[name];}
}
function dom(){
  const items={};
  const el=id=>items[id] ||= new Element();
  return {items,el,document:{getElementById:el,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),querySelectorAll:()=>[]}};
}
function response(body, ok=true){return {ok,status:ok?200:400,json:async()=>body};}

test('Settings owns the main view and hides benchmark actions despite saved Stress results',()=>{
  const {el,document}=dom();document.body={classList:{toggle(){}}};
  const names=['source','ocr','fields','random','stress','schema','settings'];
  const tabs=names.map(pane=>({dataset:{pane},classList:{toggle(){}}}));
  for(const name of names)el(name+'Actions').querySelector=()=>({});
  let loads=0;
  const ctx=vm.createContext({document,sideTabs:{querySelectorAll:()=>tabs},selectedPane:'stress',
    stressPaneOpen:true,historyOpen:true,scOpen:false,stressView:'history',cmpOverride:null,
    window:{loadUserSettings(){loads++;}},enableCompare(){},paintSchema(){},historyLoad(){},stPaint(){},rtPaint(){}});
  for(const name of ['showPane','paintMain']){
    const start=page.indexOf('function '+name+'('),end=page.indexOf('\n}',start)+2;
    vm.runInContext(page.slice(start,end),ctx);
  }
  ctx.showPane('settings');
  assert.equal(ctx.selectedPane,'settings');assert.equal(loads,1);
  assert.equal(el('settingsCard').hidden,false);
  for(const id of ['historyCard','stCard','rtCard','schemaCard','resultCard','compareCard','testActions','runlogCard'])assert.equal(el(id).hidden,true,id);
  ctx.showPane('source');assert.equal(el('settingsCard').hidden,true);assert.equal(el('resultCard').hidden,false);
  ctx.cmpOverride={case:'sol001',page:1};ctx.showPane('stress');
  assert.equal(el('compareCard').hidden,false,'Stress evidence must keep its Compare view');
  ctx.showPane('settings');assert.equal(el('compareCard').hidden,true,'Settings must hide old evidence');
});

function settingsSetup(fetch){
  const {el,document}=dom();const form=el('userSettingsForm');
  const controls=Object.fromEntries(Object.entries(config).map(([name,value])=>{
    const input=new Element('input');input.name=name;input.type=typeof value==='boolean'?'checkbox':typeof value==='number'?'number':'text';return[name,input];
  }));
  controls.shape=new Element('select');controls.shape.name='shape';controls.shape.type='select-one';
  controls.detail=new Element('select');controls.detail.name='detail';controls.detail.type='select-one';
  form.elements={...controls,namedItem:name=>controls[name]};
  el('userSettingsFields').querySelectorAll=()=>Object.values(controls);
  el('settingsCard').hidden=true;
  const window={API:'/ocr/api',OCR_BASE:'/ocr',USER_API_TOKEN:'page-proof'};
  const ctx=vm.createContext({document,window,fetch});vm.runInContext(settingsSource,ctx);
  return {window,el,controls,form};
}

test('Settings load and Save round-trip numeric and checkbox values to the prefixed API',async()=>{
  const calls=[];
  const {window,el,controls,form}=settingsSetup(async(url,options)=>{
    calls.push({url,options});return response({settings:options.method==='PUT'?JSON.parse(options.body):config,upload_ceiling_mb:32});
  });
  await window.loadUserSettings();
  assert.equal(controls.ocr_concurrency.value,50);assert.equal(controls.loop_guard.checked,true);
  assert.equal(el('userSettingsFields').disabled,false);
  controls.ocr_concurrency.value='12';controls.extras.checked=true;
  await form.listeners.submit({preventDefault(){}});
  const saved=JSON.parse(calls[1].options.body);
  assert.equal(saved.ocr_concurrency,12);assert.equal(saved.extras,true);assert.equal(saved.no_cache,false);
  assert.equal(calls[1].url,'/ocr/api/user/settings');assert.match(el('userSettingsNotice').textContent,/Saved/);
  assert.equal(calls[1].options.headers['X-OCR-User-Token'],'page-proof');
  controls.shape.value='single';controls.shape.listeners.change();assert.equal(controls.schema_first.disabled,true);
});

test('a failed Settings load can retry and a failed Save preserves the unsaved fields',async()=>{
  let fail=true;
  const {window,el,controls,form}=settingsSetup(async(_url,options)=>{
    if(fail)return response({error:'temporarily unavailable'},false);
    return response({settings:config,upload_ceiling_mb:32});
  });
  await window.loadUserSettings();assert.equal(el('userSettingsRetry').hidden,false);
  fail=false;await el('userSettingsRetry').listeners.click();
  controls.ocr_concurrency.value='7';fail=true;
  await form.listeners.submit({preventDefault(){}});
  assert.equal(controls.ocr_concurrency.value,'7');assert.equal(el('userSettingsFields').disabled,false);
  assert.match(el('userSettingsNotice').textContent,/temporarily unavailable/);
});

test('User app uploads each selected file independently and shows shared history',async()=>{
  const {el,document}=dom(),calls=[],timers=new Map();let timer=0;
  class FormData {constructor(){this.values={};}append(key,value){this.values[key]=value;}}
  const job=id=>({id,filename:id+'.png',created_at:'2026-10-26T00:00:00Z',status:'completed',progress:'done',configuration:config,
    result:{text:'OCR result',pages:['OCR result'],timing:{ocr:1,extraction:2},extracted:{fields:{buyer_tax_id:'1234'}}}});
  const fetch=async(url,options={})=>{
    calls.push({url,options});
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/batches'))return response({batch_id:'batch-id'});
    if(url.endsWith('/models'))return response({models:{ocr:{model:'reader',available:true},extraction:{model:'extractor',text_available:true}}});
    if(options.method==='POST')return response({job:job(options.body.values.upload_id)});
    if(url.includes('/jobs?'))return response({jobs:[job('shared-job')],counts:{queued_ocr:2},more:false});
    return response({job:job(url.split('/').at(-1))});
  };
  const ctx=vm.createContext({document,window:{USER_BASE:'/ocr',USER_API_TOKEN:'page-proof',renderUserOcr(target,pages,text){target.textContent=text;},addEventListener(){}},fetch,FormData,
    URL:{createObjectURL:()=> 'blob:preview',revokeObjectURL(){}},
    setTimeout:fn=>{timers.set(++timer,fn);return timer;},clearTimeout:id=>timers.delete(id),Date,console});
  vm.runInContext(appSource,ctx);for(let i=0;i<8;i++)await Promise.resolve();
  const files=[{name:'first.png',size:20*1024*1024},{name:'second.png',size:20*1024*1024}];
  el('fileInput').listeners.change({target:{files}});
  assert.equal(el('runButton').disabled,false);
  await el('uploadForm').listeners.submit({preventDefault(){}});
  const uploads=calls.filter(c=>c.options.body instanceof FormData);
  assert.equal(uploads.length,2);
  assert.ok(uploads.every(c=>c.url==='/ocr/api/user/jobs'));
  assert.deepEqual(uploads.map(c=>c.options.body.values.file.name),['first.png','second.png']);
  assert.ok(calls.filter(c=>c.options.method==='POST').every(c=>c.options.headers['X-OCR-User-Token']==='page-proof'));
  assert.match(el('uploadMessage').textContent,/2\/2/);
  assert.equal(el('historyRows').children.length,1);
  assert.equal(el('batchJobs').children.length,2);
  await el('historyRows').children[0].children.at(-1).children[0].listeners.click();
  assert.equal(el('resultFile').textContent,'shared-job.png');
  await el('uploadForm').listeners.submit({preventDefault(){}});
  for(let i=0;i<8;i++)await Promise.resolve();
  assert.notEqual(el('resultFile').textContent,'shared-job.png','a new batch must replace the previous selected result');
});

test('model status uses each capability, refreshes served names and distinguishes API failures',async()=>{
  const {el,document}=dom(),calls=[],timers=new Map(),events={};let timer=0,version=0,fail=false;
  const fetch=async(url,options={})=>{
    calls.push({url,options});
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models')){
      if(fail)throw new Error('network unavailable');
      return response({models:{ocr:{model:'ocr-'+version,available:true},
        extraction:{model:'extract-'+version,available:true,text_available:version>0,text_reason:'Text model unavailable'}}});
    }
    return response({jobs:[],counts:{},more:false});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof',addEventListener:(name,fn)=>events[name]=fn},fetch,
    setTimeout:(fn,ms)=>{timers.set(++timer,{fn,ms});return timer;},clearTimeout:id=>timers.delete(id),Date,console}));
  for(let i=0;i<10;i++)await Promise.resolve();
  assert.equal(el('ocrModelStatus').children.at(-1).textContent,'Online');
  assert.equal(el('extractionModelStatus').children.at(-1).textContent,'Offline');
  assert.match(el('extractionModelStatus').title,/Text model unavailable/);
  assert.ok([...timers.values()].some(t=>t.ms===30000));
  version=1;await el('refreshModels').listeners.click();
  assert.equal(el('extractionModelStatus').children[2].textContent,'extract-1');
  assert.equal(el('extractionModelStatus').children.at(-1).textContent,'Online');
  fail=true;await el('refreshModels').listeners.click();
  assert.equal(el('ocrModelStatus').children.at(-1).textContent,'Unknown');
  assert.equal(el('refreshModels').disabled,false);
  const probes=calls.filter(c=>c.url.endsWith('/models'));
  assert.ok(probes.every(c=>c.url==='/ocr/api/user/models' && c.options.headers['X-OCR-User-Token']==='proof'));
  events.pagehide();assert.equal(timers.size,0);
});

test('queued jobs retain measured stages, offer three result views and preview selected files without creating jobs',async()=>{
  const {el,document}=dom(),calls=[],revoked=[];let objectCount=0;
  let job={id:'job-id',filename:'three.pdf',created_at:'2026-10-27T00:00:00Z',status:'queued_ocr',progress:'กำลังรอคิว OCR',
    timeline:{queued_ocr:{started_at:'2026-10-27T00:00:00Z',progress:'กำลังรอคิว OCR'}},result:{}};
  class Clock extends Date {static now(){return Date.parse('2026-10-27T00:00:30Z');}}
  const fetch=async(url,options={})=>{
    calls.push({url,options});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/previews'))return response({preview_id:'temporary',pages:3});
    if(url.endsWith('/preview/info'))return response({pages:3});
    if(url.includes('/jobs?'))return response({jobs:[job],counts:{queued_ocr:1},more:false});
    return response({job});
  };
  const window={USER_BASE:'/ocr',USER_API_TOKEN:'proof',renderUserOcr(target,pages,text){target.textContent=text;},addEventListener(){}};
  class FormData {constructor(){this.values={};}append(key,value){this.values[key]=value;}}
  vm.runInContext(appSource,vm.createContext({document,window,fetch,Date:Clock,console,setTimeout:()=>1,clearTimeout(){},
    FormData,
    URL:{createObjectURL:()=> 'blob:local-'+(++objectCount),revokeObjectURL:url=>revoked.push(url)}}));
  for(let i=0;i<10;i++)await Promise.resolve();
  el('fileInput').listeners.change({target:{files:[{name:'local.pdf',size:1},{name:'picture.png',size:1}]}});
  for(let i=0;i<10;i++)await Promise.resolve();
  assert.equal(el('pdfPreview').children.length,3);
  assert.equal(el('pdfPreview').children[2].children[1].src,'/ocr/api/user/previews/temporary/pages/3');
  assert.equal(calls.filter(c=>c.url.endsWith('/jobs')||c.url.endsWith('/batches')).length,0);
  el('selectedFiles').children[1].children[0].listeners.click();
  assert.deepEqual(revoked,['blob:local-1']);assert.equal(el('imagePreview').hidden,false);
  el('zoomIn').listeners.click();assert.equal(el('imagePreview').style.width,'125%');
  const open=()=>el('historyRows').children[0].children.at(-1).children[0].listeners.click();
  await open();
  assert.equal(el('stageTimeline').children[0].children[0].children[1].textContent,'30.0 วิ');
  assert.equal(el('stageTimeline').children[0].children[1].textContent,'กำลังรอคิว…');
  for(let i=0;i<10;i++)await Promise.resolve();
  assert.equal(el('pdfPreview').children[2].children[1].src,'/ocr/api/user/jobs/job-id/preview/pages/3');
  job={...job,status:'ocr',progress:'กำลังอ่านเอกสารหน้า 1/3',timeline:{
    queued_ocr:{...job.timeline.queued_ocr,finished_at:'2026-10-27T00:00:30Z'},ocr:{started_at:'2026-10-27T00:00:30Z'}}};
  await open();
  assert.equal(el('stageTimeline').children[0].children[0].children[1].textContent,'30.0 วิ');
  assert.equal(el('stageTimeline').children[1].children[1].textContent,'กำลังอ่านเอกสารหน้า 1/3');
  job={...job,status:'completed',result:{text:'# Heading',pages:['# Heading'],extracted:{fields:{subtotal:'100'}}}};
  await open();el('jsonTab').listeners.click();
  assert.equal(el('jsonResult').hidden,false);assert.equal(el('textResult').hidden,true);
  assert.equal(JSON.parse(el('jsonResult').textContent).fields.subtotal,'100');
  el('fieldsTab').listeners.click();assert.equal(el('fieldsResult').hidden,false);assert.equal(el('jsonResult').hidden,true);
  el('textTab').listeners.click();assert.equal(el('textResult').hidden,false);assert.equal(el('textTab')['aria-selected'],'true');
});

test('OCR Rendered builds headings and tables while retaining untrusted markup as text',()=>{
  const {document}=dom(),window={},target=new Element();
  vm.runInContext(renderSource,vm.createContext({document,window}));
  window.renderUserOcr(target,['# Invoice\n**Total**\n| Item | Amount |\n| --- | --- |\n| A | 100 |\n<img src="https://invalid/track" onerror="alert(1)">'],null);
  const all=[];function visit(el){all.push(el);for(const child of el.children||[])visit(child);}visit(target);
  assert.equal(all.filter(e=>e.tag==='table').length,1);assert.equal(all.filter(e=>e.tag==='th').length,2);
  assert.equal(all.filter(e=>e.tag==='td').length,2);assert.equal(all.filter(e=>e.tag==='h1').length,1);
  assert.equal(all.filter(e=>e.tag==='img'||e.tag==='script').length,0);
  assert.ok(all.some(e=>e.textContent?.includes('<img src=')));
});

test('portal base paths support root, trailing slash and nested reverse-proxy mounts',()=>{
  for(const [path,base]of [['/user',''],['/user/',''],['/ocr/user','/ocr'],['/tools/ocr/user/','/tools/ocr']]){
    assert.equal(path.replace(/\/user\/?$/,''),base);
  }
  assert.ok(!appSource.includes("fetch('/api/"));
  assert.ok(!settingsSource.includes("fetch('/api/"));
});
