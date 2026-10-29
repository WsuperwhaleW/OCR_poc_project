import {benchmarkSource} from './benchmark_source.mjs';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const page = benchmarkSource();
const settingsSource = readFileSync(new URL('templates/_user_settings_script.html', root), 'utf8');
const appSource = readFileSync(new URL('templates/_user_app_script.html', root), 'utf8');
const promptsSource = readFileSync(new URL('templates/_user_prompts_script.html', root), 'utf8');
const renderSource = readFileSync(new URL('templates/_user_render_script.html', root), 'utf8');
const config = {ocr_concurrency:50,extraction_concurrency:64,detail:'medium',shape:'hybrid',
  extras:false,schema_first:false,fill_missing:true,validation_retry:false,loop_guard:true,no_cache:false,user_file_max:0,max_upload_mb:32,
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

const settle = async () => {for(let i=0;i<20;i++)await Promise.resolve();};
function lifecycleSetup(detailFetch) {
  const {el,document}=dom(),events={},timers=new Map(),calls=[];let serial=0;
  const job={id:'job-a',filename:'audit.png',created_at:'2026-10-28T00:00:00Z',
    status:'ocr',progress:'reading',configuration:config,result:{pages:[],text:''}};
  class FormData {constructor(){this.values={};}append(k,v){this.values[k]=v;}}
  const fetch=async(url,options={})=>{
    calls.push(url);
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.endsWith('/stress'))return response({runs:[]});
    if(url.includes('/jobs?'))return response({jobs:[job,{...job,id:'job-b'}],counts:{},more:false});
    if(url.endsWith('/batches'))return response({batch_id:'batch'});
    if(url.endsWith('/jobs'))return response({job});
    return detailFetch ? detailFetch(url,job) : response({job});
  };
  el('jobResult').hidden=true;
  vm.runInContext(appSource,vm.createContext({document,fetch,FormData,Date,console,
    window:{USER_BASE:'/ocr',addEventListener:(name,fn)=>events[name]=fn,renderUserOcr(){}},
    URL:{createObjectURL:()=> 'blob:audit',revokeObjectURL(){}},
    setTimeout:(fn,ms)=>{timers.set(++serial,{fn,ms});return serial;},clearTimeout:id=>timers.delete(id)}));
  return {el,events,timers,calls,job,
    open:index=>el('historyRows').children[index].children.at(-1).children[0].listeners.click()};
}

test('BFCache restore resumes history, selected job and batch polling once',async()=>{
  const {el,events,timers,calls}=lifecycleSetup();await settle();
  el('fileInput').listeners.change({target:{files:[{name:'audit.png',size:10}]}});
  await el('uploadForm').listeners.submit({preventDefault(){}});await settle();
  const delays=()=>[...timers.values()].map(t=>t.ms).sort((a,b)=>a-b);
  assert.deepEqual(delays(),[1200,2000,5000,10000,30000]);
  events.pagehide({persisted:true});assert.equal(timers.size,0);
  const previous=calls.length;events.pageshow({persisted:true});await settle();
  assert.ok(calls.slice(previous).some(url=>url.includes('/jobs?')));
  assert.equal(calls.slice(previous).filter(url=>url.endsWith('/jobs/job-a')).length,2);
  assert.deepEqual(delays(),[1200,2000,5000,10000,30000]);
  events.pageshow({persisted:true});await settle();
  assert.deepEqual(delays(),[1200,2000,5000,10000,30000]);
});

test('detail response finishing while page is hidden does not restart timers',async()=>{
  let finish;
  const {events,timers,open}=lifecycleSetup((_url,job)=>new Promise(resolve=>{finish=()=>resolve(response({job}));}));
  await settle();const pending=open(0);await settle();
  events.pagehide({persisted:true});finish();await pending;await settle();
  assert.equal(timers.size,0);
  events.pageshow({persisted:true});await settle();finish();await settle();
  assert.equal([...timers.values()].filter(t=>t.ms===1200).length,1);
});

test('first detail load failure is visible outside the hidden result and can retry',async()=>{
  let fail=true;
  const {el,open,timers}=lifecycleSetup((_url,job)=>{
    if(fail)throw new Error('network unavailable');return response({job});
  });await settle();await open(0);
  const markup=readFileSync(new URL('templates/user.html',root),'utf8');
  assert.ok(markup.indexOf('id="jobLoadMessage"')<markup.indexOf('id="jobResult"'));
  assert.equal(el('jobResult').hidden,true);
  assert.equal(el('jobLoadMessage').hidden,false);
  assert.match(el('jobLoadMessage').textContent,/network unavailable/);
  assert.equal(el('jobLoadRetry').hidden,false);
  assert.equal([...timers.values()].filter(t=>t.ms===4000).length,1);
  fail=false;el('jobLoadRetry').listeners.click();await settle();
  assert.equal(el('jobResult').hidden,false);assert.equal(el('jobLoadMessage').hidden,true);
  assert.equal(el('jobLoadRetry').hidden,true);
  assert.equal([...timers.values()].filter(t=>t.ms===4000).length,0);
});

test('HTML 404 for a missing job is explained without endless automatic retries',async()=>{
  const {el,open,timers}=lifecycleSetup(()=>({ok:false,status:404,json:async()=>{throw Error('HTML');}}));
  await settle();await open(0);
  assert.equal(el('jobLoadMessage').hidden,false);
  assert.match(el('jobLoadMessage').textContent,/ไม่พบงานนี้/);
  assert.equal([...timers.values()].filter(t=>t.ms===4000).length,0);
});

test('a late error from the previous selection cannot overwrite the current job',async()=>{
  let reject;
  const {el,open,timers}=lifecycleSetup((url,job)=>url.endsWith('job-a')
    ?new Promise((_resolve,no)=>{reject=no;}) :response({job:{...job,id:'job-b',filename:'second.png',status:'completed'}}));
  await settle();const previous=open(0);await settle();await open(1);
  reject(new Error('old request failed'));await previous;
  assert.equal(el('resultFile').textContent,'second.png');
  assert.equal(el('jobLoadMessage').hidden,true);
  assert.equal([...timers.values()].filter(t=>t.ms===4000).length,0);
});

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
  controls.fill_missing.checked=false;controls.validation_retry.checked=true;
  await form.listeners.submit({preventDefault(){}});
  const saved=JSON.parse(calls[1].options.body);
  assert.equal(saved.ocr_concurrency,12);assert.equal(saved.extras,true);assert.equal(saved.no_cache,false);
  assert.equal(saved.fill_missing,false);assert.equal(saved.validation_retry,true);
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

test('Key-Value renders row tables, footer and document types without changing saved JSON',async()=>{
  const {el,document}=dom();
  const extracted={documents:[{pages:[1],doc_types:['RECEIPT','TAX_INVOICE'],doc_type_heading:'ใบเสร็จรับเงิน / ใบกำกับภาษี',
    fields:{buyer_tax_id:'0123456789012',line_items:[],other_fields:[{name:'หมายเหตุ',value:'<img onerror="boom">'}]},
    item_table:{columns:['เลขที่เอกสาร','จำนวนเงิน'],rows:[['INV-1',0],['INV-2','100.00','extra cell']],row_pages:[1,2],
      footer:[{text:'รวม 100.00',page:2}],is_master_table:true}},
    {pages:[2,3],doc_type:'BILLING_NOTE',fields:{subtotal:'100.00'},item_table:{error:'No table',rows:[]}}]};
  const original=JSON.stringify(extracted);
  const job={id:'job',filename:'test.png',created_at:'2026-10-27',status:'completed',result:{extracted}};
  const fetch=async(url)=>{
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.includes('/jobs?'))return response({jobs:[job],counts:{},more:false});
    return response({job});
  };
  const window={USER_BASE:'/ocr',USER_DOC_TYPES:{RECEIPT:'receipt',TAX_INVOICE:'tax invoice',BILLING_NOTE:'billing note'},renderUserOcr(){},addEventListener(){}};
  vm.runInContext(appSource,vm.createContext({document,window,fetch,Date,setTimeout:()=>1,clearTimeout(){}}));
  for(let i=0;i<10;i++)await Promise.resolve();
  await el('historyRows').children[0].children.at(-1).children[0].listeners.click();
  const walk=element=>[element,...(element.children||[]).flatMap(walk)];
  const fields=walk(el('fieldsResult')),summary=walk(el('docTypesSummary'));
  assert.equal(el('docTypesPanel').hidden,false);
  assert.ok(summary.some(e=>e.textContent==='receipt'));assert.ok(summary.some(e=>e.textContent==='tax invoice'));
  assert.ok(summary.some(e=>e.textContent==='billing note'));assert.ok(summary.some(e=>e.textContent==='หน้า 2, 3'));
  assert.equal(fields.filter(e=>e.tag==='table').length,2);
  assert.ok(fields.some(e=>e.tag==='td'&&e.textContent==='0'));
  assert.ok(fields.some(e=>e.tag==='td'&&e.textContent==='extra cell'));
  assert.ok(fields.some(e=>e.tag==='th'&&e.textContent==='คอลัมน์ 3'));
  assert.ok(fields.some(e=>e.textContent==='รวม 100.00'));assert.ok(fields.some(e=>e.textContent==='No table'));
  assert.ok(fields.some(e=>e.textContent==='<img onerror="boom">'));assert.equal(fields.filter(e=>e.tag==='img').length,0);
  assert.ok(fields.some(e=>e.className==='field-key'&&e.textContent==='Buyer tax ID'));
  assert.ok(fields.some(e=>e.className==='field-key'&&e.textContent==='Billed amount'));
  assert.ok(fields.some(e=>e.textContent==='Other fields'));
  assert.ok(fields.some(e=>e.textContent==='หมายเหตุ'));
  assert.equal(fields.filter(e=>e.tag==='pre').length,0);
  assert.equal(el('jsonResult').textContent,JSON.stringify(extracted,null,2));
  assert.equal(JSON.stringify(extracted),original);
});

test('portal base paths support root, trailing slash and nested reverse-proxy mounts',()=>{
  for(const [path,base]of [['/user',''],['/user/',''],['/ocr/user','/ocr'],['/tools/ocr/user/','/tools/ocr']]){
    assert.equal(path.replace(/\/user\/?$/,''),base);
  }
  assert.ok(!appSource.includes("fetch('/api/"));
  assert.ok(!settingsSource.includes("fetch('/api/"));
});

test('User JSON loads only on selection, reports Draft and preserves the legacy JSON and prefixed download',async()=>{
  const {el,document}=dom();let conversions=0,fail=true;
  const extracted={doc_types:['RTV_SHIP_CREDIT'],fields:{document_number:'RTV-1',seller_branch_location:'มหาชัย'}};
  const job={id:'job',filename:'test.png',status:'completed',result:{extracted}};
  const candidate={schemaVersion:'7.0',file:{fileName:'test.png'},documents:[]};
  const fetch=async url=>{
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.includes('/jobs?'))return response({jobs:[job],counts:{},more:false});
    if(url.endsWith('/user-json')){conversions++;if(fail)return response({error:'retry conversion'},false);
      return response({data:candidate,schema_valid:false,schema_errors:[{path:'documents',message:'missing documents'}],
        warnings:[],missing_fields:['documents[0].buyer.name'],unmapped:[{seller_branch_location:'มหาชัย'}]});}
    return response({job});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',renderUserOcr(){},addEventListener(){}},fetch,Date,setTimeout:()=>1,clearTimeout(){}}));
  const settle=async()=>{for(let i=0;i<15;i++)await Promise.resolve();};
  await settle();await el('historyRows').children[0].children.at(-1).children[0].listeners.click();
  assert.equal(conversions,0);
  assert.equal(el('exportLink').href,'/ocr/api/user/jobs/job/export');
  assert.equal(el('userJsonDownload').hidden,false);
  assert.equal(el('userJsonDownload').href,'/ocr/api/user/jobs/job/user-json?download=1');
  assert.equal(el('userJsonReportDownload').href,'/ocr/api/user/jobs/job/user-json?download=report');
  const original=el('jsonResult').textContent;
  el('userJsonTab').listeners.click();await settle();assert.equal(el('userJsonRetry').hidden,false);
  fail=false;el('userJsonRetry').listeners.click();await settle();
  assert.equal(el('userJsonResult').hidden,false);assert.equal(el('jsonResult').hidden,true);
  assert.match(el('userJsonNotice').textContent,/Draft/);assert.equal(el('userJsonDetails').hidden,false);
  assert.equal(el('userJsonCounts').textContent,'ผิดโครงสร้าง 1 · คำเตือน 0 · ข้อมูลเสริมที่ยังไม่มี 1');
  assert.equal(el('userJsonSchemaGroup').hidden,false);assert.equal(el('userJsonWarningGroup').hidden,true);
  assert.equal(el('userJsonMissingGroup').hidden,false);
  assert.equal(el('userJsonCode').textContent,JSON.stringify(candidate,null,2));
  assert.equal(el('userJsonDownload').href,'/ocr/api/user/jobs/job/user-json?download=1');
  assert.match(el('userJsonUnmapped').textContent,/มหาชัย/);assert.equal(el('jsonResult').textContent,original);
  el('jsonTab').listeners.click();el('userJsonTab').listeners.click();await settle();assert.equal(conversions,2);
});

test('both JSON download choices are in the shared toolbar outside the User JSON tab',()=>{
  const markup=readFileSync(new URL('templates/user.html',root),'utf8');
  const toolbar=markup.slice(markup.indexOf('class="toolbar result-tools"'),markup.indexOf('id="textResult"'));
  assert.match(toolbar,/id="exportLink">JSON ปกติ ↓/);
  assert.match(toolbar,/id="userJsonDownload" hidden>JSON ของ User ↓/);
  assert.equal((markup.match(/id="userJsonDownload"/g)||[]).length,1);
});

test('schema-valid User JSON separates data warnings from optional gaps and never claims verified accuracy',async()=>{
  for(const hasWarning of [false,true]){
    const {el,document}=dom();
    const job={id:'review',filename:'review.png',status:'completed',result:{extracted:{fields:{document_number:'INV-1'}}}};
    const report={data:{documents:[]},schema_valid:true,schema_errors:[],warnings:[],
      data_issues:hasWarning?[{path:'documents[0].items[0].itemNo',message:'missing item number'}]:[],
      missing_fields:['documents[0].seller.name'],unmapped:[]};
    const fetch=async url=>{
      if(url.endsWith('/settings'))return response({settings:config});
      if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
      if(url.includes('/jobs?'))return response({jobs:[job],counts:{},more:false});
      if(url.endsWith('/user-json'))return response(report);
      return response({job});
    };
    vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',renderUserOcr(){},addEventListener(){}},fetch,Date,setTimeout:()=>1,clearTimeout(){}}));
    await settle();await el('historyRows').children[0].children.at(-1).children[0].listeners.click();
    el('userJsonTab').listeners.click();await settle();
    assert.match(el('userJsonNotice').textContent,/ผ่านโครงสร้าง/);
    assert.match(el('userJsonNotice').textContent,hasWarning?/ต้องตรวจข้อมูลก่อนใช้งาน/:/พร้อมตรวจเทียบต้นฉบับ/);
    assert.equal(el('userJsonNotice').className,hasWarning?'message warning':'message');
    assert.equal(el('userJsonCounts').textContent,`ผิดโครงสร้าง 0 · คำเตือน ${hasWarning?1:0} · ข้อมูลเสริมที่ยังไม่มี 1`);
    assert.equal(el('userJsonWarningGroup').hidden,!hasWarning);
    assert.equal(el('userJsonMissingGroup').hidden,false);
    assert.equal(el('userJsonDetails').open,hasWarning);
    assert.equal(el('userJsonDownload').hidden,false);
  }
});

test('a slow User JSON conversion cannot replace a different selected file, including returning to the previous file',async()=>{
  const {el,document}=dom();let release;const pending=new Promise(resolve=>{release=resolve;});
  const jobs=['a','b'].map(id=>({id,filename:id+'.png',status:'completed',result:{extracted:{fields:{document_number:id}}}}));
  const report=id=>({data:{file:id},schema_valid:true,warnings:[],schema_errors:[],missing_fields:[],unmapped:[]});
  const fetch=async url=>{
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.includes('/jobs?'))return response({jobs,counts:{},more:false});
    if(url.endsWith('/b/user-json'))return pending;
    if(url.endsWith('/a/user-json'))return response(report('a'));
    return response({job:url.endsWith('/b')?jobs[1]:jobs[0]});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',renderUserOcr(){},addEventListener(){}},fetch,Date,setTimeout:()=>1,clearTimeout(){}}));
  const settle=async()=>{for(let i=0;i<15;i++)await Promise.resolve();};
  await settle();const open=index=>el('historyRows').children[index].children.at(-1).children[0].listeners.click();
  await open(0);el('userJsonTab').listeners.click();await settle();
  await open(1);await settle();assert.equal(el('userJsonCode').textContent,'');
  assert.equal(el('userJsonDownload').href,'/ocr/api/user/jobs/b/user-json?download=1');
  await open(0);await settle();release(response(report('b')));await settle();
  assert.equal(el('userJsonCode').textContent,JSON.stringify(report('a').data,null,2));
});

test('Doc types menu loads the shared schema on demand, retries failures and preserves selected files',async()=>{
  const {el,document}=dom();let probes=0,fail=true;
  const schema={keys:['document_number','buyer_tax_id'],items_key:'income_items',item_keys:['amount_paid'],
    unknown:{fields:['document_number','buyer_tax_id'],mandatory:[],items:[]},types:[
      {code:'INVOICE',name:'an invoice',fields:['document_number'],mandatory:['document_number'],items:[],mandatory_items:[],headings:['ใบแจ้งหนี้'],has_requirement:true},
      {code:'TAX_INVOICE',name:'a tax invoice',fields:[],mandatory:[],items:[],headings:['ใบกำกับภาษี'],has_requirement:false}]};
  const fetch=async(url)=>{
    if(url.endsWith('/api/schema')){probes++;if(fail)return {ok:false,status:503};return response(schema);}
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    return response({jobs:[],counts:{},more:false});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/tools/ocr',addEventListener(){}},fetch,Date,
    setTimeout:()=>1,clearTimeout(){},URL:{createObjectURL:()=> 'blob:local',revokeObjectURL(){}}}));
  for(let i=0;i<10;i++)await Promise.resolve();
  el('fileInput').listeners.change({target:{files:[{name:'selected.png',size:5}]}});
  assert.equal(probes,0);
  el('navTypes').listeners.click();for(let i=0;i<10;i++)await Promise.resolve();
  assert.equal(el('typesView').hidden,false);assert.equal(el('uploadView').hidden,true);assert.equal(el('historyView').hidden,true);
  assert.equal(el('typesRetry').hidden,false);assert.match(el('typesError').textContent,/503/);
  fail=false;await el('typesRetry').listeners.click();
  assert.equal(el('typesRetry').hidden,true);assert.equal(probes,2);
  const walk=element=>[element,...(element.children||[]).flatMap(walk)],all=walk(el('typesContent'));
  assert.ok(all.some(e=>e.textContent==='invoice'));assert.ok(all.some(e=>e.textContent==='tax invoice'));
  assert.ok(all.some(e=>e.textContent==='Required'));assert.ok(all.some(e=>e.textContent==='Optional'));
  assert.ok(all.some(e=>e.textContent==='buyer_tax_id'));assert.ok(all.some(e=>e.textContent==='amount_paid'));
  assert.equal(all.filter(e=>e.tag==='table').length,1);
  el('navUpload').listeners.click();assert.equal(el('typesView').hidden,true);assert.equal(el('uploadView').hidden,false);
  assert.equal(el('selectedFiles').children[0].children[0].textContent,'selected.png');
  el('navTypes').listeners.click();assert.equal(probes,2);
});

test('Hybrid prompts page loads on demand, retries, shows grouped fields safely and refreshes settings on revisit',async()=>{
  const {el,document}=dom();let calls=0,fail=true;
  el('promptType').value='INVOICE';el('promptCombinedType').value='TAX_INVOICE';
  const data={doc_types:['INVOICE','TAX_INVOICE'],has_requirement:true,settings:{extras:false,schema_first:false},
    prompt_prefix:'Shared OCR placeholder\n',groups:[{id:'parties',title:'seller and buyer',
      keys:['seller_tax_id','buyer_tax_id'],question:'<img onerror="bad"> Read both parties'}]};
  const fetch=async url=>{
    if(url.includes('/hybrid-prompts?')){calls++;assert.equal(url,'/ocr/api/user/hybrid-prompts?doc_type=INVOICE&doc_type=TAX_INVOICE');
      return fail?response({error:'try again'},false):response(data);}
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    return response({jobs:[],counts:{},more:false});
  };
  const context=vm.createContext({document,window:{USER_BASE:'/ocr',addEventListener(){}},fetch,Date,setTimeout:()=>1,clearTimeout(){}});
  vm.runInContext(promptsSource,context);vm.runInContext(appSource,context);await settle();assert.equal(calls,0);
  el('navPrompts').listeners.click();await settle();
  assert.equal(el('promptsView').hidden,false);assert.equal(el('typesView').hidden,true);
  assert.equal(el('promptsRetry').hidden,false);assert.equal(el('promptsError').textContent,'try again');
  fail=false;await el('promptsRetry').listeners.click();
  const walk=e=>[e,...e.children.flatMap(walk)],nodes=walk(el('promptsContent'));
  assert.ok(nodes.some(e=>e.tag==='code'&&e.textContent==='seller_tax_id'));
  assert.ok(nodes.some(e=>e.tag==='code'&&e.textContent==='buyer_tax_id'));
  assert.ok(nodes.some(e=>e.tag==='pre'&&e.textContent===data.prompt_prefix+data.groups[0].question));
  assert.equal(nodes.filter(e=>e.tag==='img').length,0);
  assert.equal(el('promptsRetry').hidden,true);
  el('navUpload').listeners.click();assert.equal(el('promptsView').hidden,true);
  data.settings.extras=true;el('navPrompts').listeners.click();await settle();
  assert.match(el('promptsNotice').textContent,/ข้อมูลเพิ่มเติม เปิด/);assert.equal(calls,3);
});

test('a slow prompt response cannot replace a newer type; Unknown clears the combined type',async()=>{
  const {el,document}=dom();let release;
  const old=new Promise(resolve=>{release=resolve;});
  el('promptType').value='INVOICE';el('promptCombinedType').value='';
  const data=types=>({doc_types:types,has_requirement:false,settings:{extras:false,schema_first:false},prompt_prefix:'',groups:[]});
  const urls=[];
  const context=vm.createContext({document,window:{USER_BASE:''},fetch:async url=>{
    urls.push(url);return url.endsWith('INVOICE')?old:response(data([]));
  }});
  vm.runInContext(promptsSource,context);
  const pending=context.window.loadUserPrompts();
  el('promptType').value='UNKNOWN';el('promptCombinedType').value='TAX_INVOICE';
  await el('promptType').listeners.change();
  assert.equal(el('promptCombinedType').disabled,true);assert.equal(el('promptCombinedType').value,'');
  assert.equal(urls[1],'/api/user/hybrid-prompts?doc_type=UNKNOWN');
  release(response(data(['INVOICE'])));await pending;
  assert.match(el('promptsNotice').textContent,/^Unknown/);
});
