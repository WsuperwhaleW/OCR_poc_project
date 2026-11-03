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
    this.style={};this.dataset={};
    this.classList={toggle(){},add(){},remove(){}};}
  addEventListener(name,fn){this.listeners[name]=fn;}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  querySelectorAll(selector){const walk=e=>[e,...(e.children||[]).flatMap(walk)];return this.children.flatMap(walk).filter(e=>selector.startsWith('.')?e.className===selector.slice(1):e.tag===selector);}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  replaceChildren(...items){this.children=items;}
  setAttribute(name,value){this[name]=value;}
  removeAttribute(name){delete this[name];}
  showModal(){this.open=true;}
  close(){this.open=false;this.listeners.close?.();}
}

function promptSetData(code='INVOICE'){
  const value={instructions:{},extra_groups:[]};
  const definition={id:'document',title:'Document',fields:[{key:'document_number',edit_key:'document_number',
    linked_keys:['document_number'],default_instruction:'Read its own number',target:'documentNo',enabled:true}],enabled:true};
  const record={id:'invoice-set',name:'Invoice shared',base_code:'INVOICE',revision:2,value,
    steps:[{id:'document',keys:['document_number']}],base_steps:[]};
  return {doc_types:[code],editable_type:code,settings:{extras:false,schema_first:true},prompt_prefix:'OCR\n',
    groups:[{id:'document',title:'Document',keys:['document_number'],question:'Read its own number',enabled:true}],
    editor:{revision:2,value,groups:[definition],targets:[{path:'documentNo',type:'string'}]},
    library:{record,binding:{set_id:record.id,revision:0},source_selected:false,
      shared_types:[{code:'INVOICE',label:'380 · ใบแจ้งหนี้'},{code:'INVOICE_TAX_INVOICE',label:'T02 · ใบแจ้งหนี้/ใบกำกับภาษี'}],
      sets:[{id:record.id,name:record.name,revision:2}],types:[{code,label:code}]}};
}

function boardData(){return {revision:'board-1',groups:[
  {id:'g-inv',names:['Invoice (Sup&Pay)'],configured:true,question_count:4,field_count:13,required:['document_number'],
   types:[{code:'INVOICE',user_code:'380',label:'380 · ใบแจ้งหนี้',section:'user'},
     {code:'INVOICE_TAX_INVOICE',user_code:'T02',label:'T02 · ใบแจ้งหนี้/ใบกำกับภาษี',section:'user'}]},
  {id:'g-cn',names:['Credit Note (Sup&Pay)'],configured:true,question_count:4,field_count:13,required:[],
   types:[{code:'CREDIT_NOTE',user_code:'81',label:'81 · ใบลดหนี้',section:'user'}]},
  {id:'g-base',names:['ชุดพื้นฐาน · ยังไม่กำหนดชุด'],configured:false,question_count:15,field_count:34,required:[],
   types:[{code:'DEBIT_NOTE',user_code:'80',label:'80 · ใบเพิ่มหนี้',section:'user'}]}]};}
function boardOne(code){return {revision:'b',groups:[{id:'g',names:['Set'],configured:true,question_count:1,field_count:1,required:[],
  types:[{code,user_code:code,label:code,section:'user'}]}]};}
const dragEvent=code=>({preventDefault(){},dataTransfer:{setData(){},getData:()=>code}});

test('prompt board groups types, moves them by drag or menu, and confirms before saving the moves',async()=>{
  const {el,document}=dom(),walk=e=>[e,...e.children.flatMap(walk)],requests=[];let boardReads=0,conflict=true;
  const fetch=async(url,opts={})=>{
    if(url.endsWith('/prompt-groups')){boardReads++;return response(boardData());}
    if(url.endsWith('/prompt-groups/moves')){requests.push(JSON.parse(opts.body));
      return conflict?{ok:false,status:409,json:async()=>({error:'ก้อน prompt เปลี่ยนไปแล้ว'})}:response({saved:[]});}
    if(url.includes('/hybrid-prompts?'))return response(clean(promptSetData(url.split('doc_type=')[1].split('&')[0])));
    return response({});
  };
  const ctx=vm.createContext({document,setTimeout:()=>1,clearTimeout(){},window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof'},fetch});
  vm.runInContext(promptsSource,ctx);await ctx.window.loadUserPrompts({});await settle();
  const cards=()=>el('promptBoard').children;
  const chips=card=>walk(card).filter(e=>/^prompt-board-chip( moved)?$/.test(e.className||''));
  assert.equal(cards().length,4);
  assert.deepEqual(chips(cards()[0]).map(c=>c.title),['380 · ใบแจ้งหนี้','T02 · ใบแจ้งหนี้/ใบกำกับภาษี']);
  assert.match(el('promptBoardEditing').textContent,/กำลังแก้ก้อน Invoice \(Sup&Pay\).*380, T02/);
  assert.equal(el('promptLibrary').hidden,true);
  chips(cards()[1])[0].listeners.dragstart(dragEvent('CREDIT_NOTE'));cards()[0].listeners.drop(dragEvent('CREDIT_NOTE'));
  assert.equal(chips(cards()[0]).length,3);assert.equal(chips(cards()[1]).length,0);
  assert.equal(el('promptBoardBar').hidden,false);assert.match(el('promptBoardBarText').textContent,/ย้าย 1 ประเภท/);
  // Without dragging: tap a type, then tap the group. Moving back home cancels that move.
  const tap={stopPropagation(){}};
  const chipOf=code=>walk(el('promptBoard')).find(e=>typeof e['aria-label']==='string'&&e['aria-label'].startsWith(code+' · กดเพื่อเลือก'));
  assert.ok(!walk(el('promptBoard')).some(e=>e.tag==='select'));
  chipOf('80').listeners.click(tap);assert.equal(el('promptBoardHint').hidden,false);assert.match(el('promptBoardHint').textContent,/เลือก 80 แล้ว/);
  assert.equal(chipOf('80')['aria-pressed'],'true');
  cards()[0].listeners.click();assert.match(el('promptBoardBarText').textContent,/ย้าย 2/);assert.equal(el('promptBoardHint').hidden,true);
  chipOf('80').listeners.click(tap);cards()[2].listeners.click();assert.match(el('promptBoardBarText').textContent,/ย้าย 1/);
  chipOf('380').listeners.click(tap);chipOf('380').listeners.click(tap);assert.equal(el('promptBoardHint').hidden,true);
  chipOf('380').listeners.keydown({key:'Enter',preventDefault(){},stopPropagation(){}});assert.match(el('promptBoardHint').textContent,/เลือก 380/);
  chipOf('380').listeners.click(tap);
  walk(cards()[1]).find(e=>e.className==='prompt-board-head').listeners.click();
  assert.match(el('promptBoardError').textContent,/บันทึกหรือยกเลิกการย้าย/);
  el('promptBoardReview').listeners.click();assert.equal(el('promptMovesDialog').open,true);
  const text=walk(el('promptMovesList')).map(e=>e.textContent).join('\n');
  assert.match(text,/81 · ใบลดหนี้/);assert.match(text,/ใช้แบบเดียวกับ 380, T02/);
  await el('promptMovesSave').listeners.click();
  assert.equal(el('promptMovesDialog').open,true);assert.match(el('promptMovesMessage').textContent,/เปลี่ยนไปแล้ว/);
  conflict=false;await el('promptMovesSave').listeners.click();await settle();
  assert.deepEqual(requests.at(-1),{revision:'board-1',moves:{CREDIT_NOTE:'g-inv'}});
  assert.equal(el('promptMovesDialog').open,false);assert.equal(el('promptBoardBar').hidden,true);assert.equal(boardReads,2);
});

test('a group saves wording for every type in it, a split saves one type, and unsaved text blocks switching',async()=>{
  const {el,document}=dom(),walk=e=>[e,...e.children.flatMap(walk)],saves=[],loads=[];
  const fetch=async(url,opts={})=>{
    if(url.endsWith('/prompt-groups'))return response(boardData());
    if(url.endsWith('/prompt-groups/save')){saves.push({url,...opts,body:JSON.parse(opts.body)});return response({set_id:'s',revision:3});}
    if(url.includes('/hybrid-prompts?')){loads.push(url);return response(clean(promptSetData(url.split('doc_type=')[1].split('&')[0])));}
    return response({});
  };
  const ctx=vm.createContext({document,setTimeout:()=>1,clearTimeout(){},window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof'},fetch});
  vm.runInContext(promptsSource,ctx);await ctx.window.loadUserPrompts({});await settle();
  assert.equal(loads[0],'/ocr/api/user/hybrid-prompts?doc_type=INVOICE&template=1');
  const nodes=()=>walk(el('promptsContent')),find=text=>nodes().find(e=>e.textContent===text);
  assert.ok(!nodes().some(e=>e['aria-label']==='เปิดฟิลด์ document_number'));
  const edit=async value=>{find('แก้ไขวิธีอ่านตัวแปร').listeners.click();
    const area=nodes().find(e=>e['aria-label']==='วิธีอ่าน document_number');area.value=value;area.listeners.input();};
  await edit('Group rule');
  const cards=()=>el('promptBoard').children;
  walk(cards()[1]).find(e=>e.className==='prompt-board-head').listeners.click();
  assert.match(el('promptBoardError').textContent,/ยังไม่บันทึก/);assert.equal(loads.length,1);
  await el('promptSave').listeners.click();await settle();
  assert.equal(saves[0].headers['X-OCR-User-Token'],'proof');
  assert.deepEqual(saves[0].body.doc_types,['INVOICE','INVOICE_TAX_INVOICE']);assert.equal(saves[0].body.revision,'board-1');
  assert.equal(saves[0].body.value.instructions.document_number,'Group rule');
  cards().at(-1).listeners.drop(dragEvent('INVOICE_TAX_INVOICE'));await settle();
  assert.match(el('promptBoardEditing').textContent,/แยก T02 ออกจากก้อน Invoice/);
  assert.equal(loads.at(-1),'/ocr/api/user/hybrid-prompts?doc_type=INVOICE_TAX_INVOICE&template=1');
  await edit('T02 only');await el('promptSave').listeners.click();
  assert.deepEqual(saves[1].body.doc_types,['INVOICE_TAX_INVOICE']);
  // A type alone in its group is edited in place rather than split.
  cards().at(-1).listeners.drop(dragEvent('CREDIT_NOTE'));await settle();
  assert.match(el('promptBoardError').textContent,/81 อยู่ก้อนของตัวเองแล้ว/);assert.match(el('promptBoardEditing').textContent,/กำลังแก้ก้อน Credit Note/);
});

test('the inline User set editor keeps Retry scope explicit instead of promising a team binding change',async()=>{
  const {el,document}=dom(),data=promptSetData('INVOICE');
  data.available_types=['INVOICE'];data.job_id='job-snapshot';data.library.source_selected=true;
  const ctx=vm.createContext({document,setTimeout:()=>1,clearTimeout(){},window:{USER_BASE:'',USER_API_TOKEN:'proof'},
    fetch:async()=>response(clean(data))});
  vm.runInContext(promptsSource,ctx);
  await ctx.window.loadUserJobPrompts({id:'job-snapshot'},{},()=>{});
  assert.match(el('job_promptSharedTypes').textContent,/ต้นแบบสำหรับ Retry.*เฉพาะผลรอบใหม่ ไม่เปลี่ยนชุดกลาง/);
  assert.doesNotMatch(el('job_promptSharedTypes').textContent,/หลังบันทึก/);
});
function dom(){
  const items={};
  const el=id=>items[id] ||= new Element();
  const listeners={};
  return {items,el,document:{listeners,addEventListener(name,fn){listeners[name]=fn;},getElementById:el,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),querySelectorAll:()=>[]}};
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
    if(url.endsWith('/job-status'))return response({jobs:[{id:job.id,status:job.status,progress:job.progress}]});
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
  return {el,events,timers,calls,job,document,
    open:index=>el('historyRows').children[index].children.at(-1).children[0].listeners.click()};
}

test('BFCache restore resumes history, selected job and batch polling once',async()=>{
  const {el,events,timers,calls}=lifecycleSetup();await settle();
  el('fileInput').listeners.change({target:{files:[{name:'audit.png',size:10}]}});
  await el('uploadForm').listeners.submit({preventDefault(){}});await settle();
  const delays=()=>[...timers.values()].map(t=>t.ms).sort((a,b)=>a-b);
  assert.deepEqual(delays(),[1200,2000,10000,30000]);
  events.pagehide({persisted:true});assert.equal(timers.size,0);
  const previous=calls.length;events.pageshow({persisted:true});await settle();
  assert.ok(calls.slice(previous).some(url=>url.includes('/jobs?')));
  assert.equal(calls.slice(previous).filter(url=>url.endsWith('/jobs/job-a')).length,1);
  assert.equal(calls.slice(previous).filter(url=>url.endsWith('/job-status')).length,1);
  assert.deepEqual(delays(),[1200,2000,10000,30000]);
  events.pageshow({persisted:true});await settle();
  assert.deepEqual(delays(),[1200,2000,10000,30000]);
});

test('a hidden tab asks the server for nothing until it is visible again',async()=>{
  const {el,timers,calls,document}=lifecycleSetup();await settle();
  el('fileInput').listeners.change({target:{files:[{name:'audit.png',size:10}]}});
  await el('uploadForm').listeners.submit({preventDefault(){}});await settle();
  document.hidden=true;document.listeners.visibilitychange();
  assert.equal(timers.size,0);
  const previous=calls.length;await settle();assert.equal(calls.length,previous);
  document.hidden=false;document.listeners.visibilitychange();await settle();
  assert.deepEqual([...timers.values()].map(t=>t.ms).sort((a,b)=>a-b),[1200,2000,10000,30000]);
});

test('history polls only while the History view is showing',async()=>{
  const {el,timers}=lifecycleSetup();await settle();
  const history=()=>[...timers.values()].filter(t=>t.ms===5000).length;
  el('historyView').hidden=false;el('refreshHistory').listeners.click();await settle();
  assert.equal(history(),1);
  el('navUpload').listeners.click();assert.equal(history(),0);
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
  assert.equal(el('batchJobs').children.length,3);
  const zip=el('batchJobs').children.at(-1);
  assert.equal(zip.tag,'a');assert.equal(zip.href,'/ocr/api/user/batches/batch-id/user-json.zip');
  assert.match(zip.textContent,/ดาวน์โหลด User JSON ทั้งหมด · 2\/2 ไฟล์/);
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

test('a running stage counts up smoothly on the server clock and stops on the server time',async()=>{
  const {el,document}=dom(),intervals=new Map();let serial=0,browser=Date.parse('2026-11-03T00:00:00Z'),skew=0;
  const ahead=5000,start=browser+ahead-2000,iso=ms=>new Date(ms).toISOString();
  const running=id=>({id,filename:'scan.png',created_at:iso(start),status:'ocr',progress:'กำลังอ่านเอกสารหน้า 1/2',
    timeline:{ocr:{started_at:iso(start),progress:'กำลังอ่านเอกสารหน้า 1/2'}},result:{}});
  const jobs={'job-a':running('job-a'),'job-b':running('job-b')};
  const fetch=async url=>{
    const id=url.split('/jobs/')[1];
    const body=url.includes('/jobs?')?{jobs:Object.values(jobs),counts:{},more:false}:url.endsWith('/models')?{models:{ocr:{},extraction:{}}}:
      url.endsWith('/settings')?{settings:config}:url.endsWith('/stress')?{runs:[]}:{job:jobs[id]};
    // The server's clock reads 5 s ahead of the browser's; skew mimics a noisier sample.
    return {ok:true,status:200,json:async()=>clean(body),headers:{get:name=>name==='X-Server-Time'?String(browser+ahead+skew):null}};
  };
  class Clock extends Date {static now(){return browser;}}
  vm.runInContext(appSource,vm.createContext({document,fetch,Date:Clock,console,
    window:{USER_BASE:'/ocr',addEventListener(){},renderUserOcr(){}},URL:{createObjectURL:()=>'blob:x',revokeObjectURL(){}},
    setTimeout:()=>0,clearTimeout(){},setInterval:(fn,ms)=>{intervals.set(++serial,{fn,ms});return serial;},clearInterval:id=>intervals.delete(id)}));
  await settle();
  const shown=(stage=1)=>el('stageTimeline').children[stage].children[0].children[1].textContent;
  const tick=ms=>{for(let i=0;i<ms/100;i++){browser+=100;for(const timer of intervals.values())timer.fn();}};
  const open=async index=>{await el('historyRows').children[index].children.at(-1).children[0].listeners.click();await settle();};
  await open(0);
  assert.equal(shown(),'2.0 วิ','2 s on the server clock, not 0 s on the browser clock');
  assert.deepEqual([...intervals.values()].map(timer=>timer.ms),[100]);
  tick(100);assert.equal(shown(),'2.1 วิ');tick(200);assert.equal(shown(),'2.3 วิ');
  skew=-400;await open(0);skew=0;
  assert.equal(shown(),'2.3 วิ','a poll with a lower offset sample does not step back');
  tick(100);assert.equal(shown(),'2.3 วิ');
  await open(0);assert.equal(shown(),'2.4 วิ');
  for(let i=0;i<5;i++)await open(0);
  await open(1);await open(0);assert.equal(intervals.size,1,'re-renders and job switches keep one interval');
  document.hidden=true;document.listeners.visibilitychange();assert.equal(intervals.size,0);
  document.hidden=false;document.listeners.visibilitychange();await settle();assert.equal(intervals.size,1);
  tick(5700);assert.equal(shown(),'8.1 วิ');
  jobs['job-a']={...jobs['job-a'],status:'extraction',timeline:{ocr:{started_at:iso(start),finished_at:iso(start+7400)},
    extraction:{started_at:iso(start+7400)}}};
  await open(0);
  assert.equal(shown(),'7.4 วิ','a finished stage shows the server time, not the 8.1 s estimate');
  assert.equal(shown(3),'0.7 วิ');assert.equal(intervals.size,1);
  jobs['job-a']={...jobs['job-a'],status:'completed',timeline:{...jobs['job-a'].timeline,
    extraction:{started_at:iso(start+7400),finished_at:iso(start+9000)}}};
  await open(0);assert.equal(shown(3),'1.6 วิ');assert.equal(intervals.size,0);
  tick(1000);assert.equal(shown(3),'1.6 วิ');
  await open(1);assert.equal(intervals.size,1);
  jobs['job-b']={...jobs['job-b'],status:'failed'};
  await open(1);assert.equal(shown(),'—','a failed job without an end time neither ticks nor invents one');
  assert.equal(intervals.size,0);
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

test('an extracted result opens on User JSON, reports Draft, keeps a picked tab and the system JSON and prefixed download',async()=>{
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
  const open=async()=>{await el('historyRows').children[0].children.at(-1).children[0].listeners.click();await settle();};
  await settle();await open();
  // Converted once from the saved result; no model call.
  assert.equal(conversions,1);assert.equal(el('userJsonResult').hidden,false);assert.equal(el('textResult').hidden,true);
  assert.equal(el('exportLink').href,'/ocr/api/user/jobs/job/export');
  assert.equal(el('userJsonDownload').hidden,false);
  assert.equal(el('userJsonDownload').href,'/ocr/api/user/jobs/job/user-json?download=1');
  assert.equal(el('userJsonReportDownload').href,'/ocr/api/user/jobs/job/user-json?download=report');
  const original=el('jsonResult').textContent;
  assert.equal(el('userJsonRetry').hidden,false);
  fail=false;el('userJsonRetry').listeners.click();await settle();
  assert.equal(el('userJsonResult').hidden,false);assert.equal(el('jsonResult').hidden,true);
  assert.match(el('userJsonNotice').textContent,/Draft/);assert.equal(el('userJsonDetails').hidden,false);
  assert.equal(el('userJsonCounts').textContent,'ผิดโครงสร้าง 1 · คำเตือน 0 · ข้อมูลเสริมที่ยังไม่มี 1');
  assert.equal(el('userJsonSchemaGroup').hidden,false);assert.equal(el('userJsonWarningGroup').hidden,true);
  assert.equal(el('userJsonMissingGroup').hidden,false);
  assert.equal(el('userJsonCode').textContent,JSON.stringify(candidate,null,2));
  assert.equal(el('userJsonDownload').href,'/ocr/api/user/jobs/job/user-json?download=1');
  assert.match(el('userJsonUnmapped').textContent,/มหาชัย/);assert.equal(el('jsonResult').textContent,original);
  // A tab picked by hand stays picked when a result is opened again.
  el('jsonTab').listeners.click();await open();assert.equal(el('jsonResult').hidden,false);assert.equal(el('userJsonResult').hidden,true);
  el('userJsonTab').listeners.click();await settle();assert.equal(conversions,2);
});

test('both JSON download choices are in the shared toolbar, User JSON before the system JSON',()=>{
  const markup=readFileSync(new URL('templates/user.html',root),'utf8');
  const toolbar=markup.slice(markup.indexOf('class="toolbar result-tools"'),markup.indexOf('id="textResult"'));
  assert.match(toolbar,/id="exportLink">JSON ระบบ ↓/);
  assert.match(toolbar,/id="userJsonDownload" hidden>JSON ของ User ↓/);
  assert.ok(toolbar.indexOf('id="userJsonDownload"')<toolbar.indexOf('id="exportLink"'));
  assert.ok(toolbar.indexOf('id="userJsonTab"')<toolbar.indexOf('id="jsonTab"'));
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
  // The User's own types open first; the system catalog loads only from its own tab.
  assert.equal(el('userTypesPanel').hidden,false);assert.equal(el('systemTypesPanel').hidden,true);assert.equal(probes,0);
  el('systemTypesTab').listeners.click();for(let i=0;i<10;i++)await Promise.resolve();
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

test('User Doc types tab loads current sets independently, retries and refreshes on revisit',async()=>{
  const {el,document}=dom();let probes=0,fail=true,revision=2;
  const data=()=>({extras:false,types:[{code:'DELIVERY_TAX_INVOICE',user_code:'T04',name:'ใบส่งของ/ใบกำกับภาษี',
    field_count:1,sets:[{name:'T04 <custom>',revision,shared_types:['T04']}],
    groups:[{id:'custom',title:'Tax',fields:[{key:'custom_tax',target:'vatAmount'}]}]}]});
  const fetch=async(url)=>{
    if(url.endsWith('/doc-types')){probes++;return fail?{ok:false,status:503}:response(data());}
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    return response({jobs:[],counts:{},more:false});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',addEventListener(){}},fetch,Date,
    setTimeout:()=>1,clearTimeout(){},URL:{createObjectURL:()=> 'blob:local',revokeObjectURL(){}}}));
  await settle();assert.equal(probes,0);
  el('fileInput').listeners.change({target:{files:[{name:'selected.pdf',size:5}]}});
  el('userTypesTab').listeners.click();await settle();
  assert.equal(el('systemTypesPanel').hidden,true);assert.equal(el('userTypesPanel').hidden,false);
  assert.equal(el('userTypesTab')['aria-selected'],'true');assert.match(el('userTypesError').textContent,/503/);
  fail=false;await el('userTypesRefresh').listeners.click();
  const walk=e=>[e,...e.children.flatMap(walk)];
  assert.ok(walk(el('userTypesContent')).some(e=>e.textContent==='T04 <custom> · v2'));
  assert.ok(walk(el('userTypesContent')).some(e=>e.textContent==='→ vatAmount'));
  assert.equal(el('userTypesError').hidden,true);
  el('navUpload').listeners.click();revision=3;
  el('navTypes').listeners.click();await settle();
  assert.equal(probes,3);assert.equal(el('systemTypesPanel').hidden,true);
  assert.ok(walk(el('userTypesContent')).some(e=>e.textContent==='T04 <custom> · v3'));
  assert.equal(el('selectedFiles').children[0].children[0].textContent,'selected.pdf');
});

test('User field matrix cycles any cell of any type, reviews every change, keeps drafts after a conflict and saves all at once',async()=>{
  const {el,document}=dom();let saved=false,conflict=true,reads=0;const writes=[];
  const choices=[
    {key:'document_number',linked_keys:['document_number'],source:'Invoice',target:'documentNo'},
    {key:'payer_name',linked_keys:['payer_name'],source:'WHT',target:''},
    {key:'book_no',linked_keys:['book_no','certificate_no'],source:'WHT',target:''},
    {key:'certificate_no',linked_keys:['book_no','certificate_no'],source:'WHT',target:''},
    {key:'custom_tax',linked_keys:['custom_tax'],source:'T04',target:'vatAmount'}];
  const data=()=>({revision:saved?'snapshot-2':'snapshot-1',extras:false,
    requirable:['document_number','payer_name','book_no','certificate_no'],types:[
    {code:'INVOICE',user_code:'380',name:'ใบแจ้งหนี้',mandatory:['document_number'],choices,
     sets:[{name:'Shared Invoice',revision:2,shared_types:['380','T02']}],field_count:1,
     groups:[{title:'Fields',fields:[{key:'document_number',target:'documentNo'}]}]},
    {code:'DELIVERY_TAX_INVOICE',user_code:'T04',name:'ใบส่งของ/ใบกำกับภาษี',mandatory:[],choices,
     sets:[{name:'T04',revision:1,shared_types:['T04']}],field_count:2,
     groups:[{title:'Fields',fields:[{key:'document_number',target:'documentNo'},{key:'custom_tax',target:'vatAmount'}]}]}]});
  const fetch=async(url,opts={})=>{
    if(url.endsWith('/doc-types/fields')){writes.push({url,...opts,body:JSON.parse(opts.body)});
      if(conflict)return {ok:false,status:409,json:async()=>({error:'ชุดฟิลด์เปลี่ยนไปแล้ว'})};
      saved=true;return response({saved:[]});}
    if(url.endsWith('/doc-types')){reads++;return response(data());}
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    return response({jobs:[],counts:{},more:false});
  };
  vm.runInContext(appSource,vm.createContext({document,window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof',addEventListener(){}},fetch,Date,
    setTimeout:()=>1,clearTimeout(){}}));
  el('userTypesTab').listeners.click();await settle();
  const walk=e=>[e,...e.children.flatMap(walk)];
  const cell=(code,key)=>walk(el('userTypesContent')).find(e=>typeof e['aria-label']==='string'&&e['aria-label'].startsWith(code+' · '+key+':'));
  const click=(code,key,times=1)=>{for(let i=0;i<times;i++)cell(code,key).listeners.click();};
  const bar=()=>walk(el('userTypesContent')).find(e=>e.className==='user-fields-bar');
  assert.equal(cell('380','document_number').textContent,'Required');
  assert.equal(cell('380','payer_name').textContent,'—');
  assert.equal(bar().hidden,true);
  // No Edit button: any cell of any type cycles Required → Optional → not asked.
  assert.ok(!walk(el('userTypesContent')).some(e=>e.textContent==='Edit'));
  click('380','payer_name');assert.equal(cell('380','payer_name').textContent,'Required');
  click('380','payer_name');assert.equal(cell('380','payer_name').textContent,'Optional');
  assert.match(cell('380','payer_name').className,/changed/);
  click('380','book_no');assert.equal(cell('380','certificate_no').textContent,'Required');
  click('380','document_number');assert.equal(cell('380','document_number').textContent,'Optional');
  // A custom field cannot be Required: Optional → not asked → Optional (back to saved, no longer a change).
  click('T04','custom_tax');assert.equal(cell('T04','custom_tax').textContent,'—');
  click('T04','custom_tax');assert.equal(cell('T04','custom_tax').textContent,'Optional');
  assert.doesNotMatch(cell('T04','custom_tax').className,/changed/);
  click('T04','custom_tax');
  assert.equal(bar().hidden,false);assert.match(bar().children[0].textContent,/แก้ 5 ช่อง ใน 2 ประเภท/);
  // A reload at the same revision keeps the drafts.
  await el('userTypesRefresh').listeners.click();await settle();
  assert.equal(cell('380','book_no').textContent,'Required');assert.match(bar().children[0].textContent,/แก้ 5 ช่อง/);
  const review=()=>walk(el('userTypesContent')).find(e=>e.textContent==='ตรวจและบันทึก…').listeners.click();
  review();assert.equal(el('userFieldsEditor').open,true);assert.equal(writes.length,0);
  const text=walk(el('userFieldsTable')).map(e=>e.textContent).join('\n');
  assert.match(text,/380 · ใบแจ้งหนี้/);assert.match(text,/ใช้ชุดร่วมกับ T02/);
  assert.match(text,/เพิ่ม · 3/);assert.match(text,/เปลี่ยนเป็น Optional · 1/);assert.match(text,/เอาออก · ไม่ถาม · 1/);
  assert.match(el('userFieldsCount').textContent,/2 ประเภท · 5 ช่อง/);
  await el('userFieldsSave').listeners.click();
  assert.equal(el('userFieldsEditor').open,true);assert.match(el('userFieldsMessage').textContent,/เปลี่ยนไปแล้ว/);
  el('userFieldsCancel').listeners.click();assert.equal(el('userFieldsEditor').open,false);
  assert.equal(cell('380','payer_name').textContent,'Optional');
  conflict=false;review();await el('userFieldsSave').listeners.click();await settle();
  assert.equal(el('userFieldsEditor').open,false);
  assert.equal(writes[1].method,'PUT');assert.equal(writes[1].headers['X-OCR-User-Token'],'proof');
  assert.deepEqual(writes[1].body,{revision:'snapshot-1',changes:{
    INVOICE:{enabled_keys:['document_number','payer_name','book_no','certificate_no'],required_keys:['book_no','certificate_no']},
    DELIVERY_TAX_INVOICE:{enabled_keys:['document_number'],required_keys:[]}}});
  assert.equal(bar().hidden,true);assert.equal(reads,3);
});

test('Hybrid prompts page loads on demand, retries, shows grouped fields safely and refreshes settings on revisit',async()=>{
  const {el,document}=dom();let calls=0,fail=true;
  el('promptType').value='INVOICE';
  const data={doc_types:['INVOICE'],has_requirement:true,settings:{extras:false,schema_first:false},
    prompt_prefix:'Shared OCR placeholder\n',groups:[{id:'parties',title:'seller and buyer',
      keys:['seller_tax_id','buyer_tax_id'],question:'<img onerror="bad"> Read both parties'}]};
  const fetch=async url=>{
    if(url.endsWith('/prompt-groups'))return response(boardOne('INVOICE'));
    if(url.includes('/hybrid-prompts?')){calls++;assert.equal(url,'/ocr/api/user/hybrid-prompts?doc_type=INVOICE&template=1');
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

test('a slow prompt response cannot replace a newer supported type',async()=>{
  const {el,document}=dom();let release;
  const old=new Promise(resolve=>{release=resolve;});
  el('promptType').value='INVOICE';
  const data=types=>({doc_types:types,has_requirement:false,settings:{extras:false,schema_first:false},prompt_prefix:'',groups:[]});
  const urls=[];
  const context=vm.createContext({document,window:{USER_BASE:''},fetch:async url=>{
    urls.push(url);if(url.endsWith('/prompt-groups'))return response(boardOne('INVOICE'));
    return url.includes('doc_type=INVOICE&')?old:response(data(['WHT_CERTIFICATE']));
  }});
  vm.runInContext(promptsSource,context);
  const pending=context.window.loadUserPrompts();await settle();
  el('promptType').value='WHT_CERTIFICATE';
  await el('promptType').listeners.change();
  assert.equal(urls[2],'/api/user/hybrid-prompts?doc_type=WHT_CERTIFICATE&template=1');
  release(response(data(['INVOICE'])));await pending;
  assert.match(el('promptsNotice').textContent,/^WHT_CERTIFICATE/);
});

test('User prompt editor saves only field guidance and explicit extra destinations, protects dirty edits and keeps conflicts visible',async()=>{
  const {el,document,items}=dom(),walk=e=>[e,...e.children.flatMap(walk)];
  document.querySelector=selector=>Object.values(items).flatMap(walk).find(e=>e.className===selector.slice(1));
  const data={doc_types:['INVOICE'],has_requirement:true,settings:{extras:false,schema_first:false},prompt_prefix:'OCR\n',
    groups:[{id:'document',title:'document',keys:['document_number'],question:'Locked question'}],
    editor:{revision:0,value:{instructions:{},extra_groups:[]},groups:[{id:'document',title:'document',fields:[
      {key:'document_number',instruction:'',default_instruction:'Use its own printed number',target:'documentNo'}]}],
      targets:[{path:'documentNo',type:'string'},{path:'seller.name',type:'string'}]}};
  const calls=[];let conflict=false;
  const context=vm.createContext({document,window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof'},fetch:async(url,options={})=>{
    if(url.endsWith('/prompt-groups'))return response(boardOne('INVOICE'));
    if(options.method==='POST'&&url.endsWith('/prompt-groups/save')){
      calls.push({url,...options});
      if(conflict)return response({error:'มีผู้แก้ prompt นี้แล้ว'},false);
      const body=JSON.parse(options.body);data.editor.value=body.value;data.editor.revision++;
      return response({set_id:'s',revision:data.editor.revision});
    }
    return response(clean(data));
  }});
  el('promptType').value='INVOICE';vm.runInContext(promptsSource,context);
  await context.window.loadUserPrompts();
  const find=text=>walk(el('promptsContent')).find(e=>e.textContent===text);
  const textarea=()=>walk(el('promptsContent')).find(e=>e['aria-label']==='วิธีอ่าน document_number');
  assert.equal(textarea().readOnly,true);
  assert.equal(textarea().value,'Use its own printed number');
  assert.equal(textarea().placeholder,undefined);
  assert.equal(el('promptSave').disabled,true);
  find('แก้ไขวิธีอ่านตัวแปร').listeners.click();assert.equal(textarea().readOnly,false);
  textarea().value+=' — top right';textarea().listeners.input();
  el('promptType').value='WHT_CERTIFICATE';await el('promptType').listeners.change();
  assert.equal(el('promptType').value,'INVOICE');assert.equal(el('promptSave').disabled,false);
  find('+ เพิ่มก้อนคำถาม').listeners.click();
  const extra=document.querySelector('.prompt-extra-groups').children[0],row=extra.children.find(e=>e.className==='prompt-kv-row');
  const key=row.children[0].children[1],instruction=row.children[1].children[1],target=row.children[2].children[1];
  key.value='supplier_name';key.listeners.input();instruction.value='Read the supplier name';instruction.listeners.input();
  assert.equal(target.children.find(e=>e.value==='documentNo').disabled,true);
  target.value='seller.name';target.listeners.change();
  await el('promptSave').listeners.click();
  assert.equal(calls[0].url,'/ocr/api/user/prompt-groups/save');assert.equal(calls[0].headers['X-OCR-User-Token'],'proof');
  const sent=JSON.parse(calls[0].body);
  assert.equal(sent.revision,'b');assert.deepEqual(sent.doc_types,['INVOICE']);assert.equal(sent.value.instructions.document_number,'Use its own printed number — top right');
  assert.equal(sent.value.extra_groups[0].fields[0].target,'seller.name');
  assert.equal(el('promptSave').disabled,true);assert.match(el('promptEditStatus').textContent,/เวอร์ชัน 1/);
  find('แก้ไขวิธีอ่านตัวแปร').listeners.click();textarea().value='conflicting edit';textarea().listeners.input();
  conflict=true;await el('promptSave').listeners.click();
  assert.equal(el('promptsError').hidden,false);assert.equal(textarea().value,'conflicting edit');
  assert.equal(el('promptSave').disabled,false);
  el('promptReset').listeners.click();
  assert.equal(textarea().value,'Use its own printed number');assert.equal(document.querySelector('.prompt-extra-groups').children.length,0);
  assert.equal(el('promptSave').disabled,false);
  conflict=false;await el('promptSave').listeners.click();
  assert.deepEqual(JSON.parse(calls.at(-1).body).value.instructions,{});
});

test('linked prompt variables share one editor and collapsing or viewing keeps the saved field mapping',async()=>{
  const {el,document}=dom(),walk=e=>[e,...e.children.flatMap(walk)];
  document.querySelector=()=>null;
  el('promptsContent').querySelector=selector=>walk(el('promptsContent')).find(e=>e.className===selector.slice(1));
  const shared='- book_no and certificate_no: read both printed captions.';
  const data={doc_types:['WHT_CERTIFICATE'],settings:{extras:false,schema_first:false},prompt_prefix:'OCR\n',
    groups:[{id:'wht_certificate',title:'Certificate numbering',keys:['book_no','certificate_no'],question:'Locked question'}],
    editor:{revision:0,value:{instructions:{},extra_groups:[]},targets:[],groups:[{id:'wht_certificate',fields:
      ['book_no','certificate_no'].map(key=>({key,edit_key:'book_no',linked_keys:['book_no','certificate_no'],default_instruction:shared,target:''}))}]}};
  const requests=[];
  const context=vm.createContext({document,window:{USER_BASE:'',USER_API_TOKEN:'proof'},fetch:async(url,options={})=>{
    if(url.endsWith('/prompt-groups'))return response(boardOne('WHT_CERTIFICATE'));
    if(options.method==='POST'){const body=JSON.parse(options.body);requests.push(body);data.editor.value=body.value;data.editor.revision++;return response({set_id:'s',revision:data.editor.revision});}
    return response(clean(data));
  }});
  el('promptType').value='WHT_CERTIFICATE';vm.runInContext(promptsSource,context);await context.window.loadUserPrompts();
  const nodes=()=>walk(el('promptsContent')),find=text=>nodes().find(e=>e.textContent===text);
  assert.equal(nodes().filter(e=>e.tag==='textarea').length,1);
  assert.ok(nodes().some(e=>e.tag==='code'&&e.textContent==='book_no'));
  assert.ok(nodes().some(e=>e.tag==='code'&&e.textContent==='certificate_no'));
  find('แก้ไขวิธีอ่านตัวแปร').listeners.click();
  const area=nodes().find(e=>e.tag==='textarea');area.value=shared+' Keep the two numbers separate.';area.listeners.input();
  find('พับทุกก้อน').listeners.click();assert.equal(nodes().find(e=>e.className==='prompt-group-disclosure').open,false);
  find('เปิดทุกก้อน').listeners.click();find('กลับโหมดดู').listeners.click();
  assert.equal(area.readOnly,true);assert.equal(area.value,shared+' Keep the two numbers separate.');
  await el('promptSave').listeners.click();
  assert.deepEqual(requests[0].value.instructions,{book_no:shared+' Keep the two numbers separate.'});
  assert.equal(nodes().filter(e=>e.tag==='textarea').length,1);
});

test('result-page prompt edits retry extraction in place and keep the same request ID after a failed response',async()=>{
  const {el,document}=dom(),walk=e=>[e,...e.children.flatMap(walk)];
  document.querySelector=()=>null;
  el('job_promptsContent').querySelector=selector=>walk(el('job_promptsContent')).find(e=>e.className===selector.slice(1));
  const original={id:'original',filename:'source.png',created_at:'2026-10-30',status:'completed',configuration:{},
    result:{text:'INVOICE',pages:['INVOICE'],extracted:{doc_types:['INVOICE'],fields:{document_number:'OLD'}}}};
  const child={...clean(original),id:'retried',retry_of:'original',result:{...clean(original.result),extracted:{doc_types:['INVOICE'],fields:{document_number:'NEW'}}}};
  const data={doc_types:['INVOICE'],available_types:['INVOICE'],editable_type:'INVOICE',settings:{extras:false,schema_first:false},prompt_prefix:'OCR\n',
    groups:[{id:'document',title:'document',keys:['document_number'],question:'Original question'}],
    editor:{revision:1,value:{instructions:{},extra_groups:[]},groups:[{id:'document',title:'document',fields:[
      {key:'document_number',default_instruction:'Read own number',target:'documentNo'}]}],targets:[]}};
  const retries=[];let fail=true;
  const context=vm.createContext({document,window:{USER_BASE:'/ocr',USER_API_TOKEN:'proof',addEventListener(){},renderUserOcr(){}},Date,setTimeout:()=>1,clearTimeout(){},fetch:async(url,options={})=>{
    if(url.endsWith('/settings'))return response({settings:config});
    if(url.endsWith('/models'))return response({models:{ocr:{},extraction:{}}});
    if(url.includes('/jobs?'))return response({jobs:[original,child],counts:{},more:false});
    if(url.endsWith('/retry-history'))return response({attempts:[
      {id:'original',retry_of:'',created_at:'2026-10-30',status:'completed',prompt_changes:{}},
      {id:'retried',retry_of:'original',created_at:'2026-10-30',status:'completed',prompt_changes:{version:1,scope:'job',changes:[
        {entity:'instruction',action:'changed',doc_type:'INVOICE',key:'document_number',before:'Old reading guide',after:'Use top right number'},
        {entity:'group',action:'added',doc_type:'INVOICE',group_title:'<img onerror="bad">',before:null,after:{title:'Extra group',fields:[{key:'supplier_name',instruction:'Read seller name',target:'seller.name'}]}}
      ]}}]});
    if(url.endsWith('/original/prompts'))return response(clean(data));
    if(url.endsWith('/retry-extraction')){retries.push(JSON.parse(options.body));return fail?response({error:'connection failed'},false):response({job:child});}
    return response({job:url.endsWith('/retried')?child:original});
  }});
  vm.runInContext(promptsSource,context);vm.runInContext(appSource,context);await settle();
  await el('historyRows').children[0].children.at(-1).children[0].listeners.click();
  assert.equal(el('jobPromptPanel').hidden,false);
  await el('jobPromptOpen').listeners.click();assert.equal(el('jobPromptWorkbench').hidden,false);
  const find=text=>walk(el('job_promptsContent')).find(e=>e.textContent===text);
  find('แก้ไขวิธีอ่านตัวแปร').listeners.click();
  const area=walk(el('job_promptsContent')).find(e=>e['aria-label']==='วิธีอ่าน document_number');
  area.value='Use top right number';area.listeners.input();
  await el('job_promptSave').listeners.click();
  assert.equal(el('job_promptsError').textContent,'connection failed');
  assert.equal(area.value,'Use top right number');
  fail=false;await el('job_promptSave').listeners.click();await settle();
  assert.equal(retries.length,2);assert.equal(retries[0].request_id,retries[1].request_id);
  assert.equal(retries[1].prompt_profiles.INVOICE.instructions.document_number,'Use top right number');
  assert.equal(JSON.parse(el('jsonResult').textContent).fields.document_number,'NEW');
  assert.equal(el('jobRetryOrigin').hidden,false);
  assert.equal(el('stageTimeline').children.length,2);
  assert.equal(el('jobPromptWorkbench').hidden,true);
  el('jobPromptHistory').open=true;await el('jobPromptHistory').listeners.toggle();await settle();
  const history=walk(el('jobPromptHistoryContent'));
  assert.equal(el('jobPromptHistoryContent').children.length,2);
  assert.ok(history.some(e=>e.textContent==='Old reading guide'));
  assert.ok(history.some(e=>e.textContent==='Use top right number'));
  assert.ok(history.some(e=>e.tag==='pre'&&e.textContent.includes('seller.name')));
  assert.ok(history.some(e=>e.tag==='h4'&&e.textContent.includes('<img onerror=')));
  assert.equal(history.filter(e=>e.tag==='img').length,0);
  await el('jobPreviousResult').listeners.click();await settle();
  assert.equal(JSON.parse(el('jsonResult').textContent).fields.document_number,'OLD');
  assert.equal(el('jobPromptHistory').open,false);
  assert.equal(el('jobPromptHistoryContent').children.length,0);
});
