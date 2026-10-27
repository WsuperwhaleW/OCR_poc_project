/* Real shared bootstrap in simulated tabs; no model calls or stored user data. */
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const source = readFileSync(new URL('../templates/_workspace_session.html',import.meta.url),'utf8');
let sequence = 0;
const fresh = () => 'session'+String(++sequence).padStart(25,'0');

function tab({user=true,base='/ocr',storage=new Map(),bus=[],href,
              readBlocked=false,writeBlocked=false,channels=true,secure=true}={}){
  const address = new URL(href || 'http://example.test'+base+(user?'/user':'/'));
  const redirects=[],requests=[];
  const location={href:address.href,origin:address.origin,replace:value=>redirects.push(value)};
  class Channel {
    constructor(name){this.name=name;bus.push(this);}
    postMessage(data){for(const other of bus)if(other!==this&&other.name===this.name)other.onmessage?.({data});}
  }
  const crypto=secure?{randomUUID:fresh}:{};
  const sessionStorage={getItem(key){if(readBlocked)throw Error('blocked');return storage.get(key)??null;},
    setItem(key,value){if(writeBlocked)throw Error('full');storage.set(key,value);}};
  const window={USER_BASE:base,OCR_BASE:base,crypto,
    BroadcastChannel:channels?Channel:undefined,fetch:async(input,init)=>{requests.push({input,init});return{};}};
  vm.runInNewContext(source.replace('{{ user_session | default(false) | tojson }}',user?'true':'false'),
    {window,location,sessionStorage,crypto,BroadcastChannel:Channel,URL,Headers});
  return {window,storage,redirects,requests,bus};
}

test('a new User tab adds a session to its prefixed URL and refresh reuses only that ID',()=>{
  const first=tab({href:'http://example.test/tools/ocr/user/?other=1#result',base:'/tools/ocr'});
  assert.equal(first.redirects.length,1);
  const address=new URL(first.redirects[0]);
  assert.equal(address.pathname,'/tools/ocr/user/');assert.equal(address.searchParams.get('other'),'1');
  assert.equal(address.hash,'#result');assert.equal(address.searchParams.get('ws'),first.window.OCR_WORKSPACE);
  assert.equal(first.storage.get('ocrUserWorkspace:/tools/ocr'),first.window.OCR_WORKSPACE);
  assert.deepEqual([...first.storage.keys()],['ocrUserWorkspace:/tools/ocr']);
  const reloaded=tab({href:address.href,base:'/tools/ocr',storage:first.storage});
  assert.equal(reloaded.window.OCR_WORKSPACE,first.window.OCR_WORKSPACE);
  assert.equal(reloaded.redirects.length,0);
});

test('new tabs and duplicated tabs get independent sessions',()=>{
  const id=fresh(),bus=[],storage=new Map([['ocrUserWorkspace:/ocr',id]]);
  const first=tab({href:'http://example.test/ocr/user?ws='+id,storage,bus});
  const duplicated=tab({href:'http://example.test/ocr/user?ws='+id,storage:new Map(storage),bus});
  assert.equal(first.window.OCR_WORKSPACE,id);
  assert.notEqual(duplicated.window.OCR_WORKSPACE,id);
  assert.equal(new URL(duplicated.redirects[0]).searchParams.get('ws'),duplicated.window.OCR_WORKSPACE);
  const openedLink=tab({href:'http://example.test/ocr/user?ws='+id,bus});
  assert.notEqual(openedLink.window.OCR_WORKSPACE,id,'a copied URL must not share an existing tab session');
});

test('User sessions do not reuse the main-page key or another deployment path',()=>{
  const mainId=fresh(),storage=new Map([['ocrWorkspace',mainId]]);
  const portal=tab({storage});
  assert.notEqual(portal.window.OCR_WORKSPACE,mainId);
  const different=tab({storage,base:'/other'});
  assert.notEqual(different.window.OCR_WORKSPACE,portal.window.OCR_WORKSPACE);
  const main=tab({user:false,storage,href:'http://example.test/ocr/?ws='+mainId});
  assert.equal(main.window.OCR_WORKSPACE,mainId);assert.equal(main.redirects.length,0);
});

test('API requests carry the tab ID while preserving request data and avoiding other origins',async()=>{
  const id=fresh(),storage=new Map([['ocrUserWorkspace:/ocr',id]]);
  const current=tab({storage,href:'http://example.test/ocr/user?ws='+id});
  const body={file:'unchanged multipart object'};
  await current.window.fetch('/ocr/api/user/jobs',{method:'POST',body,headers:{'X-Custom':'keep'}});
  assert.equal(current.requests[0].init.headers.get('X-OCR-Workspace'),id);
  assert.equal(current.requests[0].init.headers.get('X-Custom'),'keep');
  assert.equal(current.requests[0].init.body,body);
  await current.window.fetch(new Request('http://example.test/ocr/api/user/settings',{headers:{'X-Custom':'request'}}));
  assert.equal(current.requests[1].init.headers.get('X-OCR-Workspace'),id);
  assert.equal(current.requests[1].init.headers.get('X-Custom'),'request');
  await current.window.fetch('http://other.test/api',{headers:{'X-Custom':'external'}});
  assert.equal(current.requests[2].init.headers['X-OCR-Workspace'],undefined);
  await current.window.fetch(new URL('http://other.test/api'),{});
  assert.equal(current.requests[3].init.headers,undefined);
});

test('blocked or full sessionStorage uses the URL without a redirect loop',()=>{
  for(const readBlocked of [false,true]){
    const first=tab({readBlocked,writeBlocked:true});
    assert.equal(first.redirects.length,1);
    const reload=tab({readBlocked,writeBlocked:true,href:first.redirects[0]});
    assert.equal(reload.redirects.length,0);
    assert.equal(reload.window.OCR_WORKSPACE,first.window.OCR_WORKSPACE);
  }
  const plainHttp=tab({secure:false,channels:false});
  assert.match(plainHttp.window.OCR_WORKSPACE,/^[A-Za-z0-9_-]{8,64}$/);
});
