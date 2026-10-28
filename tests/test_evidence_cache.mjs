import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';
const context=vm.createContext({});
vm.runInContext(readFileSync(new URL('../static/benchmark-performance.js',import.meta.url),'utf8'),context);
const response=(data,etag='v1')=>({ok:true,status:200,headers:{get:()=>etag},json:async()=>data});
test('cache reuses data, deduplicates concurrent loads and revalidates after expiry',async()=>{
  let clock=0,release;const calls=[];
  const load=context.createEvidenceCache({now:()=>clock,ttl:5,fetch:(url,options)=>{
    calls.push({url,options});return new Promise(resolve=>{release=resolve;});
  }});
  const a=load('/a'),b=load('/a');release(response({fields:[1]}));
  assert.equal(await a,await b);assert.equal(calls.length,1);
  await load('/a');assert.equal(calls.length,1);
  clock=6;const stale=load('/a');assert.equal(calls[1].options.headers['If-None-Match'],'v1');
  release({status:304});assert.deepEqual(JSON.parse(JSON.stringify(await stale)),{fields:[1]});
  await load('/a');assert.equal(calls.length,2);
});
test('revisions bypass old data, errors retry and bounded cache evicts old entries',async()=>{
  let calls=0,fail=false;
  const load=context.createEvidenceCache({maxEntries:1,fetch:async()=>{
    calls++;if(fail)return {ok:false,status:404,json:async()=>({error:'gone'})};
    return response({calls});
  }});
  await load('/a','1');await load('/a','2');assert.equal(calls,2);
  await load('/a','1');assert.equal(calls,3);
  fail=true;await assert.rejects(load('/b'),/gone/);await assert.rejects(load('/b'),/gone/);
  fail=false;await load('/b');assert.equal(calls,6);
});
