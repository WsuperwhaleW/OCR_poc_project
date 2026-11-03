import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const benchmark = readFileSync(new URL('static/benchmark.js', root), 'utf8');
const portal = readFileSync(new URL('templates/_user_app_script.html', root), 'utf8');
function extract(text, name) {
  const start = text.indexOf('async function ' + name + '(');
  assert.notEqual(start, -1);
  const end = text.indexOf('\n' + (name === 'api' ? '  ' : '') + '}', start);
  return text.slice(start, end + (name === 'api' ? 4 : 2));
}
const rejected = (status, body) => ({status, ok:false, json:async()=>{
  if(body === undefined)throw new SyntaxError('HTML response');
  return body;
}});

test('Workspace preview and OCR keep app errors and explain a non-JSON 413', async()=>{
  const context = vm.createContext({Error});
  vm.runInContext(extract(benchmark, 'uploadRequestError'), context);
  const error = await context.uploadRequestError(rejected(413), 'Could not prepare');
  assert.match(error.message, /413/);
  assert.match(error.message, /ผู้ดูแล/);
  assert.match(error.message, /เซิร์ฟเวอร์ด้านหน้า/);
  const limited = await context.uploadRequestError(rejected(413, {error:'File is larger than 32 MB.'}), 'Request failed');
  assert.equal(limited.message, 'File is larger than 32 MB.');
  assert.equal((await context.uploadRequestError(rejected(502), 'Request failed')).message, 'Request failed (502)');
  for(const name of ['refreshPrepared', 'startRead']) {
    const start = benchmark.indexOf('async function ' + name + '(');
    assert.match(benchmark.slice(start, benchmark.indexOf('\n}', start)), /throw await uploadRequestError\(/);
  }
});

test('User API handles HTML and JSON 413 responses without hiding the upload problem', async()=>{
  let response;
  const context = vm.createContext({Error, API:'/api/user', window:{USER_API_TOKEN:'token'},
    fetch:async()=>response, noteServerClock(){}});
  vm.runInContext(extract(portal, 'api'), context);
  response = rejected(413);
  await assert.rejects(context.api('/previews', {method:'POST'}), error=>
    error.status === 413 && /ผู้ดูแล/.test(error.message) && !/JSON/.test(error.message));
  response = rejected(413, {error:'ไฟล์ใหญ่เกินกำหนด (1 MB)'});
  await assert.rejects(context.api('/jobs', {method:'POST'}), /ไฟล์ใหญ่เกินกำหนด \(1 MB\)/);
  response = rejected(502);
  await assert.rejects(context.api('/jobs'), /JSON \(502\)/);
});
