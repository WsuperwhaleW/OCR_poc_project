import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = readFileSync(new URL('../templates/index.html', import.meta.url), 'utf8');
const source = page.slice(page.indexOf('let stDetailLog ='), page.indexOf('\nfunction stFileBoxes()'));
function setup(blockDownload = false) {
  const downloads = [], blobs = [], revoked = [];
  const ctx = vm.createContext({Blob, Date, stState: {plan: {run_id: 'test:run'}},
    URL: {createObjectURL(blob) {blobs.push(blob); return 'blob:download';},
          revokeObjectURL(url) {revoked.push(url);}},
    document: {body: {appendChild() {}}, createElement() {return {
      click() {if (blockDownload) throw new Error('blocked'); downloads.push(this.href);}, remove() {}};}}});
  vm.runInContext(source + '\napi = {receive: stReceiveDetailLog, finish: stFinishDetailLog, clear: stClearDetailLog, download: stDownloadDetailLog};', ctx);
  ctx.api.receive({event: 'detail_log_start', format_version: 2, configuration: {save_details: true}});
  return {ctx, api: ctx.api, downloads, blobs, revoked};
}

test('builds valid JSON, automatically downloads once, and permits a second download', async () => {
  const {api, downloads, blobs, revoked} = setup();
  const jobs = [{case: 'sol001', actual: 'ภาษาไทย "quoted"\nnext line'}, {case: 'sol002', expected: null}];
  jobs.forEach(job => api.receive({event: 'detail_log_job', job}));
  api.receive({event: 'detail_log_end', field_summary: [{field: 'tax_id'}],
    contexts: {source: 'document text'}, questions: {lookup: 'question'}, report: {completed: 2}});
  api.finish({event: 'done', report: {completed: 2, timeline: ['not for tuning']}});
  api.finish({event: 'done'});
  assert.equal(downloads.length, 1);
  const file = JSON.parse(await blobs[0].text());
  assert.deepEqual(file.jobs, jobs);
  assert.equal(file.format_version, 2);
  assert.deepEqual(file.contexts, {source: 'document text'});
  assert.deepEqual(file.questions, {lookup: 'question'});
  assert.deepEqual(file.report, {completed: 2});
  assert.equal(file.completion.finished, true);
  assert.equal(file.completion.received_jobs, 2);
  api.download();
  assert.equal(downloads.length, 2);
  api.clear();
  assert.deepEqual(revoked, ['blob:download']);
  api.download();
  assert.equal(downloads.length, 2);
});

test('interrupted download contains only received jobs and is labelled unfinished', async () => {
  const {api, blobs} = setup();
  api.receive({event: 'detail_log_job', job: {status: 'error', extraction: {raw: 'bad JSON'}}});
  api.finish({event: 'error', error: 'stream ended'});
  const file = JSON.parse(await blobs[0].text());
  assert.equal(file.completion.finished, false);
  assert.equal(file.completion.error, 'stream ended');
  assert.equal(file.jobs[0].extraction.raw, 'bad JSON');
});

test('blocked automatic download leaves the manual link available without throwing', () => {
  const {api, ctx} = setup(true);
  assert.doesNotThrow(() => api.finish({event: 'done', stopped: true}));
  assert.equal(ctx.stState.detailDownload.url, 'blob:download');
  assert.match(ctx.stState.detailDownload.name, /^extraction-details-test-run\.json$/);
});
