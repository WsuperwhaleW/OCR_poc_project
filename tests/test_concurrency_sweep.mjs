// The concurrency sweep's report: one chart per metric against the levels.
import {benchmarkSource} from './benchmark_source.mjs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
const cut = (from, to) => page.slice(page.indexOf(from), page.indexOf(to));
const source = cut('const stSweepFmt', '\nfunction stReportHtml(');
const ctx = vm.createContext({
  esc: s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'),
  encodeURIComponent,
});
vm.runInContext(cut('const stNum', '\nfunction stTile(') + source
  + ';globalThis.out = {stSweepHtml, stSweepChart, stSweepFmt};', ctx);
const {stSweepHtml, stSweepChart, stSweepFmt} = ctx.out;

const pct = (p50, p99, mean) => ({count: 4, p50, p99, mean});
const level = (lv, tps, view = {}) => ({
  level: lv, jobs: lv * 4, wall_seconds: 10,
  client: {tokens_per_second: tps * .9, latency: pct(lv, lv * 2, lv * 1.2), achieved: lv * .9,
           error_rate: 0, completed: lv * 4},
  servers: [{role: 'reader', url: 'http://vllm:8000', view: {
    generation_tokens_per_s: tps, prompt_tokens_per_s: tps * 10, requests_per_s: lv / 10,
    ttft: pct(.1 * lv, .3 * lv, .15 * lv), itl: pct(.02, .05, .03), e2e: pct(lv, 2 * lv, 1.3 * lv),
    queue: pct(0, .1 * lv, .05), prefill: pct(.1, .2, .1), decode: pct(1, 2, 1.5),
    decode_rate: 33.3, running: {mean: lv * .8, max: lv}, waiting: {mean: 0, max: 1},
    kv_cache: {mean: .1 * lv, max: .12 * lv}, preemptions: 0, ...view}}],
  errors: [],
});

test('every metric gets its own chart, against the levels', () => {
  const html = stSweepHtml({levels: [level(1, 100), level(2, 190), level(4, 300)],
    summary: {peak_level: 4, peak_tps: 300, knee_level: null}}, {sweep: []});
  for (const title of ['Output throughput', 'Time to first token', 'Time between tokens',
    'End-to-end request latency', 'Latency per job', 'Queue time', 'Requests in vLLM',
    'KV cache usage', 'Concurrency reached', 'Error rate']) {
    assert.ok(html.includes('<figcaption>' + title), title);
  }
  // Three levels on the x axis of each chart, and a point per level per line.
  assert.match(html, />1<\/text>.*>2<\/text>.*>4<\/text>/s);
  assert.ok(html.includes('4 concurrent · p99'));
  assert.ok(html.includes('peaked at <b>300 tok/s</b> with <b>4</b> concurrent'));
  assert.ok(html.includes('download="concurrency-sweep.csv"'));
  const csv = decodeURIComponent(html.match(/data:text\/csv;charset=utf-8,([^"]+)/)[1]);
  assert.equal(csv.split('\n').length, 4);
  assert.ok(csv.startsWith('Concurrent,Jobs,Wall s,Gen tok/s'));
});

test('a missing figure breaks the line instead of bridging it', () => {
  const svg = stSweepChart([{level: 1}, {level: 2}, {level: 4}], 'x', 'n',
    [{label: 'a', cls: 'st-s1', value: row => row.level === 2 ? null : row.level}]);
  assert.equal((svg.match(/ M/g) || []).length, 2);
  assert.equal((svg.match(/<circle/g) || []).length, 2);
  assert.equal(stSweepChart([{level: 1}], 'x', 'n', [{label: 'a', cls: 'st-s1', value: () => null}]), '');
});

test('before the first level ends, the plan is shown instead', () => {
  const html = stSweepHtml({levels: []}, {sweep: [{level: 1, jobs: 4}, {level: 2, jobs: 8}]});
  assert.ok(html.includes('1 (4 jobs), 2 (8 jobs)'));
  assert.ok(!html.includes('<svg'));
});

test('seconds under one read as milliseconds', () => {
  assert.equal(stSweepFmt(0.025, 's'), '25 ms');
  assert.equal(stSweepFmt(null, 's'), '—');
});
