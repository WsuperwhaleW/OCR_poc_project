// The stress flags' file names, to copy: before a retry and after it.
import {benchmarkSource} from './benchmark_source.mjs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
const cut = (from, to) => page.slice(page.indexOf(from), page.indexOf(to));
const ctx = vm.createContext({});
vm.runInContext(cut('function esc(', '\n// Fallback:')
  + cut('const ST_READ_PROBLEMS', '\n// Plain http on a server')
  + ';globalThis.out = {stFlagFileLists, stCopyBarHtml};', ctx);
const {stFlagFileLists, stCopyBarHtml} = ctx.out;

const ALERTS = [
  {id: 1, kind: 'looped', case: 'sol003', state: 'resolved'},
  {id: 2, kind: 'looped', case: 'sol009', state: 'retry failed'},
  {id: 3, kind: 'dry', case: 'sol003'},                       // flag only: no state
  {id: 4, kind: 'dry', case: 'sol012', state: 'resolved'},
  {id: 5, kind: 'error', case: 'sol015'},
  {id: 6, kind: 'looped', case: 'sol009', state: 'no retry'},  // same file twice
  {id: 7, kind: 'wrong', case: 'sol001'},                      // not a read problem
  {id: 8, kind: 'recovered', case: 'sol003'},
];

const lists = Object.fromEntries(stFlagFileLists(ALERTS, a => a.case + '.pdf')
  .map(([key, , names]) => [key, [...names]]));

test('before a retry is every file flagged, each once', () => {
  assert.deepEqual(lists['loop-before'], ['sol003.pdf', 'sol009.pdf']);
  assert.deepEqual(lists['dry-before'], ['sol003.pdf', 'sol012.pdf']);
  assert.deepEqual(lists['all-before'], ['sol003.pdf', 'sol009.pdf', 'sol012.pdf', 'sol015.pdf']);
});

test('after a retry drops the resolved ones; a flag-only DRY stays', () => {
  assert.deepEqual(lists['loop-after'], ['sol009.pdf']);
  assert.deepEqual(lists['dry-after'], ['sol003.pdf']);
  assert.deepEqual(lists['all-after'], ['sol009.pdf', 'sol003.pdf', 'sol015.pdf']);
  assert.deepEqual(lists.errors, ['sol015.pdf']);
});

test('the bar carries one name per line and is empty with no read problem', () => {
  const html = stCopyBarHtml(ALERTS);
  assert.match(html, /data-st-copy-key="loop-before" data-st-copy="sol003\nsol009"/);
  assert.match(html, /Loop · after retry \(1\)/);
  assert.equal(stCopyBarHtml([{id: 1, kind: 'wrong', case: 'sol001'}]), '');
});
