// A flagged stress read's marks: where DRY fired and where the loop is.
import {benchmarkSource} from './benchmark_source.mjs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const page = benchmarkSource();
const cut = (from, to) => page.slice(page.indexOf(from), page.indexOf(to));
const ctx = vm.createContext({});
vm.runInContext(cut('function esc(', '\n// Fallback:') + cut('function pipeTables(', '\nfunction sanitize(')
  + cut('const CMP_MARK_OPEN', '\n/* What one read of a flagged stress page came back as')
  + ';globalThis.out = {cmpMarkRaw, cmpMarkSource, cmpMarkLegend, renderOutput};', ctx);
const {cmpMarkRaw, cmpMarkSource, cmpMarkLegend, renderOutput} = ctx.out;
const OPEN = {dry: '\uE000', loop: '\uE002'}, CLOSE = {dry: '\uE001', loop: '\uE003'};
const strip = s => s.replace(/[\uE000-\uE003]/g, '');

test('the raw view wraps exactly the marked runs, escaped', () => {
  const text = 'ok <b> then repeat repeat';
  const at = text.indexOf('repeat');
  const html = cmpMarkRaw(text, [{kind: 'dry', start: at, end: text.length}]);
  assert.equal(html, 'ok &lt;b&gt; then <mark class="cmp-mk cmp-mk-dry">repeat repeat</mark>');
});

test('overlapping DRY and loop runs carry both classes', () => {
  const html = cmpMarkRaw('abcdef', [{kind: 'dry', start: 1, end: 4}, {kind: 'loop', start: 3, end: 6}]);
  assert.equal(html, 'a<mark class="cmp-mk cmp-mk-dry">bc</mark>'
    + '<mark class="cmp-mk cmp-mk-dry cmp-mk-loop">d</mark>'
    + '<mark class="cmp-mk cmp-mk-loop">ef</mark>');
});

test('markers never land between cells, in a tag or on whitespace', () => {
  const text = '<table><tr><td>a</td><td>b</td></tr>\n<tr><td>c</td><td></td></tr></table>';
  const out = cmpMarkSource(text, [{kind: 'loop', start: text.indexOf('<tr>'), end: text.length}]);
  // The text is unchanged apart from the markers and the row/cell tags.
  assert.equal(strip(out).replace(/ data-mk="[^"]*"/g, ''), text);
  // Every marker sits inside a cell's text: none is left once cells are emptied.
  assert.ok(/[\uE000-\uE003]/.test(out));
  assert.ok(!/[\uE000-\uE003]/.test(out.replace(/<td[^>]*>[^<]*<\/td>/g, '<td></td>')));
  assert.equal(out.split(OPEN.loop).length - 1, out.split(CLOSE.loop).length - 1);
  // Every row and cell in the run is tagged, the empty cell too.
  assert.equal((out.match(/data-mk="loop"/g) || []).length, 6);
});

test('a pipe table still renders as one with markers in it', () => {
  const text = '| A | B |\n|---|---|\n| x | y |\n| x | y |';
  const out = cmpMarkSource(text, [{kind: 'dry', start: 0, end: text.length}]);
  const html = renderOutput(out);
  assert.ok(html.startsWith('<table>'), html);
  assert.equal(strip(html), renderOutput(text));
});

test('plain text closes its marker where the run ends', () => {
  const text = 'line one\nline two';
  const out = cmpMarkSource(text, [{kind: 'dry', start: 5, end: 8}]);
  assert.equal(out, 'line ' + OPEN.dry + 'one' + CLOSE.dry + '\nline two');
});

test('the legend names what is marked, and an estimated place', () => {
  assert.equal(cmpMarkLegend([]), '');
  assert.equal(cmpMarkLegend([{kind: 'dry', approx: true}, {kind: 'loop'}]),
               'DRY marked in blue (place estimated) · loop marked in amber');
});
