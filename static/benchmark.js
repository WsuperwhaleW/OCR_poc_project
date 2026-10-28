
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const prepBar = document.getElementById('prepBar');
const prepCap = document.getElementById('prepCap');
const prepTrunc = document.getElementById('prepTrunc');
const prepDl = document.getElementById('prepDl');
const chip = document.getElementById('filechip');
const chipName = document.getElementById('filechipname');
const runBtn = document.getElementById('run');
const stopBtn = document.getElementById('stop');
const clearBtn = document.getElementById('clear');
const copyBtn = document.getElementById('copy');
const detailSel = document.getElementById('detail');
const renderEl = document.getElementById('render');
const rawEl = document.getElementById('raw');
const pagebar = document.getElementById('pagebar');
const warn = document.getElementById('warn');
const detailNote = document.getElementById('detailNote');
const meta = document.getElementById('meta');
const metaTrunc = document.getElementById('metaTrunc');
const statsEl = document.getElementById('stats');
const groupingEl = document.getElementById('grouping');
const tableViewEl = document.getElementById('tableview');
const tabTable = document.getElementById('tabTable');
/* Whether each table agent runs on the next extraction, as the server last
   said -- painted from its replies, never from what was clicked. */
let TABLE_AGENTS = window.BENCHMARK_BOOTSTRAP.table_agents;
/* Which Table-tab view is shown (null = the default), and the table_agent
   events of the run in flight. Up here so nothing reaches them early. */
let tvStage = null, tableLive = null;
const tabGroup = document.getElementById('tabGroup');
/* The page grouping of the run in progress, built from its `segmenting` /
   `segment_page` / `segmented` events. Outside the render because the view is
   repainted on every page, and a group left open to be checked must stay open
   while the walk moves on. */
let segWalk = null;
let segOpen = null;
const segOpenHeads = new Set();
const dot = document.getElementById('dot');
const serverText = document.getElementById('serverText');
const serverReason = document.getElementById('serverReason');
const recheckBtn = document.getElementById('recheck');
const serverGroupHead = document.getElementById('serverGroupHead');
const serverGroupNote = document.getElementById('serverGroupNote');
const modelRole = document.getElementById('modelRole');
let server = window.BENCHMARK_BOOTSTRAP.server;
let extractInfo = window.BENCHMARK_BOOTSTRAP.extract;
/* The inference engines a server can be pinned to, besides Auto. From the
   server, so the page cannot offer a kind the backend does not know. */
const SERVER_KINDS = window.BENCHMARK_BOOTSTRAP.server_kinds;
const serverKindSel = document.getElementById('serverKindSel');
const extractKindSel = document.getElementById('extractKindSel');
const extractKindField = document.getElementById('extractKindField');
const extractEndpointSel = document.getElementById('extractEndpointSel');
const fxExtractEndpoint = document.getElementById('fxExtractEndpoint');
const extractCustomRow = document.getElementById('extractCustomRow');
const extractEndpointCustom = document.getElementById('extractEndpointCustom');
const extractEndpointUse = document.getElementById('extractEndpointUse');
const extractServerNote = document.getElementById('extractServerNote');
const rtExtractEndpoint = document.getElementById('rtExtractEndpoint');
const rtExtractServerRow = document.getElementById('rtExtractServerRow');
const rtExtractServerNote = document.getElementById('rtExtractServerNote');
let pageStats = [], ticker = null, runStart = 0, liveTokens = 0, curPage = -1;
// Page index -> the Detail a loop retry re-read it at, during a read.
let pageRetried = {};
/* The cached page image for page `i` (0-based) of the current job. A page a loop
   retry re-read is a bigger image under the same URL, and the browser holds page
   images for an hour, so its Detail goes on the end to fetch the new one. */
function pageSrc(i) {
  const d = pageRetried[i] || (pageStats[i] && pageStats[i].loop_retries && pageStats[i].detail);
  return API + '/page/' + encodeURIComponent(job) + '/' + i
    + (d ? '?detail=' + encodeURIComponent(d) : '');
}
const tabRendered = document.getElementById('tabRendered');
const tabRaw = document.getElementById('tabRaw');
const cmpToggle = document.getElementById('cmpToggle');
const cmpHint = document.getElementById('cmpHint');
const tabFields = document.getElementById('tabFields');
const fieldsEl = document.getElementById('fields');
const fldStatus = document.getElementById('fldStatus');
const pipeline = document.getElementById('pipeline');
const fldBody = document.getElementById('fldBody');
const fldRerun = document.getElementById('fldRerun');
const fldMode = document.getElementById('fldMode');
const extractSel = document.getElementById('extractSel');
const extractNote = document.getElementById('extractNote');
const fldCopy = document.getElementById('fldCopy');
const fldRawChk = document.getElementById('fldRawChk');
const fldRaw = document.getElementById('fldRaw');
/* The extraction being DISPLAYED, which is what this has always meant. On a
   file holding several documents it points at the one the strip has selected,
   and `extractedFile` holds the whole result -- so every function below renders
   one document exactly as it did when a file could only ever be one. */
let extracted = null;
/* The whole pass-2 result: the same object as `extracted` for a one-document
   file, and the merged one carrying `documents` for a split file. */
let extractedFile = null;
/* Which document of it is on screen. Reset to 0 by every new extraction, so a
   two-document file followed by a one-document one cannot leave the strip
   pointing at a document that is no longer there. */
let extractedDoc = 0;
function setExtracted(result) {
  extractedFile = result;
  const docs = (result && result.documents) || null;
  extractedDoc = 0;
  /* A one-document file has no `documents` list at all, so `extracted` is the
     result itself and nothing downstream can tell the difference. That is the
     point: the split added a shape, it did not change the old one. */
  extracted = (docs && docs.length) ? docs[0] : result;
  if (docs && docs.length) {
    /* Stage 0 is per document, and the live `classified` event holds the LAST
       document's -- so a finished split file would otherwise draw document 3's
       type and confidence over document 1's values. Same reason `showDocument`
       swaps it: the strip and the pipeline have to be describing one document.
       The run is over, so nothing is running any more either. */
    classified = extracted.classified || null;
    extractSteps = extracted.steps || null;
    runningDoc = null;
  }
  tvStage = null;
  if (!result || result.pending) tableLive = null;
  paintTableTab();
}
function showDocument(index) {
  const docs = (extractedFile && extractedFile.documents) || null;
  if (!docs || !docs[index]) return;
  extractedDoc = index;
  extracted = docs[index];
  /* Stage 0 is per document, and the live event holds the LAST document's --
     so a strip that switched the fields without switching the classification
     would draw document 2's type over document 1's values. */
  classified = extracted.classified || null;
  extractSteps = extracted.steps || null;
  paintFields();
  if (tabAccuracy.classList.contains('on')) paintAccuracy();
  if (curMode === 'table') paintTableView();
}
/* The Fields-only pane: pass 2 against a fixture's hand-written transcript.
   Every one of these is absent when no case has ground truth, so everything
   that touches them checks first. */
const fxCase = document.getElementById('fxCase');
const fxMode = document.getElementById('fxMode');
const fxSchemaFirst = document.getElementById('fxSchemaFirst');
const fxRun = document.getElementById('fxRun');
const fxNote = document.getElementById('fxNote');
const fxTruthNote = document.getElementById('fxTruthNote');
const fxServerNote = document.getElementById('fxServerNote');
const fxServerText = document.getElementById('fxServerText');
const fxDot = document.getElementById('fxDot');
const FX_CASES = window.BENCHMARK_BOOTSTRAP.cases;

/* What a case will be asked for, as a phrase for a note. The form is decided by
   the document's TYPE and the types differ in what they ask -- a receipt is not
   asked its own number or date, an invoice is not asked for a cheque -- so a
   pane that names the document without naming the form is only half of what a
   reader needs before pressing Run.

   Empty for a case the manifest does not classify: the default form is asked,
   and naming a type there would claim a classification nobody made. */

function caseFormNote(id) {
  const c = (FX_CASES || []).find(x => x.id === id);
  if (!c || !(c.doc_types || []).length) return '';
  const types = c.doc_types.join(' + ').replace(/_/g, ' ').toLowerCase();
  const n = (c.fields || []).length;
  /* How many of them the requirement DEMANDS, said before anything runs: the
     count decides what the field score will be taken over, so "11 fields" alone
     names a form and not the thing that will be measured. Omitted where no
     requirement covers the type -- there, none is required and a "0 required"
     would read as a form nobody can pass. */
  const m = (c.mandatory || []).length;
  return types + (n ? ' · ' + n + ' fields' : '')
    + (m ? ' (' + m + ' required)'
      /* No requirement yet, so the form is the base field set and the score
         will be over every key the ground truth states. Named before anything
         runs, for the same reason the required count is. */
      : c.unknown_type ? ' (unknown type — scored on the base fields)' : '')
    + packNote(c)
    + ' · ';
}
/* What ELSE is in the file, where the file holds more than one document.
   Everything above it describes document 1 -- the form, the required count, the
   type -- because that is what the truth file describes and what the extractor
   is scored on. Said here so a reader knows before pressing Run: a seven-page
   pack otherwise announces itself as one billing note, and the other six
   documents turn up only in the result.

   Drawn from the manifest's own per-document list, never worked out from the
   page count: which pages are which document is a person's reading, and a
   second copy of it in JavaScript would go stale the next time a pack arrives.
   Empty for the thirteen ordinary cases, so nothing extra is drawn for them. */
function packNote(c) {
  const docs = (c && c.documents) || [];
  if (docs.length < 2) return '';
  const kinds = [];
  for (const d of docs) {
    const name = (d.doc_types || []).join(' + ').replace(/_/g, ' ').toLowerCase();
    if (name && kinds.indexOf(name) < 0) kinds.push(name);
  }
  return ' · ' + docs.length + ' documents in this file'
    + (kinds.length > 1 ? ' (' + kinds.join(', ') + ')' : '');
}
/* Set while the transcript on screen came out of `solution/<id>.md` rather than
   out of a read. It is what keeps a Re-extract from the Fields tab scored: the
   case cannot be recovered from a page job, because no page was ever read. */
let truthOnlyCase = null;
const tabAccuracy = document.getElementById('tabAccuracy');
const accuracyEl = document.getElementById('accuracy');
const accHead = document.getElementById('accHead');
const accBars = document.getElementById('accBars');
const accDiff = document.getElementById('accDiff');
const mapChip = document.getElementById('mapchip');
let truth = null, matchedCase = null;
const compareEl = document.getElementById('compare');
const cmpImg = document.getElementById('cmpImg');
const cmpImgPane = document.getElementById('cmpImgPane');
const cmpTextPane = document.getElementById('cmpTextPane');
const cmpRender = document.getElementById('cmpRender');
const cmpRawText = document.getElementById('cmpRawText');
const cmpRawChk = document.getElementById('cmpRaw');
const cmpSync = document.getElementById('cmpSync');
const cmpLabel = document.getElementById('cmpLabel');
const cmpPrev = document.getElementById('cmpPrev');
const cmpNext = document.getElementById('cmpNext');
const cmpOpen = document.getElementById('cmpOpen');
const cmpDl = document.getElementById('cmpDl');
const cmpModes = document.getElementById('cmpModes');
const cmpGrid = document.getElementById('cmpGrid');
const cmpColImg = document.getElementById('cmpColImg');
const cmpColModel = document.getElementById('cmpColModel');
const cmpColTruth = document.getElementById('cmpColTruth');
const cmpTruthPane = document.getElementById('cmpTruthPane');
const cmpTruthRender = document.getElementById('cmpTruthRender');
const cmpTruthText = document.getElementById('cmpTruthText');
const cmpTruthCap = document.getElementById('cmpTruthCap');
const cmpColFields = document.getElementById('cmpColFields');
const cmpFieldsPane = document.getElementById('cmpFieldsPane');
const cmpFieldsCap = document.getElementById('cmpFieldsCap');
const cmpImgCap = document.getElementById('cmpImgCap');
const cmpModelCap = document.getElementById('cmpModelCap');
let job = null, pageTexts = [], cmpPage = 0;
// Which two columns are on. 'source-model' is the original view, so a page that
// never touches the picker behaves exactly as it did before.
let cmpView = 'source-model';
// Ground truth, fetched once per case and kept: solution/<id>.md does not change
// under a running app, and re-fetching on every repaint would flicker the pane.
const truthCache = {};
// `current` is the workspace: one file, dropped but not yet run. A new drop
// replaces it -- reading a document is a measurement, and they are tested one at
// a time. Several documents go through compare.py or randomtest.py.
let current = null, controller = null, text = '';
let workspaceCase = '', workspaceSourceSeq = 0;

/* ---------- left-hand panes ---------- */
const sideTabs = document.getElementById('sideTabs');
const runNote = document.getElementById('runNote');
/* The pane list is derived from the tabs rather than written out beside them.
   It used to be a literal ['source', 'queue', 'fields'], and adding another tab
   left every pane hidden -- the button highlighted, nothing appeared, and there
   was no error anywhere to say why. A tab and its pane are one thing; naming
   them in two places means one of them is eventually forgotten. */
function showPane(name) {
  selectedPane = name;
  [...sideTabs.querySelectorAll('button[data-pane]')].forEach(b => {
    const on = b.dataset.pane === name;
    b.classList.toggle('on', on);
    const pane = document.getElementById(
      'pane' + b.dataset.pane[0].toUpperCase() + b.dataset.pane.slice(1));
    if (pane) pane.classList.toggle('on', on);
  });
  scOpen = name === 'schema';
  stressPaneOpen = name === 'stress';
  historyOpen = stressPaneOpen && stressView === 'history';
  paintMain();
  enableCompare();   // the workspace's run or a stress page, whichever pane this is
  if (scOpen) paintSchema();
  if (historyOpen) historyLoad();
  else if (stressPaneOpen) stPaint();
  else if (name === 'random') rtPaint();
  else if (name === 'settings' && window.loadUserSettings) window.loadUserSettings();
}

/* One pane, so nothing moves between panes any more. `paintPanes` and
   `syncEngineTab` went with the second tab: the engine is a control INSIDE the
   workspace now, and a control cannot disagree with the tab it sits in. */

/* The selected menu owns the main column. Background run updates may repaint
   their report, but cannot replace the user's current view. Keep each report
   in its card so returning to its tab restores it without clearing the run.
   `var` and element IDs allow early callers before the card consts exist. */
var scOpen = false, historyOpen = false;
var selectedPane = 'source', stressPaneOpen = false, stressView = 'run', stRunning = false;
/* A flagged stress page in the Compare card, in place of the workspace's run:
   {alert, case, page, truthCase, at, reads: [{detail, text, looped, truncated,
   dry, src}]}. Only while the stress pane is open (`cmpInStress`); the
   workspace's own run is untouched and comes back with its pane. */
var cmpOverride = null;
var stEvidenceRequest = 0;
const loadStressEvidence = createEvidenceCache({fetch: (...args) => fetch(...args)});
function paintMain() {
  const el = id => document.getElementById(id);
  const st = stressPaneOpen && !historyOpen;
  const rt = selectedPane === 'random';
  const settings = selectedPane === 'settings';
  el('settingsCard').hidden = !settings;
  el('runlogCard').hidden = settings;
  el('historyCard').hidden = !historyOpen;
  el('schemaCard').hidden = !scOpen;
  el('stCard').hidden = !st;
  el('rtCard').hidden = !rt;
  el('resultCard').hidden = scOpen || historyOpen || rt || st || settings;
  // In the stress pane, Compare shows a flagged page once one is clicked
  // (`stShowEvidence`) -- the live run's or a saved run's -- and nothing before.
  el('compareCard').hidden = scOpen || rt || settings || (stressPaneOpen && !cmpOverride);
  let hasActions = false;
  for (const name of ['source', 'ocr', 'fields', 'random', 'stress']) {
    const group = el(name + 'Actions');
    group.hidden = name !== selectedPane || !group.querySelector('button');
    if (!group.hidden) hasActions = true;
  }
  el('testActions').hidden = !hasActions;
  document.body.classList.toggle('has-test-actions', hasActions);
}

function setTestStatus(prefix, state) {
  const status = document.getElementById(prefix + 'Status');
  status.dataset.state = state;
  status.textContent = {ready: 'Ready', running: 'Running', stopping: 'Stopping',
    completed: 'Completed', stopped: 'Stopped', failed: 'Failed'}[state];
}

function setReadStatus(state) {
  setTestStatus('read', state);
  setTestStatus('oc', state);
}

// Move the existing document buttons, preserving their handlers and the
// server-rendered absence of controls when benchmark cases are unavailable.
for (const [pane, ids] of Object.entries({source: ['run', 'stop', 'clear'],
    ocr: ['ocRun', 'ocStop'], fields: ['fxRun']})) {
  const target = document.getElementById(pane + 'ActionButtons');
  for (const id of ids) {
    const button = document.getElementById(id);
    if (!button) continue;
    const row = button.parentElement;
    target.appendChild(button);
    if (!row.children.length) row.remove();
  }
}
document.getElementById('clear').textContent = 'Clear workspace';

// Keep the footer inside the visible sidebar even while the page header or
// global monitor occupies space above it. On narrow screens CSS docks it.
const sideCard = document.querySelector('.side');
let sideSizeFrame = null;
function sizeSidebar() {
  if (sideSizeFrame !== null) return;
  sideSizeFrame = requestAnimationFrame(() => {
    sideSizeFrame = null;
    const stickyTop = Math.max(18, document.querySelector('.page-head').getBoundingClientRect().bottom + 12);
    sideCard.style.setProperty('--sidebar-top', stickyTop + 'px');
    const top = Math.max(stickyTop, sideCard.getBoundingClientRect().top);
    sideCard.style.setProperty('--side-height', Math.max(180, innerHeight - top - 18) + 'px');
    const actionHeight = document.getElementById('testActions').getBoundingClientRect().height;
    document.body.style.setProperty('--run-actions-space', (actionHeight + 18) + 'px');
  });
}
window.addEventListener('resize', sizeSidebar);
window.addEventListener('scroll', sizeSidebar, {passive: true});
const sideSizeObserver = new ResizeObserver(sizeSidebar);
sideSizeObserver.observe(document.querySelector('.page-head'));
sideSizeObserver.observe(document.getElementById('monCard'));
sideSizeObserver.observe(document.getElementById('testActions'));
sizeSidebar();

// Remember the preferred width, while keeping at least 480px for the result.
// A smaller window temporarily clamps it; returning to a larger screen restores it.
(() => {
  const grid = sideCard.closest('.grid');
  const handle = document.getElementById('sidebarResize');
  const desktop = matchMedia('(min-width: 1081px)');
  const key = 'ocrSidebarWidth:' + window.OCR_BASE;
  let preferred = 420, drag = null;
  try {
    const saved = Number(localStorage.getItem(key));
    if (Number.isFinite(saved) && saved >= 320) preferred = Math.min(640, saved);
  } catch { /* Storage can be unavailable in a private window. */ }
  const limits = () => ({min: 320, max: Math.max(320, Math.min(640, grid.clientWidth - 498))});
  function apply() {
    const {min, max} = limits();
    grid.style.setProperty('--sidebar-width', preferred + 'px');
    handle.setAttribute('aria-valuemax', max);
    handle.setAttribute('aria-valuenow', Math.round(Math.max(min, Math.min(max, preferred))));
    sizeSidebar();
  }
  function save() {
    try { localStorage.setItem(key, String(preferred)); } catch { /* Keep the current layout. */ }
  }
  function change(width) {
    const {min, max} = limits();
    preferred = Math.round(Math.max(min, Math.min(max, width)));
    apply();
  }
  function finish(event) {
    if (!drag || event.pointerId !== drag.id) return;
    drag = null;
    document.body.classList.remove('resizing-sidebar');
    save();
  }
  handle.addEventListener('pointerdown', event => {
    if (!desktop.matches || event.button !== 0 || drag) return;
    event.preventDefault();
    drag = {id: event.pointerId, x: event.clientX, width: sideCard.getBoundingClientRect().width};
    handle.setPointerCapture(event.pointerId);
    handle.focus();
    document.body.classList.add('resizing-sidebar');
  });
  handle.addEventListener('pointermove', event => {
    if (drag && event.pointerId === drag.id) change(drag.width + event.clientX - drag.x);
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    handle.addEventListener(name, finish);
  }
  function reset() { preferred = 420; apply(); save(); }
  handle.addEventListener('dblclick', reset);
  handle.addEventListener('keydown', event => {
    if (!desktop.matches) return;
    const {min, max} = limits();
    const current = sideCard.getBoundingClientRect().width;
    const step = event.shiftKey ? 40 : 20;
    if (event.key === 'ArrowLeft') change(current - step);
    else if (event.key === 'ArrowRight') change(current + step);
    else if (event.key === 'Home') change(min);
    else if (event.key === 'End') change(max);
    else if (event.key === 'Enter') reset();
    else return;
    event.preventDefault();
    save();
  });
  window.addEventListener('resize', apply);
  apply();
})();
paintMain();
sideTabs.addEventListener('click', e => {
  const btn = e.target.closest('button[data-pane]');
  if (btn) showPane(btn.dataset.pane);
});

/* ---------- output renderer (offline, no CDN) ----------
   The model is prompted for a verbatim transcript: plain text plus HTML tables.
   Text runs are shown with white-space:pre-wrap so the original line breaks,
   indentation and column spacing survive; Markdown is deliberately NOT
   interpreted, because '*' and '#' in a document are literal characters. */
function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
// Fallback: the prompt asks for HTML tables, but the model sometimes emits a
// Markdown pipe table anyway. Render those as real tables rather than as
// literal pipe characters.
function pipeTables(chunk) {
  const lines = chunk.split('\n');
  const out = [];
  let i = 0;
  const isRow = l => /^\s*\|.*\|\s*$/.test(l);
  const isRule = l => /^\s*\|[\s:|-]+\|\s*$/.test(l);
  const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  while (i < lines.length) {
    if (isRow(lines[i]) && i + 1 < lines.length && isRule(lines[i + 1])) {
      const head = cells(lines[i]);
      i += 2;
      const body = [];
      while (i < lines.length && isRow(lines[i])) body.push(cells(lines[i++]));
      out.push('<table><tr>' + head.map(c => '<th>' + esc(c) + '</th>').join('') + '</tr>'
        + body.map(r => '<tr>' + r.map(c => '<td>' + esc(c) + '</td>').join('') + '</tr>').join('')
        + '</table>');
    } else {
      out.push({ text: lines[i++] });
    }
  }
  // Re-group consecutive plain lines so pre-wrap spacing survives.
  const merged = [];
  let buf = [];
  const flush = () => {
    if (!buf.length) return;
    const body = buf.join('\n').replace(/^\n+|\n+$/g, '');
    if (body.trim()) merged.push('<div class="txt">' + esc(body) + '</div>');
    buf = [];
  };
  for (const part of out) {
    if (typeof part === 'string') { flush(); merged.push(part); }
    else buf.push(part.text);
  }
  flush();
  return merged.join('\n');
}
function renderOutput(src) {
  const out = [];
  const re = /<table[\s\S]*?<\/table>/gi;
  let last = 0, m;
  while ((m = re.exec(src)) !== null) {
    out.push(pipeTables(src.slice(last, m.index)));
    out.push(m[0]);
    last = re.lastIndex;
  }
  out.push(pipeTables(src.slice(last)));
  return out.filter(Boolean).join('\n');
}
function sanitize(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  doc.body.querySelectorAll('script,style,iframe,object,embed,link,meta').forEach(n => n.remove());
  doc.body.querySelectorAll('*').forEach(el => {
    for (const attr of [...el.attributes]) {
      const n = attr.name.toLowerCase();
      if (n.startsWith('on') || (['href', 'src'].includes(n) && /^\s*javascript:/i.test(attr.value))) {
        el.removeAttribute(attr.name);
      }
    }
  });
  return doc.body.innerHTML;
}
function paint() {
  rawEl.textContent = text || 'Nothing yet.';
  rawEl.classList.toggle('empty', !text);
  if (text) {
    renderEl.innerHTML = sanitize(renderOutput(text));
    renderEl.classList.remove('empty');
  } else {
    renderEl.textContent = 'Nothing yet.';
    renderEl.classList.add('empty');
  }
}

/* ---------- tabs ---------- */
let curMode = 'raw';
function showTab(mode) {
  curMode = mode;
  renderEl.style.display = mode === 'rendered' ? 'block' : 'none';
  rawEl.style.display = mode === 'raw' ? 'block' : 'none';
  accuracyEl.style.display = mode === 'accuracy' ? 'block' : 'none';
  fieldsEl.style.display = mode === 'fields' ? 'block' : 'none';
  layoutEl.style.display = mode === 'layout' ? 'block' : 'none';
  groupingEl.style.display = mode === 'grouping' ? 'block' : 'none';
  tableViewEl.style.display = mode === 'table' ? 'block' : 'none';
  tabTable.classList.toggle('on', mode === 'table');
  tabGroup.setAttribute('aria-pressed', mode === 'grouping');
  tabGroup.classList.toggle('on', mode === 'grouping');
  tabLayout.setAttribute('aria-pressed', mode === 'layout');
  tabLayout.classList.toggle('on', mode === 'layout');
  rawoutEl.style.display = mode === 'rawout' ? 'block' : 'none';
  tabRawOut.setAttribute('aria-pressed', mode === 'rawout');
  tabRawOut.classList.toggle('on', mode === 'rawout');
  statsEl.style.display = ['accuracy', 'fields', 'layout', 'rawout', 'grouping', 'table'].includes(mode) ? 'none' : '';
  tabRendered.classList.toggle('on', mode === 'rendered');
  tabRaw.classList.toggle('on', mode === 'raw');
  tabAccuracy.classList.toggle('on', mode === 'accuracy');
  tabFields.classList.toggle('on', mode === 'fields');
  if (mode === 'accuracy') paintAccuracy();
  if (mode === 'fields') paintFields();
  if (mode === 'layout') paintLayout();
  if (mode === 'rawout') paintRawOut();
  if (mode === 'grouping') paintGrouping();
  if (mode === 'table') paintTableView();
}
tabRendered.addEventListener('click', () => showTab('rendered'));
tabRaw.addEventListener('click', () => showTab('raw'));
tabAccuracy.addEventListener('click', () => { if (!tabAccuracy.disabled) showTab('accuracy'); });
tabFields.addEventListener('click', () => { if (!tabFields.disabled) showTab('fields'); });
tabTable.addEventListener('click', () => { if (!tabTable.disabled) showTab('table'); });

/* ---------- raw output: the reply before anything touched it ----------------
   Three steps stand between what the model said and what the Markdown tab shows
   -- the fence strip, the layout flatten, and normalise_output -- and each one
   drops something deliberately. This is the only place the dropped material is
   visible: the <page_number> tags and headings on a Markdown profile, and the
   bbox coordinates on a layout one, which the transcript has no way to carry.

   Per page, because that is the unit the model actually answered in; the
   Markdown tab's page separators are this app's, not the model's. */
const rawoutEl = document.getElementById('rawout');
const tabRawOut = document.getElementById('tabRawOut');
const rawoutBody = document.getElementById('rawoutBody');
const roLabel = document.getElementById('roLabel');
const roSize = document.getElementById('roSize');
const roPrev = document.getElementById('roPrev');
const roNext = document.getElementById('roNext');
const roCopy = document.getElementById('roCopy');

/* Shared by both toggles: where to go back to when one is pressed again. */
let lastMode = 'raw';
let roPage = 0;

function roRaw(index) {
  const st = pageStats[index] || {};
  return typeof st.raw === 'string' ? st.raw : '';
}
function enableRawOut() {
  const on = pageStats.some(st => st && typeof st.raw === 'string' && st.raw !== '');
  tabRawOut.disabled = !on;
  if (!on && curMode === 'rawout') showTab(lastMode === 'rawout' ? 'raw' : lastMode);
}

function paintRawOut() {
  const total = pageStats.length || 1;
  roPage = Math.max(0, Math.min(roPage, total - 1));
  const body = roRaw(roPage);
  const paged = total > 1;
  roPrev.hidden = roNext.hidden = !paged;
  roLabel.textContent = paged ? 'Page ' + (roPage + 1) + ' of ' + total : '';
  roPrev.disabled = roPage === 0;
  roNext.disabled = roPage >= total - 1;
  roSize.textContent = body
    ? body.length.toLocaleString() + ' characters, exactly as returned'
    : '';
  rawoutBody.textContent = body || 'This page returned nothing.';
}

roPrev.addEventListener('click', () => { roPage--; paintRawOut(); });
roNext.addEventListener('click', () => { roPage++; paintRawOut(); });
roCopy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(roRaw(roPage));
    roCopy.textContent = 'Copied';
    setTimeout(() => { roCopy.textContent = 'Copy'; }, 1200);
  } catch (err) { roCopy.textContent = 'Copy failed'; }
});
tabRawOut.addEventListener('click', () => {
  if (tabRawOut.disabled) return;
  if (curMode === 'rawout') { showTab(lastMode); return; }
  lastMode = curMode;
  showTab('rawout');
});

/* ---------- layout view: where the model says it found each block ----------
   Only a profile that answers with geometry produces this, so the button stays
   disabled until a run actually returns boxes -- an empty stage would otherwise
   read as "the model found nothing" on every Markdown run ever made.

   It is a toggle rather than a fifth tab because it answers a question about the
   tab you are already on ("where on the page did that come from?"), so pressing
   it twice puts you back where you were. */
const layoutEl = document.getElementById('layout');
const tabLayout = document.getElementById('tabLayout');
const layStage = document.getElementById('layStage');
const layPop = document.getElementById('layPop');
const layLabel = document.getElementById('layLabel');
const layCount = document.getElementById('layCount');
const layLegend = document.getElementById('layLegend');
const layPrev = document.getElementById('layPrev');
const layNext = document.getElementById('layNext');

/* Stroke colours for the categories dots.ocr can return. Literals, for the
   reason the stylesheet gives: they sit on a photograph, not on the page. */
const LAY_COLOURS = {
  'Text': '#0e6f78', 'Section-header': '#4a5cae', 'Page-header': '#7a4f9e',
  'Page-footer': '#5a6470', 'Picture': '#6f7c82', 'Table': '#a8760e',
  'Title': '#b3372c', 'List-item': '#2f6b46', 'Caption': '#8a5a2b',
  'Footnote': '#5a6470', 'Formula': '#9c5f2e'
};
const layColour = c => LAY_COLOURS[c] || '#6f7c82';

let layPage = 0, laySel = null;
/* The image's own pixel size, needed to crop a Picture block out of it. Set
   once the page image loads; until then a crop is not drawn at all. */
let layNat = null, layPageSrc = '';

/* Boxes for one page, from the stats that page reported. */
function layBlocks(index) {
  const st = pageStats[index] || {};
  return (st.layout || []).filter(b => b.bbox);
}
function layAnyBoxes() {
  return pageStats.some(st => (st && st.layout || []).some(b => b.bbox));
}
function enableLayout() {
  const on = layAnyBoxes() && !!job;
  tabLayout.disabled = !on;
  /* A re-run on a Markdown profile must not strand the user in an empty view. */
  if (!on && curMode === 'layout') showTab(lastMode === 'layout' ? 'raw' : lastMode);
}

function paintLayout() {
  const total = pageStats.length || 1;
  layPage = Math.max(0, Math.min(layPage, total - 1));
  const blocks = layBlocks(layPage);
  const paged = total > 1;
  layPrev.hidden = layNext.hidden = !paged;
  layLabel.textContent = paged ? 'Page ' + (layPage + 1) + ' of ' + total : '';
  layPrev.disabled = layPage === 0;
  layNext.disabled = layPage >= total - 1;
  layCount.textContent = blocks.length + (blocks.length === 1 ? ' block' : ' blocks')
    + ' · in the order the reader returned them';

  const cats = [...new Set(blocks.map(b => b.category).filter(Boolean))];
  layLegend.innerHTML = cats.map(c =>
    '<span><i class="lay-swatch" style="background:' + layColour(c) + '"></i>'
    + esc(c) + '</span>').join('');

  /* The image is the one the model was sent, straight from the job cache, so a
     box drawn at its own coordinates lands where the model put it. The viewBox
     is set from the image's natural size once it loads, for the same reason. */
  const src = pageSrc(layPage);
  layPageSrc = src;
  layNat = null;
  const boxes = blocks.map((b, i) => {
    const x = b.bbox[0], y = b.bbox[1];
    const w = Math.max(1, b.bbox[2] - x), h = Math.max(1, b.bbox[3] - y);
    const col = layColour(b.category);
    return '<g><rect class="lay-box' + (laySel === i ? ' on' : '') + '" data-i="' + i
      + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h
      + '" stroke="' + col + '"></rect>'
      + '<rect x="' + x + '" y="' + (y - 16) + '" width="' + (String(i + 1).length * 9 + 8)
      + '" height="16" fill="' + col + '"></rect>'
      + '<text class="lay-tag" x="' + (x + 4) + '" y="' + (y - 4) + '">' + (i + 1)
      + '</text></g>';
  }).join('');

  layStage.innerHTML = '<img id="layImg" alt="the page as the model received it" src="'
    + esc(src) + '"><svg id="laySvg" preserveAspectRatio="none"></svg>';
  const img = document.getElementById('layImg');
  const svg = document.getElementById('laySvg');
  const fit = () => {
    if (!img.naturalWidth) return;
    layNat = { w: img.naturalWidth, h: img.naturalHeight };
    svg.setAttribute('viewBox', '0 0 ' + layNat.w + ' ' + layNat.h);
    svg.innerHTML = boxes;
    if (laySel !== null) layPopPin(laySel);
  };
  if (img.complete) fit(); else img.addEventListener('load', fit, { once: true });

}

/* ---------- the hover popup ----------
   Three shapes, because a block's category says what its text *is* and reading
   it as a string throws that away: a Table's text is table markup, and a
   Picture has no text at all -- the prompt tells the model to omit it, so an
   empty body there is obedience and not a gap. Rendering each as what it is
   makes a wrong box obvious at a glance, which four numbers never do. */
function layPopBody(b) {
  const isTable = b.category === 'Table' || /<table[\s>]/i.test(b.text || '');
  if (isTable && b.text) return sanitize(renderOutput(b.text));
  if (b.category === 'Picture') {
    /* Crop the region out of the page the model was actually sent, so the box
       is checked against the pixels it claims rather than against a caption. */
    if (!layNat) return '<span class="body"><em>the page image is still loading</em></span>';
    const x = b.bbox[0], y = b.bbox[1];
    const w = Math.max(1, b.bbox[2] - x), h = Math.max(1, b.bbox[3] - y);
    const s = Math.min(1, 420 / w, 300 / h);
    return '<div class="lay-crop" style="width:' + Math.round(w * s) + 'px;height:'
      + Math.round(h * s) + 'px;background-image:url(&quot;' + esc(layPageSrc)
      + '&quot;);background-size:' + Math.round(layNat.w * s) + 'px '
      + Math.round(layNat.h * s) + 'px;background-position:' + Math.round(-x * s)
      + 'px ' + Math.round(-y * s) + 'px"></div>'
      + (b.text ? '<span class="body">' + esc(b.text) + '</span>' : '');
  }
  return '<span class="body">' + (b.text ? esc(b.text)
    : '<em>no text — omitted by the prompt for this category</em>') + '</span>';
}

function layPopFill(i) {
  const b = layBlocks(layPage)[i];
  if (!b) return false;
  layPop.innerHTML = '<span class="cap">' + (i + 1) + '. ' + esc(b.category || 'block')
    + ' · [' + b.bbox.join(', ') + ']'
    + (Number.isFinite(Number(b.confidence)) ? ' · confidence '
       + (Number(b.confidence) * 100).toFixed(1) + '%' : '') + '</span>' + layPopBody(b)
    + (laySel === i ? '<span class="lay-pin">pinned — click the box again to release</span>' : '');
  layPop.classList.toggle('pinned', laySel === i);
  layPop.hidden = false;
  return true;
}

/* Clamped to the viewport after the content is in, because the popup's size is
   what decides whether it fits -- a table is not the height of a text run. */
function layPopPlace(x, y) {
  const pad = 10, r = layPop.getBoundingClientRect();
  let left = x + 16, top = y + 16;
  if (left + r.width > innerWidth - pad) left = Math.max(pad, x - 16 - r.width);
  if (top + r.height > innerHeight - pad) top = Math.max(pad, innerHeight - pad - r.height);
  layPop.style.left = Math.round(left) + 'px';
  layPop.style.top = Math.round(top) + 'px';
}
function layPopHide() {
  layPop.hidden = true;
  layPop.classList.remove('pinned');
}
/* Selecting a box repaints the marks only. A full paintLayout() re-creates the
   <img>, which re-requests the page and flickers the very thing being pointed
   at, so the selected state is toggled on the boxes already on screen. */
function layMark() {
  layStage.querySelectorAll('.lay-box').forEach(el =>
    el.classList.toggle('on', Number(el.dataset.i) === laySel));
}
/* A pinned popup is anchored to its box rather than to the cursor, so it stays
   put while the pointer goes to it. */
function layPopPin(i) {
  const box = layStage.querySelector('.lay-box[data-i="' + i + '"]');
  if (!box || !layPopFill(i)) return;
  const r = box.getBoundingClientRect();
  layPopPlace(r.right, r.top - 16);
}

layStage.addEventListener('mousemove', e => {
  if (laySel !== null) return;            // pinned: hover must not fight it
  const box = e.target.closest('.lay-box');
  if (!box) { layPopHide(); return; }
  if (!layPopFill(Number(box.dataset.i))) return;
  layPopPlace(e.clientX, e.clientY);
});
layStage.addEventListener('mouseleave', () => { if (laySel === null) layPopHide(); });
layStage.addEventListener('click', e => {
  const box = e.target.closest('.lay-box');
  if (!box) { laySel = null; layPopHide(); layMark(); return; }
  const i = Number(box.dataset.i);
  laySel = laySel === i ? null : i;
  layMark();
  if (laySel === null) layPopHide(); else layPopPin(laySel);
});
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || laySel === null || curMode !== 'layout') return;
  laySel = null; layPopHide(); layMark();
});
addEventListener('scroll', () => { if (laySel !== null) layPopPin(laySel); }, true);
addEventListener('resize', () => { if (laySel !== null) layPopPin(laySel); });

layPrev.addEventListener('click', () => { layPage--; laySel = null; layPopHide(); paintLayout(); });
layNext.addEventListener('click', () => { layPage++; laySel = null; layPopHide(); paintLayout(); });
tabLayout.addEventListener('click', () => {
  if (tabLayout.disabled) return;
  if (curMode === 'layout') { showTab(lastMode); return; }
  lastMode = curMode;
  showTab('layout');
});
tabGroup.addEventListener('click', () => {
  if (tabGroup.disabled) return;
  if (curMode === 'grouping') { showTab(lastMode === 'grouping' ? 'raw' : lastMode); return; }
  lastMode = curMode;
  showTab('grouping');
});

/* ---------- extracted fields (second LLM pass) ---------- */
/* The requirement's eleven fields and nothing else. `seller_name`, `buyer_name`,
   `reference_document`, `original_invoice_date` and `currency` were removed on
   2026-08-31 with the keys themselves -- neither requirement lists them, and a
   label for a key nothing asks for is a row that can only ever be empty. */
const FIELD_LABELS = {
  document_type: 'Document type', document_number: 'INV / CN no.',
  issue_date: 'INV date',
  po_gr_rtv_number: 'PO / GR / RTV no.',
  inv_rtv_cnr_number: 'INV / RTV / CNR no.',
  seller_tax_id: 'Supplier tax ID', seller_branch: 'Supplier branch code',
  seller_branch_location: 'Supplier branch location',
  buyer_tax_id: 'Buyer tax ID', buyer_branch: 'Buyer branch code',
  buyer_branch_location: 'Buyer branch location',
  subtotal: 'Amount ex VAT', vat_total: 'VAT amount',
  amount_incl_vat: 'Amount incl. VAT',
  remaining_amount: 'Remaining amount', remaining_vat_amount: 'Remaining VAT',
  payment_date: 'Payment date', cheque_number: 'Cheque no.',
  /* The withholding tax certificate (มาตรา 50 ทวิ), 2026-09-01. Its keys share
     nothing with the eleven above, so none of these ever renders beside them --
     `fields_asked` decides which rows a run draws. The wording is the
     requirement's own, English side, so a label on screen and a row of the
     requirement table can be matched by eye. */
  book_no: 'Book no.', certificate_no: 'Certificate no.',
  sequence_no: 'Sequence in P.N.D.',
  payer_tax_id: 'Payer tax ID', payer_name: 'Payer name',
  payer_branch: 'Payer branch code',
  payer_branch_location: 'Payer branch location', payer_address: 'Payer address',
  payee_tax_id: 'Payee tax ID', payee_name: 'Payee name',
  payee_branch: 'Payee branch code',
  payee_branch_location: 'Payee branch location', payee_address: 'Payee address',
  dividend_rate_option: 'Dividend rate option',
  total_amount_paid: 'Total amount paid', total_wht_amount: 'Total WHT amount'
};

/* One line saying what each key MEANS, shown on hover over its label.

   The labels have to fit a narrow column, so several are abbreviations or terms
   of art -- "ex VAT", "incl. VAT", "branch" -- which four characters cannot
   convey to a reader who does not already know them. These say what the field is
   and, where it matters, what it is NOT: the pair this page most often has to
   tell apart is ex-VAT against incl-VAT, two figures that look identical and
   differ by the tax.

   Written from the same definitions EXTRACT_PROMPT gives the model, so the page
   and the prompt cannot describe a key differently. */
const FIELD_HINTS = {
  document_type: 'The heading naming what this document is, copied as printed. Normalised to a standard code, and checked against the type this document was taken to be',
  document_number: 'This document\u2019s own number. Checked for a duplicate on an invoice, and against the document it credits on a credit note',
  issue_date: 'The date this document is dated \u2014 not a due or payment date. An invoice is also checked for a future date; a credit note is not',
  po_gr_rtv_number: 'The purchase order, goods receipt or return-to-vendor number this document is raised against. Mandatory on an invoice, optional on a credit note',
  inv_rtv_cnr_number: 'The invoice, return or credit-note-return this payment settles — never this receipt’s own number. Mandatory on a receipt',
  seller_tax_id: 'The issuer\u2019s tax ID. Thirteen digits, with a mod-11 check digit',
  seller_branch: 'Explicit tax branch number, normalised to five digits; head office is 00000. A digit in a place name is not a branch code',
  seller_branch_location: 'Explicit branch or site name, without the field caption. Never inferred from an ordinary address',
  buyer_tax_id: 'The customer\u2019s tax ID. Thirteen digits, with a mod-11 check digit',
  buyer_branch: 'Explicit tax branch number, normalised to five digits; head office is 00000. A digit in a place name is not a branch code',
  buyer_branch_location: 'Explicit branch or site name, without the field caption. Never inferred from an ordinary address',
  subtotal: 'The goods total printed BEFORE tax is added',
  vat_total: 'The VAT line. Checked against the standard rate of the figure it is charged on',
  amount_incl_vat: 'The total printed AFTER VAT is added. Checked as subtotal + VAT',
  remaining_amount: 'What is still owed after this payment, where the page prints a balance. Never worked out by subtraction',
  remaining_vat_amount: 'The VAT part of that balance. Checked against the standard rate of the balance',
  payment_date: 'The date the money was paid or the cheque was drawn — not the date the document was issued',
  cheque_number: 'The cheque the payment was made by. A bank account or transfer reference is not one',
  book_no: 'The book or volume this certificate was torn from (เล่มที่). Short — two to six digits',
  certificate_no: 'The certificate\u2019s own number (เลขที่). Short — not a tax ID',
  sequence_no: 'Which line of the P.N.D. filing this certificate is (ลำดับที่ในแบบ)',
  payer_tax_id: 'The tax ID of the party that paid the income and withheld the tax (ผู้จ่ายเงิน). Thirteen digits, with a mod-11 check digit',
  payer_name: 'The name of the party that withheld the tax. The requirement matches it against a Company Master at 90% — there is none here, so that check reports as not run',
  payer_branch: 'Explicit tax branch number, normalised to five digits; head office is 00000. A digit in a place name is not a branch code',
  payer_branch_location: 'Explicit branch or site name, without the field caption. Never inferred from an ordinary address',
  payer_address: 'The payer\u2019s printed address, as printed. No validation rule in the requirement',
  payee_tax_id: 'The tax ID of the party the income was paid to and the tax withheld from (ผู้ถูกหักภาษี). Thirteen digits, with a mod-11 check digit',
  payee_name: 'The name of the party the tax was withheld from. Matched against the Company Master by the same rule as the payer\u2019s',
  payee_branch: 'Explicit tax branch number, normalised to five digits; head office is 00000. A digit in a place name is not a branch code',
  payee_branch_location: 'Explicit branch or site name, without the field caption. Never inferred from an ordinary address',
  payee_address: 'The payee\u2019s printed address, as printed',
  dividend_rate_option: 'The ticked rate option against a dividend line (กรณี 40(4)(ข)). Required only where an income row states dividend income, and blank on most certificates',
  total_amount_paid: 'The total of the amounts paid (รวมเงินที่จ่าย). Reconciled against the sum of the table\u2019s own column',
  total_wht_amount: 'The total of the tax withheld and remitted (รวมภาษีที่หักนำส่ง). Reconciled against the sum of the table\u2019s own column'
};

/* Every key any document type's form can ask for, grouped the way the page is
   laid out. Which of them are RENDERED is decided per run from
   `extracted.fields_asked` -- the form the document's type actually asked for --
   so an invoice does not show a Credited-invoice-date row that nothing requested
   and nobody could fill. A key absent from this list is not rendered at all,
   which is how fifteen keys silently vanished from this tab in 2026-08-19: any
   key added to `prompts._SCALAR_KEYS` must be added here and to FIELD_LABELS and
   FIELD_HINTS as well. */
const FIELD_SECTIONS = [
  ['Document', ['document_type', 'document_number', 'issue_date']],
  /* Split from Document on 2026-08-31. Both of these name ANOTHER document --
     the order this one is raised against, or the one this payment settles -- so
     filing them under "Document" was wrong on every form and actively
     misleading on a receipt, whose form asks for nothing about itself and would
     have shown a lone reference under that heading. */
  ['References', ['po_gr_rtv_number', 'inv_rtv_cnr_number']],
  ['Supplier', ['seller_tax_id', 'seller_branch', 'seller_branch_location']],
  ['Buyer', ['buyer_tax_id', 'buyer_branch', 'buyer_branch_location']],
  ['Totals', ['subtotal', 'vat_total', 'amount_incl_vat']],
  /* The receipt's own two groups. Both drop out entirely on an invoice or a
     credit note, whose forms ask for none of these -- a section of rows that
     could only ever be empty is worse than no section. */
  ['Outstanding', ['remaining_amount', 'remaining_vat_amount']],
  ['Payment', ['payment_date', 'cheque_number']],
  /* The withholding certificate's four groups. Every one of them drops out on
     an invoice, a credit note or a receipt, whose forms ask for none of these.
     The two parties are named by what they DID rather than by the requirement's
     bare "Payer", because the requirement prints that word over both blocks and
     a page showing two identical headings would be unreadable. */
  ['Certificate', ['book_no', 'certificate_no', 'sequence_no']],
  ['Payer — withheld the tax',
   ['payer_tax_id', 'payer_name', 'payer_branch', 'payer_branch_location', 'payer_address']],
  ['Payee — tax withheld from them',
   ['payee_tax_id', 'payee_name', 'payee_branch', 'payee_branch_location', 'payee_address']],
  ['Withheld', ['total_amount_paid', 'total_wht_amount', 'dividend_rate_option']]
];
const MONEY_KEYS = new Set(['subtotal', 'vat_total', 'amount_incl_vat',
                            'remaining_amount', 'remaining_vat_amount',
                            'total_amount_paid', 'total_wht_amount']);

/* Every extracted key is priority 1 now, so the per-key tier tag would say P1
   fourteen times and carry no information. Left empty rather than removed: the
   tag renders from this map, so restoring tiers is a matter of filling it in. */
const FIELD_TIER = {};
const ITEM_COLS = [
  ['description', 'Description'], ['period', 'Period'], ['quantity', 'Qty'],
  ['unit_price', 'Unit price'], ['amount', 'Amount'], ['vat', 'VAT'],
  ['withholding_tax', 'W/T'], ['net_amount', 'Net amount']
];

/* What each of those columns holds, on hover over the heading. Qty and W/T are
   the reason this exists -- a two-letter heading over a column of money is not
   readable on its own, and W/T in particular is the one column whose figure is
   SUBTRACTED where every other money column is added. */
const ITEM_HINTS = {
  description: 'What is being charged for, as the row prints it',
  period: 'The period this row covers, where the table rules a column for it',
  quantity: 'How many — a count, not an amount of money',
  unit_price: 'Price per unit',
  amount: 'The row’s charge',
  vat: 'VAT on this row, where the table rules a VAT column',
  withholding_tax: 'Withholding tax (ภาษีหัก ณ ที่จ่าย) — deducted at source, so it comes OFF the row rather than being added to it',
  net_amount: 'What the row comes to after those, where the table rules such a column'
};

/* The withholding certificate's income table (2026-09-01), kept apart from
   ITEM_COLS above for the same reason `income_items` is kept apart from
   `line_items`: that is the charges table of a commercial document and this is
   the table of income a certificate covers. One list of columns serving both
   would draw an invoice's headings over a certificate's rows.

   The rate is LAST and is not one of the four extracted cells: it is worked out
   in Python from the two beside it (`normalise.wht_rates`), which is what the
   requirement means by calling the field "Derived". It is drawn from
   `extracted.derived`, never from the model's answer. */
const INCOME_COLS = [
  ['income_type', 'Income type'], ['payment_date', 'Payment date'],
  ['amount_paid', 'Amount paid'], ['wht_amount', 'WHT amount']
];
const INCOME_HINTS = {
  income_type: 'The kind of income as the row prints it (รายละเอียดประเภทเงินได้). Where the row is the free-text option, this is what was written against it',
  payment_date: 'The date on that row (วัน เดือน ปี ที่จ่าย), as printed. A Buddhist-era year is converted when the row is validated, never when it is read',
  amount_paid: 'The amount paid on that row (จำนวนเงินที่จ่าย). The column is added up and reconciled against the total below',
  wht_amount: 'The tax withheld and remitted on that row (ภาษีที่หักและนำส่งไว้). Checked against the amount paid — it may not exceed it — and added up against the total below'
};

/* A value is only worth reading if it came off the page. `status` is the
   server's grounding verdict for that field; without one the list renders
   exactly as it used to. */
const GROUND_TAG = {
  // "missing" says what the extractor did, not what the page contains: it came
  // back empty. Whether the page was silent or the extractor missed it is not
  // something this check can tell, so it does not claim to.
  missing: ['missing', 'left empty by the extractor — nothing was invented for it. Check the page: it is either not stated there, or it was missed'],
  ungrounded: ['not on page', 'this text does not appear in the transcript — treat it as invented'],
  computed: ['computed', 'not printed on the page; it is the sum of the other extracted figures']
};

/* The verdict a hand-written truth file passes on one value. Distinct from
   GROUND_TAG above and worth keeping distinct: grounding says a value is printed
   somewhere on the page, this says it is the value that belongs in THIS key --
   sol005's buyer_name held the page's Location Code, which grounds and is wrong,
   and only this can say so. */
const SCORE_TAG = {
  correct: ['correct', 'matches the value the ground truth says belongs in this field'],
  partial: ['partly', 'the right value, over- or under-captured — one contains the other'],
  wrong: ['wrong', 'the ground truth says a different value belongs in this field'],
  missed: ['not returned', 'the document prints a value here and nothing came back for it'],
  spurious: ['extra', 'the ground truth says this document prints nothing for this field'],
  absent: ['not stated', 'the document does not print this, and nothing was filled in — agreed']
};
const SCORE_OK = new Set(['correct', 'absent']);
/* WHAT THE RATE IS TAKEN OVER, in printing order, with the word the note under
   the fields uses for each. `fieldscore._tally` makes `expected` correct +
   partial + wrong + missed, so `spurious` and `absent` are outside the
   denominator entirely -- a value invented where the page prints nothing has no
   printed value to be judged against, and a key both sides leave blank is not a
   value at all.

   ONE statement of that list, read by the note that itemises the denominator and
   by the tooltip that tells a single value which side of it it fell on. Two
   copies would eventually disagree, and the disagreement would be a tab whose
   marks do not add up to the total printed underneath them -- which is the bug
   this whole block exists to stop. */
const SCORE_COUNTED = [['correct', 'correct'], ['partial', 'partly'],
                       ['wrong', 'wrong'], ['missed', 'not returned']];
const scoreCounts = status => SCORE_COUNTED.some(([s]) => s === status);
/* Why a verdict sits where it does, where the bare in-or-out needs a reason.
   The in-or-out itself is NOT written here: it is derived from the list above,
   so a verdict cannot be described as counted by one line and uncounted by the
   next. */
const SCORE_WHY = {
  partial: ' The strict rate counts it as a miss; the loose rate beside it is where it counts for something.',
  spurious: ' The denominator is what the PAGE prints, so a value invented where the page prints nothing cannot lower this rate — it lowers precision instead.',
  absent: ' There is no value here for either side to be judged on.'
};
function rateClause(status) {
  return (scoreCounts(status) ? ' In the field score.' : ' NOT in the field score.')
    + (SCORE_WHY[status] || '');
}
/* One formatter for every rate on this tab and the Accuracy tab's field bar, so
   the two cannot round one number two ways. */
const scPct = v => (v === null || v === undefined) ? '—' : (v * 100).toFixed(1) + '%';
/* One value's verdict, and whether it moved the headline.

   **Whether it counted is the SERVER's answer, read off the row.**
   `fieldscore.score` puts `required` on every judged row for exactly this
   reason -- its own comment says so: "the page marks it, the CLI prints it and
   the headline is taken over it, and three answers to one question drift". This
   page was the reader that gave its own, re-deriving the answer from the
   Mandatory set, and the two disagree in the one case that matters: a document
   NO requirement covers is scored over every key it asked for (`unknown_type`,
   2026-09-04) while its Mandatory set is empty -- so every value on screen said
   "not scored" underneath a headline that had just scored all of them.

   An absent `required` means counted, which is what the server means by
   `mandatory=None`: a caller that did not classify scores everything asked.

   Muting is for a value OUTSIDE the requirement, and the word is `optional`
   rather than `not scored` because the value was scored -- judged, and reported
   in the optional tally under the fields. A verdict the denominator excludes
   (`extra`, `not stated`) is said in the tooltip instead of muted: a value
   invented on a Mandatory key is a finding worth drawing loudly whatever it
   does to the arithmetic. */
function scoreMark(row) {
  const tag = row && SCORE_TAG[row.status];
  if (!tag) return '';
  const optional = row.required === false;
  return '<span class="sc ' + row.status + (optional ? ' unscored' : '') + '" title="'
    + esc(tag[1] + '.' + (optional
        ? ' Optional on this document type: the requirement does not demand this '
          + 'field, so it is judged and reported beside the field score rather '
          + 'than inside it.'
        : rateClause(row.status)))
    + '">' + esc(tag[0]) + (optional ? ' · optional' : '') + '</span>';
}
/* A table cell's half of what `scoreMark` says on a scalar: whether the cell is
   in the rate. There is no muted class for it because no shipped form makes a
   table cell Optional -- the one type that rules a table demands every cell of
   it -- so this states the truth in the tooltip and adds no CSS for a state
   nothing can currently produce. */
function cellScope(row) {
  return !row ? ''
    : row.required === false
      ? '. Optional on this document type, so it is outside the field score.'
      : '.' + rateClause(row.status);
}
/* WHICH PASS lost this value -- `blame.py`, computed in Python from the
   transcript the extraction actually read, and worth reading beside the score
   mark rather than instead of it: that one says what went wrong, this one says
   who it is down to.

   The labels are the pipeline strip's own stage names, so a chip reading
   `read` points at the box above that says how the page was read. The
   attributions themselves are the server's -- nothing here re-derives one --
   for the reason `scoreMark` has a paragraph about: a second answer to one
   question drifts from the first, and the drift is invisible. */
const BLAME_TAG = {
  ocr: ['read', 'this stage did not produce the text the field needed'],
  extraction: ['extract', 'the text was in the transcript and did not reach the field'],
  unknown: ['unclear', 'neither stage is named for it']
};
/* The attributions for the document on screen, keyed the way the values are.
   `extracted` is the SELECTED document of a split file, so these are that
   document's own paths -- the file-level ones carry a `docN.` prefix and are not
   what anything here is keyed by. */
function blameFields() {
  return ((extracted && extracted.blame) || {}).fields || {};
}
/* One chip. Silent where the value came back right, where the run was not
   attributed at all, and where there is no field truth -- the same three
   silences `scoreMark` keeps, and for the same reason: a mark on every row says
   nothing. */
/* The class that colours a flagged row and its chip. One custom property is
   set on the row (`--flag`), so the stripe and the chip inside it read the same
   value and cannot name two different culprits. Empty where nothing was
   attributed, so a clean row gains no class at all. */
function blameClass(entry) {
  return entry && BLAME_TAG[entry.blame] ? ' blamed blame-' + entry.blame : '';
}
function blameMark(entry) {
  const tag = entry && BLAME_TAG[entry.blame];
  if (!tag) return '';
  /* **The reason is the SERVER's, per value, not a sentence kept here per
     attribution.** A fixed line about what `read` means is true of the ordinary
     case and false of the one that matters most -- a value invented under a key
     the page leaves blank is `read` because the transcript holds text the page
     does not, which is the opposite of "the read did not produce this text".
     `blame.py` already worked out which of those this value is; a second
     wording here could only ever be the one that contradicts it. */
  return '<span class="bl ' + entry.blame + '" title="'
    + esc('Down to the ' + tag[0] + ' — ' + (entry.why || tag[1]) + '.'
          + (entry.counted ? '' : ' This value is outside the field score either'
             + ' way, so the attribution is reported and not counted.'))
    + '">⚑ ' + esc(tag[0]) + '</span>';
}

/* The readings of the truth file this row was NOT judged against, quoted.
   `expected` is the one the answer came closest to, so these are the rest. */
function otherReadings(row) {
  return ((row && row.accepted) || [])
    .filter(v => String(v) !== String(row.expected))
    .map(v => '“' + esc(String(v)) + '”');
}
/* Printed where the two differ -- repeating the truth under every correct value
   would double the length of the list to say nothing -- and, on a key the truth
   file gives more than one accepted reading, printed under a CORRECT value too.

   That second case is the one worth arguing for. A key with two readings was
   marked right against one of them, and the mark alone does not say that: it
   looks identical to a key with a single truth, so the reader cannot tell
   whether the other language would have been accepted or whether this run
   simply happened to answer in the one the file wanted. Saying which readings
   the key accepts is the only thing on screen that distinguishes a value that
   was right from a value that could only ever have been right one way -- and
   the readings are what a reviewer edits when they disagree with the score.

   Wrong and partial keep "Ground truth", because there the readings are what
   the answer failed to match; correct gets "Also accepted", because there they
   are not a finding and must not read as one. */
function truthLine(row) {
  if (!row) return '';
  const alt = otherReadings(row);
  if (SCORE_OK.has(row.status))
    return alt.length ? '<div class="truth">Also accepted: <i>'
                        + alt.join(', ') + '</i></div>' : '';
  if (row.status === 'spurious')
    return '<div class="truth">Ground truth: <b>the page prints nothing here</b></div>';
  return '<div class="truth">Ground truth: <b>' + esc(String(row.normalized_expected ?? row.expected)) + '</b>'
    + (alt.length ? '<i> — or ' + alt.join(', or ') + '</i>' : '')
    + '</div>';
}

/* The requirement's validation rules, keyed by the field they are about.

   Only `failed` and `warning` reach the page as a mark. `ok` deliberately draws
   nothing: a tick beside ten of thirteen values is noise on a tab whose job is
   to show what is WRONG, and the counts are in the status line for anyone who
   wants to know the checks ran at all. `unchecked` draws nothing for the same
   reason plus a stronger one -- it means the rule could not run, which is not a
   statement about this value. */
function valIndex(validation) {
  const out = {};
  if (!validation || validation.error) return out;
  for (const row of (validation.checks || [])) {
    if (row.state !== 'failed' && row.state !== 'warning') continue;
    (out[row.field] = out[row.field] || []).push(row);
  }
  return out;
}

/* A failed rule outranks a warning: the pair this most often draws is a tax ID
   that fails its checksum, which is a harder claim than a VAT rate that blends
   below 7%. */
function valMark(rows) {
  if (!rows || !rows.length) return '';
  const bad = rows.filter(r => r.state === 'failed');
  const shown = bad.length ? bad : rows;
  const state = bad.length ? 'failed' : 'warning';
  const title = shown.map(r => r.rule + ': ' + (r.why || r.detail || '')).join(' | ');
  return '<div class="val-note ' + state + '" title="' + esc(title) + '">'
    + esc(shown.map(r => r.rule).join(', ')) + ' \u2014 '
    + esc(shown[0].why || shown[0].detail || '') + '</div>';
}

function defList(pairs, truth, val, req, bl, raw) {
  if (!pairs.length) return '';
  return '<dl class="fld-grid">' + pairs.map(([label, value, money, status, key]) => {
    // Which pass lost this value, resolved once: the label's stripe, the value's
    // stripe and the chip are three marks of ONE fact and are built from it.
    const blm = (bl && key) ? bl[key] : null;
    const flag = blameClass(blm);
    const blank = (value === null || value === undefined || value === '');
    const tag = GROUND_TAG[status];
    const tier = FIELD_TIER[key];
    const sc = (truth && key) ? truth[key] : null;
    // Mandatory in the requirement for THIS document's type, which is why the
    // set is handed in rather than looked up: the same key is Mandatory on an
    // invoice and optional on a credit note, and a mark that did not follow the
    // type would be a compliance claim about the wrong document.
    const must = !!(req && key && req.has(key));
    const gap = must && (blank || status === 'missing');
    const cls = [blank ? 'empty' : '', money ? 'money' : '',
                 gap ? 'req-missing' : '',
                 status === 'ungrounded' || status === 'computed' ? status : ''].join(' ');
    let body;
    if (blank || status === 'missing') {
      // A Mandatory field that came back empty is a finding, not an absence:
      // "the page may be silent about this" is true of an Optional field and is
      // exactly what the requirement says cannot be true here. The grounding tag
      // still explains WHY it is empty, in the tooltip, because that half is
      // unchanged -- this only says the emptiness is not acceptable.
      body = gap
        ? '<span class="fld-tag req-missing" title="'
          + esc('Mandatory for this document type and it did not come back. '
                + (tag ? tag[1] : '')) + '">required, not returned</span>'
        : tag ? '<span class="fld-tag missing" title="' + esc(tag[1]) + '">'
                   + esc(tag[0]) + '</span>' : '—';
    } else {
      body = esc(String(value))
        + (tag ? '<span class="fld-tag ' + status + '" title="' + esc(tag[1]) + '">'
                 + esc(tag[0]) + '</span>' : '');
    }
    // The label is abbreviated to fit the column, so what the key actually means
    // hangs off it rather than beside it -- forty-four one-line definitions
    // printed on the page would bury the values they describe.
    const hint = key ? FIELD_HINTS[key] : '';
    return '<dt class="' + (hint ? 'has-hint' : '') + flag + '"'
      + (hint ? ' title="' + esc(hint) + '"' : '')
      + '>' + esc(label)
      + (must ? '<span class="req" title="Mandatory in the requirement for this '
                + 'document type — the page has to state it and the extraction '
                + 'has to return it.">required</span>' : '')
      + (tier ? '<span class="tier ' + tier + '" title="Priority '
                + tier.slice(1) + ' field — '
                + (tier === 'p1' ? 'prototype / MVP'
                   : tier === 'p2' ? 'production readiness'
                   : 'advanced extraction') + '">' + tier.toUpperCase()
                + '</span>' : '')
      + '</dt><dd class="' + (cls.trim() + flag).trim() + '">' + body
      + scoreMark(sc) + blameMark(blm) + truthLine(sc)
      + (raw && typeof raw[key] === 'string' && raw[key] !== String(value ?? '')
          ? '<details class="fld-raw"><summary>Raw model value</summary><div>' + esc(raw[key]) + '</div></details>' : '')
      + valMark(val && key ? val[key] : null)
      + '</dd>';
  }).join('') + '</dl>';
}

/* The truth file for this document, re-keyed the way the fields on screen are
   keyed, so every value can be marked where it is read.

   The line items are the part that needs doing at all: `line_items[i].key`
   carries the index of the TRUTH row, and rows are paired with the extractor's
   by content rather than by position (one dropped row would otherwise shift
   every row after it), so `row` -- which returned row the truth row was paired
   with -- is the only thing that lets a cell on screen find the value it is
   judged against. A truth row paired with nothing is a row the page prints and
   the extraction never returned, which is a finding in its own right and is
   drawn under the table rather than dropped. */
function truthIndex(fs) {
  const out = { scalars: {}, items: {}, missedRows: [], keys: new Set(),
                income: {}, missedIncome: [], incomeKeys: new Set() };
  if (!fs || fs.error) return out;
  for (const r of ((fs.scalars || {}).rows) || []) out.scalars[r.path] = r;
  /* Both tables index the same way and differ only in which buckets they fill,
     so one reader serves both -- `line_items` is the charges table of a
     commercial document and `income_items` the income table of a WHT
     certificate, and they are never the same document's. */
  const table = (block, prefix, byRow, missed, keys) => {
    for (const r of ((block || {}).rows) || []) {
      // A regex LITERAL, not one built from a string: `\d` inside a JS string
      // literal is just `d`, so `new RegExp('...\d...')` silently matches the
      // wrong thing. The field name is captured and compared instead.
      const m = /^([a-z_]+)\[(\d+)\]\.(.+)$/.exec(r.path || '');
      if (!m || m[1] !== prefix) continue;
      if (r.expected !== '' && r.expected !== null && r.expected !== undefined) keys.add(m[3]);
      const bucket = (r.row === null || r.row === undefined)
        ? (missed[+m[2]] = missed[+m[2]] || {})
        : (byRow[r.row] = byRow[r.row] || {});
      bucket[m[3]] = r;
    }
  };
  table(fs.line_items, 'line_items', out.items, out.missedRows, out.keys);
  table(fs.income_items, 'income_items', out.income, out.missedIncome, out.incomeKeys);
  return out;
}

/* Coverage by delivery tier. Sits above the grounding note deliberately: this
   one says how much came back, that one says how much of it is real, and reading
/* Which figures below carry tax. Two documents can print identical numbers and
   mean different things by them, and nothing else on this tab would say so —
   the amounts are copied verbatim from the page either way, which is right, and
/* The field score, where this document has hand-written field ground truth
   (solution/<id>.fields.json). A stronger statement than the grounding note
   that follows it: grounding says a value is printed somewhere on the page,
   this says it is the value that belongs in that key. sol005's buyer_name came
   back as the page's Location Code -- grounded, and wrong.

   It sits BELOW the fields, not above them: every value it scores is already
   marked where it is read, so this is the summary of marks the reader has just
   been through rather than a verdict handed down before they see any of them.
   Absent on every document nobody has written a truth file for. */
/* Why this run's fields carry no rate, or '' where they do.

   Shared by the two tabs that would otherwise each draw one, so they cannot
   describe the same run differently. The score is still on the result and every
   value above is still marked with it -- what the read floor drops is the
   headline, which over a broken transcript is the READ's number wearing the
   extractor's name. Silent where there was never a rate to drop: a document with
   no field ground truth has nothing to say here. */
/* Which tally is THE score for this document, and what it is over.

   The headline is the requirement's Mandatory set. **Some documents have no
   requirement at all** -- a page that is only a tax invoice, or one nothing
   could classify -- and for those `overall.expected` is 0 while every value the
   truth file states sits in `optional`, judged and correct. Until 2026-09-04
   that rendered as NOTHING: no bar, no list, no sentence, which is
   indistinguishable from an extraction that found nothing and is what sent the
   user looking at the read floor for a document the floor never touched.

   Since 2026-09-04 the server does that half: a document no requirement covers
   is scored over the BASE field set and comes back with `unknown_type` set, so
   `overall` is populated and the fallback below is left for the other case that
   empties it -- a document whose truth file happens to state only keys the
   requirement marks Optional.

   `required` decides whether the values are called required or merely stated;
   `unknown` decides whether the page says no requirement covers the document.
   They are not the same question, which is why the fallback sets one and not
   the other. */
function scoreTally(fs) {
  const o = (fs && fs.overall) || {}, op = (fs && fs.optional) || {};
  const unknown = !!(fs && fs.unknown_type);
  if (o.expected) return { t: o, scope: fs.scored_scope || '',
    required: !unknown, unknown: unknown };
  if (op.expected) return { t: op, required: false, unknown: unknown,
    scope: 'the keys the ground truth states that the requirement marks Optional '
      + '— it states none of the Mandatory ones' };
  return null;
}

/* Who lost the values the score charged for, in one sentence.

   It says what the marks above cannot: the marks are per value and a reader
   wants the shape of the run. Placed with the score and never instead of it --
   `fieldscore` says how much went wrong, this says whose it was, and neither
   answers the other's question.

   **It renders under a run the read floor suppressed as well**, which is the
   case it is worth the most in: there the rate is deliberately blank, and this
   is the only thing on the tab that says why in terms of values rather than
   percentages. */
function blameNote() {
  const b = (extracted && extracted.blame) || {};
  const c = b.counts;
  if (!c || (!b.counted && !b.uncounted)) return '';
  const say = [[c.ocr, 'the read', 'ocr'], [c.extraction, 'the extraction', 'extraction'],
               [c.unknown, 'unclear', 'unknown']]
    .filter(([n]) => n)
    .map(([n, who, kind]) => '<b class="blame-' + kind + ' bl-count">'
                             + n + ' ' + who + '</b>');
  return '<p class="ground-note">'
    + '<strong>Who lost it</strong> — '
    + (b.counted
        ? 'of the ' + b.counted + ' value(s) in the rate that did not come back '
          + 'right, ' + say.join(', ') + '.'
        : 'nothing in the rate went wrong.')
    + ' <span class="how">Each value the ground truth states was looked for in '
    + 'the transcript this run actually read — so a value that is not in it was '
    + 'lost by the read whatever the extraction did, and one that is in it '
    + 'reached the model and did not reach the field. '
    + 'This attributes what the score already charged for and changes none of it.'
    + (b.uncounted ? ' ' + b.uncounted + ' more value(s) are attributed above and '
        + 'counted in neither: an Optional field, or one filled where the page '
        + 'states nothing — the same two the rate itself leaves out.' : '')
    /* Said only when it is missing, because its absence is what weakens the
       two attributions that need it -- see `blame.py`. Both then fall back to
       claiming less, so the effect is one-directional and worth naming. */
    + (b.page_checked ? '' : ' There was no ground-truth transcript to check an '
        + 'invented value against, so an attribution that would have named the '
        + 'read names the extraction instead.')
    + '</span></p>';
}

function unscoredReason() {
  if (!extracted || !extracted.fields_unscored) return '';
  const o = ((extracted.field_score || {}).overall) || {};
  return o.expected ? extracted.fields_unscored : '';
}

function fieldScoreNote(fs) {
  if (!fs || fs.error) return '';
  const pick = scoreTally(fs);
  if (!pick) return '';
  const o = pick.t;
  const c = o.counts || {};
  const pct = scPct;
  const cov = fs.coverage || {};
  let html = '<p class="ground-note' + (c.correct === o.expected ? '' : ' bad') + '">'
    + '<strong>Field score ' + pct(o.accuracy) + '</strong> — ' + c.correct + ' of '
    + o.expected + ' value(s) the ground truth says this page prints came back '
    + 'correct'
    + (c.partial ? ', ' + c.partial + ' partly (' + pct(o.accuracy_loose) + ' counting those)' : '')
    /* The denominator, itemised, so it adds up on screen. `expected` is these
       four and NOTHING else -- printing the invented count in the same breath is
       what made the arithmetic look wrong, because a reader adding correct,
       wrong and invented together never reached the total beside them. */
    + '. The ' + o.expected + ' it is taken over: '
    + SCORE_COUNTED.map(([k, word]) => (c[k] || 0) + ' ' + word).join(', ') + '.'
    /* And what sits outside it, said as outside it. Both are real findings and
       neither can move this rate: an invention lowers precision, which is
       printed here because a rate quoted alone is the one place that difference
       cannot show. */
    + ((c.spurious || c.absent)
        ? ' <span class="how">Outside that count: '
          + (c.spurious ? c.spurious + ' filled where the page states nothing, '
              + 'which lowers precision (' + pct(o.precision) + ') and cannot '
              + 'lower this rate' : '')
          + (c.spurious && c.absent ? '; ' : '')
          + (c.absent ? c.absent + ' the page and the extraction agree is blank, '
              + 'which is no value for either side to be judged on' : '')
          + '.</span>'
        : '');
  /* What the rate is over. Since 2026-09-02 it is the requirement's Mandatory
     set, so a document can score 100% with an Optional field wrong -- which is
     correct, and unreadable unless the scope is printed beside the number. */
  if (pick.scope) {
    /* Said outright: a rate with no Mandatory set behind it is not a compliance
       figure and must not be read as one. The unknown-type case states the
       scope and the caveat in ONE sentence rather than printing the scope and
       then explaining it -- the scope string already names the reason, and two
       sentences saying it is the second-copy-of-the-page shape this tab had a
       whole block deleted for. */
    html += ' <span class="how">'
      + (pick.unknown
        ? 'No requirement covers this document type yet, so the rate is over the '
          + 'base field set — every key the ground truth states. Nothing on it '
          + 'is Mandatory: this is what the extractor found, not what it owed.'
        : 'Scored over ' + esc(pick.scope) + '.'
          + (pick.required ? '' : ' None of these is Mandatory here, so this is '
            + 'what the extractor found, not what it owed.'))
      + '</span>';
  }
  /* The Optional fields, judged the same way and deliberately outside the rate:
     nobody is held to them, so a document that leaves one out is compliant and
     must not read as incomplete. Reported rather than dropped, for the reason
     `other_fields` is -- a model good at the eleven that matter and poor at the
     rest is a different proposition from one that is evenly mediocre. */
  const op = fs.optional || {};
  /* `scored_paths` is how many values carry the muted `optional` mark above, and
     it is the number a reader is actually looking for: `op.expected` is smaller,
     because it drops the invented and the agreed-blank exactly as the headline
     does. Saying only the second left the marks on screen unaccounted for. */
  const opRows = op.scored_paths || 0;
  if (pick.required && opRows) {
    const oc = op.counts || {};
    html += ' <span class="how">' + opRows + ' value(s) below are marked '
      + '<b>optional</b> — judged, and outside the rate above because the '
      + 'requirement does not demand them of this document type'
      + (op.expected ? ': ' + oc.correct + ' of ' + op.expected + ' correct ('
                       + pct(op.accuracy) + ')' : '')
      + '.</span>';
  }
  if (cov.scalars_checked < cov.scalars_total) {
    html += ' <span class="how">Scored over the ' + cov.scalars_checked + ' of '
      + cov.scalars_total + ' key(s) the truth file fills in'
      + (cov.line_items_checked ? ', plus the table' : '')
      + '; the rest are unscored.</span>';
  }
  html += '</p>';
  const d = (fs.line_items || {}).derived;
  if (d) {
    // The rows were read out of the .md's own table, and which printed column
    // was taken for which key is a guess made from the heading. A column it
    // could not name makes a correct extraction look wrong, so it is said here
    // rather than left to the terminal.
    html += '<p class="ground-note"><b>Line items</b> are scored against the table in '
      + '<code>' + esc(d.source || '') + '</code> — '
      + Object.entries(d.columns || {}).map(([h, k]) => esc(k) + ' ← ' + esc(h)).join(', ')
      + (d.dropped && d.dropped.length
          ? '. ' + d.dropped.length + ' row(s) dropped as totals or notes' : '')
      + ((d.unmapped || []).length
          ? '. Not scored, no key matches the heading: ' + d.unmapped.map(esc).join(', ')
          : '')
      + '.</p>';
  }
  return html;
}

/* The same score read as a list, and the counterpart to the marks beside the
   fields: those answer "is this one right" with the document open, this answers
   "what is left" once they have been read. Kept below the fields for that
/* ---------- raw pass output ---------- */
// Both later passes ask the model for JSON and then parse it, so when a value on
// screen looks wrong the question is which of those two steps lost it. Showing
// the verbatim reply beside the parsed response answers that without a server
// log: if the value is in the reply but not the response, the parse dropped it.
function paintRaw(target, result, reply) {
  const prompt = result && result.prompt;
  target.innerHTML =
    // Collapsed: it is the whole transcript with instructions round it, and it
    // is the same on every run until a prompt in `prompts.py` changes. Worth
    // being able to read, not worth reading past.
    (prompt ? '<details class="raw-prompt"><summary>Prompt sent — '
              + prompt.length.toLocaleString() + ' characters</summary><pre>'
              + esc(prompt) + '</pre></details>' : '')
    + (reply ? '<div class="fld-sec">Model reply (verbatim)</div><pre>'
             + esc(reply) + '</pre>' : '')
    + '<div class="fld-sec">Parsed response</div><pre>'
    + esc(JSON.stringify(result, null, 2)) + '</pre>';
}

/* ---------- extraction mode: one shot, or a few fields at a time ----------
   The switch lives on the server, not in this page: it applies to the random test and
   to a run started from another tab as well as to Re-extract here, and every
   result says which mode produced it, so the button and the result cannot
   disagree about what just happened. */
let extractMode = window.BENCHMARK_BOOTSTRAP.extract_mode;
const EXTRACT_STEP_COUNT = window.BENCHMARK_BOOTSTRAP.extract_steps;
const HYBRID_STEP_COUNT = window.BENCHMARK_BOOTSTRAP.hybrid_steps;
function isStepMode(mode) { return mode === 'agentic' || mode === 'hybrid'; }
let extractSteps = null;
/* Stage 0's answer, from the `classified` event. Kept outside `extracted`
   because it arrives BEFORE the extraction does and is what the strip draws
   while pass 2 is still running; the finished result carries its own copy for
   anything rendered after the fact. */
let classified = null;

/* Two controls for one switch: the picker in the sidebar, where a run is set up
   before it starts, and the button beside Re-extract, where you are standing when
   a document comes back with a value in the wrong field. Both paint from the same
   variable, so they cannot show different things. */
function paintMode() {
  const on = isStepMode(extractMode);
  const hybrid = extractMode === 'hybrid';
  fldMode.value = extractMode;
  fldMode.classList.toggle('on', on);
  fldMode.title = hybrid ? 'Hybrid: tuned prompts with related fields grouped together.'
    : on ? 'Agentic: original prompts and separate field groups.'
    : 'Single: the whole form in one request.';
  if (extractSel.value !== extractMode) extractSel.value = extractMode;
  if (fxMode && fxMode.value !== extractMode) fxMode.value = extractMode;
  if (fxSchemaFirst) fxSchemaFirst.disabled = !on;
  extractNote.textContent = hybrid
    ? 'Tuned prompts with related totals and payment fields grouped together. '
      + HYBRID_STEP_COUNT + ' steps depending on document type.'
    : on ? 'Original Agentic prompts and separate field groups. '
      + EXTRACT_STEP_COUNT + ' steps depending on document type.'
    : 'The whole form in one request.';
}

/* The server owns the setting -- it applies to the random test and to a run started
   from another tab as well as to Re-extract here -- so the controls only ever
   show what came back from it. */
async function setMode(next) {
  fldMode.disabled = extractSel.disabled = true;
  if (fxMode) fxMode.disabled = true;
  try {
    const res = await fetch(API + '/extract/mode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: next })
    });
    const body = await res.json();
    if (body.mode) extractMode = body.mode;
  } catch (err) { /* leave it where it was */ }
  fldMode.disabled = extractSel.disabled = false;
  if (fxMode) fxMode.disabled = false;
  paintMode();
}

fldMode.addEventListener('change', () => setMode(fldMode.value));
extractSel.addEventListener('change', () => setMode(extractSel.value));
paintMode();

/* Whether pass 2 asks for `other_fields`. Server-owned like the mode: it applies
   to Re-extract, the random test and the stress test alike. */
const extrasChk = document.getElementById('extrasChk');
const extrasNote = document.getElementById('extrasNote');

function paintExtras() {
  extrasNote.innerHTML = extrasChk.checked
    ? 'Every labelled fact the form has no key for comes back under its own label. '
      + 'Nothing scores it, and it is the longest part of most replies.'
    : '<b>Off.</b> Only the form is asked for: the single prompt and its grammar '
      + 'leave <code>other_fields</code> out, and agentic skips the <i>other</i> step.';
}

async function setExtras(on) {
  extrasChk.disabled = true;
  try {
    const res = await fetch(API + '/extract/extras', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ extras: on })
    });
    const body = await res.json();
    if (typeof body.extras === 'boolean') extrasChk.checked = body.extras;
  } catch (err) { /* leave it where it was */ }
  extrasChk.disabled = false;
  paintExtras();
}

extrasChk.addEventListener('change', () => setExtras(extrasChk.checked));
paintExtras();

/* ---------- pass 1: which prompt, and whether a system message goes with it ----
   The two move together on purpose. A prompt written for one model sent to
   another has been measured returning nothing at all at HTTP 200, and so has the
   right prompt sent with the wrong system slot -- so this is one control, not
   two, and the server owns it for the same reason it owns the extraction mode. */
const ocrProfileSel = document.getElementById('ocrProfileSel');
const ocrProfileNote = document.getElementById('ocrProfileNote');
const OCR_PROFILES = window.BENCHMARK_BOOTSTRAP.ocr_profiles;

function paintOcrProfile() {
  const spec = OCR_PROFILES.find(p => p.id === ocrProfileSel.value);
  ocrProfileNote.textContent = spec ? spec.note : '';
}

async function setOcrProfile(next) {
  ocrProfileSel.disabled = true;
  try {
    const res = await fetch(API + '/ocr/profile', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile: next })
    });
    const body = await res.json();
    /* Show what the server accepted, never what was clicked. */
    if (body.profile) ocrProfileSel.value = body.profile;
  } catch (err) { /* leave it where it was */ }
  ocrProfileSel.disabled = false;
  paintOcrProfile();
}

ocrProfileSel.addEventListener('change', () => setOcrProfile(ocrProfileSel.value));
paintOcrProfile();

/* ---------- the loop backstop ------------------------------------------------
   Unticked turns off the ABORT and nothing else. A read that cycles the whole way
   is still detected and still comes back flagged `looped`, so it still reads as a
   failed run and still stays out of every mean -- what changes is that the
   transcript is all of what the model produced rather than the part before the
   backstop fired, and that a true runaway costs the whole token budget. */
const loopGuardChk = document.getElementById('loopGuardChk');
const loopGuardNote = document.getElementById('loopGuardNote');
/* The page-read token cap, as the SERVER reports it. Rendered into the page and
   refreshed from every loop-guard reply, so an open page follows a server that
   was restarted with a different MAX_NEW_TOKENS instead of showing the old one. */
let READ_TOKEN_CAP = window.BENCHMARK_BOOTSTRAP.max_new_tokens;

function paintLoopGuard() {
  loopGuardNote.innerHTML = loopGuardChk.checked
    ? 'A read whose tail is cycling is stopped where it starts repeating and '
      + 'flagged <b>looped</b>. Leave this on unless it is cutting short a read '
      + 'you want to see in full &mdash; lowering Detail is the usual fix for a '
      + 'page that loops.'
    : '<b>Off.</b> A cycling read now runs to the ' + READ_TOKEN_CAP + '-token '
      + 'cap, which on a real runaway is minutes of wall clock. It is still '
      + 'detected: the transcript comes back whole and still flagged '
      + '<b>looped</b>, so it still counts as a failed run and is kept out of '
      + 'every mean.';
}

async function setLoopGuard(on) {
  loopGuardChk.disabled = true;
  try {
    const res = await fetch(API + '/ocr/loop-guard', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ loop_guard: on })
    });
    const body = await res.json();
    /* Show what the server accepted, never what was clicked. */
    if (typeof body.loop_guard === 'boolean') loopGuardChk.checked = body.loop_guard;
    if (Number.isFinite(body.max_tokens)) READ_TOKEN_CAP = body.max_tokens;
  } catch (err) { /* leave it where it was */ }
  loopGuardChk.disabled = false;
  paintLoopGuard();
}

/* What the server holds now: the guard and the cap. Asked on load and when the
   tab comes back into view; it is the app's own state, never the model server. */
async function refreshLoopGuard() {
  try {
    const body = await (await fetch(API + '/ocr/loop-guard')).json();
    if (typeof body.loop_guard === 'boolean' && !loopGuardChk.disabled)
      loopGuardChk.checked = body.loop_guard;
    if (Number.isFinite(body.max_tokens)) READ_TOKEN_CAP = body.max_tokens;
  } catch (err) { return; }
  paintLoopGuard();
}
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshLoopGuard();
});

loopGuardChk.addEventListener('change', () => setLoopGuard(loopGuardChk.checked));
paintLoopGuard();
refreshLoopGuard();

/* ---------- the page-read sampler -------------------------------------------
   Temperature, top_p and top_k of the PAGE READ (pass 1); extraction stays greedy.
   top_k 0 means the field is not sent and the server own top_k applies. One process
   setting, painted from what the server accepted, never from what was typed. */
const sampTemp = document.getElementById('sampTemp');
const sampTopP = document.getElementById('sampTopP');
const sampTopK = document.getElementById('sampTopK');
const sampReset = document.getElementById('sampReset');
const samplingNote = document.getElementById('samplingNote');
let SAMPLING = window.BENCHMARK_BOOTSTRAP.ocr_sampling;

function paintSampling() {
  sampTemp.value = SAMPLING.temperature;
  sampTopP.value = SAMPLING.top_p;
  sampTopK.value = SAMPLING.top_k;
  const d = SAMPLING.defaults || {};
  const greedy = SAMPLING.temperature === 0;
  const changed = d.temperature !== SAMPLING.temperature
    || d.top_p !== SAMPLING.top_p || d.top_k !== SAMPLING.top_k;
  sampReset.disabled = !changed;
  samplingNote.innerHTML = (greedy
      ? '<b>Greedy read</b> (temperature 0): the same page transcribes the same way '
        + 'every time.'
      : 'Reads are <b>not reproducible byte for byte</b> at temperature above 0, so a '
        + 'small difference between two runs is not evidence of anything.')
    + ' Top k <b>0</b> sends no top_k (the server own applies); top k 1 makes the other '
    + 'two moot. Page reading only &mdash; extraction stays greedy. Takes effect on the '
    + 'next read; a run log row does not record the sampler.'
    + (changed ? ' Started at temperature ' + d.temperature + ', top p ' + d.top_p
      + ', top k ' + d.top_k + '.' : '');
}

async function pushSampling(payload) {
  let refusal = '';
  [sampTemp, sampTopP, sampTopK, sampReset].forEach(el => { el.disabled = true; });
  try {
    const res = await fetch(API + '/ocr/sampling', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const body = await res.json();
    if (res.ok) SAMPLING = body;
    else refusal = body.error || 'The sampler was not changed.';
  } catch (err) { /* leave it where it was */ }
  [sampTemp, sampTopP, sampTopK].forEach(el => { el.disabled = false; });
  paintSampling();
  if (refusal) samplingNote.innerHTML = '<b>Not changed:</b> ' + refusal + ' '
    + samplingNote.innerHTML;
}

function samplingFromBoxes() {
  return { temperature: Number(sampTemp.value), top_p: Number(sampTopP.value),
           top_k: Number(sampTopK.value) };
}
/* Awaited before a read so the run uses what is on screen. Sends only a difference. */
async function ensureSampling() {
  const want = samplingFromBoxes();
  if (want.temperature === SAMPLING.temperature && want.top_p === SAMPLING.top_p
      && want.top_k === SAMPLING.top_k) return;
  await pushSampling(want);
}
[sampTemp, sampTopP, sampTopK].forEach(el => el.addEventListener('change', ensureSampling));
sampReset.addEventListener('click', () => pushSampling({ reset: true }));
paintSampling();

/* ---------- the read floor ---------------------------------------------------
   The transcript score a read has to reach before the extraction taken from it
   is worth a field score. Pass 2 can only map values pass 1 produced, so a score
   over a broken transcript marks down whatever extracted and lets whatever read
   get away with it -- measured: dots.mocr agentic scored 32.7% over its own
   reads against 46.8% over the ground truth.

   **Two floors, and only one of them is reversible.** This one decides whether a
   figure is WRITTEN, at the moment a run happens, and a score never taken is not
   in the log to recover. The Summary card's floor decides whether a figure
   already in the log is AVERAGED, and moves freely because every table
   re-derives trust from the row's own transcript score. So this is the one to
   leave low and that is the one to raise.

   ONE setting behind TWO boxes -- here and on the Random test pane -- both
   painted from what the server accepted, never kept in step by each other's
   handlers. Same rule as the model pickers: a control refused by the server has
   to end up showing what is really in force, not what was typed. */
const readFloorInp = document.getElementById('readFloorInp');
const readFloorNote = document.getElementById('readFloorNote');
const rtReadFloorInp = document.getElementById('rtReadFloor');
let READ_FLOOR = window.BENCHMARK_BOOTSTRAP.min_read_for_fields;          /* a FRACTION, like the server's */
const READ_FLOOR_DEFAULT = window.BENCHMARK_BOOTSTRAP.min_read_default;   /* what the process started at */

function paintReadFloor() {
  const pct = (READ_FLOOR * 100).toFixed(0);
  readFloorInp.value = pct;
  if (rtReadFloorInp) rtReadFloorInp.value = pct;
  readFloorNote.innerHTML = READ_FLOOR > 0
    ? 'A read scoring under <b>' + pct + '%</b> still extracts and still logs — '
      + 'the fields come back, each one still marked against the ground truth — '
      + 'but <b>no field score is written</b> for it. That figure would be a '
      + 'measurement of the read that produced it rather than of the extractor: '
      + 'a value pass 1 got wrong cannot then be '
      + 'extracted right. Applies to every path that reads a page — this pane, '
      + 'the Workspace and the random test alike. Set it to <b>0</b> to score '
      + 'everything and let the run-log card decide what to average with its own '
      + 'floor; '
      + 'that one can be moved afterwards and this one cannot.'
    : '<b>Off.</b> Every extraction is scored whatever the read did, so the '
      + 'figure is in the log, and the read floor on the run-log card decides '
      + 'which runs it averages. A looped or truncated read is still left '
      + 'unscored: that transcript is a fragment whatever it scored.'
      + (READ_FLOOR_DEFAULT > 0
          ? ' This process started at ' + (READ_FLOOR_DEFAULT * 100).toFixed(0)
            + '%.' : '');
  /* The random test's scope note quotes the floor. A hoisted function, so this
     is safe to call from here -- which runs long before that section. */
  if (typeof rtPaintScope === 'function' && window.rtScopeNote) rtPaintScope();
}

/* Returns once the server holds the value on screen, so a run started straight
   after a keystroke is read under the floor the user is looking at. Sending it
   only when it differs keeps a Run click from paying for a request that changes
   nothing. */
async function pushReadFloor(box) {
  const asked = Number((box || readFloorInp).value);
  if (!isFinite(asked) || Math.round(asked) === Math.round(READ_FLOOR * 100)) {
    paintReadFloor();
    return;
  }
  readFloorInp.disabled = true;
  if (rtReadFloorInp) rtReadFloorInp.disabled = true;
  try {
    const res = await fetch(API + '/ocr/read-floor', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ min_read_pct: Math.max(0, Math.min(asked, 100)) })
    });
    const body = await res.json();
    /* What the server accepted, never what was typed. */
    if (typeof body.min_read_pct === 'number') READ_FLOOR = body.min_read_pct / 100;
  } catch (err) { /* leave it where it was */ }
  readFloorInp.disabled = false;
  if (rtReadFloorInp) rtReadFloorInp.disabled = false;
  paintReadFloor();
}

readFloorInp.addEventListener('change', () => pushReadFloor(readFloorInp));
if (rtReadFloorInp) {
  rtReadFloorInp.addEventListener('change', () => pushReadFloor(rtReadFloorInp));
}
paintReadFloor();

/* Step progress, from either the Re-extract stream or a full run's stream.
   Returns true when the message was a step message and has been consumed. */
function stepEvent(msg) {
  if (msg.event === 'classified') {
    const { event, ...rest } = msg;
    classified = rest;
  } else if (msg.event === 'segmenting') {
    /* A multi-page file is about to be grouped. Everything the Grouping view
       draws starts here, so a view left open from the last file is emptied
       rather than shown under this one. */
    segWalk = { pages: msg.pages, walk: !!msg.walk, steps: [], heads: {},
                final: null, how: '', asked: 0, answered: 0, refused: 0 };
    segOpen = null; segOpenHeads.clear();
    enableGrouping();
  } else if (msg.event === 'segment_page') {
    if (!segWalk) {
      segWalk = { pages: msg.pages, walk: false, steps: [], heads: {},
                  final: null, how: '', asked: 0, answered: 0, refused: 0 };
    }
    segWalk.steps.push(msg);
    segWalk.heads[msg.page] = msg.head || '';
    if ('asked_total' in msg) {
      segWalk.asked = msg.asked_total; segWalk.answered = msg.answered_total;
      segWalk.refused = msg.refused_total || 0;
    }
    enableGrouping();
    /* Said on the progress bar as well: between the last page read and the
       first field extracted, this is the only thing that says the run moves. */
    pagebar.dataset.grouping = '1';
    pagebar.innerHTML = '<span class="spin"></span> Grouping pages — page ' + msg.page
      + ' of ' + msg.pages + ' · ' + msg.remaining + ' left · session ' + msg.document
      + ' (page' + (msg.doc_pages.length > 1 ? 's ' : ' ') + segRange(msg.doc_pages) + ', '
      + esc(segTypes(msg.doc_codes)) + ')'
      + (segWalk.walk ? ' · ' + segWalk.asked + ' asked' : '')
      + ' <button type="button" class="ghost-sm" data-go="grouping">View</button>';
  } else if (msg.event === 'segmented') {
    /* How many documents the file was read as, and on what authority. Kept
       even when it is one: "one document" is a finding about a three-page file
       quite as much as "three documents" is, and the strip below decides on
       the count rather than on this event existing. */
    fileSplit = msg;
    if (segWalk) { segWalk.final = msg.segments || []; segWalk.how = msg.split_from || ''; }
    if (pagebar.dataset.grouping) { pagebar.textContent = ''; delete pagebar.dataset.grouping; }
  } else if (msg.event === 'document' || msg.event === 'document_done') {
    /* Which document the steps arriving now belong to. Without it a page
       receiving twenty-one step events across three documents cannot say which
       of them any one of them is about. */
    runningDoc = msg;
    if (msg.event === 'document') { classified = null; extractSteps = null; }
  } else if (msg.event === 'extract_steps') {
    extractSteps = msg.steps.map(s => Object.assign({}, s, { status: 'waiting' }));
    // The block every step's message opens with. Kept here so a step's question
    // can be read in context while the run is still going; the finished result
    // carries its own copy for anything loaded after the fact.
    extractPrefix = msg.prompt_prefix || '';
  } else if (msg.event === 'extract_step') {
    if (!extractSteps) extractSteps = [];
    const i = msg.step - 1;
    extractSteps[i] = Object.assign({}, extractSteps[i] || {}, msg);
  } else if (msg.event === 'table_agent') {
    /* The item table's two agents, as they run. Kept per agent so the Table tab
       can say which one is working; the finished result replaces all of it. */
    tableLive = Object.assign({}, tableLive || {}, { [msg.agent]: msg });
    if (msg.status === 'running') {
      pagebar.innerHTML = '<span class="spin"></span> Item table'
        + (msg.document ? ' (document ' + msg.document + ')' : '') + ' — '
        + (msg.agent === 'fix' ? 'fix agent checking ' + (msg.rows || 0) + ' rows'
           : msg.agent === 'concat' ? 'concat agent: are the other ' + (msg.offered || 0)
             + ' tables part of this one?'
           : 'identify agent: master table or item list?')
        + (msg.model ? ' · ' + esc(msg.model) : '') + '…'
        + ' <button type="button" class="ghost-sm" data-go="table">View</button>';
      pagebar.dataset.table = '1';
    } else if (msg.agent === 'identify' && pagebar.dataset.table) {
      // The last table event: the line it put up comes down, on every stream.
      pagebar.textContent = ''; delete pagebar.dataset.table;
    }
    paintTableTab();
  } else {
    return false;
  }
  if (curMode === 'table') paintTableView();
  if (tabFields.classList.contains('on')) paintFields();
  if (curMode === 'grouping') scheduleGrouping();
  return true;
}

/* Which steps have their reply open. Kept outside the render because the list is
   repainted on every step event, and a panel that closed itself each time a later
   step finished would be unreadable during the run it is there to explain. */
const openStepRaw = new Set();
/* And which have their question open. Two sets rather than one panel holding
   both: the question is the same on every run until a prompt changes, the reply
   is different every time, and wanting one is not wanting the other. */
const openStepPrompt = new Set();
/* The shared prefix is opened once for the whole list, not per step -- it is the
   same string in all of them, and a copy under each step would say the opposite
   of what the design does. */
let showStepPrefix = false;
let extractPrefix = '';
/* The `segmented` event: how many documents this file was read as, and how that
   was decided. Cleared with the rest of a run's state. */
let fileSplit = null;
/* The `document` / `document_done` event, while a split file is being
   extracted one document at a time. */
let runningDoc = null;

/* Which pages each document of a split file is, and how sure the split is.
   Nothing here is worked out from the pages -- `segment.py` decided it and said
   why, and a second reading of the same evidence in JavaScript would eventually
   disagree with the one the fields were actually extracted under. */
/* ---------- the Grouping view ---------------------------------------------
   Which pages became which documents, drawn from the run's own events: live
   while the walk goes -- the page it is on, how many are left, the session it
   is building -- and settled once `segmented` arrives, which is authoritative
   (the one-shot model may have regrouped what the rules walked). Each group
   opens to its own pages, so a boundary can be checked against the page it was
   decided on while the run is still going. */
function segRange(pages) {
  if (!pages || !pages.length) return '';
  return pages.length === 1 ? String(pages[0])
                            : pages[0] + '–' + pages[pages.length - 1];
}
function segTypes(codes) {
  return (codes || []).join(' + ').replace(/_/g, ' ').toLowerCase() || 'untyped';
}
function enableGrouping() {
  tabGroup.disabled = !(segWalk && segWalk.pages > 1);
}
function resetGrouping() {
  segWalk = null; segOpen = null; segOpenHeads.clear();
  enableGrouping();
  if (curMode === 'grouping') paintGrouping();
}
let groupingFrame = 0;
function scheduleGrouping() {
  if (groupingFrame) return;
  groupingFrame = requestAnimationFrame(() => { groupingFrame = 0; paintGrouping(); });
}
/* How a boundary was settled, in three words and a colour. `rules` is a
   reading of the page, `asked` is the model's answer (quoted back from the page
   before it was believed), `guess` is what is left when neither settled it. */
function segSettled(entry, how) {
  if (how === 'model') return ['asked', 'model decided'];
  if (entry.asked && entry.answered) return ['asked', 'model decided'];
  if (entry.certain === false) return ['guess', entry.asked ? 'guess (model refused)' : 'guess'];
  return ['rules', 'read off the page'];
}
function segGroups() {
  const w = segWalk;
  if (!w) return [];
  const stepOf = {};
  for (const st of w.steps) stepOf[st.page] = st;
  if (w.final && w.final.length) {
    return w.final.map(r => {
      const open = stepOf[r.pages[0]] || {};
      return { document: r.document, pages: r.pages, codes: r.doc_types,
               why: r.start_reason, certain: r.certain,
               asked: !!open.asked, answered: !!open.answered,
               joins: (r.joins || []).map(j => Object.assign({}, j, {
                 asked: !!(stepOf[j.page] || {}).asked,
                 answered: !!(stepOf[j.page] || {}).answered })) };
    });
  }
  const by = new Map();
  for (const st of w.steps) {
    if (!by.has(st.document)) {
      by.set(st.document, { document: st.document, pages: [], codes: [], why: '',
                            certain: true, asked: false, answered: false, joins: [] });
    }
    const g = by.get(st.document);
    g.pages = st.doc_pages; g.codes = st.doc_codes;
    if (st.opened) {
      g.why = st.why; g.certain = st.certain; g.asked = st.asked; g.answered = st.answered;
    } else {
      g.joins.push({ page: st.page, why: st.why, certain: st.certain,
                     asked: st.asked, answered: st.answered });
    }
  }
  return [...by.values()];
}
function segPagesHtml(g) {
  const how = segWalk.how;
  const cards = g.pages.map(n => {
    const join = g.joins.find(j => j.page === n);
    const entry = n === g.pages[0] ? g : (join || { certain: true });
    const why = n === g.pages[0] ? g.why : (join ? join.why : '');
    const [cls, word] = segSettled(entry, how);
    const head = segWalk.heads[n] || '';
    return '<div class="grp-page">'
      + '<div class="grp-pn"><span>Page ' + n + (n === g.pages[0] ? ' · opens' : '') + '</span>'
      + '<span class="grp-badge ' + cls + '">' + esc(word) + '</span></div>'
      + (job ? '<a href="' + pageSrc(n - 1)
               + '" target="_blank" rel="noopener" title="Open this page full size">'
               + '<img loading="lazy" alt="Page ' + n + '" src="'
               + pageSrc(n - 1) + '"></a>' : '')
      + '<div class="grp-why">' + esc(why || '') + '</div>'
      + (head ? '<details data-page="' + n + '"' + (segOpenHeads.has(n) ? ' open' : '')
                + '><summary>Text the grouping read</summary><pre>' + esc(head)
                + '</pre></details>' : '')
      + '</div>';
  }).join('');
  return '<tr class="grp-verify"><td colspan="5">'
    + (job ? '' : '<p class="note">No page images for this run -- the transcript did not come '
                  + 'from a read. The text each decision was taken on is under each page.</p>')
    + '<div class="grp-pages">' + cards + '</div></td></tr>';
}
function paintGrouping() {
  const w = segWalk;
  if (!w) {
    groupingEl.innerHTML = '<p class="note">Nothing grouped yet. A file of more than one '
      + 'page is split into documents before its fields are extracted, and that '
      + 'grouping appears here as it happens.</p>';
    return;
  }
  const scroller = groupingEl.querySelector('.grp-scroll');
  const keep = scroller ? scroller.scrollTop : 0;
  const groups = segGroups();
  const settled = !!(w.final && w.final.length);
  const last = w.steps[w.steps.length - 1];
  const done = settled ? w.pages : (last ? last.page : 0);
  const howText = { rules: 'every boundary read off the pages',
                    model: 'regrouped by the model in one request',
                    chat: 'walked page by page with the model',
                    guess: 'some boundaries are guesses nothing settled',
                    whole: 'read as one document' }[w.how] || '';
  let html = '<div class="grp-head"><b>' + (settled
      ? groups.length + ' document' + (groups.length === 1 ? '' : 's') + ' from '
        + w.pages + ' pages'
      : 'Grouping ' + w.pages + ' pages — page ' + done + ' of ' + w.pages + ', '
        + (w.pages - done) + ' left') + '</b>'
    + (settled && howText ? '<span class="muted-inline">' + esc(howText) + '</span>' : '')
    + (w.walk ? '<span class="muted-inline" title="Questions put to the model at boundaries the pages did not settle. A boundary the pages settle costs none.">'
                + w.asked + ' question' + (w.asked === 1 ? '' : 's') + ' asked · '
                + w.answered + ' answered' + (w.refused ? ' · ' + w.refused + ' refused' : '')
                + '</span>' : '')
    + '</div>'
    + '<div class="grp-bar' + (settled ? ' done' : '') + '"><span style="width:'
    + (w.pages ? Math.round(100 * done / w.pages) : 0) + '%"></span></div>';
  if (!settled && last) {
    const cur = groups.find(g => g.document === last.document) || {};
    html += '<div class="grp-now"><b>Session ' + last.document + '</b>'
      + '<span>page' + ((cur.pages || []).length > 1 ? 's ' : ' ') + segRange(cur.pages)
      + ' · ' + esc(segTypes(cur.codes)) + '</span>'
      + '<span class="grp-why">page ' + last.page + ' ' + (last.opened ? 'opened it' : 'was kept')
      + ' — ' + esc(last.why) + '</span>'
      + '<button type="button" class="ghost-sm" data-doc="' + last.document + '">'
      + (segOpen === last.document ? 'Hide pages' : 'View pages') + '</button></div>';
  }
  html += '<div class="grp-scroll"><table class="grp"><thead><tr><th>Document</th><th>Pages</th>'
    + '<th>Type</th><th>Started because</th><th>Settled</th></tr></thead><tbody>';
  for (const g of groups) {
    const [cls, word] = segSettled(g, w.how);
    const isCur = !settled && last && g.document === last.document;
    const asked = g.joins.filter(j => j.asked).length + (g.asked ? 1 : 0);
    html += '<tr class="grp-row' + (isCur ? ' cur' : '') + (segOpen === g.document ? ' open' : '')
      + '" data-doc="' + g.document + '" title="Click to ' + (segOpen === g.document ? 'hide' : 'check')
      + ' this group’s pages">'
      + '<td>' + g.document + (isCur ? ' <span class="grp-badge asked">now</span>' : '') + '</td>'
      + '<td>' + segRange(g.pages) + ' <span class="muted-inline">(' + g.pages.length + ')</span></td>'
      + '<td>' + esc(segTypes(g.codes)) + '</td>'
      + '<td class="grp-why">' + esc(g.why || '') + '</td>'
      + '<td><span class="grp-badge ' + cls + '">' + esc(word) + '</span>'
      + (asked ? ' <span class="muted-inline">' + asked + ' asked</span>' : '') + '</td></tr>';
    if (segOpen === g.document) html += segPagesHtml(g);
  }
  html += '</tbody></table></div>';
  groupingEl.innerHTML = html;
  const box = groupingEl.querySelector('.grp-scroll');
  if (box) box.scrollTop = keep;
}
groupingEl.addEventListener('click', e => {
  const hit = e.target.closest('[data-doc]');
  if (!hit || !groupingEl.contains(hit)) return;
  const doc = Number(hit.dataset.doc);
  segOpen = segOpen === doc ? null : doc;
  paintGrouping();
});
/* `toggle` does not bubble, so it is caught on the way down. */
groupingEl.addEventListener('toggle', e => {
  const d = e.target;
  if (!d.dataset || !d.dataset.page) return;
  const n = Number(d.dataset.page);
  if (d.open) segOpenHeads.add(n); else segOpenHeads.delete(n);
}, true);
pagebar.addEventListener('click', e => {
  const go = e.target.closest('[data-go="grouping"]');
  if (!go || tabGroup.disabled || curMode === 'grouping') return;
  lastMode = curMode;
  showTab('grouping');
});

function docStrip() {
  const docs = (extractedFile && extractedFile.documents) || null;
  if (!docs || docs.length < 2) return '';
  const segs = (extractedFile.segments) || [];
  /* `chat` is the rolling walk: too long for one question about the whole
     file, so one question per unsettled boundary with the document so far as
     the conversation. Worth its own wording rather than folding into `model`,
     because the two were asked different questions. */
  const how = {rules: 'read off the pages', model: 'decided by the model',
               chat: 'walked page by page with the model',
               guess: 'a guess -- nothing on the pages settled it',
               whole: 'not split'}[extractedFile.split_from] || '';
  return '<div class="docs">' + docs.map((d, i) => {
      const seg = segs[i] || {};
      const types = (d.doc_types || []).join(' + ').replace(/_/g, ' ').toLowerCase();
      const pages = d.page_range || (seg.page_range || '');
      /* This document's own field score, where it has one. On the button
         because the strip is where a reader decides which document to look at,
         and "which of the seven went wrong" is the question they are asking. */
      const own = ((d.field_score || {}).overall) || {};
      const rate = (!d.fields_unscored && own.expected)
        ? ' · ' + (own.accuracy * 100).toFixed(0) + '% of ' + own.expected : '';
      return '<button type="button" data-doc="' + i + '"'
        + (i === extractedDoc ? ' class="on' : ' class="')
        + (d.error ? ' bad' : (seg.certain === false ? ' guess' : '')) + '"'
        + ' title="' + esc(seg.start_reason || '') + '">'
        + 'Document ' + (i + 1)
        + '<small>' + esc(types || 'unclassified')
        + ' · page' + (String(pages).indexOf('-') >= 0 ? 's ' : ' ') + esc(pages)
        + esc(rate)
        + (d.error ? ' · failed' : '')
        + (seg.certain === false ? ' · uncertain start' : '') + '</small></button>';
    }).join('') + '</div>'
    /* Said once above the strip rather than on every button: it is a fact about
       the FILE, and repeating it under each document would read as a separate
       claim about each of them. */
    + '<p class="split-note">This file was read as ' + docs.length
    + ' documents' + (how ? ' — ' + esc(how) : '') + '. '
    + 'Each is extracted from its own pages only, so a value on one page cannot '
    + 'fill a key of another document.'
    + fileScoreNote() + '</p>';
}
/* The FILE's score, where its documents were scored. The marks and the rate
   above the fields are one document's -- the one on screen -- and this is the
   other question: how did the file do. Pooled, so it follows from the counts
   beside it; the unweighted mean of the documents' own rates is named after it
   as the other reading, because a file of one big document and six small ones
   answers the two differently. */
function fileScoreNote() {
  const score = (extractedFile || {}).field_score || {};
  const o = score.overall || {};
  if (!o.expected || !score.scored_documents) return '';
  /* **The headline is the mean of the documents since 2026-09-11**, so it is
     NOT a share of `expected` and must not be printed as one -- that count is
     the sum of the documents' denominators and belongs to the pooled figure
     beside it. Printing the mean "of N values" was true of the pooled rate and
     is false of this one, which is the kind of small lie that gets a number
     quoted wrongly. */
  const pooled = o.accuracy_pooled;
  return ' Over the whole file, <b>' + (o.accuracy * 100).toFixed(0)
    + '% counting each document once</b>, across ' + score.scored_documents
    + ' scored document' + (score.scored_documents > 1 ? 's' : '')
    + (pooled === null || pooled === undefined ? ''
       : ' (' + (pooled * 100).toFixed(0) + '% of all ' + o.expected
         + ' values pooled, where a document with more of them counts for more)')
    + '.';
}
function stepPrefixText() {
  return (extracted && extracted.prompt_prefix) || extractPrefix || '';
}

function stepList(steps) {
  if (!steps || !steps.length) return '';
  return '<div class="fld-sec">Extraction steps</div><ul class="step-list">'
    + steps.map((s, i) => {
        const keys = s.keys || [];
        // The live events carry a status; the finished record does not, because
        // by the time it is in the result every step it lists has run.
        const status = s.status || 'done';
        const done = status === 'done';
        const filled = Object.values(s.values || {})
          .filter(v => v !== '' && v !== null && v !== undefined).length;
        // Did this step leave any of what it owns empty? A list step reports its
        // length instead of its keys, so an empty list is what "empty" means
        // there -- `values` deliberately excludes lists, and 0/1 on the `other`
        // step would otherwise call every run short.
        const short = keys.length ? (s.items ? false : filled < keys.length) : false;
        let note = '', cls = '';
        if (s.error) {
          note = esc(s.error);
          cls = 'bad';
        } else if (s.extras_error && done) {
          /* An extras-only step that broke. Flagged, never `bad`: it costs
             entries nothing scores, so colouring it like a failed step would
             say the extraction went wrong when the form came back whole. */
          note = 'extra fields — ' + esc(s.extras_error)
               + (s.items ? ' · kept ' + s.items + (s.items === 1 ? ' entry' : ' entries')
                          : ' · nothing kept');
          cls = 'flagged';
        } else if (done) {
          const bits = [];
          // A step that owns a list reports its length; one that owns scalars
          // reports how many of its own keys the page filled, which is not a
          // score -- most documents leave whole steps legitimately empty.
          if (s.items) bits.push(s.items + (s.items === 1 ? ' row' : ' rows'));
          // "0/0" is not a count, it is a step that did not say what it owns.
          else if (keys.length) bits.push(filled + '/' + keys.length);
          // The reply stopped mid-entry and what had finished was kept. After
          // the count rather than inside it -- and NOT between those two
          // branches, which silently detaches the `else` from its `if` and
          // prints a row count and a key count on the same clean step.
          if (s.salvaged) bits.push('cut off — kept what finished');
          if (s.retry_error) bits.push('re-ask failed');
          else if (s.retried) bits.push('re-asked');
          if (s.ungrounded) { bits.push(s.ungrounded + ' not on page'); cls = 'flagged'; }
          if (s.seconds) bits.push(s.seconds + 's');
          note = esc(bits.join(' · '));
        } else if (status === 'running') {
          note = '<span class="spin"></span>';
        } else {
          cls = 'waiting';
        }
        // The reply this step actually got, verbatim: one block per request, so
        // the plain attempt, the schema-constrained rescue and a re-ask are told
        // apart rather than concatenated into one wall of JSON.
        const raws = s.raw || [];
        const id = s.id || String(i);
        const open = openStepRaw.has(id);
        const openAsk = openStepPrompt.has(id);
        // The questions this step sent, deduplicated: the plain attempt and the
        // schema-constrained one send identical text and differ only in the
        // request body, so they are one question, not two. A re-ask is a
        // different question and appears as one.
        const asks = [];
        if (s.prompt) asks.push({ label: 'question', text: s.prompt });
        raws.forEach(r => {
          if (r.prompt && !asks.some(a => a.text === r.prompt)) {
            asks.push({ label: r.label + ' — question', text: r.prompt });
          }
        });
        const btn = (kind, on, label, tip) =>
          '<button type="button" class="step-' + kind + '-btn ghost-sm' + (on ? ' on' : '')
          + '" data-step="' + esc(id) + '" aria-expanded="' + on + '" title="' + esc(tip)
          + '">' + label + '</button>';
        /* Each button is offered where it answers something, and not otherwise.
           The question is for rechecking an answer that is already in, so it
           appears once the step has answered -- there is nothing to check it
           against while the step is still running. */
        const askable = done && asks.length;
        /* The reply is for when the answer is wrong or absent: a step that
           failed, was re-asked, needed the schema rescue, answered with
           something that is not on the page, or left one of its own keys empty.
           A step that answered cleanly has nothing to explain, and its reply is
           still in the whole-reply pane above if anyone wants it. */
        const why = s.error ? 'this step failed'
                  : s.extras_error ? 'its reply broke before it finished listing '
                                     + 'the extra fields'
                  : s.salvaged ? 'its reply was cut off and only the entries that '
                                 + 'finished were kept'
                  : s.retry_error ? 'the re-ask failed'
                  : s.ungrounded ? 'it answered with a value that is not on the page'
                  : s.schema_retry ? 'its first reply would not parse'
                  : s.retried ? 'it was asked again'
                  : short ? 'it left ' + (keys.length - filled > 1 ? 'keys' : 'a key')
                            + ' empty'
                  : '';
        const showRaw = done && raws.length && why;
        const toggles =
          (askable ? btn('prompt', openAsk, 'prompt',
             'what this step asked for, exactly as it was sent — to read against '
             + 'the answer it produced') : '')
          + (showRaw ? btn('raw', open, 'raw',
             'shown because ' + why + '. The reply, exactly as it arrived.') : '');
        return '<li class="' + cls + '"><span class="n">' + (i + 1) + '/' + steps.length
          + '</span><span class="t">' + esc(s.title || s.id || '') + '</span>'
          + '<span class="s">' + note + '</span>' + toggles + '</li>'
          // Question first, then answer, in the order they happened. Both panels
          // are gated on the same test as their button: a panel left open by an
          // earlier run whose button this one does not offer would have no way
          // to be closed again.
          + (s.chats && s.chats.length ? '<li class="raw-row"><details><summary>Chat context ('
              + s.chats.length + ' request(s))</summary>'
              + agenticChatTable(s.chats.map(chat => ({...chat, step: s.id}))) + '</details></li>' : '')
          + (openAsk && askable ? '<li class="raw-row"><div class="step-raw">'
              + '<div class="step-raw-head">shared prefix'
              + (stepPrefixText()
                  ? ' · ' + stepPrefixText().length.toLocaleString() + ' chars · the same '
                    + 'instructions and transcript open every step'
                    + '<button type="button" class="step-prefix-btn ghost-sm">'
                    + (showStepPrefix ? 'hide' : 'show') + '</button>'
                  : ' · not recorded for this run')
              + '</div>'
              + (showStepPrefix && stepPrefixText()
                  ? '<pre>' + esc(stepPrefixText()) + '</pre>' : '')
              + asks.map(a => '<div class="step-raw-head">' + esc(a.label)
                  + ' · sent after the prefix · ' + a.text.length.toLocaleString()
                  + ' chars</div><pre>' + esc(a.text) + '</pre>').join('')
              + '</div></li>' : '')
          + (open && showRaw ? '<li class="raw-row"><div class="step-raw">'
              + raws.map(r => '<div class="step-raw-head">' + esc(r.label || '')
                  + ' · ' + (r.text || '').length.toLocaleString() + ' chars</div>'
                  + ((r.text || '').trim()
                      ? '<pre>' + esc(r.text) + '</pre>'
                      : '<p class="empty-reply">The model returned nothing at all '
                        + 'for this request.</p>')).join('')
              + '</div></li>' : '');
      }).join('') + '</ul>';
}

/* Said above the fields rather than only in the step list, because a step that
   failed leaves its keys empty and they are otherwise indistinguishable from
   fields the document does not state -- which is the one distinction the
   grounding note below is careful to make. */
/* How an extraction was asked for, in the words the picker uses. `agentic` says
   how many steps it actually ran rather than how many the table has, because a
   result carried over from an older run may have had a different step list. */
function modeLabel(r) {
  const mode = r && r.mode;
  if (isStepMode(mode)) {
    const steps = (r.steps || []).length;
    return mode + (steps ? ' · ' + steps + ' steps' : '')
      + (r.schema_first ? ' · Schema first' : ' · JSON first');
  }
  if (mode === 'single') return 'single prompt';
  return mode ? esc(mode) : 'extraction';
}
function stepNote(r) {
  const failed = (r && r.steps_failed) || [];
  const retried = ((r && r.steps) || []).filter(s => s.retried).length;
  if (!r || !isStepMode(r.mode) || !r.steps) return '';
  let html = '<p class="ground-note' + (failed.length ? ' bad' : '') + '">Extracted in '
    + r.steps.length + ' steps of one to three fields.';
  if (retried) {
    html += ' ' + retried + ' step(s) returned a value that is not on the page and were '
      + 'asked again.';
  }
  if (failed.length) {
    html += ' <strong>' + failed.length + ' step(s) failed</strong>, so their fields are '
      + 'empty because they were never answered — not because the document is silent:'
      + '<ul>' + failed.map(f =>
          '<li>' + esc(f.title) + ' — ' + esc(f.error) + '</li>').join('') + '</ul>';
  }
  /* The extras step, said separately and without the failure wording. A reply
     that stopped part-way through `other_fields` costs entries nothing scores,
     so it is a statement about how much of the page came back — not about
     whether the extraction worked. The distinction matters the same way
     `partial` does: it stops a short list reading as a page with little on it. */
  const extras = (r.steps_extras || []).length;
  const salvaged = r.steps.filter(s => s.salvaged).length;
  if (extras || salvaged) {
    html += ' The extra-fields step stopped part-way through its list, so what is '
      + 'under “other fields” is what it had named before it stopped'
      + (extras && !salvaged ? ' — nothing, in this run' : '')
      + '. The fields above are unaffected and this is not counted as a failure.';
  }
  return html + '</p>';
}

/* A newline for the tooltips below. Written this way rather than as an escape
   because these strings pass through a Jinja template on the way out. */
const NL = String.fromCharCode(10);

/* What this stage is blamed for, on its own box. The strip is the one place
   that already draws the two passes side by side, so a count of the values each
   lost belongs on it -- and putting it here rather than only under the fields is
   what makes the question answerable at a glance, which is what it was asked
   for. Silent where nothing was attributed to the stage, so a clean run draws
   an ordinary strip. */
function pipeBlame(who) {
  const c = ((extracted && extracted.blame) || {}).counts;
  const n = c && c[who];
  return n ? ' · <b class="pipe-bl">' + n + ' field' + (n > 1 ? 's' : '')
             + ' lost</b>' : '';
}
function pipeBlameHint(who) {
  const c = ((extracted && extracted.blame) || {}).counts;
  const n = c && c[who];
  /* A statement about the STAGE, which is what this box is, so it cannot
     contradict any one value: the per-value reasons are on the chips beside the
     values themselves. */
  return n ? NL + NL + n + ' of the values the field score charged for '
             + (n > 1 ? 'are' : 'is') + ' down to this stage — '
             + BLAME_TAG[who][1] + '. Each one is marked on the Fields tab with '
             + 'what was looked for and where.'
           : '';
}

/* One stage box. `state` is done / run / wait / skip. */
function pipeBox(n, name, state, value, detail, hint) {
  return '<div class="pipe-st ' + state + '"' + (hint ? ' title="' + esc(hint) + '"' : '')
    + '><div class="pipe-h"><span class="pipe-n">' + n + '</span>' + esc(name) + '</div>'
    + '<div class="pipe-v">' + (value || '') + '</div>'
    + (detail ? '<div class="pipe-d">' + detail + '</div>' : '') + '</div>';
}

/* Where the type came from, in words. The same four authorities the tooltip
   names, shortened to fit a box. */
/* Whose answer stage 0 took, in the order it asks. `model` is the ordinary case
   for a real document since 2026-09-02 and `transcript` is the fallback under
   it -- so `transcript` is not a lesser answer, it is the one Python reached
   when the model's could not be believed. */
const CLASSIFY_FROM = {
  case: 'from the benchmark manifest',
  transcript: "matched from the page's own heading",
  model: 'read by the model, quoting the page back',
  caller: 'given by the request',
};

/* Read → classify → extract, the three stages of a run, in order.
   Painted from three different places on purpose: pass 1 from the page state,
   stage 0 from its event or from the finished result, pass 2 from `extracted`.
   None of the three can report the other two, and a strip assembled from one of
   them would go stale for the other two. */
function paintPipeline() {
  const c = classified || (extracted && extracted.classified) || null;
  const done = extracted && !extracted.pending && !extracted.error;
  if (!c && !extracted && !truth && !pageStats.length) { pipeline.innerHTML = ''; return; }

  // 1. Read. A fields-only run never read a page, and saying so is the point:
  //    a blank box there reads as a read that failed.
  let read;
  if (truthOnlyCase) {
    read = pipeBox(1, 'Read', 'skip', 'not read',
                   'fed <b>solution/' + esc(truthOnlyCase) + '.md</b>',
                   'This run was fed the ground-truth transcript, so pass 1 '
                   + 'contributes no errors of its own to the fields below.');
  } else if (pageStats.length) {
    const secs = pageStats.reduce((a, p) => a + (Number(p.seconds) || 0), 0);
    read = pipeBox(1, 'Read', 'done',
                   pageStats.length + ' page' + (pageStats.length > 1 ? 's' : ''),
                   Math.round(secs) + 's'
                     + (truth && truth.char_accuracy !== undefined
                        ? ' · ' + (truth.char_accuracy * 100).toFixed(1) + '% chars' : '')
                     + pipeBlame('ocr'),
                   'Pass 1: the page image in, a verbatim transcript out.'
                     + pipeBlameHint('ocr'));
  } else {
    read = pipeBox(1, 'Read', 'wait', 'no page read', '',
                   'Pass 1 has not run in this tab.');
  }

  // 2. Classify. The value is what the document IS; the detail is what that
  //    bought, which is the half a reader needs to make sense of stage 3.
  let cls;
  if (c) {
    const types = (c.doc_types || []).join(' + ').replace(/_/g, ' ').toLowerCase();
    const bits = [];
    /* How many documents the file was read as, ahead of the field count and
       only where it is more than one. It has to come first because it changes
       what every figure after it is ABOUT: "11 fields" is the whole form on a
       one-document file and one document's form on a split one. Neither good
       nor bad, so it takes neither the green nor the amber -- a file really
       holding three documents is the ordinary case for a file of three. */
    const found = (extractedFile && extractedFile.documents_found)
      || (fileSplit && fileSplit.documents) || 1;
    if (found > 1) {
      /* While the run is going, the document the events are arriving for;
         once it is finished, the one the strip has selected. In that order:
         `runningDoc` is cleared the moment the result lands, so a finished file
         can never report itself as still on its last document. */
      const at = (runningDoc && runningDoc.document) || (extractedDoc + 1);
      bits.push('<b>' + found + ' documents</b> ('
                + (runningDoc ? 'on ' : 'showing ') + at + ')');
    }
    bits.push(esc(c.fields.length + ' fields'
                  // What the score will be over. On the box rather than only in
                  // its tooltip: this stage chose the form, and how much of that
                  // form is compulsory is the half stage 3 is judged on.
                  + ((c.mandatory || []).length
                     ? ', ' + c.mandatory.length + ' required' : '')));
    /* No requirement covers this type yet, so the form is the base field set and
       nothing on it is Mandatory. Said rather than left as a missing "N
       required" clause: an absent clause reads as a form nobody can pass, and
       the fact it stands for is the one that explains both why there are thirty
       rows below and why the score beside them is not a compliance figure. */
    if (c.unknown_type) bits.push('<b class="pipe-warn">unknown type</b>');
    if (c.steps && c.steps.length) bits.push(esc(c.steps.length + ' steps'));
    if (c.rules && c.rules.length) bits.push(esc(c.rules.length + ' extra checks'));
    /* The manifest's own answer, where this run did not use it and the two
       differ. A finding, not a field: a fixture whose form was chosen by the
       model against a person's reading is the one case where a moved field
       score has an explanation that is not the extractor's. */
    const disagree = (c.doc_type_expected || []).length
      ? ' · <b class="pipe-warn">manifest says '
        + esc(c.doc_type_expected.join(' + ').replace(/_/g, ' ').toLowerCase())
        + '</b>' : '';
    /* How sure the page's own heading made this, 0-1, or null where a person
       gave the answer and there is nothing on the page to score. It decides
       whether the model is asked at all, so it belongs beside the answer rather
       than in its tooltip -- and a run answered by the code at 43% is a
       different thing from one answered by the code at 100%. */
    const conf = c.doc_type_confidence;
    const sure = (conf === null || conf === undefined) ? ''
      : ' · <b class="' + (conf >= MIN_CLASSIFY_CONFIDENCE ? 'pipe-sure' : 'pipe-warn')
        + '">' + Math.round(conf * 100) + '% sure</b>';
    cls = pipeBox(2, 'Classify', disagree ? 'warn' : 'done',
                  esc(types || 'unclassified'),
                  esc(CLASSIFY_FROM[c.doc_type_from] || 'not classified')
                    + sure + ' · ' + bits.join(' · ') + disagree,
                  (c.doc_type_escalated
                     ? 'The page-heading table was not taken on its own here: '
                       + c.doc_type_escalated + '.' + NL
                     : '')
                    + ((conf === null || conf === undefined)
                     ? 'No confidence figure: a person gave this answer -- the '
                       + 'request or the benchmark manifest -- so there is no '
                       + 'heading on the page to score.' + NL
                     : 'How much of the line it was read off is actually the '
                       + 'heading, discounted a little further down the page. It '
                       + 'measures the evidence on the page, not how sure the '
                       + 'classifier feels, '
                       + 'self-belief, which is what lets the same figure describe '
                       + 'the Python table and the model. At or above '
                       + Math.round(MIN_CLASSIFY_CONFIDENCE * 100) + '% the table '
                       + 'answers on its own and no request is made; below it the '
                       + 'model is asked.' + NL)
                    + ((c.doc_type_expected || []).length
                     ? 'The benchmark manifest says this document is '
                       + c.doc_type_expected.join(' + ') + ', and this run did not '
                       + 'use that answer. The form below follows the answer above,'
                       + ' so a field score taken here is over a different form '
                       + 'from the one the truth file was written against.' + NL
                     : '')
                    + (c.doc_type_heading ? 'Heading read: ' + c.doc_type_heading + NL : '')
                    + 'Chose the form below: ' + c.fields.join(', ')
                    + (c.mandatory && c.mandatory.length
                       ? NL + 'Mandatory on this type: ' + c.mandatory.join(', ')
                       : c.unknown_type
                       ? NL + 'No requirement covers this type yet, so this is the '
                         + 'base field set -- the union of every requirement this '
                         + 'project has. Nothing on it is Mandatory, no extra '
                         + 'validation rule runs, and the field score below is '
                         + 'taken over every key the ground truth states rather '
                         + 'than over a compliance set.' : '')
                    + (c.rules && c.rules.length
                       ? NL + 'Validation rules these types add: '
                         + c.rules.join(', ').replace(/_/g, ' ') : '')
                    + NL + (c.framed
                       ? 'The extract prompt was told this.'
                       : 'The extract prompt was NOT told this — it chose the '
                         + 'form only (TYPE_FRAMING is off for this mode).'));
  } else {
    cls = pipeBox(2, 'Classify', 'wait', '—', '',
                  'Which kind of document this is, decided in Python from the '
                  + 'printed heading before either prompt is built. It chooses '
                  + 'the fields asked for below.');
  }

  // 3. Extract.
  let ext;
  if (extracted && extracted.error) {
    ext = pipeBox(3, 'Extract', 'skip', 'failed', esc((extracted.error || '').slice(0, 60)),
                  extracted.error);
  } else if (extracted && extracted.pending) {
    const steps = extractSteps || [];
    const at = steps.filter(x => x.status === 'done').length;
    ext = pipeBox(3, 'Extract', 'run', 'running…',
                  steps.length ? esc(at + ' of ' + steps.length + ' steps') : '',
                  'Pass 2: the transcript back in as text, the form out as JSON.');
  } else if (done) {
    const tiers = extracted.tiers || {};
    const p = Number(tiers.p1_present), a = Number(tiers.p1_absent);
    const filled = Number.isFinite(p) && p + a > 0 ? p + '/' + (p + a) + ' filled' : '';
    ext = pipeBox(3, 'Extract', 'done', esc(modeLabel(extracted)),
                  [filled, extracted.seconds + 's'].filter(Boolean).join(' · ')
                    + pipeBlame('extraction'),
                  'Pass 2 asked for exactly the form stage 2 chose.'
                    + pipeBlameHint('extraction'));
  } else {
    ext = pipeBox(3, 'Extract', 'wait', 'not run', '', '');
  }
  pipeline.innerHTML = read + cls + ext;
}

function paintFields() {
  paintPipeline();
  paintFieldBody();
  /* Prepended here rather than at each of `paintFieldBody`'s five exits: that
     function rewrites the whole body on every one of them, and a strip added at
     four of the five would vanish on whichever path nobody tested. */
  const strip = docStrip();
  if (strip) fldBody.insertAdjacentHTML('afterbegin', strip);
  if (extracted && !extracted.pending) {
    if (extracted.image_filled?.length) fldBody.insertAdjacentHTML('beforeend',
      '<p class="note">Filled from images: ' + extracted.image_filled.map(esc).join(', ') + '</p>');
    for (const retry of extracted.validation_image_rechecks || []) fldBody.insertAdjacentHTML('beforeend',
      '<details><summary>Validation image recheck · ' + (retry.accepted ? 'accepted' : 'original values kept')
      + '</summary><pre>' + esc(JSON.stringify(retry, (key, value) => key === 'chats' ? undefined : value, 2)) + '</pre></details>');
    const rechecks = (extractedFile || extracted).image_rechecks;
    if (rechecks) fldBody.insertAdjacentHTML('beforeend', '<details><summary>Image recheck summary</summary><pre>'
      + esc(JSON.stringify(rechecks, null, 2)) + '</pre></details>');
  }
  // Left on the normal body until there is a reply to show: an empty pane while
  // the pass is still running reads as "the model returned nothing", when the
  // body underneath is saying it has not been asked yet.
  const raw = fldRawChk.checked && !!extracted && !extracted.pending;
  fldBody.hidden = raw;
  fldRaw.hidden = !raw;
  if (raw) paintRaw(fldRaw, extracted, extracted.raw);
}
function paintFieldBody() {
  if (!extracted) {
    fldStatus.textContent = 'Not extracted yet.';
    fldBody.innerHTML = '';
    return;
  }
  if (extracted.pending) {
    const steps = extractSteps || [];
    const done = steps.filter(s => s.status === 'done').length;
    fldStatus.innerHTML = '<span class="spin"></span> ' + (steps.length
      ? 'Extracting field by field — ' + done + ' of ' + steps.length + ' steps done…'
      : 'Asking the model for structured fields…');
    fldBody.innerHTML = stepList(steps);
    return;
  }
  if (extracted.error) {
    fldStatus.textContent = '';
    fldBody.innerHTML = '<p class="acc-note err">' + esc(extracted.error) + '</p>'
      + stepList(extracted.steps || extractSteps)
      + (extracted.raw ? '<pre id="accDiff">' + esc(extracted.raw) + '</pre>' : '');
    return;
  }

  const f = {...(extracted.fields || {}), ...(extracted.normalized || {})};
  /* Which of them the requirement marks Mandatory, from stage 0 -- the live
     event while a run is in flight, the result's own copy for one read back
     afterwards, exactly as the pipeline strip resolves it. Empty on a run that
     predates per-type forms and on a document nothing could place, and empty is
     the honest answer there: no requirement is in play, so nothing is demanded.
     Never derived here from the type -- which keys a requirement demands is
     stated once, in `prompts.MANDATORY_FIELDS`, and a second copy in JavaScript
     would go stale the next time a requirement arrives. */
  const stage0 = classified || extracted.classified || {};
  const req = new Set(stage0.mandatory || []);
  const reqItems = new Set(stage0.mandatory_items || []);
  const g = extracted.grounding || {};
  const st = g.statuses || {};
  const ratio = g.grounded_ratio;
  // The shape that produced these fields, named first. Two extractions of one
  // transcript are not comparable without it -- the same 14 keys mean different
  // things filled by one request and by seven -- and the picker in the sidebar
  // shows the *current* setting, which may already have been switched since.
  // Counts only the tiers this build actually asks for, so a schema that covers
  // one tier does not report two empty ones. This is what is left of the tier
  // bar that used to sit at the foot of the tab.
  const tiers = extracted.tiers || {};
  const filled = ['p1', 'p2', 'p3'].reduce((acc, t) => {
    const present = Number(tiers[t + '_present']), absent = Number(tiers[t + '_absent']);
    if (!Number.isFinite(present) || present + absent === 0) return acc;
    return {present: acc.present + present, total: acc.total + present + absent};
  }, {present: 0, total: 0});
  /* The document type is on the status line rather than in a banner, because
     it is not a finding -- it is the name of the form the values below were
     asked for, and it changes which rows exist. `doc_type_from` rides in the
     tooltip: a type taken off the heading and one taken from the manifest are
     the same answer with different authority behind them, and a run should
     never have to be trusted about a choice it cannot show its working for. */
  const vc = (extracted.validation && extracted.validation.counts) || null;
  /* Every type the page names, not just the one it is filed under: the form
     below is the UNION of their field sets, so naming one of them would leave a
     reader unable to work out why a key is on screen. */
  const docTypes = extracted.doc_types
    || (extracted.doc_type ? [extracted.doc_type] : []);
  /* Over every model request pass 2 made, timed by the server as well; the
     count says how many of them the server could put a figure on. */
  const srvT = (extractedFile || extracted).server_time;
  fldStatus.textContent = `${modeLabel(extracted)} · ${extracted.seconds}s · `
    + `${extracted.tokens} tokens · ${extracted.model || ''}`
    + (srvT ? serverTimeText(srvT, srvT.source)
       + (srvT.server_seconds !== null && srvT.attributed < srvT.requests
          ? ` [${srvT.attributed} of ${srvT.requests} requests timed]` : '') : '')
    + (docTypes.length ? ` · ${docTypes.join(' + ').replace(/_/g, ' ').toLowerCase()}` : '')
    + (filled.total ? ` · ${filled.present}/${filled.total} fields` : '')
    + (ratio === null || ratio === undefined ? ''
       : ` · ${(ratio * 100).toFixed(0)}% grounded`)
    + (vc && vc.scored ? ` · ${vc.ok}/${vc.scored} checks pass` : '')
    /* Mandatory fields that came back empty, counted by `validate` over
       DISTINCT fields rather than over check rows -- one field with two
       rules against it contributes two absent rows. Beside the check
       count and never folded into it: an absent field is neither a pass
       nor a failure of a rule, because the rule never ran on it, and it
       is the one figure on this line that says the document cannot be
       filed as it stands. Absent rather than zero where no requirement
       covers the type: nothing is demanded, so nothing can be missing. */
    + (vc && vc.mandatory_absent
       ? ` · ${vc.mandatory_absent} required missing` : '')
    /* What KIND of table the document rules, where it rules one. On this line
       because it is a fact about the document rather than about a value: a
       master table says the rows below are other documents, each with items of
       its own somewhere else. */
    + (extracted.item_table && Array.isArray(extracted.item_table.rows)
       ? ` · ${extracted.item_table.is_master_table === true ? 'master table'
             : extracted.item_table.is_master_table === false ? 'item table'
             : 'table'} (${extracted.item_table.rows.length} row`
         + `${extracted.item_table.rows.length === 1 ? '' : 's'})` : '');
  /* Which rules a document is held to is a property of its TYPE, not of the
     schema -- an invoice's number is checked for a duplicate and a credit
     note's against the document it credits -- so the check count is unreadable
     without them. A rule that passed and one this document was never held to
     look identical in a bare "13/15". */
  const vrules = (extracted.validation && extracted.validation.rules) || [];
  /* Where the answer came from, and -- since 2026-09-01 -- the line it was
     read off. The type is settled before either prompt is built and both are
     told what it is, so this is not only a label on the form: it is text the
     model was sent. A reader looking at a value in the wrong key has to be able
     to see whether the run was told the wrong thing about the document. */
  const fromWord = {case: 'from the benchmark manifest',
                    transcript: 'matched in Python from the heading on the page'
                                + ' -- the fallback, so the model either was not'
                                + ' asked or quoted a heading the page does not print',
                    model: 'classified by the model, from a heading it quoted back'
                           + ' off the page and Python checked is really printed there',
                    caller: 'given by the request'}[extracted.doc_type_from]
                   || 'not classified';
  fldStatus.title = (docTypes.length
    ? `Field set: the union of ${docTypes.join(' + ')} (${fromWord})`
      + (extracted.doc_type_heading ? `\nHeading read: ${extracted.doc_type_heading}` : '')
      + '\nBoth prompts were told this, and carry the rules that go with it'
    : 'The document type was not identified, so the default field set was asked'
      + ' for and neither prompt was told what kind of document this is')
    + (vrules.length
       ? `\nRules these types add: ${vrules.join(', ').replace(/_/g, ' ')}`
       : '\nNo type-specific validation rules apply')
    + (req.size
       ? `\nMandatory for this type (${req.size}): `
         + [...req].map(k => FIELD_LABELS[k] || k).join(', ')
       : '\nNo requirement covers this type, so no field is Mandatory');

  // Which figures below already carry tax. The labels are built from it rather
  // than fixed, because `subtotal` is only "Amount ex VAT" on a page that
  // prices VAT-exclusive, and calling it that on an inclusive one invites
  // reading a figure that contains ฿672.30 of VAT as if it did not.
  const basis = extracted.vat_basis;
  const label = k => (basis && basis.inclusive && k === 'subtotal'
                        && basis.subtotal_includes_vat) ? 'Amount (VAT included)'
                   : FIELD_LABELS[k];

  const truth = truthIndex(extracted.field_score);
  const val = valIndex(extracted.validation);
  // Which pass lost each value the score charged for, keyed by the same paths.
  const bl = blameFields();
  /* The form this document's TYPE asked for. Falls back to every key when a
     result predates per-type forms, so an old run read back from the log still
     renders in full rather than blank. */
  const asked = new Set(extracted.fields_asked
                        || Object.keys(FIELD_LABELS));

  /* Two banners stay above the fields and everything else moved below them.
     Both of these say the values under them are incomplete for a reason that is
     not the document's -- an extraction that stopped, and a step that was never
     answered -- so an empty field below reads as "the model never got here"
     rather than as "the page is silent", which is the one distinction the
     grounding note is careful to make the other way round. Everything that
     COMMENTS on the fields -- the score, the grounding audit, the tier counts,
     the basis -- now follows them, so the tab opens with what was extracted
     instead of with four paragraphs about it. */
  let html = (extracted.partial
      ? '<p class="ground-note bad"><strong>Partial extraction.</strong> '
        + esc(extracted.partial) + '</p>' : '')
    + stepNote(extracted);

  for (const [title, allKeys, hideWhenEmpty] of FIELD_SECTIONS) {
    // Only the keys this run's form asked for. A key the form omits is not an
    // empty answer, it is a question nobody put -- and drawing it as an empty
    // row would read as a document that says nothing about it.
    const keys = allKeys.filter(k => asked.has(k));
    if (!keys.length) continue;
    // A section of keys the document simply may not have drops when it is empty
    // -- unless the truth file says one of them IS printed, in which case the
    // whole point of this tab is to show that nothing came back for it.
    if (hideWhenEmpty && keys.every(k => !f[k]
        && !(truth.scalars[k] && truth.scalars[k].status === 'missed'))) continue;
    html += '<div class="fld-sec">' + title + '</div>'
      + defList(keys.map(k => [label(k), f[k], MONEY_KEYS.has(k), st[k], k]),
                truth.scalars, val, req, bl, extracted.fields);
  }

  // Filtered exactly as the grounding pass filters it, so the row index in
  // `line_items[i].key` addresses the same row on both sides.
  const items = (Array.isArray(f.line_items) ? f.line_items : [])
    .filter(it => it && typeof it === 'object' && !Array.isArray(it));
  const missedRows = truth.missedRows.filter(Boolean);
  if (items.length || missedRows.length) {
    // Only show columns that at least one row actually fills -- plus any the
    // truth file rules on, because a column every row left empty is exactly the
    // one worth seeing when the page prints it.
    const cols = ITEM_COLS.filter(([k]) =>
      items.some(it => it && it[k]) || truth.keys.has(k));
    // The Amount header says which basis its column is on, because that is
    // where the difference actually lives — every figure under it either
    // carries the tax or does not, and they look identical.
    const colHead = (k, text) => (k === 'amount' && basis)
      ? text + (basis.inclusive ? ' (incl. VAT)' : ' (ex VAT)') : text;
    html += '<div class="fld-sec">Line items (' + items.length
      + (missedRows.length ? ', ' + missedRows.length + ' not returned' : '') + ')</div>'
      + '<div style="overflow-x:auto"><table><tr>'
      + cols.map(([k, label]) => '<th'
          + (ITEM_HINTS[k] ? ' class="has-hint" title="' + esc(ITEM_HINTS[k]) + '"' : '')
          + '>' + esc(colHead(k, label)) + '</th>').join('') + '</tr>'
      + items.map((it, i) => {
          // Which truth row this returned row was paired with -- by content, so
          // a row the extractor dropped does not shift the marks along by one.
          const scored = truth.items[i] || {};
          return '<tr>' + cols.map(([k]) => {
            const v = it && it[k] ? String(it[k]) : '';
            const num = /^[\d.,\- ]+$/.test(v) && v.trim() !== '';
            const s = st['line_items[' + i + '].' + k];
            const bad = s === 'ungrounded' || s === 'computed';
            const sc = scored[k];
            const off = sc && !SCORE_OK.has(sc.status);
            const tip = bad ? GROUND_TAG[s][1]
                      : off ? SCORE_TAG[sc.status][1] + cellScope(sc) : '';
            return '<td class="' + (num ? 'num ' : '') + (bad ? 'ungrounded' : '')
              + (off ? ' sc-' + sc.status : '')
              + blameClass(bl['line_items[' + i + '].' + k]) + '"'
              + (tip ? ' title="' + esc(tip) + '"' : '') + '>' + esc(v)
              + (off && sc.status !== 'spurious'
                  ? '<div class="truth">' + esc(String(sc.expected)) + '</div>' : '')
              + blameMark(bl['line_items[' + i + '].' + k])
              + '</td>';
          }).join('') + '</tr>';
        }).join('')
      // Rows the page prints that the extraction never returned. Drawn from the
      // truth file, dashed, so nothing in them can be mistaken for model output
      // -- and drawn at all because a dropped row is the observed failure on
      // sol005 and leaving it out is how it stayed invisible.
      + missedRows.map(row => '<tr class="row-missed">' + cols.map(([k], ci) =>
          '<td>' + esc(String((row[k] || {}).expected || ''))
          + (ci === 0 ? '<span class="fld-tag missing" title="the ground truth prints '
              + 'this row and the extraction did not return it">not returned</span>' : '')
          + '</td>').join('') + '</tr>').join('')
      + '</table></div>';
  }

  /* The certificate's income table. Drawn only where the form asked for one --
     `items_asked` is the row shape the server chose from the document's type --
     so an invoice never shows an empty one, and a certificate that returned no
     rows shows the heading with a note rather than nothing, because "the table
     did not come back" and "this document has no table" are different failures
     and only the first is worth acting on.

     Cells are marked against the truth exactly as the line-items table above
     is, and for the same reason -- the requirement makes all four Mandatory of
     every row, so they are in the headline and a reader has to be able to see
     which one went wrong. Each also carries its grounding verdict and the
     derived rate. */
  const incomeCols = (extracted.items_asked || [])
    .map(k => INCOME_COLS.find(([c]) => c === k)).filter(Boolean);
  if (incomeCols.length) {
    const rows = (Array.isArray(f.income_items) ? f.income_items : [])
      .filter(r => r && typeof r === 'object' && !Array.isArray(r));
    const rates = ((extracted.derived || {}).wht_rates) || [];
    // Truth rows the extraction never returned. A certificate's table is the
    // reconciliation the two totals below are checked against, so a dropped row
    // is a finding rather than a smaller answer.
    const missedIncome = truth.missedIncome.filter(Boolean);
    html += '<div class="fld-sec">Income rows (' + rows.length
      + (missedIncome.length ? ', ' + missedIncome.length + ' not returned' : '')
      + ')</div>';
    if (!rows.length && !missedIncome.length) {
      html += '<p class="ground-note">No income rows came back. The certificate'
        + ' rules a table of income and this run returned none of it, so the two'
        + ' totals below have nothing to be reconciled against.</p>';
    } else {
      html += '<div style="overflow-x:auto"><table><tr>'
        + incomeCols.map(([k, lab]) => '<th class="has-hint" title="'
            + esc(INCOME_HINTS[k] || '') + '">' + esc(lab)
            // A cell of this table is Mandatory of every ROW, so the mark goes
            // on the heading rather than on each cell -- the same fact said
            // once instead of once per row.
            + (reqItems.has(k) ? '<span class="req" title="Mandatory in the '
                + 'requirement — every row has to state it.">required</span>'
                : '') + '</th>').join('')
        + '<th class="has-hint" title="Worked out here from the two figures on'
        + ' the row, not read off the page — the requirement calls it a derived'
        + ' rate. Blank where either figure could not be read, or where the'
        + ' amount paid is zero.">Rate %</th></tr>'
        + rows.map((r, i) => {
            const rate = (rates[i] || {}).rate;
            // Which truth row this returned row was paired with -- by content,
            // so a dropped row does not shift the marks along by one.
            const scored = truth.income[i] || {};
            return '<tr>' + incomeCols.map(([k]) => {
              const v = r && r[k] ? String(r[k]) : '';
              const num = /^[\d.,\- ]+$/.test(v) && v.trim() !== '';
              const st2 = st['income_items[' + i + '].' + k];
              const bad = st2 === 'ungrounded' || st2 === 'computed';
              const sc = scored[k];
              const off = sc && !SCORE_OK.has(sc.status);
              // Empty in a column the requirement demands, where nothing scores
              // it. Marked in the CELL as well as on the heading: the heading
              // says the column is required of every row, and only the cell can
              // say which row failed it. Where there IS a truth file the score
              // says it better -- `missed`, with the expected value under it --
              // so this stands down rather than saying it twice.
              const gap = !v && reqItems.has(k) && !sc;
              const tip = bad ? GROUND_TAG[st2][1]
                        : off ? SCORE_TAG[sc.status][1] + cellScope(sc)
                        : gap ? 'Mandatory on every row of this table, and this'
                                + ' row left it empty' : '';
              return '<td class="' + (num ? 'num ' : '')
                + (bad ? 'ungrounded' : gap ? 'sc-missed' : '')
                + (off ? ' sc-' + sc.status : '')
                + blameClass(bl['income_items[' + i + '].' + k]) + '"'
                + (tip ? ' title="' + esc(tip) + '"' : '')
                + '>' + (v ? esc(v) : gap ? 'required' : '')
                + (off && sc.status !== 'spurious'
                    ? '<div class="truth">' + esc(String(sc.expected)) + '</div>' : '')
                // The four cells of this table are Mandatory of every row, so
                // they are IN the headline -- which is exactly why a reader has
                // to be able to see which pass lost one.
                + blameMark(bl['income_items[' + i + '].' + k])
                + '</td>';
            }).join('')
            + '<td class="num derived">'
            + (rate === null || rate === undefined ? '—' : esc(String(rate)))
            + '</td></tr>';
          }).join('')
        // Drawn from the truth file, dashed, so nothing in them reads as model
        // output -- the same treatment a dropped charges row gets above.
        + missedIncome.map(row => '<tr class="row-missed">'
            + incomeCols.map(([k], ci) =>
                '<td>' + esc(String((row[k] || {}).expected || ''))
                + (ci === 0 ? '<span class="fld-tag missing" title="the ground '
                    + 'truth prints this row and the extraction did not return '
                    + 'it">not returned</span>' : '') + '</td>').join('')
            + '<td class="num derived">—</td></tr>').join('')
        + '</table></div>';
    }
  }

  // The item table has its own tab now; one line here says it exists.
  if (extracted && 'item_table' in extracted) {
    const it = extracted.item_table;
    html += '<div class="fld-sec">Item table</div><p class="tbl-note">'
      + (it && Array.isArray(it.rows)
         ? it.rows.length + ' row' + (it.rows.length === 1 ? '' : 's') + ', '
           + (it.is_master_table === true ? 'a master table' : it.is_master_table === false
              ? 'an item list' : 'kind not determined')
         : 'none found')
      + ' — see the <a href="#" data-go="table">Table</a> tab for what the read cut, '
      + 'what the agents and the code changed, and the ground truth.</p>';
  }

  // Paired with its original index for the same reason as the line items above.
  const other = (Array.isArray(f.other_fields) ? f.other_fields : [])
    .map((o, i) => [o, i]).filter(([o]) => o && (o.label || o.value));
  if (other.length) {
    html += '<div class="fld-sec">Other fields the model found (' + other.length + ')</div>'
      + defList(other.map(([o, i]) => [o.label || '—', o.value, false,
                                       st['other_fields[' + i + '].value']]));
  }

  // Then the score: how did this run do, over the values just read. Nothing is
  // rendered where there is no truth file, which is most documents.
  const unscored = unscoredReason();
  const score = unscored
    ? '<p class="ground-note bad"><strong>Not scored</strong> — ' + esc(unscored)
      + '. The extraction ran and is logged, and every value above is still '
      + 'marked against the ground truth; what is left out is the rate, here and '
      + 'in the run log, because over a transcript this poor it would be a '
      + 'measurement of the read. Lower <b>Score fields only above</b> to score '
      + 'runs like this one.</p>'
    : fieldScoreNote(extracted.field_score);
  // Under the score and under the not-scored banner alike: which pass lost a
  // value is a question the suppressed rate makes MORE worth answering, not
  // less.
  const who = blameNote();
  if (score || who) html += '<div class="fld-sec">Score</div>' + score + who;

  // There is no "Warnings and findings" block any more, and its absence is the
  // point rather than an omission. Every line of it restated something already
  // marked against the value it was about: `missing` and `not on page` are tags
  // on the field itself, a wrong value already prints the truth underneath it,
  // and the money labels already say which VAT basis they are on. Read down the
  // page it was a second copy of the page. The one figure in it that was not a
  // repeat -- how many keys came back filled -- is now in the status line above,
  // where it costs a phrase instead of a section.
  html += stepList(extracted.steps);

  fldBody.innerHTML = html;
}

/* THE ITEM TABLE (2026-09-30). Read out of the transcript in Python -- no model
   wrote a cell of it -- and classified there too: `is_master_table` is true
   where the rows are other DOCUMENTS (a receipt settling twelve invoices, a
   billing note) and false where they are goods or services.

   Drawn with the document's OWN headings, because that is the whole of what
   fixed it: the old eight-key row made the model smear values across columns
   the page never ruled.

   Three states for the section, and the difference between the last two is the
   one worth drawing: the table, "this document has none", and nothing at all
   where it was not looked for (the key is absent from the result -- the type
   rules a table of its own, or the setting is off). */
function itemTableHtml(x, onTableTab) {
  if (!x || !('item_table' in x)) return '';
  const t = x.item_table;
  if (!t || t.error || !Array.isArray(t.rows)) {
    return '<div class="fld-sec">Item table</div><p class="ground-note">'
      + (t && t.error ? esc(t.error)
         : 'No item table was found in the transcript of this document.')
      + '</p>';
  }
  const cols = t.columns || [];
  const ragged = new Set(t.misaligned || []);
  const recut = new Set(t.realigned || []);
  const kind = t.is_master_table === true ? ['master', 'is_master_table: true']
             : t.is_master_table === false ? ['items', 'is_master_table: false']
             : ['unsure', 'is_master_table: not determined'];
  // Marks against the ground truth, where the case has one. Keyed by the
  // RETURNED row and column, which the server works out: rows are paired by
  // content, and the two tables need not be the same width.
  const marks = {}, missed = {};
  const sc = x.table_score && Array.isArray(x.table_score.rows) ? x.table_score : null;
  (sc ? sc.rows : []).forEach(r => {
    const m = /^item_table\[(\d+)\]/.exec(r.path || '');
    if (!m) return;
    if (r.row === null || r.row === undefined) {
      (missed[m[1]] = missed[m[1]] || []).push(r);
    } else if (r.column !== null && r.column !== undefined) {
      marks[r.row + ':' + r.column] = r;
    }
  });
  const missedRows = Object.keys(missed).map(k => missed[k]);
  const width = Math.max(cols.length, ...t.rows.map(r => r.length), 1);
  const heads = [];
  for (let i = 0; i < width; i++) heads.push(cols[i] || '');

  let html = '<div class="fld-sec">Item table (' + t.rows.length + ' row'
    + (t.rows.length === 1 ? '' : 's')
    + (missedRows.length ? ', ' + missedRows.length + ' not returned' : '') + ')'
    + '<span class="tbl-kind ' + kind[0] + '" title="' + esc(t.master_why || '')
    + '">' + kind[1] + '</span></div>'
    // Why, in the words of the code that decided it. One sentence and not two:
    // the chip has already said which answer it is.
    + '<p class="tbl-note">' + esc((t.master_why || '').replace(/^./, c => c.toUpperCase()))
    + '.</p>'
    + '<div style="overflow-x:auto"><table><tr>'
    + heads.map(h => '<th>' + (h ? esc(h) : '<span style="color:var(--muted)">—</span>')
        + '</th>').join('') + '</tr>'
    + t.rows.map((row, i) => {
        const cls = ragged.has(i) ? 'row-ragged' : recut.has(i) ? 'row-recut' : '';
        const why = ragged.has(i)
          ? 'The read cut this row into the wrong number of cells and nothing '
            + 'could settle where the cuts belong, so it is shown as read.'
          : recut.has(i)
          ? 'The read cut this row into the wrong number of cells. It has been '
            + 'put back under the headings; every character is the read’s own.'
          : '';
        const cells = [];
        for (let c = 0; c < Math.max(width, row.length); c++) {
          const v = row[c] || '';
          const num = /^[\d.,\-() ]+$/.test(v) && v.trim() !== '';
          const mark = marks[i + ':' + c];
          const off = mark && !SCORE_OK.has(mark.status);
          cells.push('<td class="' + (num ? 'num' : '')
            + (off ? ' sc-' + mark.status : '') + '"'
            + (off ? ' title="' + esc(SCORE_TAG[mark.status][1]) + '"' : '') + '>'
            + esc(v)
            + (off && mark.status !== 'spurious'
                ? '<div class="truth">' + esc(String(mark.expected)) + '</div>' : '')
            + '</td>');
        }
        return '<tr' + (cls ? ' class="' + cls + '" title="' + esc(why) + '"' : '')
          + '>' + cells.join('') + '</tr>';
      }).join('')
    // Rows the page prints and the read did not return, from the ground truth.
    // Dashed, so nothing in them can be taken for the read.
    + missedRows.map(cellsOf => '<tr class="row-missed">' + heads.map((h, c) => {
        const hit = cellsOf.find(r => r.column === c
          || (r.column === null && r.heading === h));
        return '<td>' + esc(String((hit || {}).expected || ''))
          + (c === 0 ? '<span class="fld-tag missing" title="the ground truth '
              + 'prints this row and the read did not return it">not returned'
              + '</span>' : '') + '</td>';
      }).join('') + '</tr>').join('')
    + '</table></div>';

  // What was done to the table as read, and what was taken out of it. Said,
  // because a table that quietly lost six rows reads as a short table.
  const notes = [];
  if (t.pages && t.pages.length) notes.push('page' + (t.pages.length === 1 ? ' ' : 's ')
    + t.pages.join(', '));
  notes.push('read from the ' + esc(t.source || 'transcript'));
  (t.repairs || []).forEach(r => notes.push(esc(r)));
  if (ragged.size) notes.push('<span style="color:var(--warn)">' + ragged.size + ' row'
    + (ragged.size === 1 ? '' : 's') + ' still cut wrongly, shown as read</span>');
  html += '<p class="tbl-note">' + notes.join(' · ') + '</p>';
  html += tableAgentsHtml(t, onTableTab);
  if (sc && sc.expected) {
    const m = sc.master || {};
    html += '<p class="tbl-note">Against the ground truth: <b>'
      + (sc.counts.correct || 0) + ' of ' + sc.expected + '</b> cells correct, '
      + sc.rows_matched + ' of ' + sc.rows_expected + ' rows matched'
      + (m.status && m.status !== 'unchecked'
          ? '; <code>is_master_table</code> <b>' + esc(m.status) + '</b> (the truth '
            + 'file says ' + esc(String(m.expected)) + ')' : '')
      + '. Not part of the field score above: no requirement makes a row of this '
      + 'table Mandatory, and a wrong cell here is the read, not the extraction.</p>';
  }
  if ((t.dropped || []).length) {
    html += '<details class="tbl-note"><summary>' + t.dropped.length + ' row'
      + (t.dropped.length === 1 ? '' : 's') + ' of the table that '
      + (t.dropped.length === 1 ? 'is not an item' : 'are not items')
      + ' — totals, section headings, notes</summary><table>'
      + t.dropped.map(d => '<tr><td style="color:var(--muted)">' + esc(d.why) + '</td>'
          + (d.cells || []).map(c => '<td>' + esc(c) + '</td>').join('') + '</tr>')
        .join('') + '</table></details>';
  }
  return html;
}

/* What the two table agents did. Each was sent the table and nothing else, and
   the code checked each answer, so the line says what the agent said AND what
   the code took -- an agent that was overruled reads differently from one that
   was believed. Silent where neither ran. */
function tableAgentsHtml(t, compact) {
  const f = t.fix_agent, d = t.identify_agent;
  if (!f && !d) return '';
  const rowList = a => a.map(i => i + 1).join(', ');
  const lines = [];
  if (f) {
    let s = '<b>Fix agent</b>' + (f.model ? ' (' + esc(f.model) + ')' : '') + ': ';
    const moved = f.removed || [], refused = (f.refusals || []).slice();
    if (f.skipped && !f.asked) s += esc(f.skipped) + ' — the cells were settled by code alone';
    else if (f.error) s += 'no usable answer (' + esc(f.error) + ') — nothing removed';
    else {
      const ln = f.lines || {};
      const ls = (ln.stray || []).length + (ln.stray_before || []).length;
      const lo = (ln.own || []).length + (ln.own_before || []).length;
      const said = [];
      if (ls) said.push(ls + ' line' + (ls === 1 ? '' : 's') + ' around the table stray');
      if (lo) said.push(lo + ' line' + (lo === 1 ? '' : 's') + ' outside the rows the table’s own');
      if (moved.length) said.push(moved.length + ' piece' + (moved.length === 1 ? '' : 's')
        + ' of stray text taken out of cells');
      s += said.length ? said.join('; ') : 'found nothing in or around the table that does not belong';
      ((ln.refusals) || []).forEach(r => refused.push({ entry: { text: r.text }, why: r.why }));
    }
    if ((f.rows_emptied || []).length) s += '; ' + f.rows_emptied.length + ' row'
      + (f.rows_emptied.length === 1 ? ' was' : 's were') + ' nothing but that text and left the table';
    if (refused.length) s += ' · <span style="color:var(--warn)">' + refused.length
      + ' not taken: ' + refused.map(r => '“' + esc(String((r.entry || {}).text || ''))
        + '” (' + esc(r.why) + ')').join('; ') + '</span>';
    lines.push(s + '.');
  }
  if (d) {
    let s = '<b>Identify agent</b>' + (d.model ? ' (' + esc(d.model) + ')' : '') + ': ';
    if (d.error) s += 'no answer (' + esc(d.error) + ')';
    else if (d.answer === null) s += 'no usable answer';
    else s += 'said <code>' + d.answer + '</code>'
      + (d.column ? ', naming column ' + d.column : '');
    if (d.refused) s += ' — <span style="color:var(--warn)">not taken: '
      + esc(d.refused) + '</span>';
    s += '. The answer above is the ' + (t.master_from === 'model' ? 'agent'
      : 'Python rule') + '’s';
    if (t.master_from === 'model' && t.master_rules !== t.is_master_table) {
      s += '; the rule says <code>' + (t.master_rules === null ? 'not determined'
        : t.master_rules) + '</code> (' + esc(t.master_rules_why || '') + ')';
    }
    lines.push(s + '.');
  }
  const asks = [];
  if (f && f.prompt) asks.push(['fix', f.prompt, f.raw]);
  if (t.recut && t.recut.prompt) asks.push(['re-cut', t.recut.prompt, t.recut.raw]);
  if (d && d.prompt) asks.push(['identify', d.prompt, d.raw]);
  // The text the fix agent took out, as plain text: it was printed on the
  // page, so it is kept and shown -- where it was read from, and nothing more.
  const out = t.outside_text || [], foot = t.footer || [];
  const pre = a => '<pre style="white-space:pre-wrap;margin:4px 0">'
    + a.map(o => esc(o.text)).join(String.fromCharCode(10)) + '</pre>';
  const where = o => (o.where ? esc(o.where) : 'row ' + (o.row + 1)
    + (o.heading ? ', ' + esc(o.heading) : '')) + (o.page ? ', page ' + o.page : '');
  // The table's own lines the read had put after it: its totals, a note.
  const footer = foot.length ? '<div class="tbl-note"><b>Part of the table, read '
    + 'outside its rows:</b>' + pre(foot) + '<span style="color:var(--muted)">'
    + foot.map(where).join(' · ') + '</span></div>' : '';
  const plain = out.length ? '<div class="tbl-note"><b>Not part of the table — a stamp, '
    + 'a barcode, handwriting — kept as plain text:</b>' + pre(out)
    + '<span style="color:var(--muted)">' + out.map(where).join(' · ') + '</span></div>' : '';
  return (compact ? '' : '<p class="tbl-note">' + lines.join('<br>') + '</p>') + footer + plain
    + (asks.length ? '<details class="tbl-note"><summary>What the agents were sent, '
      + 'and what they answered</summary>' + asks.map(a => '<div><b>' + a[0]
        + '</b><pre style="white-space:pre-wrap;max-height:16em;overflow:auto;font-size:12px">' + esc(a[1]) + '</pre><pre style="white-space:pre-wrap;max-height:16em;overflow:auto;font-size:12px">'
        + esc(a[2] || '(no reply)') + '</pre></div>').join('') + '</details>' : '');
}

/* ---------- the Table tab --------------------------------------------------
   The item table step by step. The server snapshots it after every step
   (`item_table.stages`) and says who took each one: the read, the fix agent's
   cleaning, the code's re-cut, the model's re-cut. A cell is marked in the
   colour of the step that changed it, with what it was before, so "what did the
   AI do" and "what did the code do" are two colours rather than one paragraph.
   The ground truth is `table_score.truth_table` -- the same parser run over
   `solution/<id>.md`, so it is in exactly the shape a perfect read would be. */
const TV_STAGE = {
  read:    ['As read', 'read', 'The rows exactly as the OCR read cut them -- after the '
            + 'structural repairs only (spans expanded, a two-row heading merged, '
            + 'totals and notes taken out of the items).'],
  concat:  ['Pieces joined', 'agent', 'The read had cut this table into pieces -- its '
            + 'totals, or rows continued under other headings, came out as tables of '
            + 'their own. A piece holding one of the table’s column totals is joined by the code; '
            + 'for the rest the concat agent named which belong to it, and the code '
            + 'joined them, placing every cell by what it is; a totals line is kept as a '
            + 'totals line, not as an item.'],
  clean:   ['AI agent cleaned', 'agent', 'The fix agent named text in a cell that is '
            + 'not table data -- a stamp, a barcode, handwriting -- and the code took it '
            + 'out where it found it in that cell and was allowed to. Nothing it took '
            + 'out is dropped: it is listed under the final table as plain text.'],
  realign: ['Code re-cut', 'code', 'Python put rows cut into the wrong cells back '
            + 'under their headings, by what each cell is -- an amount under a money '
            + 'heading, a number under a reference. No character changed.'],
  recut:   ['AI model re-cut', 'agent', 'For the rows Python could not settle, the '
            + 'model said which column each cell belongs under; the code checked the '
            + 'answer against the cells and did the cutting itself, so no character '
            + 'of the reply reaches the table.'],
};
function hasItemTable(r) {
  if (!r) return false;
  if (r.pending) return true;
  return ('item_table' in r) || (r.documents || []).some(d => d && 'item_table' in d);
}
function paintTableTab() {
  const on = hasItemTable(extractedFile) || !!tableLive;
  tabTable.disabled = !on;
  if (!on && curMode === 'table') showTab('raw');
}
/* Which cells changed between two snapshots. Rows are compared by position --
   every step keeps the row order -- except where the agent emptied a row and it
   left the table, which `removed_rows` says. A row whose column count changed
   was re-cut whole, so every cell of it is marked and the old row is its title. */
function tvDiff(prev, cur) {
  const out = {};
  if (!prev || !cur) return out;
  let prows = prev.rows;
  const gone = new Set(cur.removed_rows || []);
  if (gone.size) prows = prows.filter((_, i) => !gone.has(i));
  const sameCols = prev.columns.length === cur.columns.length;
  cur.rows.forEach((row, i) => {
    const was = prows[i] || [];
    if (JSON.stringify(was) === JSON.stringify(row)) return;
    const cells = new Set();
    const whole = !sameCols || was.length !== row.length;
    row.forEach((v, c) => { if (whole || (v || '') !== (was[c] || '')) cells.add(c); });
    out[i] = { cells, was, whole };
  });
  return out;
}
function tvCount(diff) {
  return Object.values(diff).reduce((a, d) => a + d.cells.size, 0);
}
function tvTable(columns, rows, opts) {
  opts = opts || {};
  const diff = opts.diff || {}, by = opts.by || 'read';
  const bad = new Set(opts.misaligned || []), missed = new Set(opts.missed || []);
  const width = Math.max(columns.length, ...rows.map(r => r.length), 1);
  let h = '<div style="overflow-x:auto"><table><tr>';
  for (let c = 0; c < width; c++) {
    h += '<th>' + (columns[c] ? esc(columns[c]) : '<span style="color:var(--muted)">—</span>') + '</th>';
  }
  h += '</tr>';
  rows.forEach((row, i) => {
    const d = diff[i];
    const cls = [bad.has(i) ? 'row-ragged' : '', missed.has(i) ? 'tv-missed' : ''].join(' ').trim();
    const title = d && d.whole ? 'before: ' + d.was.map(x => x || '·').join(' | ')
      : missed.has(i) ? 'the read did not return this row'
      : bad.has(i) ? 'still cut into the wrong number of cells' : '';
    h += '<tr' + (cls ? ' class="' + cls + '"' : '') + (title ? ' title="' + esc(title) + '"' : '') + '>';
    for (let c = 0; c < Math.max(width, row.length); c++) {
      const v = row[c] || '';
      const ch = d && d.cells.has(c);
      const num = /^[\d.,\-() ]+$/.test(v) && v.trim() !== '';
      h += '<td class="' + (num ? 'num' : '') + (ch ? ' tv-ch' : '') + '"'
        + (ch ? ' style="--by:var(--by-' + by + ')"' : '') + '>' + esc(v)
        + (ch && !d.whole && (d.was[c] || '') !== v
           ? '<span class="was">' + (d.was[c] ? esc(d.was[c]) : 'empty') + '</span>' : '')
        + '</td>';
    }
    h += '</tr>';
    /* A row re-cut whole has no cell-to-cell "before"; the row as it was is
       drawn under it, struck, so the move is visible without hovering. */
    if (d && d.whole && d.was.length) {
      h += '<tr class="tv-before"><td colspan="' + Math.max(width, row.length)
        + '"><span>before</span> ' + d.was.map(x => esc(x || '·')).join(' <b>|</b> ')
        + '</td></tr>';
    }
  });
  (opts.gone || []).forEach(row => {
    h += '<tr class="tv-gone" title="taken out of the table by the fix agent: nothing '
      + 'in it was table data">' + row.map(v => '<td>' + esc(v || '') + '</td>').join('')
      + '</tr>';
  });
  return h + '</table></div>';
}
/* Who did what, one card per hand. Counts and the pieces themselves; the
   tables below show where. */
function tvCards(t, stages, sc) {
  const f = t.fix_agent, d = t.identify_agent, rc = t.recut, cc = t.concat;
  const li = a => a.length ? '<ul>' + a.map(x => '<li>' + x + '</li>').join('') + '</ul>'
                           : '<p class="none">nothing</p>';
  const q = x => '<q>' + esc(String(x || '')) + '</q>';
  const ai = [], code = [], over = [];
  if (cc) {
    const head = 'Concat agent' + (cc.model ? ' (' + esc(cc.model) + ')' : '')
      + (cc.seconds ? ', ' + cc.seconds + 's' : '');
    if (cc.skipped && !cc.asked) ai.push(head + ': not asked — ' + esc(cc.skipped));
    else if (cc.error) ai.push(head + ': no usable answer — ' + esc(cc.error));
    else ai.push(head + ': ' + (cc.joined || []).length + ' of ' + (cc.offered || 0)
      + ' other table' + (cc.offered === 1 ? '' : 's') + ' joined'
      + ((cc.joined || []).length ? ' (' + (cc.rows_added || 0) + ' row'
         + (cc.rows_added === 1 ? '' : 's') + ', ' + (cc.dropped_added || 0)
         + ' totals or notes)' : ''));
    (cc.refusals || []).forEach(r => over.push('refused joining table T' + r.number
      + ' — ' + esc(r.why)));
  }
  if (f) {
    const head = 'Fix agent' + (f.model ? ' (' + esc(f.model) + ')' : '')
      + (f.seconds ? ', ' + f.seconds + 's' : '');
    if (f.skipped && !f.asked) ai.push(head + ': not asked — ' + esc(f.skipped));
    else if (f.error) ai.push(head + ': no usable answer — ' + esc(f.error));
    else {
      const ln = f.lines || {};
      const moved = f.removed || [];
      const n = k => (ln[k] || []).length;
      const lines = (count, side, what) => ai.push(count + ' line' + (count === 1 ? '' : 's')
        + ' ' + side + ' the table called ' + what);
      ai.push(head + ': ' + (moved.length || n('stray') || n('own') || n('stray_before')
        || n('own_before') ? 'see below' : 'found nothing that does not belong'));
      moved.forEach(m => ai.push('took ' + q(m.text) + ' out of row ' + (m.row + 1)
        + (m.heading ? ', ' + esc(m.heading) : '')));
      if (n('stray_before')) lines(n('stray_before'), 'before', 'stray (stamp, barcode, handwriting)');
      if (n('own_before')) lines(n('own_before'), 'before', 'the table’s own (headings or a row it cut off)');
      if (n('stray')) lines(n('stray'), 'after', 'stray (stamp, barcode, handwriting)');
      if (n('own')) lines(n('own'), 'after', 'the table’s own (totals, a note)');
      (f.refusals || []).forEach(r => over.push('refused the fix agent’s '
        + q((r.entry || {}).text) + ' — ' + esc(r.why)));
      ((ln.refusals) || []).forEach(r => over.push('refused moving line ' + q(r.text)
        + ' — ' + esc(r.why)));
    }
  } else {
    ai.push('Fix agent: off');
  }
  if (rc && (rc.asked || rc.skipped || rc.error)) {
    if (rc.skipped) ai.push('Model re-cut: not asked — ' + esc(rc.skipped));
    else if (rc.error) ai.push('Model re-cut: no usable answer — ' + esc(rc.error));
    else ai.push('Model re-cut' + (rc.model ? ' (' + esc(rc.model) + ')' : '') + ': asked about '
      + rc.asked + ' row' + (rc.asked === 1 ? '' : 's') + ', ' + rc.taken + ' taken');
    (rc.refusals || []).forEach(r => over.push('refused the model’s cut of row '
      + (r.row + 1) + ' — ' + esc(r.why)));
  }
  if (d) {
    let s = 'Identify agent' + (d.model ? ' (' + esc(d.model) + ')' : '') + ': ';
    if (d.error) s += 'no answer — ' + esc(d.error);
    else if (d.answer === null || d.answer === undefined) s += 'no usable answer';
    else s += 'said <code>is_master_table: ' + d.answer + '</code>'
      + (d.column ? ', column ' + d.column : '') + (d.taken ? ' — taken' : '');
    ai.push(s);
    if (d.refused) over.push('refused the identify agent — ' + esc(d.refused));
  } else {
    ai.push('Identify agent: off');
  }
  // The code's own repairs, in its own words. The ones the agents caused are
  // already in the AI card.
  // `sort_tail` and `remove_text` write their own repair lines, and those are
  // the fix agent's work -- already in the AI card.
  (t.repairs || []).filter(r => !/model said|did not belong|plain text|kept with it/.test(r))
    .forEach(r => code.push(esc(r)));
  const realign = stages.find(x => x.stage === 'realign');
  const prev = realign ? stages[stages.indexOf(realign) - 1] : null;
  if (realign && prev) {
    const n = Object.keys(tvDiff(prev, realign)).length;
    if (n && !code.some(x => /put back under the headings/.test(x))) {
      code.push(n + ' row' + (n === 1 ? '' : 's') + ' re-cut under the headings');
    }
  }
  code.push(...over.map(x => '<span style="color:var(--warn)">' + x + '</span>'));
  (t.misaligned || []).length && code.push('<span style="color:var(--warn)">'
    + t.misaligned.length + ' row' + (t.misaligned.length === 1 ? '' : 's')
    + ' still cut wrongly — shown as read</span>');
  let res = [];
  if (sc && sc.expected) {
    res.push('<b>' + (sc.counts.correct || 0) + ' of ' + sc.expected + '</b> cells correct');
    res.push(sc.rows_matched + ' of ' + sc.rows_expected + ' rows matched');
    const m = sc.master || {};
    if (m.status && m.status !== 'unchecked') res.push('<code>is_master_table</code> '
      + esc(m.status) + ' (sol says ' + esc(String(m.expected)) + ')');
  } else if (sc && sc.truth_table === undefined && sc.expected === 0) {
    res.push('sol rules no item table on this document');
  } else {
    res.push('<span class="none">no ground truth for this document</span>');
  }
  return '<div class="tv-who">'
    + '<div style="--by:var(--by-agent)"><h4>AI agents</h4>' + li(ai) + '</div>'
    + '<div style="--by:var(--by-code)"><h4>Code</h4>' + li(code) + '</div>'
    + '<div style="--by:var(--by-truth)"><h4>Against sol</h4>' + li(res) + '</div>'
    + '</div>';
}
/* The fix agent's own score (`fix_score`, from fixscore.py against
   solution/tables/<id>.md). Drawn apart from the table score on purpose: it says
   whether the agent's decisions were right -- what it moved out, what it kept --
   and never feeds the cells-correct figure above it. */
function fixScoreHtml(fs) {
  if (!fs) return '';
  if (fs.error) return '<div class="fld-sec">Fix agent score</div><p class="tbl-note">'
    + esc(fs.error) + '</p>';
  const st = fs.strays || {}, rm = fs.removals || {}, own = fs.own_lines || {};
  const cells = fs.cells, agent = fs.agent || {};
  const pct = v => v === null || v === undefined ? '–' : v.toFixed(1) + '%';
  const sign = v => v === null || v === undefined ? '–' : (v > 0 ? '+' : '') + v;
  const lines = [], bad = [];
  if (!agent.asked) lines.push('<span class="none">The fix agent was not asked'
    + (agent.skipped ? ' — ' + esc(agent.skipped) : '') + '; everything below is what '
    + 'was left as read.</span>');
  lines.push('<b>Strays found: ' + pct(st.recall) + '</b> — moved ' + (st.moved || 0)
    + ' of ' + (st.scored || 0) + ' stamp, barcode or handwriting lines it was shown'
    + (st.left_in_table ? '; <span style="color:var(--danger)">' + st.left_in_table
       + ' still in the table</span>' : '')
    + (st.missed ? '; ' + st.missed + ' left as page text' : '')
    + (st.not_read ? ' · ' + st.not_read + ' the read never produced (not counted)' : '')
    + (st.not_shown ? ' · ' + st.not_shown + ' outside the lines the agent was shown (not counted)' : '')
    + (st.marks ? ' · ' + st.marks + ' mark' + (st.marks === 1 ? '' : 's') + ' with no text' : ''));
  lines.push('<b>Removals right: ' + pct(rm.precision) + '</b> — ' + (rm.stray || 0) + ' of '
    + (rm.total || 0) + ' moved out were stray'
    + (rm.page_text ? '; ' + rm.page_text + ' were ordinary page text' : '')
    + (rm.real_value ? '; <span style="color:var(--danger)">' + rm.real_value
       + ' were values the clean table holds</span>' : ''));
  if (own.total) lines.push('<b>Table’s own lines:</b> ' + own.table + ' of ' + own.total
    + ' named the table’s are in the clean table');
  const tot = fs.totals || {}, cc = fs.concat || {};
  if (tot.expected) lines.push('<b>Totals held:</b> ' + (tot.in_table || 0) + ' of ' + tot.expected
    + ' of the table’s totals rows are part of the returned table');
  if (cc.offered) lines.push('<b>Joined:</b> ' + (cc.joined || 0) + ' of ' + cc.offered
    + ' other table' + (cc.offered === 1 ? '' : 's') + ' the read made'
    + (cc.by_sum ? ', ' + cc.by_sum + ' by a column total' : '')
    + (cc.joined ? ' (' + (cc.rows_added || 0) + ' rows, ' + (cc.dropped_added || 0)
       + ' totals or notes)' : ''));
  (tot.missing || []).forEach(m => bad.push('<li>totals row not in the table: <q>'
    + esc(m) + '</q></li>'));
  if (cells) lines.push('<b>Cells:</b> ' + cells.read + ' correct as read → '
    + (cells.after_agent === null ? '–' : cells.after_agent) + ' after the agent → '
    + cells.final + ' final, of ' + cells.expected + ' (agent ' + sign(cells.agent_delta)
    + ', all repairs ' + sign(cells.final_delta) + ')');
  (st.items || []).forEach(i => {
    if (i.status === 'left_in_table' || i.status === 'missed') bad.push(
      '<li>' + (i.status === 'left_in_table' ? 'still in the table' : 'not moved')
      + ' — ' + esc(i.kind) + ': <q>' + esc(i.text) + '</q></li>');
  });
  (rm.items || []).forEach(i => {
    if (i.status !== 'stray') bad.push('<li>' + (i.status === 'real_value'
      ? '<span style="color:var(--danger)">a real value taken out</span>'
      : 'page text called stray') + ': <q>' + esc(i.text) + '</q>'
      + (i.where ? ' (' + esc(i.where) + ')' : '') + '</li>');
  });
  (own.items || []).forEach(i => {
    if (i.status !== 'table') bad.push('<li>named the table’s own, and is not: <q>'
      + esc(i.text) + '</q></li>');
  });
  return '<div class="fld-sec" title="Scored against solution/tables. Separate from '
    + 'the table score above: this measures the fix agent’s decisions only.">'
    + 'Fix agent score <span class="tbl-note" style="margin:0">— separate from the '
    + 'table score</span></div><p class="tbl-note">' + lines.join('<br>') + '</p>'
    + (bad.length ? '<ul class="tbl-note">' + bad.join('') + '</ul>' : '');
}
function tvSwitches() {
  const a = TABLE_AGENTS || {};
  return '<span class="tv-sw" title="Whether each agent runs on the NEXT extraction. '
    + 'Off is the code alone: the table is still read and re-cut by Python, and '
    + 'the master-table answer is the Python rule.">'
    + '<label><input type="checkbox" data-agent="concat"' + (a.concat ? ' checked' : '')
    + '> concat agent</label>'
    + '<label><input type="checkbox" data-agent="fix"' + (a.fix ? ' checked' : '')
    + '> fix agent</label><label><input type="checkbox" data-agent="identify"'
    + (a.identify ? ' checked' : '') + '> identify agent</label>'
    + '<span>(next run)</span></span>';
}
function paintTableView() {
  if (curMode !== 'table') return;
  const strip = docStrip();
  if (!extracted || extracted.pending) {
    const L = tableLive || {};
    const line = (k, name) => {
      const e = L[k];
      if (!e) return '<li class="none">' + name + ': waiting</li>';
      if (e.status === 'running') return '<li><span class="spin"></span> ' + name
        + ' running' + (e.model ? ' · ' + esc(e.model) : '') + '…</li>';
      if (e.status === 'off') return '<li>' + name + ': off</li>';
      if (e.status === 'skipped') return '<li>' + name + ': nothing to join — the read '
        + 'made no other table after this one</li>';
      if (e.status === 'none') return '<li>' + name + ': no item table in this document</li>';
      return '<li>' + name + ': done' + (e.seconds ? ' in ' + e.seconds + 's' : '')
        + (k === 'fix' && e.asked ? ' · ' + (e.removed || 0) + ' taken out of cells, '
           + (e.stray_lines || 0) + ' stray lines' : '')
        + (k === 'concat' && e.asked ? ' · ' + (e.joined || 0) + ' of ' + (e.offered || 0)
           + ' other tables joined' : '')
        + (k === 'identify' && e.is_master_table !== undefined
           ? ' · is_master_table: ' + e.is_master_table + ' (' + esc(e.master_from || '') + ')' : '')
        + (e.refused ? ' · <span style="color:var(--warn)">refused: ' + esc(e.refused) + '</span>' : '')
        + '</li>';
    };
    tableViewEl.innerHTML = '<div class="tv-head"><h3>Item table</h3><span class="spacer"></span>'
      + tvSwitches() + '</div>'
      + (tableLive ? '<ul class="tbl-note">' + line('read', 'Reading the table')
           + line('concat', 'Concat agent') + line('fix', 'Fix agent') + line('identify', 'Identify agent') + '</ul>'
         : '<p class="tbl-note">Waiting for the extraction to reach the table…</p>');
    return;
  }
  const x = extracted;
  if (!('item_table' in x)) {
    tableViewEl.innerHTML = strip + '<div class="tv-head"><h3>Item table</h3>'
      + '<span class="spacer"></span>' + tvSwitches() + '</div><p class="tbl-note">'
      + 'This document was not looked at for an item table — its type rules a table '
      + 'of its own (a withholding certificate’s income rows are on the Fields tab), '
      + 'or table extraction is off.</p>';
    return;
  }
  const t = x.item_table;
  const sc = x.table_score && !x.table_score.error ? x.table_score : null;
  const truth = sc && sc.truth_table ? sc.truth_table : null;
  if (!t || t.error || !Array.isArray(t.rows)) {
    tableViewEl.innerHTML = strip + '<div class="tv-head"><h3>Item table</h3><span class="spacer"></span>'
      + tvSwitches() + '</div><p class="tbl-note">' + (t && t.error ? esc(t.error)
        : 'No item table was found in the transcript of this document.') + '</p>'
      + (truth ? '<div class="fld-sec">What sol says it should be</div>'
         + tvTable(truth.columns, truth.rows) : '');
    return;
  }
  const stages = (t.stages || []).filter(st => TV_STAGE[st.stage]);
  const kind = t.is_master_table === true ? ['master', 'is_master_table: true']
             : t.is_master_table === false ? ['items', 'is_master_table: false']
             : ['unsure', 'is_master_table: not determined'];
  let html = strip + '<div class="tv-head"><h3>Item table · ' + t.rows.length + ' row'
    + (t.rows.length === 1 ? '' : 's') + '</h3><span class="tbl-kind ' + kind[0]
    + '" title="' + esc(t.master_why || '') + '">' + kind[1] + '</span>'
    + '<span class="tbl-note" style="margin:0">answered by the '
    + (t.master_from === 'model' ? 'identify agent' : 'Python rule') + '</span>'
    + '<span class="spacer"></span>' + tvSwitches() + '</div>';
  html += tvCards(t, stages, sc) + fixScoreHtml(x.fix_score);

  // The steps, each with how many cells it changed.
  const views = [];
  stages.forEach((st, i) => {
    const prev = i ? stages[i - 1] : null;
    const diff = tvDiff(prev, st);
    const n = tvCount(diff), gone = (st.removed_rows || []).length;
    const meta = TV_STAGE[st.stage];
    const notAsked = st.stage === 'recut' && !st.asked;
    const label = i === 0 ? st.rows.length + ' rows' + (st.misaligned.length
        ? ', ' + st.misaligned.length + ' cut wrongly' : '')
      : notAsked ? 'not asked' : (n || gone ? n + ' cell' + (n === 1 ? '' : 's') + ' changed'
        + (gone ? ', ' + gone + ' row' + (gone === 1 ? '' : 's') + ' out' : '') : 'no change');
    const joinedNote = st.stage === 'concat' && t.concat && label === 'no change'
      && t.concat.dropped_added ? t.concat.dropped_added + ' totals or notes joined' : '';
    views.push({ key: st.stage, title: (i + 1) + ' ' + meta[0], by: st.by || meta[1],
                 label: joinedNote || label,
                 changed: n + gone, st, prev, diff });
  });
  if (truth) {
    views.push({ key: 'truth', title: 'Ground truth (sol)', by: 'truth',
                 label: truth.rows.length + ' rows · ' + truth.columns.length + ' columns' });
    views.push({ key: 'side', title: 'Final vs sol', by: 'truth', label: 'side by side' });
  }
  let pick = tvStage && views.find(v => v.key === tvStage);
  if (!pick) pick = views.find(v => v.key === 'side')
    || views.slice().reverse().find(v => v.changed) || views[views.length - 1];
  if (views.length) {
    html += '<div class="fld-sec">Step by step</div><div class="tv-steps">'
      + views.map((v, i) => (i && v.key === 'truth' ? '<span class="arrow">|</span>'
          : i && v.key !== 'side' ? '<span class="arrow">›</span>' : '')
          + '<button type="button" data-tv="' + v.key + '" style="--by:var(--by-' + v.by + ')"'
          + (v === pick ? ' class="on"' : '') + '>' + esc(v.title) + '<small>' + esc(v.label)
          + '</small></button>').join('') + '</div>';
    if (pick.st) {
      const st = pick.st;
      const goneRows = pick.prev && (st.removed_rows || []).length
        ? st.removed_rows.map(i => pick.prev.rows[i] || []) : [];
      html += '<p class="tv-what">' + esc(TV_STAGE[st.stage][2])
        + (pick.prev ? ' Marked cells are what this step changed, with the cell before '
           + 'it struck through underneath.' : '') + '</p>'
        + tvTable(st.columns, st.rows, { diff: pick.diff, by: pick.by,
                                          misaligned: st.misaligned, gone: goneRows });
    } else if (pick.key === 'truth') {
      html += '<p class="tv-what">The item table of <b>solution/' + esc(sc.case || '')
        + '.md</b>, read by the same code that reads a model’s transcript — so this is '
        + 'exactly what a perfect read would come back as. sol says <code>is_master_table: '
        + esc(String((sc.master || {}).expected)) + '</code>.</p>'
        + tvTable(truth.columns, truth.rows, { missed: tvMissedTruth(sc) });
    } else {
      const last = stages[stages.length - 1] || { columns: t.columns, rows: t.rows };
      html += '<p class="tv-what">The table this run returned beside the one sol says '
        + 'the page prints. Dashed rows in sol are rows the read did not return.</p>'
        + '<div class="tv-side"><div><h4 style="--by:var(--by-agent);color:var(--text)">Final</h4>'
        + tvTable(t.columns, t.rows, { misaligned: t.misaligned }) + '</div>'
        + '<div><h4 style="--by:var(--by-truth)">sol</h4>'
        + tvTable(truth.columns, truth.rows, { missed: tvMissedTruth(sc) }) + '</div></div>';
    }
  }
  html += '<div class="fld-sec">Final table, marked against sol</div>'
    + itemTableHtml(x, true).replace(/^<div class="fld-sec">[\s\S]*?<\/div>/, '');
  tableViewEl.innerHTML = html;
}
/* The truth rows the read never returned, by their index in the truth table. */
function tvMissedTruth(sc) {
  const out = new Set();
  ((sc && sc.rows) || []).forEach(r => {
    const m = /^item_table\[(\d+)\]/.exec(r.path || '');
    if (m && (r.row === null || r.row === undefined)) out.add(Number(m[1]));
  });
  return [...out];
}
tableViewEl.addEventListener('click', ev => {
  const doc = ev.target.closest('.docs button');
  if (doc) { showDocument(Number(doc.dataset.doc)); paintTableView(); return; }
  const b = ev.target.closest('[data-tv]');
  if (b) { tvStage = b.dataset.tv; paintTableView(); }
});
document.addEventListener('click', ev => {
  const go = ev.target.closest('[data-go="table"]');
  if (!go) return;
  ev.preventDefault();
  if (!tabTable.disabled) showTab('table');
});
tableViewEl.addEventListener('change', async ev => {
  const box = ev.target.closest('[data-agent]');
  if (!box) return;
  box.disabled = true;
  try {
    const res = await fetch(API + '/table-agents', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [box.dataset.agent]: box.checked })
    });
    const body = await res.json();
    if (typeof body.fix === 'boolean') TABLE_AGENTS = body;
  } catch (err) { /* leave it as the server last said */ }
  paintTableView();
});

/* NDJSON rather than a plain POST: agentic mode is a request per step, and a
   button that sits silent for two minutes is indistinguishable from one that has
   hung. Single mode sends no step events and this reads it exactly as before. */
async function runExtract() {
  if (!truthOnlyCase && !text.trim()) return;
  /* A transcript that came out of `solution/<id>.md` is still that document's,
     so Re-extract asks for it by case and keeps the field score. It is re-read
     server-side rather than posted back: the same bytes either way, and the
     run-log row then names the file that was actually fed in. */
  const schema_first = !!(extracted && extracted.schema_first);
  return extractInto(truthOnlyCase ? { case: truthOnlyCase, from_truth: true, schema_first }
                                   : { text, job, schema_first });
}
/* The streaming half, shared by Re-extract and by the Fields-only pane. */
async function extractInto(payload) {
  setExtracted({ pending: true });
  extractSteps = null;
  fileSplit = null; runningDoc = null; resetGrouping();
  // Cleared with the steps, and for the same reason: last document's answer
  // sitting over this run's fields would name the wrong form.
  classified = null;
  // Cleared with the steps: a prefix left from the last document would sit under
  // this run's questions claiming to be the text they were asked against.
  extractPrefix = '';
  tabFields.disabled = false;
  paintFields();
  try {
    const res = await fetch(API + '/extract/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // The page job id, or the case, names the document in the run-log row
      // this writes.
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      // A 400 from the request itself -- an unknown case, an empty transcript --
      // never opens a stream, so this is the whole reply. Returned as well as
      // painted: the caller reports it where the button that started it is.
      setExtracted(await res.json());
      paintFields();
      return extracted;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.event === 'fields') {
          const { event, ...rest } = msg;
          setExtracted(rest);
        } else if (msg.event === 'error') {
          setExtracted({ error: msg.error });
        } else {
          stepEvent(msg);
        }
      }
      paintFields();
    }
  } catch (err) {
    setExtracted({ error: err.message });
  }
  paintFields();
  // A re-extraction moves the field score too, and the Accuracy tab shows it.
  if (tabAccuracy.classList.contains('on')) paintAccuracy();
  // Every extraction writes its own run-log row, so the card below is stale the
  // moment this finishes -- and the whole point of re-extracting is comparing
  // this run's field counts and grounded ratio with the last one's.
  refreshRuns();
  return extracted;
}
/* Delegated, because the step list is rebuilt by innerHTML on every repaint. */
fldBody.addEventListener('click', e => {
  if (e.target.closest('.step-prefix-btn')) {
    showStepPrefix = !showStepPrefix;
    paintFields();
    return;
  }
  const btn = e.target.closest('.step-raw-btn, .step-prompt-btn');
  if (!btn) return;
  const set = btn.classList.contains('step-raw-btn') ? openStepRaw : openStepPrompt;
  const id = btn.dataset.step;
  if (set.has(id)) set.delete(id); else set.add(id);
  paintFields();
});
fldRerun.addEventListener('click', runExtract);
fldRawChk.addEventListener('change', paintFields);
/* Delegated, because the strip is rebuilt on every repaint and a listener bound
   to a button would go with the button it was bound to. */
fldBody.addEventListener('click', ev => {
  const button = ev.target.closest('.docs button');
  if (button) showDocument(Number(button.dataset.doc));
});
fldCopy.addEventListener('click', async () => {
  if (!extracted || !extracted.fields) return;
  await navigator.clipboard.writeText(JSON.stringify(extracted.fields, null, 2));
  fldCopy.textContent = 'Copied';
  setTimeout(() => fldCopy.textContent = 'Copy JSON', 1200);
});

/* ---------- compare card ---------- */
cmpToggle.addEventListener('click', () => {
  compareEl.hidden = !compareEl.hidden;
  cmpToggle.textContent = compareEl.hidden ? 'Show' : 'Hide';
  if (!compareEl.hidden) paintCompare();
});

/* The job queue was removed on 2026-09-09, tab and subsystem both. What its
   409 protected did NOT go with it: a run is attributed to the server that ran
   it, so a batch split across two servers would be logged as one. The counter
   moved onto the random test, which is the batch this process still runs --
   `app.sweep_running`. Several documents go through `compare.py` or
   `randomtest.py`. */

/* ---------- run log ----------
   Measurements only: the CSV behind this holds no transcript, no extracted
   fields and no page images, so it stays small and is safe to keep around. */
const runBody = document.getElementById('runBody');
const runBest = document.getElementById('runBest');
const runStats = document.getElementById('runStats');
const runPath = document.getElementById('runPath');
const runRefresh = document.getElementById('runRefresh');
const runlogCard = document.getElementById('runlogCard');
const lvFloor = document.getElementById('lvFloor');
const lvWindow = document.getElementById('lvWindow');
const lvApply = document.getElementById('lvApply');
const lvReset = document.getElementById('lvReset');
const lvNote = document.getElementById('lvNote');
const lvFilters = document.getElementById('lvFilters');
const lvMainFacets = document.getElementById('lvMainFacets');
const lvExtraFacets = document.getElementById('lvExtraFacets');
const lvFilterHelp = document.getElementById('lvFilterHelp');
const lvApplied = document.getElementById('lvApplied');
const lvAdvancedCount = document.getElementById('lvAdvancedCount');
const logGroups = document.getElementById('logGroups');
const logViewNote = document.getElementById('logViewNote');
const rawScope = document.getElementById('rawScope');
const lvPipe = document.getElementById('lvPipe');
const lvEmpty = document.getElementById('lvEmpty');
const errDomToggle = document.getElementById('errDomToggle');
const logTabs = document.getElementById('logTabs');
const presentSum = document.getElementById('presentSum');
const presBias = document.getElementById('presBias');
const rawSel = document.getElementById('rawSel');
const rawNone = document.getElementById('rawNone');
const rawDelete = document.getElementById('rawDelete');

/* ---------- sortable tables ----------
   **Every column on this card sorts, the ranked one included** (2026-08-24, at
   the user's request: *allow sort each column, dont lock accuracy sort*). The
   order the server returns is the DEFAULT -- best first by whatever that table
   ranks on -- and it is now a starting point rather than the only reading.

   Sort state is per table and lives outside the paint functions, because those
   repaint on every refresh and a column that reset itself whenever the log
   drained would be unusable during exactly the run it was sorting for.

   `logSort[id]` is `{key, dir}`; a table sets its default the first time it
   paints, so the arrow always says what the order actually is. */
let logSort = {};

function lvHead(id, cols) {
  const st = logSort[id] || {};
  return '<tr>' + cols.map(c => {
    const on = st.key === c.key;
    const cls = [c.n ? 'n' : '', c.sortable === false ? '' : 'lv-th',
                 on ? 'lv-on' : ''].filter(Boolean).join(' ');
    const arrow = on ? '<span class="lv-arrow">'
                       + (st.dir === 'asc' ? '&#9650;' : '&#9660;') + '</span>' : '';
    return '<th' + (cls ? ' class="' + cls + '"' : '')
      + (c.title ? ' title="' + esc(c.title) + '"' : '')
      + (c.sortable === false ? ''
         : ' data-table="' + esc(id) + '" data-key="' + esc(c.key) + '"')
      + '>' + c.label + arrow + '</th>';
  }).join('') + '</tr>';
}

/* Not-measured sorts LAST whichever way the column points. "Not scored" is not a
   low score -- it is a document with no ground truth, or a run whose figure this
   view does not trust -- and a table sorted ascending that opened with a column
   of dashes would bury every row that has a figure. Ties keep the order the
   server sent, which is that table's own ranking. */
function lvBlank(v) { return v === null || v === undefined || v === ''; }

function lvSorted(id, rows, cols) {
  const st = logSort[id] || {};
  const col = cols.find(c => c.key === st.key && c.value);
  if (!col) return rows;
  const dir = st.dir === 'asc' ? 1 : -1;
  return rows.map((r, i) => [r, i]).sort((a, b) => {
    const x = col.value(a[0]), y = col.value(b[0]);
    if (lvBlank(x) && lvBlank(y)) return a[1] - b[1];
    if (lvBlank(x)) return 1;
    if (lvBlank(y)) return -1;
    const cmp = (typeof x === 'number' && typeof y === 'number')
      ? x - y : String(x).localeCompare(String(y));
    return cmp * dir || a[1] - b[1];
  }).map(pair => pair[0]);
}

/* One table from a column spec. The cells are unchanged from what each summary
   drew before -- what the spec adds is a sortable VALUE beside the markup, so
   the two cannot drift: a column that sorted on something other than what it
   displays would be worse than one that did not sort at all. */
function lvTable(id, rows, cols, opts) {
  opts = opts || {};
  if (!logSort[id] && opts.sort) logSort[id] = Object.assign({}, opts.sort);
  const body = lvSorted(id, rows, cols).map(e =>
      '<tr' + (opts.rowClass ? opts.rowClass(e) : '') + '>'
    // One class attribute, built from both sources. `n` and `cls` used to emit
    // two, and a second `class=` on an element is silently ignored by every
    // parser -- so a column that wanted both got whichever came first and
    // nothing said so. No column set both until the Summary tab's model column
    // did; the comment above already claimed this invariant, so it is made true
    // rather than documented around.
    + cols.map(c => {
        const cls = [c.n ? 'n' : '', c.cls || ''].filter(Boolean).join(' ');
        return '<td' + (cls ? ' class="' + cls + '"' : '') + '>'
             + c.cell(e) + '</td>';
      }).join('')
    + '</tr>').join('');
  return '<table class="queue">' + lvHead(id, cols) + body + '</table>';
}

/* Delegated, because every table on this card is innerHTML and is replaced
   whole on each repaint -- a listener bound to a `th` would not survive the
   first refresh. */
runlogCard.addEventListener('click', ev => {
  const th = ev.target.closest ? ev.target.closest('th[data-key]') : null;
  if (!th || !runlogCard.contains(th)) return;
  const id = th.dataset.table, key = th.dataset.key;
  const st = logSort[id];
  // First click on a column sorts it descending: on this card every column is
  // "more is more" except the clocks, and a reader chasing the best of anything
  // wants the top of the list, not the bottom.
  logSort[id] = (st && st.key === key)
    ? {key: key, dir: st.dir === 'asc' ? 'desc' : 'asc'}
    : {key: key, dir: 'desc'};
  paintLog();
});

function runRow(r) {
  const acc = r.char_accuracy ? Number(r.char_accuracy) : null;
  const when = (r.timestamp || '').replace('T', ' ').slice(5);   // drop the year
  const size = r.file_size_mb ? Number(r.file_size_mb).toFixed(2) + ' MB' : '—';
  const cls = r.status === 'error' ? 'q-failed'
            : r.status === 'cancelled' ? 'q-cancelled' : '';
  // A pass-2-only row: no page was read, so its pass-1 cells are blank by
  // design. The Tokens and Time cells show the extraction's own figures rather
  // than an em dash, which would read as "nothing was measured".
  const only = r.run_type === 'extract';
  return '<tr class="' + cls + '">'
    // The row's position in the file, which is the only unique key the log has:
    // two rows of one batch share a timestamp to the second and a file name. It
    // is checked against the timestamp and file server-side before anything is
    // removed, so a stale page deletes nothing rather than its neighbour.
    // Not escaped, and it does not need to be: `_row` is an integer the server
    // stamped on with `enumerate`, never anything a document or a model wrote.
    + '<td class="pick"><input type="checkbox" data-pick="' + Number(r._row)
      + '"' + (rawPicked.has(r._row) ? ' checked' : '') + '></td>'
    + '<td class="n">' + esc(when) + '</td>'
    + '<td>' + esc(r.file || '—')
      + (only ? ' <span class="tier p2">re-extract</span>' : '')
      + '<br><span style="color:var(--muted);font-size:.74rem">' + esc(size)
      + (r.pages ? ' · ' + esc(r.pages) + 'p' : '') + ' · ' + esc(r.detail || '')
      + (r.source ? ' · ' + esc(r.source) : '') + '</span></td>'
    + '<td>' + esc(r.model || '—')
      + '<br><span style="color:var(--muted);font-size:.74rem">'
      + esc(r.backend || '') + ' ' + esc(r.server || '') + '</span></td>'
    + '<td class="n">' + esc((only ? r.extract_tokens : r.tokens) || '—') + '</td>'
    // Prefill and decode are the two halves of the run and they move for
    // different reasons -- prefill scales with pixels, decode with output length
    // -- so a total alone cannot say why a run was slow. Both are already in the
    // CSV; this is where they become readable without opening it.
    + '<td class="n">'
      + (only
          ? (r.extract_seconds ? Number(r.extract_seconds).toFixed(1) + 's' : '—')
          : (r.seconds ? Number(r.seconds).toFixed(1) + 's' : '—'))
      + '<br><span style="color:var(--muted);font-size:.74rem">'
      + (only ? 'pass 2 only'
              : ((r.prefill_seconds || r.decode_seconds
                   ? Number(r.prefill_seconds || 0).toFixed(1) + 's prefill + '
                     + Number(r.decode_seconds || 0).toFixed(1) + 's decode<br>'
                   : '')
                 + (r.tokens_per_second ? Number(r.tokens_per_second).toFixed(1) + ' tok/s' : '')))
      + '</span></td>'
    // Blank, not 0/14, when the run never got as far as extracting: an
    // unmeasured tier is not a tier measured at zero, and the CSV keeps them
    // apart for the same reason.
    + '<td class="n">' + (r.p1_present === '' || r.p1_present === undefined ? '—'
        : [['p1', r.p1_present, r.p1_absent], ['p2', r.p2_present, r.p2_absent],
           ['p3', r.p3_present, r.p3_absent]]
          // A tier the row never recorded is skipped, and so is one this build
          // asked no keys for: `0/0` in a column of counts reads as a failure
          // rather than as an absence. Blank-vs-zero is still kept -- a row with
          // no p1 count at all renders the dash above.
          .filter(([, present, absent]) => present !== '' && present !== undefined
                  && Number(present) + Number(absent) > 0)
          .map(([tier, present, absent]) => '<span class="tier ' + tier + '">'
               + tier.toUpperCase() + '</span> ' + esc(present) + '/'
               + (Number(present) + Number(absent))).join('<br>'))
      // Which shape filled them. Beside the counts rather than in a column of
      // its own because it is what makes two rows of counts comparable: one
      // request and twenty fill the schema in different ways, and the CSV has
      // carried `extract_mode` from the start for exactly this reason.
      + (r.extract_mode
          ? '<br><span class="tier ' + (isStepMode(r.extract_mode) ? 'p1' : 'p2')
            + '" title="pass 2 ran as '
            + (isStepMode(r.extract_mode) ? 'a step per group of one to three fields'
                                            : 'one request for the whole form')
            + '">' + esc(r.extract_mode) + '</span>'
          : '')
      // Everything the fourteen keys do not cover comes back here under the
      // document's own printed label, so a run that found a great deal of the
      // page and one that found almost none of it are otherwise identical in
      // this column. Never a percentage: there is no denominator -- the model
      // decides how many extra fields a page has.
      + (r.other_fields === '' || r.other_fields === undefined ? ''
         : '<br><span style="color:var(--muted);font-size:.72rem" title="extra '
           + 'fields the model named itself, under the labels the page prints">+'
           + esc(r.other_fields) + ' other</span>')
      // These figures came from a later re-extraction that scored better than
      // the one this read ran with, so the row's timestamp does not describe
      // them. Said here rather than left silent: a coverage count that quietly
      // changed under a fixed timestamp is the kind of thing that gets believed.
      + (r.extract_updated
          ? '<br><span style="color:var(--muted);font-size:.7rem" title="replaced by a '
            + 'better re-extraction at ' + esc(r.extract_updated) + '">updated</span>'
          : '')
      + '</td>'
    + '<td class="n acc" style="' + (acc === null ? '' : 'color:' + accColour(acc / 100))
      + '">' + (acc === null ? '—' : acc.toFixed(1) + '%')
      + (r.case ? '<br><span style="color:var(--muted);font-size:.74rem;font-weight:400">'
                  + esc(r.case) + '</span>' : '') + '</td>'
    // How much of what was extracted was actually found on the page. An
    // invented field costs no time and no tokens, so it shows up in no other
    // column: a run can look fast, clean and wrong.
    + '<td class="n">' + (r.grounded_pct === '' || r.grounded_pct === undefined ? '—'
        : Number(r.grounded_pct).toFixed(0) + '%')
      + (Number(r.ungrounded) > 0
         ? '<br><span style="color:var(--danger);font-size:.72rem">'
           + esc(r.ungrounded) + ' ungrounded</span>' : '')
      // The field score, in the same cell as grounding because the two answer
      // the same kind of question about pass 2 and the stronger one belongs
      // beside the weaker: grounded says a value is printed somewhere on the
      // page, this says it is the value that belongs in that key. Blank for
      // every document without a field truth file, which is most of them.
      //
      // The count of values comes with it, always. This number does NOT read
      // down the column -- its denominator is whatever that document's truth
      // file rules on -- so 100% of 3 values and 100% of 39 have to be
      // distinguishable at a glance.
      + (r.field_acc === '' || r.field_acc === undefined ? ''
         : '<br><span style="font-size:.72rem" title="of the ' + esc(r.field_expected)
           + ' value(s) the field ground truth says this document prints AND the '
           + 'requirement demands of its type, the share that came back correct '
           + '— a different question from whether they are on the page. Fields '
           + 'the requirement marks optional are extracted and shown, and are '
           + 'not in this rate"><b style="color:' + accColour(Number(r.field_acc) / 100)
           + '">' + Number(r.field_acc).toFixed(0) + '%</b> of '
           + esc(r.field_expected) + ' required</span>')
      + '</td>'
    + '<td><span class="qbadge ' + (r.status === 'ok' ? 'done' : esc(r.status || '')) + '">'
      + esc(r.status || '') + '</span>'
      + (r.error ? '<br><span style="color:var(--danger);font-size:.72rem">'
                   + esc(r.error) + '</span>' : '') + '</td>'
    + '<td><button type="button" class="ghost-sm" data-run-detail="' + Number(r._row)
      + '" aria-expanded="' + (rawExpanded === runIdentity(r)) + '" aria-label="'
      + logAttr('Details for ' + (r.file || 'run') + ' at ' + (r.timestamp || 'unknown time'))
      + '">' + (rawExpanded === runIdentity(r) ? 'Close' : 'Details') + '</button></td>'
    + '</tr>';
}

/* Best score per document, over the whole log. Kept in a variable as well as on
   screen because the Accuracy tab compares the run it is showing against it, and
   the two must not be able to disagree. */
let logBests = {};
/* Every table on this card is compiled over the most recent runs OF EACH ROW --
   `settings.SUMMARY_RUNS`, applied per setting, per model and per document, not
   as a slice off the end of the file. That has to be said on each table rather
   than once at the top: "best ever" over twenty runs and "best ever" over two
   hundred are different claims, and nobody reading a ranking is looking at the
   card header. The noun differs per table because the key does. */
let logWindow = null, logRows = 0;
/* **The label does not state the number** (2026-08-21, at the user's request:
   *if its 20 dont say its 20 in the summary, if not says how many*). The window
   is a ceiling, not a count: a setting with three runs is summarised over three,
   and a title reading "last 20 runs" over it is simply false. Every row already
   prints what it actually covers -- `N run(s)`, `N read(s)`, `2 of 7` -- so the
   honest label says only that the figures are recent, and the ceiling goes in
   the tooltip where it belongs. */
function windowNote(what) {
  if (!logWindow) return '';
  return '<span class="cs-of" title="each row is summarised over that ' + what
    + "'s most recent runs, up to " + logWindow
    + ' (SUMMARY_RUNS) — the count on each row is what it actually covers">'
    + ' &middot; recent runs per ' + what + '</span>';
}
function paintBests(t) {
  logWindow = t.window || null;
  logRows = t.logged || 0;
  logBests = t.best_by_case || {};
  const cases = Object.keys(logBests).sort();
  if (!cases.length) { runBest.innerHTML = ''; return; }
  runBest.innerHTML = '<span title="'
      + (logWindow ? "the best score in each document's recent runs (up to "
                     + logWindow + '), under any setting'
                   : 'the best score each document has ever reached')
      + '">Best' + (logWindow ? ' recently' : ' ever') + ', any setting:</span>'
    + cases.map(c => {
        const b = logBests[c];
        // The condition is the tooltip rather than the line: five documents
        // times model, backend and detail is more text than the numbers it
        // would be explaining.
        const how = [b.model, b.backend, b.detail, b.extract_mode, b.timestamp]
          .filter(Boolean).join(' · ');
        return '<span class="rb" title="' + esc(how) + '">' + esc(c)
          + ' <b>' + Number(b.char_accuracy).toFixed(1) + '%</b></span>';
      }).join('');
}

/* ---------- per-document summary ---------- */
/* Which setting to use for a document, answered three ways, because the run log
   answers it three ways and they rarely agree: the transcript score, the field
   score, and the clock. The chips above give the first of those only, and a
   percentage with no setting attached is not something anyone can act on.

   Everything here is computed in `runlog.by_case` -- the rules about what is
   eligible for each column live there, beside the file format they read. */

/* A model name short enough to sit under a number. The full one is the title
   attribute wherever this is used, because two `hf.co/...` repacks of one model
   differ only in the part this drops. */
function csModel(name) {
  return String(name || '')
    .replace(/^hf\.co\//, '').replace(/:latest$/, '').split('/').pop();
}
/* ---------- the reading engine, on a table row ----------

   The two pipelines are stored and scored in one place on purpose: a transcript
   is a transcript, and the whole reason to run a library beside a model server
   is to put the two accuracies on one scale. What the tables need is not a
   separate store but a LABEL, so a row can be read as one or the other without
   inferring it from a backend string.

   Blank draws nothing rather than an "LLM" default. A document read by both
   engines has no single answer, and a run that read no page has none at all --
   in both cases a tag would be a claim the row cannot support. */
function engineLabel(id) {
  return id === 'library' ? 'Library' : id === 'llm' ? 'LLM' : '';
}
function engineTag(id) {
  const text = engineLabel(id);
  if (!text) return '';
  return '<span class="eng-tag ' + esc(id) + '" title="'
    + (id === 'library'
       ? 'read by an OCR library in the processes of this machine — no prompt, '
         + 'no sampler, no token budget'
       : 'read by a model server over HTTP — a prompt, a sampler and a token budget')
    + '">' + text + '</span>';
}
/* The setting a figure was reached under, on one line. `run_type` is spelled out
   for a re-extraction: that row scored pass 2 against a transcript some earlier
   run read, so the detail and profile beside it did not produce the number. */
function csSetting(b) {
  if (!b) return '';
  // **Two models is two names.** `extract_model` is blank on the one-model
  // setup -- the CSV column’s own rule -- so one name means both passes ran on
  // it and the arrow is drawn only where the row really carried a second model.
  // Printing `model` alone made a typhoon-reads/qwen-extracts run look as though
  // typhoon had done the extracting, which is the pass this table ranks in two
  // of its four columns.
  const two = b.extract_model && b.extract_model !== b.model;
  const models = two ? csModel(b.model) + ' → ' + csModel(b.extract_model)
                     : csModel(b.model);
  // What each pass ran ON, where the row says: the quantisation beside the
  // model it belongs to, and the engine with its version in place of the bare
  // backend. Two passes on two servers name both, in pass order.
  const quant = two
    ? [b.model_quant || '?', b.extract_quant || '?'].join(' → ')
    : b.model_quant;
  const engine = two && b.extract_inference_engine
      && b.extract_inference_engine !== b.inference_engine
    ? (b.inference_engine || b.backend || '?') + ' → ' + b.extract_inference_engine
    : (b.inference_engine || b.backend);
  const bits = [models, (two && !b.model_quant && !b.extract_quant) ? '' : quant,
                engine, b.detail, b.ocr_profile, b.extract_mode]
    .filter(Boolean).join(' · ');
  // Which engine produced it, beside the setting rather than inside the joined
  // string: it is a category the row belongs to, not another setting to read.
  const tag = engineTag(b.ocr_engine);
  const when = (b.timestamp || '').replace('T', ' ');
  const full = two ? b.model + ' read → ' + b.extract_model + ' extracted'
                   : (b.model || '');
  return '<span class="cs-set" title="' + esc(full) + (when ? ' — ' + esc(when) : '')
    + '">' + esc(bits || '—')
    + (b.run_type === 'extract' ? ' <b>· re-extract only</b>' : '') + '</span>' + tag;
}
function csSecs(v) { return v === null || v === undefined ? '—' : Number(v).toFixed(1) + 's'; }
/* A mean clock with what fed it underneath. The run count is not optional --
   these cells are two or three runs deep as often as not -- and the range is
   printed beside the SD because a tight SD with a wide range is one stray run,
   which a mean over three cannot say on its own. Same reading as `anFig` on the
   analysis tab, in this table’s own type. */
function csFig(f) {
  if (!f || f.mean === null || f.mean === undefined)
    return '<span class="cs-none">no complete run</span>';
  const parts = [f.runs + ' run' + (f.runs === 1 ? '' : 's')];
  if (f.sd !== null && f.sd !== undefined)
    parts.push('±' + Number(f.sd).toFixed(1) + 's');
  if (f.min !== null && f.min !== undefined && f.min !== f.max)
    parts.push(csSecs(f.min) + '–' + csSecs(f.max));
  return '<span class="cs-val">' + csSecs(f.mean) + '</span>'
    + '<span class="cs-set">' + parts.join(' · ') + '</span>';
}
function csPct(v) {
  return '<span class="cs-val" style="color:' + accColour(Number(v) / 100) + '">'
    + Number(v).toFixed(1) + '%</span>';
}
/* ---------- standouts: best and worst, four ways ---------- */
/* Ranked on `runlog._standout_score` -- accuracy x (1 - failure rate), which is
   what one attempt is worth. Read the score as a claim about attempts, not about
   quality: a model that reads at 95% and fails half the time is worth less per
   run than one that reads at 70% and always finishes, and that is the ordering
   the user asked for (failures first, accuracy second). */
function stFig(e, unit) {
  if (e.accuracy === null || e.accuracy === undefined) {
    // Two different absences. Nothing scored AND runs failed is a result -- not
    // one attempt survived to be measured; nothing scored and nothing failed is
    // a document nobody has ground truth for.
    return '<span class="cs-none">' + (e.failed ? 'none survived' : 'not scored') + '</span>';
  }
  return '<span class="cs-val" style="color:' + accColour(Number(e.accuracy) / 100) + '">'
    + Number(e.accuracy).toFixed(1) + unit + '</span>';
}
function stFails(e) {
  if (!e.runs) return '<span class="cs-none">&mdash;</span>';
  if (!e.failed) return '<span class="cs-none">0 of ' + e.runs + '</span>';
  return '<span style="color:var(--danger)">' + e.failed + ' of ' + e.runs
    + ' (' + Number(e.failure_rate).toFixed(0) + '%)</span>';
}
function stSentence(e, tag, cls, unit, noun) {
  if (!e) return '';
  return '<p class="st-line"><span class="st-tag ' + cls + '">' + tag + '</span>'
    + '<b title="' + esc(e.key) + '">' + esc(csModel(e.key)) + '</b>'
    // Blank on a DOCUMENT that both engines have read, which is the honest
    // answer: "sol003 is a library document" would be a claim about the page
    // made out of who happened to read it.
    + engineTag(e.ocr_engine) + ' &mdash; '
    // The noun only belongs to a figure. "none survived transcript" is not a
    // sentence; "none survived" is the whole of what that entry has to say.
    + stFig(e, unit) + (e.accuracy === null || e.accuracy === undefined ? '' : ' ' + noun)
    + ', ' + stFails(e) + ' failed'
    + (e.thin ? ' <span class="st-thin">(only ' + e.runs + ' run)</span>' : '')
    + '</p>';
}
function stBox(id, title, data, unit, noun, hint) {
  if (!data || !data.ranked || !data.ranked.length) return '';
  // Best and worst are marked by KEY, not by position and not by identity, so
  // they stay on the right rows when the reader sorts the table by something
  // else -- the ranking is a claim the sort must not be able to rewrite by
  // accident.
  //
  // **Identity is what this used to compare and it could never match.** The
  // server puts the same entry in `ranked` and in `best`, but JSON has no
  // references: `JSON.parse` hands back two separate objects, so `e === data.best`
  // was false on every row and neither mark had ever appeared. The key is unique
  // within a list -- it is the model or the document the row is about -- and it
  // survives the round trip.
  const bestKey = data.best ? data.best.key : null;
  const worstKey = data.worst ? data.worst.key : null;
  const cols = [
    {key: 'name', label: 'Name', value: e => csModel(e.key),
     // The engine badge sits in the ranked table as well as in the two
     // sentences above it. This is the one table where a library reader and a
     // model server are ranked against each other on one score, which is the
     // whole point of keeping them in one log -- and a reader cannot act on
     // rank 2 without knowing which of the two it is.
     cell: e => '<b title="' + esc(e.key) + '">' + esc(csModel(e.key)) + '</b>'
       + engineTag(e.ocr_engine)
       + (e.thin ? ' <span class="st-thin">' + e.runs + ' run</span>' : '')},
    {key: 'score', label: 'Score', n: true, value: e => e.score,
     title: 'accuracy x (1 - failure rate): what one attempt is worth',
     cell: e => e.score === null || e.score === undefined
       ? '<span class="cs-none">&mdash;</span>'
       : '<span class="cs-val">' + Number(e.score).toFixed(1) + '</span>'},
    {key: 'accuracy', label: 'Accuracy', n: true, value: e => e.accuracy,
     cell: e => stFig(e, unit)},
    // **No Runs column, at the user's request**: `stFails` already prints the run
    // count as its own denominator -- `0 of 47`, `5 of 47 (11%)` -- so a column
    // beside it repeating 47 was the same number twice in one row.
    {key: 'failed', label: 'Failed', n: true, value: e => e.failure_rate,
     title: 'runs that did not finish, out of the runs this row is summarised '
          + 'over — its own recent ones',
     cell: e => stFails(e)}
  ];
  return '<div class="st-box"><h4 title="' + esc(hint) + '">' + title + '</h4>'
    + stSentence(data.best, 'best', 'st-good', unit, noun)
    + stSentence(data.worst, 'worst', 'st-bad', unit, noun)
    + lvTable(id, data.ranked, cols, {
        sort: {key: 'score', dir: 'desc'},
        rowClass: e => e.key === bestKey ? ' class="ms-top"'
                     : e.key === worstKey ? ' class="q-failed"' : ''
      })
    + '</div>';
}
function paintStandouts(t) {
  const s = t.standouts;
  if (!s) { standoutSum.innerHTML = ''; return; }
  const boxes = [
    stBox('st-ocr-model', 'Reading &mdash; by model', s.ocr.models, '%', 'transcript',
          'Every read this model has done, at any Detail or profile'),
    stBox('st-ocr-case', 'Reading &mdash; by document', s.ocr.cases, '%', 'transcript',
          'Every read of this document, by any model'),
    stBox('st-ext-model', 'Extracting &mdash; by model', s.extract.models, '%', 'fields',
          'Every extraction this model has done, in either shape'),
    stBox('st-ext-case', 'Extracting &mdash; by document', s.extract.cases, '%', 'fields',
          'Every extraction from this document, by any model')
  ].filter(Boolean).join('');
  if (!boxes) { standoutSum.innerHTML = ''; return; }
  standoutSum.innerHTML = '<span class="cs-title">Standouts &mdash; what is '
    + 'carrying this, and what is failing it' + windowNote('row') + '</span>'
    + '<div class="st-grid">' + boxes + '</div>'
    + '<p class="note"><b>Score is accuracy &times; (1 &minus; failure rate)</b> '
    + '&mdash; what one attempt is worth, so a failed run counts as zero and '
    + 'failures outweigh a few points of accuracy. Accuracy is taken over the '
    + 'runs that finished, which is why it is printed beside the failures and '
    + 'never alone. Models are grouped <b>without</b> their Detail, profile or '
    + 'shape: the tables below answer which setting to run, this answers which '
    + 'model is carrying its weight across everything it has been asked to do. '
    + '<b>A document&rsquo;s row is not a verdict on the document</b> &mdash; a '
    + 'fixture every model reads badly may be the hardest page here, or its '
    + 'hand-written ground truth may be wrong.'
    + (logWindow
        ? ' Every row on this card covers <b>that model\'s or that document\'s own '
          + 'recent runs</b> &mdash; up to ' + logWindow + ' each '
          + '(<code>SUMMARY_RUNS</code>), never the log\'s last ' + logWindow
          + ' lines. The count beside each row is what it actually covers, which '
          + 'is often fewer. A mean taken across a change to the scorer, the '
          + 'schema or the presets describes a build nobody is running, and one '
          + 'busy evening on one setting must not push every other setting out of '
          + 'the table. The CSV keeps everything.'
        : '')
    + '</p>';
}

/* ---------- errors: what is failing, ranked three ways ---------- */
/* From `runlog.errors`. The setting tables print a failure RATE per setting and
   the standouts fold failures into a score with accuracy; neither answers *what
   is failing, and is it one model, one document, or one pairing*. Ranked on the
   count -- "most error" -- with the rate beside it and every column sortable.

   A document that fails nine times all under one model is an artefact of that
   pairing, not a hard document, and it would sit at the top of the by-document
   list looking like one. `sources` is how many distinct contributors on the
   OTHER axis produced a row's failures, so `sources === 1` is exactly "dominated
   by one". Those rows are HIGHLIGHTED here, never hidden -- they are the finding.

   The toggle does not touch this table. It asks the server to recompile every
   OTHER table on the card -- the ranking, standouts, best reading, best
   extraction and per-document -- with those single-source failures removed, so a
   bad pairing stops skewing the tables that answer "which model, which document".
   That is a round-trip (`runlog.drop_single_source`), not a client filter, which
   is why it goes through `refreshRuns`. */
let errDropSingle = false;
/* **Off by default, like every other control on this card** (2026-08-25, at the
   user's request: *show all if i do not toggle to exclude them like the other
   page*). It shipped defaulting ON earlier the same day and that was reversed:
   a page whose whole job is to be read at a glance must not also be a page you
   have to remember is hiding something. Excluding a model is the chips' job
   here as everywhere else, and this toggle is the shortcut that does it by
   rule instead of by hand.

   It stays a request parameter rather than a client-side filter because the
   rule needs the whole set to decide -- "delivers less than 60% of what the
   leader delivers" is not a question a page holding one table can answer. */
let presBiased = false;

const ERR_LABEL = {
  error: 'crashed', cancelled: 'cancelled', truncated: 'hit the token cap',
  looped: 'read looped', extract_loop: 'extraction cycled'
};
function errKindsNote(kinds) {
  return Object.keys(kinds || {}).sort((a, b) => kinds[b] - kinds[a])
    .map(k => kinds[k] + ' ' + (ERR_LABEL[k] || k)).join(', ');
}
// "Dominated by one": every failure this row has came from a single contributor
// on the other axis. Only meaningful where a row HAS another axis -- the pairs
// list carries no concentration, so it is never asked.
function errDominated(e) { return e.sources === 1; }

function errBox(id, title, entries, opts) {
  opts = opts || {};
  if (!entries || !entries.length) return '';
  const rows = entries;   // never filtered here -- single-source rows are the finding
  const cols = [
    {key: 'name', label: 'Name', value: e => e.key,
     cell: e => '<b title="' + esc(e.key) + '">'
       + esc(opts.pair ? e.parts.map(csModel).join(' · ') : csModel(e.key)) + '</b>'
       + '<span class="cs-set">' + e.runs + ' run' + (e.runs === 1 ? '' : 's') + '</span>'},
    {key: 'failed', label: 'Errors', n: true, value: e => e.failed,
     title: 'runs that failed — a crash, cancellation, truncation, a read loop '
          + 'or a cycled extraction',
     cell: e => (e.failed
         ? '<span class="cs-val" style="color:var(--danger)">' + e.failed + '</span>'
           + ' <span class="cs-of">of ' + e.runs
           + (e.failure_rate !== null ? ' · ' + Number(e.failure_rate).toFixed(0) + '%' : '')
           + '</span>'
         : '<span class="cs-none">0 of ' + e.runs + '</span>')
       + (e.failed ? '<span class="cs-set">' + esc(errKindsNote(e.kinds)) + '</span>' : '')}
  ];
  if (opts.blame) {
    cols.push({key: 'top', label: 'Mostly from', value: e => e.top ? e.top.share : -1,
      title: 'the single ' + opts.blame + ' behind most of this row’s failures, '
           + 'and how many ' + opts.blame + 's produced any — one is dominated, '
           + 'not difficult',
      cell: e => e.top
        ? '<b title="' + esc(e.top.key) + '">' + esc(csModel(e.top.key)) + '</b>'
          + ' <span class="cs-of">' + e.top.failed + ' (' + Number(e.top.share).toFixed(0)
          + '%)</span><span class="cs-set">' + e.sources + ' ' + opts.blame
          + (e.sources === 1 ? '' : 's')
          + (errDominated(e) ? ' · <span class="err-dom-tag">single source</span>' : '')
          + '</span>'
        : '<span class="cs-none">—</span>'});
  }
  if (opts.empty) {
    cols.push({key: 'empty', label: 'Empty', n: true, value: e => e.empty || 0,
      title: 'runs that finished at HTTP 200 and returned an empty transcript — '
           + 'not a failure, a different problem, and fixed in a different place',
      cell: e => e.empty
        ? '<span class="cs-val">' + e.empty + '</span>'
        : '<span class="cs-none">0</span>'});
  }
  cols.push({key: 'share', label: 'Share', n: true, value: e => e.share === null ? -1 : e.share,
    title: 'this row’s share of all the failures in this list',
    cell: e => e.share === null || e.share === undefined
      ? '<span class="cs-none">—</span>'
      : '<span class="cs-val">' + Number(e.share).toFixed(0) + '%</span>'});
  return '<div class="st-box"><h4>' + title + '</h4>'
    + lvTable(id, rows, cols, {
        sort: {key: 'failed', dir: 'desc'},
        rowClass: e => (opts.blame && errDominated(e)) ? ' class="err-dom"' : ''
      })
    + '</div>';
}

function errPass(prefix, data, isRead) {
  if (!data) return '';
  const head = '<p class="st-line" style="margin-bottom:2px">'
    + (data.failed
        ? '<b style="color:var(--danger)">' + data.failed + '</b> of ' + data.runs
          + ' runs failed'
          + (Object.keys(data.kinds || {}).length ? ' — ' + esc(errKindsNote(data.kinds)) : '')
        : '<span class="cs-none">no failures in ' + data.runs + ' runs</span>')
    + '</p>';
  // Models blame the document behind their failures; documents blame the model.
  // Pairs carry no concentration -- they are already one of each.
  const boxes = [
    errBox(prefix + '-model', 'By model', data.models, {blame: 'document', empty: isRead}),
    errBox(prefix + '-case', 'By document', data.cases, {blame: 'model', empty: isRead}),
    errBox(prefix + '-pair', 'By pairing', data.pairs, {pair: true})
  ].filter(Boolean).join('');
  return head + '<div class="st-grid">' + boxes + '</div>';
}

function paintErrors(t) {
  const e = t.errors;
  if (!e || (!e.read && !e.extract)) { errorSum.innerHTML = ''; return; }
  errorSum.innerHTML = '<span class="cs-title">Errors — what is failing'
      + windowNote('row') + '</span>'
    + (e.read ? '<p class="err-head" style="margin-top:12px">Reading</p>'
        + errPass('err-r', e.read, true) : '')
    + (e.extract ? '<p class="err-head">Extracting</p>'
        + errPass('err-x', e.extract, false) : '')
    + '<p class="note"><b>Ranked on the failure COUNT</b>, not the rate — “most '
    + 'error” is what a count answers, and one run that failed is 100% and rarely '
    + 'the thing to look at first. Every column sorts, so either reading is a '
    + 'click away. <b>By pairing</b> is the row the other two cannot be read off: '
    + 'a document at the top of <i>by document</i> and a model at the top of <i>by '
    + 'model</i> may be one pairing failing repeatedly or two separate problems, '
    + 'and only the pair says which. <b>Mostly from</b> names the single '
    + 'contributor on the other axis behind most of a row’s failures; a row whose '
    + 'every failure came from one contributor is a <i>single source</i> — '
    + 'dominated by that pairing rather than a hard model or document, and it is '
    + '<b>highlighted</b> above. <b>Drop single-source failures</b> (in the '
    + 'controls at the top of the card) leaves those rows here and removes their '
    + 'failures from every <b>other</b> table, so a bad pairing stops skewing '
    + 'which model and which document look worst. A run counts under every pass it '
    + 'performed, so a two-model run is a failure of whichever pass broke.</p>';
}

// The toggle asks the server to recompile the other tables without the
// single-source failures, so it goes through `refreshRuns` rather than a local
// repaint. It lives in the general control bar (static markup), so a direct
// listener is enough -- unlike the numeric controls, a checkbox applies on its
// own change rather than waiting for Apply, the same as the Pipeline segment.
errDomToggle.addEventListener('change', () => {
  errDropSingle = errDomToggle.checked;
  refreshRuns();
});

presBias.addEventListener('change', () => {
  presBiased = presBias.checked;
  refreshRuns();
});

function paintCaseSummary(t) {
  const cases = t.by_case || {};
  const ids = Object.keys(cases).sort();
  if (!ids.length) { caseSum.innerHTML = ''; return; }
  const rows = ids.map(id => Object.assign({id: id}, cases[id]));
  const cols = [
    {key: 'doc', label: 'Document', value: e => e.id,
     cell: e => '<b>' + esc(e.id) + '</b><span class="cs-set">' + e.runs + ' read(s)'
       + (e.extracts ? ' · ' + e.extracts + ' re-extract(s)' : '') + '</span>'},
    // Pass 2's correctness. The count of values comes with the percentage
    // everywhere it is printed: the denominator is this document's truth file
    // and nothing else, so 40% of 15 and 44% of 18 are not one column.
    {key: 'fields', label: 'Best fields',
     title: 'highest field score: the share of the values the field ground truth '
          + 'says this document prints that came back correct',
     value: e => e.best_field ? e.best_field.field_acc : null,
     cell: e => e.best_field
       ? csPct(e.best_field.field_acc) + ' <span class="cs-of">of '
         + esc(e.best_field.field_expected) + '</span>' + csSetting(e.best_field)
         + '<span class="cs-set">transcript '
         + (e.best_field.char_accuracy === null ? 'not scored'
            : Number(e.best_field.char_accuracy).toFixed(1) + '%')
         + ' · ' + csSecs(e.best_field.total_seconds) + '</span>'
       // Blank is not zero, the same rule the CSV column follows: no field truth
       // file, or none of this document's runs has been scored against one yet.
       : '<span class="cs-none">not scored yet</span>'},
    {key: 'fastest', label: 'Fastest full run',
     title: 'quickest run that finished and did both passes; a read that skipped '
          + 'extraction is not in this column, because it did less work',
     value: e => e.fastest ? e.fastest.total_seconds : null,
     cell: e => e.fastest
       ? '<span class="cs-val">' + csSecs(e.fastest.total_seconds) + '</span>'
         + csSetting(e.fastest)
         + '<span class="cs-set">' + csSecs(e.fastest.seconds) + ' read + '
         + csSecs(e.fastest.extract_seconds) + ' extract · transcript '
         + Number(e.fastest.char_accuracy).toFixed(1) + '%'
         + (e.fastest.field_acc === null ? ''
            : ' · fields ' + Number(e.fastest.field_acc).toFixed(0) + '%')
         + '</span>'
       : '<span class="cs-none">no complete run</span>'},
    // What the pipeline COSTS on this page on an ordinary run, beside what it
    // cost on its best one. The fast column flatters and cannot say so: both
    // servers cache -- llama.cpp reuses a prompt prefix and Ollama keeps a model
    // resident -- so the quickest run of a document is very often a warm repeat
    // nobody gets cold. Same population as Fastest, so the range under this mean
    // opens on the figure in the column to its left.
    {key: 'avgrun', label: 'Avg full run',
     title: 'mean end-to-end clock, read plus extract, over the same runs the '
          + 'Fastest column ranks: runs that finished, scored above zero and did '
          + 'both passes',
     value: e => e.pipeline_seconds ? e.pipeline_seconds.mean : null,
     cell: e => csFig(e.pipeline_seconds)},
    // Character accuracy alone. Word accuracy is still computed and logged, and
    // is a useful diagnostic in the CSV, but it is order-sensitive where this
    // number deliberately is not -- printing both invites reading the gap as an
    // error rather than as a page walked in another order.
    {key: 'transcript', label: 'Best transcript',
     title: 'highest character accuracy of the transcript, over every run and '
          + 'every setting in this view',
     value: e => e.best_char ? e.best_char.char_accuracy : null,
     cell: e => e.best_char
       ? csPct(e.best_char.char_accuracy) + csSetting(e.best_char)
       : '<span class="cs-none">not scored</span>'}
  ];
  caseSum.innerHTML = '<span class="cs-title">Best setting per document'
    + windowNote('document') + '</span>'
    // Documents in name order by default. Unlike the two setting tables there is
    // no server ranking here to preserve -- each column is a different question
    // and the rows are a list, not a leaderboard.
    + lvTable('case', rows, cols, {sort: {key: 'doc', dir: 'asc'}})
    + '<p class="note">Each column is the best row <b>in this view</b> for that '
    + 'document, with the setting that produced it — they are three different '
    + 'runs more often than one. Fields and transcript are separate scores: a '
    + 'run can read the page well and still put the values in the wrong keys. '
    + 'Fastest and Avg full run count reads that finished, scored above zero '
    + 'and ran both passes — the same runs, so the average is what this page '
    + 'usually costs and the fastest is the best it has managed. Prefer the '
    + 'average: llama.cpp caches prompts and Ollama keeps a model resident, so '
    + 'a record time is often a warm repeat nobody gets cold. Where a run used '
    + 'two models the setting reads <b>reader → extractor</b>; one name means '
    + 'both passes ran on it. A run that <b>did not '
    + 'finish</b> — looped, truncated, cancelled or errored — is counted in the '
    + 'read count and can hold no record; a run that finished and read badly is '
    + 'not a failure and is eligible for all three. Any column here sorts.</p>';
}


/* ---------- time x document x accuracy ---------- */
/* Pass 1 only. Figures over reads, grouped: the model x document grid, the two
   Detail tables, time per model, the per-document and per-model spreads, what
   each inferred mark is worth, and the reads the outlier fence dropped.

   **There are no correlation or share-of-spread bars, removed 2026-08-25 at the
   user's request.** They summarised the same reads the grid and the tables show
   directly, in two abstractions (Pearson r, eta-squared) that a reader has to
   translate back into reads before they mean anything -- and on a log measured
   on one machine the splits behind them were degenerate. Do not put them back
   without being asked; the grouped figures below are the material they were
   computed from.

   From `runlog.ocr_analysis`; the eligibility rules live there. */

// A hex colour with an alpha, for the heatmap tint. accColour returns hex.
function hexA(hex, a) {
  const n = parseInt(String(hex).slice(1), 16);
  return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
}
// Strength as a colour: four neutral tiers, used by the Detail cost table and
// the outlier list. Neutral on purpose -- a strong effect is neither good nor
// bad, only strong.
function anTierColour(t) {
  return ['#5b6270', '#d9a441', '#4c8dff', '#7c5cff'][t] || '#5b6270';
}
// Read time for a heatmap cell: minutes past 100s, so a grid of 300s reads is
// legible without a column of five-digit second counts.
function anSecs(s) {
  if (s === null || s === undefined) return '—';
  return s >= 100 ? (s / 60).toFixed(1) + 'm' : s.toFixed(0) + 's';
}
function anMeanOf(vals) {
  const v = vals.filter(x => x !== null && x !== undefined);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function paintAnalysis(t) {
  const a = t.ocr_analysis;
  if (!a || !a.points) {
    analysisSum.innerHTML = '<span class="cs-title">Time × document × accuracy'
      + windowNote('cell') + '</span>'
      + '<p class="note">Nothing to analyse yet. This needs pass-1 reads that '
      + 'finished, are timed, and have a ground-truth document to score against '
      + '&mdash; drop a benchmark file, or run the random test in <code>ocr</code> '
      + 'scope, and the reads will appear here.</p>';
    return;
  }
  analysisSum.innerHTML = '<span class="cs-title">Time × document × accuracy '
      + '— pass 1' + windowNote('cell') + '</span>'
    + pipeNote('ocr')
    + anScopeNote()
    + anMarksBar(a)
    + anHeat(a)
    + anDetailAcc(a)
    + anTimeDetail(a)
    + anModelTime(a)
    + anSpread(a)
    + anMarkStats(a)
    + anOutliers(a)
    + '<p class="note">Each cell in the grid is one model on one document, '
    + 'over that pairing’s recent reads; the tint is accuracy and the small figure '
    + 'is time. Reading across a row is how one model fared on every document, '
    + 'down a column is how every model fared on one. <b>Nothing here is a claim '
    + 'of cause</b> — a document that takes longer is mostly a longer document. '
    + '<b>Time outliers are excluded</b> before any figure here is computed: a '
    + 'read whose time is more than ' + esc(String(a.outlier_mads || '')) + '×MAD '
    + 'from the middle of its own model×document cell — a cold model load or a '
    + 'runaway — is dropped, because one of them moves a mean on its own. '
    + 'They are still in the raw table and the CSV. '
    + '<b>GPU/CPU and cold/hot are inferred</b>, not recorded: the app talks HTTP '
    + 'to a server it did not launch, so hardware is read off the decode rate '
    + '(≥' + esc(String(a.gpu_min_tps || '')) + ' tok/s reads as GPU) and warmth '
    + 'off prefill (under ' + esc(String(a.warm_prefill_max || '')) + 's reads as '
    + 'hot, i.e. a cache hit or a resident model). Both are proxies and both are '
    + 'filterable from the chips at the top of the card.</p>';
}

/* ---- what this tab is for, and what it is not ----
   Set 2026-08-25 at the user's request: *this page focuses on comparing the
   detail, time and accuracy across the same model and the same doc -- model
   accuracy comparison is already on another page*.

   Said at the top rather than left implicit, because the tables below CAN be
   misread as a model ranking and there are three better places for that on this
   very card.

   **The per-(model x document) table that held both still was removed the same
   day**, at the user's request: *I only want detail vs accuracy and time vs
   detail*. So every table here pools one axis -- Detail-vs-accuracy pools a
   model's documents, Detail-vs-time pools a document's models -- and the notes
   under them say which. */
function anScopeNote() {
  return '<p class="an-scope">What the <b>Detail</b> preset buys and what it '
    + 'costs: accuracy per model across the presets, time per document across '
    + 'them. <span class="an-scope-not">Not a model ranking</span> — for which '
    + 'model reads best, use <b>Full rank</b>, <b>Best reading</b> or <b>Per '
    + 'document</b>, which compare like with like. Pass 1 only.</p>';
}
/* The two inferred marks as readable words. `unknown` is a real third state --
   a run with no decode rate or no prefill figure -- and is never folded into
   either bucket: guessing is the one thing these marks must not do. */
const AN_MARK = {
  hardware: {gpu: 'GPU', cpu: 'CPU', unknown: 'unrecorded'},
  start: {cold: 'cold start', hot: 'warm start', unknown: 'unrecorded'}
};
function anMarkLabel(kind, key) {
  return (AN_MARK[kind] || {})[key] || key;
}
/* The composition of what is being analysed, above the tables: how many reads
   survived the outlier fence, and how they split across the two marks. It is
   here rather than in the note because it QUALIFIES every figure below -- an
   all-cold, all-GPU log means the marks explain nothing and the splits will be
   degenerate, and that should be visible before the tables, not after them. */
function anMarksBar(a) {
  const chip = (kind, counts) => Object.keys(counts || {})
    .sort((x, y) => counts[y] - counts[x])
    .map(k => '<span class="an-chip' + (k === 'unknown' ? ' an-chip-un' : '') + '">'
      + esc(anMarkLabel(kind, k)) + ' <b>' + counts[k] + '</b></span>').join('');
  const hw = chip('hardware', a.hardware_counts);
  const st = chip('start', a.start_counts);
  if (!hw && !st && !a.excluded) return '';
  return '<div class="an-marks">'
    + '<span class="an-marks-label">' + a.points + ' read(s) analysed</span>'
    + (a.excluded
        ? '<span class="an-chip an-chip-out" title="reads whose time was a robust '
          + 'outlier within their own model×document×Detail cell — a cold load or '
          + 'a runaway. Dropped from every figure on this tab; still in the log. '
          + 'Listed in full at the bottom of this tab.">'
          + a.excluded + ' outlier(s) excluded</span>'
        : '')
    // What the fence could NOT check. A cell of two reads cannot be tested at
    // all -- the median sits between them, so neither is far from it however
    // different they are -- and this log has a 25-minute read surviving for
    // exactly that reason. A fence that hid how much it skipped would be worse
    // than none, because the mean behind it would look guarded when it is not.
    + (a.untested
        ? '<span class="an-chip an-chip-un" title="reads in a cell with too few '
          + 'runs at that model, document and Detail for the outlier test to '
          + 'mean anything — a two-read cell cannot be tested at all. They are '
          + 'kept, so a runaway in one of them is still in the figures above. '
          + 'Repeat a setting to make its cell testable.">'
          + a.untested + ' not tested</span>'
        : '')
    + (hw ? '<span class="an-marks-sep"></span>' + hw : '')
    + (st ? '<span class="an-marks-sep"></span>' + st : '')
    + '<span class="an-inf" title="GPU/CPU is inferred from the decode rate and '
    + 'cold/warm from prefill — neither is recorded by the server">inferred</span>'
    + '</div>';
}

/* A mean with its run count under it, the shape every figure on this card uses:
   a mean with no denominator is not a figure anyone can read, and these cells
   are often one or two runs deep. */
function anFig(f, fmt) {
  if (!f || f.mean === null || f.mean === undefined)
    return '<span class="an-none">—</span>';
  return '<b>' + fmt(f.mean) + '</b><span class="an-t">n=' + f.runs
    + (f.sd !== null && f.sd !== undefined ? ' ±' + Number(f.sd).toFixed(1) : '')
    + '</span>' + anRange(f, fmt);
}
/* The range under the mean. **Printed even when it is a single run**, as a
   dash — a cell that cannot show a spread should say so rather than leave the
   reader wondering whether it was omitted or was zero. Suppressed only when min
   and max are the same value, where a range says nothing the mean did not. */
function anRange(f, fmt) {
  if (!f || f.min === null || f.min === undefined) return '';
  if (f.min === f.max) return '';
  return '<span class="an-t an-rng">' + fmt(f.min) + '–' + fmt(f.max) + '</span>';
}
/* Mean, SD and range as four cells of one row, for the spread tables. Kept apart
   from `anFig` because a table whose whole point is the four figures should put
   them in four sortable columns, not stack them in one. */
function anStatCols(prefix, label, pick, fmt, colour) {
  const val = (f, k) => (f && f[k] !== null && f[k] !== undefined) ? f[k] : null;
  return [
    {key: prefix + '_mean', label: label + ' avg', n: true,
     title: 'mean ' + label.toLowerCase() + ' over this row’s reads',
     value: e => val(pick(e), 'mean'),
     cell: e => {
       const f = pick(e), m = val(f, 'mean');
       if (m === null) return '<span class="an-none">—</span>';
       return '<b' + (colour ? ' style="color:' + accColour(m / 100) + '"' : '')
         + '>' + fmt(m) + '</b><span class="an-t">n=' + f.measured + '</span>';
     }},
    {key: prefix + '_sd', label: label + ' SD', n: true,
     title: 'sample standard deviation — blank where the row has a single read, '
          + 'because one read has no spread (which is not the same as zero)',
     value: e => val(pick(e), 'sd'),
     cell: e => {
       const s = val(pick(e), 'sd');
       return s === null ? '<span class="an-none">—</span>'
         : '<b>' + fmt(s) + '</b>';
     }},
    {key: prefix + '_min', label: label + ' min', n: true,
     value: e => val(pick(e), 'min'),
     cell: e => {
       const v = val(pick(e), 'min');
       return v === null ? '<span class="an-none">—</span>' : fmt(v);
     }},
    {key: prefix + '_max', label: label + ' max', n: true,
     value: e => val(pick(e), 'max'),
     cell: e => {
       const v = val(pick(e), 'max');
       return v === null ? '<span class="an-none">—</span>' : fmt(v);
     }}
  ];
}
/* The two inferred marks as a compact cell: the split where a row mixes them,
   the single value where it does not. `unknown` is shown rather than dropped --
   a row half of whose reads carry no rate is not a row whose hardware is known. */
function anMarkCell(counts, kind) {
  const keys = Object.keys(counts || {});
  if (!keys.length) return '<span class="an-none">—</span>';
  if (keys.length === 1) return esc(anMarkLabel(kind, keys[0]));
  return keys.sort((a, b) => counts[b] - counts[a])
    .map(k => esc(anMarkLabel(kind, k)) + '&nbsp;<b>' + counts[k] + '</b>')
    .join('<span class="an-t"></span>');
}
function anPct(v) { return Number(v).toFixed(1) + '%'; }

/* ---- Spread per document and per model ----
   Mean, SD, min and max for both numbers, at the user's request. Everything
   here could be read off the model × document grid by eye; what a row adds is
   the POOLED figure for one document across every model that read it (or one
   model across every document), which the grid deliberately does not compute.

   **The range is not decoration beside the SD.** An SD over three reads is
   barely a statistic, while min and max are facts about the reads rather than
   estimates from them, and the two disagree usefully: a tight SD with a wide
   range is one stray read. The marks ride along per row so the inferred pair is
   visible where it might vary, not only in aggregate. */
function anSpread(a) {
  const build = (id, rows, label, hint) => {
    if (!rows || !rows.length) return '';
    const cols = [
      {key: 'name', label: label, value: e => csModel(e.key),
       cell: e => '<b title="' + esc(e.key) + '">' + esc(csModel(e.key)) + '</b>'
         + '<span class="cs-set">' + e.runs + ' read(s) · ' + e.others + ' '
         + (label === 'Document' ? 'model' : 'document') + (e.others === 1 ? '' : 's')
         + (e.details && e.details.length
             ? ' · ' + e.details.map(esc).join(', ') : '') + '</span>'}
    ]
      .concat(anStatCols('acc', 'Accuracy', e => e.accuracy, anPct, true))
      .concat(anStatCols('t', 'Time', e => e.seconds, anSecs))
      .concat([
        {key: 'marks', label: 'Run type', sortable: false,
         title: 'GPU/CPU and cold/warm, both inferred — a split is shown where '
              + 'this row mixes them',
         cell: e => '<span class="an-t2">' + anMarkCell(e.hardware, 'hardware')
           + '</span><span class="an-t">' + anMarkCell(e.start, 'start') + '</span>'}
      ]);
    return '<p class="an-h">' + hint + '</p>'
      + lvTable(id, rows, cols, {sort: {key: 'name', dir: 'asc'}});
  };
  const doc = build('an-sp-case', a.case_stats, 'Document',
    'Spread per document — every model that read it');
  const mod = build('an-sp-model', a.model_stats, 'Model',
    'Spread per model — every document it read');
  if (!doc && !mod) return '';
  return doc + mod
    + '<p class="note"><b>Average, SD, min and max</b> for both numbers. The SD '
    + 'is a sample one and is blank on a single read — one read has no spread, '
    + 'which is not the same claim as a spread of zero. <b>Read the range beside '
    + 'the SD, not instead of it</b>: min and max are facts about the reads while '
    + 'an SD over three of them is barely a statistic, and a tight SD with a wide '
    + 'range is one stray read. A document row pools every model that read it and '
    + 'a model row pools every document it read, so the other axis’s count is on '
    + 'every row — a mean over one model, or one document, is not a general '
    + 'claim. Outliers are already excluded. Every column sorts.</p>';
}

/* ---- What each inferred mark is worth ----
   The marks shown as figures rather than only used to split a correlation. A
   single-row mark is the expected case on a one-machine log and is still worth
   printing: it says outright that the mark explains nothing here, which a
   missing table would leave the reader to infer. */
function anMarkStats(a) {
  const rows = a.mark_stats || [];
  if (!rows.length) return '';
  const cols = [
    {key: 'mark', label: 'Mark', value: e => e.kind + e.key,
     cell: e => '<b>' + esc(anMarkLabel(e.kind, e.key)) + '</b>'
       + '<span class="cs-set">' + (e.kind === 'hardware' ? 'hardware' : 'start')
       + ' · ' + e.runs + ' read(s) · ' + e.models + ' model'
       + (e.models === 1 ? '' : 's') + ', ' + e.documents + ' document'
       + (e.documents === 1 ? '' : 's') + '</span>'}
  ]
    .concat(anStatCols('acc', 'Accuracy', e => e.accuracy, anPct, true))
    .concat(anStatCols('t', 'Time', e => e.seconds, anSecs));
  // Whether either mark actually splits this log. A mark with one bucket cannot
  // explain anything, and saying so is more useful than a table that looks like
  // a comparison but has nothing to compare.
  const kinds = {};
  rows.forEach(r => { kinds[r.kind] = (kinds[r.kind] || 0) + 1; });
  const flat = Object.keys(kinds).filter(k => kinds[k] < 2)
    .map(k => k === 'hardware' ? 'hardware' : 'start');
  return '<p class="an-h">What each run type is worth <span class="an-inf">inferred</span></p>'
    + lvTable('an-marks', rows, cols, {sort: {key: 'mark', dir: 'asc'}})
    + '<p class="note">Each row is every read carrying that mark, so the two '
    + 'kinds are two separate splits of the same reads and their counts both sum '
    + 'to the total. '
    + (flat.length
        ? '<b>This log has only one ' + flat.join(' and one ') + ' value</b>, so '
          + 'that mark explains nothing here and its row is a restatement of the '
          + 'whole set — expected on a single machine, and the reason the '
          + 'correlation split above it is degenerate rather than absent. '
        : '')
    + '<b>Both marks are inferred</b>, not reported: hardware from the decode '
    + 'rate and warmth from prefill. A difference between two rows here is a '
    + 'difference between the reads that happened to carry each mark — the '
    + 'models and documents behind them are not held equal, and the counts on '
    + 'each row are what says so.</p>';
}

/* ---- The excluded reads, shown ----
   An exclusion nobody can inspect is indistinguishable from a bug, and this
   fence is the one thing on the tab that silently changes every figure. Each
   dropped read comes back with the evidence against it: its time, its cell's
   median, the ratio and the MADs. Worst first. */
function anOutliers(a) {
  const rows = a.outliers || [];
  if (!rows.length) return '';
  const cols = [
    {key: 'run', label: 'Excluded read', value: e => csModel(e.model) + e.case,
     cell: e => '<b title="' + esc(e.model) + '">' + esc(csModel(e.model)) + '</b>'
       + '<span class="cs-set">' + esc(e.case) + ' · ' + esc(e.detail || '—')
       + ' · ' + esc(anMarkLabel('hardware', e.hardware || 'unknown'))
       + ', ' + esc(anMarkLabel('start', e.start || 'unknown')) + '</span>'},
    {key: 'time', label: 'Its time', n: true, value: e => e.seconds,
     title: 'wall clock for this read',
     cell: e => '<b style="color:var(--danger)">' + anSecs(e.seconds) + '</b>'},
    {key: 'median', label: 'Cell median', n: true, value: e => e.cell_median,
     title: 'the middle time of the reads at this model, document AND Detail — '
          + 'the group this read was judged against',
     cell: e => anSecs(e.cell_median)},
    {key: 'ratio', label: 'Times off', n: true, value: e => e.cell_ratio,
     title: 'how many times the cell median this read took. Both this and the '
          + 'MAD figure must be past their thresholds before a read is dropped.',
     cell: e => e.cell_ratio === null || e.cell_ratio === undefined
       ? '<span class="an-none">—</span>'
       : '<b>×' + Number(e.cell_ratio).toFixed(2) + '</b>'},
    {key: 'mads', label: 'MADs out', n: true, value: e => e.cell_mads,
     title: 'distance from the cell median in median-absolute-deviations — the '
          + 'scale-free shape test',
     cell: e => e.cell_mads === null || e.cell_mads === undefined
       ? '<span class="an-none">—</span>' : Number(e.cell_mads).toFixed(1)},
    {key: 'char', label: 'It scored', n: true, value: e => e.char,
     title: 'the transcript accuracy of the excluded read — a slow read that '
          + 'still scored well is worth a second look',
     cell: e => e.char === null || e.char === undefined
       ? '<span class="an-none">—</span>'
       : '<b style="color:' + accColour(e.char / 100) + '">' + anPct(e.char) + '</b>'}
  ];
  return '<p class="an-h">Excluded as time outliers — ' + rows.length + ' read(s)</p>'
    + lvTable('an-outliers', rows, cols, {sort: {key: 'ratio', dir: 'desc'}})
    + '<p class="note">These are left out of <b>every figure on this tab</b> and '
    + 'are still in the run log, the raw table and the CSV — nothing is deleted. '
    + 'A read is dropped only when it fails <b>both</b> tests: more than '
    + esc(String(a.outlier_mads || '')) + '×MAD from its cell median '
    + '<i>and</i> at least ×' + esc(String(a.outlier_ratio || '')) + ' (or ÷'
    + esc(String(a.outlier_ratio || '')) + ') of it. The second test is what '
    + 'stops a tightly clustered cell — where the MAD is a fraction of a second '
    + '— flagging ordinary wall-clock jitter as a runaway. <b>The cell includes '
    + 'the Detail</b>, because raising Detail legitimately costs ×1.5 to ×2.6 of '
    + 'the clock: judging an <code>original</code> read against a median made of '
    + '<code>low</code> ones would drop it for being exactly what it should be, '
    + 'and would make <code>original</code> look cheaper than it is. '
    + (a.untested
        ? '<b>' + a.untested + ' read(s) were not tested at all</b> — their cell '
          + 'has too few runs at that model, document and Detail for the test to '
          + 'mean anything, and a two-read cell cannot be tested in principle '
          + '(the median sits between the pair, so neither is far from it however '
          + 'different they are). Those reads are kept, so a runaway among them is '
          + 'still in the figures above; repeating a setting is what makes its '
          + 'cell testable. '
        : '')
    + 'Set <code>ANALYSIS_OUTLIER_MADS=0</code> to turn the fence off entirely.</p>';
}

/* ---- Detail vs accuracy: what raising Detail BUYS ----
   Drawn as a model × Detail grid rather than three pooled numbers, because the
   pooled trend is a mixing effect: the models have not been run equally at each
   preset, so pooling ranks whichever models were run most at each. Measured on
   this log the pooled figures fall steadily with more pixels while the main OCR
   model is flat across all three and two others collapse only at `original` --
   the same trap the accuracy-vs-time split exists for. The pooled row is kept as
   a row of the same table, marked, so it can be compared with the models rather
   than read instead of them. */
function anDetailAcc(a) {
  const details = a.details || [], rows = a.acc_detail || [];
  if (!details.length || !rows.length) return '';
  const at = {};
  rows.forEach(r => { at[r.detail] = r; });
  const models = Object.keys((rows[0] || {}).by_model || {}).sort();

  const head = '<tr><th class="an-corner">Model \\ Detail</th>'
    + details.map(d => '<th title="' + esc(anDetailHint(d)) + '">' + esc(d)
        + '</th>').join('')
    + '<th title="best minus worst across the presets this model ran at">Range</th></tr>';

  const body = models.map(m => {
    const vals = details.map(d => ((at[d] || {}).by_model || {})[m] || null);
    const real = vals.filter(v => v && v.mean !== null && v.mean !== undefined);
    const cells = vals.map(v => v && v.mean !== null && v.mean !== undefined
      ? '<td class="an-cell" style="background:' + hexA(accColour(v.mean / 100), 0.16)
        + '"><b style="color:' + accColour(v.mean / 100) + '">' + anPct(v.mean)
        + '</b><span class="an-t">n=' + v.runs + '</span></td>'
      : '<td class="an-empty">—</td>').join('');
    // The spread across presets is the per-model finding: near zero means Detail
    // does not matter for this model, and a big one means it does.
    let range = '<td class="an-mean-col an-empty">—</td>';
    if (real.length > 1) {
      const hi = Math.max(...real.map(v => v.mean)), lo = Math.min(...real.map(v => v.mean));
      const d = hi - lo;
      range = '<td class="an-mean-col"><b style="color:'
        + anTierColour(d >= 20 ? 3 : d >= 8 ? 2 : d >= 3 ? 1 : 0) + '">'
        + d.toFixed(1) + ' pts</b></td>';
    }
    return '<tr><td class="an-rowh" title="' + esc(m) + '">' + esc(csModel(m))
      + '</td>' + cells + range + '</tr>';
  }).join('');

  // **There is no pooled "All models" row, and its removal is the point**
  // (2026-08-25, at the user's request). It averaged every model at a preset,
  // which is a cross-model accuracy figure -- the comparison this tab explicitly
  // defers to Full rank, Best reading and Per document. It was also the most
  // misleading number on the page: pooled, accuracy fell steadily with more
  // pixels, while the main OCR model is flat across all three presets and the
  // decline was two other models collapsing at `original`. A row that has to be
  // argued with in its own tooltip is a row to delete.
  return '<p class="an-h">Detail vs accuracy — what more pixels buy</p>'
    + '<div class="an-grid"><table class="an-heat">' + head + body
    + '</table></div>'
    + '<p class="note">Presets run cheapest first, so read each row left to right '
    + 'as <b>more pixels →</b>. <b>Read along a row, never down a column</b> — a '
    + 'row is one model across the presets, which is the comparison this tab is '
    + 'for; a column is several models that were not run equally at each preset, '
    + 'and ranking them is another tab’s job. This is why there is no pooled '
    + '“all models” row: it fell steadily with more pixels while the main OCR '
    + 'model is flat across all three, and the decline was two other models '
    + 'collapsing at <code>original</code>. <b>Range</b> is how much that model '
    + 'moved across the presets it ran at: near zero means Detail does not decide '
    + 'accuracy for it, a large one means it does. More pixels is <b>not '
    + 'monotonically better</b> — past roughly 4 MP the extra visual tokens '
    + 'degrade some models rather than helping. <b>Each row pools that model’s '
    + 'documents</b>, so a row is that model over everything it read at each '
    + 'preset, not one page.</p>';
}
function anDetailHint(d) {
  return {low: 'low — 2 MP', medium: 'medium — 4 MP, the default',
          original: 'original — uncapped, the page at native resolution'}[d] || d;
}

/* ---- Time vs Detail per document: what raising Detail COSTS ----
   Per document rather than pooled, because time scales with the page: a pooled
   time column would mostly rank the fixtures by length and say nothing about the
   preset. The factor column prices the whole climb in one number, which is the
   thing anyone choosing a preset actually wants. */
function anTimeDetail(a) {
  const details = a.details || [], rows = a.time_detail_by_case || [];
  if (!details.length || !rows.length) return '';
  const cols = [
    {key: 'doc', label: 'Document', value: e => e.case,
     cell: e => '<b>' + esc(e.case) + '</b><span class="cs-set">' + e.runs
       + ' read(s)</span>'}
  ];
  details.forEach(d => cols.push({
    key: 'd_' + d, label: d, n: true,
    title: anDetailHint(d) + ' — mean wall clock for the whole document',
    value: e => (e.cells[d] || {}).mean,
    cell: e => anFig(e.cells[d], anSecs)
  }));
  cols.push({key: 'factor', label: 'Cost of the climb', n: true,
    title: 'how many times longer the dearest preset this document ran at took '
         + 'than the cheapest one',
    value: e => e.factor,
    cell: e => e.factor === null || e.factor === undefined
      ? '<span class="an-none">one preset only</span>'
      : '<b style="color:' + anTierColour(e.factor >= 3 ? 3 : e.factor >= 2 ? 2 : 1)
        + '">×' + Number(e.factor).toFixed(2) + '</b><span class="an-t">'
        + esc(e.from) + ' → ' + esc(e.to) + '</span>'});
  return '<p class="an-h">Time vs Detail, per document — what more pixels cost</p>'
    + lvTable('an-time-detail', rows, cols, {sort: {key: 'doc', dir: 'asc'}})
    + '<p class="note">Per document because <b>time scales with the page</b> — '
    + 'pooling these would mostly rank the fixtures by length rather than say '
    + 'anything about the preset. Each cell is the mean wall clock over that '
    + 'document’s recent reads at that preset, across whatever models read it, so '
    + 'a cell mixing a fast and a slow model is an average of both. Read it '
    + 'beside the accuracy table above: those two are the cost and the benefit of '
    + 'the same knob, and they are deliberately separate because the trade is '
    + 'not monotone. Every column sorts.</p>';
}

/* ---- Time per model, with accuracy beside it ----
   Accuracy rides along for the reason `by_case`'s speed column carries it: fast
   is a claim about cost alone, and the cheapest model here is not the one to
   run. */
function anModelTime(a) {
  const rows = a.model_time || [];
  if (!rows.length) return '';
  const cols = [
    {key: 'model', label: 'Model', value: e => csModel(e.model),
     cell: e => '<b title="' + esc(e.model) + '">' + esc(csModel(e.model)) + '</b>'
       + '<span class="cs-set">' + e.seconds.runs + ' read(s) · ' + e.documents
       + ' document(s)</span>'},
    {key: 'time', label: 'Mean time', n: true, value: e => e.seconds.mean,
     title: 'wall clock for the whole document, meaned over this model’s reads',
     cell: e => anFig(e.seconds, anSecs)},
    {key: 'acc', label: 'Accuracy', n: true, value: e => e.accuracy.mean,
     title: 'transcript accuracy over the same reads — fast is a claim about '
          + 'cost alone, so it is never printed without this beside it',
     cell: e => e.accuracy.mean === null || e.accuracy.mean === undefined
       ? '<span class="an-none">—</span>'
       : '<b style="color:' + accColour(e.accuracy.mean / 100) + '">'
         + anPct(e.accuracy.mean) + '</b><span class="an-t">n='
         + e.accuracy.runs + '</span>'},
    {key: 'perpt', label: 'Seconds per point', n: true,
     title: 'mean time divided by mean accuracy — what a point of transcript '
          + 'accuracy costs on this model. Lower is better value; it is a ratio '
          + 'of two means, not a measured rate.',
     value: e => (e.seconds.mean && e.accuracy.mean) ? e.seconds.mean / e.accuracy.mean : null,
     cell: e => (e.seconds.mean && e.accuracy.mean)
       ? '<b>' + (e.seconds.mean / e.accuracy.mean).toFixed(2) + 's</b>'
       : '<span class="an-none">—</span>'}
  ];
  return '<p class="an-h">Time per model</p>'
    + lvTable('an-model-time', rows, cols, {sort: {key: 'time', dir: 'asc'}})
    + '<p class="note">Meaned over each model’s recent reads, outliers excluded. '
    + '<b>The models have not all read the same documents</b> — the document '
    + 'count is on every row for that reason, and a model that has only read the '
    + 'short fixtures will look fast. <b>Seconds per point</b> divides the two '
    + 'means to price what accuracy costs on each model; it is a ratio of means '
    + 'rather than a measured rate, so read it as a rough value ranking and not '
    + 'as a per-run figure. Every column sorts.</p>';
}

// The model x document grid the numbers above summarise, with a per-model and
// per-document average in the margins (unweighted over the documents / models,
// the same "average over documents" the setting tables use).
function anHeat(a) {
  const models = a.models || [], cases = a.cases || [];
  if (!models.length || !cases.length) return '';
  const at = {};
  (a.cells || []).forEach(c => { (at[c.model] = at[c.model] || {})[c.case] = c; });

  const head = '<tr><th class="an-corner">Model \\ Document</th>'
    + cases.map(c => '<th>' + esc(c) + '</th>').join('')
    + '<th title="this model’s mean accuracy over the documents it read">Avg</th></tr>';

  const body = models.map(m => {
    const rowCells = cases.map(c => {
      const cell = (at[m] || {})[c];
      if (!cell || cell.char_mean === null || cell.char_mean === undefined)
        return '<td class="an-empty">—</td>';
      const acc = cell.char_mean;
      const tip = m + ' · ' + c + ' — ' + cell.runs + ' read(s)'
        + (cell.char_sd !== null && cell.char_sd !== undefined
           ? ', ±' + Number(cell.char_sd).toFixed(1) + '% accuracy' : '')
        + ', ' + anSecs(cell.time_mean) + ' mean';
      return '<td class="an-cell" title="' + esc(tip) + '" style="background:'
        + hexA(accColour(acc / 100), 0.16) + '"><b style="color:'
        + accColour(acc / 100) + '">' + acc.toFixed(0) + '%</b>'
        + '<span class="an-t">' + anSecs(cell.time_mean) + '</span></td>';
    }).join('');
    const rowMean = anMeanOf(cases.map(c => {
      const cell = (at[m] || {})[c];
      return cell ? cell.char_mean : null;
    }));
    const meanCell = rowMean === null ? '<td class="an-mean-col an-empty">—</td>'
      : '<td class="an-mean-col"><b style="color:' + accColour(rowMean / 100) + '">'
        + rowMean.toFixed(0) + '%</b></td>';
    return '<tr><td class="an-rowh" title="' + esc(m) + '">' + esc(csModel(m))
      + '</td>' + rowCells + meanCell + '</tr>';
  }).join('');

  // The per-document average, over the models that read it.
  const foot = '<tr class="an-mean-row"><td class="an-rowh">Avg</td>'
    + cases.map(c => {
        const mean = anMeanOf(models.map(m => {
          const cell = (at[m] || {})[c];
          return cell ? cell.char_mean : null;
        }));
        return mean === null ? '<td class="an-empty">—</td>'
          : '<td><b style="color:' + accColour(mean / 100) + '">'
            + mean.toFixed(0) + '%</b></td>';
      }).join('')
    + '<td class="an-mean-col"></td></tr>';

  return '<div class="an-grid"><table class="an-heat">' + head + body + foot
    + '</table></div>';
}


/* ---------- which setting to run at all ---------- */
/* The per-document table above answers "what should I run on THIS file"; these
   two answer "what should I run", which it scatters across one row per document.

   TWO tables, not one, because the two passes are scored on different things and
   paid for differently: pass 1 is judged on the transcript and costs prefill and
   decode, pass 2 is judged on whether a value reached the key it belongs in and
   costs neither. A single row averaging both would answer neither question.

   Everything here is computed in `runlog.by_ocr` and `runlog.by_extract`; the
   rules about what is eligible live there, beside the file format they read. */

/* One aggregate figure with its two spreads under it.

   The spreads are labelled rather than merged into one ±, because they are not
   the same claim: `repeat` is how far the setting moved when it was run twice on
   ONE document -- reproducibility, which under greedy decoding should be near
   zero -- while `across docs` is how much the documents differ from each other,
   which is a property of the fixtures as much as of the setting. Printing one
   number would invite reading it as an error bar on the mean, which neither is. */
function msFig(block, unit, digits, colour, outer) {
  if (!block || block.mean === null || block.mean === undefined)
    return '<span class="cs-none">&mdash;</span>';
  const d = digits === undefined ? 1 : digits;
  const text = Number(block.mean).toFixed(d) + (unit || '');
  const head = '<span class="cs-val"'
    + (colour ? ' style="color:' + accColour(Number(block.mean) / 100) + '"' : '')
    + '>' + text + '</span>';
  const sd = [];
  if (block.sd_repeat !== null && block.sd_repeat !== undefined)
    sd.push('&plusmn;' + Number(block.sd_repeat).toFixed(d) + ' repeat');
  // What the outer spread is ACROSS is not always documents: the same figure
  // grouped by document is spread across the models that read it, and calling
  // that "across docs" would name the wrong axis on half the tables that use
  // this. Defaulted, because it is documents everywhere it was before.
  if (block.sd_case !== null && block.sd_case !== undefined)
    sd.push('&plusmn;' + Number(block.sd_case).toFixed(d) + ' '
            + (outer || 'across docs'));
  return head + (sd.length ? '<span class="cs-set">' + sd.join(' &middot; ') + '</span>' : '');
}

/* The setting itself: the model on top, the knobs that make it a distinct thing
   to run underneath. The full model name is the tooltip, because two `hf.co/...`
   repacks of one model differ only in the part `csModel` drops. */
function msCell(e, knobs, extra) {
  // A run that failed before it reached a model has none recorded. Named rather
  // than rendered as an empty cell: the row is a real record of real failures,
  // and a blank there reads as a rendering bug instead of as what it is.
  //
  // Returns the cell's CONTENTS, not the `<td>`: the sortable table renderer
  // owns the element, so a column cannot quietly grow a second class attribute.
  return '<b title="' + esc(e.model || '') + '">'
    + (e.model ? esc(csModel(e.model))
               : '<span class="cs-none">model not recorded</span>') + '</b>'
    // Which pipeline this setting belongs to, beside its name. The two are on
    // one scale by design -- that is the point of storing them together -- and a
    // reader comparing a 90% row with an 86% one has to be able to see which is
    // which without decoding `paddleocr` out of the backend column.
    + engineTag(e.ocr_engine)
    + '<span class="cs-set">' + knobs.filter(Boolean).map(esc).join(' &middot; ') + '</span>'
    + '<span class="cs-set">' + e.runs + ' run(s) &middot; ' + e.documents
    + ' document(s)</span>'
    // Already-built markup, not text: the caller passes a coloured warning, so
    // this one is deliberately not escaped. Nothing user-supplied reaches it.
    + (extra ? '<span class="cs-set">' + extra + '</span>' : '');
}

function paintOcrSummary(t) {
  const settings = (t.by_ocr) || [];
  // An empty table with retired rows behind it is the one case where the note
  // matters most -- every read this log holds ran at a preset that is gone, and
  // without it the card reads as a log nobody has written to.
  if (!settings.length) { ocrSum.innerHTML = emptyPassNote('ocr', t) + retiredNote(t) + legacyNote(t); return; }
  // The server's ranking, kept before any sort: the top row is the best setting
  // by transcript accuracy, and it stays marked as such however the reader
  // reorders the table. A highlight that followed the sort would say "best" of
  // whatever happens to be on top, which is a different and false claim.
  const top = settings[0];
  const cols = [
    {key: 'setting', label: 'Setting', value: e => csModel(e.model),
     cell: e => msCell(e, [e.quant, e.engine || e.backend, e.detail, e.ocr_profile],
             // Reads that did not finish, or finished with nothing. Beside the
             // setting rather than under the clock, because it qualifies every
             // other number on the row: a mean taken over runs that mostly
             // looped is a mean of the survivors.
             //
             // A failure is a loop or a crash. A run that finished and read the
             // page badly is neither: it is in the mean, as the score it got.
             // The two are printed side by side and never added up -- they are
             // found and fixed in different places.
             [e.failed
                ? '<span style="color:var(--danger)">' + e.failed + ' of ' + e.runs
                  + ' incomplete (' + Number(e.failure_rate).toFixed(0) + '% failure)'
                  + '</span>'
                : '',
              e.empty
                ? '<span class="cs-none">' + e.empty + ' finished empty (in the mean '
                  + 'as 0%)</span>'
                : ''].filter(Boolean).join(' &middot; '))},
    {key: 'char', label: 'Transcript', n: true, value: e => e.char.mean,
     title: 'character accuracy against solution/<id>.md, meaned over documents '
          + '— content in whatever order the model put it',
     cell: e => msFig(e.char, '%', 1, true)},
    // The one number a headline cannot give: WHICH of the three scripts this
    // setting is losing. Numbers first, against the order used elsewhere,
    // because every Mandatory field in the requirement is a figure, a date or an
    // ID -- a setting can read 92% overall and 71% of the digits, and only this
    // column says so. Blank where the documents printed too little of a script
    // to measure it, never zero.
    {key: 'num', label: 'Numbers', n: true, value: e => e.digit.mean,
     title: 'digits only, Arabic and Thai numerals — amounts, dates, tax IDs '
          + 'and document numbers. Read this column first: the three script '
          + 'rates do not add up to the transcript score, they are recall over '
          + 'three subsets of it',
     cell: e => msFig(e.digit, '%', 1, true)},
    {key: 'thai', label: 'Thai', n: true, value: e => e.thai.mean,
     title: 'Thai letters, tone marks and vowels included — the first thing to '
          + 'go when a page is downscaled',
     cell: e => msFig(e.thai, '%', 1, true)},
    {key: 'eng', label: 'English', n: true, value: e => e.latin.mean,
     title: 'the Latin alphabet — not a claim that the text is English, since a '
          + 'romanised Thai name is Latin script too',
     cell: e => msFig(e.latin, '%', 1, true)},
    // Prefill and decode apart, never summed. They move for different reasons --
    // prefill with pixels, decode with output length -- and the entire Detail
    // trade lives in the first of them, invisible in a total.
    {key: 'prefill', label: 'Prefill', n: true, value: e => e.prefill.mean,
     title: 'vision tower plus prompt encode, paid once per page before the '
          + 'first token — this is what scales with Detail',
     cell: e => msFig(e.prefill, 's', 1)},
    {key: 'decode', label: 'Decode', n: true, value: e => e.decode.mean,
     title: 'generating the transcript; the rate excludes prefill, so it '
          + 'compares across pages of different resolutions',
     cell: e => msFig(e.decode, 's', 1)
       + '<span class="cs-set">'
       + (e.tps.mean === null ? '&mdash;' : Number(e.tps.mean).toFixed(1) + ' tok/s')
       + '</span>'},
    {key: 'total', label: 'Total', n: true, value: e => e.seconds.mean,
     title: 'wall clock for the whole document',
     cell: e => msFig(e.seconds, 's', 1)},
    {key: 'failure', label: 'Failed', n: true, value: e => e.failure_rate,
     title: 'runs that did not finish — looped, truncated, cancelled or errored. '
          + 'A run that finished and scored badly is not one of these.',
     cell: e => e.failed
       ? '<span style="color:var(--danger)">' + Number(e.failure_rate).toFixed(0)
         + '%</span><span class="cs-set">' + e.failed + ' of ' + e.runs + '</span>'
       : '<span class="cs-none">0 of ' + e.runs + '</span>'}
    // **There is no Runs column**, at the user's request. This cell already
    // carries the run count as its denominator, and the setting cell on the left
    // carries it again with the document count beside it -- so a third copy in a
    // column of its own said nothing the row was not already saying twice.
  ];
  ocrSum.innerHTML = '<span class="cs-title">Pass 1 &mdash; best setting to read a page'
    + windowNote('setting') + '</span>'
    + pipeNote('ocr')
    + lvTable('ocr', settings, cols, {
        sort: {key: 'char', dir: 'desc'},
        rowClass: e => e === top ? ' class="ms-top"' : ''
      })
    + '<p class="note">A setting is the model, backend, Detail and pass-1 profile '
    + 'together &mdash; the same model at a different Detail is a different thing '
    + 'to run, and the profile decides the prompt and the system slot, which on '
    + 'some models is the difference between a transcript and an empty string at '
    + 'HTTP&nbsp;200. Repeats of one document are averaged before the documents '
    + 'are, so a setting run five times on one page is not five samples of it. '
    + '<b>llama.cpp caches prompts</b>, so a prefill figure that includes a '
    + 're-read of a page it had already seen is lower than anyone gets cold. '
    + 'Every column sorts; the highlighted row is the best by transcript '
    + 'accuracy whichever column you sort by.</p>'
    + retiredNote(t)
    + legacyNote(t);
}


/* Rows the log holds at a Detail this build no longer offers.
   Said here rather than left implicit: this table and the per-document one both
   drop those rows, and a table quietly missing a quarter of the log reads as a
   table nobody has run much. `max`, `accurate` and `balanced` are NOT in here --
   they were renamed, the budget did not move, and their rows are folded onto
   `original`/`medium`/`low` and counted. Only a preset that was DELETED lands
   here, and there is one: `fast`, 1 MP, dropped because at that budget the model
   fabricates rather than misreads. Folding it into `low` would average 1 MP and
   2 MP under one name. */
/* How many reads in view still carry the OLD character score -- an edit distance,
   which charged text the page does not print exactly as much as text it lost.
   Since 2026-09-04 the score is recall of the ground truth and that text is free,
   which on this project's own fixtures is worth about +7.6 points of mean.

   Said out loud for the reason `retiredNote` is: while both eras sit in one
   window the pass-1 means are over two different questions, and a table that does
   not admit it reads as a model that suddenly got better. **The log was
   deliberately not reset** -- every table covers each setting's own most recent
   runs, so the old rows leave on their own as new ones arrive, and this note
   empties itself with them.

   The test is exact and free: `invented_chars` was added in the same change, so
   on a row that has a transcript score at all, a blank one is the old metric. */
function legacyNote(t) {
  const g = t.legacy_char || {};
  if (!g.legacy || !g.scored) return '';
  const pct = (g.legacy / g.scored) * 100;
  return '<p class="note"><b>' + g.legacy + ' of ' + g.scored + ' scored read(s)</b> ('
    + pct.toFixed(0) + '%) still carry the character score as it was computed before '
    + '2026-09-04 &mdash; an edit distance, which charged text the page does not '
    + 'print. It is recall of the ground truth now, so those rows read '
    + '<b>lower</b> than the same transcript would score today. Nothing was reset: '
    + 'each table covers its own setting&rsquo;s most recent runs, so these leave as '
    + 'new ones arrive and this note goes with them. Until then, treat a rise in a '
    + 'mean that mixes the two as the metric moving rather than the model.</p>';
}

function retiredNote(t) {
  const gone = t.retired_detail || {};
  const names = Object.keys(gone);
  if (!names.length) return '';
  const parts = names.map(n => gone[n] + ' at <b>' + esc(n) + '</b>');
  return '<p class="note">' + parts.join(', ')
    + ' &mdash; ' + (names.length > 1 ? 'presets' : 'a preset')
    + ' this build no longer offers. Those runs are real and stay in the log, in '
    + 'the model and document rankings and in the counts above; they are left out '
    + 'of this table and of the per-document bests, because a setting nobody can '
    + 'select is not one to recommend.</p>';
}

function paintExtractSummary(t) {
  const settings = (t.by_extract) || [];
  if (!settings.length) { extractSum.innerHTML = emptyPassNote('extract', t); return; }
  const scored = settings.some(e => e.scored_runs);
  // The server's ranking, held before any sort. See `paintOcrSummary`.
  const top = scored ? settings[0] : null;
  const cols = [
    {key: 'setting', label: 'Setting', value: e => csModel(e.model),
     cell: e => msCell(e, [e.quant, e.engine || e.backend, e.extract_mode,
                           isStepMode(e.extract_mode)
                             ? (e.extract_schema_first === '1' ? 'Schema first' : 'JSON first') : ''],
             // Two lines, both about what the row does not otherwise say.
             //
             // `read_by` names the model that produced the transcript, where
             // that is a different model from the one extracting. It matters
             // because a field score taken over a real transcript is partly a
             // measurement of PASS 1 -- a value the OCR misread cannot then be
             // extracted correctly -- so the row is only as good as the read
             // under it.
             ((e.read_by || []).length
               ? 'reading: ' + e.read_by.map(m => esc(csModel(m))).join(', ')
               : '')
             // Runs whose reply cycled. They are dropped from the extras contest
             // rather than allowed to win it, so without this line they would
             // read as a setting that simply had less to say.
             + (e.looped
               ? (((e.read_by || []).length) ? '<br>' : '')
                 + '<span style="color:var(--danger)">' + e.looped + ' of ' + e.runs
                 // "incomplete", not "looped": this counts replies that cycled
                 // AND runs that never produced a reply at all, and calling an
                 // outright error a loop sends anyone reading it to the wrong
                 // place.
                 + ' incomplete (' + Number(e.failure_rate).toFixed(0) + '% failure)</span>'
               : ''))},
    // The headline: priority-1 values that are the value the truth file says
    // belongs in that key. The denominator is about 12 and is per document, so
    // the raw counts come with the rate.
    {key: 'field', label: 'Field accuracy', n: true,
     value: e => e.scored_runs ? e.field.mean : null,
     title: 'priority-1 values that are the value the field ground truth says '
          + 'belongs in that key, as a share of the values that file rules on',
     cell: e => e.scored_runs
       ? msFig(e.field, '%', 1, true)
         // A total ACROSS the scored runs, not a single denominator: two runs of
         // 1/11 read 2/22 here. Said as "over N runs" rather than joined with a
         // middot, which would read as two separate facts.
         + '<span class="cs-set">' + e.correct + '/' + e.expected + ' values over '
         + e.scored_runs + ' scored run(s)</span>'
       // Blank is not zero, the rule the CSV column follows: no field truth file
       // for anything this setting has run on -- or every run of it was read too
       // badly for this view's floor to credit the extractor with the result.
       : '<span class="cs-none">not scored yet</span>'},
    // `other_fields` has no ground truth and cannot have one -- the labels are
    // the model's own wording. So it is scored against the other settings that
    // read the SAME document: points are how many of them you beat there.
    //
    // Shown PER DOCUMENT, because the settings have not entered the same number
    // of contests: +10 from one document and +26 from five are not the same
    // achievement, and the raw totals rank whichever setting was run most.
    {key: 'extras_rank', label: 'Extra fields, ranked', n: true,
     value: e => e.other_rate,
     title: 'other_fields has no ground truth, so it is ranked against the other '
          + 'settings run on the same document: points are how many of them this '
          + 'setting beat, divided by the documents it was compared on',
     cell: e => e.other_contests
       ? '<span class="cs-val">' + Number(e.other_rate).toFixed(1) + '</span>'
         + '<span class="cs-of"> pts/doc</span>'
         + '<span class="cs-set">+' + e.other_points + ' of ' + e.other_possible
         + ' over ' + e.other_contests + ' document(s)</span>'
       : '<span class="cs-none">nothing to compare</span>'},
    // Distinct entries, not the raw total: a reply that repeats one invented
    // line a hundred times returned one field, not a hundred. The raw total is
    // under it, so a gap between the two is visible where it has not yet crossed
    // the threshold that flags the run outright.
    {key: 'extras', label: 'Extra fields', n: true, value: e => e.other_distinct.mean,
     title: 'distinct entries under other_fields, meaned over documents — a reply '
          + 'that repeats one line a hundred times returned one field, not a hundred',
     cell: e => msFig(e.other_distinct, '', 1)
       + (e.other_count.mean !== null && e.other_distinct.mean !== null
          && e.other_count.mean > e.other_distinct.mean
          ? '<span class="cs-set">' + Number(e.other_count.mean).toFixed(1)
            + ' before de-duplication</span>' : '')},
    // Coverage, and deliberately not what the table is ranked on: read it
    // against the score two columns left. Many filled and few correct is a model
    // inventing; few filled is one giving up, and they share an accuracy.
    {key: 'filled', label: 'Filled', n: true, value: e => e.filled.mean,
     // No fixed number: the form is per document type now -- eleven keys on any
     // of the three requirements, sixteen where no requirement applies -- so a
     // mean over rows of different forms has no single denominator to name.
     title: 'how many priority-1 keys came back filled with anything at all — '
          + 'coverage, not correctness. The form is per document type, so rows '
          + 'here can be out of different totals',
     cell: e => msFig(e.filled, '', 1)},
    {key: 'failure', label: 'Failed', n: true, value: e => e.failure_rate,
     title: 'extractions whose reply cycled, and runs that produced no reply at '
          + 'all. Both are counted as failures and neither is scored.',
     cell: e => e.looped
       ? '<span style="color:var(--danger)">' + Number(e.failure_rate).toFixed(0)
         + '%</span><span class="cs-set">' + e.looped + ' of ' + e.runs + '</span>'
       : '<span class="cs-none">0 of ' + e.runs + '</span>'}
    // No Runs column here either -- see `paintOcrSummary`.
  ];
  extractSum.innerHTML = '<span class="cs-title">Pass 2 &mdash; best setting to extract fields'
    + windowNote('setting') + '</span>'
    + pipeNote('extract')
    + lvTable('extract', settings, cols, {
        sort: {key: 'field', dir: 'desc'},
        rowClass: e => e === top ? ' class="ms-top"' : ''
      })
    + '<p class="note"><b>Single, Agentic and Hybrid are separate rows</b>: some '
    + 'models do well in one shape and badly in the other, so averaging the two '
    + 'reports a figure neither produces. Everything else about a setting is '
    + 'pooled, including whether pass&nbsp;2 was fed a ground-truth transcript or '
    + 'one pass&nbsp;1 produced &mdash; worth remembering when reading a low '
    + 'score, because a run fed a real transcript is being marked on the OCR as '
    + 'well as the extraction. That is what the <b>read floor</b> above is for: '
    + 'a field score from a transcript below it is left out, because it measures '
    + 'the read rather than the extractor. Repeats of one '
    + 'document are averaged before the documents are, and the spread of those '
    + 'repeats is printed as <b>&plusmn; repeat</b>. Restricted agentic runs are '
    + 'left out &mdash; they answer part of the form. A <b>partial</b> match '
    + 'scores half a value: the model found the right thing and took too much or '
    + 'too little of it, which is neither a hit nor a miss. A run whose reply '
    + '<b>cycled</b>, and one that produced no reply at all, are counted as '
    + 'failures and <b>not scored</b> &mdash; they stay out of every mean and '
    + 'out of the extras ranking rather than being allowed to win it. A setting '
    + 'whose every run failed shows no score at all. '
    + '<b>Field accuracy</b> and <b>filled</b> are '
    + 'different claims: a run can fill every key and get most of them wrong, '
    + 'which is the whole reason the field ground truth exists.</p>';
}


/* ---------- the presentation summary ----------
   Six blocks on one tab, at the user's request, in the order they were asked
   for: reading ranked, reading timed, Detail priced, extraction ranked, the two
   extraction shapes, extraction timed. Everything comes from
   `runlog.presentation`, which does the grouping and applies the bias.

   **This tab is the one on this card that is meant to be SHOWN**, and three
   things follow from that which the other tabs deliberately do the other way:

   - **No failure flags in the setting cell.** `paintOcrSummary` prints a red
     incomplete count under the model because it qualifies every other figure on
     that row. Here the failure rate is one column and nothing else, as asked --
     a slide with a warning under every row reads as a system that does not work,
     whatever the numbers beside it say.
   - **Its own colour scale**, at the user's request: red under 70, amber to 90,
     green at 90 and above. `accColour`'s 80/95 stays everywhere else, because
     changing it there would silently re-mark the accuracy bars and every other
     table on the page against a scale nobody asked to move.
   - **No prose** (2026-08-25, at the user's request: *less useless text, more
     summary for easy presentation / no text note*). Every explanatory paragraph
     is gone and what it said lives in `title` attributes on the headings and
     column headers instead. The rule for this tab is that a reader gets the
     answer from the figures and the colour, and the caveats on hover -- the
     opposite of the analysis tab, where the caveat IS the finding.

   **Colour is carried by a bar beside every number**, not by more coloured text
   (*more color but keep it formal and clean*). A bar is comparable down a column
   at a glance and stays formal because it is one weight of one hue per column;
   colouring more words would have made the tab louder without making it faster
   to read.

   **Every bar reads "longer is better" except the error column**, which is the
   rate itself and is the only one drawn from the danger end of the scale, so
   that reads as deliberate rather than as an inconsistency. There is no
   inverted column any more: `cost of accuracy` was removed on 2026-08-25 at the
   user's request as useless, and it was -- a ratio of two means that are both
   already columns on the same row, so it added a number to read without adding
   anything to know. */

/* Red under 70, amber to 90, green at 90 and above -- this tab only.
   **Returns CSS variables, not hex**, so the scale darkens on a light page --
   the amber especially, which at #d9a441 is a pale smear on white at the size
   this tab prints numbers. Safe here and NOT safe in `accColour`, which the
   analysis heatmap feeds to `hexA` to make a tint from: that one has to stay a
   parsable hex string. Two functions with two jobs, and the reason is one line
   of arithmetic in `hexA`. */
function presColour(pct) {
  if (pct === null || pct === undefined) return 'inherit';
  if (pct >= 90) return 'var(--ok)';
  if (pct >= 70) return 'var(--warn)';
  return 'var(--danger)';
}

/* An error rate on the same scale read from the other end: 10% error is 90%
   clean and green, over 30% is red. One rule for both columns rather than a
   second set of thresholds nobody could hold in mind beside the first. */
function presErrColour(rate) {
  return rate === null || rate === undefined ? 'inherit' : presColour(100 - rate);
}

/* The magnitude bar under a figure. `share` is 0..1 of the column's own scale,
   so a bar is only ever comparable DOWN its column -- which is the one reading
   it is for. Clamped at both ends: a zero-width bar reads as a missing cell
   rather than as a small value, and an over-long one would break the column. */
function prBar(share, colour) {
  const w = Math.max(3, Math.min(100, (share || 0) * 100));
  return '<span class="pr-bar"><span style="width:' + w.toFixed(1) + '%;background:'
    + colour + '"></span></span>';
}

/* One figure: the number, its bar, and everything else on hover.
   **The spread and the range moved into the tooltip** -- they are detail, and
   this tab is a summary. Nothing is lost: `±sd` and the min-max are still one
   pointer away, and the count that a mean must never be read without is still
   printed under the model name. */
function prFig(block, unit, digits, colour, share) {
  if (!block || block.mean === null || block.mean === undefined)
    return '<span class="cs-none">&mdash;</span>';
  const d = digits === undefined ? 1 : digits;
  const hint = [];
  if (block.sd !== null && block.sd !== undefined)
    hint.push('±' + Number(block.sd).toFixed(d) + ' across documents');
  if (block.min !== null && block.min !== undefined
      && block.max !== null && block.max !== undefined
      && Number(block.max) - Number(block.min) > 0)
    hint.push('range ' + Number(block.min).toFixed(d) + '–' + Number(block.max).toFixed(d));
  if (block.measured) hint.push(block.measured + ' document(s) measured');
  const tone = colour ? colour(Number(block.mean)) : 'var(--accent)';
  return '<span class="cs-val"' + (hint.length ? ' title="' + esc(hint.join(' · ')) + '"' : '')
    + (colour ? ' style="color:' + tone + '"' : '') + '>'
    + Number(block.mean).toFixed(d) + (unit || '') + '</span>'
    + (share === null || share === undefined ? '' : prBar(share, tone));
}

/* The one error column every table here carries. The bar is the rate itself, so
   a long red bar is a bad row -- the only column on the tab where more bar is
   worse, which is why it is the only one drawn from the danger end of the scale.
   The denominator stays printed: a rate with no count under it is not a figure
   anyone can read. */
function prErr(e) {
  if (e.failure_rate === null || e.failure_rate === undefined)
    return '<span class="cs-none">&mdash;</span>';
  const tone = presErrColour(e.failure_rate);
  return '<span class="cs-val" style="color:' + tone + '">'
    + Number(e.failure_rate).toFixed(0) + '%</span>'
    + prBar(e.failure_rate / 100, tone)
    + '<span class="cs-set">' + e.failed + ' of ' + e.runs + '</span>';
}

/* The model, with its rank and its counts. The rank badge is the server's
   ordering, passed in -- it is NOT the row's position after a sort, for the same
   reason the highlight is not: a badge that renumbered when somebody sorted by
   time would be ranking models on the clock. */
function prModel(e, rank) {
  return '<span class="pr-rank' + (rank === 1 ? ' pr-rank-1' : '') + '">'
    + (rank || '·') + '</span>'
    + '<span class="pr-name"><b title="' + esc(e.model || '') + '">'
    + esc(csModel(e.model)) + '</b>'
    // What it ran ON -- quantisation and engine version -- where the runs say.
    + ((e.quant || e.engine)
       ? '<span class="cs-set" title="quantisation · inference engine">'
         + [e.quant, e.engine].filter(Boolean).map(esc).join(' &middot; ') + '</span>'
       : '')
    + '<span class="cs-set">' + e.runs + ' run(s) &middot; ' + e.documents
    + ' doc(s)</span></span>';
}

/* Rank by the order the server returned, keyed by the row object, so a column
   spec can look one up without knowing where the row ended up on screen. */
function prRanks(list) {
  const at = new Map();
  (list || []).forEach((e, i) => at.set(e, i + 1));
  return at;
}

/* The largest value in a column, for the bars. Zero and a missing column both
   give 1, so a bar divides by something rather than by nothing. */
function prMax(list, pick) {
  let top = 0;
  (list || []).forEach(e => {
    const v = pick(e);
    if (v !== null && v !== undefined && v > top) top = v;
  });
  return top || 1;
}

/* What the bias dropped, as chips rather than as a paragraph.
   **Nothing at all when the bias is off**, which is the default: a banner
   explaining that a page is showing everything is the "useless text" this tab
   was told to lose. When it IS on, the models it removed have to be named --
   a summary that leaves things out and cannot say what is indistinguishable from
   a broken one -- so they are named, in the smallest form that still names
   them. */
const PR_RULE = {
  single_source: 'failures from one pairing',
  weak_model: 'weak models',
  time_outlier: 'time outliers'
};

function prBiasNote(p) {
  if (!p.biased) return '';
  const why = {};
  ['ocr', 'extract'].forEach(pass => (p.weak[pass] || []).forEach(w => {
    why[w.model] = (why[w.model] ? why[w.model] + '; ' : '')
      + (pass === 'ocr' ? 'reading' : 'extraction (Extra fields '
        + (w.extras === false ? 'off' : 'on') + ')') + ': ' + w.reason;
  }));
  const chips = Object.keys(why).sort().map(m =>
    '<span class="pr-chip pr-chip-out" title="' + esc(why[m]) + '">'
    + esc(csModel(m)) + '</span>').join('');
  const counts = (p.bias || []).map(s =>
    '<span class="pr-chip" title="' + esc(PR_RULE[s.rule] || s.rule)
    + (s.pass ? ' — ' + esc(s.pass === 'ocr' ? 'reading' : 'extraction') : '')
    + '">' + esc(PR_RULE[s.rule] || s.rule) + '<b>' + s.dropped + '</b></span>').join('');
  return '<div class="pr-excl"><span class="pr-excl-k">Excluded</span>'
    + (chips || '<span class="cs-none">nothing met the rules</span>')
    + counts + '</div>';
}

/* The Detail grids. **Two, not one** (2026-08-25, at the user's request:
   *separate to 2 table, 1 is accuracy another will be time*). It was one grid
   with the accuracy in each cell and the clock in small type under it, plus a
   change column and a cost column side by side -- four numbers per row read in
   two directions, one table doing two jobs.

   Split, each grid asks one question and reads straight across: what did the
   presets SCORE, and what did they COST. Same reasoning the analysis tab's own
   Detail tables follow -- a single table of "Detail vs both" invites a diagonal
   reading that hides the case where more pixels cost more and scored less, and
   here that case IS the finding.

   **Cells are tinted, not just coloured.** The accuracy grid tints by the score
   and the time grid by the cost RELATIVE TO THAT ROW's cheapest preset, so a row
   darkens left to right exactly as far as it actually got dearer. Per row rather
   than per grid, because the models differ in absolute speed by 3x and a
   grid-wide scale would paint the fast model's whole row the same shade.

   Still model x preset, never pooled across models: the pooled row was deleted
   from the analysis tab as the most misleading number on the page. */
function prDetailGrid(p, which) {
  const details = p.ocr_details || [];
  const rows = p.ocr_detail || [];
  if (!details.length || !rows.length) return '';
  const acc = which === 'accuracy';
  const head = '<tr><th class="an-corner">Model</th>'
    + details.map(d => '<th title="the pixel budget the page was rasterised to">'
        + esc(d) + '</th>').join('')
    + (acc
       ? '<th title="what the dearest preset this model ran at scored, minus '
         + 'what the cheapest one scored. Negative means more pixels read WORSE.">'
         + 'Change</th>'
       : '<th title="what the dearest preset this model ran at cost in wall '
         + 'clock, as a multiple of the cheapest">Cost</th>')
    + '</tr>';
  const body = rows.map(r => {
    // The row's own cheapest timed preset, which is what the time tint is a
    // multiple of. Undefined on the accuracy grid, where the tint is the score.
    let base = null;
    if (!acc) {
      details.forEach(d => {
        const c = r.cells[d];
        if (c && c.seconds.mean !== null && base === null) base = c.seconds.mean;
      });
    }
    const cells = details.map(d => {
      const c = r.cells[d];
      const figure = c && (acc ? c.value : c.seconds);
      if (!figure || figure.mean === null)
        return '<td class="an-empty">&mdash;</td>';
      const tint = acc
        ? 'background:' + presTint(figure.mean)
        : 'background:' + prCostTint(base ? figure.mean / base : 1);
      return '<td class="an-cell" style="' + tint
        + (acc ? ';color:' + presColour(figure.mean) : '') + '"'
        + ' title="' + esc(csModel(r.model)) + ' at ' + esc(d) + ' — '
        + c.runs + ' run(s) over ' + c.documents + ' document(s)">'
        + Number(figure.mean).toFixed(1) + (acc ? '%' : 's')
        + '<span class="an-t">' + c.runs + ' run(s)</span></td>';
    }).join('');
    let tail;
    if (acc) {
      // Signed, and green only where more pixels actually bought something. The
      // standing finding is that they usually do not, and a delta drawn in one
      // colour would let that read as noise.
      tail = r.acc_delta === null || r.acc_delta === undefined
        ? '<td class="an-empty">&mdash;</td>'
        : '<td class="an-cell" style="color:' + (r.acc_delta > 0.5 ? 'var(--ok)'
            : r.acc_delta < -0.5 ? 'var(--danger)' : 'inherit') + '"'
          + ' title="' + esc(r.from) + ' → ' + esc(r.to) + '">'
          + (r.acc_delta > 0 ? '+' : '') + Number(r.acc_delta).toFixed(1)
          + '<span class="an-t">' + esc(r.from) + ' → ' + esc(r.to) + '</span></td>';
    } else {
      tail = r.time_factor === null || r.time_factor === undefined
        ? '<td class="an-empty">&mdash;</td>'
        : '<td class="an-cell" style="color:var(--info)" title="' + esc(r.time_from)
          + ' → ' + esc(r.time_to) + '">&times;' + Number(r.time_factor).toFixed(2)
          + '<span class="an-t">' + esc(r.time_from) + ' → ' + esc(r.time_to)
          + '</span></td>';
    }
    return '<td class="an-rowh" title="' + esc(r.model) + '">'
      + esc(csModel(r.model)) + '</td>' + cells + tail;
  }).map(r => '<tr>' + r + '</tr>').join('');
  return '<div class="an-grid"><table class="an-heat">' + head + body + '</table></div>';
}

/* Cell washes for the two grids. Deliberately weak (10-18%): the number on top
   has to stay the thing being read, and a saturated grid is the opposite of
   formal. Both return `color-mix` against the card so the wash sits on the
   panel in either theme rather than on an assumed dark ground. */
function presTint(pct) {
  const tone = pct >= 90 ? 'var(--ok)' : pct >= 70 ? 'var(--warn)' : 'var(--danger)';
  return 'color-mix(in srgb, ' + tone + ' 14%, transparent)';
}

/* The time grid's wash: how many times the row's cheapest preset this cell cost.
   x1 is clear, x4 and over is full strength. */
function prCostTint(factor) {
  const share = Math.max(0, Math.min(1, ((factor || 1) - 1) / 3));
  return 'color-mix(in srgb, var(--info) ' + (share * 22).toFixed(1)
    + '%, transparent)';
}

/* ---------- the headline cards ----------
   **Added 2026-08-25 at the user's request**: *have a summerize banner card for
   best ocr at x% and also have a banner card for my hardware and this env*.

   The six tables answer *which and by how much*; a slide also has to answer
   *what is the number* from across a room, and a table cannot do that. So the
   two results this tab exists to deliver are lifted out at the top, at a size
   that reads without leaning in.

   **The winner is the server's ranking, never the sorted table.** `lvTable`
   reorders on a click and the card must not follow it -- a headline that changed
   when somebody sorted by time would be naming the fastest model "best". Same
   rule as the `ms-top` highlight, which is why both read `[0]` of the list the
   server returned rather than anything in the DOM. */
/* `badge` is already-built markup (the shape pill) or empty; `extra` is a
   leading fact for the sub-line. Both are assembled by the caller with `esc`
   applied to anything that came from the log. */
function prHero(label, entry, unit, extra, badge) {
  const head = '<span class="pr-hero-label">' + esc(label)
    + (badge || '') + '</span>';
  if (!entry || entry.value.mean === null || entry.value.mean === undefined)
    return '<div class="pr-hero pr-hero-none">' + head
      + '<span class="pr-hero-num">&mdash;</span>'
      + '<span class="pr-hero-sub">nothing scored in this view</span></div>';
  const pct = Number(entry.value.mean);
  return '<div class="pr-hero" style="--hero:' + presColour(pct) + '">' + head
    + '<span class="pr-hero-model" title="' + esc(entry.model || '') + '">'
      + esc(csModel(entry.model)) + '</span>'
    + '<span class="pr-hero-num">' + pct.toFixed(1) + (unit || '') + '</span>'
    + ((entry.quant || entry.engine)
       ? '<span class="pr-hero-sub">'
         + [entry.quant, entry.engine].filter(Boolean).map(esc).join(' &middot; ')
         + '</span>'
       : '')
    + '<span class="pr-hero-sub">' + (extra ? extra + ' &middot; ' : '')
      + entry.runs + ' run(s) &middot; ' + entry.documents + ' doc(s)'
      + ' &middot; <span style="color:' + presErrColour(entry.failure_rate) + '">'
      + (entry.failure_rate === null ? '&mdash;'
         : Number(entry.failure_rate).toFixed(0) + '% error')
      + '</span></span></div>';
}

/* Two cards: the best reader, and **the single best (model, shape) pairing**.

   This went through two shapes on 2026-08-25, both at the user's request, and
   the pair of them is the design:

   1. *separate agentic and single in the big title card since the performance is
      very different* -- the extraction card had been the POOLED per-model score,
      which is the one thing `by_extract` and `_pres_group` refuse to compute.
      The shapes are not two samples of one setting: the gap on typhoon was 2.5x
      on the 29-key schema, and `qwen3.5:2b` is the one model that prefers the
      shape the others do not. A pooled headline averaged a model's strong shape
      with its weak one and named neither.
   2. *only show the best not both* -- so one card, not one per shape.

   **What survives from the first is the number, not the layout.** The card is
   `extract_shapes[0]`, a model AND a shape, so it still never pools the two --
   it just shows the winner instead of a card each. On the live log that is
   agentic `qwen3.5:4b` at 76.0%, where the pooled figure had been 67.9%: the
   headline now names something you can actually go and run.

   The pill is not decoration. Without it the card would say "qwen3.5:4b, 76.0%"
   and that number is unreachable in single mode -- the shape is half of what is
   being recommended. Block 5 has the full grid for the shape that lost. */
function prHeroes(p) {
  const ocr = (p.ocr_models || [])[0];
  // The server's ordering, sorted by field accuracy -- never the sorted table.
  const best = (p.extract_shapes || [])[0];
  const secs = e => e && e.seconds.mean !== null
    ? Number(e.seconds.mean).toFixed(0) + 's per doc' : '';
  return '<div class="pr-heroes">'
    + prHero('Best at reading a page', ocr, '%', secs(ocr))
    + prHero('Best extractor', best, '%', secs(best),
             best ? '<span class="pr-shape pr-shape-' + esc(best.mode)
                    + ' pr-hero-pill">' + esc(best.mode) + '</span> '
                    + prExtrasLabel(best.extras) + ' ' + prSchemaLabel(best.schema_first) : '')
    + '</div>';
}

/* ---------- the environment card ----------
   **Three claims, drawn as three columns, and the separation is the point.**
   The machine is the box this PROCESS runs on; the server is what it would send
   a request to NOW; the runs are what the rows in view were actually made under.
   None of the three implies another -- the log outlives the machine, and
   switching the endpoint does not rewrite a row -- so a reader must never be
   able to take "this machine has an RTX 3060" together with "these runs used
   Ollama" as "these runs used this card". The column headings are what keep
   those apart, and they are not decoration: the paragraph that used to say it in
   words is gone with the rest of the prose, so the headings and their tooltips
   now carry it alone. */

/* The top few values of a tally, with the rest counted rather than listed. */
function prTally(list, shorten) {
  if (!list || !list.length) return '<span class="cs-none">&mdash;</span>';
  const shown = list.slice(0, 3).map(v =>
    '<span class="pr-pill" title="' + esc(v.value) + ' — ' + v.runs + ' run(s)">'
    + esc(shorten ? csModel(v.value) : v.value)
    + '<span class="n">' + v.runs + '</span></span>').join('');
  return shown + (list.length > 3
    ? '<span class="pr-pill pr-pill-more" title="'
      + esc(list.slice(3).map(v => v.value + ' (' + v.runs + ')').join(', '))
      + '">+' + (list.length - 3) + ' more</span>' : '');
}

/* **Neither `label` nor `value` is escaped**, and both are literal markup written
   in `prEnvCard` below -- the label so it can carry the `inferred` badge, the
   value because every one of them is assembled there with `esc` already applied
   to each piece of text that came from the log or the machine. Nothing
   user-supplied reaches this function directly, and the badge was being printed
   as its own source until this was split out. */
function prEnvRow(label, value, hint) {
  return '<div class="pr-env-row"><span class="pr-env-k"'
    + (hint ? ' title="' + esc(hint) + '"' : '') + '>' + label + '</span>'
    + '<span class="pr-env-v">' + value + '</span></div>';
}

function prEnvCard(env) {
  if (!env) return '';
  const m = env.machine || {}, sv = env.server || {}, st = env.settings || {};
  const runs = env.runs || {};
  const gpus = (m.gpus || []).length
    ? m.gpus.map(g => '<b>' + esc(g.name) + '</b>'
        + (g.memory_gb ? ' ' + g.memory_gb + ' GB' : '')
        + (g.driver ? '<span class="cs-of"> driver ' + esc(g.driver) + '</span>' : ''))
        .join('<br>')
    // Not an empty cell: "no GPU" and "could not ask" are different states, and
    // `nvidia-smi` is the only probe here, so an AMD or Apple machine lands in
    // the same branch as a missing binary. Saying which probe was used is what
    // lets a reader tell those apart.
    : '<span class="cs-none">not detected <span class="cs-of">('
      + esc(m.gpu_probe || 'no probe') + ')</span></span>';
  const when = runs.first && runs.last
    ? esc(String(runs.first).slice(0, 10)) + ' &rarr; ' + esc(String(runs.last).slice(0, 10))
    : '<span class="cs-none">&mdash;</span>';
  return '<div class="pr-env">'
    + '<div class="pr-env-col">'
      + '<div class="pr-env-h" title="probed on the machine this app is running '
        + 'on, right now. It says nothing about where the rows were produced — '
        + 'the log outlives the machine.">This machine</div>'
      + prEnvRow('GPU', gpus)
      + prEnvRow('CPU', '<b>' + esc(m.cpu || '—') + '</b>'
          + (m.cpu_threads ? '<span class="cs-of"> ' + m.cpu_threads
             + ' threads</span>' : ''),
          'logical processors, which is what llama.cpp’s -t is counted in — '
          + 'not the P-core count the performance notes recommend pinning to')
      + prEnvRow('Memory', m.memory_gb
          ? '<b>' + m.memory_gb + ' GB</b>' : '<span class="cs-none">&mdash;</span>')
      + prEnvRow('System', esc(m.os || '—')
          + '<span class="cs-of"> ' + esc(m.arch || '') + ' &middot; Python '
          + esc(m.python || '') + '</span>')
    + '</div>'
    + '<div class="pr-env-col">'
      + '<div class="pr-env-h" title="the model server this process would send a '
        + 'request to now, and the settings it would send. Switching the endpoint '
        + 'does not rewrite a row, so this does not describe the runs either.">'
        + 'Server now</div>'
      // Three states, not two. **Never probed** and **probed and not answering**
      // are different problems -- the first is a card drawn before anything
      // asked, the second a server that is down -- and this card is forbidden
      // from asking, so it must be able to say the first out loud.
      + prEnvRow('Endpoint',
          '<b>' + (sv.backend ? esc(sv.backend)
                   : sv.probed ? '<span class="cs-none">not reachable</span>'
                   : '<span class="cs-none">not probed yet</span>') + '</b>'
          + '<span class="cs-of"> ' + esc(sv.url || '') + '</span>',
          'as last seen. This card never contacts the model server — it is '
          + 'repainted every few seconds, and polling a llama.cpp server cancels '
          + 'work in flight. Use Re-check beside the server picker to refresh it.')
      + prEnvRow('Reading model', sv.model
          ? '<b title="' + esc(sv.model) + '">' + esc(csModel(sv.model)) + '</b>'
          : '<span class="cs-none">none resolved</span>')
      + prEnvRow('Extracting model', sv.extract_model
          ? '<b title="' + esc(sv.extract_model) + '">' + esc(csModel(sv.extract_model))
            + '</b>'
          // Blank means the one-model setup every measurement in this project
          // was taken under, not "unknown" -- the same rule the CSV column
          // follows, said in words because a dash here would read as missing.
          : '<span class="cs-of">same as reading</span>',
          'blank in the log means the two passes ran on one model, which is the '
          + 'normal case')
      + prEnvRow('Context', (sv.num_ctx ? '<b>' + sv.num_ctx + '</b> tokens' : '—')
          + (sv.slots ? '<span class="cs-of"> ' + sv.slots + ' slot(s)</span>' : ''))
      + prEnvRow('Defaults', '<b>' + esc(st.detail || '—') + '</b>'
          + '<span class="cs-of"> Detail &middot; ' + esc(st.ocr_profile || '')
          + ' &middot; ' + esc(st.extract_mode || '') + ' &middot; DRY '
          + (st.dry === undefined ? '—' : st.dry) + '</span>')
    + '</div>'
    + '<div class="pr-env-col">'
      + '<div class="pr-env-h" title="counted from the rows this view matched, so '
        + 'it narrows with the chips above like every table here. This is the only '
        + 'one of the three columns that describes the numbers below.">'
        + 'These runs <span class="cs-of">(' + (runs.runs || 0) + ')</span></div>'
      + prEnvRow('Dates', when)
      + prEnvRow('Backend', prTally(runs.backends))
      + prEnvRow('Read by', prTally(runs.read_models, true))
      + prEnvRow('Extracted by', prTally(runs.extract_models, true))
      + prEnvRow('Detail', prTally(runs.details))
      // Inferred, and marked as such wherever it appears -- the app talks HTTP
      // to a server it did not launch and knows neither the GPU layer count nor
      // whether the model was resident.
      + prEnvRow('Ran on <span class="an-inf">inferred</span>',
                 prTally(runs.hardware_inferred),
                 'read off the decode rate, not reported by the server')
      + prEnvRow('Start <span class="an-inf">inferred</span>',
                 prTally(runs.start_inferred),
                 'read off prefill: a warm read reuses a cached prefix or an '
                 + 'already-resident model and pays almost none')
    + '</div>'
    + '</div>';
}

/* A block heading: a numbered accent chip and the question it answers. The
   caveats that used to sit under each block are on the heading's tooltip now.

   `what` names the key THIS block groups on, because the window is per group
   like everywhere else on the card -- block 1 covers each model's own recent
   runs, block 3 each model-and-preset's. The sentence is plain text and joins
   the tooltip rather than becoming a `cs-of` span as it does on the other tabs:
   this tab has no prose by rule, and `prHead` escapes its hint. */
function prHead(n, text, hint, what) {
  const note = logWindow
    ? ' Figures cover each ' + (what || 'model') + "'s most recent runs, up to "
      + logWindow + ' — the run count on each row is what it actually covers.'
    : '';
  hint = (hint || '') + note;
  return '<div class="pr-h"' + (hint ? ' title="' + esc(hint) + '"' : '') + '>'
    + '<span class="pr-n">' + n + '</span>' + esc(text) + '</div>';
}

function prExtrasLabel(extras) {
  return '<span class="pr-shape">Extra fields ' + (extras === false ? 'off' : 'on') + '</span>';
}

function prSchemaLabel(on) {
  return '<span class="pr-shape">' + (on ? 'Schema first' : 'JSON first') + '</span>';
}

function paintPresent(t) {
  const p = (t && t.presentation) || null;
  if (!presentSum) return;
  if (!p) { presentSum.innerHTML = ''; return; }
  presBias.checked = presBiased;
  const ocr = p.ocr_models || [];
  const ex = p.extract_models || [];
  const shapes = p.extract_shapes || [];
  const ocrRank = prRanks(ocr), exRankAt = prRanks(ex), shapeRank = prRanks(shapes);
  // Column scales for the bars. Accuracy is a percentage and scales to 100, so
  // a 60% row is visibly two thirds of a 90% one; the clocks scale to their own
  // column's slowest, because seconds have no natural ceiling.
  const ocrSecsMax = prMax(ocr, e => e.seconds.mean);
  const ocrTpsMax = prMax(ocr, e => e.tps.mean);
  const ocrTokMax = prMax(ocr, e => e.tokens.mean);
  const exSecsMax = prMax(ex, e => e.seconds.mean);
  const shapeSecsMax = prMax(shapes, e => e.seconds.mean);
  const extrasColumn = {
    key: 'extras', label: 'Extra fields', value: e => e.extras === false ? 'off' : 'on',
    title: 'Whether other_fields was requested. Older runs used Extra fields on. '
      + 'Zero extra values found still counts as on.',
    cell: e => prExtrasLabel(e.extras)
  };
  const schemaColumn = {
    key: 'schema_first', label: 'JSON schema first', value: e => !!e.schema_first,
    title: 'Agentic schema from the first request. Older runs used JSON first.',
    cell: e => prSchemaLabel(e.schema_first)
  };

  // 1. Which model to read a page with.
  const rank = [
    {key: 'model', label: 'Model', cls: 'pr-model', value: e => csModel(e.model),
     cell: e => prModel(e, ocrRank.get(e))},
    {key: 'char', label: 'Transcript accuracy', n: true, value: e => e.value.mean,
     title: 'character accuracy against the hand-written transcript ground '
          + 'truth, meaned over each document first and then across documents',
     cell: e => prFig(e.value, '%', 1, presColour, (e.value.mean || 0) / 100)},
    // WHICH of the three scripts the model loses, beside how much it loses.
    // Numbers first, against the order used elsewhere: every Mandatory field in
    // the requirement is a figure, a date or an ID, so a model at 92% overall
    // and 71% on the digits is unusable in a way the column to the left cannot
    // say. The three are recall over subsets of that column and do NOT add up
    // to it — punctuation belongs to no script.
    {key: 'digit', label: 'Numbers', n: true, value: e => e.digit.mean,
     title: 'digits only, Arabic and Thai numerals — amounts, dates, tax IDs '
          + 'and document numbers. Read this before the column to its left: '
          + 'every Mandatory field in the requirement is one of these.',
     cell: e => prFig(e.digit, '%', 1, presColour, (e.digit.mean || 0) / 100)},
    {key: 'thai', label: 'Thai', n: true, value: e => e.thai.mean,
     title: 'Thai letters, tone marks and vowels included — the first thing to '
          + 'go when a page is downscaled',
     cell: e => prFig(e.thai, '%', 1, presColour, (e.thai.mean || 0) / 100)},
    {key: 'latin', label: 'English', n: true, value: e => e.latin.mean,
     title: 'the Latin alphabet. Not a claim that the text is English: a '
          + 'romanised Thai company name is Latin script too.',
     cell: e => prFig(e.latin, '%', 1, presColour, (e.latin.mean || 0) / 100)},
    {key: 'err', label: 'Error rate', n: true, value: e => e.failure_rate,
     title: 'runs that did not finish — looped, truncated, cancelled or errored. '
          + 'A run that finished and read the page badly is not one of these; it '
          + 'is in the accuracy column as the score it got.',
     cell: prErr}
  ];
  // 2. What each of them costs.
  const time = [
    {key: 'model', label: 'Model', cls: 'pr-model', value: e => csModel(e.model),
     cell: e => prModel(e, ocrRank.get(e))},
    {key: 'secs', label: 'Time per document', n: true, value: e => e.seconds.mean,
     title: 'wall clock for the whole document, meaned over documents',
     cell: e => prFig(e.seconds, 's', 1, null, (e.seconds.mean || 0) / ocrSecsMax)},
    {key: 'tps', label: 'Decode rate', n: true, value: e => e.tps.mean,
     title: 'tokens per second, excluding prefill — so it compares across pages '
          + 'of different resolutions',
     cell: e => prFig(e.tps, ' tok/s', 1, null, (e.tps.mean || 0) / ocrTpsMax)},
    // **Accuracy is not in this table** (2026-08-25, at the user's request:
    // *change accuracy to average token generated*). It was here to say "fast
    // is a claim about cost alone", which the heading now says at no width --
    // and it is the whole of block 1 two rows up, so the column was the same
    // number printed twice. What replaces it is the other half of what a read
    // costs: how much the model actually wrote.
    {key: 'tokens', label: 'Tokens generated', n: true, value: e => e.tokens.mean,
     title: 'output tokens per document, meaned over documents — the length of '
          + 'the transcript the model produced. The page image and the prompt '
          + 'are prefill and are NOT in this figure, so it is what the model '
          + 'wrote, not what the request cost.',
     cell: e => prFig(e.tokens, '', 0, null, (e.tokens.mean || 0) / ocrTokMax)}
  ];
  // 4. Which model to extract with.
  const exRank = [
    {key: 'model', label: 'Model', cls: 'pr-model', value: e => csModel(e.model),
     cell: e => prModel(e, exRankAt.get(e))},
    extrasColumn, schemaColumn,
    {key: 'field', label: 'Field accuracy', n: true, value: e => e.value.mean,
     title: 'priority-1 values that are the value the field ground truth says '
          + 'belongs in that key. A partial match counts half.',
     cell: e => prFig(e.value, '%', 1, presColour, (e.value.mean || 0) / 100)},
    {key: 'err', label: 'Error rate', n: true, value: e => e.failure_rate,
     title: 'extractions whose reply cycled, and runs that produced no reply at '
          + 'all. Neither is scored.',
     cell: prErr}
  ];
  // 5. Single against agentic, per model, with the clock -- the shape is a row
  // rather than something averaged over, because the two are not two samples of
  // one setting and some models prefer one and some the other.
  const shape = [
    {key: 'model', label: 'Model', cls: 'pr-model', value: e => csModel(e.model),
     cell: e => prModel(e, shapeRank.get(e))},
    {key: 'mode', label: 'Shape', value: e => e.mode,
     title: 'single: one request for the whole form. hybrid: tuned grouped requests. agentic: original steps '
          + 'of one to three fields, then the answers merged.',
     cell: e => '<span class="pr-shape pr-shape-' + esc(e.mode) + '">'
       + esc(e.mode) + '</span>'},
    extrasColumn, schemaColumn,
    {key: 'field', label: 'Field accuracy', n: true, value: e => e.value.mean,
     title: 'the same score as the table above, split by the shape of the request',
     cell: e => prFig(e.value, '%', 1, presColour, (e.value.mean || 0) / 100)},
    {key: 'secs', label: 'Time per document', n: true, value: e => e.seconds.mean,
     title: 'wall clock for the extraction alone — the read is not in this figure',
     cell: e => prFig(e.seconds, 's', 1, null, (e.seconds.mean || 0) / shapeSecsMax)},
    {key: 'err', label: 'Error rate', n: true, value: e => e.failure_rate,
     cell: prErr}
  ];
  // 6. What each extractor costs.
  const exTime = [
    {key: 'model', label: 'Model', cls: 'pr-model', value: e => csModel(e.model),
     cell: e => prModel(e, exRankAt.get(e))},
    extrasColumn, schemaColumn,
    {key: 'secs', label: 'Time per document', n: true, value: e => e.seconds.mean,
     title: 'wall clock for the extraction alone, meaned over documents',
     cell: e => prFig(e.seconds, 's', 1, null, (e.seconds.mean || 0) / exSecsMax)},
    {key: 'field', label: 'Field accuracy', n: true, value: e => e.value.mean,
     cell: e => prFig(e.value, '%', 1, presColour, (e.value.mean || 0) / 100)}
  ];

  const none = '<div class="pr-empty">Nothing in this view.</div>';
  presentSum.innerHTML = prHeroes(p)
    + prEnvCard((t || {}).environment)
    + prBiasNote(p)
    + prHead(1, 'Which model reads a page best',
             'Every mean is taken over each document first and then across the '
             + 'documents, so a model run five times on one page is not five '
             + 'samples of it. A run that did not finish is counted in the error '
             + 'column and scored nowhere. Every column sorts; the highlighted '
             + 'row and the rank badges stay the server ranking whichever column '
             + 'you sort by.')
    + (ocr.length ? lvTable('prOcr', ocr, rank,
        {sort: {key: 'char', dir: 'desc'},
         rowClass: e => e === ocr[0] ? ' class="ms-top"' : ''}) : none)
    + prHead(2, 'What each reader costs',
             'The clock, the decode rate, and how much the model actually wrote. '
             + 'Fast is a claim about cost alone — read it beside block 1, '
             + 'because the cheapest reader here is not the one to run. Tokens '
             + 'generated is output only: the page image and the prompt are '
             + 'prefill and are not counted in it.')
    + (ocr.length ? lvTable('prOcrTime', ocr, time, {sort: {key: 'secs', dir: 'asc'}}) : none)
    + prHead(3, 'What raising Detail buys, and what it costs',
             'Read along a row, never down a column: the models have not been '
             + 'run equally at every preset, so a figure pooled across them is '
             + 'partly a ranking of whichever model was run most at each — which '
             + 'is why there is no pooled row. A row that falls in the first grid '
             + 'and rises in the second is a preset paying for nothing.',
             'model and preset')
    + '<div class="pr-sub">Accuracy at each preset</div>'
    + (prDetailGrid(p, 'accuracy') || none)
    + '<div class="pr-sub">Time at each preset</div>'
    + (prDetailGrid(p, 'time') || none)
    + prHead(4, 'Which model extracts fields best',
             'Scored against hand-written field ground truth. A partial match '
             + 'counts half a value. Extra fields on and off are separate rows.',
             'model and Extra fields setting')
    + (ex.length ? lvTable('prExtract', ex, exRank,
        {sort: {key: 'field', dir: 'desc'},
         rowClass: e => e === ex[0] ? ' class="ms-top"' : ''}) : none)
    + prHead(5, 'Extraction modes',
             'Single, Agentic and Hybrid are separate rows, never averaged. Extra fields on and off '
             + 'also stay separate for accuracy, time and error rate.',
             'model, shape and Extra fields setting')
    + (shapes.length ? lvTable('prShape', shapes, shape, {sort: {key: 'field', dir: 'desc'}}) : none)
    + prHead(6, 'What each extractor costs',
             'The pass-2 clock alone — reading the page is not in these figures. '
             + 'Extra fields on and off are separate rows.',
             'model and Extra fields setting')
    + (ex.length ? lvTable('prExtractTime', ex, exTime, {sort: {key: 'secs', dir: 'asc'}}) : none);
}


/* ---------- random test ---------- */
/* Reads the NDJSON stream `/api/randomtest/stream` produces, one event per
   round, and paints the table as it goes. The rows are the point: a run that
   only reported a total would say "3 of 6 failed" without saying which
   combination did it, and the combination IS the finding.

   `rtRows` lives outside the render because the table is repainted on every
   event -- a round that scrolled away when the next one started would be
   unreadable during exactly the run it exists to explain. */
let rtRows = [];
let rtAbort = null;
let rtRunning = false;
let rtContinuous = false;
/* True between pressing Stop and the stream ending. Stop is a REQUEST to the
   server now, not a hang-up on our own fetch: an abandoned stream is not
   stopped on that server, so the old Stop left every remaining round reading
   and the next Run was refused by a 409 that was telling the truth. `rtAbort`
   stays as the fallback for a Stop the server could not be told about. */
let rtStopping = false;
/* From the server, never repeated here: the bar the Python classifier has
   to clear before the model is asked. */
const MIN_CLASSIFY_CONFIDENCE = window.BENCHMARK_BOOTSTRAP.min_classify_confidence;
/* `scoring.SCRIPT_MIN_CHARS`: below this many characters of a script on
   one page, no rate is reported for it and the count is shown instead. */
const scriptMinChars = window.BENCHMARK_BOOTSTRAP.script_min_chars;

function rtPct(value, digits) {
  if (value === null || value === undefined) return '<span class="cs-none">&mdash;</span>';
  return '<span class="cs-val" style="color:' + accColour(Number(value) / 100) + '">'
    + Number(value).toFixed(digits === undefined ? 1 : digits) + '%</span>';
}
/* What a round ran, naming only the settings its scope actually used. A Detail
   printed under a fields-only round names a setting that had no part in the
   result -- no image was made -- and an extraction shape under a read-only one
   reads as a pass that ran and found nothing. */
function rtPlanLines(p) {
  const line = t => '<span class="rt-plan">' + t + '</span>';
  const reader = esc(csModel(p.reader)) + ' (' + esc(p.profile) + ')';
  // Which end of the ranking a contender came from. Absent on a random round,
  // and absent on a contest where every ranked model is running -- there is no
  // top half when everyone is in it.
  const rank = p.contender
    ? '<span class="rt-seen" title="from the ' + p.contender
      + ' of the Standouts ranking">' + p.contender + '</span> '
    : '';
  if (p.scope === 'fields') {
    return line('fields only &middot; no page read')
      + line(rank + esc(csModel(p.extractor)) + ' &middot; ' + esc(p.mode));
  }
  if (p.scope === 'ocr') {
    return line('read only &middot; no extraction')
      + line(rank + reader + ' &middot; ' + esc(p.detail));
  }
  return line(rank + reader
              + ' &rarr; ' + esc(p.extractor ? csModel(p.extractor) : 'same'))
    + line(esc(p.detail) + ' &middot; ' + esc(p.mode));
}

/* The two model locks are painted from the same lists the pickers above use, so
   a lock can only ever name something this endpoint serves -- the server refuses
   the rest, and a refusal that could have been prevented in the picker is a
   refusal nobody learns anything from.

   The extraction list depends on the scope: a fields round runs ONE model and
   any served model may be it, including an OCR fine-tune; every other scope puts
   pass 2 beside a reader, where `select_extract` refuses a second OCR model. */
function rtOption(name, chosen, suffix) {
  return '<option value="' + esc(name) + '"' + (name === chosen ? ' selected' : '')
    + '>' + esc(csModel(name)) + (suffix || '') + '</option>';
}
/* Excluded names live outside the render, like every other set on this page: the
   chips are repainted whenever the model list is, and a toggle that reset itself
   on a status refresh would be a control that forgets. A name that stops being
   served stays in the set harmlessly -- the server ignores an exclusion for a
   model it does not have. */
const rtExcluded = { readers: new Set(), extractors: new Set(), cases: new Set() };
function rtPaintSequence() {
  const sequential = rtStrategy.value === 'sequential';
  const files = [...rtFiles.querySelectorAll('input')];
  const selected = files.filter(box => box.checked).length;
  rtFilesField.style.display = sequential ? 'block' : 'none';
  rtRoundsField.style.display = sequential ? 'none' : 'flex';
  rtLockCaseField.style.display = sequential ? 'none' : 'flex';
  rtFileCount.textContent = selected + ' / ' + files.length + ' files selected'
    + (selected ? '' : ' — select at least one file');
  rtRun.textContent = sequential ? 'Run sequential test' : 'Run random test';
  rtRun.disabled = rtRunning || (sequential && !selected);
  rtStrategy.disabled = rtRunning;
  rtFilesAll.disabled = rtFilesNone.disabled = rtRunning;
  files.forEach(box => { box.disabled = rtRunning; });
}
function rtChips(box, names, set) {
  box.innerHTML = names.map(name =>
    '<label class="rt-chip' + (set.has(name) ? ' off' : '') + '" title="'
    + esc(name) + '"><input type="checkbox" value="' + esc(name) + '"'
    + (set.has(name) ? '' : ' checked') + '>' + esc(csModel(name)) + '</label>').join('');
  for (const box_ of box.querySelectorAll('input')) {
    box_.addEventListener('change', () => {
      if (box_.checked) set.delete(box_.value); else set.add(box_.value);
      box_.parentElement.classList.toggle('off', !box_.checked);
    });
  }
}
function rtPaintLocks() {
  const models = server.models || [];
  const readers = models.filter(m => m.vision !== false).map(m => m.name);
  /* A fields round runs ONE model, and where pass 2 has a server of its own
     that model is drawn from THAT server (`randomtest.pools(extract_models=)`)
     -- a name from the reading server would not be served where it is sent. */
  const fieldModels = (extractInfo && extractInfo.separate)
    ? (extractInfo.models || [])
    : models.map(m => m.name);
  const keep = (sel, names) => {
    const chosen = sel.value;
    sel.innerHTML = '<option value="">— random —</option>'
      + names.map(n => rtOption(n, chosen)).join('');
    // A lock that is no longer on the list is dropped rather than silently
    // pointing at something else.
    if (sel.value !== chosen) sel.value = '';
  };
  keep(rtLockReader, readers);
  keep(rtLockExtract, rtScope.value === 'fields'
    ? fieldModels
    : ((extractInfo && extractInfo.choices) || []).filter(Boolean));
  // A read-only run never extracts, and a fields run never reads.
  rtLockReaderField.style.display = rtScope.value === 'fields' ? 'none' : 'flex';
  rtLockExtractField.style.display = rtScope.value === 'ocr' ? 'none' : 'flex';
  // A fields run makes no image, so it has no Detail to pin.
  rtLockDetailField.style.display = rtScope.value === 'fields' ? 'none' : 'flex';
  // A read-only run never extracts, so it has no shape to pin.
  rtLockModeField.style.display = rtScope.value === 'ocr' ? 'none' : 'flex';
  // The chips are painted from the same two pools as the locks, so a model this
  // endpoint cannot use for a pass is not offered as one to exclude from it.
  rtChips(rtExRead, readers, rtExcluded.readers);
  rtChips(rtExExtract, rtScope.value === 'fields'
    ? fieldModels
    : ((extractInfo && extractInfo.choices) || []).filter(Boolean),
    rtExcluded.extractors);
  rtExReadRow.style.display = rtScope.value === 'fields' ? 'none' : 'block';
  rtExExtractRow.style.display = rtScope.value === 'ocr' ? 'none' : 'block';
  // A read-only run never extracts, so where pass 2 goes is not its question.
  const noExtract = rtScope.value === 'ocr';
  if (rtExtractServerRow) rtExtractServerRow.style.display = noExtract ? 'none' : 'flex';
  if (rtExtractServerNote) rtExtractServerNote.style.display = noExtract ? 'none' : '';
}


/* What the chosen scope will and will not measure, said before the run rather
   than inferred from empty columns afterwards.

   A hoisted function rather than the const map it used to be: the read floor it
   quotes moves at runtime now, and `paintReadFloor` repaints this note when it
   does -- from the workspace section, which runs first. A const there would be
   in its temporal dead zone at that moment, which throws and takes the whole
   inline script down with it. */
function rtScopeText(scope) {
  if (scope === 'ocr') {
    return 'Pass 1 only. No extraction runs, so the row it logs has blank pass-2 '
         + 'columns and there is no field score.';
  }
  if (scope === 'fields') {
    return 'Pass 2 only, on <code>solution/&lt;id&gt;.md</code>. No page is read, so '
         + 'nothing the extractor gets wrong belongs to the read — and any model '
         + 'can be drawn, vision or not.';
  }
  return 'Both passes. The field score is partly a measurement of the read — a '
       + 'value pass 1 got wrong cannot then be extracted right — so a round '
       + 'whose read failed'
       + (READ_FLOOR > 0 ? ' or scored under ' + (READ_FLOOR * 100).toFixed(0) + '%'
                         : '')
       + ' still extracts and still logs, and its fields are left '
       + '<b>unscored</b>.';
}
/* What the chosen draw will and will not do, beside the choice rather than
   inferred from a plan afterwards. Both are legitimate and they answer opposite
   questions, so neither is described here as the correct one. */
function rtStrategyText(strategy) {
  if (strategy === 'sequential') {
    return ' Selected files run in the order shown, then repeat from the first '
         + 'file until you press Stop. Runs and Lock document do not apply.';
  }
  if (strategy === 'uniform') {
    return ' Every document is equally likely, and no two rounds of one plan are '
         + 'the same document, reader, Detail and shape, so a repeat happens only '
         + 'once every combination has been used. That is a fair sample, which '
         + 'also means a document the log is already full of can be drawn again '
         + 'while one that has never run is missed.';
  }
  return ' Each round goes to whichever document the run log has read fewest '
       + 'times, so the corpus stays level and a fixture nobody has run is picked '
       + 'up by the next test. Not a fair sample: a document that is ahead is not '
       + 'drawn at all until the others catch up.';
}
function rtPaintScope() {
  rtScopeNote.innerHTML = (rtScopeText(rtScope.value) || '')
    + rtStrategyText(rtStrategy.value);
}

/* The Result card and the random test take turns in the same place. Both belong
   to "what just ran", and only one thing runs at a time here -- the alternative
   was a fifty-row table stacked above a Result card still showing a document
   somebody read by hand an hour ago. Compare goes with the Result card, since
   it holds that same read's page images. */
function rtPaint() {
  paintMain();
  if (!rtRows.length) {
    rtOut.innerHTML = '<p class="note">Run a random test to see its rounds here.</p>';
    return;
  }
  const body = rtRows.map(r => {
    const p = r.plan, out = r.result || {};
    const cls = r.error ? ' class="rt-bad"' : (r.pending ? ' class="rt-now"' : '');
    return '<tr' + cls + '>'
      + '<td><b>' + esc(p.case) + '</b>'
        + (p.cycle ? '<span class="rt-plan">Round ' + r.index + ' &middot; Cycle '
          + p.cycle + ' &middot; File ' + p.position + '</span>' : '')
        // How many times the run log had read this document before the round
        // was planned. It is why this case was chosen -- rounds go to whatever
        // is furthest behind -- and without it a plan that reads one fixture
        // twice and another not at all looks like a bad shuffle.
        + (p.seen === undefined ? ''
           : '<span class="rt-seen" title="reads of this document in the run log '
             + 'before this round'
             // Under `balanced` this number is WHY the document is here. Under
             // `uniform` it is not -- the draw ignores it -- and a tooltip that
             // went on claiming otherwise would describe a rule the run was not
             // planned by.
             + (p.strategy === 'sequential'
                ? ' — selected files run in order'
                : p.strategy === 'uniform'
                ? ' — drawn at random, so this is what the round adds to, '
                  + 'not why it was chosen'
                : ' — the least-read case is chosen every time') + '">'
             + p.seen + '</span>')
        + rtPlanLines(p) + '</td>'
      + (r.pending
          ? '<td colspan="6"><span class="cs-none">running&hellip;</span></td>'
          : !r.result && !r.error
          // Planned but not reached yet. Shown rather than hidden so the whole
          // plan is visible from the start -- that is what makes a long run
          // watchable, and what a seed reproduces.
          ? '<td colspan="6"><span class="cs-none">queued</span></td>'
          : r.error
            ? '<td colspan="6"><span style="color:var(--danger)">' + esc(r.error) + '</span></td>'
            // A pass the scope did not run is `not run`, never a zero: an
            // empty field count under a read-only round reads as an extraction
            // that returned nothing, which is a result and not what happened.
            : (p.scope === 'fields'
                ? '<td><span class="cs-none">not run</span></td>'
                : '<td>' + rtPct(out.char_accuracy === null || out.char_accuracy === undefined
                    ? null : out.char_accuracy * 100)
                  // A looped or truncated read scored what it scored, and the
                  // score alone cannot say the run did not finish.
                  + (out.status && out.status !== 'ok'
                     ? '<span class="rt-plan" style="color:var(--danger)">' + esc(out.status)
                       + '</span>' : '') + '</td>')
            + (p.scope === 'ocr'
                ? '<td><span class="cs-none">not run</span></td>'
                  + '<td><span class="cs-none">&mdash;</span></td>'
                // A read too poor to judge its own extraction by: the fields
                // came back and are logged, the score is not. The reason is
                // printed under it, because `unscored` with no because reads as
                // a defect in the harness rather than a rule of it.
                : (out.unscored
                    ? '<td><span class="cs-none">unscored</span>'
                      + '<span class="rt-plan">' + esc(out.unscored) + '</span></td>'
                    : '<td>' + rtPct(out.field)
                      // What the rate is over -- the requirement's Mandatory
                      // set -- and, where the truth file rules on any, the
                      // optional keys judged beside it and left out of it.
                      + '<span class="rt-plan"' + (out.scope
                          ? ' title="scored over ' + esc(out.scope) + '"' : '') + '>'
                      + out.correct + '+' + out.partial + 'p/' + out.expected
                      + (out.unknown_type ? ' base' : ' req')
                      + (out.optional_expected
                          ? ', ' + out.optional_correct + '/' + out.optional_expected
                            + ' opt' : '')
                      + '</span></td>')
                + '<td>' + out.other
                  + (out.partial_reply
                     ? '<span class="rt-plan" style="color:var(--danger)">salvaged</span>' : '')
                  + '</td>')
            + rtThroughputCells(out)
            + '<td class="n">' + Number(r.seconds || 0).toFixed(0) + 's</td>')
      + '</tr>';
  }).join('');
  rtOut.innerHTML = '<table class="queue"><tr><th>Round</th>'
    + '<th title="character accuracy of the transcript against solution/&lt;id&gt;.md">Read</th>'
    + '<th title="the fields the requirement marks Mandatory for this document '
      + 'type, a partial counting half. Optional fields are extracted and judged '
      + 'beside it and are not in the rate. A type no requirement covers is '
      + 'scored over the base field set instead and reads &quot;base&quot;'
      + '">Fields</th>'
    + '<th title="entries returned under other_fields">Extra</th>'
    + '<th title="pages read over the read and extract time, not the round&#39;s '
      + 'wall time: model switching between rounds is left out">Pages/s</th>'
    + '<th title="tokens sent plus tokens generated, both passes; per page under it">'
      + 'Tokens</th>'
    + '<th>Time</th></tr>' + body + '</table>';
}

/* Pages, pages/s and tokens, a round's or the whole run's -- `randomtest.
   summarise_round` and `run_totals` hold the same names so one painter fits both. */
function rtThroughputText(t) {
  const parts = [];
  if (t.pages) parts.push(t.pages + ' page(s)');
  if (t.pages_per_second != null) parts.push(t.pages_per_second.toFixed(2) + ' pages/s');
  if (t.tokens != null) {
    parts.push(t.tokens.toLocaleString() + ' tokens (in '
      + (t.in_tokens || 0).toLocaleString() + ' / out '
      + (t.out_tokens || 0).toLocaleString() + ')');
  }
  if (t.tokens_per_page != null) parts.push(Math.round(t.tokens_per_page).toLocaleString() + ' tokens/page');
  return parts.join(', ');
}
function rtThroughputCells(out) {
  const none = '<span class="cs-none">&mdash;</span>';
  return '<td class="n">' + (out.pages_per_second == null ? none
      : out.pages_per_second.toFixed(2)
        + '<span class="rt-plan">' + out.pages + ' page(s)</span>') + '</td>'
    + '<td class="n">' + (out.tokens == null ? none
      : '<span title="in ' + (out.in_tokens || 0).toLocaleString() + ' / out '
        + (out.out_tokens || 0).toLocaleString() + '">'
        + out.tokens.toLocaleString() + '</span>'
        + (out.tokens_per_page == null ? ''
           : '<span class="rt-plan">' + Math.round(out.tokens_per_page).toLocaleString()
             + '/page</span>')) + '</td>';
}

/* Both Stop buttons move with the Run button: the run is one thing, and a Stop
   that looks live in the pane while the card's is greyed out is a control that
   lies about what it will do. */
function rtBusy(on) {
  rtRunning = on;
  if (on) { rtStopping = false; setTestStatus('rt', 'running'); }
  rtClear.disabled = on;
  rtPaintSequence();
  // A Stop already asked for is not a Stop that can be asked for again: the
  // request is in, and a live-looking button would invite a second one that
  // does nothing.
  rtStop.disabled = !on || rtStopping;
  rtStop2.disabled = !on || rtStopping;
  // The server refuses to move pass 2 mid-sweep (409), so the picker says so
  // by being unavailable rather than by an error after the click.
  if (rtExtractEndpoint) rtExtractEndpoint.disabled = on;
  // One sweep at a time: the server refuses the second with a 409 anyway.
  if (typeof stRun !== 'undefined' && stRun) stPaintFiles();
  // Reads and sweeps share server settings, so do not start a workspace read
  // while a sweep is changing those settings between rounds.
  // `paintServer` owns this button otherwise and puts it back on its own terms.
  if (on) runBtn.disabled = true; else paintServer();
}

async function rtStart() {
  const rounds = Math.max(1, Math.min(Number(rtRounds.value) || 5, 50));
  rtRounds.value = rounds;
  rtRows = [];
  rtContinuous = false;
  rtPaint();
  rtBusy(true);
  rtStop.disabled = rtStop2.disabled = true;
  // Same reason as the Read button: the rounds below are judged by this floor,
  // so it is pushed before the first one rather than whenever the box lost
  // focus.
  rtNote.classList.remove('err');
  rtNote.textContent = 'Planning…';
  paintMain();
  rtAbort = new AbortController();
  try {
    await pushReadFloor(rtReadFloorInp || readFloorInp);
    await ensureSampling();
    const res = await fetch(API + '/randomtest/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      signal: rtAbort.signal,
      // Exclusions go with both shapes: "do not test that model" is a statement
      // about the run, not about which button started it.
      body: JSON.stringify({
        // Exclusions are a statement about the run: do not draw that model.
        exclude: { readers: [...rtExcluded.readers],
                   extractors: [...rtExcluded.extractors],
                   cases: rtStrategy.value === 'sequential' ? [...rtExcluded.cases] : [] },
        rounds, seed: rtSeed.value.trim() || null, scope: rtScope.value,
        // The same seed under the other draw is a different plan, so this goes
        // with the seed rather than being remembered anywhere.
        strategy: rtStrategy.value,
        lock: { case: rtStrategy.value === 'sequential' ? '' : rtLockCase.value,
                reader: rtLockReader.value,
                extractor: rtLockExtract.value, mode: rtLockMode.value,
                detail: rtScope.value === 'fields' ? '' : rtLockDetail.value }
      })
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    rtStop.disabled = rtStop2.disabled = rtStopping;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.event === 'plan') {
          rtContinuous = !!ev.continuous;
          // The seed goes into the box, not just the note: the next thing
          // anyone wants after an interesting run is to run it again.
          rtSeed.value = ev.seed;
          const locked = Object.entries(ev.lock || {})
            .map(([k, v]) => k + '=' + csModel(v)).join(', ');
          rtNote.textContent = 'Seed ' + ev.seed + ' — ' + ev.rounds.length
            + ' round(s). Re-run with this seed and the same draw for the same '
            + 'plan.'
            + (locked ? ' Locked: ' + locked + '.' : '')
            // How many different rounds there were to choose from, and how many
            // of this plan could not be different because there were not enough.
            // A plan that repeats a scenario should say why on its own face
            // rather than looking like a bad shuffle.
            + (ev.scenarios
               ? ' Drawn from ' + ev.scenarios.toLocaleString()
                 + ' distinct scenario(s).' : '')
            + (ev.repeats
               ? ' ' + ev.repeats + ' round(s) repeat one: there are not that '
                 + 'many different rounds to run.' : '');
          // Only the round actually in flight is `pending`; the rest are
          // planned-but-not-started, which is a different thing to show. Every
          // row read "running…" when this said `pending: true` for all of them.
          if (rtContinuous) {
            rtRows = [];
            rtNote.textContent = 'Sequential — ' + ev.rounds.length
              + ' selected files per cycle. Repeats until Stop. Seed ' + ev.seed + '.';
          } else {
            rtRows = ev.rounds.map(p => ({ plan: p }));
            rtRows[0].pending = true;
            rtPaint();
          }
        } else if (ev.event === 'round_start') {
          rtRows.push({ ...ev, pending: true });
          if (rtRows.length > 200) rtRows.splice(0, rtRows.length - 200);
          rtNote.textContent = 'Sequential — cycle ' + ev.plan.cycle + ', file '
            + ev.plan.position + ': ' + ev.plan.case + ' (round ' + ev.index
            + '). Repeats until Stop.';
          rtPaint();
        } else if (ev.event === 'round') {
          if (rtContinuous) {
            const index = rtRows.findIndex(row => row.index === ev.index);
            if (index >= 0) rtRows[index] = { ...ev, pending: false };
            else rtRows.push({ ...ev, pending: false });
          } else {
            rtRows[ev.index - 1] = { ...ev, pending: false };
            if (rtRows[ev.index]) rtRows[ev.index].pending = true;
          }
          rtPaint();
          // The run log is where these rows land, so it is repainted as they
          // arrive rather than only at the end.
          refreshRuns();
        } else if (ev.event === 'stopped') {
          // The server says where it stopped. Nothing to paint: `done` follows
          // immediately and writes the one summary line a run gets.
          rtStopping = true;
          setTestStatus('rt', 'stopping');
        } else if (ev.event === 'done') {
          setTestStatus('rt', ev.stopped ? 'stopped' : 'completed');
          rtNote.textContent = 'Seed ' + ev.seed + ' — ' + ev.completed
            + (ev.continuous ? '' : '/' + ev.total) + ' round(s) '
            + (ev.stopped ? 'run before stopping' : 'completed')
            + (ev.failed ? ', ' + ev.failed + ' failed' : '')
            + ' in ' + Math.round(ev.seconds) + 's.'
            + (ev.totals && ev.totals.rounds
               ? ' Total: ' + rtThroughputText(ev.totals)
                 + (ev.totals.token_rounds < ev.totals.rounds
                    ? ' (tokens from ' + ev.totals.token_rounds + '/'
                      + ev.totals.rounds + ' rounds)' : '') + '.'
               : '')
            // The guard is released by the time this arrives, so this is also
            // the page saying another test may be started now.
            + (ev.stopped ? ' Stopped — the rounds above are in the run log, '
                          + 'and the one cut short is logged as cancelled. You '
                          + 'can start another test.' : '');
          rtNote.classList.toggle('err', ev.failed > 0);
        }
      }
    }
  } catch (err) {
    setTestStatus('rt', 'failed');
    if (err.name === 'AbortError') {
      // Only reached when the server could not be told to stop -- an ordinary
      // Stop now ends with the stream's own `done`. Hanging up frees the page
      // and nothing else, so this says what is still true of the server rather
      // than claiming the test is over.
      rtNote.textContent = 'Hung up on the test, but the server was not told '
        + 'to stop, so it is still running. Rounds already finished are above '
        + 'and in the run log; another test will be refused until it ends.';
      rtNote.classList.add('err');
    } else {
      rtNote.textContent = err.message;
      rtNote.classList.add('err');
    }
  } finally {
    if (['running', 'stopping'].includes(document.getElementById('rtStatus').dataset.state))
      setTestStatus('rt', 'failed');
    rtAbort = null;
    rtBusy(false);
    // Drop rows that never ran -- on a stop or a failure, a "queued" row that
    // will now never happen reads as part of the result.
    rtRows = rtRows.filter(r => r.result || r.error);
    rtPaint();
    refreshRuns();
  }
}

if (rtRun) {
  rtRun.addEventListener('click', () => rtStart());
  /* Stop asks the SERVER to stop and then waits for it. The stream is left
     open on purpose: it is what reports the round that was cut short and
     releases the sweep guard, and hanging up early is exactly what used to make
     a stopped test look over while it was still reading. */
  const stop = async () => {
    if (!rtAbort || rtStopping) return;
    rtStopping = true;
    setTestStatus('rt', 'stopping');
    rtStop.disabled = rtStop2.disabled = true;
    rtNote.classList.remove('err');
    // The bound, measured rather than guessed: a read notices between tokens,
    // so one still in PREFILL finishes its prefill -- Ollama sends no response
    // headers until generation starts, so there is nothing to hang up on. That
    // was 178s and 134s on uncapped pages here. Saying "stopping" with no idea
    // how long is what makes a slow stop read as a broken one.
    rtNote.textContent = 'Stopping — the rest of the plan is dropped now, and '
      + 'the round in flight ends at its next token. One still waiting on the '
      + 'model for its first token has to finish that wait first, which is '
      + 'slowest at high Detail; an extraction already running finishes.';
    try {
      const res = await fetch(API + '/randomtest/stop', { method: 'POST' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
    } catch (err) {
      // The server could not be told. Hang up so the page is usable, and let
      // the AbortError branch say the sweep is still going -- because it is.
      if (rtAbort) rtAbort.abort();
    }
  };
  rtStop.addEventListener('click', stop);
  rtStop2.addEventListener('click', stop);
  rtClear.addEventListener('click', () => {
    if (rtRunning) return;
    rtRows = []; rtNote.textContent = ''; rtNote.classList.remove('err');
    setTestStatus('rt', 'ready');
    rtPaint();
  });

  // Painted on change and once at load: the caveat belongs beside the choice,
  // not in the results of a run that already made it.
  rtScope.addEventListener('change', () => { rtPaintScope(); rtPaintLocks(); });
  rtStrategy.addEventListener('change', () => { rtPaintScope(); rtPaintSequence(); });
  const selectFile = box => {
    if (box.checked) rtExcluded.cases.delete(box.value);
    else rtExcluded.cases.add(box.value);
    box.parentElement.classList.toggle('off', !box.checked);
  };
  for (const box of rtFiles.querySelectorAll('input')) {
    selectFile(box);
    box.addEventListener('change', () => { selectFile(box); rtPaintSequence(); });
  }
  const selectFiles = checked => {
    for (const box of rtFiles.querySelectorAll('input')) {
      box.checked = checked;
      selectFile(box);
    }
    rtPaintSequence();
  };
  rtFilesAll.addEventListener('click', () => selectFiles(true));
  rtFilesNone.addEventListener('click', () => selectFiles(false));
  rtPaintScope();
  rtPaintLocks();
  rtPaintSequence();
}

/* ---------- the log view ----------
   **What is stored and what is averaged are two different things**, and this is
   the whole of the 2026-08-24 request. The CSV keeps every run whatever these
   controls say -- a read that scored 20% is written, is passed to pass 2, and is
   in the Raw data panel. What the controls decide is which rows reach a MEAN,
   and under what rules.

   Four pieces of state, none of which the server remembers between requests:

   - `logView`   -- the read floor and the window, `null` for "use the process
                    setting", so moving one control cannot silently reset the other.
   - `logFilters`-- include/exclude per field. A filtered row is ABSENT from the
                    tables, not merely hidden: `runs`, the failure rate and the
                    window all behave as though it had never happened, which is
                    the only reading under which "how does my best model do on
                    sol001" is answerable.
   - `logSort`   -- per table, above.
   - `rawPicked` -- which raw rows are ticked for deletion, by their position in
                    the file. */
let logData = null;
let logView = {window: null, min_read_pct: null};
let logFilters = {include: {}, exclude: {}};
let rawPicked = new Set();
let rawExpanded = null;
let currentLogGroup = 'overview';
const lastLogPanel = {overview: 'present', compare: 'rank', history: 'raw', advanced: 'case'};
let logBusy = false;

/* The axes worth filtering on, in the user's own terms: model, sol, mode. The
   two models are separate rows of chips because they are separate questions --
   this project has a model that reads badly and extracts best, and one that
   cannot read at all. `runlog.FILTER_FIELDS` knows more fields than these; what
   is not here is not worth the width. */
const LV_FIELDS = [
  {field: 'case', label: 'Document',
   hint: 'the ground-truth case, or the file name where there is none'},
  // What READ the page, which is not the same question as `pipeline` below --
  // that one is which PASSES a run performed. Blank on a row that read nothing,
  // so a fields-only row is not offered here and an include still drops it.
  {field: 'ocr_engine', label: 'Reading engine',
   hint: 'LLM: a model server read the page over HTTP. Library: PaddleOCR or '
       + 'EasyOCR read it in the processes of this machine. Rows written before '
       + 'the column are derived from their backend, so the whole log is here'},
  {field: 'model', label: 'Reading model',
   hint: 'the model that read the page (pass 1)'},
  {field: 'extract_model', label: 'Extraction model',
   hint: 'the model that extracted the fields (pass 2) — the reading model, '
       + 'where the two passes ran on one'},
  // What each pass ran ON. Blank on rows written before 2026-10-08, so those
  // rows are in none of these chips and keeping one drops them.
  {field: 'model_quant', label: 'Reading quant',
   hint: 'the quantisation of the reading model: what Ollama states for it, the '
       + 'GGUF type in the llama-server model file, or what the model name says. '
       + 'Rows logged before 2026-10-08 have none'},
  {field: 'inference_engine', label: 'Reading engine version',
   hint: 'the server kind and version that read the page (ollama 0.32.14, '
       + 'llama.cpp b1-67a17c1, vllm 0.11.0)'},
  {field: 'extract_quant', label: 'Extraction quant',
   hint: 'the quantisation of the model that extracted the fields'},
  {field: 'extract_inference_engine', label: 'Extraction engine version',
   hint: 'the server kind and version pass 2 ran on'},
  {field: 'extract_mode', label: 'Extraction shape',
   hint: 'single: one request for the whole form. hybrid: tuned grouped requests. agentic: original steps of '
       + 'one to three fields'},
  {field: 'extract_schema_first', label: 'JSON schema first',
   hint: 'on: Agentic/Hybrid steps start with a JSON schema. Older runs are off'},
  {field: 'detail', label: 'Detail',
   hint: 'the pixel budget the page was rasterised to: low 2 MP, medium 4 MP '
       + '(the default), original uncapped'},
  // The two inferred marks. Filterable like anything else, so "GPU cold reads
  // only" is one pair of chips -- and worth having as chips rather than only as
  // a column, because the question they answer is usually "show me the rest".
  {field: 'hardware', label: 'Hardware (inferred)',
   hint: 'GPU or CPU, inferred from the decode rate — the server does not '
       + 'report it, so this is a proxy and not a probe'},
  {field: 'start', label: 'Start (inferred)',
   hint: 'cold or warm, inferred from prefill: a warm read reuses a cached '
       + 'prefix or an already-resident model and pays almost none'},
  // **`status` has been a filter field with facets since the view controls
  // landed and had no chip**, so "show me this without the loops" was
  // expressible through the API and not from the page. It is last because it is
  // the one group that changes what a figure MEANS rather than which runs it is
  // over -- see the caveat under the chips.
  // A stress-test job saved to the log, or an ordinary run. Its timings were
  // taken under the load of the other jobs in flight, so this is the chip to
  // drop them with when a table is about one run at a time.
  {field: 'stress', label: 'Run kind',
   hint: 'stress: a job of a stress test saved to the log, timed under the load '
       + 'of the other jobs in flight. normal: every other run'},
  {field: 'status', label: 'Outcome',
   hint: 'how the run ended. Dropping the failures does not clean the data: a '
       + 'failure is already counted and never scored, so what a failure chip '
       + 'removes is the failure RATE beside every mean, not a bad number '
       + 'inside it'}
];

/* A chip's visible text. Model names get shortened; the two inferred marks get
   the same words the analysis tab prints for them, so one vocabulary covers the
   chips and the tables rather than `gpu` in one place and `GPU` in the other. */
function lvChipLabel(field, value) {
  if (field === 'hardware' || field === 'start') return anMarkLabel(field, value);
  if (field === 'ocr_engine') return engineLabel(value);
  if (field === 'status') return LV_STATUS[value] || value || 'not recorded';
  if (field === 'stress') return value === 'stress' ? 'stress test' : 'normal run';
  // As stored: `csModel` would cut a version at its last `/`.
  if (/quant$|inference_engine$/.test(field)) return value;
  return csModel(value);
}

/* How a run ended, in the words the rest of the card uses. `ok` is the only one
   a reader would not recognise from a table, and "finished" is what every other
   panel means by it -- a run that came back, whatever it scored, because a 0%
   read is a measurement and not a failure (2026-08-24). */
const LV_STATUS = {
  ok: 'finished', error: 'error', looped: 'looped', cancelled: 'cancelled',
  truncated: 'truncated', partial: 'partial (salvaged)'
};

function lvState(field, value) {
  if ((logFilters.include[field] || []).indexOf(value) >= 0) return 'inc';
  if ((logFilters.exclude[field] || []).indexOf(value) >= 0) return 'exc';
  return '';
}

/* Neutral -> kept -> dropped -> neutral. One control for both directions,
   because "only sol001" and "anything but dots.ocr" are the same question asked
   from opposite ends and a reader switching between them should not have to find
   a different widget. */
function lvCycle(field, value) {
  const inc = logFilters.include[field] || [];
  const exc = logFilters.exclude[field] || [];
  const at = lvState(field, value);
  logFilters.include[field] = inc.filter(v => v !== value);
  logFilters.exclude[field] = exc.filter(v => v !== value);
  if (at === '') logFilters.include[field].push(value);
  else if (at === 'inc') logFilters.exclude[field].push(value);
  refreshRuns();
}

function lvActive() {
  return LV_FIELDS.some(f => (logFilters.include[f.field] || []).length
                          || (logFilters.exclude[f.field] || []).length);
}

function logAttr(value) {
  return esc(String(value)).replace(/"/g, '&quot;');
}

function paintFilters() {
  const facets = (logData && logData.facets) || {};
  const primary = new Set(['model', 'extract_model', 'extract_mode']);
  const blocks = LV_FIELDS.map(f => {
    const values = facets[f.field] || [];
    if (!values.length) return null;
    return {field: f.field, html: '<div class="lv-field"><b title="' + logAttr(f.hint) + '">' + esc(f.label)
      + '</b><div class="lv-chips">'
      + values.map(v => {
          const at = lvState(f.field, v.value);
          return '<button type="button" class="lv-chip' + (at ? ' ' + at : '')
            + '" aria-label="' + logAttr(f.label + ': ' + lvChipLabel(f.field, v.value)
              + (at === 'inc' ? ' (included)' : at === 'exc' ? ' (excluded)' : ' (any)'))
            + '" data-field="' + logAttr(f.field) + '" data-value="' + logAttr(v.value)
            + '" title="' + logAttr(v.value) + ' — ' + v.runs + ' row(s) in the log. '
            + (at === 'inc' ? 'Kept. Click to drop it.'
               : at === 'exc' ? 'Dropped. Click to clear.'
               : 'Click to keep only this, again to drop it.') + '">'
            + esc(lvChipLabel(f.field, v.value)) + '<span class="n">' + v.runs
            + '</span></button>';
        }).join('')
      + '</div></div>'};
  }).filter(Boolean);
  lvMainFacets.innerHTML = blocks.filter(b => primary.has(b.field)).map(b => b.html).join('');
  lvExtraFacets.innerHTML = blocks.filter(b => !primary.has(b.field)).map(b => b.html).join('');
  lvFilterHelp.innerHTML = '<p class="note" style="margin-top:8px">Click a chip to keep <b>only</b> '
    + 'that value, again to <b>drop</b> it, again to clear. Chips are counted '
    + 'over the whole log, never over what survived the filter, so a value you '
    + 'have just dropped is still here to put back. <b>A dropped row is absent '
    + 'from every table</b> — not hidden but uncounted: it is out of the run '
    + 'counts, the failure rates and the window, exactly as if it had not been '
    + 'run. The Run history panel is never filtered, because a row nobody can see '
    + 'is a row nobody can delete. <b>Reading model is blank on a run that read '
    + 'no page</b> — a fields-only run carries the model that extracted, not one '
    + 'that read — so those runs are not in that picker at all. To see one half '
    + 'of the pipeline, use <b>Pipeline</b> above rather than turning off every '
    + 'model. <b>Dropping an outcome is the one narrowing that changes what the '
    + 'figures mean</b> — a failed run is already counted and never scored, so '
    + 'dropping the failures does not remove a bad number from a mean, it '
    + 'removes the failure rate printed beside it and leaves every accuracy '
    + 'reading as though nothing had gone wrong.'
    + (lvActive() ? ' <b>A filter is in force.</b>' : '') + '</p>';
}

/* ---------- pipeline ----------
   One choice, not a chip group: the three kinds are mutually exclusive and a row
   is exactly one of them, so a tri-state chip per kind would let a reader ask for
   a combination that means the same as "Any" by three different routes.

   It rides in `include.pipeline` like any other filter -- the server treats it as
   a field like the rest (`runlog._pipeline`), so nothing downstream needs to know
   it is special. */
const PIPE_LABEL = {
  '': 'Any', both: 'Both passes', read: 'Read only', extract: 'Extract only'
};

function pipeNow() {
  return (logFilters.include.pipeline || [])[0] || '';
}

/* What the pipeline choice means for one of the two setting tables. Printed in
   the panel rather than left to be inferred from an empty table, because both
   emptinesses have an innocent reading that is wrong here:

   - **Pass 1 under "Extract only" is empty because nothing read a page**, not
     because no setting has been measured.
   - **Pass 2 under "Read only" is not empty, and that is the misleading one.**
     A read-only round that LOOPED is pulled into `by_extract` by its status --
     the rule there is that an incomplete run is one pass 2 never got to -- and a
     round that never asked for pass 2 is indistinguishable in the CSV from one
     that died before reaching it: both have blank pass-2 columns and a blank
     `extract_mode`. So the table shows a handful of settings at a 100% failure
     rate and no score, which reads as "these settings cannot extract" when what
     it means is "these runs never tried". */
function pipeNote(pass) {
  const now = pipeNow();
  if (!now || now === 'both') return '';
  if (pass === 'ocr' && now === 'extract')
    return '<p class="note"><b>Nothing to show under &ldquo;Extract only&rdquo;.</b> '
      + 'Those runs extracted from a transcript they did not read, so they have no '
      + 'pass-1 figures at all — this table is empty because no page was read, not '
      + 'because no setting has been measured.</p>';
  if (pass === 'extract' && now === 'read')
    return '<p class="note"><b>Under &ldquo;Read only&rdquo; every row here is a '
      + 'failure, and none of them tried to extract.</b> A read-only run that '
      + 'looped or errored is counted as an extraction that never got started, '
      + 'because a run that never asked for pass&nbsp;2 and one that died before '
      + 'reaching it look identical in the log — both leave the pass-2 columns and '
      + '<code>extract_mode</code> blank. Read these as pass-1 failures; the '
      + 'extraction figures to compare are under <b>Both passes</b> or '
      + '<b>Extract only</b>.</p>';
  return '';
}

/* The same explanation for the route that reaches the same place WITHOUT the
   pipeline picker: excluding every reading model leaves only runs that read no
   page, which empties this table exactly as "Extract only" does. The note has to
   fire on the state, not on the control that produced it -- there are two ways
   in and only one of them was covered. */
function emptyPassNote(pass, t) {
  const specific = pipeNote(pass);
  if (specific) return specific;
  if (!t || !t.matched) return '';       // the no-rows banner above says it
  const noun = pass === 'ocr' ? 'read a page' : 'extracted any fields';
  const fix = pass === 'ocr'
    ? 'Every run left in this view read no page — check the <b>Reading model</b> '
      + 'chips and <b>Pipeline</b>.'
    : 'Every run left in this view extracted nothing — check the <b>Extraction '
      + 'model</b> chips and <b>Pipeline</b>.';
  return '<p class="note"><b>No ' + (pass === 'ocr' ? 'pass-1' : 'pass-2')
    + ' figures in this view.</b> None of the ' + t.matched + ' runs that match '
    + 'the filter ' + noun + '. ' + fix + '</p>';
}

function paintPipe() {
  const now = pipeNow();
  lvPipe.querySelectorAll('button').forEach(b =>
    b.classList.toggle('on', (b.dataset.pipe || '') === now));
}

lvPipe.addEventListener('click', ev => {
  const btn = ev.target.closest ? ev.target.closest('button[data-pipe]') : null;
  if (!btn || logBusy) return;
  const want = btn.dataset.pipe || '';
  // Clicking the choice already in force clears it, so the control can always be
  // undone from itself rather than only from Reset.
  logFilters.include.pipeline = (want && want !== pipeNow()) ? [want] : [];
  refreshRuns();
});

/* Every narrowing in force, in the words the controls use. Printed only when the
   result is empty, where it is the whole of what a reader needs: the card cannot
   say what is missing, but it can say what it was told to leave out. */
function lvInForce() {
  const out = [];
  const pipe = pipeNow();
  if (pipe) out.push('<b>' + PIPE_LABEL[pipe] + '</b>');
  LV_FIELDS.forEach(f => {
    const inc = logFilters.include[f.field] || [];
    const exc = logFilters.exclude[f.field] || [];
    // Named the way its own chip is, not through `csModel` -- that is a model
    // name trim, and it printed a raw `ok` where the chip beside it reads
    // "finished". One vocabulary for the control and for the sentence about it.
    if (inc.length) out.push('only ' + f.label.toLowerCase() + ' <b>'
      + inc.map(v => esc(lvChipLabel(f.field, v))).join(', ') + '</b>');
    if (exc.length) out.push('no ' + f.label.toLowerCase() + ' <b>'
      + exc.map(v => esc(lvChipLabel(f.field, v))).join(', ') + '</b>');
  });
  return out;
}

function paintEmpty(t) {
  lvEmpty.hidden = currentLogGroup === 'history';
  if (t.matched !== 0) { lvEmpty.innerHTML = ''; return; }
  const forced = lvInForce();
  lvEmpty.innerHTML = '<b>No runs match the summary filters.</b> The summary tables are empty because '
    + 'nothing in the log got past the filter'
    + (forced.length ? ' — in force: ' + forced.join(' &middot; ') : '')
    + '. The log itself is untouched (' + t.logged + ' rows) and the <b>Run history</b> '
    + 'panel still shows it. Clear a chip above, or press <b>Reset filters</b>.';
}

function paintLogControls() {
  const t = (logData && logData.totals) || {};
  const d = (logData && logData.defaults) || {};
  // Painted from what came BACK, never from what was asked for -- the server
  // clamps both, and a control showing a figure the tables were not compiled
  // under is a control that lies. Same rule the model pickers follow.
  if (document.activeElement !== lvFloor)
    lvFloor.value = t.min_read_pct === null || t.min_read_pct === undefined
      ? '' : t.min_read_pct;
  if (document.activeElement !== lvWindow)
    lvWindow.value = t.window === null || t.window === undefined ? 0 : t.window;
  // The single-source toggle is client state, so it is reflected here rather than
  // read back from the server. Disabled when the current view has no single-source
  // failure to drop -- a live checkbox that would change nothing is a control that
  // lies -- but never while it is on, or it could not be turned back off.
  const anyDom = t.errors && ['read', 'extract'].some(p => t.errors[p]
    && [...(t.errors[p].models || []), ...(t.errors[p].cases || [])]
        .some(r => r.sources === 1));
  errDomToggle.checked = errDropSingle;
  errDomToggle.disabled = !anyDom && !errDropSingle;
  const bits = [];
  if (t.matched !== undefined && t.logged !== undefined && t.matched !== t.logged)
    bits.push('<b>' + t.matched + ' of ' + t.logged + ' rows</b> match the filter');
  else if (t.logged !== undefined)
    bits.push(t.logged + ' rows in the log');
  bits.push(t.window ? 'up to ' + t.window + ' recent runs per setting, model and '
                       + 'document' : 'every run, no window');
  bits.push(t.min_read_pct
    ? 'field scores from reads under ' + Number(t.min_read_pct).toFixed(0)
      + '% are left out'
    : 'every field score counted, whatever the read scored');
  // A run-level narrowing set from the Errors tab, worth naming here because it
  // changes every table except the raw log and the Errors one, and nothing else
  // on the card would say why the numbers moved.
  if (errDropSingle)
    bits.push('<b>' + (t.single_source_dropped || 0) + ' single-source failure'
      + ((t.single_source_dropped === 1) ? '' : 's') + '</b> excluded from these '
      + 'tables (Errors tab)');
  const moved = (d.window !== undefined && t.window !== (d.window || null))
    || (d.min_read_pct !== undefined && t.min_read_pct !== d.min_read_pct);
  lvNote.innerHTML = bits.join(' &middot; ')
    + (moved ? ' &middot; <b>not the server default</b> (' + d.window + ' runs, '
               + Number(d.min_read_pct).toFixed(0) + '%)' : '');
  const applied = lvInForce();
  if (t.min_read_pct) applied.push('Read floor: <b>' + esc(String(t.min_read_pct)) + '%</b>');
  if (t.window) applied.push('Runs averaged: <b>' + esc(String(t.window)) + '</b> per group');
  if (errDropSingle) applied.push('Drop single-source failures');
  if (presBiased) applied.push('Exclude weak models <b>(Overview only)</b>');
  lvApplied.innerHTML = '<span>' + (t.matched ?? 0) + ' / ' + (t.logged ?? 0)
    + ' rows match summary filters</span>'
    + (applied.length ? applied.map(x => '<span class="active">' + x + '</span>').join('')
       : '<span>No summary filters · all runs averaged</span>');
  const primary = new Set(['model', 'extract_model', 'extract_mode']);
  const extraCount = LV_FIELDS.filter(f => !primary.has(f.field)).reduce((n, f) => n
    + (logFilters.include[f.field] || []).length + (logFilters.exclude[f.field] || []).length, 0)
    + (t.min_read_pct ? 1 : 0) + (t.window ? 1 : 0)
    + (errDropSingle ? 1 : 0) + (presBiased ? 1 : 0);
  lvAdvancedCount.textContent = extraCount ? '· ' + extraCount + ' active' : '';
  paintLogScope();
}

/* ---------- raw rows ---------- */
function lvNum(v) {
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  return isNaN(n) ? null : n;
}
/* Header labels and sortable values for the log itself. Values correspond to
   runRow; the selection and details controls are deliberately not sortable. */
const RAW_COLS = [
  {key: 'pick', label: '', sortable: false, cls: 'pick'},
  {key: 'when', label: 'When', n: true, value: r => r.timestamp || ''},
  {key: 'doc', label: 'Document', value: r => r.file || ''},
  {key: 'model', label: 'Model', value: r => r.model || ''},
  {key: 'tokens', label: 'Tokens', n: true,
   value: r => lvNum(r.run_type === 'extract' ? r.extract_tokens : r.tokens)},
  {key: 'time', label: 'Time', n: true,
   value: r => lvNum(r.run_type === 'extract' ? r.extract_seconds : r.seconds)},
  {key: 'fields', label: 'Fields', n: true, value: r => lvNum(r.p1_present)},
  {key: 'accuracy', label: 'Accuracy', n: true, value: r => lvNum(r.char_accuracy)},
  {key: 'grounded', label: 'Grounded &middot; fields', n: true,
   title: 'grounded: found on the page. fields: correct against the field ground '
        + 'truth, where there is one',
   value: r => lvNum(r.grounded_pct)},
  {key: 'status', label: 'Status', value: r => r.status || ''},
  {key: 'details', label: 'Details', sortable: false}
];

function runIdentity(r) {
  return JSON.stringify([r._row, r.timestamp, r.file]);
}

function runDetails(r, jsonOpen) {
  const fields = [
    ['Timestamp', r.timestamp], ['Document', r.file], ['Run type', r.run_type],
    ['Logged model', r.model], ['Extraction model', r.extract_model],
    ['Extraction shape', r.extract_mode],
    ['JSON schema first', ['1', 'true', 'on'].includes(String(r.extract_schema_first).toLowerCase())],
    ['Detail', r.detail], ['Status', r.status],
    ['Reading time (s)', r.seconds], ['Extraction time (s)', r.extract_seconds],
    ['Reading output tokens', r.tokens], ['Extraction output tokens', r.extract_tokens]
  ].filter(([, value]) => value !== undefined && value !== null && value !== '');
  return '<tr class="run-detail"><td colspan="' + RAW_COLS.length + '">'
    + '<b>Saved run details</b><p class="note">Only values stored in this log row are shown. '
    + 'Full Stress test settings and reports are in Stress test → History.</p><dl>'
    + fields.map(([label, value]) => '<div><dt>' + esc(label) + '</dt><dd>'
      + esc(String(value)) + '</dd></div>').join('') + '</dl>'
    + '<details' + (jsonOpen ? ' open' : '') + '><summary>All saved values (JSON)</summary><pre>'
    + esc(JSON.stringify(r, null, 2)) + '</pre></details></td></tr>';
}

function paintRunRows() {
  const jsonOpen = runBody.querySelector('.run-detail details')?.open || false;
  const rows = (logData && logData.runs) || [];
  runBody.innerHTML = rows.length
    ? '<table class="queue">' + lvHead('raw', RAW_COLS)
      + lvSorted('raw', rows, RAW_COLS).map(r => runRow(r)
        + (rawExpanded === runIdentity(r) ? runDetails(r, jsonOpen) : '')).join('') + '</table>'
    : '<p class="note">No runs have been logged yet.</p>';
  const n = rawPicked.size;
  rawSel.textContent = n ? n + ' row(s) selected — deleting archives them to '
                           + 'logs/runs.deleted.csv'
                         : 'Tick a row to delete it. Nothing is selected.';
  rawDelete.disabled = !n || logBusy;
  rawNone.disabled = !n || logBusy;
}

runBody.addEventListener('click', ev => {
  const button = ev.target.closest ? ev.target.closest('button[data-run-detail]') : null;
  if (!button) return;
  const row = ((logData && logData.runs) || []).find(r => r._row === Number(button.dataset.runDetail));
  if (!row) return;
  rawExpanded = rawExpanded === runIdentity(row) ? null : runIdentity(row);
  paintRunRows();
  runBody.querySelector('button[data-run-detail="' + Number(row._row) + '"]')?.focus({preventScroll: true});
});

runBody.addEventListener('change', ev => {
  const box = ev.target.closest ? ev.target.closest('input[data-pick]') : null;
  if (!box) return;
  const index = Number(box.dataset.pick);
  if (box.checked) rawPicked.add(index); else rawPicked.delete(index);
  const n = rawPicked.size;
  rawSel.textContent = n ? n + ' row(s) selected — deleting archives them to '
                           + 'logs/runs.deleted.csv'
                         : 'Tick a row to delete it. Nothing is selected.';
  rawDelete.disabled = !n;
  rawNone.disabled = !n;
});

rawNone.addEventListener('click', () => { rawPicked.clear(); paintRunRows(); });

rawDelete.addEventListener('click', async () => {
  const rows = ((logData && logData.runs) || [])
    .filter(r => rawPicked.has(r._row))
    // The whole key, not the index alone: the server checks it against the row
    // it finds there and refuses one that has moved. A delete is permanent and
    // silent when it lands on the wrong row.
    .map(r => ({_row: r._row, timestamp: r.timestamp, file: r.file}));
  if (!rows.length) return;
  if (!confirm('Delete ' + rows.length + ' run(s) from the log?\n\n'
               + 'They are removed from every table and every mean, and a copy '
               + 'is kept in logs/runs.deleted.csv.')) return;
  logBusy = true; rawDelete.disabled = true;
  let out = null;
  try {
    out = await (await fetch(API + '/runs/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rows: rows})
    })).json();
  } catch { out = null; }
  logBusy = false;
  rawPicked.clear();
  await refreshRuns();
  if (!out || out.error) {
    rawSel.textContent = 'Could not delete: ' + ((out && out.error) || 'no reply.');
  } else if (out.stale) {
    // The log moved under the page -- a run finished while rows were ticked.
    // Named rather than retried: the row that is there now is not the row that
    // was ticked, and deleting it anyway is the mistake this guard exists for.
    rawSel.textContent = out.deleted + ' deleted, ' + out.stale
      + ' skipped — the log changed since this table was drawn. Refreshed.';
  }
});

/* ---------- panels ---------- */
/* One panel per question rather than one long column, at the user's request.
   Only one of them is the log: the summaries are compiled and filtered, the last
   is what was written.

   **The list is derived from the buttons, never written out beside them.** It
   was a literal array until 2026-08-25, and adding the Summary tab is exactly
   the edit that array was waiting for -- the same failure the side tabs already
   hit, where a fourth tab left every pane hidden with the button highlighted and
   no error anywhere. A tab and its pane are one thing; naming them in two places
   means one is eventually forgotten. */
function paintLogScope() {
  const t = (logData && logData.totals) || {};
  const rows = (logData && logData.runs) || [];
  const history = currentLogGroup === 'history';
  logViewNote.textContent = history
    ? 'Individual logged runs. Summary filters above do not apply here.'
    : 'Aggregated results using the summary filters above. Open Run history to inspect individual runs.';
  rawScope.textContent = 'Showing the latest ' + rows.length + ' of ' + (t.logged ?? rows.length)
    + ' logged rows (up to 50). Summary filters do not hide these rows. Download CSV for the complete log.';
  lvEmpty.hidden = history;
  runStats.hidden = history;
}

function showLogPanel(want) {
  const selected = logTabs.querySelector('button[data-panel="' + want + '"]');
  if (!selected) return;
  currentLogGroup = selected.dataset.group;
  lastLogPanel[currentLogGroup] = want;
  let count = 0;
  logTabs.querySelectorAll('button[data-panel]').forEach(b => {
    const name = b.dataset.panel;
    const active = name === want;
    b.hidden = b.dataset.group !== currentLogGroup;
    if (!b.hidden) count++;
    b.classList.toggle('on', active);
    b.setAttribute('aria-pressed', String(active));
    const el = document.getElementById('panel' + name[0].toUpperCase() + name.slice(1));
    if (el) el.hidden = !active;
  });
  logTabs.hidden = count < 2;
  logGroups.querySelectorAll('button[data-log-group]').forEach(b => {
    const active = b.dataset.logGroup === currentLogGroup;
    b.classList.toggle('on', active);
    b.setAttribute('aria-pressed', String(active));
  });
  paintLogScope();
}
logGroups.addEventListener('click', ev => {
  const button = ev.target.closest ? ev.target.closest('button[data-log-group]') : null;
  if (button) showLogPanel(lastLogPanel[button.dataset.logGroup]);
});
logTabs.addEventListener('click', ev => {
  const button = ev.target.closest ? ev.target.closest('button[data-panel]') : null;
  if (button) showLogPanel(button.dataset.panel);
});
showLogPanel(lastLogPanel[currentLogGroup]);

lvFilters.addEventListener('click', ev => {
  const chip = ev.target.closest ? ev.target.closest('.lv-chip') : null;
  if (!chip || logBusy) return;
  lvCycle(chip.dataset.field, chip.dataset.value);
});

lvApply.addEventListener('click', () => {
  const floor = lvFloor.value === '' ? null : Number(lvFloor.value);
  const window_ = lvWindow.value === '' ? null : Number(lvWindow.value);
  logView.min_read_pct = floor === null || isNaN(floor) ? null : floor;
  logView.window = window_ === null || isNaN(window_) ? null : window_;
  refreshRuns();
});
[lvFloor, lvWindow].forEach(el => el.addEventListener('keydown', ev => {
  if (ev.key === 'Enter') { ev.preventDefault(); lvApply.click(); }
}));

lvReset.addEventListener('click', () => {
  // Back to the process settings AND to no filter: "reset" meaning half of it
  // would leave a view nobody asked for looking like the default one.
  logView = {window: null, min_read_pct: null};
  // `include.pipeline` lives in here too, so Reset clears the pipeline choice
  // with everything else -- a "reset" that left one narrowing standing would
  // leave a view nobody asked for looking like the default one.
  logFilters = {include: {}, exclude: {}};
  // The single-source exclusion is a narrowing like any other, so Reset clears it
  // too -- a "reset" that left it on would leave tables nobody asked for looking
  // like the default ones.
  errDropSingle = false;
  // Cleared like every other narrowing, now that showing everything is this
  // tab's default too. It was deliberately exempt while the bias was the
  // default and that exemption went with it -- a Reset that left one narrowing
  // standing leaves a view nobody asked for looking like the default one.
  presBiased = false;
  refreshRuns();
});

/* ---------- what each model and each document is WEAK at ----------
   Added 2026-09-04 at the user's request -- *add to summary to check each model
   strength and weakness, also each doc to see in each doc the model weakness is
   in what field*.

   Every other tab on this card answers *which is best*, and a mean is the wrong
   shape for *what is it bad at*: a model reading 92% overall can be reading 71%
   of the digits, and the digits are every Mandatory field in the requirement.
   Same one level down in pass 2 -- a field score of 8 of 11 says nothing about
   WHICH three, and it is the same three on every document until somebody looks.

   Four tables, two per pass, and each pass asks the same question of a model and
   of a document. From `runlog.script_accuracy` and `runlog.field_weakness`; the
   eligibility rules live there. */

/* One field name as the page spells it. `FIELD_LABELS` is the Fields tab's own
   map, so a key is named the same on both -- a key it does not know (one this
   build no longer asks for, still in the log) falls back to its raw name rather
   than being dropped, because a run that happened is a run that happened. */
function wkField(key) {
  return (FIELD_LABELS && FIELD_LABELS[key]) || key;
}

/* A rate as a coloured cell of a grid, on the same scale as everything else on
   this card. `null` is NOT a zero -- it is a cell nothing measured -- so it is
   an em dash rather than a red 0%. */
function wkCell(rate, tip, sub) {
  // A cell with no rate still has something to say where the reason is that
  // the model INVENTED a value: on the keys this corpus leaves empty by
  // construction that is the whole story, and a bare dash would report the one
  // measurable failure of those keys as nothing having been measured.
  if (rate === null || rate === undefined)
    return '<td class="an-empty" title="' + esc(tip || '') + '">&mdash;'
      + (sub ? '<span class="an-t">' + esc(sub) + '</span>' : '') + '</td>';
  return '<td class="an-cell" title="' + esc(tip || '') + '" style="background:'
    + hexA(accColour(rate / 100), 0.16) + '"><b style="color:'
    + accColour(rate / 100) + '">' + rate.toFixed(0) + '%</b>'
    + (sub ? '<span class="an-t">' + esc(sub) + '</span>' : '') + '</td>';
}

/* The three script rates for one model or one document, as a sortable table.
   Numbers first, against the order used elsewhere: every Mandatory field in the
   requirement is a figure, a date or an ID, so that is the column to read first
   and the one a reader should not have to hunt for. */
function wkScriptTable(id, entries, what, labels) {
  // A document row is meaned over the MODELS that read it, so its spread is
  // across models and not across documents. Naming the wrong axis in a tooltip
  // is the kind of small lie that gets a table quoted wrongly.
  const spread = what === 'model' ? 'across docs' : 'across models';
  const cols = [
    {key: 'key', label: what === 'model' ? 'Model' : 'Document',
     value: e => csModel(e.key),
     cell: e => '<b title="' + esc(e.key) + '">' + esc(csModel(e.key)) + '</b>'
       + '<span class="cs-set">' + e.runs + ' read(s) &middot; ' + e.documents
       + (what === 'model' ? ' document(s)' : ' model(s)') + '</span>'},
    {key: 'char', label: 'Transcript', n: true, value: e => e.char.mean,
     title: 'every content character — the headline this card ranks on '
          + 'elsewhere. The three columns to its right are recall over three '
          + 'subsets of it and do NOT add up to it: punctuation and symbols '
          + 'belong to no script',
     cell: e => msFig(e.char, '%', 1, true, spread)},
    {key: 'digit', label: 'Numbers', n: true, value: e => e.digit.mean,
     title: 'digits, Arabic and Thai numerals — amounts, dates, tax IDs and '
          + 'document numbers',
     cell: e => msFig(e.digit, '%', 1, true, spread)
       + wkChars(e.digit_chars, 'digits')},
    {key: 'thai', label: 'Thai', n: true, value: e => e.thai.mean,
     title: 'Thai letters, tone marks and vowels included',
     cell: e => msFig(e.thai, '%', 1, true, spread)
       + wkChars(e.thai_chars, 'Thai chars')},
    {key: 'latin', label: 'English', n: true, value: e => e.latin.mean,
     title: 'the Latin alphabet. Not a claim that the text is English: a '
          + 'romanised Thai company name is Latin script too',
     cell: e => msFig(e.latin, '%', 1, true, spread)
       + wkChars(e.latin_chars, 'Latin chars')},
    // The answer said out loud. A reader can find the lowest of three numbers;
    // what the numbers do not say is whether the gap is worth acting on, and a
    // two-point gap is noise where a twenty-point one is a finding.
    {key: 'gap', label: 'Weakest', n: true, value: e => e.gap,
     title: 'the lowest of the three, and how far below the highest. Blank '
          + 'where fewer than two of the three could be measured',
     cell: e => e.weakest
       ? '<span class="cs-val" style="color:' + (e.gap >= 10 ? 'var(--danger)'
           : e.gap >= 5 ? 'var(--warn)' : 'var(--muted)') + '">'
         + esc(labels[e.weakest] || e.weakest) + '</span>'
         + '<span class="cs-set">' + e.gap.toFixed(1) + ' pts below its best</span>'
       : '<span class="cs-none">&mdash;</span>'}
  ];
  return lvTable(id, entries, cols, {sort: {key: 'char', dir: 'desc'}});
}

/* The denominator under a script rate. "68% of ~1900 Thai characters" is a
   finding and "68% of six" is not, and only the count separates them. */
function wkChars(mean, what) {
  if (mean === null || mean === undefined) return '';
  return '<span class="cs-set">of ~' + Math.round(mean) + ' ' + esc(what)
    + '</span>';
}

/* A model x field, document x field or type x field grid. The field ORDER is
   the ranking -- weakest overall first -- and it is the same order in every
   grid, so a column means the same thing wherever it is read.

   `opts.labels` names a document-type row (`prompts.TYPE_NAMES`, trimmed of its
   article by `tmType` exactly as the Model x type tab does, so a type reads the
   same on both tabs); a model or document row is its own key. */
function wkFieldGrid(rows, order, what, required, opts) {
  if (!rows.length || !order.length) return '';
  const o = opts || {};
  const corner = what === 'model' ? 'Model' : what === 'type'
    ? 'Document type' : 'Document';
  const rowName = e => what === 'type' ? tmType(e.key, o.labels) : csModel(e.key);
  const head = '<tr><th class="an-corner">' + corner + ' \\ Field</th>'
    + order.map(f => '<th title="' + esc(f
        + (required[f] ? ' — the requirement demands it ' + required[f] : ''))
        + '">' + esc(wkField(f)) + '</th>').join('')
    + '<th title="this row&rsquo;s mean over the fields it was judged on">Mean</th></tr>';
  const body = rows.map(e => {
    const cells = order.map(f => {
      const c = e.cells[f];
      if (!c) return '<td class="an-empty" title="not asked of this '
        + (what === 'model' ? 'model' : what === 'type' ? 'document type'
           : 'document') + '">&mdash;</td>';
      const counts = c.counts || {};
      // The counts in the tooltip rather than in the cell: `missed` and
      // `spurious` are OPPOSITE mistakes wanting opposite fixes, and a cell
      // showing one rate cannot say which of them it is made of.
      const tip = rowName(e) + ' · ' + f + ' — '
        + ['correct', 'partial', 'wrong', 'missed', 'spurious', 'absent']
            .filter(k => counts[k]).map(k => counts[k] + ' ' + k).join(', ');
      return wkCell(c.rate, tip,
                    c.spurious ? c.spurious + ' invented'
                    : c.rate !== null && c.rate < 100 && c.worst ? c.worst : '');
    }).join('');
    // A grouped type column must name what it holds, or the catch-all is a row
    // nobody can take apart; a model or document row is its own name.
    const members = (e.members || [])
      .map(m => tmType(m[0], o.labels) + ' ×' + m[1]).join(', ');
    const sub = e.runs + ' run(s)'
      + (what === 'type' ? ' · ' + e.inner + ' doc(s)' : '');
    return '<tr><td class="an-rowh" title="' + esc(e.key
        + ((e.members || []).length > 1 ? ' — ' + members : '')) + '">'
      + esc(rowName(e)) + '<span class="an-t">' + esc(sub) + '</span></td>'
      + cells
      + (e.mean === null
         ? '<td class="an-mean-col an-empty">&mdash;</td>'
         : '<td class="an-mean-col"><b style="color:' + accColour(e.mean / 100)
           + '">' + e.mean.toFixed(0) + '%</b></td>')
      + '</tr>';
  }).join('');
  return '<div class="an-grid"><table class="an-heat">' + head + body
    + '</table></div>';
}

/* Every key, weakest first, pooled over everything -- what is hard, before the
   question of who finds it hard. */
function wkFieldTable(fields) {
  const cols = [
    {key: 'field', label: 'Field', value: e => wkField(e.field),
     cell: e => '<b title="' + esc(e.field) + '">' + esc(wkField(e.field)) + '</b>'
       + '<span class="cs-set">' + (e.required === 'always'
           ? '<span class="tier p2">required</span>'
           : e.required === 'sometimes'
             ? 'required on ' + e.required_runs + ' of ' + e.asked_runs + ' run(s)'
             : 'not required') + '</span>'},
    {key: 'rate', label: 'Correct', n: true, value: e => e.rate,
     title: 'correct, counting a partial as half — the same arithmetic the '
          + 'field score uses, so this and field accuracy cannot disagree. '
          + 'Blank where every verdict was absent or spurious: there was '
          + 'nothing on the page to be right about',
     cell: e => e.rate === null
       ? '<span class="cs-none">&mdash;</span>'
       : '<span class="cs-val" style="color:' + accColour(e.rate / 100) + '">'
         + e.rate.toFixed(0) + '%</span><span class="cs-set">of ' + e.scored
         + ' scored</span>'},
    {key: 'worst', label: 'How it fails', value: e => e.worst,
     title: 'missed and spurious are OPPOSITE mistakes and neither means '
          + 'empty: missed is the page stating it and the extractor returning '
          + 'nothing, spurious is the extractor filling a key the page leaves '
          + 'blank. They want opposite fixes',
     cell: e => {
       const c = e.counts || {};
       const bits = ['wrong', 'missed', 'spurious'].filter(k => c[k])
         .map(k => '<b>' + c[k] + '</b> ' + k);
       return bits.length ? bits.join(' &middot; ')
         : '<span class="cs-none">nothing wrong</span>';
     }},
    {key: 'runs', label: 'Judged', n: true, value: e => e.runs,
     title: 'how many extractions this key was judged on at all, including the '
          + 'ones where both sides were empty and it was scored neither way',
     cell: e => e.runs}
  ];
  return lvTable('wkFields', fields, cols, {sort: {key: 'rate', dir: 'asc'}});
}

function paintWeak(t) {
  const sa = t.script_accuracy || {};
  const fw = t.field_weakness || {};
  const labels = sa.labels || {};
  const required = {};
  (fw.fields || []).forEach(e => {
    required[e.field] = e.required === 'always' ? 'of every document here'
      : e.required === 'sometimes'
        ? 'on ' + e.required_runs + ' of ' + e.asked_runs + ' run(s)' : '';
  });

  let html = '<span class="cs-title">Pass 1 &mdash; which script each model '
    + 'loses' + windowNote('model') + '</span>' + pipeNote('ocr');
  // Each grid is windowed by the thing it groups on -- a model row is that
  // model's own recent reads, a document row is that document's -- so the note
  // has to name a different axis per table rather than once at the top.

  if (!(sa.by_model || []).length) {
    html += '<p class="note">No read in view carries the per-script figures. '
      + 'They were added on 2026-09-04, so a run from before that has a '
      + 'transcript score and no breakdown &mdash; which is not a low score, and '
      + 'is left out rather than counted as zero. Read a benchmark document and '
      + 'it will appear here.</p>';
  } else {
    html += wkScriptTable('wkScriptModel', sa.by_model, 'model', labels)
      + '<span class="cs-title">&hellip; and which document loses it'
      + windowNote('document') + '</span>'
      + wkScriptTable('wkScriptCase', sa.by_case, 'case', labels)
      + '<p class="note">The same recall as the transcript score, taken over the '
      + 'ground truth&rsquo;s own characters of each script. <b>The three do not '
      + 'add up to the transcript column</b> &mdash; punctuation, symbols and '
      + 'currency signs belong to no script and are in the headline only. A '
      + 'document printing fewer than ' + esc(String(sa.min_chars || ''))
      + ' characters of a script reports no rate for it: three characters score '
      + '0% or 100% and nothing in between. Models are keyed on the model alone, '
      + 'without Detail or profile &mdash; this asks what a model is bad at '
      + 'across everything it has read, not which setting to run, which is '
      + '<b>Best reading</b>&rsquo;s question. Documents are ordered worst first: '
      + 'this table is a work list, not a ranking.</p>';
  }

  html += '<span class="cs-title">Pass 2 &mdash; which field goes wrong'
    + windowNote('document') + '</span>' + pipeNote('extract');
  if (!(fw.fields || []).length) {
    html += '<p class="note">No extraction in view carries per-field verdicts. '
      + 'They were added on 2026-09-04 and are written only where the document '
      + 'has a field ground truth (<code>solution/&lt;id&gt;.fields.json</code>) '
      + 'and the read was good enough to judge the extraction by. Run a '
      + 'benchmark document, or a fields-only test, and it will appear here.</p>';
  } else {
    const typeGrid = wkFieldGrid(fw.by_type || [], fw.order || [], 'type',
                                 required, {labels: fw.type_labels});
    html += wkFieldTable(fw.fields)
      + '<span class="cs-title">Each document type, field by field'
      + windowNote('document type') + '</span>'
      + (typeGrid || '<p class="note">No extraction in view could be placed '
          + 'against a document type.</p>')
      + (fw.packs ? '<p class="note">' + fw.packs + ' extraction(s) of a '
          + 'multi-document file are left out of <b>this grid alone</b>: one '
          + 'row of the log carries one set of verdicts for several documents '
          + 'of <i>different</i> types, so unlike the transcript and field '
          + 'scores there is nothing per document to take apart. They are in '
          + 'every other table here.</p>' : '')
      + '<span class="cs-title">Each model, field by field'
      + windowNote('model') + '</span>'
      + wkFieldGrid(fw.by_model || [], fw.order || [], 'model', required)
      + '<span class="cs-title">Each document, field by field'
      + windowNote('document') + '</span>'
      + wkFieldGrid(fw.by_case || [], fw.order || [], 'case', required)
      + '<p class="note">Every cell is the share of that key the extractor got '
      + 'right, a partial counting half &mdash; the same arithmetic the field '
      + 'score uses, so a cell here and the field accuracy on the same run '
      + 'cannot disagree. <b>The column order is the ranking</b>, weakest key '
      + 'first, and it is the same in every grid so a column means the same '
      + 'thing in each. <b>The type grid is the axis the requirement is '
      + 'written on</b> &mdash; a key that is hard on a KIND of page is a '
      + 'prompt or a ground-truth question, where a key one model alone loses '
      + 'is a model question. Its columns are the four requirement types and '
      + 'one catch-all, the same grouping <b>Model &times; type</b> uses, so a '
      + 'type reads the same on both tabs. A cell is blank where nothing measured it: the key was '
      + 'not asked of that document&rsquo;s type, or every verdict on it was '
      + '<i>absent</i> (both sides empty, which is agreement) or <i>spurious</i> '
      + '(the page states nothing, so there is nothing to be right about) '
      + '&mdash; the second is called out under the cell, because on the keys '
      + 'this corpus leaves empty by construction it is the whole story. '
      + 'Extractions whose read scored under the floor above are excluded: a '
      + 'field score taken over a broken transcript is the read&rsquo;s mistake '
      + 'wearing the extractor&rsquo;s name. <b>This is not a model ranking</b> '
      + '&mdash; <b>Best extraction</b> and <b>Full rank</b> answer that, over '
      + 'the same runs.</p>';
  }
  weakSum.innerHTML = html;
}

/* ---------- model x document type ----------
   From `runlog.type_models`, added 2026-09-11 at the user's request -- *a tab
   summary model vs doc type, and a summary of what doc type performs best and
   what model is best for what type, in both OCR and extraction*.

   Every figure here is one the card already holds. What is new is the AXIS:
   **Best reading**, **Best extraction** and **Full rank** all average over the
   KIND of page, so a model that reads printed invoices well and handwritten
   receipts badly reports one middling number and nothing on this card said why.

   The winning cell of each column is ringed, and it is the SERVER&rsquo;s `best`
   rather than the highest number in the column -- so a model with one lucky run
   does not take the ring off a better-evidenced one. Same rule as the `ms-top`
   highlight: a ranking is a claim the reader&rsquo;s sort must not be able to
   rewrite by accident. */

/* A type named for a heading: the noun, without the article a prompt uses.
   `scTypeName` is the same trim the Doc types tab applies, so one string from
   `prompts.TYPE_NAMES` reads the same on both tabs. */
function tmType(code, labels) {
  return scTypeName((labels || {})[code] || code) || code;
}

/* A grid row named. **`key` is the model AND the extraction shape on pass 2**,
   joined by a separator the server owns, so it is never printed -- `model` is
   the half a reader wants and `csModel` trims it as everywhere else. */
function tmName(e) {
  return csModel((e && e.model) || (e && e.key) || '');
}

/* The whole of what a row is, for a `title`. `csModel` trims a repack prefix
   that is the only thing separating two builds of one model, so the tooltip has
   to carry the untrimmed name -- and the shape with it, or two rows of one model
   have the same tooltip. */
function tmTitle(e) {
  const name = (e && e.model) || (e && e.key) || '';
  return e && e.shape ? name + ' · ' + e.shape : name;
}

/* The extraction shape beside a name, where the pass has one.

   **Single, Agentic and Hybrid are separate rows, not one averaged pair** -- the same
   rule `by_extract` follows, and the reason is that the two are two shapes with
   different failure modes rather than two samples of one setting. Pass 1 has no
   shape at all, so `shaped` is false there and this draws nothing; on pass 2 a
   blank shape is a run that died before pass 2 chose one, which is a real state
   and is named rather than left as an empty gap. */
function tmShape(e, shaped) {
  if (!shaped) return '';
  const s = e && e.shape;
  if (!s) return '<span class="tm-shape tm-shape-none" title="this run failed '
    + 'before pass 2 chose a shape, so the log records none. It counts as a '
    + 'failure and can carry no score">no shape</span>';
  return '<span class="tm-shape" title="the extraction shape: ' + esc(s)
    + '. Single asks for the whole form; Agentic uses original field groups; '
    + 'Hybrid uses tuned prompts and related field groups. Each mode is ranked '
    + 'separately">' + esc(s) + '</span>';
}

/* The best or worst KIND of page, as a sentence. Not `stSentence`: that one
   names a model and this names a type, and the two carry different denominators
   -- a type is described by how many documents and how many models fed it, which
   is the only thing that says whether its rank is evidence or an accident of
   what has been run. */
function tmSentence(e, tag, cls, unit, noun, labels, shaped) {
  if (!e) return '';
  const best = e.best
    ? ' &middot; best on it: <b>' + esc(tmName(e.best)) + '</b>'
      + tmShape(e.best, shaped) + ' ' + stFig(e.best, unit)
    : '';
  return '<p class="st-line"><span class="st-tag ' + cls + '">' + tag + '</span>'
    + '<b title="' + esc(e.key) + '">' + esc(tmType(e.key, labels)) + '</b> &mdash; '
    + stFig(e, unit)
    + (e.accuracy === null || e.accuracy === undefined ? '' : ' ' + noun)
    + ', ' + stFails(e) + ' failed, over ' + e.documents + ' document(s) and '
    + tmCovers(e, shaped) + best
    + (e.thin ? ' <span class="st-thin">(only ' + e.runs + ' run)</span>' : '')
    + '</p>';
}

/* How many things fed a type. **On pass 2 that is settings AND models**, and
   both are worth saying: seven models in fourteen settings is a different
   claim from fourteen models, and the figure above is meaned over the
   settings. Pass 1 has one of each, so it prints one. */
function tmCovers(e, shaped) {
  const settings = (e.settings === undefined ? e.models : e.settings);
  return shaped && settings !== e.models
    ? settings + ' setting(s) across ' + e.models + ' model(s)'
    : e.models + ' model(s)';
}

/* The types folded into one column, named with their own document counts.

   **A column is a requirement form, not a document type** -- the four types a
   requirement covers are four columns and every other type is one catch-all,
   because those all get the same 30-key base form and are all scored the same
   way. That grouping loses the ranking BETWEEN the folded types and must not
   also lose which they are, so the catch-all says so wherever it is named.

   Empty on a column holding one type: its own name is directly above it, and
   repeating it is the second-copy-of-the-page shape this card deletes on
   sight. */
function tmMembers(e, labels) {
  const m = (e && e.members) || [];
  if (m.length < 2) return '';
  // A trailing parenthetical is the label explaining itself -- "a billing note
  // (a covering slip listing invoices)" -- and in a list that is already
  // parenthesising a count it reads as two brackets fighting. The gloss is on
  // the Doc types tab, where there is room for it.
  return m.map(x => tmType(x.key, labels).replace(/\s*\([^)]*\)\s*$/, '')
                    + ' (' + x.documents + ')').join(', ');
}

/* One type per row: how it is read or extracted, and which model does it best.
   This is the half that answers *what should I run on a credit note* -- the grid
   below answers *what is this model good at*, which is the same data read the
   other way and rarely the same shape of answer. */
function tmTypeTable(id, entries, labels, unit, noun, shaped) {
  const cols = [
    {key: 'type', label: 'Document type', value: e => tmType(e.key, labels),
     cell: e => '<b title="' + esc(e.key) + '">' + esc(tmType(e.key, labels))
       + '</b>'
       // What the column holds, where it holds more than one type -- the
       // catch-all, and nothing else. Above `also`, because "this column is
       // eight kinds of page" has to be read before "and some of them are
       // also tax invoices".
       + (tmMembers(e, labels)
          ? '<span class="cs-set">holds ' + esc(tmMembers(e, labels)) + '</span>'
          : '')
       // The other types these pages carry. A document is filed under ONE type
       // -- the most specific it has -- so without this a receipt column reads
       // as a claim that those pages are not also tax invoices.
       + (e.also.length ? '<span class="cs-set">also '
           + e.also.map(c => esc(tmType(c, labels))).join(', ') + '</span>' : '')
       + '<span class="cs-set">' + e.runs + ' run(s) &middot; ' + e.documents
       + ' document(s) &middot; ' + tmCovers(e, shaped) + '</span>'},
    {key: 'score', label: 'Score', n: true, value: e => e.score,
     title: 'accuracy x (1 - failure rate) — what one attempt at this kind of '
          + 'page is worth, the same arithmetic Full rank uses',
     cell: e => e.score === null || e.score === undefined
       ? '<span class="cs-none">&mdash;</span>'
       : '<span class="cs-val">' + Number(e.score).toFixed(1) + '</span>'},
    {key: 'accuracy', label: noun === 'fields' ? 'Fields' : 'Transcript',
     n: true, value: e => e.accuracy,
     title: 'meaned over the SETTINGS that have been pointed at this type — '
          + 'each meaned over the documents of it first, so a type is not '
          + 'described by whichever setting was run at it most. On pass 2 a '
          + 'setting is a model AND a shape, so a model that is strong single '
          + 'and weak agentic counts as both rather than as their average',
     cell: e => stFig(e, unit)
       + (e.spread === null || e.spread === undefined ? ''
          : '<span class="cs-set">&plusmn;' + Number(e.spread).toFixed(1)
            + (shaped ? ' across settings' : ' across models') + '</span>')},
    {key: 'failed', label: 'Failed', n: true, value: e => e.failure_rate,
     title: 'runs on this type that did not finish, out of the runs this row '
          + 'covers. A failure is counted and never scored, which is why the '
          + 'accuracy beside it is over the runs that finished',
     cell: e => stFails(e)},
    {key: 'best', label: shaped ? 'Best setting' : 'Best model',
     value: e => e.best ? e.best.accuracy : null,
     title: 'the model — and on pass 2 the shape — this type is read or '
          + 'extracted best by. Picked the way Full rank picks one, so a single '
          + 'lucky run does not win it while a better-evidenced setting exists',
     cell: e => e.best
       ? '<b title="' + esc(tmTitle(e.best)) + '">' + esc(tmName(e.best)) + '</b>'
         + tmShape(e.best, shaped) + engineTag(e.best.ocr_engine)
         + '<span class="cs-set">' + stFig(e.best, unit) + ' &middot; '
         + e.best.runs + ' run(s)'
         + (e.best.thin ? ' &middot; thin' : '') + '</span>'
       : '<span class="cs-none">&mdash;</span>'},
    {key: 'worst', label: shaped ? 'Weakest setting' : 'Weakest model',
     value: e => e.worst ? e.worst.accuracy : null,
     title: 'the other end of the same ranking. Blank where only one setting '
          + 'has been run on this type — worst over one thing is a slur rather '
          + 'than a ranking',
     cell: e => e.worst
       ? '<b title="' + esc(tmTitle(e.worst)) + '">' + esc(tmName(e.worst))
         + '</b>' + tmShape(e.worst, shaped) + engineTag(e.worst.ocr_engine)
         + '<span class="cs-set">' + stFig(e.worst, unit) + ' &middot; '
         + e.worst.runs + ' run(s)</span>'
       : '<span class="cs-none">&mdash;</span>'}
  ];
  return lvTable(id, entries, cols, {sort: {key: 'score', dir: 'desc'}});
}

/* The model x type grid. Columns are in the server&rsquo;s ranked type order --
   best-performing kind of page first -- and that order is the same in both
   passes&rsquo; grids only by coincidence, so each is drawn from its own. */
function tmGrid(pass, labels, unit) {
  const order = pass.order || [];
  const rows = pass.models || [];
  const shaped = !!pass.shaped;
  if (!order.length || !rows.length) return '';
  const ring = {};
  (pass.types || []).forEach(e => { if (e.best) ring[e.key] = e.best.key; });
  const head = '<tr><th class="an-corner">'
    + (shaped ? 'Model &amp; shape' : 'Model') + ' \\ Document type</th>'
    + order.map(code => {
        const e = (pass.types || []).find(x => x.key === code);
        const held = tmMembers(e, labels);
        return '<th title="' + esc(tmType(code, labels)) + ' — ' + esc(code)
          + (held ? '. Holds ' + esc(held) : '')
          + '">' + esc(tmType(code, labels)) + '</th>';
      }).join('')
    + '<th title="the unweighted mean of this row&rsquo;s own cells — one per '
      + 'type, so a model measured on six types and one measured on two are '
      + 'compared on what they did rather than on which types happened to be '
      + 'busy. NOT the figure Best reading or Full rank prints for this model">'
      + 'Mean of its types</th></tr>';
  const body = rows.map(e => {
    const cells = order.map(code => {
      const c = e.cells[code];
      if (!c) return '<td class="an-empty" title="this model has not been run '
        + 'on a ' + esc(tmType(code, labels)) + '">&mdash;</td>';
      const tip = tmTitle(e) + ' on ' + tmType(code, labels) + ' — '
        + c.runs + ' reading(s) of ' + c.groups + ' document(s)'
        + (c.failed ? ', ' + c.failed + ' failed' : '')
        + (ring[code] === e.key
           ? ' — best ' + (shaped ? 'setting' : 'model') + ' for this type' : '');
      // A cell with no rate is not the same absence twice: every run failing is
      // a result, and nothing having a ground truth is not. The run count is
      // printed either way and the failures are added to it rather than
      // replacing it -- "1 failed" with no denominator is not a figure anyone
      // can read, which is the rule the whole card follows.
      const sub = (c.accuracy === null || c.accuracy === undefined
                   ? (c.failed ? 'none survived, ' : 'not scored, ') : '')
        + c.runs + ' run(s)' + (c.failed ? ' · ' + c.failed + ' failed' : '');
      const td = wkCell(c.accuracy, tip, sub);
      return ring[code] === e.key
        ? td.replace('<td class="', '<td class="tm-best ') : td;
    }).join('');
    return '<tr><td class="an-rowh" title="' + esc(tmTitle(e)) + '">'
      + esc(tmName(e)) + tmShape(e, shaped) + engineTag(e.ocr_engine)
      + '<span class="an-t">' + e.types + ' type(s) &middot; ' + e.runs
      + ' run(s)</span></td>'
      + cells
      + (e.mean === null || e.mean === undefined
         ? '<td class="an-mean-col an-empty">&mdash;</td>'
         : '<td class="an-mean-col"><b style="color:'
           + accColour(e.mean / 100) + '">' + e.mean.toFixed(0) + '%</b></td>')
      + '</tr>';
  }).join('');
  return '<div class="an-grid"><table class="an-heat">' + head + body
    + '</table></div>';
}

/* What a model is best and worst at, said out loud under the grid. A reader can
   find the highest of six numbers; what six numbers do not say is whether the
   spread between them is worth acting on. */
function tmStrengths(pass, labels) {
  const rows = (pass.models || []).filter(e => e.best_types.length);
  if (!rows.length) return '';
  const shaped = !!pass.shaped;
  return '<ul class="tm-list">' + rows.map(e =>
    '<li><b title="' + esc(tmTitle(e)) + '">' + esc(tmName(e)) + '</b>'
    + tmShape(e, shaped) + ' '
    + 'is strongest on ' + e.best_types.map(c => '<b>' + esc(tmType(c, labels))
        + '</b>').join(' and ')
    + (e.worst_types.length
       ? ', weakest on ' + e.worst_types.map(c => esc(tmType(c, labels))).join(' and ')
       : '')
    + '<span class="cs-set">' + e.types + ' type(s) &middot; ' + e.runs
    + ' run(s)' + (e.failed ? ' &middot; ' + e.failed + ' failed' : '') + '</span>'
    + '</li>').join('') + '</ul>';
}

/* How many runs this half of the tab could not place, and why. Three different
   absences, and only the first is a property of the document -- so they are
   counted apart rather than summed into one "excluded" number. */
function tmExcluded(pass) {
  const x = pass.excluded || {};
  const bits = [];
  if (x.pack) bits.push('<b>' + x.pack + '</b> run(s) on a file holding several '
    + 'documents, read before per-document scores were recorded. Its row has one '
    + 'transcript score over every page and one field score across all of them, '
    + 'and the log holds no transcript to take them apart again &mdash; so those '
    + 'runs wait until the file is read once more. <b>Every pack read since is '
    + 'split into its documents</b>, so this count empties itself');
  if (x.untyped) bits.push('<b>' + x.untyped + '</b> run(s) on a page nothing '
    + 'says the type of &mdash; an upload with no benchmark entry whose run '
    + 'never extracted, so the classifier&rsquo;s answer was never written down');
  if (x.unattributed) bits.push('<b>' + x.unattributed + '</b> run(s) with no '
    + 'model recorded, which is a run that failed before one was resolved');
  if (!bits.length) return '';
  return '<p class="note">Left out of the figures above: ' + bits.join('; ')
    + '. Counted here rather than dropped silently, and still in the log.</p>';
}

function tmPass(key, pass, labels, unit, noun, title, hint, t) {
  const shaped = !!(pass && pass.shaped);
  // The window is per grid CELL, and on pass 2 a cell is a type, a model and a
  // shape -- so the note has to name what it is actually per rather than
  // repeating pass 1's wording.
  const per = shaped ? 'setting and type' : 'model and type';
  // Taken once and reused: `emptyPassNote` opens with this same paragraph, so
  // computing it in both places printed it twice on a half that is BOTH narrowed
  // by the pipeline picker and empty -- the second-copy-of-the-page shape the
  // Warnings block was deleted for. It still belongs above a half that is
  // narrowed and NOT empty, which is why it is not simply moved inside.
  const piped = pipeNote(key);
  let html = '<span class="cs-title" title="' + esc(hint) + '">' + title
    + windowNote(per) + '</span>' + piped;
  if (!pass || !(pass.types || []).length) {
    /* **Three reasons this half can be empty and only one of them is about
       document types**, so the filter is named first: a reader who has just
       dropped a model or an outcome must not be told the corpus has no types.
       `emptyPassNote` covers the pipeline picker and the "every run here read no
       page" case card-wide; `lvInForce` names every narrowing; the type
       resolution is the fallback. */
    const shared = piped ? '' : emptyPassNote(key, t);
    const forced = lvInForce();
    html += shared
      || (piped ? ''
          : forced.length
          ? '<p class="note"><b>Nothing left to place against a document type.</b> '
            + 'In force: ' + forced.join(' &middot; ')
            + '. Clear a chip above, or press <b>Reset filters</b>.</p>'
          : '<p class="note">No run in view could be placed against a document '
            + 'type. The type comes from the benchmark manifest for a fixture and '
            + 'from the run&rsquo;s own classification otherwise, so an upload '
            + 'that never extracted and is in no manifest has none. Read a '
            + 'benchmark document and it will appear here.</p>')
      + tmExcluded(pass || {});
    // `tmExcluded` still prints: a pack backlog is a fact about the log, not
    // about the filter, and it is worth saying even where nothing else survived.
    return html;
  }
  html += tmSentence(pass.best, 'best', 'st-good', unit, noun, labels, shaped)
    + tmSentence(pass.worst, 'worst', 'st-bad', unit, noun, labels, shaped)
    + tmTypeTable('tm-types-' + key, pass.types, labels, unit, noun, shaped)
    + '<span class="cs-title">&hellip; and every '
    + (shaped ? 'model and shape' : 'model') + ' across the types'
    + windowNote(per) + '</span>'
    + tmGrid(pass, labels, unit)
    + tmStrengths(pass, labels)
    + tmExcluded(pass);
  return html;
}

// The Tokens tab: what each model costs in tokens, per pass. Two views of one
// compilation -- by model (each document counted once) and by document.
const tkView = { pass: 'ocr', by: 'models' };
function tkCell(d) {
  if (!d) return '<td class="n muted-inline">—</td>';
  return '<td class="n" title="min ' + d.min.toLocaleString() + ' · max '
    + d.max.toLocaleString() + ' · over ' + d.n + '">' + d.mean.toLocaleString()
    + ' <small class="muted-inline">' + d.min.toLocaleString() + '–'
    + d.max.toLocaleString() + '</small></td>';
}
function paintTokens(t) {
  const el = document.getElementById('tokenSum');
  const tu = t && t.token_usage;
  if (!el) return;
  if (!tu) { el.innerHTML = ''; return; }
  document.querySelectorAll('#panelTokens .tk-seg').forEach(seg =>
    seg.querySelectorAll('button').forEach(b =>
      b.classList.toggle('on', b.dataset.v === tkView[seg.dataset.tk])));
  const ocr = tkView.pass === 'ocr';
  const unit = ocr ? 'per page' : 'per file';
  let html = '';
  if (tkView.by === 'models') {
    const list = ocr ? tu.ocr_models : tu.extract_models;
    html += '<table class="queue"><tr><th>Model</th><th>' + (ocr ? 'Detail' : 'Shape')
      + '</th><th class="n">Docs</th><th class="n">Runs</th>'
      + '<th class="n" title="the mean over documents of each document\u2019s own mean; min\u2013max across documents">In, ' + unit + '</th>'
      + '<th class="n" title="the mean over documents of each document\u2019s own mean; min\u2013max across documents">Out, ' + unit + '</th></tr>'
      + list.map(m => '<tr><td>' + esc(csModel(m.model)) + '</td><td>' + esc(m.setting)
        + '</td><td class="n">' + m.documents + '</td><td class="n">' + m.runs + '</td>'
        + tkCell(m.in) + tkCell(m.out) + '</tr>').join('') + '</table>';
  } else {
    const list = ocr ? tu.ocr : tu.extract;
    html += '<table class="queue"><tr><th>Document</th><th>Model</th><th>'
      + (ocr ? 'Detail' : 'Shape') + '</th>' + (ocr ? '<th class="n">Pages</th>' : '')
      + '<th class="n">Runs</th><th class="n">In, whole document</th>'
      + '<th class="n">Out, whole document</th>'
      + (ocr ? '<th class="n">Out per page</th>' : '') + '</tr>'
      + list.map(e => '<tr><td>' + esc(e.case) + '</td><td>' + esc(csModel(e.model))
        + '</td><td>' + esc(e.setting) + '</td>' + (ocr ? '<td class="n">' + e.pages + '</td>' : '')
        + '<td class="n">' + e.runs + '</td>' + tkCell(e.in) + tkCell(e.out)
        + (ocr ? tkCell(e.out_per_page) : '') + '</tr>').join('') + '</table>';
  }
  html += '<p class="note">' + (ocr
    ? 'In is the prompt the server counted, page image included; it depends on the model and '
      + 'the Detail far more than on the document. Older rows only have it where the server timed '
      + 'the request, and an Ollama row whose prefix was cached shows only the computed part.'
    : 'In and out count every pass-2 request of the file &mdash; classify, segmentation, the '
      + 'steps or the single request, and the table agents. <b>In exists only for runs from '
      + '2026-10-15 on</b>; older rows have out only, and there it counts the extraction '
      + 'requests alone.')
    + ' Finished runs only, the last runs of each group. Mean, then min&ndash;max.</p>';
  el.innerHTML = html;
}
document.getElementById('panelTokens').addEventListener('click', ev => {
  const b = ev.target.closest('.tk-seg button');
  if (!b) return;
  tkView[b.closest('.tk-seg').dataset.tk] = b.dataset.v;
  paintTokens(logData && logData.totals);
});

function paintTypes(t) {
  const tm = t.type_models;
  if (!tm) { typeSum.innerHTML = ''; return; }
  const labels = tm.labels || {};
  // Read off the server's own test rather than listed here: a requirement
  // arriving for a fifth type splits it into its own column, and a hand-written
  // list in the page would go on naming four.
  const reqList = (tm.requirement_types || [])
    .map(c => '<b>' + esc(tmType(c, labels)) + '</b>').join(', ');
  typeSum.innerHTML =
      tmPass('ocr', tm.ocr, labels, '%', 'transcript',
             'Pass 1 &mdash; which kind of page is read well, and by what',
             'Every read grouped by the KIND of document rather than by the '
             + 'document or the setting', t)
    + tmPass('extract', tm.extract, labels, '%', 'fields',
             'Pass 2 &mdash; which kind of page is extracted well, and by what',
             'Every extraction grouped by the KIND of document it was mapping', t)
    + '<p class="note"><b>The type is the document&rsquo;s, not the '
    + 'run&rsquo;s.</b> It comes from the benchmark manifest &mdash; where a '
    + 'person states what each fixture is &mdash; and from the run&rsquo;s own '
    + 'classification only for a file the manifest does not describe. A run that '
    + 'read a page and never extracted records no type of its own, so going off '
    + 'the log column alone would throw nine reads in ten out of the pass-1 half '
    + 'of this tab. <b>A page is often more than one type at once</b> &mdash; '
    + 'four fixtures here are a receipt <i>and</i> a tax invoice &mdash; and each '
    + 'document is filed under the most specific one it carries, so the columns '
    + 'do not overlap and a document is counted once. The others are named under '
    + 'the type in the table. <b>The columns are the requirement forms, not the '
    + 'fourteen types</b>: a type a requirement covers &mdash; ' + reqList
    + ' &mdash; gets its own column, and every other type shares one, because '
    + 'those are all asked the same 30-key base form and all scored as an '
    + 'unknown type. That column names what it holds and how many documents of '
    + 'each, so the grouping loses the ranking between those types and nothing '
    + 'about which they are. <b>Single, Agentic and Hybrid are separate rows on the '
    + 'pass-2 half</b>, never one averaged pair &mdash; they are two settings '
    + 'with different failure modes rather than two samples of one, which is why '
    + '<b>Best extraction</b> keys on the shape too, and a model that is strong '
    + 'in one shape and weak in the other would otherwise be reported as neither. '
    + 'Pass 1 has no shape, so its rows are the model alone. <b>A file holding '
    + 'several documents contributes one '
    + 'measurement per document</b>, each with its own transcript and field score '
    + 'under its own type: the row&rsquo;s own figures are the average of those '
    + 'and describe no single kind of page, which is why the split is needed '
    + 'rather than convenient. <b>Score is accuracy &times; (1 &minus; failure rate)</b>, '
    + 'the same arithmetic as <b>Full rank</b>, and a failure is counted and '
    + 'never scored &mdash; so a model&rsquo;s place here is its place there '
    + 'restricted to one kind of page, and where the two disagree that difference '
    + 'is the finding. <b>This is not a second model ranking</b>: '
    + '<b>Full rank</b>, <b>Best reading</b> and <b>Best extraction</b> answer '
    + 'that over the same runs, and a model&rsquo;s mean here is over its types '
    + 'rather than over its runs.</p>';
}

/* ---------- Trend: accuracy and error rate over time ----------

   Built 2026-10-07 at the user's request. The server sends per-DAY sums
   (`[date, runs, failed, scored, acc_sum]`, see `runlog.trend`), so a week is
   the days added up and a mean is always acc_sum / scored over that bucket --
   folding here cannot drift from what the server counted.

   **One shared time axis across every card**, from the pass's first bucket to
   its last, so a model that only ran last week sits at the right-hand end
   rather than stretched across the whole width. Both lines are percentages on
   one 0-100 axis, so there is no second scale; the error line is dashed so the
   two are told apart without colour. A bucket where every run failed has an
   error point and no accuracy point: a failure is counted, never scored. */
const trendState = { pass: 'ocr', by: 'models', bucket: 'week', per: 10 };
let trendCards = {};

/* The run-number axis (2026-10-07, at the user's request: *0-10 run got 99% but
   at 11-20 run got 95%*). The server sends every run of a series in order, one
   value each -- -1 a failure, null finished but not scored, otherwise the score
   -- and this cuts them into consecutive groups of `per`. Each card has its OWN
   axis, run 1 to its last run: a model run 400 times and one run 20 times have
   nothing in common on a time scale and less on a run scale. A short last group
   is drawn and its tooltip says how many runs it holds. */
function trendChunks(seq, per) {
  const out = [];
  for (let i = 0; i < (seq || []).length; i += per) {
    const part = seq.slice(i, i + per);
    let failed = 0, scored = 0, sum = 0;
    part.forEach(v => {
      if (v === -1) failed += 1;
      else if (v !== null) { scored += 1; sum += v; }
    });
    const a = i + 1, b = i + part.length;
    out.push({ k: a === b ? 'run ' + a : 'runs ' + a + '–' + b, t: (a + b) / 2,
               runs: part.length, failed, scored, sum,
               acc: scored ? sum / scored : null, err: 100 * failed / part.length });
  }
  return out;
}

function trendKey(day, unit) {
  if (unit === 'day') return day;
  const d = new Date(day + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7);   // the Monday
  return d.toISOString().slice(0, 10);
}

function trendFold(days, unit) {
  const m = new Map();
  (days || []).forEach(([d, runs, failed, scored, sum]) => {
    const k = trendKey(d, unit);
    const c = m.get(k) || { k, runs: 0, failed: 0, scored: 0, sum: 0 };
    c.runs += runs; c.failed += failed; c.scored += scored; c.sum += sum;
    m.set(k, c);
  });
  return [...m.values()].sort((a, b) => (a.k < b.k ? -1 : 1)).map(c => ({
    ...c, t: Date.parse(c.k + 'T00:00:00Z'),
    acc: c.scored ? c.sum / c.scored : null,
    err: c.runs ? 100 * c.failed / c.runs : null,
  }));
}

function trendTotals(b) {
  let runs = 0, failed = 0, scored = 0, sum = 0;
  b.forEach(c => { runs += c.runs; failed += c.failed; scored += c.scored; sum += c.sum; });
  return { runs, failed, scored, acc: scored ? sum / scored : null,
           err: runs ? 100 * failed / runs : null };
}

function trendSvg(id, b, t0, t1, W, H, lo, hi) {
  const L = 30, R = 10, T = 8, B = 18, iw = W - L - R, ih = H - T - B;
  const x = t => L + (t1 > t0 ? (t - t0) / (t1 - t0) : 0.5) * iw;
  const y = v => T + (1 - v / 100) * ih;
  let g = '';
  [0, 50, 100].forEach(v => {
    g += '<line class="st-grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(v)
      + '" y2="' + y(v) + '"/><text class="st-axis" x="' + (L - 4) + '" y="' + (y(v) + 3)
      + '" text-anchor="end">' + v + '%</text>';
  });
  g += '<text class="st-axis" x="' + L + '" y="' + (H - 4) + '">' + esc(lo) + '</text>'
    + '<text class="st-axis" x="' + (W - R) + '" y="' + (H - 4) + '" text-anchor="end">'
    + esc(hi) + '</text>';
  const line = (key, cls) => {
    const pts = b.filter(c => c[key] !== null)
      .map(c => x(c.t).toFixed(1) + ',' + y(c[key]).toFixed(1));
    return pts.length > 1 ? '<polyline class="' + cls + '" points="' + pts.join(' ') + '"/>' : '';
  };
  const dots = (key, cls) => b.filter(c => c[key] !== null).map(c =>
    '<circle class="' + cls + '" r="3" cx="' + x(c.t).toFixed(1) + '" cy="'
    + y(c[key]).toFixed(1) + '"/>').join('');
  trendCards[id] = { b, x, W };
  return '<svg class="tr-svg" data-tr-card="' + id + '" viewBox="0 0 ' + W + ' ' + H + '">'
    + g + line('err', 'tr-err') + line('acc', 'tr-acc')
    + dots('err', 'tr-dot-err') + dots('acc', 'tr-dot-acc')
    + '<line class="st-cross" data-tr-cross x1="0" x2="0" y1="' + T + '" y2="' + (T + ih)
    + '" style="display:none"/></svg>';
}

function trendCard(id, name, full, b, t0, t1, all, lo, hi) {
  const s = trendTotals(b);
  const pct = v => (v === null ? '—' : v.toFixed(1) + '%');
  const sub = s.runs + ' run' + (s.runs === 1 ? '' : 's') + ' · accuracy ' + pct(s.acc)
    + ' · errors ' + s.failed + ' (' + pct(s.err) + ')';
  return '<div class="tr-card' + (all ? ' tr-all' : '') + '">'
    + '<div class="tr-name" title="' + esc(full || name) + '">' + esc(name) + '</div>'
    + '<div class="tr-sub">' + sub + '</div>'
    + trendSvg(id, b, t0, t1, all ? 900 : 300, all ? 150 : 128, lo, hi)
    + '<div class="st-tip tr-tip" hidden></div></div>';
}

function paintTrend(t) {
  const host = document.getElementById('trendSum');
  if (!host) return;
  document.querySelectorAll('#panelTrend .tr-seg').forEach(seg => {
    const want = trendState[seg.dataset.tr];
    seg.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === want));
  });
  document.getElementById('trendPerBox').hidden = trendState.bucket !== 'runs';
  trendCards = {};
  const tr = (t && t.trend) || {};
  const pass = tr[trendState.pass] || {};
  const unit = trendState.bucket;
  const byRun = unit === 'runs';
  const per = trendState.per;
  const seq = pass.seq || {};
  const all = byRun ? trendChunks(seq.all, per) : trendFold(pass.all, unit);
  if (!all.length) {
    host.innerHTML = emptyPassNote(trendState.pass, t)
      || '<p class="note">No runs in this view.</p>';
    return;
  }
  // Time: one axis shared by every card. Runs: each card its own, 1 to n.
  const span = (b, n) => byRun
    ? [1, Math.max(n, 2), 'run 1', 'run ' + n]
    : [all[0].t, all[all.length - 1].t, all[0].k.slice(5), all[all.length - 1].k.slice(5)];
  const group = pass[trendState.by] || {};
  const isModel = trendState.by === 'models';
  const entries = Object.keys(group).map(k => {
    const b = byRun ? trendChunks((seq[trendState.by] || {})[k], per) : trendFold(group[k], unit);
    return { k, b, runs: trendTotals(b).runs };
  });
  entries.sort((a, b) => isModel
    ? b.runs - a.runs || (a.k < b.k ? -1 : 1)
    : a.k.localeCompare(b.k, undefined, { numeric: true }));
  const passName = trendState.pass === 'ocr' ? 'reading' : 'extraction';
  const card = (id, name, full, b, big) => {
    const [t0, t1, lo, hi] = span(b, trendTotals(b).runs);
    return trendCard(id, name, full, b, t0, t1, big, lo, hi);
  };
  host.innerHTML = card('all', 'All ' + passName + ' runs', '', all, true)
    + '<div class="tr-grid">'
    + entries.map((e, i) => card('c' + i, isModel ? csModel(e.k) : e.k, e.k, e.b, false)).join('')
    + '</div>'
    + '<p class="note"><b>Every run the filters match, over its whole history</b> — '
    + 'not the recent-runs window the other tabs use, because a trend with its past '
    + 'cut off has nothing to show. Each point is '
    + (byRun ? per + ' consecutive runs of that card, oldest first, whenever they '
        + 'happened; each card has its own run axis, 1 to its last run, and a short '
        + 'last group is drawn with the runs it has'
        : unit === 'day' ? 'one day' : 'one week (labelled by its Monday)')
    + ': <b>accuracy</b> is the mean over the runs in it that finished and were scored ('
    + (trendState.pass === 'ocr' ? 'transcript character accuracy'
        : 'required-field score, a partial counting half, under the read floor')
    + '), and the <b>error rate</b> is the share that looped, crashed, was cut off or '
    + 'was cancelled — a failure is counted, never scored, so a bucket where every run '
    + 'failed has an error point and no accuracy point. A line joins the buckets that '
    + 'have runs; a stretch of time with none is simply crossed. On Day and Week all '
    + 'cards share one time axis. Hover a chart for the figures.</p>';
}

document.getElementById('panelTrend').addEventListener('click', ev => {
  const btn = ev.target.closest ? ev.target.closest('.tr-seg button[data-v]') : null;
  if (!btn) return;
  trendState[btn.closest('.tr-seg').dataset.tr] = btn.dataset.v;
  paintTrend(logData && logData.totals);
});

document.getElementById('trendPer').addEventListener('change', ev => {
  const n = Math.round(Number(ev.target.value));
  trendState.per = n >= 1 ? Math.min(n, 1000) : 10;
  ev.target.value = trendState.per;
  paintTrend(logData && logData.totals);
});

function trendHideTips(keep) {
  document.querySelectorAll('#panelTrend .tr-tip').forEach(tip => {
    if (!keep || tip.parentNode !== keep.parentNode) tip.hidden = true;
  });
  document.querySelectorAll('#panelTrend [data-tr-cross]').forEach(l => {
    if (!keep || l.ownerSVGElement !== keep) l.style.display = 'none';
  });
}

document.getElementById('panelTrend').addEventListener('mousemove', ev => {
  const svg = ev.target.closest ? ev.target.closest('svg[data-tr-card]') : null;
  trendHideTips(svg);
  if (!svg) return;
  const card = trendCards[svg.dataset.trCard];
  if (!card || !card.b.length) return;
  const box = svg.getBoundingClientRect();
  const vx = (ev.clientX - box.left) * card.W / box.width;
  let best = card.b[0];
  card.b.forEach(c => {
    if (Math.abs(card.x(c.t) - vx) < Math.abs(card.x(best.t) - vx)) best = c;
  });
  const cross = svg.querySelector('[data-tr-cross]');
  cross.setAttribute('x1', card.x(best.t));
  cross.setAttribute('x2', card.x(best.t));
  cross.style.display = '';
  const tip = svg.parentNode.querySelector('.tr-tip');
  const label = trendState.bucket === 'week' ? 'week of ' + best.k : best.k;
  tip.innerHTML = '<b>' + label + '</b><br>' + best.runs + ' run' + (best.runs === 1 ? '' : 's')
    + '<br>accuracy ' + (best.acc === null ? '— (none scored)'
        : best.acc.toFixed(1) + '% over ' + best.scored)
    + '<br>errors ' + best.failed + ' (' + best.err.toFixed(0) + '%)';
  tip.hidden = false;
  const cardBox = svg.parentNode.getBoundingClientRect();
  const px = box.left - cardBox.left + card.x(best.t) * box.width / card.W;
  const cw = svg.parentNode.clientWidth;
  tip.style.left = Math.max(4, Math.min(px + 10, cw - tip.offsetWidth - 4)) + 'px';
});
document.getElementById('panelTrend').addEventListener('mouseleave', () => trendHideTips(null));

function paintLog() {
  if (!logData) return;
  const t = logData.totals || {};
  paintBests(t);
  paintStandouts(t);
  paintErrors(t);
  paintTrend(t);
  paintCaseSummary(t);
  paintAnalysis(t);
  paintOcrSummary(t);
  paintExtractSummary(t);
  paintWeak(t);
  paintTypes(t);
  paintTokens(t);
  paintPresent(t);
  paintRunRows();
  paintFilters();
  paintPipe();
  paintEmpty(t);
  paintLogControls();
  paintRunStats(t);
}

function paintRunStats(t) {
  const rows = (logData && logData.runs) || [];
  runStats.title = t.window
    ? 'Every summary on this card covers the most recent runs of that setting, '
      + 'model or document — up to ' + t.window + ' each. Each row prints how '
      + 'many it actually covers.'
    : 'Every summary on this card covers every run in this view.';
  runStats.textContent = [
    // Counted over each document's own window, the same rule the tables use --
    // a count over "the log's last twenty rows" would be a count of whatever ran
    // last night. The file's true size is beside it.
    t.runs + ' run(s)',
    // Counted apart from reads: a re-extraction is a row, but it is not another
    // document read, and every other figure on this line is read against `runs`.
    t.extracts ? t.extracts + ' re-extract(s)' : null,
    // No number here either: this line is read against tables whose rows each
    // cover a different amount. The ceiling is on the tooltip.
    t.window ? 'recent runs per setting/document' : null,
    // Said out loud, because every figure on this line is over the filtered rows
    // and the raw table below is not.
    t.matched !== t.logged ? 'filtered to ' + t.matched + ' of ' + t.logged
                             + ' rows' : null,
    rows.length < (t.logged || 0)
      ? 'showing ' + rows.length + ' of ' + t.logged + ' rows' : null,
    t.scored ? t.scored + ' scored' : null,
    t.mean_accuracy !== null && t.mean_accuracy !== undefined
      ? 'mean ' + Number(t.mean_accuracy).toFixed(1) + '%' : null,
    // The ceiling as well as the average: the mean moves with whatever was being
    // tried lately, and on its own it hides that a document has scored higher.
    t.best_accuracy !== null && t.best_accuracy !== undefined
      ? 'best ' + Number(t.best_accuracy).toFixed(1) + '%'
        + (t.best_case ? ' (' + t.best_case + ')' : '') : null,
    t.seconds ? Math.round(t.seconds) + 's total' : null,
    t.tokens ? t.tokens + ' tokens' : null
  ].filter(Boolean).join(' · ');
  runPath.textContent = t.path ? 'Written to ' + t.path
    + ' — time, tokens, model, runtime, file name, file size, accuracy and the field '
    + 'score. No document text.' : '';
}

/* **Responses are applied in request order, not arrival order** (2026-08-25).
   Two refreshes can be in flight at once -- a chip clicked while Refresh is
   fetching, or two clicks in a row -- and the reply that lands LAST used
   to win whichever was asked first. The symptom is a card that is exactly one
   change behind: tick a box, watch the previous state paint. Found on the
   Summary tab's toggle, but it was never specific to it -- every control here
   goes through this one function.

   A counter rather than an AbortController: the request is cheap and read-only,
   and abandoning a reply is simpler than cancelling a request that may already
   have been served. */
let logSeq = 0;

async function refreshRuns() {
  const seq = ++logSeq;
  let data;
  try {
    data = await (await fetch(API + '/runs/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({limit: 50, window: logView.window,
                            min_read_pct: logView.min_read_pct,
                            include: logFilters.include,
                            exclude: logFilters.exclude,
                            drop_single_source: errDropSingle,
                            bias: presBiased})
    })).json();
  } catch { return; }
  // A newer refresh has been asked for since this one went out, so this answer
  // is about a view nobody is looking at any more. Dropped rather than painted.
  if (seq !== logSeq) return;
  logData = data;
  // A row that has been deleted, or has fallen off the end of the 50 shown, can
  // no longer be selected. Dropped rather than kept hopefully: a tick nobody can
  // see would delete a row nobody chose.
  const live = {};
  (data.runs || []).forEach(r => { live[r._row] = true; });
  Array.from(rawPicked).forEach(i => { if (!live[i]) rawPicked.delete(i); });
  paintLog();
}
runRefresh.addEventListener('click', refreshRuns);

/* ---------- no timer ----------
   **The card queries only when asked** (2026-10-27). It used to poll
   `/api/runs/stat` every five seconds and re-query whenever the CSV moved; now
   rows written from anywhere else -- a `curl`, a second tab, another process --
   show up when you press Refresh. Filter changes and this page's own runs still
   refresh it, since those are things you did. */

/* ---------- run mode ---------- */
/* ---------- theme ----------
   Three states: `auto` follows the desktop and is the default, `light` and
   `dark` stamp `data-theme` on the root and win over it. **`auto` REMOVES the
   attribute rather than setting it to "auto"** -- the CSS keys off the attribute
   being absent, so a stored "auto" written into the DOM would pin the page to
   the dark fallback on a light desktop, which is the exact bug the three-state
   design exists to avoid.

   Stored in `localStorage`, so a projector session survives a reload. A read
   that throws (private mode, a locked-down browser) falls back to `auto` rather
   than taking the page with it -- a theme is not worth an exception. */
const themeSeg = document.getElementById('themeSeg');

function applyTheme(name) {
  if (name === 'light' || name === 'dark')
    document.documentElement.setAttribute('data-theme', name);
  else
    document.documentElement.removeAttribute('data-theme');
  themeSeg.querySelectorAll('button').forEach(b =>
    b.classList.toggle('on', b.dataset.theme === (name || 'auto')));
}

function storedTheme() {
  try { return localStorage.getItem('ocrTheme') || 'auto'; } catch { return 'auto'; }
}

themeSeg.addEventListener('click', ev => {
  const btn = ev.target.closest ? ev.target.closest('button[data-theme]') : null;
  if (!btn) return;
  const name = btn.dataset.theme;
  try { localStorage.setItem('ocrTheme', name); } catch { /* not worth failing over */ }
  applyTheme(name);
});

applyTheme(storedTheme());

/* The run-mode and worker controls went with the queue itself: with nothing
   running batches in-process there is no pool to size. */

/* ---------- accuracy vs ground truth ---------- */
function accColour(v) {
  // red below 80%, amber to 95%, green above — matches how these read in practice.
  // **Hex, deliberately**: `hexA` parses this to build the analysis heatmap's
  // cell tint, so a `var(--ok)` here would be an unparsable colour rather than a
  // theme-aware one. `presColour` is the theme-aware twin, on a different scale
  // and with no heatmap behind it.
  if (v >= 0.95) return '#3fb950';
  if (v >= 0.80) return '#d9a441';
  return '#ff6b6b';
}
/* The same recall over the three scripts the page is written in, under the
   headline. Three rows rather than three numbers in a sentence because the whole
   point of them is the COMPARISON -- a model at 96% Thai and 71% numbers is a
   different problem from one evenly at 85%, and three bars side by side say that
   at a glance where three figures in prose do not.

   The digits row is FIRST, against the order everywhere else: every Mandatory
   field in the requirement is a figure, a date or an ID, so it is the one to
   read first and the one a reader should not have to look for.

   A script the page prints too little of to measure has no bar and says so with
   its count -- blank here means *not enough of it*, never *scored zero*. */
const ACC_SCRIPTS = [
  ['digit', 'Numbers', 'digits, in Arabic or Thai numerals — amounts, dates, '
   + 'tax IDs and document numbers. Every Mandatory field in the requirement is '
   + 'one of these, so a page can score well overall and still be unusable'],
  ['thai', 'Thai', 'Thai letters, including the tone marks and the upper and '
   + 'lower vowels — the first thing to go when a page is downscaled'],
  ['latin', 'English', 'the Latin alphabet. Not a claim that the text is '
   + 'English: a romanised Thai company name is Latin script too']
];
function accScriptRows(t) {
  const rows = ACC_SCRIPTS.map(spec => {
    const name = spec[0], value = t[name + '_accuracy'], total = t[name + '_chars'];
    if (total === undefined) return '';
    if (value === null || value === undefined)
      return '<div class="acc-row"><span class="label" title="' + esc(spec[2])
        + '">' + spec[1] + '</span><span class="acc-track"></span>'
        + '<span class="acc-val cs-none">' + total + ' chars</span></div>';
    return accRow(spec[1], value, spec[2],
                  Math.round(value * total) + ' of ' + total);
  }).join('');
  if (!rows) return '';
  return '<div class="acc-scripts">' + rows
    + '<p class="acc-note">The same measure over the three scripts, on the '
    + 'ground truth&rsquo;s own characters. <b>They do not add up to the figure '
    + 'above</b> — punctuation and symbols belong to no script and are in it '
    + 'only. A script printed fewer than ' + (scriptMinChars || 20) + ' times on '
    + 'the page shows its count and no rate: three characters score 0% or 100% '
    + 'and nothing in between, which is not a measurement of anything.</p></div>';
}
function accRow(label, value, hint, extra) {
  return '<div class="acc-row">'
    + '<span class="label" title="' + (hint || '') + '">' + label + '</span>'
    + '<span class="acc-track"><span class="acc-fill" style="width:'
    + (value * 100).toFixed(1) + '%;background:' + accColour(value) + '"></span></span>'
    + '<span class="acc-val">' + (value * 100).toFixed(1) + '%'
    + (extra ? '<span class="cs-set">' + esc(extra) + '</span>' : '')
    + '</span></div>';
}
/* This run against the best this document has ever scored, under any model,
   backend, detail or prompt. A percentage on its own says nothing about whether
   it is good: 93.1% is the ceiling on the handwritten scan and a poor result on
   the printed ones, and the only thing on the page that knows the difference is
   the run log. Read from `logBests`, which the run-log card fills on load. */
function bestNote(t) {
  const b = t && t.case && logBests[t.case];
  if (!b) return '';
  const best = Number(b.char_accuracy);
  const now = t.char_accuracy * 100;
  const how = [b.model, b.backend, b.detail].filter(Boolean).join(' · ');
  // A run can beat the record and still be reading a row written moments ago:
  // the log is refreshed on the `logged` event, so treat within-noise as equal.
  const verdict = now >= best - 0.05
    ? '<strong>the best this document has scored</strong>'
    : 'below its best of <strong>' + best.toFixed(1) + '%</strong>, set by '
      + esc(how) + (b.timestamp ? ' on ' + esc(b.timestamp.replace('T', ' ')) : '');
  return '<p class="acc-note">This run is ' + verdict
    + '. The figure is the highest in the run log for ' + esc(t.case)
    + ' across every run and every setting — not a mean.</p>';
}
/* Pass 2's score on the Accuracy tab, beside pass 1's. The headline only --
   which values were wrong, missed or invented is the Fields tab's job, and
   repeating the list here would make the tab that answers "how well was this
   document read" twice as long to answer a different question.

   Shown only for a document with a solution/<id>.fields.json behind it, and only
   once that file checks something: a bar reading 0.0% because nobody has filled
   the truth file in yet would be a score, and there is no score. */
function fieldAccNote(fs) {
  if (!fs || fs.error) return '';
  const pick = scoreTally(fs);
  if (!pick) return '';
  const o = pick.t;
  const c = o.counts || {};
  const cov = fs.coverage || {};
  const items = fs.line_items;
  const op = fs.optional || {};
  return accRow('Field extraction', o.accuracy,
                'Extracted values against the value the field ground truth says '
                + 'belongs in that key -- a different question from whether the '
                + 'text was read correctly'
                + (pick.scope ? '\nScored over ' + pick.scope : ''))
    + '<p class="acc-note">' + c.correct + ' of ' + o.expected
    // What the bar is OVER, on the bar's own line. The Fields tab says it too,
    // and this tab is read without that one: a rate whose scope is only stated
    // elsewhere is a rate that will be read as covering everything.
    + (pick.required ? ' required' : ' stated') + ' value(s) correct'
    + (c.partial ? ', ' + c.partial + ' partly' : '')
    // The bar's own denominator first, then what the bar does not count -- in
    // that order and with that wording, because the three used to be one list
    // and read as one denominator. An invented value cannot lower a rate whose
    // denominator is what the page prints; it lowers precision.
    + ' — ' + c.wrong + ' wrong, ' + c.missed + ' not returned'
    + (c.spurious ? '. Outside the bar: ' + c.spurious + ' filled where the page '
        + 'states nothing, which lowers precision (' + scPct(o.precision)
        + ') rather than this rate' : '')
    + (pick.required && op.expected
        ? '. ' + (op.counts || {}).correct + ' of ' + op.expected
          + ' optional value(s) correct, also outside this bar' : '')
    + (pick.unknown
        ? '. <b>No requirement covers this document type yet</b>, so this bar is '
          + 'over the base field set: nothing here is Mandatory, and it is what '
          + 'the extractor found rather than what it owed'
        : '')
    + (items ? '; ' + items.rows_matched + '/' + items.rows_expected
               + ' line-item row(s) matched, from the table in '
               + esc(((items.derived || {}).source) || 'the ground truth') : '')
    + '. Scored over the ' + cov.scalars_checked + ' of ' + cov.scalars_total
    + ' key(s) <code>' + esc(fs.case || '') + '.fields.json</code> fills in'
    + (cov.line_items_checked ? ', plus the table' : '')
    + '. The <b>Fields</b> tab lists each one.</p>';
}
/* What the page cap dropped from this read, or ''. One function, three callers,
   so the meta line, the Accuracy tab and a re-render cannot word it differently.
   `pages_total` is ABSENT rather than equal where nothing knows -- a fields-only
   run read no page -- so a missing value is never read as "nothing was dropped". */
function truncatedNote(res) {
  const read = Number((res || {}).page_count || 0);
  const total = Number((res || {}).pages_total || 0);
  if (!read || !total || total <= read) return '';
  return 'Only ' + read + ' of ' + total + ' pages were read — MAX_PAGES cut the '
       + 'file short. Any score below is against the whole document, so it is not '
       + 'a measurement of this read.';
}

/* **The cap truncates, it does not refuse**, and this is the one place that can
   be said BEFORE an hour of OCR is paid for. A named function rather than inline
   in the fetch handler so the banner has one wording and can be exercised on its
   own -- the same reason `truncatedNote` is one function with three callers. */
function paintPrepTruncated(read, total, cap) {
  const cut = total > read;
  prepTrunc.innerHTML = cut
    ? '<b>Only ' + read + ' of ' + total + ' pages will be read.</b> '
      + 'The rest are dropped, and a score would be against the whole document. '
      + 'Raise <code>MAX_PAGES</code>' + (cap ? ' (now ' + cap + ')' : '')
      + ' and restart to read all of it.'
    : '';
  prepTrunc.hidden = !cut;
}

/* The run's own summary, kept because the Accuracy tab is painted long after
   the `done` event -- on a tab switch, and again after a re-extraction. */
let lastSummary = null;

function paintTruncated(res) {
  lastSummary = res;
  const note = truncatedNote(res);
  metaTrunc.textContent = note;
  metaTrunc.hidden = !note;
}

function paintAccuracy() {
  if (!truth) { accHead.innerHTML = ''; accBars.innerHTML = ''; accDiff.textContent = ''; return; }
  if (truth.error) {
    accHead.innerHTML = '<span class="case">Scoring unavailable</span>';
    accBars.innerHTML = '<p class="acc-note">' + esc(truth.error) + '</p>';
    accDiff.textContent = '';
    return;
  }
  accHead.innerHTML = '<span class="case">' + esc(truth.case) + '</span> '
    + '<span class="kind">· ' + esc(truth.pdf) + (truth.kind ? ' · ' + esc(truth.kind) : '') + '</span>';

  // **Above the bar, not under it.** A truncated read is scored against the whole
  // document, so the number below belongs to pages nobody read -- and a caveat
  // printed after the figure it disqualifies is one that arrives too late. Same
  // reasoning as the `partial` banner above the fields.
  const cut = truncatedNote(lastSummary);
  accHead.innerHTML += cut ? '<p class="note warn">' + esc(cut) + '</p>' : '';

  // ONE number, not three. `word_accuracy` and `char_accuracy_no_marks` are
  // still computed and still logged -- they are useful diagnostics when reading
  // the CSV -- but on screen they were three bars saying almost the same thing,
  // and the two extra ones invited being read as scores in their own right when
  // neither is the score. Character accuracy is the headline everywhere else in
  // this project; the page now says only that.
  accBars.innerHTML =
      accRow('Character accuracy', truth.char_accuracy,
             'How much of the ground truth the transcript actually produced. '
             + 'Content characters only, in whatever order the model put them — '
             + 'whitespace and zero-width marks come off both sides first, table '
             + 'markup is normalised, and blocks are matched to the ground truth '
             + 'by content rather than by position. Text the transcript has and '
             + 'the page does not costs nothing')
    + '<p class="acc-note">Expected ' + truth.expected_chars + ' content chars · got '
    + truth.actual_chars
    + (truth.matched_chars === undefined ? '.'
        // Both halves of the score on one line: what it charges for, and what
        // it deliberately does not. The invented count is ALWAYS printed, even
        // when it is two characters -- a number that appears only past a
        // threshold is one a reader cannot trust the absence of.
        : ' · <b>' + (truth.expected_chars - truth.matched_chars)
          + '</b> missed (dropped or misread) · <b>' + (truth.invented_chars || 0)
          + '</b> invented (not charged).')
    + ' Scored on <b>content, not layout</b>: spacing, line breaks, HTML-vs-pipe'
    + ' tables and the order the model walked the page in cannot lower it, and'
    + ' neither can <b>extra</b> text — a value the read dropped cannot then be'
    + ' extracted, while text the page does not print leaves every real value'
    + ' where it was. What lowers it is content that is missing or misread.</p>'
    + accScriptRows(truth)
    // The count is on the line above whatever it is; this paragraph is the
    // explanation, and it appears only where the number is material. Two stray
    // characters do not need three sentences of warning, and a warning printed
    // under every read is one nobody reads by the third document.
    + (truth.invented_chars > Math.max(20, truth.expected_chars * 0.02)
        ? '<p class="acc-note' + (truth.invented_chars > truth.expected_chars * 0.1
                                  ? ' err' : '') + '"><b>'
          + (truth.invented_chars / Math.max(truth.expected_chars, 1) * 100).toFixed(0)
          + '% of the page length is text the page does not print.</b> That '
          + 'costs the score nothing by design — extra text leaves every real '
          + 'value where it was — and it is the half worth looking at anyway: a '
          + 'figure caption or a stamp described in words is harmless, an '
          + 'invented value is not, and pass 2 will ground an invented value '
          + 'happily because it is in the transcript. The <b>differences</b> '
          + 'below show it as <span class="add">+</span> lines.</p>'
        : '')
    + bestNote(truth)
    + (unscoredReason()
        // No bar, because there is no figure -- a bar drawn at whatever the
        // suppressed score happened to be is the exact number this rule exists
        // to keep off the board.
        ? '<p class="acc-note"><b>Fields not scored</b> — ' + esc(unscoredReason())
          + '. The extraction ran and is in the run log with its coverage and '
          + 'grounding; the correctness figure is not, because it would be a '
          + 'measurement of this read rather than of the extractor. The '
          + '<b>Fields</b> tab still marks each value.</p>'
        : fieldAccNote(extracted && extracted.field_score));

  if (!truth.diff || !truth.diff.length) {
    // "the same content", not "byte for byte the same page": `scoring.diff_lines`
    // folds out any group whose two sides differ only in where the line breaks
    // and spaces fell, which is the same rule `char_accuracy` scores under.
    accDiff.innerHTML = '<span class="add">Every character of content matches.</span>';
    return;
  }
  accDiff.innerHTML = truth.diff.map(line => {
    const cls = line.startsWith('+') && !line.startsWith('+++') ? 'add'
      : line.startsWith('-') && !line.startsWith('---') ? 'del'
      : line.startsWith('@@') ? 'hunk' : '';
    return cls ? '<span class="' + cls + '">' + esc(line) + '</span>' : esc(line);
  }).join('\n');
}

/* ---------- compare view ----------
   Three columns exist, two are shown: the picker chooses which pair. The pair
   that is off keeps its rendered content, so flipping between OCR and ground
   truth against the same source page is instant and re-fetches nothing.

   Ground truth is `solution/<id>.md` rendered through the same renderer as a
   transcript. That is deliberate: the two are only comparable by eye if a pipe
   table in the fixture and an HTML table from the model come out as the same
   shape of table -- which is exactly what `scoring.normalise` does before it
   scores them. */
const CMP_VIEWS = {
  'source-model': [cmpColImg, cmpColModel],
  'source-truth': [cmpColImg, cmpColTruth],
  'model-truth':  [cmpColModel, cmpColTruth],
  // Only for a missing-required stress flag: the form it came back with.
  'model-fields': [cmpColModel, cmpColFields],
  'source-fields': [cmpColImg, cmpColFields],
};
const CMP_FIELD_VIEWS = ['model-fields', 'source-fields'];
function hasRun() { return !!(job && pageTexts.length); }
function cmpInStress() { return !!(cmpOverride && stressPaneOpen); }
/* What the per-page columns step through: the workspace run's pages, or, for a
   flagged stress page, each read of that one page (a looped read, then its retry). */
function cmpTotal() { return cmpInStress() ? cmpOverride.reads.length : pageTexts.length; }
function cmpText(i) { return cmpInStress() ? cmpOverride.reads[i].text : pageTexts[i]; }
function cmpSrc(i) { return cmpInStress() ? cmpOverride.reads[i].src : pageSrc(i); }
/* Which case's ground truth to show. What the finished run was scored against
   wins; failing that, whatever the workspace is pointing at -- so a fixture can
   be read before anything has been run against it. A stress page's is its own
   case's, and none for a realData file. */
function truthCase() {
  if (cmpInStress()) return cmpOverride.truthCase || null;
  if (truth && truth.case) return truth.case;
  return matchedCase || null;
}
/* What one read of a flagged stress page came back as, for its caption. */
function stReadOutcome(r) {
  const parts = [r.detail || ''];
  // A read that failed or never came back (an error or stalled flag).
  if (r.note) parts.push(r.note);
  else if (r.looped) parts.push('looped — cut off by the loop guard');
  else if (r.truncated) parts.push('hit the token cap');
  else parts.push('read cleanly');
  if (r.dry) parts.push('DRY fired from token ' + r.dry.first + ', longest repeat ' + r.dry.longest);
  return parts.filter(Boolean).join(' · ');
}
/* The source image and the OCR text both come out of a run; ground truth comes
   out of the fixture and is there before one. So a pair is offered only when
   both of its halves exist -- and where no pair does, the one column that does
   exist is shown on its own rather than beside an empty box. */
function colAvailable(col) {
  if (col === cmpColTruth) return !!truthCase();
  if (col === cmpColFields) return cmpInStress() && !!cmpOverride.fields;
  // An extract-only job read no page, so its flag has text and no image.
  if (col === cmpColImg && cmpInStress()) return cmpOverride.reads.some(r => r.src)
    || !!window.StressLocalPreview?.hasCandidate(cmpOverride.filename);
  return cmpInStress() ? cmpTotal() > 0 : hasRun();
}
const ST_WRONG_TEXT = { wrong: 'wrong', partial: 'partly right',
  missed: 'not returned', spurious: 'invented' };
/* Each document's form: every key asked, its value, Required where the
   requirement demands it, and the flagged ones marked. */
function cmpFieldsHtml(docs) {
  const many = docs.length > 1;
  return docs.map(d => {
    const head = many || d.error ? '<h4>Document ' + d.document
      + (d.pages && d.pages.length ? ' · pages ' + d.pages.join(', ') : '')
      + (d.doc_types && d.doc_types.length ? ' · ' + esc(d.doc_types.join(', ')) : '')
      + (d.error ? ' · <span style="color:var(--danger)">' + esc(d.error) + '</span>' : '')
      + '</h4>' : '';
    if (!d.rows.length) return head + '<p class="note">No fields came back.</p>';
    // A value that does not match the ground truth (`wrong`) shows what the
    // truth says under it, and who lost it where that is known.
    const lost = { ocr: 'read lost it', extract: 'extraction lost it', unknown: 'unclear' };
    return head + '<table><tr><th>Field</th><th>Value</th><th></th></tr>'
      + d.rows.map(r => '<tr' + (r.missing || r.wrong ? ' class="miss"' : '') + '><td>'
        + esc(r.key) + '</td><td class="v">'
        + (r.value ? esc(r.value) : r.missing ? 'missing' : '—')
        + (r.wrong ? '<br><small>ground truth: '
          + (r.expected ? esc(r.expected) : 'nothing printed') + '</small>' : '')
        + '</td><td>' + (r.required ? '<span class="st-tag '
          + (r.missing ? 'st-bad' : 'st-info') + '">required</span>' : '')
        + (r.wrong ? ' <span class="st-tag st-bad">'
          + esc(ST_WRONG_TEXT[r.wrong] || r.wrong) + '</span>'
          + (lost[r.blame] ? ' <small>' + lost[r.blame] + '</small>' : '') : '')
        + '</td></tr>').join('') + '</table>'
      + ((d.validation_rechecks || []).map(retry => '<details><summary>Validation image recheck · '
          + (retry.accepted ? 'accepted' : 'original values kept') + '</summary><pre>'
          + esc(JSON.stringify(retry, (key, value) => key === 'chats' ? undefined : value, 2))
          + '</pre></details>').join(''));
  }).join('');
}
function viewFull(view) { return CMP_VIEWS[view].every(colAvailable); }

/* Ground truth, fetched once per case. A failure is cached too, as an error --
   a missing fixture is not going to appear on the next repaint, and retrying on
   every one of them would hammer the endpoint from a scroll. */
async function loadTruth(id) {
  if (truthCache[id] === undefined) {
    truthCache[id] = (async () => {
      try {
        const res = await fetch(API + '/truth/' + encodeURIComponent(id));
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Ground truth unavailable.');
        return data;
      } catch (err) {
        return { error: err.message };
      }
    })();
  }
  return truthCache[id];
}
function paintTruthPane() {
  const id = truthCase();
  if (!id) {
    cmpTruthCap.textContent = '';
    cmpTruthRender.innerHTML = '<em>No ground truth for this document.</em>';
    cmpTruthText.textContent = '';
    return;
  }
  cmpTruthCap.textContent = '· ' + id;
  loadTruth(id).then(data => {
    // The pane may have moved on to another case while this was in flight.
    if (truthCase() !== id) return;
    const raw = cmpRawChk.checked;
    cmpTruthRender.style.display = raw ? 'none' : 'block';
    cmpTruthText.style.display = raw ? 'block' : 'none';
    if (data.error) {
      cmpTruthCap.textContent = '· ' + id;
      cmpTruthRender.innerHTML = '<em>' + esc(data.error) + '</em>';
      cmpTruthText.textContent = data.error;
      return;
    }
    cmpTruthCap.textContent = '· ' + data.file
      + (data.pages > 1 ? ' · whole document, ' + data.pages + ' pages' : '');
    if (raw) cmpTruthText.textContent = data.text;
    else cmpTruthRender.innerHTML = sanitize(renderOutput(data.text));
  });
}
/* The view actually painted. A chosen view whose half is missing falls back for
   as long as it is missing -- but `cmpView` itself is left alone, so starting a
   new run (no pages yet, ground truth still known) does not quietly move the
   reader to a different pair and leave them there once the pages arrive. */
function effectiveView() {
  if (viewFull(cmpView)) return cmpView;
  return Object.keys(CMP_VIEWS).find(viewFull) || null;
}
function paintCompare() {
  stPaintFlagLinks();
  const view = effectiveView();
  const cols = view ? CMP_VIEWS[view]
                    : [cmpColImg, cmpColModel, cmpColTruth].filter(colAvailable);
  [cmpColImg, cmpColModel, cmpColTruth, cmpColFields].forEach(c => { c.hidden = !cols.includes(c); });
  cmpGrid.classList.toggle('one', cols.length < 2);
  const withFields = colAvailable(cmpColFields);
  [...cmpModes.querySelectorAll('button')].forEach(b => {
    b.classList.toggle('on', b.dataset.view === view);
    b.disabled = !viewFull(b.dataset.view);
    // The Fields pairs exist only for a missing-required flag's window.
    if (CMP_FIELD_VIEWS.includes(b.dataset.view)) b.hidden = !withFields;
  });
  if (cols.includes(cmpColFields)) {
    const docs = cmpOverride.fields;
    const gaps = docs.reduce((n, d) => n + d.rows.filter(r => r.missing).length, 0);
    const bad = docs.reduce((n, d) => n + d.rows.filter(r => r.wrong).length, 0);
    cmpFieldsCap.textContent = '· ' + gaps + ' required missing'
      + (bad ? ' · ' + bad + ' wrong against the ground truth' : '');
    cmpFieldsPane.innerHTML = cmpFieldsHtml(docs);
  }

  const total = cmpTotal();
  // The page controls belong to the two columns that are per-page; ground truth
  // is one file for the whole document and says so in its own caption.
  const paged = cols.some(c => c !== cmpColTruth) && total > 0;
  if (paged && cmpInStress()) {
    // One page of a stress run: the controls step through its reads instead.
    const o = cmpOverride;
    o.at = Math.max(0, Math.min(o.at, total - 1));
    const r = o.reads[o.at];
    // A loop or DRY flag steps through the reads of one page; a missing-required
    // flag through the pages of its document, each as the read that was kept.
    // A stalled flag's reads are the pages still out, one each.
    const unit = o.fields || o.kind === 'stalled' ? 'Page' : 'Read';
    const page = r.page || o.page;
    [cmpPrev, cmpNext, cmpLabel].forEach(el => { el.hidden = false; });
    cmpLabel.textContent = total > 1 ? `${unit} ${o.at + 1} of ${total}` : unit + ' 1';
    cmpPrev.disabled = o.at === 0;
    cmpNext.disabled = o.at >= total - 1;
    window.StressLocalPreview?.show(o, r, !cmpColImg.hidden);
    cmpModelCap.textContent = o.loading ? '· waiting for comparison results'
      : o.loadError ? '· comparison unavailable'
      : r.page === null && o.fields ? '· the transcript the extraction was fed'
      : '· ' + stReadOutcome(r);
    const raw = cmpRawChk.checked;
    cmpRender.style.display = raw ? 'none' : 'block';
    cmpRawText.style.display = raw ? 'block' : 'none';
    const pendingText = o.loading ? 'Loading OCR and extraction results from server…' : o.loadError;
    if (raw) cmpRawText.textContent = pendingText || r.text || '';
    else cmpRender.innerHTML = pendingText ? '<em>' + esc(pendingText) + '</em>'
      : r.text ? sanitize(renderOutput(r.text)) : '<em>No text.</em>';
    if (cols.includes(cmpColTruth)) paintTruthPane();
    return;
  }
  window.StressLocalPreview?.hide();
  [cmpPrev, cmpNext, cmpLabel].forEach(el => { el.hidden = !paged; });
  // "Open image" belongs to the image column, not to paging: it is meaningless
  // in the view that shows the transcript beside the fixture.
  cmpOpen.hidden = cmpColImg.hidden || !paged;
  // The page in this pane is the PREPARED one -- trimmed and scaled to the Detail
  // that ran -- so this saves the picture the model was actually given, which is
  // the same thing the workspace bar offers before a read.
  cmpDl.hidden = cmpOpen.hidden;

  if (paged) {
    cmpPage = Math.max(0, Math.min(cmpPage, total - 1));
    cmpLabel.textContent = total > 1 ? `Page ${cmpPage + 1} of ${total}` : 'Page 1';
    cmpPrev.disabled = cmpPage === 0;
    cmpNext.disabled = cmpPage >= total - 1;

    const src = pageSrc(cmpPage);
    if (cmpImg.getAttribute('src') !== src) cmpImg.src = src;
    cmpOpen.href = src;
    cmpDl.href = src;
    cmpDl.download = (matchedCase || 'page') + '-p' + (cmpPage + 1) + '.png';
    cmpImgCap.textContent = total > 1 ? '· page ' + (cmpPage + 1) : '';
    cmpModelCap.textContent = total > 1 ? '· page ' + (cmpPage + 1) : '';

    const body = pageTexts[cmpPage] || '';
    const raw = cmpRawChk.checked;
    cmpRender.style.display = raw ? 'none' : 'block';
    cmpRawText.style.display = raw ? 'block' : 'none';
    if (raw) cmpRawText.textContent = body;
    else cmpRender.innerHTML = body ? sanitize(renderOutput(body)) : '<em>No text.</em>';
  }

  if (cols.includes(cmpColTruth)) paintTruthPane();
}
function enableCompare() {
  if (!cmpInStress()) window.StressLocalPreview?.hide();
  const ready = cmpInStress() || hasRun() || !!truthCase();
  cmpToggle.disabled = !ready;
  cmpHint.textContent = !ready ? 'Run a document first.'
    : cmpInStress() && cmpOverride.loading ? 'Opening PC original · loading comparison results…'
    : cmpInStress() && cmpOverride.loadError ? cmpOverride.loadError
    : cmpInStress() && cmpOverride.fields ? 'Stress test · ' + cmpOverride.case
      + ' — the page and the fields its extraction came back with, required ones that are empty in red.'
    : cmpInStress() ? 'Stress test · ' + cmpOverride.case + ' page ' + cmpOverride.page
      + ' — every read of it, the page it was sent and the text it returned.'
    : hasRun() ? 'Source page, transcript and ground truth, two at a time.'
    : 'Ground truth only — run the document to compare it against a transcript.';
  if (!ready) { compareEl.hidden = true; cmpToggle.textContent = 'Show'; }
  else if (!compareEl.hidden) paintCompare();
}
cmpModes.addEventListener('click', e => {
  const btn = e.target.closest('button[data-view]');
  if (!btn || btn.disabled) return;
  cmpView = btn.dataset.view;
  paintCompare();
});
cmpPrev.addEventListener('click', () => {
  if (cmpInStress()) cmpOverride.at--; else cmpPage--;
  paintCompare();
});
cmpNext.addEventListener('click', () => {
  if (cmpInStress()) cmpOverride.at++; else cmpPage++;
  paintCompare();
});
cmpRawChk.addEventListener('change', paintCompare);

// Proportional scroll linking: the panes have different natural heights, so
// match by fraction of scrollable distance rather than by pixel offset.
//
// Echoes are suppressed by remembering the exact offset we wrote and ignoring
// the scroll event that reports it back. A time- or frame-based guard would
// also swallow a real user scroll that lands in the same frame as a
// programmatic one, which silently breaks syncing in one direction.
const echo = new WeakMap();
function applySync(from, to) {
  if (!cmpSync.checked) return 'off';
  const expected = echo.get(from);
  if (expected !== undefined) {
    // Consume unconditionally: a stale entry must never block a later scroll.
    echo.delete(from);
    if (Math.abs(from.scrollTop - expected) < 2) return 'echo';
  }
  const fromMax = from.scrollHeight - from.clientHeight;
  const toMax = to.scrollHeight - to.clientHeight;
  if (fromMax <= 0 || toMax <= 0) return 'nothing-to-scroll';
  const target = (from.scrollTop / fromMax) * toMax;
  echo.set(to, target);
  to.scrollTop = target;
  return 'synced';
}
// The pair being synced follows the view mode, so the link is resolved per
// scroll event rather than wired once: a listener bound to a column that is
// currently off would sync a pane nobody can see.
const CMP_PANES = [cmpImgPane, cmpTextPane, cmpTruthPane];
function otherVisiblePane(from) {
  return CMP_PANES.find(p => p !== from && !p.parentElement.hidden);
}
CMP_PANES.forEach(pane => {
  pane.addEventListener('scroll', () => {
    if (pane.parentElement.hidden) return;
    const to = otherVisiblePane(pane);
    if (to) applySync(pane, to);
  });
});

/* ---------- model server: status and switching ----------
   Two server kinds are supported and they are not interchangeable in one
   respect: Ollama holds many models and needs one named per request, while
   llama-server holds exactly one and ignores the name. So the model picker is
   shown only when the active endpoint actually offers a choice. */
const endpointSel = document.getElementById('endpointSel');
const modelSel = document.getElementById('modelSel');
const modelField = document.getElementById('modelField');
// The same two pickers again, in the Fields-only pane. Comparing two models on
// one transcript is most of what that pane is for, so it cannot send you to
// another tab to choose one -- and there is a single setting behind both pairs.
const fxEndpoint = document.getElementById('fxEndpoint');
const fxModel = document.getElementById('fxModel');
const fxModelField = document.getElementById('fxModelField');
const fxExtract = document.getElementById('fxExtract');
const fxExtractField = document.getElementById('fxExtractField');
const customRow = document.getElementById('customRow');
const endpointCustom = document.getElementById('endpointCustom');
const endpointUse = document.getElementById('endpointUse');
const serverNote = document.getElementById('serverNote');
const subKind = document.getElementById('subKind');
const subUrl = document.getElementById('subUrl');
const serverFieldLabel = document.getElementById('serverFieldLabel');
const ctxFieldLabel = document.getElementById('ctxFieldLabel');

/* The MODEL SERVER group's heading and labels. With a separate extraction
   server the two passes are on two boxes, and the heading must not claim this
   one does both. */
function paintServerGroup() {
  const split = !!(extractInfo && extractInfo.separate);
  serverGroupHead.textContent = split ? 'Model servers · reads, then extracts elsewhere'
                                      : 'Model server · reads and extracts';
  serverGroupNote.textContent = split
    ? 'The model server reads each prepared page image; the extraction server '
      + 'below maps that transcript onto the form.'
    : 'This server reads each prepared page image and then extracts the fields '
      + 'from its own transcript.';
  modelRole.textContent = 'Model · reads the page';
  serverFieldLabel.textContent = 'Model server';
  ctxFieldLabel.textContent = 'Context';
  subKind.textContent = server.kind || 'no server';
  subUrl.textContent = server.url || '';
}

function paintGlobalModels() {
  for (const [role, info, ready] of [
    ['Ocr', server, !!server.available],
    ['Extract', extractInfo || server, !!(extractInfo ? extractInfo.available : server.text_available)]
  ]) {
    const model = document.getElementById('global' + role + 'Model');
    const url = document.getElementById('global' + role + 'Url');
    const health = document.getElementById('global' + role + 'Health');
    model.textContent = info.model || 'No model';
    model.title = info.model || '';
    url.textContent = info.url || '';
    url.title = info.url || '';
    health.textContent = ready ? 'Ready' : 'Unavailable';
    health.dataset.ready = String(ready);
    health.title = ready ? 'Ready' : (info.reason || info.text_reason || 'Server unavailable');
  }
}

const globalModelSettings = document.getElementById('globalModelSettings');
document.addEventListener('pointerdown', event => {
  if (globalModelSettings.open && !globalModelSettings.contains(event.target)) globalModelSettings.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && globalModelSettings.open) {
    globalModelSettings.open = false;
    globalModelSettings.querySelector('summary').focus();
  }
});

// Add the active URL to a list if it was typed in rather than configured.
function addUrlOption(sel, url) {
  if (!sel || !url || [...sel.options].some(o => o.value === url)) return;
  const last = sel.options[sel.options.length - 1];
  // Ahead of "Other address…" where there is one, so a typed-in URL does not sit
  // under the entry that offers to type one.
  if (last && last.value === '__custom__') sel.insertBefore(new Option(url, url), last);
  else sel.add(new Option(url, url));
}
/* The extraction model, offered as "same as reading model" plus every model the
   endpoint serves that is NOT an OCR fine-tune.

   The one combination left out is reading with one OCR model and extracting with
   a different one: it pays for a second set of weights to get a second model
   that is bad at pass 2. Extracting with the READING model is always offered,
   because that is the one-model setup every measurement in this project was
   taken under -- it is the default, not a fallback.

   The server decides which models qualify (`backends.is_ocr_model`) and refuses
   the combination outright; this list only makes the refusal visible before it
   is triggered rather than after. */
function paintExtractPicker(sel, field, ex) {
  if (!sel || !field) return;
  const choices = (ex && ex.choices) || [];
  /* On a server of its own there is no reading model to share, so the empty
     choice means that server's default instead -- named, so it is visible
     which model "default" is. */
  const separate = !!(ex && ex.separate);
  field.style.display = choices.length > 1 || (ex && !ex.same_as_reading && !separate)
    ? 'flex' : 'none';
  if (field.style.display === 'none') return;
  sel.innerHTML = (separate
      ? '<option value="">server default'
        + (ex.model && !ex.chosen ? ' (' + esc(ex.model) + ')' : '') + '</option>'
      : '<option value="">same as reading model'
        + (server.model ? ' (' + esc(server.model) + ')' : '') + '</option>')
    + choices.map(name => '<option value="' + esc(name) + '"'
        + (name === (ex.chosen || '') ? ' selected' : '') + '>' + esc(name)
        + (name === server.model && !separate ? ' · also reading' : '')
        + '</option>').join('');
  sel.value = (ex && ex.chosen) || '';
}

/* The extraction SERVER: "same as reading server" (the one-URL setup), every
   configured endpoint, and "Other address…". Painted from the reading picker's
   own list so the two cannot offer different servers, plus whatever pass 2 is
   pointed at now. The value is the CHOICE (`configured`), never the resolved
   URL: "the reading server" and "this URL, which happens to be the reading
   server" are different choices, and only the first follows the reader. */
function paintExtractEndpoint(sel) {
  if (!sel) return;
  const ex = extractInfo || {};
  const urls = [...endpointSel.options].map(o => o.value)
    .filter(v => v && v !== '__custom__');
  if (ex.configured && !urls.includes(ex.configured)) urls.push(ex.configured);
  const want = ex.configured || '';
  sel.innerHTML = '<option value="">same as reading server</option>'
    + urls.map(u => '<option value="' + esc(u) + '">' + esc(u) + '</option>').join('')
    + (sel === extractEndpointSel ? '<option value="__custom__">Other address…</option>' : '');
  sel.value = want;
}
function paintExtractServerNote() {
  const ex = extractInfo || {};
  let text = '', bad = false;
  if (!ex.separate) {
    text = ex.configured
      ? 'The extraction server is the reading server right now, so both passes go to '
        + (ex.url || '') + '.'
      : '';
  } else {
    bad = !ex.available;
    text = ex.available
      ? 'Fields are extracted on ' + (ex.kind || 'server') + ' ' + ex.url
        + (ex.model ? ' · ' + ex.model : '') + ' · page images still go to '
        + (server.url || 'the reading server') + '.'
      : '⚠ Extraction server ' + ex.url + ': ' + (ex.reason || 'not reachable.');
  }
  // One sentence, two panes: the random test's copy says the same thing about
  // the same setting, so it cannot say anything different.
  for (const el of [extractServerNote, rtExtractServerNote]) {
    if (!el) continue;
    el.textContent = text;
    el.classList.toggle('err', bad);
  }
}

function paintModelPicker(sel, field, models) {
  if (!sel) return;
  // One model is not a choice, it is a fact — shown in the status line instead.
  field.style.display = models.length > 1 ? 'flex' : 'none';
  if (models.length < 2) return;
  sel.innerHTML = models.map(m =>
    '<option value="' + esc(m.name) + '"' + (m.name === server.model ? ' selected' : '') + '>'
    + esc(m.name) + (m.vision === false ? ' · no vision' : '')
    + (m.size_gb ? ' · ' + m.size_gb + ' GB' : '') + '</option>').join('');
  sel.value = server.model || '';
}
/* Both pairs of pickers are painted from the status that came back, never kept
   in step by each other's change handlers: a switch that is refused -- the 409
   while a random test is running is the one that bites -- has to leave every picker
   on the page showing what is really active, not what was asked for. */
function paintEndpoint() {
  paintServerGroup();
  paintGlobalModels();
  for (const sel of [endpointSel, fxEndpoint]) {
    if (!sel) continue;
    addUrlOption(sel, server.url);
    if (sel.value !== '__custom__') sel.value = server.url || '';
  }
  const models = server.models || [];
  paintModelPicker(modelSel, modelField, models);
  paintModelPicker(fxModel, fxModelField, models);
  paintExtractPicker(extractModelSel, extractModelField, extractInfo);
  paintExtractPicker(fxExtract, fxExtractField, extractInfo);
  paintExtractEndpoint(extractEndpointSel);
  paintExtractEndpoint(fxExtractEndpoint);
  paintExtractEndpoint(rtExtractEndpoint);
  paintServerKind(serverKindSel, server);
  /* The extraction server has a type of its own only when it IS its own
     server; in the one-URL setup the picker above already covers it. */
  if (extractKindField) {
    extractKindField.style.display = extractInfo && extractInfo.separate ? 'flex' : 'none';
    if (extractInfo && extractInfo.separate) paintServerKind(extractKindSel, extractInfo);
  }
  paintExtractServerNote();
  // The random test's locks are painted from the same two lists, so a model that
  // stops being served stops being lockable at the same moment.
  if (typeof rtPaintLocks === 'function') rtPaintLocks();
  // The /metrics button follows the server kind: only vLLM is ever asked.
  if (typeof stPaintServer === 'function') stPaintServer();
}

/* Auto names what it detected, so a reader can see what Auto currently means
   and whether pinning would change anything. Painted from the status that came
   back, like every other picker here -- a refused pin leaves it showing the
   setting really in force. */
function kindLabel(kind) {
  const hit = SERVER_KINDS.find(k => k.kind === kind);
  return hit ? hit.label : (kind || 'nothing');
}
function paintServerKind(sel, info) {
  if (!sel) return;
  const setting = (info && info.kind_setting) || 'auto';
  const detected = info && info.kind;
  sel.innerHTML = '<option value="auto">Auto'
    + (setting === 'auto' ? (detected ? ' · found ' + esc(kindLabel(detected))
                                       : ' · nothing found') : '')
    + '</option>'
    + SERVER_KINDS.map(k => '<option value="' + esc(k.kind) + '">'
        + esc(k.label) + '</option>').join('');
  sel.value = setting;
}
async function setServerKind(url, kind) {
  setServerNote('Setting server type…');
  try {
    const res = await fetch(API + '/servers/kind', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url, kind: kind })
    });
    const data = await res.json();
    if (!res.ok) {
      setServerNote(data.error || 'Could not set the server type.', true);
      paintEndpoint();
      return;
    }
    server = data.server;
    if (data.extract) extractInfo = data.extract;
    const info = url === server.url ? server : extractInfo;
    setServerNote(kind === 'auto'
      ? url + ' is detected automatically' + (info.kind ? ' · found ' + kindLabel(info.kind) : ' · nothing answered')
      : url + ' is pinned to ' + kindLabel(kind)
        + (info.kind ? '' : ' · it did not answer as one')
        + (info.kind_warning ? ' · ⚠ ' + info.kind_warning : ''),
      !info.kind || !!info.kind_warning);
    paintEndpoint();
    paintServer();
  } catch (err) {
    setServerNote(err.message, true);
    paintEndpoint();
  }
}
if (serverKindSel) serverKindSel.addEventListener('change',
  () => setServerKind(server.url, serverKindSel.value));
if (extractKindSel) extractKindSel.addEventListener('change',
  () => setServerKind(extractInfo.url, extractKindSel.value));

/* One switch, two panes: the note goes to both, because the pane you are not
   looking at is the one that would otherwise hold a stale "Now using …". */
function setServerNote(msg, bad) {
  for (const el of [serverNote, fxServerNote]) {
    if (!el) continue;
    el.textContent = msg;
    el.classList.toggle('err', !!bad);
  }
}
/* What pass 2 will run on, as the server last reported it. Held beside `server`
   rather than inside it because it is a separate choice with its own refusal,
   and a picker painted from a merged blob cannot show which half was rejected.
   Seeded at render, so the pickers are right on first paint. */

/* `extractModel` is a tri-state and the distinction matters: undefined means
   "leave it alone", '' means "same as the reading model", and a name means that
   model. Sent as a JSON null-vs-string, which the route reads the same way. */
async function switchServer(url, model, extractModel, extractUrl) {
  setServerNote('Switching…');
  runBtn.disabled = true;
  try {
    const body = { url: url || '', model: model || '' };
    if (extractModel !== undefined) body.extract_model = extractModel;
    // Same tri-state: undefined leaves pass 2's server alone, '' puts it back on
    // the reading server, a URL gives it one of its own.
    if (extractUrl !== undefined) body.extract_url = extractUrl;
    const res = await fetch(API + '/servers', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) {
      setServerNote(data.error || 'Could not switch server.', true);
      // Put the dropdowns back on whatever is really active.
      paintEndpoint();
      paintServer();
      return;
    }
    server = data.server;
    if (data.extract) extractInfo = data.extract;
    // The pass-1 profile follows the reading model, so the picker has to be
    // repainted from what came back rather than left on what was selected
    // before the switch -- the same rule every other picker here follows.
    if (data.ocr_profile) {
      ocrProfileSel.value = data.ocr_profile;
      paintOcrProfile();
    }
    // The models Ollama was holding are stopped on a switch, so the new one
    // loads onto a free card. Said out loud: an unload that happened and one
    // that did not look the same on this line otherwise, and they leave very
    // different amounts of VRAM behind.
    const stopped = data.unloaded || [];
    setServerNote('Now using ' + server.url
      + (server.model ? ' · ' + server.model : '')
      // Named only when the two passes differ, so the usual one-model line stays
      // as short as it was.
      + (extractInfo.separate
         ? ' · extracting on ' + extractInfo.url
           + (extractInfo.model ? ' with ' + extractInfo.model : '')
         : extractInfo.same_as_reading ? ''
         : ' · extracting with ' + extractInfo.model)
      + (stopped.length ? ' · stopped ' + stopped.join(', ') + ' to free the GPU' : ''));
    customRow.style.display = 'none';
    if (extractCustomRow) extractCustomRow.style.display = 'none';
    paintEndpoint();
    paintServer();
  } catch (err) {
    setServerNote(err.message, true);
    paintServer();
  }
}

endpointSel.addEventListener('change', () => {
  if (endpointSel.value === '__custom__') {
    customRow.style.display = 'flex';
    endpointCustom.focus();
    return;
  }
  customRow.style.display = 'none';
  switchServer(endpointSel.value);
});
endpointUse.addEventListener('click', () => {
  const url = endpointCustom.value.trim();
  if (url) switchServer(url);
});
endpointCustom.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); endpointUse.click(); }
});
modelSel.addEventListener('change', () => switchServer(null, modelSel.value));
if (fxEndpoint) fxEndpoint.addEventListener('change', () => switchServer(fxEndpoint.value));
if (fxModel) fxModel.addEventListener('change', () => switchServer(null, fxModel.value));
for (const sel of [extractModelSel, fxExtract]) {
  if (sel) sel.addEventListener('change', () => switchServer(null, null, sel.value));
}
for (const sel of [extractEndpointSel, fxExtractEndpoint, rtExtractEndpoint]) {
  if (!sel) continue;
  sel.addEventListener('change', () => {
    if (sel.value === '__custom__') {
      extractCustomRow.style.display = 'flex';
      extractEndpointCustom.focus();
      return;
    }
    if (extractCustomRow) extractCustomRow.style.display = 'none';
    switchServer(null, null, undefined, sel.value);
  });
}
if (extractEndpointUse) extractEndpointUse.addEventListener('click', () => {
  const url = extractEndpointCustom.value.trim();
  if (url) switchServer(null, null, undefined, url);
});
if (extractEndpointCustom) extractEndpointCustom.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); extractEndpointUse.click(); }
});

function paintServer() {
  const kind = server.kind || 'server';
  const ok = !!server.available;
  dot.className = 'dot ' + (ok ? 'ok' : 'bad');
  serverText.textContent = ok
    ? (server.model || kind) + ' · ' + kind
      + (server.vision ? ' · vision ready'
                       : (server.vision_known ? ' · no vision'
                                              : ' · vision not reported'))
    : (server.model ? server.model + ' · unusable'
                    : 'No model server at ' + server.url);
  serverReason.textContent = ok ? '' : ('⚠ ' + (server.reason || ''));
  paintGlobalModels();
  runBtn.disabled = !ok || (!current && !workspaceCase);
  // Whether the context picker does anything depends on which kind is active.
  paintCtx();
  paintFieldsOnly();
  paintOcrOnly();
}
/* Workspace can read a named benchmark case, a realData file or a local upload.
   All three use the same full pipeline and prepared-image preview. */

async function refreshServer() {
  recheckBtn.disabled = true;
  document.getElementById('globalModelNotice').hidden = true;
  recheckBtn.textContent = 'Refreshing…';
  document.querySelector('.model-toolbar').setAttribute('aria-busy', 'true');
  const was = serverText.textContent;
  serverText.textContent = 'Checking…';
  try {
    // probe=1 bypasses the status cache and re-reads the model list too, which
    // is what "Re-check" is for after starting or stopping a server.
    const res = await fetch(API + '/servers?probe=1', {cache: 'no-store'});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not re-check the servers.');
    server = data.server;
    if (data.extract) extractInfo = data.extract;
    if (data.ocr_profile) {
      ocrProfileSel.value = data.ocr_profile;
      paintOcrProfile();
    }
    // The last switch's "Now using ..." note may name a model that was just
    // replaced on this port. The freshly painted status is authoritative.
    setServerNote('');
    paintEndpoint();
  } catch (err) {
    setServerNote(err.message || 'Could not re-check the servers.', true);
    const notice = document.getElementById('globalModelNotice');
    notice.textContent = err.message || 'Could not refresh the servers.';
    notice.hidden = false;
    serverText.textContent = was;
  } finally {
    recheckBtn.disabled = false;
    recheckBtn.textContent = 'Refresh';
    document.querySelector('.model-toolbar').removeAttribute('aria-busy');
    paintServer();
  }
}
recheckBtn.addEventListener('click', refreshServer);

/* ---------- live run counters ---------- */
function startTicker() {
  runStart = Date.now();
  liveTokens = 0;
  clearInterval(ticker);
  ticker = setInterval(() => {
    const s = ((Date.now() - runStart) / 1000).toFixed(1);
    const rate = liveTokens > 0 ? ' · ' + (liveTokens / ((Date.now() - runStart) / 1000)).toFixed(2) + ' tok/s' : '';
    meta.textContent = `${s}s · ${liveTokens} tokens${rate}`;
  }, 200);
}
function stopTicker() { clearInterval(ticker); ticker = null; }

/* The model server's own figures (servertime.py), as it gave them: prefill and
   decode time with their tokens and tokens per second, queue time, and its
   end-to-end latency. The rates are tokens over the SERVER's seconds -- prefill
   counts the tokens actually computed (image tokens in, prefix-cache hits out),
   decode the tokens generated in the decode phase -- so they are the GPU's raw
   speed. A dash where the server gave no figure: Ollama's /v1 sends none,
   llama.cpp has no queue figure, a vLLM /metrics change spanning concurrent
   requests cannot be attributed, and a prefill rate needs the cached count. */
const SERVER_COLS = [
  ['server_prefill_seconds', 'Server prefill', 'secs',
   'Prompt processing on the server: the vision encoder and the prompt'],
  ['server_prefill_tps', 'Prefill tok/s', 'rate',
   'Tokens computed in prefill (image tokens included, prefix-cache hits excluded) per second of server prefill'],
  ['server_decode_seconds', 'Server decode', 'secs', 'Token generation on the server'],
  ['server_decode_tps', 'Decode tok/s', 'rate',
   'Tokens generated in the decode phase per second of server decode'],
  ['server_queue_seconds', 'Server queue', 'secs',
   'Time the request waited in the server queue before it was scheduled'],
  ['server_seconds', 'Server latency', 'secs',
   'The server end-to-end latency (on vLLM: arrival to last token, queue and image decoding included; on llama.cpp: prefill + decode)'],
];
function fmtTok(n) { return Number(n).toLocaleString('en-US'); }
function fmtBytes(n) {
  if (n === null || n === undefined) return '—';
  return n >= 1048576 ? (n / 1048576).toFixed(2) + ' MB' : (n / 1024).toFixed(0) + ' KB';
}
function serverTimeText(fig, src) {
  if (!fig || fig.server_seconds === null || fig.server_seconds === undefined) return '';
  const has = k => fig[k] !== null && fig[k] !== undefined;
  const parts = [];
  if (has('server_prefill_seconds')) {
    const tok = [];
    if (has('server_prefill_tokens')) tok.push(fmtTok(fig.server_prefill_tokens) + ' tok');
    if (has('server_cached_tokens') && fig.server_cached_tokens)
      tok.push(fmtTok(fig.server_cached_tokens) + ' cached');
    if (has('server_prefill_tps')) tok.push(fmtTok(fig.server_prefill_tps) + ' tok/s');
    parts.push(`prefill ${Number(fig.server_prefill_seconds).toFixed(2)}s`
      + (tok.length ? ` (${tok.join(', ')})` : ''));
  }
  if (has('server_decode_seconds')) {
    const tok = [];
    if (has('server_decode_tokens')) tok.push(fmtTok(fig.server_decode_tokens) + ' tok');
    if (has('server_decode_tps')) tok.push(fmtTok(fig.server_decode_tps) + ' tok/s');
    parts.push(`decode ${Number(fig.server_decode_seconds).toFixed(2)}s`
      + (tok.length ? ` (${tok.join(', ')})` : ''));
  }
  if (has('server_queue_seconds')) parts.push(`queue ${Number(fig.server_queue_seconds).toFixed(2)}s`);
  parts.push(`latency ${Number(fig.server_seconds).toFixed(2)}s`);
  return ' · server ' + parts.join(', ') + (src ? ` (${src})` : '')
    + (has('image_bytes') ? ` · image sent ${fmtBytes(fig.image_bytes)}` : '');
}
function serverCell(v, kind) {
  if (v === null || v === undefined) return '—';
  return kind === 'rate' ? fmtTok(v) : Number(v).toFixed(2) + 's';
}

function renderStats(summary) {
  if (!pageStats.length) { statsEl.innerHTML = ''; return; }
  const timed = pageStats.some(s => s.server_seconds !== null && s.server_seconds !== undefined);
  const sent = pageStats.some(s => s.image_bytes);
  const head = '<thead><tr><th>Page</th><th>Resolution</th><th>Tokens</th>'
    + '<th>Prefill</th><th>Decode</th><th>tok/s</th>'
    + (sent ? '<th title="The page as sent: a base64 PNG in the JSON body. Uploading it and the server decoding it are in no server phase">Image sent</th>' : '')
    + (timed ? SERVER_COLS.map(([, label, , hint]) =>
        `<th title="${hint}">${label}</th>`).join('') : '')
    + '</tr></thead>';
  const extra = row => (sent ? `<td>${fmtBytes(row.image_bytes)}</td>` : '')
    + (timed ? SERVER_COLS.map(([key, , kind]) =>
        `<td>${serverCell(row[key], kind)}</td>`).join('') : '');
  const body = pageStats.map((s, i) =>
    `<tr><td>${i + 1}</td><td>${s.resolution || '—'}</td><td>${s.new_tokens ?? 0}</td>`
    + `<td>${(s.prefill_seconds ?? 0).toFixed(1)}s</td>`
    + `<td>${(s.decode_seconds ?? 0).toFixed(1)}s</td>`
    + `<td>${(s.tokens_per_second ?? 0).toFixed(2)}</td>`
    + extra(s)
    + '</tr>').join('');
  const f = summary || {};
  const foot = pageStats.length > 1 || summary
    ? `<tfoot><tr><td>Total</td><td>${f.model || server.model || ''}</td>`
      + `<td>${f.tokens ?? ''}</td><td>${(f.prefill_seconds ?? 0).toFixed(1)}s</td>`
      + `<td>${(f.decode_seconds ?? 0).toFixed(1)}s</td>`
      + `<td>${(f.tokens_per_second ?? 0).toFixed(2)}</td>`
      + extra(f)
      + '</tr></tfoot>' : '';
  statsEl.innerHTML = head + '<tbody>' + body + '</tbody>' + foot;
}

/* ---------- detail hint ---------- */
const DETAIL_NOTE = {
  original: 'No downscaling — the page exactly as rasterised. A 300 DPI A4 page '
          + 'is ~8.7 MP and can take 10+ minutes, and past ~4 MP the extra visual '
          + 'tokens measurably make this model worse, not better.',
  medium: 'Caps pages at ~4.0 MP. The measured best for dense or small print, and '
        + 'the default.',
  low: 'Caps pages at ~2.0 MP. Roughly half the prefill; small tone marks are the '
     + 'first thing lost.'
};
function updateNote() { detailNote.textContent = DETAIL_NOTE[detailSel.value] || ''; }
detailSel.addEventListener('change', updateNote);
// The preset IS the picture: moving this picker changes what the model is given,
// so the preview follows it rather than describing the last one chosen.
detailSel.addEventListener('change', refreshPrepared);
updateNote();

/* ---------- context window ----------
   Sent with every request, to whichever server is active: Ollama reads it as
   options.num_ctx, llama-server as n_ctx. Changeable at any time, including
   mid-sweep -- a read already in flight keeps the window it started on. */
const ctxSel = document.getElementById('ctxSel');
const ctxNote = document.getElementById('ctxNote');
function paintCtx() {
  if (server.num_ctx) ctxSel.value = String(server.num_ctx);
  ctxNote.textContent = server.kind === 'ollama'
    ? 'Sent as options.num_ctx on every request. Bigger holds a longer transcript '
      + 'for passes 2 and 3, but costs KV-cache memory whether or not it is used.'
    : 'Sent as n_ctx on every request. A build that does not accept it keeps the '
      + 'window from its own -c instead.';
}
ctxSel.addEventListener('change', async () => {
  const want = ctxSel.value;
  try {
    const res = await fetch(API + '/context', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_ctx: Number(want) })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not set the context size.');
    server = data.server;
    meta.textContent = `Context window set to ${data.num_ctx} tokens.`;
    meta.classList.remove('err');
  } catch (err) {
    // Put the picker back on what the server is really using.
    if (server.num_ctx) ctxSel.value = String(server.num_ctx);
    meta.textContent = err.message;
    meta.classList.add('err');
  } finally {
    paintCtx();
  }
});

/* ---------- file intake ---------- */
function showChip(name) {
  preview.style.display = 'none';
  chipName.textContent = name;
  chip.style.display = 'block';
}
/* ---------- auto-map an upload to its ground-truth case ---------- */
async function sha256Hex(file) {
  if (!crypto.subtle) return '';           // needs a secure context; name-match still works
  const buf = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
}
async function mapFile(file) {
  mapChip.className = '';
  mapChip.textContent = 'Looking for a matching solution…';
  try {
    const sha256 = await sha256Hex(file);
    const res = await fetch(API + '/match', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, sha256 })
    });
    const m = await res.json();
    // A stale reply from a previously selected file must not overwrite this one.
    if (current !== file) return;
    if (m.case) {
      matchedCase = m.case;
      mapChip.className = '';
      mapChip.innerHTML = '✓ Ground truth <strong>' + esc(m.case) + '</strong> — will be scored'
        + ' <span class="how">(matched by ' + esc(m.matched_by) + ')</span>';
    } else {
      matchedCase = null;
      mapChip.className = 'none';
      mapChip.textContent = 'No matching solution file — this run will not be scored.';
    }
  } catch {
    if (current !== file) return;
    matchedCase = null;
    mapChip.className = 'none';
    mapChip.textContent = '';
  }
  enableCompare();   // a matched upload can show its ground truth before it runs
}

/* A single file gets the preview and the ground-truth lookup, exactly as before.
   A batch cannot be streamed into one result panel, so it goes to the queue
   instead -- listed in full first, then worked by however many workers there
   are. That is what makes a concurrency measurement mean anything. */
// Each preview holds a blob URL alive until it is released; the workspace can be
// repainted many times before a run, so the previous one goes first.
let previewUrl = null;
function dropPreview() {
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  preview.style.display = 'none'; preview.removeAttribute('src');
}
/* The workspace preview with nothing prepared: the uploaded file as it is, or
   the chip naming whatever cannot be drawn as an image (a PDF, a HEIC, a
   fixture the server holds). Split out of `previewOne` because the prepared
   preview has to be able to put it BACK -- switching Detail to `original` after
   looking at `low` would otherwise leave a scaled-down picture on screen with
   nothing left on the page to say so. */
function localPreview() {
  paintPrepTruncated(1, 1, 0);      // a banner from the last file is about that file
  dropPreview();
  if (!current) {
    if (workspaceCase) showChip(workspaceCase + ' · test file');
    else chip.style.display = 'none';
    return;
  }
  if (current.type === 'application/pdf' || /\.(pdf|heic|heif)$/i.test(current.name)) {
    showChip(current.name);
    return;
  }
  preview.onerror = () => showChip(current.name);
  preview.onload = () => { chip.style.display = 'none'; preview.style.display = 'block'; };
  previewUrl = URL.createObjectURL(current);
  preview.src = previewUrl;
}
function previewOne(file) {
  mapFile(file);
  chip.style.display = 'none';
  localPreview();
  meta.textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(2) + ' MB';
  meta.classList.remove('err');
  // The local blob above is instant feedback on the file. What the model gets is
  // a different picture at every Detail but `original`, and that one comes back
  // from the server.
  refreshPrepared();
}

/* ---------- the page as the model will actually see it ----------
   A capped Detail trims the margins and scales the page down before anything is
   sent, so at `medium` and `low` the file on disk is NOT what gets read -- and
   the difference is the whole reason this project deleted its 1 MP preset, where
   the model stopped misreading and started inventing. The preview is therefore
   fetched back from `/api/preview`, prepared by the same two functions the read
   calls, and offered as a download.

   It is never resized in the browser. A canvas downscale here would be plausible,
   free and WRONG -- a picture no run ever saw, differing from the real one by
   whatever Pillow does that canvas does not.

   `original` is uncapped, so there is nothing to show: the local preview stands
   and the bar stays hidden. (The margin trim still applies there; the bar is
   about the pixel budget, and a trim of 0-18% of blank border is not a thing
   anyone needs to inspect before paying for a read.) */
let prepSeq = 0;                 // a stale reply must never overwrite a newer one
let prepCtl = null;
function prepBody() {
  // The same three shapes `request_input` resolves, chosen the same way the Read
  // button chooses them, so the preview cannot be of a different document.
  const body = new FormData();
  if (current) body.append('image', current);
  else if (workspaceCase) body.append('case', workspaceCase);
  else return null;
  body.append('detail', detailSel.value);
  return body;
}
function prepStem() {
  const raw = (current && current.name) || workspaceCase || 'page';
  // Only the extension comes off. The browser sanitises a `download` name
  // itself, and these documents have Thai names worth keeping.
  return raw.replace(/\.[^.]+$/, '');
}
function hidePrepared() {
  prepSeq++;                                   // orphan whatever is in flight
  if (prepCtl) { prepCtl.abort(); prepCtl = null; }
  prepBar.classList.remove('on');
  // Emptied, not just hidden: a caption left behind describes a Detail nobody is
  // on any more, and it is one CSS class away from being read.
  prepCap.textContent = '';
  prepCap.classList.remove('err');
  prepDl.hidden = true;
  prepDl.removeAttribute('href');
}
async function refreshPrepared() {
  hidePrepared();
  const body = prepBody();
  if (!body) return;
  // Uncapped: the file itself is what gets sent, so there is nothing to prepare
  // and the plain preview stands.
  if (detailSel.value === 'original') { localPreview(); return; }
  const mine = ++prepSeq;
  prepCtl = new AbortController();
  prepBar.classList.add('on');
  prepCap.textContent = 'Preparing this page at ' + detailSel.value + '…';
  try {
    const res = await fetch(API + '/preview', { method: 'POST', body, signal: prepCtl.signal });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.error || ('Could not prepare the preview (' + res.status + ').'));
    }
    const blob = await res.blob();
    if (mine !== prepSeq) return;
    const head = n => res.headers.get(n) || '';
    const w = head('X-Preview-Width'), h = head('X-Preview-Height');
    const sw = head('X-Preview-Source-Width'), sh = head('X-Preview-Source-Height');
    const pages = Number(head('X-Preview-Pages')) || 1;
    // What the file HOLDS, against what would be read. The preview is the only
    // place this can be said BEFORE an hour of OCR is paid for.
    const pagesTotal = Number(head('X-Preview-Pages-Total')) || pages;
    const maxPages = Number(head('X-Preview-Max-Pages')) || 0;
    dropPreview();
    previewUrl = URL.createObjectURL(blob);
    // This one cannot fail to decode -- the server just encoded it -- so the
    // upload's fall-back-to-a-chip handler would only ever fire on the next file.
    preview.onerror = null;
    preview.onload = () => { preview.style.display = 'block'; };
    preview.src = previewUrl;
    // A fixture keeps its chip: nothing else on the pane names it. An upload has
    // its name and size on the meta line already.
    if (current) chip.style.display = 'none';
    prepCap.innerHTML = 'The model will see this — <b>' + esc(head('X-Preview-Detail'))
      + '</b> · ' + esc(w + '×' + h)
      + (sw && sw + 'x' + sh !== w + 'x' + h
          ? ' <span class="muted-inline">(from ' + esc(sw + '×' + sh) + ')</span>'
          : ' <span class="muted-inline">(unchanged — under the cap)</span>')
      + (pages > 1 ? ' · page 1 of ' + pages : '');
    paintPrepTruncated(pages, pagesTotal, maxPages);
    prepDl.href = previewUrl;
    prepDl.download = prepStem() + '.' + head('X-Preview-Detail') + '.png';
    prepDl.hidden = false;
  } catch (err) {
    if (mine !== prepSeq) return;              // superseded, including by an abort
    // A preview that will not build must not stop anyone reading the document,
    // so this says so quietly and leaves the Read button alone.
    prepCap.textContent = err.message || 'Could not prepare the preview.';
    prepCap.classList.add('err');
  } finally {
    if (mine === prepSeq) prepCtl = null;
  }
}
/* The workspace holds exactly one file, and a new one replaces it. Reading a
   document is a measurement -- one page, one set of settings, one row in the log
   -- and testing them one at a time is what the workspace is for. Several
   documents belong in the Queue tab, which is built for batches and starts on its
   own Run. */
function addFiles(list) {
  if (controller) return;
  const incoming = [...(list || [])].filter(Boolean);
  if (!incoming.length) return;
  // The first of a multi-file drop is kept, and the rest are named rather than
  // silently dropped: a file that is quietly ignored reads as one that ran.
  const file = incoming[0];
  const ignored = incoming.length - 1;
  // Hand-picked files override the folder and benchmark pickers.
  workspaceSourceSeq++;
  workspaceCase = '';
  document.getElementById('workspaceCase').value = '';
  document.getElementById('realPick').value = '';
  current = file;
  previewOne(file);
  paintRun();
  paintServer();
  runNote.textContent = ignored
    ? `One file at a time — kept ${file.name}, ignored ${ignored} other`
      + `${ignored > 1 ? 's' : ''}. Use the Queue tab to run several.`
    : '';
}
function clearWorkspaceFile() {
  workspaceSourceSeq++;
  workspaceCase = '';
  document.getElementById('workspaceCase').value = '';
  document.getElementById('realPick').value = '';
  current = null;
  dropPreview();
  hidePrepared();
  chip.style.display = 'none';
  mapChip.textContent = ''; mapChip.className = '';
  matchedCase = null;
  runNote.textContent = '';
  paintRun();
  paintServer();
}
function paintRun() {
  runBtn.textContent = 'Read document';
}

drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { addFiles(e.target.files); fileInput.value = ''; });
document.getElementById('workspaceCase').addEventListener('change', e => {
  const value = e.target.value;
  if (controller) return;
  clearWorkspaceFile();
  workspaceCase = value;
  e.target.value = value;
  if (value) {
    matchedCase = value;
    mapChip.textContent = 'Ground truth ' + value + ' — will be scored';
    localPreview(); refreshPrepared();
  }
  paintServer(); enableCompare();
});
document.getElementById('realPick').addEventListener('change', e => {
  if (e.target.value) realPickLoad(e.target.value);
});
['dragenter', 'dragover'].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => addFiles(e.dataTransfer.files));
window.addEventListener('paste', e => {
  if (e.clipboardData.files.length) return addFiles(e.clipboardData.files);
  const item = [...e.clipboardData.items].find(i => i.type.startsWith('image/'));
  if (item) addFiles([item.getAsFile()]);
});
// A file dropped anywhere else would otherwise be opened by the browser, which
// navigates away from the page and loses whatever is on screen.
['dragover', 'drop'].forEach(ev => window.addEventListener(ev, e => {
  if (e.target && e.target.closest && e.target.closest('#drop')) return;
  e.preventDefault();
}));

clearBtn.addEventListener('click', () => {
  if (controller || fxBusy) return;
  setReadStatus('ready');
  setTestStatus('fx', 'ready');
  if (fxNote) fxNote.textContent = '';
  text = ''; fileInput.value = '';
  // The folder picker was left selected here, so a "cleared" page would still
  // run whatever was chosen in it.
  clearWorkspaceFile();
  copyBtn.disabled = true;
  pagebar.textContent = ''; warn.textContent = '';
  meta.textContent = ''; meta.classList.remove('err');
  paintTruncated(null);            // a banner from the last run is a lie about this one
  stopTicker(); pageStats = []; renderStats();
  setExtracted(null); extractSteps = null; tabFields.disabled = true;
  fileSplit = null; runningDoc = null; resetGrouping();
  truthOnlyCase = null;
  job = null; pageTexts = []; curPage = -1; cmpPage = 0;
  cmpImg.removeAttribute('src');
  enableCompare();
  paint();
});
copyBtn.addEventListener('click', async () => {
  await navigator.clipboard.writeText(text);
  copyBtn.textContent = 'Copied';
  setTimeout(() => copyBtn.textContent = 'Copy', 1200);
});
stopBtn.addEventListener('click', () => {
  if (controller) { setReadStatus('stopping'); controller.abort(); }
});

/* ---------- streaming run ---------- */
/* ---------- one read driver, two buttons ----------

   The Workspace reads what is DROPPED on it and extracts; the OCR only tab
   reads a NAMED case and stops after pass 1. Everything between those two ends
   -- the stream, the page events, the transcript, the layout blocks, the
   scoring, the ticker, the Stop -- is identical, so it is one function.

   Two copies would drift, and the half that drifted would be the half nobody
   was looking at: this is the same rule `_reply_label` follows server-side, and
   the reason `_read_case` shares `prepare_input` and `log_run` with the real
   endpoint rather than reimplementing them. A test path that differs from the
   real one is a test of the test path.

   `opts` is `{case, image, detail, extract}`. `extract: false` sends
   `extract=0`, which the route already understands -- it is what a read-only
   random-test round sends. */
async function startRead(opts) {
  if (rtRows.length) { rtRows = []; setTestStatus('rt', 'ready'); rtPaint(); }
  // The read floor on screen, before the read that will be judged by it. The
  // change handler has almost always sent it already; this makes "the floor this
  // run used" the one the reader is looking at rather than the one that happened
  // to have landed. No-ops when they match.
  await pushReadFloor(readFloorInp);
  await ensureSampling();
  readBusy(true); copyBtn.disabled = true;
  text = ''; pageStats = []; pageTexts = []; curPage = -1; cmpPage = 0; pageRetried = {};
  layPage = 0; laySel = null; enableLayout();
  roPage = 0; enableRawOut();
  truth = null; tabAccuracy.disabled = true;
  setExtracted(null); extractSteps = null; tabFields.disabled = true;
  fileSplit = null; runningDoc = null; resetGrouping();
  if (['accuracy','fields'].some(m => document.querySelector('#tab' + m[0].toUpperCase() + m.slice(1)).classList.contains('on'))) showTab('rendered');
  paint(); renderStats();
  pagebar.innerHTML = '<span class="spin"></span> Starting…';
  warn.textContent = ''; meta.textContent = ''; meta.classList.remove('err');
  paintTruncated(null);
  startTicker();

  const body = new FormData();
  if (opts.case) body.append('case', opts.case);   // the server loads the PDF
  else body.append('image', opts.image);
  body.append('detail', opts.detail);
  // Pass 2 is skipped rather than run and discarded, so the row's pass-2 columns
  // are blank because nothing extracted -- not because an extraction returned
  // nothing. Those are different states and the run log has to keep them apart.
  if (opts.extract === false) body.append('extract', '0');
  else {
    body.append('fill_missing', document.getElementById('workspaceFillImages').checked ? '1' : '0');
    body.append('validation_retry', document.getElementById('workspaceValidationImages').checked ? '1' : '0');
  }
  controller = new AbortController();

  try {
    const res = await fetch(API + '/ocr/stream', {
      method: 'POST', body, signal: controller.signal
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.error || ('Request failed (' + res.status + ')'));
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', painted = 0, readParallel = 0;

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.event === 'pages_parallel') {
          // Every page is in flight at once; each arrives whole, in page order.
          readParallel = msg.parallel;
          job = msg.job;
          pagebar.innerHTML = '<span class="spin"></span> Reading ' + msg.total
            + ' pages at once (' + msg.parallel + ' in flight)…';
        } else if (msg.event === 'page') {
          pagebar.innerHTML = '<span class="spin"></span> ' + (readParallel
            ? 'Read page ' + msg.page + ' of ' + msg.total + ' · ' + readParallel
              + ' pages at once…'
            : 'Reading page ' + msg.page + ' of ' + msg.total
              + ' (' + msg.resolution + ')…');
          if (msg.page > 1) text += '\n\n--- page ' + msg.page + ' ---\n';
          job = msg.job;
          curPage = msg.page - 1;
          pageTexts[curPage] = '';
          cmpPage = curPage;
          enableCompare();
          if (!compareEl.hidden) paintCompare();
        } else if (msg.event === 'loop_retry') {
          // The page looped; the server threw that read away and is streaming it
          // again one Detail up, so its text so far comes off the transcript.
          const dropped = pageTexts[curPage] || '';
          text = text.slice(0, text.length - dropped.length);
          pageTexts[curPage] = '';
          pageRetried[curPage] = msg.detail;
          painted = text.length;
          paint();
          if (!compareEl.hidden) paintCompare();
          pagebar.innerHTML = '<span class="spin"></span> Page ' + msg.page
            + ' looped — re-reading at ' + esc(msg.detail) + ' (' + msg.resolution + ')…';
        } else if (msg.event === 'page_done') {
          pageStats.push(msg);
          if (msg.loop_retries && msg.loop_retries.length && !msg.looped) {
            warn.textContent = 'Page ' + msg.page + ' looped at '
              + msg.loop_retries.map(a => a.detail).join(', then ')
              + ' and was re-read at ' + msg.detail + '.';
          }
          enableLayout();
          enableRawOut();
          if (curMode === 'layout') paintLayout();
          if (curMode === 'rawout') paintRawOut();
          renderStats();
        } else if (msg.event === 'truncated') {
          warn.textContent = 'Page ' + msg.page + ' hit the output limit and may be cut off. '
            + 'Raise MAX_NEW_TOKENS in settings.py, or split the page.';
        } else if (msg.event === 'looped') {
          warn.textContent = 'Page ' + msg.page + ' started repeating itself and was stopped early — '
            + 'the transcript is incomplete. Try a lower Detail, or a stronger model.';
        } else if (msg.event === 'token') {
          // The server batches pieces (STREAM_FLUSH_MS); n is how many.
          liveTokens += msg.n || 1;
          text += msg.text;
          if (curPage >= 0) pageTexts[curPage] = (pageTexts[curPage] || '') + msg.text;
          // Repainting the rendered view every token is wasteful; throttle it.
          if (text.length - painted > 120) { painted = text.length; paint(); }
          else { rawEl.textContent = text; rawEl.classList.remove('empty'); }
        } else if (msg.event === 'done') {
          text = msg.text;
          stopTicker();
          pagebar.textContent = '';
          // Authoritative per-page text: fence-stripped server-side.
          if (msg.pages && msg.pages.length) pageTexts = msg.pages;
          if (msg.job) job = msg.job;
          enableCompare();
          if (!compareEl.hidden) paintCompare();
          pageStats = msg.page_stats || pageStats;
          enableLayout();
          enableRawOut();
          if (curMode === 'layout') paintLayout();
          if (curMode === 'rawout') paintRawOut();
          renderStats(msg);
          truth = msg.truth || null;
          tabAccuracy.disabled = !truth;
          enableCompare();   // the scored case is what the truth pane shows
          if (truth && !truth.error) {
            // Surface the headline number without needing a tab switch.
            meta.textContent += ` · ${(truth.char_accuracy * 100).toFixed(1)}% vs ${truth.case}`;
          }
          if (tabAccuracy.classList.contains('on')) paintAccuracy();
          meta.textContent = `${msg.page_count} page(s) · ${msg.detail} · `
              + `${msg.tokens} tokens · ${msg.seconds}s total `
              + `(${msg.prefill_seconds}s prefill + ${msg.decode_seconds}s decode) · `
              + `${msg.tokens_per_second} tok/s · ${msg.model}`
              + serverTimeText(msg, msg.server_timing);
          // A read the cap cut short is not a read of that document, and the
          // accuracy beside it is against the whole of it. Said on the meta line
          // AND on the Accuracy tab, because the two are read at different times.
          paintTruncated(msg);
          copyBtn.disabled = !text;
        } else if (msg.event === 'extracting') {
          // OCR is finished and on screen; the second pass runs after it.
          setExtracted({ pending: true });
          extractSteps = null;
          classified = null; fileSplit = null; runningDoc = null; resetGrouping();
          tabFields.disabled = false;
          pagebar.innerHTML = '<span class="spin"></span> Extracting fields…';
          if (tabFields.classList.contains('on')) paintFields();
        } else if (msg.event === 'image_recheck') {
          pagebar.innerHTML = '<span class="spin"></span> ' + esc(msg.message);
        } else if (msg.event === 'table_agent') {
          stepEvent(msg);
        } else if (msg.event === 'segmented' || msg.event === 'document'
                   || msg.event === 'document_done' || msg.event === 'segmenting'
                   || msg.event === 'segment_page') {
          // How many documents the file holds, and which of them pass 2 is on.
          // A three-document file otherwise spends three classifications and
          // twenty-one steps saying nothing about which document any of it is.
          stepEvent(msg);
          if (msg.event === 'document') {
            pagebar.innerHTML = '<span class="spin"></span> Document ' + msg.document
              + ' of ' + msg.documents + ' (page' + (msg.pages.length > 1 ? 's ' : ' ')
              + esc(msg.page_range) + ')...';
          }
        } else if (msg.event === 'classified') {
          // Stage 2 of three, and on a full run it is the only moment the page
          // can say WHY the fields about to appear are the ones they are.
          stepEvent(msg);
          pagebar.innerHTML = '<span class="spin"></span> '
            + (msg.documents > 1 ? 'Document ' + msg.document + ' of ' + msg.documents + ': ' : '')
            + esc((msg.doc_types || []).join(' + ').replace(/_/g, ' ').toLowerCase()
                  || 'unclassified')
            + ' — asking for ' + msg.fields.length + ' fields…';
        } else if (msg.event === 'extract_steps' || msg.event === 'extract_step') {
          // Agentic mode only. The bar names the step because 15 requests with
          // one label between them is the same silence as no progress at all.
          stepEvent(msg);
          if (msg.event === 'extract_steps' && msg.parallel > 1 && extractSteps) {
            extractSteps.parallel = msg.parallel;
          }
          if (extractSteps && extractSteps.parallel > 1) {
            /* On vLLM the steps are asked at once, so "step 7 of 7" would name
               whichever was submitted last. Count the answers instead. */
            const answered = extractSteps.filter(s => s && s.status === 'done').length;
            pagebar.innerHTML = '<span class="spin"></span> Extracting fields — '
              + extractSteps.length + ' steps at once: ' + answered + ' of '
              + extractSteps.length + ' answered';
          } else if (msg.event === 'extract_step' && msg.status === 'running') {
            pagebar.innerHTML = '<span class="spin"></span> Extracting fields — step '
              + msg.step + ' of ' + msg.total + ': ' + esc(msg.title);
          }
        } else if (msg.event === 'fields') {
          const { event, ...rest } = msg;
          setExtracted(rest);
          tabFields.disabled = false;
          pagebar.textContent = '';
          if (tabFields.classList.contains('on')) paintFields();
          // The Accuracy tab now has a second bar to draw, and extraction
          // finishes long after the scores it sits under were painted.
          if (tabAccuracy.classList.contains('on')) paintAccuracy();
        } else if (msg.event === 'logged') {
          // The row for this run is on disk; show it without waiting.
          refreshRuns();
        } else if (msg.event === 'error') {
          throw new Error(msg.error);
        }
      }
    }
    paint();
    setReadStatus('completed');
  } catch (err) {
    setReadStatus(err.name === 'AbortError' ? 'stopped' : 'failed');
    stopTicker();
    pagebar.textContent = '';
    if (err.name === 'AbortError') {
      const s = ((Date.now() - runStart) / 1000).toFixed(1);
      meta.textContent = `Stopped after ${s}s · ${liveTokens} tokens · partial result kept.`;
      copyBtn.disabled = !text;
    } else {
      meta.textContent = err.message;
      meta.classList.add('err');
    }
    paint();
  } finally {
    // **`controller` is cleared BEFORE the repaint, not after.** `paintOcrOnly`
    // reads it to decide whether a read is in flight, so clearing it second left
    // the OCR only tab's Run button disabled for good once a run had finished --
    // the button was painted against a controller that was about to be dropped.
    controller = null;
    readBusy(false);
  }
}

/* Both Run buttons and both Stops move together, because there is one Result
   card and one `controller`: a Stop that looks live on the tab you are not
   looking at is a control lying about what it will do. `paintServer` owns the
   Workspace button's enabled state otherwise, so it is asked rather than set. */
function readBusy(on) {
  if (on) setReadStatus('running');
  clearBtn.disabled = on;
  for (const id of ['workspaceCase', 'realPick', 'workspaceFillImages', 'workspaceValidationImages'])
    document.getElementById(id).disabled = on;
  if (on) { runBtn.disabled = true; if (ocRun) ocRun.disabled = true; }
  else { paintServer(); paintOcrOnly(); }
  stopBtn.disabled = !on;
  if (ocStop) ocStop.disabled = !on;
}

runBtn.addEventListener('click', () => {
  if (!current && !workspaceCase) return;
  startRead({ case: workspaceCase, image: current, detail: detailSel.value, extract: true });
});

/* Workspace preferences are separate from Stress defaults and server User settings. */
const workspaceImageDefaultsKey = 'ocrWorkspaceImageRechecks:v1:' + API;
for (const id of ['workspaceFillImages', 'workspaceValidationImages']) {
  const input = document.getElementById(id);
  try {
    const saved = JSON.parse(localStorage.getItem(workspaceImageDefaultsKey) || '{}');
    if (typeof saved[id] === 'boolean') input.checked = saved[id];
  } catch { /* Defaults still work if browser storage is unavailable. */ }
  input.addEventListener('change', () => {
    try { localStorage.setItem(workspaceImageDefaultsKey, JSON.stringify({
      workspaceFillImages: document.getElementById('workspaceFillImages').checked,
      workspaceValidationImages: document.getElementById('workspaceValidationImages').checked
    })); } catch { /* Keep the current page's selection. */ }
  });
}

/* ---------- Fields only: pass 2 against the hand-written transcript ---------
   `compare.py --fields-only --from-truth` with a button on it. Feeding
   `solution/<id>.md` straight into extraction is the only way a wrong field is
   attributable: a value that never comes back may be one the extractor missed
   or one the read never produced, and text that is right by construction tells
   those apart. Every pass-2 measurement in CLAUDE.md was taken this way.

   The server does the reading (`case` + `from_truth`), not this page. What is
   fetched here is the same file for display -- so the input is on screen beside
   the fields it produced -- out of the cache the compare view already fills. */
let fxBusy = false;

/* ---------- OCR only: pass 1 against the hand-written transcript ----------

   The mirror of the Fields pane, and the two exist for one reason between them:
   a wrong field is otherwise unattributable. Fields only removes the read from
   the question; this removes the extraction, so a transcript score here is the
   reader's alone with nothing downstream to blame. */
const ocCase = document.getElementById('ocCase');
const ocDetail = document.getElementById('ocDetail');
const ocRun = document.getElementById('ocRun');
const ocStop = document.getElementById('ocStop');
const ocDot = document.getElementById('ocDot');
const ocServerText = document.getElementById('ocServerText');
const ocReason = document.getElementById('ocReason');
const ocNote = document.getElementById('ocNote');

function paintOcrOnly() {
  if (!ocRun) return;
  if (ocDetail.value !== detailSel.value) ocDetail.value = detailSel.value;
  /* Vision IS required here, unlike the Fields pane: this pass sends an image. */
  const ok = !!server.available;
  ocDot.className = 'dot ' + (ok ? 'ok' : 'bad');
  ocServerText.textContent = ok
    ? (server.model || server.kind || 'server') + ' · ' + (server.kind || '')
    : 'No model server at ' + server.url;
  ocReason.classList.toggle('err', !ok);
  ocReason.textContent = ok
    ? 'No extraction runs, so the row keeps its pass-1 figures and leaves the '
      + 'pass-2 columns blank.'
    : ('⚠ ' + (server.reason || ''));
  ocRun.disabled = !ok || !ocCase.value || !!controller;
}

if (ocRun) {
  // The Detail is process-wide, so moving it here moves it in the Workspace too.
  ocDetail.addEventListener('change', () => {
    detailSel.value = ocDetail.value;
    detailSel.dispatchEvent(new Event('change'));
    paintOcrOnly();
  });
  ocCase.addEventListener('change', () => {
    /* What the READ is about, not what a form would ask for. The field count
       and the required count belong to pass 2, and printing them on a tab that
       never extracts would describe work this run is not going to do. */
    const c = (FX_CASES || []).find(x => x.id === ocCase.value);
    ocNote.textContent = c
      ? c.kind + ' · scored against solution/' + c.id + '.md'
        + ((c.documents || []).length > 1
           ? ' · ' + c.documents.length + ' documents in this file' : '')
      : '';
    paintOcrOnly();
  });
  ocRun.addEventListener('click', () => {
    if (!ocCase.value) return;
    startRead({ case: ocCase.value, detail: ocDetail.value, extract: false });
  });
  ocStop.addEventListener('click', () => {
    if (controller) { setReadStatus('stopping'); controller.abort(); }
  });
}

function paintFieldsOnly() {
  if (!fxRun) return;
  clearBtn.disabled = fxBusy || !!controller;
  // Vision is deliberately NOT required, unlike the workspace's Run button:
  // pass 2 sends text and gets text back, so a model that cannot read a page can
  // still be measured on the form -- and measuring exactly that is what this
  // pane is for. Hence `text_available`; this read `available` until 2026-08-20,
  // which greyed the button out for every text-only model while the comment
  // above claimed otherwise.
  // Where pass 2 has a server of its own, THAT is the one this pane needs: it
  // reads no page, so the reading server being down is not its problem.
  const ex = extractInfo || {};
  const sep = !!ex.separate;
  const ok = sep ? !!ex.available : !!server.text_available;
  fxDot.className = 'dot ' + (ok ? 'ok' : 'bad');
  fxServerText.textContent = sep
    ? (ok ? (ex.model || ex.kind || 'server') + ' · ' + (ex.kind || '')
            + ' · extraction server ' + ex.url
          : (ex.reason || 'No extraction server at ' + ex.url))
    : ok
    ? (server.model || server.kind || 'server') + ' · ' + (server.kind || '')
      + ' · text only, no image'
    : 'No model server at ' + server.url;
  fxRun.disabled = !ok || !fxCase.value || fxBusy;
  fxMode.disabled = fxBusy;
  if (fxSchemaFirst) fxSchemaFirst.disabled = fxBusy || !isStepMode(extractMode);
}

/* What is about to be fed in, and whether the result can be scored. The two are
   different files -- `solution/<id>.md` is the transcript, `<id>.fields.json`
   says which value belongs in which key -- and a case can have the first without
   the second, in which case the fields come back and nothing marks them. */
async function fxDescribe() {
  if (!fxCase) return;
  const id = fxCase.value;
  fxTruthNote.classList.remove('err');
  if (!id) { fxTruthNote.textContent = ''; return; }
  fxTruthNote.textContent = 'Reading ' + id + '.md …';
  const t = await loadTruth(id);
  if (fxCase.value !== id) return;   // a second choice overtook this one
  if (t.error) {
    fxTruthNote.textContent = t.error;
    fxTruthNote.classList.add('err');
    return;
  }
  const meta = FX_CASES.find(c => c.id === id) || {};
  fxTruthNote.textContent = caseFormNote(id) + t.file + ' · '
    + t.text.length.toLocaleString()
    + ' characters · ' + (meta.field_truth
      ? 'scored against ' + id + '.fields.json'
      : 'no ' + id + '.fields.json — the fields come back unscored');
}

async function runFieldsOnly() {
  if (!fxCase) return;
  const id = fxCase.value;
  if (!id || fxBusy) return;
  const mode = extractMode;
  const schemaFirst = isStepMode(mode) && fxSchemaFirst.checked;
  fxBusy = true; paintFieldsOnly();
  setTestStatus('fx', 'running');
  fxNote.classList.remove('err');
  fxNote.textContent = 'Extracting from ' + id + '.md …';

  const t = await loadTruth(id);
  if (t.error) {
    fxNote.textContent = t.error; fxNote.classList.add('err');
    fxBusy = false; paintFieldsOnly();
    setTestStatus('fx', 'failed');
    return;
  }

  /* Take down everything that belongs to a read before showing what this run
     works on. Pass 1 did not run, so a page image, an accuracy bar or a layout
     left from the last document would be describing something else entirely --
     and sitting above these fields it reads as a property of them. */
  if (controller) controller.abort();
  clearWorkspaceFile();
  stopTicker(); pageStats = []; renderStats();
  job = null; pageTexts = []; curPage = -1; cmpPage = 0; truth = null;
  cmpImg.removeAttribute('src');
  layPage = 0; laySel = null; enableLayout();
  roPage = 0; enableRawOut();
  pagebar.textContent = ''; warn.textContent = '';
  tabAccuracy.disabled = true;
  enableCompare();

  // The input, shown where a transcript is shown. It is the only thing on the
  // page that says these values came out of a hand-written page rather than out
  // of a read, so it is not left as the previous document's text.
  text = t.text; paint(); copyBtn.disabled = false;
  meta.classList.remove('err');
  meta.textContent = 'Ground truth ' + t.file
    + ' — fed straight to pass 2. The page itself was not read, so there is no '
    + 'transcript accuracy for this run.';
  truthOnlyCase = id;
  showTab('fields');

  const r = await extractInto({ case: id, from_truth: true, mode, schema_first: schemaFirst }) || {};
  fxBusy = false; paintFieldsOnly();
  fxNote.classList.toggle('err', !!r.error);
  if (r.error) { setTestStatus('fx', 'failed'); fxNote.textContent = r.error; return; }
  setTestStatus('fx', 'completed');
  const o = (r.field_score || {}).overall || {};
  fxNote.textContent = modeLabel(r) + ' · ' + r.seconds + 's · ' + (r.model || '')
    + (o.expected
        ? ' · field score ' + (o.accuracy * 100).toFixed(1) + '% ('
          + (o.counts || {}).correct + '/' + o.expected + ')'
        : ' · no field truth to score against')
    + ' — on the Fields tab.';
}

if (fxCase) {
  fxCase.addEventListener('change', () => { fxDescribe(); paintFieldsOnly(); });
  fxMode.addEventListener('change', () => setMode(fxMode.value));
  fxRun.addEventListener('click', runFieldsOnly);
}

/* ---------- the Doc types tab (2026-09-07) ----------
   What each document type asks pass 2 for, and what it demands. Reference: it
   reads nothing from the run log, contacts no model server and logs nothing.

   **Every fact in it comes from GET /api/schema.** Which keys a requirement
   asks for, which of them it makes Mandatory, which validation rules a type
   adds and what wording classifies a page as that type are each stated once, in
   `prompts` / `normalise` / `validate`, and a second copy in this file is
   exactly how fifteen keys silently vanished from the Fields tab on 2026-08-19.
   What this file owns is the WORDING of a label, the grouping into sections and
   nothing else -- `FIELD_LABELS`, `FIELD_HINTS`, `FIELD_SECTIONS` and
   `INCOME_COLS`, which the Fields tab already draws from, so one key reads the
   same on both tabs.

   Cached for the life of the page: the requirement cannot change while the
   process runs. */
var scData = null, scPending = false;

/* `TYPE_NAMES` is the article form a prompt uses in a sentence -- "an invoice",
   "a withholding tax certificate (50 ทวิ)". A heading wants the noun. Trimmed
   here rather than sent twice from the server: one string, two presentations. */
function scTypeName(name) {
  return (name || '').replace(/^(an|a|the) /, '');
}

function paintSchema() {
  if (scData) { scRender(); return; }
  if (scPending) return;
  scPending = true;
  document.getElementById('scOut').innerHTML =
    '<p class="note">Loading the field requirements…</p>';
  fetch(API + '/schema')
    .then(r => r.json())
    .then(d => { scData = d; scPending = false; scRender(); })
    .catch(err => {
      scPending = false;
      document.getElementById('scOut').innerHTML =
        '<p class="note err">Could not load the field requirements: '
        + esc(String(err)) + '</p>';
    });
}

/* One matrix cell. Three states, and deliberately not three words: REQUIRED is
   what a reader is scanning for, so it is the only one that gets a chip. The
   word "optional" printed ninety times would be the loudest thing on the page. */
function scCell(col, key, mandatory, asked) {
  const tint = col.unknown ? ' sc-unknown-col' : '';
  if (mandatory.has(key)) {
    return '<td class="sc-req' + tint + '"><span class="req" title="Mandatory in '
      + 'the ' + esc(col.label) + ' requirement">required</span></td>';
  }
  if (asked.has(key)) {
    return '<td class="sc-ask' + tint + '" title="asked for, and the requirement '
      + 'does not demand it">●</td>';
  }
  return '<td class="sc-no' + tint + '" title="not asked for on this type — the '
    + 'prompt never mentions it, so the model cannot answer with it">–</td>';
}

function scRender() {
  const d = scData;
  const withReq = d.types.filter(t => t.has_requirement);
  const without = d.types.filter(t => !t.has_requirement);

  document.getElementById('scHead').textContent =
    withReq.length + ' of ' + d.types.length + ' types carry a requirement · '
    + d.keys.length + ' keys in the schema';

  /* The narrow pane: the legend for the matrix, and the two rules that decide
     how to read it. */
  document.getElementById('scSide').innerHTML =
    '<div class="sc-legend">'
    + '<span><span class="req" style="margin-left:0">required</span> demanded</span>'
    + '<span><b style="color:var(--accent)">●</b> asked, optional</span>'
    + '<span><b style="color:var(--muted)">–</b> not asked</span>'
    + '</div>'
    + '<p class="note">A document is often more than one type at once — a receipt '
    + 'that is also a tax invoice — and its form is the <b>union</b> of theirs. '
    + 'The type is read off the printed heading in Python, never out of an '
    + 'extracted field.</p>'
    + '<p class="note">A type no requirement covers adds no key of its own. Where '
    + 'nothing in play has one, the whole base set is asked and none of it is '
    + 'demanded — the <b>Unknown type</b> column.</p>';

  /* Columns: the types that have a requirement, then what an unplaced document
     gets. The unknown column is last and tinted because it is not a peer of the
     four -- it is the fallback, and reading it as a fifth requirement is the
     misreading worth designing against. */
  const cols = withReq.map(t => ({
      code: t.code, label: scTypeName(t.name), unknown: false,
      asked: new Set(t.fields), mandatory: new Set(t.mandatory),
      items: new Set(t.items), mandatoryItems: new Set(t.mandatory_items)
    })).concat([{
      code: '', label: 'Unknown type', unknown: true,
      asked: new Set(d.unknown.fields), mandatory: new Set(d.unknown.mandatory),
      items: new Set(d.unknown.items), mandatoryItems: new Set()
    }]);

  let rows = '';
  for (const [title, keys] of FIELD_SECTIONS) {
    const live = keys.filter(k => d.keys.indexOf(k) !== -1);
    if (!live.length) continue;
    rows += '<tr class="sc-sec"><td colspan="' + (cols.length + 1) + '">'
      + esc(title) + '</td></tr>';
    for (const key of live) {
      rows += '<tr><td class="sc-key'
        + (FIELD_HINTS[key] ? ' has-hint" title="' + esc(FIELD_HINTS[key]) : '')
        + '">' + esc(FIELD_LABELS[key] || key) + '</td>'
        + cols.map(c => scCell(c, key, c.mandatory, c.asked)).join('') + '</tr>';
    }
  }

  /* The certificate's income table, in the same matrix rather than beside it.
     A cell of that table is not a key -- it is one column of a table, Mandatory
     of EVERY row -- so it gets its own section, and the point it makes is
     visible in the Unknown column: the base set asks for no table at all. A
     scalar nobody rules comes back empty; a table nobody rules comes back
     FILLED, from whatever was nearest. */
  const itemRows = (d.item_keys || []).filter(
    k => INCOME_COLS.some(c => c[0] === k));
  if (itemRows.length) {
    rows += '<tr class="sc-sec"><td colspan="' + (cols.length + 1) + '">'
      + 'Income table — one row per line, every cell required of every row'
      + '</td></tr>';
    for (const key of itemRows) {
      const col = INCOME_COLS.find(c => c[0] === key) || [key, key];
      rows += '<tr><td class="sc-key'
        + (INCOME_HINTS[key] ? ' has-hint" title="' + esc(INCOME_HINTS[key]) : '')
        + '">' + esc(col[1]) + '</td>'
        + cols.map(c => scCell(c, key, c.mandatoryItems, c.items)).join('')
        + '</tr>';
    }
  }

  const head = '<tr><th class="sc-corner">Field</th>'
    + cols.map(c => '<th' + (c.unknown
        ? ' class="sc-unknown-col" title="no requirement covers the type, so the '
          + 'base field set is asked and nothing is Mandatory"' : '')
      + '>' + esc(c.label) + '</th>').join('') + '</tr>';

  /* A card per type, in the order everything else ranks them: most specific
     first. The three with no requirement are listed rather than left out --
     being classifiable and being scoreable are different things, and a reader
     looking for TAX INVOICE should find it here rather than conclude the page
     is broken. */
  const cards = d.types.map(t => {
    const rules = (t.rules || []).map(name => {
      const note = (d.rule_notes || {})[name] || {};
      return '<li' + (note.runs ? '' : ' class="sc-dead"') + '><b>'
        + esc(name.replace(/_/g, ' ')) + '</b> — ' + esc(note.what || '')
        + (note.runs ? ''
           : '. <i>Cannot run here: ' + esc(note.why || '') + '</i>')
        + '</li>';
    }).join('');
    return '<div class="sc-type' + (t.has_requirement ? '' : ' sc-noreq') + '">'
      + '<h4>' + esc(scTypeName(t.name)) + '</h4>'
      + '<div class="sc-count">' + (t.has_requirement
          ? t.fields.length + ' fields · ' + t.mandatory.length + ' required'
            + (t.items.length ? ' · plus a table, ' + t.items.length
                                + ' cells per row' : '')
          : 'no requirement — adds no field of its own')
        + '</div>'
      + (t.headings.length
         ? '<div class="sc-headings" title="the printed wordings that classify a '
           + 'page as this, matched against the heading with spacing and '
           + 'punctuation removed">'
           + t.headings.map(h => '<span>' + esc(h) + '</span>').join('')
           + '</div>'
         : '')
      + (rules ? '<ul class="sc-rules">' + rules + '</ul>' : '')
      + (t.has_requirement ? ''
         : '<p class="note">Classified, and it contributes nothing to a form. A '
           + 'page that is <i>only</i> this is asked the base field set and is '
           + 'held to none of it.</p>')
      + '</div>';
  }).join('');

  document.getElementById('scOut').innerHTML =
    (window.renderDocTypeMapping?.(d.user_mapping) || '')
    + '<div class="sc-grid"><table class="sc-matrix"><thead>' + head
    + '</thead><tbody>' + rows + '</tbody></table></div>'
    + '<div class="sc-types">' + cards + '</div>'
    + '<p class="note">These requirements are the whole of what is extracted. '
    + 'Everything else a page states comes back under its own printed label in '
    + '<b>other_fields</b>, unscored — nothing is lost, it is simply not on a '
    + 'form. ' + (without.length
        ? without.map(t => esc(scTypeName(t.name))).join(', ')
          + ' are recognised and carry no requirement yet, so a page that is only '
          + 'one of them is asked all ' + d.unknown.fields.length + ' keys and '
          + 'scored on what the ground truth states rather than on what it owes.'
        : '') + '</p>';
}

paint();
showTab('raw');
enableCompare();
paintEndpoint();
paintServer();
refreshRuns();

/* ---------- stress test ----------
   One report per run: the server streams a progress event a second (in flight
   by stage, done, live throughput) and one `done` event carrying the report.
   The one per-job event is `alert` -- an error, a loop, a reply at the token
   cap, a salvaged partial reply or a stalled job, naming the document -- shown
   the moment it arrives rather than at the end. Nothing here writes to the run
   log, so nothing here refreshes it. */
const stEl = id => document.getElementById(id);
const stMode = stEl('stMode'), stRun = stEl('stRun'), stStop = stEl('stStop');
const stStop2 = stEl('stStop2'), stClear = stEl('stClear');
const stOut = stEl('stOut'), stNote = stEl('stNote');
let stAbort = null, stStopping = false;
let stState = null;          // {plan, progress, report, metrics, draining, alerts}
let stDetailLog = {parts: null, count: 0, url: '', name: '', summary: [], catalog: {}, report: null};
function stClearDetailLog() {
  if (stDetailLog.url) URL.revokeObjectURL(stDetailLog.url);
  stDetailLog = {parts: null, count: 0, url: '', name: '', summary: [], catalog: {}, report: null};
}
function stDownloadDetailLog() {
  if (!stDetailLog.url) return;
  const link = document.createElement('a');
  link.href = stDetailLog.url; link.download = stDetailLog.name;
  document.body.appendChild(link);
  try { link.click(); }
  catch { /* The download link remains available if automatic download is blocked. */ }
  finally { link.remove(); }
}
function stReceiveDetailLog(ev) {
  if (ev.event === 'detail_log_start') {
    const header = {format: 'ocr-extraction-detail-log', format_version: ev.format_version,
      diagnostics_revision: ev.diagnostics_revision,
      configuration: ev.configuration, plan: ev.plan, provenance: ev.provenance};
    stDetailLog.parts = [JSON.stringify(header).slice(0, -1) + ',"jobs":['];
  } else if (ev.event === 'detail_log_job' && stDetailLog.parts) {
    stDetailLog.parts.push((stDetailLog.count ? ',' : '') + JSON.stringify(ev.job));
    stDetailLog.count++;
  } else if (ev.event === 'detail_log_end') {
    stDetailLog.summary = ev.field_summary || [];
    stDetailLog.catalog = {contexts: ev.contexts || {}, questions: ev.questions || {}};
    stDetailLog.report = ev.report || null;
  }
}
function stFinishDetailLog(ev) {
  if (!stDetailLog.parts) return;
  stDetailLog.parts.push('],"field_summary":' + JSON.stringify(stDetailLog.summary)
    + ',"contexts":' + JSON.stringify(stDetailLog.catalog.contexts || {})
    + ',"questions":' + JSON.stringify(stDetailLog.catalog.questions || {})
    + ',"completion":' + JSON.stringify({received_jobs: stDetailLog.count,
      finished: ev.event === 'done', stopped: !!ev.stopped,
      history_id: ev.history_id || null, error: ev.error || null})
    + ',"report":' + JSON.stringify(stDetailLog.report) + '}');
  stDetailLog.url = URL.createObjectURL(new Blob(stDetailLog.parts, {type: 'application/json'}));
  stDetailLog.name = 'extraction-details-' + ((stState?.plan?.run_id || new Date().toISOString())
    .replace(/[^a-zA-Z0-9_-]/g, '-')) + '.json';
  stDetailLog.parts = null;
  stDetailLog.summary = [];
  stDetailLog.catalog = {};
  stDetailLog.report = null;
  stState.detailDownload = {url: stDetailLog.url, name: stDetailLog.name, count: stDetailLog.count};
  stDownloadDetailLog();
}

function stFileBoxes() {
  return [...stEl('stFiles').querySelectorAll('input')];
}
function stEligibleFile(box) {
  return stMode.value === 'classify' || box.dataset.fieldTruth === '1';
}
function stSelectedCases() {
  return stFileBoxes().filter(box => stEligibleFile(box) && box.checked)
    .map(box => box.value);
}
function stFileInSet(box, name) {
  if (name === 'all') return true;
  const number = Number(/^sol(\d+)$/.exec(box.value)?.[1]);
  return name === '1-20' ? number >= 1 && number <= 20 : number >= 21 && number <= 23;
}
/* realData/: files with no solution. `stReal` is the folder listing
   ({folder, files: [{name, type, size_mb}], types}); the Source pane's picker
   reads it too. Random draws from every file of a ticked type; Choose files
   draws only from the ticked files of a ticked type. */
var stSource = 'benchmark', stRealPick = 'random', stReal = null;
const REAL_MODES = ['ocr', 'full', 'image'];
// The modes that extract a form, with the shape, extras and detail log they take.
const ST_EXTRACT_MODES = ['extract', 'full', 'image'];
/* Explicitly saved per-mode defaults, scoped to this deployment and browser. */
const ST_DEFAULTS_KEY = 'ocrStressDefaults:v1:' + API;
const ST_MODES = ['ocr', 'extract', 'classify', 'full', 'image'];
let stDefaultBase = null, stPendingRealDefaults = null;
function stReadDefaultStore() {
  try {
    const value = JSON.parse(localStorage.getItem(ST_DEFAULTS_KEY));
    if (value?.version === 1 && value.modes && typeof value.modes === 'object'
        && !Array.isArray(value.modes)) return value;
  } catch { /* Missing, invalid or unavailable browser storage uses built-in defaults. */ }
  return {version: 1, modes: {}};
}
function stDefaultControls() {
  return [...stEl('stSetup').querySelectorAll('input[id], select[id]')]
    .filter(el => el.id !== 'stMode');
}
function stCaptureDefaults() {
  const selected = id => [...stEl(id).querySelectorAll('input:checked')].map(el => el.value);
  return {controls: Object.fromEntries(stDefaultControls().map(el =>
      [el.id, el.type === 'checkbox' ? el.checked : el.value])),
    source: stSource, files: selected('stFiles'), batch: stBatchOn, fillImages: stFillOn,
    realPick: stRealPick, realTypes: stReal && !stReal.error ? selected('stRealTypes') : null,
    realFiles: stReal && !stReal.error ? selected('stRealFiles') : null};
}
function stApplyDefaultControls(values) {
  for (const el of stDefaultControls()) {
    const value = values?.[el.id];
    if (el.type === 'checkbox') {
      if (typeof value === 'boolean') el.checked = value;
    } else if (typeof value === 'string') {
      if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value === value)) continue;
      if (el.type === 'number' && (!value.trim() || !Number.isFinite(Number(value))
          || (el.min !== '' && Number(value) < Number(el.min))
          || (el.max !== '' && Number(value) > Number(el.max)))) continue;
      el.value = value;
    }
  }
}
function stApplyRealDefaults() {
  if (!stPendingRealDefaults || !stReal || stReal.error) return;
  for (const [id, key] of [['stRealTypes', 'realTypes'], ['stRealFiles', 'realFiles']]) {
    const values = stPendingRealDefaults[key];
    const selected = Array.isArray(values) ? new Set(values) : null;
    for (const box of stEl(id).querySelectorAll('input')) box.checked = !selected || selected.has(box.value);
  }
  stPendingRealDefaults = null;
}
function stLoadModeDefaults(mode, sourceOverride = null) {
  if (!ST_MODES.includes(mode) || stRunning) return;
  const saved = stReadDefaultStore().modes[mode];
  stMode.value = mode;
  stApplyDefaultControls(stDefaultBase.controls);
  const [concurrent, jobs] = mode === 'ocr' ? [50, 100] : mode === 'extract' ? [64, 40] : [4, 20];
  stEl('stConc').value = String(concurrent);
  stEl('stJobs').value = String(jobs);
  if (saved && typeof saved === 'object') stApplyDefaultControls(saved.controls);
  const files = new Set(Array.isArray(saved?.files) ? saved.files : stDefaultBase.files);
  for (const box of stFileBoxes()) box.checked = files.has(box.value);
  stBatchOn = saved?.batch === true;
  stFillOn = saved?.fillImages !== false;
  stRealPick = saved?.realPick === 'choose' ? 'choose' : 'random';
  stPendingRealDefaults = saved && typeof saved === 'object' ? saved : {};
  stApplyRealDefaults();
  const source = sourceOverride || saved?.source;
  stSetSource(source === 'realdata' && REAL_MODES.includes(mode) ? 'realdata' : 'benchmark');
  const note = stEl('stDefaultsNote');
  note.hidden = true;
  note.classList.remove('err');
}
function stSaveModeDefaults() {
  if (stRunning) return;
  const note = stEl('stDefaultsNote');
  note.hidden = false;
  note.classList.remove('err');
  if (stSource === 'realdata' && (!stReal || stReal.error)) {
    note.textContent = 'Load the realData file list before saving defaults.';
    note.classList.add('err');
    return;
  }
  const invalid = stDefaultControls().find(el => !el.disabled && !el.checkValidity());
  if (invalid) { invalid.reportValidity(); note.hidden = true; return; }
  // Re-read so saving one mode cannot overwrite another tab's saved modes.
  const store = stReadDefaultStore();
  store.modes[stMode.value] = stCaptureDefaults();
  store.lastMode = stMode.value;
  try {
    localStorage.setItem(ST_DEFAULTS_KEY, JSON.stringify(store));
    const name = stMode.options[stMode.selectedIndex].textContent;
    note.textContent = 'Saved defaults for ' + name + ' in this browser.';
  } catch {
    note.textContent = 'Defaults could not be saved. Browser storage is unavailable or full.';
    note.classList.add('err');
  }
}
function stRealTypes() {
  return [...stEl('stRealTypes').querySelectorAll('input:checked')].map(box => box.value);
}
function stRealPool() {
  const types = new Set(stRealTypes());
  return ((stReal && stReal.files) || []).filter(f => types.has(f.type)).map(f => f.name);
}
function stRealSelected() {
  if (stRealPick === 'random') return stRealPool();
  const types = new Set(stRealTypes());
  return [...stEl('stRealFiles').querySelectorAll('input:checked')]
    .filter(box => types.has(box.dataset.type)).map(box => box.value);
}
function stSelectedCount() {
  return stSource === 'realdata' ? stRealSelected().length : stSelectedCases().length;
}
async function stLoadReal() {
  // Ticks survive a Refresh; a type or file new to the listing starts ticked.
  const offTypes = new Set([...stEl('stRealTypes').querySelectorAll('input:not(:checked)')].map(b => b.value));
  const offFiles = new Set([...stEl('stRealFiles').querySelectorAll('input:not(:checked)')].map(b => b.value));
  stEl('stRealCount').textContent = 'Listing realData…';
  try {
    const res = await fetch(API + '/realdata');
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    stReal = data;
  } catch (err) {
    stReal = { folder: '', files: [], types: [], error: err.message };
  }
  const chip = (value, label, title, off, type) => {
    const box = Object.assign(document.createElement('input'), { type: 'checkbox', value, checked: !off });
    if (type) box.dataset.type = type;
    const span = document.createElement('span');
    span.textContent = label;
    const el = document.createElement('label');
    el.className = 'rt-chip';
    el.title = title;
    el.append(box, span);
    return el;
  };
  const count = type => stReal.files.filter(f => f.type === type).length;
  stEl('stRealTypes').replaceChildren(...stReal.types.map(t =>
    chip(t, t.toUpperCase() + ' · ' + count(t), count(t) + ' ' + t + ' file(s)', offTypes.has(t))));
  stEl('stRealFiles').replaceChildren(...stReal.files.map(f =>
    chip(f.name, f.name + ' · ' + f.size_mb + ' MB', f.name, offFiles.has(f.name), f.type)));
  stApplyRealDefaults();
  stPaintFiles();
  realPickPaint();
}
function stPaintReal(busy) {
  const types = new Set(stRealTypes());
  for (const box of stEl('stRealTypes').querySelectorAll('input')) {
    box.disabled = busy;
    box.parentElement.classList.toggle('off', !box.checked);
  }
  for (const box of stEl('stRealFiles').querySelectorAll('input')) {
    box.disabled = busy;
    box.parentElement.style.display = types.has(box.dataset.type) ? '' : 'none';
    box.parentElement.classList.toggle('off', !box.checked);
  }
  for (const button of stEl('stRealPick').querySelectorAll('button')) {
    const on = button.dataset.stRealPick === stRealPick;
    button.disabled = busy;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  }
  stEl('stRealRefresh').disabled = busy;
  stEl('stRealFiles').hidden = stRealPick !== 'choose';
  const pool = stRealPool(), selected = stRealSelected();
  stEl('stRealCount').textContent = !stReal ? 'Listing realData…'
    : stReal.error ? 'Could not list realData: ' + stReal.error
    : !stReal.files.length ? 'realData/ is empty — put PDFs or images in ' + (stReal.folder || 'realData/') + ' and press Refresh.'
    : !pool.length ? 'Tick at least one file type.'
    : stRealPick === 'random'
      ? 'Random — every job draws one of ' + pool.length + ' file(s). Nothing is scored.'
      : selected.length + ' / ' + pool.length + ' files selected'
        + (selected.length ? '. Nothing is scored.' : ' — select at least one file');
  stMode.disabled = busy;
  stRun.disabled = busy || !selected.length || !REAL_MODES.includes(stMode.value);
}
function stSetSource(source) {
  stSource = source;
  for (const button of stEl('stSource').querySelectorAll('button')) {
    const on = button.dataset.stSource === source;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  }
  const real = source === 'realdata';
  stEl('stRealField').hidden = !real;
  stEl('stFilesField').hidden = real;
  // realData has no transcript truth, so the modes that start from one go.
  for (const option of stMode.options) option.disabled = real && !REAL_MODES.includes(option.value);
  if (real && !REAL_MODES.includes(stMode.value)) {
    stLoadModeDefaults('ocr', 'realdata');
    return;
  }
  if (real && !stReal) stLoadReal();
  stPaintMode();
}

/* The Source pane's realData picker: a chosen file is fetched and handed to
   addFiles, so it is read exactly as a dropped file would be. */
function realPickPaint() {
  const sel = document.getElementById('realPick');
  if (!sel || !stReal) return;
  const selected = sel.value;
  document.getElementById('realPickRow').hidden = !stReal.files.length;
  const groups = stReal.types.map(type => {
    const group = document.createElement('optgroup');
    group.label = type.toUpperCase();
    for (const f of stReal.files.filter(f => f.type === type)) {
      group.append(new Option(f.name + ' · ' + f.size_mb + ' MB', f.name));
    }
    return group;
  });
  sel.replaceChildren(new Option('— select a real file —', ''), ...groups);
  sel.value = selected;
}
async function realPickLoad(name) {
  const sel = document.getElementById('realPick');
  if (controller) return;
  clearWorkspaceFile();
  const mine = workspaceSourceSeq;
  let applied = false;
  sel.value = name;
  sel.disabled = true;
  runNote.textContent = 'Loading ' + name + ' from realData…';
  try {
    const res = await fetch(API + '/realdata/file?name=' + encodeURIComponent(name));
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    const blob = await res.blob();
    if (mine !== workspaceSourceSeq) return;
    addFiles([new File([blob], name.split('/').pop(), { type: blob.type })]);
    sel.value = name;
    applied = true;
  } catch (err) {
    if (mine === workspaceSourceSeq) runNote.textContent = 'Could not load ' + name + ': ' + err.message;
  } finally {
    if (applied || mine === workspaceSourceSeq) {
      sel.disabled = !!controller;
      if (!applied) sel.value = '';
    }
  }
}

function stPaintFiles() {
  const busy = !!stAbort || rtRunning;
  for (const button of stEl('stSource').querySelectorAll('button')) button.disabled = busy;
  if (stSource === 'realdata') { stPaintReal(busy); return; }
  const boxes = stFileBoxes();
  const eligible = boxes.filter(stEligibleFile);
  const selected = new Set(stSelectedCases());
  for (const box of boxes) {
    const usable = stEligibleFile(box);
    box.parentElement.style.display = usable ? '' : 'none';
    box.parentElement.classList.toggle('off', !box.checked);
    box.disabled = busy || !usable;
  }
  stEl('stFileCount').textContent = selected.size + ' / ' + eligible.length
    + ' files selected' + (selected.size ? '' : ' — select at least one file');
  for (const button of stEl('stFileSets').querySelectorAll('button')) {
    const name = button.dataset.stFileSet;
    const available = eligible.some(box => stFileInSet(box, name));
    const on = available && eligible.every(box => box.checked === stFileInSet(box, name));
    button.disabled = busy || !available;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  }
  stMode.disabled = busy;
  stRun.disabled = busy || !selected.size;
}

function stPaintMode() {
  const m = stMode.value;
  stEl('stDetailField').style.display = ['ocr', 'full', 'image'].includes(m) ? '' : 'none';
  stEl('stShapeField').style.display = ST_EXTRACT_MODES.includes(m) ? '' : 'none';
  stEl('stExtrasField').style.display = ST_EXTRACT_MODES.includes(m) ? '' : 'none';
  stEl('stSchemaFirstField').style.display = ST_EXTRACT_MODES.includes(m) ? '' : 'none';
  stEl('stSchemaFirst').disabled = stRunning || !ST_EXTRACT_MODES.includes(m)
    || !isStepMode(stEl('stShape').value);
  stEl('stDetailsField').style.display = ST_EXTRACT_MODES.includes(m) ? '' : 'none';
  stEl('stSplitPagesField').style.display = ['ocr', 'full'].includes(m) ? '' : 'none';
  stEl('stSaveDetails').disabled = stRunning || !ST_EXTRACT_MODES.includes(m);
  stEl('stPagesField').style.display = m === 'classify' ? '' : 'none';
  stEl('stStrategyField').style.display = m === 'classify' ? 'none' : '';
  // Classification writes no row, so the box is not offered there; nor does an
  // image-fed job, whose row would have no transcript to hold its fields to.
  const busy = stRunning;
  stEl('stSaveLog').disabled = m === 'classify' || m === 'image' || busy;
  stEl('stSaveLogRow').title = m === 'classify' ? 'Unavailable for classification'
    : m === 'image' ? 'Unavailable for Image to extractor: a run-log row is a read and its transcript' : '';
  stPaintVision();
  stPaintBatch();
  stPaintFill();
  stPaintFiles();
}

/* Image mode sends page images to the EXTRACTION model, so say whether it takes
   them: refused on Run where its server says no, tried where it does not say.
   Hoisted like stPaintBatch, and looks its elements up. */
function stPaintVision() {
  const note = document.getElementById('stVisionNote');
  const mode = document.getElementById('stMode');
  if (!note || !mode) return;
  note.hidden = mode.value !== 'image';
  if (note.hidden) return;
  const info = extractInfo || {};
  const name = info.model || '';
  const entry = (info.models || []).find(m => m.name === name);
  const sees = entry ? entry.vision : null;
  note.classList.toggle('err', sees === false);
  note.textContent = !name ? 'No extraction model is selected.'
    : sees === false ? 'The extraction model ' + name + ' reports no vision, so it cannot be sent page images. Choose a vision model for extraction.'
    : sees === true ? 'No OCR: each job sends its page images to ' + name + ', which reports vision.'
    : 'No OCR: each job sends its page images to ' + name + '. Its server does not say whether it takes images; a model that cannot fails its first request, and the run flags it.';
}

/* The batch toggle follows the servers THIS MODE sends to: only vLLM has the
   /batch route. Hoisted like stPaintServer, and looks its elements up. */
var stBatchOn = false;   // var, not let: stPaintBatch can run before this line
function stPaintBatch() {
  const btn = document.getElementById('stBatch');
  const mode = document.getElementById('stMode');
  if (!btn || !mode) return;
  const readVllm = !!(server && server.kind === 'vllm');
  const extractVllm = extractInfo && extractInfo.separate
    ? extractInfo.kind === 'vllm' : readVllm;
  const m = mode.value;
  const ok = m === 'ocr' ? readVllm : m === 'full' ? (readVllm || extractVllm) : extractVllm;
  const busy = stRunning;
  if (!ok && !busy) stBatchOn = false;
  btn.disabled = !ok || busy;
  btn.classList.toggle('on', stBatchOn);
  btn.setAttribute('aria-pressed', String(stBatchOn));
  btn.textContent = 'Batch requests: ' + (stBatchOn ? 'on' : 'off');
  btn.title = !ok ? 'Requires vLLM' : '';
}

/* Full pipeline only: a required field left empty is asked of the page images
   with its Hybrid step. Hoisted and looks its elements up, like stPaintBatch. */
var stFillOn = true;   // var, not let: stPaintFill can run before this line
function stPaintFill() {
  const btn = document.getElementById('stFillImages');
  const mode = document.getElementById('stMode');
  if (!btn || !mode) return;
  btn.hidden = mode.value !== 'full';
  btn.disabled = stRunning;
  btn.classList.toggle('on', stFillOn);
  btn.setAttribute('aria-pressed', String(stFillOn));
  btn.textContent = 'Fill missing from images: ' + (stFillOn ? 'on' : 'off');
  const validation = document.getElementById('stValidationField');
  if (validation) validation.hidden = mode.value !== 'full';
  const input = document.getElementById('stValidationImages');
  if (input) input.disabled = stRunning || mode.value !== 'full';
}

/* Live monitoring and saved snapshots are offered only for vLLM servers. */
// A function declaration, so it is hoisted and `paintEndpoint` can call it at
// parse time -- which is why it looks its elements up rather than reading the
// consts below, whose temporal dead zone would take the whole script down.
function stPaintServer() {
  const vllm = server && server.kind === 'vllm';
  const extractVllm = !!(extractInfo && extractInfo.separate && extractInfo.kind === 'vllm');
  const monVllm = vllm || extractVllm;
  const snapshots = document.getElementById('stHistoryMonitor');
  if (snapshots) {
    snapshots.disabled = stRunning || !monVllm;
    snapshots.parentElement.title = monVllm ? '' : 'Requires vLLM';
  }
  const monBtn = document.getElementById('monToggle');
  if (monBtn) {
    monBtn.disabled = !monVllm;
    monBtn.title = monVllm ? 'Show live server metrics on any tab' : 'Live monitor requires a vLLM server';
    if (!monVllm && typeof monState !== 'undefined' && monState && monState.on) monSet(false);
  }
  if (typeof monState !== 'undefined' && monState && monState.on) { monSync(); paintMon(); }
  stPaintVision();
  stPaintBatch();
}

function showStressView(view) {
  stressView = view;
  historyOpen = stressPaneOpen && view === 'history';
  stEl('stSetup').hidden = view !== 'run';
  stEl('stHistoryPanel').hidden = view !== 'history';
  stEl('stViews').querySelectorAll('button[data-st-view]').forEach(button => {
    const on = button.dataset.stView === view;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  });
  stPaint();
  if (historyOpen) historyLoad();
}

function stBusy(on) {
  stRunning = on;
  stEl('stSaveDefaults').disabled = on;
  if (on) { stStopping = false; setTestStatus('st', 'running'); }
  stClear.disabled = on;
  stRun.disabled = on;
  stEl('stNoCache').disabled = on;
  stEl('stSplitPages').disabled = on;
  stEl('stSchemaFirst').disabled = on || !ST_EXTRACT_MODES.includes(stMode.value)
    || !isStepMode(stEl('stShape').value);
  stEl('stSaveLog').disabled = on || ['classify', 'image'].includes(stMode.value);
  stEl('stSaveDetails').disabled = on || !ST_EXTRACT_MODES.includes(stMode.value);
  stPaintBatch();
  stPaintFill();
  stStop.disabled = stStop2.disabled = !on || stStopping;
  if (rtRun) rtRun.disabled = on;
  if (on) runBtn.disabled = true; else paintServer();
  stPaintServer();
  stPaintFiles();
}

const stNum = (v, d = 1) => (v === null || v === undefined) ? '—'
  : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
const stSec = v => (v === null || v === undefined) ? '—'
  : (v >= 100 ? Math.round(v) + ' s' : Number(v).toFixed(v >= 10 ? 1 : 2) + ' s');
const stPct = v => (v === null || v === undefined) ? '—' : (v * 100).toFixed(1) + '%';

function stTile(label, value, sub, cls) {
  return '<div class="st-tile' + (cls ? ' ' + cls : '') + '"><div class="st-tile-label">'
    + esc(label) + '</div><div class="st-tile-num">' + value + '</div>'
    + (sub ? '<div class="st-tile-sub">' + sub + '</div>' : '') + '</div>';
}

function stLatencyRows(rows) {
  const cols = ['n', 'mean', 'p50', 'p90', 'p95', 'p99', 'max'];
  const body = rows.filter(r => r[1] && r[1].n).map(([name, d, tip, unit]) =>
    '<tr><td title="' + esc(tip || '') + '">' + esc(name) + '</td>'
    + cols.map(c => '<td class="n">' + (c === 'n' ? d.n
        : unit === 'tps' ? stNum(d[c], 1) : stSec(d[c])) + '</td>').join('') + '</tr>'
  ).join('');
  if (!body) return '';
  return '<table class="queue st-lat"><tr><th>Stage</th>'
    + cols.map(c => '<th class="n">' + c + '</th>').join('') + '</tr>' + body + '</table>';
}

/* In-flight jobs over time, stacked by stage. One axis (jobs in flight), two
   series at most -- only two stages can hold jobs in any one mode -- with a
   legend and a hover readout, and the numbers are in the tables beside it. */
function stChart(report, host = stOut) {
  const t = report.timeline || [];
  const cols = report.timeline_columns || [];
  if (t.length < 2) return '';
  const idx = n => cols.indexOf(n);
  const mode = report.mode;
  const series = (mode === 'classify' ? [['segment', 'Splitting'], ['classify', 'Classifying']]
    : mode === 'extract' ? [['extract', 'Extracting']]
    : mode === 'ocr' ? [['ocr', 'Reading']]
    : [['ocr', 'Reading (OCR)'], ['extract', 'Extracting']]);
  // Drawn at the card's real width, so the axis text is never stretched.
  const W = Math.max(320, Math.round(host.clientWidth || 640)), H = 190;
  const L = 34, R = 10, T = 10, B = 22;
  const tMax = t[t.length - 1][0] || 1;
  const yMax = Math.max(1, report.concurrency || 1,
    ...t.map(p => series.reduce((a, [k]) => a + (p[idx(k)] || 0), 0)));
  const x = v => L + (W - L - R) * v / tMax;
  const y = v => T + (H - T - B) * (1 - v / yMax);
  let base = t.map(() => 0);
  const paths = series.map(([k, label], si) => {
    const top = t.map((p, i) => base[i] + (p[idx(k)] || 0));
    const up = t.map((p, i) => x(p[0]) + ',' + y(top[i])).join(' L');
    const down = t.map((p, i) => x(p[0]) + ',' + y(base[i])).reverse().join(' L');
    base = top;
    return '<path d="M' + up + ' L' + down + ' Z" class="st-area st-s' + (si + 1) + '"/>';
  }).join('');
  const ticks = [0, Math.round(yMax / 2), yMax].filter((v, i, a) => a.indexOf(v) === i)
    .map(v => '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y(v) + '" y2="' + y(v)
      + '" class="st-grid"/><text x="' + (L - 6) + '" y="' + (y(v) + 4)
      + '" class="st-axis" text-anchor="end">' + v + '</text>').join('');
  const xt = [0, tMax / 2, tMax].map(v => '<text x="' + x(v) + '" y="' + (H - 6)
    + '" class="st-axis" text-anchor="middle">' + stSec(v) + '</text>').join('');
  const conc = '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y(report.concurrency)
    + '" y2="' + y(report.concurrency) + '" class="st-conc"/>';
  const legend = series.map(([, label], si) => '<span class="st-key"><i class="st-sw st-s'
    + (si + 1) + '"></i>' + esc(label) + '</span>').join('')
    + '<span class="st-key"><i class="st-sw st-dash"></i>concurrency limit</span>';
  return '<div class="st-chart-wrap"><div class="st-legend">' + legend + '</div>'
    + '<svg class="st-chart" viewBox="0 0 ' + W + ' ' + H + '" data-w="' + W + '" '
    + 'data-series="' + esc(JSON.stringify(series.map(s => [idx(s[0]), s[1]]))).replace(/"/g, '&quot;') + '">'
    + ticks + paths + conc + xt
    + '<line class="st-cross" y1="' + T + '" y2="' + (H - B) + '" x1="-10" x2="-10"/></svg>'
    + '<div class="st-tip" hidden></div></div>';
}

function stWireChart(host = stOut, rep = (stState || {}).report) {
  const svg = host.querySelector('.st-chart');
  if (!svg || !rep) return;
  const t = rep.timeline, cols = rep.timeline_columns;
  const series = JSON.parse(svg.dataset.series);
  const tip = svg.parentElement.querySelector('.st-tip');
  const cross = svg.querySelector('.st-cross');
  const tMax = t[t.length - 1][0] || 1;
  svg.onmousemove = e => {
    const box = svg.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    const W = Number(svg.dataset.w) || 640, L = 34, R = 10;
    const tv = Math.max(0, Math.min(tMax, (frac * W - L) / (W - L - R) * tMax));
    let best = t[0];
    for (const p of t) if (Math.abs(p[0] - tv) < Math.abs(best[0] - tv)) best = p;
    const cx = L + (W - L - R) * best[0] / tMax;
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx);
    tip.hidden = false;
    tip.innerHTML = '<b>' + stSec(best[0]) + '</b>'
      + series.map(([i, label]) => '<div>' + esc(label) + ': ' + best[i] + '</div>').join('')
      + '<div>Queued: ' + best[cols.indexOf('queued')] + '</div>'
      + '<div>Done: ' + best[cols.indexOf('done')] + '</div>';
    const px = e.clientX - box.left;
    tip.style.left = Math.min(px + 12, box.width - 150) + 'px';
  };
  svg.onmouseleave = () => {
    tip.hidden = true; cross.setAttribute('x1', -10); cross.setAttribute('x2', -10);
  };
}

function stMetricsTable(view, title, since) {
  if (!view) return '';
  const rows = [
    ['Requests running', view.running, 'n'], ['Requests waiting', view.waiting, 'n'],
    ['KV cache usage', view.kv_cache, 'pct'],
    ['Prompt tokens' + since, view.prompt_tokens, 'n'],
    ['Generation tokens' + since, view.generation_tokens, 'n'],
    ['Prompt tokens / s', view.prompt_tokens_per_s, 'n'],
    ['Generation tokens / s', view.generation_tokens_per_s, 'n'],
    ['Requests finished' + since, view.requests_ok, 'n'],
    ['Preemptions' + since, view.preemptions, 'n'],
    ['Prefix cache hit rate', view.prefix_hit_rate, 'pct'],
    ['Mean time to first token', view.ttft, 's'],
    ['Mean time per output token', view.tpot, 's'],
    ['Mean queue time', view.queue_time, 's'],
    ['Mean prefill time', view.prefill_time, 's'],
    ['Mean decode time', view.decode_time, 's'],
    ['Mean end-to-end', view.e2e, 's'],
  ].filter(r => r[1] !== null && r[1] !== undefined);
  if (!rows.length) return '<p class="note">The server answered /metrics but none of '
    + 'the vLLM series this page knows were in it.</p>';
  return '<h3 class="st-h">' + esc(title) + '</h3><table class="queue st-kv">'
    + rows.map(([k, v, u]) => '<tr><td>' + esc(k) + '</td><td class="n">'
      + (u === 'pct' ? stPct(v) : u === 's' ? stSec(v) : stNum(v, 1)) + '</td></tr>').join('')
    + '</table>';
}

const ST_ALERT_TEXT = {
  error: 'error', looped: 'loop', truncated: 'token cap', partial: 'partial reply',
  stalled: 'stalled', recovered: 'retry read cleanly', dry: 'DRY',
  missing: 'missing required', wrong: 'wrong field', validation: 'validation image recheck'
};

/* The running score (`_StressLive.score`): what the jobs finished so far
   scored, the same populations the report uses. Nothing where no job could be
   scored -- a realData run, or none finished yet. */
function stLiveScore(sc) {
  if (!sc) return '';
  const parts = [];
  if (sc.char_scored) parts.push('transcript <b>' + stPct(sc.char_accuracy) + '</b> of '
    + sc.char_scored + ' read(s)');
  if (sc.field_scored) parts.push('fields <b>' + stPct(sc.field_accuracy) + '</b> of '
    + sc.field_scored + ' job(s) · pooled ' + stPct(sc.field_pooled) + ' of '
    + sc.field_values + ' values');
  if (sc.wrong_fields) parts.push('<span style="color:var(--danger)">' + sc.wrong_fields
    + ' wrong field(s) in ' + sc.wrong_jobs + ' job(s)</span>');
  if (sc.files) parts.push(sc.doc_sets_extracted + ' of ' + sc.doc_sets
    + ' document set(s) extracted from ' + sc.files + ' file(s)');
  return parts.length ? '<div class="st-live"><b>Score so far</b> · '
    + parts.join(' · ') + '</div>' : '';
}
// A looped page's one row, by where its retries got to (`_StressLive.update_flag`).
const ST_LOOP_STATE = { retrying: 'loop retrying', resolved: 'loop resolved',
  'retry failed': 'loop retry failed', 'no retry': 'loop, no retry' };
// A DRY page is read again one Detail up as well (STRESS_DRY_RETRY).
const ST_DRY_STATE = { retrying: 'DRY retrying', resolved: 'DRY resolved',
  'retry failed': 'DRY retry failed', 'no retry': 'DRY, no retry' };
// Image mode extracts a form again one Detail up when a required field came
// back empty or the reply ran to the cap (`_stress_image_extract`).
const ST_MISSING_STATE = { retrying: 'missing, retrying', resolved: 'missing resolved',
  'retry failed': 'missing retry failed', 'no retry': 'missing, no retry' };
const ST_CAP_STATE = { retrying: 'token cap, retrying', resolved: 'token cap resolved',
  'retry failed': 'token cap retry failed', 'no retry': 'token cap, no retry' };
const ST_STAGE_TEXT = { ocr: 'reading', extract: 'extracting', segment: 'splitting',
  classify: 'classifying' };

function stAlertTag(a) {
  if (a.kind === 'looped' && ST_LOOP_STATE[a.state]) return ST_LOOP_STATE[a.state];
  if (a.kind === 'dry' && ST_DRY_STATE[a.state]) return ST_DRY_STATE[a.state];
  if (a.kind === 'missing' && ST_MISSING_STATE[a.state]) return ST_MISSING_STATE[a.state];
  if (a.kind === 'truncated' && ST_CAP_STATE[a.state]) return ST_CAP_STATE[a.state];
  if (a.kind === 'validation') return 'validation · ' + (a.state || 'flagged');
  return ST_ALERT_TEXT[a.kind] || a.kind;
}

function stAlertClass(a) {
  if (a.kind === 'error' || a.kind === 'wrong' || a.state === 'retry failed') return 'st-bad';
  if (a.kind === 'recovered' || a.state === 'resolved') return 'st-good';
  if (a.kind === 'dry') return 'st-info';
  return 'st-warn';
}

/* The flags in two parts (2026-10-26, at the user's request): the read's, as
   they always were, and everything after it -- classify and extraction, where a
   looped read still goes and where a required field that came back empty is
   flagged. Split by the stage each alert names. */
const ST_ALERT_PARTS = [
  ['ocr', 'OCR', a => a.stage === 'ocr'],
  ['extract', 'Extraction', a => a.stage !== 'ocr'],
];

// `mode` is the run's (`plan.mode`): only a full run extracts what it read.
function stAlertsHtml(alerts, mode) {
  return ST_ALERT_PARTS.map(([part, title, test]) =>
    stAlertBox(alerts.filter(test), part, title, mode)).join('');
}

function stAlertCounts(alerts) {
  return ST_ALERT_PARTS.map(([, title, test]) => [title, alerts.filter(test).length])
    .filter(([, n]) => n);
}

// Newest first. Only the first rows are drawn: a run where every job fails
// would otherwise redraw thousands of rows a second.
function stAlertBox(alerts, part, title, mode) {
  if (!alerts.length) return '';
  const counts = {};
  for (const a of alerts) {
    const key = stAlertTag(a);
    counts[key] = (counts[key] || 0) + 1;
  }
  const shown = alerts.slice(-100).reverse();
  return '<div class="st-alerts" data-st-part="' + part + '" data-st-mode="'
    + esc(mode || '') + '"><h3>' + esc(title)
    + ' problems, as they happen · '
    + Object.entries(counts).map(([k, n]) => n + ' ' + esc(k)).join(' · ')
    + '</h3><table class="queue"><tr><th>At</th><th>Job</th><th>Document</th><th>What</th>'
    + '<th>Where</th><th>Detail</th></tr>'
    + shown.map(a => (a.evidence
        ? '<tr class="st-alert-link' + (cmpOverride && cmpOverride.alert === a.id ? ' on' : '')
          + '" data-alert-id="' + esc(String(a.id)) + '" data-alert-kind="' + esc(a.kind)
          + '" data-alert-job="' + esc(String(a.job)) + '" data-alert-part="' + part
          + '" data-alert-case="' + esc(a.case || '')
          + '" data-alert-revision="' + esc(String(a.updated ?? a.elapsed ?? ''))
          + '" data-alert-page="' + esc(String(a.page || '')) + '" data-alert-tag="'
          + esc(stAlertTag(a)) + '" title="Show this page in Compare below">'
        : '<tr>') + '<td class="n">' + stSec(a.elapsed)
      + (a.phase === 'warmup' ? ' <small>warm-up</small>' : '')
      + (a.updated !== undefined && a.updated !== null
        ? '<br><small>updated ' + stSec(a.updated) + '</small>' : '') + '</td>'
      + '<td class="n">#' + a.job + '</td><td><b>' + esc(a.case || '') + '</b></td>'
      + '<td><span class="st-tag ' + stAlertClass(a) + '">'
      + esc(stAlertTag(a)) + '</span></td>'
      + '<td>' + esc(ST_STAGE_TEXT[a.stage] || a.stage || '')
      + (a.page ? ', page ' + a.page : '') + '</td>'
      + '<td class="msg">' + esc(a.message || '') + '</td></tr>').join('')
    + '</table>'
    + (alerts.length > shown.length ? '<p class="note">' + (alerts.length - shown.length)
      + ' older row(s) not shown.</p>' : '')
    + '</div>';
}

/* A flagged page into the Compare card below: the page each read was sent and
   the text it returned, starting at the read the alert is about. */
async function stShowEvidence(alertId, kind, host, job) {
  const ticket = ++stEvidenceRequest;
  const base = API + '/stress/evidence/' + encodeURIComponent(alertId);
  // The row already identifies the source. Open a selected PC file immediately;
  // large extraction payloads must not stand between a click and the original.
  const row = [...(host?.querySelectorAll('tr.st-alert-link') || [])]
    .find(row => row.dataset.alertId === String(alertId));
  const previous = cmpOverride?.host === host && String(cmpOverride?.job) === String(job) ? cmpOverride : null;
  const caseName = row?.dataset.alertCase || previous?.case;
  const fixture = FX_CASES.find(c => c.id === caseName);
  const filename = fixture?.pdf || caseName || previous?.filename;
  let openedLocal = false;
  if (filename && window.StressLocalPreview?.hasCandidate(filename)) {
    const page = Number(row?.dataset.alertPage) || previous?.page || 1;
    cmpOverride = {alert: alertId, kind, case: caseName, filename, page,
      truthCase: fixture?.id || null, reads: [{page, text: '', src: null}], at: 0,
      fields: null, host, job, loading: true};
    cmpView = 'source-model';
    stOpenEvidenceCompare(alertId);
    openedLocal = true;
  }
  let data;
  try {
    data = await loadStressEvidence(base, row?.dataset.alertRevision || '');
  } catch (err) {
    if (ticket !== stEvidenceRequest) return;
    stNote.textContent = err.message;
    stNote.classList.add('err');
    if (openedLocal) {
      cmpOverride.loading = false;
      cmpOverride.loadError = 'Comparison results unavailable: ' + err.message;
      enableCompare();
    }
    // Switched to from the bar in Compare, where the reader is looking.
    if (cmpOverride && !document.getElementById('cmpFlagLinks').hidden) {
      stPaintFlagLinks(err.message);
    }
    return;
  }
  if (ticket !== stEvidenceRequest) return;
  // `image` is false for a read with no page (an extract-only job's transcript).
  const reads = data.attempts.map((a, n) => Object.assign({}, a,
    { src: a.image === false ? null : base + '/' + n + '.png' }));
  // An error opens on the read that failed, which is the last.
  const first = kind === 'dry' ? reads.findIndex(r => r.dry)
              : kind === 'error' ? reads.length - 1
              : reads.findIndex(r => r.looped || r.truncated);
  cmpOverride = { alert: alertId, kind, case: data.case, page: data.page,
                  filename: data.filename || FX_CASES.find(c => c.id === (data.truth_case || data.case))?.pdf || data.case,
                  truthCase: data.truth_case, reads, at: Math.max(0, first),
                  fields: data.fields || null, host, job };
  // An extraction flag opens on the page beside its form, so a missing field
  // can be looked for on the image; on the transcript where no page was read.
  cmpView = !data.fields ? 'source-model'
    : reads.some(r => r.src) || window.StressLocalPreview?.hasCandidate(cmpOverride.filename)
      ? 'source-fields' : 'model-fields';
  stOpenEvidenceCompare(alertId, !openedLocal);
}
function stOpenEvidenceCompare(alertId, scroll = true) {
  paintMain();
  compareEl.hidden = false;
  cmpToggle.textContent = 'Hide';
  enableCompare();
  document.querySelectorAll('tr.st-alert-link').forEach(tr =>
    tr.classList.toggle('on', tr.dataset.alertId === String(alertId)));
  if (scroll) document.getElementById('compareCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
for (const host of [stOut, document.getElementById('historyDetail')]) {
  host.addEventListener('click', e => {
    const row = e.target.closest('tr.st-alert-link');
    if (row) stShowEvidence(row.dataset.alertId, row.dataset.alertKind, host,
                            row.dataset.alertJob);
  });
}

/* Every flag of the open window's file, to switch between (2026-10-27, at the
   user's request): a looped or failed read still goes on to extraction, so an
   OCR flag switches to that file's extraction and an extraction flag back to
   its reads. A file whose extraction raised no flag of its own has it kept
   under the job's key (`_StressLive.job_key`): the alert id with its number
   swapped for `job<n>`. Read off the rows of the box the flag was clicked in,
   so it works the same for the live run and a saved one. */
function stPaintFlagLinks(note) {
  // Looked up here: `paintCompare` calls this, and can run before this point.
  const cmpFlagLinks = document.getElementById('cmpFlagLinks');
  const o = cmpInStress() ? cmpOverride : null;
  const rows = o && o.host && o.job ? [...o.host.querySelectorAll('tr.st-alert-link')]
    .filter(tr => tr.dataset.alertJob === String(o.job)) : [];
  // The same flag can be drawn twice (a saved run in History and the live one).
  const seen = new Set();
  const links = rows.filter(tr => !seen.has(tr.dataset.alertId) && seen.add(tr.dataset.alertId))
    .map(tr => ({ id: tr.dataset.alertId, kind: tr.dataset.alertKind,
                  part: tr.dataset.alertPart, page: Number(tr.dataset.alertPage) || 0,
                  label: (tr.dataset.alertPart === 'ocr' ? 'OCR' : 'Extraction')
                    + (tr.dataset.alertPage ? ' p.' + tr.dataset.alertPage : '')
                    + ' · ' + tr.dataset.alertTag }))
    .sort((a, b) => (a.part === 'ocr' ? 0 : 1) - (b.part === 'ocr' ? 0 : 1) || a.page - b.page);
  const ocr = links.find(l => l.part === 'ocr');
  const box = rows.length && rows[0].closest('[data-st-mode]');
  // A saved run from before the mode was on the box: offer it, a 404 says why.
  const extracts = !box || !box.dataset.stMode || box.dataset.stMode === 'full';
  if (ocr && extracts && !links.some(l => l.part !== 'ocr')) {
    links.push({ id: String(ocr.id).replace(/\d+$/, '') + 'job' + o.job,
                 kind: 'extraction', label: 'Extraction · no flag' });
  }
  cmpFlagLinks.hidden = links.length < 2 && !note;
  if (cmpFlagLinks.hidden) return;
  cmpFlagLinks.innerHTML = '<span>This file (job #' + esc(String(o.job)) + '):</span>'
    + '<div class="cmp-modes">' + links.map(l => '<button class="'
      + (l.id === String(o.alert) ? 'on' : '') + '" data-go-id="' + esc(l.id)
      + '" data-go-kind="' + esc(l.kind) + '">' + esc(l.label) + '</button>').join('')
    + '</div>' + (note ? '<span style="color:var(--danger)">' + esc(note) + '</span>' : '');
}
document.getElementById('cmpFlagLinks').addEventListener('click', e => {
  const b = e.target.closest('button[data-go-id]');
  if (b && cmpOverride && !b.classList.contains('on')) {
    stShowEvidence(b.dataset.goId, b.dataset.goKind, cmpOverride.host, cmpOverride.job);
  }
});

function stPaint() {
  paintMain();
  if (!stState) {
    stOut.innerHTML = '<p class="note">Run a stress test to see its report here, or open History to review a saved run.</p>';
    return;
  }
  // Patch changed nodes while retaining existing rows and controls. Keep the
  // reader where they were: newest rows go in at the top, so
  // a box scrolled down is pushed by however much it grew, and one at the top
  // stays there to show the newest.
  // One box per part (`ST_ALERT_PARTS`), each kept where it was.
  const scrolled = {};
  for (const box of stOut.querySelectorAll('.st-alerts')) {
    scrolled[box.dataset.stPart] = [box.scrollTop, box.scrollHeight];
  }
  patchStressReport(stOut, stReportHtml(stState, null, stOut));
  for (const box of stOut.querySelectorAll('.st-alerts')) {
    const [oldTop, oldHeight] = scrolled[box.dataset.stPart] || [0, 0];
    if (oldTop > 0) box.scrollTop = oldTop + (box.scrollHeight - oldHeight);
  }
  stWireChart();
  // The open window's file may have raised more flags since, or finished extracting.
  if (cmpOverride && cmpOverride.host === stOut) stPaintFlagLinks();
  stWireChatContext(stOut, stOutlierReport(stState || {}));
  stWireOutliers(stOut, stState, stPaint);
}

function chatToken(value) {
  return value === null || value === undefined ? '—' : Number(value).toLocaleString();
}

function agenticChatTable(chats) {
  return '<div class="an-grid"><table class="queue chat-context-table"><tr>'
    + '<th>Document / Step</th><th>Request</th><th>Input</th><th>Output</th><th>Used total</th>'
    + '<th>Reply cap</th><th>Context limit</th><th>Used %</th><th>Status</th></tr>'
    + chats.map(chat => '<tr><td title="' + esc((chat.keys || []).join(', ')) + '">'
      + (chat.document ? '#' + chat.document + ' · ' : '') + esc(chat.step || '') + '</td>'
      + '<td class="n">' + chatToken(chat.request) + (chat.schema ? ' · schema' : '') + '</td>'
      + ['in_tokens', 'out_tokens', 'used_tokens', 'reply_cap'].map(key =>
        '<td class="n">' + chatToken(chat[key]) + '</td>').join('')
      + '<td class="n">' + chatToken(chat.context_limit)
      + (chat.context_source ? ' <small>(' + esc(chat.context_source) + ')</small>' : '') + '</td>'
      + '<td class="n">' + (chat.used_pct == null ? '—' : chatToken(chat.used_pct) + '%') + '</td>'
      + '<td title="' + esc(chat.error || '') + '">' + (chat.error ? 'Failed'
        : chat.http_status ? 'HTTP ' + chat.http_status : '—') + '</td></tr>').join('')
    + '</table></div>';
}

function stChatContextHtml(context) {
  const requests = context.requests || [];
  const jobs = new Map(requests.map(chat => [chat.job, chat.case]));
  const spread = d => d ? [d.min, d.mean, d.max].map(chatToken).join(' / ') : '—';
  return '<h3 class="st-h">Agentic chat context</h3>'
    + '<p class="note">Each step sends its own transcript and question, without a conversation '
    + 'history from previous steps. Input includes the cached prompt. Used total is input + output; '
    + 'Reply cap is the requested output budget, not the context window. Limits marked server '
    + 'come from model metadata; requested limits come from the request. Missing usage is —.</p>'
    + '<div class="an-grid"><table class="queue"><tr><th>Step</th><th>Chats</th>'
    + '<th>Input: min / mean / max</th><th>Output: min / mean / max</th>'
    + '<th>Peak used total</th><th>Context limit(s)</th><th>Peak used %</th></tr>'
    + (context.steps || []).map(step => '<tr><td title="' + esc((step.keys || []).join(', '))
      + '"><b>' + esc(step.step) + '</b><br>' + esc(step.title || '') + '</td>'
      + '<td class="n">' + step.requests + '</td><td class="n">' + spread(step.tokens.in)
      + '</td><td class="n">' + spread(step.tokens.out) + '</td><td class="n">'
      + chatToken(step.used.max) + '</td><td class="n">'
      + ((step.context_limits || []).map(chatToken).join(' / ') || '—')
      + '</td><td class="n">' + (step.used_pct.max == null ? '—'
        : chatToken(step.used_pct.max) + '%') + '</td></tr>').join('') + '</table></div>'
    + '<details class="chat-context-jobs"><summary>Inspect each job — ' + requests.length
    + ' chat request(s)</summary><label class="field">Job / file<select data-chat-job>'
    + [...jobs].map(([job, name]) => '<option value="' + job + '">#' + job + ' · '
      + esc(name || '') + '</option>').join('') + '</select></label><div data-chat-rows></div></details>';
}

function stWireChatContext(host, report) {
  const context = report && report.agentic_context;
  const select = host.querySelector('[data-chat-job]');
  if (!context || !select) return;
  const paint = () => {
    patchStressReport(host.querySelector('[data-chat-rows]'), agenticChatTable(
      context.requests.filter(chat => String(chat.job) === select.value)));
  };
  select.onchange = paint;
  paint();
}

// The report without the N longest jobs (`report.outliers[N - 1]`, computed by
// the server while it still had every job). The run-wide parts that are not
// per job -- the timeline, /metrics, batches, prepare time -- stay the full
// run's, and the per-chat list is the full one minus the excluded jobs.
function stOutlierReport(s) {
  const full = s.report;
  const n = s.excludeN || 0;
  const cut = full && n && (full.outliers || [])[n - 1];
  if (!cut) return full;
  const gone = new Set(cut.excluded.map(e => e.job));
  const ctx = cut.agentic_context && full.agentic_context
    ? {...cut.agentic_context, requests: (full.agentic_context.requests || [])
        .filter(chat => !gone.has(chat.job))} : cut.agentic_context;
  return {...cut, agentic_context: ctx, concurrency: full.concurrency, timeline: full.timeline,
    timeline_columns: full.timeline_columns, prepare_seconds: full.prepare_seconds,
    vllm_batch: full.vllm_batch, servers_metrics: full.servers_metrics,
    server_metrics: full.server_metrics};
}

function stOutlierPicker(s) {
  const options = (s.report.outliers || []);
  if (!options.length) return '';
  const n = s.excludeN || 0;
  const cut = n ? options[n - 1] : null;
  return '<div class="st-outliers"><label class="field">Exclude outliers'
    + '<select data-st-exclude title="Recompute the report as if the slowest jobs (end to end) had never run">'
    + '<option value="0">None: every run</option>'
    + options.map((o, i) => '<option value="' + (i + 1) + '"' + (n === i + 1 ? ' selected' : '')
      + '>The ' + (i + 1) + ' longest run' + (i ? 's' : '') + '</option>').join('')
    + '</select></label>'
    + (cut ? '<p class="note">Left out: ' + cut.excluded.map(e => '#' + e.job + ' '
        + esc(e.case || '') + ' (' + stSec(e.total_s)
        + (e.status !== 'ok' ? ', ' + esc(e.status) : '') + ')').join(' · ')
      + '. They leave every count and figure, error rate included; the wall clock ends with the '
      + 'last job kept. The chart and /metrics below are still the whole run.</p>' : '')
    + '</div>';
}

// The picker re-draws the report it sits in; `rerender` is how this host draws it.
function stWireOutliers(host, s, rerender) {
  const select = host.querySelector('[data-st-exclude]');
  if (!select) return;
  select.onchange = () => {
    s.excludeN = Number(select.value) || 0;
    rerender();
  };
}

function stReportHtml(s, snapshot, host) {
  let html = '';
  if (s.detailDownload) {
    html += '<p class="note"><a class="ghost-sm" href="' + esc(s.detailDownload.url)
      + '" download="' + esc(s.detailDownload.name) + '">Download detail log</a> · '
      + s.detailDownload.count + ' job(s) · kept in this browser tab only</p>';
  }
  if (s.monitor_samples && s.monitor_samples.length) {
    html += historyMonitorHtml(s.monitor_samples, s.monitor_samples_total);
  }
  const plan = s.plan;
  if (plan) {
    html += '<p class="st-plan">' + esc(plan.mode === 'full' ? 'Full pipeline'
        : plan.mode === 'ocr' ? 'OCR only' : plan.mode === 'extract' ? 'Extract only'
        : plan.mode === 'image' ? 'Image to extractor (no OCR)'
        : 'Classify ' + plan.pack_pages + '-page files')
      + ' · ' + plan.concurrency + ' concurrent · ' + plan.jobs + ' job(s)'
      + (plan.warmup ? ' + ' + plan.warmup + ' warm-up' : '')
      + ' · ' + esc(plan.server.kind || '') + ' ' + esc(plan.server.url || '')
      + (plan.reader ? ' · reads with <b>' + esc(csModel(plan.reader)) + '</b>' : '')
      + (plan.extractor ? ' · extracts with <b>' + esc(csModel(plan.extractor)) + '</b>' : '')
      + (plan.detail ? ' · ' + esc(plan.detail) : '')
      + (plan.extract_mode ? ' · ' + esc(plan.extract_mode) : '')
      + (plan.extras === false ? ' · <b>no extra fields</b>' : '')
      + (plan.mode === 'image' && plan.extractor_vision === null
        ? ' · the extraction server does not say whether the model takes images' : '')
      + (plan.metrics ? ' · sampling vLLM /metrics on '
          + (plan.metrics_servers || []).map(t => esc(t.role) + ' ' + esc(t.url)).join(' and ') : '')
      + (plan.no_cache ? ' · prompt cache bypassed ('
          + Object.entries(plan.no_cache_effect || {})
              .map(([k, v]) => esc(k) + ': ' + esc(v)).join('; ') + ')' : '')
      + (plan.split_pages ? ' · long files spread on idle workers' : '')
      + (plan.mode === 'full' ? (plan.fill_missing
          ? ' · missing required fields asked of the page images (Hybrid)'
          : ' · <b>no fill from images</b>') : '')
      + (plan.validation_retry ? ' · validation findings rechecked from images (Hybrid)' : '')
      + (plan.vllm_batch ? ' · <b>batched</b>, up to ' + plan.batch_size
          + ' per /batch request on ' + (plan.batch_servers || []).map(esc).join(' and ') : '')
      + (plan.save_log ? ' · saving each job to the run log as stress run <b>'
          + esc(plan.run_id) + '</b>' : '')
      + ' · seed ' + plan.seed + '</p>';
    if (plan.cases) html += '<details><summary>Files selected (' + plan.cases.length
      + ')</summary><p class="note">' + plan.cases.map(esc).join(', ') + '</p></details>';
  }
  html += stAlertsHtml(s.alerts || [], plan && plan.mode);
  const p = s.progress;
  if (p && !s.report) {
    const frac = p.total ? p.done / p.total : 0;
    const st = p.stages || {};
    html += '<div class="st-progress"><div class="st-bar"><i style="width:'
      + (frac * 100).toFixed(1) + '%"></i></div><div class="st-live">'
      + '<b>' + (p.phase === 'warmup' ? 'Warming up' : 'Running') + '</b> · '
      + p.done + ' of ' + p.total + ' done · ' + stSec(p.elapsed)
      + ' · queued ' + (st.queued || 0)
      + (st.ocr ? ' · reading ' + st.ocr : '') + (st.extract ? ' · extracting ' + st.extract : '')
      + (st.segment ? ' · splitting ' + st.segment : '')
      + (st.classify ? ' · classifying ' + st.classify : '')
      + (p.failed ? ' · <span style="color:var(--danger)">' + p.failed + ' failed</span>' : '')
      + ((s.alerts || []).length ? ' · <span style="color:var(--warn)">'
        + stAlertCounts(s.alerts).map(([title, n]) => n + ' ' + title).join(', ')
        + ' flagged</span>' : '')
      + (p.pages ? ' · ' + stNum(p.pages_per_second, 2) + ' pages/s · '
        + stNum(p.tokens_per_second, 0) + ' tok/s' : '')
      + '</div>' + stLiveScore(p.score) + '</div>';
    if (s.metricsNow) {
      // The last sample of each server, kept between ticks: only every Nth
      // progress event carries one, and drawing only those made the line flicker.
      const many = s.metricsNow.length > 1;
      for (const srv of s.metricsNow) {
        const m = srv.view;
        html += '<p class="note">vLLM ' + (many ? esc(srv.role) + ' ' : '') + 'now: '
          + stNum(m.running, 0) + ' running, ' + stNum(m.waiting, 0) + ' waiting'
          + (m.kv_cache !== null && m.kv_cache !== undefined ? ', KV cache ' + stPct(m.kv_cache) : '')
          + (m.generation_tokens_per_s ? ', ' + stNum(m.generation_tokens_per_s, 0)
            + ' gen tok/s since start' : '') + '.</p>';
      }
    }
    if (p.metrics_error) html += '<p class="note err">/metrics: ' + esc(p.metrics_error) + '</p>';
  }
  const r = stOutlierReport(s);
  if (r) {
    html += stOutlierPicker(s);
    const c = r.counts || {};
    const tiles = [
      stTile('Completed', r.completed + '<small> / ' + r.attempted + '</small>',
        [c.looped ? c.looped + ' looped' : '', c.truncated ? c.truncated + ' at token cap' : '',
         c.cancelled ? c.cancelled + ' cancelled' : '', c.skipped ? c.skipped + ' past the time limit' : '',
         r.ocr && r.ocr.pages_lent ? r.ocr.pages_lent + ' page(s) read on idle workers' : '',
         r.ocr && r.ocr.loop_retries ? r.ocr.loop_retries + ' loop retr' + (r.ocr.loop_retries === 1 ? 'y' : 'ies')
           + ', ' + (r.ocr.loop_recovered || 0) + ' page(s) recovered' : '',
         r.ocr && r.ocr.dry_checked ? 'DRY fired on ' + (r.ocr.dry_pages || 0) + ' of '
           + r.ocr.dry_checked + ' page(s)' : '',
         r.ocr && r.ocr.dry_retries ? r.ocr.dry_retries + ' DRY retr' + (r.ocr.dry_retries === 1 ? 'y' : 'ies')
           + ', ' + (r.ocr.dry_recovered || 0) + ' page(s) cleared' : '',
         r.image && r.image.retries ? r.image.retries + ' job(s) extracted again one Detail up, '
           + (r.image.recovered || 0) + ' came back whole' : '']
          .filter(Boolean).join(' · ')),
      stTile('Error rate', stPct(r.error_rate), (c.error || 0) + ' error(s)',
        r.error_rate ? 'st-bad' : ''),
      stTile('Wall clock', stSec(r.wall_seconds), 'prepare ' + stSec(r.prepare_seconds) + ', not timed'),
      stTile('Throughput', stNum(r.jobs_per_minute, 2) + '<small> jobs/min</small>',
        r.ocr ? stNum(r.ocr.pages_per_second, 3) + ' pages/s'
          : r.image ? stNum(r.image.pages_per_second, 3) + ' pages/s sent as images'
          : r.classify ? stNum(r.classify.pages_per_second, 2) + ' pages/s' : ''),
    ];
    if (r.vllm_batch) tiles.push(stTile('Batches', stNum(r.vllm_batch.batches, 0)
      + '<small> sent</small>', 'mean ' + stNum(r.vllm_batch.mean_size, 1) + ' · largest '
      + r.vllm_batch.max_size + ' of ' + r.vllm_batch.size + ' requests per batch'
      + (r.vllm_batch.failed ? ' · ' + r.vllm_batch.failed + ' failed' : ''),
      r.vllm_batch.failed ? 'st-bad' : ''));
    if (r.ocr) tiles.push(stTile('OCR tokens / s', stNum(r.ocr.tokens_per_second, 1),
      'all requests together; one request decodes at '
      + stNum((r.ocr.decode_tps || {}).p50, 0) + ' (median)'));
    if (r.extract) tiles.push(stTile('Extract tokens / s', stNum(r.extract.tokens_per_second, 1),
      stNum(r.extract.tokens, 0) + ' tokens'));
    if (r.ocr) tiles.push(stTile('Transcript score', stPct(r.ocr.char_accuracy),
      'mean character accuracy of ' + r.ocr.char_scored + ' clean read(s)'));
    if (r.extract) tiles.push(stTile('Field score', stPct(r.extract.field_accuracy),
      'mean of ' + r.extract.field_scored + ' job(s) · pooled ' + stPct(r.extract.field_pooled)
      + ' of ' + r.extract.field_values + ' values'
      + (r.extract.unscored ? ' · ' + r.extract.unscored + ' under the read floor' : '')
      + (r.extract.wrong_fields ? ' · ' + r.extract.wrong_fields + ' wrong field(s) in '
        + r.extract.wrong_field_jobs + ' job(s)' : ''),
      r.extract.wrong_fields ? 'st-warn' : ''));
    if (r.extract && r.extract.missing_required_jobs !== undefined) tiles.push(stTile(
      'Missing required', r.extract.missing_required_jobs + '<small> job(s)</small>',
      r.extract.missing_required_fields + ' required field(s) came back empty'
        + (r.extract.fill_jobs ? ' · ' + r.extract.fill_filled + ' of ' + r.extract.fill_asked
          + ' filled from the page images (Hybrid)' : ''),
      r.extract.missing_required_jobs ? 'st-warn' : ''));
    if (r.extract?.validation_retry_jobs) tiles.push(stTile('Validation image recheck',
      r.extract.validation_retry_changed + '<small> field(s) changed</small>',
      r.extract.validation_retry_jobs + ' job(s) · ' + r.extract.validation_retry_asked + ' fields asked'
      + ' · ' + r.extract.validation_retry_accepted + ' document(s) accepted'
      + ' · ' + r.extract.validation_retry_resolved + ' resolved'
      + ' · ' + stSec(r.extract.validation_retry_s) + ' additional work'));
    // Document sets: one file can hold several documents, each extracted on
    // its own (stress.doc_sets).
    const ds = r.extract && r.extract.documents;
    if (ds && ds.files) tiles.push(stTile('Document sets',
      ds.extracted + '<small> / ' + ds.found + ' extracted</small>',
      [ds.found + ' found in ' + ds.files + ' file(s)',
       ds.multi_files ? ds.multi_files + ' file(s) held more than one' : '',
       (ds.per_file || {}).max > 1 ? 'up to ' + ds.per_file.max + ' per file' : '',
       ds.failed ? ds.failed + ' failed' : '',
       ds.split_checked ? 'split matched the manifest on ' + ds.split_matched + ' of '
         + ds.split_checked + ' file(s) (' + ds.expected + ' expected)' : '']
        .filter(Boolean).join(' · '),
      ds.failed ? 'st-bad' : ds.split_checked && ds.split_matched < ds.split_checked ? 'st-warn' : ''));
    if (r.classify) {
      tiles.push(stTile('Grouped exactly', stPct(r.classify.grouping_exact),
        r.classify.documents_found + ' found of ' + r.classify.documents_expected + ' documents'));
      tiles.push(stTile('Typed correctly', stPct(r.classify.types_exact),
        'same form ' + stPct(r.classify.form_match) + ' · of the exactly grouped'));
      tiles.push(stTile('Walk questions', stNum(r.classify.asks_per_100_pages, 1)
        + '<small> / 100 pages</small>', r.classify.asks + ' asked · '
        + r.classify.refused + ' refused · ' + r.classify.model_classified + ' typed by the model'));
    }
    html += '<div class="st-tiles">' + tiles.join('') + '</div>';
    if (r.stage_share && r.stage_share.ocr !== null) {
      html += '<p class="note">Of the time jobs spent working, <b>' + stPct(r.stage_share.ocr)
        + '</b> was reading and <b>' + stPct(r.stage_share.extract) + '</b> was extracting.</p>';
    }
    html += stChart(r, host);
    const lat = r.latency || {};
    const latency = stLatencyRows([
      ['Waiting in the queue', lat.queue_wait, 'time a job waited for a free worker on this side'],
      ['End to end', lat.end_to_end, 'one whole job, from starting work to finishing'],
      ...(r.ocr ? [
        ['OCR, per job', r.ocr.latency_job, 'every page of the document read'],
        ['OCR, per page', r.ocr.latency_page, 'one page request'],
        ['Time to first token', r.ocr.ttft, 'measured here, so it includes time the request waited in the server queue'],
        ['Decode tokens/s, per request', r.ocr.decode_tps, 'one request own decode rate', 'tps'],
      ] : []),
      ...(r.extract ? [['Extraction, per job', r.extract.latency_job, r.image
        ? 'every request of the page images, a retry one Detail up included'
        : 'pass 2, all of its requests']] : []),
      ...(r.classify ? [['Splitting into documents', r.classify.latency_segment, 'the page walk, including every question asked'],
                        ['Classifying the documents', r.classify.latency_classify, 'one type per document found']] : []),
    ]);
    if (latency) html += '<h3 class="st-h">Latency</h3>' + latency;
    // Tokens sent (in) and generated (out): per OCR page, and per file for
    // pass 2 -- every request of it, classify included. In classify mode the
    // file is the glued pack and its requests are the walk and classify questions.
    const shape = (s.plan && s.plan.extract_mode) || '';
    const tokenRows = [];
    if (r.ocr && r.ocr.tokens_per_page) tokenRows.push(['OCR, per page', r.ocr.tokens_per_page]);
    if (r.extract_tokens_per_file) tokenRows.push([
      r.mode === 'classify' ? 'Extraction, per file (split + classify)'
        : 'Extraction, per file' + (shape ? ' (' + shape + ', classify included)' : ''),
      r.extract_tokens_per_file]);
    const tokCell = (d, k) => d && d[k] ? stNum(d[k].min, 0) + ' / ' + stNum(d[k].mean, 0)
      + ' / ' + stNum(d[k].max, 0) + ' <small>(' + d[k].n + ')</small>' : '—';
    if (tokenRows.length) {
      html += '<h3 class="st-h">Tokens</h3><table class="queue"><tr><th></th>'
        + '<th title="prompt tokens sent, cached part included">In: min / mean / max</th>'
        + '<th title="tokens generated">Out: min / mean / max</th></tr>'
        + tokenRows.map(([label, d]) => '<tr><td>' + esc(label) + '</td><td class="n">'
          + tokCell(d, 'in') + '</td><td class="n">' + tokCell(d, 'out') + '</td></tr>').join('')
        + '</table>';
    }
    if (r.agentic_context) html += stChatContextHtml(r.agentic_context);
    else if (plan && isStepMode(plan.extract_mode)) {
      html += '<p class="note">Per-chat context usage was not recorded for this run.</p>';
    }
    if (r.errors && r.errors.length) {
      html += '<h3 class="st-h">Errors</h3><table class="queue">' + r.errors.map(e =>
        '<tr><td class="n">' + e.count + '×</td><td>' + esc(e.error) + '</td></tr>').join('') + '</table>';
    }
    const runServers = r.servers_metrics
      || (r.server_metrics ? [{ role: '', url: '', view: r.server_metrics }] : []);
    for (const srv of runServers) {
      html += stMetricsTable(srv.view, 'vLLM /metrics during the run'
        + (runServers.length > 1 ? ' · ' + srv.role + ' · ' + srv.url : ''), ' (this run)');
    }
  }
  if (s.draining) {
    html += '<p class="note err">Stopped. ' + s.draining + ' request(s) are still running '
      + 'on the server; another test can start when they finish.</p>';
  }
  for (const snap of snapshot || []) {
    html += stMetricsTable(snap.data.view, 'vLLM /metrics snapshot · '
      + (snapshot.length > 1 ? snap.label + ' · ' + snap.data.url + ' · ' : '')
      + new Date(snap.data.at * 1000).toLocaleTimeString(), ' (since server start)');
  }
  return html;
}

function stBody() {
  const val = id => stEl(id).value;
  return {
    mode: stMode.value, concurrency: Number(val('stConc')) || 1,
    jobs: Number(val('stJobs')) || 1, warmup: Number(val('stWarm')) || 0,
    max_seconds: Number(val('stMax')) || 0, detail: val('stDetail'),
    extract_mode: val('stShape'), extras: stEl('stExtras').checked,
    schema_first: stEl('stSchemaFirst').checked && ST_EXTRACT_MODES.includes(stMode.value)
      && isStepMode(val('stShape')),
    pack_pages: Number(val('stPages')) || 100,
    strategy: val('stStrategy'), seed: val('stSeed').trim() || null,
    source: stSource,
    // realData: Random sends only the types, so every file of them is drawn
    // from; Choose files sends the files as well.
    types: stSource === 'realdata' ? stRealTypes() : undefined,
    cases: stSource !== 'realdata' ? stSelectedCases()
      : stRealPick === 'choose' ? stRealSelected() : undefined,
    metrics: false,
    no_cache: stEl('stNoCache').checked,
    split_pages: stEl('stSplitPages').checked,
    fill_missing: stFillOn && stMode.value === 'full',
    validation_retry: !!stEl('stValidationImages')?.checked && stMode.value === 'full',
    vllm_batch: stBatchOn && !stEl('stBatch').disabled,
    save_log: stEl('stSaveLog').checked && !['classify', 'image'].includes(stMode.value),
    save_details: stEl('stSaveDetails').checked && ST_EXTRACT_MODES.includes(stMode.value),
    history_monitor: stEl('stHistoryMonitor').checked && !stEl('stHistoryMonitor').disabled,
    gpu_url: stEl('monGpu').value.trim(),
    gpu_extract_url: stEl('monGpuExtract').value.trim(),
    metrics_every: Math.max(1, Math.min(60, Number(val('monEvery')) || 2))
  };
}

async function stStart() {
  // Read the form before busy disables the snapshot option.
  const body = stBody();
  if (!stSelectedCount()) {
    stNote.textContent = 'Select at least one file to stress test.';
    stNote.classList.add('err');
    return;
  }
  stClearDetailLog();
  stState = { draining: 0, alerts: [], received: 0 };
  stNote.textContent = 'Preparing…';
  stNote.classList.remove('err');
  await stWatch(signal => fetch(API + '/stress/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    signal, body: JSON.stringify(body)
  }));
}

/* The run belongs to the server, not to this page (2026-10-27): it goes on in
   the background whoever is watching, keeps every line it sends, and names
   itself in `X-Stress-Run`. So a dropped connection is followed again from the
   line it stopped at, and a page that reloads picks the run up where it is
   (`stResume`). Stop is the only thing that ends a run early. */
const ST_REATTACH_TRIES = 30;
const ST_SEEN_KEY = 'stSeenRun';
function stSeenRun() {
  try { return sessionStorage.getItem(ST_SEEN_KEY); } catch { return null; }
}
function stMarkSeen(id) {
  try { if (id) sessionStorage.setItem(ST_SEEN_KEY, id); } catch { /* private window */ }
}

// Read one response of a run into stState: 'done' when its report came,
// 'ended' when the server closed it without one. A dropped connection throws
// an error marked `lost`; anything else thrown here is a real failure.
async function stReadStream(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', lastPaint = 0, reported = false;
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (err) {
      if (err.name !== 'AbortError') err.lost = true;
      throw err;
    }
    const { value, done } = chunk;
    if (done) return reported ? 'done' : 'ended';
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      stState.received++;
      if (['detail_log_start', 'detail_log_job', 'detail_log_end'].includes(ev.event)) {
        stReceiveDetailLog(ev);
        stNote.textContent = 'Preparing detail log — ' + stDetailLog.count + ' job(s)';
        continue;
      }
      if (ev.history_id) stState.historyId = ev.history_id;
      if (ev.history_error) stState.historyError = ev.history_error;
      if (ev.event === 'plan') {
        stState.plan = ev;
        stEl('stSeed').value = ev.seed;
        stNote.textContent = 'Seed ' + ev.seed + '.';
      } else if (ev.event === 'preparing') {
        stNote.textContent = 'Preparing ' + ev.done + ' of ' + ev.total
          + (stMode.value === 'classify' ? ' file(s)…' : ' document(s)…');
      } else if (ev.event === 'progress') {
        stState.progress = ev;
        if (ev.servers_metrics) stState.metricsNow = ev.servers_metrics;
        stNote.textContent = (ev.phase === 'warmup' ? 'Warming up — ' : 'Running — ')
          + ev.done + '/' + ev.total;
      } else if (ev.event === 'done') {
        stFinishDetailLog(ev);
        setTestStatus('st', ev.stopped ? 'stopped' : 'completed');
        stState.report = ev.report;
        stState.monitor_samples = ev.monitor_samples;
        stState.monitor_samples_total = ev.monitor_samples_total;
        stState.draining = ev.draining || 0;
        stNote.textContent = (ev.stopped ? 'Stopped' : 'Finished') + ' — seed ' + ev.seed
          + (ev.history_error ? '. ' + ev.history_error
            : ev.history_id ? '. Saved to History.' : '. No History record was saved.');
        stNote.classList.toggle('err', !!ev.history_error);
        if (stState.plan && stState.plan.save_log) stNote.textContent += " Each measured job was saved to the run log.";
        stMarkSeen(stState.runId);
        reported = true;
      } else if (ev.event === 'alert') {
        // The same id again is the same issue, updated (a loop retried and
        // resolved, or not): replace its row rather than add one.
        const at = ev.id ? stState.alerts.findIndex(a => a.id === ev.id) : -1;
        if (at >= 0) stState.alerts[at] = ev; else stState.alerts.push(ev);
        lastPaint = 0;
      } else if (ev.event === 'draining') {
        stState.draining = ev.in_flight;
      } else if (ev.event === 'error') {
        stMarkSeen(stState.runId);
        throw new Error(ev.error);
      }
    }
    const now = Date.now();
    if (now - lastPaint > 300) { lastPaint = now; stPaint(); }
  }
}

// Follow a run to its report: `open(signal)` is the first response; a
// connection that drops is followed again, from the next line, while the run
// goes on on the server.
async function stWatch(open) {
  stAbort = new AbortController();
  const signal = stAbort.signal;
  stBusy(true);
  stPaint();
  try {
    let res = await open(signal), lost = 0;
    for (;;) {
      if (res) {
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || ('HTTP ' + res.status));
        }
        stState.runId = res.headers.get('X-Stress-Run') || stState.runId;
        const before = stState.received;
        let outcome = 'lost';
        try {
          outcome = await stReadStream(res);
        } catch (err) {
          if (!err.lost || !stState.runId) throw err;
        }
        if (outcome === 'done') break;
        if (outcome === 'ended') throw new Error('The test ended without a report.');
        if (stState.received > before) lost = 0;
      }
      if (!stState.runId || ++lost > ST_REATTACH_TRIES) {
        throw new Error('Lost the connection to the test. It is still running on the server: '
          + 'reload the page to follow it again.');
      }
      stNote.textContent = 'Connection lost — the test goes on on the server. Reconnecting…';
      stNote.classList.add('err');
      await new Promise(resolve => setTimeout(resolve, Math.min(10000, 1000 * lost)));
      if (signal.aborted) throw new DOMException('Hung up', 'AbortError');
      res = await fetch(API + '/stress/attach?id=' + encodeURIComponent(stState.runId)
        + '&from=' + stState.received, { signal })
        .catch(err => { if (err.name === 'AbortError') throw err; return null; });
      if (res && res.ok) stNote.classList.remove('err');
    }
  } catch (err) {
    stFinishDetailLog({event: 'error', error: err.message});
    setTestStatus('st', 'failed');
    stNote.textContent = err.name === 'AbortError'
      ? 'Hung up on the test; the server may still be running it — reload the page to follow it again.'
      : err.message;
    stNote.classList.add('err');
  } finally {
    if (['running', 'stopping'].includes(stEl('stStatus').dataset.state))
      setTestStatus('st', 'failed');
    if (stState) stState.draining = 0;
    stAbort = null;
    stBusy(false);
    stPaint();
  }
}

// On load: this tab's run, if one is still going on the server -- or one that
// finished while the page was away and whose report it has not shown yet.
async function stResume() {
  let run = null;
  try {
    const res = await fetch(API + '/stress/active');
    if (res.ok) run = (await res.json()).run;
  } catch { return; }
  if (!run || stRunning || stAbort || (!run.running && stSeenRun() === run.id)) return;
  if (ST_MODES.includes(run.mode) && run.mode !== stMode.value) stLoadModeDefaults(run.mode);
  stClearDetailLog();
  stState = { draining: 0, alerts: [], received: 0, runId: run.id };
  stNote.textContent = run.running ? 'Reconnecting to the stress test running on the server…'
    : 'Loading the stress test that finished while this page was away…';
  stNote.classList.remove('err');
  await stWatch(signal => fetch(API + '/stress/attach?id=' + encodeURIComponent(run.id)
    + '&from=0', { signal }));
}

if (stRun) {
  stEl('stFiles').addEventListener('change', stPaintFiles);
  const scrollToSelection = () => {
    if (stEl('stFilePicker').hidden) return;
    const list = stEl('stFiles');
    const first = stFileBoxes().find(box => stEligibleFile(box) && box.checked);
    if (first) list.scrollTop += first.parentElement.getBoundingClientRect().top - list.getBoundingClientRect().top;
  };
  stEl('stFileAdvanced').addEventListener('click', () => {
    const panel = stEl('stFilePicker'), button = stEl('stFileAdvanced');
    panel.hidden = !panel.hidden;
    button.setAttribute('aria-expanded', String(!panel.hidden));
    button.classList.toggle('on', !panel.hidden);
    scrollToSelection();
  });
  stEl('stFileSets').addEventListener('click', event => {
    const button = event.target.closest('button[data-st-file-set]');
    if (!button || button.disabled) return;
    for (const box of stFileBoxes()) box.checked = stFileInSet(box, button.dataset.stFileSet);
    stPaintFiles();
    scrollToSelection();
  });
  stDefaultBase = stCaptureDefaults();
  stMode.addEventListener('change', () => stLoadModeDefaults(stMode.value));
  stEl('stSaveDefaults').addEventListener('click', stSaveModeDefaults);
  stEl('stShape').addEventListener('change', stPaintMode);
  stEl('stSource').addEventListener('click', event => {
    const button = event.target.closest('button[data-st-source]');
    if (button && !button.disabled && !stRunning) stSetSource(button.dataset.stSource);
  });
  stEl('stRealPick').addEventListener('click', event => {
    const button = event.target.closest('button[data-st-real-pick]');
    if (!button || button.disabled) return;
    stRealPick = button.dataset.stRealPick;
    stPaintFiles();
  });
  stEl('stRealTypes').addEventListener('change', stPaintFiles);
  stEl('stRealFiles').addEventListener('change', stPaintFiles);
  stEl('stRealRefresh').addEventListener('click', () => stLoadReal());
  stEl('stBatch').addEventListener('click', () => { stBatchOn = !stBatchOn; stPaintBatch(); });
  stEl('stFillImages').addEventListener('click', () => { stFillOn = !stFillOn; stPaintFill(); });
  stEl('stConc').addEventListener('input', stPaintBatch);
  stRun.addEventListener('click', () => stStart());
  const stop = async () => {
    if (!stAbort || stStopping) return;
    stStopping = true;
    setTestStatus('st', 'stopping');
    stStop.disabled = stStop2.disabled = true;
    stNote.textContent = 'Stopping — no new job starts, reads in flight are hung up…';
    try {
      const res = await fetch(API + '/stress/stop', { method: 'POST' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
    } catch { if (stAbort) stAbort.abort(); }
  };
  stStop.addEventListener('click', stop);
  stStop2.addEventListener('click', stop);
  stClear.addEventListener('click', () => {
    if (stAbort) return;
    stClearDetailLog();
    stState = null; stNote.textContent = '';
    stNote.classList.remove('err');
    setTestStatus('st', 'ready');
    stPaint();
  });
  const initialMode = stReadDefaultStore().lastMode;
  stLoadModeDefaults(ST_MODES.includes(initialMode) ? initialMode : stMode.value);
  if (stSource !== 'realdata') stLoadReal(); // realData defaults already start the listing
  stPaintServer();
  stResume();
}

/* ---------- live /metrics monitor ----------
   Polls GET /api/monitor while it is switched on and the page is visible, one
   request at a time (the next is scheduled when the last has answered). Rates
   come from the server, which diffs its own last two scrapes; the page only
   keeps a short history for the sparklines. */
var monState = { on: false, panes: {}, minEvery: 2, idleMax: 30 };
const MON_HIST = 300;
const monToggleBtn = stEl('monToggle'), monEveryInp = stEl('monEvery');
const monGpuInp = stEl('monGpu'), monGpuExInp = stEl('monGpuExtract'), monOut = stEl('stMon');

function monStore(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* private window */ }
}
try {
  const g = localStorage.getItem('monGpu'), e = localStorage.getItem('monEvery');
  const gx = localStorage.getItem('monGpuExtract');
  if (g) monGpuInp.value = g;
  if (gx) monGpuExInp.value = gx;
  if (e) monEveryInp.value = e;
} catch (e) { /* storage unavailable: defaults stand */ }

/* The servers to watch, each that says it is vLLM: the OCR (reading) server
   and the extraction server, side by side. One pane where both passes share a
   URL. A function declaration, so `stPaintServer` can call it at parse time. */
function monTargets() {
  const sep = !!(extractInfo && extractInfo.separate);
  const out = [];
  if (server && server.kind === 'vllm') {
    out.push({ key: 'ocr', label: sep ? 'OCR server' : 'OCR + extraction server', url: server.url });
  }
  if (sep && extractInfo.kind === 'vllm') {
    out.push({ key: 'extract', label: 'Extraction server', url: extractInfo.url });
  }
  return out;
}

/* The box's value, never below the server's MONITOR_MIN_INTERVAL: asking
   faster only gets the cached reading back. */
function monEvery() {
  const asked = Math.max(1, Math.min(Number(monEveryInp.value) || 2, 60));
  return Math.max(asked, monState.minEvery || 0);
}

/* Idle or failing readings double the gap up to the server's MONITOR_IDLE_MAX;
   any sign of work puts it straight back to the box's value. */
function monIdle(d) {
  const s = (d && d.snapshot) || {}, r = (d && d.rates) || {};
  return !(s.running > 0 || s.waiting > 0 || r.generation_tps > 0
           || r.prompt_tps > 0 || r.requests_per_s > 0);
}

/* Each server is polled on its own clock, so a slow or idle one never holds up
   the other: its own backoff, its own one-request-at-a-time. */
function monSchedule(p) {
  clearTimeout(p.timer);
  if (!monState.on || monState.panes[p.key] !== p) return;
  const base = monEvery(), cap = Math.max(base, monState.idleMax || 0);
  p.delay = p.idleRuns > 1 && monState.idleMax > 0
    ? Math.min(cap, base * Math.pow(2, p.idleRuns - 1)) : base;
  p.timer = setTimeout(() => monTick(p), p.delay * 1000);
}

async function monTick(p) {
  if (!monState.on || p.busy || monState.panes[p.key] !== p) return;
  if (document.hidden) { monSchedule(p); return; }
  p.busy = true;
  try {
    const gpu = (p.key === 'extract' ? monGpuExInp : monGpuInp).value.trim();
    const res = await fetch(API + '/monitor?server=' + p.key
      + (gpu ? '&gpu=' + encodeURIComponent(gpu) : ''));
    const d = await res.json();
    if (d.min_interval !== undefined) monState.minEvery = d.min_interval;
    if (d.idle_max !== undefined) monState.idleMax = d.idle_max;
    if (!res.ok) throw new Error(d.error || ('HTTP ' + res.status));
    const s = d.snapshot || {}, r = d.rates || {}, g = d.gpu;
    // Another tab scraped a moment ago: the same reading, so no new point.
    const repeat = p.last && p.last.at === d.at;
    p.idleRuns = monIdle(d) ? p.idleRuns + 1 : 0;
    p.last = d;
    p.error = '';
    if (!repeat) p.hist.push({
      t: d.at, kv_free: s.kv_free, running: s.running, waiting: s.waiting,
      vram_free: g ? g.free_mb / 1024 : null,
      gen: r.generation_tps, prompt: r.prompt_tps, hit: r.prefix_hit_rate,
      rps: r.requests_per_s, mm: r.mm_hit_rate, batch: r.batch,
      fits: d.fit ? d.fit.more_now : null
    });
    if (p.hist.length > MON_HIST) p.hist.shift();
  } catch (err) {
    p.error = err.message;
    p.idleRuns += 1;
  } finally {
    p.busy = false;
  }
  paintMon();
  monSchedule(p);
}

/* Brings the panes in line with the servers now set: a server switched away
   from loses its pane, a new one gets a fresh one, read at once. */
function monSync() {
  const want = monState.on ? monTargets() : [];
  for (const key of Object.keys(monState.panes)) {
    const p = monState.panes[key], t = want.find(w => w.key === key);
    if (!t || t.url !== p.url) { clearTimeout(p.timer); delete monState.panes[key]; }
  }
  for (const t of want) {
    if (monState.panes[t.key]) { monState.panes[t.key].label = t.label; continue; }
    const p = monState.panes[t.key] = { ...t, hist: [], last: null, error: '',
                                         timer: null, delay: 0, idleRuns: 0, busy: false };
    monTick(p);
  }
}

function monEach(fn) { Object.values(monState.panes).forEach(fn); }

function monSet(on) {
  monState.on = on;
  monEach(p => clearTimeout(p.timer));
  monToggleBtn.textContent = on ? 'Stop live monitor' : 'Start live monitor';
  monToggleBtn.classList.toggle('on', on);
  monToggleBtn.setAttribute('aria-pressed', String(on));
  monToggleBtn.setAttribute('aria-expanded', String(on));
  stEl('monCard').hidden = !on;
  if (on) {
    monState.panes = {};
    monSync();
  }
  paintMon();
}

/* One series per tile, so no legend: the tile label names it. The line is the
   last readings left to right; a gap is a reading that had no value. */
function monSpark(hist, key, fixedMax) {
  const vals = hist.map(h => h[key]);
  const nums = vals.filter(v => v !== null && v !== undefined);
  if (nums.length < 2) return '<div class="mon-spark mon-spark-empty"></div>';
  const max = fixedMax || Math.max(...nums) || 1;
  const W = 120, H = 30, n = vals.length;
  const x = i => (n === 1 ? W : i * W / (n - 1));
  const y = v => H - 2 - (H - 4) * Math.max(0, Math.min(v / max, 1));
  const segs = [];
  let cur = [];
  vals.forEach((v, i) => {
    if (v === null || v === undefined) { if (cur.length) segs.push(cur); cur = []; return; }
    cur.push(x(i).toFixed(1) + ',' + y(v).toFixed(1));
  });
  if (cur.length) segs.push(cur);
  // A reading with empty readings either side is drawn as a short dash, or it
  // would not show at all (a one-point line has no length).
  const lines = segs.map(p => {
    if (p.length > 1) return '<polyline points="' + p.join(' ') + '"/>';
    const [px, py] = p[0].split(',').map(Number);
    return '<polyline points="' + (px - 1.5) + ',' + py + ' ' + (px + 1.5) + ',' + py + '"/>';
  }).join('');
  const lo = Math.min(...nums), hi = Math.max(...nums);
  const fmt = v => fixedMax === 1 ? stPct(v) : stNum(v, 1);
  return '<svg class="mon-spark" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">'
    + '<title>last ' + nums.length + ' readings: ' + fmt(lo) + ' to ' + fmt(hi) + '</title>'
    + lines + '</svg>';
}

function monTile(hist, label, value, sub, key, fixedMax, cls) {
  return '<div class="st-tile mon-tile' + (cls ? ' ' + cls : '') + '">'
    + '<div class="st-tile-label">' + esc(label) + '</div>'
    + '<div class="st-tile-num">' + value + '</div>'
    + (sub ? '<div class="st-tile-sub">' + sub + '</div>' : '')
    + monSpark(hist, key, fixedMax) + '</div>';
}

function paintMon() {
  if (!monOut) return;
  const order = ['ocr', 'extract'];
  const panes = Object.values(monState.panes)
    .sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
  const exRow = stEl('monGpuExtractRow');
  if (exRow) exRow.style.display = monTargets().some(t => t.key === 'extract') ? '' : 'none';
  if (!monState.on && !panes.some(p => p.last)) { monOut.innerHTML = ''; return; }
  monOut.innerHTML = panes.map(p => '<section class="mon-server">'
    + monPaneHtml(p, panes.length > 1) + '</section>').join('');
}

function monPaneHtml(p, many, on = monState.on) {
  const d = p.last || {}, hist = p.hist;
  const s = d.snapshot || {}, r = d.rates || {}, g = d.gpu;
  const every = p.delay || monEvery();
  const slowed = on && every > monEvery()
    ? (p.error ? ' (backing off after an error)' : ' (server idle, slowed down)') : '';
  let head = (many ? '<b>' + esc(p.label) + '</b> · ' : '') + (on
    ? '<span class="mon-dot"></span><b>Live</b> · every ' + every + ' s' + slowed
    : '<b>' + (p.saved ? 'Saved monitor snapshot' : 'Stopped') + '</b>');
  if (d.url) head += ' · ' + esc(d.kind || '') + ' at ' + esc(d.url)
    + (d.model ? ' · ' + esc(d.model) : '')
    + ' · updated ' + new Date(d.at * 1000).toLocaleTimeString();
  let html = '<div class="mon-head">' + head + '</div>';
  if (p.error) html += '<p class="note err">' + esc(p.error) + '</p>';
  if (!d.snapshot) return html;
  const noRate = 'needs a second reading';

  const kvFree = s.kv_free;
  const kvCls = kvFree === null || kvFree === undefined ? ''
    : kvFree < 0.1 ? 'mon-bad' : kvFree < 0.25 ? 'mon-warn' : '';
  const kvSub = (s.kv_tokens_free !== null && s.kv_tokens_free !== undefined
      ? stNum(s.kv_tokens_free, 0) + ' of ' + stNum(s.kv_tokens_total, 0) + ' tokens free'
      : 'block count not exported')
    + (r.preemptions ? ' · <b>' + r.preemptions + ' preempted</b>' : '');

  // With only the vLLM port reachable there is no figure for the card itself:
  // what the port does say is the share vLLM took at start-up, and the part of
  // that which moves under load is the KV cache, in the tile beside this one.
  const share = s.reserved_share;
  const vramCls = g && g.total_mb && g.free_mb / g.total_mb < 0.05 ? 'mon-warn' : '';
  const vramLabel = g ? 'VRAM free' : 'VRAM held by vLLM';
  const vramVal = g ? stNum(g.free_mb / 1024, 1) + '<small> GB</small>'
    : (share !== null && share !== undefined ? stPct(share) + '<small> of the card</small>' : '—');
  const vramSub = g
    ? 'of ' + stNum(g.total_mb / 1024, 1) + ' GB · ' + esc(g.source)
      + (g.gpus > 1 ? ' · ' + g.gpus + ' GPUs' : '')
      + ' · vLLM reserves most of it up front'
    : (share !== null && share !== undefined
        ? 'reserved at start-up and held, so it does not move. The card size is not '
          + 'readable over the vLLM port: KV cache free is the part that moves'
        : esc(d.gpu_reason || 'not reported'));

  // Requests per second: finished, from vLLM counter; received, from that plus
  // how much the backlog grew. How they finished says whether replies ran to
  // their cap (length -- where a loop ends if nothing cuts it) or were cut off.
  const why = r.finished_by_reason || {};
  const whyText = Object.keys(why).filter(k => why[k])
    .map(k => why[k] + ' ' + esc(k)).join(' · ');
  const la = d.loop_aborts || {};
  const reqCls = why.length ? 'mon-warn' : '';
  const reqSub = (r.seconds
      ? 'received ' + stNum(r.arrivals_per_s, 2) + '/s · '
        + (whyText ? 'finished ' + whyText : 'none finished') + ' in ' + r.seconds + ' s'
      : noRate)
    + ' · loop cut-off ' + (d.loop_abort_on ? 'on' : '<b>off</b>')
    + (la.count ? ' · <b>' + la.count + ' cut by this app</b>' : '');

  const waitCls = s.waiting ? 'mon-warn' : '';
  const has = v => v !== null && v !== undefined;

  // Batch size: generated tokens per engine step, i.e. how many requests one
  // step decoded. Far below "running" means the GPU is decoding few at a time.
  const batchSub = (r.seconds
      ? (r.steps ? 'requests decoded per engine step'
          + (has(r.tokens_per_step) ? ' · ' + stNum(r.tokens_per_step, 1) + ' tokens/step with prefill' : '')
        : 'no engine steps in this window')
      : noRate)
    + (has(d.lifetime_batch) ? ' · ' + stNum(d.lifetime_batch, 1) + ' since start' : '');

  // Multimodal cache: images whose preprocessing was reused. Counted in items,
  // and cache_salt does not bypass it.
  const mmSub = (r.seconds
      ? (r.mm_queries ? 'of ' + stNum(r.mm_queries, 0) + ' images looked up'
                      : 'no images in this window')
      : noRate)
    + (has(d.lifetime_mm_hit_rate) ? ' · ' + stPct(d.lifetime_mm_hit_rate) + ' since start' : '');

  // Requests that fit: the KV cache over what one finished request costs
  // (prompt + whole output). Prefix sharing is ignored, so it leans low.
  const f = d.fit;
  const fitCls = !f || !has(f.more_now) ? '' : f.more_now < 1 ? 'mon-bad' : f.more_now < 5 ? 'mon-warn' : '';
  const fitVal = f ? '≈' + stNum(f.requests, 0) + '<small> requests</small>' : '—';
  const fitSub = f
    ? (has(f.more_now) ? '≈' + stNum(f.more_now, 0) + ' more fit now · ' : '')
      + stNum(f.per_request, 0) + ' tokens each ('
      + (f.source === 'window' ? 'requests finished in the last ' + r.seconds + ' s' : 'mean since start')
      + ')' + (has(f.at_full_context) ? ' · ' + stNum(f.at_full_context, 1) + ' at the full context' : '')
    : 'needs the KV block count and a finished request';

  html += '<div class="st-tiles mon-tiles">'
    + monTile(hist, 'KV cache free', stPct(kvFree), kvSub, 'kv_free', 1, kvCls)
    + monTile(hist, vramLabel, vramVal, vramSub, 'vram_free', 0, vramCls)
    + monTile(hist, 'Running (concurrent)', stNum(s.running, 0), 'requests being decoded', 'running')
    + monTile(hist, 'Waiting', stNum(s.waiting, 0),
        'queued inside vLLM' + (s.swapped ? ' · ' + stNum(s.swapped, 0) + ' swapped' : ''),
        'waiting', 0, waitCls)
    + monTile(hist, 'Generation tok/s', stNum(r.generation_tps, 1),
        r.seconds ? 'over the last ' + r.seconds + ' s' : noRate, 'gen')
    + monTile(hist, 'Prompt tok/s', stNum(r.prompt_tps, 1),
        r.seconds ? 'prefill, over the last ' + r.seconds + ' s' : noRate, 'prompt')
    + monTile(hist, 'Requests/s (finished)', stNum(r.requests_per_s, 2), reqSub, 'rps', 0, reqCls)
    + monTile(hist, 'Prefix cache hit', stPct(r.prefix_hit_rate),
        (r.seconds
          ? (r.prefix_queries ? 'of ' + stNum(r.prefix_queries, 0) + ' tokens looked up'
                              : 'no lookups in this window')
          : noRate)
        + (d.lifetime_hit_rate !== null && d.lifetime_hit_rate !== undefined
            ? ' · ' + stPct(d.lifetime_hit_rate) + ' since start' : ''),
        'hit', 1)
    + monTile(hist, 'Image cache hit', stPct(r.mm_hit_rate), mmSub, 'mm', 1)
    + monTile(hist, 'Batch size', stNum(r.batch, 1), batchSub, 'batch')
    + monTile(hist, 'Fit before preemption', fitVal, fitSub, 'fits', 0, fitCls)
    + '</div>';
  return html;
}

function monReportHtml(state, every) {
  return monPaneHtml({...state, delay: every}, false, !!state.on);
}

if (monToggleBtn) {
  monToggleBtn.addEventListener('click', () => monSet(!monState.on));
  monEveryInp.addEventListener('change', () => {
    monStore('monEvery', monEveryInp.value);
    monEach(p => { p.idleRuns = 0; monSchedule(p); });
    paintMon();
  });
  monGpuInp.addEventListener('change', () => monStore('monGpu', monGpuInp.value.trim()));
  monGpuExInp.addEventListener('change', () => monStore('monGpuExtract', monGpuExInp.value.trim()));
  // Back from a hidden tab: read now rather than waiting out the interval.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && monState.on) monEach(p => { clearTimeout(p.timer); monTick(p); });
  });
  stPaintServer();
}
/* ---------- persistent Stress test history ---------- */
let historyOffset = 0, historySelected = null, historyCurrent = null;
let historyListSeq = 0, historyDetailSeq = 0;
stEl('stViews').addEventListener('click', event => {
  const button = event.target.closest('button[data-st-view]');
  if (button) showStressView(button.dataset.stView);
});
const historyPageSize = 50;
const historyLabel = status => ({running: 'Running', completed: 'Completed',
  completed_with_errors: 'Completed with errors', stopped: 'Stopped',
  interrupted: 'Interrupted', unfinished: 'Unfinished / other instance', error: 'Error'}[status] || status);
const historyDate = value => value ? new Date(value).toLocaleString() : '—';

async function historyLoad() {
  const mine = ++historyListSeq;
  const note = stEl('historyStatus');
  note.textContent = 'Loading…';
  note.classList.remove('err');
  try {
    const res = await fetch(API + '/history?limit=' + historyPageSize + '&offset=' + historyOffset,
      {cache: 'no-store'});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    if (mine !== historyListSeq) return;
    const runs = data.runs || [];
    note.textContent = data.total ? (historyOffset + 1) + '–'
      + (historyOffset + runs.length) + ' of ' + data.total : 'No saved runs yet.';
    stEl('historyPrev').disabled = historyOffset === 0;
    stEl('historyNext').disabled = historyOffset + runs.length >= data.total;
    stEl('historyList').innerHTML = runs.length ? runs.map(run => {
      const p = run.plan || run.configuration || {};
      return '<button type="button" class="history-item" data-history-id="' + esc(run.id) + '">'
        + '<b>' + esc(historyDate(run.started_at)) + '</b>'
        + '<span>' + esc(historyLabel(run.status)) + ' · ' + esc(p.mode || '') + '</span>'
        + '<span>Concurrent ' + esc(String(p.concurrency ?? '—'))
        + ' · Queue ' + esc(String(p.jobs ?? '—')) + ' · Warm-up ' + esc(String(p.warmup ?? '—')) + '</span>'
        + '<span>' + esc(p.reader || p.extractor || ((run.configuration.reader
          || run.configuration.extractor || {}).model) || '') + '</span></button>';
    }).join('') : '<p class="note">New Stress test runs will appear here automatically. '
      + 'Earlier runs were not saved.</p>';
    if (runs.length) {
      const selected = runs.some(r => r.id === historySelected) ? historySelected : runs[0].id;
      await historySelect(selected);
    } else {
      ++historyDetailSeq;
      historySelected = null; historyCurrent = null;
      stEl('historyTitle').textContent = '';
      stEl('historyDownload').hidden = true;
      stEl('historyDetail').innerHTML = '<p class="note">No saved Stress test runs on this page.</p>';
    }
  } catch (err) {
    if (mine !== historyListSeq) return;
    note.textContent = err.message;
    note.classList.add('err');
  }
}

async function historySelect(id) {
  const mine = ++historyDetailSeq;
  historySelected = id;
  stEl('historyDownload').hidden = true;
  stEl('historyTitle').textContent = 'Loading run…';
  document.querySelectorAll('[data-history-id]').forEach(button => {
    const on = button.dataset.historyId === id;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  });
  try {
    const res = await fetch(API + '/history/' + encodeURIComponent(id), {cache: 'no-store'});
    const item = await res.json();
    if (!res.ok) throw new Error(item.error || ('HTTP ' + res.status));
    if (mine !== historyDetailSeq) return;
    historyCurrent = item;
    historyRender(item);
  } catch (err) {
    if (mine !== historyDetailSeq) return;
    stEl('historyTitle').textContent = 'Could not load run';
    stEl('historyDetail').innerHTML = '<p class="note err">' + esc(err.message) + '</p>';
  }
}

function historyRender(item) {
  const cfg = item.configuration || {}, plan = item.plan || {}, result = item.result || {};
  const read = cfg.reader || {}, extract = cfg.extractor || {};
  stEl('historyTitle').textContent = historyDate(item.started_at) + ' · ' + historyLabel(item.status);
  const download = stEl('historyDownload');
  download.href = (API + '/history/' + encodeURIComponent(item.id) + '?download=1');
  download.hidden = false;
  const value = v => v === undefined || v === null ? '—' : typeof v === 'object'
    ? JSON.stringify(v) : String(v);
  const cells = [
    ['Mode', cfg.mode], ['Concurrent', cfg.concurrency], ['Queue (jobs)', cfg.jobs],
    ['Warm-up', cfg.warmup], ['Time limit (s)', cfg.max_seconds],
    ['Detail', cfg.detail], ['Extraction shape', cfg.extract_mode],
    ['Extra fields', cfg.extras ?? true],
    ['JSON schema first', cfg.schema_first ?? false],
    ['Seed', plan.seed ?? cfg.seed],
    ['Pages per file', cfg.pack_pages], ['Draw strategy', cfg.strategy],
    ['Lock document', cfg.lock || 'None'],
    ['Document source', cfg.source === 'realdata' ? 'realData (no solution)' : 'Benchmark'],
    ['Files selected', cfg.cases ? cfg.cases.join(', ') : undefined],
    ['OCR model', read.model], ['OCR server', read.url],
    ['Extraction model', extract.model], ['Extraction server', extract.url],
    ['OCR profile', cfg.ocr_profile], ['Context', cfg.context], ['Sampling', cfg.sampling],
    ['Loop guard', cfg.loop_guard], ['Read floor', cfg.read_floor], ['Table agents', cfg.table_agents],
    ['Sample /metrics', cfg.metrics], ['Monitor snapshots', cfg.history_monitor],
    ['Sampling interval (s)', cfg.metrics_every], ['GPU exporter', cfg.gpu_url || 'Default / local'],
    ['Extraction GPU exporter', cfg.gpu_extract_url || 'Default / local'],
    ['Bypass prompt cache', cfg.no_cache], ['Batch requests (vLLM /batch)', cfg.vllm_batch ?? false],
    ['Fill missing from images (Hybrid)', cfg.mode === 'full' ? cfg.fill_missing ?? true : undefined],
    ['Recheck validation from images (Hybrid)', cfg.mode === 'full' ? cfg.validation_retry ?? false : undefined],
    ['Save per-job accuracy rows', cfg.save_log],
    ['Detail log destination', cfg.save_details ? 'Browser download only; not stored here' : 'Off'],
    ['Workspace', cfg.workspace || 'Default'],
    ['Started', historyDate(item.started_at)], ['Finished', historyDate(item.finished_at)]
  ];
  const root = stEl('historyDetail');
  root.innerHTML = (item.error ? '<p class="note err">' + esc(item.error) + '</p>' : '')
    + (item.status === 'running' ? '<p class="note">This run is still in progress. Refresh to check its final report.</p>' : '')
    + '<h3 class="st-h">Settings used for this run</h3><dl class="history-config">'
    + cells.slice(0, 8).map(([key, v]) => '<div><dt>' + esc(key) + '</dt><dd>' + esc(value(v)) + '</dd></div>').join('')
    + '</dl><details><summary>More settings: models, servers, sampling and monitor</summary><dl class="history-config">'
    + cells.slice(8).map(([key, v]) => '<div><dt>' + esc(key) + '</dt><dd>' + esc(value(v)) + '</dd></div>').join('')
    + '</dl></details><div id="historyReport" class="history-report"></div>'
    + '<details><summary>All saved settings and metrics (JSON)</summary><pre class="history-json"></pre></details>';
  const host = stEl('historyReport');
  const saved = {plan: item.plan, report: result.report,
    alerts: result.alerts || [], monitor_samples: result.monitor_samples,
    monitor_samples_total: result.monitor_samples_total};
  const draw = () => {
    host.innerHTML = stReportHtml(saved, null, host)
      + (!result.report ? '<p class="note">No final report was saved for this run.</p>' : '');
    stWireChart(host, result.report);
    stWireChatContext(host, stOutlierReport(saved));
    stWireOutliers(host, saved, draw);
  };
  draw();
  root.querySelector('.history-json').textContent = JSON.stringify(item, null, 2);
}

function historyMonitorState(samples, index) {
  const d = samples[index] || {};
  return {saved: true, on: false, last: d, error: d.error || '',
    hist: samples.slice(0, index + 1).filter(sample => sample.url === d.url).map(sample => {
      const s = sample.snapshot || {}, r = sample.rates || {}, g = sample.gpu;
      return {t: sample.at, kv_free: s.kv_free, running: s.running, waiting: s.waiting,
        vram_free: g ? g.free_mb / 1024 : null, gen: r.generation_tps,
        prompt: r.prompt_tps, hit: r.prefix_hit_rate, rps: r.requests_per_s,
        mm: r.mm_hit_rate, batch: r.batch, fits: sample.fit ? sample.fit.more_now : null};
    })};
}

function historyMonitorHtml(samples, total) {
  const options = samples.map((d, i) => '<option value="' + i + '"'
    + (i === samples.length - 1 ? ' selected' : '') + '>'
    + esc(d.at ? new Date(d.at * 1000).toLocaleTimeString() : 'Snapshot ' + (i + 1))
    + (d.role ? ' · ' + esc(d.role) : '')
    + (d.error ? ' · unavailable' : ' · running ' + esc(String((d.snapshot || {}).running ?? '—')))
    + '</option>').join('');
  return '<section class="history-monitor"><h3 class="st-h">Server monitor during this run</h3>'
    + '<label class="field">Saved snapshot<select class="history-snapshot">' + options + '</select></label>'
    + '<p class="note">' + samples.length + ' saved snapshot(s)'
    + (total > samples.length ? ' · latest ' + samples.length + ' of ' + total : '')
    + '. These are recorded values, not live readings.</p><div class="history-monitor-body">'
    + monReportHtml(historyMonitorState(samples, samples.length - 1), 0) + '</div></section>';
}

stEl('historyList').addEventListener('click', event => {
  const button = event.target.closest('[data-history-id]');
  if (button) historySelect(button.dataset.historyId);
});
stEl('historyRefresh').addEventListener('click', historyLoad);
stEl('historyPrev').addEventListener('click', () => {
  historyOffset = Math.max(0, historyOffset - historyPageSize); historyLoad();
});
stEl('historyNext').addEventListener('click', () => { historyOffset += historyPageSize; historyLoad(); });
document.addEventListener('change', event => {
  if (!event.target.matches('.history-snapshot')) return;
  const saved = !!event.target.closest('#historyCard');
  const samples = saved ? ((historyCurrent || {}).result || {}).monitor_samples
    : (stState || {}).monitor_samples;
  if (!samples) return;
  event.target.closest('.history-monitor').querySelector('.history-monitor-body').innerHTML =
    monReportHtml(historyMonitorState(samples, Number(event.target.value)), 0);
});
