/* A stress run's raw results, kept in THIS browser and never sent back.

   The report and the flag list are in server History, but what a flag points at
   is not: each flagged page's images and text are in server memory (newest
   STRESS_EVIDENCE_MAX, gone on restart) and the detail log only ever reached
   the tab's memory. So the page copies them into IndexedDB as they appear.

   `evidence`  a flagged page's text, by alert id (ids are unique per run)
   `images`    each read's page PNG, `<alert id>|<read number>`
   `details`   the detail log, by run
   `runs`      what the report needs to be drawn again, by run */
const STRESS_RESULT_STORES = ['runs', 'evidence', 'images', 'details'];

function stressResultsIdb(idb, name = 'ocrStressResults') {
  let opening = null;
  const db = () => opening || (opening = new Promise((resolve, reject) => {
    const open = idb.open(name, 1);
    open.onupgradeneeded = () => {
      for (const store of STRESS_RESULT_STORES) {
        open.result.createObjectStore(store, {keyPath: 'key'}).createIndex('run', 'run');
      }
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => { opening = null; reject(open.error); };
  }));
  const run = async (store, mode, work) => {
    const database = await db();
    return new Promise((resolve, reject) => {
      const tx = database.transaction(store, mode);
      const request = work(tx.objectStore(store));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  };
  return {
    get: (store, key) => run(store, 'readonly', s => s.get(key)),
    put: (store, value) => run(store, 'readwrite', s => s.put(value)),
    remove: (store, key) => run(store, 'readwrite', s => s.delete(key)),
    all: store => run(store, 'readonly', s => s.getAll()),
    byRun: (store, id) => run(store, 'readonly', s => s.index('run').getAll(id)),
  };
}

function createStressResults({backend, fetch: request, api, now = Date.now,
                              keepRuns = 20, maxBytes = 1024 * 1024 * 1024}) {
  const queue = new Map();
  let draining = null, full = false;
  const evidenceUrl = id => api + '/stress/evidence/' + encodeURIComponent(id);
  const isFull = err => err && err.name === 'QuotaExceededError';

  async function sizeOf(id) {
    const [evidence, images, details] = await Promise.all(['evidence', 'images', 'details']
      .map(store => backend.byRun(store, id)));
    return evidence.reduce((n, e) => n + JSON.stringify(e.data).length, 0)
      + [...images, ...details].reduce((n, r) => n + (r.blob ? r.blob.size : 0), 0);
  }

  async function deleteRun(id) {
    for (const store of ['evidence', 'images', 'details']) {
      for (const row of await backend.byRun(store, id)) await backend.remove(store, row.key);
    }
    await backend.remove('runs', id);
  }

  // Oldest first, never the run being kept.
  async function prune(current) {
    const runs = (await backend.all('runs')).sort((a, b) => b.at - a.at);
    let bytes = 0, kept = 0;
    for (const row of runs) {
      bytes += row.bytes || 0;
      kept += 1;
      if (row.key !== current && (kept > keepRuns || bytes > maxBytes)) await deleteRun(row.key);
    }
  }

  async function captureOne(run, id, revision) {
    const old = await backend.get('evidence', id);
    if (old && old.revision === revision) return;
    const res = await request(evidenceUrl(id));
    if (!res.ok) return;              // not finished yet, or no longer on the server
    const data = await res.json();
    // Images first, so a stored page always has its images. A read's image
    // never changes, so an update only fetches the reads that are new.
    for (const [n, attempt] of (data.attempts || []).entries()) {
      const key = id + '|' + n;
      if (attempt.image === false || await backend.get('images', key)) continue;
      const image = await request(evidenceUrl(id) + '/' + n + '.png');
      if (image.ok) await backend.put('images', {key, run, blob: await image.blob()});
    }
    await backend.put('evidence', {key: id, run, revision, data});
  }

  async function drain() {
    while (queue.size) {
      const [id, {run, revision}] = queue.entries().next().value;
      queue.delete(id);
      try {
        await captureOne(run, id, revision);
      } catch (err) {
        if (!isFull(err)) continue;   // a dropped request: the final sweep asks again
        try {
          await prune(run);
          await captureOne(run, id, revision);
        } catch {
          full = true;
          queue.clear();
        }
      }
    }
  }

  const api_ = {
    available: true,
    get full() { return full; },
    capture(run, id, revision = '') {
      if (full || !run || id === undefined || id === null) return;
      queue.set(String(id), {run, revision: String(revision)});
      if (!draining) draining = drain().finally(() => { draining = null; });
    },
    async idle() { while (draining) await draining; },
    // A flagged page: `{revision, data, blobs}`, blobs by read number.
    async read(id) {
      const row = await backend.get('evidence', String(id));
      if (!row) return null;
      const blobs = [];
      for (let n = 0; n < (row.data.attempts || []).length; n++) {
        const image = await backend.get('images', id + '|' + n);
        blobs.push(image ? image.blob : null);
      }
      return {revision: row.revision, data: row.data, blobs};
    },
    async saveRun(id, fields) {
      const old = await backend.get('runs', id);
      const row = {at: now(), ...old, ...fields, key: id, run: id};
      row.bytes = await sizeOf(id);
      await backend.put('runs', row);
      await prune(id);
      return row;
    },
    async saveDetail(run, blob, name, count) {
      await backend.put('details', {key: run, run, blob, name, count});
    },
    detail: run => backend.get('details', run),
    async detailForHistory(historyId) {
      const row = (await backend.all('runs')).find(r => r.historyId === historyId);
      return row ? backend.get('details', row.key) : null;
    },
    async runs() {
      return (await backend.all('runs')).filter(r => r.finished).sort((a, b) => b.at - a.at);
    },
    async latest() {
      return (await api_.runs()).find(r => !r.dismissed) || null;
    },
    run: id => backend.get('runs', id),
    deleteRun,
    async clear() {
      queue.clear();
      full = false;
      for (const row of await backend.all('runs')) await deleteRun(row.key);
    },
  };
  return api_;
}

/* What the page uses when the browser has no IndexedDB (or refuses it): every
   read finds nothing and the page falls back to the server, as it always did. */
const STRESS_RESULTS_OFF = {
  available: false, full: false, capture() {}, idle: async () => {}, read: async () => null,
  saveRun: async () => null, saveDetail: async () => {}, detail: async () => null,
  detailForHistory: async () => null, runs: async () => [], latest: async () => null,
  run: async () => null, deleteRun: async () => {}, clear: async () => {},
};

if (typeof window !== 'undefined') {
  window.StressResults = window.indexedDB
    ? createStressResults({backend: stressResultsIdb(window.indexedDB),
                           fetch: (...args) => fetch(...args), api: window.API})
    : STRESS_RESULTS_OFF;
}
