"""A stress run's raw results are copied into the browser (2026-11-02).

Runs `static/benchmark-results.js` in node with a Map in place of IndexedDB and a
fake server. The IndexedDB adapter itself needs a browser and is not covered here.
"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const context = { Map, Promise, JSON, String, Date, console };
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') + `
  this.create = createStressResults; this.off = STRESS_RESULTS_OFF;`, context);

function memory(failPuts) {
  const stores = {runs: new Map(), evidence: new Map(), images: new Map(), details: new Map()};
  return {
    stores,
    failPuts,
    get: async (s, k) => stores[s].get(k),
    async put(s, v) {
      if (this.failPuts && this.failPuts-- > 0) {
        const err = new Error('full'); err.name = 'QuotaExceededError'; throw err;
      }
      stores[s].set(v.key, v);
    },
    remove: async (s, k) => { stores[s].delete(k); },
    all: async s => [...stores[s].values()],
    byRun: async (s, id) => [...stores[s].values()].filter(r => r.run === id),
  };
}

function server(pages) {
  const calls = [];
  const fetch = async url => {
    calls.push(url);
    const m = url.match(/evidence\/([^/]+)(?:\/(\d+)\.png)?$/);
    const page = m && pages[decodeURIComponent(m[1])];
    if (!page) return { ok: false, status: 404 };
    if (m[2] !== undefined) return { ok: true, blob: async () => ({ size: 100, id: m[1] + m[2] }) };
    return { ok: true, json: async () => page };
  };
  return { fetch, calls };
}

(async () => {
  const out = {};
  const attempts = n => ({ attempts: Array.from({ length: n }, () => ({ text: 'x' })) });
  let clock = 1000;
  const now = () => clock++;

  // Copy on flag, read it back with the server gone.
  let backend = memory(), srv = server({ a1: attempts(2) });
  let store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  store.capture('r1', 'a1', 5);
  await store.idle();
  const copy = await store.read('a1');
  out.read = { revision: copy.revision, attempts: copy.data.attempts.length,
               blobs: copy.blobs.map(b => b && b.id) };
  out.first_calls = srv.calls.length;

  // The same revision costs nothing.
  srv.calls.length = 0;
  store.capture('r1', 'a1', 5);
  await store.idle();
  out.same_revision_calls = srv.calls.length;

  // A retry adds a read: only the new image is fetched.
  srv.calls.length = 0;
  srv = server({ a1: attempts(3) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  store.capture('r1', 'a1', 9);
  await store.idle();
  out.update_calls = srv.calls;
  out.update_revision = (await store.read('a1')).revision;

  // A read with no image is not asked for, a page the server dropped is skipped.
  backend = memory();
  srv = server({ t: { attempts: [{ image: false }] } });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  store.capture('r', 't');
  store.capture('r', 'gone');
  await store.idle();
  out.no_image_calls = srv.calls.length;
  out.gone = await store.read('gone');

  // Runs: only finished ones list, newest first, dismissed ones are not latest.
  backend = memory();
  srv = server({ a: attempts(1) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  await store.saveRun('unfinished', { finished: false, state: {} });
  await store.saveRun('old', { finished: true, state: {}, historyId: 'h1' });
  await store.saveRun('new', { finished: true, state: {} });
  out.listed = (await store.runs()).map(r => r.key);
  out.latest = (await store.latest()).key;
  await store.saveRun('new', { dismissed: true });
  out.latest_after_dismiss = (await store.latest()).key;
  await store.saveDetail('old', { size: 7 }, 'log.json', 3);
  out.detail_for_history = (await store.detailForHistory('h1')).name;
  out.detail_unknown = await store.detailForHistory('nope');

  // Past keepRuns the oldest go, with their pages; the run being saved stays.
  backend = memory();
  srv = server({ p1: attempts(1), p2: attempts(1) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now, keepRuns: 1 });
  store.capture('one', 'p1');
  await store.idle();
  await store.saveRun('one', { finished: true, state: {} });
  store.capture('two', 'p2');
  await store.idle();
  await store.saveRun('two', { finished: true, state: {} });
  out.pruned_runs = [...backend.stores.runs.keys()];
  out.pruned_evidence = [...backend.stores.evidence.keys()];
  out.pruned_images = [...backend.stores.images.keys()];

  // Storage full: older runs go first, and only then does capture stop.
  backend = memory();
  srv = server({ q: attempts(1) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  await store.saveRun('older', { finished: true, state: {} });
  backend.failPuts = 1;
  store.capture('now', 'q');
  await store.idle();
  out.full_recovered = { full: store.full, stored: !!(await store.read('q')),
                         older: !!(await store.run('older')) };

  backend = memory();
  srv = server({ z: attempts(1) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  backend.failPuts = 50;
  store.capture('now', 'z');
  await store.idle();
  out.full_stops = store.full;

  // Clear empties every store.
  backend = memory();
  srv = server({ c: attempts(1) });
  store = context.create({ backend, fetch: srv.fetch, api: '/api', now });
  store.capture('r', 'c');
  await store.idle();
  await store.saveRun('r', { finished: true, state: {} });
  await store.clear();
  out.cleared = Object.values(backend.stores).map(s => s.size);

  // No IndexedDB: every read finds nothing.
  out.off = { available: context.off.available, read: await context.off.read('a'),
              latest: await context.off.latest() };

  process.stdout.write(JSON.stringify(out));
})().catch(err => { console.error(err); process.exit(1); });
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class BrowserResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        done = subprocess.run(
            ["node", "-e", HARNESS, str(ROOT / "static" / "benchmark-results.js")],
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert done.returncode == 0, done.stderr
        cls.out = json.loads(done.stdout)

    def test_a_flagged_page_is_copied_and_read_back(self):
        self.assertEqual(self.out["read"], {"revision": "5", "attempts": 2,
                                            "blobs": ["a10", "a11"]})
        self.assertEqual(self.out["first_calls"], 3)

    def test_the_same_revision_asks_the_server_nothing(self):
        self.assertEqual(self.out["same_revision_calls"], 0)

    def test_an_update_only_fetches_the_new_image(self):
        self.assertEqual(self.out["update_calls"], ["/api/stress/evidence/a1",
                                                    "/api/stress/evidence/a1/2.png"])
        self.assertEqual(self.out["update_revision"], "9")

    def test_no_image_and_a_dropped_page(self):
        self.assertEqual(self.out["no_image_calls"], 2)
        self.assertIsNone(self.out["gone"])

    def test_listing_latest_and_dismissing(self):
        self.assertEqual(self.out["listed"], ["new", "old"])
        self.assertEqual(self.out["latest"], "new")
        self.assertEqual(self.out["latest_after_dismiss"], "old")

    def test_a_history_run_finds_its_detail_log(self):
        self.assertEqual(self.out["detail_for_history"], "log.json")
        self.assertIsNone(self.out["detail_unknown"])

    def test_old_runs_are_removed_with_their_pages(self):
        self.assertEqual(self.out["pruned_runs"], ["two"])
        self.assertEqual(self.out["pruned_evidence"], ["p2"])
        self.assertEqual(self.out["pruned_images"], ["p2|0"])

    def test_full_storage_removes_older_runs_then_stops(self):
        self.assertEqual(self.out["full_recovered"],
                         {"full": False, "stored": True, "older": True})
        self.assertTrue(self.out["full_stops"])

    def test_clear_empties_everything(self):
        self.assertEqual(self.out["cleared"], [0, 0, 0, 0])

    def test_without_indexeddb_nothing_is_found(self):
        self.assertEqual(self.out["off"], {"available": False, "read": None, "latest": None})


if __name__ == "__main__":
    unittest.main()
