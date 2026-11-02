"""A tab closed during a stress run can come back to it (2026-11-02).

Runs the real page code in node against fake browser storage: the workspace
script from the template, and the helpers from benchmark.js that list a run.
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
const [template, bench, scenario] = process.argv.slice(1);
const steps = JSON.parse(scenario);

function storage(values) {
  return {
    getItem: k => (k in values ? values[k] : null),
    setItem: (k, v) => { values[k] = String(v); },
    removeItem: k => { delete values[k]; },
  };
}

const local = {};
const channels = [];
let loading = [];
class Channel {
  constructor() { this.onmessage = null; channels.push(this); loading.push(this); }
  postMessage(data) {
    this.posted = data;
    for (const other of channels) {
      if (other !== this && other.onmessage) other.onmessage({ data });
    }
  }
}

// One page load: returns what the tab ended up with, and where it navigated.
function load(session, href, isUser) {
  loading = [];
  const context = {
    sessionStorage: storage(session), localStorage: storage(local),
    location: { href, origin: 'http://app', replace(u) { this.next = u; } },
    fetch: () => null, URL, Headers, BroadcastChannel: Channel, crypto: undefined,
    Math, Date, JSON, Object, console,
  };
  context.window = context;
  context.OCR_BASE = '/OCR';
  context.USER_BASE = '/user';
  const source = fs.readFileSync(template, 'utf8')
    .replace('{{ user_session | default(false) | tojson }}', isUser ? 'true' : 'false');
  vm.runInNewContext(source, context);
  context.channels = loading;
  return context;
}

// A page that has gone no longer answers.
function leave(page) {
  for (const channel of page.channels) channels.splice(channels.indexOf(channel), 1);
}

// Load a tab until it stops navigating, as a browser would.
function open(session, isUser) {
  let href = 'http://app/OCR/', page;
  for (let i = 0; i < 5; i++) {
    page = load(session, href, isUser);
    if (!page.location.next) return page;
    href = page.location.next;
    leave(page);
  }
  throw new Error('the tab never settled');
}

function helpers(ws) {
  const source = fs.readFileSync(bench, 'utf8');
  const from = source.indexOf('const ST_SEEN_KEY');
  const to = source.indexOf('// Read one response of a run');
  const context = {
    sessionStorage: storage({}), localStorage: storage(local),
    OCR_BASE: '/OCR', OCR_WORKSPACE: ws, JSON, Array,
  };
  context.window = context;
  vm.runInNewContext(
    source.slice(from, to) + ';this.park = stPark; this.seen = stMarkSeen;', context);
  return context;
}

const out = {};
const tabs = {};
for (const step of steps) {
  if (step.op === 'open') {
    const session = step.session || {};
    tabs[step.name] = { session, page: open(session, step.user) };
    out[step.name] = {
      ws: tabs[step.name].page.OCR_WORKSPACE,
      adopted: !!tabs[step.name].page.OCR_ADOPTED_RUN,
    };
  } else if (step.op === 'close') {
    leave(tabs[step.name].page);
    delete tabs[step.name];
  } else if (step.op === 'park') {
    helpers(step.ws || out[step.name].ws).park(step.run);
  } else if (step.op === 'seen') {
    helpers(step.ws || out[step.name].ws).seen(step.run);
  }
}
out.parked = JSON.parse(local['ocrStressRuns:/OCR'] || '[]');
process.stdout.write(JSON.stringify(out));
"""


def workspace_script():
    return (ROOT / "templates" / "_workspace_session.html").read_text(encoding="utf-8")


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ReopenTabTests(unittest.TestCase):
    def run_steps(self, *steps):
        done = subprocess.run(
            ["node", "-e", HARNESS, str(self.script), str(ROOT / "static" / "benchmark.js"),
             json.dumps(list(steps))],
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.directory = tempfile.TemporaryDirectory()
        cls.script = Path(cls.directory.name) / "workspace.js"
        cls.script.write_text(workspace_script(), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_a_new_tab_with_no_run_starts_a_new_workspace(self):
        out = self.run_steps({"op": "open", "name": "a"})
        self.assertRegex(out["a"]["ws"], r"^[A-Za-z0-9]{8,}$")
        self.assertFalse(out["a"]["adopted"])

    def test_reopening_after_closing_takes_the_runs_workspace_back(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "close", "name": "a"},
            {"op": "open", "name": "b"},
        )
        self.assertEqual(out["b"]["ws"], out["a"]["ws"])
        self.assertTrue(out["b"]["adopted"])

    def test_a_tab_still_open_keeps_its_run_and_the_new_tab_starts_empty(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "open", "name": "b"},
        )
        self.assertNotEqual(out["b"]["ws"], out["a"]["ws"])
        self.assertFalse(out["b"]["adopted"])

    def test_a_tab_still_open_does_not_hide_a_run_whose_tab_closed(self):
        out = self.run_steps(
            {"op": "open", "name": "gone"},
            {"op": "park", "name": "gone", "run": "run1"},
            {"op": "close", "name": "gone"},
            {"op": "open", "name": "held", "session": {"ocrWorkspace": "heldworkspace1"}},
            {"op": "park", "ws": "heldworkspace1", "run": "run2"},
            {"op": "open", "name": "new"},
        )
        self.assertEqual(out["new"]["ws"], out["gone"]["ws"])
        self.assertTrue(out["new"]["adopted"])

    def test_a_seen_report_ends_the_adoption(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "seen", "name": "a", "run": "run1"},
            {"op": "close", "name": "a"},
            {"op": "open", "name": "b"},
        )
        self.assertEqual(out["parked"], [])
        self.assertNotEqual(out["b"]["ws"], out["a"]["ws"])
        self.assertFalse(out["b"]["adopted"])

    def test_parking_twice_keeps_one_entry_per_workspace(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "park", "name": "a", "run": "run2"},
        )
        self.assertEqual(out["parked"], [{"ws": out["a"]["ws"], "run": "run2"}])

    def test_a_tab_that_has_a_workspace_keeps_it(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "close", "name": "a"},
            {"op": "open", "name": "b", "session": {"ocrWorkspace": "existingworkspace"}},
        )
        self.assertEqual(out["b"]["ws"], "existingworkspace")
        self.assertFalse(out["b"]["adopted"])

    def test_the_user_page_does_not_adopt(self):
        out = self.run_steps(
            {"op": "open", "name": "a"},
            {"op": "park", "name": "a", "run": "run1"},
            {"op": "close", "name": "a"},
            {"op": "open", "name": "u", "user": True},
        )
        self.assertNotEqual(out["u"]["ws"], out["a"]["ws"])


if __name__ == "__main__":
    unittest.main()
