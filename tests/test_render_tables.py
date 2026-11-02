"""The page's table renderer, run under node: a table a read never closed is
still drawn as a table rather than as literal <tr><td> text."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROW = "<tr><td>a</td><td>1.00</td></tr>"

SCRIPT = """
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
const code = src.slice(src.indexOf('function esc(s)'), src.indexOf('function sanitize(html)'));
const renderOutput = new Function(code + '; return renderOutput;')();
const inputs = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(inputs.map(renderOutput)));
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class RenderOutputTests(unittest.TestCase):
    def render(self, *inputs):
        done = subprocess.run(
            ["node", "-e", SCRIPT, str(ROOT / "static" / "benchmark.js")],
            input=json.dumps(inputs), capture_output=True, text=True,
            encoding="utf-8", check=True)
        return json.loads(done.stdout)

    def test_a_closed_table_is_passed_through_unchanged(self):
        table = f"<table>{ROW}</table>"
        self.assertEqual(self.render(table), [table])

    def test_a_table_cut_off_is_closed(self):
        out = self.render(f"<table>{ROW}{ROW}")[0]
        self.assertEqual(out, f"<table>{ROW}{ROW}</table>")

    def test_a_tag_the_read_died_inside_is_dropped(self):
        out = self.render(f"<table>{ROW}<tr><td>b</td><td")[0]
        self.assertEqual(out, f"<table>{ROW}<tr><td>b</td></table>")

    def test_a_closed_table_then_a_cut_off_one(self):
        out = self.render(f"<table>{ROW}</table>\nnote\n<table>{ROW}<tr><td>")[0]
        self.assertEqual(out.count("<table>"), 2)
        self.assertEqual(out.count("</table>"), 2)
        self.assertIn('<div class="txt">note</div>', out)
        self.assertNotIn("&lt;", out)

    def test_rows_with_no_table_around_them_are_wrapped(self):
        out = self.render(f"before\n{ROW}{ROW}\nafter")[0]
        self.assertIn(f"<table>{ROW}{ROW}</table>", out)
        self.assertIn('<div class="txt">before</div>', out)
        self.assertIn('<div class="txt">after</div>', out)

    def test_text_with_no_table_is_still_escaped(self):
        self.assertEqual(self.render("a < b <b>c</b>"),
                         ['<div class="txt">a &lt; b &lt;b&gt;c&lt;/b&gt;</div>'])


if __name__ == "__main__":
    unittest.main()
