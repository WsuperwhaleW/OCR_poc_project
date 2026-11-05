"""Where a flagged read went wrong: DRY's runs and the loop, marked in its text."""
import unittest

import dryreplay
import engine


class DryReplaySpansTests(unittest.TestCase):
    def test_fired_steps_come_back_as_runs(self):
        ids = [1, 2, 3, 1, 2, 3, 1, 2, 3, 9, 1, 2, 3]
        result = dryreplay.replay(ids, {"dry_allowed_length": 2, "dry_multiplier": 0.8})
        self.assertTrue(result["fired"])
        self.assertEqual(sum(b - a for a, b in result["spans"]), result["fired"])
        self.assertEqual(result["spans"][0][0], result["first"])


class DryRawSpansTests(unittest.TestCase):
    def test_exact_where_each_token_is_placed(self):
        dry = {"tokens": 4, "spans": [[2, 4]]}
        self.assertEqual(engine._dry_raw_spans(dry, [0, 3, 6, 9], 12),
                         {"raw_spans": [[6, 12]]})

    def test_estimated_where_the_reply_came_in_one_piece(self):
        out = engine._dry_raw_spans({"tokens": 10, "spans": [[5, 10]]}, None, 100)
        self.assertEqual(out, {"raw_spans": [[50, 100]], "raw_spans_approx": True})


class LoopSpansTests(unittest.TestCase):
    def test_repeated_lines_marked_from_the_second_copy(self):
        line = "<tr><td>same row</td></tr>\n"
        text = "Header\n" + line * 8
        [(a, b)] = engine.loop_spans(text)
        self.assertEqual(a, len("Header\n") + len(line))
        self.assertEqual(text[a:b], (line * 7).rstrip("\n"))

    def test_a_dead_tail_of_empty_rows_is_marked(self):
        text = "Real words here " + "<tr><td></td><td></td></tr>" * 200
        [(a, b)] = engine.loop_spans(text)
        self.assertEqual(b, len(text))
        self.assertTrue(text[a:].startswith("<tr>"))

    def test_a_clean_read_has_no_loop(self):
        self.assertEqual(engine.loop_spans("one\ntwo\nthree"), [])


class ReadMarksTests(unittest.TestCase):
    def test_dry_offsets_follow_the_reply_into_the_transcript(self):
        raw = "# Title\nabc &amp; def repeat repeat"
        text = engine.read_reply(raw)
        stats = {"raw": raw, "dry": {"fired": 2,
                                     "raw_spans": [[raw.index("repeat"), len(raw)]]}}
        [mark] = engine.read_marks(text, stats)
        self.assertEqual(text[mark["start"]:mark["end"]], "repeat repeat")
        self.assertNotIn("approx", mark)

    def test_a_mark_never_cuts_a_tag(self):
        text = '<td colspan="4">x</td>'
        stats = {"raw": text, "dry": {"fired": 1, "raw_spans": [[5, 12]],
                                      "raw_spans_approx": True}}
        [mark] = engine.read_marks(text, stats)
        self.assertEqual((mark["start"], mark["end"]), (0, len('<td colspan="4">')))
        self.assertTrue(mark["approx"])

    def test_a_looped_read_is_marked_and_a_clean_one_is_not(self):
        text = "x\n" + "same line again\n" * 6
        self.assertEqual([m["kind"] for m in engine.read_marks(text, {"looped": True})],
                         ["loop"])
        self.assertEqual(engine.read_marks(text, {}), [])


if __name__ == "__main__":
    unittest.main()
