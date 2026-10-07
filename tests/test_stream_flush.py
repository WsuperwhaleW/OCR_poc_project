import unittest
from unittest import mock

import app


class BatchTokensTests(unittest.TestCase):
    def test_zero_interval_is_one_piece_a_batch(self):
        self.assertEqual(list(app.batch_tokens(iter("abc"), 0)), [["a"], ["b"], ["c"]])

    def test_first_piece_alone_then_coalesced(self):
        clock = iter([0.0, 0.01, 0.02, 0.15, 0.16])
        with mock.patch.object(app.time, "perf_counter", lambda: next(clock)):
            out = list(app.batch_tokens(iter("abcde"), 0.1))
        self.assertEqual(out, [["a"], ["b", "c", "d"], ["e"]])
        self.assertEqual("".join("".join(b) for b in out), "abcde")

    def test_pending_flushed_before_error(self):
        def pieces():
            yield "a"
            yield "b"
            raise ValueError("boom")

        got = []
        with self.assertRaises(ValueError):
            for batch in app.batch_tokens(pieces(), 10):
                got.append(batch)
        self.assertEqual(got, [["a"], ["b"]])


if __name__ == "__main__":
    unittest.main()
