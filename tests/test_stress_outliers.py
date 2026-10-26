"""The stress report without its N longest jobs (`stress.without_longest`)."""

import unittest

import stress


def job(n, total, end, status="ok", pages=1):
    return {"job": n, "who": f"sol00{n}", "status": status, "total_s": total,
            "end_s": end, "wait_s": 0, "ocr_s": total, "pages": pages,
            "ocr_tokens": 100, "page_s": [total], "char_accuracy": 0.9}


class WithoutLongestTests(unittest.TestCase):
    def setUp(self):
        self.records = [job(1, 10, 10), job(2, 12, 12), job(3, 90, 90, "error"),
                        job(4, 11, 21), {"status": "cancelled"}]

    def test_drops_the_slowest_first_and_shortens_the_wall(self):
        cuts = stress.without_longest(self.records, 95.0, "ocr", 2, 10)
        # four jobs ran, so at most three can go and one is always kept
        self.assertEqual(len(cuts), 3)
        one = cuts[0]
        self.assertEqual([e["job"] for e in one["excluded"]], [3])
        self.assertEqual(one["excluded"][0]["status"], "error")
        self.assertEqual(one["attempted"], 3)
        self.assertEqual(one["error_rate"], 0.0)
        # the wall ends with the last kept job (#4 at 21 s), not the straggler
        self.assertEqual(one["wall_seconds"], 21.0)
        self.assertEqual([e["job"] for e in cuts[1]["excluded"]], [3, 2])

    def test_limit_and_nothing_to_rank(self):
        self.assertEqual(len(stress.without_longest(self.records, 95.0, "ocr", 2, 1)), 1)
        self.assertEqual(stress.without_longest(self.records, 95.0, "ocr", 2, 0), [])
        self.assertEqual(stress.without_longest([job(1, 5, 5)], 5.0, "ocr", 1, 10), [])

    def test_a_record_without_end_keeps_the_run_wall(self):
        records = [job(1, 10, None), job(2, 50, 50), job(3, 12, 30)]
        self.assertEqual(stress.without_longest(records, 60.0, "ocr", 2, 1)[0]["wall_seconds"], 60.0)

    def test_the_full_report_is_unchanged(self):
        before = stress.report(self.records, 95.0, "ocr", 2)
        stress.without_longest(self.records, 95.0, "ocr", 2, 10)
        self.assertEqual(stress.report(self.records, 95.0, "ocr", 2), before)

    def test_agentic_requests_are_dropped_from_each_cut(self):
        chat = {"step": "document", "in_tokens": 1, "out_tokens": 1, "job": 1}
        records = [dict(job(1, 10, 10), agentic_chats=[chat]),
                   dict(job(2, 20, 20), agentic_chats=[dict(chat, job=2)])]
        cut = stress.without_longest(records, 20.0, "extract", 1, 5)[0]
        self.assertNotIn("requests", cut["agentic_context"])
        self.assertEqual(cut["agentic_context"]["steps"][0]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
