"""The type-change gate, and the page-by-page grouping stream (2026-09-16).

Two changes that ship together because the second is how the first is seen:

* **the gate** -- a change of type the classifier is under
  `CLASSIFY_MIN_CONFIDENCE` about is no longer a CERTAIN cut. It is still a cut,
  and it is now a question. **On the walk only**: gated on a short file, one
  unsure boundary sent the whole file to the one-shot question, and qwen3.5:9b
  then regrouped sol015 and sol022 wrongly where the rules alone are right;
* **the stream** -- `segment.walk` yields one step per page, and
  `app._segment_stream` forwards each as a `segment_page` event carrying how many
  pages remain, the session the page landed in, and the head of the page, so the
  Grouping view can be checked while the run is still going.

The model server is stubbed at `requests.post`, as in `test_segment_chat`.
"""
import unittest
from unittest.mock import patch

import app
import segment
from tests.test_segment_chat import STATUS, Server, answer


def classify(text):
    """A receipt the table is sure of; a credit note it is not."""
    low = text.lower()
    if "credit note" in low:
        return ["CREDIT_NOTE"], "credit note", 0.5, {}
    if "receipt" in low:
        return ["RECEIPT"], "receipt", 1.0, {}
    return [], "", 0.0, {}


def drain(gen):
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as done:
            return events, done.value


def seg(codes, confidence):
    return {"pages": [1], "codes": codes, "confidence": confidence,
            "numbers": set()}


class GateTests(unittest.TestCase):
    def boundary(self, page_conf, doc_conf, bar):
        return segment._boundary("credit note", "receipt",
                                 seg(["RECEIPT"], doc_conf), ["CREDIT_NOTE"],
                                 page_conf, bar)

    def test_an_unsure_change_of_type_is_still_a_cut_but_a_guess(self):
        new, certain, why = self.boundary(0.5, 1.0, 0.9)
        self.assertTrue(new)
        self.assertFalse(certain)
        self.assertIn("guess", why)

    def test_a_sure_change_of_type_is_settled(self):
        self.assertEqual(self.boundary(0.95, 1.0, 0.9)[:2], (True, True))

    def test_an_unsure_DOCUMENT_type_counts_as_well(self):
        self.assertFalse(self.boundary(1.0, 0.4, 0.9)[1])

    def test_no_bar_is_no_gate(self):
        self.assertEqual(self.boundary(0.1, 0.1, 0.0)[:2], (True, True))

    def test_an_unmeasured_confidence_is_not_held_against_it(self):
        self.assertTrue(segment._confident(None, None, 0.9))


class WalkTests(unittest.TestCase):
    PAGES = ["receipt", "Page 2 of 2", "credit note"]

    def test_one_step_per_page_and_the_same_answer_as_segment(self):
        steps, segments = drain(segment.walk(self.PAGES, classify))
        self.assertEqual([s["page"] for s in steps], [1, 2, 3])
        self.assertEqual(segments, segment.segment(self.PAGES, classify))

    def test_a_step_says_which_session_the_page_landed_in(self):
        steps, _ = drain(segment.walk(self.PAGES, classify))
        self.assertEqual([s["document"] for s in steps], [1, 1, 2])
        self.assertEqual([s["opened"] for s in steps], [True, False, True])
        self.assertEqual(steps[1]["doc_pages"], [1, 2])
        self.assertEqual(steps[2]["doc_codes"], ["CREDIT_NOTE"])


class StreamTests(unittest.TestCase):
    def stream(self, pages, server):
        text = "\n\n".join("--- page %d ---\n%s" % (n, p)
                           for n, p in enumerate(pages, 1))
        with patch.object(app, "classify_transcript", classify), \
                patch.object(app.requests, "post", server):
            return drain(app._segment_stream(text, pages, STATUS))

    def test_a_short_file_is_not_gated_and_asks_nothing(self):
        server = Server()
        events, (segments, how) = self.stream(["receipt", "credit note"], server)
        self.assertEqual(how, "rules")
        self.assertEqual(server.sent, [])
        self.assertEqual([s["pages"] for s in segments], [[1], [2]])
        self.assertEqual([e["event"] for e in events],
                         ["segmenting", "segment_page", "segment_page"])
        self.assertFalse(events[0]["walk"])
        self.assertNotIn("asked_total", events[-1])

    def test_a_long_file_asks_about_the_unsure_change_only(self):
        pages = (["receipt Page %d of 12" % n for n in range(1, 13)]
                 + ["credit note"]
                 + ["credit note Page %d of 13" % n for n in range(2, 14)])
        server = Server(answer(False, "credit note"))
        with patch.object(app, "SEGMENT_MAX_PAGES", 20):
            events, (segments, how) = self.stream(pages, server)
        self.assertEqual(len(server.sent), 1)
        self.assertEqual(how, "chat")
        self.assertEqual([s["pages"] for s in segments],
                         [list(range(1, 13)), list(range(13, 26))])
        steps = [e for e in events if e["event"] == "segment_page"]
        self.assertTrue(events[0]["walk"])
        self.assertEqual([s["remaining"] for s in steps], list(range(24, -1, -1)))
        self.assertEqual(steps[12]["asked"], True)
        self.assertEqual(steps[-1]["asked_total"], 1)
        self.assertEqual(steps[-1]["answered_total"], 1)
        self.assertEqual(steps[12]["head"], "credit note")


if __name__ == "__main__":
    unittest.main()
