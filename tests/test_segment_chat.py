"""The rolling boundary walk: one question per unsettled boundary, with a memory.

`segment._selftest` already covers the WALK -- which boundaries are offered to
the asker, that an answer is acted on in both directions, and that the memory is
thrown away at every cut. What is here is the other half: `app._SegmentChat`,
the object that holds the conversation and talks to the model server.

Four things, in the order they matter:

* **the reset**, seen from inside the request: the messages a question carries
  after a cut hold the new document's opening page and nothing of the old one.
  This is the whole design, and the failure it prevents is silent -- document 7
  judged partly against document 1's parties and totals;
* **the window**, which is what makes a long document answerable at all: both
  backends are stateless, so every turn is resent, and the opening page is the
  one turn that is never allowed to fall off;
* **the guards** -- a quote that is not printed on the page, a reply that is not
  JSON, a missing `same`, an HTTP error. Each discards the answer WHOLE and
  leaves Python's own reading standing, so a refusal costs one boundary;
* **that no number is read**, here as anywhere in this project.

The model server is stubbed at `requests.post`, so nothing here needs one --
which also means `runlog.live_transport` would refuse to write a row from this
process, which is the standing rule after the stubbed rows of 2026-09-03.
"""
import json
import unittest
from unittest.mock import patch

import app
import prompts
import segment


STATUS = {"kind": "ollama", "model": "stub", "url": "http://stub"}


class Reply:
    """Enough of a `requests` response for `backends.structured_reply`."""

    status_code = 200

    def __init__(self, text):
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def answer(same, evidence):
    return Reply(json.dumps({"same": same, "evidence": evidence},
                            ensure_ascii=False))


class Server:
    """A stub that answers in order and keeps what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, url, json=None, timeout=None, **kw):
        self.sent.append(json)
        return self.replies.pop(0) if self.replies else Reply("{}")

    def messages(self, index=-1):
        return self.sent[index]["messages"]

    def text(self, index=-1):
        return "\n".join(m["content"] for m in self.messages(index))


def classify(text):
    for code in ("RECEIPT", "CREDIT_NOTE"):
        if code.lower().replace("_", " ") in text.lower():
            return [code], code.lower(), 1.0, {}
    return [], "", 0.0, {}


def walk(pages, server):
    """Run the walk over `pages` with the stub in place. Returns (groups, chat)."""
    chat = app._SegmentChat(STATUS)
    with patch.object(app.requests, "post", server):
        segments = segment.segment(list(pages), classify, ask=chat)
    return [s["pages"] for s in segments], chat


class MemoryTests(unittest.TestCase):
    def test_a_cut_throws_the_previous_document_away(self):
        """The reset, seen from inside the request rather than assumed.

        Pages 1-2 are one receipt, 3-4 a credit note. The boundary at 3 is a
        type change, which Python settles, so the only questions are at 2 and 4
        -- and the question at 4 must carry page 3 and nothing of pages 1-2.
        """
        server = Server(answer(True, "receipt A"), answer(True, "credit note B"))
        groups, chat = walk(["receipt A total 100", "receipt A page two",
                             "credit note B total 200", "credit note B more"],
                            server)
        self.assertEqual(groups, [[1, 2], [3, 4]])
        self.assertEqual(chat.asked, 2)
        self.assertEqual(chat.answered, 2)

        first, second = server.text(0), server.text(1)
        self.assertIn("receipt A total 100", first)
        # The second document's question knows its own opening page ...
        self.assertIn("credit note B total 200", second)
        # ... and nothing whatever of the one before it.
        self.assertNotIn("receipt A total 100", second)
        self.assertNotIn("receipt A page two", second)

    def test_a_settled_page_is_in_the_memory_and_costs_no_request(self):
        """Python's readings go in as USER turns, never as assistant ones.

        A page the rules settled still has to be in the conversation or the next
        question is asked against a document missing its middle -- but putting
        Python's reading in the model's mouth would make the next answer
        conditioned on words it did not use.
        """
        server = Server(answer(True, "receipt A opening"))
        groups, chat = walk(["receipt A opening", "Page 2 of 3", "Page 3 of 3",
                             "receipt A opening"], server)
        self.assertEqual(groups, [[1, 2, 3, 4]])
        # One question: the boundaries at 2 and 3 are settled by their markers.
        self.assertEqual(chat.asked, 1)
        sent = server.messages(0)
        self.assertIn("Page 2 of 3", server.text(0))
        self.assertIn("Page 3 of 3", server.text(0))
        # Nothing in the conversation claims the model said anything yet.
        self.assertEqual([m for m in sent if m["role"] == "assistant"], [])

    def test_the_opening_page_never_falls_out_of_the_window(self):
        """What identifies a document is its first page. The middle drops."""
        pages = ["receipt OPENING-MARKER"] + [
            "receipt filler %d" % n for n in range(2, 16)]
        server = Server(*[answer(True, "receipt filler %d" % n)
                          for n in range(2, 16)])
        groups, chat = walk(pages, server)
        self.assertEqual(groups, [list(range(1, 16))])
        last = server.messages(-1)
        self.assertIn("OPENING-MARKER", server.text(-1))
        # Bounded: the conversation does not grow with the document.
        self.assertLessEqual(len(last), app.SEGMENT_CHAT_WINDOW + 2)
        # And an early middle page really has dropped out.
        self.assertNotIn("receipt filler 2", server.text(-1))


class GuardTests(unittest.TestCase):
    """Each of these must leave Python's own reading standing, unchanged.

    Python SPLITS two same-type unnumbered pages, and says it was a guess, so a
    refused answer is visible as `certain is False` rather than only as a count.
    """

    def refusal(self, reply):
        server = Server(reply)
        chat = app._SegmentChat(STATUS)
        with patch.object(app.requests, "post", server):
            segments = segment.segment(["receipt A one", "receipt B two"],
                                       classify, ask=chat)
        self.assertEqual([s["pages"] for s in segments], [[1], [2]])
        self.assertIs(segments[1]["certain"], False)
        self.assertEqual((chat.asked, chat.answered, chat.refused), (1, 0, 1))

    def test_a_quote_that_is_not_on_the_page_is_refused(self):
        # Fluent, well-formed, and pointing at nothing. This is the guard that
        # makes acting on a sequential answer safe at all.
        self.refusal(answer(False, "INVOICE No. 12345"))

    def test_an_empty_quote_is_refused(self):
        self.refusal(answer(False, ""))

    def test_a_reply_that_is_not_json_is_refused(self):
        self.refusal(Reply("Yes, page 2 starts a new document."))

    def test_a_missing_verdict_is_refused(self):
        self.refusal(Reply(json.dumps({"evidence": "receipt B two"})))

    def test_a_verdict_that_is_not_a_boolean_is_refused(self):
        # "probably" is not an answer, and neither is 0.9.
        self.refusal(Reply(json.dumps({"same": "probably",
                                       "evidence": "receipt B two"})))
        self.refusal(Reply(json.dumps({"same": 0.9,
                                       "evidence": "receipt B two"})))

    def test_an_http_error_is_refused(self):
        bad = Reply("{}")
        bad.status_code = 500
        self.refusal(bad)

    def test_a_raising_server_is_refused(self):
        def boom(*a, **kw):
            raise OSError("connection reset")
        chat = app._SegmentChat(STATUS)
        with patch.object(app.requests, "post", boom):
            segments = segment.segment(["receipt A one", "receipt B two"],
                                       classify, ask=chat)
        self.assertEqual([s["pages"] for s in segments], [[1], [2]])
        self.assertEqual(chat.refused, 1)

    def test_a_quote_is_matched_the_way_every_extracted_value_is(self):
        """Loose about presentation, strict about content -- `grounding.squash`.

        A quote differing only in spacing and punctuation is the same quote, and
        refusing it would throw away correct answers for the way they were
        typed. It is quoted off page 2, which is the page being ASKED about --
        evidence for "same document" is the thing on THIS page that says so.
        """
        server = Server(answer(True, "receipt  B,  two"))
        groups, chat = walk(["receipt A one", "receipt B two"], server)
        self.assertEqual(groups, [[1, 2]])
        self.assertEqual(chat.answered, 1)


class BudgetTests(unittest.TestCase):
    def test_the_ask_cap_stops_asking_and_says_so(self):
        """Reached, the rest of the walk is Python's reading -- and reported."""
        pages = ["receipt %d" % n for n in range(1, 6)]
        server = Server(*[answer(True, "receipt %d" % n) for n in range(2, 6)])
        chat = app._SegmentChat(STATUS)
        with patch.object(app, "SEGMENT_CHAT_MAX_ASKS", 2), \
             patch.object(app.requests, "post", server):
            segment.segment(pages, classify, ask=chat)
        self.assertEqual(chat.asked, 2)
        self.assertTrue(chat.capped)
        self.assertIn("SEGMENT_CHAT_MAX_ASKS", chat.note())

    def test_a_walk_that_asked_nothing_says_so(self):
        server = Server()
        groups, chat = walk(["receipt A", "Page 2 of 2"], server)
        self.assertEqual(groups, [[1, 2]])
        self.assertEqual(chat.asked, 0)
        self.assertEqual(server.sent, [])
        self.assertEqual(chat.note(), "every boundary was read off the pages")


class ProvenanceTests(unittest.TestCase):
    """`resolve_segments` must never claim the model settled what it did not."""

    def long_file(self, *replies):
        pages = ["receipt %d" % n for n in range(1, 26)]
        text = "\n\n".join("--- page %d ---\n%s" % (n, p)
                           for n, p in enumerate(pages, 1))
        server = Server(*replies)
        with patch.object(app.requests, "post", server):
            return app.resolve_segments(text, pages, STATUS)

    def test_a_long_file_is_walked_rather_than_read_as_one(self):
        segments, how = self.long_file(
            *[answer(True, "receipt %d" % n) for n in range(2, 26)])
        self.assertEqual(how, "chat")
        self.assertEqual([s["pages"] for s in segments], [list(range(1, 26))])

    def test_a_walk_nobody_answered_is_a_guess_not_a_chat(self):
        # Every question refused: the answer IS Python's own reading, and the
        # provenance has to say so or a split nobody can check reads as one the
        # model stood behind.
        segments, how = self.long_file(
            *[Reply("nonsense") for _ in range(2, 26)])
        self.assertEqual(how, "guess")

    def test_with_the_walk_off_a_long_file_is_one_document(self):
        pages = ["receipt %d" % n for n in range(1, 26)]
        text = "\n\n".join(pages)
        with patch.object(app, "SEGMENT_CHAT", False):
            segments, how = app.resolve_segments(text, pages, STATUS)
        self.assertEqual(how, "whole")
        self.assertEqual(len(segments), 1)
        self.assertIn("SEGMENT_MAX_PAGES", segments[0]["start_reason"])

    def test_with_no_server_a_long_file_is_one_document(self):
        pages = ["receipt %d" % n for n in range(1, 26)]
        segments, how = app.resolve_segments("\n\n".join(pages), pages, None)
        self.assertEqual(how, "whole")

    def test_a_short_file_is_untouched_by_any_of_this(self):
        """The one-shot path and the rules path, exactly as before."""
        pages = ["receipt A", "Page 2 of 2"]
        segments, how = app.resolve_segments("\n\n".join(pages), pages, STATUS)
        self.assertEqual(how, "rules")
        self.assertEqual([s["pages"] for s in segments], [[1, 2]])


class PromptTests(unittest.TestCase):
    def test_the_walk_never_asks_for_a_number(self):
        """The standing rule, asserted of this prompt as `prompts._selftest`
        asserts it of the other two: a model never exports a figure."""
        for text in (prompts.SEGMENT_CHAT_PROMPT, prompts.SEGMENT_CHAT_ASK,
                     prompts.SEGMENT_CHAT_KEPT):
            low = text.lower()
            for word in ("confidence", "certainty", "probability", "percent",
                         "how sure", "score", "%"):
                self.assertNotIn(word, low)

    def test_no_number_in_a_reply_is_ever_read(self):
        """A reply volunteering one has it dropped on the floor."""
        server = Server(Reply(json.dumps(
            {"same": True, "evidence": "receipt B two",
             "confidence": 0.91, "score": 88})))
        groups, chat = walk(["receipt A one", "receipt B two"], server)
        self.assertEqual(groups, [[1, 2]])
        segments_note = chat.note()
        self.assertNotIn("0.91", segments_note)
        self.assertNotIn("88", segments_note)


if __name__ == "__main__":
    unittest.main()
