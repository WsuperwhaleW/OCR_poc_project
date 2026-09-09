"""How many documents are in one file, and which pages are which.

**One upload is not one document.** A three-page file can be three separate
documents of three different types, three separate documents of the SAME type,
or one document that happens to run to three pages -- and the three want
completely different things from pass 2. Reading the first as one document puts
three issue dates and three totals in front of one eleven-key form, where the
model fills each key from whichever page it saw first; reading the third as
three documents asks a continuation page, which prints no heading and no
totals, for a form it cannot answer.

**Nothing here is asked of the model.** Same rule the type classifier is built
on, and it matters for the same reason one level up: this answer decides how
many FORMS are asked for and what text each of them sees, so a split the model
made would be a schema the model chose for itself. `app.py` may put an
AMBIGUOUS boundary to the model -- and checks the answer covers every page
exactly once before believing it -- exactly as it does for a type.

The unit is the PAGE, never a span of text. A document boundary in a scanned
file falls on a page break by construction: two documents are two pieces of
paper. Splitting mid-page would also make every page-indexed thing in this
project (the images the compare view holds, `page_stats`, the run log's
`pages`) mean something different from the transcript beside it.

The evidence, most certain first:

  1. **the page numbers itself** -- `Page 2 of 2`, `หน้า 1/3`. A page that says
     it is the second of two is a continuation and says so; a page that says it
     is the first of anything starts a document. This is the ONLY signal that
     gets both of this project's multi-page fixtures right, and each of the
     other two rules gets exactly one of them wrong:
       - sol004 page 2 REPRINTS the letterhead and the full heading
         `ใบเสร็จรับเงิน/ใบกำกับภาษี`, so "a page with its own heading opens a
         document" splits a document that is one. `Page 2 of 2` saves it.
       - sol012 page 2 is a `ใบนัดจ่าย`, a payment schedule the needle table
         cannot place at all, so "a page the classifier cannot type is a
         continuation" merges two documents into one. `Page 1 / 1` saves it.
  2. **the type** -- a page whose heading names other types is another document.
  3. **no heading at all** -- a page that does not say what it is is part of
     what came before. A continuation page is exactly the page with no heading.

What is left after those three is the genuinely hard case, and it is reported
rather than settled: **two pages of the same type, each with its own heading,
neither numbered**. Two receipts back to back and one receipt whose letterhead
is reprinted are the same text at this resolution, and no rule over the text
separates them. Those boundaries come back `certain=False` with a reason, and
the caller decides whether to spend a request on them.
"""

import re

# `app.py` joins the pages of one read with this, and `scoring.PAGE_MARKER`
# strips it before scoring. Parsed here rather than assumed away, because a
# transcript reaches pass 2 from two places -- a read that still has its page
# list, and `solution/<id>.md` read off disk, which carries the marker only
# where a page actually breaks.
PAGE_MARKER = re.compile(r"^[ \t]*-{2,}[ \t]*page[ \t]+(\d+)[ \t]*-{2,}[ \t]*$",
                         re.I | re.M)

# `Page 2 of 2`, `Page 1 / 1`, `หน้า 1/3`, `หน้าที่ 2 จาก 3`, `หน้า 2 ของ 3`.
#
# **The word is required, and that is the whole of what keeps this safe.** A
# bare `1 of 2` is a quantity on an invoice as often as it is a page number, and
# reading a line item as a page count would split a document down the middle of
# its own table.
_POSITION = re.compile(
    r"(?:page|หน้าที่|หน้า)\s*(?:ที่\s*)?(\d{1,3})\s*(?:/|of|จาก|ของ)\s*(\d{1,3})",
    re.I)

# How far into a page to look for that marker. It is printed in the head band
# beside the heading (sol004 line 12, sol012 line 6) or in a footer, so both
# ends are searched and the middle is not -- which keeps a stray `2 of 3` inside
# a body row out of it.
POSITION_SCAN_LINES = 25


def split_pages(text: str) -> list:
    """The joined transcript, back into one string per page.

    Returns `[text]` unchanged where there are no markers -- a single-page read,
    or a truth file that never breaks. **A one-element list is a real answer**,
    not a failure to split: it is what a one-page document looks like.

    **Pages are placed by the NUMBER in their marker, not by position in the
    split.** Positionally it would be one line shorter and wrong on two real
    shapes this project produces:

      * `app.py` writes `--- page 1 ---` above the first page, so splitting on
        the marker leaves an empty string in front of it -- a phantom page that
        every page number after it is then off by;
      * that same join drops a page whose transcript came back empty, so an
        empty page 2 of three leaves markers reading 1 and 3, and the pages
        after the gap would shift up.

    A gap is filled with "" so the list index really is the page number. Text
    before the first marker is page 1: that is the hand-written truth files'
    shape, which numbers only the breaks.
    """
    if not text:
        return [""]
    parts = PAGE_MARKER.split(text)
    if len(parts) == 1:
        return [text]
    numbered = {}
    for number, body in zip(parts[1::2], parts[2::2]):
        numbered[int(number)] = body.strip("\n")
    lead = parts[0].strip("\n")
    if lead:
        numbered[1] = (lead + "\n\n" + numbered[1]).strip("\n") if 1 in numbered else lead
    return [numbered.get(number, "") for number in range(1, max(numbered) + 1)]


def page_position(text: str):
    """`(index, total)` where the page numbers itself, else None.

    Both ends of the page are searched and nothing in between: the marker sits
    in the head band or in the footer, and a `2 of 3` in the middle of a table
    is a quantity.

    `total` of 0 is refused, and so is an index past its total: `หน้า 3 จาก 2`
    is a misread, and a misread page number that is believed splits a document
    in the wrong place -- which is worse than not splitting it at all.
    """
    lines = (text or "").splitlines()
    head = lines[:POSITION_SCAN_LINES]
    foot = lines[-POSITION_SCAN_LINES:] if len(lines) > POSITION_SCAN_LINES else []
    for line in head + foot:
        found = _POSITION.search(line)
        if not found:
            continue
        index, total = int(found.group(1)), int(found.group(2))
        if total and 1 <= index <= total:
            return index, total
    return None


def _same_types(left, right) -> bool:
    return sorted(left or []) == sorted(right or [])


# A number printed under a label that means "this document's number". Label-led
# on purpose: an unlabelled code is a customer number as often as a document
# number, and the whole value of this signal is that it is the same PRINTED
# identity on two pages rather than a coincidence of digits.
_DOC_NUMBER = re.compile(
    r"(?:เลขที่เอกสาร|ใบกำกับเลขที่|เอกสารเลขที่|เลขที่ใบกำกับ|เลขที่ใบวางบิล"
    r"|เลขที่การส่งคืน|เลขที่|เลขที|doc(?:ument)?\s*(?:no|number)"
    r"|invoice\s*no|running\s*no|no)\s*[.:]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9\-/]{2,})", re.I)

# The head band only. A document number is printed with the heading; the same
# regex over a charge table would pick up every row's reference and make two
# pages of one document look like two documents that share nothing.
NUMBER_SCAN_LINES = 25


def document_numbers(text: str) -> set:
    """The labelled document numbers printed near the top of a page.

    **Evidence that two pages are the same transaction**, which is the one
    question the layout cannot answer: a multi-page invoice reprints its own
    number on every page, and two invoices back to back print two different
    ones. Compared as SETS and never parsed further -- what matters is whether
    the same printed identity appears on both pages, not what it means.

    Must contain a digit: `เลขที่ ...` over a blank is a caption, and a purely
    alphabetic capture is a word that followed the label.
    """
    numbers = set()
    for line in (text or "").splitlines()[:NUMBER_SCAN_LINES]:
        for found in _DOC_NUMBER.finditer(line):
            value = found.group(1).strip(".:-/")
            if len(value) >= 3 and any(c.isdigit() for c in value):
                numbers.add(value.upper())
    return numbers


def _boundary(page: str, previous: str, seg: dict, codes):
    """Does this page open a new document? `(new, certain, why)`.

    `seg` is the document it would otherwise continue, so the type test compares
    this page with what the document was OPENED as rather than with the page in
    front of it -- which is what lets a middle page that prints no heading still
    be judged against the type its document started with.
    """
    here = page_position(page)
    there = page_position(previous)
    if here:
        index, total = here
        if index > 1:
            # It says it is not the first page of its document. Whether the page
            # before agrees is worth REPORTING -- `1 of 2` followed by `2 of 3`
            # is either two numbered documents or two misread markers -- but not
            # worth acting on: a page's claim about itself is the better
            # evidence, and the alternative is to split a document on a
            # disagreement between two readings of small print.
            agrees = bool(there and there[1] == total and there[0] == index - 1)
            return (False, True,
                    "the page numbers itself %d of %d%s"
                    % (index, total, "" if agrees else
                       ", though the page before it does not number itself %d of %d"
                       % (index - 1, total)))
        # `1 of N` on a page that is not the first of the file: a document
        # starts here, and the page is the thing saying so.
        return True, True, "the page numbers itself 1 of %d" % total
    if there and there[0] < there[1]:
        # The page before promised a page after it and this one is unnumbered:
        # the number was printed once, or the marker here was not read. The
        # promise is evidence and it is the page's own.
        return (False, True,
                "the page before it numbers itself %d of %d, so more follows"
                % there)
    if not codes:
        # It does not say what it is, and a continuation page is exactly that
        # page -- no heading, because its document already has one.
        #
        # **This is also the rule that would have merged sol012's two
        # documents**, and the page marker above is what stops it. The ORDER of
        # these tests is the whole of their correctness, not a tidying.
        #
        # **Continued, and NOT certain** (2026-09-08, at the user's request:
        # *some page need multipage to be complete, llm needs to decide is the
        # doc is the same transaction or not*). An unheaded page is the one
        # shape where "part of the document before it" and "another document
        # whose heading did not survive the read" look identical, and no rule
        # over the text separates them -- what separates them is whether the two
        # pages are the same TRANSACTION, which is a reading of the document
        # numbers, the parties and the totals rather than of the layout. So
        # Python continues it, which is right far more often than not, and says
        # the boundary is unsettled so the caller can put it to the model.
        return False, False, "the page heads itself with nothing"
    if not _same_types(codes, seg["codes"]):
        return (True, True,
                "it heads itself %s, and the document so far is %s"
                % (" + ".join(codes), " + ".join(seg["codes"]) or "untyped"))
    # Same types, its own heading, neither page numbered. **The layout has
    # nothing left to say, so the DOCUMENT NUMBER is asked instead** -- a
    # multi-page invoice reprints its own number on every page, and two invoices
    # back to back print two different ones. That is a reading of printed text
    # rather than a guess about it, which is why it is here and not left to the
    # model; what stays the model's is the verdict, since neither answer is
    # certain enough to settle a boundary on its own.
    here_numbers = document_numbers(page)
    there_numbers = document_numbers(previous)
    shared = here_numbers & there_numbers
    if shared:
        return (False, False,
                "it heads itself %s again and prints the same document number, %s"
                % (" + ".join(codes), ", ".join(sorted(shared))))
    if here_numbers and there_numbers:
        return (True, False,
                "it heads itself %s again and prints a different document "
                "number, %s against %s"
                % (" + ".join(codes), ", ".join(sorted(here_numbers)),
                   ", ".join(sorted(there_numbers))))
    # Neither page names itself. It SPLITS, and that is the direction that loses
    # least when it is wrong: a continuation page split off wrongly still
    # extracts whatever it prints, and says on the page that it was a guess --
    # while two documents merged wrongly answer one form out of two documents'
    # figures and nothing anywhere says so.
    return (True, False,
            "it heads itself %s again, and neither page is numbered"
            % " + ".join(codes))


def segment(pages, classify, whole: str = None) -> list:
    """The pages of one file, grouped into documents.

    `classify(text)` is injected rather than imported: the type classifier lives
    in `app.py` because it is the thing the model may be asked to second-guess,
    and a second copy here would be a second answer to drift from the first. It
    returns `(codes, heading, confidence, detail)` -- `app.classify_transcript`'s
    own shape.

    Always at least one segment. A one-page file is one document and is reported
    as one, so **the single-document case is a real answer from this function
    rather than a path around it**: every caller reads a list and there is no
    second shape for one of them to get wrong.
    """
    pages = [p if p is not None else "" for p in (pages or [""])]
    segments = []
    for number, page in enumerate(pages, 1):
        codes, heading, confidence, _ = classify(page)
        if not segments:
            segments.append(_open(number, page, codes, heading, confidence,
                                  "the first page of the file"))
            continue
        seg = segments[-1]
        new, certain, why = _boundary(page, pages[number - 2], seg, codes)
        if new:
            segments.append(_open(number, page, codes, heading, confidence,
                                  why, certain))
            continue
        seg["pages"].append(number)
        seg["page_texts"].append(page)
        # `certain` on a JOIN as well as on an opening, because the two are one
        # question asked from opposite ends: an unheaded page KEPT with the
        # document before it is exactly as unsettled as a headed page SPLIT off
        # from it, and only the second used to be reportable. What settles
        # either is whether the two pages are the same transaction.
        seg["joins"].append({"page": number, "why": why, "certain": certain})
        # A continuation page can carry the heading its document never printed
        # on page 1. Taken only where there is nothing to overwrite: the first
        # heading is the document's own, and one reprinted on page 2 must not
        # relabel it.
        if codes and not seg["codes"]:
            seg["codes"], seg["heading"] = list(codes), heading
            seg["confidence"] = confidence
    for seg in segments:
        seg["text"] = _join(seg)
    if len(segments) == 1 and whole is not None:
        # **A file that is one document hands pass 2 the transcript it was
        # given, byte for byte.** Rebuilt from its pages it would not be the
        # same string: `app.py` writes `--- page 1 ---` above the first page of
        # a multi-page read and a hand-written truth file does not, so a round
        # trip through here would quietly change what sol004 is extracted from
        # -- and every truth-fed number in CLAUDE.md was taken on the file as
        # written. The one-document case has to be a no-op, not a rebuild that
        # usually agrees.
        segments[0]["text"] = whole
    return segments


def _open(number, page, codes, heading, confidence, why, certain=True) -> dict:
    return {"pages": [number], "page_texts": [page], "codes": list(codes or []),
            "heading": heading, "confidence": confidence,
            # Why this document was opened where it was, and whether that is a
            # reading of the page or a guess. Both travel to the browser and on
            # to the result: a file that came back as three documents when it is
            # one has to be able to say on what evidence.
            "start_reason": why, "certain": bool(certain),
            # Why each later page was kept, one entry per page after the first.
            "joins": []}


def _join(seg: dict) -> str:
    """A segment's pages, joined the way `app.py` joins a whole read's.

    **The page numbers are the FILE's, not the segment's**, so the first page of
    a document that starts on page 2 is still `--- page 2 ---`. The marker names
    where the text came from, and renumbering it would put the transcript out of
    step with the page images the compare view holds under the same numbers.
    """
    if len(seg["page_texts"]) == 1:
        return seg["page_texts"][0]
    return "\n\n".join("--- page %d ---\n%s" % (number, text)
                       for number, text in zip(seg["pages"], seg["page_texts"])
                       if text)


def one(pages, classify, whole: str = None, why: str = "read as one document") -> list:
    """Every page as one document, whatever the pages say.

    The answer where nothing was asked -- a one-page file, the setting off, a
    file too long to put a boundary question about. It is built here rather than
    by the caller so that a run that did not split still carries a segment with
    the same fields as one that did, and no reader downstream needs a second
    shape for it. `why` goes on `start_reason`, so the reason it was not split
    is on the record beside the reasons a split one was.
    """
    pages = [p if p is not None else "" for p in (pages or [""])]
    codes, heading, confidence, _ = classify(pages[0])
    seg = _open(1, pages[0], codes, heading, confidence, why)
    for number, page in enumerate(pages[1:], 2):
        seg["pages"].append(number)
        seg["page_texts"].append(page)
        seg["joins"].append({"page": number, "why": why})
        if not seg["codes"]:
            codes, heading, confidence, _ = classify(page)
            if codes:
                seg["codes"], seg["heading"] = list(codes), heading
                seg["confidence"] = confidence
    seg["text"] = whole if whole is not None else _join(seg)
    return [seg]


def page_range(seg: dict) -> str:
    """`3` or `2-4` -- which pages of the file this document is."""
    pages = seg["pages"]
    return str(pages[0]) if len(pages) == 1 else "%d-%d" % (pages[0], pages[-1])


def describe(segments) -> str:
    """One line saying what the split found, for a log line or a status note."""
    if len(segments) <= 1:
        return "one document"
    return "%d documents: %s" % (len(segments),
                                 ", ".join("pages " + page_range(s)
                                           if len(s["pages"]) > 1
                                           else "page " + page_range(s)
                                           for s in segments))


def uncertain(segments) -> list:
    """The segments carrying a boundary nothing on the page settled, in file order.

    What `app.py` may put to the model, and it is BOTH kinds of unsettled
    boundary (2026-09-08):

    * an OPENING that was a guess -- the same types again, neither page
      numbered, which is two receipts in a row or one receipt whose letterhead
      is reprinted;
    * a JOIN that was a guess -- an unheaded page kept with the document before
      it, which is a continuation or a document whose heading did not survive
      the read.

    Both are the same question -- **are these two pages the same transaction** --
    and it is not a question about layout, so no rule here answers it. A segment
    opened on a page marker or on a change of type is not in this list however
    unusual it looks: those are readings of what the page prints, and a request
    that could only disagree with printed text can only make the answer worse.
    """
    return [s for s in segments
            if not s["certain"]
            or any(not j.get("certain", True) for j in s.get("joins") or [])]


def unsettled(segments) -> list:
    """One line per boundary nothing settled, for a status line or a log.

    Named apart from `uncertain` because they answer different questions: that
    one is "is a request worth making", this one is "what would it be about".
    """
    lines = []
    for seg in segments:
        if not seg["certain"]:
            lines.append("page %d opens a document: %s"
                         % (seg["pages"][0], seg["start_reason"]))
        for join in seg.get("joins") or []:
            if not join.get("certain", True):
                lines.append("page %d continues the document before it: %s"
                             % (join["page"], join["why"]))
    return lines


def apply_groups(pages, groups, classify) -> list:
    """Rebuild the segments from an explicit grouping of page numbers.

    How an answer from the model is taken: it names the groups, and the segments
    are then built here so a model-decided split and a Python-decided one are
    the same object with the same fields. **The grouping is not trusted to be
    sane** -- `valid_groups` below is what checks it, and this assumes it passed.
    """
    segments = []
    for group in groups:
        first = group[0]
        codes, heading, confidence, _ = classify(pages[first - 1])
        seg = _open(first, pages[first - 1], codes, heading, confidence,
                    "the model read the file as %d documents" % len(groups))
        for number in group[1:]:
            seg["pages"].append(number)
            seg["page_texts"].append(pages[number - 1])
            seg["joins"].append({"page": number,
                                 "why": "grouped with page %d by the model" % first})
            if not seg["codes"]:
                codes, heading, confidence, _ = classify(pages[number - 1])
                if codes:
                    seg["codes"], seg["heading"] = list(codes), heading
                    seg["confidence"] = confidence
        seg["text"] = _join(seg)
        segments.append(seg)
    return segments


def valid_groups(groups, count: int):
    """The grouping, cleaned, or None if it is not one.

    Four things are demanded, and each is a way a plausible-looking answer would
    quietly lose or duplicate a page of the file:

      * every group is a non-empty list of page numbers in range;
      * **every page appears exactly once** -- a page in two documents would be
        extracted twice and a page in none would be silently dropped;
      * **every group is CONSECUTIVE** -- pages 1 and 3 are not one document
        with page 2 inside it belonging to another, and a model that answers so
        has not understood the question;
      * the groups are in file order, so document 1 is the one at the front.

    Returned cleaned rather than as a yes/no, because the caller needs the
    normalised form and re-deriving it separately is how the check and the thing
    checked drift apart.
    """
    if not isinstance(groups, list) or not groups:
        return None
    cleaned, seen = [], set()
    for group in groups:
        if isinstance(group, int):
            group = [group]
        if not isinstance(group, list) or not group:
            return None
        numbers = []
        for page in group:
            if not isinstance(page, int) or not 1 <= page <= count:
                return None
            if page in seen:
                return None
            seen.add(page)
            numbers.append(page)
        numbers.sort()
        if numbers != list(range(numbers[0], numbers[-1] + 1)):
            return None
        cleaned.append(numbers)
    if len(seen) != count:
        return None
    cleaned.sort(key=lambda g: g[0])
    return cleaned


def _selftest():
    """Asserted at import, like `prompts._selftest`.

    The two fixtures are in here as the shapes they are rather than by name: a
    reprinted heading over `Page 2 of 2`, and an untyped page over `Page 1 / 1`.
    They are the two cases this module exists for and each defeats one of the
    other rules, so a change that breaks either is a change that has undone the
    reason for the ordering in `_boundary`.
    """
    def classify(text):
        for code in ("RECEIPT", "CREDIT_NOTE"):
            if code.lower().replace("_", " ") in text.lower():
                return [code], code.lower(), 1.0, {}
        return [], "", 0.0, {}

    assert page_position("Page 2 of 2") == (2, 2)
    assert page_position("หน้า 1/3") == (1, 3)
    assert page_position("หน้าที่ 2 จาก 3") == (2, 3)
    # No word, so no page number: this is a quantity.
    assert page_position("| widget | 1 of 2 |") is None
    # A misread that would split in the wrong place.
    assert page_position("Page 3 of 2") is None

    # sol004's shape: the heading is reprinted, and the marker is what says the
    # document is one.
    four = ["receipt\nPage 1 of 2\nlines", "receipt\nPage 2 of 2\ntotals"]
    assert len(segment(four, classify)) == 1
    # sol012's shape: page 2 is a kind the table cannot place, and the marker is
    # what says it is its own document.
    twelve = ["receipt\nlines and totals", "a schedule\nPage 1 / 1\nrows"]
    split = segment(twelve, classify)
    assert len(split) == 2 and split[1]["pages"] == [2], split
    # A continuation page with no heading of its own: kept, and NOT settled --
    # whether it is the same transaction is not a question about layout.
    cont = segment(["receipt\ntotals", "more rows"], classify)
    assert len(cont) == 1 and cont[0]["pages"] == [1, 2], cont
    assert cont[0]["joins"][0]["certain"] is False
    assert len(uncertain(cont)) == 1 and len(unsettled(cont)) == 1
    # A numbered continuation is settled and is not put to the model.
    numbered = segment(["receipt Page 1 of 2", "Page 2 of 2\nmore rows"], classify)
    assert len(numbered) == 1 and not uncertain(numbered), numbered
    # A different type opens a document, certainly.
    two = segment(["receipt\nx", "credit note\ny"], classify)
    assert len(two) == 2 and two[1]["certain"]
    # The same type twice, unnumbered and naming no document number: split, and
    # say it was a guess.
    same = segment(["receipt\nx", "receipt\ny"], classify)
    assert len(same) == 2 and not same[1]["certain"]
    assert len(uncertain(same)) == 1
    # The same type twice printing the SAME document number: one document, and
    # still a guess -- whether two pages are one transaction is the model's.
    one = segment(["receipt\nเลขที่ RC-7\nrows", "receipt\nเลขที่ RC-7\nmore"],
                  classify)
    assert len(one) == 1 and one[0]["pages"] == [1, 2], one
    assert not one[0]["joins"][0]["certain"]
    # The same type twice printing DIFFERENT document numbers: two documents.
    two_n = segment(["receipt\nเลขที่ RC-7", "receipt\nเลขที่ RC-8"], classify)
    assert len(two_n) == 2, two_n
    assert document_numbers("ใบเสร็จรับเงิน\nเลขที่ RC-7") == {"RC-7"}
    # A caption over nothing, and a word after the label, are not numbers.
    assert document_numbers("เลขที่\nNo. :") == set()
    assert document_numbers("Invoice No. draft") == set()

    # One document hands back the string it was given, markers and all.
    marked = ("--- page 1 ---\nreceipt Page 1 of 2"
              "\n\n--- page 2 ---\nPage 2 of 2")
    once = segment(split_pages(marked), classify, whole=marked)
    assert len(once) == 1 and once[0]["text"] == marked, once

    assert split_pages("a\n\n--- page 2 ---\nb") == ["a", "b"]
    assert split_pages("only") == ["only"]
    # `app.py`'s own join, which numbers the first page too: no phantom page 0.
    assert split_pages("--- page 1 ---\na\n\n--- page 2 ---\nb") == ["a", "b"]
    # A page whose transcript came back empty is dropped from the join, and the
    # pages after the gap must not shift up into it.
    assert split_pages("--- page 1 ---\na\n\n--- page 3 ---\nc") == ["a", "", "c"]
    # The file's own page numbers survive the split.
    assert "--- page 2 ---" in _join({"pages": [2, 3],
                                      "page_texts": ["b", "c"]})

    assert valid_groups([[1, 2], [3]], 3) == [[1, 2], [3]]
    assert valid_groups([[3], [1, 2]], 3) == [[1, 2], [3]]
    assert valid_groups([[1, 3], [2]], 3) is None      # not consecutive
    assert valid_groups([[1, 2]], 3) is None           # page 3 lost
    assert valid_groups([[1, 1], [2]], 2) is None      # page 1 twice
    assert valid_groups([[1], [4]], 3) is None         # out of range
    assert valid_groups([], 1) is None


_selftest()
