"""Normalised and derived values, computed in Python from what pass 2 copied.

Deterministic and model-free, like `grounding` and `verify`. Everything here is
worked out from values the extraction already copied verbatim, plus the
transcript -- nothing in this module is ever asked of the model, and that is the
whole point of it.

The OCR field requirement asks for several values that are *classifications*
rather than readings:

* a standard document type -- INVOICE, CREDIT_NOTE, RECEIPT, TAX_INVOICE,
  RECEIPT_TAX_INVOICE -- kept separate from the raw printed heading, because a
  real page heads itself INVOICE, Statement of Account, or
  ใบเสร็จรับเงิน/ใบกำกับภาษี, and the requirement asks for both the raw title and
  the normalised code in as many words;
* a standard branch **code**, with สำนักงานใหญ่ / Head Office normalised to 00000;
* a tax ID reduced to digits;
* a service period split into its two ends;
* references collected as a LIST, because one receipt legitimately settles
  several invoices and the requirement rules out packing them into one text
  field.

None of that may be asked of the extraction pass. Pass 2 is a lookup: it copies
printed values onto keys and reasons about nothing, and the reason is recorded at
length in CLAUDE.md -- a 2B model asked to classify while it reads starts
adjusting what it reads to fit the classification, and a conclusion cannot be
grounded against the page the way a copied value can. So the model copies the
heading and the branch line as printed, `grounding.py` checks those against the
transcript, and everything below is derived from them afterwards.

That split is also why these values ride on the extraction result as `derived`
rather than being merged into `fields`: `grounding.check` walks `fields`, and a
computed value has nothing to be grounded against. It would be reported as
`ungrounded` -- an invention -- when it is the one class of value that provably
is not one.
"""

import re

import grounding
import prompts
import verify

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def _needles(*words):
    """The needles below, reduced the same way the text they are tested against is.

    Not decoration: `grounding.squash` keeps only alphanumerics, and every Thai
    vowel and tone mark is a combining character that `str.isalnum` rejects. So
    สำนักงานใหญ่ squashes to สานกงานใหญ, and a needle written the way a human
    types it matches nothing at all. Squashing both sides at import is what makes
    the tables below readable in their printed form and correct at run time.
    """
    return tuple(grounding.squash(w) for w in words)


# --------------------------------------------------------------------------
# document type
# --------------------------------------------------------------------------

# Raw printed heading -> the standard codes it names. **A document can be more
# than one type at once, and in this corpus six of the ten are.**
# `ใบเสร็จรับเงิน/ใบกำกับภาษี` is a receipt AND a tax invoice;
# `ใบลดหนี้/ใบกำกับภาษี` is a credit note AND a tax invoice. That is not a
# quirk of these fixtures -- Thai practice routinely issues one page that
# discharges two documents, and the slash in the heading is the page saying so.
#
# **This table used to return the FIRST match and stop, and that was wrong in two
# separate ways** (both found 2026-08-31, by the user):
#
#  * a compound entry `RECEIPT_TAX_INVOICE` papered over one combination by
#    hand, which only works for combinations somebody thought of. It did not
#    exist for `ใบลดหนี้/ใบกำกับภาษี`, so sol006 matched CREDIT_NOTE, stopped,
#    and its tax-invoice half was **silently dropped** -- taking with it every
#    field a tax invoice is required to carry;
#  * `statement of account` was a needle under INVOICE, so sol001 -- which heads
#    itself `ใบแจ้งหนี้ STATEMENT OF ACCOUNT` -- came back INVOICE and the
#    statement half disappeared the same way.
#
# So every entry is now tested and ALL that match are returned. There are no
# compound codes: a receipt/tax-invoice is `["RECEIPT", "TAX_INVOICE"]`, which
# composes for combinations nobody enumerated, and the field set is the union of
# the two forms (`prompts.fields_for_types`). Order in this list is the order the
# codes come back in -- most specific first, so `primary_type` reads off the
# front -- but it no longer decides which single answer wins, because there is
# no longer a single answer.
#
# Each entry is (code, all-of, any-of): the code applies when every needle in
# all-of is present and at least one from any-of is, an empty any-of meaning
# nothing further is required. Needles are matched against the squashed heading,
# so spacing, punctuation and slashes do not matter.
DOCUMENT_TYPES = [
    # FIRST, and the order matters here more than anywhere else in this table:
    # a withholding tax certificate shares not one field with the commercial
    # documents below it, so a page that is one is nothing else, and it is what
    # `primary_type` should file it under whatever else its heading happens to
    # brush against.
    #
    # The Thai section number is a needle in its own right because the printed
    # form carries it on its own line under the heading -- and it survives
    # squashing as "50ทว", the vowel being a combining mark that
    # `grounding.squash` drops. That is short, and it is safe only because
    # classification reads the heading band alone: `app._TYPE_SCAN_LINES` looks
    # at the top of the page and at lines short enough to be a heading, so a
    # "50" and a "ทว" adrift in body text are never seen.
    ("WHT_CERTIFICATE", (), _needles(
        "หนังสือรับรองการหักภาษี ณ ที่จ่าย",
        "หนังสือรับรองการหักภาษี",
        "ใบรับรองการหักภาษี",
        "มาตรา 50 ทวิ",
        "50 ทวิ",
        "withholding tax certificate",
        "certificate of withholding tax",
        "tax withholding certificate")),
    ("CREDIT_NOTE", (), _needles("ใบลดหนี้", "ใบหักหนี้", "credit note",
                                 "credit notice")),
    ("DEBIT_NOTE", (), _needles("ใบเพิ่มหนี้", "debit note")),
    # Its own code since 2026-08-31. It was a needle under INVOICE, on a reading
    # of the requirement, and a statement of account is a different document: it
    # summarises what is owed rather than charging for one delivery. sol001
    # prints ใบแจ้งหนี้ in Thai and STATEMENT OF ACCOUNT in English, so it now
    # comes back as both -- which is what the page says, and the manifest is
    # where a human overrules it.
    ("STATEMENT_OF_ACCOUNT", (), _needles("statement of account",
                                          "ใบแจ้งยอด", "ใบสรุปยอด")),
    ("INVOICE", (), _needles("ใบแจ้งหนี้", "ใบวางบิล", "ใบกำกับสินค้า",
                             "invoice", "billing note")),
    ("RECEIPT", (), _needles("ใบเสร็จรับเงิน", "ใบรับเงิน", "receipt")),
    # LAST on purpose, and this decides `primary_type`. Being a tax invoice is
    # the least distinguishing thing about a document here -- six of the ten
    # fixtures are one -- so a receipt/tax-invoice files as a receipt and a
    # credit-note/tax-invoice as a credit note. It is a qualifier that changes
    # which FIELDS are required, not a family to file under.
    ("TAX_INVOICE", (), _needles("ใบกำกับภาษี", "tax invoice")),
]

# `invoice` is a substring of `tax invoice` once both are squashed, so a page
# headed only "TAX INVOICE" would come back as both unless the narrower reading
# suppresses the wider one. These are not exclusive in general -- a real
# ใบกำกับภาษี/ใบแจ้งหนี้ is genuinely both -- so this is a needle-collision
# table, not a statement about documents: a code is dropped only when the ONLY
# evidence for it is text that belongs to the more specific code.
_SUBSUMED = {
    "TAX_INVOICE": (_needles("tax invoice"), _needles("invoice")),
}


def _drop_subsumed(codes, squashed):
    """Remove a code whose only evidence was a substring of a narrower code's."""
    out = list(codes)
    for narrow, (narrow_needles, wide_needles) in _SUBSUMED.items():
        if narrow not in out:
            continue
        for wide_code, _, wide_any in DOCUMENT_TYPES:
            if wide_code not in out:
                continue
            # Evidence for the wider code that is NOT inside the narrower one.
            independent = [n for n in wide_any
                           if n in squashed
                           and not any(n in w for w in narrow_needles)]
            if not independent and set(wide_any) & set(wide_needles):
                out.remove(wide_code)
    return out


def match_types(raw_title):
    """(codes, needles, squashed) -- the table's answer and the EVIDENCE for it.

    `document_types` is this with the evidence dropped. The needles that actually
    matched are what `heading_confidence` measures: a line that IS a heading is
    almost entirely made of them, and a line of body text that happens to mention
    a document kind is not.
    """
    squashed = grounding.squash(raw_title)
    if not squashed:
        return [], [], ""
    codes, hits = [], []
    for code, required, any_of in DOCUMENT_TYPES:
        if not all(n in squashed for n in required):
            continue
        matched = [n for n in tuple(required) + tuple(any_of) if n in squashed]
        if any_of and not any(n in squashed for n in any_of):
            continue
        if code not in codes:
            codes.append(code)
        hits.extend(matched)
    return _drop_subsumed(codes, squashed), hits, squashed


def heading_confidence(raw_title, line_no=1):
    """How much of a line is the heading it was classified by, 0.0-1.0.

    **It measures the EVIDENCE ON THE PAGE, not anybody's self-belief** -- not a
    probability and not the model's opinion of itself. That is what lets one
    function score the Python table's answer and the model's quoted heading on
    the same scale, which is the only way the two can be compared at all.

    `coverage` is the share of the line's squashed characters that the matched
    needles account for, intervals merged so two needles overlapping one another
    are not counted twice. A line that is only the heading scores ~1.0;
    `ใบเสร็จรับเงิน/ใบกำกับภาษี` is two needles covering the whole of it. A
    sentence that mentions a document kind in passing scores low, which is the
    case worth separating -- a credit note's body text regularly names the
    invoice it credits.

    `line_no` costs a little confidence further down the page, because a heading
    is normally near the top and a match at line 30 is likelier to be body text.
    It is a gentle discount, not a cut-off: sol002 prints its heading eighteen
    lines down under two letterheads, and a rule that punished that would be
    wrong on a real document in this very corpus.

    Returns (confidence, detail) so a caller can report WHY -- a bare number
    nobody can take apart is exactly the confidently-wrong figure this project
    refuses elsewhere.
    """
    codes, hits, squashed = match_types(raw_title)
    if not codes or not squashed:
        return 0.0, {"coverage": 0.0, "position": 1.0, "matched": [],
                     "chars": len(squashed)}
    if any(squashed.startswith(lead) for lead in REFERENCE_LEADINS):
        return 0.0, {"coverage": 0.0, "position": 1.0, "matched": [],
                     "chars": len(squashed), "why": "refers to another document"}
    spans = []
    for needle in hits:
        start = squashed.find(needle)
        while start != -1:
            spans.append((start, start + len(needle)))
            start = squashed.find(needle, start + 1)
    spans.sort()
    covered, end = 0, -1
    for a, b in spans:
        a = max(a, end)
        if b > a:
            covered += b - a
            end = b
    coverage = covered / len(squashed)
    position = 1.0 if line_no <= PROMINENT_LINES else POSITION_DISCOUNT
    detail = {"coverage": round(coverage, 3), "position": position,
              "matched": sorted(set(hits)), "chars": len(squashed)}
    return round(min(1.0, coverage) * position, 3), detail


# A heading this far down the page is still a heading -- sol002 prints its own
# eighteen lines below the top, under two letterheads and a logo block -- so the
# discount below applies only past that, and it is small.
PROMINENT_LINES = 20
POSITION_DISCOUNT = 0.9

# A line that OPENS with one of these is a sentence ABOUT another document, not
# this document's own heading -- and it is the failure that made scoring by
# coverage alone unsafe. sol009 prints `อ้างถึงใบกำกับภาษี` ("with reference to
# tax invoice"), which is 64% needle and beat its real heading
# `ใบลดหนี้ / รับคืนสินค้า` at 35%: the page would have been classified a tax
# invoice off a line whose whole job is to name a DIFFERENT one.
#
# **It disqualifies the line rather than discounting it**, because there is no
# score at which "this sentence is about another document" should win: a credit
# note naming the invoice it credits is the single most common way a page
# mentions a type it is not. Anchored at the START only -- `ใบกำกับภาษี` inside
# a heading is ordinary, and a needle merely appearing after one of these words
# somewhere in a long line is already handled by coverage.
REFERENCE_LEADINS = _needles(
    "อ้างถึง", "อ้างอิง", "ตามที่", "ตามใบ", "เพื่อชำระ",
    "with reference to", "reference to", "refer to", "ref.", "against invoice",
    "as per", "in respect of")


def document_types(raw_title):
    """Every standard code a printed heading names, most specific first.

    An empty list is a real answer and is returned as one. Guessing a code from a
    heading this table does not know would put a document into a matching flow on
    the strength of a coin flip, and the requirement is explicit that
    classification belongs to review rather than to a default.
    """
    return match_types(raw_title)[0]


def primary_type(codes):
    """The one code to file a document under, from the codes it carries.

    The front of `DOCUMENT_TYPES`' order, which runs most specific to least: a
    credit note that is also a tax invoice is filed as a credit note, because
    "credit note" is what distinguishes it and nearly everything here is also a
    tax invoice. Used for a filename and for grouping a picker -- never for
    choosing a field set, which is the union of ALL the codes.
    """
    order = [code for code, _, _ in DOCUMENT_TYPES]
    ranked = sorted(codes, key=lambda c: order.index(c) if c in order else 99)
    return ranked[0] if ranked else ""


def document_type_code(raw_title):
    """The single primary code for a heading, or "".

    Kept for callers that genuinely want one label -- the filename, a picker
    heading, the `derived` block. **Do not use it to choose a field set**: it
    discards the other types the page names, which is the bug this whole section
    exists to record.
    """
    return primary_type(document_types(raw_title))


# --------------------------------------------------------------------------
# branch codes
# --------------------------------------------------------------------------

# The standard code for a head office. The requirement names 00000 for it.
HEAD_OFFICE_CODE = "00000"

# Thai revenue branch codes are five digits, so a page printing สาขาที่ 226 means
# branch 00226. Padding is what makes two spellings of one branch compare equal
# downstream.
BRANCH_CODE_WIDTH = 5

_HEAD_OFFICE = _needles("สำนักงานใหญ่", "สนญ", "head office", "ho office",
                        "headquarter")

# A run of digits that is a branch number rather than something else on the line.
# Capped at five AND bounded on both sides, and the bounding is the load-bearing
# half: `\d{1,5}` alone happily chews a thirteen-digit tax ID that has been copied
# into the branch key into its first five digits, turning a known extraction
# failure into a confident-looking branch code. Bounded, a run that long matches
# nothing and the branch code comes back "" -- which is the honest answer, and the
# one a reviewer can act on.
_BRANCH_DIGITS = re.compile(r"(?<!\d)\d{1,5}(?!\d)")


def branch_code(branch_text):
    """A printed branch line reduced to a standard branch code, or "".

    Empty for a branch the page names but does not number -- sol003's buyer block
    prints "(สาขาอำพัน แพร่)", a branch identified by place. That is a missing
    code, not a head office, and the two must not be confused: defaulting an
    unnumbered branch to 00000 would file a branch's documents against the head
    office, which is the one error this normalisation exists to prevent.
    """
    text = (branch_text or "").strip()
    if not text:
        return ""
    squashed = grounding.squash(text)
    digits = _BRANCH_DIGITS.findall(text.translate(_THAI_DIGITS))
    head_office = any(n in squashed for n in _HEAD_OFFICE)
    # A branch number wins over a head-office word where both are printed: a line
    # reading "สำนักงานใหญ่ สาขาที่ 00068" names a numbered branch, and the
    # number is the more specific statement. A zero-valued run is not a branch
    # number -- it is how a head office writes itself.
    for run in digits:
        if int(run):
            return run.zfill(BRANCH_CODE_WIDTH)
    if head_office or (digits and not any(int(d) for d in digits)):
        return HEAD_OFFICE_CODE
    return ""


# --------------------------------------------------------------------------
# tax IDs
# --------------------------------------------------------------------------

# A Thai tax identification number is thirteen digits. A value shorter than this
# is reported as it stands rather than padded -- padding would invent digits, and
# a short tax ID is a read to review, not one to repair.
TAX_ID_DIGITS = 13


def tax_id_digits(value):
    """A printed tax ID reduced to digits, so 0-5454-54545-54-5 compares.

    The requirement asks for exactly this and nothing more: the separators a page
    chooses are presentation, and a matching engine keyed on the printed form
    would miss the same company written two ways.
    """
    return re.sub(r"\D", "", (value or "").translate(_THAI_DIGITS))


def tax_id_valid(value):
    """Whether a normalised tax ID is the right length to be one.

    Reported rather than enforced. A thirteen-digit check is the cheapest signal
    that a tax ID key holds something that is not a tax ID -- the failure the
    extraction prompt spends a paragraph on -- and it belongs beside the value on
    screen, not in a rule that drops it.
    """
    return len(tax_id_digits(value)) == TAX_ID_DIGITS


# --------------------------------------------------------------------------
# service period
# --------------------------------------------------------------------------

# The two ends of a printed period. Kept to a separator surrounded by two
# date-shaped runs, so that a single date, or a sentence containing a dash, does
# not come back as half a period. The extraction pass is told to copy a period
# WHOLE and never to split one across two keys -- CLAUDE.md records what happened
# when it did -- so the splitting is done here instead, where a failed split
# leaves the whole value intact rather than losing half of it.
_PERIOD = re.compile(
    r"^\s*(?P<start>\d[\d/.\-]{4,}|\d{1,2}[-\s][A-Za-z]{3}[-\s]\d{2,4})"
    r"\s*(?:-|–|—|ถึง|to|~)\s*"
    r"(?P<end>\d[\d/.\-]{4,}|\d{1,2}[-\s][A-Za-z]{3}[-\s]\d{2,4})\s*$",
    re.I)


def period_ends(period):
    """A printed period as (from, to). Both "" when it is not a two-ended one."""
    match = _PERIOD.match((period or "").translate(_THAI_DIGITS).strip())
    if not match:
        return "", ""
    return match.group("start").strip(), match.group("end").strip()


# --------------------------------------------------------------------------
# electronic tax
# --------------------------------------------------------------------------

# Wording that says the document was filed with the Revenue Department
# electronically. This is the one derived value read off the transcript rather
# than off an extracted field, because the requirement puts the legal sentence
# itself outside the MVP -- what is wanted is the flag, not the prose, and a flag
# is cheaper to derive than a sentence is to extract.
_ETAX_WORDING = [
    re.compile(r"ส่งข้อมูล(?:ให้แก่|ให้กับ|แก่)?\s*กรมสรรพากร"),
    re.compile(r"ด้วยวิธีการทางอิเล็กทรอนิกส์"),
    re.compile(r"\be-?tax\s*invoice\b", re.I),
    re.compile(r"\be-?receipt\b", re.I),
]


def electronic_tax(transcript):
    """True where the page says it was filed electronically."""
    text = transcript or ""
    return any(p.search(text) for p in _ETAX_WORDING)


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------

# Which extracted key contributes which kind of reference. The requirement asks
# for references as a LIST rather than as a text field, because one receipt
# legitimately settles several invoices -- sol004 settles five distinct invoice
# numbers across fourteen rows, and packing those into one string is precisely
# what it rules out.
#
# The list is ASSEMBLED here rather than asked of the model: every entry in it is
# a value pass 2 already copied and `grounding.py` already checked, so a
# reference cannot appear in this list without appearing on the page.
REFERENCE_SOURCES = [
    ("reference_document", "REFERENCE"),
]

# The line-item cells that would name the document a charge was billed under.
# Pass 2 does not extract line items while `prompts.EXTRACT_LINE_ITEMS` is false,
# so nothing feeds these today. They are kept because the shape of the list is
# the part worth preserving: the requirement asks for references as an ARRAY
# precisely because one receipt can settle several invoices, and that is a
# per-row fact whenever the table comes back.
ITEM_REFERENCE_KEY = "reference_no"
ITEM_REFERENCE_DATE = "reference_date"


def references(fields):
    """Every document this one cites, as a list, deduplicated and in order.

    Each entry is {type, number, date, source}. `source` says whether the value
    came from a header key or from a row of the table, because a reference read
    off a table column and one read off a labelled header line are worth
    different amounts to a reviewer.
    """
    found, seen = [], set()

    def add(kind, number, date, source):
        squashed = grounding.squash(number)
        if not squashed or (kind, squashed) in seen:
            return
        seen.add((kind, squashed))
        found.append({"type": kind, "number": (number or "").strip(),
                      "date": (date or "").strip(), "source": source})

    if not isinstance(fields, dict):
        return found

    for key, kind in REFERENCE_SOURCES:
        add(kind, fields.get(key),
            fields.get("original_invoice_date") if kind == "ORIGINAL_INVOICE"
            else "", "header")

    for row in fields.get("line_items") or []:
        if isinstance(row, dict):
            add("INVOICE", row.get(ITEM_REFERENCE_KEY),
                row.get(ITEM_REFERENCE_DATE), "line_item")
    return found


# --------------------------------------------------------------------------
# the withholding certificate's one derived figure
# --------------------------------------------------------------------------

def wht_rates(fields):
    """The requirement's "Derived WHT Rate %", one entry per income row.

    **Derived, and the requirement says so in the field's own name.** It is
    worked out here rather than asked of the model for the reason this module
    exists: pass 2 is a lookup, and a model asked to compute a percentage while
    it reads starts adjusting the figures it reads to fit the percentage. The
    field is Optional in the requirement, which is what makes deriving it free.

    Each entry is {row, rate}: the row's index in `income_items`, and the tax as
    a percentage of the amount paid, rounded to two places. A row whose figures
    do not both read as amounts, or whose amount paid is zero, contributes
    `None` rather than being left out -- the caller needs the rate to line up
    with the row it belongs to, and a zero payment has no rate rather than a
    rate of zero.
    """
    out = []
    rows = (fields or {}).get(prompts.INCOME_ITEMS_KEY) or []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        paid = verify.parse_amount(row.get("amount_paid"))
        tax = verify.parse_amount(row.get("wht_amount"))
        rate = None
        if paid not in (None, 0) and tax is not None:
            rate = round(tax / paid * 100, 2)
        out.append({"row": index, "rate": rate})
    return out


# --------------------------------------------------------------------------
# the whole derivation
# --------------------------------------------------------------------------

def derive(fields, transcript=""):
    """Every derived value for one extraction, as a flat dict.

    Returned separately from `fields` and never merged into it. `grounding.check`
    walks `fields` and would report each of these as an invention: they are not
    on the page, by construction -- that is what makes them derived.
    """
    fields = fields if isinstance(fields, dict) else {}
    return {
        # The classification the requirement asks for, kept separate from the raw
        # printed heading the model copied. This is the single most useful thing
        # in here: it is what turns "ใบเสร็จรับเงิน/ใบกำกับภาษี" into a value a
        # matching flow can branch on.
        # Both: the list is what the page names and what the field set is built
        # from, the primary is the one label to file it under. A single code
        # here would repeat the bug fixed on 2026-08-31 -- it drops every type
        # after the first.
        "document_types": document_types(fields.get("document_type")),
        "document_type_code": document_type_code(fields.get("document_type")),
        "seller_branch_code": branch_code(fields.get("seller_branch")),
        "buyer_branch_code": branch_code(fields.get("buyer_branch")),
        "seller_tax_id_digits": tax_id_digits(fields.get("seller_tax_id")),
        "buyer_tax_id_digits": tax_id_digits(fields.get("buyer_tax_id")),
        "seller_tax_id_valid": tax_id_valid(fields.get("seller_tax_id")),
        "buyer_tax_id_valid": tax_id_valid(fields.get("buyer_tax_id")),
        # The withholding certificate's two parties get the same treatment as
        # the commercial documents' two, from the same functions. A key the
        # extraction never asked for is absent from `fields`, so these come back
        # empty on every other type rather than being conditional here -- the
        # derivation is a flat dict by design, and a key that appears and
        # disappears is worse to read than one that is empty.
        "payer_branch_code": branch_code(fields.get("payer_branch")),
        "payee_branch_code": branch_code(fields.get("payee_branch")),
        "payer_tax_id_digits": tax_id_digits(fields.get("payer_tax_id")),
        "payee_tax_id_digits": tax_id_digits(fields.get("payee_tax_id")),
        "payer_tax_id_valid": tax_id_valid(fields.get("payer_tax_id")),
        "payee_tax_id_valid": tax_id_valid(fields.get("payee_tax_id")),
        "wht_rates": wht_rates(fields),
        "electronic_tax": electronic_tax(transcript),
        "references": references(fields),
    }


def _selftest():
    """The two tables that name document types must agree, in order.

    `prompts` may not import this module -- normalise -> grounding -> prompts is
    the existing chain and the fourth edge would close a cycle -- so the phrase
    each code is named by in a prompt, and the specificity order both
    `primary_type` and `prompts.type_bullets` rank by, are written out over
    there and checked from here. A code added to `DOCUMENT_TYPES` alone would
    otherwise classify a page and then be silently unnameable in the prompt that
    page is read with, which is the same failure as a schema key missing from
    the template's four maps.
    """
    ours = [code for code, _, _ in DOCUMENT_TYPES]
    theirs = list(prompts.TYPE_SPECIFICITY)
    assert ours == theirs, (
        f"normalise.DOCUMENT_TYPES is {ours} and prompts.TYPE_SPECIFICITY is "
        f"{theirs}: the two must name the same codes in the same order")


_selftest()
