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

# Raw printed heading -> the standard code. Tested in order, and the ORDER is the
# whole of the correctness here, exactly as in `fieldscore.HEADER_MAP`: a receipt
# that is also a tax invoice heads itself ใบเสร็จรับเงิน/ใบกำกับภาษี, which
# contains both single readings inside it, so the compound has to be asked for
# before either half. A credit note is asked before all of them, because one
# often names the invoice it corrects in its own heading.
#
# Each entry is (code, all-of, any-of): the code applies when every needle in
# all-of is present and at least one from any-of is, an empty any-of meaning
# nothing further is required. Needles are matched against the squashed heading,
# so spacing, punctuation and slashes do not matter.
DOCUMENT_TYPES = [
    ("CREDIT_NOTE", (), _needles("ใบลดหนี้", "ใบหักหนี้", "credit note",
                                 "credit notice")),
    ("DEBIT_NOTE", (), _needles("ใบเพิ่มหนี้", "debit note")),
    # Both halves present -> the compound. Thai and English are tested
    # independently so a page printing only one language still resolves.
    ("RECEIPT_TAX_INVOICE", _needles("ใบเสร็จรับเงิน", "ใบกำกับภาษี"), ()),
    ("RECEIPT_TAX_INVOICE", _needles("receipt", "tax invoice"), ()),
    ("TAX_INVOICE", (), _needles("ใบกำกับภาษี", "tax invoice")),
    ("RECEIPT", (), _needles("ใบเสร็จรับเงิน", "ใบรับเงิน", "receipt")),
    # Statement of Account is named by the requirement as an invoice heading, and
    # ใบวางบิล / ใบแจ้งหนี้ are the two ordinary Thai wordings for one.
    ("INVOICE", (), _needles("ใบแจ้งหนี้", "ใบวางบิล", "ใบกำกับสินค้า",
                             "invoice", "statement of account",
                             "billing note")),
]


def document_type_code(raw_title):
    """The standard code for a printed document heading, or "" if unrecognised.

    Unrecognised is a real answer and is returned as one. Guessing a code from a
    heading this table does not know would put a document into a matching flow on
    the strength of a coin flip, and the requirement is explicit that
    classification belongs to review rather than to a default.
    """
    squashed = grounding.squash(raw_title)
    if not squashed:
        return ""
    for code, required, any_of in DOCUMENT_TYPES:
        if all(n in squashed for n in required) and (
                not any_of or any(n in squashed for n in any_of)):
            return code
    return ""


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
        "document_type_code": document_type_code(fields.get("document_type")),
        "seller_branch_code": branch_code(fields.get("seller_branch")),
        "buyer_branch_code": branch_code(fields.get("buyer_branch")),
        "seller_tax_id_digits": tax_id_digits(fields.get("seller_tax_id")),
        "buyer_tax_id_digits": tax_id_digits(fields.get("buyer_tax_id")),
        "seller_tax_id_valid": tax_id_valid(fields.get("seller_tax_id")),
        "buyer_tax_id_valid": tax_id_valid(fields.get("buyer_tax_id")),
        "electronic_tax": electronic_tax(transcript),
        "references": references(fields),
    }
