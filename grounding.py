"""Grounding check: every extracted value must be findable in the transcript.

The extraction prompt tells the model to copy values verbatim and to leave a
field empty when the document does not state one. This module is what makes that
enforceable rather than hopeful. After the fields come back, each value is looked
for in the transcript it was supposedly read from. A value that is not there was
not read, it was written -- so it is flagged instead of being displayed as if it
were data. A field that came back empty is reported as missing, which is the right
answer when the document is silent and is worth showing either way.

`missing` describes the extractor, not the page: it says the field came back empty.
Whether the document was silent or the extractor simply missed a value that *is*
printed there is beyond what this check can decide, and it does not pretend
otherwise -- measured on sol005, five of eleven missing fields were on the page.

Deliberately deterministic, like `verify`: no second model pass, nothing to
hallucinate its way into the audit itself.

Matching is loose about presentation and strict about content. Punctuation,
spacing and Thai digits are normalised away, so "0-1055-43041-88-7" grounds
"0105543041887", and a baht|satang cell printed "9,741 60" grounds "9,741.60".
Numbers are additionally compared by value, so "1,200" grounds "1,200.00".

Known limit: a one- or two-character value ("7", "฿") will nearly always be found
somewhere in a page of text, so grounding says little about it. The check is a
hallucination detector for substantive values -- names, identifiers, dates,
amounts -- not a proof of correctness for short ones.
"""

import re
import unicodedata

import prompts
import verify

# Every scalar key the extraction schema can ask for -- the UNION of the per-type
# field sets, imported rather than copied so the two cannot drift. What any one
# extraction actually asks for is a subset of this, chosen by document type: see
# `prompts.DOC_TYPE_FIELDS`. Anything outside it that the model returns is still
# checked for grounding, just never demanded.
SCALAR_FIELDS = list(prompts._SCALAR_KEYS)

# Of those, the ones whose absence is worth reporting -- and since 2026-08-31
# that is a question about the DOCUMENT TYPE, not about the schema. The
# requirement marks a different set Mandatory for an invoice than for a credit
# note, and a key the requirement does not demand of this kind of document
# should not stand as a complaint against it.
#
# `SCALAR_REQUIRED` is what a caller that has not classified anything gets, and
# it is the list this module used before the split: every key except
# `reference_document`, which is legitimately empty on most documents.
# `required_for_type` is what a caller with a type should use.
SCALAR_REQUIRED = list(prompts.DEFAULT_MANDATORY)


def required_for_type(code):
    """The keys whose absence is worth reporting, for a document-type code."""
    return list(prompts.mandatory_for_type(code))


# The delivery tiers of the field requirement. Pass 2 extracts TIER 1 ONLY, so
# tier 1 is the whole schema and the other two are empty here rather than
# deleted.
#
# They are kept as empty lists on purpose, and this is not tidiness. `tier_counts`
# writes `p2_present`/`p2_absent` into every run-log row, and the run log is
# append-only across builds: a reader comparing a row written today with one
# written last week needs today's row to say `0/0` -- "this build asked for none
# of them" -- rather than to leave a blank that means "this run extracted
# nothing". Present and absent are both written, so the sum is what the tier was
# on that row.
#
# The keys that used to be in tiers 2 and 3 -- addresses, due dates, withholding
# tax, net payable, VAT rate, service period, customer, contract and location
# codes, payment and bank details, the amount in words -- are not lost. Pass 2
# puts anything it finds under its own printed label into `other_fields`, which
# is exactly where they go until a later phase asks for them by name.
PRIORITY_1 = list(SCALAR_FIELDS)
PRIORITY_2 = []
PRIORITY_3 = []

# Line-item cells that are normally blank on a real document (a receipt rarely
# rules a period or a withholding column). Missing ones are not worth reporting;
# a missing description or amount is.
ITEM_REQUIRED = ["description", "amount"]


# The house rule for "this is a loop, not a document": `app._repeated_list` uses
# the same figure, and the two must agree or one detector will clear a reply the
# other condemns. A genuine page does not list one entry three times over.
REPEAT_THRESHOLD = 3


def _entry_text(entry) -> str:
    """One list entry reduced to the text that identifies it.

    Handles both shapes deliberately. `other_fields` is *supposed* to be
    {label, value} objects, and the loop that motivated this returned 104 bare
    STRINGS instead -- so a check that assumes the schema was obeyed misses
    exactly the replies worth catching.
    """
    if isinstance(entry, dict):
        parts = [entry.get("label"), entry.get("value")]
    else:
        parts = [entry]
    return "|".join(squash(part) for part in parts if part is not None)


def list_repetition(fields) -> dict:
    """How much of what came back under `other_fields` is one entry over again.

    **Why this exists, and why it is not `app._repeated_list`.** That detector
    reads the RAW reply for `"label"`/`"description"` keys, which is the right
    place to catch a loop before the JSON is parsed -- and it is blind to a reply
    whose list holds bare strings, because there are no keys in it to match. That
    is the shape dots.mocr returns: 104 entries, 5 distinct, the same invented
    company name a hundred times, salvaged as a clean `partial` and counted as
    104 extra fields found. This reads the PARSED list instead, so the shape of
    the entries cannot hide the repetition.

    Returns entries, distinct, and `looped`. `distinct` is the honest count of
    what the reply actually contributed, and it is what a comparison between
    models should use: the alternative rewards the failure it should penalise.

    **It compares squashed text, so it catches exact repetition and not a
    counter loop** (`ปี 1`, `ปี 2`, ...) -- digits survive `squash`. That is the
    same limit `looks_repetitive` handles separately for streams, and no
    `other_fields` reply observed so far needs it.
    """
    entries = [e for e in ((fields or {}).get("other_fields") or [])]
    seen = {}
    for entry in entries:
        text = _entry_text(entry)
        if text:
            seen[text] = seen.get(text, 0) + 1
    return {
        "entries": len(entries),
        "distinct": len(seen),
        "looped": any(count >= REPEAT_THRESHOLD for count in seen.values()),
    }


def tier_counts(fields, keys=None) -> dict:
    """How many tier-1 and tier-2 fields came back filled, and how many did not.

    Counts only -- never the values -- so the run log can carry field coverage
    without becoming a copy of the documents. "Present" means the extractor
    returned something; whether that something is right is what `grounded_pct`
    and the number checks are for.
    """
    # `keys` is the field set this extraction actually asked for. Counting
    # against the union instead would report a document as missing keys nobody
    # requested of it -- an invoice is not incomplete for having no
    # `original_invoice_date`. `p1_present + p1_absent` is therefore the size of
    # the form that ran, which is what makes the pair readable on a run-log row
    # whose build asked for a different number of keys.
    tier1 = list(keys) if keys is not None else PRIORITY_1
    counts = {}
    for label, tier in (("p1", tier1), ("p2", PRIORITY_2), ("p3", PRIORITY_3)):
        present = sum(1 for k in tier
                      if isinstance(fields, dict) and not _is_blank(fields.get(k)))
        counts[f"{label}_present"] = present
        counts[f"{label}_absent"] = len(tier) - present
    return counts

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# A nil cell, not an absent one: the document printed a dash where a figure would
# go. Grounding it against the page is meaningless, so it is reported as its own
# state rather than as an invention.
_NIL = {"-", "--", "---", "–", "—", "n/a", "na", "nil", "null", "none"}

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# The same baht|satang column split `verify.parse_amount` handles on the field
# side, applied to the transcript so both ends of the comparison agree.
_BAHT_SATANG = re.compile(r"(\d[\d,]*)\s+(\d{2})(?!\d)")

# Money fields, and how each could be arrived at by adding up other extracted
# numbers rather than by reading it off the page. Used only to say *how* an
# ungrounded total probably came about -- the prompt forbids computing them.
_MONEY_KEYS = ("subtotal", "vat_total", "amount_incl_vat",
               "withholding_tax_total", "net_payable")


def _squash(value):
    """Reduce text to comparable content: letters and digits, nothing else."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(_THAI_DIGITS).lower()
    return "".join(ch for ch in text if ch.isalnum())


def _is_blank(value):
    return _squash(value) == "" and not str(value or "").strip()


def _is_nil(value):
    return str(value or "").strip().lower() in _NIL


# Public because the agentic extractor grounds each step's answer as the step
# comes back, rather than only grounding the finished set at the end, and it has
# to skip the same two states this module skips. Two copies of what counts as a
# nil cell would drift apart, and the copy in `app.py` would be the wrong one.
def is_blank(value):
    """True when a value is absent -- the field came back empty."""
    return _is_blank(value)


def is_nil(value):
    """True when the document printed a dash where a figure would go."""
    return _is_nil(value)


def squash(value):
    """Value reduced to comparable content: letters and digits, nothing else.

    Public for the same reason as the two above: `fieldscore.py` compares an
    extracted value with a hand-written one and has to be loose about exactly the
    same things this module is loose about -- punctuation, spacing, Thai digits.
    A second implementation of that would let the audit and the score disagree
    about whether two spellings of one value are the same value.
    """
    return _squash(value)


def _transcript_numbers(transcript):
    """Every number the page prints, as floats, for value-wise comparison."""
    text = (transcript or "").translate(_THAI_DIGITS)
    found = set()
    for match in _NUMBER.finditer(text):
        number = verify.parse_amount(match.group(0))
        if number is not None:
            found.add(round(abs(number), 2))
    for match in _BAHT_SATANG.finditer(text):
        number = verify.parse_amount(f"{match.group(1)}.{match.group(2)}")
        if number is not None:
            found.add(round(abs(number), 2))
    return found


class Source:
    """The transcript, indexed for the two ways a value can be grounded in it."""

    def __init__(self, transcript):
        self.squashed = _squash(transcript)
        self.numbers = _transcript_numbers(transcript)

    def holds(self, value):
        squashed = _squash(value)
        if squashed and squashed in self.squashed:
            return True
        # A number can be printed one way and extracted another -- "1,200" for
        # "1,200.00", or ruled into baht and satang -- so fall back to comparing
        # what it is worth. Sign is verify's business, not grounding's.
        number = verify.parse_amount(value)
        if number is not None and round(abs(number), 2) in self.numbers:
            return True
        return False


def _sum_of(items, key):
    values = [verify.parse_amount(i.get(key)) for i in items if isinstance(i, dict)]
    values = [v for v in values if v is not None]
    return round(sum(values), 2) if values else None


def _looks_computed(key, value, fields, items):
    """True when an ungrounded money field equals a sum of the other figures.

    Worth separating from a plain invention: it means the model added the page up
    instead of reading a total the page never printed, which the prompt forbids
    and which reads as authoritative on screen.
    """
    actual = verify.parse_amount(value)
    if actual is None:
        return False
    subtotal = verify.parse_amount(fields.get("subtotal"))
    vat = verify.parse_amount(fields.get("vat_total"))
    incl = verify.parse_amount(fields.get("amount_incl_vat"))
    wht = verify.parse_amount(fields.get("withholding_tax_total"))

    candidates = []
    if key == "subtotal":
        candidates.append(_sum_of(items, "amount"))
        # On a VAT-inclusive page the goods total is not reached by adding the
        # lines up but by taking the VAT back out of the grand total, so that is
        # the route a subtotal the page never printed would have come by.
        if incl is not None:
            candidates.append(round(incl - (vat or 0), 2))
    elif key == "vat_total":
        candidates.append(_sum_of(items, "vat"))
    elif key == "withholding_tax_total":
        candidates.append(_sum_of(items, "withholding_tax"))
    elif key == "amount_incl_vat":
        if subtotal is not None:
            candidates.append(round(subtotal + (vat or 0), 2))
    elif key == "net_payable":
        candidates.append(_sum_of(items, "net_amount"))
        # Either route to the same figure: down from the VAT-inclusive total, or
        # up from the subtotal when the page never printed that total either.
        if incl is not None:
            candidates.append(round(incl - (wht or 0), 2))
        if subtotal is not None:
            candidates.append(round(subtotal + (vat or 0) - (wht or 0), 2))
    return any(c is not None and abs(c - actual) <= verify.TOLERANCE
               for c in candidates)


def check(fields, transcript, keys=None, required=None, items=None):
    """Audit extracted fields against the transcript they came from.

    Returns a per-path status map plus the two lists worth acting on: what the
    document does not state, and what the model produced that the document does
    not support.

    `keys` is the field set this extraction asked for and `required` the subset
    the requirement makes Mandatory for that document type; both default to the
    whole schema and the untyped required list, which is what this did before
    per-type field sets. Reporting a key nobody asked for as `missing` would put
    a standing complaint against every document of the wrong type -- and
    `missing` already means "the extractor did not return this", never "the page
    does not print it".

    `items` is the row shape of a TABLE this extraction asked for --
    `prompts.items_for_types` -- and it follows exactly the same rule one level
    down, which is why it had to be added when the withholding certificate
    brought a table back (2026-09-01):

    * a non-empty shape means the certificate's `income_items` were asked for,
      so an empty list is a real `missing` and every cell is audited;
    * `()` means this form asks for no table at all, so nothing is demanded;
    * `None` is what a caller that has not been taught about tables gets, and it
      keeps the behaviour this had before: `line_items` demanded whether or not
      anything asked for it. That was a standing false complaint on every
      document -- pass 2 has not asked for the charges table since the schema
      narrowed to priority 1 -- and it is fixed by the app passing the shape,
      not by changing what a caller who says nothing gets.
    """
    if not isinstance(fields, dict):
        return {}
    if not (transcript or "").strip():
        return {"error": "no transcript to check the fields against"}

    source = Source(transcript)
    items_shape = tuple(items) if items is not None else None
    rows = [i for i in (fields.get("line_items") or []) if isinstance(i, dict)]
    statuses = {}
    missing = []
    flagged = []

    def judge(path, value, required, key=None):
        if _is_blank(value):
            statuses[path] = "missing"
            if required:
                missing.append(path)
            return
        if _is_nil(value):
            statuses[path] = "nil"
            return
        if source.holds(value):
            statuses[path] = "grounded"
            return
        computed = key in _MONEY_KEYS and _looks_computed(key, value, fields, rows)
        statuses[path] = "computed" if computed else "ungrounded"
        flagged.append({
            "path": path,
            "value": str(value)[:120],
            "status": statuses[path],
            "why": ("does not appear on the page; it is the sum of the extracted "
                    "figures, so the model added it up rather than reading it"
                    if computed else
                    "no matching text on the page -- the model produced this"),
        })

    asked = list(keys) if keys is not None else SCALAR_FIELDS
    demanded = set(required if required is not None else SCALAR_REQUIRED)
    for key in asked:
        judge(key, fields.get(key), required=key in demanded, key=key)

    # Keys outside the schema are the model's own additions and are the most
    # likely place for an invention, so they are checked too -- just not demanded.
    for key, value in fields.items():
        if key in asked or key in ("line_items", prompts.INCOME_ITEMS_KEY,
                                   "other_fields"):
            continue
        if isinstance(value, (str, int, float)):
            judge(key, value, required=False)

    if items_shape is None:
        # The pre-2026-09-01 behaviour, kept for a caller that says nothing.
        if not rows:
            statuses["line_items"] = "missing"
            missing.append("line_items")
        for index, item in enumerate(rows):
            for key, value in item.items():
                if isinstance(value, (str, int, float)):
                    judge(f"line_items[{index}].{key}", value,
                          required=key in ITEM_REQUIRED)
    elif items_shape:
        income = [r for r in (fields.get(prompts.INCOME_ITEMS_KEY) or [])
                  if isinstance(r, dict)]
        if not income:
            statuses[prompts.INCOME_ITEMS_KEY] = "missing"
            missing.append(prompts.INCOME_ITEMS_KEY)
        for index, row in enumerate(income):
            for key, value in row.items():
                if isinstance(value, (str, int, float)):
                    # Never demanded here, although the requirement marks all
                    # four Mandatory. A per-ROW obligation is `validate`'s to
                    # report -- it is the module that knows which rules this
                    # document is held to -- and one fault reported twice under
                    # two names is the mistake the ABSENT/FAILED split exists to
                    # avoid.
                    judge(f"{prompts.INCOME_ITEMS_KEY}[{index}].{key}", value,
                          required=False)

    for index, entry in enumerate(fields.get("other_fields") or []):
        if isinstance(entry, dict):
            judge(f"other_fields[{index}].value", entry.get("value"), required=False)

    counts = {"grounded": 0, "ungrounded": 0, "computed": 0, "missing": 0, "nil": 0}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    checked = counts["grounded"] + counts["ungrounded"] + counts["computed"]

    return {
        "statuses": statuses,
        "missing": missing,
        "flagged": flagged,
        "counts": counts,
        "checked": checked,
        # Share of the values that were actually traceable back to the page. The
        # headline number: 1.0 means nothing on screen was invented.
        "grounded_ratio": round(counts["grounded"] / checked, 4) if checked else None,
        "verdict": "flagged" if flagged else ("grounded" if checked else "empty"),
    }
