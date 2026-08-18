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

import verify

# The scalar keys of the extraction schema. Anything else the model returns is
# checked for grounding but never demanded.
SCALAR_FIELDS = [
    "document_type", "document_number", "issue_date", "due_date",
    "reference_document", "po_number", "original_invoice_number",
    "contract_number", "customer_code", "location_code", "service_period",
    "seller_name", "seller_tax_id", "seller_branch", "seller_address",
    "buyer_name", "buyer_tax_id", "buyer_branch", "buyer_address",
    "currency", "vat_rate", "subtotal", "vat_total", "amount_incl_vat",
    "withholding_tax_total", "net_payable", "amount_in_words",
    "payment_method", "payment_reference",
]

# Of those, the ones whose absence is worth reporting. The rest are ordinary on a
# real document -- a cash receipt has no PO number, no contract and no branch, and
# listing all thirteen as missing on every run would bury the ones that matter.
# Same reasoning as ITEM_REQUIRED below, applied to the scalars.
SCALAR_REQUIRED = [
    "document_type", "document_number", "issue_date", "due_date",
    "seller_name", "seller_tax_id", "buyer_name", "buyer_tax_id",
    "currency", "vat_rate", "subtotal", "vat_total", "amount_incl_vat",
    "withholding_tax_total", "amount_in_words", "payment_method",
]

# The same scalars split into the two delivery tiers, so a run can be scored as
# "how much of the MVP set did we get" separately from the production set. Order
# is the mapping's, not the schema's. amount_in_words is tier 3 and in neither.
PRIORITY_1 = [
    "document_type", "document_number", "issue_date",
    "seller_name", "buyer_name", "seller_tax_id", "buyer_tax_id",
    "seller_branch", "buyer_branch", "subtotal", "vat_total",
    "amount_incl_vat", "currency", "reference_document",
]
PRIORITY_2 = [
    "seller_address", "buyer_address", "due_date", "po_number",
    "original_invoice_number", "withholding_tax_total", "net_payable",
    "vat_rate", "service_period", "customer_code", "contract_number",
    "location_code", "payment_method", "payment_reference",
]

# Line-item cells that are normally blank on a real document (a receipt rarely
# rules a period or a withholding column). Missing ones are not worth reporting;
# a missing description or amount is.
ITEM_REQUIRED = ["description", "amount"]


def tier_counts(fields) -> dict:
    """How many tier-1 and tier-2 fields came back filled, and how many did not.

    Counts only -- never the values -- so the run log can carry field coverage
    without becoming a copy of the documents. "Present" means the extractor
    returned something; whether that something is right is what `grounded_pct`
    and the number checks are for.
    """
    counts = {}
    for label, keys in (("p1", PRIORITY_1), ("p2", PRIORITY_2)):
        present = sum(1 for k in keys
                      if isinstance(fields, dict) and not _is_blank(fields.get(k)))
        counts[f"{label}_present"] = present
        counts[f"{label}_absent"] = len(keys) - present
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


def check(fields, transcript):
    """Audit extracted fields against the transcript they came from.

    Returns a per-path status map plus the two lists worth acting on: what the
    document does not state, and what the model produced that the document does
    not support.
    """
    if not isinstance(fields, dict):
        return {}
    if not (transcript or "").strip():
        return {"error": "no transcript to check the fields against"}

    source = Source(transcript)
    items = [i for i in (fields.get("line_items") or []) if isinstance(i, dict)]
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
        computed = key in _MONEY_KEYS and _looks_computed(key, value, fields, items)
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

    for key in SCALAR_FIELDS:
        judge(key, fields.get(key), required=key in SCALAR_REQUIRED, key=key)

    # Keys outside the schema are the model's own additions and are the most
    # likely place for an invention, so they are checked too -- just not demanded.
    for key, value in fields.items():
        if key in SCALAR_FIELDS or key in ("line_items", "other_fields"):
            continue
        if isinstance(value, (str, int, float)):
            judge(key, value, required=False)

    if not items:
        statuses["line_items"] = "missing"
        missing.append("line_items")
    for index, item in enumerate(items):
        for key, value in item.items():
            if isinstance(value, (str, int, float)):
                judge(f"line_items[{index}].{key}", value,
                      required=key in ITEM_REQUIRED)

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
