"""Reading document amounts, and deciding whether they already carry VAT.

Deterministic and model-free, like `grounding`: everything here is decided in
Python from figures the extraction copied verbatim.

Two callers, both in the extraction pass:

* `grounding.py` compares extracted figures to the transcript by value rather
  than by text, so `1,200` grounds `1,200.00` -- `parse_amount` and `TOLERANCE`
  are what make that comparison possible.
* `app.py` puts `vat_basis` on the extraction result, because the Fields tab
  labels the totals from it. Most Thai tax invoices price VAT-exclusive -- the
  lines add up to the goods total and VAT is added below it. But a receipt whose
  money column is headed จำนวนเงิน (รวมภาษี) prices VAT-inclusive: each figure in
  that column already contains the tax. Labelling a figure that contains ฿672.30
  of VAT as "ex VAT" is a plain lie, values right and read wrong.
"""

import re

# Tolerance in currency units. Documents round per-line, so exact equality is
# too strict; anything above this is a real discrepancy rather than rounding.
TOLERANCE = 0.02

# Wording that says which basis the figures are on. Kept deliberately narrow,
# because the obvious pattern is a trap: "รวมภาษีมูลค่าเพิ่ม 672.30" at the foot
# of an ordinary exclusive invoice is the label of the VAT row -- "total VAT" --
# not a claim that anything includes it. So an inclusive reading is only accepted
# where the wording is bracketed as a column heading, attached to a price word,
# or closed with แล้ว ("already"). Exclusive wording is tested first: every Thai
# way of denying inclusion contains the inclusive phrase inside it.
_INCLUSIVE_WORDING = [
    re.compile(r"[(\[（]\s*รวม\s*(?:ภาษีมูลค่าเพิ่ม|ภาษี|VAT)[^)\]）]{0,10}[)\]）]", re.I),
    re.compile(r"(?:ราคา|มูลค่า|จำนวนเงิน|ยอดเงิน)\s*รวม\s*(?:ภาษีมูลค่าเพิ่ม|ภาษี|VAT)", re.I),
    re.compile(r"รวม\s*(?:ภาษีมูลค่าเพิ่ม|ภาษี|VAT)\s*(?:\d{1,2}(?:\.\d+)?\s*%\s*)?แล้ว", re.I),
    re.compile(r"[(\[]?\s*(?:incl\.?|including|inclusive\s+of)\s+(?:of\s+)?VAT\b", re.I),
    re.compile(r"\bVAT\s+includ(?:ed|ing)\b", re.I),
    re.compile(r"\bprices?\s+(?:are\s+)?includ\w*\s+VAT\b", re.I),
]
_EXCLUSIVE_WORDING = [
    re.compile(r"(?:ไม่|ยังไม่)\s*รวม\s*(?:ภาษีมูลค่าเพิ่ม|ภาษี|VAT)", re.I),
    re.compile(r"ก่อน\s*ภาษีมูลค่าเพิ่ม", re.I),
    re.compile(r"[(\[]?\s*(?:excl\.?|excluding|exclusive\s+of)\s+(?:of\s+)?VAT\b", re.I),
    re.compile(r"\bbefore\s+VAT\b", re.I),
    re.compile(r"\bVAT\s+not\s+included\b", re.I),
]


def parse_amount(value):
    """Turn a document amount into a float. None when it is not a number.

    Handles thousands separators, a bare '-' meaning nil, parenthesised
    negatives, trailing currency words, and Thai digits.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "--", "—", "N/A", "n/a"}:
        return None
    text = text.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    negative = text.startswith("(") and text.endswith(")")
    inner = text.strip("()").strip()
    # Thai receipts often rule the amount column into baht | satang, so a cell can
    # arrive as "9,741 60" meaning 9,741.60. Only treat a trailing 2-digit group
    # this way when there is no decimal point already -- "1,731,118.40 51,933.55"
    # is two separate amounts and must NOT be merged.
    if "." not in inner:
        split = re.fullmatch(r"(\d{1,3}(?:,\d{3})*|\d+)\s+(\d{2})", inner)
        if split:
            inner = f"{split.group(1)}.{split.group(2)}"
    text = re.sub(r"[^\d.\-]", "", inner)
    if text in {"", "-", ".", "-."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _close(a, b):
    return abs(a - b) <= TOLERANCE


def _sum_of(items, key):
    values = [parse_amount(i.get(key)) for i in items if isinstance(i, dict)]
    values = [v for v in values if v is not None]
    return round(sum(values), 2) if values else None


def _haystack(fields, transcript):
    """Everywhere a statement about the VAT basis could have landed.

    The column heading that carries it is usually in the transcript, but on a
    page the extractor read well it can also arrive inside a line description or
    an other_fields label, so all three are searched.
    """
    parts = [transcript or ""]
    for item in (fields.get("line_items") or []):
        if isinstance(item, dict):
            parts.extend(str(item.get(k) or "") for k in ("description", "period"))
    for extra in (fields.get("other_fields") or []):
        if isinstance(extra, dict):
            parts.extend(str(extra.get(k) or "") for k in ("label", "value"))
    return "\n".join(p for p in parts if p)


def _wording_basis(text):
    """The basis the page states in words, with the phrase that said so."""
    for basis, patterns in (("exclusive", _EXCLUSIVE_WORDING),
                            ("inclusive", _INCLUSIVE_WORDING)):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return basis, " ".join(match.group(0).split()).strip("()[]（） ")
    return None, None


def vat_basis(fields, transcript=None):
    """Decide whether the printed amounts already carry VAT, and say how we know.

    The numbers are consulted before the words, because they are the stronger
    evidence: line amounts that reach the VAT-inclusive total while missing the
    subtotal by exactly the VAT settle the question outright, whatever any
    heading says. Wording decides it when the totals cannot -- one of them is
    absent, or both were printed the same. Failing both, Thai tax invoices are
    VAT-exclusive, and that is the assumption, stated as an assumption.
    """
    fields = fields if isinstance(fields, dict) else {}
    items = [i for i in (fields.get("line_items") or []) if isinstance(i, dict)]
    line_total = _sum_of(items, "amount")
    subtotal = parse_amount(fields.get("subtotal"))
    vat = parse_amount(fields.get("vat_total"))
    incl = parse_amount(fields.get("amount_incl_vat"))

    basis = source = evidence = None
    if (line_total is not None and subtotal is not None and incl is not None
            and not _close(subtotal, incl)):
        hits_incl = _close(line_total, incl)
        hits_subtotal = _close(line_total, subtotal)
        if hits_incl != hits_subtotal:
            basis = "inclusive" if hits_incl else "exclusive"
            source = "inferred from the totals"
            evidence = (f"line amounts sum to {line_total:,.2f}, which is the "
                        + ("VAT-inclusive total" if hits_incl else "subtotal"))

    stated, phrase = _wording_basis(_haystack(fields, transcript))
    if basis is None and stated:
        basis, source, evidence = stated, "stated on the document", phrase
    elif stated and stated == basis:
        evidence += f" · the page says “{phrase}”"
    elif stated:
        # The figures and the page's own wording disagree. The figures win --
        # they are the ones being labelled -- but the disagreement is reported
        # rather than swallowed, because one of the two is a mis-read: either a
        # line amount landed in the wrong column, or that phrase is not about
        # the money column at all.
        source += ", against the page's wording"
        evidence += f" — but the page says “{phrase}”"
    if basis is None:
        basis, source = "exclusive", "assumed (the document does not say)"

    # Whether the figure captured as `subtotal` is itself VAT-inclusive. It is
    # when the page printed one grand total and nothing before VAT: the same
    # figure then lands in both total fields, or the inclusive lines add up to
    # the one the extractor called the subtotal. Worth separating from the line
    # basis, because sol003-style pages print inclusive lines AND a genuine
    # goods total below them -- inclusive lines do not imply an inclusive subtotal.
    subtotal_includes_vat = False
    if subtotal is not None and vat:
        if incl is not None:
            subtotal_includes_vat = _close(subtotal, incl)
        else:
            subtotal_includes_vat = (basis == "inclusive" and line_total is not None
                                     and _close(line_total, subtotal))

    return {
        "basis": basis,
        "inclusive": basis == "inclusive",
        "source": source,
        "evidence": evidence,
        "subtotal_includes_vat": subtotal_includes_vat,
        "summary": ("line amounts already include VAT" if basis == "inclusive"
                    else "line amounts exclude VAT")
                   + (" · subtotal includes VAT too" if subtotal_includes_vat else ""),
    }
