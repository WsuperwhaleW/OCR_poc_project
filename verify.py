"""Arithmetic verification of extracted document fields.

Deliberately deterministic. A 2B model is unreliable at addition, so every
check that *can* be decided by arithmetic is decided here, in Python, and the
model is only asked afterwards for the judgement calls arithmetic cannot make.

Also parses Thai amount-in-words, which is the one field that cannot be checked
by summing anything else -- and the field where a wrong reading is easiest to
miss, because it reads fluently.

Every check here depends on one thing being settled first: whether the printed
figures already carry VAT. Most Thai tax invoices price VAT-exclusive -- the
lines add up to the goods total, and VAT is added below it. But a receipt whose
money column is headed จำนวนเงิน (รวมภาษี) prices VAT-inclusive: each figure in
that column already contains the tax, the lines add up to the GRAND total, and
the VAT line at the foot is backed out of it rather than added to it. Checked
against the exclusive relations, such a page fails on every line while being
perfectly correct. `vat_basis` decides which it is before anything is compared.
"""

import re

# Tolerance in currency units. Documents round per-line, so exact equality is
# too strict; anything above this is a real discrepancy rather than rounding.
TOLERANCE = 0.02

# Thai VAT is 7%. Used only when the document does not state its own rate.
DEFAULT_VAT_RATE = 7.0
# How far the computed ratio may sit from the expected rate before it is flagged.
RATE_TOLERANCE = 0.15

# "อัตราภาษีมูลค่าเพิ่ม 7.00%", "VAT 7%", "อัตราร้อยละ 7", "Vat 7 %"
_RATE_PATTERNS = [
    re.compile(r"อัตราภาษีมูลค่าเพิ่ม\s*(\d{1,2}(?:\.\d+)?)\s*%?", re.I),
    re.compile(r"อัตราร้อยละ\s*(\d{1,2}(?:\.\d+)?)", re.I),
    re.compile(r"ภาษีมูลค่าเพิ่ม\s*(\d{1,2}(?:\.\d+)?)\s*%", re.I),
    re.compile(r"\bVAT\b[^0-9%\n]{0,12}(\d{1,2}(?:\.\d+)?)\s*%", re.I),
    re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%\s*VAT\b", re.I),
]

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

_THAI_DIGITS = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
}
_THAI_PLACES = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10_000, "แสน": 100_000}
_THAI_TOKEN = re.compile(
    "(" + "|".join(sorted(list(_THAI_DIGITS) + list(_THAI_PLACES) + ["ล้าน"],
                          key=len, reverse=True)) + ")"
)


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


def parse_rate(value):
    """Read a VAT rate from a field value. None when absent or implausible.

    Accepts "7", "7%", "7.00%", "ร้อยละ 7". A rate outside 0-30% is rejected as a
    misread rather than believed -- 0% is real (zero-rated), 700% is not.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        rate = float(value)
    else:
        text = str(value).translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
        # Match the WHOLE value, not the first digits in it. A loose search would
        # read a tax ID like "0-5454-54545-54-5" as a 0% rate and then quietly
        # check every line against 0%.
        text = re.sub(r"(อัตรา|ภาษีมูลค่าเพิ่ม|ร้อยละ|VAT|vat)", " ", text)
        match = re.fullmatch(r"\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*", text)
        if not match:
            return None
        rate = float(match.group(1))
    return rate if 0 <= rate <= 30 else None


def find_vat_rate(text):
    """Pull a VAT rate out of a raw transcript. None when not stated."""
    if not text:
        return None
    for pattern in _RATE_PATTERNS:
        match = pattern.search(text)
        if match:
            rate = parse_rate(match.group(1))
            if rate is not None:
                return rate
    return None


def resolve_vat_rate(fields, transcript=None):
    """Decide which VAT rate to check against, and say where it came from."""
    rate = parse_rate((fields or {}).get("vat_rate"))
    if rate is not None:
        return rate, "stated on the document"
    rate = find_vat_rate(transcript)
    if rate is not None:
        return rate, "found in the transcript"
    return DEFAULT_VAT_RATE, "assumed (document does not state one)"


def thai_words_to_number(text):
    """Parse a Thai baht amount-in-words. None if it cannot be read."""
    parts = thai_words_parts(text)
    return None if parts is None else parts[0] + parts[1] / 100.0


def thai_words_parts(text):
    """The baht and satang the words spell out, separately. None if unreadable.

    Kept apart from the total so the check can show its working: a reading that
    is out by a factor of a hundred, or that lost the satang clause entirely, is
    obvious from the two parts and invisible from the sum.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = text.replace("จำนวนเงินตัวอักษร", "").replace("(", " ").replace(")", " ")
    if "บาท" not in cleaned:
        return None
    baht_part, _, rest = cleaned.partition("บาท")
    satang_part = rest.split("สตางค์")[0] if "สตางค์" in rest else ""

    def chunk(part):
        tokens = _THAI_TOKEN.findall(part)
        if not tokens:
            return None
        total = current = digit = 0
        for tok in tokens:
            if tok == "ล้าน":
                total = (total + current + digit) * 1_000_000
                current = digit = 0
            elif tok in _THAI_PLACES:
                current += (digit or 1) * _THAI_PLACES[tok]
                digit = 0
            else:
                digit = _THAI_DIGITS[tok]
        return total + current + digit

    baht = chunk(baht_part)
    if baht is None:
        return None
    satang = chunk(satang_part) if satang_part.strip() else 0
    return baht, (satang or 0)


def _close(a, b):
    return abs(a - b) <= TOLERANCE


def _check(name, detail, expected, actual, severity="error", tolerance=TOLERANCE,
           working=""):
    testable = expected is not None and actual is not None
    ok = testable and abs(expected - actual) <= tolerance
    if not testable:
        status = "skip"
    elif ok:
        status = "pass"
    else:
        # A warning-severity check that misses is a warning, not a failure --
        # otherwise a rounded per-line VAT would sink the whole verdict.
        status = "warn" if severity == "warning" else "fail"
    return {
        "name": name,
        "detail": detail,
        "expected": expected,
        "actual": actual,
        "difference": None if not testable else round(actual - expected, 2),
        "status": status,
        "severity": severity,
        # The sum itself, written out with the figures in it. A check that only
        # reports expected-vs-found leaves you to reconstruct where the expected
        # came from -- and on a document with a dropped row, seeing which
        # addends went into it is how you find the row.
        "working": working,
    }


def _money(value):
    return f"{value:,.2f}"


def _sum_of(items, key):
    values = [parse_amount(i.get(key)) for i in items if isinstance(i, dict)]
    values = [v for v in values if v is not None]
    return round(sum(values), 2) if values else None


def _sum_working(items, key, limit=8):
    """A column addition written out: '9,741.60 + 535.00 = 10,276.60'.

    Long tables are truncated: past eight rows the string stops being readable
    and the row count carries the same information.
    """
    values = [parse_amount(i.get(key)) for i in items if isinstance(i, dict)]
    values = [v for v in values if v is not None]
    if not values:
        return ""
    shown = " + ".join(_money(v) for v in values[:limit])
    if len(values) > limit:
        shown += f" + … [{len(values) - limit} more row(s)]"
    return f"{shown} = {_money(round(sum(values), 2))}"


def _vatable_base(items):
    """Line amounts that actually carry VAT, and the VAT charged on them.

    A document mixing VAT and non-VAT charges has a blended ratio well below its
    headline rate, so comparing VAT against the whole subtotal would flag every
    such document. Comparing against only the VAT-bearing lines is the fair test.
    The VAT is returned alongside because on a VAT-inclusive page it has to come
    back out of the base before the ratio means anything.
    """
    total = vat_total = 0.0
    found = False
    for item in items:
        amount = parse_amount(item.get("amount"))
        vat = parse_amount(item.get("vat"))
        if amount is not None and vat:
            total += amount
            vat_total += vat
            found = True
    return (round(total, 2), round(vat_total, 2)) if found else (None, None)


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
        # they are the ones being checked -- but the disagreement is reported
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


def run_checks(fields, transcript=None, basis=None):
    """Every arithmetic relation we can test on an invoice/receipt.

    `transcript` is optional; when given it is searched for a stated VAT rate if
    the extracted fields do not carry one, and for a statement of the VAT basis.
    `basis` is the `vat_basis` result when the caller already has one -- it is
    reported to the user alongside these checks, and both must come from the
    same decision.
    """
    if not isinstance(fields, dict):
        return []

    items = fields.get("line_items") or []
    items = [i for i in items if isinstance(i, dict)]

    def total_of(key):
        return _sum_of(items, key)

    if basis is None:
        basis = vat_basis(fields, transcript)
    inclusive = basis["inclusive"]

    subtotal = parse_amount(fields.get("subtotal"))
    vat = parse_amount(fields.get("vat_total"))
    incl = parse_amount(fields.get("amount_incl_vat"))
    wht = parse_amount(fields.get("withholding_tax_total"))
    net_total = parse_amount(fields.get("net_payable"))

    checks = []

    # 1. Line items must add up to whichever total they are stated on. On a
    #    VAT-exclusive page that is the subtotal; on a VAT-inclusive one the
    #    lines already carry the tax and reach the grand total instead, so
    #    testing them against the subtotal would fail the page by the VAT.
    if inclusive and not basis["subtotal_includes_vat"]:
        against = incl if incl is not None else (
            round(subtotal + (vat or 0), 2) if subtotal is not None else None)
        against_note = "amount incl. VAT" if incl is not None else "subtotal + VAT"
        checks.append(_check(
            "Line items sum to the VAT-inclusive total",
            f"sum of {len(items)} VAT-inclusive line amount(s) vs {against_note}"
            f" · basis {basis['source']}",
            total_of("amount"), against,
            working=_sum_working(items, "amount")))
    else:
        checks.append(_check(
            "Line items sum to subtotal",
            f"sum of {len(items)} line amount(s) vs subtotal"
            + (" (VAT included in both)" if basis["subtotal_includes_vat"] else ""),
            total_of("amount"), subtotal,
            working=_sum_working(items, "amount")))

    # 2. and 3. Same for the tax columns.
    checks.append(_check("Line VAT sums to VAT total",
                         "sum of per-line VAT vs vat_total", total_of("vat"), vat,
                         working=_sum_working(items, "vat")))
    checks.append(_check("Line W/T sums to W/T total",
                         "sum of per-line withholding vs withholding_tax_total",
                         total_of("withholding_tax"), wht,
                         working=_sum_working(items, "withholding_tax")))

    # 4. and 5. The headline relation, one rung at a time. Splitting it is what
    #    lets a break name itself: VAT added wrong is a different error from
    #    withholding deducted wrong, and the old combined check could not say
    #    which of the two had gone.
    #    Where the subtotal itself already carries VAT, adding the VAT to it
    #    would double-count: the relation to test is then that the two totals
    #    are the same figure, with the VAT contained inside it.
    if basis["subtotal_includes_vat"]:
        checks.append(_check(
            "Subtotal already includes VAT = amount incl. VAT",
            "the subtotal contains the VAT rather than adding it"
            f" · basis {basis['source']}",
            subtotal, incl,
            working=f"{_money(subtotal)} − {_money(vat or 0)} VAT inside it"
                    f" = {_money(round(subtotal - (vat or 0), 2))} before tax"
                    f" · nothing to add on top"))
    elif subtotal is not None and incl is not None:
        checks.append(_check(
            "Subtotal + VAT = amount incl. VAT",
            "subtotal + vat_total vs amount_incl_vat",
            round(subtotal + (vat or 0), 2), incl,
            working=f"{_money(subtotal)} + {_money(vat or 0)}"
                    f" = {_money(round(subtotal + (vat or 0), 2))}"))
    else:
        checks.append(_check("Subtotal + VAT = amount incl. VAT",
                             "needs subtotal and amount_incl_vat", None, None))

    # A document with no withholding line prints no net payable either, so the
    # second rung skips there rather than failing. Where the page never printed
    # the VAT-inclusive total, the subtotal stands in for it -- plus the VAT when
    # the subtotal excludes it, as it is, untouched, when it already includes it.
    if incl is not None:
        base, base_note = incl, "amount incl. VAT"
    elif subtotal is None:
        base, base_note = None, ""
    elif basis["subtotal_includes_vat"]:
        base, base_note = subtotal, "subtotal, VAT already in it"
    else:
        base, base_note = round(subtotal + (vat or 0), 2), "subtotal + VAT"
    if base is not None and net_total is not None:
        checks.append(_check(
            "Amount incl. VAT − W/T = net payable",
            f"{base_note} − withholding_tax_total vs net_payable",
            round(base - (wht or 0), 2), net_total,
            working=f"{_money(base)} [{base_note}] − {_money(wht or 0)} withheld"
                    f" = {_money(round(base - (wht or 0), 2))}"))
    else:
        checks.append(_check("Amount incl. VAT − W/T = net payable",
                             "needs a VAT-inclusive total and net_payable",
                             None, None))

    # 5b. Net column should also reach whichever payable total the page printed.
    payable = net_total if net_total is not None else incl
    checks.append(_check(
        "Line net amounts sum to the payable total",
        "sum of per-line net_amount vs "
        + ("net_payable" if net_total is not None else "amount_incl_vat"),
        total_of("net_amount"), payable,
        working=_sum_working(items, "net_amount")))

    # 6. VAT ratio against the rate the document itself states, falling back to
    #    the Thai standard. Only ever a warning: a document that mixes VAT and
    #    non-VAT lines legitimately blends below its headline rate, so the ratio
    #    is evidence, not proof.
    #    A rate is a fraction of the price BEFORE tax, so a base that already
    #    contains the tax has to have it taken back out first -- 7% VAT is 7% of
    #    the goods total but only 6.54% of the total it is included in, and
    #    comparing the second against 7 would flag every inclusive document.
    expected_rate, rate_source = resolve_vat_rate(fields, transcript)
    vatable, vatable_vat = _vatable_base(items)
    if vatable:
        base, base_note, contained = vatable, "VAT-bearing lines", vatable_vat if inclusive else 0.0
    elif subtotal:
        base, base_note = subtotal, "subtotal"
        contained = vat if basis["subtotal_includes_vat"] else 0.0
    else:
        base = None
    if base and vat is not None:
        net_base = round(base - (contained or 0), 2)
        if contained:
            base_note += f", less the {contained:,.2f} VAT inside it"
        if net_base:
            rate = vat / net_base * 100
            checks.append({
                "name": f"VAT rate is {expected_rate:g}%",
                "detail": f"vat_total against the {base_note} · rate {rate_source}",
                "expected": expected_rate, "actual": round(rate, 2),
                "difference": round(rate - expected_rate, 2),
                "status": "pass" if abs(rate - expected_rate) < RATE_TOLERANCE else "warn",
                "severity": "warning",
                # Written with the backing-out visible, because that step is the
                # one a reader would not have done in their head.
                "working": (f"{_money(vat)} ÷ ({_money(base)} − {_money(contained)}"
                            f" VAT inside it) × 100 = {rate:.2f}%" if contained else
                            f"{_money(vat)} ÷ {_money(net_base)} × 100 = {rate:.2f}%"),
            })

    # 6b. VAT charged on each line should match the rate too -- again as a share
    #     of the amount before tax, which on an inclusive line is the printed
    #     amount minus the VAT within it: r/(100+r) of what is printed.
    for index, item in enumerate(items, 1):
        amount = parse_amount(item.get("amount"))
        line_vat = parse_amount(item.get("vat"))
        if amount is None or line_vat is None or amount == 0 or line_vat == 0:
            continue  # a nil VAT cell is a non-VAT line, not a discrepancy
        if inclusive:
            name = f"Line {index}: VAT is the {expected_rate:g}% inside the amount"
            detail = f"the VAT cell vs the {expected_rate:g}/{100 + expected_rate:g} within a VAT-inclusive amount"
            expected = round(amount * expected_rate / (100 + expected_rate), 2)
            working = (f"{_money(amount)} × {expected_rate:g} ÷ {100 + expected_rate:g}"
                       f" = {_money(expected)}")
        else:
            name = f"Line {index}: VAT is {expected_rate:g}% of amount"
            detail = f"the VAT cell vs {expected_rate:g}% of a pre-tax amount"
            expected = round(amount * expected_rate / 100, 2)
            working = (f"{_money(amount)} × {expected_rate:g} ÷ 100 = {_money(expected)}")
        checks.append(_check(name, detail, expected, line_vat, working=working,
                             severity="warning", tolerance=0.5))  # per-line rounding is normal

    # 7. Amount in words vs the numeral -- catches a fluent misreading.
    words = fields.get("amount_in_words")
    parts = thai_words_parts(words) if words else None
    parsed = None if parts is None else parts[0] + parts[1] / 100.0
    # Thai forms usually spell out the VAT-inclusive total, but a document that
    # withholds sometimes spells the net instead. Either is legitimate, so the
    # check passes on a match with either and reports which one it matched.
    totals = [(t, name) for t, name in
              ((incl, "amount incl. VAT"), (net_total, "net payable")) if t is not None]
    if words and parsed is not None and totals:
        matched = [pair for pair in totals if _close(parsed, pair[0])]
        target, label = matched[0] if matched else totals[0]
        checks.append({
            "name": "Amount in words matches total",
            "detail": f"“{words}” reads as {_money(parsed)} · vs {label}",
            "expected": target, "actual": parsed,
            "difference": round(parsed - target, 2),
            "status": "pass" if matched else "fail",
            "severity": "error",
            "working": (f"{parts[0]:,} baht + {parts[1]:g} satang = {_money(parsed)}"
                        f"  ·  {label} {_money(target)}"),
        })
    else:
        checks.append({
            "name": "Amount in words matches total",
            "detail": ("not stated or could not be parsed" if not parsed
                       else "no total to compare against"),
            "expected": totals[0][0] if totals else None,
            "actual": parsed, "difference": None,
            "status": "skip", "severity": "error",
            "working": "",
        })

    # 8. Per-line internal consistency.
    for index, item in enumerate(items, 1):
        amount = parse_amount(item.get("amount"))
        net = parse_amount(item.get("net_amount"))
        if amount is None or net is None:
            continue
        # VAT is added to an exclusive amount to reach the net, but on an
        # inclusive one it is already there -- adding it again would charge the
        # tax twice and fail a line that is right.
        line_vat = 0 if inclusive else (parse_amount(item.get("vat")) or 0)
        line_wht = parse_amount(item.get("withholding_tax")) or 0
        expected = amount + line_vat - line_wht
        label = (item.get("description") or f"line {index}")[:40]
        working = _money(amount) + (" [VAT already in it]" if inclusive
                                    else f" + {_money(line_vat)} VAT")
        checks.append(_check(
            f"Line {index}: amount − W/T = net" if inclusive
            else f"Line {index}: amount + VAT − W/T = net",
            label + (" · amount already includes VAT" if inclusive else ""),
            round(expected, 2), net,
            working=f"{working} − {_money(line_wht)} W/T = {_money(round(expected, 2))}"))

    # 9. Quantity x unit price, where both are given.
    for index, item in enumerate(items, 1):
        qty = parse_amount(item.get("quantity"))
        unit = parse_amount(item.get("unit_price"))
        amount = parse_amount(item.get("amount"))
        if qty is None or unit is None or amount is None:
            continue
        checks.append(_check(f"Line {index}: qty × unit price = amount",
                             (item.get("description") or f"line {index}")[:40],
                             round(qty * unit, 2), amount,
                             working=f"{qty:g} × {_money(unit)}"
                                     f" = {_money(round(qty * unit, 2))}"))

    return checks


def summarise_checks(checks):
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    testable = counts["pass"] + counts["fail"] + counts["warn"]
    return {
        "counts": counts,
        "testable": testable,
        "verdict": "fail" if counts["fail"] else ("warn" if counts["warn"] else
                   ("pass" if counts["pass"] else "insufficient data")),
    }
