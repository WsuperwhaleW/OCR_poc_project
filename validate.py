"""The field requirement's validation rules, checked in Python after extraction.

Deterministic and model-free, the same standing as `normalise.py` and
`grounding.py`, and for the same reason recorded at length in CLAUDE.md: pass 2
is a lookup, and a model asked to *check* what it reads starts adjusting what it
reads to fit the check. Every rule below runs on values pass 2 copied verbatim
and `grounding.py` has already tested against the page.

Three things this module is not, and each is a deliberate limit:

* **It never rewrites a value.** A failed check is reported beside the value,
  which stays exactly as the model returned it. That is `grounding.py`'s standing
  rule -- flags, never rewrites -- applied to a second kind of finding, and the
  reasoning is the same: silently repairing a value is worse than showing it with
  a flag, because the repair cannot be reviewed.
* **It never gates anything.** No result is dropped, no run fails, and nothing
  here reaches an accuracy denominator. The corpus is the reason it must not:
  seventeen of the twenty tax IDs in `solution/` fail the checksum below, because
  they are anonymised mocks carrying invented numbers. A rule that rejected them
  would be scoring the fixture generator.
* **It does not know anything the page does not.** Two of the requirement's rules
  need data this process has not got -- a duplicate check needs the corpus of
  documents already filed, and reference matching needs the PO/GR/RTV system.
  Those return `unchecked` with the reason, never `ok`. **`unchecked` is a real
  answer and is never counted as a pass**; the whole point of separating it from
  `ok` is that a validation summary reading "all green" while half the rules
  never ran is the confidently-wrong number this project refuses everywhere.

The result rides on the extraction as `validation`, outside `fields`, exactly as
`normalise`'s `derived` does -- `grounding.check` walks `fields` and would report
a computed verdict as an invention.
"""

import datetime
import re

import grounding
import normalise
import prompts
import verify

# --------------------------------------------------------------------------
# the states a check can be in
# --------------------------------------------------------------------------

OK = "ok"
FAILED = "failed"
WARNING = "warning"          # worth a look, but a correct document can fail it
UNCHECKED = "unchecked"      # the rule could not be run -- see `why`
ABSENT = "absent"            # the field came back empty, so there is nothing to check

# `absent` is separated from `failed` because they are fixed in different places
# and by different people. An absent Mandatory field is an EXTRACTION problem and
# `grounding.check` already reports it as `missing`; a failed check is a problem
# with the value that came back. Merging them would report one fault twice under
# two names, which is the mistake the Fields tab's wording notes are all about.

# Only these two count towards the headline pass rate. `warning` is deliberately
# outside it: the VAT-rate rule is the only one that uses it, and it was
# warning-only in the verification pass this project removed, for a reason the
# corpus proves three times over -- a document mixing taxed and untaxed lines
# blends legitimately below the standard rate. sol001, sol004 and sol005 all do,
# at 4.4% to 4.8%, and every one of them is correct. Counting those as failures
# would make the rule's own headline useless on exactly the documents it is
# hardest to read.
_SCORED = (OK, FAILED)

# The states, in the order a reader should meet them.
STATES = (OK, FAILED, WARNING, UNCHECKED, ABSENT)

# The standard Thai VAT rate and how far a computed rate may sit from it. Both
# were `verify.py`'s and went out with the verification pass on 2026-08-17; the
# requirement asks for a VAT calculation by name, so they come back here rather
# than being reintroduced to a module whose remaining job is `parse_amount` and
# `vat_basis`. The tolerance is wide because documents round per line.
DEFAULT_VAT_RATE = 7.0
RATE_TOLERANCE = 0.15


# --------------------------------------------------------------------------
# Thai tax identification numbers
# --------------------------------------------------------------------------

TAX_ID_DIGITS = normalise.TAX_ID_DIGITS


def tax_id_checksum_ok(value):
    """Whether a 13-digit Thai tax ID satisfies its mod-11 check digit.

    The first twelve digits are weighted 13 down to 2, summed, and the check
    digit is `(11 - sum % 11) % 10`. Returns None where the value is not thirteen
    digits at all, because "wrong length" and "wrong check digit" are different
    findings with different causes -- a short ID is usually a misread or a
    genuinely malformed number on the page, and `solution/sol004` prints one.
    """
    digits = normalise.tax_id_digits(value)
    if len(digits) != TAX_ID_DIGITS or not digits.isdigit():
        return None
    total = sum(int(digits[i]) * (TAX_ID_DIGITS - i) for i in range(12))
    return (11 - total % 11) % 10 == int(digits[12])


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

# Thai documents date themselves in the Buddhist era as often as not, and in this
# corpus BOTH two-digit conventions appear on real pages: sol001 prints 05/02/26
# for February 2026 (a two-digit CE year) and sol003 prints 17/3/69 for the same
# year (a two-digit BE year, 2569). One heuristic separates them, and it is the
# ordinary Thai one:
#
#   yy in 60..99  ->  BE 25yy      ->  CE 25yy - 543
#   yy in 00..59  ->  CE 20yy
#
# It is a heuristic and it is stated as one. It is safe for every year this
# project will meet -- it breaks for a CE year of 60-99, i.e. 2060 onwards -- and
# the alternative, refusing every two-digit year, would leave the future-date
# rule unable to run on half the corpus.
_BE_OFFSET = 543
_BE_SHORT_FROM = 60
_BE_FULL_FROM = 2400

_DATE_SEPARATORS = r"[/.\-\s]"
_NUMERIC_DATE = re.compile(
    rf"^\s*(\d{{1,2}}){_DATE_SEPARATORS}(\d{{1,2}}){_DATE_SEPARATORS}(\d{{2}}|\d{{4}})\s*$")

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "ม.ค": 1, "ก.พ": 2, "มี.ค": 3, "เม.ย": 4, "พ.ค": 5, "มิ.ย": 6,
    "ก.ค": 7, "ส.ค": 8, "ก.ย": 9, "ต.ค": 10, "พ.ย": 11, "ธ.ค": 12,
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
    "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
    "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
_NAMED_DATE = re.compile(
    r"^\s*(\d{1,2})\s*[/.\-\s]?\s*([A-Za-z]{3,}|[\u0e00-\u0e7f.]{2,})\s*[/.\-\s]?\s*(\d{2}|\d{4})\s*$")

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def _year_to_ce(year: int) -> int:
    """A printed year as a Gregorian one. See the heuristic above."""
    if year >= _BE_FULL_FROM:
        return year - _BE_OFFSET
    if year >= 100:
        return year
    if year >= _BE_SHORT_FROM:
        return 2500 + year - _BE_OFFSET
    return 2000 + year


def parse_date(value):
    """A printed date as `datetime.date`, or None where it is not one.

    Day-first throughout: every date in this corpus is, and a month-first reading
    of 05/02/26 is a different day rather than a failure to parse -- which is
    exactly the kind of silently wrong answer that must not be guessed at. A
    value whose first number is above 12 confirms day-first; one where both are
    12 or under is ambiguous in principle and read day-first anyway, because the
    documents are Thai.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().translate(_THAI_DIGITS)
    if not text:
        return None

    match = _NUMERIC_DATE.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
    else:
        match = _NAMED_DATE.match(text)
        if not match:
            return None
        day = int(match.group(1))
        name = match.group(2).lower().rstrip(".")
        month = _MONTHS.get(name) or _MONTHS.get(name[:3])
        if not month:
            return None
        year = int(match.group(3))

    try:
        return datetime.date(_year_to_ce(year), month, day)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

# A document, order or reference number: something with at least one letter or
# digit in it, on one line, and not absurdly long. Deliberately loose -- the
# corpus alone prints `CR2024/030`, `BYD875419`, `510260007343`, `6611/2023` and
# `2026-10301489`, and a tighter pattern would reject a real number rather than
# catch a wrong one. What it is really for is the failure it CAN catch: a key
# that came back holding a sentence, a label, or a line of the page.
_NUMBER_PATTERN = re.compile(r"^[^\r\n]{1,64}$")
_HAS_ALNUM = re.compile(r"[0-9A-Za-z\u0e00-\u0e7f]")

# The five-digit branch code the requirement asks for, after `normalise` has
# turned สำนักงานใหญ่ into 00000 and padded a printed branch number.
_BRANCH_CODE = re.compile(r"^\d{5}$")


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def _result(field, rule, state, why="", detail="", mandatory=False):
    row = {"field": field, "rule": rule, "state": state}
    if why:
        row["why"] = why
    if detail:
        row["detail"] = detail
    # Set here only by a check whose field is not a scalar key -- an income row's
    # cell. `validate` marks the scalars itself, from the form's own Mandatory
    # list, and cannot match a path like income_items[0].amount_paid against it.
    if mandatory:
        row["mandatory"] = True
    return row


def _blank(value):
    return not str(value or "").strip()


def _check_document_type(fields, expected):
    """The heading names a known type, and one the document was expected to be.

    Compared as SETS and satisfied by an OVERLAP, not by equality. A page headed
    ใบเสร็จรับเงิน/ใบกำกับภาษี names two types; the form was built from the union
    of them, and a heading that names any of the expected types is the heading
    that form was asked for. Demanding equality would fail a correct extraction
    whenever the manifest records a reading narrower than the page's own -- which
    is exactly what the manifest is for.
    """
    raw = fields.get("document_type")
    if _blank(raw):
        return [_result("document_type", "pattern", ABSENT)]
    codes = normalise.document_types(raw)
    if not codes:
        return [_result("document_type", "pattern", FAILED,
                        "the heading does not match any known document type",
                        str(raw)[:80])]
    found = " + ".join(codes)
    if expected and not (set(codes) & set(expected)):
        return [_result("document_type", "pattern", FAILED,
                        f"reads as {found} where "
                        f"{' + '.join(expected)} was expected", found)]
    return [_result("document_type", "pattern", OK, "", found)]


def _check_number(fields, key, rules, seen=None):
    """Pattern, plus whichever second rule this document's type asks for.

    The two requirements name different ones for the same field, and that is a
    real difference rather than two words for one check:

      Invoice      Pattern Check + **Duplicate Check** -- has this number been
                   filed before? A question about every document already filed.
      Credit Note  Pattern Check + **Original Document Matching** -- does the
                   invoice this credits exist? A question about a different
                   document in a different system.

    Neither is answerable from one page, so both return `unchecked` unless the
    caller supplies what they need. A type whose requirement names neither gets
    the pattern check alone: running a rule nobody asked for would report a
    finding against a document that is not held to it.
    """
    value = fields.get(key)
    extra = [r for r in ("duplicate", "original_document") if r in rules]
    if _blank(value):
        return ([_result(key, "pattern", ABSENT)]
                + [_result(key, _RULE_NAMES[r], ABSENT) for r in extra])
    text = str(value).strip()
    rows = []
    if _NUMBER_PATTERN.match(text) and _HAS_ALNUM.search(text):
        rows.append(_result(key, "pattern", OK, "", text))
    else:
        rows.append(_result(key, "pattern", FAILED,
                            "not a document number -- it holds a line break, or "
                            "no letter or digit at all", text[:80]))

    if "duplicate" in rules:
        # `seen` is the caller's own corpus of numbers already filed, where it
        # has one. This process does not, and must not pretend to.
        if seen is None:
            rows.append(_result(key, "duplicate", UNCHECKED,
                                "needs the set of numbers already filed; none "
                                "was supplied to this run"))
        elif grounding.squash(text) in {grounding.squash(s) for s in seen}:
            rows.append(_result(key, "duplicate", FAILED,
                                "this number has been seen before", text))
        else:
            rows.append(_result(key, "duplicate", OK, "", text))

    if "original_document" in rules:
        rows.append(_result(key, "original document matching", UNCHECKED,
                            "needs the document this one credits, in the system "
                            "that holds it; this process has no access to one"))
    return rows


# The rule names as they are reported, keyed by the internal name.
_RULE_NAMES = {"duplicate": "duplicate",
               "original_document": "original document matching"}


def _check_issue_date(fields, rules, today=None):
    """Date format always; the future-date rule only where a type asks for it.

    The invoice requirement says "Date Format + Future Date Validation"; the
    credit note says "Date Format Validation" and stops there. That is not an
    oversight to tidy up -- a credit note legitimately carries a date after the
    invoice it corrects, and holding one to an invoice's rule would report a
    finding against a correct document.
    """
    future = "future_date" in rules
    value = fields.get("issue_date")
    if _blank(value):
        rows = [_result("issue_date", "date format", ABSENT)]
        if future:
            rows.append(_result("issue_date", "not in the future", ABSENT))
        return rows
    parsed = parse_date(value)
    if parsed is None:
        rows = [_result("issue_date", "date format", FAILED,
                        "does not read as a date", str(value)[:80])]
        if future:
            rows.append(_result("issue_date", "not in the future", UNCHECKED,
                                "the date could not be read"))
        return rows
    today = today or datetime.date.today()
    rows = [_result("issue_date", "date format", OK, "", parsed.isoformat())]
    if not future:
        return rows
    if parsed > today:
        rows.append(_result("issue_date", "not in the future", FAILED,
                            f"dated {parsed.isoformat()}, after {today.isoformat()}",
                            parsed.isoformat()))
    else:
        rows.append(_result("issue_date", "not in the future", OK, "",
                            parsed.isoformat()))
    return rows


def _check_tax_id(fields, key):
    value = fields.get(key)
    if _blank(value):
        return [_result(key, "13-digit format", ABSENT),
                _result(key, "checksum", ABSENT)]
    digits = normalise.tax_id_digits(value)
    rows = []
    if len(digits) == TAX_ID_DIGITS:
        rows.append(_result(key, "13-digit format", OK, "", digits))
    else:
        rows.append(_result(key, "13-digit format", FAILED,
                            f"{len(digits)} digits, not {TAX_ID_DIGITS}", digits))
    ok = tax_id_checksum_ok(value)
    if ok is None:
        rows.append(_result(key, "checksum", UNCHECKED,
                            "the check digit can only be tested on 13 digits"))
    elif ok:
        rows.append(_result(key, "checksum", OK, "", digits))
    else:
        rows.append(_result(key, "checksum", FAILED,
                            "the 13th digit is not the mod-11 check digit of the "
                            "first twelve", digits))
    return rows


def _check_branch(fields, key):
    value = fields.get(key)
    if _blank(value):
        return [_result(key, "5-digit code", ABSENT)]
    code = normalise.branch_code(value)
    if code and _BRANCH_CODE.match(code):
        return [_result(key, "5-digit code", OK, "", code)]
    # `normalise.branch_code` deliberately refuses to default an unnumbered
    # branch to 00000 -- filing a branch's documents against the head office is
    # the one error that normalisation exists to prevent. So this is a real
    # finding about the PAGE: it names a branch and prints no code for it.
    return [_result(key, "5-digit code", FAILED,
                    "the page names a branch but prints no number for it, so no "
                    "five-digit code can be derived -- and an unnumbered branch "
                    "is not the head office",
                    str(value)[:80])]


def _check_amounts(fields, asked, transcript=None):
    """Numeric format on the three totals, then the two arithmetic rules.

    The arithmetic is the one part of the removed verification pass that the
    requirement asks for by name, and it is restored here in its narrowest form:
    two identities over three printed figures, never over the line-item table.
    """
    rows = []
    amounts = {}
    # Only the totals this document's form actually asked for. A receipt is not
    # asked for the with-tax total -- its requirement does not list one -- and
    # checking a key nobody requested reports an absent field against a document
    # that was never going to fill it.
    for key in [k for k in ("subtotal", "vat_total", "amount_incl_vat")
                if k in asked]:
        value = fields.get(key)
        if _blank(value):
            rows.append(_result(key, "numeric", ABSENT))
            continue
        if grounding.is_nil(value):
            # The page printed a dash where a figure would go. That is a stated
            # nil, not an unreadable amount -- sol009 prints its VAT line that
            # way because it charges no tax -- so it reads as zero and says so.
            amounts[key] = 0.0
            rows.append(_result(key, "numeric", OK, "", "nil (printed as a dash)"))
            continue
        parsed = verify.parse_amount(value)
        amounts[key] = parsed
        if parsed is None:
            rows.append(_result(key, "numeric", FAILED,
                                "does not read as an amount", str(value)[:80]))
        else:
            rows.append(_result(key, "numeric", OK, "", f"{parsed:,.2f}"))

    sub = amounts.get("subtotal")
    vat = amounts.get("vat_total")
    incl = amounts.get("amount_incl_vat")

    # subtotal + VAT = total including VAT. Not run at all where the form has no
    # with-tax total: it is not that the page failed to print one, it is that
    # nothing asked.
    if "amount_incl_vat" not in asked:
        pass
    elif sub is None or vat is None or incl is None:
        rows.append(_result("amount_incl_vat", "total = subtotal + VAT", UNCHECKED,
                            "the page does not print all three figures"))
    elif abs(sub + vat - incl) <= verify.TOLERANCE:
        rows.append(_result("amount_incl_vat", "total = subtotal + VAT", OK, "",
                            f"{sub:,.2f} + {vat:,.2f} = {incl:,.2f}"))
    else:
        rows.append(_result("amount_incl_vat", "total = subtotal + VAT", FAILED,
                            f"{sub:,.2f} + {vat:,.2f} = {sub + vat:,.2f}, "
                            f"but the page prints {incl:,.2f}",
                            f"out by {abs(sub + vat - incl):,.2f}"))

    # VAT is the standard rate of the figure it is charged on. Which figure that
    # is depends on the basis, which `verify.vat_basis` settles from the numbers
    # where it can and from the wording where it cannot -- the one surviving
    # piece of the removed pass, and the reason it survived.
    if sub is None or vat is None:
        rows.append(_result("vat_total", "VAT rate", UNCHECKED,
                            "the page does not print both figures"))
    elif vat == 0:
        # A zero or dashed VAT line is a real state -- sol009 charges no tax --
        # and testing a rate against it would fail every exempt document.
        rows.append(_result("vat_total", "VAT rate", UNCHECKED,
                            "the page charges no VAT, so there is no rate to check"))
    else:
        basis = verify.vat_basis(fields, transcript)
        inclusive = basis.get("basis") == "inclusive"
        base = (incl - vat if inclusive and incl is not None else sub)
        rate = (vat / base * 100) if base else None
        if rate is None:
            rows.append(_result("vat_total", "VAT rate", UNCHECKED,
                                "no figure to charge the VAT on"))
        elif abs(rate - DEFAULT_VAT_RATE) <= RATE_TOLERANCE:
            rows.append(_result("vat_total", "VAT rate", OK, "",
                                f"{rate:.2f}% of {base:,.2f}"))
        else:
            # A warning rather than an error in the removed pass, and for the
            # same reason: a document mixing VAT and non-VAT lines legitimately
            # blends below the standard rate. It is reported as failed here
            # because nothing gates on it, and a rate that is not 7% is the one
            # signal that a total landed in the wrong key.
            rows.append(_result("vat_total", "VAT rate", WARNING,
                                f"{rate:.2f}% of {base:,.2f}, not "
                                f"{DEFAULT_VAT_RATE:.0f}% -- legitimate on a page "
                                f"that mixes taxed and untaxed lines, and "
                                f"otherwise a sign that a figure is in the wrong "
                                f"key",
                                f"{rate:.2f}%"))
    return rows


def _check_po_gr_rtv(fields):
    value = fields.get("po_gr_rtv_number")
    if _blank(value):
        return [_result("po_gr_rtv_number", "pattern", ABSENT),
                _result("po_gr_rtv_number", "reference matching", ABSENT)]
    text = str(value).strip()
    rows = []
    if _NUMBER_PATTERN.match(text) and _HAS_ALNUM.search(text):
        rows.append(_result("po_gr_rtv_number", "pattern", OK, "", text))
    else:
        rows.append(_result("po_gr_rtv_number", "pattern", FAILED,
                            "not an order or receipt number", text[:80]))
    rows.append(_result("po_gr_rtv_number", "reference matching", UNCHECKED,
                        "needs the PO/GR/RTV system to match against; this "
                        "process has no access to one"))
    return rows


def _check_inv_rtv_cnr(fields):
    """Pattern, and the reference matching the receipt requirement names.

    Matching is the same shape as the invoice's PO/GR/RTV rule and unanswerable
    for the same reason: it asks whether a document in another system exists.
    """
    value = fields.get("inv_rtv_cnr_number")
    if _blank(value):
        return [_result("inv_rtv_cnr_number", "pattern", ABSENT),
                _result("inv_rtv_cnr_number", "reference matching", ABSENT)]
    text = str(value).strip()
    rows = []
    if _NUMBER_PATTERN.match(text) and _HAS_ALNUM.search(text):
        rows.append(_result("inv_rtv_cnr_number", "pattern", OK, "", text))
    else:
        rows.append(_result("inv_rtv_cnr_number", "pattern", FAILED,
                            "not a document number", text[:80]))
    rows.append(_result("inv_rtv_cnr_number", "reference matching", UNCHECKED,
                        "needs the document this payment settles, in the system "
                        "that holds it; this process has no access to one"))
    return rows


def _check_remaining(fields, rules):
    """The two balance figures: numeric, and the two rules the receipt asks for.

    **`vat_balance` is the only one of the three external-looking rules that can
    actually be run here**, and only when the page prints both figures: the tax
    part of a balance should be the standard rate of the balance it is part of.
    `outstanding_balance` cannot -- it asks what was owed BEFORE this payment,
    which is a fact about another document.
    """
    rows = []
    amounts = {}
    for key in ("remaining_amount", "remaining_vat_amount"):
        value = fields.get(key)
        if _blank(value):
            rows.append(_result(key, "numeric", ABSENT))
            continue
        if grounding.is_nil(value):
            amounts[key] = 0.0
            rows.append(_result(key, "numeric", OK, "", "nil (printed as a dash)"))
            continue
        parsed = verify.parse_amount(value)
        amounts[key] = parsed
        if parsed is None:
            rows.append(_result(key, "numeric", FAILED,
                                "does not read as an amount", str(value)[:80]))
        else:
            rows.append(_result(key, "numeric", OK, "", f"{parsed:,.2f}"))

    if "outstanding_balance" in rules:
        if "remaining_amount" not in amounts:
            rows.append(_result("remaining_amount", "outstanding balance", ABSENT))
        else:
            rows.append(_result("remaining_amount", "outstanding balance", UNCHECKED,
                                "needs what was owed before this payment, which "
                                "is a fact about another document"))

    if "vat_balance" in rules:
        base = amounts.get("remaining_amount")
        vat = amounts.get("remaining_vat_amount")
        if base is None or vat is None:
            rows.append(_result("remaining_vat_amount", "VAT balance", ABSENT
                                if not amounts else UNCHECKED,
                                "" if not amounts else
                                "the page does not print both balance figures"))
        elif vat == 0 and base == 0:
            rows.append(_result("remaining_vat_amount", "VAT balance", OK, "",
                                "nothing outstanding"))
        elif not base:
            rows.append(_result("remaining_vat_amount", "VAT balance", FAILED,
                                "a VAT balance is outstanding against a zero "
                                "balance", f"{vat:,.2f}"))
        else:
            rate = vat / base * 100
            if abs(rate - DEFAULT_VAT_RATE) <= RATE_TOLERANCE:
                rows.append(_result("remaining_vat_amount", "VAT balance", OK, "",
                                    f"{rate:.2f}% of {base:,.2f}"))
            else:
                # A warning for the same reason the VAT-rate rule is one: a
                # balance covering both taxed and untaxed lines blends below the
                # standard rate, and that is a correct document.
                rows.append(_result("remaining_vat_amount", "VAT balance", WARNING,
                                    f"{rate:.2f}% of the outstanding "
                                    f"{base:,.2f}, not {DEFAULT_VAT_RATE:.0f}%",
                                    f"{rate:.2f}%"))
    return rows


def _check_payment_date(fields):
    value = fields.get("payment_date")
    if _blank(value):
        return [_result("payment_date", "date format", ABSENT)]
    parsed = parse_date(value)
    if parsed is None:
        return [_result("payment_date", "date format", FAILED,
                        "does not read as a date", str(value)[:80])]
    # Date Format Validation only -- the receipt requirement names no future-date
    # rule for it, and a post-dated cheque is an ordinary thing.
    return [_result("payment_date", "date format", OK, "", parsed.isoformat())]


# --------------------------------------------------------------------------
# the withholding tax certificate (2026-09-01)
# --------------------------------------------------------------------------
# Its party keys reuse `_check_tax_id` and `_check_branch` unchanged: the
# requirement asks the same two things of them -- thirteen digits and a
# five-digit branch code -- and a second copy of either rule would drift from the
# first. What is new here is the three rules no other requirement states.

def _check_company_master(fields, key, rules):
    """Fuzzy match against the Company Master, which this process has not got.

    The requirement asks for a name matched at 90% or better against a company
    master, after normalising the Thai company-form abbreviations. There is no
    company master in this repo and no service to ask, so the honest answer is
    `unchecked` with the reason -- **never `ok`**. That is the same standing as
    the duplicate and reference-matching rules, and the module docstring's whole
    argument for keeping `unchecked` out of the pass rate.
    """
    if "company_master" not in rules:
        return []
    value = fields.get(key)
    if _blank(value):
        return [_result(key, "company master match", ABSENT)]
    return [_result(key, "company master match", UNCHECKED,
                    "there is no Company Master to match against here -- the "
                    "rule needs the list of companies already on file",
                    str(value)[:80])]


def _income_rows(fields):
    """The income rows as a list of dicts, however the model returned them."""
    rows = (fields or {}).get(prompts.INCOME_ITEMS_KEY) or []
    return [r for r in rows if isinstance(r, dict)]


def _check_income_items(fields, demanded=()):
    """Every rule the requirement states against a row of the income table.

    A blank cell the requirement demands is reported as an absent Mandatory field
    of that ROW rather than of the document -- `mandatory` is set here rather than
    by `validate`'s scalar loop, which matches paths against the form's key list
    and could never match `income_items[0].amount_paid`.

    `demanded` is `prompts.mandatory_items_for_types` and is read rather than
    assumed: which cells a requirement marks Mandatory is stated once, in
    `prompts.MANDATORY_ITEMS`, and a second copy here would eventually disagree
    with the page drawing the same fact on the column headings.

    The one cross-field rule is the requirement's own: the tax withheld on a row
    may not exceed the amount paid on it.
    """
    demanded = set(demanded or ())
    rows = _income_rows(fields)
    if not rows:
        # Not `absent` per cell -- there are no cells. One row saying the table
        # never came back is worth more than four saying nothing about nothing.
        return [_result(prompts.INCOME_ITEMS_KEY, "table extraction", ABSENT,
                        "no income rows came back",
                        mandatory=bool(demanded))]
    out = []
    for index, row in enumerate(rows):
        path = f"{prompts.INCOME_ITEMS_KEY}[{index}]"
        amounts = {}
        for key in ("amount_paid", "wht_amount"):
            value = row.get(key)
            field = f"{path}.{key}"
            if _blank(value):
                out.append(_result(field, "amount", ABSENT,
                                   mandatory=key in demanded))
                continue
            parsed = verify.parse_amount(value)
            amounts[key] = parsed
            if parsed is None:
                out.append(_result(field, "amount", FAILED,
                                   "does not read as an amount", str(value)[:80]))
            elif parsed < 0:
                out.append(_result(field, "amount", FAILED,
                                   "an amount on this form cannot be negative",
                                   f"{parsed:,.2f}"))
            else:
                out.append(_result(field, "amount", OK, "", f"{parsed:,.2f}"))

        paid, tax = amounts.get("amount_paid"), amounts.get("wht_amount")
        field = f"{path}.wht_amount"
        if paid is None or tax is None:
            out.append(_result(field, "tax not above the amount paid", UNCHECKED,
                               "both figures on the row have to read as amounts"))
        elif tax <= paid + verify.TOLERANCE:
            out.append(_result(field, "tax not above the amount paid", OK, "",
                               f"{tax:,.2f} of {paid:,.2f}"))
        else:
            out.append(_result(field, "tax not above the amount paid", FAILED,
                               "the tax withheld is larger than the amount it "
                               "was withheld from",
                               f"{tax:,.2f} of {paid:,.2f}"))

        # The description and the date. The date carries the requirement's
        # "Date Format (BE to AD)", which `parse_date` already does for the whole
        # project -- a Thai two-digit year and a Buddhist-era four-digit one both
        # land on the Western year here.
        value = row.get("income_type")
        if _blank(value):
            out.append(_result(f"{path}.income_type", "stated", ABSENT,
                               mandatory="income_type" in demanded))
        else:
            out.append(_result(f"{path}.income_type", "stated", OK, "",
                               str(value)[:80]))

        value = row.get("payment_date")
        field = f"{path}.payment_date"
        if _blank(value):
            out.append(_result(field, "date format", ABSENT,
                               mandatory="payment_date" in demanded))
        else:
            parsed = parse_date(value)
            if parsed is None:
                out.append(_result(field, "date format", FAILED,
                                   "does not read as a date", str(value)[:80]))
            else:
                out.append(_result(field, "date format", OK, "",
                                   parsed.isoformat()))
    return out


def _check_wht_totals(fields, asked, rules):
    """Numeric format on the two totals, then the requirement's Sum Reconciliation.

    The only arithmetic rule in any of these requirements that runs over a TABLE
    rather than over printed totals alone: each total must equal the sum of its
    own column. It is skipped rather than failed where a row's figure could not
    be read -- a sum missing one addend is not evidence that the total is wrong,
    and reporting it as such would blame the totals line for a bad row.
    """
    rows = _income_rows(fields)
    columns = {"total_amount_paid": "amount_paid",
               "total_wht_amount": "wht_amount"}
    out = []
    for key in [k for k in ("total_amount_paid", "total_wht_amount")
                if k in asked]:
        value = fields.get(key)
        if _blank(value):
            out.append(_result(key, "numeric", ABSENT))
            continue
        parsed = verify.parse_amount(value)
        if parsed is None:
            out.append(_result(key, "numeric", FAILED,
                               "does not read as an amount", str(value)[:80]))
            continue
        out.append(_result(key, "numeric", OK, "", f"{parsed:,.2f}"))

        if "sum_reconciliation" not in rules:
            continue
        if not rows:
            out.append(_result(key, "sum reconciliation", UNCHECKED,
                               "no income rows came back to add up"))
            continue
        cells = [verify.parse_amount(r.get(columns[key])) for r in rows]
        if any(c is None for c in cells):
            out.append(_result(key, "sum reconciliation", UNCHECKED,
                               "at least one row's figure does not read as an "
                               "amount, so the column cannot be added up"))
            continue
        total = round(sum(cells), 2)
        working = f"{' + '.join(f'{c:,.2f}' for c in cells[:8])}"
        if len(cells) > 8:
            working += f" + {len(cells) - 8} more"
        working += f" = {total:,.2f}"
        if abs(total - parsed) <= verify.TOLERANCE:
            out.append(_result(key, "sum reconciliation", OK, "", working))
        else:
            out.append(_result(key, "sum reconciliation", FAILED,
                               f"the rows come to {total:,.2f} and the total "
                               f"line reads {parsed:,.2f}", working))
    return out


# What an income row has to say for the dividend rate option to be required.
# Squashed, so the brackets and spacing of 40(4)(kho) do not matter -- see
# `normalise._needles` for why a Thai needle has to be squashed at all.
_DIVIDEND_MARKS = tuple(grounding.squash(w) for w in ("4(ข)", "เงินปันผล",
                                                      "dividend"))


def _check_dividend_option(fields, rules):
    """The rate option, which is required only on a dividend row.

    "required when Income Type = 4(kho)" -- so on every other certificate an
    empty answer is correct and reporting it as absent would put a standing
    finding against the great majority of them. The condition is read off the
    income rows, which is the only place the page states it.
    """
    if "dividend_option" not in rules:
        return []
    value = fields.get("dividend_rate_option")
    rows = _income_rows(fields)
    dividend = any(any(m in grounding.squash(r.get("income_type") or "")
                       for m in _DIVIDEND_MARKS) for r in rows)
    if not _blank(value):
        return [_result("dividend_rate_option", "checkbox detection", OK, "",
                        str(value)[:80])]
    if not rows:
        return [_result("dividend_rate_option", "checkbox detection", UNCHECKED,
                        "the rule applies only to a dividend row, and no income "
                        "rows came back to say whether there is one")]
    if dividend:
        return [_result("dividend_rate_option", "checkbox detection", ABSENT,
                        "a row states dividend income, which is the one case "
                        "the requirement makes this field required",
                        mandatory=True)]
    return [_result("dividend_rate_option", "checkbox detection", OK,
                    "", "no dividend row, so no option is required")]


# Which checks a field asks for. A field that is not in the extraction's field
# set is not checked at all -- validating a key nobody asked for would report a
# fault against a document of the wrong type.
_CHECKS = {
    "document_type": lambda f, ctx: _check_document_type(f, ctx["expected_type"]),
    "document_number": lambda f, ctx: _check_number(f, "document_number",
                                                    ctx["rules"], ctx["seen"]),
    "issue_date": lambda f, ctx: _check_issue_date(f, ctx["rules"], ctx["today"]),
    "po_gr_rtv_number": lambda f, ctx: _check_po_gr_rtv(f),
    "seller_tax_id": lambda f, ctx: _check_tax_id(f, "seller_tax_id"),
    "buyer_tax_id": lambda f, ctx: _check_tax_id(f, "buyer_tax_id"),
    "seller_branch": lambda f, ctx: _check_branch(f, "seller_branch"),
    "buyer_branch": lambda f, ctx: _check_branch(f, "buyer_branch"),
    # The three totals are one check group -- the arithmetic needs all of them --
    # so it hangs off the first of them and the other two are covered by it.
    "subtotal": lambda f, ctx: _check_amounts(f, ctx["asked"], ctx["transcript"]),
    "inv_rtv_cnr_number": lambda f, ctx: _check_inv_rtv_cnr(f),
    # Both balance figures are one group, for the same reason as the totals: the
    # VAT-balance rule needs the pair.
    "remaining_amount": lambda f, ctx: _check_remaining(f, ctx["rules"]),
    "payment_date": lambda f, ctx: _check_payment_date(f),
    # `cheque_number` has no validation rule in the requirement -- the cell is
    # blank -- so it has no entry here. A field with nothing to check is not a
    # field that passes; it simply is not checked, and inventing a rule for it
    # would report findings nobody asked for.

    # ---- the withholding tax certificate ----------------------------------
    # The two parties reuse the commercial documents' rules unchanged: the WHT
    # requirement asks the same Length Validation of a tax ID (13 digits) and of
    # a branch (5 digits).
    "payer_tax_id": lambda f, ctx: _check_tax_id(f, "payer_tax_id"),
    "payee_tax_id": lambda f, ctx: _check_tax_id(f, "payee_tax_id"),
    "payer_branch": lambda f, ctx: _check_branch(f, "payer_branch"),
    "payee_branch": lambda f, ctx: _check_branch(f, "payee_branch"),
    # The requirement writes the fuzzy Company Master match against the Payer
    # Name only. It is run against BOTH names here: the two blocks hold the same
    # four things, both are companies on the same master, and a rule checking one
    # side of a two-sided form would read as an oversight rather than as a
    # reading. It cannot run either way -- see `_check_company_master`.
    "payer_name": lambda f, ctx: _check_company_master(f, "payer_name",
                                                       ctx["rules"]),
    "payee_name": lambda f, ctx: _check_company_master(f, "payee_name",
                                                       ctx["rules"]),
    # The two totals are one check group, like the three commercial ones: Sum
    # Reconciliation needs the rows and both columns, so it hangs off the first
    # of them and covers the second.
    "total_amount_paid": lambda f, ctx: _check_wht_totals(f, ctx["asked"],
                                                          ctx["rules"]),
    "dividend_rate_option": lambda f, ctx: _check_dividend_option(f, ctx["rules"]),
    # `book_no`, `certificate_no`, `sequence_no`, `payer_address` and
    # `payee_address` have no validation cell in the requirement, so like
    # `cheque_number` they have no entry.
}


def validate(fields, keys=None, doc_types=(), mandatory=(), transcript=None,
             seen=None, today=None, items=()):
    """Run every rule the requirement states for the fields that were asked for.

    `keys` is the extraction's field set, `doc_types` every type the document
    was taken to be, `mandatory` the union of what the requirement demands of
    those types, `seen` an optional collection of document numbers already filed
    (for the duplicate rule), `today` an override for the future-date rule so a
    test can be written that does not expire, and `items` the row shape of a TABLE
    the form asked for (`prompts.items_for_types`).

    `items` is separate from `keys` because a row is not a key: its rules run per
    row and report against a path rather than a field name, so they cannot hang
    off the `_CHECKS` table, which is keyed by scalar. Empty means this form
    asked for no table, and none of those rules runs.

    Returns {"checks": [...], "counts": {...}, "failed": [...], "doc_type": ...}.
    Never raises: a validation that fell over would take an extraction result
    with it, and the extraction is the thing the caller asked for.
    """
    fields = fields if isinstance(fields, dict) else {}
    asked = list(keys) if keys is not None else list(_CHECKS)
    # A list, because a document is regularly more than one type and the form
    # was built from the union of them.
    doc_types = ([doc_types] if isinstance(doc_types, str) and doc_types
                 else list(doc_types or []))
    # Which of the type-specific rules this document is held to -- the union
    # over its types, from the requirement's own tables.
    rules = set(prompts.rules_for_types(doc_types))
    ctx = {"expected_type": doc_types, "seen": seen, "today": today,
           "transcript": transcript, "rules": rules, "asked": set(asked)}

    checks = []
    for key in asked:
        runner = _CHECKS.get(key)
        if not runner:
            continue
        try:
            checks.extend(runner(fields, ctx))
        except Exception as err:                     # noqa: BLE001 - see docstring
            checks.append(_result(key, "check", UNCHECKED,
                                  f"the check itself failed: {err}"))

    # The table's rules, where this form asked for a table. Outside the loop
    # above because they run per ROW and report against a path -- see the
    # docstring.
    if items:
        try:
            checks.extend(_check_income_items(
                fields, prompts.mandatory_items_for_types(doc_types)))
        except Exception as err:                     # noqa: BLE001
            checks.append(_result(prompts.INCOME_ITEMS_KEY, "check", UNCHECKED,
                                  f"the check itself failed: {err}"))

    demanded = set(mandatory or ())
    for row in checks:
        # A Mandatory field that came back empty is worth separating from an
        # Optional one that did: the first is a compliance failure of the run and
        # the second is a document that legitimately says nothing.
        if row["state"] == ABSENT and row["field"] in demanded:
            row["mandatory"] = True

    counts = {state: sum(1 for r in checks if r["state"] == state)
              for state in STATES}
    counts["scored"] = sum(1 for r in checks if r["state"] in _SCORED)
    # A rate over the rules that actually RAN. `unchecked` and `absent` are not
    # passes and are not failures, and folding them either way would make this
    # number a claim nobody can act on.
    counts["pass_rate"] = (counts[OK] / counts["scored"]
                           if counts["scored"] else None)
    # Distinct FIELDS, not check rows. A field with two rules against it
    # contributes two absent rows -- `inv_rtv_cnr_number` has a pattern rule and
    # a matching rule -- and counting those as two missing Mandatory fields
    # overstates by exactly as much as the rule table happens to say.
    counts["mandatory_absent"] = len({r["field"] for r in checks
                                      if r["state"] == ABSENT
                                      and r.get("mandatory")})

    return {
        "doc_types": doc_types,
        # Which type-specific rules ran, so a reader can tell a rule that passed
        # from one this document was never held to.
        "rules": sorted(rules),
        "checks": checks,
        "counts": counts,
        "failed": [r for r in checks if r["state"] == FAILED],
        "warnings": [r for r in checks if r["state"] == WARNING],
    }
