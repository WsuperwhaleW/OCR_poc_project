"""Score extracted fields against hand-written field ground truth.

Pass 1 has had a score since the beginning -- `scoring.py` compares a transcript
with `solution/<id>.md` and reports character and word accuracy. Pass 2 had
nothing of the kind. `grounding.py` answers a different and weaker question: it
says whether a value is *on the page*, not whether it is the *right* value for
the key it landed in. sol005's `buyer_name` came back as the page's Location
Code and grounded happily, because it is printed there. This module is what
catches that: a value is scored against the value a human says belongs in that
key.

The ground truth lives beside the transcript truth, one file per case:

    solution/<id>.fields.json

and is hand-written, exactly like `solution/<id>.md`. `skeleton()` writes an
empty one to fill in.

Three states per key, and the difference between the last two is the whole point
of the format:

    null   not checked -- this key has not been filled in yet, so it is left out
           of every count. A half-filled truth file scores its filled half and
           says how much of the file that was.
    ""     the document does not state this. The extractor is expected to leave
           it empty; a value here is scored as spurious.
    "..."  the document states this. Copied as printed, the same rule the
           extraction prompt gives the model.
    [...]  the document states this in more than one way -- typically a name
           printed in Thai and again in English -- and a reply matching any of
           them is correct. One or two readings is the intended size; a list
           long enough to catch anything is a key that is no longer scored.

Comparison is loose about presentation and strict about content, deliberately the
same looseness `grounding.py` applies (`grounding.squash`, `verify.parse_amount`),
so the audit and the score cannot disagree about whether two spellings are the
same value. Presentation is not what pass 2 is being scored on: a truth file
written by hand cannot be expected to guess whether the model will write 1,200 or
1,200.00.

Two numbers are reported and the gap between them is informative:

    accuracy         exact matches only
    accuracy_loose   counting partial matches -- one value contains the other, so
                     the model found the right thing and took too much or too
                     little of it. A wide gap means truncation and over-capture,
                     which is a prompt problem, rather than misreading, which is
                     a model-choice one.

Line-item rows are matched to truth rows by cell similarity rather than by
position, because a dropped row would otherwise mis-score every row after it.
`other_fields` is scored separately and never folded into the headline: its
labels are the model's own wording, and punishing a model for calling a thing
something else is not what this measures.
"""

import json
import re
from pathlib import Path

import config
import grounding
import prompts
import tables
import verify

SOLUTION = config.SOLUTION_DIR

# One file per case, beside the transcript ground truth it belongs to.
TRUTH_SUFFIX = ".fields.json"

# The schema, from the two modules that already own it. Not re-listed here: a
# third copy of the key names would be a third thing to keep in step.
SCALARS = list(grounding.SCALAR_FIELDS)
ITEM_KEYS = list(prompts.ITEM_KEYS)

# A row counts as the same row as a truth row when this share of the truth row's
# filled cells match. Deliberately low: a row the model read badly is still that
# row, and scoring its cells as wrong is more useful than reporting the row as
# missing and a spurious one beside it.
ROW_MATCH_MIN = 0.3

# Text that is only a figure -- with separators, a currency mark, a percent sign
# or a trailing currency word. Only these are compared by value; comparing
# "Room 2" with "Room 3" by value would call them equal, because both reduce to 2.
_NUMERIC_TEXT = re.compile(
    r"^[\s฿$€£]*\(?\s*-?[\d๐-๙][\d๐-๙,.\s]*\)?"
    r"[\s%฿$€£]*(?:บาท|สตางค์|thb|baht)?\s*$",
    re.I)

# A containment match on one or two characters is noise -- every page contains
# "7" somewhere inside some longer value. Same reasoning as grounding's note
# about short values, applied to the comparison rather than to the search.
PARTIAL_MIN_CHARS = 4

# Outcomes best first. A key whose truth is a LIST is judged against every
# reading in it and keeps the best one -- see `accepted`.
_STATUS_ORDER = ("correct", "partial", "wrong", "missed", "spurious", "absent")

# What each of those six words claims, in one line each. Printed under the report
# table by `format_report`, because the column itself is six bare words and the
# two that matter most are the two least obvious: `missed` and `spurious` are
# statements about opposite mistakes -- one is the page saying something the
# extractor did not, the other the extractor saying something the page did not --
# and neither is a synonym for "empty". `absent` is agreement rather than a
# score, which is why the totals do not count it.
STATUS_MEANING = {
    "correct": "matches the value the truth file gives for that key",
    "partial": "right thing found, too much or too little of it taken"
               " -- one value contains the other",
    "wrong": "both sides filled, and they are different values",
    "missed": "the page states it, the extractor returned nothing",
    "spurious": "the page does not state it, the extractor filled it in anyway",
    "absent": "both empty -- agreed, and not counted either way",
}


# Keys of the truth file that are configuration rather than a value to score.
_CONFIG_KEYS = ("other_fields", "table_columns", "score_table", "line_items",
                "income_items", "is_master_table")

# Column heading -> schema key, tried in this order and each key taken by the
# LEFTMOST column that claims it. The order is the whole of the mapping's
# correctness: every needle here is a substring test against a squashed heading,
# so the specific readings have to be asked for before the general ones that
# contain them. "จำนวนเงินสุทธิ Net Amount" holds both `netamount` and
# `จำนวนเงิน`; "ภาษีหัก ณ ที่จ่าย W/T" and "ภาษีมูลค่าเพิ่ม VAT" both open with
# ภาษี; and `จำนวน` alone is a quantity while `จำนวนเงิน` is money, which is why
# quantity is asked last rather than first.
#
# Deliberately not a catalogue of every heading these five documents print --
# the same reasoning that keeps specific Thai labels out of the prompts. A
# heading this misses is reported as unmapped and its column is simply not
# scored, and `table_columns` in the truth file names it in one line.
# Headings a key must NOT contain, whatever its needles say. Only quantity needs
# one, and it needs it badly: จำนวน means "number of" and จำนวนเงิน means "amount
# of money", so every money heading in these documents contains the quantity
# needle. sol004 rules two money columns, and without this the second one
# (จำนวนเงินรับ, amount received) lands in `quantity` -- money in the count key,
# and the model marked wrong for leaving the count empty.
HEADER_BLOCKERS = {"quantity": ("เงิน", "amount", "price", "ราคา")}

HEADER_MAP = [
    ("withholding_tax", ("wt", "withholding", "หักณที่จ่าย", "ภาษีหัก")),
    ("vat", ("vat", "ภาษีมูลค่าเพิ่ม")),
    ("net_amount", ("netamount", "จำนวนเงินสุทธิ", "จำนวนสุทธิ", "ยอดสุทธิ")),
    ("unit_price", ("unitprice", "priceperunit", "ราคาต่อหน่วย", "ราคาหน่วย")),
    ("amount", ("grossamount", "amount", "จำนวนเงิน", "ยอดเงิน", "ราคารวม")),
    ("period", ("period", "ประจำงวด", "งวด")),
    ("description", ("description", "particular", "รายการ", "รายละเอียด", "สินค้า")),
    ("quantity", ("quantity", "qty", "จำนวน")),
]

# The money columns. A derived row with nothing in any of them and no quantity is
# a note printed inside the table rather than a charge -- sol003 rules three of
# them ("186902 มีใบกำกับภาษี 2 ใบ", "1.ใบนี้", "2.215/10736"), and an extractor
# that leaves them out is right to.
_MONEY_CELLS = ("amount", "unit_price", "vat", "withholding_tax", "net_amount")

# A row that totals the rows above it. The extraction prompt says in as many
# words that these are not line items and that their figures belong in the totals
# keys, so counting them as expected rows would score the documented behaviour as
# a miss. Anchored at the start of the cell: a description that merely contains
# รวม somewhere is a charge.
_TOTAL_ROW = re.compile(r"^\s*(?:total|sub\s*-?\s*total|grand\s*total|less|"
                        r"รวม|ยอดรวม|ยอดสุทธิ|จำนวนเงินรวม|บวก|หัก)", re.I)

# One reader of a Markdown pipe table, shared with `tables.py`, which takes the
# same syntax out of a transcript. They lived here until 2026-09-30; a second
# copy would eventually disagree with the first about what a cell is, and the
# two would then be scoring a table against a different reading of itself.
_cells = tables.pipe_cells
_tables = tables.pipe_tables


def _map_headers(headers, overrides=None) -> dict:
    """Column index -> schema key, for the headings this can name."""
    overrides = {grounding.squash(k): v for k, v in (overrides or {}).items()}
    mapped, taken = {}, set()
    for index, heading in enumerate(headers):
        squashed = grounding.squash(heading)
        if not squashed:
            continue
        if squashed in overrides and overrides[squashed] not in taken:
            mapped[index] = overrides[squashed]
            taken.add(overrides[squashed])
            continue
        for key, needles in HEADER_MAP:
            if key in taken:
                continue
            if any(grounding.squash(n) in squashed
                   for n in HEADER_BLOCKERS.get(key, ())):
                continue
            if any(grounding.squash(n) in squashed for n in needles):
                mapped[index] = key
                taken.add(key)
                break
    return mapped


def table_rows(case_id: str, overrides=None) -> dict:
    """The line-item ground truth, read out of `solution/<id>.md`.

    That file already holds a hand-checked transcription of the charge table, so
    the rows are taken from it rather than typed a second time into the field
    truth. What the field truth adds is the one thing the .md cannot carry: which
    printed column is which schema key -- and even that is usually derived, from
    HEADER_MAP.

    Returns {"rows": [...], "columns": {...}, "unmapped": [...], "dropped": [...],
             "source": name}, or {"error": ...} when there is no table to read.

    Two shapes are handled here because both are in the fixtures and both would
    otherwise be scored wrongly:

    * a **totals row inside the table** (sol001, sol005) is not a line item, and
    * a **satang column** (sol003) is a blank heading beside a money column whose
      cell holds the two digits after the point. Merged back into the money cell
      as "9,741 60", which `verify.parse_amount` already reads as 9741.60 -- and
      only when it really is two digits, so the "-" meaning no satang is dropped
      rather than glued on to make an unparsable "535 -".

    Keys no column maps to are `""` on every row, not absent: the prompt tells the
    model that a key the table rules no column for is empty on every row, and this
    is what measures whether it obeyed.
    """
    path = ground_truth_path(case_id)
    if path is None:
        return {"error": f"no transcript ground truth for {case_id}"}
    tables = _tables(path.read_text("utf-8"))
    if not tables:
        return {"error": f"no Markdown table in {path.name}"}

    # The charge table is the one whose headings name the most schema keys, ties
    # to the longer table. A totals block ("Vatable Amount | 11,638.64", sol002)
    # names one key at most and loses on both counts.
    best, best_map = None, {}
    for headers, rows in tables:
        mapped = _map_headers(headers, overrides)
        if (len(mapped), len(rows)) > (len(best_map), len(best[1]) if best else 0):
            best, best_map = (headers, rows), mapped
    if len(best_map) < 2:
        return {"error": f"no table in {path.name} whose headings name two or more "
                         f"line-item keys -- name them in table_columns"}

    headers, raw_rows = best
    # A blank heading directly after a money column is that column's satang half.
    satang = {i for i in range(1, len(headers))
              if not grounding.squash(headers[i])
              and best_map.get(i - 1) in _MONEY_CELLS}

    out, dropped = [], []
    for cells in raw_rows:
        row = {key: "" for key in ITEM_KEYS}
        for index, key in best_map.items():
            value = cells[index] if index < len(cells) else ""
            if index + 1 in satang:
                tail = cells[index + 1] if index + 1 < len(cells) else ""
                if re.fullmatch(r"\d{2}", tail.strip()):
                    value = f"{value} {tail.strip()}"
            row[key] = value
        label = row.get("description") or (cells[0] if cells else "")
        if _TOTAL_ROW.match(label or ""):
            dropped.append({"why": "totals row", "text": label})
            continue
        # Figures under no description at all, in a table that rules a
        # description column: the unlabelled total sol004 prints under its last
        # charge. A charge without a name is not something these documents print.
        if "description" in best_map.values()                 and grounding.is_blank(row.get("description")):
            dropped.append({"why": "figures with no description",
                            "text": " ".join(c for c in cells if c)[:60]})
            continue
        if all(grounding.is_blank(row[k]) for k in _MONEY_CELLS) \
                and grounding.is_blank(row.get("quantity")):
            dropped.append({"why": "no figure in any money column", "text": label})
            continue
        out.append(row)

    return {
        "rows": out,
        "columns": {headers[i]: key for i, key in sorted(best_map.items())},
        "unmapped": [h for i, h in enumerate(headers)
                     if i not in best_map and i not in satang
                     and grounding.squash(h)],
        "dropped": dropped,
        "source": path.name,
    }


def ground_truth_path(case_id: str):
    """`solution/<id>.md`, via scoring so the two agree on where truth lives.

    Imported here rather than at the top of the module: `scoring` is the heavier
    of the two and nothing else in this file needs it, and keeping the import
    graph one-directional is what lets `app.py` import both in either order.
    """
    import scoring

    return scoring.ground_truth_path(case_id)


# --------------------------------------------------------------------------
# the ground-truth file
# --------------------------------------------------------------------------

def truth_path(case_id: str) -> Path:
    return SOLUTION / f"{case_id}{TRUTH_SUFFIX}"


def has_truth(case_id: str) -> bool:
    return truth_path(case_id).exists()


def _readings(where: str, value, warnings) -> list:
    """One truth entry as the list of readings that count as correct, or None.

    A plain string is a list of one, so nothing downstream has to know which of
    the two shapes the file used. A list is for a value the page prints in more
    than one way -- the seller's name in Thai on the letterhead and again in
    English below it -- where a reply in either is right and there is no single
    string that could say so: `Jo-Jo TRAT CO., Ltd.` and `Jo-Jo TRAT COMPANY
    LIMITED` are not substrings of one another, so the partial match cannot
    reach across them.

    Two shapes are refused rather than repaired, because both would quietly
    widen the key instead of describing it:

    * an empty list -- `null` is not-checked and `""` is not-printed, and a list
      of nothing says neither;
    * `""` inside a list -- that would claim the page both prints this and does
      not, which reads to the scorer as licence for any answer at all.

    Two readings that squash to the same content are one reading; the duplicate
    is dropped so the file cannot claim to accept more than it does.
    """
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if not isinstance(value, list):
        warnings.append(f"{where}: expected a string or a list of strings, got "
                        f"{type(value).__name__} -- ignored")
        return None
    out = []
    for item in value:
        if not isinstance(item, (str, int, float)):
            warnings.append(f"{where}: a list of accepted readings holds strings, "
                            f"got {type(item).__name__} -- key ignored")
            return None
        text = str(item)
        if grounding.is_blank(text):
            warnings.append(f"{where}: an empty string inside the list of accepted "
                            f'readings -- write "" on its own for a value the '
                            f"document does not print. Key ignored")
            return None
        if any(grounding.squash(text) == grounding.squash(seen) for seen in out):
            continue                  # two spellings of one reading; count it once
        out.append(text)
    if not out:
        warnings.append(f"{where}: empty list -- null is not checked and \"\" is "
                        f"not printed, and this says neither. Key ignored")
        return None
    return out


def _mandatory_drift(case_id: str, note, codes=None) -> list:
    """Warn where a truth file's `_mandatory` note disagrees with the requirement.

    The note is a convenience for whoever is filling the file in by hand -- which
    of its keys have to be answered and which are found-only -- and it is a COPY
    of `prompts.MANDATORY_FIELDS`, which is the one statement of the fact. A copy
    that has drifted is worse than no copy at all, because it is read as the rule;
    so it is checked rather than trusted, and never used to score anything.

    Silent when there is no note, when the case is not in the manifest, or when
    the note is not the shape this writes: it is an annotation, and a malformed
    one must not stop a file being scored.
    """
    if not isinstance(note, dict):
        return []
    # **The types of the DOCUMENT this note is about**, where the caller knows
    # them: a pack's blocks are of different types, and checking every one of
    # them against the case's own -- which are document 1's -- reported a
    # correct note as drift on every other document in the file.
    if codes is None:
        import scoring                        # local: nothing else here needs it
        case = scoring.cases_index().get(case_id)
        if not case:
            return []
        codes = list(case.get("doc_types") or [])
    codes = list(codes)
    want = list(prompts.mandatory_for_types(codes))
    got = note.get("required")
    if not isinstance(got, list) or sorted(got) == sorted(want):
        return []
    return [f"_mandatory in the truth file lists {sorted(got)}, but the "
            f"requirement demands {sorted(want)} of {' + '.join(codes) or 'this'}"
            " -- the note is out of date, prompts.MANDATORY_FIELDS is the rule"]


# A truth file for a file holding SEVERAL documents lists them here, each with
# its own pages, its own types and its own values. A file holding one document
# has no such key and is read exactly as it always was -- which is the thirteen
# ordinary fixtures, untouched.
_DOCUMENTS_KEY = "documents"


def truth_documents(case_id: str) -> list:
    """[{pages, doc_types}] for a case whose file holds several documents, else [].

    Read without scoring anything, so a caller can ask what a truth file covers
    before it has an extraction to score -- which is what the page's pickers and
    `compare.py`'s header need.
    """
    path = truth_path(case_id)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, dict):
        return []
    return [{"pages": list(d.get("pages") or []),
             "doc_types": list(d.get("doc_types") or [])}
            for d in (raw.get(_DOCUMENTS_KEY) or []) if isinstance(d, dict)]


def _pick_document(raw: dict, pages, warnings) -> dict:
    """The entry of a pack's truth file that describes the given pages.

    **Matched on the FIRST page**, not on the whole list: the truth file states
    where a document begins, and a read that split it one page longer or shorter
    still means that document. Falling back to the first entry is deliberate --
    a caller with no pages in hand (the CLI, a re-score) means "the document this
    case is about", which is document 1 and is what `_document` says it is.
    """
    documents = raw.get(_DOCUMENTS_KEY) or []
    if not documents:
        return raw
    # `pages or ()` because None is a documented input, not a mistake -- see the
    # docstring: a caller with no pages in hand means "the document this case is
    # about". `list(None)` raised a TypeError, which `evaluate` does not catch
    # (it catches ValueError), so `load_truth('sol015')` crashed rather than
    # falling back to document 1 the way this says it does.
    first = (list(pages or ()) or [None])[0]
    if first is not None:
        for entry in documents:
            if first in (entry.get("pages") or []):
                return entry
        # A page no document of the truth file covers. Not an error and not
        # document 1's truth: scoring one document's values against another's
        # would mark a correct extraction wrong, which is worse than not scoring.
        warnings.append("no truth for page %s of this file" % first)
        return {}
    return documents[0]


def load_truth(case_id: str, pages=None) -> dict:
    """Read one case's field ground truth. Raises ValueError if unusable.

    `pages` selects which document of a PACK -- a file holding several documents
    -- is wanted. A truth file for one document ignores it, so every existing
    caller and all thirteen single-document fixtures behave exactly as before.

    Returns {"scalars": {...}, "other_fields": [...] or None, "table_columns": {},
             "score_table": bool, "warnings": [...]}.

    **The line-item table is not in this file.** It is derived from the Markdown
    table in `solution/<id>.md`, which is already a transcription of the same rows
    -- see `table_rows`. Transcribing them a second time here would be the same
    work done twice, and two hand-written copies of one table disagree eventually.

    Keys beginning with an underscore are notes to the person filling the file in
    and are ignored here -- JSON has no comments, and a format nobody can annotate
    is a format that gets filled in wrongly.
    """
    path = truth_path(case_id)
    if not path.exists():
        raise ValueError(f"no field ground truth: {path}")
    try:
        raw = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path.name} is not valid JSON: {err}") from err
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must hold a JSON object")

    warnings = []
    # For a pack, everything below reads the chosen document's own block --
    # its values, its `_mandatory` note, its `other_fields` and table settings.
    # A pack's top level holds only the list and the file-wide notes.
    whole, raw = raw, _pick_document(raw, pages, warnings)
    warnings.extend(_mandatory_drift(
        case_id, raw.get("_mandatory"),
        # A block of a pack carries its own types; a single-document file has
        # none of its own and falls back to the manifest's.
        list(raw.get("doc_types") or []) if whole is not raw else None))
    scalars = {}
    for key, value in raw.items():
        if key in ("pages", "doc_types") and whole is not raw:
            continue                      # the document's own address, not a value
        if key.startswith("_") or key in _CONFIG_KEYS:
            continue
        if key in prompts.RETIRED_KEYS:
            # A value for a key the schema used to ask for. Not a mistake and
            # not a warning: these files are the human's record of the page, and
            # a key leaving the schema does not make the value on the page wrong.
            # It is simply not scored while nothing asks for it.
            continue
        if key not in SCALARS:
            warnings.append(f"unknown key {key!r} -- ignored")
            continue
        if value is None:
            continue                      # not checked
        readings = _readings(key, value, warnings)
        if readings is None:
            continue
        # One reading stays a plain string, so a file that uses none of this
        # produces exactly the truth dict it produced before.
        scalars[key] = readings[0] if len(readings) == 1 else readings

    if "line_items" in raw:
        warnings.append("line_items: the table is read from the .md now and this "
                        "key is ignored -- delete it")

    # The income table of a WHT certificate, and the one table this module reads
    # from the truth file rather than deriving from the .md.
    #
    # **That is not an exception to the derive-it rule, it is that rule's own
    # reasoning.** The charges table is derived because "what the derivation
    # adds is the one thing the .md cannot carry: which printed column is which
    # schema key" -- a per-document question `HEADER_MAP` exists to answer. A
    # 50 tavi is a government form: its four columns are fixed by the form, in
    # a fixed order, so there is no mapping to derive. What DOES vary across the
    # fixtures is the heading wording (ประเภทเงินได้พึงประเมินที่จ่าย against
    # ประเภทเงินได้ที่จ่ายและเงินคง against ประเภทเงินได้ที่จ่าย), so a heading map
    # here would need a needle per fixture to answer a question the form has
    # already answered -- all cost, no information.
    income = raw.get("income_items")
    if income is not None:
        if not isinstance(income, list):
            warnings.append("income_items: expected a list -- ignored")
            income = None
        else:
            rows = []
            for index, entry in enumerate(income):
                if not isinstance(entry, dict):
                    warnings.append(f"income_items[{index}]: expected an object")
                    continue
                row = {}
                for key, value in entry.items():
                    if key.startswith("_"):
                        continue
                    if key not in prompts.INCOME_ITEM_KEYS:
                        warnings.append(f"income_items[{index}]: unknown cell "
                                        f"{key!r} -- ignored")
                        continue
                    if value is None:
                        continue          # not checked, as a scalar's null is
                    readings = _readings(f"income_items[{index}].{key}", value,
                                         warnings)
                    if readings is None:
                        continue
                    row[key] = readings[0] if len(readings) == 1 else readings
                if row:
                    rows.append(row)
            income = rows

    # Only needed where a column heading beats the header map below. Written as
    # the heading exactly as the .md prints it -> the schema key it fills.
    columns = raw.get("table_columns") or {}
    if not isinstance(columns, dict):
        warnings.append("table_columns: expected an object -- ignored")
        columns = {}
    else:
        bad = [k for k, v in columns.items() if v not in ITEM_KEYS]
        for key in bad:
            warnings.append(f"table_columns[{key!r}]: {columns[key]!r} is not a "
                            f"line-item key -- ignored")
            columns.pop(key)

    others = raw.get("other_fields")
    if others is not None:
        if not isinstance(others, list):
            warnings.append("other_fields: expected a list -- ignored")
            others = None
        else:
            entries = []
            for index, entry in enumerate(others):
                if not isinstance(entry, dict):
                    warnings.append(f"other_fields[{index}]: expected an object")
                    continue
                label = str(entry.get("label") or "")
                value = entry.get("value")
                if grounding.is_blank(label):
                    continue
                if value is None:
                    stated = ""
                else:
                    readings = _readings(f"other_fields[{index}].value", value,
                                         warnings)
                    if readings is None:
                        continue
                    stated = readings[0] if len(readings) == 1 else readings
                entries.append({"label": label, "value": stated})
            others = entries

    # Whether this document's item table is a list of OTHER DOCUMENTS rather than
    # of items -- a person's reading, and the only thing about the item table
    # this file holds. The rows themselves are in the .md and are read from it
    # (`table_truth`), for the reason the charges table always was.
    #
    # Three states like every other key here: true, false, and null or absent
    # for "not checked" -- which is also the honest answer for a document that
    # rules no item table at all.
    master = raw.get("is_master_table")
    if master is not None and not isinstance(master, bool):
        warnings.append("is_master_table: expected true, false or null -- ignored")
        master = None

    return {"scalars": scalars, "other_fields": others,
            "is_master_table": master,
            # None where the file states none, so "this document rules no income
            # table" and "nobody has transcribed it yet" stay distinguishable --
            # the same reason `other_fields` is None rather than [].
            "income_items": income,
            "table_columns": columns,
            # The truth file may switch the table off, and so may the schema:
            # pass 2 does not ask for line items at all while
            # `prompts.EXTRACT_LINE_ITEMS` is false, and scoring a table the
            # extractor was never asked for would mark every row of it missed and
            # report the resulting collapse as an accuracy figure. Either veto is
            # enough, so the schema's is applied here rather than being written
            # into five truth files by hand.
            "score_table": (prompts.EXTRACT_LINE_ITEMS
                            and raw.get("score_table", True) is not False),
            "warnings": warnings}


# The legend that ships INSIDE every truth file, as `_readme`.
#
# A truth file is read and corrected by a person, and it states a lot of its
# meaning in one or two characters: "" is not the same claim as null, and a
# lone "-" is a third thing again. JSON has no comments, so the explanation has
# to be a key -- and the loader ignores underscored keys precisely so that one
# can sit here without being scored.
#
# A module constant rather than a literal inside `skeleton()`, because the five
# shipped files carry it too. Two copies of a legend drift, and a legend that
# disagrees with the format it describes is worse than none.
README_LINES = [
        "Hand-written ground truth for pass 2 (field extraction), scored by",
        "fieldscore.py. The transcript ground truth for this case is the .md",
        "file beside this one; this file is about which VALUE belongs in which",
        "KEY, which the .md cannot say and grounding.py cannot check.",
        "",
        "Three states per key:",
        "  null   not checked yet -- left out of every count. This is the",
        "         default, so fill in what you are sure of and leave the rest.",
        "  \"\"     the document does not print this. A value here is scored as",
        "         spurious -- the extractor invented it, or took it from",
        "         another key's label.",
        "  \"text\" the document prints this. Copy it EXACTLY as printed: same",
        "         digits, same separators, same language, no tidying. The model",
        "         is told to copy verbatim, so the truth has to be verbatim too.",
        "  \"-\"    the document prints a dash where a figure would go. A dash",
        "         and an empty answer both count as correct, which is what the",
        "         extraction prompt asks the model for.",
        "  [ .. ]  the document prints this value in more than one way and a",
        "         reply matching any of them is correct -- a name printed in",
        "         Thai on the letterhead and again in English below it is the",
        "         case it is for. One or two readings; a list long enough to",
        "         catch anything is a key that is no longer being scored, and",
        "         every reading in it still has to be printed on the page.",
        "",
        "Scoring is loose about presentation and strict about content:",
        "punctuation, spacing and Thai digits are normalised away, and a figure",
        "is compared by value, so 1,200 and 1,200.00 score equal. You do not",
        "have to guess which way the model will write a number.",
        "",
        "The line-item table is NOT in this file. It is read out of the",
        "Markdown table in the .md beside it, which already transcribes those",
        "rows -- typing them again here would be the same work twice, and two",
        "hand-written copies of one table disagree eventually. Each column is",
        "matched to one of these eight cells by its heading:",
        "  " + ", ".join(ITEM_KEYS),
        "Totals rows printed inside the table are dropped, and a cell no column",
        "maps to is expected empty on every row. Rows are matched to the",
        "extractor's rows by content, not by position, so one row it drops does",
        "not mis-score every row after it.",
        "",
        "table_columns: only needed when a heading is not recognised -- the",
        "run prints which columns it mapped and which it did not. Write the",
        "heading exactly as the .md prints it against the cell it fills, e.g.",
        "  \"table_columns\": { \"จำนวนเงินรับ\": \"amount\" }",
        "score_table: set false to leave the table out of the score entirely.",
        "",
        "is_master_table: what KIND of table the document's item table is.",
        "  true   its rows are OTHER DOCUMENTS -- each identified by an invoice",
        "         number or another reference, with that document's figures: a",
        "         receipt settling several invoices, a billing note, a payment",
        "         schedule.",
        "  false  its rows are items -- goods or services. A list of goods that",
        "         cites a document on each row is still a list of goods.",
        "  null   not checked, or the document rules no item table at all.",
        "The rows themselves are read from the .md, like the line items above;",
        "this one flag is the only thing about the item table written here,",
        "because it is a person's reading of the table and cannot be derived",
        "without marking the rule against itself.",
        "",
        "other_fields: null means not scored. A list means these labels and",
        "values are what the page prints outside the schema. Scored and",
        "reported separately -- the labels are the model's own wording, so they",
        "never move the headline number.",
        "",
        "_mandatory is a NOTE, not values. It records which keys the field",
        "requirement marks Mandatory for this document's types, and which it",
        "asks for without demanding. A required key decides the headline score;",
        "a found-only key is still extracted, still judged and still shown, and",
        "simply does not move the rate -- nobody is held to it. The list is",
        "derived from prompts.MANDATORY_FIELDS, which is the single statement of",
        "the fact; a copy here that disagrees with it is reported as a warning,",
        "so fix it there rather than here.",
]


def skeleton(case_id: str, pdf: str = "", kind: str = "") -> str:
    """An empty ground-truth file for one case, as text ready to write.

    Every key starts null -- not checked -- so a file that has been half filled in
    scores its filled half honestly instead of reporting the rest of the schema as
    fields the document does not state.
    """
    body = {
        "_case": case_id,
        "_source": pdf,
        "_kind": kind,
        "_readme": README_LINES,
        **{key: None for key in SCALARS},
        # The shape of one entry, kept where it will be needed rather than only
        # described in the notes above. Underscored, so the loader ignores it
        # however it is edited -- copy it down into the list below and fill it in.
        "_other_fields_template": [{"label": "", "value": ""}],
        "other_fields": None,
        "table_columns": None,
        "score_table": True,
        # Not checked until a person says which it is.
        "is_master_table": None,
    }
    return json.dumps(body, ensure_ascii=False, indent=2) + "\n"


# --------------------------------------------------------------------------
# comparing one value
# --------------------------------------------------------------------------

def _numeric(value) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(_NUMERIC_TEXT.match(text))


def compare_value(expected, actual) -> str:
    """correct | partial | wrong, for two values both known to be filled."""
    exp, act = grounding.squash(expected), grounding.squash(actual)
    if exp and exp == act:
        return "correct"
    if _numeric(expected) and _numeric(actual):
        a = verify.parse_amount(expected)
        b = verify.parse_amount(actual)
        if a is not None and b is not None and abs(a - b) <= verify.TOLERANCE:
            return "correct"
    if min(len(exp), len(act)) >= PARTIAL_MIN_CHARS and (exp in act or act in exp):
        return "partial"
    return "wrong"


def accepted(expected) -> list:
    """One key's truth as the list of readings that count as correct.

    A plain value is a list of one, so a caller never has to ask which shape the
    truth file used.
    """
    if isinstance(expected, (list, tuple)):
        return [str(v) for v in expected]
    return [expected]


def judge_best(expected, actual):
    """(status, the accepted reading that produced it).

    The reading is what a caller should show as `expected`: with two readings on
    a key, printing an arbitrary one beside the answer would report the model as
    wrong against a value it was never closest to. Ties go to the first reading
    listed, which by the truth files' convention is the page's own first
    printing of the value.
    """
    best = None
    for option in accepted(expected):
        status = judge(option, actual)
        rank = _STATUS_ORDER.index(status)
        if best is None or rank < best[0]:
            best = (rank, status, option)
        if rank == 0:
            break
    return (best[1], best[2]) if best else ("absent", "")


def judge(expected, actual) -> str:
    """One key's outcome.

    correct/partial/wrong  both sides filled
    missed                 the document states it, the extractor left it empty
    spurious               the document does not state it, the extractor filled it
    absent                 both empty -- agreed, and not counted as an achievement

    `expected` may be a list of accepted readings, in which case the best of them
    is the key's outcome. Use `judge_best` where the reading itself is wanted too.
    """
    if isinstance(expected, (list, tuple)):
        return judge_best(expected, actual)[0]
    exp_blank = grounding.is_blank(expected)
    act_blank = grounding.is_blank(actual)
    if exp_blank and act_blank:
        return "absent"
    # A cell the document prints as a dash. The extraction prompt allows either
    # answer for it -- "write it as a dash or as ''" -- so both are correct here,
    # and only a figure invented in its place is not. Without this, sol005's
    # dashed VAT cells score as missed against an extractor doing as it was told.
    if grounding.is_nil(expected):
        return "correct" if (act_blank or grounding.is_nil(actual)) else "wrong"
    if exp_blank:
        return "spurious"
    if act_blank:
        return "missed"
    return compare_value(expected, actual)


def _tally(rows) -> dict:
    """Counts and the three rates, over a list of judged rows."""
    counts = {k: 0 for k in
              ("correct", "partial", "wrong", "missed", "spurious", "absent")}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    expected = counts["correct"] + counts["partial"] + counts["wrong"] + counts["missed"]
    returned = counts["correct"] + counts["partial"] + counts["wrong"] + counts["spurious"]
    accuracy = counts["correct"] / expected if expected else None
    loose = (counts["correct"] + counts["partial"]) / expected if expected else None
    # Half credit for a partial: the model found the right thing and took too
    # much or too little of it, which is neither a hit nor a miss. Added
    # 2026-08-20 at the user's request as the figure the setting comparison is
    # ranked on. It sits BESIDE the strict and loose rates rather than replacing
    # either -- those two are the documented pair whose *gap* is the diagnosis
    # (over-capture against misreading), and collapsing them into one number
    # would throw that away.
    half = ((counts["correct"] + 0.5 * counts["partial"]) / expected
            if expected else None)
    precision = counts["correct"] / returned if returned else None
    return {
        "counts": counts,
        # How many values the truth file says the document states -- the
        # denominator of `accuracy`, and not the size of the schema.
        "expected": expected,
        "returned": returned,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "accuracy_loose": round(loose, 4) if loose is not None else None,
        "accuracy_half": round(half, 4) if half is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
    }


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

def _row_similarity(truth_row: dict, actual_row: dict) -> float:
    """Share of a truth row's filled cells the actual row gets right."""
    filled = [(k, v) for k, v in truth_row.items() if not grounding.is_blank(v)]
    if not filled:
        return 0.0
    score = 0.0
    for key, value in filled:
        status = judge(value, (actual_row or {}).get(key))
        score += 1.0 if status == "correct" else 0.5 if status == "partial" else 0.0
    return score / len(filled)


def _pair_rows(truth_rows, actual_rows):
    """Match truth rows to extracted rows by content, best pair first.

    Positional matching is wrong here: one row the extractor drops, or one
    invented row it inserts, shifts every row after it and would score a correct
    table as entirely wrong. Ties are broken towards the diagonal, so two rows
    that genuinely look alike are matched in printed order.
    """
    pairs = []
    for i, truth_row in enumerate(truth_rows):
        for j, actual_row in enumerate(actual_rows):
            sim = _row_similarity(truth_row, actual_row)
            if sim >= ROW_MATCH_MIN:
                pairs.append((sim, -abs(i - j), i, j))
    pairs.sort(reverse=True)

    matched = {}
    taken = set()
    for _, _, i, j in pairs:
        if i in matched or j in taken:
            continue
        matched[i] = j
        taken.add(j)
    return matched


def _score_items(truth_rows, actual_rows, required_cells=None,
                 field="line_items", shape=None) -> dict:
    """One table's cells, scored row by row.

    `field` is the key the extraction returns this table under and the prefix of
    every path here, because there are two tables and they are not
    interchangeable: `line_items` is the charges table of a commercial document
    and `income_items` the income table of a WHT certificate. `shape` is the
    keys that table rules, used to count cells in rows the truth has no
    counterpart for. Both default to the charges table, so the caller that had
    no choice before this still gets exactly what it got.
    """
    shape = tuple(shape) if shape else ITEM_KEYS
    actual_rows = [r for r in (actual_rows or []) if isinstance(r, dict)]
    matched = _pair_rows(truth_rows, actual_rows)
    rows = []

    for i, truth_row in enumerate(truth_rows):
        j = matched.get(i)
        source = actual_rows[j] if j is not None else {}
        for key, value in truth_row.items():
            status, reading = judge_best(
                value, None if j is None else source.get(key))
            rows.append({
                "path": f"{field}[{i}].{key}",
                # The reading that produced the outcome, for the reason a
                # scalar's `expected` is: a cell may carry accepted readings
                # too, and printing an arbitrary one beside the answer reports
                # the model wrong against a value it was never judged against.
                "expected": reading,
                "actual": "" if j is None else str(source.get(key) or ""),
                # Which RETURNED row this truth row was paired with, or None if
                # none was. The path above carries the truth row's index, and
                # rows are matched by content, so without this a caller holding
                # the extractor's rows has no way to put a truth cell beside the
                # cell it judges -- which is what the Fields tab marks with.
                "row": j,
                # A row that was never matched is a row the extractor did not
                # return: every cell the document prints in it was missed, and
                # saying so cell by cell keeps one dropped row costing what it
                # actually cost rather than one point.
                "status": status,
                # Mandatory of every ROW, where the requirement says so. Same
                # flag as a scalar's so one rule can take the headline over both.
                "required": (True if required_cells is None
                             else key in required_cells),
            })

    spurious_cells = 0
    for j, actual_row in enumerate(actual_rows):
        if j in matched.values():
            continue
        spurious_cells += sum(1 for k in shape
                              if not grounding.is_blank(actual_row.get(k)))

    order = [matched[i] for i in sorted(matched)]
    result = _tally(rows)
    result.update(
        rows=rows,
        rows_expected=len(truth_rows),
        rows_returned=len(actual_rows),
        rows_matched=len(matched),
        rows_missed=len(truth_rows) - len(matched),
        rows_spurious=len(actual_rows) - len(matched),
        # Cells inside rows the truth has no counterpart for. Not folded into the
        # counts above: they belong to no key the truth file rules on, so calling
        # them wrong would score the extractor against a row nobody transcribed.
        spurious_cells=spurious_cells,
        in_order=order == sorted(order),
    )
    return result


def _score_others(truth_entries, actual_entries) -> dict:
    """`other_fields`, matched by label. Never part of the headline."""
    actual = [e for e in (actual_entries or []) if isinstance(e, dict)]
    used = set()
    rows = []
    for entry in truth_entries:
        want = grounding.squash(entry["label"])
        found = None
        for index, candidate in enumerate(actual):
            if index in used:
                continue
            got = grounding.squash(candidate.get("label"))
            if want and got and (want == got or want in got or got in want):
                found = index
                break
        if found is None:
            status, reading = judge_best(entry["value"], "")
            rows.append({"path": f"other_fields[{entry['label']}]",
                         "expected": reading, "actual": "",
                         "status": "missed", "label_found": False})
            continue
        used.add(found)
        status, reading = judge_best(entry["value"], actual[found].get("value"))
        rows.append({"path": f"other_fields[{entry['label']}]",
                     "expected": reading,
                     "actual": str(actual[found].get("value") or ""),
                     "status": status,
                     "label_found": True})
    result = _tally(rows)
    result.update(rows=rows,
                  labels_expected=len(truth_entries),
                  labels_returned=len(actual),
                  labels_matched=len(used))
    return result


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------

def score(truth: dict, fields, table: dict = None, keys=None,
          mandatory=None, items_mandatory=None, item_keys=None) -> dict:
    """Score one extraction against one loaded truth file.

    `fields` is the `fields` object of an extraction result -- what the model
    returned, before any of it is displayed. `table` is what `table_rows` read out
    of the .md, passed in rather than fetched so this stays a pure function of its
    arguments and can be tested without a solution directory.

    `keys` is the field set the extraction was ASKED for -- `prompts.fields_for_type`
    of the document's type. A truth value for a key outside it is not scored, and
    that is the whole of how a per-type schema reaches this module.

    **It filters what is scored; it never edits the truth file.** Those files are
    the human's -- a value a person has entered is not to be changed -- so a key
    the current type does not ask for stays on disk and simply falls out of the
    denominator, ready for the day the requirement asks for it again. Scoring it
    anyway would mark every invoice wrong for not returning a `currency` that
    nothing requested.

    `mandatory` is the requirement's Mandatory set for this document's types
    (`prompts.mandatory_for_types`), and **it is the headline's denominator**:
    a field the requirement marks Yes is scored, one it marks No is extracted,
    grounded, shown and reported under `optional` -- found, not marked. The two
    are kept apart rather than pooled because they are different claims: the
    first is whether this document can be filed at all, and mixing an Optional
    field nobody is held to into that rate makes a compliant document look
    incomplete. `optional` is reported for the same reason `other_fields` is --
    excluded from the headline, never thrown away.

    **`None` and `()` are different, and the difference is the usual one here.**
    `None` is *nobody said*, and scores everything asked, which is what this did
    before the requirement tables arrived and is still right for a caller that
    did not classify. `()` is *no requirement covers this type yet* -- a tax
    invoice on its own, or a page nothing could place -- and **it scores the
    base field set, marked as an unknown type** (2026-09-04, at the user's
    request: *on the unknown doc type just score with the base field we have*,
    since the ground truth exists and the requirement is coming later).

    That reverses what `()` did until then, which was to score NOTHING on the
    grounds that a document no table covers has no compliance rate. The rate is
    honest either way; what made scoring nothing the wrong answer is that the
    truth file states values, the extractor either got them or did not, and a
    headline of 0 of 0 threw a real measurement away and rendered as no score at
    all -- indistinguishable from an extraction that found nothing.

    **What it must NOT do is turn into a compliance claim**, so the reversal is
    confined to this module: `prompts.mandatory_for_types` still returns `()`,
    `validate` still demands nothing and the page still marks no key REQUIRED.
    `unknown_type` rides on the result and `scored_scope` says so, because a
    rate over the base fields and a rate over a requirement's Mandatory set are
    different claims and only the second says whether the document can be filed.
    """
    fields = fields if isinstance(fields, dict) else {}
    asked = list(keys) if keys is not None else list(SCALARS)
    required = None if mandatory is None else set(mandatory)
    # An empty-but-given Mandatory set is the third state: the caller classified
    # the document and no requirement covers what it found. Scored like `None`
    # -- every key asked for, which is the base field set -- and reported apart
    # from it, so nothing downstream reads the rate as a compliance figure.
    unknown_type = required is not None and not required
    required_cells = None if items_mandatory is None else set(items_mandatory)
    if unknown_type:
        required = None
        # The table's cells follow the scalars, or one form would put its
        # scalars in the headline and its cells outside it. No type does both
        # today -- the only one that rules a table has a requirement -- and the
        # day one does, a silent split between the two halves of one form is
        # exactly the kind of inconsistency that is found months later.
        required_cells = None
    scalar_rows = []
    for key, value in truth["scalars"].items():
        if key not in asked:
            continue
        status, reading = judge_best(value, fields.get(key))
        row = {
            "path": key,
            # The reading that produced the outcome, not the whole truth entry:
            # every caller of this prints `expected` beside `actual`, and a list
            # printed there says nothing about which of its readings was meant.
            "expected": reading,
            "actual": str(fields.get(key) or ""),
            "status": status,
            "tier": ("p1" if key in grounding.PRIORITY_1 else
                     "p2" if key in grounding.PRIORITY_2 else "p3"),
            # Whether the requirement demands this key of this document. On the
            # row rather than worked out again by each reader: the page marks it,
            # the CLI prints it and the headline is taken over it, and three
            # answers to one question drift.
            "required": True if required is None else key in required,
        }
        readings = accepted(value)
        # Present only where there is genuinely a choice, so a caller can say
        # "or" without having to compare the list with the value beside it.
        if len(readings) > 1:
            row["accepted"] = readings
        scalar_rows.append(row)

    scalars = _tally(scalar_rows)
    scalars["rows"] = scalar_rows
    scalars["checked"] = len(scalar_rows)
    scalars["unchecked"] = len(asked) - len(scalar_rows)
    for tier in ("p1", "p2"):
        scalars[tier] = _tally([r for r in scalar_rows if r["tier"] == tier])

    warnings = list(truth.get("warnings") or [])
    items = None
    # A form that rules a table of its OWN rules no charges table, so the
    # charges table's absence is not a finding about it -- and saying "table not
    # scored" beside a certificate whose income table WAS scored is worse than
    # saying nothing, because the reader has one table on screen and no way to
    # tell which one the sentence is about.
    charges_asked = not item_keys or tuple(item_keys) == ITEM_KEYS
    if not charges_asked:
        pass
    elif not truth.get("score_table", True):
        warnings.append(
            "charges table not scored: pass 2 does not extract line items"
            if not prompts.EXTRACT_LINE_ITEMS
            else "charges table not scored: score_table is false in the truth file")
    elif table and table.get("error"):
        warnings.append(f"table not scored: {table['error']}")
    elif table:
        items = _score_items(table["rows"], fields.get("line_items"),
                             required_cells)
        # Where these rows came from, carried with the score. A derived truth has
        # to say what it derived, or a wrong column mapping looks like a wrong
        # extraction -- and the mapping is the one part of this nobody wrote down
        # by hand.
        items["derived"] = {key: table.get(key) for key in
                            ("source", "columns", "unmapped", "dropped")}

    # The income table of a WHT certificate. Two conditions, and both are
    # deliberate: the FORM has to ask for this shape (`prompts.items_for_types`
    # of the document's types), and the truth file has to state the rows.
    #
    # **`score_table` does not gate it, and must not.** That flag is the charges
    # table's veto and is ANDed with `prompts.EXTRACT_LINE_ITEMS`, which is
    # false while pass 2 asks for no charges table -- so it reads false for
    # every case in the corpus. Gating this on it would mean the income table
    # could never be scored at all, which is the state this replaces.
    income = None
    if item_keys and truth.get("income_items"):
        income = _score_items(truth["income_items"],
                              fields.get("income_items"), required_cells,
                              field="income_items", shape=item_keys)

    others = (_score_others(truth["other_fields"], fields.get("other_fields"))
              if truth["other_fields"] is not None else None)

    # The headline is what the REQUIREMENT demands: the Mandatory scalars plus
    # the Mandatory cells of the table. `other_fields` is excluded on purpose --
    # see the module docstring -- and an Optional field is excluded for a
    # different reason with the same shape: nobody is held to it, so a document
    # that leaves it out is compliant and must not read as incomplete.
    # The requirement marks all four income cells Mandatory OF EVERY ROW, so
    # they belong in the headline exactly as a Mandatory scalar does -- which is
    # what `required_cells` already marked them.
    judged = (scalar_rows + (items["rows"] if items else [])
              + (income["rows"] if income else []))
    wanted = [r for r in judged if r["required"]]
    spare = [r for r in judged if not r["required"]]
    overall = _tally(wanted)
    overall["scored_paths"] = len(wanted)
    # The other half, reported and never in the headline. A reader comparing the
    # two rates learns something the headline cannot say: a model that is good at
    # the eleven that matter and poor at the rest is a different proposition from
    # one that is evenly mediocre.
    optional = _tally(spare)
    optional["scored_paths"] = len(spare)

    return {
        "overall": overall,
        "optional": optional,
        # Which of the three states the caller put this in. 100% of nothing and
        # 100% of eleven are the same number and not the same claim, so the scope
        # travels with the rate rather than being inferred from a count.
        "scored_scope": ("every key asked for -- no requirement covers this "
                         "document type yet" if unknown_type
                         else "everything asked" if required is None
                         else "the requirement's Mandatory fields"),
        # Whether that rate is over a requirement's Mandatory set or over the
        # base field set for a type no requirement covers. A boolean rather than
        # a string match on the scope: the page, the CLI and the round table all
        # have to word themselves differently for it, and three of them parsing
        # one sentence is three chances to disagree with it.
        "unknown_type": unknown_type,
        "scalars": scalars,
        "line_items": items,
        # Kept apart from `line_items` for the reason the two keys exist at all:
        # they are different tables with different cells, and one key holding
        # both would score a certificate's income rows against an invoice's
        # charge rows the day a document rules both.
        "income_items": income,
        "other_fields": others,
        "warnings": warnings,
        # What the truth file actually rules on. A score taken over 6 of 29 keys
        # is a different claim from one taken over all of them, and the number
        # alone cannot say which it is.
        "coverage": {
            "scalars_checked": len(scalar_rows),
            # The size of the form that RAN, not of the union of every form. A
            # rate over 11 of an invoice's keys and one over 11 of the whole
            # schema are the same number and not the same claim.
            "scalars_total": len(asked),
            # Truth values this document's type did not ask for. Not a fault --
            # the file is allowed to know more than the form does -- but a
            # reader comparing two documents needs to see the form changed.
            "scalars_not_asked": sum(1 for k in truth["scalars"]
                                     if k not in asked),
            # Of the form that ran, how many keys the requirement demands and
            # how many of those the truth file rules on. The headline's
            # denominator is the second, and the gap is truth nobody has written.
            "required_total": (len(asked) if required is None
                               else len([k for k in asked if k in required])),
            "required_checked": sum(1 for r in scalar_rows if r["required"]),
            "line_items_checked": items is not None,
            "income_items_checked": income is not None,
            "other_fields_checked": truth["other_fields"] is not None,
        },
    }


def evaluate(case_id: str, fields, keys=None, doc_types=(),
             mandatory=None, items_mandatory=None, pages=None) -> dict:
    """Load the truth for a case and score `fields` against it.

    Returns {"error": ...} rather than raising, so a caller on a request path can
    put it on a response without a try/except of its own.

    `keys` is the field set the extraction asked for; `doc_types` are the types
    it was built from, recorded on the result so a score can say which form it
    was taken over. Passing neither scores every key the truth file rules on,
    which is what this did before per-type field sets and is still right for a
    caller that did not classify.

    `mandatory` and `items_mandatory` are the requirement's Mandatory sets and
    decide what the headline is taken over -- see `score`.
    """
    try:
        truth = load_truth(case_id, pages)
    except ValueError as err:
        return {"error": str(err)}
    table = (table_rows(case_id, truth.get("table_columns"))
             if truth.get("score_table", True) else None)
    # Which table shape this document's form asks for, derived here from the
    # types the caller already passes rather than added to every call site --
    # `items_mandatory` is the same answer's other half and arrives that way.
    result = score(truth, fields, table, keys=keys, mandatory=mandatory,
                   items_mandatory=items_mandatory,
                   item_keys=prompts.items_for_types(doc_types))
    result["case"] = case_id
    result["doc_types"] = list(doc_types or [])
    # Which document of the file this score is of, where the file holds several.
    # On the score rather than left to the caller: a rate travels further than
    # the run that produced it, and "56% of 11" says nothing about which of
    # seven documents was being marked.
    if pages:
        result["pages"] = list(pages)
    return result


def _rates(counts: dict, expected: int, returned: int) -> dict:
    """The four rates a scored block carries, from its counts.

    One function so a pooled block and a scored one cannot compute them
    differently -- the pooled dict is read by everything that reads a score, and
    a rate that does not follow from the counts beside it is unrecoverable.
    """
    correct, partial = counts.get("correct", 0), counts.get("partial", 0)
    return {
        "accuracy": round(correct / expected, 4) if expected else None,
        "accuracy_loose": (round((correct + partial) / expected, 4)
                           if expected else None),
        "accuracy_half": (round((correct + 0.5 * partial) / expected, 4)
                          if expected else None),
        "precision": round(correct / returned, 4) if returned else None,
    }


def _pool_block(blocks) -> dict:
    """Several {counts, expected, returned} blocks summed into one."""
    counts, expected, returned, paths = {}, 0, 0, 0
    for block in blocks:
        expected += block.get("expected") or 0
        returned += block.get("returned") or 0
        paths += block.get("scored_paths") or 0
        for verdict, count in (block.get("counts") or {}).items():
            counts[verdict] = counts.get(verdict, 0) + count
    out = {"counts": counts, "expected": expected, "returned": returned}
    out.update(_rates(counts, expected, returned))
    if paths:
        out["scored_paths"] = paths
    return out


def pool(scores) -> dict:
    """Several documents of ONE file, as one score for the file.

    **At the user's request** (2026-09-08): *if the doc got more we still score
    them as 1 doc but we do more of an average way.* A pack used to be scored on
    its first document and the rest not at all, which reported a seven-document
    file on a seventh of itself.

    **The rates are the MEAN OF THE DOCUMENTS and the counts are pooled**
    (2026-09-11, at the user's request: *if a file has multiple type the score
    will be calculated in total, each page then average*). Every document counts
    once whatever its size, so a seven-document file is the average of seven
    readings rather than a figure the two biggest documents decide.

    This reverses the 2026-09-08 rule, whose objection was real and is answered
    rather than dismissed: `field_acc` sits beside `p1_correct` and `p1_scored`
    in the run log, those two are COUNTS and stay pooled, so on a pack row the
    rate no longer follows from the pair beside it. What makes that recoverable
    is that the row says so -- `documents` is 2 or more on exactly those rows
    and on no others, and `doc_scores` carries each document's own rate, so both
    readings are re-derivable from the row. The pooled rates are kept as
    `accuracy_pooled` / `accuracy_half_pooled` / `accuracy_loose_pooled` for the
    same reason: a figure this project has quoted before must stay reachable.

    Read the two together the way `accuracy` and `accuracy_loose` are read
    together -- pooled well above the mean means the big documents carried the
    file.

    **One score is returned untouched.** Every single-document case therefore
    produces exactly the dict it produced before this existed, which is what
    keeps the thirteen ordinary fixtures comparable across the change.

    Shared by `app._merge_field_scores` and `compare._pool_scores` rather than
    written twice: the app scores through its own truth files and the CLI
    through the working copy's, but the ARITHMETIC over the results is one fact,
    and two copies of it disagree eventually -- which this project has already
    paid for once in `case_payload`.
    """
    scores = [s for s in scores if s and not s.get("error")]
    if not scores:
        return {"error": "no document returned fields"}
    if len(scores) == 1:
        return scores[0]

    pooled = {
        "case": scores[0].get("case"),
        "overall": _pool_block([s.get("overall") or {} for s in scores]),
        "optional": _pool_block([s.get("optional") or {} for s in scores]),
        # Every type in the file, in the order the documents appear. Not
        # deduplicated: sol015 holds two billing notes and two payment
        # schedules, and a list that hid that would misdescribe the file.
        "doc_types": [c for s in scores for c in (s.get("doc_types") or [])],
        "warnings": [w for s in scores for w in (s.get("warnings") or [])],
        # True only where NO document of the file is covered by a requirement.
        # A pack mixing a credit note with a goods-return note is partly a
        # compliance figure and partly not, and calling the whole of it an
        # unknown type would be false about the half that is not.
        "unknown_type": all(s.get("unknown_type") for s in scores),
        "scored_documents": len(scores),
    }
    scalars = _pool_block([s.get("scalars") or {} for s in scores])
    for tier in ("p1", "p2", "p3"):
        blocks = [(s.get("scalars") or {}).get(tier) for s in scores]
        if any(isinstance(b, dict) for b in blocks):
            scalars[tier] = _pool_block([b for b in blocks if isinstance(b, dict)])
    # Every document's rows, in document order, so a per-value report still
    # lists what each document was marked on.
    scalars["rows"] = [row for s in scores
                       for row in ((s.get("scalars") or {}).get("rows") or [])]
    for name in ("checked", "unchecked"):
        scalars[name] = sum((s.get("scalars") or {}).get(name) or 0 for s in scores)
    pooled["scalars"] = scalars
    for name in ("line_items", "income_items", "other_fields"):
        blocks = [s.get(name) for s in scores if isinstance(s.get(name), dict)]
        pooled[name] = _pool_block(blocks) if blocks else None
    coverage = {}
    for s in scores:
        for key, value in (s.get("coverage") or {}).items():
            if isinstance(value, (int, float)):
                coverage[key] = coverage.get(key, 0) + value
    pooled["coverage"] = coverage
    # The headline swaps: each document's own rate, meaned, with the pooled
    # figure kept beside it under a name that says what it is. `accuracy_macro`
    # stays as an alias of the new headline -- it is what the page and
    # `compare.py` already print as "macro", and renaming it would move a label
    # without moving a number.
    for name in ("accuracy", "accuracy_half", "accuracy_loose"):
        rates = [(s.get("overall") or {}).get(name) for s in scores]
        rates = [r for r in rates if r is not None]
        pooled["overall"][name + "_pooled"] = pooled["overall"].get(name)
        pooled["overall"][name] = (round(sum(rates) / len(rates), 4)
                                   if rates else None)
    pooled["overall"]["accuracy_macro"] = pooled["overall"]["accuracy"]
    pooled["per_document"] = [{
        "pages": list(s.get("pages") or []),
        "doc_types": list(s.get("doc_types") or []),
        "accuracy": (s.get("overall") or {}).get("accuracy"),
        # The half-credit rate as well as the strict one, because that is what
        # the run log means by a run's field score (`runlog._p1_rate`) and what
        # the per-document cell carries. Two spellings of "how did this document
        # do" that followed different arithmetic would be the one disagreement a
        # reader cannot recover from.
        "accuracy_half": (s.get("overall") or {}).get("accuracy_half"),
        "expected": (s.get("overall") or {}).get("expected") or 0,
        "unknown_type": bool(s.get("unknown_type")),
    } for s in scores]
    pooled["scored_scope"] = (
        "%d documents in this file, each scored on its own and the rates "
        "averaged -- every value their requirements demand of them, with each "
        "document counting once whatever its size" % len(scores))
    return pooled


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# the item table (2026-09-30)
# --------------------------------------------------------------------------
#
# Scored BESIDE the field score and never inside it. No requirement makes a row
# of this table Mandatory, so it has no place in a headline that is the
# requirement's Mandatory set -- the standing `other_fields` has, for a
# different reason with the same shape. And it measures a different thing: the
# table is read out of the transcript in Python (`tables.py`), so a wrong cell
# here is the READ's, or a repair's, and never the extraction model's.

def _document_text(case_id: str, pages=None) -> str:
    """The ground-truth transcript of one document of a case, page markers kept.

    `pages` picks the document out of a file holding several; without it the
    whole file is the document. The markers are rebuilt rather than sliced out
    of the text, so a table's rows still say which page they came off.
    """
    path = ground_truth_path(case_id)
    if path is None:
        return ""
    text = path.read_text("utf-8")
    if not pages:
        return text
    import segment
    every = segment.split_pages(text)
    return "\n".join("--- page %d ---\n%s" % (n, every[n - 1])
                     for n in pages if 0 < n <= len(every))


def table_truth(case_id: str, pages=None) -> dict:
    """What a document's item table should come back as.

    {"table": the item table of the ground-truth transcript, or None,
     "is_master_table": what a person recorded in the truth file, or None}

    **The rows are derived from the .md by the same code that reads them out of
    a model's transcript**, which is the rule the charges table has always
    followed and for the same reason: the .md is already a hand-checked
    transcription of those rows, and a second copy typed into JSON would
    disagree with it eventually. It also means the truth can never be in a
    shape the extractor could not have produced.

    **The flag is NOT derived.** `tables.classify` would give the same answer
    for the truth and for a perfect read by construction, so marking it against
    itself would measure nothing; what it is marked against is a person saying
    which tables list documents. None where nobody has said.
    """
    table = tables.item_table(_document_text(case_id, pages))
    expected = None
    path = truth_path(case_id)
    if path.exists():
        try:
            raw = json.loads(path.read_text("utf-8"))
            block = _pick_document(raw, pages, []) if isinstance(raw, dict) else {}
            if isinstance(block.get("is_master_table"), bool):
                expected = block["is_master_table"]
        except (json.JSONDecodeError, OSError):
            pass
    return {"table": table, "is_master_table": expected}


def _map_columns(truth_columns, actual_columns) -> dict:
    """truth column index -> the returned column that answers it.

    Positional where the two tables are the same width, which is nearly always
    and is the robust reading: a read that garbled a heading still put its cells
    in the right column, and matching on the garbled wording would lose a column
    the read got entirely right. By heading only where the widths differ, so a
    table that lost or gained a column still has the others marked.
    """
    if len(truth_columns) == len(actual_columns):
        return {i: i for i in range(len(truth_columns))}
    import difflib
    pairs = []
    for i, want in enumerate(truth_columns):
        for j, got in enumerate(actual_columns):
            a, b = grounding.squash(want), grounding.squash(got)
            if a and b:
                ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                if ratio >= 0.6:
                    pairs.append((ratio, -abs(i - j), i, j))
    pairs.sort(reverse=True)
    mapped, taken = {}, set()
    for _, _, i, j in pairs:
        if i not in mapped and j not in taken:
            mapped[i] = j
            taken.add(j)
    return mapped


def score_table(truth: dict, actual: dict, expected_master=None) -> dict:
    """One document's item table against the table its page prints.

    `truth` and `actual` are tables as `tables.item_table` returns them, either
    of which may be None: a page that rules no item table, or a read that
    produced none.

    Rows are paired by content and cells judged one by one, by the functions the
    field score uses -- `_pair_rows`, `judge` -- so a dropped row costs the cells
    it printed and a cell is "the same" on exactly the terms a field is.

    `master` is the one verdict here that is a person's: `is_master_table`
    against the truth file's. `undetermined` is a returned table the classifier
    could not place, and is not a correct answer.
    """
    truth_cols = list((truth or {}).get("columns") or [])
    actual_cols = list((actual or {}).get("columns") or [])
    mapping = _map_columns(truth_cols, actual_cols) if truth and actual else {}
    keys = ["c%d" % i for i in range(len(truth_cols))]

    truth_rows = [{keys[i]: (row[i] if i < len(row) else "") for i in range(len(keys))}
                  for row in (truth or {}).get("rows") or []]
    actual_rows = []
    for row in (actual or {}).get("rows") or []:
        actual_rows.append({keys[i]: (row[j] if j < len(row) else "")
                            for i, j in mapping.items()})
    if truth:
        result = _score_items(truth_rows, actual_rows, field="item_table", shape=keys)
        # Where each judged cell sits in the RETURNED table, and under which
        # printed heading. The path names the truth's own column, and the two
        # tables need not be the same width -- so without this nothing holding
        # the returned rows can put a verdict on the cell it is about.
        for row in result["rows"]:
            index = int(row["path"].rsplit(".c", 1)[1])
            row["column"] = mapping.get(index)
            row["heading"] = truth_cols[index]
    else:
        # The page rules no item table. Nothing to be right about; what the read
        # returned anyway is counted and kept out of every rate.
        result = _tally([])
        result.update(rows=[], rows_expected=0, rows_matched=0, rows_missed=0,
                      rows_returned=len((actual or {}).get("rows") or []),
                      rows_spurious=len((actual or {}).get("rows") or []),
                      spurious_cells=sum(1 for r in (actual or {}).get("rows") or []
                                         for c in r if not grounding.is_blank(c)),
                      in_order=True)
    got = (actual or {}).get("is_master_table")
    if expected_master is None:
        status = "unchecked"
    elif not actual:
        status = "missed"
    elif got is None:
        status = "undetermined"
    else:
        status = "correct" if got == expected_master else "wrong"
    result.update(
        columns=truth_cols,
        columns_expected=len(truth_cols),
        columns_returned=len(actual_cols),
        columns_matched=len(mapping),
        table_expected=bool(truth),
        table_returned=bool(actual),
        misaligned=len((actual or {}).get("misaligned") or []),
        master={"expected": expected_master, "actual": got, "status": status},
    )
    return result


def evaluate_table(case_id: str, actual, pages=None):
    """`score_table` for a benchmark case, or None where there is nothing to say.

    None -- not a score of zero -- where the page rules no item table, nobody
    recorded a flag for it and the read returned none: three absences agreeing
    is not a measurement.
    """
    try:
        truth = table_truth(case_id, pages)
    except Exception as err:        # pragma: no cover - a score is never worth a 500
        return {"error": f"table scoring failed: {err}"}
    if not truth["table"] and truth["is_master_table"] is None and not actual:
        return None
    result = score_table(truth["table"], actual, truth["is_master_table"])
    # What the table SHOULD come back as, for the page to draw beside the one
    # that did. Columns and rows only: the rest of the truth's table is the
    # parser's working, and nothing reads it.
    if truth["table"]:
        result["truth_table"] = {"columns": list(truth["table"]["columns"]),
                                 "rows": [list(r) for r in truth["table"]["rows"]]}
    result["case"] = case_id
    if pages:
        result["pages"] = list(pages)
    return result


def transcript_tables(case_id: str, text: str) -> dict:
    """Every document of a case: its item table read from a transcript, scored.

    {"tables": [the table or None, one per document], "score": pooled or None}

    **The pages come from the manifest, not from the read**, the rule
    `scoring.score_documents` follows and for its reason: asking `segment` here
    would make a table's score depend on whether that run's segmentation agreed,
    which is two things to be wrong at once.

    Needs no extraction and no model: the table is a function of the transcript,
    so `compare.py --no-run` can report it off a saved read.
    """
    import scoring
    import segment
    case = scoring.cases_index().get(case_id) or {}
    documents = case.get("documents") or []
    pages = segment.split_pages(text or "")
    found, scores = [], []
    for entry in (documents if len(documents) > 1 else [None]):
        numbers = list(entry["pages"]) if entry else None
        # A type that rules a table of its own -- a withholding certificate's
        # income rows -- is not looked at, exactly as the app does not look.
        if prompts.items_for_types((entry or case).get("doc_types") or []):
            continue
        body = (text if numbers is None else
                "\n".join("--- page %d ---\n%s" % (n, pages[n - 1])
                          for n in numbers if 0 < n <= len(pages)))
        table = tables.item_table(body)
        found.append(table)
        scores.append(evaluate_table(case_id, table, numbers))
    return {"tables": found, "score": pool_tables(scores)}


def format_table_report(found, score) -> list:
    """The item-table line(s) of a report, for one case."""
    found = [t for t in (found or []) if isinstance(t, dict) and "rows" in t]
    lines = []
    if not found and not score:
        return lines
    rows = sum(len(t.get("rows") or []) for t in found)
    masters = sum(1 for t in found if t.get("is_master_table") is True)
    unsure = sum(1 for t in found if t.get("is_master_table") is None)
    fixed = sum(len(t.get("realigned") or []) for t in found)
    bad = sum(len(t.get("misaligned") or []) for t in found)
    if len(found) == 1:
        kind = ("master table (a list of other documents)" if masters
                else "could not be classified" if unsure else "item list")
        head = "%d row%s, %s" % (rows, "" if rows == 1 else "s", kind)
    else:
        head = "%d table%s, %d row%s, %d master table%s" % (
            len(found), "" if len(found) == 1 else "s", rows,
            "" if rows == 1 else "s", masters, "" if masters == 1 else "s")
    if fixed:
        head += ", %d row%s re-cut to fit the headings" % (fixed, "" if fixed == 1 else "s")
    if bad:
        head += ", %d row%s still misaligned" % (bad, "" if bad == 1 else "s")
    lines.append("  item table           " + head)
    if isinstance(score, dict) and "counts" in score:
        parts = []
        if score.get("expected"):
            parts.append("cells %d of %d correct (%s)" % (
                score["counts"].get("correct", 0), score["expected"],
                pct(score.get("accuracy"))))
            parts.append("rows %d of %d matched" % (
                score.get("rows_matched") or 0, score.get("rows_expected") or 0))
        if "master_scored" in score:
            if score["master_scored"]:
                parts.append("is_master_table %d of %d correct" % (
                    score.get("master_correct") or 0, score["master_scored"]))
        else:
            master = score.get("master") or {}
            if master.get("status") not in (None, "unchecked"):
                parts.append("is_master_table %s (truth says %s)" % (
                    master["status"], str(master.get("expected")).lower()))
        if parts:
            lines.append("                       " + "; ".join(parts))
    for table in found if len(found) == 1 else []:
        if table.get("master_why"):
            lines.append("                       why: " + table["master_why"])
    return lines


def pool_tables(scores) -> dict:
    """Several documents' table scores as one for the file. Counts are summed.

    One score is handed back untouched. The rate follows from the pooled counts
    here -- unlike the field headline there is no per-document mean, because a
    file of seven tables of one to thirteen rows has no sensible "average
    table", and the counts are what a reader can check.
    """
    scores = [s for s in (scores or []) if isinstance(s, dict) and "counts" in s]
    if not scores:
        return None
    if len(scores) == 1:
        return scores[0]
    counts = {}
    for one in scores:
        for name, value in (one.get("counts") or {}).items():
            counts[name] = counts.get(name, 0) + value
    expected = sum(one.get("expected") or 0 for one in scores)
    returned = sum(one.get("returned") or 0 for one in scores)
    checked = [one["master"] for one in scores
               if (one.get("master") or {}).get("status") not in (None, "unchecked")]
    out = {"counts": counts, "expected": expected, "returned": returned,
           **_rates(counts, expected, returned)}
    for name in ("rows_expected", "rows_returned", "rows_matched", "rows_missed",
                 "rows_spurious", "spurious_cells", "misaligned"):
        out[name] = sum(one.get(name) or 0 for one in scores)
    out["documents"] = len(scores)
    out["master_scored"] = len(checked)
    out["master_correct"] = sum(1 for m in checked if m["status"] == "correct")
    out["per_document"] = [
        {"pages": one.get("pages") or [], "expected": one.get("expected") or 0,
         "accuracy": one.get("accuracy"), "master": one.get("master")}
        for one in scores]
    return out


def pct(value):
    """A rate as a percentage, or n/a when nothing was scored.

    Public because `compare.py` prints its summary table with it: two spellings
    of "no truth file for this" -- one reading n/a and one reading 0.0% -- would
    say opposite things about the same run.
    """
    return "   n/a" if value is None else f"{value:6.1%}"


def _clip(value, width=42):
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def format_report(result: dict, show: int = 40) -> list:
    """The score as printable lines. Shared so a script and the CLI agree."""
    if result.get("error"):
        return [f"  field score unavailable: {result['error']}"]

    lines = []
    overall, scalars = result["overall"], result["scalars"]
    counts = overall["counts"]
    lines.append(f"  field accuracy       {pct(overall['accuracy'])}"
                 f"   ({counts['correct']}/{overall['expected']} values"
                 f", half {pct(overall['accuracy_half']).strip()}"
                 f", loose {pct(overall['accuracy_loose']).strip()})")
    lines.append(f"  field precision      {pct(overall['precision'])}"
                 f"   ({counts['correct']}/{overall['returned']} of what it filled)")
    lines.append(f"  correct {counts['correct']}  partial {counts['partial']}  "
                 f"wrong {counts['wrong']}  missed {counts['missed']}  "
                 f"spurious {counts['spurious']}  agreed-absent {counts['absent']}")
    # What the two rates above are OVER. Printed every run rather than on demand:
    # since 2026-09-02 the headline is the requirement's Mandatory set, so a rate
    # of 100% is a claim about compliance and not about the whole form, and the
    # number alone cannot say which.
    if result.get("scored_scope"):
        lines.append(f"  scored over: {result['scored_scope']}")
    optional = result.get("optional") or {}
    if optional.get("expected"):
        oc = optional["counts"]
        lines.append(f"  optional fields      {pct(optional['accuracy'])}"
                     f"   ({oc['correct']}/{optional['expected']} values"
                     " -- found, not in the headline)")

    tiers = "  ".join(
        f"{tier} {pct(scalars[tier]['accuracy']).strip()}"
        f" ({scalars[tier]['counts']['correct']}/{scalars[tier]['expected']})"
        for tier in ("p1", "p2") if scalars[tier]["expected"])
    if tiers:
        lines.append(f"  by tier: {tiers}")

    items = result["line_items"]
    if items is not None:
        order = "" if items["in_order"] else ", OUT OF ORDER"
        lines.append(f"  line items: {items['rows_matched']}/{items['rows_expected']}"
                     f" rows matched, {items['rows_returned']} returned"
                     f", {items['rows_spurious']} spurious{order}"
                     f" -- cells {pct(items['accuracy']).strip()}")
        # What was derived from the .md, said every run rather than on demand: a
        # column this failed to recognise makes a correct extraction look wrong,
        # and there is nothing else on screen that would give that away.
        got = items.get("derived") or {}
        if got:
            lines.append(f"    rows from {got.get('source', '?')}: "
                         + ", ".join(f"{key} ← {head}"
                                     for head, key in (got.get("columns") or {}).items())
                         + (f"; {len(got['dropped'])} row(s) dropped ("
                            + ", ".join(sorted({d["why"] for d in got["dropped"]}))
                            + ")" if got.get("dropped") else ""))
            if got.get("unmapped"):
                lines.append("    columns not scored, no key matches their heading: "
                             + ", ".join(got["unmapped"])
                             + " — name them in table_columns if one belongs to a key")

    income = result.get("income_items")
    if income is not None:
        order = "" if income["in_order"] else ", OUT OF ORDER"
        lines.append(f"  income items: {income['rows_matched']}"
                     f"/{income['rows_expected']} rows matched, "
                     f"{income['rows_returned']} returned, "
                     f"{income['rows_spurious']} spurious{order}"
                     f" -- cells {pct(income['accuracy']).strip()}")
        # No `derived` line: unlike the charges table these rows are stated in
        # the truth file rather than read out of the .md, so there is no column
        # mapping that could silently be wrong.

    others = result["other_fields"]
    if others is not None:
        lines.append(f"  other fields: {others['labels_matched']}"
                     f"/{others['labels_expected']} labels matched"
                     f", values {pct(others['accuracy']).strip()}"
                     f" (not in the headline)")

    coverage = result["coverage"]
    if coverage["scalars_checked"] < coverage["scalars_total"]:
        lines.append(f"  truth covers {coverage['scalars_checked']}"
                     f"/{coverage['scalars_total']} scalar keys"
                     " -- the rest are null and unscored")
    for warning in result["warnings"]:
        lines.append(f"  ! {warning}")

    bad = [r for r in (scalars["rows"] + (items["rows"] if items else []))
           if r["status"] in ("wrong", "partial", "missed", "spurious")]
    if bad:
        lines.append("")
        lines.append(f"  {'field':34} {'status':9} {'expected':44} got")
        for row in bad[:show]:
            # A row outside the headline is marked rather than dropped: it is
            # still a value the page prints and the extraction got wrong, and it
            # is still worth correcting -- it simply does not move the rate.
            mark = "" if row.get("required", True) else "  (optional)"
            lines.append(f"  {row['path'][:34]:34} {row['status']:9} "
                         f"{_clip(row['expected']):44} {_clip(row['actual'])}{mark}")
        if len(bad) > show:
            lines.append(f"  ... and {len(bad) - show} more")
        # Only the words actually in the table above, in the order the column
        # ranks them. A key to six terms none of which appeared is noise, and the
        # table is read by whoever is correcting the truth file rather than by
        # someone who already knows the vocabulary.
        seen = {r["status"] for r in bad[:show]}
        for name in _STATUS_ORDER:
            if name in seen:
                lines.append(f"    {name:9} {STATUS_MEANING[name]}")
        # Where a key accepts more than one reading, the column above holds the
        # one the answer came closest to. Said once, and only when there is such
        # a key in the table, so an ordinary run does not carry a footnote about
        # a format it does not use.
        multi = sorted({r["path"] for r in bad[:show] if len(r.get("accepted") or ()) > 1})
        if multi:
            lines.append("  expected shows the accepted reading the answer came "
                         "closest to; these keys accept more than one: "
                         + ", ".join(multi))
    return lines


# --------------------------------------------------------------------------
# creating the files
# --------------------------------------------------------------------------

def init(case_ids=None, force: bool = False) -> list:
    """Write an empty truth file for every case that has none. Never overwrites.

    Overwriting is refused rather than confirmed: these files are hand-written
    over hours and there is no other copy of one. `--force` exists for the case
    of a file that was created and never touched, and it says so first.
    """
    import scoring                       # only needed here; keeps the import graph thin

    index = scoring.cases_index()
    written = []
    for case_id in (case_ids or list(index)):
        case = index.get(case_id, {})
        path = truth_path(case_id)
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(skeleton(case_id, case.get("pdf", ""), case.get("kind", "")),
                        "utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "init"]
    force = "--force" in args
    ids = [a for a in args if not a.startswith("-")] or None
    made = init(ids, force=force)
    for path in made:
        config.say(f"wrote {path}")
    if not made:
        config.say("nothing to write -- every case already has a field truth file"
                   " (--force overwrites, and there is no undo)")
