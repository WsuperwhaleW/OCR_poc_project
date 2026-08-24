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
_CONFIG_KEYS = ("other_fields", "table_columns", "score_table", "line_items")

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

_PIPE_SPLIT = re.compile(r"(?<!\\)\|")
_RULE_CELL = re.compile(r"^:?-{2,}:?$")


def _cells(line: str) -> list:
    """One Markdown table row as its cells, outer pipes and escapes removed."""
    parts = _PIPE_SPLIT.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [re.sub(r"<br\s*/?>", " ", cell).replace("\\|", "|").strip()
            for cell in parts]


def _tables(text: str) -> list:
    """Every Markdown pipe table in a document, as (header cells, row cell lists)."""
    found = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        block = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index])
            index += 1
        if len(block) >= 3:
            rule = _cells(block[1])
            if rule and all(_RULE_CELL.match(c) for c in rule if c):
                found.append((_cells(block[0]), [_cells(ln) for ln in block[2:]]))
    return found


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


def load_truth(case_id: str) -> dict:
    """Read one case's field ground truth. Raises ValueError if unusable.

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
    scalars = {}
    for key, value in raw.items():
        if key.startswith("_") or key in _CONFIG_KEYS:
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

    return {"scalars": scalars, "other_fields": others,
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
        "other_fields: null means not scored. A list means these labels and",
        "values are what the page prints outside the schema. Scored and",
        "reported separately -- the labels are the model's own wording, so they",
        "never move the headline number.",
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


def _score_items(truth_rows, actual_rows) -> dict:
    actual_rows = [r for r in (actual_rows or []) if isinstance(r, dict)]
    matched = _pair_rows(truth_rows, actual_rows)
    rows = []

    for i, truth_row in enumerate(truth_rows):
        j = matched.get(i)
        source = actual_rows[j] if j is not None else {}
        for key, value in truth_row.items():
            rows.append({
                "path": f"line_items[{i}].{key}",
                "expected": value,
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
                "status": judge(value, None if j is None else source.get(key)),
            })

    spurious_cells = 0
    for j, actual_row in enumerate(actual_rows):
        if j in matched.values():
            continue
        spurious_cells += sum(1 for k in ITEM_KEYS
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

def score(truth: dict, fields, table: dict = None) -> dict:
    """Score one extraction against one loaded truth file.

    `fields` is the `fields` object of an extraction result -- what the model
    returned, before any of it is displayed. `table` is what `table_rows` read out
    of the .md, passed in rather than fetched so this stays a pure function of its
    arguments and can be tested without a solution directory.
    """
    fields = fields if isinstance(fields, dict) else {}
    scalar_rows = []
    for key, value in truth["scalars"].items():
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
    scalars["unchecked"] = len(SCALARS) - len(scalar_rows)
    for tier in ("p1", "p2"):
        scalars[tier] = _tally([r for r in scalar_rows if r["tier"] == tier])

    warnings = list(truth.get("warnings") or [])
    items = None
    if not truth.get("score_table", True):
        warnings.append(
            "table not scored: pass 2 does not extract line items"
            if not prompts.EXTRACT_LINE_ITEMS
            else "table not scored: score_table is false in the truth file")
    elif table and table.get("error"):
        warnings.append(f"table not scored: {table['error']}")
    elif table:
        items = _score_items(table["rows"], fields.get("line_items"))
        # Where these rows came from, carried with the score. A derived truth has
        # to say what it derived, or a wrong column mapping looks like a wrong
        # extraction -- and the mapping is the one part of this nobody wrote down
        # by hand.
        items["derived"] = {key: table.get(key) for key in
                            ("source", "columns", "unmapped", "dropped")}

    others = (_score_others(truth["other_fields"], fields.get("other_fields"))
              if truth["other_fields"] is not None else None)

    # The headline is the schema's own keys: the scalars plus the table's cells.
    # `other_fields` is excluded on purpose -- see the module docstring.
    overall = _tally(scalar_rows + (items["rows"] if items else []))
    overall["scored_paths"] = len(scalar_rows) + (len(items["rows"]) if items else 0)

    return {
        "overall": overall,
        "scalars": scalars,
        "line_items": items,
        "other_fields": others,
        "warnings": warnings,
        # What the truth file actually rules on. A score taken over 6 of 29 keys
        # is a different claim from one taken over all of them, and the number
        # alone cannot say which it is.
        "coverage": {
            "scalars_checked": len(scalar_rows),
            "scalars_total": len(SCALARS),
            "line_items_checked": items is not None,
            "other_fields_checked": truth["other_fields"] is not None,
        },
    }


def evaluate(case_id: str, fields) -> dict:
    """Load the truth for a case and score `fields` against it.

    Returns {"error": ...} rather than raising, so a caller on a request path can
    put it on a response without a try/except of its own.
    """
    try:
        truth = load_truth(case_id)
    except ValueError as err:
        return {"error": str(err)}
    table = (table_rows(case_id, truth.get("table_columns"))
             if truth.get("score_table", True) else None)
    result = score(truth, fields, table)
    result["case"] = case_id
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

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
            lines.append(f"  {row['path'][:34]:34} {row['status']:9} "
                         f"{_clip(row['expected']):44} {_clip(row['actual'])}")
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
