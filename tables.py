"""The item table of a document, read out of its transcript -- and what KIND of table it is.

Added 2026-09-30 at the user's request, in two lines:

    1. fix table and also extract table
    2. identify item table -> real item list or ref number list, e.g. this table
       is a master table for another table. only identify them: is_master_table

**Nothing here asks a model to read the table.** Pass 1 has already read it: a
model-server transcript carries the page's table as `<table>` HTML or as a
Markdown pipe table, so the rows are IN the transcript and this module takes
them out again. That is the whole design, and three things follow from it that
an extraction prompt could not give:

  * every cell is grounded by construction -- it is a substring of the
    transcript, so there is nothing for `grounding.py` to audit and nothing a
    model could have invented;
  * it cannot loop. Pass 2 cycling over near-identical line items until the
    token cap is the oldest failure in CLAUDE.md, and it is unreachable here;
  * **no pass-2 prompt changes**, so not one field baseline moves. The table was
    taken OUT of pass 2 on 2026-08-19 because a fixed eight-key row made the
    model smear values across columns the page never ruled. The row shape here
    is the document's OWN columns, which is the fix that entry asked for.

What a transcript's table needs before it is usable is REPAIR, and that is the
"fix table" half. An OCR model's table is frequently not a grid:

  * `colspan` / `rowspan` cells (expanded here, the ordinary grid algorithm);
  * a heading printed over two rows (merged into one heading per column);
  * totals, section headings and notes ruled as rows of the table (taken out of
    the item rows and KEPT, under `dropped`, with the reason -- nothing is
    silently removed);
  * one table continued over several pages (joined, where the headings agree);
  * and rows cut into the wrong number of cells -- a reference split in two, a
    blank cell lost. sol007's header has five columns and every one of its rows
    has six.

The last of those is settled in two steps, and neither changes a character:

  1. `realign`, in Python, by what KIND of thing each cell is. An amount belongs
     under a money heading and a thirteen-digit number is not a description, and
     that pins most rows on its own. Taken only where one cut is clearly better
     than every other.
  2. `recut`, for a row the cells do not settle: **a model is asked which COLUMN
     each cell belongs under, and Python does the cutting.** No character of the
     model's answer ever reaches the table -- only column numbers do -- so a
     repaired row holds exactly the characters the read produced for it, by
     construction rather than by a check afterwards.

A row neither settles is never forced into its headings. It is kept as read and
flagged (`misaligned`).

**Since 2026-10-01 two table AGENTS sit in front of that** (`app._table_fix_agent`,
`app._table_identify_agent`), at the user's request. The fix agent looks at the
rows as the read cut them (`item_table(..., align=False)`) and QUOTES text in
them that does not belong to the table -- a stamp's words, a QR code's
description, stray text; `remove_text` cuts it out where the quote is found in
that cell and keeps it, as plain text, under `outside_text`. `removal_refused`
is the code's veto. The cuts are then settled exactly as before: `realign`, and
the column question for what it leaves (with `check_cut`). The identify agent answers `is_master_table` from the table, the
document's type and a few extracted values, and `reference_column_ok` is the
evidence a `true` must have. Neither is sent the document, and neither answer
reaches the table as text. `classify` below still runs every time, and its answer
is kept beside the agent's.

`is_master_table` is decided here too, from the headings and the cells, for the
reason every other classification in this project is decided in Python: it is a
fact about printed text that can be re-derived, and the evidence rides back with
the answer (`master_why`, `reference_columns`, `item_columns`).

  * a MASTER table lists other documents: each row is identified by a reference
    to one -- an invoice number, a tax invoice number, a reference document --
    and its figures are that document's. The rows of a receipt that settles
    twelve invoices, a billing note, a payment schedule.
  * a REAL ITEM list lists goods or services: a description, and usually a
    quantity and a unit price.

The rule is two tests, and both have to hold: the table rules a reference
column that is actually filled in, and it rules NO unit-price and NO
product-code column. The second test is what keeps a credit note whose rows are
returned goods -- each citing the tax invoice it came off -- a real item list,
which it is.

A QUANTITY column is deliberately not part of the second test, and a real read
is why. จำนวน is "number of" and จำนวนเงิน is "amount of money", one syllable
apart, and sol004 at 4 MP came back with จำนวนเงินรับ read as จำนวนสิทธิ์ -- a money
column wearing a quantity's heading, over rows of invoice numbers. A price per
unit and a product code are what only a list of goods rules; a count on its
own is something a list of documents can carry too.

A reference column is known by its HEADING, and failing that by its CELLS: a
column whose heading says nothing recognisable and whose cells are document
numbers on most rows. That second reading exists because a read garbles
headings -- see `classify` -- and it never overrules a heading that does say
something.

Three states, never two: True, False, and None where it cannot be told (the
read lost the headings, or not one row lines up with them). "Not determined"
is not "not a master table".

No I/O, no model, no app state. `app.py` owns the one request `recut` answers
and the settings; `fieldscore.py` owns the scoring.
"""
import math
import re
from html.parser import HTMLParser

import grounding
import segment
import verify


def _needles(*words):
    """Needles reduced the way the headings they are tested against are.

    `grounding.squash` keeps only alphanumerics and every Thai vowel and tone
    mark is a combining character `str.isalnum` rejects, so a needle written the
    way a person types it matches nothing, silently. See `normalise._needles`.
    """
    return tuple(grounding.squash(w) for w in words)


# --------------------------------------------------------------------------
# what a column heading says the column is
# --------------------------------------------------------------------------
# Substring tests against a squashed heading, the same mechanism as
# `fieldscore.HEADER_MAP` and deliberately not the same table: that one maps a
# heading onto one of eight schema keys for a scorer, this one asks four
# yes/no questions about a heading in order to choose a table and to tell a
# reference list from an item list.

# A column that names ANOTHER document. Compound wordings only: bare เลขที่ and
# bare No. are a row number as often as a document number (sol008, sol014), so
# neither is a needle, and `Invoice Date` / `Invoice Amount` / `Invoice
# Descriptions` (sol015 p7) must not be reference columns for mentioning an
# invoice.
_REFERENCE = _needles(
    "เลขที่ใบแจ้งหนี้", "เลขที่ใบกำกับ", "ใบกำกับภาษีเลขที่", "ใบกำกับเลขที่",
    "เลขที่เอกสาร", "เอกสารเลขที่", "เอกสารอ้างอิง", "อ้างอิง", "อ้างถึง",
    "เลขที่ใบส่งของ", "ใบส่งของเลขที่", "เลขที่ใบเสร็จ", "เลขที่ใบลดหนี้",
    "เลขที่ใบสั่งซื้อ", "เลขที่ใบวางบิล",
    "invoice no", "invoice number", "inv no", "tax invoice no",
    "reference", "ref no", "document no", "doc no", "purchasing no",
    "purchase order no", "po no", "cn no", "bill no", "receipt no",
)
# `Ref.` on its own, which squashes to three letters and would match inside
# "prefer" or "reference" as a bare needle. Tested against the RAW heading.
_REFERENCE_WORD = re.compile(r"(?<![A-Za-z])ref(?![A-Za-z])", re.I)

# Columns only a list of goods or services rules. Any one of them makes the
# table a real item list however many reference columns sit beside it.
_QUANTITY = _needles("quantity", "qty", "จำนวน")
# จำนวน is "number of" and จำนวนเงิน is "amount of money", so every money heading
# contains the quantity needle. Same blocker as `fieldscore.HEADER_BLOCKERS`,
# for the same reason: sol004 rules two money columns and neither is a count.
_QUANTITY_BLOCKERS = _needles("เงิน", "amount", "price", "ราคา")
_UNIT_PRICE = _needles("unit price", "price per unit", "unit/price", "ราคาต่อหน่วย",
                       "ราคา/หน่วย", "ราคาหน่วย", "หน่วยละ")
_PRODUCT = _needles("รหัสสินค้า", "item code", "item no", "product code", "sku",
                    "barcode", "หน่วยนับ", "unitmsr", "uom")

_DESCRIPTION = _needles("description", "particular", "รายการ", "รายละเอียด",
                        "สินค้า", "details")
# A date, a period, a due date. Asked for one reason only: a heading that says
# DATE has said what its column is, so the column is not a candidate for the
# reading-by-cells below -- and a period printed as two dates is a long run of
# digits that would otherwise look exactly like a document number.
_DATE = _needles("วันที่", "date", "period", "งวด", "ครบกำหนด", "due")
# Things a column of numbers can be that are NOT another document: a party's tax
# ID, an account, a telephone. Named so that a heading which says one of these is
# taken at its word rather than second-guessed from its cells.
_OTHER_NUMBER = _needles("เลขประจำตัว", "tax id", "taxid", "บัญชี", "account",
                         "โทร", "tel", "phone", "รหัสลูกค้า", "customer")
_MONEY = _needles("amount", "จำนวนเงิน", "ยอด", "ราคา", "รวมเงิน", "total",
                  "มูลค่า", "price")


def _has(squashed: str, needles) -> bool:
    return any(n and n in squashed for n in needles)


def column_roles(heading: str) -> set:
    """What one printed heading says its column holds. Empty where it says nothing."""
    squashed = grounding.squash(heading)
    roles = set()
    if not squashed:
        return roles
    if _has(squashed, _REFERENCE) or _REFERENCE_WORD.search(heading or ""):
        roles.add("reference")
    if _has(squashed, _UNIT_PRICE):
        roles.add("unit_price")
    if _has(squashed, _PRODUCT):
        roles.add("product")
    if _has(squashed, _QUANTITY) and not _has(squashed, _QUANTITY_BLOCKERS):
        roles.add("quantity")
    if _has(squashed, _DESCRIPTION):
        roles.add("description")
    if _has(squashed, _MONEY):
        roles.add("money")
    if _has(squashed, _DATE):
        roles.add("date")
    if _has(squashed, _OTHER_NUMBER):
        roles.add("other_number")
    return roles


# What only a list of goods or services rules. Not `quantity`: see the module
# docstring.
_ITEM_ROLES = ("unit_price", "product")


# --------------------------------------------------------------------------
# reading a table out of text
# --------------------------------------------------------------------------

_TABLE_BLOCK = re.compile(r"<table\b[^>]*>(.*?)(?:</table\s*>|\Z)", re.I | re.S)
_PIPE_SPLIT = re.compile(r"(?<!\\)\|")
_RULE_CELL = re.compile(r"^:?-{2,}:?$")
_SPACE = re.compile(r"\s+")

# A span an OCR model invented is not a reason to build a ten-thousand-cell row.
_MAX_SPAN = 60


def pipe_cells(line: str) -> list:
    """One Markdown table row as its cells, outer pipes and escapes removed."""
    parts = _PIPE_SPLIT.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [re.sub(r"<br\s*/?>", " ", cell).replace("\\|", "|").strip()
            for cell in parts]


def pipe_tables(text: str, min_lines: int = 3) -> list:
    """Every Markdown pipe table in a text, as (header cells, row cell lists).

    `min_lines` is 3 for the scorer that has always used this -- a heading, its
    rule and at least one row -- and 2 here, where a heading with nothing under
    it is still a table: sol004 reprints its headings on page 2 over no rows.
    """
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
        if len(block) >= max(2, min_lines):
            rule = pipe_cells(block[1])
            if rule and all(_RULE_CELL.match(c) for c in rule if c):
                found.append((pipe_cells(block[0]),
                              [pipe_cells(ln) for ln in block[2:]]))
    return found


def _span(value) -> int:
    try:
        return max(1, min(_MAX_SPAN, int(str(value).strip())))
    except (TypeError, ValueError):
        return 1


class _Rows(HTMLParser):
    """The `<tr>`/`<td>` structure of one table body, tolerant of what OCR emits.

    Tolerant on purpose. A model's HTML is frequently not well formed -- a `</th>`
    closing a `<td>`, a stray `</td>` with text in front of it, a `<tr>` never
    closed -- and the rows are still plainly there. A cell or a row left open is
    closed by the next one that starts.

    **Text outside any cell is two different things, and they are told apart by
    where it stands.** At the START of a row, before its first cell, it is a
    first cell whose tag the model forgot -- sol021 writes a row number that way
    -- and it is kept as one. Anywhere else it is debris between two cells that
    are both already there (sol015 repeats a date after a `<br/>`), and keeping
    it would push every cell after it one column right. That is dropped, and
    `stray` says what was dropped: nothing leaves a table without a record.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self._row, self._cell = [], None, None
        self.stray = []

    def _end_cell(self):
        if self._cell is not None and self._row is not None:
            self._cell[0] = _SPACE.sub(" ", self._cell[0]).strip()
            self._row.append(tuple(self._cell))
        self._cell = None

    def _end_row(self):
        self._end_cell()
        if self._row is not None:
            self.rows.append(self._row)
        self._row = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._end_row()
            self._row = []
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._end_cell()
            attrs = dict(attrs)
            self._cell = ["", _span(attrs.get("colspan")), _span(attrs.get("rowspan"))]
        elif tag == "br" and self._cell is not None:
            self._cell[0] += " "

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._end_cell()
        elif tag == "tr":
            self._end_row()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[0] += data
            return
        text = _SPACE.sub(" ", data).strip()
        if not text or self._row is None:
            return
        if not self._row:
            self._row.append((text, 1, 1))
        else:
            self.stray.append(text)

    def close(self):
        super().close()
        self._end_row()


def _grid(raw_rows):
    """Spans expanded: every row a plain list of strings, one per grid column.

    A `colspan` leaves its text in the first column it covers and "" in the
    rest. A `rowspan` REPEATS its text down the rows it covers -- a cell ruled
    across two rows says the same thing about both -- and those repeats are
    returned as `inherited`, per row, so a heading printed over two rows is not
    mistaken for a second heading with the same words.
    """
    grid, inherited, pending = [], [], {}
    for raw in raw_rows:
        row, mine, col = [], set(), 0

        def carry():
            nonlocal col
            while col in pending:
                text, left = pending[col]
                row.append(text)
                mine.add(col)
                if left <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, left - 1)
                col += 1

        for text, colspan, rowspan in raw:
            carry()
            for offset in range(colspan):
                value = text if offset == 0 else ""
                row.append(value)
                if rowspan > 1:
                    pending[col] = (value, rowspan - 1)
                col += 1
        carry()
        grid.append(row)
        inherited.append(mine)
    return grid, inherited


def _html_tables(text: str) -> list:
    found = []
    for block, match in enumerate(_TABLE_BLOCK.finditer(text)):
        parser = _Rows()
        try:
            parser.feed(match.group(1))
            parser.close()
        except Exception:   # pragma: no cover - HTMLParser is lenient already
            continue
        grid, inherited = _grid(parser.rows)
        if grid:
            found.append({"kind": "html", "grid": grid, "inherited": inherited,
                          "at": match.start(), "stray": list(parser.stray),
                          "block": block})
    return found


# What the fix agent is shown around a table: the table in the middle, and
# CONTEXT_SHARE of the page's lines on each side of it -- 20% before and 20%
# after, a page's lines counting every line of text and every row of every
# table on it. At the user's request (2026-10-01): *sometimes the table got cut
# or the total got cut*. An OCR model ends a table early and writes its last
# rows, or its totals, as loose lines or as a second small table; it starts one
# late and writes the headings or the first row above it. A fixed run of lines
# AFTER the table, stopping at the next table, saw neither the part above nor a
# totals block the read had ruled as a table of its own.
#
# A share of the PAGE rather than a fixed count, so a dense page shows more and
# a sparse one does not hand over half of itself. CONTEXT_MIN_LINES is the
# floor under it, for a page so short that 20% is a line or two; the window
# never leaves the page the table is on -- a page break is not inside a frame.
# `app` passes the setting (`TABLE_CONTEXT_SHARE`); the constants are the default.
CONTEXT_SHARE = 0.20
CONTEXT_MIN_LINES = 3
_CONTEXT_LINE_CHARS = 300
# A line that is the start of another table: what an HTML table is blanked to,
# so the walk over a page can tell where each sat.
_NEXT_TABLE = "\x00table\x00"
_PIPE_RULE_LINE = re.compile(r"^\|?[\s:\-|]+\|?$")


def _row_line(cells) -> str:
    """One row of a table as a line of text, its filled cells joined by `|`."""
    return " | ".join(c.strip() for c in cells if (c or "").strip())


def _page_items(blanked: str, tables: list) -> list:
    """The page as a list of lines, every table expanded to its rows.

    Each item is {text, table}: `table` is the parsed table a row belongs to, or
    None for a line of text. A table that never parsed into one -- a block with
    no rows, a pipe run too short to be a table -- contributes its rows as
    lines of text, which is what it is to a reader.
    """
    html = {t["block"]: t for t in tables if t["kind"] == "html"}
    pipes = {t["line_start"]: t for t in tables if t["kind"] == "pipe"}
    lines = blanked.split("\n")
    items, block, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        if _NEXT_TABLE in line:
            table = html.get(block)
            block += 1
            for row in (table or {}).get("grid") or []:
                text = _row_line(row)
                if text:
                    items.append({"text": text[:_CONTEXT_LINE_CHARS], "table": table})
            i += 1
            continue
        if i in pipes:
            table = pipes[i]
            for row in table["grid"]:
                text = _row_line(row)
                if text:
                    items.append({"text": text[:_CONTEXT_LINE_CHARS], "table": table})
            i = max(table["line_end"], i + 1)
            continue
        text = line.strip()
        if text and not segment.PAGE_MARKER.match(text) \
                and not (text.startswith("|") and _PIPE_RULE_LINE.match(text)):
            items.append({"text": text[:_CONTEXT_LINE_CHARS], "table": None})
        i += 1
    return items


def context_window(part_first: dict, part_last: dict, share=None) -> tuple:
    """(lines before, lines after) the table: `share` of its page's lines each side.

    `part_first` and `part_last` are the first and last pieces of the table --
    one table read off one page is both. The window before comes from the page
    the table starts on, the window after from the page it ends on.
    """
    share = CONTEXT_SHARE if share is None else share

    def size(items):
        return max(CONTEXT_MIN_LINES, math.ceil(round(share * len(items), 6)))

    before, after = [], []
    items = part_first.get("ctx") or []
    span = part_first.get("ctx_span")
    if span:
        before = [it["text"] for it in items[max(0, span[0] - size(items)):span[0]]]
    items = part_last.get("ctx") or []
    span = part_last.get("ctx_span")
    if span:
        after = [it["text"] for it in items[span[1]:span[1] + size(items)]]
    return before, after


def _pipe_tables_at(text: str) -> list:
    found = []
    # Located as well as parsed, so tables come back in the order the page
    # prints them whichever syntax each is written in -- and so the text printed
    # AFTER each can be kept with it. A table is found by its heading LINE: the
    # first pipe line, from where the last table ended, whose cells are this
    # table's headings. (Until 2026-10-01 it searched for the headings joined
    # with a bare `|`, which a `| a | b |` line never contains, so every pipe
    # table was placed at the top of its page.)
    lines = text.split("\n")
    starts, at = [], 0
    for line in lines:
        starts.append(at)
        at += len(line) + 1
    cursor = 0
    for header, rows in pipe_tables(text, min_lines=2):
        index = next((i for i in range(cursor, len(lines))
                      if lines[i].strip().startswith("|")
                      and pipe_cells(lines[i]) == list(header)), None)
        if index is None:
            index = cursor
        end = index
        while end < len(lines) and lines[end].strip().startswith("|"):
            end += 1
        cursor = end
        offset = starts[index] if index < len(starts) else len(text)
        grid = [list(header)] + [list(r) for r in rows]
        found.append({"kind": "pipe", "grid": grid, "stray": [],
                      "inherited": [set() for _ in grid], "at": offset,
                      "line_start": index, "line_end": end})
    return found


def parse(text: str) -> list:
    """Every table in a document's text, page by page, in printed order.

    Each is {page, kind, grid, inherited}: `grid` is rows of cells with spans
    expanded, the first row being whatever the table opens with. Nothing is
    interpreted yet -- no heading is chosen and no row is dropped.
    """
    out = []
    for number, page in enumerate(segment.split_pages(text or ""), 1):
        # Pipe tables are looked for OUTSIDE the HTML ones: a `|` inside a
        # `<td>` is cell text, not a second table.
        # Each HTML table leaves a marker line behind, so the text read after a
        # pipe table can still see where the next table starts.
        blanked = _TABLE_BLOCK.sub(
            lambda m: _NEXT_TABLE + "\n" * max(1, m.group(0).count("\n")), page)
        tables = _html_tables(page) + _pipe_tables_at(blanked)
        # Where each table sits among the page's lines, for the fix agent's
        # window (`context_window`).
        items = _page_items(blanked, tables)
        for table in tables:
            mine = [k for k, it in enumerate(items) if it["table"] is table]
            table["ctx"] = items
            table["ctx_span"] = (mine[0], mine[-1] + 1) if mine else None
        for table in sorted(tables, key=lambda t: t["at"]):
            table["page"] = number
            out.append(table)
    return out


# --------------------------------------------------------------------------
# headings and rows
# --------------------------------------------------------------------------

# A figure as a money column prints it: separators, a decimal part, brackets or
# a minus for a negative. A bare integer is NOT one -- that is a row number, a
# quantity or a year, and calling it money would make every numbered list a
# table of amounts.
_MONEY_CELL = re.compile(r"^\(?\s*-?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,4})\s*\)?$"
                         r"|^\(?\s*-?\s*\d+\.\d{1,4}\s*\)?$")
_NIL_CELL = re.compile(r"^[-–—]+$")
_DIGIT = re.compile(r"\d")

# A row that totals the rows above it. The same wording `fieldscore._TOTAL_ROW`
# anchors on, tested against every cell rather than the first: sol015's billing
# notes print the amount in words in column one and รวมเงินทั้งสิ้น in column five.
#
# Plus the VAT line, which that scorer never needed: a totals block ruled INSIDE
# the table prints its tax line as a row (`ภาษีมูลค่าเพิ่ม - % | 0.00` on sol010,
# `Vat 7% | 814.71` on sol002), and that is a line of the totals, not a charge.
# ภาษี alone is NOT here -- a land and building tax is a real charge, and sol004
# bills three of them.
_TOTAL_ROW = re.compile(r"^\s*(?:total|sub\s*-?\s*total|grand\s*total|less|"
                        r"vat(?:able)?\b|"
                        r"รวม|ยอดรวม|ยอดสุทธิ|จำนวนเงินรวม|จำนวนเงินทั้งสิ้น|"
                        r"ภาษีมูลค่าเพิ่ม|บวก|หัก|"
                        # The lines of a totals block that open with neither
                        # รวม nor ภาษี (2026-10-01, when the item table and its
                        # totals became one table): what a credit note
                        # reconciles -- the original value, the corrected value
                        # and the difference -- what a receipt says it received,
                        # and a credit note's own three money lines.
                        r"มูลค่าตาม|มูลค่าที่ถูกต้อง|มูลค่าของสินค้า|มูลค่าสินค้า|"
                        r"ผลต่าง|รับชำระทั้งสิ้น|ราคารวม|จำนวนเงินใบส่งของคืน|"
                        r"จำนวนเงินลดหนี้|จำนวนเงินใหม่|จำนวนเงินหลัง|จำนวนเงินที่รวม|"
                        r"จำนวนภาษี|ส่วนลด\s*[\d.]+\s*%)", re.I)


def _blank(cell) -> bool:
    return not str(cell or "").strip()


_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def _money(cell) -> bool:
    return bool(_MONEY_CELL.match(str(cell or "").translate(_THAI_DIGITS).strip()))


def _header_ok(header) -> bool:
    """Whether a table's first row is headings rather than a label and its value.

    `Vatable Amount | 11,638.64` and `รวม | 208,839.60` are ruled like a table
    and are a block of totals: their "heading" row carries a figure. Two named
    columns is the least a table of rows can have.
    """
    named = [c for c in header if not _blank(c)]
    return len(named) >= 2 and not any(_money(c) for c in named)


def _subheader(row, inherited, header) -> bool:
    """Whether a body row is the second line of a heading printed over two rows.

    Only the shape that can be told apart from a data row: no digit anywhere,
    and every filled cell sits under a heading that spans several columns --
    that heading's own column, or the blank ones ruled to its right. A second
    heading row that repeats every column in another language is NOT merged: it
    is indistinguishable from a row of text, and guessing would eat a line item.
    """
    filled = [i for i, c in enumerate(row) if not _blank(c) and i not in inherited]
    if not filled or any(_DIGIT.search(row[i]) for i in filled):
        return False
    for index in filled:
        if index >= len(header):
            return False
        under_blank = _blank(header[index])
        opens_span = (index + 1 < len(header) and _blank(header[index + 1])
                      and not _blank(header[index]))
        if not (under_blank or opens_span):
            return False
    # At least one of them under a blank heading, or it is just a row that
    # happens to have text in a spanning column.
    return any(_blank(header[i]) for i in filled)


def _merge_subheader(header, row, inherited=()):
    """One heading per column: the spanning heading, then what is printed under it.

    A cell the second row merely inherits from a `rowspan` above is the heading
    itself over again, not something printed under it, and adds nothing.
    """
    out, parent = [], ""
    for index, heading in enumerate(header):
        if not _blank(heading):
            parent = heading
        sub = row[index] if index < len(row) and index not in inherited else ""
        if _blank(sub):
            out.append(heading)
        else:
            out.append(((parent + " ") if parent else "") + sub.strip())
    return out


def _naming_columns(columns) -> set:
    """The columns that say WHAT a row is: a description, a reference, a product."""
    return {i for i, h in enumerate(columns)
            if column_roles(h) & {"description", "reference", "product"}}


def _row_kind(cells, columns) -> str:
    """"" for an item row, else why it is not one."""
    width = len(columns)
    filled = [c for c in cells if not _blank(c)]
    if not filled:
        return "blank"
    if any(_TOTAL_ROW.match(c) for c in filled):
        return "totals row"
    if len(filled) <= 2 and any(_AMOUNT_IN_WORDS.search(c.strip(" .")) for c in filled) \
            and not any(_money(c) for c in filled[1:]):
        # The amount spelled out in words, ruled as a row of the table's foot.
        return "amount in words"
    if all(_money(c) or _NIL_CELL.match(c.strip()) for c in filled) \
            and any(_money(c) for c in filled) and len(filled) < len(cells):
        # Figures standing under no description, no reference and no row number:
        # the unlabelled total sol004 prints under its last charge.
        return "figures with no label"
    naming = _naming_columns(columns)
    fitted = _fit(cells, width)
    if naming and fitted is not None and len(filled) < width \
            and all(_blank(fitted[i]) for i in naming) \
            and any(_money(c) for c in filled):
        # A label and its figure printed in the figure columns, under the last
        # item: a line of the totals block the table was ruled around (sol014's
        # VAT line). An item says what it is; this row says nothing in any
        # column that could.
        return "figures with no description or reference"
    lead = next((n for n, c in enumerate(cells) if not _blank(c)), 0)
    if fitted is None and len(cells) >= 4 and lead * 2 >= len(cells) \
            and len(filled) >= 2 and not _money(filled[0]) and all(_money(c) for c in filled[1:]):
        # The same line where the row does not fit its headings, so no column
        # can be consulted: the left half of the row empty, then a label, then
        # nothing but its figures.
        return "figures with no description or reference"
    if len(filled) == 1 and max(width, len(cells)) >= 3 and not _money(filled[0]):
        # One cell of text in a table of three or more columns is a section
        # heading or a note ruled as a row (sol015 p7's branch line, sol003's
        # three notes), not a charge.
        return "section heading or note"
    return ""


def _reach(cells) -> int:
    """How many columns a row actually uses: its length without trailing blanks."""
    count = len(cells)
    while count and _blank(cells[count - 1]):
        count -= 1
    return count


def _fit(cells, width: int):
    """The row at the table's width where that loses nothing, else None.

    The only change made without reading the cells: blank cells past the last
    column come off. A row that is SHORT is not padded -- where the missing cell
    belongs is exactly what is not known -- and a row that is long with text in
    the overhang is not cut.
    """
    cells = list(cells)
    while len(cells) > width and _blank(cells[-1]):
        cells.pop()
    return cells if len(cells) == width else None


def _prepare(table: dict) -> dict:
    """One parsed table with its headings settled and its rows sorted.

    Returns {page, columns, header_ok, rows, dropped, blank, notes} where every
    row is {cells, page, fits} and `dropped` keeps what was taken out of the
    item rows, with the reason.
    """
    grid, inherited = table["grid"], table["inherited"]
    header = [c.strip() for c in grid[0]]
    body = list(zip(grid[1:], inherited[1:]))
    notes = []
    while body and _header_ok(header) and _subheader(body[0][0], body[0][1], header):
        header = _merge_subheader(header, body[0][0], body[0][1])
        body = body[1:]
        notes.append("a heading printed over two rows was joined into one per column")
    # A heading that is blank and that no row reaches is not a column. An OCR
    # model that writes one `colspan` too wide leaves exactly that behind, and
    # without this every row of the table is one cell short of its headings.
    reach = max((_reach(cells) for cells, _ in body
                 if not any(_TOTAL_ROW.match(c) for c in cells if not _blank(c))),
                default=len(header))
    while len(header) > max(reach, 2) and _blank(header[-1]):
        header.pop()
    width = len(header)
    if table.get("stray"):
        notes.append("text the read put outside any cell was left out: %s"
                     % "; ".join(table["stray"][:6]))
    rows, dropped, blank = [], [], 0
    for cells, _ in body:
        kind = _row_kind(cells, header)
        if kind == "blank":
            blank += 1
            continue
        if kind:
            dropped.append({"why": kind, "cells": [c for c in cells],
                            "page": table["page"]})
            continue
        fitted = _fit(cells, width)
        rows.append({"cells": fitted if fitted is not None else list(cells),
                     "page": table["page"], "fits": fitted is not None})
    return {"page": table["page"], "kind": table["kind"], "columns": header,
            "ctx": table.get("ctx") or [], "ctx_span": table.get("ctx_span"),
            "grid": table.get("grid") or [],
            "header_ok": _header_ok(header), "rows": rows, "dropped": dropped,
            "blank": blank, "notes": notes}


def _money_columns(rows, width: int) -> set:
    """Columns whose cells are figures, judged by the cells rather than the heading."""
    found = set()
    fitting = [r["cells"] for r in rows if r["fits"]]
    for index in range(width):
        cells = [r[index] for r in fitting if index < len(r) and not _blank(r[index])]
        if cells and sum(1 for c in cells if _money(c)) * 2 >= len(cells):
            found.add(index)
    return found


def _candidate(prepared: dict):
    """How good an item table this is, or None where it is not one.

    An item table names WHAT each row is (a description or a reference) and
    carries a figure for it (an amount, a quantity, a unit price) -- by its
    headings, or for the figure by its cells. That excludes the two things most
    often ruled beside it: a strip of header fields printed as a one-row table
    (`PO No. | SO No. | Terms | Due Date`, no figure) and a totals block
    (`Vatable Amount | 11,638.64`, no headings at all).
    """
    if not prepared["rows"]:
        return None
    columns = prepared["columns"]
    roles = [column_roles(h) for h in columns]
    every = set().union(*roles) if roles else set()
    money_by_cell = _money_columns(prepared["rows"], len(columns))
    figure = bool(every & {"money", "quantity", "unit_price"}) or bool(money_by_cell)
    what = bool(every & {"description", "reference", "product"})
    if prepared["header_ok"] and figure and what:
        return (2, len(prepared["rows"]), len(columns))
    # The read lost or garbled the headings, and the rows are still a table of
    # figures. Offered only where nothing better is, and it says so.
    widths = [len(r["cells"]) for r in prepared["rows"]]
    modal = max(sorted(set(widths)), key=widths.count)
    if modal >= 3 and len(prepared["rows"]) >= 2 and any(
            sum(1 for c in r["cells"] if _money(c)) for r in prepared["rows"]):
        return (1 if prepared["header_ok"] else 0, len(prepared["rows"]), modal)
    return None


def _same_headings(a, b) -> bool:
    return ([grounding.squash(c) for c in a] == [grounding.squash(c) for c in b]
            and any(grounding.squash(c) for c in a))


# --------------------------------------------------------------------------
# master table or item list
# --------------------------------------------------------------------------

def _document_number(cell) -> bool:
    """Whether a cell reads as the number of a document: long, and mostly digits.

    `510210009577`, `IV0005887`, `RO 1885555448955`. Not a row number (too
    short), not an amount, not a date -- and not a description that happens to
    contain a figure, which is mostly words.
    """
    if cell_kind(cell) not in ("code", "number"):
        return False
    squashed = grounding.squash(cell)
    digits = sum(1 for ch in squashed if ch.isdigit())
    return len(squashed) >= 6 and digits >= 4 and digits * 2 >= len(squashed)


def _share(rows, index, test) -> bool:
    """Whether at least half the rows pass `test` in one column."""
    return bool(rows) and sum(
        1 for r in rows if index < len(r) and test(r[index])) * 2 >= len(rows)


def classify(columns, rows):
    """(is_master_table, why, reference_columns, item_columns) for one table.

    `rows` are cell lists that FIT `columns`; a row that does not line up says
    nothing reliable about which column a value is in and must not be passed.

    None -- not False -- where it cannot be told. See the module docstring for
    the rule and for why both of its tests are needed.

    **A reference column is known by its HEADING, and failing that by its
    CELLS.** The heading is the page's own statement and is asked first. The
    cells are asked only about a column whose heading says nothing this
    recognises, and only where they are document numbers on most rows -- which
    is what is left when the read garbled the one heading that mattered. sol004
    prints เลขที่ใบแจ้งหนี้ over a column of twelve-digit invoice numbers; read at
    4 MP the heading came back as เลขทะเบียนหนี้, which names nothing, over the
    same twelve digits on every row. A heading that DOES say something else --
    a tax ID, an account, a date -- is taken at its word and never overruled by
    what is under it: that would be this module deciding it knows better than
    the page. `why` says which of the two the answer rests on.
    """
    named = [(i, h) for i, h in enumerate(columns) if not _blank(h)]
    if len(named) < 2:
        return (None, "the read kept no column headings, and the headings are "
                      "what says whether a row is a document or an item", [], [])
    roles = {i: column_roles(h) for i, h in named}
    item_cols = [columns[i] for i, _ in named if roles[i] & set(_ITEM_ROLES)]
    ref_cols = [i for i, _ in named if "reference" in roles[i]]
    if ref_cols and not rows:
        return (None, "a reference column is ruled, and no row lines up with the "
                      "headings to say whether it is filled in",
                [columns[i] for i in ref_cols], item_cols)
    filled = [i for i in ref_cols
              if _share(rows, i, lambda cell: not _blank(cell))]
    by_cells = False
    if not filled:
        # No heading names another document, or the one that does is blank.
        # The cells of a column whose heading says NOTHING are the only other
        # evidence there is.
        filled = [i for i, _ in named
                  if not roles[i] and _share(rows, i, _document_number)]
        by_cells = bool(filled)
    names = [columns[i] for i in filled]
    if not filled:
        if ref_cols:
            return (False, "a reference column is ruled and left blank on most "
                           "rows, so the rows are not identified by another "
                           "document", [], item_cols)
        return (False, "no column of this table names another document", [],
                item_cols)
    how = ("carries a document number under %s, a heading that does not say "
           "what the column is -- read from the cells, not from the heading"
           if by_cells else "is identified by another document (%s)")
    if item_cols:
        return (False, ("each row " + how + ", but the table also rules %s -- the "
                        "rows are goods or services, not documents")
                % (", ".join(names), ", ".join(item_cols)), names, item_cols)
    return (True, ("each row " + how + " and the table rules no unit price and "
                   "no product code -- a list of documents, not of items")
            % ", ".join(names), names, item_cols)


# --------------------------------------------------------------------------
# the item table of one document
# --------------------------------------------------------------------------

def item_table(text: str, align: bool = True, context_share=None):
    """The item table of one document's transcript, or None where it has none.

    Returns {
      columns            the headings as printed, one per column
      rows               the item rows, each a list of cell strings. Every row is
                         exactly len(columns) long EXCEPT those in `misaligned`
      row_pages          the page each row was read off
      misaligned         indexes into `rows` of rows cut into the wrong number of
                         cells that nothing could settle. Kept as read
      realigned          indexes of rows that WERE cut wrongly and have been put
                         back under the headings -- by `realign`, or by `recut`.
                         Every character is the read's; only the cuts moved
      dropped            rows of the table that are not items -- totals, section
                         headings, notes -- each {why, cells, page}
      is_master_table    True / False / None
      master_why         the sentence behind that answer
      reference_columns  the headings that name another document
      item_columns       the headings only a list of goods rules
      headings_lost      True where the read kept no usable headings
      pages              the pages the table runs over
      tables_found       how many tables the document's text holds in all
      repairs            what was done to the table as read, in words
      source             "transcript"
    }
    """
    parsed = parse(text)
    prepared = [_prepare(t) for t in parsed]
    best, rank = None, None
    for one in prepared:
        score = _candidate(one)
        if score is not None and (rank is None or score > rank):
            best, rank = one, score
    if best is None:
        return None

    columns = list(best["columns"])
    headings_lost = rank[0] == 0
    repairs = list(best["notes"])
    # One table continued over several pages prints its headings again on each.
    # Joined where the headings are the same; a continuation whose headings the
    # read garbled stays a table of its own, which is the safe direction.
    parts = [p for p in prepared
             if p is best or (not headings_lost
                              and _same_headings(p["columns"], columns))]
    if headings_lost:
        widths = [len(r["cells"]) for r in best["rows"]]
        width = max(sorted(set(widths)), key=widths.count)
        columns = [""] * width
        for row in best["rows"]:
            fitted = _fit(row["cells"], width)
            row["fits"] = fitted is not None
            if fitted is not None:
                row["cells"] = fitted
        repairs.append("the read kept no usable headings for this table; the "
                       "columns are unnamed")
    rows, pages_of, dropped, pages = [], [], [], []
    for part in parts:
        for row in part["rows"]:
            rows.append(row)
            pages_of.append(row["page"])
        dropped.extend(part["dropped"])
        if (part["rows"] or part is best) and part["page"] not in pages:
            pages.append(part["page"])
    if len([p for p in parts if p["rows"]]) > 1:
        repairs.append("joined across pages %s, which reprint the same headings"
                       % ", ".join(str(p) for p in pages))
    kinds = {}
    for entry in dropped:
        kinds[entry["why"]] = kinds.get(entry["why"], 0) + 1
    for why, count in kinds.items():
        repairs.append("%d row%s taken out of the items: %s"
                       % (count, "" if count == 1 else "s", why))

    shown = [p for p in parts if p["rows"]] or [best]
    window = context_window(shown[0], shown[-1], context_share)
    out = {
        "columns": columns,
        "rows": [list(r["cells"]) for r in rows],
        "row_pages": pages_of,
        "misaligned": [i for i, r in enumerate(rows) if not r["fits"]],
        "dropped": dropped,
        "headings_lost": headings_lost,
        "pages": pages,
        "tables_found": len(parsed),
        # The page around the table: the lines the fix agent is asked about,
        # CONTEXT_SHARE of the page each side. Working state -- `public`
        # leaves both out.
        "context_before": window[0],
        "context_page": shown[0]["page"],
        "tail": window[1],
        "tail_page": shown[-1]["page"],
        "repairs": repairs,
        "source": "transcript",
        # The other tables the read made of this document, AFTER this one: the
        # pieces a cut table leaves -- its totals ruled as a table of their own,
        # its rows continued under headings the read garbled. `concat` joins the
        # ones it is told to. Working state, like the window: `public` leaves it.
        "_others": [p for p in prepared[prepared.index(best) + 1:]
                    if not any(p is q for q in parts)],
        "_shown": shown,
        "_tail_from": shown[-1],
        "_share": context_share,
    }
    # `align=False` stops after the structural repairs, so a caller that has a
    # model look at the table first (`app._table_fix_agent`) sees the rows as the
    # read cut them, and `realign` runs afterwards as the fallback.
    return reclassify(realign(out) if align else out)


def reclassify(table: dict) -> dict:
    """Set `is_master_table` and its evidence from the table as it now stands.

    Called again after a repair: a row that has been re-cut to fit is a row that
    can now say which column its reference is in.
    """
    bad = set(table.get("misaligned") or [])
    fitting = [r for i, r in enumerate(table["rows"]) if i not in bad]
    master, why, refs, items = classify(table["columns"], fitting)
    table["is_master_table"] = master
    table["master_why"] = why
    table["reference_columns"] = refs
    table["item_columns"] = items
    return table


# --------------------------------------------------------------------------
# re-cutting a row that does not fit its headings
# --------------------------------------------------------------------------
#
# Two hands may do it, in this order, and both leave every character alone:
#
#   1. `realign` -- Python, by what KIND of thing each cell is. An amount belongs
#      under a money heading and nowhere else, which pins most of a row on its
#      own; what is left is usually one pair of cells and one column for them.
#      Taken only where one cut is clearly better than every other.
#   2. `recut` -- the model, for the rows the first could not settle, asked
#      which column each cell is under. `app.py` makes the request.
#
# The order is the measurement, not a preference. On sol007 -- five headings,
# six cells, the reference read as `RO` and `1885555448955` -- `gemma4:e4b` put
# the number with the DESCRIPTION, six rows out of six, twice running, in a
# reply that was exactly the right shape; `qwen3.5:9b` put it with the reference,
# six out of six. So the model's answer is only as good as the model, and the
# shape check cannot tell a right cut from a wrong one. A thirteen-digit number
# is not a description, and telling those apart is not a judgement: it is a
# property of the cell. So the part that can be decided from the cells is
# decided from the cells, and the model is left the rows they do not decide.

_DATE_CELL = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}$"
                        r"|^\d{1,2}[-/. ][A-Za-z]{3,9}[-/. ]\d{2,4}$")

# How many ways of cutting one row are worth comparing. A row that is off by one
# cell has a handful; past this the row is not a bad cut, it is not a row.
_MAX_CUTS = 400
# How much better the best cut has to be than the next before it is taken.
_CUT_MARGIN = 1.0


def cell_kind(text: str) -> str:
    """What sort of thing a cell holds: money, nil, date, number, code, text or "".

    Deliberately coarse. It is used to compare a cell with the column it might
    belong under, and the only distinction that has to be sharp is money against
    everything else -- an amount is the one kind of cell whose column is never
    in doubt.

    A dash is `nil`, its own kind and evidence of nothing: a page prints one
    wherever a figure would go and there is none, under any heading at all.
    Counting it as an amount made a satang cell printed `-` argue that its
    column was money.
    """
    text = str(text or "").translate(_THAI_DIGITS).strip()
    if not text:
        return ""
    if _NIL_CELL.match(text):
        return "nil"
    if _money(text):
        return "money"
    if _DATE_CELL.match(text):
        return "date"
    digits = sum(1 for ch in text if ch.isdigit())
    letters = sum(1 for ch in text if ch.isalpha())
    if digits and not letters:
        return "number"
    if digits and letters:
        # A code is mostly digits with a prefix on it; a description that
        # mentions a number is mostly words.
        return "code" if digits * 10 >= (digits + letters) * 4 else "text"
    return "text" if letters else ""


def _heading_kind(heading: str) -> str:
    """What a heading says its column holds, as a `cell_kind`, or "" if nothing."""
    roles = column_roles(heading)
    return ("code" if roles & {"reference", "product"}
            else "text" if "description" in roles
            else "money" if roles & {"money", "unit_price"}
            else "number" if "quantity" in roles
            else "")


def _expected_kinds(columns, fitting) -> list:
    """What each column holds: what its HEADING says, else what its cells are.

    The heading first, and that order was forced by a real read. A row can fit
    its headings by width and still be one column out -- sol007 at 2 MP came
    back with `colspan`s the page does not rule, and its first row was exactly
    the table's width with the description under Amount. Learning the columns
    from "the rows that fit" learnt them from that row. A heading cannot be
    shifted; the rows are consulted only for a column whose heading says nothing.
    """
    out = []
    for index, heading in enumerate(columns):
        kind = _heading_kind(heading)
        if not kind:
            seen = [cell_kind(r[index]) for r in fitting if index < len(r)]
            seen = [k for k in seen if k and k != "nil"]
            # The commonest, ties to the first seen. NOT `max(set(seen))`: the
            # order of a set of strings changes from one process to the next,
            # and a repair that depends on it is a repair that differs between
            # two runs of one transcript.
            kind = max(dict.fromkeys(seen), key=seen.count) if seen else ""
        out.append(kind)
    return out


def _violates(cells, columns) -> bool:
    """Whether a row that fits by WIDTH has a cell under a heading it cannot be.

    Two shapes only, the two that are never a matter of taste: words under a
    heading for money, and an amount under a heading for a description, a
    reference or a product. A bare number under a money heading is fine -- half
    the amounts in a handwritten receipt are written without satang -- and so is
    anything under a heading that says nothing.
    """
    for cell, heading in zip(cells, columns):
        if _blank(cell):
            continue
        expected, kind = _heading_kind(heading), cell_kind(cell)
        if expected == "money" and kind in ("text", "code"):
            return True
        if kind == "money" and expected in ("text", "code"):
            return True
    return False


def _fit_score(group, expected: str) -> float:
    """How well the cells cut into one column suit what that column holds."""
    if not group:
        return 0.0
    kinds = [cell_kind(c) for c in group]
    # Two amounts are never one value, whatever column they are offered to.
    score = -4.0 if kinds.count("money") > 1 else 0.0
    # Each join is a claim, so a cut that needs fewer is preferred between two
    # that are otherwise as good.
    score -= 0.25 * (len(group) - 1)
    kind = cell_kind(" ".join(group))
    if kind == "nil" or not expected:
        return score                # a dash, or a column that says nothing
    if expected == "money":
        return score + (2.0 if kind == "money" else 0.0 if kind == "number" else -3.0)
    if kind == "money":
        return score - 3.0          # an amount under a heading that is not money
    if expected == kind or (expected == "code" and kind == "number"):
        return score + 1.0
    return score - 1.0


def _cuts(count: int, width: int):
    """Every way to put `count` cells, in order, under `width` columns with the
    fewest changes: only joins where there are too many, only blanks where there
    are too few. Yields one column number (0-based) per cell."""
    import itertools
    if count >= width:
        # `count - width` joins: choose which cell boundaries are NOT cuts.
        for kept in itertools.combinations(range(1, count), width - 1):
            column, out = 0, []
            for cell in range(count):
                if cell in kept:
                    column += 1
                out.append(column)
            yield out
    else:
        for used in itertools.combinations(range(width), count):
            yield list(used)


def _place(cells, expected):
    """The one clearly best way to put a row's filled cells under the columns.

    Returns the row, or None where no cut is clearly better than the next --
    a row the cells do not settle is left alone rather than guessed at.
    """
    import math
    width, count = len(expected), len(cells)
    if not cells or width < 2:
        return None
    ways = (math.comb(count - 1, width - 1) if count >= width
            else math.comb(width, count))
    if not 0 < ways <= _MAX_CUTS:
        return None
    ranked = []
    for cut in _cuts(count, width):
        groups = [[c for c, col in zip(cells, cut) if col == column]
                  for column in range(width)]
        ranked.append((sum(_fit_score(g, e) for g, e in zip(groups, expected)), cut))
    ranked.sort(key=lambda pair: -pair[0])
    clear = len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= _CUT_MARGIN
    # A lone candidate still has to make sense: a row cut the only way it can
    # be cut, with an amount under a description, is not a repair.
    if not clear or ranked[0][0] < 0:
        return None
    row = [""] * width
    for cell, column in zip(cells, ranked[0][1]):
        row[column] = (row[column] + " " + cell.strip()).strip()
    return row


def _settle(rows, misfits, columns):
    """One reading of the grid, carried through: (rows, unsettled, moved).

    `unsettled` are rows still cut wrongly. `moved` are rows that were and have
    been put back. A row that fits by width and sits wrongly by KIND is tried
    too, and counted `doubtful` while no clear placement exists for it.
    `agreement` is how well the placed cells suit their headings, summed.
    """
    sound = [r for i, r in enumerate(rows)
             if i not in misfits and not _violates(r, columns)]
    expected = _expected_kinds(columns, sound)
    out, unsettled, moved, doubtful = [], [], [], 0
    for index, row in enumerate(rows):
        misfit = index in misfits
        if not misfit and not _violates(row, columns):
            out.append(list(row))
            continue
        placed = _place(filled_cells(row), expected)
        if placed is not None and not _violates(placed, columns):
            out.append(placed)
            if placed != list(row):
                moved.append(index)
            continue
        out.append(list(row))
        if misfit:
            unsettled.append(index)
        else:
            doubtful += 1
    # How well the cells agree with the headings under this reading, summed over
    # the rows that ended up placed. It is what two readings are compared on.
    agreement = sum(_fit_score([cell] if not _blank(cell) else [], kind)
                    for index, row in enumerate(out) if index not in unsettled
                    for cell, kind in zip(row, expected))
    return out, unsettled, moved, doubtful, agreement


def realign(table: dict) -> dict:
    """Re-cut the rows Python can settle on its own. See above.

    Two readings of the grid are tried and the one the CELLS agree with is kept:

      * the grid as read, every column the header rules, named or not;
      * the grid with its UNNAMED columns taken out, each row being its filled
        cells in order. An OCR model that invents a `colspan` leaves a column
        with no heading behind, and every row then sits one cell out of true.

    The second is adopted only where it settles at least as many rows AND the
    cells agree with the headings clearly better under it. A column with no
    heading accepts anything, so the grid as read can "settle" every row of a
    shifted table without complaint -- which is why the comparison is on
    agreement and not on whether a placement was found. A real column that
    happens to have no heading (sol003 rules the satang beside the baht with
    nothing over it) ties, and a tie keeps the grid as read.

    A row is re-cut only where ONE cut scores clearly above every other; a row
    with two cuts as good as each other is a row the cells do not settle, and it
    is left for `recut` -- or left flagged.
    """
    columns = list(table["columns"])
    if table.get("headings_lost") or len(columns) < 2 or not table["rows"]:
        return table
    misfits = set(table.get("misaligned") or [])
    rows, unsettled, moved, doubtful, agreement = _settle(
        table["rows"], misfits, columns)
    note = ""
    named = [i for i, h in enumerate(columns) if not _blank(h)]
    if 2 <= len(named) < len(columns):
        compact = [columns[i] for i in named]
        packed = [filled_cells(r) for r in table["rows"]]
        odd = {i for i, r in enumerate(packed) if len(r) != len(compact)}
        c_rows, c_unsettled, c_moved, c_doubtful, c_agreement = _settle(
            packed, odd, compact)
        if (len(c_unsettled) + c_doubtful <= len(unsettled) + doubtful
                and c_agreement >= agreement + _CUT_MARGIN):
            # A row counts as put back where it was wrong in the grid as read
            # -- the wrong width, or a cell under a heading it cannot be -- or
            # where two of its cells were joined on the way.
            c_moved = sorted(
                i for i, r in enumerate(c_rows)
                if i not in c_unsettled
                and (i in misfits or _violates(table["rows"][i], columns)
                     or filled_cells(r) != packed[i]))
            dropped = len(columns) - len(compact)
            columns, rows = compact, c_rows
            unsettled, moved = c_unsettled, c_moved
            note = ("%d column%s the read ruled with no heading %s taken out: the "
                    "cells do not line up under %s"
                    % (dropped, "" if dropped == 1 else "s",
                       "was" if dropped == 1 else "were",
                       "it" if dropped == 1 else "them"))
    table["columns"] = columns
    table["rows"] = rows
    table["misaligned"] = sorted(unsettled)
    if note:
        table.setdefault("repairs", []).append(note)
    if moved:
        table["realigned"] = sorted(set(table.get("realigned") or []) | set(moved))
        table.setdefault("repairs", []).append(
            "%d row%s put back under the headings by what each cell is -- an "
            "amount under a money heading, a number under a reference -- with "
            "no character changed" % (len(moved), "" if len(moved) == 1 else "s"))
    return table


def content(cells) -> str:
    """What a row SAYS, with the cell boundaries taken away."""
    return "".join(grounding.squash(c) for c in cells)


def filled_cells(row) -> list:
    """A misaligned row's cells that hold something, in the order they were read.

    The blanks are left out because their POSITIONS are exactly what cannot be
    trusted in a row that does not fit: a blank says a column is empty only
    where the cells around it are in the right columns, and these are not.
    """
    return [c for c in row if not _blank(c)]


def _cut_row(cells, proposal, width: int) -> list:
    """A row of `width` cells from filled cells and one column number each."""
    row = [""] * width
    for cell, column in zip(cells, proposal):
        row[column - 1] = (row[column - 1] + " " + cell.strip()).strip()
    return row


def _cut_score(cells, proposal, expected) -> float:
    """How well a cut -- one 1-based column per cell -- suits the columns."""
    groups = [[c for c, col in zip(cells, proposal) if col == n + 1]
              for n in range(len(expected))]
    return sum(_fit_score(g, e) for g, e in zip(groups, expected))


def _best_cut(cells, expected):
    """(score, cut) of the best way the cells can be cut, or None where there are
    too many ways to compare. The cut is 1-based, the shape `recut` takes."""
    import math
    width, count = len(expected), len(cells)
    if not cells or width < 2:
        return None
    ways = (math.comb(count - 1, width - 1) if count >= width
            else math.comb(width, count))
    if not 0 < ways <= _MAX_CUTS:
        return None
    best = None
    for cut in _cuts(count, width):
        cut = [c + 1 for c in cut]
        score = _cut_score(cells, cut, expected)
        if best is None or score > best[0]:
            best = (score, cut)
    return best


def check_cut(cells, proposal, columns, expected) -> str:
    """Why a proposed cut is refused, or "" where the code lets it stand.

    This is the CODE checking the model, and it can only refuse -- it never
    substitutes a cut of its own here. Two refusals, both properties of the
    cells rather than judgements about them:

      * the cut puts words under a money heading or an amount under a
        description, a reference or a product (`_violates`);
      * the cells fit another cut CLEARLY better -- by `_CUT_MARGIN`, the margin
        `realign` itself needs before it moves a row. That is the sol007 case:
        `gemma4:e4b` joined a thirteen-digit reference to the DESCRIPTION, and a
        number is not a description.

    Where the cells do not settle it -- two cuts as good as each other -- the
    model's answer stands. That is exactly the question it is asked for.
    """
    row = _cut_row(cells, proposal, len(columns))
    if _violates(row, columns):
        return "it would put an amount under words, or words under an amount"
    best = _best_cut(cells, expected)
    if best is not None and best[1] != list(proposal) \
            and best[0] - _cut_score(cells, proposal, expected) >= _CUT_MARGIN:
        return ("the cells fit another cut clearly better (%s)"
                % " | ".join(" ".join(g) for g in
                             ([c for c, col in zip(cells, best[1]) if col == n + 1]
                              for n in range(len(columns))) if g))
    return ""


def recut(table: dict, answers, check: bool = False) -> dict:
    """Put the misaligned rows under their headings, from a column per cell.

    `answers` is one entry per row in `table["misaligned"]`, in that order, and
    each entry is a list of COLUMN NUMBERS (1-based) -- one for every filled cell
    of that row, as `filled_cells` lists them. An answer is taken only where

      * it has exactly one number per filled cell,
      * every number names a column the table has, and
      * **the numbers never go down along the row.**

    Python then does the cutting: cells given the same column are joined with a
    space, and a column no cell was given is empty. So the answer decides where
    the boundaries fall and nothing else -- **no character of the row can change,
    because no character of the answer is ever copied into the table.** That is
    what makes it safe to take the columns from a model, and it is a stronger
    guarantee than checking a rewritten row afterwards: there is nothing to
    check. The third test is what stops an answer reordering a row.

    `check` adds `check_cut`: a well-formed answer the CELLS contradict is
    refused too, and the reason is kept (`recut["refusals"]`). Used where the
    model decides which rows need fixing at all (`app._table_fix_agent`), so a
    row it volunteered can be put back only where the cells do not argue.

    What it cannot do is SPLIT a cell that holds two values. A split is a claim
    about where inside one cell a boundary falls, and that is a different
    question with a different failure -- a value cut in half reads as two
    plausible values. Such a row keeps its flag.

    Returns the table, with `recut` counting {asked, taken, refused}.
    """
    columns = table["columns"]
    width = len(columns)
    wanted = list(table.get("misaligned") or [])
    expected = None
    if check:
        skip = set(wanted)
        sound = [r for i, r in enumerate(table["rows"])
                 if i not in skip and len(r) == width and not _violates(r, columns)]
        expected = _expected_kinds(columns, sound)
    taken, still, refusals = 0, [], []
    answers = list(answers or [])
    for position, index in enumerate(wanted):
        proposal = answers[position] if position < len(answers) else None
        cells = filled_cells(table["rows"][index])
        ok = (isinstance(proposal, (list, tuple)) and cells
              and len(proposal) == len(cells)
              and all(isinstance(n, int) and not isinstance(n, bool)
                      and 1 <= n <= width for n in proposal)
              and all(a <= b for a, b in zip(proposal, proposal[1:])))
        if not ok:
            still.append(index)
            if check:
                refusals.append({"row": index, "why": "no usable answer for "
                                 "every cell of this row"})
            continue
        if check:
            why = check_cut(cells, list(proposal), columns, expected)
            if why:
                still.append(index)
                refusals.append({"row": index, "why": why})
                continue
        row = _cut_row(cells, proposal, width)
        if row != list(table["rows"][index]):
            table["rows"][index] = row
            table["realigned"] = sorted(set(table.get("realigned") or []) | {index})
        taken += 1
    table["misaligned"] = still
    table["recut"] = {"asked": len(wanted), "taken": taken,
                      "refused": len(wanted) - taken}
    if check:
        table["recut"]["refusals"] = refusals
    moved = len([i for i in wanted if i in set(table.get("realigned") or [])
                 and i not in still])
    if moved:
        table["source"] = "transcript, re-cut by the model"
        table.setdefault("repairs", []).append(
            "%d row%s put back under the headings: the model said which column "
            "each cell belongs to, and no character was changed"
            % (moved, "" if moved == 1 else "s"))
    return reclassify(table)


# A column the identify agent names as the reference must not be one whose
# heading says it holds something else. `reference` itself overrides: a heading
# like "Invoice No. / Date" names both.
_NOT_A_REFERENCE = {"description", "money", "quantity", "unit_price", "date",
                    "other_number", "product"}


def reference_column_ok(table: dict, index: int) -> str:
    """Why column `index` (0-based) cannot be the reference column, or "".

    The evidence the code demands before a model's `is_master_table: true` is
    taken: the column exists, its heading does not say it is something else, and
    it is FILLED on most rows that line up -- a master table's rows are each
    identified by another document, so a reference column left blank is not one.
    Where the heading says nothing, the cells have to be document numbers, the
    same reading `classify` falls back to.
    """
    columns = table.get("columns") or []
    if not isinstance(index, int) or isinstance(index, bool) \
            or not 0 <= index < len(columns):
        return "it names no column the table has"
    roles = column_roles(columns[index])
    if roles & _NOT_A_REFERENCE and "reference" not in roles:
        return ("its heading (%s) says the column holds something else"
                % (columns[index] or "none"))
    bad = set(table.get("misaligned") or [])
    rows = [r for i, r in enumerate(table["rows"]) if i not in bad]
    if not _share(rows, index, lambda cell: not _blank(cell)):
        return "that column is blank on most rows"
    if not roles and not _share(rows, index, _document_number):
        return ("that column has no heading saying what it is, and its cells are "
                "not document numbers")
    return ""


def _locate(cell: str, quote: str):
    """(start, end) of `quote` inside `cell`, or None.

    Exact first. Failing that, a match that ignores how whitespace was written --
    a model copying a cell reliably collapses a line break or doubles a space --
    mapped back onto the cell's own characters, so what is removed is always the
    CELL's text, never the quote's. Nothing looser: a quote that differs in any
    visible character is not the text in the cell, and is not found.
    """
    # The rows are shown as `1: cell | 2: cell`, and a quote copied off that line
    # can carry the separator with it.
    cell, quote = cell or "", (quote or "").strip().strip("|").strip()
    if not quote:
        return None
    at = cell.find(quote)
    if at >= 0:
        return at, at + len(quote)
    keep = [i for i, ch in enumerate(cell) if not ch.isspace()]
    squeezed = "".join(cell[i] for i in keep)
    target = "".join(ch for ch in quote if not ch.isspace())
    at = squeezed.find(target) if target else -1
    if at < 0:
        return None
    return keep[at], keep[at + len(target) - 1] + 1


def _matches_heading(text: str, heading: str) -> bool:
    """Whether a value is the KIND of thing its heading asks for, in the three
    kinds that are never a stamp's words: an amount, a date, a document number."""
    expected, kind = _heading_kind(heading), cell_kind(text)
    if "date" in column_roles(heading) and kind == "date":
        return True
    if expected == "money" and kind in ("money", "number"):
        return True
    return expected == "code" and _document_number(text)


def _has_words(cell) -> bool:
    return cell_kind(cell) in ("text", "code")


def removal_refused(cell: str, part: str, heading: str) -> str:
    """Why taking `part` out of `cell` under `heading` is refused, or "".

    The code checking the fix agent, and it can only say no. The refusals are
    properties of the text rather than judgements about it:

      * **the part holds an amount, a date or a document number, under ANY
        heading** -- as a whole or as one of its words. A stamp's words, a QR
        code's description and a note are words; a figure in a table is the
        table's. Under any heading and not only its own, because the rows stray
        text gets into are exactly the rows it shifts: a cell inserted after the
        first pushes every amount one column over, under a heading that is not
        money, and gemma4:e4b then took the amount out as stray. And "one of its
        words" because the same model, on real tables with nothing planted,
        took `ใบสั่งซื้อเลขที่ :2260007911.Rev.0` -- a purchase-order number
        inside a description -- out of all four of sol022's documents (both
        measured 2026-10-01). A date stamp written as a bare date is kept by
        this too; that is the cheaper error;
      * a whole cell that is the kind of thing its heading asks for.

    Everything else is the agent's to decide: whether `RECEIVED` at the end of a
    description belongs there is a reading of the page, and the reason the
    agent is asked at all. `remove_text` adds one more refusal that needs the
    whole row to see.
    """
    part = (part or "").strip()
    if not part:
        return "nothing to take out"
    # Too little to be a stamp's words or a description of anything: a bare
    # number, a dash, a letter or two. Each is how a table writes a value --
    # sol003's satang column is `60` and `-`, and gemma4:e4b took both out of a
    # clean table, along with a lone `W`.
    if len(grounding.squash(part)) < 3:
        return "it is too short to be anything but part of a value"
    if cell_kind(part) in ("number", "nil"):
        return "it is a number or a dash, which is how a table writes a value"
    for piece in [part] + part.split():
        piece = piece.strip(" ,;:()[]")
        kind = cell_kind(piece)
        if kind == "money":
            return "it holds an amount"
        if kind == "date":
            return "it holds a date"
        if _document_number(piece):
            return "it holds a document number"
    if part == (cell or "").strip() and _matches_heading(part, heading):
        return "the whole cell is the kind of value its heading asks for"
    return ""


def remove_text(table: dict, entries) -> dict:
    """Take the text the fix agent named out of the table, into plain text.

    `entries` are `{row, column, text}`, 1-based, as the agent answered. Each is
    taken only where the text is FOUND in that cell (`_locate`) and
    `removal_refused` lets it go; the characters removed are the cell's own.

    **One refusal needs the whole row**: emptying a cell is refused where it
    would leave the row's figures with no words saying what they are for. A row
    of items always says what each is; a row that is a stamp from end to end
    has no figures to orphan and may go. Measured: gemma4:e4b took the whole
    description out of sol013's and sol002's only row, on clean tables.

    A row left with nothing in it leaves `rows` altogether. Everything removed
    is kept, as plain text: `outside_text`, one entry per piece with where it
    was read from, and a line in `repairs`. Text that did not belong in the
    table is still text the page printed, and dropping it silently is the one
    thing this must not do.

    Returns the table with `table["removal"]` = {removed, refusals, rows_emptied}.
    """
    rows, columns = table["rows"], table["columns"]
    refusals, wanted = [], {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            refusals.append({"entry": {"text": str(entry)[:200]},
                             "why": "not a row, column and text"})
            continue
        r, c, quote = entry.get("row"), entry.get("column"), entry.get("text")
        r = r if isinstance(r, int) and not isinstance(r, bool) else None
        c = c if isinstance(c, int) and not isinstance(c, bool) else None
        if r is None or c is None or not 1 <= r <= len(rows) \
                or not 1 <= c <= len(rows[r - 1]) or not isinstance(quote, str):
            refusals.append({"entry": entry, "why": "it names no cell the table has"})
            continue
        cell = rows[r - 1][c - 1] or ""
        span = _locate(cell, quote)
        if span is None:
            refusals.append({"entry": entry, "why": "that text is not in that cell"})
            continue
        part = cell[span[0]:span[1]]
        heading = columns[c - 1] if c - 1 < len(columns) else ""
        why = removal_refused(cell, part, heading)
        if why:
            refusals.append({"entry": entry, "why": why})
            continue
        wanted.setdefault(r - 1, {}).setdefault(c - 1, (entry, span))

    def cut(cell, span):
        return _SPACE.sub(" ", cell[:span[0]] + " " + cell[span[1]:]).strip()

    removed = []
    for r, by_column in sorted(wanted.items()):
        row = rows[r]
        after = [cut(cell, by_column[c][1]) if c in by_column else cell
                 for c, cell in enumerate(row)]
        left = [c for c in after if not _blank(c)]
        if left and not any(_has_words(c) for c in left):
            # It would orphan the figures. Keep whatever empties a cell; a part
            # taken from inside a cell that keeps its words may still go.
            for c in [c for c in by_column if _blank(after[c])]:
                refusals.append({"entry": by_column.pop(c)[0],
                                 "why": "it would leave the row's figures with "
                                        "nothing saying what they are for"})
        for c, (entry, span) in sorted(by_column.items()):
            removed.append({"text": row[c][span[0]:span[1]].strip(), "row": r,
                            "column": c,
                            "heading": columns[c] if c < len(columns) else "",
                            "page": (table.get("row_pages") or [None] * len(rows))[r]})
            row[c] = cut(row[c], span)
    # A row with nothing left in it was stray text from end to end. Out of the
    # rows, and every index that points into them moves with it.
    touched = {e["row"] for e in removed}
    empty = {i for i in touched if all(_blank(c) for c in rows[i])}
    if empty:
        keep = [i for i in range(len(rows)) if i not in empty]
        moved = {old: new for new, old in enumerate(keep)}
        table["rows"] = [rows[i] for i in keep]
        if table.get("row_pages"):
            table["row_pages"] = [table["row_pages"][i] for i in keep]
        for key in ("misaligned", "realigned"):
            if table.get(key):
                table[key] = [moved[i] for i in table[key] if i in moved]
    if removed:
        table.setdefault("outside_text", []).extend(removed)
        table.setdefault("repairs", []).append(
            "%d piece%s of text that did not belong in the table moved out of it "
            "as plain text%s" % (len(removed), "" if len(removed) == 1 else "s",
                                 "; %d row%s left empty and taken out"
                                 % (len(empty), "" if len(empty) == 1 else "s")
                                 if empty else ""))
    table["removal"] = {"removed": removed, "refusals": refusals,
                        "rows_emptied": sorted(empty)}
    return table


_TOTAL_WORD = re.compile(r"total|รวม", re.I)
_TOTAL_LABEL_CHARS = 32


def totals_line(tail):
    """The index of the first line after a table that is its totals line, or None.

    Total wording -- at the start the way a totals row opens, or anywhere, since
    sol011's reads write `ราคารวมทั้งสิ้น.../TOTAL (BEFORE VAT) 196,612.52` -- AND
    an amount. Found by the code so the fix agent can be TOLD where the frame
    ends: everything read between the table's last row and this line came off
    the empty space inside the frame, which on a printed form holds nothing the
    form printed. That is what makes sol001's pasted slip findable at all -- its
    nine lines read like printed payment instructions, and gemma4:e4b left every
    one of them in place until it was told where they sit.
    """
    for index, line in enumerate(tail or []):
        if (_TOTAL_ROW.match(line) or _TOTAL_WORD.search(line)) and any(
                cell_kind(p.strip(" ,;:()[]")) == "money" for p in line.split()):
            return index
        # A totals LABEL on a line of its own: a fresh typhoon read of sol001 at
        # 4 MP printed `รวมเงิน / Total Amount` alone and its four figures in a
        # separate table further down. Short, so a sentence that opens with รวม
        # is not taken for it.
        if _TOTAL_ROW.match(line) and len(grounding.squash(line)) <= _TOTAL_LABEL_CHARS:
            return index
    return None


# Two kinds of line the form itself prints after a table, which gemma4:e4b
# called stray on the clean fixtures (2026-10-01): the amount spelled out in Thai
# words (sol009, sol015, sol021) and a note under its own label (sol006, sol023).
# A stamp, a barcode or a handwritten mark is neither.
_AMOUNT_IN_WORDS = re.compile(r"(?:บาท|สตางค์|ถ้วน)\s*\)?\s*\.?\s*$"
                              r"|baht\s+only\s*\)?\s*\.?\s*$",
                              re.I)
_NOTE_LABEL = re.compile(r"^\s*(?:หมายเหตุ|remarks?\b|notes?\b)", re.I)


def _line_refs(value, before: int, after: int) -> list:
    """The valid line labels in a reply's list, as (side, 0-based index).

    `B3` is the third line shown before the table, `L3` -- or a bare 3, which is
    what a reply gave while only the lines after were shown -- the third after.
    Anything else, a bool among them, is not a label.
    """
    out = set()
    for n in value if isinstance(value, list) else []:
        side = "L"
        if isinstance(n, str):
            label = n.strip().upper()
            if label[:1] in ("B", "L"):
                side, label = label[0], label[1:]
            if not label.isdigit():
                continue
            n = int(label)
        if isinstance(n, int) and not isinstance(n, bool) \
                and 1 <= n <= (before if side == "B" else after):
            out.add((side, n - 1))
    return sorted(out)


def sort_tail(table: dict, stray, own) -> dict:
    """File the lines printed around the table, as the fix agent labelled them.

    The lines shown before the table (`context_before`, labelled B) are filed by
    the same rules as the lines after it (`tail`, labelled L); each filed line
    says which side it was read on.

    `stray` lines -- a stamp, a barcode, handwriting stuck in the table's frame --
    go to `outside_text`, as plain text, with the line they were read on.
    `own` lines -- the table's totals, a note -- go to `footer`: they belong to
    the table, and the read had only put them outside it.

    The code refuses two things: a line in both lists (the answer contradicts
    itself, so neither is taken), and calling a line stray that holds an amount
    (a line of figures inside the frame is the table's totals, not a stamp).
    Lines in neither list are ordinary page text after the table and are left
    alone. Nothing here changes the transcript; it says what the lines ARE.
    """
    lines = {"B": list(table.get("context_before") or []),
             "L": list(table.get("tail") or [])}
    page = {"B": table.get("context_page"), "L": table.get("tail_page")}
    where = {"B": "before the table", "L": "after the table"}
    stray_i = _line_refs(stray, len(lines["B"]), len(lines["L"]))
    own_i = _line_refs(own, len(lines["B"]), len(lines["L"]))
    both = set(stray_i) & set(own_i)
    refusals, moved, footer = [], [], []
    for side, i in stray_i:
        text = lines[side][i]
        entry = {"line": i, "side": side, "text": text}
        if (side, i) in both:
            refusals.append({**entry, "why": "named both stray and part of the table"})
            continue
        # The lines before the table are above its frame. They are shown so a
        # heading or a row the read cut off can be named the table's own; on
        # sol007 gemma4:e4b called the whole issuer block above the table stray.
        if side == "B":
            refusals.append({**entry, "why": "it is above the table, outside its "
                                             "frame -- page text, not a stamp in it"})
            continue
        if any(cell_kind(p.strip(" ,;:()[]")) == "money"
               for p in [text] + text.split()):
            refusals.append({**entry, "why": "it holds an amount, which is the table's"})
            continue
        if _AMOUNT_IN_WORDS.search(text):
            refusals.append({**entry, "why": "it spells out an amount in words, which "
                                             "the form prints"})
            continue
        if _NOTE_LABEL.match(text):
            refusals.append({**entry, "why": "it opens with a printed note label"})
            continue
        moved.append({"text": text, "line": i, "side": side, "page": page[side],
                      "where": where[side]})
    for side, i in own_i:
        if (side, i) not in both:
            footer.append({"text": lines[side][i], "line": i, "side": side,
                           "page": page[side], "where": where[side]})
    if moved:
        table.setdefault("outside_text", []).extend(moved)
        table.setdefault("repairs", []).append(
            "%d line%s printed with the table but not part of it (a stamp, a barcode, "
            "handwriting) kept as plain text" % (len(moved), "" if len(moved) == 1 else "s"))
    if footer:
        table["footer"] = footer
        table.setdefault("repairs", []).append(
            "%d line%s the read put outside the table belong to it -- its headings, "
            "totals or notes -- and are kept with it"
            % (len(footer), "" if len(footer) == 1 else "s"))
    # `stray` and `own` index the lines AFTER the table, as they always have;
    # the lines before it are counted on their own.
    table["tail_sorted"] = {
        "stray": [m["line"] for m in moved if m["side"] == "L"],
        "own": [f["line"] for f in footer if f["side"] == "L"],
        "stray_before": [m["line"] for m in moved if m["side"] == "B"],
        "own_before": [f["line"] for f in footer if f["side"] == "B"],
        "refusals": refusals}
    return table


# --------------------------------------------------------------------------
# concat: the read cut one table into several
# --------------------------------------------------------------------------

# How much of each other table the concat question shows: its first rows say
# what it is, and a totals block is short.
CONCAT_SHOW_ROWS = 8


def concat_parts(table: dict) -> list:
    """The other tables after this one, numbered from 1, as the question shows them."""
    out = []
    for number, part in enumerate(table.get("_others") or [], 1):
        rows = [r for r in part.get("grid") or [] if any(not _blank(c) for c in r)]
        out.append({"number": number, "page": part.get("page"),
                    "rows": [[c.strip() for c in r] for r in rows]})
    return out


def _order(part) -> tuple:
    """Where a part sits in the document: page, then line."""
    span = part.get("ctx_span") or (0, 0)
    return (part.get("page") or 0, span[0])


def _tail_after(source: dict, skip, share=None) -> list:
    """The lines after `source` on its page, `share` of the page, stepping over
    the lines of the parts in `skip` -- the same count `context_window` takes."""
    items = source.get("ctx") or []
    span = source.get("ctx_span")
    if not span:
        return []
    share = CONTEXT_SHARE if share is None else share
    size = max(CONTEXT_MIN_LINES, math.ceil(round(share * len(items), 6)))
    skipped = set()
    for part in skip:
        if part.get("ctx") is items and part.get("ctx_span"):
            skipped.update(range(*part["ctx_span"]))
    out, index = [], span[1]
    while index < len(items) and len(out) < size:
        if index not in skipped:
            out.append(items[index]["text"])
        index += 1
    return out


def sum_parts(table: dict) -> list:
    """The other tables that hold one of this table's COLUMN TOTALS.

    A figure equal to the sum of a money column's cells is that column's total
    -- arithmetic, not a judgement -- so the piece it sits in is this table's
    totals block, cut off by the read. sol007: the read ruled 711,900.73 /
    49,833.05 / 761,733.78 as a table of their own, and the first is exactly the
    Amount column. Numbers as `concat_parts` numbers them.
    """
    # Counted by position from the RIGHT of each row: the money columns are
    # the last ones, and a row the read cut into the wrong number of cells
    # (sol007's reference split in two) still has its figures at its right end.
    by_offset = {}
    for row in table.get("rows") or []:
        values = [verify.parse_amount(c) for c in row if _money(c)]
        values = [v for v in values if v is not None]
        for offset, value in enumerate(reversed(values)):
            by_offset.setdefault(offset, []).append(value)
    sums = [round(sum(v), 2) for v in by_offset.values() if len(v) >= 2]
    if not sums:
        return []
    out = []
    for number, part in enumerate(table.get("_others") or [], 1):
        figures = [verify.parse_amount(c) for r in part.get("grid") or [] for c in r
                   if _money(c)]
        if any(f is not None and any(abs(f - s) < 0.005 for s in sums) for f in figures):
            out.append(number)
    return out


def _place_part_row(cells, columns):
    """One row of another table at this table's width, or None.

    Same width: as it is. Otherwise by what each cell IS, the way a person
    copies a totals line across: figures right-aligned into the last columns,
    words from the first column that names what a row is. A row with more
    filled cells than the table has columns cannot be placed and is refused.
    """
    width = len(columns)
    cells = [c or "" for c in cells]
    while cells and _blank(cells[-1]) and len(cells) > width:
        cells.pop()
    if len(cells) == width:
        return list(cells)
    filled = [c.strip() for c in cells if not _blank(c)]
    if not filled or len(filled) > width:
        return None
    out = [""] * width
    figures = [c for c in filled if cell_kind(c) in ("money", "nil")]
    words = [c for c in filled if cell_kind(c) not in ("money", "nil")]
    for k, cell in enumerate(reversed(figures)):
        out[width - 1 - k] = cell
    described = [i for i, h in enumerate(columns) if "description" in column_roles(h)]
    referenced = [i for i, h in enumerate(columns) if "reference" in column_roles(h)]
    position = (described or referenced or [0])[0]
    for word in words:
        while position < width and out[position]:
            position += 1
        if position >= width:
            return None
        out[position] = word
        position += 1
    return out


def concat(table: dict, picks, totals=()) -> dict:
    """Join the other tables named in `picks` (numbers from `concat_parts`).

    Each picked table's rows are placed under this table's columns
    (`_place_part_row`) and sorted the way the table's own rows were: an item row joins
    the rows, a totals line or a note joins `dropped` with its reason. A
    repeated heading row is skipped. The window after the table moves to after
    the last table joined, so the fix agent is asked about what follows the
    WHOLE table. Nothing is changed in a cell; what cannot be placed is refused
    and named.
    """
    others = table.get("_others") or []
    joined_parts = []
    numbers = sorted({n for n in picks if isinstance(n, int) and not isinstance(n, bool)
                      and 1 <= n <= len(others)})
    columns = table["columns"]
    record = {"joined": [], "rows_added": 0, "dropped_added": 0, "refusals": []}
    last = None
    for number in numbers:
        part = others[number - 1]
        grid = [r for r in part.get("grid") or [] if any(not _blank(c) for c in r)]
        placed = []
        for cells in grid:
            if _same_headings([c.strip() for c in cells], columns):
                continue
            row = _place_part_row(cells, columns)
            if row is None:
                placed = None
                break
            placed.append(row)
        if placed is None:
            record["refusals"].append({"number": number,
                                       "why": "a row has more filled cells than the "
                                              "table has columns"})
            continue
        # A row that would join as an ITEM must line up with the columns: words
        # under a money heading, or an amount under a description, is a block
        # of the page that is not this table (sol015's payment box, which
        # gemma4:e4b joined to the receipt's items).
        money_cols = [i for i, h in enumerate(columns) if _heading_kind(h) == "money"]
        if any(not _row_kind(row, columns)
               and (_violates(row, columns)
                    or (money_cols and not any(_money(row[i]) for i in money_cols)))
               for row in placed):
            record["refusals"].append({"number": number,
                                       "why": "its rows do not line up with the "
                                              "table's columns"})
            continue
        for row in placed:
            kind = _row_kind(row, columns)
            if kind == "blank":
                continue
            # A piece known to be the totals block (it holds a column total) is
            # totals from end to end, whatever its labels say.
            if not kind and number in totals:
                kind = "totals block"
            if kind:
                table.setdefault("dropped", []).append(
                    {"why": kind, "cells": row, "page": part.get("page")})
                record["dropped_added"] += 1
            else:
                table["rows"].append(row)
                table.setdefault("row_pages", []).append(part.get("page"))
                record["rows_added"] += 1
        if part.get("page") not in table.get("pages", []):
            table.setdefault("pages", []).append(part.get("page"))
        record["joined"].append(number)
        joined_parts.append(part)
    # The window after the table: from the LATEST of the table and its joined
    # pieces -- never earlier, so sol023's empty summary box on page 4 cannot
    # pull it back from page 5 -- and stepping over the joined pieces rather than
    # starting after them, so lines the read wrote between the table and a piece
    # of it (sol023 writes half its summary box as lines, half as a table) stay
    # in front of the fix agent.
    if joined_parts:
        current = table.get("_tail_from")
        latest = max([p.get("page") or 0 for p in joined_parts]
                     + ([current.get("page") or 0] if current else []))
        if current and (current.get("page") or 0) == latest:
            source = current
        else:
            source = min([p for p in joined_parts if (p.get("page") or 0) == latest],
                         key=_order)
        table["tail"] = _tail_after(source, joined_parts, table.get("_share"))
        table["tail_page"] = source.get("page")
        table["_tail_from"] = source
    if record["joined"]:
        table.setdefault("repairs", []).append(
            "%d more table%s the read cut from this one joined to it (%d row%s, %d "
            "totals or notes)" % (len(record["joined"]),
                                  "" if len(record["joined"]) == 1 else "s",
                                  record["rows_added"],
                                  "" if record["rows_added"] == 1 else "s",
                                  record["dropped_added"]))
    table["concat_done"] = record
    return table


def public(table):
    """The table as a result carries it: a copy, with no working state in it."""
    if not table:
        return None
    keys = ("is_master_table", "master_why", "columns", "rows", "row_pages",
            "misaligned", "realigned", "dropped", "reference_columns",
            "item_columns",
            "headings_lost", "pages", "tables_found", "repairs", "source",
            "recut", "outside_text", "footer", "fix_agent", "identify_agent",
            "master_from",
            "master_rules", "master_rules_why", "stages", "concat")
    return {k: table[k] for k in keys if k in table}


def _selftest():
    """The shapes this module exists for, asserted at import. Cheap, and each is
    a failure that would otherwise surface as a quietly different table."""
    # Spans: a heading over two rows, and a totals row ruled across the table.
    html = ('<table><tr><td>Date</td><td>Invoice No.</td><td>Amount</td>'
            '<th colspan="2">Documents</th></tr>'
            '<tr><td></td><td></td><td></td><td>Tax</td><td>Receipt</td></tr>'
            '<tr><td>05-FEB-26</td><td>IV0005908</td><td>7,752.15</td><td>Y</td><td>Y</td></tr>'
            '<tr><td colspan="2">Total:</td><td>7,752.15</td><td></td><td></td></tr></table>')
    found = item_table(html)
    assert found["columns"][3:] == ["Documents Tax", "Documents Receipt"], found["columns"]
    assert found["rows"] == [["05-FEB-26", "IV0005908", "7,752.15", "Y", "Y"]]
    assert found["dropped"][0]["why"] == "totals row"
    assert found["is_master_table"] is True

    # A real item list that cites a document is still an item list.
    md = ("| No. | Description | Refer to Tax Invoice No. | Quantity | Unit Price | Amount |\n"
          "| --- | --- | --- | --- | --- | --- |\n"
          "| 1 | Widget | IV001 | 2 | 10.00 | 20.00 |\n")
    assert item_table(md)["is_master_table"] is False

    # A reference column nobody filled in is not a reference list.
    md = ("| No. | Reference | Description | Amount |\n| --- | --- | --- | --- |\n"
          "| 1 |  | Box | 170,430.00 |\n| 2 |  | Fee | 39,945.40 |\n")
    assert item_table(md)["is_master_table"] is False

    # A label and its value ruled as a table is not an item table.
    assert item_table("| Vatable Amount | 11,638.64 |\n| --- | --- |\n"
                      "| Vat 7% | 814.71 |\n") is None

    # Five headings, six cells, the reference read in two pieces. Settled from
    # the cells: a thirteen-digit number is not a description.
    ragged = ('<table><tr><td>Reference</td><td>Description</td><td>Amount</td>'
              '<td>Vat</td><td>Total Amount</td></tr>'
              '<tr><td>RO</td><td>1885555448955</td><td>Power</td><td>360,672.33</td>'
              '<td>25,247.06</td><td>385,919.39</td></tr></table>')
    table = item_table(ragged)
    assert table["misaligned"] == [] and table["realigned"] == [0], table
    assert table["rows"][0] == ["RO 1885555448955", "Power", "360,672.33",
                                "25,247.06", "385,919.39"]
    assert table["is_master_table"] is True

    # A row the cells do NOT settle -- three amounts for four money columns --
    # is left alone rather than guessed at, and says so.
    short = ("| Description | Amount | VAT | W/T | Net |\n| --- | --- | --- | --- | --- |\n"
             "| Rent | 100.00 | 7.00 | 3.00 | 104.00 |\n")
    table = item_table(short)
    table["rows"].append(["Fee", "50.00", "3.50", "53.50"])
    table["misaligned"] = [1]
    assert realign(table)["misaligned"] == [1]
    # The model's turn: a column for each cell, and only column numbers.
    for bad in ([[1, 2, 3]],                  # a cell with no column
                [[1, 2, 3, 6]],               # a column the table has not got
                [[2, 1, 3, 5]],               # reorders the row
                [["1", 2, 3, 5]],             # not a number
                [None], []):                  # no answer at all
        refused = recut(dict(table, rows=[list(r) for r in table["rows"]]), bad)
        assert refused["misaligned"] == [1] and refused["recut"]["taken"] == 0, bad
    before = content(table["rows"][1])
    right = recut(table, [[1, 2, 3, 5]])
    assert right["misaligned"] == [] and right["realigned"] == [1]
    assert right["rows"][1] == ["Fee", "50.00", "3.50", "", "53.50"]
    assert content(right["rows"][1]) == before


_selftest()
