"""The two table agents: one that fixes the table, one that identifies it.

Both are sent the TABLE and never the document, and neither answer reaches the
table as text -- the fix agent quotes stray text and `tables.remove_text` cuts
the cell's own characters out where the quote is found (into `outside_text`),
the identify agent's `true` must name a filled reference column. These pin the shape of each request and every way the code refuses an
answer. `_chat` is stubbed throughout: nothing here talks to a model server.
"""
import json
import unittest
from unittest import mock

import app
import tables


def html(header, *rows):
    tr = lambda cells: "<tr>" + "".join("<td>%s</td>" % c for c in cells) + "</tr>"
    return "<table>" + tr(header) + "".join(tr(r) for r in rows) + "</table>"


LETTERHEAD = "บริษัท ตัวอย่าง จำกัด\nเลขประจำตัวผู้เสียภาษี 0105555555555\n"
# sol007's shape: four headings, and every row's reference read as two cells.
RAGGED = LETTERHEAD + html(
    ["เอกสารอ้างอิง Reference", "รายการ Description", "จำนวนเงิน Amount",
     "ภาษีมูลค่าเพิ่ม Vat", "จำนวนเงินรวม Total Amount"],
    ["RO", "1885555448955", "ค่าไฟ", "360,672.33", "25,247.06", "385,919.39"],
    ["RO", "1885415445575", "ค่าน้ำ", "24,926.05", "1,744.82", "26,670.87"])
RAGGED_1 = ["RO 1885555448955", "ค่าไฟ", "360,672.33", "25,247.06", "385,919.39"]
RAGGED_2 = ["RO 1885415445575", "ค่าน้ำ", "24,926.05", "1,744.82", "26,670.87"]
GRID = LETTERHEAD + html(
    ["No.", "Invoice No.", "Date", "Amount"],
    ["1", "IV0005887", "20/11/68", "14,487.91"],
    ["2", "IV0005888", "20/11/68", "541.42"])
ITEMS = LETTERHEAD + html(
    ["No.", "Description", "Quantity", "Unit Price", "Amount"],
    ["1", "Widget", "2", "10.00", "20.00"],
    ["2", "Gadget", "1", "5.00", "5.00"])
FORM = {"items": [], "doc_types": ["RECEIPT", "TAX_INVOICE"]}
FIELDS = {"document_number": "RC-0001", "subtotal": "1,250.00", "issue_date": ""}
CUT_RIGHT = {"rows": [{"c1": 1, "c2": 1, "c3": 2, "c4": 3, "c5": 4, "c6": 5}] * 2}
CUT_WRONG = {"rows": [{"c1": 1, "c2": 2, "c3": 2, "c4": 3, "c5": 4, "c6": 5}] * 2}
MASTER_2 = {"is_master_table": True, "reference_column": 2}
MASTER_1 = {"is_master_table": True, "reference_column": 1}


class Stub:
    """Answers each agent request by what the request is asking for."""

    def __init__(self, clean=None, cut=None, identify=None, concat=None):
        self.replies = {"clean": clean, "cut": cut, "identify": identify,
                        "concat": {"same_table": []} if concat is None else concat}
        self.asked = []

    def __call__(self, message, max_tokens, status, schema=None):
        kind = ("clean" if '"stray_lines": null' in message
                else "identify" if '"is_master_table"' in message
                else "concat" if '"same_table": null' in message
                else "cut")
        self.asked.append((kind, message))
        reply = self.replies[kind]
        if isinstance(reply, Exception):
            raise reply
        return (reply if isinstance(reply, str) else json.dumps(reply), False, 7)

    def kinds(self):
        return [k for k, _ in self.asked]


def run(text, stub, form=FORM, fields=FIELDS):
    with mock.patch.object(app, "_chat", stub), \
            mock.patch.object(app, "TABLE_FIX_AGENT", True), \
            mock.patch.object(app, "TABLE_IDENTIFY_AGENT", True):
        return app._item_table(text, form, {"model": "stub"}, None, fields)


HEAD = ["เอกสารอ้างอิง Reference", "รายการ Description", "จำนวนเงิน Amount",
        "ภาษีมูลค่าเพิ่ม Vat", "จำนวนเงินรวม Total Amount"]
CLEAN_ROW_1 = ["RO 1885555448955", "ค่าไฟ", "360,672.33", "25,247.06", "385,919.39"]
CLEAN_ROW_2 = ["RO 1885415445575", "ค่าน้ำ", "24,926.05", "1,744.82", "26,670.87"]
# A stamp read INTO the description cell.
STAMPED = LETTERHEAD + html(
    HEAD, ["RO 1885555448955", "ค่าไฟ RECEIVED 25 APR 2024", "360,672.33",
           "25,247.06", "385,919.39"], CLEAN_ROW_2)
# A stamp read as a cell of its own, pushing every cell after it one column over.
INSERTED = LETTERHEAD + html(
    HEAD, ["RO 1885555448955", "ภาพตราประทับวงกลม PAID", "ค่าไฟ", "360,672.33",
           "25,247.06", "385,919.39"], CLEAN_ROW_2)
# A row that is nothing but a stamp.
STAMP_ROW = LETTERHEAD + html(
    HEAD, CLEAN_ROW_1, ["ตราประทับบริษัท", "RECEIVED", "", "", ""], CLEAN_ROW_2)
NOTHING = {"remove": []}


class FixAgentTests(unittest.TestCase):
    def test_a_clean_table_costs_one_question_and_changes_nothing(self):
        stub = Stub(clean=NOTHING, identify=MASTER_1)
        table = run(STAMPED.replace(" RECEIVED 25 APR 2024", ""), stub)
        self.assertEqual(stub.kinds(), ["clean", "identify"])
        self.assertEqual(table["rows"], [CLEAN_ROW_1, CLEAN_ROW_2])
        self.assertNotIn("outside_text", table)

    def test_the_fix_agent_sees_the_page_around_the_table_and_identify_does_not(self):
        # The fix agent is sent a window of the page around the table (2026-10-01:
        # a table the read cut short leaves its headings above it or its totals
        # below it); the identify agent is still sent the table alone.
        stub = Stub(clean=NOTHING, identify=MASTER_1)
        run(STAMPED, stub)
        asked = dict(stub.asked)
        self.assertIn("B1 บริษัท ตัวอย่าง จำกัด", asked["clean"])
        self.assertIn("1885555448955", asked["clean"])
        self.assertNotIn("ตัวอย่าง", asked["identify"])
        self.assertNotIn("0105555555555", asked["identify"])
        self.assertIn("1885555448955", asked["identify"])

    def test_stamp_text_in_a_cell_moves_out_as_plain_text(self):
        stub = Stub(clean={"remove": [{"row": 1, "column": 2,
                                       "text": "RECEIVED 25 APR 2024"}]},
                    identify=MASTER_1)
        table = run(STAMPED, stub)
        self.assertEqual(table["rows"][0], CLEAN_ROW_1)
        self.assertEqual([o["text"] for o in table["outside_text"]],
                         ["RECEIVED 25 APR 2024"])
        self.assertEqual(table["outside_text"][0]["row"], 0)
        self.assertEqual(len(table["fix_agent"]["removed"]), 1)

    def test_an_inserted_stamp_cell_comes_out_and_the_row_then_fits(self):
        stub = Stub(clean={"remove": [{"row": 1, "column": 2,
                                       "text": "ภาพตราประทับวงกลม PAID"}]},
                    identify=MASTER_1)
        table = run(INSERTED, stub)
        self.assertEqual(table["rows"][0], CLEAN_ROW_1)
        self.assertEqual(table["misaligned"], [])
        self.assertNotIn("cut", stub.kinds())      # code settled the cuts

    def test_a_row_that_is_only_a_stamp_leaves_the_table(self):
        stub = Stub(clean={"remove": [{"row": 2, "column": 1, "text": "ตราประทับบริษัท"},
                                      {"row": 2, "column": 2, "text": "RECEIVED"}]},
                    identify=MASTER_1)
        table = run(STAMP_ROW, stub)
        self.assertEqual(table["rows"], [CLEAN_ROW_1, CLEAN_ROW_2])
        self.assertEqual(len(table["row_pages"]), 2)
        self.assertEqual(table["fix_agent"]["rows_emptied"], [1])
        self.assertEqual([o["text"] for o in table["outside_text"]],
                         ["ตราประทับบริษัท", "RECEIVED"])

    def test_a_quote_the_model_reworded_is_not_found(self):
        stub = Stub(clean={"remove": [{"row": 1, "column": 2, "text": "RECEIVE 25 APR"}]},
                    identify=MASTER_1)
        table = run(STAMPED, stub)
        self.assertIn("RECEIVED", table["rows"][0][1])
        self.assertIn("not in that cell", table["fix_agent"]["refusals"][0]["why"])

    def test_spacing_is_not_a_rewording(self):
        stub = Stub(clean={"remove": [{"row": 1, "column": 2,
                                       "text": "RECEIVED  25 APR\n2024"}]},
                    identify=MASTER_1)
        table = run(STAMPED, stub)
        self.assertEqual(table["rows"][0], CLEAN_ROW_1)

    def test_the_code_keeps_real_values(self):
        stub = Stub(clean={"remove": [
            {"row": 1, "column": 3, "text": "360,672.33"},       # an amount
            {"row": 2, "column": 1, "text": "RO 1885415445575"},  # the reference
            {"row": 9, "column": 1, "text": "x"},                 # no such row
            {"row": 1, "column": 2, "text": ""}]},                # a placeholder
            identify=MASTER_1)
        table = run(STAMPED, stub)
        self.assertEqual(table["rows"][0][2], "360,672.33")
        self.assertEqual(table["rows"][1][0], "RO 1885415445575")
        whys = [r["why"] for r in table["fix_agent"]["refusals"]]
        self.assertEqual(len(whys), 3)
        self.assertIn("it holds an amount", whys[0])
        self.assertIn("it holds a document number", whys[1])
        self.assertIn("no cell", whys[2])

    def test_an_amount_pushed_under_another_heading_is_still_kept(self):
        # The QR cell shifts the row, so the last amount sits under no heading.
        stub = Stub(clean={"remove": [{"row": 1, "column": 6, "text": "385,919.39"},
                                      {"row": 1, "column": 2,
                                       "text": "ภาพตราประทับวงกลม PAID"}]},
                    identify=MASTER_1)
        table = run(INSERTED, stub)
        self.assertEqual(table["rows"][0], CLEAN_ROW_1)
        self.assertEqual(table["fix_agent"]["refusals"][0]["why"], "it holds an amount")

    def test_a_quote_carrying_the_cell_separator_is_still_found(self):
        stub = Stub(clean={"remove": [{"row": 1, "column": 2,
                                       "text": "RECEIVED 25 APR 2024 | "}]},
                    identify=MASTER_1)
        self.assertEqual(run(STAMPED, stub)["rows"][0], CLEAN_ROW_1)

    def test_a_document_number_inside_a_description_is_kept(self):
        # gemma4:e4b on sol022, all four documents, nothing planted.
        text = LETTERHEAD + html(
            HEAD, ["RO 1885555448955",
                   "ค่าบริการทำความสะอาด ใบสั่งซื้อเลขที่ :2260007911.Rev.0",
                   "360,672.33", "25,247.06", "385,919.39"], CLEAN_ROW_2)
        stub = Stub(clean={"remove": [{"row": 1, "column": 2,
                                       "text": "ใบสั่งซื้อเลขที่ :2260007911.Rev.0"}]},
                    identify=MASTER_1)
        table = run(text, stub)
        self.assertIn("2260007911", table["rows"][0][1])
        self.assertIn("document number", table["fix_agent"]["refusals"][0]["why"])

    def test_figures_are_never_left_with_nothing_saying_what_they_are(self):
        # gemma4:e4b took sol013's and sol002's only description out whole.
        text = LETTERHEAD + html(["รายการ Description", "จำนวนเงิน Amount"],
                                 ["ค่าบริการทำความสะอาด ธันวาคม 2568", "151,353.08"],
                                 ["ค่าบริการพิเศษ", "1,000.00"])
        stub = Stub(clean={"remove": [{"row": 1, "column": 1,
                                       "text": "ค่าบริการทำความสะอาด ธันวาคม 2568"}]},
                    identify=MASTER_1)
        table = run(text, stub)
        self.assertEqual(table["rows"][0][0], "ค่าบริการทำความสะอาด ธันวาคม 2568")
        self.assertIn("nothing saying", table["fix_agent"]["refusals"][0]["why"])

    def test_the_cells_are_sent_one_to_a_line(self):
        stub = Stub(clean=NOTHING, identify=MASTER_1)
        run(STAMPED, stub)
        message = dict(stub.asked)["clean"]
        self.assertIn("row 1, column 2 (รายการ Description): ค่าไฟ RECEIVED 25 APR 2024",
                      message)

    def test_a_failed_request_removes_nothing_and_code_still_fixes_cuts(self):
        stub = Stub(clean=ValueError("HTTP 500"), identify=MASTER_1)
        table = run(RAGGED, stub)
        self.assertIn("HTTP 500", table["fix_agent"]["error"])
        self.assertEqual(table["rows"][0], RAGGED_1)
        self.assertNotIn("outside_text", table)

    def test_a_cut_the_cells_contradict_is_still_refused(self):
        # What is left unsettled after the cleaning goes to the column question,
        # with `check_cut`. Here realign would settle it, so force it unsettled.
        stub = Stub(clean=NOTHING, cut=CUT_WRONG, identify=MASTER_1)
        with mock.patch.object(app.tables, "realign", side_effect=lambda t: t):
            table = run(RAGGED, stub)
        self.assertIn("cut", stub.kinds())
        self.assertEqual(table["recut"]["refused"], 2)
        self.assertIn("clearly better", table["recut"]["refusals"][0]["why"])

    def test_a_long_table_is_not_sent(self):
        stub = Stub(identify=MASTER_1)
        with mock.patch.object(app, "TABLE_FIX_MAX_ROWS", 1):
            table = run(STAMPED, stub)
        self.assertNotIn("clean", stub.kinds())
        self.assertIn("over the 1", table["fix_agent"]["skipped"])


# sol005's shape, as typhoon reads it: the rows, then the barcode and the
# handwriting that sit in the empty rows of the frame, then the table's own totals
# split onto lines, then ordinary page text.
FRAME = LETTERHEAD + html(HEAD, CLEAN_ROW_1, CLEAN_ROW_2) + (
    "\n\nA01$T*-\n"
    "ภาพบาร์โค้ดสีดำบนพื้นขาว มีตัวเลข \"10260202364\" อยู่ด้านล่าง\n"
    "รวมเงิน / Total Amount : 387,598.38\n"
    "สามแสนแปดหมื่นเจ็ดพันห้าร้อยเก้าสิบแปดบาทสามสิบแปดสตางค์\n\n"
    "<table><tr><td>ชำระโดย</td><td>เช็ค</td></tr></table>\n"
    "after the next table\n")


class TailTests(unittest.TestCase):
    # FRAME's page is 11 lines: 2 of letterhead, the 3 rows of the table, 4
    # lines after it, a one-row table and one more line.
    def test_the_window_is_a_share_of_the_page_on_each_side(self):
        stub = Stub(clean={"stray_lines": [], "table_lines": [], "remove": []},
                    identify=MASTER_1)
        run(FRAME, stub)
        message = dict(stub.asked)["clean"]
        # 20% of 11 is 3 lines each side.
        self.assertIn("B1 บริษัท ตัวอย่าง จำกัด", message)
        self.assertIn("L1 A01$T*-", message)
        self.assertIn("L3 รวมเงิน / Total Amount : 387,598.38", message)
        self.assertNotIn("L4 สามแสน", message)

    def test_a_wider_share_reaches_past_the_next_table(self):
        stub = Stub(clean={"stray_lines": [], "table_lines": [], "remove": []},
                    identify=MASTER_1)
        with mock.patch.object(app, "TABLE_CONTEXT_SHARE", 0.5):
            run(FRAME, stub)
        message = dict(stub.asked)["clean"]
        self.assertIn("L4 สามแสน", message)
        # The next table is written out row by row, its cells joined by |.
        self.assertIn("L5 ชำระโดย | เช็ค", message)
        self.assertIn("L6 after the next table", message)

    def test_the_window_never_leaves_the_page(self):
        text = FRAME + "\n--- page 2 ---\nบนหน้าถัดไป\n"
        table = tables.item_table(text, align=False, context_share=1.0)
        self.assertNotIn("บนหน้าถัดไป", table["tail"])
        self.assertEqual(table["tail"][-1], "after the next table")

    def test_a_line_before_the_table_can_be_named_the_tables_own(self):
        # A heading the read put above the table instead of in it.
        text = ("ใบเสร็จรับเงิน\nรายการสินค้า\n"
                + html(HEAD, CLEAN_ROW_1, CLEAN_ROW_2) + "\nรวม 3,000.00\n")
        stub = Stub(clean={"stray_lines": [], "table_lines": ["B2", "L1"],
                           "remove": []}, identify=MASTER_1)
        table = run(text, stub)
        self.assertEqual([(f["text"], f["where"]) for f in table["footer"]],
                         [("รายการสินค้า", "before the table"),
                          ("รวม 3,000.00", "after the table")])
        self.assertEqual(table["fix_agent"]["lines"]["own_before"], [1])
        self.assertEqual(table["fix_agent"]["lines"]["own"], [0])

    def test_a_line_before_the_table_is_never_called_stray(self):
        # gemma4:e4b called sol007's issuer block, above the table, stray.
        stub = Stub(clean={"stray_lines": ["B1", "L1"], "table_lines": [], "remove": []},
                    identify=MASTER_1)
        table = run(FRAME, stub)
        self.assertEqual([o["text"] for o in table["outside_text"]], ["A01$T*-"])
        refused = table["fix_agent"]["lines"]["refusals"]
        self.assertEqual([(r["side"], r["text"]) for r in refused],
                         [("B", "บริษัท ตัวอย่าง จำกัด")])
        self.assertIn("above the table", refused[0]["why"])

    def test_stray_lines_go_to_plain_text_and_the_totals_stay_with_the_table(self):
        stub = Stub(clean={"stray_lines": [1, 2], "table_lines": [3], "remove": []},
                    identify=MASTER_1)
        table = run(FRAME, stub)
        self.assertEqual([o["text"] for o in table["outside_text"]],
                         ["A01$T*-", "ภาพบาร์โค้ดสีดำบนพื้นขาว มีตัวเลข \"10260202364\" "
                          "อยู่ด้านล่าง"])
        self.assertEqual([f["text"] for f in table["footer"]],
                         ["รวมเงิน / Total Amount : 387,598.38"])
        self.assertEqual(table["rows"], [CLEAN_ROW_1, CLEAN_ROW_2])
        self.assertEqual(table["fix_agent"]["lines"]["stray"], [0, 1])

    def test_a_line_of_figures_is_never_called_stray(self):
        stub = Stub(clean={"stray_lines": [1, 3], "table_lines": [], "remove": []},
                    identify=MASTER_1)
        table = run(FRAME, stub)
        self.assertEqual([o["text"] for o in table["outside_text"]], ["A01$T*-"])
        self.assertIn("amount", table["fix_agent"]["lines"]["refusals"][0]["why"])

    def test_a_line_in_both_lists_is_in_neither(self):
        stub = Stub(clean={"stray_lines": [2, "L1"], "table_lines": [2, 9, True],
                           "remove": []}, identify=MASTER_1)
        table = run(FRAME, stub)
        self.assertEqual([o["text"] for o in table["outside_text"]], ["A01$T*-"])
        self.assertNotIn("footer", table)

    def test_a_totals_block_the_read_cut_into_its_own_table_is_shown(self):
        # The case the window exists for: the totals came out as a second small
        # table, which the fixed tail before 2026-10-01 stopped at.
        text = ("Header line\n\n| Description | Amount |\n| --- | --- |\n| x | 1,000.00 |\n"
                "| y | 2,000.00 |\n\n*A04$CN*\n3:1\n\n| รวม | 3,000.00 |\n| --- | --- |\n")
        table = tables.item_table(text, align=False)
        self.assertEqual(table["context_before"], ["Header line"])
        self.assertEqual(table["tail"], ["*A04$CN*", "3:1", "รวม | 3,000.00"])
        self.assertEqual(tables.totals_line(table["tail"]), 2)


class TotalsLineTests(unittest.TestCase):
    def test_the_totals_line_is_found_with_figures_or_as_a_label_alone(self):
        self.assertEqual(tables.totals_line(
            ["*A01$TX*", "10260202364", "รวมเงิน / Total Amount 3,646,128.51 174,443.47"]), 2)
        # A fresh read of sol001: the label alone, its figures in a later table.
        self.assertEqual(tables.totals_line(
            ["วันที่รับวางบิล 6/2/69", "เวลารับฝากเช็ค 08:30 - 15:30 น.",
             "รวมเงิน / Total Amount"]), 2)
        self.assertIsNone(tables.totals_line(
            ["รวมทั้งหมดนี้บริษัทจะชำระภายในสามสิบวันนับจากวันที่ได้รับเอกสาร"]))

    def test_the_agent_is_told_where_the_frame_ends(self):
        stub = Stub(clean={"stray_lines": [], "table_lines": [], "remove": []},
                    identify=MASTER_1)
        run(FRAME, stub)
        self.assertIn("L3 is the table's totals line. So L1 to L2 were read",
                      dict(stub.asked)["clean"])


class IdentifyAgentTests(unittest.TestCase):
    sound = {"remove": []}

    def test_it_is_sent_the_type_and_a_few_values(self):
        stub = Stub(clean=self.sound, identify=MASTER_2)
        run(GRID, stub)
        message = dict(stub.asked)["identify"]
        self.assertIn("a receipt and a tax invoice", message)
        self.assertIn("this document's own number: RC-0001", message)
        self.assertIn("total before VAT: 1,250.00", message)
        self.assertNotIn("this document's date", message)   # empty: not shown

    def test_true_with_a_filled_reference_column_is_taken(self):
        table = run(GRID, Stub(clean=self.sound, identify=MASTER_2))
        self.assertIs(table["is_master_table"], True)
        self.assertEqual(table["master_from"], "model")
        self.assertEqual(table["reference_columns"], ["Invoice No."])

    def test_true_naming_a_column_that_is_not_a_reference_is_refused(self):
        stub = Stub(clean=self.sound,
                    identify={"is_master_table": True, "reference_column": 4})
        table = run(ITEMS, stub)
        self.assertIs(table["is_master_table"], False)        # the rule's answer
        self.assertEqual(table["master_from"], "rules")
        self.assertIn("something else", table["identify_agent"]["refused"])

    def test_false_is_taken_and_the_rule_is_kept_beside_it(self):
        stub = Stub(clean=self.sound,
                    identify={"is_master_table": False, "reference_column": 0})
        table = run(GRID, stub)
        self.assertIs(table["is_master_table"], False)
        self.assertIs(table["master_rules"], True)
        self.assertEqual(table["master_from"], "model")

    def test_the_word_in_quotes_is_the_same_answer(self):
        # gemma4:e4b, measured: the literal on one call, the word on the next.
        stub = Stub(clean=self.sound,
                    identify={"is_master_table": "true", "reference_column": "2"})
        table = run(GRID, stub)
        self.assertIs(table["is_master_table"], True)
        self.assertEqual(table["master_from"], "model")

    def test_anything_but_a_boolean_leaves_the_rule(self):
        for reply in ({"is_master_table": "yes", "reference_column": 2},
                      "not json at all", ValueError("HTTP 500")):
            table = run(GRID, Stub(clean=self.sound, identify=reply))
            self.assertIs(table["is_master_table"], True, reply)
            self.assertEqual(table["master_from"], "rules", reply)

    def test_off_asks_nothing(self):
        stub = Stub(clean=self.sound)
        with mock.patch.object(app, "_chat", stub), \
                mock.patch.object(app, "TABLE_FIX_AGENT", True), \
                mock.patch.object(app, "TABLE_IDENTIFY_AGENT", False):
            table = app._item_table(GRID, FORM, {"model": "stub"}, None, FIELDS)
        self.assertEqual(stub.kinds(), ["clean"])
        self.assertEqual(table["master_from"], "rules")


class CheckCutTests(unittest.TestCase):
    def test_a_tie_is_the_models_to_decide(self):
        columns = ["Description", "Remarks"]
        expected = tables._expected_kinds(columns, [])
        # Two cells of words, and nothing in them says which cut is right.
        self.assertEqual(tables.check_cut(["ค่าไฟ", "เดือนมกราคม"], [1, 1],
                                          columns, expected), "")


if __name__ == "__main__":
    unittest.main()
