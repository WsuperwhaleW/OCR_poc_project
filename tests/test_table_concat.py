"""The concat agent: a table the read cut into pieces, joined back into one.

sol007's shape as typhoon reads it: the item table, then its three totals
figures ruled as a table of their own. And sol023's: a header box on a later
page, which is a different block and must not be joined.
"""

import unittest
from unittest import mock

import app
import fixscore
import tables
from tests.test_table_agents import FORM, FIELDS, Stub, html

NL = chr(10)
HEAD = ["เอกสารอ้างอิง Reference", "รายการ Description", "จำนวนเงิน Amount",
        "ภาษีมูลค่าเพิ่ม Vat", "จำนวนเงินรวม Total Amount"]
ROW_1 = ["RO 1885555448955", "ค่าไฟ", "360,672.33", "25,247.06", "385,919.39"]
ROW_2 = ["RO 1885415445575", "ค่าน้ำ", "24,926.05", "1,744.82", "26,670.87"]
CUT = ("บริษัท ตัวอย่าง จำกัด\n" + html(HEAD, ROW_1, ROW_2)
       + "\n\nบริษัทฯได้หักภาษี ณ ที่จ่าย\n"
       + html(["", "", "385,598.38", "26,991.88", "412,590.26"])
       + "\n\n" + html(["ภาษีหัก ณ ที่จ่าย W/H"], ["รับชำระทั้งสิ้น Total Receipts", "400,000.00"])
       + "\nลงชื่อ ผู้รับเงิน\n")
HEADER_BOX = (html(HEAD, ROW_1) + "\n--- page 2 ---\n"
              + html(["PO No.", "SO No.", "Sales"], ["R250001589", "SO25120112", "ck"])
              + "\n" + html(HEAD, ROW_2))


def run(text, stub):
    with mock.patch.object(app, "_chat", stub), \
            mock.patch.object(app, "TABLE_CONCAT_AGENT", True), \
            mock.patch.object(app, "TABLE_FIX_AGENT", False), \
            mock.patch.object(app, "TABLE_IDENTIFY_AGENT", False):
        return app._item_table(text, FORM, {"model": "stub"}, None, FIELDS)


class ConcatCodeTests(unittest.TestCase):
    def test_the_other_tables_after_the_item_table_are_offered(self):
        table = tables.item_table(CUT, align=False)
        parts = tables.concat_parts(table)
        self.assertEqual([p["number"] for p in parts], [1, 2])
        self.assertEqual(parts[0]["rows"][0][2:], ["385,598.38", "26,991.88", "412,590.26"])

    def test_a_totals_table_joins_as_totals_and_not_as_items(self):
        table = tables.concat(tables.item_table(CUT, align=False), [1])
        self.assertEqual(table["rows"], [ROW_1, ROW_2])
        last = table["dropped"][-1]
        self.assertEqual(last["cells"][2:], ["385,598.38", "26,991.88", "412,590.26"])
        self.assertEqual(table["concat_done"]["joined"], [1])

    def test_a_short_part_is_placed_by_what_each_cell_is(self):
        table = tables.concat(tables.item_table(CUT, align=False), [2])
        cells = [d["cells"] for d in table["dropped"]]
        self.assertIn(["", "รับชำระทั้งสิ้น Total Receipts", "", "", "400,000.00"], cells)

    def test_the_window_steps_over_the_pieces_joined(self):
        # The line between the table and a joined piece stays in front of the
        # fix agent; the piece's own rows, now part of the table, do not.
        table = tables.item_table(CUT, align=False)
        self.assertNotIn("ลงชื่อ ผู้รับเงิน", table["tail"])
        table = tables.concat(table, [2])
        self.assertIn("บริษัทฯได้หักภาษี ณ ที่จ่าย", table["tail"])
        self.assertFalse(any("Total Receipts" in line for line in table["tail"]))
        self.assertIn("ลงชื่อ ผู้รับเงิน", table["tail"])

    def test_a_number_naming_no_table_is_ignored(self):
        table = tables.concat(tables.item_table(CUT, align=False), [0, 9, True, "1"])
        self.assertEqual(table["concat_done"]["joined"], [])

    def test_a_part_wider_than_the_table_is_refused(self):
        wide = html(HEAD, ROW_1) + "\n" + html(["a", "b", "c", "d", "e", "f", "g"])
        table = tables.concat(tables.item_table(wide, align=False), [1])
        self.assertEqual(table["concat_done"]["joined"], [])
        self.assertIn("more filled cells", table["concat_done"]["refusals"][0]["why"])

    def test_a_figure_equal_to_a_column_total_marks_the_totals_piece(self):
        table = tables.item_table(CUT, align=False)
        self.assertEqual(tables.sum_parts(table), [1])
        box = tables.item_table(HEADER_BOX, align=False)
        self.assertEqual(tables.sum_parts(box), [])

    def test_a_piece_whose_rows_do_not_line_up_is_refused(self):
        payment = (html(HEAD, ROW_1, ROW_2) + NL
                   + html(["เช็คธนาคาร", "SCB", "เช็คเลขที่ 18661994", "ลงวันที่ 25/3/26",
                           "จำนวนเงิน"]))
        table = tables.concat(tables.item_table(payment, align=False), [1])
        self.assertEqual(table["concat_done"]["joined"], [])
        self.assertIn("line up", table["concat_done"]["refusals"][0]["why"])

    def test_a_piece_from_earlier_never_pulls_the_window_back(self):
        text = (html(HEAD, ROW_1) + NL + html(["รวม", "999.00"]) + NL + "--- page 2 ---" + NL
                + html(HEAD, ROW_2) + NL + "หลังตาราง" + NL)
        table = tables.item_table(text, align=False)
        self.assertIn("หลังตาราง", table["tail"])
        table["_others"] = [p for p in table["_others"] if p["page"] == 1] or table["_others"]
        table = tables.concat(table, [1])
        self.assertIn("หลังตาราง", table["tail"])

    def test_working_state_never_reaches_the_result(self):
        table = tables.public(tables.item_table(CUT, align=False))
        self.assertFalse(any(k.startswith("_") for k in table))


class ConcatAgentTests(unittest.TestCase):
    def test_a_column_total_is_joined_by_the_code_and_the_rest_asked(self):
        stub = Stub(concat={"same_table": ["T2"]})
        table = run(CUT, stub)
        self.assertEqual(table["concat"]["by_sum"], [1])
        message = dict(stub.asked)["concat"]
        self.assertNotIn("T1 (page", message)
        self.assertIn("T2 (page 1)", message)
        self.assertEqual(table["concat"]["joined"], [1, 2])
        self.assertIn("concat", [s["stage"] for s in table["stages"]])

    def test_a_reply_that_is_not_a_list_keeps_only_what_the_code_joined(self):
        stub = Stub(concat={"same_table": "all of them"})
        table = run(CUT, stub)
        self.assertIn("error", table["concat"])
        self.assertEqual(table["concat"]["joined"], [1])

    def test_a_number_the_code_already_joined_is_not_joined_twice(self):
        stub = Stub(concat={"same_table": [1, 2]})
        table = run(CUT, stub)
        self.assertEqual(table["concat"]["joined"], [1, 2])

    def test_no_other_table_is_no_question(self):
        stub = Stub()
        run("บริษัท\n" + html(HEAD, ROW_1, ROW_2), stub)
        self.assertNotIn("concat", stub.kinds())

    def test_the_question_shows_every_other_table(self):
        stub = Stub()
        run(HEADER_BOX, stub)
        message = dict(stub.asked)["concat"]
        self.assertIn("PO No. | SO No. | Sales", message)
        self.assertIn("T1 (page 2)", message)

    def test_switched_off_it_is_never_asked(self):
        stub = Stub()
        with mock.patch.object(app, "_chat", stub), \
                mock.patch.object(app, "TABLE_CONCAT_AGENT", False), \
                mock.patch.object(app, "TABLE_FIX_AGENT", False), \
                mock.patch.object(app, "TABLE_IDENTIFY_AGENT", False):
            app._item_table(CUT, FORM, {"model": "stub"}, None, FIELDS)
        self.assertNotIn("concat", stub.kinds())


class TruthFileTests(unittest.TestCase):
    def test_a_table_over_several_pages_is_one_section(self):
        sections = fixscore.load("sol023")
        self.assertEqual(len(sections), 8)
        self.assertEqual(sections[1]["pages"], [1, 2, 3, 4, 5])
        self.assertEqual(len(sections[1]["rows"]), 53 + 9)

    def test_an_item_table_and_its_totals_are_one_section(self):
        sections = fixscore.load("sol009")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["rows"][-1][-1], "208,839.60")


if __name__ == "__main__":
    unittest.main()
