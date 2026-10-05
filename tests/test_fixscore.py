"""The fix agent's score against solution/tables -- fixscore.py.

Built from sol005's saved shape: a table read with its rows, then a barcode
and handwriting the OCR model wrote as lines after it. The agent's output is
put on the table by hand, so every verdict is the scorer's alone.
"""

import copy
import unittest

import fixscore
import runlog
import tables

HEAD = ["รายการ Description", "ประจำงวด Period", "จำนวนเงิน Gross Amount",
        "ภาษีมูลค่าเพิ่ม VAT", "ภาษีหัก ณ ที่จ่าย W/T", "จำนวนสุทธิ Net Amount"]
ROWS = [
    ["ค่าบริการ Consign", "01/01/2026 - 31/01/2026", "1,731,118.40", "121,178.29",
     "51,933.55", "1,800,363.14"],
    ["ค่าบริการสาธารณูปโภค Consign", "01/01/2026 - 31/01/2026", "760,931.17",
     "53,265.18", "22,827.94", "791,368.41"],
    ["ค่าเช่า Consign", "01/01/2026 - 31/01/2026", "1,154,078.94", "-", "57,703.95",
     "1,096,374.99"],
]
TRANSCRIPT = ("| " + " | ".join(HEAD) + " |\n| " + " | ".join("---" for _ in HEAD) + " |\n"
              + "\n".join("| " + " | ".join(r) + " |" for r in ROWS)
              + "\n\nA01$T*-\n10260202364\nรวมเงิน / Total Amount 3,646,128.51\n")


def read_table():
    table = tables.item_table(TRANSCRIPT, align=False)
    table["stages"] = [{"stage": "read", "columns": list(table["columns"]),
                        "rows": [list(r) for r in table["rows"]]}]
    table["fix_agent"] = {"asked": True, "model": "stub"}
    return table


class LoadTests(unittest.TestCase):
    def test_a_truth_file_is_its_tables_and_their_strays(self):
        sections = fixscore.load("sol005")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["page"], 1)
        self.assertEqual(sections[0]["columns"], HEAD)
        self.assertEqual([(s["kind"], s["lines"]) for s in sections[0]["strays"]],
                         [("barcode sticker", ["*A01$TX*"]),
                          ("handwriting", ["10260202364"])])

    def test_a_mark_with_no_text_has_no_lines(self):
        strays = fixscore.load("sol001")[0]["strays"]
        self.assertEqual(strays[0]["kind"], "mark")
        self.assertEqual(strays[0]["lines"], [])
        self.assertGreater(len(strays[1]["lines"]), 10)

    def test_no_truth_file_is_no_score(self):
        self.assertEqual(fixscore.load("sol999"), [])
        self.assertIsNone(fixscore.score("sol999", read_table(), TRANSCRIPT))


class StrayTests(unittest.TestCase):
    def test_nothing_moved_is_missed_not_zero_scored(self):
        result = fixscore.score("sol005", read_table(), TRANSCRIPT)
        self.assertEqual(result["strays"]["scored"], 2)
        self.assertEqual(result["strays"]["missed"], 2)
        self.assertEqual(result["strays"]["recall"], 0.0)
        self.assertIsNone(result["removals"]["precision"])

    def test_a_garbled_barcode_still_counts_as_moved(self):
        table = read_table()
        table["outside_text"] = [{"text": "A01$T*-", "where": "after the table"},
                                 {"text": "10260202364", "where": "after the table"}]
        result = fixscore.score("sol005", table, TRANSCRIPT)
        self.assertEqual(result["strays"]["moved"], 2)
        self.assertEqual(result["strays"]["recall"], 100.0)
        self.assertEqual(result["removals"]["precision"], 100.0)

    def test_a_stray_the_read_never_produced_is_not_counted(self):
        transcript = TRANSCRIPT.replace("A01$T*-\n", "")
        result = fixscore.score("sol005", read_table(), transcript)
        self.assertEqual(result["strays"]["not_read"], 1)
        self.assertEqual(result["strays"]["scored"], 1)

    def test_a_short_stray_counts_as_read_only_where_printed_alone(self):
        self.assertFalse(fixscore._found("3:1", "31"))
        self.assertTrue(fixscore._found("3:1", "ลงชื่อ 3:1 x"))
        self.assertFalse(fixscore._found("ย", "กรุณารีบแจ้ง"))
        self.assertTrue(fixscore._found("ย", "ย"))

    def test_a_stray_inside_a_longer_word_is_not_read(self):
        self.assertFalse(fixscore._found("SPRING", "ชื่อร้าน : SPRINGROLL"))
        self.assertTrue(fixscore._found("*A01$TX*", "A01$T*-"))

    def test_a_stray_outside_the_window_is_not_shown(self):
        # The read produced it, at the top of the page, where the agent is not sent.
        transcript = "10260202364\n" + TRANSCRIPT.replace("10260202364\n", "")
        table = tables.item_table(transcript, align=False)
        table["stages"] = [{"stage": "read", "columns": list(table["columns"]),
                            "rows": [list(r) for r in table["rows"]]}]
        table["tail"] = [l for l in table["tail"] if l != "10260202364"]
        table["context_before"] = []
        table["fix_agent"] = {"asked": True, "model": "stub"}
        result = fixscore.score("sol005", table, transcript)
        self.assertEqual(result["strays"]["not_shown"], 1)
        self.assertEqual(result["strays"]["scored"], 1)

    def test_a_stray_left_in_a_cell_is_left_in_table(self):
        table = read_table()
        table["rows"][2][0] = "ค่าเช่า Consign 10260202364"
        result = fixscore.score("sol005", table, TRANSCRIPT)
        statuses = {i["text"]: i["status"] for i in result["strays"]["items"]}
        self.assertEqual(statuses["10260202364"], "left_in_table")


class RemovalTests(unittest.TestCase):
    def test_a_value_the_clean_table_holds_is_a_real_value(self):
        table = read_table()
        table["outside_text"] = [{"text": "ค่าบริการ Consign", "row": 0, "column": 0,
                                  "heading": HEAD[0]}]
        result = fixscore.score("sol005", table, TRANSCRIPT)
        self.assertEqual(result["removals"]["real_value"], 1)
        self.assertEqual(result["removals"]["precision"], 0.0)

    def test_other_page_text_is_page_text(self):
        table = read_table()
        table["outside_text"] = [{"text": "ผู้รับเงิน / Collector", "where": "after the table"}]
        result = fixscore.score("sol005", table, TRANSCRIPT)
        self.assertEqual(result["removals"]["page_text"], 1)

    def test_the_totals_line_named_the_tables_own_is_right(self):
        table = read_table()
        table["footer"] = [{"text": "รวมเงิน / Total Amount 3,646,128.51", "where": "after the table"},
                           {"text": "กรณีชำระเป็นเช็ค", "where": "after the table"}]
        result = fixscore.score("sol005", table, TRANSCRIPT)
        self.assertEqual((result["own_lines"]["table"], result["own_lines"]["not_table"]), (1, 1))


class CellTests(unittest.TestCase):
    def test_the_agents_effect_is_the_difference_its_stage_made(self):
        table = read_table()
        table["stages"][0]["rows"][1][0] = "ค่าบริการสาธารณูปโภค Consign PAID"
        clean = copy.deepcopy(table["stages"][0])
        clean["stage"] = "clean"
        clean["rows"][1][0] = "ค่าบริการสาธารณูปโภค Consign"
        table["stages"].append(clean)
        result = fixscore.score("sol005", table, TRANSCRIPT)
        cells = result["cells"]
        self.assertEqual(cells["expected"], 18)
        self.assertEqual(cells["agent_delta"], 1)
        self.assertEqual(cells["after_agent"], cells["read"] + 1)


class PoolAndLogTests(unittest.TestCase):
    def test_one_score_pools_to_itself_and_two_recompute_the_rates(self):
        one = fixscore.score("sol005", read_table(), TRANSCRIPT)
        self.assertIs(fixscore.pool([one]), one)
        moved = read_table()
        moved["outside_text"] = [{"text": "A01$T*-"}, {"text": "10260202364"}]
        two = fixscore.pool([one, fixscore.score("sol005", moved, TRANSCRIPT)])
        self.assertEqual((two["strays"]["moved"], two["strays"]["scored"]), (2, 4))
        self.assertEqual(two["strays"]["recall"], 50.0)
        self.assertIsNone(fixscore.pool([None, {"error": "x"}]))

    def test_the_run_log_writes_pairs_or_blanks(self):
        table = read_table()
        table["outside_text"] = [{"text": "A01$T*-"}]
        cells = runlog._fix_cells(fixscore.score("sol005", table, TRANSCRIPT))
        self.assertEqual((cells["fix_moved"], cells["fix_strays"]), (1, 2))
        self.assertEqual((cells["fix_stray_removals"], cells["fix_removals"]), (1, 1))
        self.assertEqual(cells["fix_cells"], 18)
        blank = runlog._fix_cells(None)
        self.assertTrue(all(v == "" for v in blank.values()))
        none_moved = runlog._fix_cells(fixscore.score("sol005", read_table(), TRANSCRIPT))
        self.assertEqual(none_moved["fix_removals"], "")


if __name__ == "__main__":
    unittest.main()
