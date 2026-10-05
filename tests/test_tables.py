"""The item table: read out of a transcript, repaired, and classified.

What these cover, in the order the module is read:

* reading a table out of text -- spans, a heading over two rows, HTML an OCR
  model left malformed, and the two things most often ruled beside the item
  table that are not it;
* what is taken OUT of the item rows, and that it is kept with the reason;
* `is_master_table`: a list of other documents against a list of items, the
  three answers it can give, and the evidence each rests on;
* re-cutting a row that does not fit its headings -- by what its cells are, and
  then by a model that may say only which column each cell is under;
* the ground truth: the flag a person recorded against the rule, on every
  document of every fixture, and the score a perfect read has to reach;
* the run-log cells, where blank is not zero.
"""
import json
import unittest
from unittest import mock

import app
import fieldscore
import prompts
import runlog
import scoring
import tables


def html(header, *rows):
    def tr(cells):
        return "<tr>" + "".join("<td>%s</td>" % c for c in cells) + "</tr>"
    return "<table>" + tr(header) + "".join(tr(r) for r in rows) + "</table>"


def md(header, *rows):
    line = lambda cells: "| " + " | ".join(cells) + " |"
    return "\n".join([line(header), line(["---"] * len(header))]
                     + [line(r) for r in rows]) + "\n"


MASTER = md(["No.", "Invoice No.", "Date", "Amount"],
            ["1", "IV0005887", "20/11/68", "14,487.91"],
            ["2", "IV0005888", "20/11/68", "541.42"])
ITEMS = md(["No.", "Description", "Quantity", "Unit Price", "Amount"],
           ["1", "Widget", "2", "10.00", "20.00"])


class ReadingTests(unittest.TestCase):
    def test_a_colspan_and_a_rowspan_are_expanded(self):
        text = ('<table><tr><td>Reference</td><td>Description</td><td>Amount</td></tr>'
                '<tr><td rowspan="2">IV001</td><td>Rent</td><td>100.00</td></tr>'
                '<tr><td>Service</td><td>50.00</td></tr></table>')
        table = tables.item_table(text)
        # The spanned reference says the same thing about both rows.
        self.assertEqual(table["rows"], [["IV001", "Rent", "100.00"],
                                         ["IV001", "Service", "50.00"]])
        self.assertEqual(table["misaligned"], [])

    def test_a_heading_over_two_rows_is_one_heading_per_column(self):
        text = ('<table><tr><td>Date</td><td>Invoice No.</td><td>Amount</td>'
                '<th colspan="2">Documents</th></tr>'
                '<tr><td></td><td></td><td></td><td>Tax</td><td>Receipt</td></tr>'
                '<tr><td>05-FEB-26</td><td>IV0005908</td><td>7,752.15</td>'
                '<td>Y</td><td>Y</td></tr></table>')
        table = tables.item_table(text)
        self.assertEqual(table["columns"][3:],
                         ["Documents Tax", "Documents Receipt"])
        self.assertEqual(len(table["rows"]), 1)

    def test_a_rowspan_heading_is_not_repeated_into_its_own_subheading(self):
        text = ('<table><tr><th rowspan="2">Description<th rowspan="2">Qty'
                '<th colspan="2">Amount</th></tr>'
                '<tr><td>Goods</td><td>Other</td></tr>'
                '<tr><td>Soap</td><td>12</td><td>756.00</td><td>0.00</td></tr></table>')
        table = tables.item_table(text)
        self.assertEqual(table["columns"],
                         ["Description", "Qty", "Amount Goods", "Amount Other"])

    def test_malformed_html_still_yields_its_rows(self):
        # A `</th>` closing a `<td>`, debris between two cells, a row whose
        # first cell lost its tag: all three are in the saved reads.
        text = ('<table><tr><td>No.</td><td>Item No.</td><td>Description</td>'
                '<td>Qty</td></tr>'
                '<tr><td>1</td><td>A-100</th></td><td>Bolt</td><td>20</td></tr>'
                '<tr><td>2</td><br/>2</td><td>A-200</td><td>Nut</td><td>20</td></tr>'
                '<tr>\n 3\n<td>A-300</td><td>Washer</td><td>20</td></tr></table>')
        table = tables.item_table(text)
        self.assertEqual([r[0] for r in table["rows"]], ["1", "2", "3"])
        self.assertEqual(table["misaligned"], [])
        # What was dropped is said, not hidden.
        self.assertTrue(any("outside any cell" in note for note in table["repairs"]))

    def test_a_trailing_heading_no_row_reaches_is_not_a_column(self):
        text = ('<table><tr><td>Invoice No.</td><td>Amount</td>'
                '<th colspan="2"></th></tr>'
                '<tr><td>IV0005908</td><td>7,752.15</td></tr></table>')
        table = tables.item_table(text)
        self.assertEqual(table["columns"], ["Invoice No.", "Amount"])
        self.assertEqual(table["misaligned"], [])

    def test_a_label_and_its_value_is_not_an_item_table(self):
        self.assertIsNone(tables.item_table(
            md(["Vatable Amount", "11,638.64"], ["Vat 7%", "814.71"])))

    def test_a_strip_of_header_fields_is_not_the_item_table(self):
        strip = md(["PO No.", "SO No.", "Terms of payment", "Due Date"],
                   ["R250001589", "SO25120112", "30 days", "12/06/2026"])
        table = tables.item_table(strip + "\n\n" + ITEMS)
        self.assertEqual(table["columns"][1], "Description")
        self.assertEqual(table["tables_found"], 2)

    def test_no_table_at_all_is_none(self):
        self.assertIsNone(tables.item_table("Invoice INV-1\nTotal 100.00\n"))

    def test_one_table_over_two_pages_is_joined(self):
        text = ("--- page 1 ---\n" + MASTER + "\n--- page 2 ---\n"
                + md(["No.", "Invoice No.", "Date", "Amount"],
                     ["3", "IV0005889", "20/11/68", "184.58"]))
        table = tables.item_table(text)
        self.assertEqual(len(table["rows"]), 3)
        self.assertEqual(table["row_pages"], [1, 1, 2])
        self.assertEqual(table["pages"], [1, 2])


class RowTests(unittest.TestCase):
    def test_totals_notes_and_blanks_are_taken_out_and_kept(self):
        text = md(["No.", "Invoice No.", "Description", "Amount", "Received"],
                  ["1", "510210009577", "Turnover Rent", "56,656.60", "56,656.60"],
                  ["", "", "", "", ""],
                  ["", "", "", "56,656.60", "56,656.60"],
                  ["", "", "Total before tax", "", "56,656.60"],
                  ["Branch No.: 012001", "", "", "", ""])
        table = tables.item_table(text)
        self.assertEqual(len(table["rows"]), 1)
        self.assertEqual(sorted(d["why"] for d in table["dropped"]),
                         ["figures with no label", "section heading or note",
                          "totals row"])

    def test_a_vat_line_is_a_total_and_a_tax_charge_is_an_item(self):
        text = md(["Description", "Quantity", "Amount"],
                  ["Land and Building Tax", "1", "9,177.49"],
                  ["Vat 7%", "", "642.42"])
        table = tables.item_table(text)
        self.assertEqual([r[0] for r in table["rows"]], ["Land and Building Tax"])


class MasterTableTests(unittest.TestCase):
    def test_rows_that_are_other_documents(self):
        table = tables.item_table(MASTER)
        self.assertIs(table["is_master_table"], True)
        self.assertEqual(table["reference_columns"], ["Invoice No."])

    def test_a_reference_beside_a_description_is_still_a_master_table(self):
        # The three the user named all look like this: a reference AND a
        # description on every row.
        table = tables.item_table(md(
            ["Reference", "Description", "Amount", "Vat", "Total Amount"],
            ["RO 1885555448955", "Power", "360,672.33", "25,247.06", "385,919.39"]))
        self.assertIs(table["is_master_table"], True)

    def test_goods_that_cite_a_document_are_an_item_list(self):
        table = tables.item_table(md(
            ["No.", "Description", "Refer to Tax Invoice No.", "Quantity",
             "Unit Price", "Amount"],
            ["1", "Widget", "IV001", "2", "10.00", "20.00"]))
        self.assertIs(table["is_master_table"], False)
        self.assertEqual(table["item_columns"], ["Unit Price"])

    def test_a_plain_item_list(self):
        table = tables.item_table(ITEMS)
        self.assertIs(table["is_master_table"], False)
        self.assertEqual(table["reference_columns"], [])

    def test_a_reference_column_nobody_filled_in(self):
        table = tables.item_table(md(
            ["No.", "Reference", "Description", "Amount"],
            ["1", "", "Celebrations box", "170,430.00"],
            ["2", "", "Delivery fee", "39,945.40"]))
        self.assertIs(table["is_master_table"], False)

    def test_a_quantity_column_alone_does_not_make_an_item_list(self):
        # จำนวนเงินรับ read as จำนวนสิทธิ์ is a money column wearing a quantity's
        # heading. A count is something a list of documents can carry too.
        table = tables.item_table(md(
            ["No.", "Invoice No.", "Description", "จำนวน", "Amount"],
            ["1", "510210009577", "Turnover Rent", "56,656.60", "56,656.60"]))
        self.assertIs(table["is_master_table"], True)

    def test_a_garbled_heading_over_document_numbers_is_read_by_its_cells(self):
        table = tables.item_table(md(
            ["No.", "เลขทะเบียนหนี้", "รายละเอียด", "จำนวนเงิน"],
            ["1", "510210009577", "Turnover Rent", "56,656.60"],
            ["2", "510210008957", "Land Tax", "9,177.49"]))
        self.assertIs(table["is_master_table"], True)
        self.assertIn("read from the cells", table["master_why"])

    def test_a_heading_that_says_something_else_is_taken_at_its_word(self):
        table = tables.item_table(md(
            ["No.", "Tax ID", "Description", "Amount"],
            ["1", "0105557874178", "Consulting", "56,656.60"],
            ["2", "0105533353335", "Consulting", "9,177.49"]))
        self.assertIs(table["is_master_table"], False)

    def test_a_period_is_not_a_document_number(self):
        table = tables.item_table(md(
            ["Description", "Period", "Amount"],
            ["Rent", "01/01/2026 - 31/01/2026", "1,154,078.94"]))
        self.assertIs(table["is_master_table"], False)

    def test_not_determined_is_its_own_answer(self):
        master, why, _, _ = tables.classify(["", ""], [["a", "b"]])
        self.assertIsNone(master)
        master, _, refs, _ = tables.classify(["Invoice No.", "Amount"], [])
        self.assertIsNone(master)
        self.assertEqual(refs, ["Invoice No."])


class RealignTests(unittest.TestCase):
    HEAD = ["เอกสารอ้างอิง Reference", "รายการ Description", "จำนวนเงิน Amount",
            "ภาษีมูลค่าเพิ่ม Vat", "จำนวนเงินรวม Total Amount"]
    WANT = ["RO 1885555448955", "ค่าไฟ", "360,672.33", "25,247.06", "385,919.39"]

    def test_a_reference_read_in_two_cells(self):
        """sol007 as typhoon read it at 4 MP: five headings, six cells."""
        table = tables.item_table(html(
            self.HEAD, ["RO", "1885555448955", "ค่าไฟ", "360,672.33",
                        "25,247.06", "385,919.39"]))
        self.assertEqual(table["rows"], [self.WANT])
        self.assertEqual((table["misaligned"], table["realigned"]), ([], [0]))
        self.assertIs(table["is_master_table"], True)

    def test_colspans_the_page_does_not_rule(self):
        """The same table at 2 MP: spans on the headings and different spans on
        every row, so a row can fit by width and be one column out."""
        text = ('<table><tr><td>%s</td><td>%s</td><th colspan="2">%s'
                '<th colspan="2">%s<td>%s</td></th></tr>'
                '<tr><td>RO</td><td>1885555448955</td><td>ค่าไฟ</td>'
                '<td>360,672.33</td><td colspan="2">25,247.06</td>'
                '<td>385,919.39</td></tr>'
                '<tr><td>RO</td><td>1885415445575</td><td>ค่าน้ำ</td>'
                '<td colspan="2">24,926.05</td><td colspan="2">1,744.82</td>'
                '<td>26,670.87</td></tr></table>') % tuple(self.HEAD)
        table = tables.item_table(text)
        self.assertEqual(table["columns"], self.HEAD)
        self.assertEqual(table["rows"][0], self.WANT)
        self.assertEqual(table["rows"][1][0], "RO 1885415445575")
        self.assertEqual(table["misaligned"], [])
        self.assertIs(table["is_master_table"], True)

    def test_no_character_changes(self):
        ragged = html(self.HEAD, ["RO", "1885555448955", "ค่าไฟ", "360,672.33",
                                  "25,247.06", "385,919.39"])
        raw = tables.parse(ragged)[0]["grid"][1]
        table = tables.item_table(ragged)
        self.assertEqual(tables.content(table["rows"][0]), tables.content(raw))

    def test_a_real_column_with_no_heading_is_kept(self):
        """sol003 rules the satang beside the baht with nothing over it."""
        table = tables.item_table(md(
            ["จำนวน", "รายการสินค้าหรือบริการ", "ราคาต่อหน่วย",
             "จำนวนเงิน (รวมภาษี)", ""],
            ["", "ค่าไฟฟ้า กพ.69", "", "9,741", "60"],
            ["", "ค่าไฟฟ้าป้ายแบนเนอร์", "", "535", "-"]))
        self.assertEqual(len(table["columns"]), 5)
        self.assertEqual(table["rows"][0][3:], ["9,741", "60"])
        self.assertNotIn("realigned", table)

    def test_a_row_the_cells_do_not_settle_is_left_alone(self):
        table = tables.item_table(md(
            ["Description", "Amount", "VAT", "W/T", "Net"],
            ["Rent", "100.00", "7.00", "3.00", "104.00"]))
        table["rows"].append(["Fee", "50.00", "3.50", "53.50"])
        table["misaligned"] = [1]
        table = tables.realign(table)
        # Three amounts and four money columns: which one is blank is not
        # something the cells say.
        self.assertEqual(table["misaligned"], [1])
        self.assertEqual(table["rows"][1], ["Fee", "50.00", "3.50", "53.50"])


class RecutTests(unittest.TestCase):
    """The agents-OFF path: Python first, the model only for what it left.

    The two table agents are switched off here so these keep covering the path
    `TABLE_FIX_AGENT=0` restores; `AgentTests` covers them.
    """
    def setUp(self):
        for name in ("TABLE_FIX_AGENT", "TABLE_IDENTIFY_AGENT"):
            patch = mock.patch.object(app, name, False)
            patch.start()
            self.addCleanup(patch.stop)
        self.table = tables.item_table(md(
            ["Description", "Amount", "VAT", "W/T", "Net"],
            ["Rent", "100.00", "7.00", "3.00", "104.00"]))
        self.table["rows"].append(["Fee", "50.00", "3.50", "53.50"])
        self.table["misaligned"] = [1]

    def copy(self):
        return dict(self.table, rows=[list(r) for r in self.table["rows"]])

    def test_a_column_for_every_cell_is_taken(self):
        table = tables.recut(self.copy(), [[1, 2, 3, 5]])
        self.assertEqual(table["rows"][1], ["Fee", "50.00", "3.50", "", "53.50"])
        self.assertEqual(table["misaligned"], [])
        self.assertEqual(table["realigned"], [1])
        self.assertEqual(table["recut"], {"asked": 1, "taken": 1, "refused": 0})
        self.assertIn("model", table["source"])

    def test_anything_else_is_refused_and_the_row_stays_as_read(self):
        for answer in ([[1, 2, 3]],             # a cell with no column
                       [[1, 2, 3, 9]],          # a column the table has not got
                       [[2, 1, 3, 5]],          # reorders the row
                       [[True, 2, 3, 5]],       # a bool is not a column number
                       [["Fee", "50.00", "3.50", "53.50"]],   # text, not numbers
                       [None], []):
            table = tables.recut(self.copy(), answer)
            self.assertEqual(table["misaligned"], [1], answer)
            self.assertEqual(table["rows"][1], ["Fee", "50.00", "3.50", "53.50"])
            self.assertEqual(table["recut"]["taken"], 0)

    def test_the_request_is_made_only_for_a_row_that_does_not_fit(self):
        with mock.patch.object(app, "_chat") as chat:
            table = app._item_table(MASTER, {"items": []}, {"model": "stub"})
        chat.assert_not_called()
        self.assertNotIn("recut", table)

    def test_the_reply_names_every_cell_and_python_does_the_cutting(self):
        ragged = self.copy()
        reply = json.dumps({"rows": [{"c1": 1, "c2": 2, "c3": 3, "c4": 5}]})
        with mock.patch.object(app.tables, "item_table", return_value=ragged), \
                mock.patch.object(app, "_chat", return_value=(reply, False, 9)) as chat:
            table = app._item_table("x", {"items": []}, {"model": "stub"})
        message = chat.call_args[0][0]
        # Every cell is named in the question, so the answer cannot be the right
        # shape without placing each one.
        for name in ("c1: Fee", "c2: 50.00", "c3: 3.50", "c4: 53.50"):
            self.assertIn(name, message)
        self.assertEqual(table["rows"][1], ["Fee", "50.00", "3.50", "", "53.50"])
        self.assertEqual(table["recut"]["taken"], 1)

    def test_a_reply_that_leaves_a_cell_out_is_refused(self):
        ragged = self.copy()
        reply = json.dumps({"rows": [{"c1": 1, "c2": 2, "c3": 3}]})
        with mock.patch.object(app.tables, "item_table", return_value=ragged), \
                mock.patch.object(app, "_chat", return_value=(reply, False, 9)):
            table = app._item_table("x", {"items": []}, {"model": "stub"})
        self.assertEqual(table["misaligned"], [1])
        self.assertEqual(table["recut"]["refused"], 1)

    def test_a_failed_request_costs_the_repair_and_nothing_else(self):
        ragged = self.copy()
        with mock.patch.object(app.tables, "item_table", return_value=ragged), \
                mock.patch.object(app, "_chat", side_effect=ValueError("HTTP 500")):
            table = app._item_table("x", {"items": []}, {"model": "stub"})
        self.assertEqual(table["misaligned"], [1])
        self.assertIn("HTTP 500", table["recut"]["error"])

    def test_a_type_with_a_table_of_its_own_is_not_looked_at(self):
        form = {"items": list(prompts.INCOME_ITEM_KEYS)}
        self.assertIs(app._item_table(MASTER, form, {"model": "stub"}), ...)

    def test_one_page_of_a_pack_says_which_page_of_the_file_it_is(self):
        # A single page arrives as bare text with no marker in it.
        table = app._item_table(MASTER, {"items": []}, {"model": "stub"}, pages=[7])
        self.assertEqual(table["pages"], [7])
        self.assertEqual(set(table["row_pages"]), {7})

    def test_the_table_score_is_beside_the_field_score_and_never_in_it(self):
        table = app._item_table(MASTER, {"items": []}, {"model": "stub"})
        result = {"fields": {"subtotal": "1.00"}, "field_score": {"kept": True},
                  "item_table": table}
        scored = app._score_item_table(result, "sol007")
        self.assertEqual(scored["field_score"], {"kept": True})
        self.assertIn("table_score", scored)
        # Not looked for is not scored: the key is absent, not empty.
        self.assertNotIn("table_score",
                         app._score_item_table({"fields": {}}, "sol007"))


def fixture_documents():
    """(case id, pages or None) for every document of every scored case."""
    for case_id, case in sorted(scoring.cases_index().items()):
        documents = case.get("documents") or []
        if len(documents) > 1:
            for entry in documents:
                yield case_id, list(entry["pages"]), entry.get("doc_types") or []
        else:
            yield case_id, None, case.get("doc_types") or []


class TruthTests(unittest.TestCase):
    def test_the_three_the_requirement_names(self):
        for case_id, pages in (("sol004", None), ("sol007", None), ("sol015", [7])):
            truth = fieldscore.table_truth(case_id, pages)
            self.assertIs(truth["is_master_table"], True, case_id)
            self.assertIs(truth["table"]["is_master_table"], True, case_id)

    def test_the_rule_agrees_with_the_flag_a_person_recorded(self):
        """On the ground-truth transcripts, every document of every fixture."""
        checked = masters = 0
        for case_id, pages, codes in fixture_documents():
            if prompts.items_for_types(codes):
                continue                  # its type rules a table of its own
            truth = fieldscore.table_truth(case_id, pages)
            if truth["is_master_table"] is None:
                # Nobody recorded a flag: the page rules no item table.
                self.assertIsNone(truth["table"], (case_id, pages))
                continue
            self.assertIs(truth["table"]["is_master_table"],
                          truth["is_master_table"], (case_id, pages))
            checked += 1
            masters += truth["is_master_table"]
        self.assertGreaterEqual(checked, 30)
        self.assertEqual(masters, 9)

    def test_a_perfect_read_scores_every_cell(self):
        for case_id, pages, codes in fixture_documents():
            if prompts.items_for_types(codes):
                continue
            truth = fieldscore.table_truth(case_id, pages)
            score = fieldscore.evaluate_table(case_id, truth["table"], pages)
            if score is None:
                continue
            self.assertEqual(score["counts"]["correct"], score["expected"],
                             (case_id, pages))
            self.assertEqual(score["rows_matched"], score["rows_expected"])
            self.assertEqual(score["master"]["status"], "correct")

    def test_no_truth_file_warns_about_the_new_key(self):
        for case_id, pages, _ in fixture_documents():
            self.assertEqual(fieldscore.load_truth(case_id, pages)["warnings"], [],
                             (case_id, pages))


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.truth = tables.item_table(MASTER)

    def test_a_dropped_row_costs_its_cells(self):
        actual = tables.item_table(md(
            ["No.", "Invoice No.", "Date", "Amount"],
            ["1", "IV0005887", "20/11/68", "14,487.91"]))
        score = fieldscore.score_table(self.truth, actual, True)
        self.assertEqual((score["rows_matched"], score["rows_missed"]), (1, 1))
        self.assertEqual(score["counts"]["missed"], 4)
        self.assertEqual(score["master"]["status"], "correct")

    def test_a_garbled_heading_does_not_cost_the_column_under_it(self):
        actual = tables.item_table(MASTER.replace("Date", "Dale"))
        score = fieldscore.score_table(self.truth, actual, True)
        self.assertEqual(score["counts"]["correct"], score["expected"])

    def test_no_table_returned(self):
        score = fieldscore.score_table(self.truth, None, True)
        self.assertEqual(score["counts"]["missed"], score["expected"])
        self.assertEqual(score["master"]["status"], "missed")

    def test_the_wrong_kind_of_table(self):
        score = fieldscore.score_table(self.truth, tables.item_table(ITEMS), True)
        self.assertEqual(score["master"]["status"], "wrong")

    def test_undetermined_is_not_correct(self):
        actual = dict(tables.item_table(MASTER), is_master_table=None)
        self.assertEqual(
            fieldscore.score_table(self.truth, actual, True)["master"]["status"],
            "undetermined")

    def test_nobody_recorded_a_flag(self):
        score = fieldscore.score_table(self.truth, self.truth, None)
        self.assertEqual(score["master"]["status"], "unchecked")

    def test_every_judged_cell_says_where_it_is(self):
        score = fieldscore.score_table(self.truth, self.truth, True)
        self.assertEqual({r["column"] for r in score["rows"]}, {0, 1, 2, 3})
        self.assertEqual(score["rows"][1]["heading"], "Invoice No.")

    def test_a_file_of_several_is_pooled_by_count(self):
        one = fieldscore.score_table(self.truth, self.truth, True)
        two = fieldscore.score_table(self.truth, None, True)
        pooled = fieldscore.pool_tables([one, two, None, {"error": "x"}])
        self.assertEqual(pooled["expected"], one["expected"] * 2)
        self.assertEqual(pooled["counts"]["correct"], one["expected"])
        self.assertEqual((pooled["master_correct"], pooled["master_scored"]), (1, 2))
        self.assertIs(fieldscore.pool_tables([one]), one)
        self.assertIsNone(fieldscore.pool_tables([None]))


class RunLogTests(unittest.TestCase):
    NAMES = ("item_tables", "master_tables", "table_rows", "table_realigned",
             "table_misaligned", "table_cells_ok", "table_cells", "master_ok",
             "master_scored")

    def test_not_looked_for_is_blank(self):
        cells = runlog._table_cells([{"fields": {}}])
        self.assertEqual([cells[n] for n in self.NAMES], [""] * 9)

    def test_looked_for_and_none_found_is_zero(self):
        cells = runlog._table_cells([{"item_table": None}])
        self.assertEqual((cells["item_tables"], cells["master_tables"],
                          cells["table_rows"]), (0, 0, 0))
        self.assertEqual(cells["table_cells"], "")

    def test_one_document(self):
        table = tables.item_table(MASTER)
        score = fieldscore.score_table(table, table, True)
        cells = runlog._table_cells([{"item_table": tables.public(table)}], score)
        self.assertEqual([cells[n] for n in self.NAMES],
                         [1, 1, 2, 0, 0, 8, 8, 1, 1])

    def test_a_pack_is_counted_over_its_documents(self):
        master, items = tables.item_table(MASTER), tables.item_table(ITEMS)
        score = fieldscore.pool_tables([
            fieldscore.score_table(master, master, True),
            fieldscore.score_table(items, items, False)])
        cells = runlog._table_cells(
            [{"item_table": master}, {"item_table": items}, {"item_table": None}],
            score)
        self.assertEqual((cells["item_tables"], cells["master_tables"],
                          cells["table_rows"]), (2, 1, 3))
        self.assertEqual((cells["master_ok"], cells["master_scored"]), (2, 2))

    def test_the_columns_are_in_the_extraction_half(self):
        for name in self.NAMES:
            self.assertIn(name, runlog.COLUMNS)
            self.assertIn(name, runlog.EXTRACT_COLUMNS)


class PromptTests(unittest.TestCase):
    def test_the_question_asks_for_columns_and_never_for_text(self):
        table = tables.item_table(MASTER)
        table["rows"].append(["3", "IV", "0005889", "20/11/68", "184.58"])
        table["misaligned"] = [2]
        message = app._recut_message(table)
        self.assertIn('"c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0', message)
        self.assertIn("2. Invoice No.", message)
        # The rows that fit are shown as the pattern, and are not asked about.
        self.assertIn("IV0005887", message)
        self.assertEqual(message.count("Row 1 -- "), 1)
        self.assertNotIn("Row 2 -- ", message)


if __name__ == "__main__":
    unittest.main()
