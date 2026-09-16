"""Attributing a wrong field to the pass that lost it.

What these cover, in the order the module is read:

* the four attributions, each from the transcript rather than from a guess;
* the two claims the ground-truth transcript is consulted to REFUSE, and the
  fallback when there is no page to consult;
* the run-log invariant, on a real case rather than a hand-made score:

      blame_ocr + blame_extract + blame_unknown == p1_scored - p1_correct

  which is the whole reason the counted population is the headline's own;
* that none of it moves a score.
"""
import unittest

import blame
import fieldscore
import prompts
import runlog
import scoring


def row(path, expected, actual, status, required=True, **extra):
    return dict(path=path, expected=expected, actual=actual, status=status,
                required=required, **extra)


def score_of(*rows):
    return {"scalars": {"rows": list(rows)}}


class AttributionTests(unittest.TestCase):
    def test_a_value_the_transcript_lost_is_the_reads(self):
        s = score_of(row("buyer_tax_id", "0155737222723", "", "missed"))
        # The read dropped a digit, so the value is not findable.
        out = blame.check(s, "TAX ID 015573722723 total 1,200.00")
        self.assertEqual(out["fields"]["buyer_tax_id"]["blame"], blame.OCR)
        self.assertEqual(out["counts"], {"ocr": 1, "extraction": 0, "unknown": 0})
        self.assertEqual(out["mostly"], blame.OCR)

    def test_a_value_the_transcript_holds_is_the_extractions(self):
        s = score_of(row("issue_date", "17/3/69", "", "missed"))
        out = blame.check(s, "Receipt dated 17/3/69, total 1,200.00")
        self.assertEqual(out["fields"]["issue_date"]["blame"], blame.EXTRACTION)

    def test_a_wrong_value_is_the_extractions_when_the_right_one_was_there(self):
        s = score_of(row("document_number", "INV-9001", "INV-9002", "wrong"))
        out = blame.check(s, "Invoice no. INV-9001")
        self.assertEqual(out["fields"]["document_number"]["blame"],
                         blame.EXTRACTION)

    def test_any_accepted_reading_in_the_transcript_answers_for_the_key(self):
        # A name printed in Thai and again in English: a read that produced
        # either of them handed the extraction something it could answer with.
        s = score_of(row("seller_name", "บริษัท โจโจ้ จำกัด", "", "missed",
                         accepted=["บริษัท โจโจ้ จำกัด", "Jo-Jo TRAT CO., Ltd."]))
        out = blame.check(s, "Jo-Jo TRAT CO., Ltd.  99 Real Road")
        self.assertEqual(out["fields"]["seller_name"]["blame"], blame.EXTRACTION)

    def test_a_printed_dash_is_never_a_loss_the_read_can_be_blamed_for(self):
        # The truth is a dash; a figure in its place is the extraction's doing,
        # and there is nothing here a transcript could have held or lost.
        s = score_of(row("vat_total", "-", "672.30", "wrong"))
        out = blame.check(s, "VAT -")
        self.assertEqual(out["fields"]["vat_total"]["blame"], blame.EXTRACTION)


class InventedValueTests(unittest.TestCase):
    """`spurious`: the page states nothing and the extraction filled it anyway."""

    PAGE = "ABC Co Ltd  99 Real Road  total 1,200.00"

    def test_text_in_the_transcript_and_not_on_the_page_is_the_reads(self):
        # The 1 MP failure: the read invents an address and the extraction
        # copies it faithfully.
        s = score_of(row("buyer_branch", "", "88 Phantom Road", "spurious"))
        out = blame.check(s, self.PAGE + "  88 Phantom Road", self.PAGE)
        self.assertEqual(out["fields"]["buyer_branch"]["blame"], blame.OCR)

    def test_text_on_the_page_in_the_wrong_key_is_the_extractions(self):
        s = score_of(row("buyer_branch", "", "99 Real Road", "spurious"))
        out = blame.check(s, self.PAGE, self.PAGE)
        self.assertEqual(out["fields"]["buyer_branch"]["blame"],
                         blame.EXTRACTION)

    def test_text_in_neither_is_the_extractions(self):
        s = score_of(row("buyer_branch", "", "Invented Road", "spurious"))
        out = blame.check(s, self.PAGE, self.PAGE)
        self.assertEqual(out["fields"]["buyer_branch"]["blame"],
                         blame.EXTRACTION)

    def test_without_the_page_the_accusation_is_not_made(self):
        # The same invented address, with no ground-truth transcript to check it
        # against: the attribution falls back to the one that claims less.
        s = score_of(row("buyer_branch", "", "88 Phantom Road", "spurious"))
        out = blame.check(s, self.PAGE + "  88 Phantom Road")
        self.assertEqual(out["fields"]["buyer_branch"]["blame"],
                         blame.EXTRACTION)
        self.assertFalse(out["page_checked"])

    def test_an_invented_value_is_attributed_and_counted_nowhere(self):
        s = score_of(row("buyer_branch", "", "Invented Road", "spurious"))
        out = blame.check(s, self.PAGE, self.PAGE)
        self.assertFalse(out["fields"]["buyer_branch"]["counted"])
        self.assertEqual(out["counted"], 0)
        self.assertEqual(out["uncounted"], 1)


class RefusalTests(unittest.TestCase):
    def test_a_key_the_two_ground_truths_disagree_about_blames_nobody(self):
        # The field truth says the page prints this and the transcript truth
        # does not. Blaming a read for failing to produce text the page does not
        # print would accuse every model that ever reads it.
        s = score_of(row("gr_number", "GR-0001", "", "missed"))
        out = blame.check(s, "a transcript", "a page that says nothing of it")
        entry = out["fields"]["gr_number"]
        self.assertEqual(entry["blame"], blame.UNKNOWN)
        self.assertTrue(entry["counted"])
        self.assertEqual(out["counts"]["unknown"], 1)

    def test_without_the_page_the_same_key_is_the_reads(self):
        s = score_of(row("gr_number", "GR-0001", "", "missed"))
        out = blame.check(s, "a transcript")
        self.assertEqual(out["fields"]["gr_number"]["blame"], blame.OCR)

    def test_a_truth_fed_run_is_not_attributed_at_all(self):
        # Its transcript IS the ground truth, so every verdict would read
        # `extraction` whether or not the extraction was at fault.
        s = score_of(row("issue_date", "17/3/69", "", "missed"))
        out = blame.check(s, "Receipt dated 17/3/69", truth_fed=True)
        self.assertIn("skipped", out)
        self.assertNotIn("counts", out)

    def test_no_score_and_no_transcript_are_skipped_rather_than_guessed(self):
        self.assertIn("skipped", blame.check({"error": "no truth"}, "text"))
        self.assertIn("skipped", blame.check(score_of(), "   "))

    def test_a_correct_value_is_not_attributed(self):
        s = score_of(row("issue_date", "17/3/69", "17/3/69", "correct"),
                     row("currency", "", "", "absent"))
        out = blame.check(s, "Receipt dated 17/3/69")
        self.assertEqual(out["fields"], {})
        self.assertIsNone(out["mostly"])


class TableTests(unittest.TestCase):
    def test_both_tables_cells_are_attributed(self):
        s = {"scalars": {"rows": []},
             "income_items": {"rows": [
                 row("income_items[0].amount_paid", "10,000.00", "", "missed")]},
             "line_items": {"rows": [
                 row("line_items[0].description", "Rent", "", "missed")]}}
        out = blame.check(s, "income 10,000.00 baht")
        self.assertEqual(out["fields"]["income_items[0].amount_paid"]["blame"],
                         blame.EXTRACTION)
        self.assertEqual(out["fields"]["line_items[0].description"]["blame"],
                         blame.OCR)

    def test_a_table_cell_never_reaches_the_per_field_column(self):
        # `field_blame` is scalars and field names only, the rule
        # `field_verdicts` already follows: one cell of one row going wrong is
        # not a weakness of a key.
        out = blame.check(
            {"income_items": {"rows": [
                row("income_items[0].amount_paid", "10,000.00", "", "missed")]},
             "scalars": {"rows": [row("issue_date", "17/3/69", "", "missed")]}},
            "nothing here")
        self.assertEqual(runlog.field_blame(out), "issue_date=o")


class MergeTests(unittest.TestCase):
    def test_paths_are_prefixed_with_their_document_and_counts_sum(self):
        one = blame.check(score_of(row("buyer_name", "A Ltd", "", "missed")),
                          "nothing")
        two = blame.check(score_of(row("buyer_name", "B Ltd", "", "missed")),
                          "B Ltd is here")
        merged = blame.merge([(1, one), (2, two)])
        self.assertEqual(sorted(merged["fields"]), ["doc1.buyer_name",
                                                    "doc2.buyer_name"])
        self.assertEqual(merged["counts"],
                         {"ocr": 1, "extraction": 1, "unknown": 0})
        self.assertEqual(merged["mostly"], "mixed")
        # The prefix comes off for the per-field column, so it is keyed the way
        # `field_verdicts` is keyed and the two can be read side by side. Both
        # documents are written and the reader keeps the last, which is exactly
        # what that column already does on a pack.
        self.assertEqual(runlog.field_blame(merged),
                         "buyer_name=o;buyer_name=e")
        self.assertEqual(runlog.parse_blame(runlog.field_blame(merged)),
                         {"buyer_name": "extraction"})

    def test_a_file_whose_documents_were_all_skipped_reports_the_reason(self):
        skipped = blame.check(score_of(), "text", truth_fed=True)
        merged = blame.merge([(1, skipped), (2, skipped)])
        self.assertIn("skipped", merged)
        self.assertNotIn("counts", merged)


class RunLogTests(unittest.TestCase):
    def test_an_attribution_that_did_not_run_is_blank_not_zero(self):
        cells = runlog._blame_cells({"skipped": "fed the ground truth"})
        self.assertEqual(cells, {"blame_ocr": "", "blame_extract": "",
                                 "blame_unknown": "", "field_blame": ""})

    def test_a_run_that_lost_nothing_writes_zeroes(self):
        # "Attributed, and nothing went wrong" and "nobody asked" are different
        # statements, and only the first is a measurement.
        clean = blame.check(score_of(row("issue_date", "1/1/26", "1/1/26",
                                         "correct")), "dated 1/1/26")
        self.assertEqual(runlog._blame_cells(clean),
                         {"blame_ocr": 0, "blame_extract": 0,
                          "blame_unknown": 0, "field_blame": ""})

    def test_the_read_floor_blanks_the_score_and_not_the_attribution(self):
        # The case these columns are worth the most in: the rate is deliberately
        # blank, and the attribution is the only thing left that says why.
        attribution = blame.check(
            score_of(row("buyer_tax_id", "0155737222723", "", "missed")),
            "a transcript that lost it")
        cells = runlog._extract_cells({"extracted": {
            "fields": {"buyer_tax_id": ""},
            "fields_unscored": "the read scored 40.0%, under 75%",
            "field_score": {"overall": {"expected": 1,
                                        "counts": {"correct": 0, "missed": 1}}},
            "blame": attribution,
        }})
        self.assertEqual(cells["field_acc"], "")
        self.assertEqual(cells["blame_ocr"], 1)


class InvariantTests(unittest.TestCase):
    """The property the run-log columns rest on, on a real case."""

    CASE = "sol002"

    def _score(self, fields):
        codes = ["INVOICE"]
        truth = fieldscore.load_truth(self.CASE)
        return fieldscore.score(truth, fields, None,
                                keys=prompts.fields_for_types(codes),
                                mandatory=prompts.mandatory_for_types(codes),
                                item_keys=prompts.items_for_types(codes)), truth

    def test_the_counts_are_the_headlines_own_population(self):
        page = scoring.cases_index()[self.CASE]["ground_truth"].read_text("utf-8")
        truth = fieldscore.load_truth(self.CASE)
        stated = {k: v for k, v in truth["scalars"].items()
                  if isinstance(v, str)}
        # A perfect extraction, then three ways of being wrong: a value the read
        # lost, a value the extraction did not return, and a value invented.
        fields = dict(stated)
        fields["seller_tax_id"] = ""
        fields["document_number"] = "NOT-THIS-ONE"
        lost = stated["seller_tax_id"]
        transcript = page.replace(lost, lost[:-2])

        score, _ = self._score(fields)
        out = blame.check(score, transcript, page)
        cells = runlog._blame_cells(out)
        overall = score["overall"]
        self.assertEqual(
            cells["blame_ocr"] + cells["blame_extract"] + cells["blame_unknown"],
            overall["expected"] - overall["counts"].get("correct", 0))
        # And the read really is named for the value it dropped.
        self.assertEqual(out["fields"]["seller_tax_id"]["blame"], blame.OCR)

    def test_attributing_moves_no_score(self):
        fields = {k: v for k, v in fieldscore.load_truth(self.CASE)["scalars"].items()
                  if isinstance(v, str)}
        fields["document_number"] = "NOT-THIS-ONE"
        score, _ = self._score(fields)
        before = dict(score["overall"])
        blame.check(score, "a transcript")
        self.assertEqual(score["overall"], before)


if __name__ == "__main__":
    unittest.main()
