"""RTV request routing and isolation; model accuracy still needs an AWS run."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import prompts
import segment


RTV = ["RTV_SHIP_CREDIT"]
ROOT = Path(__file__).resolve().parents[1]
STATUS = {"kind": "vllm", "url": "http://mock:8001", "reachable": True,
          "model": "gemma-test"}


class ReturnPromptTests(unittest.TestCase):
    def test_return_form_covers_identity_parties_and_totals_without_payment_or_wht(self):
        keys = prompts.fields_for_types(RTV)
        self.assertEqual(set(keys), {
            "document_type", "document_number", "issue_date", "seller_tax_id",
            "seller_branch", "seller_branch_location", "buyer_tax_id", "buyer_branch", "buyer_branch_location", "subtotal",
            "vat_total", "amount_incl_vat"})
        table = prompts.steps_for_types(RTV, extras=False)
        self.assertEqual([s["id"] for s in table],
                         ["document", "seller", "buyer", "totals"])
        self.assertEqual([k for s in table for k in s["keys"]], list(keys))
        self.assertEqual(prompts.mandatory_for_types(RTV), ())
        self.assertEqual(prompts.steps_for_types(RTV)[-1]["id"], "other")

    def test_agentic_questions_replace_conflicting_issuer_and_addressee_roles(self):
        table = {s["id"]: s for s in prompts.steps_for_types(RTV, extras=False)}
        for party in ("seller", "buyer"):
            question = app._step_question(table[party], RTV)
            self.assertIn("ORIGINAL SALE", question)
            self.assertIn("RECEIVING", question)
            self.assertIn("SENDING", question)
            self.assertIn("examples, not required wording", question)
            self.assertIn("A branch name is also valid", question)
            self.assertIn("not the store/location code", question)
            self.assertNotIn("who issued this document", question)
            self.assertNotIn("who this document is addressed to", question)
            self.assertNotIn("The issuer is the party this document is FROM", question)
            self.assertNotIn("00803", question)
            self.assertNotIn("0905550001570", question)

    def test_single_request_uses_the_same_return_roles(self):
        text = prompts.extract_prompt(prompts.fields_for_types(RTV), RTV, extras=False)
        self.assertIn("ORIGINAL SALE", text)
        self.assertIn("seller_tax_id: copy only that party", text)
        self.assertIn("buyer_tax_id: copy only that party", text)
        self.assertNotIn("The issuer is the party the document is FROM", text)
        self.assertNotIn('"certificate_no"', text)
        self.assertNotIn('"po_gr_rtv_number"', text)

    def test_return_document_step_distinguishes_own_number_and_return_date(self):
        document = prompts.steps_for_types(RTV, extras=False)[0]
        question = app._step_question(document, RTV)
        self.assertIn("เลขที่การส่งคืน", question)
        self.assertIn("วันที่ส่งคืน", question)
        self.assertIn("print timestamp", question)

    def test_credit_note_with_return_heading_keeps_credit_note_roles_and_form(self):
        # sol009's heading is a credit note accepting returned goods, not a
        # buyer-issued return shipment like sol014 page 2.
        mixed = RTV + ["CREDIT_NOTE"]
        self.assertEqual(prompts.fields_for_types(mixed),
                         prompts.fields_for_types(["CREDIT_NOTE"]))
        self.assertEqual(prompts.steps_for_types(mixed),
                         prompts.steps_for_types(["CREDIT_NOTE"]))
        self.assertEqual(prompts.type_bullets(mixed, prompts.fields_for_types(mixed)),
                         prompts.type_bullets(["CREDIT_NOTE"], prompts.fields_for_types(mixed)))
        text = prompts.extract_prompt(prompts.fields_for_types(mixed), mixed)
        self.assertNotIn("ORIGINAL SALE", text)
        self.assertIn("The issuer is the party the document is FROM", text)
        self.assertEqual(prompts.fields_for_types(RTV + ["DEBIT_NOTE"]),
                         prompts.fields_for_types(["DEBIT_NOTE"]))

    def test_document_specific_steps_do_not_mutate_shared_definitions(self):
        saved = copy.deepcopy(prompts.EXTRACT_STEPS)
        rtv = prompts.steps_for_types(RTV)
        seller = next(s for s in rtv if s["id"] == "seller")
        seller["title"] = "local change"
        self.assertEqual(prompts.EXTRACT_STEPS, saved)
        for codes in (["INVOICE"], ["CREDIT_NOTE"], ["RECEIPT"], []):
            step = next(s for s in prompts.steps_for_types(codes) if s["id"] == "seller")
            self.assertEqual(step["title"], "who issued this document")

    def test_real_return_page_routes_four_schema_requests_with_grounded_values(self):
        truth = json.loads((ROOT / "solution/sol014.fields.json").read_text(encoding="utf-8"))
        block = truth["documents"][1]
        expected = block
        # Stub only inference: exercise real form selection, step messages,
        # schemas, parsing, grounding and merge. This is not an accuracy test.
        text = segment.split_pages((ROOT / "solution/sol014.md").read_text(encoding="utf-8"))[1]
        form = app._field_set(text, doc_type=RTV, extras=False)
        form["schema_first"] = True
        requests = []

        def chat(message, cap, status, schema=None):
            keys = schema["properties"]
            values = {key: expected[key][0] if isinstance(expected[key], list)
                      else expected[key] for key in keys}
            requests.append((message, schema, values))
            return json.dumps(values, ensure_ascii=False), False, 10

        with patch.object(app, "_chat", side_effect=chat), \
             patch.object(app, "AGENTIC_PARALLEL", False):
            result = app._drain(app._extract_agentic(text, STATUS, form))
        self.assertNotIn("error", result)
        self.assertEqual(len(requests), 4)
        self.assertEqual(set().union(*(set(schema["properties"]) for _, schema, _ in requests)),
                         set(form["keys"]))
        for key in form["keys"]:
            want = expected[key][0] if isinstance(expected[key], list) else expected[key]
            self.assertEqual(result["fields"][key], want)
        self.assertTrue(all(not s.get("retried") for s in result["steps"]))


if __name__ == "__main__":
    unittest.main()
