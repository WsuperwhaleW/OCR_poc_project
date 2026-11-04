"""The `other_fields` switch: off, no part of pass 2 asks for the list.

Pinned: the single prompt and its grammar leave it out, the agentic `other`
step is not sent, a list the model writes anyway is dropped, a stress run takes
the switch from its request, and the run log leaves the count blank.
"""

import json
import unittest
from unittest import mock

import app
import engine
import prompts
import runlog

CODES = ["INVOICE"]
KEYS = list(prompts.fields_for_types(CODES))
FORM = {"doc_types": CODES, "items": (), "keys": KEYS,
        "mandatory": list(prompts.mandatory_for_types(CODES))}
STATUS = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m"}


class PromptTests(unittest.TestCase):
    def test_on_is_unchanged(self):
        self.assertEqual(prompts.extract_prompt(KEYS, CODES),
                         prompts.extract_prompt(KEYS, CODES, extras=True))
        self.assertIn("other_fields", prompts.extract_schema(KEYS)["required"])

    def test_off_leaves_the_list_out_of_skeleton_rules_and_grammar(self):
        text = prompts.extract_prompt(KEYS, CODES, extras=False)
        self.assertNotIn('"other_fields": [', text)
        self.assertNotIn("other_fields:\n", text)
        skeleton = prompts._skeleton_lines(KEYS, extras=False)
        # Still a JSON object once the `//` notes are taken off.
        bare = "\n".join(line.split("//")[0] for line in skeleton.splitlines())
        self.assertEqual(set(json.loads(bare)), set(KEYS))
        schema = prompts.extract_schema(KEYS, extras=False)
        self.assertNotIn("other_fields", schema["properties"])
        self.assertNotIn("other_fields", schema["required"])

    def test_off_drops_the_other_step(self):
        on = [s["id"] for s in prompts.steps_for_types(CODES)]
        off = [s["id"] for s in prompts.steps_for_types(CODES, extras=False)]
        self.assertIn("other", on)
        self.assertEqual(off, [i for i in on if i != "other"])


class SingleTests(unittest.TestCase):
    def _once(self, extras):
        reply = {"fields": None}
        body = {"choices": [{"message": {"content": json.dumps(
            {**{k: "" for k in KEYS},
             "other_fields": [{"label": "Tel", "value": "02"}]})},
            "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}
        res = mock.Mock(status_code=200)
        res.json.return_value = body
        sent = {}

        def post(url, payload, status):
            sent["payload"] = payload
            return res
        with mock.patch.object(engine, "_timed_post", side_effect=post):
            reply = app._extract_once("Tel 02", STATUS, None,
                                      {**FORM, "extras": extras})
        return reply, json.dumps(sent["payload"])

    def test_off_asks_for_none_and_keeps_none(self):
        reply, sent = self._once(False)
        self.assertEqual(reply["fields"]["other_fields"], [])
        self.assertNotIn('\\"other_fields\\": [', sent)

    def test_on_keeps_what_came_back(self):
        reply, sent = self._once(True)
        self.assertEqual(len(reply["fields"]["other_fields"]), 1)
        self.assertIn('\\"other_fields\\": [', sent)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.saved = app._extract_extras
        self.addCleanup(setattr, app, "_extract_extras", self.saved)

    def test_set_and_read(self):
        self.assertFalse(app.set_extract_extras(False))
        self.assertFalse(app.extract_extras())
        self.assertFalse(app._field_set("x", doc_type=CODES)["extras"])
        self.assertTrue(app._field_set("x", doc_type=CODES, extras=True)["extras"])

    def test_route(self):
        client = app.app.test_client()
        self.assertEqual(client.post("/api/extract/extras",
                                     json={"extras": "no"}).status_code, 400)
        self.assertEqual(client.post("/api/extract/extras",
                                     json={"extras": False}).get_json(),
                         {"extras": False})
        self.assertEqual(client.get("/api/extract/extras").get_json(),
                         {"extras": False})

    def test_run_log_leaves_the_count_blank(self):
        extracted = {"fields": {**{k: "" for k in KEYS}, "other_fields": []},
                     "extras": False}
        cells = runlog._extract_cells({"extracted": extracted})
        self.assertEqual((cells["other_fields"], cells["other_distinct"]), ("", ""))
        cells = runlog._extract_cells({"extracted": {**extracted, "extras": True}})
        self.assertEqual(cells["other_fields"], 0)


if __name__ == "__main__":
    unittest.main()
