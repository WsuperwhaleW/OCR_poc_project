"""Download diagnostics preserve evidence without putting it in History."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import config
import extractionlog
import runhistory
import workspace

MARKER = "PRIVATE_RAW_REPLY_NOT_FOR_SERVER_HISTORY"
RESULT = {"mode": "agentic", "tokens": 10, "model": "mock", "documents": [
    {"document": i, "pages": [i], "doc_types": ["INVOICE"],
     "fields": {"seller_tax_id": value},
     "steps": [{"id": "seller", "keys": ["seller_tax_id"],
                "values": {"seller_tax_id": value}, "raw": [{"text": MARKER, "prompt": "question"}]}],
     "field_score": {"scalars": {"rows": [{"path": "seller_tax_id", "actual": value,
        "expected": "seller", "status": verdict, "required": True}]}}}
    for i, value, verdict in ((1, "seller", "correct"), (2, "buyer", "wrong"))]}
STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock", "model": "mock"}


class FormatTests(unittest.TestCase):
    def test_correct_steps_keep_compact_stats_and_questions_without_raw_replies(self):
        result = copy.deepcopy(RESULT)
        result['documents'] = [result['documents'][0]]
        step = result['documents'][0]['steps'][0]
        step.update(prompt='shared question', seconds=1.2, attempts=1, chats=[{
            'in_tokens': 7000, 'out_tokens': 12, 'used_tokens': 7012,
            'context_limit': 8192, 'reply_cap': 220, 'schema': True,
            'seconds': 1.2, 'http_status': 200, 'unwanted': MARKER}])
        step['raw'][0]['text'] = MARKER * 10000
        events = list(extractionlog.events(
            [{'_extraction_result': result}] * 2,
            [{'index': 0}, {'index': 1}], {}, 0, {}))
        self.assertEqual(events[0]['diagnostics_revision'], 1)
        for event in events[1:-1]:
            doc = event['job']['documents'][0]
            self.assertEqual(doc['steps'], [])
            stats = doc['step_stats'][0]
            self.assertEqual(stats['chats'][0]['used_tokens'], 7012)
            self.assertTrue(stats['chats'][0]['schema'])
            self.assertNotIn('raw', stats)
            self.assertNotIn('values', stats)
            self.assertEqual(events[-1]['questions'][stats['question_id']], 'shared question')
        self.assertNotIn(MARKER, json.dumps(events))
        self.assertLess(len(json.dumps(events)), 10000)

    def test_provenance_hashes_code_and_truth_without_embedding_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, transcript, fields = root/'code.py', root/'a.md', root/'a.fields.json'
            code.write_text('private code source')
            transcript.write_text('private transcript')
            fields.write_text(json.dumps({'documents': [
                {'pages': [1], 'doc_types': ['INVOICE'], 'seller_tax_id': 'PRIVATE_ID'},
                {'pages': [2], 'doc_types': ['RECEIPT']}]}))
            kwargs = {'extra_files': {'normalise': code}, 'benchmarks': {
                'a': {'transcript': transcript, 'fields': fields, 'pages': 2}}}
            first = extractionlog.provenance(code, code, {}, **kwargs)
            self.assertEqual(first['benchmarks']['a']['expected_documents'][1]['pages'], [2])
            self.assertEqual(first['snapshot'], 'before_benchmark')
            self.assertNotIn('PRIVATE_ID', json.dumps(first))
            self.assertNotIn('private transcript', json.dumps(first))
            fields.write_text('{"seller_tax_id": "changed"}')
            second = extractionlog.provenance(code, code, {}, **kwargs)
            self.assertNotEqual(first['benchmarks']['a']['fields'], second['benchmarks']['a']['fields'])
            self.assertEqual(second['benchmarks']['a']['expected_documents'][0]['pages'], [1, 2])
            fields.unlink()
            missing = extractionlog.provenance(code, code, {}, **kwargs)
            self.assertIn('error', missing['benchmarks']['a']['fields'])

    def test_alignment_flags_crossing_truth_documents_without_rewriting_results(self):
        benchmark = {'expected_documents': [{'pages': [6]}, {'pages': [7]}]}
        result = {'documents': [{'document': 1, 'pages': [6, 7], 'fields': {'issue_date': 'date'}}]}
        before = copy.deepcopy(result)
        job = extractionlog.job_detail({'_extraction_result': result}, {'index': 0}, {}, benchmark=benchmark)
        issue = job['document_alignment'][0]
        self.assertEqual(issue['truth_pages_by_first_page'], [6])
        self.assertTrue(issue['crosses_truth_documents'])
        self.assertEqual(issue['overlapping_truth_documents'], [[6], [7]])
        self.assertEqual(result, before)
        result['documents'] = [{'pages': [6]}, {'pages': [7]}]
        job = extractionlog.job_detail({'_extraction_result': result}, {'index': 0}, {}, benchmark=benchmark)
        self.assertNotIn('document_alignment', job)

    def test_score_scope_form_and_unscored_normalized_values_are_preserved(self):
        fields = {'buyer_branch_location': 'สาขาอำพัน แพร่'}
        score = {'overall': {'accuracy_half': .9, 'accuracy_half_pooled': .8, 'expected': 10},
                 'optional': {'expected': 1}, 'coverage': {'required_checked': 10},
                 'unknown_type': True, 'warnings': ['scope warning'], 'scalars': {'rows': []}}
        result = {'fields': fields, 'normalized': {'buyer_branch_location': 'อำพัน แพร่'},
                  'field_score': score, 'classified': {'fields': list(fields), 'mandatory': [],
                                                     'unknown_type': True}}
        job = extractionlog.job_detail({'_extraction_result': result}, {'index': 0}, {})
        self.assertEqual(job['score']['overall']['accuracy_half_pooled'], .8)
        self.assertEqual(job['documents'][0]['score']['warnings'], ['scope warning'])
        self.assertEqual(job['documents'][0]['form']['keys'], list(fields))
        self.assertEqual(job['documents'][0]['form']['mandatory'], [])
        self.assertNotIn('scalars', job['score'])
        row = job['field_rows'][0]
        self.assertIsNone(row['expected'])
        self.assertEqual(row['actual'], 'สาขาอำพัน แพร่')
        self.assertEqual(row['normalized_actual'], 'อำพัน แพร่')

    def test_repeated_documents_share_contexts_and_questions_but_keep_retry_evidence(self):
        result = copy.deepcopy(RESULT)
        for doc in result["documents"]:
            doc["prompt_prefix"] = "shared input document"
            doc["steps"][0]["prompt"] = "shared question"
            doc["steps"][0]["raw"][0]["prompt"] = "shared question"
        # A successful final answer still needs its evidence when it was retried.
        result["documents"][0]["steps"][0]["retried"] = True
        events = list(extractionlog.events(
            [{"status": "ok", "_extraction_result": result}] * 2,
            [{"index": i, "case": "a"} for i in range(2)], {}, 0, {}))
        end = events[-1]
        self.assertEqual(list(end["contexts"].values()), ["shared input document"])
        self.assertEqual(list(end["questions"].values()), ["shared question"])
        for event in events[1:-1]:
            for doc in event["job"]["documents"]:
                self.assertIn(doc["context_id"], end["contexts"])
                self.assertEqual(doc["steps"][0]["raw"][0]["text"], MARKER)
                self.assertIn(doc["steps"][0]["raw"][0]["question_id"], end["questions"])

    def test_old_download_conversion_preserves_verdicts_and_removes_duplicate_dumps(self):
        old = {"format": "ocr-extraction-detail-log", "format_version": 1,
               "provenance": {"files": {"prompts": {"sha256": "abc", "source": "full source"}}},
               "report": {"completed": 1, "timeline": [[0, 1]], "agentic_context": {"requests": [1]}},
               "jobs": [{"job": 1, "queue_position": 1, "case": "a", "status": "ok",
                         "extraction": RESULT, "field_rows": [{"path": "seller_tax_id",
                         "expected": "seller", "actual": "buyer", "status": "wrong"}]}]}
        compact = extractionlog.compact_download(old)
        self.assertEqual(compact["format_version"], 2)
        self.assertEqual(compact["jobs"][0]["field_rows"], old["jobs"][0]["field_rows"])
        self.assertNotIn("extraction", compact["jobs"][0])
        self.assertEqual(compact["report"], {"completed": 1})
        self.assertEqual(compact["provenance"]["files"]["prompts"], {"sha256": "abc"})
        self.assertIn("source", old["provenance"]["files"]["prompts"])

    def test_documents_values_and_replies_are_not_merged(self):
        events = list(extractionlog.events([{"status": "ok", "_extraction_result": RESULT}],
            [{"index": 3, "case": "sol014"}], {}, 2, {}))
        detail = events[1]["job"]
        self.assertEqual((detail["job"], detail["queue_position"]), (4, 2))
        self.assertEqual([(r["document"], r["pages"], r["actual"], r["status"], r["step"])
            for r in detail["field_rows"]], [(1, [1], "seller", "correct", "seller"),
                                             (2, [2], "buyer", "wrong", "seller")])
        self.assertEqual(detail["documents"][1]["steps"][0]["raw"][0]["text"], MARKER)
        self.assertEqual(detail["documents"][0]["steps"], [])
        self.assertNotIn("extraction", detail)
        self.assertEqual(events[-1]["field_summary"][0]["counts"], {"correct": 1, "wrong": 1})
        self.assertEqual(events[-1]["field_summary"][0]["accuracy"], .5)

    def test_missing_truth_and_cancelled_jobs_do_not_invent_expected_values(self):
        result = {"fields": {"buyer_tax_id": "A"}, "raw": MARKER, "segments": [{"pages": [1, 2]}]}
        row = extractionlog.job_detail({"status": "ok", "_extraction_result": result},
            {"index": 0, "case": "real.pdf"}, {})["field_rows"][0]
        self.assertEqual((row["status"], row["expected"], row["pages"]), ("unscored", None, [1, 2]))
        cancelled = extractionlog.job_detail({"status": "cancelled"}, {"index": 0}, {})
        self.assertEqual(cancelled["documents"], [])
        self.assertEqual(cancelled["field_rows"], [])


class StreamTests(unittest.TestCase):
    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        truth = self.root / "a.md"
        truth.write_text("source transcript", encoding="utf-8")
        for target, name, kwargs in (
            (config, "LOG_DIR", {"new": self.root / "logs"}),
            (app.scoring, "cases_index", {"return_value": {"a": {"id": "a", "ground_truth": truth,
                "pdf": "a.pdf", "doc_types": ["INVOICE"]}}}),
            (app.fieldscore, "has_truth", {"return_value": True}),
            (app, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.app.test_client()

    def run_stress(self, **overrides):
        with patch.object(app, "extract_fields", side_effect=lambda *a, **kw: copy.deepcopy(RESULT)):
            response = self.client.post("/api/stress/stream", json={"mode": "extract", "cases": ["a"],
                "jobs": 2, "warmup": 1, "concurrency": 2, "save_details": True,
                "save_log": False, "history_monitor": False, **overrides})
            self.assertEqual(response.status_code, 200)
            return [json.loads(line) for line in response.data.splitlines()]

    def test_details_download_but_never_persist_in_history_database_or_csv(self):
        events = self.run_stress()
        jobs = [e["job"] for e in events if e["event"] == "detail_log_job"]
        self.assertEqual([j["job"] for j in jobs], [2, 3])
        self.assertIn(MARKER, json.dumps(jobs))
        start = next(e for e in events if e["event"] == "detail_log_start")
        self.assertNotIn("source", start["provenance"]["files"]["prompts"])
        self.assertIn("sha256", start["provenance"]["files"]["prompts"])
        self.assertEqual(start['diagnostics_revision'], 1)
        for name in ('normalise', 'fieldscore', 'segment', 'backends', 'grounding', 'stress'):
            self.assertIn('sha256', start['provenance']['files'][name])
        self.assertTrue(start["configuration"]["save_details"])
        self.assertEqual(start["plan"]["jobs"], 2)
        run_id = events[-1]["history_id"]
        self.assertNotIn(MARKER, self.client.get("/api/history/" + run_id + "?download=1").get_data(as_text=True))
        self.assertNotIn(MARKER, json.dumps(runhistory.get(run_id)))
        self.assertNotIn(MARKER.encode(), (self.root / "logs/run_history.sqlite3").read_bytes())
        self.assertFalse((self.root / "logs/runs.csv").exists())

    def test_disabled_option_emits_no_diagnostics(self):
        with patch.object(extractionlog, 'provenance', side_effect=AssertionError('unrequested hashing')):
            self.assertFalse(any(e["event"].startswith("detail_log")
                                 for e in self.run_stress(save_details=False)))

    def test_api_validates_boolean_and_ignores_option_for_non_extraction_modes(self):
        response = self.client.post("/api/stress/stream", json={"save_details": "yes"})
        self.assertEqual(response.status_code, 400)
        for mode in ("ocr", "classify"):
            self.assertFalse(app._stress_config({"mode": mode, "save_details": True})["save_details"])

    def test_export_delay_is_outside_benchmark_wall_clock(self):
        original = extractionlog.events
        def slow(*args, **kwargs):
            time.sleep(.1)
            yield from original(*args, **kwargs)
        with patch.object(extractionlog, "events", side_effect=slow):
            began = time.perf_counter()
            events = self.run_stress(jobs=1, warmup=0)
            elapsed = time.perf_counter() - began
        self.assertGreater(elapsed - events[-1]["report"]["wall_seconds"], .08)


if __name__ == "__main__":
    unittest.main()
