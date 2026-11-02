"""One PDF can fill several independent forms without mixing pages or requests."""
import copy
import json
import re
import threading
import time
import unittest
from unittest.mock import patch

import app
import extractionlog
import prompts
import servertime
import userpipeline
import workspace


VLLM = {'kind': 'vllm', 'url': 'http://mock', 'model': 'mock', 'text_available': True}


def segments(count):
    keys = prompts.fields_for_types(['INVOICE'])
    return [{'pages': [i * 2 - 1, i * 2], 'codes': ['INVOICE'],
             'text': 'INVOICE\nDOC%02d\n' % i + '\n'.join('DOC%02d_%s' % (i, key) for key in keys)}
            for i in range(1, count + 1)]


def records(parts):
    return [app._segment_record(part, i, len(parts)) for i, part in enumerate(parts, 1)]


class DocumentParallelTests(unittest.TestCase):
    def setUp(self):
        for target, name, value in [(app, 'EXTRACT_DOCUMENT_PARALLEL_MAX', 4),
                (app, 'EXTRACT_REQUEST_CONCURRENCY', 8), (app, 'TABLE_EXTRACT', False),
                (app, 'VLLM_PREFIX_WARMUP', False), (app, 'AGENTIC_PARALLEL', True),
                (app, 'AGENTIC_PARALLEL_MAX', 0), (servertime.settings, 'SERVER_TIMING_METRICS', False)]:
            guard = patch.object(target, name, value)
            guard.start(); self.addCleanup(guard.stop)
        guard = patch('requests.sessions.Session.request', side_effect=AssertionError('No real inference'))
        guard.start(); self.addCleanup(guard.stop)

    def run_documents(self, parts, status=VLLM, each=None):
        events = []
        result = app._drain(app._extract_documents(parts, records(parts), 'hybrid', None,
                            None, ['INVOICE'], status, False, False, False),
                            lambda event: (events.append(event), each(event) if each else None))
        return result, events

    def test_documents_overlap_finish_out_of_order_and_merge_their_own_pages_in_order(self):
        gate = threading.Barrier(4)
        lock = threading.Lock()
        active = peak = 0
        def extract(text, mode, case, only, doc_type, status, record, *args):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            yield {'event': 'classified', 'doc_types': ['INVOICE']}
            if record['document'] <= 4:
                gate.wait(timeout=3)
            time.sleep(.04 if record['document'] == 1 else .005)
            with lock:
                active -= 1
            return {'fields': {'document_number': re.search(r'DOC\d+', text)[0]}, 'seconds': .04}
        parts = segments(7)
        with patch.object(app, '_extract_document', side_effect=extract):
            result, events = self.run_documents(parts)
        self.assertEqual(peak, 4)
        self.assertEqual([d['fields']['document_number'] for d in result['documents']],
                         ['DOC%02d' % i for i in range(1, 8)])
        self.assertEqual([d['pages'] for d in result['documents']], [p['pages'] for p in parts])
        done = [e['document'] for e in events if e['event'] == 'document_done']
        self.assertEqual(sorted(done), list(range(1, 8)))
        self.assertNotEqual(done[0], 1)
        for event in events:
            if event.get('event') == 'classified':
                self.assertEqual(event['pages'], parts[event['document'] - 1]['pages'])

    def test_nested_hybrid_steps_share_a_request_cap_and_keep_sequential_answers(self):
        parts = segments(5)
        requests, active, peak = [], 0, 0
        labels = []
        lock = threading.Lock()
        def post(url, payload, status):
            nonlocal active, peak
            with app._watched():
                with lock:
                    active += 1; peak = max(peak, active)
                    requests.append(payload['text'])
                time.sleep(.015)
                servertime.Clock(status).finish(client_seconds=.015)
                with lock:
                    active -= 1
            return app._Reply(200, {})
        def ask(content, step, status, collect=None):
            doc = re.search(r'DOC\d+', content)[0]
            app._timed_post('http://mock/chat', {'text': doc}, status)
            values = {key: doc + '_' + key for key in step['keys']}
            raw = json.dumps(values)
            if collect is not None:
                collect.append(raw)
            return values, [raw], False, 7, False
        prior = app._progress_swap(lambda event, request, what: labels.append(what) if event == 'begin' else None)
        self.addCleanup(app._progress_swap, prior)
        with patch.object(app, '_ask_step', side_effect=ask), patch.object(app, '_post_once', side_effect=post):
            previous = servertime.open_tally()
            parallel, _ = self.run_documents(parts)
            tally = servertime.close_tally(previous)
            self.assertEqual(parallel['document_requests_peak'], 8)
            self.assertEqual(peak, 8)
            self.assertEqual(tally['requests'], len(requests))
            self.assertEqual(tally['requests'], len(parts) * len(prompts.steps_for_types(['INVOICE'], False, 'hybrid')))
            with patch.object(app, 'EXTRACT_DOCUMENT_PARALLEL_MAX', 1):
                serial, _ = self.run_documents(parts)
        for old, new in zip(serial['documents'], parallel['documents']):
            for key in ('fields', 'raw', 'tokens', 'pages', 'grounding', 'validation', 'classified'):
                self.assertEqual(new.get(key), old.get(key), key)
            self.assertTrue(all(value.startswith('DOC%02d_' % new['document'])
                                for value in new['fields'].values() if isinstance(value, str) and value))
        self.assertEqual(serial['tokens'], parallel['tokens'])
        self.assertEqual({int(re.search(r'document (\d+)', label)[1]) for label in labels if label.startswith('document ')},
                         set(range(1, 6)))

    def test_request_limit_one_does_not_deadlock_nested_steps_and_returns_slots(self):
        budget = app._DocumentRequests()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with budget.request():
                raise RuntimeError('failed')
        self.assertEqual(budget.active, 0)
        with budget.request():
            self.assertEqual(budget.active, 1)
        with patch.object(app, 'EXTRACT_REQUEST_CONCURRENCY', 1), \
                patch.object(app, '_extract_document', side_effect=self.fake_document):
            result, _ = self.run_documents(segments(3))
        self.assertEqual(result['document_parallel'], 1)
        self.assertEqual(len(result['documents']), 3)

    @staticmethod
    def fake_document(text, *args, **kwargs):
        yield {'event': 'classified', 'doc_types': ['INVOICE']}
        return {'fields': {'document_number': re.search(r'DOC\d+', text)[0]}, 'seconds': .01}

    def test_failure_is_scoped_to_one_document_and_later_documents_still_finish(self):
        def extract(text, *args):
            if 'DOC02\n' in text:
                raise ValueError('document two failed')
            yield from ()
            return {'fields': {'document_number': re.search(r'DOC\d+', text)[0]}}
        with patch.object(app, '_extract_document', side_effect=extract):
            result, events = self.run_documents(segments(6))
        self.assertEqual(result['documents_failed'], 1)
        self.assertEqual(result['documents'][1]['error'], 'document two failed')
        self.assertEqual(result['documents'][-1]['fields']['document_number'], 'DOC06')
        self.assertEqual(len([e for e in events if e['event'] == 'document_done']), 6)

    def test_closing_stream_does_not_start_the_remaining_documents(self):
        begun = []
        def extract(text, *args):
            begun.append(text)
            time.sleep(.02)
            yield {'event': 'classified'}
            return {'fields': {}}
        with patch.object(app, '_extract_document', side_effect=extract):
            gen = app._extract_documents(segments(150), records(segments(150)), 'hybrid', None,
                                         None, ['INVOICE'], VLLM)
            next(gen)  # Parallel configuration; no workers have started yet.
            next(gen)  # First worker's event.
            gen.close()
        self.assertLessEqual(len(begun), 4)

    def test_non_vllm_and_disabled_paths_keep_sequential_events(self):
        for status, limit in [({**VLLM, 'kind': 'ollama'}, 4), (VLLM, 1)]:
            with self.subTest(status=status, limit=limit), \
                    patch.object(app, 'EXTRACT_DOCUMENT_PARALLEL_MAX', limit), \
                    patch.object(app, '_extract_document', side_effect=self.fake_document):
                result, events = self.run_documents(segments(3), status)
            self.assertNotIn('document_parallel', result)
            self.assertEqual([e['document'] for e in events if e['event'] == 'document_done'], [1, 2, 3])

    def test_workspace_and_user_prompt_snapshot_are_carried_into_each_document(self):
        seen = []
        def extract(text, *args):
            seen.append((workspace.current_id(), copy.deepcopy(workspace.get('user_prompt_profiles', {}))))
            yield {'event': 'classified'}
            return {'fields': {}}
        with workspace.use(None):
            ws = workspace.enter('parallel_doc_user_snapshot')
            ws.put('user_prompt_profiles', {'INVOICE': {'revision': 7}})
            with patch.object(app, '_extract_document', side_effect=extract):
                self.run_documents(segments(4))
        self.assertEqual(seen, [('parallel_doc_user_snapshot', {'INVOICE': {'revision': 7}})] * 4)

    def test_execution_metadata_survives_user_compaction_and_detail_download(self):
        with patch.object(app, '_extract_document', side_effect=self.fake_document):
            result, _ = self.run_documents(segments(3))
        compact = userpipeline.compact_extraction(result)
        self.assertEqual(compact['document_parallel'], 3)
        detail = extractionlog.job_detail({'_extraction_result': result}, {'index': 0}, None)
        self.assertEqual(detail['document_execution']['extraction_request_limit'], 8)
        json.dumps(detail)

    def test_file_entry_point_uses_parallel_documents_but_single_document_stays_unchanged(self):
        def split(text, pages, status):
            yield from ()
            return parts, 'rules'
        for count in (1, 5):
            parts = segments(count)
            with self.subTest(count=count), \
                    patch.object(app.backends, 'extract_status', return_value=VLLM), \
                    patch.object(app, '_segment_stream', side_effect=split), \
                    patch.object(app, '_extract_document', side_effect=self.fake_document) as extract:
                result = app.extract_fields('whole PDF', 'hybrid', pages=['page'] * count)
            self.assertEqual(result['documents_found'], count)
            if count == 1:
                self.assertNotIn('document_parallel', result)
                self.assertEqual(extract.call_args.args[0], parts[0]['text'])
            else:
                self.assertEqual(result['document_parallel'], 4)


if __name__ == '__main__':
    unittest.main()
