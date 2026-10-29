"""Original Agentic is frozen; Hybrid carries tuning, with isolated requests/logs."""
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app
import config
import prompts
import runhistory
import runlog
from tests.test_summary_extras import row


STATUS = {'kind': 'vllm', 'model': 'mock', 'url': 'http://mock:8001',
          'available': True, 'text_available': True, 'reason': '', 'text_reason': ''}


class VariantTests(unittest.TestCase):
    def test_agentic_matches_the_frozen_pre_tuning_commit(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/agentic_baseline_sha256.json').read_text())
        self.assertTrue(fixture['revision'].startswith('9880dce'))
        for profile, expected in fixture['profiles'].items():
            codes, extras = json.loads(profile)
            payload = {'task': prompts.EXTRACT_STEP_TASK,
                       'steps': prompts.steps_for_types(codes, extras, mode='agentic')}
            actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(actual, expected, profile)

    def test_only_hybrid_groups_related_steps(self):
        for code, original, hybrid in [('INVOICE', 6, 4), ('RECEIPT', 6, 4),
                                       ('RTV_SHIP_CREDIT', 5, 3), ('WHT_CERTIFICATE', 10, 6)]:
            a = prompts.steps_for_types([code], extras=False, mode='agentic')
            h = prompts.steps_for_types([code], extras=False, mode='hybrid')
            self.assertEqual((len(a), len(h)), (original, hybrid))
            self.assertEqual({k for s in a for k in s['keys']}, {k for s in h for k in s['keys']})
        self.assertEqual(app._steps_for(['total_incl'], ['INVOICE'], False)[0]['keys'],
                         ('amount_incl_vat',))
        self.assertEqual(set(app._steps_for(['total_incl'], ['INVOICE'], False, mode='hybrid')[0]['keys']),
                         {'subtotal', 'vat_total', 'amount_incl_vat'})

    def test_document_driver_uses_the_selected_task_schema_and_result_label(self):
        for mode, count, task in [('agentic', 6, prompts.EXTRACT_STEP_TASK),
                                  ('hybrid', 4, prompts.HYBRID_STEP_TASK)]:
            requests = []

            def chat(message, cap, status, schema=None):
                requests.append((message, schema))
                return json.dumps({key: '' for key in schema['properties']}), False, 10

            events = []
            with patch.object(app, '_chat', side_effect=chat), \
                 patch.object(app, 'AGENTIC_PARALLEL', False), patch.object(app, 'TABLE_EXTRACT', False):
                gen = app._extract_document('Invoice with no amounts', mode, None, None,
                                            ['INVOICE'], STATUS, extras=False, schema_first=True)
                while True:
                    try:
                        events.append(next(gen))
                    except StopIteration as stop:
                        result = stop.value
                        break
            self.assertEqual(result['mode'], mode)
            self.assertTrue(result['schema_first'])
            self.assertEqual(len(requests), count)
            self.assertEqual(len(events[0]['steps']), count)
            first = prompts.steps_for_types(['INVOICE'], extras=False, mode=mode)[0]
            framing = app._framing({'doc_types': ['INVOICE']}, mode)
            expected = task.format(title=first['title'], context=prompts.type_line(framing),
                                   skeleton=first['skeleton'], rules=prompts.step_rules(
                                       first, framing, app.TYPE_FRAMING_BULLETS))
            self.assertIn(expected, requests[0][0])
            self.assertTrue(all(schema is not None for _, schema in requests))
            self.assertEqual(runlog._extract_cells({'extracted': result})['extract_mode'], mode)

    def test_concurrent_variants_keep_their_own_steps_and_parallelism(self):
        def ask(content, step, status, collect=None):
            values = {key: '' for key in step['keys']}
            return values, [json.dumps(values)], False, 1, False

        def run(mode):
            form = app._field_set('Invoice', doc_type=['INVOICE'], extras=False)
            return app._drain(app._extract_agentic('Invoice', {**STATUS, 'reachable': True},
                                                  form, mode=mode))

        with patch.object(app, '_ask_step', side_effect=ask), \
             patch.object(app, 'AGENTIC_PARALLEL', True), \
             patch.object(app, 'AGENTIC_PARALLEL_MAX', 0), \
             patch.object(app, 'VLLM_PREFIX_WARMUP', False), ThreadPoolExecutor(max_workers=2) as pool:
            original, hybrid = list(pool.map(run, ['agentic', 'hybrid']))
        self.assertEqual((original['mode'], hybrid['mode']), ('agentic', 'hybrid'))
        self.assertEqual((original['parallel'], hybrid['parallel']), (6, 4))
        self.assertIn('total_incl', {s['id'] for s in original['steps']})
        self.assertNotIn('total_incl', {s['id'] for s in hybrid['steps']})


class ApiAndLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        context = patch.object(config, 'LOG_DIR', Path(directory.name))
        context.start()
        self.addCleanup(context.stop)
        self.client = app.app.test_client()

    def test_mode_setting_accepts_all_three_and_refuses_unknown_values(self):
        for mode in ('single', 'agentic', 'hybrid'):
            response = self.client.post('/api/extract/mode', json={'mode': mode})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['mode'], mode)
            self.assertEqual(self.client.get('/api/extract/mode').get_json()['mode'], mode)
        self.assertEqual(self.client.post('/api/extract/mode', json={'mode': 'unknown'}).status_code, 400)

    def test_extract_api_passes_hybrid_and_schema_first(self):
        def stream(*args, **kwargs):
            yield {'event': 'extract_steps'}
            return {'mode': 'hybrid', 'fields': {}}

        for url in ('/api/extract', '/api/extract/stream'):
            with patch.object(app, 'log_extract'), \
                 patch.object(app, 'extract_fields', return_value={'mode': 'hybrid', 'fields': {}}) as blocking, \
                 patch.object(app, 'extract_fields_stream', side_effect=stream) as streaming:
                response = self.client.post(url, json={'text': 'A', 'mode': 'hybrid', 'schema_first': True})
                self.assertEqual(response.status_code, 200)
                response.get_data()
                called = streaming if url.endswith('/stream') else blocking
                self.assertEqual(called.call_args.args[1], 'hybrid')
                self.assertTrue(called.call_args.kwargs['schema_first'])

    def test_stress_history_and_jobs_keep_hybrid_separate(self):
        def stream(cfg, cancel):
            yield json.dumps({'event': 'plan', 'extract_mode': cfg['extract_mode']})
            yield json.dumps({'event': 'done', 'report': {}})

        with patch.object(app.backends, 'extract_status', return_value=STATUS), \
             patch.object(app, '_stress_run', side_effect=stream):
            response = self.client.post('/api/stress/stream', json={
                'mode': 'extract', 'extract_mode': 'hybrid', 'schema_first': True,
                'history_monitor': False})
            events = [json.loads(line) for line in response.data.splitlines()]
        saved = runhistory.get(events[-1]['history_id'])
        self.assertEqual(saved['configuration']['extract_mode'], 'hybrid')
        self.assertEqual(saved['plan']['extract_mode'], 'hybrid')
        self.assertTrue(saved['configuration']['schema_first'])
        with patch.object(app.backends, 'extract_status', return_value=STATUS), \
             patch.object(app, 'extract_fields', return_value={'mode': 'hybrid', 'fields': {}}) as extract:
            cfg = app._stress_config({'mode': 'extract', 'extract_mode': 'hybrid',
                                     'schema_first': True, 'cases': ['sol001'], 'jobs': 1})
            list(app._stress_run(cfg, app.MultiCancel()))
        self.assertEqual(extract.call_args.args[1], 'hybrid')
        self.assertTrue(extract.call_args.kwargs['schema_first'])

    def test_summary_never_pools_hybrid_with_agentic_or_relabels_older_rows(self):
        old = row(mode='agentic', seconds=100)
        rows = [old, row(mode='hybrid', seconds=10),
                row(mode='hybrid', seconds=20, extract_schema_first='1'), row(mode='single', seconds=30)]
        with runlog.view(window=0):
            result = runlog.presentation(rows, bias=False)
        shapes = {(r['mode'], r['schema_first']):r for r in result['extract_shapes']}
        self.assertEqual(len(shapes), 4)
        self.assertEqual(shapes[('agentic', False)]['seconds']['mean'], 100)
        self.assertEqual(shapes[('hybrid', False)]['seconds']['mean'], 10)
        self.assertEqual(shapes[('hybrid', True)]['seconds']['mean'], 20)
        self.assertEqual(old['extract_mode'], 'agentic')
        self.assertNotIn('extract_schema_first', old)


if __name__ == '__main__':
    unittest.main()
