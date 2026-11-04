"""The Service dashboard's numbers (servicestats.py), from jobs with known times.

Built 2026-11-04 with the Service tab (worker-service plan, step 2b). Jobs are
written straight into a UserStore with hand-made timelines and results, so every
count, bucket and percentile below can be checked by hand.
"""
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

import engine
import extractionlimit
import servicestats
import userweb
from userstore import UserStore

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def at(**delta):
    return (NOW - timedelta(**delta)).isoformat()


class Seeded:
    def __init__(self, directory):
        self.store = UserStore(Path(directory), 32)
        self.count = 0

    def job(self, status, created, updated, timeline=None, result=None, error='', retry_of=''):
        self.count += 1
        job_id = '%032x' % self.count
        with self.store.connect() as db:
            db.execute('INSERT INTO batches VALUES (?,?,?)', ('b%031x' % self.count, 1, '{}'))
            db.execute('INSERT INTO jobs(id,batch_id,upload_id,filename,created_at,updated_at,status,progress,error,'
                       'configuration,result,timeline,retry_of) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (job_id, 'b%031x' % self.count, 'upload_%04d' % self.count, 'file%d.pdf' % self.count,
                        created, updated, status, '', error, '{}', json.dumps(result or {}),
                        json.dumps(timeline or {}), retry_of))
        return job_id


def seed(directory):
    s = Seeded(directory)
    ids = {}
    ids['c1'] = s.job('completed', at(hours=1), at(minutes=59, seconds=30),
                      {'queued_ocr': {'started_at': at(hours=1)}, 'ocr': {'started_at': at(minutes=59, seconds=58)}},
                      {'timing': {'ocr': 10, 'extraction': 15}, 'extracted': {'doc_types': ['INVOICE']}})
    ids['c2'] = s.job('completed', at(minutes=30), at(minutes=29),
                      {'queued_ocr': {'started_at': at(minutes=30)}, 'ocr': {'started_at': at(minutes=29, seconds=50)}},
                      {'timing': {'ocr': 20, 'extraction': 25}, 'extracted': {'doc_types': ['INVOICE', 'RECEIPT']}},
                      retry_of='f' * 32)
    ids['p1'] = s.job('partial', at(hours=3), at(hours=2, minutes=45), None, {'extracted': {'doc_types': []}})
    ids['f1'] = s.job('failed', at(minutes=11), at(minutes=10),
                      {'ocr': {'started_at': at(minutes=11), 'finished_at': at(minutes=10), 'outcome': 'failed'},
                       'failed': {'started_at': at(minutes=10)}}, {}, error='boom\nsecond line')
    s.job('completed', at(days=3), at(days=3), None, {'timing': {'ocr': 1}})
    s.job('completed', at(days=10), at(days=10), None, {})
    ids['q1'] = s.job('queued_ocr', at(minutes=5), at(minutes=5), {'queued_ocr': {'started_at': at(minutes=5)}})
    # Waiting for extraction since 2 min ago, though uploaded an hour ago.
    ids['q2'] = s.job('queued_extraction', at(hours=1), at(minutes=2),
                      {'queued_extraction': {'started_at': at(minutes=2)}})
    ids['r1'] = s.job('ocr', at(minutes=1), at(minutes=1), {'ocr': {'started_at': at(minutes=1)}})
    return s.store, ids


def service_for(store):
    running, finished = Future(), Future()
    finished.set_result(None)
    return SimpleNamespace(store=store, lock=threading.Lock(),
                           active={'ocr': {running, finished}, 'extraction': set()})


def engine_stub(limits=None, probes=None):
    probes = probes or {}
    return SimpleNamespace(backends=SimpleNamespace(last_probe=lambda url: probes.get(url)),
                           _EXTRACTION_LIMITS=limits or extractionlimit.ServerLimits(),
                           EXTRACT_REQUEST_CONCURRENCY=128)


CHECKS = {}


def fake_convert(job):
    return CHECKS[job['id']]


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store, self.ids = seed(self.directory.name)
        CHECKS.clear()
        CHECKS.update({self.ids['c1']: {'schema_valid': True, 'warnings': [], 'missing_fields': ['a']},
                       self.ids['c2']: {'schema_valid': False, 'warnings': [], 'missing_fields': []},
                       self.ids['p1']: {'schema_valid': True, 'warnings': ['x', 'y'], 'missing_fields': ['a', 'b']}})
        patcher = patch.object(servicestats.userjson, 'convert', side_effect=fake_convert)
        self.convert = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)

    def overview(self, **engine_args):
        return servicestats.Overview(service_for(self.store), engine_stub(**engine_args),
                                     started=(NOW - timedelta(hours=2)).timestamp())

    def test_queue_counts_live_runs_and_oldest_wait(self):
        queue = self.overview().compute(NOW)['queue']
        self.assertEqual(queue['open_jobs'], 3)
        ocr, extraction = queue['stages']['ocr'], queue['stages']['extraction']
        self.assertEqual((ocr['waiting'], ocr['running'], ocr['running_recorded'], ocr['concurrency']), (1, 1, 1, 50))
        self.assertEqual((extraction['waiting'], extraction['running']), (1, 0))
        self.assertEqual(ocr['oldest_wait_seconds'], 300)
        # From when it joined the extraction queue, not from the upload.
        self.assertEqual(extraction['oldest_wait_seconds'], 120)
        self.assertEqual(queue['oldest_wait_seconds'], 300)

    def test_throughput_windows_and_hourly_buckets(self):
        t = self.overview().compute(NOW)['throughput']
        self.assertEqual(t['last_24h'], {'completed': 2, 'partial': 1, 'failed': 1, 'total': 4})
        self.assertEqual(t['last_7d']['total'], 5)
        self.assertEqual(t['last_hour'], {'completed': 2, 'partial': 0, 'failed': 1, 'total': 3})
        self.assertEqual(t['retries_24h'], 1)
        self.assertEqual(len(t['hours']), 24)
        # The last bar is the hour in progress; 11:xx is the one before it.
        self.assertEqual(t['hours'][-1]['hour'], NOW.isoformat())
        self.assertEqual((t['hours'][-2]['completed'], t['hours'][-2]['failed']), (2, 1))
        self.assertEqual(t['hours'][-4]['partial'], 1)
        self.assertEqual(sum(h['completed'] + h['partial'] + h['failed'] for h in t['hours']), 4)

    def test_speed_percentiles(self):
        speed = self.overview().compute(NOW)['speed']
        self.assertEqual(speed['queue'], {'count': 2, 'p50': 2.0, 'p95': 10.0, 'max': 10.0})
        self.assertEqual(speed['ocr'], {'count': 2, 'p50': 10, 'p95': 20, 'max': 20})
        self.assertEqual(speed['extraction'], {'count': 2, 'p50': 15, 'p95': 25, 'max': 25})
        self.assertEqual(speed['total'], {'count': 3, 'p50': 60.0, 'p95': 900.0, 'max': 900.0})
        self.assertEqual(servicestats.percentiles(list(range(1, 101))), {'count': 100, 'p50': 50, 'p95': 95, 'max': 100})
        self.assertIsNone(servicestats.percentiles([None]))

    def test_quality_rates_doc_types_and_user_json(self):
        quality = self.overview().compute(NOW)['quality']
        self.assertEqual((quality['finished'], quality['completed_rate'], quality['partial_rate'], quality['failed_rate']),
                         (4, .5, .25, .25))
        self.assertEqual(quality['unknown_type_rate'], .3333)
        self.assertEqual(quality['top_doc_types'], [{'code': 'INVOICE', 'count': 2}, {'code': 'RECEIPT', 'count': 1}])
        self.assertEqual(quality['user_json'], {'checked': 3, 'schema_valid_rate': .6667, 'avg_warnings': .67,
                                                'avg_missing_fields': 1.0})

    def test_user_json_is_converted_once_per_job_version(self):
        overview = self.overview()
        overview.compute(NOW)
        overview.compute(NOW)
        self.assertEqual(self.convert.call_count, 3)

    def test_failures_show_first_error_line_and_stage(self):
        failures = self.overview().compute(NOW)['failures']
        self.assertEqual(len(failures), 1)
        self.assertEqual((failures[0]['id'], failures[0]['error'], failures[0]['stage']), (self.ids['f1'], 'boom', 'ocr'))

    def test_alerts(self):
        out = self.overview().compute(NOW)
        texts = [a['text'] for a in out['alerts']]
        self.assertTrue(any('ยังไม่ถูกตรวจ' in t for t in texts))
        self.assertTrue(any('ล้มเหลว 1 จาก 3' in t for t in texts))
        self.assertFalse(any('รอคิวนานกว่า' in t for t in texts))
        later = self.overview().compute(NOW + timedelta(minutes=10))
        self.assertTrue(any('รอคิวนานกว่า 15 นาที' in a['text'] for a in later['alerts']))

    def test_model_health_from_the_last_probe_only(self):
        cfg = self.store.settings()
        probes = {cfg['ocr_url']: (4.2, {'reachable': True, 'kind': 'vllm', 'model': None,
                                         'models': [{'name': 'typhoon-ocr'}], 'reason': ''}),
                  cfg['extraction_url']: (9.0, {'reachable': False, 'kind': None, 'models': [],
                                                'reason': 'connection refused'})}
        health = self.overview(probes=probes).compute(NOW)['health']
        self.assertEqual(health['models']['ocr']['model'], 'typhoon-ocr')
        self.assertTrue(health['models']['ocr']['available'])
        self.assertEqual(health['models']['ocr']['probe_age_seconds'], 4.2)
        self.assertFalse(health['models']['extraction']['available'])
        self.assertEqual(health['models']['extraction']['reason'], 'connection refused')
        self.assertTrue(health['app_version'])
        self.assertGreater(health['data_bytes'], 0)

    def test_request_gates_snapshot(self):
        limits = extractionlimit.ServerLimits()
        with limits.request('http://127.0.0.1:8001/v1/chat/completions', 4, priority=0):
            requests = self.overview(limits=limits).compute(NOW)['requests']
        self.assertEqual(requests['extraction'], [{'url': 'http://127.0.0.1:8001', 'limit': 4, 'active': 1, 'peak': 1,
                                                   'waiting_user': 0, 'waiting_background': 0}])
        self.assertIsNone(requests['ocr'])

    def test_cached_for_ttl(self):
        overview = self.overview()
        first = overview.get()
        self.assertIs(overview.get(), first)
        overview.cached = (0.0, first)
        self.assertIsNot(overview.get(), first)

    def test_planned_cards(self):
        planned = self.overview().compute(NOW)['planned']
        self.assertEqual([(p['id'], p['step']) for p in planned], [('capacity', 5), ('callbacks', 6), ('retention', 7)])


class RouteTests(unittest.TestCase):
    def test_route_answers_without_asking_a_model_server(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OCR_USER_DATA_DIR': directory}), \
                patch.object(engine.backends, 'probe', side_effect=AssertionError('no network')):
            app = Flask('overview_test')
            userweb.register(app, engine)
            client = app.test_client()
            response = client.get('/api/user/service/overview')
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(set(body), {'generated_at', 'alerts', 'health', 'queue', 'requests', 'throughput',
                                         'speed', 'quality', 'failures', 'planned'})
            self.assertEqual(body['health']['data_dir'], str(Path(directory)))
            app.extensions['user_service'].close()


if __name__ == '__main__':
    unittest.main()
