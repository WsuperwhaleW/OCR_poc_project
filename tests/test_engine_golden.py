"""The real engine, end to end, against a fake model server: the refactor guard.

Built 2026-11-04 for the worker-service plan. `app.py` is about to be split into
a shared engine plus `service_app.py` and the benchmark; nothing about what the
User pipeline sends to the model or returns may change while that happens. Only
the model HTTP is fake (`tests/fakemodel.py`); image preparation, OCR streaming,
grouping, Hybrid extraction, image rechecks, validation and User JSON all run.

The golden file holds, per input file, the job's status and result (minus
timings, timestamps and server URLs), its User JSON, and the sorted SHA-256 of
every chat request body. Refresh it only on purpose:
    UPDATE_ENGINE_GOLDEN=1 python -m unittest tests.test_engine_golden
"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import runlog
import userjson
from tests.fakemodel import OCR_MODEL, TEXT_MODEL, FakeModelServer
from userpipeline import EngineAdapter, UserService
from userstore import UserStore

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / 'fixtures' / 'engine_golden.json'
FILES = ('mockOcr/invoice_sol002.pdf', 'mockOcr/receipt_tax_invoice_sol003.pdf',
         'mockOcr/wht_certificate_sol017.png', 'mockOcr/multiple_sol015.pdf')
FINISHED = ('completed', 'partial', 'failed', 'interrupted')
VOLATILE = {'seconds', 'elapsed', 'wall', 'latency', 'busy', 'url', 'urls', 'server_time',
            'tokens_per_second', 'decode_tps', 'prefill_tps'}


def _volatile(key):
    # Clock readings and how many requests happened to overlap: both vary run to
    # run with thread scheduling, never with what the engine asks or answers.
    key = str(key).lower()
    return (key in VOLATILE or any(part in key for part in ('time', 'timing', 'seconds', 'peak'))
            or key.endswith(('_s', '_ms', '_at', '_per_second', '_tps')))


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in sorted(value.items()) if not _volatile(k)}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def run_pipeline(files=FILES, timeout=240):
    """Run `files` through UserService with the real engine; return the projection."""
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(runlog, 'LOG_DIR', Path(directory)), \
            patch.object(runlog, 'LOG_PATH', Path(directory) / 'runs.csv'), \
            patch.object(runlog, 'DELETED_PATH', Path(directory) / 'runs.deleted.csv'), \
            FakeModelServer(OCR_MODEL) as ocr, FakeModelServer(TEXT_MODEL) as text:
        store = UserStore(Path(directory) / 'user', 32)
        store.save_settings({**store.settings(), 'ocr_url': ocr.url, 'extraction_url': text.url})
        adapter = EngineAdapter(app)
        cfg = adapter.snapshot(store.settings())
        cfg['prompt_profiles'] = store.prompt_profiles()
        cfg['prompt_sets'] = store.prompt_set_snapshot()
        batch = store.create_batch(len(files), cfg)
        jobs = [store.add_job(batch, 'golden_%04d' % index, Path(name).name, (ROOT / name).read_bytes())
                for index, name in enumerate(files)]
        service = UserService(store, adapter)
        service.start()
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if all(store.get(job['id'])['status'] in FINISHED for job in jobs):
                    break
                time.sleep(0.2)
        finally:
            service.close()
        out = {'files': {}, 'requests': {'ocr': sorted(ocr.requests), 'extraction': sorted(text.requests)}}
        for job in jobs:
            done = store.get(job['id'])
            out['files'][done['filename']] = {
                'status': done['status'], 'error': done['error'], 'result': sanitize(done['result']),
                'user_json': sanitize(userjson.convert(done).get('data'))}
        text_out = json.dumps(out, ensure_ascii=False, sort_keys=True)
        for url, name in ((ocr.url, 'http://fake-ocr'), (text.url, 'http://fake-extract')):
            text_out = text_out.replace(url, name)
        return json.loads(text_out)


class EngineGoldenTests(unittest.TestCase):
    maxDiff = None

    def test_real_engine_matches_the_golden_run(self):
        actual = run_pipeline()
        if os.environ.get('UPDATE_ENGINE_GOLDEN') == '1' or not GOLDEN.exists():
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(json.dumps(actual, ensure_ascii=False, indent=1, sort_keys=True) + '\n', 'utf-8')
            if os.environ.get('UPDATE_ENGINE_GOLDEN') != '1':
                self.skipTest('golden file created; run again to compare')
        expected = json.loads(GOLDEN.read_text('utf-8'))
        for name in expected['files']:
            with self.subTest(file=name):
                self.assertEqual(actual['files'].get(name), expected['files'][name])
        self.assertEqual(actual['requests'], expected['requests'],
                         'the engine sent different model requests than the golden run')

    def test_every_golden_file_runs_through_extraction(self):
        golden = json.loads(GOLDEN.read_text('utf-8')) if GOLDEN.exists() else run_pipeline()
        for name, entry in golden['files'].items():
            with self.subTest(file=name):
                self.assertIn(entry['status'], ('completed', 'partial'))
                self.assertTrue(entry['result'].get('extracted'))
                self.assertTrue(entry['user_json'].get('documents'))


if __name__ == '__main__':
    unittest.main()
