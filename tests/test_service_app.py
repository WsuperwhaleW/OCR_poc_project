"""`service_app.py` runs the User pipeline with no benchmark code at all.

Built 2026-11-04 for the worker-service plan (step 2): production starts
`service_app.py`, which must import the shared `engine` and the User pages only.
The check runs in a fresh interpreter with every benchmark module blocked
(`sys.modules[name] = None` makes any import of it fail), then sends one real
file through `/api/user` against the fake model server -- so a stray benchmark
import anywhere on the User path, even inside a function, fails here. The file
holds several documents (the merge path) and must end as the golden run of the
shared engine did (`tests/fixtures/engine_golden.json`).
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = 'multiple_sol015.pdf'

BENCHMARK = ('app', 'stress', 'randomtest', 'concurrencysweep', 'vllmbatch', 'monitor',
             'machine', 'runhistory', 'blame', 'fixscore', 'fieldscore', 'compare', 'test')

SCRIPT = textwrap.dedent('''
    import json, sys, time
    for name in %(blocked)r:
        sys.modules[name] = None
    sys.path.insert(0, %(root)r)
    import service_app
    from tests.fakemodel import OCR_MODEL, TEXT_MODEL, FakeModelServer

    out = {'rules': sorted(rule.rule for rule in service_app.app.url_map.iter_rules())}
    client = service_app.app.test_client()
    out['root'] = [client.get('/').status_code, client.get('/').headers.get('Location')]
    out['pages'] = {path: client.get(path).status_code
                    for path in ('/user', '/OCR/user', '/api/schema', '/api/user/settings')}
    with FakeModelServer(OCR_MODEL) as ocr, FakeModelServer(TEXT_MODEL) as text:
        store = service_app.get_user_service().store
        store.save_settings({**store.settings(), 'ocr_url': ocr.url, 'extraction_url': text.url})
        batch = client.post('/api/user/batches', json={'file_count': 1})
        out['batch'] = batch.status_code
        with open(%(file)r, 'rb') as handle:
            job = client.post('/api/user/jobs', data={'batch_id': batch.get_json()['batch_id'],
                                                      'upload_id': 'service_0001',
                                                      'file': (handle, %(name)r)})
        out['submit'] = job.status_code
        job_id = job.get_json()['job']['id']
        deadline = time.time() + 240
        while time.time() < deadline:
            status = client.get('/api/user/jobs/' + job_id).get_json()['job']['status']
            if status in ('completed', 'partial', 'failed', 'interrupted'):
                break
            time.sleep(0.2)
        out['status'] = status
        out['error'] = client.get('/api/user/jobs/' + job_id).get_json()['job'].get('error')
        out['user_json'] = client.get('/api/user/jobs/%%s/user-json' %% job_id).get_json()['data'].get('schemaVersion')
        service_app.get_user_service().stop() if hasattr(service_app.get_user_service(), 'stop') else None
    out['loaded'] = sorted(name for name in %(blocked)r if sys.modules.get(name) is not None)
    print('RESULT ' + json.dumps(out))
''')


class ServiceAppTests(unittest.TestCase):
    def test_runs_a_user_job_with_benchmark_modules_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            script = SCRIPT % {'blocked': BENCHMARK, 'root': str(ROOT),
                               'file': str(ROOT / 'mockOcr' / FILE), 'name': FILE}
            env = {**os.environ, 'OCR_LOG_DIR': directory,
                   'OCR_USER_DATA_DIR': str(Path(directory) / 'user'), 'PYTHONIOENCODING': 'utf-8'}
            done = subprocess.run([sys.executable, '-c', script], cwd=directory, env=env,
                                  capture_output=True, text=True, encoding='utf-8', timeout=300)
            lines = [line for line in done.stdout.splitlines() if line.startswith('RESULT ')]
            self.assertTrue(lines, done.stdout[-2000:] + done.stderr[-4000:])
            out = json.loads(lines[-1][len('RESULT '):])

        # Only the User pages, their API and the shared schema: no benchmark route, no /static.
        for rule in out['rules']:
            self.assertTrue(rule == '/' or rule == '/api/schema' or rule.startswith(('/user', '/api/user/')), rule)
        self.assertEqual(out['root'], [302, '/user/'])
        self.assertEqual(out['pages'], {'/user': 200, '/OCR/user': 200, '/api/schema': 200,
                                        '/api/user/settings': 200})
        self.assertEqual((out['batch'], out['submit']), (201, 202))
        golden = json.loads((ROOT / 'tests' / 'fixtures' / 'engine_golden.json').read_text(encoding='utf-8'))
        self.assertEqual(out['status'], golden['files'][FILE]['status'], out['error'])
        self.assertEqual(out['user_json'], '7.0')
        self.assertEqual(out['loaded'], [])


if __name__ == '__main__':
    unittest.main()
