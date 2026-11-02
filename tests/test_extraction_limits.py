"""All files/workspaces use the same extraction admission gate per endpoint."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import app
import extractionlimit
import workspace


class ExtractionLimitTests(unittest.TestCase):
    def setUp(self):
        self.limits = extractionlimit.ServerLimits()
        guard = patch.object(app, '_EXTRACTION_LIMITS', self.limits)
        guard.start(); self.addCleanup(guard.stop)

    def test_different_files_workspaces_and_models_share_the_same_server_cap(self):
        active = peak = 0
        lock = threading.Lock()
        def post(*args):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.015)
            with lock:
                active -= 1
            return app._Reply(200, {})
        def file(index):
            with workspace.use(workspace.Workspace('limit_file_%03d' % index)):
                status = {'url':'http://GPU:8001/', 'kind':'vllm', 'model':str(index % 2)}
                return app._timed_post('http://gpu:8001/v1/chat/completions', {}, status)
        with patch.object(app, 'EXTRACT_REQUEST_CONCURRENCY', 3), \
                patch.object(app, '_post_once', side_effect=post), ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(file, range(24)))
        self.assertEqual(len(results), 24)
        self.assertEqual(peak, 3)
        self.assertEqual(len(self.limits.servers), 1)
        self.assertEqual(next(iter(self.limits.servers.values())).active, 0)

    def test_different_endpoints_do_not_block_each_other(self):
        gate = threading.Barrier(2)
        def request(url):
            with self.limits.request(url, 1):
                gate.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(request, ['http://gpu:8001','http://gpu:8002']))

    def test_endpoint_spelling_and_model_routes_use_a_single_key(self):
        aliases = ['HTTP://GPU/', 'http://gpu:80', 'http://gpu/v1/chat/completions',
                   'http://gpu/api/chat', 'gpu/']
        self.assertEqual(len({extractionlimit.server_key(url) for url in aliases}), 1)
        self.assertNotEqual(extractionlimit.server_key('http://gpu/a'), extractionlimit.server_key('http://gpu/b'))

    def test_waiting_user_is_admitted_before_background_work(self):
        order = []
        gate = self.limits.gate('http://gpu', 1)
        def request(name, rank):
            with self.limits.request('http://gpu', 1, priority=rank):
                order.append(name)
        def wait_for_waiters(count):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with gate.condition:
                    if len(gate.waiters) == count:
                        return
                time.sleep(.005)
            self.fail('request did not enter admission queue')
        with ThreadPoolExecutor(max_workers=2) as pool:
            with self.limits.request('http://gpu', 1):
                background = pool.submit(request, 'background', 10)
                wait_for_waiters(1)
                user = pool.submit(request, 'user', 0)
                wait_for_waiters(2)
            background.result(timeout=2); user.result(timeout=2)
        self.assertEqual(order, ['user', 'background'])

    def test_failure_and_cancelled_waiter_return_capacity_without_stopping_other_files(self):
        with self.assertRaisesRegex(ValueError, 'offline'):
            with self.limits.request('http://gpu', 1):
                raise ValueError('offline')
        cancel = threading.Event()
        gate = self.limits.gate('http://gpu', 1)
        def blocked():
            with self.limits.request('http://gpu', 1, cancel=cancel):
                self.fail('cancelled file must not send another request')
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.limits.request('http://gpu', 1):
                future = pool.submit(blocked)
                cancel.set()
                with self.assertRaises(extractionlimit.Cancelled):
                    future.result(timeout=2)
            with self.limits.request('http://gpu', 1):
                self.assertEqual(gate.active, 1)
        self.assertEqual(gate.active, 0)
        self.assertEqual(gate.waiters, [])

    def test_context_retry_reuses_one_shared_slot_and_preserves_payload(self):
        payload = {'max_tokens':1000}
        gate = self.limits.gate('http://gpu', 1)
        seen = []
        def post(*args):
            seen.append(gate.active)
            return app._Reply(400, text='maximum context length') if len(seen) == 1 else app._Reply(200, {})
        with patch.object(app, 'EXTRACT_REQUEST_CONCURRENCY', 1), \
                patch.object(app, '_post_once', side_effect=post), \
                patch.object(app, '_context_overflow', return_value=True), \
                patch.object(app, '_fit_max_tokens', return_value={'max_tokens':500}):
            response = app._timed_post('http://gpu/chat', payload, {'url':'http://gpu'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, [1,1])
        self.assertEqual(payload, {'max_tokens':1000})
        self.assertEqual(gate.active, 0)

    def test_waiting_reports_progress_without_starting_or_leaking_a_model_request(self):
        cancel = threading.Event()
        heard = threading.Event()
        with self.limits.request('http://gpu', 1):
            def wait():
                with self.assertRaises(extractionlimit.Cancelled):
                    with self.limits.request('http://gpu', 1, cancel=cancel, on_wait=heard.set):
                        self.fail('must not be admitted while full')
            thread = threading.Thread(target=wait)
            thread.start()
            self.assertTrue(heard.wait(2))
            cancel.set(); thread.join(2)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
