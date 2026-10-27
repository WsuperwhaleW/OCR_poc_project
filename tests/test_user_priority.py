"""User portal jobs before every other model request, and the stress-test alert."""
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import app
import priority
import workspace
from tests.test_user_portal import FakeAdapter, until
from userpipeline import UserService
from userstore import UserStore


def user_workspace():
    """Run as a User job does: its own workspace with `user_pipeline` set."""
    ws = workspace.Workspace('userjob_test')
    ws.put('user_pipeline', True)
    return workspace.use(ws)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(priority.set_user_busy, False)

    def held(self, fn):
        """Run `fn` on a thread; return (thread, event set once fn returns)."""
        done = threading.Event()
        thread = threading.Thread(target=workspace.bind(lambda: (fn(), done.set())), daemon=True)
        thread.start()
        return thread, done

    def test_nothing_waits_while_no_user_job_is_pending(self):
        self.assertEqual(priority.wait_turn(), 0.0)

    def test_benchmark_request_waits_until_user_jobs_are_done(self):
        priority.set_user_busy(True)
        _, done = self.held(priority.wait_turn)
        self.assertFalse(done.wait(.3))
        self.assertEqual(priority.state()['waiting'], 1)
        priority.set_user_busy(False)
        self.assertTrue(done.wait(2))
        self.assertEqual(priority.state()['waiting'], 0)

    def test_user_request_never_waits(self):
        priority.set_user_busy(True)
        with user_workspace():
            self.assertEqual(priority.wait_turn(), 0.0)

    def test_stop_reaches_a_held_request(self):
        priority.set_user_busy(True)
        cancel = threading.Event()
        _, done = self.held(lambda: priority.wait_turn(cancel))
        self.assertFalse(done.wait(.2))
        cancel.set()
        self.assertTrue(done.wait(2))

    def test_held_request_reports_progress_so_it_is_not_flagged_stalled(self):
        priority.set_user_busy(True)
        beats = []
        _, done = self.held(lambda: priority.wait_turn(progress=lambda: beats.append(1)))
        until(lambda: beats, timeout=3)
        priority.set_user_busy(False)
        self.assertTrue(done.wait(2))

    def test_stall_clock_counts_only_from_when_the_gate_opened(self):
        priority.set_user_busy(True)
        self.assertEqual(priority.quiet_for(), 0.0)
        priority.set_user_busy(False)
        self.assertLess(priority.quiet_for(), 1.0)

    def test_extraction_request_from_a_stress_worker_is_held(self):
        sent = threading.Event()
        priority.set_user_busy(True)
        with patch.object(app, '_batcher_for', return_value=None), \
                patch.object(app, '_guards_extraction', return_value=False), \
                patch.object(app, '_plain_post', side_effect=lambda *a: sent.set()):
            _, done = self.held(lambda: app._post_dispatch('http://x/v1/chat/completions', {}, {}))
            self.assertFalse(sent.wait(.3))
            priority.set_user_busy(False)
            self.assertTrue(done.wait(2))
            self.assertTrue(sent.is_set())

    def test_extraction_request_from_a_user_job_is_not_held(self):
        priority.set_user_busy(True)
        with patch.object(app, '_batcher_for', return_value=None), \
                patch.object(app, '_guards_extraction', return_value=False), \
                patch.object(app, '_plain_post', return_value='reply'), user_workspace():
            self.assertEqual(app._post_dispatch('http://x/v1/chat/completions', {}, {}), 'reply')


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = UserStore(self.temp.name, 32)

    def test_pending_user_jobs_hold_the_gate_until_they_finish(self):
        release = threading.Event()
        self.addCleanup(release.set)
        adapter = FakeAdapter()
        normal = adapter.ocr
        adapter.ocr = lambda job, store: (release.wait(5), normal(job, store))[1]
        service = UserService(self.store, adapter)
        self.addCleanup(service.close)
        batch = self.store.create_batch(1, self.store.settings())
        job = self.store.add_job(batch, 'upload_01', 'one.pdf', b'x')
        service.start()
        until(priority.user_busy)
        release.set()
        until(lambda: self.store.get(job['id'])['status'] == 'completed')
        until(lambda: not priority.user_busy())


class StatusTests(unittest.TestCase):
    def test_user_page_is_told_about_a_running_stress_test(self):
        client = app.app.test_client()
        self.assertEqual(client.get('/api/user/status').json['stress'], [])
        began, finish = threading.Event(), threading.Event()
        self.addCleanup(finish.set)

        def stress_run():
            priority.stress_begin(mode='full', concurrency=8)
            began.set()
            finish.wait(5)
            priority.stress_end()
        thread = threading.Thread(target=stress_run, daemon=True)
        thread.start()
        self.assertTrue(began.wait(2))
        runs = client.get('/api/user/status').json['stress']
        self.assertEqual([(r['mode'], r['concurrency']) for r in runs], [('full', 8)])
        finish.set(); thread.join(2)
        self.assertEqual(client.get('/api/user/status').json['stress'], [])

    def test_a_stress_thread_that_died_is_not_reported(self):
        thread = threading.Thread(target=lambda: priority.stress_begin(mode='ocr'))
        thread.start(); thread.join()
        self.assertEqual(priority.stress_runs(), [])


if __name__ == '__main__':
    unittest.main()
