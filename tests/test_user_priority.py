"""User portal jobs first by vLLM's priority scheduling, and the stress-test alert."""
import threading
import unittest
from unittest.mock import MagicMock, patch

import app
import backends
import priority
import workspace

CHAT = 'http://gpu:8000/v1/chat/completions'
VLLM = {'kind': 'vllm'}


def user_workspace():
    """Run as a User job does: its own workspace with `user_pipeline` set."""
    ws = workspace.Workspace('userjob_test')
    ws.put('user_pipeline', True)
    return workspace.use(ws)


def reply(status=200, text=''):
    return MagicMock(status_code=status, text=text)


class PriorityFieldTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(priority._fcfs.clear)
        patcher = patch.object(backends, '_post', return_value=reply())
        self.post = patcher.start()
        self.addCleanup(patcher.stop)

    def sent(self):
        return [call.kwargs['json'] for call in self.post.call_args_list]

    def test_benchmark_chat_request_is_sent_with_a_later_priority(self):
        backends.http_post(CHAT, VLLM, json={'messages': []})
        self.assertEqual(self.sent(), [{'messages': [], 'priority': priority.BACKGROUND}])

    def test_user_chat_request_keeps_the_default_priority(self):
        with user_workspace():
            backends.http_post(CHAT, VLLM, json={'messages': []})
        self.assertEqual(self.sent(), [{'messages': []}])

    def test_user_priority_reaches_worker_threads(self):
        with user_workspace():
            worker = threading.Thread(target=workspace.bind(
                lambda: backends.http_post(CHAT, VLLM, json={'messages': []})))
            worker.start(); worker.join(2)
        self.assertEqual(self.sent(), [{'messages': []}])

    def test_the_callers_payload_is_not_changed(self):
        payload = {'messages': []}
        backends.http_post(CHAT, VLLM, json=payload)
        self.assertEqual(payload, {'messages': []})

    def test_only_vllm_chat_requests_are_tagged(self):
        backends.http_post(CHAT, {'kind': 'ollama'}, json={'messages': []})
        backends.http_post('http://gpu:8000/tokenize', VLLM, json={'prompt': 'x'})
        self.assertEqual(self.sent(), [{'messages': []}, {'prompt': 'x'}])

    def test_fcfs_server_is_asked_again_without_priority_and_remembered(self):
        self.post.side_effect = [
            reply(400, '{"message":"Got priority 10 but Priority scheduling is not enabled."}'),
            reply(), reply()]
        self.assertEqual(backends.http_post(CHAT, VLLM, json={'messages': []}).status_code, 200)
        backends.http_post(CHAT, VLLM, json={'messages': []})
        self.assertEqual(self.sent(), [{'messages': [], 'priority': priority.BACKGROUND},
                                       {'messages': []}, {'messages': []}])

    def test_another_400_is_returned_as_it_is(self):
        self.post.return_value = reply(400, 'maximum context length is 16384')
        self.assertEqual(backends.http_post(CHAT, VLLM, json={'messages': []}).status_code, 400)
        self.assertEqual(len(self.sent()), 1)


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
