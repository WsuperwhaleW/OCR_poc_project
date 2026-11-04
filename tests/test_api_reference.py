"""The API tab's list (apireference.py) matches the routes the service really has.

Built 2026-11-04 with the API tab (worker-service plan, step 2b): a route added
without an entry, or an entry marked live for a route that is gone, fails here,
so the page never documents an API that does not exist or misses one that does.
"""
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import apireference

# The reference route never opens the User store; the data dir is only a guard.
with patch.dict(os.environ, {'OCR_USER_DATA_DIR': tempfile.mkdtemp(prefix='apiref_')}):
    import service_app


def flask_path(rule):
    """`/api/user/jobs/<job_id>/preview/pages/<int:page>` → `/api/user/jobs/{job_id}/preview/pages/{page}`."""
    return re.sub(r'<(?:[a-z]+:)?([a-z_]+)>', r'{\1}', rule)


def routes():
    out = set()
    for rule in service_app.app.url_map.iter_rules():
        if rule.rule.startswith(('/api/user/', '/api/schema', '/api/v1/')):
            for method in rule.methods - {'HEAD', 'OPTIONS'}:
                out.add((method, flask_path(rule.rule)))
    return out


class ApiReferenceTests(unittest.TestCase):
    def test_every_live_route_is_documented_and_every_live_entry_exists(self):
        documented = {(e['method'], e['path']) for e in apireference.ENDPOINTS if e['status'] == 'live'}
        self.assertEqual(routes() - documented, set(), 'routes without an API tab entry')
        self.assertEqual(documented - routes(), set(), 'entries marked live with no route')

    def test_only_the_callback_is_still_planned(self):
        # /api/v1 is built (2026-11-04); the callback comes with step 6.
        self.assertEqual([e['id'] for e in apireference.ENDPOINTS if e['status'] != 'live'], [])
        self.assertEqual(apireference.CALLBACK['status'], 'planned')

    def test_entries_are_complete_and_ids_unique(self):
        ids = [e['id'] for e in apireference.ENDPOINTS]
        self.assertEqual(len(ids), len(set(ids)))
        groups = {group['id'] for group in apireference.GROUPS}
        for entry in apireference.ENDPOINTS:
            with self.subTest(entry['id']):
                self.assertIn(entry['group'], groups)
                self.assertIn(entry['method'], ('GET', 'POST', 'PUT', 'DELETE'))
                self.assertTrue(entry['title'] and entry['summary'] and entry['auth'])
                self.assertTrue(entry['responses'])
                for response in entry['responses']:
                    self.assertIn(response['kind'], ('json', 'file', 'empty'))
                    json.dumps(response['body'], allow_nan=False)
                # Every {name} in the path is a documented path parameter.
                names = set(re.findall(r'{([a-z_]+)}', entry['path']))
                self.assertEqual(names, {p['name'] for p in entry['params'] if p['in'] == 'path'})

    def test_snippets_carry_method_path_and_the_right_auth(self):
        for entry in apireference.ENDPOINTS:
            with self.subTest(entry['id']):
                snippets = apireference.snippets(entry)
                self.assertEqual(set(snippets), {'curl', 'python', 'javascript'})
                path = apireference.example_path(entry)
                self.assertNotIn('{', path)
                for text in snippets.values():
                    self.assertIn('{BASE}' + path, text)
                self.assertIn('-X ' + entry['method'], snippets['curl'])
                write = entry['group'] == 'user' and entry['method'] != 'GET'
                self.assertEqual('X-OCR-User-Token' in snippets['curl'], write)
                if entry['group'] == 'v1' and entry['path'] != '/api/v1/health':
                    self.assertIn('Authorization: Bearer', snippets['curl'])
                if entry.get('multipart'):
                    self.assertIn('-F ', snippets['curl'])
                    self.assertIn('FormData', snippets['javascript'])
                    self.assertNotIn('Content-Type', snippets['javascript'])

    def test_python_snippets_compile(self):
        for entry in apireference.ENDPOINTS:
            with self.subTest(entry['id']):
                compile(apireference.snippets(entry)['python'], entry['id'], 'exec')
        compile(apireference.CALLBACK['verify']['python'], 'callback', 'exec')

    def test_callback_body_is_the_result_body(self):
        result = next(e for e in apireference.ENDPOINTS if e['id'] == 'v1-result')
        self.assertEqual(apireference.CALLBACK['body'],
                         next(r['body'] for r in result['responses'] if r['code'] == 200))

    def test_route_answers_in_written_order(self):
        response = service_app.app.test_client().get('/api/user/api-reference')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual([group['id'] for group in body['groups']], ['v1', 'user'])
        self.assertEqual(len(body['endpoints']), len(apireference.ENDPOINTS))
        self.assertTrue(all('snippets' in entry for entry in body['endpoints']))
        # Not key-sorted: an example reads in the order it was written.
        first = next(e for e in body['endpoints'] if e['id'] == 'v1-status')['responses'][0]['body']
        self.assertEqual(list(first)[:2], ['job_id', 'client_ref'])


if __name__ == '__main__':
    unittest.main()
