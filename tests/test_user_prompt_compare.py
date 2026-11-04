"""Compare must execute User snapshots, without changing team or Benchmark state."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import config
import prompts
import userpromptcompare
import workspace


class UserPromptCompareTests(unittest.TestCase):
    def test_in_process_adapter_compares_both_scopes_and_restores_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            original_requests = object()
            original_log = app.log_extract
            original_output = Path(directory) / 'original'
            counts = {}
            def fake_model(text,*args,**kwargs):
                self.assertTrue(workspace.get('user_pipeline',False))
                table = app.steps_for_types(['DELIVERY_TAX_INVOICE','TAX_INVOICE'],False,mode='hybrid')
                return {'fields_asked':[key for step in table for key in step['keys']],
                        'groups':[step['id'] for step in table]}
            for variant in ('before','after'):
                with self.subTest(variant=variant):
                    args = types.SimpleNamespace(app='http://127.0.0.1:5000',mode=None,user_prompt_defaults=variant)
                    module = types.SimpleNamespace(requests=original_requests,
                        scoring=types.SimpleNamespace(OUT=original_output),say=lambda *a:None)
                    def run(opts):
                        response = module.requests.post(opts.app+'/api/extract',json={'text':'ใบส่งของ/ใบกำกับภาษี','mode':'hybrid'})
                        self.assertTrue(response.ok,response.text)
                        counts[variant] = response.json()
                        self.assertEqual(opts.mode,'hybrid')
                        self.assertTrue((module.scoring.OUT / 'prompt-snapshot.json').exists())
                        return 0
                    module.run = run
                    with patch.object(config,'BASE_DIR',Path(directory)), \
                            patch.object(app,'extract_fields',side_effect=fake_model), \
                            patch('requests.sessions.Session.request',side_effect=AssertionError('No model/network')):
                        self.assertEqual(userpromptcompare.run(args,module),0)
                    self.assertIs(module.requests,original_requests)
                    self.assertEqual(module.scoring.OUT,original_output)
                    self.assertIs(app.log_extract,original_log)
                    self.assertEqual(args.app,'http://127.0.0.1:5000')
                    self.assertIsNone(args.mode)
            # Before migration 2 the type used the base form, which Hybrid no longer
            # widens with the WHT certificate's keys on a classified page.
            self.assertEqual(len(counts['before']['fields_asked']),
                             len(set(prompts.DEFAULT_FIELDS)-set(prompts.DOC_TYPE_FIELDS['WHT_CERTIFICATE'])))
            self.assertEqual(len(counts['after']['fields_asked']),13)
            self.assertFalse(any(g.startswith('wht') for g in counts['before']['groups']))
            self.assertFalse(any(g.startswith('wht') for g in counts['after']['groups']))

    def test_failed_comparison_also_restores_the_transport_and_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            original_requests, original_log = object(), app.log_extract
            args = types.SimpleNamespace(app='original',mode=None,user_prompt_defaults='after')
            module = types.SimpleNamespace(requests=original_requests,scoring=types.SimpleNamespace(OUT=Path(directory)),say=lambda *a:None)
            def fail(args):
                raise RuntimeError('model unavailable')
            module.run = fail
            with patch.object(config,'BASE_DIR',Path(directory)), self.assertRaisesRegex(RuntimeError,'model unavailable'):
                userpromptcompare.run(args,module)
            self.assertIs(module.requests,original_requests)
            self.assertIs(app.log_extract,original_log)
            self.assertEqual(args.app,'original')


if __name__ == '__main__':
    unittest.main()
