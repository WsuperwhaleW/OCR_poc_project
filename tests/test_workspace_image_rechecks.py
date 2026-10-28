"""Workspace and durable User jobs share the Stress image rescue policies."""
import copy
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import app
from userpipeline import EngineAdapter
from userstore import UserStore
from tests.test_stress_validation_retry import GOOD, SHIFTED, document, page
from tests.test_user_portal import StoreCase


def completed(value):
    yield {'event': 'extract_steps', 'total': 1}
    return copy.deepcopy(value)


class SharedRecheckTests(unittest.TestCase):
    def test_missing_then_validation_use_the_same_hybrid_steps_with_independent_switches(self):
        value = document({**SHIFTED, 'vat_total': ''})
        events = []
        with patch.object(app, '_fill_ask', side_effect=[{'fields': GOOD}, {'fields': GOOD}]) as ask:
            app.recheck_fields_from_images(value, [page()], ['OCR'], fill_missing=True,
                                           validation_retry=True, each=events.append)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(value['fields'], GOOD)
        self.assertEqual([c.args[2] for c in ask.call_args_list], [['totals'], ['totals']])
        self.assertEqual(value['image_filled'], ['vat_total'])
        self.assertTrue(value['validation_image_rechecks'][0]['accepted'])
        self.assertEqual({e['kind'] for e in events}, {'missing', 'validation'})
        self.assertTrue(all(f['state'] == 'resolved' for f in value['image_rechecks']['flags']))

    def test_disabled_options_do_not_use_images_or_modify_the_result(self):
        value = document(); before = copy.deepcopy(value)
        with patch.object(app, '_fill_ask') as ask:
            app.recheck_fields_from_images(value, [page()], ['OCR'])
        ask.assert_not_called()
        self.assertEqual(value, before)

    def test_missing_retry_does_not_send_other_documents_pages_when_boundaries_are_unknown(self):
        value = {'documents': [document(GOOD, document=1, pages=[1]),
                               document({**GOOD, 'vat_total': ''}, document=2)]}
        with patch.object(app, '_fill_ask') as ask:
            app.recheck_fields_from_images(value, [page(), page()], ['one', 'two'], fill_missing=True)
        ask.assert_not_called()
        self.assertEqual(value['documents'][1]['fields']['vat_total'], '')
        self.assertIn('page boundaries', value['image_rechecks']['flags'][0]['message'])


class WorkspaceRouteTests(unittest.TestCase):
    def request(self, stream, options):
        value = document()
        def read(pages, texts, stats, **kwargs):
            texts.append('OCR'); stats.append({})
        with patch.object(app, 'EXTRACT', True), \
                patch.object(app, 'prepare', return_value=([page()], 'low', None, None, {})), \
                patch.object(app, '_page_parallelism', return_value=1), \
                patch.object(app, 'stream_page', return_value=iter(['OCR'])), \
                patch.object(app, 'loop_retry_page', return_value=None), \
                patch.object(app, 'read_all_pages', side_effect=read), \
                patch.object(app, 'extract_fields', return_value=value), \
                patch.object(app, 'extract_fields_stream', side_effect=lambda *a, **k: completed(value)), \
                patch.object(app, '_fill_ask', return_value={'fields': GOOD}) as ask, \
                patch.object(app, 'log_run'), patch.object(app, 'evaluate_if_known', return_value=None):
            response = app.app.test_client().post('/ocr/api/ocr' + ('/stream' if stream else ''), data=options)
            if stream:
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertFalse([e for e in events if e['event'] == 'error'], events)
                result = next(e for e in events if e['event'] == 'fields') if options.get('extract') != '0' else None
            else:
                self.assertEqual(response.status_code, 200, response.text)
                events, result = [], response.json.get('extracted')
        return result, events, ask.call_count

    def test_blocking_and_streaming_apply_the_opt_in_policy_before_returning_fields(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                result, events, calls = self.request(stream, {'validation_retry': '1'})
                self.assertEqual(result['fields'], GOOD); self.assertEqual(calls, 1)
                if stream:
                    kinds = [e['event'] for e in events]
                    self.assertLess(kinds.index('image_recheck'), kinds.index('fields'))

    def test_an_ordinary_api_call_or_ocr_only_never_enables_the_recheck_implicitly(self):
        for stream in (False, True):
            result, _, calls = self.request(stream, {})
            self.assertEqual(result['fields'], SHIFTED); self.assertEqual(calls, 0)
            _, _, calls = self.request(stream, {'extract': '0', 'validation_retry': '1', 'fill_missing': '1'})
            self.assertEqual(calls, 0)


class UserImageRecheckTests(StoreCase):
    def job(self, value, **options):
        cfg = self.store.save_settings(options)
        batch = self.store.create_batch(1, cfg)
        image = Image.new('RGB', (20, 20), 'white'); content = io.BytesIO()
        image.save(content, format='TIFF', save_all=True, append_images=[image])
        job = self.store.add_job(batch, 'upload_01', 'two.tiff', content.getvalue())
        job['result'] = {'text': 'one\ntwo', 'pages': ['one', 'two'], 'warnings': []}
        return job

    def test_saved_flags_persist_and_each_job_keeps_its_own_settings_snapshot(self):
        job = self.job(None, fill_missing=False, validation_retry=True)
        self.store.save_settings({'fill_missing': True, 'validation_retry': False})
        reopened = UserStore(self.temp.name, 32)
        self.assertTrue(reopened.settings()['fill_missing'])
        self.assertFalse(reopened.settings()['validation_retry'])
        self.assertEqual((job['configuration']['fill_missing'], job['configuration']['validation_retry']), (False, True))
        for flags in ({'fill_missing': 'true'}, {'validation_retry': 1}):
            with self.assertRaises(ValueError):
                self.store.save_settings(flags)

    def test_user_uses_the_original_only_when_needed_and_retains_audits_without_benchmark_cache(self):
        value = {'documents': [document(GOOD, document=1, pages=[1]), document(document=2, pages=[2])]}
        job = self.job(value, fill_missing=False, validation_retry=True, schema_first=True)
        cache = set(app._jobs)
        with patch.object(app.backends, 'extract_status', return_value={'text_available': True}), \
                patch.object(app, 'extract_fields_stream', side_effect=lambda *a, **k: completed(value)) as extract, \
                patch.object(app, 'prepare_input', wraps=app.prepare_input) as prepare, \
                patch.object(app, '_fill_ask', return_value={'fields': GOOD}) as ask, \
                patch.object(self.store, 'update', wraps=self.store.update) as update:
            result = EngineAdapter(app).extraction(job, self.store)
        self.assertEqual(prepare.call_count, 1); self.assertFalse(prepare.call_args.kwargs['cache'])
        self.assertEqual(len(ask.call_args.args[0]), 1)
        self.assertTrue(extract.call_args.kwargs['schema_first'])
        self.assertEqual(result['pages'], ['one', 'two'])
        self.assertEqual(result['extracted']['documents'][1]['fields'], GOOD)
        audit = result['extracted']['documents'][1]['validation_image_rechecks'][0]
        self.assertEqual(audit['pages'], [2]); self.assertTrue(audit['accepted'])
        self.assertEqual(set(app._jobs), cache)
        self.assertTrue(any('กำลังอ่านภาพซ้ำ' in c.kwargs.get('progress', '') for c in update.call_args_list))

    def test_no_findings_or_disabled_switches_do_not_prepare_images(self):
        for value, flags in ((document(GOOD), {'fill_missing': True, 'validation_retry': True}),
                             (document(), {'fill_missing': False, 'validation_retry': False})):
            job = self.job(value, **flags)
            with patch.object(app.backends, 'extract_status', return_value={'text_available': True}), \
                    patch.object(app, 'extract_fields_stream', side_effect=lambda *a, **k: completed(value)), \
                    patch.object(app, 'prepare_input') as prepare:
                EngineAdapter(app).extraction(job, self.store)
            prepare.assert_not_called()

    def test_failed_image_preparation_keeps_ocr_and_original_fields(self):
        job = self.job(None, fill_missing=False, validation_retry=True)
        with patch.object(app.backends, 'extract_status', return_value={'text_available': True}), \
                patch.object(app, 'extract_fields_stream', side_effect=lambda *a, **k: completed(document())), \
                patch.object(app, 'prepare_input', side_effect=ValueError('image unavailable')):
            result = EngineAdapter(app).extraction(job, self.store)
        self.assertEqual(result['text'], 'one\ntwo')
        self.assertEqual(result['extracted']['fields'], SHIFTED)
        self.assertIn('image unavailable', result['warnings'][0])


if __name__ == '__main__':
    unittest.main()
