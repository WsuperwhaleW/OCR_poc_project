"""Image rechecks use runtime validation, never benchmark truth as an oracle."""
import copy
import time
import unittest
from unittest.mock import patch

from PIL import Image

import app
import extractionlog
import stress


GOOD = {"subtotal": "196,612.52", "vat_total": "13,762.88", "amount_incl_vat": "210,375.40"}
SHIFTED = {"subtotal": "13,762.88", "vat_total": "210,375.40", "amount_incl_vat": "210,375.40"}


def document(fields=None, **extra):
    out = {"fields": dict(fields or SHIFTED), "tokens": 5, "doc_types": ["INVOICE"],
           "classified": {"doc_types": ["INVOICE"], "fields": list(GOOD),
                          "mandatory": list(GOOD), "mandatory_items": []}, **extra}
    app._revalidate(out, app._fill_form(out), "")
    return out


def page(colour=0):
    value = Image.new("RGB", (8, 8), (colour, colour, colour))
    value._data_uri = app.image_data_uri(value)
    return value


def run(result, answer, enabled=True, images=None):
    images = images or [page()]
    asked = []
    def read(pages, cancel, rec, live, stats_out=None, job=0, case="", kept=None):
        rec.update(ocr_s=.1, pages=len(pages), ocr_tokens=2)
        kept.extend({"number": n, "text": "OCR page", "page": p, "detail": "low",
                     "looped": False, "truncated": False, "dry": None} for n,p in enumerate(pages,1))
        return ["OCR page"] * len(pages), False, False
    def ask(pages, form, ids):
        asked.append((pages, copy.deepcopy(form), list(ids)))
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)
    live = app._StressLive(1)
    cfg = {"mode": "full", "concurrency": 1, "extract_mode": "hybrid", "detail": "low",
           "fill_missing": False, "validation_retry": enabled}
    spec = {"index": 0, "case": "sol002", "submitted": time.perf_counter(), "detail": True}
    prepared = {"pages": {"sol002": images}, "sources": {"sol002": "source"}}
    with patch.object(app, "_STRESS_EVIDENCE", app._StressEvidence(10)), \
            patch.object(app, "_stress_read", side_effect=read), \
            patch.object(app, "extract_fields", return_value=result), \
            patch.object(app, "_fill_ask", side_effect=ask), \
            patch.object(app, "_score_fields"):
        rec = app._stress_job(spec, cfg, prepared, app.MultiCancel(), live)
        alerts = live.take_alerts()
        ids = {a['id'] for a in alerts if a['kind']=='validation'}
        bodies = [app.app.test_client().get('/api/stress/evidence/'+str(id)).get_json() for id in ids]
    return rec, alerts, bodies, asked


class ValidationRetryTests(unittest.TestCase):
    def test_shifted_totals_are_read_together_corrected_and_audited_with_costs(self):
        result=document(); result['fields']['seller_branch']='00016'
        rec,alerts,bodies,asked=run(result,{'fields':{**GOOD,'seller_branch':'99999'},'tokens':4,
            'steps':[{'id':'totals','keys':list(GOOD),'chats':[{'in_tokens':123,'out_tokens':4,'request':1}]}]})
        self.assertEqual(rec['status'],'ok')
        self.assertEqual(len(asked),1);self.assertEqual(asked[0][2],['totals'])
        self.assertTrue(asked[0][1]['validation_recheck'])
        self.assertEqual({k:result['fields'][k] for k in GOOD},GOOD)
        self.assertEqual(result['fields']['seller_branch'],'00016')
        self.assertEqual(rec['extract_tokens'],9)
        self.assertEqual((rec['validation_retry_jobs'],rec['validation_retry_asked'],rec['validation_retry_changed']), (1,3,2))
        self.assertEqual((rec['validation_retry_accepted'],rec['validation_retry_resolved']),(1,1))
        self.assertEqual([a['state'] for a in alerts if a['kind']=='validation'],['retrying','resolved'])
        audit=bodies[0]['fields'][0]['validation_rechecks'][0]
        self.assertEqual(audit['before'],SHIFTED);self.assertEqual(audit['after'],GOOD)
        self.assertTrue(audit['accepted']);self.assertEqual(audit['checks_candidate'],[])
        self.assertEqual(rec['agentic_chats'][0]['step'],'validation_image.totals')
        report=stress.report([rec],1,'full',1)['extract']
        self.assertEqual(report['validation_retry_changed'],2)
        self.assertEqual(report['validation_retry_tokens'],4)
        detail=extractionlog.job_detail(rec, {'index':0,'case':'sol002'}, {})
        self.assertTrue(detail['documents'][0]['validation_image_rechecks'][0]['accepted'])

    def test_toggle_off_leaves_existing_results_and_does_not_call_images(self):
        result=document();before=copy.deepcopy(result)
        rec,_,_,asked=run(result,{'fields':GOOD},enabled=False)
        self.assertFalse(asked);self.assertEqual(result['fields'],before['fields'])
        self.assertEqual(result['validation'],before['validation'])
        self.assertNotIn('validation_image_rechecks',result)
        self.assertNotIn('validation_retry_jobs',rec)

    def test_correct_runtime_validation_does_not_use_wrong_ground_truth_to_trigger(self):
        result=document(GOOD)
        result['field_score']={'scalars':{'rows':[{'path':'subtotal','actual':GOOD['subtotal'],
            'expected':'a different value','status':'wrong','required':True}]}}
        rec,_,_,asked=run(result,{'fields':GOOD})
        self.assertFalse(asked);self.assertNotIn('validation_retry_jobs',rec)

    def test_unchanged_mixed_vat_warning_is_kept_and_does_not_force_a_standard_rate(self):
        fields={'subtotal':'100','vat_total':'4','amount_incl_vat':'104'}
        result=document(fields)
        rec,alerts,_,asked=run(result,{'fields':fields,'tokens':3})
        self.assertEqual(result['fields'],fields);self.assertEqual(len(asked),1)
        self.assertEqual(rec['validation_retry_accepted'],0)
        self.assertEqual([a['state'] for a in alerts if a['kind']=='validation'][-1],'retry failed')
        self.assertIn('did not improve',result['validation_image_rechecks'][0]['reason'])

    def test_worse_incomplete_and_failed_answers_keep_originals_and_do_not_loop(self):
        for answer in ({'fields':{**GOOD,'vat_total':'not an amount'}},
                       {'fields':{'vat_total':'13,762.88'}}, {'fields':{**GOOD,'vat_total':''}},
                       {'error':'vision unsupported'}, RuntimeError('timeout')):
            with self.subTest(answer=answer):
                result=document();rec,_,_,asked=run(result,answer)
                self.assertEqual(result['fields'],SHIFTED);self.assertEqual(len(asked),1)
                self.assertEqual(rec['status'],'ok');self.assertEqual(rec['validation_retry_accepted'],0)
                self.assertFalse(result['validation_image_rechecks'][0]['accepted'])

    def test_only_the_suspicious_documents_own_pages_are_sent(self):
        first,second=page(10),page(20)
        result={'documents':[document(GOOD,document=1,pages=[1]),document(document=2,pages=[2])]}
        rec,_,_,asked=run(result,{'fields':GOOD},images=[first,second])
        self.assertEqual([item[0] for item in asked],[[second]])
        self.assertNotIn('validation_image_rechecks',result['documents'][0])
        self.assertEqual(rec['validation_retry_resolved'],1)

    def test_missing_multi_document_page_boundaries_never_fall_back_to_other_pages(self):
        result={'documents':[document(GOOD,document=1,pages=[1]),document(document=2)]}
        rec,_,_,asked=run(result,{'fields':GOOD},images=[page(10),page(20)])
        self.assertFalse(asked);self.assertEqual(rec['validation_retry_asked'],0)
        self.assertIn('page boundaries',result['documents'][1]['validation_image_rechecks'][0]['error'])

    def test_scalar_failure_uses_its_hybrid_step_without_overwriting_other_party_values(self):
        fields={'seller_tax_id':'123','buyer_tax_id':'0105559000000'}
        result={'fields':fields.copy(),'doc_types':['INVOICE'],'classified':{'doc_types':['INVOICE'],
            'fields':list(fields),'mandatory':list(fields),'mandatory_items':[]}}
        app._revalidate(result,app._fill_form(result),'')
        # An explicitly valid tax ID with its check digit; no truth is consulted.
        rec,_,_,asked=run(result,{'fields':{'seller_tax_id':'0105559000000','buyer_tax_id':'wrong'}})
        self.assertEqual(asked[0][2],['parties']);self.assertEqual(result['fields']['buyer_tax_id'],fields['buyer_tax_id'])
        self.assertEqual(rec['validation_retry_asked'],1)

    def test_image_prompt_preserves_empty_cells_and_does_not_receive_the_bad_ocr_values(self):
        seen=[]
        def agentic(text,status,form,only=None,mode='agentic'):
            seen.append((text,mode,only,app._image_fed()))
            yield {'event':'done','fields':{}}
        form=app._fill_form(document());form['validation_recheck']=True
        with patch.object(app.backends,'extract_status',return_value={'text_available':True}), \
                patch.object(app,'_extract_agentic',side_effect=agentic):
            app._fill_ask([page()],form,['totals'])
        self.assertEqual(seen[0][1:],('hybrid',['totals'],True))
        self.assertIn('blank cell must stay blank',seen[0][0])
        self.assertNotIn(SHIFTED['subtotal'],seen[0][0])


class ValidationToggleTests(unittest.TestCase):
    def config(self,**body):
        status={'available':True,'text_available':True,'model':'m','url':'http://x','kind':'ollama'}
        with patch.object(app,'llama_status',return_value=status),patch.object(app.backends,'extract_status',return_value=status):
            return app._stress_config(body)

    def test_opt_in_and_full_pipeline_only(self):
        self.assertFalse(self.config(mode='full')['validation_retry'])
        self.assertTrue(self.config(mode='full',validation_retry=True)['validation_retry'])
        for mode in ('ocr','extract','image'):
            self.assertFalse(self.config(mode=mode,validation_retry=True)['validation_retry'])


if __name__=='__main__':
    unittest.main()
