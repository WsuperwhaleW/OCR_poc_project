"""Seek a real start, leave the unknown prefix to the preceding worker."""
import re
import json
import threading
import time
import unittest
from unittest.mock import patch

import app
import segmentparallel
import workspace


STATUS = {'kind':'vllm', 'url':'http://group-test', 'model':'mock'}


def pages_for(starts, count=100):
    return ['receipt\nInvoice No DOC%03d\nbody page %d' %
            (sum(start <= n for start in starts), n) for n in range(1, count + 1)]


class ParallelGroupingTests(unittest.TestCase):
    def setUp(self):
        for name, value in [('SEGMENT_PARALLEL_MAX',4), ('SEGMENT_PARALLEL_MIN_PAGES',32),
                            ('SEGMENT_PARALLEL_OVERLAP',3), ('SEGMENT_MAX_PAGES',5),
                            ('SEGMENT_CHAT_MAX_ASKS',200), ('EXTRACT_REQUEST_CONCURRENCY',128)]:
            guard = patch.object(app,name,value); guard.start(); self.addCleanup(guard.stop)
        self.active = self.peak = self.calls = 0
        self.lock = threading.Lock()
        owner = self
        def ask(chat, number, page):
            with owner.lock:
                owner.active += 1; owner.peak = max(owner.peak,owner.active); owner.calls += 1
            time.sleep(.002)
            current = re.search(r'DOC\d+',page)[0]
            first = re.search(r'DOC\d+',chat._first['content'])[0]
            with owner.lock:
                owner.active -= 1
            return current == first, current
        self.ask = ask
        self.real_ask = app._SegmentChat._ask
        guard=patch.object(app._SegmentChat,'_ask',ask); guard.start(); self.addCleanup(guard.stop)
        guard=patch('requests.sessions.Session.request',side_effect=AssertionError('No model/network'))
        guard.start(); self.addCleanup(guard.stop)

    def run_pages(self, pages, status=STATUS):
        events=[]
        result=app._drain(app._segment_stream(app.join_page_texts(pages),pages,status),events.append)
        return result[0],events

    def assert_coverage(self, groups, count):
        self.assertEqual([n for g in groups for n in g['pages']],list(range(1,count+1)))

    def test_worker_at_23_starts_at_24_or_25_only_after_a_confirmed_cut(self):
        for start in (24,25):
            with self.subTest(start=start):
                pages=pages_for([1,start,61])
                groups,events=self.run_pages(pages)
                self.assertEqual([g['pages'][0] for g in groups],[1,start,61])
                self.assertEqual(groups[0]['pages'][-1],start-1)
                metadata=groups[0]['_parallel_execution']
                self.assertEqual(metadata['worker_starts'],[1,23,48,73])
                self.assertEqual(metadata['worker_ownership'][1]['starts_at'],start)
                self.assertFalse(metadata['fallback'])
                self.assert_coverage(groups,100)
                steps=[e for e in events if e['event']=='segment_page']
                self.assertEqual([s['page'] for s in steps],list(range(1,101)))
                self.assertEqual(steps[start-2]['document'],1)
                self.assertEqual(steps[start-1]['document'],2)

    def test_many_documents_run_in_parallel_and_match_a_forward_walk(self):
        pages=pages_for(list(range(1,151,7)),150)
        parallel,events=self.run_pages(pages)
        self.assertGreater(self.peak,1)
        self.assertFalse(parallel[0]['_parallel_execution']['fallback'])
        with patch.object(app,'SEGMENT_PARALLEL_MAX',1):
            forward,_=self.run_pages(pages)
        for actual, expected in zip(parallel,forward):
            for key in ('pages','text','codes','heading','confidence','certain','joins','start_reason'):
                self.assertEqual(actual[key],expected[key],key)
        self.assertEqual(len(parallel),len(forward))
        self.assert_coverage(parallel,150)
        self.assertTrue(any(e['event']=='segment_progress' for e in events))

    def test_one_long_document_is_not_cut_at_worker_seeds_and_leader_keeps_question_reserve(self):
        with patch.object(app,'SEGMENT_CHAT_MAX_ASKS',110):
            groups,_=self.run_pages(pages_for([1]))
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['pages'],list(range(1,101)))
        meta=groups[0]['_parallel_execution']
        self.assertLessEqual(meta['questions_sent'],110)
        self.assertEqual(meta['confirmed_starts'],[1])
        self.assertTrue(all(row['starts_at'] is None for row in meta['worker_ownership'][1:]))
        self.assertTrue(all(join['certain'] for join in groups[0]['joins']))

    def test_handoffs_follow_page_order_when_workers_finish_in_different_orders(self):
        starts = [1, 15, 24, 47, 49, 73, 90]
        for slow_region in (1, 23, 48, 73):
            with self.subTest(slow_region=slow_region):
                def delayed(chat, number, page):
                    opening = int(re.search(r'body page (\d+)', chat._first['content'])[1])
                    if opening == slow_region:
                        time.sleep(.004)
                    return self.ask(chat, number, page)
                with patch.object(app._SegmentChat, '_ask', delayed):
                    groups, _ = self.run_pages(pages_for(starts))
                self.assert_coverage(groups, 100)
                self.assertEqual([g['pages'][0] for g in groups], starts)
                self.assertFalse(groups[0]['_parallel_execution']['fallback'])

    def test_printed_first_page_at_a_worker_seed_is_a_confirmed_start(self):
        pages = pages_for([1, 23, 48, 73])
        for first, following in ((1, 23), (23, 48), (48, 73), (73, 101)):
            for n in range(first, following):
                pages[n - 1] += '\nPage %d of %d' % (n - first + 1, following - first)
        groups, _ = self.run_pages(pages)
        self.assert_coverage(groups, 100)
        self.assertEqual([g['pages'][0] for g in groups], [1, 23, 48, 73])
        self.assertFalse(groups[0]['_parallel_execution']['fallback'])
        self.assertTrue(all(g['certain'] for g in groups))

    def test_conflicting_handoff_falls_back_without_an_invented_cut(self):
        def ask(chat, number, page):
            if number==24 and 'Page 23' in chat._first['content']:
                return False,'DOC001'
            # Match the actual template's global number without assuming casing.
            if number==24 and re.search(r'\b23\b',chat._first['content']):
                return False,'DOC001'
            return self.ask(chat,number,page)
        with patch.object(app._SegmentChat,'_ask',ask):
            groups,_=self.run_pages(pages_for([1]))
        self.assertEqual(len(groups),1)
        self.assert_coverage(groups,100)
        self.assertIn('disagree',groups[0]['_parallel_execution']['fallback'])
        self.assertFalse(groups[0]['_parallel_execution']['ownership_applied'])
        self.assertEqual(groups[0]['_parallel_execution']['final_document_starts'],[1])
        self.assertTrue(all(join['certain'] for join in groups[0]['joins']))
        self.assertGreater(groups[0]['_parallel_execution']['question_cache_hits'],0)

    def test_refused_or_unbudgeted_questions_do_not_create_a_worker_owner(self):
        with patch.object(app._SegmentChat,'_ask',return_value=None), patch.object(app,'SEGMENT_CHAT_MAX_ASKS',4):
            groups,_=self.run_pages(pages_for([1]))
        self.assert_coverage(groups,100)
        meta=groups[0]['_parallel_execution']
        self.assertLessEqual(meta['questions_sent'],4)
        self.assertEqual(meta['confirmed_starts'],[1])

    def test_short_files_and_other_backends_keep_the_existing_path(self):
        for pages,status in [(pages_for([1],10),STATUS), (pages_for([1]),{**STATUS,'kind':'ollama'})]:
            with self.subTest(pages=len(pages),status=status):
                groups,_=self.run_pages(pages,status)
                self.assertNotIn('_parallel_execution',groups[0])

    def test_closing_parallel_grouping_does_not_keep_sending_questions(self):
        gen=app._segment_stream('whole',pages_for(list(range(1,151,7)),150),STATUS)
        self.assertEqual(next(gen)['event'],'segmenting')
        self.assertEqual(next(gen)['event'],'segment_progress')
        gen.close()
        calls=self.calls
        time.sleep(.03)
        self.assertEqual(self.calls,calls)
        self.assertEqual(self.active,0)

    def test_boundary_model_requests_from_multiple_files_share_the_server_ceiling(self):
        from concurrent.futures import ThreadPoolExecutor
        import extractionlimit
        active = peak = 0
        lock = threading.Lock()
        def post(url,payload,status):
            nonlocal active,peak
            with lock:
                active+=1; peak=max(peak,active)
            time.sleep(.004)
            messages=[m for m in payload['messages'] if m['role']=='user']
            first=re.search(r'DOC\d+',messages[0]['content'])[0]
            current=re.search(r'DOC\d+',messages[-1]['content'])[0]
            with lock:
                active-=1
            return app._Reply(200,{'choices':[{'message':{'content':json.dumps(
                {'same':first==current,'evidence':current})}}]})
        with patch.object(app,'_EXTRACTION_LIMITS',extractionlimit.ServerLimits()), \
                patch.object(app,'EXTRACT_REQUEST_CONCURRENCY',3), \
                patch.object(app._SegmentChat,'_ask',self.real_ask), \
                patch.object(app,'_post_once',side_effect=post), ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _: self.run_pages(pages_for(list(range(1,101,7)))),range(2)))
        self.assertEqual(peak,3)
        for grouped,_ in results:
            self.assert_coverage(grouped,100)
            self.assertFalse(grouped[0]['_parallel_execution']['fallback'])

    def test_exact_question_cache_deduplicates_in_flight_requests_and_refuses_only_new_questions_at_cap(self):
        questions=segmentparallel.Questions(1)
        started,release=threading.Event(),threading.Event()
        calls=[]
        def request():
            calls.append(1); started.set(); release.wait(2)
            return True,'printed'
        results=[]
        a=threading.Thread(target=lambda:results.append(questions.call('same',request)))
        b=threading.Thread(target=lambda:results.append(questions.call('same',request)))
        a.start(); self.assertTrue(started.wait(2)); b.start(); release.set(); a.join(2); b.join(2)
        self.assertEqual(calls,[1])
        self.assertEqual(results,[(True,(True,'printed'))]*2)
        self.assertEqual(questions.call('different',request),(False,None))


if __name__=='__main__':
    unittest.main()
