"""Two durable file queues using the live engine, never a copied engine snapshot."""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import workspace

workspace.field('user_pipeline', lambda: False)


def compact_extraction(value):
    """Keep user results, not benchmark truth, prompts or repeated raw replies."""
    keys = ('fields', 'normalized', 'mode', 'extras', 'schema_first', 'doc_types', 'doc_type',
            'doc_type_heading', 'pages', 'document', 'documents_found', 'documents_failed',
            'partial', 'error', 'seconds', 'tokens', 'segments', 'split_from',
            'steps_failed', 'steps_extras', 'model', 'backend', 'url', 'normalization_version')
    out = {key: value[key] for key in keys if key in value}
    if isinstance(value.get('item_table'), dict):
        table = value['item_table']
        out['item_table'] = {key: table[key] for key in ('columns', 'rows', 'row_pages', 'pages',
            'footer', 'is_master_table', 'master_why', 'reference_columns', 'item_columns', 'error') if key in table}
    elif 'item_table' in value:
        out['item_table'] = None
    if 'documents' in value:
        out['documents'] = [compact_extraction(doc) for doc in value['documents']]
    return out


class EngineAdapter:
    def __init__(self, engine):
        self.engine = engine

    @contextmanager
    def context(self, job_id, configuration):
        # A background task must not inherit the benchmark tab's workspace.
        with workspace.use(None):
            ws = workspace.enter('userjob_' + job_id)
            workspace.hold()
            try:
                ws.put('user_pipeline', True)
                ws.put('active', configuration['ocr_url'])
                ws.put('extract_url', configuration['extraction_url'])
                models = configuration.get('models') or {}
                ws.put('chosen', {configuration['ocr_url']: models['ocr']['model']}
                       if models.get('ocr', {}).get('model') else {})
                ws.put('extract_chosen', {configuration['extraction_url']: models['extraction']['model']}
                       if models.get('extraction', {}).get('model') else {})
                self.engine.set_loop_guard(configuration['loop_guard'])
                if configuration.get('ocr_profile'):
                    self.engine.set_ocr_profile(configuration['ocr_profile'])
                if configuration['no_cache']:
                    self.engine.backends.bypass_prefix_cache()
                try:
                    yield
                finally:
                    if configuration['no_cache']:
                        self.engine.backends.restore_prefix_cache()
            finally:
                workspace.release()

    def models(self, configuration, force=True):
        with self.context('probe_' + uuid.uuid4().hex, configuration):
            return {'ocr': self._info(self.engine.backends.status(force=force)),
                    'extraction': self._info(self.engine.backends.extract_status(force=force))}

    @staticmethod
    def _info(info):
        return {k: info.get(k) for k in ('url', 'kind', 'model', 'available', 'text_available',
                                        'reason', 'text_reason')}

    def snapshot(self, configuration):
        models = self.models(configuration)
        for key, available, reason in (('ocr', 'available', 'reason'),
                                       ('extraction', 'text_available', 'text_reason')):
            if not models[key].get(available) or not models[key].get('model'):
                raise RuntimeError(f"{key}: {models[key].get(reason) or 'Model unavailable'}")
        return {**configuration, 'models': models,
                'ocr_profile': self.engine.backends.profile_for_model(models['ocr']['model'])}

    def check_model(self, job, stage):
        info = (self.engine.backends.status(force=True) if stage == 'ocr'
                else self.engine.backends.extract_status(force=True))
        expected = job['configuration'].get('models', {}).get(stage, {}).get('model')
        names = {m['name'] for m in info.get('models') or []}
        if expected and (info.get('model') != expected or (names and expected not in names)):
            raise ValueError(f'{stage}: โมเดลเปลี่ยนหลังรับไฟล์ กรุณา Refresh แล้วส่งไฟล์ใหม่')
        available = 'available' if stage == 'ocr' else 'text_available'
        if not info.get(available):
            raise ValueError(info.get('reason' if stage == 'ocr' else 'text_reason') or 'Model unavailable')

    def ocr(self, job, store):
        engine, cfg = self.engine, job['configuration']
        with self.context(job['id'], cfg):
            self.check_model(job, 'ocr')
            source = {'kind': 'user_upload', 'name': job['filename'], 'filename': job['filename']}
            context = {}
            images, detail, cache_id, _ = engine.prepare_input(
                (store.uploads / job['id']).read_bytes(), cfg['detail'], source=source,
                cache=False, context_out=context)
            if not images:
                raise ValueError('ไฟล์ไม่มีหน้าที่อ่านได้')
            total = context.get('pages_total', len(images))
            result = {'text': '', 'pages': [], 'page_count': len(images), 'pages_total': total,
                      'warnings': [], 'page_stats': [], 'timing': {}}
            if total > len(images):
                result['warnings'].append(f'อ่านได้ {len(images)} จากทั้งหมด {total} หน้า')
            parallel = engine._page_parallelism(len(images))
            if parallel > 1:
                reads = engine.read_pages_at_once(images, parallel)
            else:
                def sequential():
                    for index, image in enumerate(images, 1):
                        stats = {}
                        yield index, engine.read_page(image, stats), stats
                reads = sequential()
            try:
                store.update(job['id'], progress=f'กำลังอ่านเอกสาร 0/{len(images)} หน้า', result=result)
                for index, text, stats in reads:
                    result['pages'].append(text)
                    result['page_stats'].append(stats)
                    result['text'] = engine.join_page_texts(result['pages'])
                    if stats.get('truncated') or stats.get('looped'):
                        result['warnings'].append(f'ข้อความหน้า {index} อาจอ่านได้ไม่ครบ')
                    store.update(job['id'], progress=f'อ่านข้อความแล้ว {index}/{len(images)} หน้า', result=result)
            finally:
                reads.close()
                for image in images:
                    image.close()
                images.clear()
            if not result['text'].strip():
                raise ValueError('ไม่พบข้อความที่อ่านได้จากเอกสาร')
            return result

    def extraction(self, job, store):
        cfg, result = job['configuration'], job['result']
        with self.context(job['id'], cfg):
            self.check_model(job, 'extraction')
            stream = self.engine.extract_fields_stream(
                result['text'], pages=result['pages'], mode=cfg['shape'],
                extras=cfg['extras'], schema_first=cfg['schema_first'])
            document, documents, completed, total = 1, 1, 0, 0
            try:
                while True:
                    event = next(stream)
                    kind = event.get('event')
                    if kind == 'segmented':
                        documents = event.get('documents', 1)
                    elif kind in ('document', 'classified'):
                        document = event.get('document', 1)
                        completed, total = 0, 0
                    elif kind == 'extract_steps':
                        total, completed = event['total'], 0
                    elif kind == 'extract_step' and event.get('status') == 'done':
                        completed += 1
                    elif kind in ('classifying', 'segmenting'):
                        store.update(job['id'], progress='กำลังจำแนกและแยกเอกสาร')
                        continue
                    else:
                        continue
                    text = f'ดึงข้อมูลเอกสาร {document}/{documents}'
                    if total:
                        text += f' · เสร็จ {completed}/{total} steps'
                    store.update(job['id'], progress=text)
            except StopIteration as stop:
                extracted = stop.value
            finally:
                stream.close()
            if not isinstance(extracted, dict):
                raise ValueError('ระบบไม่สามารถดึงข้อมูลได้')
            result['extracted'] = compact_extraction(extracted)
            if extracted.get('error'):
                store.update(job['id'], result=result)
                raise ValueError(extracted['error'])
            docs = extracted.get('documents') or [extracted]
            if (extracted.get('partial') or extracted.get('documents_failed') or
                    any(doc.get('partial') or doc.get('error') or doc.get('steps_failed') or
                        doc.get('steps_extras') or (doc.get('item_table') or {}).get('error') for doc in docs) or
                    not any(doc.get('fields') for doc in docs)):
                result['warnings'].append('ดึงข้อมูลบางส่วนได้ไม่ครบ กรุณาตรวจสอบกับต้นฉบับ')
            return result


class UserService:
    def __init__(self, store, adapter):
        self.store, self.adapter = store, adapter
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.closed = False
        self.active = {'ocr': set(), 'extraction': set()}
        # Pools grow lazily. Dispatch, rather than the executor's unbounded queue,
        # enforces the currently saved file concurrency.
        self.pools = {stage: ThreadPoolExecutor(max_workers=256, thread_name_prefix='user-' + stage)
                      for stage in self.active}

    def start(self):
        with self.lock:
            if self.thread is None:
                self.store.recover()
                self.thread = threading.Thread(target=self._dispatch, name='user-dispatch', daemon=True)
                self.thread.start()
        self.wake.set()

    def _dispatch(self):
        while not self.closed:
            try:
                cfg = self.store.settings()
                for stage in ('extraction', 'ocr'):
                    key = 'ocr_concurrency' if stage == 'ocr' else 'extraction_concurrency'
                    with self.lock:
                        self.active[stage] = {f for f in self.active[stage] if not f.done()}
                        while not self.closed and len(self.active[stage]) < cfg[key]:
                            job = self.store.claim(stage)
                            if not job:
                                break
                            self.active[stage].add(self.pools[stage].submit(self._run, stage, job))
            except Exception as err:
                # A transient DB failure must not permanently kill the dispatcher.
                import logging
                logging.getLogger(__name__).exception('User queue dispatcher: %s', err)
            self.wake.wait(0.5)
            self.wake.clear()

    def _run(self, stage, job):
        started = time.monotonic()
        try:
            result = getattr(self.adapter, stage)(job, self.store)
            result.setdefault('timing', {})[stage] = round(time.monotonic() - started, 3)
            if stage == 'ocr':
                status, progress = 'queued_extraction', 'รอดึงข้อมูล'
            else:
                status = 'partial' if result.get('warnings') else 'completed'
                progress = 'เสร็จบางส่วน' if status == 'partial' else 'ประมวลผลเสร็จแล้ว'
            self.store.update(job['id'], status=status, progress=progress, result=result)
        except Exception as err:
            current = self.store.get(job['id'])
            result = current['result']
            result.setdefault('timing', {})[stage] = round(time.monotonic() - started, 3)
            self.store.update(job['id'], status='failed', progress='ประมวลผลไม่สำเร็จ',
                              error=str(err)[:1200], result=result)
        finally:
            self.wake.set()

    def close(self):
        self.closed = True
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)
        for pool in self.pools.values():
            pool.shutdown(wait=True)
