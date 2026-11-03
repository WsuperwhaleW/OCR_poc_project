"""Register the user portal on the benchmark Flask application."""
import io
import json
import re
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from flask import abort, jsonify, render_template, request, send_file
from PIL import Image, ImageOps

import config
import userjson
import userprompts
import userpromptsets
from userpipeline import EngineAdapter, UserService
from userstore import UserStore, validate_settings


def register(application, engine):
    lock = threading.Lock()
    # Page-issued proof survives proxy Host rewriting without trusting arbitrary
    # forwarded hosts. Shared across tabs, not a session/ownership identifier.
    page_token = secrets.token_urlsafe(32)

    @application.context_processor
    def user_page_context():
        return {'user_api_token': page_token}

    def service():
        with lock:
            if 'user_service' not in application.extensions:
                directory = config.env_path('OCR_USER_DATA_DIR', config.LOG_DIR / 'user')
                store = UserStore(directory, engine.MAX_UPLOAD_MB)
                application.extensions['user_service'] = UserService(store, EngineAdapter(engine))
            return application.extensions['user_service']

    application.extensions['user_service_factory'] = service

    @application.before_request
    def user_request():
        if request.path.startswith('/api/user/'):
            if request.method not in ('GET', 'HEAD', 'OPTIONS'):
                origin = request.headers.get('Origin')
                proof = request.headers.get('X-OCR-User-Token', '')
                if (origin and urlsplit(origin).netloc != request.host
                        and not secrets.compare_digest(proof.encode('utf-8'), page_token.encode('ascii'))):
                    return jsonify(error='ไม่สามารถยืนยันคำขอได้ กรุณาโหลดหน้าใหม่แล้วลองอีกครั้ง'), 403
            # Permit a file at the deployment limit plus the multipart envelope.
            if request.path in ('/api/user/jobs', '/api/user/previews') and request.method == 'POST':
                request.max_content_length = (engine.MAX_UPLOAD_MB + 1) * 1024 * 1024

    @application.get('/user')
    @application.get('/user/')
    def user_index():
        return render_template('user.html', user_doc_types={
            code: re.sub(r'^(an|a|the) ', '', name)
            for code, name in engine.prompts.TYPE_NAMES.items()}, user_prompt_types={
            code: re.sub(r'^(an|a|the) ', '', name)
            for code, name in engine.prompts.TYPE_NAMES.items()
            if engine.prompts.DOC_TYPE_FIELDS.get(code)}, user_prompt_catalog=userpromptsets.catalog())

    @application.get('/api/user/settings')
    def user_settings():
        store = service().store
        return jsonify(settings=store.settings(), upload_ceiling_mb=store.initial['max_upload_mb'])

    @application.get('/api/user/hybrid-prompts')
    def user_hybrid_prompts():
        """Current User prompt reference and the editable field-level profile."""
        codes = list(dict.fromkeys(request.args.getlist('doc_type') or ['INVOICE']))
        if any(code not in engine.prompts.TYPE_NAMES for code in codes):
            return jsonify(error='ไม่รู้จักประเภทเอกสารนี้'), 400
        cfg = service().store.settings()
        cfg['prompt_sets'] = service().store.prompt_set_snapshot()
        if len(codes) > 1:
            if any(not engine.prompts.DOC_TYPE_FIELDS.get(code) for code in codes):
                return jsonify(error='เลือกประเภทเดี่ยวเพื่อจัดการชุด prompt'), 400
            try:
                return jsonify(set_reference(codes[0], cfg, codes_override=codes))
            except ValueError as err:
                return jsonify(error=str(err)), 400
        try:
            return jsonify(set_reference(codes[0], cfg, request.args.get('source_set_id')))
        except ValueError as err:
            return jsonify(error=str(err)), 400

    def set_reference(code, cfg, source_id=None, codes_override=None):
        import copy
        snapshot = copy.deepcopy(cfg['prompt_sets'])
        actual = snapshot['bindings'].get(code, {'set_id':None, 'revision':0})
        if source_id:
            if source_id not in snapshot['sets']:
                raise ValueError('ไม่พบชุดต้นแบบในเวอร์ชันของงานนี้')
            snapshot['bindings'][code] = {'set_id':source_id, 'revision':actual['revision']}
        binding = snapshot['bindings'].get(code)
        record = snapshot['sets'][binding['set_id']] if binding else None
        codes = codes_override or userpromptsets.codes_for(code)
        table = userpromptsets.runtime(engine.prompts.steps_for_types(codes, cfg['extras'], mode='hybrid'),
                                         codes, snapshot, cfg['extras'])
        framing = engine._framing({'doc_types':codes}, 'hybrid')
        groups = [{'id':step['id'], 'title':step['title'], 'keys':list(step['keys']), 'enabled':True,
                   'question':engine._step_question(step, framing, mode='hybrid')} for step in table]
        editor = None
        if record:
            definitions = userpromptsets.view(record)
            present = {group['id'] for group in groups}
            for group in definitions:
                if group['id'] not in present:
                    groups.append({'id':group['id'], 'title':group['title'], 'keys':[f['key'] for f in group['fields']],
                                   'enabled':False, 'question':''})
            order = {group['id']:i for i,group in enumerate(definitions)}
            groups.sort(key=lambda group:order.get(group['id'],len(order)))
            editor = {'revision':record['revision'], 'value':record['value'], 'groups':definitions,
                      'targets':[{'path':path, 'type':schema.get('type')} for path,schema in userprompts.targets().items()]}
        labels = {r['code']:r['label'] for r in userpromptsets.catalog()}
        shared = [c for c,b in cfg['prompt_sets']['bindings'].items()
                  if record and b['set_id'] == record['id']]
        library = {'binding':actual, 'record':record, 'source_selected':bool(source_id),
                   'shared_types':[{'code':c, 'label':labels[c]} for c in shared],
                   'sets':[{'id':r['id'], 'name':r['name'], 'revision':r['revision'], 'base_code':r['base_code']}
                           for r in snapshot['sets'].values()], 'types':userpromptsets.catalog()}
        return {'doc_types':codes, 'editable_type':code, 'mode':'hybrid', 'groups':groups,
                'prompt_prefix':engine._step_prefix('[ข้อความ OCR ของเอกสาร]'),
                'settings':{'extras':cfg['extras'], 'schema_first':cfg['schema_first']},
                'has_requirement':bool(engine.prompts.DOC_TYPE_FIELDS.get(code)), 'editor':editor, 'library':library}

    @application.get('/api/user/prompt-sets')
    def user_prompt_sets():
        state = service().store.prompt_set_snapshot()
        return jsonify(**state, types=userpromptsets.catalog())

    @application.post('/api/user/prompt-sets')
    def user_change_prompt_set():
        try:
            record = service().store.change_prompt_set(request.get_json(silent=True))
            return jsonify(record=record)
        except userpromptsets.Conflict as err:
            return jsonify(error=str(err)), 409
        except (ValueError, TypeError) as err:
            return jsonify(error=str(err)), 400

    @application.get('/api/user/prompt-sets/<set_id>/history')
    def user_prompt_set_history(set_id):
        return jsonify(versions=service().store.prompt_set_history(set_id))

    @application.post('/api/user/hybrid-prompts/preview')
    def user_preview_hybrid_prompts():
        import copy
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {'doc_type','set_id','value','job_id'}:
            return jsonify(error='ข้อมูลตัวอย่าง prompt ไม่ถูกต้อง'), 400
        try:
            cfg = (copy.deepcopy(job_or_404(body['job_id'])['configuration']) if body.get('job_id') else
                   {**service().store.settings(), 'prompt_sets':service().store.prompt_set_snapshot()})
            snapshot = cfg['prompt_sets']
            code, set_id = body['doc_type'], body['set_id']
            if body.get('job_id') and code not in prompt_types_for_job(job_or_404(body['job_id']))[0]:
                raise ValueError('แก้ได้เฉพาะประเภทที่พบในงานนี้')
            record = snapshot['sets'][set_id]
            record['value'] = userpromptsets.validate(record['base_code'], body['value'], record['base_steps'])
            record['steps'] = userpromptsets.compile_record(record)
            data = set_reference(code, cfg, set_id)
            return jsonify(data)
        except (ValueError, KeyError, TypeError) as err:
            return jsonify(error=str(err)), 400

    def prompt_reference(codes, cfg, profiles, editable=None):
        selected = editable or (codes[0] if codes else 'UNKNOWN')
        profile = profiles.get(selected, {'revision': 0, 'value': {'instructions': {}, 'extra_groups': []}})
        table = userprompts.compile_steps(engine.prompts.steps_for_types(codes, cfg['extras'], mode='hybrid'),
                                          codes, profiles)
        framing = engine._framing({'doc_types': codes}, 'hybrid')
        groups = []
        for step in table:
            groups.append({'id': step['id'], 'title': step['title'],
                           'keys': list(step['keys']) or ['other_fields'],
                           'question': engine._step_question(step, framing, mode='hybrid')})
        return dict(doc_types=codes, mode='hybrid', groups=groups,
                       prompt_prefix=engine._step_prefix('[ข้อความ OCR ของเอกสาร]'),
                       settings={'extras': cfg['extras'], 'schema_first': cfg['schema_first']},
                       has_requirement=any(engine.prompts.DOC_TYPE_FIELDS.get(code) for code in codes),
                       editor=({'revision': profile['revision'], 'value': userprompts.editor_value(editable, profile['value']),
                               'groups': userprompts.view(editable, profile),
                               'targets': [{'path': path, 'type': schema.get('type')} for path, schema in userprompts.targets().items()]}
                               if editable else None))

    @application.put('/api/user/hybrid-prompts')
    def user_save_hybrid_prompts():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) != {'doc_type', 'revision', 'value'}:
            return jsonify(error='ส่งประเภทเอกสาร เวอร์ชัน และค่า prompt ให้ครบ'), 400
        try:
            value = userprompts.validate(body['doc_type'], body['value'])
            profile = service().store.save_prompt_profile(body['doc_type'], value, body['revision'])
        except (ValueError, TypeError) as err:
            return jsonify(error=str(err)), 400
        if profile is None:
            return jsonify(error='มีผู้แก้ prompt นี้แล้ว กรุณาโหลดเวอร์ชันล่าสุดก่อนบันทึก'), 409
        return jsonify(profile=profile)

    @application.put('/api/user/settings')
    def user_save_settings():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Invalid settings object.'), 400
        try:
            value = service().store.save_settings(body)
        except ValueError as err:
            return jsonify(error=str(err)), 400
        service().wake.set()
        return jsonify(settings=value)

    @application.post('/api/user/models')
    def user_models():
        try:
            body = request.get_json(silent=True)
            cfg = validate_settings(body if body is not None else service().store.settings(),
                                    service().store.initial)
            return jsonify(models=service().adapter.models(cfg))
        except ValueError as err:
            return jsonify(error=str(err)), 400

    @application.get('/api/user/stress')
    def user_stress():
        # In-process state only; no model server is asked anything.
        runs = engine.stress_runs_live()
        return jsonify(running=bool(runs), runs=runs)

    @application.post('/api/user/batches')
    def user_batch():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Invalid file batch.'), 400
        current = service()
        cfg = current.store.settings()
        count = body.get('file_count')
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or
                (cfg['user_file_max'] and count > cfg['user_file_max'])):
            return jsonify(error=f"เลือกอย่างน้อย 1 ไฟล์" +
                           (f" และไม่เกิน {cfg['user_file_max']} ไฟล์ต่อครั้ง" if cfg['user_file_max'] else '')), 400
        try:
            cfg = current.adapter.snapshot(cfg)
        except RuntimeError as err:
            return jsonify(error=str(err)), 503
        cfg['prompt_profiles'] = current.store.prompt_profiles()
        cfg['prompt_sets'] = current.store.prompt_set_snapshot()
        batch_id = current.store.create_batch(count, cfg)
        return jsonify(batch_id=batch_id, configuration=cfg), 201

    @application.post('/api/user/jobs')
    def user_submit():
        current = service()
        upload = request.files.get('file')
        batch = current.store.batch(request.form.get('batch_id', ''))
        if not batch or not upload or not upload.filename:
            return jsonify(error='กรุณาเลือกไฟล์และเริ่มส่งใหม่'), 400
        filename = re.sub(r'[\x00-\x1f\x7f]', '', upload.filename.replace('\\', '/').rsplit('/', 1)[-1])[:220]
        if Path(filename).suffix.lower() not in engine.ACCEPTED_SUFFIXES:
            return jsonify(error='รองรับ PDF และรูปภาพเท่านั้น'), 400
        limit = min(batch['configuration']['max_upload_mb'], engine.MAX_UPLOAD_MB) * 1024 * 1024
        content = upload.read(limit + 1)
        if len(content) > limit:
            return jsonify(error=f'ไฟล์ใหญ่เกินกำหนด ({limit // 1024 // 1024} MB)'), 413
        if not content:
            return jsonify(error='ไฟล์ว่าง กรุณาเลือกไฟล์ใหม่'), 400
        try:
            job = current.store.add_job(batch['id'], request.form.get('upload_id'), filename, content)
        except ValueError as err:
            return jsonify(error=str(err)), 400
        current.start()
        return jsonify(job=job), 202

    def pdf_info(path, page=None):
        if not getattr(engine, 'fitz', None):
            raise ValueError('ต้องติดตั้ง PyMuPDF เพื่อแสดงตัวอย่าง PDF')
        # PyMuPDF shares the engine's rendering lock; previews never call OCR.
        with engine._PDF_LOCK:
            try:
                document = engine.fitz.open(stream=path.read_bytes(), filetype='pdf')
            except FileNotFoundError:
                abort(404)
            except Exception:
                raise ValueError('ไม่สามารถเปิด PDF นี้ได้ โปรดลองเปิดต้นฉบับ') from None
            with document:
                if document.needs_pass:
                    raise ValueError('PDF มีรหัสผ่าน กรุณาปลดล็อกก่อน')
                if page is None:
                    if not document.page_count:
                        raise ValueError('PDF ไม่มีหน้าเอกสาร')
                    return document.page_count
                if page < 1 or page > document.page_count:
                    abort(404)
                source = document[page - 1]
                scale = min(2, 1800 / max(source.rect.width, source.rect.height, 1))
                pixmap = source.get_pixmap(matrix=engine.fitz.Matrix(scale, scale), alpha=False)
                # JPEG is far smaller for a scan; PNG can still win on a sparse page.
                return min(pixmap.tobytes('png'), pixmap.tobytes('jpeg', jpg_quality=80), key=len)

    def page_image(data):
        return send_file(io.BytesIO(data), mimetype='image/jpeg' if data[:2] == b'\xff\xd8' else 'image/png')

    def fitted_image(path):
        """A large upload as the preview panel needs it, or None to send the original."""
        try:
            if path.stat().st_size <= 1024 * 1024:
                return None
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                image.thumbnail((1800, 1800))
                buffer = io.BytesIO()
                image.convert('RGB').save(buffer, 'JPEG', quality=80)
            return buffer.getvalue() if buffer.tell() < path.stat().st_size else None
        except Exception:
            return None

    def preview_directory():
        directory = service().store.directory / 'previews'
        directory.mkdir(exist_ok=True)
        return directory

    @application.post('/api/user/previews')
    def user_prepare_preview():
        upload = request.files.get('file')
        if not upload or not upload.filename.lower().endswith('.pdf'):
            return jsonify(error='รองรับตัวอย่าง PDF เท่านั้น'), 400
        maximum = min(service().store.settings()['max_upload_mb'], engine.MAX_UPLOAD_MB) * 1024 * 1024
        content = upload.read(maximum + 1)
        if len(content) > maximum:
            return jsonify(error='PDF ใหญ่เกินขนาดที่อนุญาต'), 413
        directory = preview_directory()
        # Temporary previews have no jobs/history. Remove expired files on use,
        # also covering closed tabs whose best-effort DELETE never arrived.
        for old in directory.glob('*.pdf'):
            try:
                if time.time() - old.stat().st_mtime > 1800:
                    old.unlink(missing_ok=True)
            except OSError:
                pass
        preview_id = secrets.token_hex(16)
        path = directory / (preview_id + '.pdf')
        try:
            path.write_bytes(content)
            pages = pdf_info(path)
        except Exception:
            path.unlink(missing_ok=True)
            return jsonify(error='ไม่สามารถแสดงตัวอย่าง PDF นี้ได้ โปรดลองเปิดต้นฉบับ'), 400
        return jsonify(preview_id=preview_id, pages=pages), 201

    def temporary_preview(preview_id):
        if not re.fullmatch('[0-9a-f]{32}', preview_id):
            abort(404)
        path = preview_directory() / (preview_id + '.pdf')
        if not path.is_file():
            abort(404)
        return path

    @application.delete('/api/user/previews/<preview_id>')
    def user_remove_preview(preview_id):
        temporary_preview(preview_id).unlink(missing_ok=True)
        return '', 204

    @application.get('/api/user/previews/<preview_id>/pages/<int:page>')
    def user_temporary_page(preview_id, page):
        return page_image(pdf_info(temporary_preview(preview_id), page))

    @application.get('/api/user/jobs')
    def user_history():
        current = service()
        current.start()  # Also resumes durable queued work after a restart.
        return jsonify(current.store.list(request.args.get('offset', 0, type=int)))

    def job_or_404(job_id):
        job = service().store.get(job_id)
        if job is None:
            abort(404)
        return job

    @application.get('/api/user/jobs/<job_id>')
    def user_job(job_id):
        return jsonify(job=job_or_404(job_id))

    @application.post('/api/user/job-status')
    def user_job_status():
        body = request.get_json(silent=True)
        ids = body.get('ids') if isinstance(body, dict) else None
        if not isinstance(ids, list) or len(ids) > 500:
            return jsonify(error='Send up to 500 job IDs.'), 400
        return jsonify(jobs=service().store.statuses(ids))

    @application.get('/api/user/jobs/<job_id>/retry-history')
    def user_retry_history(job_id):
        job = job_or_404(job_id)
        return jsonify(job_id=job_id, attempts=service().store.retry_history(job))

    def prompt_types_for_job(job):
        result = job['result']
        extracted = result.get('extracted') or {}
        docs = extracted.get('documents') or ([extracted] if extracted else result.get('classifications') or [])
        codes = set()
        for doc in docs:
            named = doc.get('doc_types') or ([doc['doc_type']] if doc.get('doc_type') else [])
            codes.update(named)
            for code in tuple(named):
                codes.update(engine.prompts.TYPE_COMPONENTS.get(code, ()))
        return [code for code in engine.prompts.TYPE_SPECIFICITY if code in codes], docs

    @application.get('/api/user/jobs/<job_id>/prompts')
    def user_job_prompts(job_id):
        job = job_or_404(job_id)
        codes, docs = prompt_types_for_job(job)
        if job['configuration'].get('prompt_sets') and codes:
            available = set(codes)
            for code in codes:
                available.difference_update(engine.prompts.TYPE_COMPONENTS.get(code, ()))
            available = [code for code in engine.prompts.TYPE_SPECIFICITY if code in available]
            selected = request.args.get('doc_type') or (available[0] if available else None)
            if selected not in available:
                return jsonify(error='ประเภทที่เลือกไม่ได้อยู่ในผลของงานนี้'), 400
            try:
                data = set_reference(selected, job['configuration'], request.args.get('source_set_id'))
            except ValueError as err:
                return jsonify(error=str(err)), 400
            data.update(available_types=available, job_id=job_id, preview_document=next((i+1 for i,d in enumerate(docs)
                if selected in (d.get('doc_types') or [d.get('doc_type')])), None))
            return jsonify(data)
        available = [code for code in codes if engine.prompts.DOC_TYPE_FIELDS.get(code)]
        selected = request.args.get('doc_type') or (available[0] if available else None)
        if selected and selected not in available:
            return jsonify(error='ประเภทที่เลือกไม่ได้อยู่ในผลของงานนี้'), 400
        # A profile is edited per type and reused for every document of that type.
        # Other types retain the versions captured by the original upload.
        preview_codes, preview_document = codes, None
        if selected:
            for index, doc in enumerate(docs, 1):
                named = set(doc.get('doc_types') or ([doc['doc_type']] if doc.get('doc_type') else []))
                for kind in tuple(named):
                    named.update(engine.prompts.TYPE_COMPONENTS.get(kind, ()))
                if selected in named:
                    preview_codes = [kind for kind in engine.prompts.TYPE_SPECIFICITY if kind in named]
                    preview_document = index
                    break
        data = prompt_reference(preview_codes, job['configuration'],
                                job['configuration'].get('prompt_profiles', {}), selected)
        data.update(available_types=available, job_id=job_id, editable_type=selected,
                    preview_document=preview_document,
                    documents=[{'document': i + 1, 'pages': doc.get('pages', []),
                                'doc_types': doc.get('doc_types', [])} for i, doc in enumerate(docs)])
        return jsonify(data)

    @application.post('/api/user/jobs/<job_id>/retry-extraction')
    def user_retry_extraction(job_id):
        parent = job_or_404(job_id)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) not in ({'request_id', 'prompt_profiles'}, {'request_id','prompt_sets'}):
            return jsonify(error='ส่งรหัสคำขอและ prompt สำหรับ retry ให้ครบ'), 400
        request_id, edits = body['request_id'], body.get('prompt_sets', body.get('prompt_profiles'))
        if not isinstance(request_id, str) or not re.fullmatch('[a-zA-Z0-9_-]{8,64}', request_id):
            return jsonify(error='รหัสคำขอ retry ไม่ถูกต้อง'), 400
        old = service().store.extraction_retry(job_id, request_id)
        if old:
            return jsonify(job=old), 202
        if parent['status'] not in ('completed', 'partial', 'failed', 'interrupted'):
            return jsonify(error='รอให้งานเดิมหยุดประมวลผลก่อน retry'), 409
        if not parent['result'].get('text') or not parent['result'].get('pages'):
            return jsonify(error='งานนี้ยังไม่มีข้อความ OCR ที่บันทึกไว้สำหรับ retry'), 400
        current = service()
        codes, _ = prompt_types_for_job(parent)
        available = (set(codes) if 'prompt_sets' in body else
                     {code for code in codes if engine.prompts.DOC_TYPE_FIELDS.get(code)})
        if not isinstance(edits, dict) or set(edits) - available:
            return jsonify(error='แก้ได้เฉพาะประเภทที่พบในงานนี้'), 400
        cfg = {**current.store.settings(), **parent['configuration'], 'shape': 'hybrid',
               'extraction_url': current.store.settings()['extraction_url']}
        profiles = dict(cfg.get('prompt_profiles', {}))
        try:
            if 'prompt_sets' in body:
                import copy
                if not cfg.get('prompt_sets'):
                    raise ValueError('งานรุ่นเก่าให้ใช้การแก้ Prompt แบบเดิมของงานนั้น')
                snapshot = copy.deepcopy(cfg['prompt_sets'])
                for code, edit in edits.items():
                    if not isinstance(edit, dict) or set(edit) != {'set_id','value'} or edit['set_id'] not in snapshot['sets']:
                        raise ValueError('เลือกชุดและค่า prompt จากเวอร์ชันของงานนี้')
                    snapshot['bindings'][code] = {'set_id':edit['set_id'], 'revision':0}
                cfg['prompt_sets'] = userpromptsets.retry(snapshot, {c:e['value'] for c,e in edits.items()})
            for code, value in (edits.items() if 'prompt_profiles' in body else []):
                normalized = userprompts.validate(code, value)
                profiles[code] = {'revision': profiles.get(code, {}).get('revision', 0) + 1,
                                  'scope': 'job', 'value': normalized}
            cfg['prompt_profiles'] = profiles
            if 'prompt_profiles' in body and cfg.get('prompt_sets'):
                cfg['prompt_sets'] = userpromptsets.retry(cfg['prompt_sets'], edits)
            cfg = current.adapter.snapshot_extraction(cfg)
            retried = current.store.retry_extraction(job_id, request_id, cfg)
        except (ValueError, TypeError, KeyError) as err:
            return jsonify(error=str(err)), 400
        except RuntimeError as err:
            return jsonify(error=str(err)), 503
        current.start()
        return jsonify(job=retried), 202

    @application.get('/api/user/jobs/<job_id>/file')
    def user_file(job_id):
        job = job_or_404(job_id)
        return send_file(service().store.original(job), as_attachment=True, download_name=job['filename'])

    @application.get('/api/user/jobs/<job_id>/preview')
    def user_preview(job_id):
        job = job_or_404(job_id)
        suffix = Path(job['filename']).suffix.lower()
        types = {'.pdf': 'application/pdf', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                 '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp'}
        if suffix not in types:
            return jsonify(error='โปรดดาวน์โหลดต้นฉบับเพื่อเปิดไฟล์ประเภทนี้'), 415
        if request.args.get('fit') == '1' and suffix in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            fitted = fitted_image(service().store.original(job))
            if fitted:
                return send_file(io.BytesIO(fitted), mimetype='image/jpeg')
        return send_file(service().store.original(job), mimetype=types[suffix],
                         as_attachment=False, download_name=job['filename'])

    @application.get('/api/user/jobs/<job_id>/preview/info')
    def user_preview_info(job_id):
        job = job_or_404(job_id)
        if Path(job['filename']).suffix.lower() != '.pdf':
            abort(415)
        try:
            return jsonify(pages=pdf_info(service().store.original(job)))
        except ValueError as err:
            return jsonify(error=str(err)), 400

    @application.get('/api/user/jobs/<job_id>/preview/pages/<int:page>')
    def user_preview_page(job_id, page):
        job = job_or_404(job_id)
        if Path(job['filename']).suffix.lower() != '.pdf':
            abort(415)
        return page_image(pdf_info(service().store.original(job), page))

    @application.get('/api/user/jobs/<job_id>/export')
    def user_export(job_id):
        job = job_or_404(job_id)
        output = json.dumps(job, ensure_ascii=False, indent=2).encode('utf-8')
        return send_file(io.BytesIO(output), mimetype='application/json', as_attachment=True,
                         download_name=Path(job['filename']).stem + '.json')

    @application.get('/api/user/jobs/<job_id>/user-json')
    def user_json_preview(job_id):
        job = job_or_404(job_id)
        report = userjson.convert(job)
        if request.args.get('download') == 'report':
            output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
            return send_file(io.BytesIO(output), mimetype='application/json', as_attachment=True,
                             download_name=Path(job['filename']).stem + '.user-v7.report.json')
        if request.args.get('download') == '1':
            output = json.dumps(report['data'], ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
            return send_file(io.BytesIO(output), mimetype='application/json', as_attachment=True,
                             download_name=Path(job['filename']).stem + '.user-v7.preview.json')
        return jsonify(report)

    # A job's original never changes and a temporary preview lasts 30 minutes, so
    # these need not be fetched again each time the same document is opened.
    kept_seconds = {'user_file': 86400, 'user_preview': 86400, 'user_preview_page': 86400,
                    'user_temporary_page': 1800}

    @application.after_request
    def user_headers(response):
        if request.path.startswith('/api/user/') or request.path.rstrip('/') == '/user':
            response.headers['X-Content-Type-Options'] = 'nosniff'
            if request.endpoint in kept_seconds and response.status_code in (200, 206, 304):
                response.headers['Cache-Control'] = f'private, max-age={kept_seconds[request.endpoint]}'
            elif (request.method == 'GET' and response.status_code == 200 and not response.is_streamed
                    and response.mimetype in ('application/json', 'text/html')):
                # Always checked with the server, but polling a job that has not
                # moved costs a 304 rather than all of its OCR text again.
                response.headers['Cache-Control'] = 'private, no-cache'
                response.add_etag(weak=True)
                response.make_conditional(request)
            else:
                response.headers['Cache-Control'] = 'no-store'
        return response

    return service
