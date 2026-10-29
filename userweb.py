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

import config
import userjson
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
            for code, name in engine.prompts.TYPE_NAMES.items()})

    @application.get('/api/user/settings')
    def user_settings():
        store = service().store
        return jsonify(settings=store.settings(), upload_ceiling_mb=store.initial['max_upload_mb'])

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
                return source.get_pixmap(matrix=engine.fitz.Matrix(scale, scale), alpha=False).tobytes('png')

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
        return send_file(io.BytesIO(pdf_info(temporary_preview(preview_id), page)), mimetype='image/png')

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

    @application.get('/api/user/jobs/<job_id>/file')
    def user_file(job_id):
        job = job_or_404(job_id)
        return send_file(service().store.uploads / job['id'], as_attachment=True, download_name=job['filename'])

    @application.get('/api/user/jobs/<job_id>/preview')
    def user_preview(job_id):
        job = job_or_404(job_id)
        suffix = Path(job['filename']).suffix.lower()
        types = {'.pdf': 'application/pdf', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                 '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp'}
        if suffix not in types:
            return jsonify(error='โปรดดาวน์โหลดต้นฉบับเพื่อเปิดไฟล์ประเภทนี้'), 415
        return send_file(service().store.uploads / job['id'], mimetype=types[suffix],
                         as_attachment=False, download_name=job['filename'])

    @application.get('/api/user/jobs/<job_id>/preview/info')
    def user_preview_info(job_id):
        job = job_or_404(job_id)
        if Path(job['filename']).suffix.lower() != '.pdf':
            abort(415)
        try:
            return jsonify(pages=pdf_info(service().store.uploads / job['id']))
        except ValueError as err:
            return jsonify(error=str(err)), 400

    @application.get('/api/user/jobs/<job_id>/preview/pages/<int:page>')
    def user_preview_page(job_id, page):
        job = job_or_404(job_id)
        if Path(job['filename']).suffix.lower() != '.pdf':
            abort(415)
        return send_file(io.BytesIO(pdf_info(service().store.uploads / job['id'], page)), mimetype='image/png')

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

    @application.after_request
    def user_headers(response):
        if request.path.startswith('/api/user/') or request.path.rstrip('/') == '/user':
            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    return service
