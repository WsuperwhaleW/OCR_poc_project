"""Register the user portal on the benchmark Flask application."""
import io
import json
import re
import secrets
import threading
from pathlib import Path
from urllib.parse import urlsplit

from flask import abort, jsonify, render_template, request, send_file

import config
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
            if request.path == '/api/user/jobs' and request.method == 'POST':
                request.max_content_length = (engine.MAX_UPLOAD_MB + 1) * 1024 * 1024

    @application.get('/user')
    @application.get('/user/')
    def user_index():
        return render_template('user.html')

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

    @application.get('/api/user/jobs/<job_id>/export')
    def user_export(job_id):
        job = job_or_404(job_id)
        output = json.dumps(job, ensure_ascii=False, indent=2).encode('utf-8')
        return send_file(io.BytesIO(output), mimetype='application/json', as_attachment=True,
                         download_name=Path(job['filename']).stem + '.json')

    @application.after_request
    def user_headers(response):
        if request.path.startswith('/api/user/') or request.path.rstrip('/') == '/user':
            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    return service
