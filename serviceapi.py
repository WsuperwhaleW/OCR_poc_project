"""The Worker API: /api/v1 for the main service, plus API-key management for /user.

Built 2026-11-04 at the user's request (worker-service plan, steps 3 + 4, reduced
to "ระบบมี firewall อยู่แล้ว ทำแต่ api key พอ"): the main service sends a file,
polls its status, reads the User JSON v7 result and acknowledges it. Every
/api/v1 route except `/health` needs `Authorization: Bearer <API key>`; a key
sees only the jobs it submitted (anything else is 404, so job ids are not
probed). Keys are made and revoked in the `/user` API keys tab, which the
firewall protects.

A job here is an ordinary User job (same queue, engine, prompt sets and Retry
as `/user`), with `client_id` / `client_ref` recorded. When `/user` retries an
API job, `GET …/result` answers with the newest successful attempt and
`result_version` counts the successful attempts, so the main service can tell
a re-extraction from the first answer. The result is built from the stored
job on each request (`userjson.convert` is deterministic; converted reports are
cached by attempt and `updated_at`); step 6 stores a snapshot when callbacks
need one.

Not yet: `callback_url` (step 6) is refused with 400 rather than accepted and
ignored; the open-job limit, spool cap (step 5) and retention (step 7).
"""
import hashlib
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import g, jsonify, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import config
import servicestore
import userjson

STATE = {'queued_ocr': 'queued', 'ocr': 'processing', 'queued_extraction': 'processing',
         'extraction': 'processing', 'completed': 'succeeded', 'partial': 'partial',
         'failed': 'failed', 'interrupted': 'failed'}
SUCCEEDED = ('completed', 'partial')
FINISHED = ('completed', 'partial', 'failed', 'interrupted')
CACHE_SIZE = 256
EVENT_NAMESPACE = uuid.UUID('6f0c2a52-6b0e-4d55-9a57-2f8c3c1f7a10')


def _seconds(start, end):
    try:
        return round(max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()), 3)
    except (TypeError, ValueError):
        return None


def register(application, engine, service):
    lock = threading.Lock()
    reports = {}

    def error(http_status, code, message, **extra):
        return jsonify(error=code, message=message, **extra), http_status

    # ------------------------------------------------------------------ auth
    @application.before_request
    def api_key():
        if not request.path.startswith('/api/v1/') or request.path == '/api/v1/health':
            return None
        client = servicestore.authenticate(service().store, request.headers.get('Authorization'))
        if client is None:
            response, status = error(401, 'unauthorized', 'ต้องส่ง Authorization: Bearer <API key> ที่ยังไม่ถูกยกเลิก')
            response.headers['WWW-Authenticate'] = 'Bearer'
            return response, status
        g.api_client = client
        return None

    # ------------------------------------------------------------------ jobs
    def own_root(job_id):
        """The client's own job by its v1 id (the first attempt), or None."""
        job = service().store.get(job_id)
        if job is None or job.get('client_id') != g.api_client['id'] or job.get('original_id'):
            return None
        return job

    def attempts(root):
        with service().store.connect() as db:
            rows = db.execute('SELECT id,status FROM jobs WHERE id=? OR original_id=? ORDER BY created_at,rowid',
                              (root['id'], root['id'])).fetchall()
        return [dict(row) for row in rows]

    def current(root):
        """(the attempt to report, result_version, a newer attempt still running)."""
        rows = attempts(root)
        succeeded = [row for row in rows if row['status'] in SUCCEEDED]
        chosen = succeeded[-1] if succeeded else rows[-1]
        index = next(i for i, row in enumerate(rows) if row['id'] == chosen['id'])
        running = any(row['status'] not in FINISHED for row in rows[index + 1:])
        job = root if chosen['id'] == root['id'] else service().store.get(chosen['id'])
        return job, len(succeeded), running

    def report(job):
        key = (job['id'], job['updated_at'])
        with lock:
            if key in reports:
                return reports[key]
        value = userjson.convert(job)
        with lock:
            reports[key] = value
            while len(reports) > CACHE_SIZE:
                reports.pop(next(iter(reports)))
        return value

    def accepted(job, replay=False):
        url = url_for('v1_status', job_id=job['id'])
        response = jsonify(job_id=job['id'], status='queued', status_url=url)
        response.status_code = 202
        response.headers['Location'] = url
        if replay:
            response.headers['Idempotent-Replayed'] = 'true'
        return response

    @application.post('/api/v1/jobs')
    def v1_submit():
        current_service = service()
        store = current_service.store
        try:
            # Read here, not in an app-wide 413 handler: the benchmark app has its own.
            request.form, request.files
        except RequestEntityTooLarge:
            return error(413, 'file_too_large', 'ไฟล์ใหญ่เกินกำหนด', max_mb=engine.MAX_UPLOAD_MB)
        if request.form.get('callback_url'):
            return error(400, 'callback_not_available',
                         'ยังไม่รองรับ callback (ขั้นที่ 6) ให้ดึงผลด้วย GET /api/v1/jobs/{job_id}/result')
        client_ref = request.form.get('client_ref', '')
        if len(client_ref) > 200 or re.search(r'[\x00-\x1f\x7f]', client_ref):
            return error(400, 'invalid_client_ref', 'client_ref ยาวไม่เกิน 200 ตัวอักษรและไม่มีอักขระควบคุม')
        upload = request.files.get('file')
        if not upload or not upload.filename:
            return error(400, 'invalid_file', 'ต้องส่งไฟล์ใน field "file" (multipart/form-data)')
        filename = re.sub(r'[\x00-\x1f\x7f]', '', upload.filename.replace('\\', '/').rsplit('/', 1)[-1])[:220]
        if Path(filename).suffix.lower() not in engine.ACCEPTED_SUFFIXES:
            return error(400, 'invalid_file', 'รองรับ PDF และรูปภาพเท่านั้น',
                         accepted=sorted(engine.ACCEPTED_SUFFIXES))
        cfg = store.settings()
        limit_mb = min(cfg['max_upload_mb'], engine.MAX_UPLOAD_MB)
        content = upload.read(limit_mb * 1024 * 1024 + 1)
        if len(content) > limit_mb * 1024 * 1024:
            return error(413, 'file_too_large', f'ไฟล์ใหญ่เกิน {limit_mb} MB', max_mb=limit_mb)
        if not content:
            return error(400, 'invalid_file', 'ไฟล์ว่าง')

        client = g.api_client['id']
        idempotency = request.headers.get('Idempotency-Key')
        if idempotency is not None:
            if not re.fullmatch(r'[\x21-\x7e]{1,200}', idempotency):
                return error(400, 'invalid_idempotency_key', 'Idempotency-Key ยาว 1–200 ตัว เป็น ASCII ที่พิมพ์ได้')
            digest = hashlib.sha256(filename.encode('utf-8') + b'\0' + client_ref.encode('utf-8') + b'\0' + content).hexdigest()
            held = servicestore.reserve(store, client, idempotency, digest)
            if held == 'conflict':
                return error(409, 'idempotency_conflict', 'Idempotency-Key นี้ใช้กับคำขออื่นแล้ว')
            if held == 'pending':
                return error(409, 'idempotency_in_progress', 'คำขอเดียวกันกำลังถูกรับอยู่ ลองใหม่อีกครู่')
            if held:
                return accepted(store.get(held), replay=True)
        try:
            try:
                snapshot = current_service.adapter.snapshot(cfg)
            except RuntimeError as err:
                if idempotency is not None:
                    servicestore.release(store, client, idempotency)
                return error(503, 'model_unavailable', str(err))
            snapshot['prompt_profiles'] = store.prompt_profiles()
            snapshot['prompt_sets'] = store.prompt_set_snapshot()
            batch = store.create_batch(1, snapshot)
            job = store.add_job(batch, 'api_' + uuid.uuid4().hex[:16], filename, content,
                                client_id=client, client_ref=client_ref)
        except Exception:
            if idempotency is not None:
                servicestore.release(store, client, idempotency)
            raise
        if idempotency is not None:
            servicestore.complete(store, client, idempotency, job['id'])
        current_service.start()
        return accepted(job)

    @application.get('/api/v1/jobs/<job_id>')
    def v1_status(job_id):
        root = own_root(job_id)
        if root is None:
            return error(404, 'not_found', 'ไม่พบงานนี้')
        job, version, running = current(root)
        result = job.get('result') or {}
        root_timing = (root.get('result') or {}).get('timing') or {}
        timing = result.get('timing') or {}
        done = job['status'] in FINISHED
        body = {
            'job_id': root['id'], 'client_ref': root.get('client_ref', ''),
            'status': STATE.get(job['status'], job['status']), 'stage': 'done' if done else job['status'],
            'created_at': root['created_at'], 'finished_at': job['updated_at'] if done else None,
            'acknowledged_at': root.get('acknowledged_at') or None,
            'timing': {'queue_seconds': _seconds((root.get('timeline') or {}).get('queued_ocr', {}).get('started_at'),
                                                 (root.get('timeline') or {}).get('ocr', {}).get('started_at')),
                       'ocr_seconds': timing.get('ocr', root_timing.get('ocr')),
                       'extraction_seconds': timing.get('extraction')},
            'result_version': version, 'retry_in_progress': running,
        }
        if job['status'] in SUCCEEDED:
            converted = report(job)
            body['summary'] = {'documents': len(converted['data'].get('documents') or []),
                               'schema_valid': converted['schema_valid'],
                               'warnings': len(converted['warnings']),
                               'missing_fields': len(converted['missing_fields'])}
        elif job['status'] in FINISHED:
            body['error'] = job.get('error') or ''
        return jsonify(body)

    @application.get('/api/v1/jobs/<job_id>/result')
    def v1_result(job_id):
        root = own_root(job_id)
        if root is None:
            return error(404, 'not_found', 'ไม่พบงานนี้')
        job, version, _ = current(root)
        if job['status'] not in SUCCEEDED:
            if job['status'] in FINISHED:
                return error(422, 'job_failed', job.get('error') or 'ประมวลผลไม่สำเร็จ', status='failed')
            return error(409, 'not_finished', 'งานยังไม่เสร็จ', status=STATE.get(job['status'], job['status']))
        converted = report(job)
        cfg = job.get('configuration') or {}
        extracted = (job.get('result') or {}).get('extracted') or {}
        found = set(extracted.get('doc_types') or [])
        for document in extracted.get('documents') or []:
            found.update(document.get('doc_types') or [])
        bindings = (cfg.get('prompt_sets') or {}).get('bindings') or {}
        timeline = job.get('timeline') or {}
        models = cfg.get('models') or {}
        body = {
            'job_id': root['id'], 'client_ref': root.get('client_ref', ''),
            'event_id': str(uuid.uuid5(EVENT_NAMESPACE, f"{root['id']}:{version}")),
            'result_version': version,
            'processing': {
                'app_version': config.app_version(),
                'ocr_model': (models.get('ocr') or {}).get('model'),
                'extraction_model': (models.get('extraction') or {}).get('model'),
                'prompt_sets': [{'doc_type': code, 'set_id': bindings[code].get('set_id'),
                                 'revision': bindings[code].get('revision')}
                                for code in sorted(found) if code in bindings],
                'started_at': ((root.get('timeline') or {}).get('ocr') or {}).get('started_at')
                              or (timeline.get('queued_extraction') or {}).get('started_at'),
                'finished_at': job['updated_at'],
            },
            'user_json': converted['data'],
        }
        if request.args.get('report') == '1':
            body['report'] = {key: converted.get(key) for key in (
                'schema_valid', 'schema_errors', 'warnings', 'missing_fields', 'unmapped', 'data_issues', 'review_status')}
        return application.response_class(
            json.dumps(body, ensure_ascii=False, allow_nan=False), mimetype='application/json')

    @application.post('/api/v1/jobs/<job_id>/ack')
    def v1_ack(job_id):
        root = own_root(job_id)
        if root is None:
            return error(404, 'not_found', 'ไม่พบงานนี้')
        with service().store.connect() as db:
            db.execute("UPDATE jobs SET acknowledged_at=? WHERE id=? AND acknowledged_at=''",
                       (servicestore.now(), root['id']))
        return '', 204

    @application.get('/api/v1/health')
    def v1_health():
        current_service = service()
        cfg = current_service.store.settings()
        try:
            # Cached probes (a few seconds), not a request per health check.
            models = current_service.adapter.models(cfg, force=False)
        except Exception as err:  # a broken probe is a degraded service, not a 500
            models = {'ocr': {'available': False, 'reason': str(err)},
                      'extraction': {'text_available': False, 'reason': str(err)}}
        state = {'ocr': 'online' if models['ocr'].get('available') else 'offline',
                 'extraction': 'online' if models['extraction'].get('text_available') else 'offline'}
        with current_service.store.connect() as db:
            open_jobs = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN "
                                   "('queued_ocr','ocr','queued_extraction','extraction')").fetchone()[0]
        healthy = all(value == 'online' for value in state.values())
        return jsonify(status='ok' if healthy else 'degraded', models=state,
                       queue={'open_jobs': open_jobs, 'limit': None, 'accepting': healthy}), 200 if healthy else 503

    # ------------------------------------------------------------------ keys (the /user API keys tab)
    @application.get('/api/user/api-keys')
    def user_api_keys():
        return jsonify(keys=servicestore.list_keys(service().store))

    @application.post('/api/user/api-keys')
    def user_create_api_key():
        body = request.get_json(silent=True)
        try:
            key, record = servicestore.create_key(service().store, (body or {}).get('name'))
        except ValueError as err:
            return jsonify(error=str(err)), 400
        return jsonify(key=key, record=record), 201

    @application.post('/api/user/api-keys/<key_id>/revoke')
    def user_revoke_api_key(key_id):
        record = servicestore.revoke_key(service().store, key_id)
        if record is None:
            return jsonify(error='ไม่พบ key นี้'), 404
        return jsonify(record=record)
