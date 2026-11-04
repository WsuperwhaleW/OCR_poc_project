"""Every HTTP endpoint of the User service, with example requests and responses.

Built 2026-11-04 at the user's request: "อยากให้เพิ่มหน้า api ทั้งหมดก่อน เพื่อใช้ดู"
(worker-service plan, step 2b). The **API** tab of `/user` draws this list
(`GET /api/user/api-reference`), so the page and the code cannot drift apart:
`tests/test_api_reference.py` fails when a route exists without an entry here,
or an entry says `live` for a route that does not exist.

Two groups:

* `v1` -- the Worker API the main service calls (serviceapi.py): submit a file,
  read the status and result, acknowledge. Live since 2026-11-04 with API keys
  only ("ระบบมี firewall อยู่แล้ว ทำแต่ api key พอ"). Responses that later steps
  add (429 queue_full, 507 spool_full, 410 expired, the callback) are listed
  with the step that brings them.
* `user` -- what the `/user` page itself calls, behind the firewall. Examples
  follow the real response shapes (`userstore`, `userjson.convert`), shortened
  with "…".

Data only, plus `snippets()`, which writes the curl / Python / JavaScript for an
entry. `{BASE}` in a snippet stands for the service's address; the page fills in
its own.
"""
import json

EXAMPLE_JOB = '3f2a9c1e5b7d4e6f8a0b1c2d3e4f5a6b'
EXAMPLE_BATCH = '9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e'
EXAMPLE_SET = 'invoice-a1b2c3'
EXAMPLE_KEY_ID = '5d4c3b2a19087f6e5d4c3b2a19087f6e'

V1_AUTH = ('Authorization: Bearer <API key> — สร้างและยกเลิก key ได้ในแท็บ API keys ของ /user · '
           'key ที่ไม่ถูกต้องหรือถูกยกเลิกได้ 401 · เห็นเฉพาะงานที่ key นี้ส่ง')
USER_AUTH = ('ไม่มี login (หน้า /user อยู่หลัง firewall) · คำขอที่แก้ข้อมูลจากต่างโดเมนต้องส่ง '
             'X-OCR-User-Token ที่หน้า /user ออกให้')

_ERROR = {'error': 'ข้อความอธิบายสาเหตุ (ภาษาไทย)'}

USER_JSON_EXAMPLE = {
    'schemaVersion': '7.0',
    'file': {'fileName': 'invoice_0425.pdf', 'pageCount': 2},
    'documents': [{'documentType': {'code': 'INV', 'name': 'ใบแจ้งหนี้'},
                   'pages': [1, 2], 'fields': {'documentNumber': 'INV-2026-0425',
                                               'documentDate': '2026-10-28',
                                               'totalAmount': 10700.0}, 'items': ['…']}],
}

JOB_EXAMPLE = {
    'id': EXAMPLE_JOB, 'batch_id': EXAMPLE_BATCH, 'filename': 'invoice_0425.pdf',
    'created_at': '2026-11-04T08:20:11.402+00:00', 'updated_at': '2026-11-04T08:21:03.118+00:00',
    'status': 'completed', 'progress': 'ประมวลผลเสร็จแล้ว', 'error': '',
    'configuration': {'detail': 'medium', 'shape': 'hybrid', 'models': {'ocr': {'model': 'typhoon-ocr'},
                                                                       'extraction': {'model': 'qwen3'}}, '…': '…'},
    'timeline': {'queued_ocr': {'started_at': '…', 'finished_at': '…'}, 'ocr': {'started_at': '…', 'finished_at': '…'},
                 'queued_extraction': {'…': '…'}, 'extraction': {'…': '…'}, 'completed': {'started_at': '…'}},
    'result': {'text': '# ใบแจ้งหนี้ …', 'pages': ['…'], 'page_count': 2, 'warnings': [],
               'timing': {'ocr': 21.4, 'extraction': 30.2},
               'extracted': {'doc_types': ['INVOICE'], 'documents': ['…'], 'fields': {'…': '…'}}},
    'retry_of': '', 'original_id': '', 'prompt_changes': {},
}

MODEL_INFO = {'url': 'http://127.0.0.1:8000', 'kind': 'vllm', 'model': 'typhoon-ocr',
              'available': True, 'text_available': True, 'reason': '', 'text_reason': ''}

SETTINGS_EXAMPLE = {'ocr_concurrency': 50, 'extraction_concurrency': 64, 'detail': 'medium',
                    'loop_guard': True, 'shape': 'hybrid', 'extras': False, 'schema_first': False,
                    'fill_missing': True, 'validation_retry': False,
                    'ocr_url': 'http://127.0.0.1:8000', 'extraction_url': 'http://127.0.0.1:8001',
                    'no_cache': False, 'user_file_max': 0, 'max_upload_mb': 50}


def _param(name, where, kind, desc, required=True):
    return {'name': name, 'in': where, 'type': kind, 'required': required, 'desc': desc}


def _ok(code, body, note='', kind='json'):
    return {'code': code, 'body': body, 'note': note, 'kind': kind}


def _err(code, note):
    return {'code': code, 'body': _ERROR if code != 404 else None, 'note': note,
            'kind': 'json' if code != 404 else 'empty'}


JOB_ID = _param('job_id', 'path', 'string (32 hex)', 'รหัสงาน')

ENDPOINTS = [
    # ------------------------------------------------------------------ v1 (live)
    {'id': 'v1-submit', 'group': 'v1', 'status': 'live', 'method': 'POST', 'path': '/api/v1/jobs',
     'title': 'ส่งไฟล์เข้าคิว',
     'summary': 'รับไฟล์ 1 ไฟล์ (PDF หรือรูป) แล้วตอบทันทีด้วย job_id งานเข้าคิวเดียวกับหน้า /user '
                '(User priority) · poll สถานะที่ status_url แล้วดึงผลจาก /result',
     'auth': V1_AUTH,
     'headers': [{'name': 'Authorization', 'value': 'Bearer <API key>', 'required': True},
                 {'name': 'Idempotency-Key', 'value': '<รหัสไม่ซ้ำของคำขอนี้ เช่น uuid>', 'required': False,
                  'desc': 'ส่งซ้ำด้วย key + ไฟล์ + client_ref เดิม = ได้ job เดิมคืน (header Idempotent-Replayed: true, '
                          'เก็บ 7 วัน) · key เดิมแต่คำขอต่าง = 409'}],
     'params': [_param('file', 'form', 'file', 'ไฟล์ PDF / JPG / PNG / HEIC'),
                _param('client_ref', 'form', 'string ≤ 200', 'รหัสอ้างอิงของ Service หลัก ส่งกลับมาในทุกคำตอบ', False),
                _param('callback_url', 'form', 'URL', 'ยังไม่รองรับ (ขั้นที่ 6) ส่งมาจะได้ 400 callback_not_available',
                       False)],
     'multipart': True,
     'responses': [
         _ok(202, {'job_id': EXAMPLE_JOB, 'status': 'queued', 'status_url': '/api/v1/jobs/' + EXAMPLE_JOB},
             'รับงานแล้ว · header Location = status_url'),
         _ok(400, {'error': 'invalid_file', 'message': 'รองรับ PDF และรูปภาพเท่านั้น'},
             'invalid_file · invalid_client_ref · invalid_idempotency_key · callback_not_available'),
         _ok(401, {'error': 'unauthorized', 'message': 'ต้องส่ง Authorization: Bearer <API key> ที่ยังไม่ถูกยกเลิก'},
             'ไม่มี key หรือ key ถูกยกเลิก'),
         _ok(409, {'error': 'idempotency_conflict', 'message': 'Idempotency-Key นี้ใช้กับคำขออื่นแล้ว'},
             'Idempotency-Key เดิมแต่คำขอต่างกัน · idempotency_in_progress = คำขอเดียวกันกำลังถูกรับ'),
         _ok(413, {'error': 'file_too_large', 'message': 'ไฟล์ใหญ่เกิน 50 MB', 'max_mb': 50}, 'ไฟล์ใหญ่เกินกำหนด'),
         _ok(429, {'error': 'queue_full', 'open_jobs': 500, 'limit': 500},
             'ขั้นที่ 5 (ยังไม่ส่ง): งานค้างครบเพดาน ค่าเริ่ม 500 · มี header Retry-After'),
         _ok(503, {'error': 'model_unavailable', 'message': 'ocr: connection refused'}, 'model server ไม่พร้อม'),
         _ok(507, {'error': 'spool_full'}, 'ขั้นที่ 5 (ยังไม่ส่ง): พื้นที่เก็บไฟล์ชั่วคราวเต็ม'),
     ]},
    {'id': 'v1-status', 'group': 'v1', 'status': 'live', 'method': 'GET', 'path': '/api/v1/jobs/{job_id}',
     'title': 'สถานะงาน',
     'summary': 'สถานะและเวลาของงาน ไม่มีผลลัพธ์เต็ม (เบา เหมาะกับการ poll ทุก 2–5 วินาที) · '
                'งานของ client อื่นตอบ 404',
     'auth': V1_AUTH,
     'headers': [{'name': 'Authorization', 'value': 'Bearer <API key>', 'required': True}],
     'params': [JOB_ID],
     'responses': [
         _ok(200, {'job_id': EXAMPLE_JOB, 'client_ref': 'ERP-000123',
                   'status': 'succeeded', 'stage': 'done',
                   'created_at': '2026-11-04T08:20:11.402+00:00', 'finished_at': '2026-11-04T08:21:03.118+00:00',
                   'acknowledged_at': None,
                   'timing': {'queue_seconds': 0.8, 'ocr_seconds': 21.4, 'extraction_seconds': 30.2},
                   'result_version': 1, 'retry_in_progress': False,
                   'summary': {'documents': 1, 'schema_valid': True, 'warnings': 0, 'missing_fields': 2}},
             'status: queued | processing | succeeded | partial | failed · stage = ขั้นที่กำลังทำ หรือ done · '
             'งานที่ล้มเหลวมี error แทน summary'),
         _ok(401, {'error': 'unauthorized', 'message': '…'}, 'ไม่มี key หรือ key ถูกยกเลิก'),
         _ok(404, {'error': 'not_found', 'message': 'ไม่พบงานนี้'}, 'ไม่พบงาน หรือเป็นงานของ client อื่น'),
     ]},
    {'id': 'v1-result', 'group': 'v1', 'status': 'live', 'method': 'GET',
     'path': '/api/v1/jobs/{job_id}/result',
     'title': 'ผลลัพธ์ (User JSON v7)',
     'summary': 'User JSON v7 ของไฟล์นี้ (1 bundle ต่อ 1 ไฟล์) · ถ้ามีการ Retry ในหน้า /user จะได้ผลรอบล่าสุดที่สำเร็จ '
                'และ result_version เพิ่มขึ้น · event_id คงที่ต่อ 1 version',
     'auth': V1_AUTH,
     'headers': [{'name': 'Authorization', 'value': 'Bearer <API key>', 'required': True}],
     'params': [JOB_ID, _param('report', 'query', '0 | 1',
                               'ใส่ 1 เพื่อแนบ report: schema_valid, schema_errors, warnings, missing_fields, unmapped', False)],
     'responses': [
         _ok(200, {'job_id': EXAMPLE_JOB, 'client_ref': 'ERP-000123',
                   'event_id': '7c1e2d3f-4a5b-5c6d-8e7f-9a0b1c2d3e4f', 'result_version': 1,
                   'processing': {'app_version': '2026-11-04+038fe4b', 'ocr_model': 'typhoon-ocr',
                                  'extraction_model': 'qwen3',
                                  'prompt_sets': [{'doc_type': 'INVOICE', 'set_id': EXAMPLE_SET, 'revision': 4}],
                                  'started_at': '2026-11-04T08:20:12.010+00:00',
                                  'finished_at': '2026-11-04T08:21:03.118+00:00'},
                   'user_json': USER_JSON_EXAMPLE}),
         _ok(409, {'error': 'not_finished', 'message': 'งานยังไม่เสร็จ', 'status': 'processing'}, 'งานยังไม่เสร็จ'),
         _ok(410, {'error': 'expired'}, 'ขั้นที่ 7 (ยังไม่ส่ง): ผลถูกล้างตามระยะเก็บประวัติ'),
         _ok(422, {'error': 'job_failed', 'message': 'Unsupported or corrupt file format.', 'status': 'failed'},
             'ประมวลผลไม่สำเร็จ ไม่มีผล'),
         _ok(404, {'error': 'not_found', 'message': 'ไม่พบงานนี้'}, 'ไม่พบงาน'),
     ]},
    {'id': 'v1-ack', 'group': 'v1', 'status': 'live', 'method': 'POST', 'path': '/api/v1/jobs/{job_id}/ack',
     'title': 'ยืนยันว่ารับผลแล้ว',
     'summary': 'บันทึกเวลาที่ Service หลักเก็บผลเรียบร้อย (ครั้งแรกเท่านั้น) ไม่ลบอะไร ผลยังดึงซ้ำได้ตามระยะเก็บประวัติ',
     'auth': V1_AUTH,
     'headers': [{'name': 'Authorization', 'value': 'Bearer <API key>', 'required': True}],
     'params': [JOB_ID],
     'responses': [_ok(204, None, 'บันทึกแล้ว (ส่งซ้ำได้)', 'empty'),
                   _ok(404, {'error': 'not_found', 'message': 'ไม่พบงานนี้'}, 'ไม่พบงาน')]},
    {'id': 'v1-health', 'group': 'v1', 'status': 'live', 'method': 'GET', 'path': '/api/v1/health',
     'title': 'สถานะ service',
     'summary': 'ไม่ต้องใช้ key · ใช้กับ load balancer หรือดูว่ารับงานได้ไหมก่อนส่ง · ผลตรวจ model ถูก cache ไม่กี่วินาที',
     'auth': 'ไม่ต้องใช้',
     'headers': [], 'params': [],
     'responses': [_ok(200, {'status': 'ok', 'models': {'ocr': 'online', 'extraction': 'online'},
                             'queue': {'open_jobs': 12, 'limit': None, 'accepting': True}},
                       'limit เป็น null จนกว่าขั้นที่ 5 จะเพิ่มเพดาน'),
                   _ok(503, {'status': 'degraded', 'models': {'ocr': 'offline', 'extraction': 'online'},
                             'queue': {'open_jobs': 0, 'limit': None, 'accepting': False}}, 'model server ไม่พร้อม')]},

    # ------------------------------------------------------------------ user (live)
    {'id': 'api-keys', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/api-keys',
     'title': 'รายการ API key', 'summary': 'ไม่มีตัว key (เก็บแค่ hash) มีแค่ชื่อ ตัวย่อ เวลาใช้ล่าสุด และจำนวนงานที่ส่ง',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'keys': [{'id': EXAMPLE_KEY_ID, 'name': 'ERP หลัก', 'hint': 'ocr_AbCd…9xYz',
                                       'created_at': '…', 'last_used_at': '…', 'revoked_at': '', 'jobs': 412}]})]},
    {'id': 'api-keys-create', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/api-keys',
     'title': 'สร้าง API key', 'summary': 'ตอบตัว key ครั้งเดียวเท่านั้น เก็บไว้ทันที',
     'auth': USER_AUTH, 'headers': [], 'params': [], 'body': {'name': 'ERP หลัก'},
     'responses': [_ok(201, {'key': 'ocr_…(43 ตัว)', 'record': {'id': EXAMPLE_KEY_ID, 'name': 'ERP หลัก',
                                                                 'hint': 'ocr_AbCd…9xYz', 'created_at': '…',
                                                                 'last_used_at': '', 'revoked_at': ''}}),
                   _err(400, 'ชื่อว่างหรือยาวเกิน 80 ตัว')]},
    {'id': 'api-keys-revoke', 'group': 'user', 'status': 'live', 'method': 'POST',
     'path': '/api/user/api-keys/{key_id}/revoke',
     'title': 'ยกเลิก API key', 'summary': 'มีผลกับคำขอถัดไปทันที กู้คืนไม่ได้ (สร้างใหม่แทน)',
     'auth': USER_AUTH, 'headers': [], 'params': [_param('key_id', 'path', 'string (32 hex)', 'รหัส key')],
     'responses': [_ok(200, {'record': {'id': EXAMPLE_KEY_ID, 'name': 'ERP หลัก', 'revoked_at': '…', '…': '…'}}),
                   _err(404, 'ไม่พบ key')]},
    {'id': 'schema', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/schema',
     'title': 'ประเภทเอกสารและ field ของระบบ',
     'summary': 'ข้อมูลอ้างอิง ไม่เรียก model server · หน้า Doc types ใช้วาดตาราง',
     'auth': 'ไม่ต้องใช้', 'headers': [], 'params': [],
     'responses': [_ok(200, {'types': [{'code': 'INVOICE', 'name': 'an invoice', 'headings': ['ใบแจ้งหนี้'],
                                        'fields': ['document_number', '…'], 'mandatory': ['…'], 'items': ['…'],
                                        'mandatory_items': [], 'rules': ['…'], 'has_requirement': True}],
                             'unknown': {'fields': ['…'], 'mandatory': [], 'items': [], 'rules': []},
                             'rule_notes': {'…': '…'}, 'keys': ['document_number', '…'],
                             'item_keys': ['…'], 'items_key': 'income_items', 'user_mapping': {'…': '…'}})]},
    {'id': 'api-reference', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/api-reference',
     'title': 'รายการ API นี้',
     'summary': 'ข้อมูลที่หน้า API ใช้แสดง (มาจาก apireference.py)',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'endpoints': ['…'], 'callback': {'…': '…'}, 'groups': [{'id': 'v1', 'title': '…', 'note': '…'}]})]},
    {'id': 'service-overview', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/service/overview',
     'title': 'ข้อมูลหน้า Service dashboard',
     'summary': 'สุขภาพระบบ คิว ปริมาณงาน ความเร็ว คุณภาพ และงานที่ล้มเหลวล่าสุด · คำนวณจากข้อมูลในเครื่อง '
                'ไม่เรียก model server · cache 2 วินาที',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'generated_at': '2026-11-04T08:30:00+00:00', 'alerts': [{'level': 'warn', 'text': '…'}],
                             'health': {'…': '…'}, 'queue': {'…': '…'}, 'requests': ['…'],
                             'throughput': {'…': '…'}, 'speed': {'…': '…'}, 'quality': {'…': '…'},
                             'failures': ['…'], 'planned': ['…']})]},
    {'id': 'settings-get', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/settings',
     'title': 'อ่าน Settings', 'summary': 'ค่าที่ใช้กับงานใหม่ (model server, ความละเอียด, concurrency)',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'settings': SETTINGS_EXAMPLE, 'upload_ceiling_mb': 50})]},
    {'id': 'settings-put', 'group': 'user', 'status': 'live', 'method': 'PUT', 'path': '/api/user/settings',
     'title': 'บันทึก Settings', 'summary': 'ส่งเฉพาะ key ที่จะเปลี่ยน มีผลกับงานถัดไปภายใน ~1 วินาที',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'ocr_concurrency': 32, 'detail': 'medium'},
     'responses': [_ok(200, {'settings': SETTINGS_EXAMPLE}), _err(400, 'key ไม่รู้จักหรือค่าไม่ถูกต้อง')]},
    {'id': 'models', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/models',
     'title': 'ตรวจ model server',
     'summary': 'ถาม OCR และ extraction server ตอนนี้ (มีการเรียกเครือข่าย) · ไม่ส่ง body = ใช้ Settings ที่บันทึกไว้',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'ocr_url': 'http://127.0.0.1:8000', 'extraction_url': 'http://127.0.0.1:8001'},
     'responses': [_ok(200, {'models': {'ocr': MODEL_INFO, 'extraction': {**MODEL_INFO, 'url': 'http://127.0.0.1:8001',
                                                                           'model': 'qwen3'}}}),
                   _err(400, 'URL ไม่ถูกต้อง')]},
    {'id': 'stress', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/stress',
     'title': 'Stress test ที่กำลังรัน',
     'summary': 'ใน service_app ตอบว่างเสมอ (ไม่มี benchmark) · ใน app.py บอก stress run ที่แย่ง model server อยู่',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'running': False, 'runs': []})]},
    {'id': 'batch-create', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/batches',
     'title': 'เริ่มชุดอัปโหลด',
     'summary': 'ขั้นแรกของการส่งไฟล์: ล็อก Settings, model และ prompt sets ของชุดนี้ไว้ แล้วส่งไฟล์ทีละไฟล์ด้วย /api/user/jobs',
     'auth': USER_AUTH, 'headers': [], 'params': [], 'body': {'file_count': 3},
     'responses': [_ok(201, {'batch_id': EXAMPLE_BATCH, 'configuration': {**SETTINGS_EXAMPLE, 'models': '…',
                                                                         'prompt_sets': '…'}}),
                   _err(400, 'file_count ไม่ถูกต้องหรือเกินจำนวนต่อครั้ง'),
                   _ok(503, _ERROR, 'model server ไม่พร้อม')]},
    {'id': 'job-submit', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/jobs',
     'title': 'ส่งไฟล์เข้าชุด',
     'summary': 'ส่ง 1 ไฟล์ต่อคำขอ · upload_id เดิมในชุดเดิม = ได้งานเดิมคืน (ส่งซ้ำได้ปลอดภัย)',
     'auth': USER_AUTH, 'headers': [],
     'params': [_param('batch_id', 'form', 'string', 'จาก /api/user/batches'),
                _param('upload_id', 'form', 'string 8–64 ตัว [a-zA-Z0-9_-]', 'รหัสไฟล์นี้ในชุด'),
                _param('file', 'form', 'file', 'PDF หรือรูปภาพ')],
     'multipart': True,
     'responses': [_ok(202, {'job': {k: JOB_EXAMPLE[k] for k in ('id', 'batch_id', 'filename', 'created_at',
                                                                 'updated_at')} | {'status': 'queued_ocr',
                                                                                    'progress': 'กำลังรอคิว OCR'}}),
                   _err(400, 'ไม่มีไฟล์ ชุดไม่ถูกต้อง หรือชนิดไฟล์ไม่รองรับ'),
                   _err(413, 'ไฟล์ใหญ่เกินกำหนด')]},
    {'id': 'jobs-list', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs',
     'title': 'ประวัติงาน', 'summary': 'ทีละ 50 งาน ใหม่สุดก่อน พร้อมจำนวนงานแยกตามสถานะ',
     'auth': USER_AUTH, 'headers': [],
     'params': [_param('offset', 'query', 'int', 'ข้ามกี่งาน (ค่าเริ่ม 0)', False)],
     'responses': [_ok(200, {'jobs': [{'id': EXAMPLE_JOB, 'batch_id': EXAMPLE_BATCH, 'filename': 'invoice_0425.pdf',
                                       'created_at': '…', 'updated_at': '…', 'status': 'completed',
                                       'progress': 'ประมวลผลเสร็จแล้ว', 'error': '', 'retry_of': ''}],
                             'more': False, 'counts': {'completed': 41, 'partial': 3, 'failed': 1}})]},
    {'id': 'job-status', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/job-status',
     'title': 'สถานะหลายงานพร้อมกัน', 'summary': 'เบา ไม่มีข้อความ OCR · ใช้ poll ชุดที่กำลังอัปโหลด (สูงสุด 500 id)',
     'auth': USER_AUTH, 'headers': [], 'params': [], 'body': {'ids': [EXAMPLE_JOB]},
     'responses': [_ok(200, {'jobs': [{'id': EXAMPLE_JOB, 'status': 'ocr', 'progress': 'กำลังอ่านเอกสารหน้า 1/2',
                                       'updated_at': '…'}]}),
                   _err(400, 'ids ไม่ใช่ list หรือเกิน 500')]},
    {'id': 'job-get', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs/{job_id}',
     'title': 'รายละเอียดงาน', 'summary': 'ทั้งงาน: configuration, timeline, ข้อความ OCR และผล extraction',
     'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'responses': [_ok(200, {'job': JOB_EXAMPLE}), _err(404, 'ไม่พบงาน')]},
    {'id': 'job-user-json', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/jobs/{job_id}/user-json',
     'title': 'User JSON v7 ของงาน',
     'summary': 'แปลงผลเป็น User JSON v7 พร้อมผลตรวจ schema · download=1 ได้ไฟล์ JSON · download=report ได้ทั้งรายงาน',
     'auth': USER_AUTH, 'headers': [],
     'params': [JOB_ID, _param('download', 'query', '1 | report', 'ดาวน์โหลดเป็นไฟล์', False)],
     'responses': [_ok(200, {'data': USER_JSON_EXAMPLE, 'schema_valid': True, 'schema_errors': [], 'warnings': [],
                             'missing_fields': ['documents[0].fields.dueDate'], 'unmapped': [], 'data_issues': [],
                             'review_status': 'ready_for_review', 'mappings': ['…'],
                             'counts': {'schema_errors': 0, 'warnings': 0, 'missing_optional': 1},
                             'source_job': {'id': EXAMPLE_JOB, 'filename': 'invoice_0425.pdf',
                                            'status': 'completed', 'updated_at': '…'},
                             'conversion_version': 4}),
                   _err(404, 'ไม่พบงาน')]},
    {'id': 'job-file', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs/{job_id}/file',
     'title': 'ดาวน์โหลดไฟล์ต้นฉบับ', 'summary': 'ไฟล์ที่อัปโหลด (แนบเป็น attachment)',
     'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'responses': [_ok(200, None, 'ไฟล์ต้นฉบับ', 'file'), _err(404, 'ไม่พบงาน')]},
    {'id': 'job-preview', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs/{job_id}/preview',
     'title': 'เปิดไฟล์ในเบราว์เซอร์', 'summary': 'PDF หรือรูปแบบ inline · fit=1 ย่อรูปใหญ่ให้เหลือ ≤1800 px',
     'auth': USER_AUTH, 'headers': [],
     'params': [JOB_ID, _param('fit', 'query', '1', 'ย่อรูปใหญ่', False)],
     'responses': [_ok(200, None, 'PDF / รูปภาพ', 'file'), _ok(415, _ERROR, 'ชนิดไฟล์ที่เปิดในเบราว์เซอร์ไม่ได้')]},
    {'id': 'job-preview-info', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/jobs/{job_id}/preview/info',
     'title': 'จำนวนหน้า PDF', 'summary': 'เฉพาะ PDF', 'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'responses': [_ok(200, {'pages': 2}), _ok(415, None, 'ไม่ใช่ PDF', 'empty')]},
    {'id': 'job-preview-page', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/jobs/{job_id}/preview/pages/{page}',
     'title': 'ภาพหน้า PDF', 'summary': 'หน้าเดียวเป็น PNG/JPEG (เลือกไฟล์ที่เล็กกว่า)',
     'auth': USER_AUTH, 'headers': [],
     'params': [JOB_ID, _param('page', 'path', 'int', 'หน้าที่ (เริ่ม 1)')],
     'responses': [_ok(200, None, 'PNG / JPEG', 'file'), _err(404, 'ไม่มีหน้านี้')]},
    {'id': 'job-export', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs/{job_id}/export',
     'title': 'ดาวน์โหลดงานทั้งก้อน (JSON)', 'summary': 'เหมือน /api/user/jobs/{job_id} แต่เป็นไฟล์',
     'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'responses': [_ok(200, None, '<ชื่อไฟล์>.json', 'file'), _err(404, 'ไม่พบงาน')]},
    {'id': 'job-prompts', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/jobs/{job_id}/prompts',
     'title': 'Prompt ที่งานนี้ใช้', 'summary': 'คำถาม Hybrid ตามเวอร์ชันที่ล็อกไว้กับงาน ใช้ก่อน Retry แบบแก้ prompt',
     'auth': USER_AUTH, 'headers': [],
     'params': [JOB_ID, _param('doc_type', 'query', 'code', 'ประเภทที่จะดู', False),
                _param('source_set_id', 'query', 'string', 'ลองใช้ชุดอื่นเป็นต้นแบบ', False)],
     'responses': [_ok(200, {'doc_types': ['INVOICE'], 'editable_type': 'INVOICE', 'mode': 'hybrid',
                             'groups': [{'id': 'parties', 'title': '…', 'keys': ['…'], 'enabled': True, 'question': '…'}],
                             'available_types': ['INVOICE'], 'job_id': EXAMPLE_JOB, 'editor': {'…': '…'},
                             'library': {'…': '…'}}),
                   _err(400, 'ประเภทไม่อยู่ในผลของงานนี้')]},
    {'id': 'job-retry-history', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/jobs/{job_id}/retry-history',
     'title': 'ประวัติการ Retry', 'summary': 'ทุกครั้งที่ Retry ไฟล์เดียวกัน พร้อม prompt ที่เปลี่ยน',
     'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'responses': [_ok(200, {'job_id': EXAMPLE_JOB, 'attempts': [{'id': EXAMPLE_JOB, 'retry_of': '', 'created_at': '…',
                                                                 'status': 'completed', 'prompt_changes': {}}]})]},
    {'id': 'job-retry', 'group': 'user', 'status': 'live', 'method': 'POST',
     'path': '/api/user/jobs/{job_id}/retry-extraction',
     'title': 'Retry เฉพาะ extraction',
     'summary': 'ใช้ข้อความ OCR เดิม สร้างงานใหม่ด้วย prompt ที่แก้ · request_id เดิม = ได้งานเดิมคืน',
     'auth': USER_AUTH, 'headers': [], 'params': [JOB_ID],
     'body': {'request_id': 'retry_8f3a2b1c', 'prompt_sets': {'INVOICE': {'set_id': EXAMPLE_SET, 'value': {'…': '…'}}}},
     'responses': [_ok(202, {'job': {'id': '…', 'status': 'queued_extraction', 'retry_of': EXAMPLE_JOB, '…': '…'}}),
                   _err(400, 'body ไม่ครบหรือแก้ประเภทที่ไม่อยู่ในงาน'),
                   _ok(409, _ERROR, 'งานเดิมยังประมวลผลไม่เสร็จ'),
                   _ok(503, _ERROR, 'extraction server ไม่พร้อม')]},
    {'id': 'batch-zip', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/batches/{batch_id}/user-json.zip',
     'title': 'ดาวน์โหลด User JSON ทั้งชุด (ZIP)',
     'summary': '1 ไฟล์ JSON ต่อ 1 เอกสารต้นฉบับ พร้อม manifest.json · ใช้ผล Retry ล่าสุดถ้ามี',
     'auth': USER_AUTH, 'headers': [],
     'params': [_param('batch_id', 'path', 'string (32 hex)', 'รหัสชุด')],
     'responses': [_ok(200, None, 'user-json-YYYYMMDD-HHMM.zip', 'file'),
                   _ok(409, _ERROR, 'ยังไม่มีไฟล์ที่เสร็จในชุดนี้'), _err(404, 'ไม่พบชุด')]},
    {'id': 'preview-create', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/previews',
     'title': 'ตัวอย่าง PDF ก่อนส่ง', 'summary': 'เก็บชั่วคราว 30 นาที ไม่สร้างงานและไม่เข้าประวัติ',
     'auth': USER_AUTH, 'headers': [], 'params': [_param('file', 'form', 'file', 'PDF')], 'multipart': True,
     'responses': [_ok(201, {'preview_id': '0f1e2d3c4b5a69788796a5b4c3d2e1f0', 'pages': 2}),
                   _err(400, 'ไม่ใช่ PDF หรือเปิดไม่ได้'), _err(413, 'ไฟล์ใหญ่เกิน')]},
    {'id': 'preview-page', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/previews/{preview_id}/pages/{page}',
     'title': 'ภาพหน้าของตัวอย่าง', 'summary': 'หน้าเดียวเป็น PNG/JPEG',
     'auth': USER_AUTH, 'headers': [],
     'params': [_param('preview_id', 'path', 'string (32 hex)', 'จาก /api/user/previews'),
                _param('page', 'path', 'int', 'หน้าที่ (เริ่ม 1)')],
     'responses': [_ok(200, None, 'PNG / JPEG', 'file'), _err(404, 'หมดอายุหรือไม่มีหน้านี้')]},
    {'id': 'preview-delete', 'group': 'user', 'status': 'live', 'method': 'DELETE',
     'path': '/api/user/previews/{preview_id}',
     'title': 'ลบตัวอย่าง', 'summary': 'หน้าเว็บเรียกตอนปิดตัวอย่าง',
     'auth': USER_AUTH, 'headers': [], 'params': [_param('preview_id', 'path', 'string (32 hex)', 'รหัสตัวอย่าง')],
     'responses': [_ok(204, None, 'ลบแล้ว', 'empty'), _err(404, 'ไม่พบ')]},
    {'id': 'doc-types', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/doc-types',
     'title': 'Doc types ของ User', 'summary': 'ประเภทเอกสารตามสัญญา User JSON พร้อม field ที่ถามและ Required',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'types': [{'code': 'INVOICE', 'user_code': 'INV', 'label': 'ใบแจ้งหนี้', 'groups': ['…'],
                                        'choices': {'…': '…'}, 'mandatory': ['…'], 'field_count': 24, 'sets': ['…']}],
                             'mode': 'hybrid', 'extras': False, 'revision': 17, 'requirable': ['…']})]},
    {'id': 'doc-type-fields', 'group': 'user', 'status': 'live', 'method': 'PUT', 'path': '/api/user/doc-types/fields',
     'title': 'บันทึก field ของ Doc types', 'summary': 'เปิด/ปิด field และ Required หลายประเภทพร้อมกัน · revision กันการเขียนทับ',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'revision': 17, 'changes': [{'doc_type': 'INVOICE', 'field': 'due_date', 'state': 'required'}]},
     'responses': [_ok(200, {'saved': [{'set_id': EXAMPLE_SET, 'revision': 5, 'name': 'Invoice'}]}),
                   _err(400, 'ข้อมูลไม่ถูกต้อง'), _ok(409, _ERROR, 'มีคนแก้ก่อน ให้โหลดใหม่')]},
    {'id': 'prompt-sets', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/prompt-sets',
     'title': 'ชุด Hybrid prompt ทั้งหมด', 'summary': 'ชุดที่มี การผูกกับประเภทเอกสาร และรายการประเภท',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'sets': {EXAMPLE_SET: {'id': EXAMPLE_SET, 'name': 'Invoice', 'revision': 4, '…': '…'}},
                             'bindings': {'INVOICE': {'set_id': EXAMPLE_SET, 'revision': 4}}, 'types': ['…']})]},
    {'id': 'prompt-sets-change', 'group': 'user', 'status': 'live', 'method': 'POST', 'path': '/api/user/prompt-sets',
     'title': 'สร้าง/แก้/ผูกชุด prompt', 'summary': 'การกระทำตาม action ใน body · revision กันการเขียนทับ',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'action': 'bind', 'doc_type': 'INVOICE', 'set_id': EXAMPLE_SET, 'revision': 4},
     'responses': [_ok(200, {'record': {'id': EXAMPLE_SET, 'revision': 5, '…': '…'}}),
                   _err(400, 'ข้อมูลไม่ถูกต้อง'), _ok(409, _ERROR, 'มีคนแก้ก่อน')]},
    {'id': 'prompt-set-history', 'group': 'user', 'status': 'live', 'method': 'GET',
     'path': '/api/user/prompt-sets/{set_id}/history',
     'title': 'ประวัติเวอร์ชันของชุด', 'summary': 'ทุก revision ของชุดนี้',
     'auth': USER_AUTH, 'headers': [], 'params': [_param('set_id', 'path', 'string', 'รหัสชุด')],
     'responses': [_ok(200, {'versions': [{'revision': 4, 'created_at': '…', 'value': {'…': '…'}}]})]},
    {'id': 'prompt-groups', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/prompt-groups',
     'title': 'กลุ่มประเภทที่ใช้ prompt เดียวกัน', 'summary': 'ข้อมูลของกระดานลากวาง Hybrid prompts',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'responses': [_ok(200, {'groups': [{'set_id': EXAMPLE_SET, 'types': ['INVOICE', 'TAX_INVOICE'], '…': '…'}],
                             'revision': 17})]},
    {'id': 'prompt-groups-moves', 'group': 'user', 'status': 'live', 'method': 'POST',
     'path': '/api/user/prompt-groups/moves',
     'title': 'ย้ายประเภทระหว่างกลุ่ม', 'summary': 'ผลของการลากการ์ดบนกระดาน',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'revision': 17, 'moves': [{'doc_type': 'TAX_INVOICE', 'to': 'new'}]},
     'responses': [_ok(200, {'saved': [{'set_id': '…', 'revision': 1, 'name': '…'}]}),
                   _err(400, 'ข้อมูลไม่ถูกต้อง'), _ok(409, _ERROR, 'มีคนแก้ก่อน')]},
    {'id': 'prompt-groups-save', 'group': 'user', 'status': 'live', 'method': 'POST',
     'path': '/api/user/prompt-groups/save',
     'title': 'บันทึก prompt ของกลุ่ม', 'summary': 'คำสั่งและ field ของทั้งกลุ่ม',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'set_id': EXAMPLE_SET, 'revision': 4, 'value': {'…': '…'}},
     'responses': [_ok(200, {'set_id': EXAMPLE_SET, 'revision': 5, 'name': 'Invoice'}),
                   _err(400, 'ข้อมูลไม่ถูกต้อง'), _ok(409, _ERROR, 'มีคนแก้ก่อน')]},
    {'id': 'hybrid-prompts', 'group': 'user', 'status': 'live', 'method': 'GET', 'path': '/api/user/hybrid-prompts',
     'title': 'คำถาม Hybrid ของประเภทเอกสาร', 'summary': 'คำถามที่จะส่งให้ model จริง พร้อมตัวแก้ไข',
     'auth': USER_AUTH, 'headers': [],
     'params': [_param('doc_type', 'query', 'code (ซ้ำได้)', 'ประเภท (ค่าเริ่ม INVOICE)', False),
                _param('template', 'query', '1', 'ใช้สิ่งที่ประเภทนี้ถูกถามอยู่ตอนนี้เป็นต้นแบบ', False),
                _param('source_set_id', 'query', 'string', 'ใช้ชุดอื่นเป็นต้นแบบ', False)],
     'responses': [_ok(200, {'doc_types': ['INVOICE'], 'editable_type': 'INVOICE', 'mode': 'hybrid',
                             'groups': ['…'], 'prompt_prefix': '…', 'settings': {'extras': False, 'schema_first': False},
                             'has_requirement': True, 'editor': {'…': '…'}, 'library': {'…': '…'}}),
                   _err(400, 'ไม่รู้จักประเภท')]},
    {'id': 'hybrid-prompts-put', 'group': 'user', 'status': 'live', 'method': 'PUT', 'path': '/api/user/hybrid-prompts',
     'title': 'บันทึก prompt (รูปแบบเดิม)', 'summary': 'API รุ่นก่อนกระดานกลุ่ม ยังรองรับ',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'doc_type': 'INVOICE', 'revision': 4, 'value': {'instructions': {'…': '…'}, 'extra_groups': []}},
     'responses': [_ok(200, {'profile': {'revision': 5, 'value': {'…': '…'}}}),
                   _err(400, 'ข้อมูลไม่ครบ'), _ok(409, _ERROR, 'มีคนแก้ก่อน')]},
    {'id': 'hybrid-prompts-preview', 'group': 'user', 'status': 'live', 'method': 'POST',
     'path': '/api/user/hybrid-prompts/preview',
     'title': 'ดูตัวอย่างคำถามก่อนบันทึก', 'summary': 'ไม่บันทึกอะไร · ใส่ job_id เพื่อดูตามเวอร์ชันของงานนั้น',
     'auth': USER_AUTH, 'headers': [], 'params': [],
     'body': {'doc_type': 'INVOICE', 'set_id': EXAMPLE_SET, 'value': {'…': '…'}},
     'responses': [_ok(200, {'groups': ['…'], 'editor': {'…': '…'}, '…': '…'}), _err(400, 'ข้อมูลไม่ถูกต้อง')]},
]

CALLBACK = {
    'status': 'planned',
    'title': 'Callback เมื่องานเสร็จ',
    'summary': ('ถ้าส่ง callback_url มากับ POST /api/v1/jobs เมื่องานเสร็จ service จะ POST ผลไปที่ URL นั้น '
                'body เหมือน GET /api/v1/jobs/{job_id}/result ทุกไบต์ · ตอบ 2xx = ส่งสำเร็จ · '
                'ไม่ตาม redirect · ผู้รับควรกันซ้ำด้วย X-OCR-Event-Id'),
    'headers': [{'name': 'Content-Type', 'value': 'application/json'},
                {'name': 'X-OCR-Event', 'value': 'job.finished | job.updated (หลัง Retry)'},
                {'name': 'X-OCR-Event-Id', 'value': '7c1e2d3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f'},
                {'name': 'X-OCR-Signature', 'value': 'sha256=<HMAC-SHA256 ของ body ด้วย OCR_CALLBACK_SECRET>'}],
    'retries': ['30 วินาที', '2 นาที', '10 นาที', '1 ชั่วโมง', '6 ชั่วโมง', 'แล้วหยุด (กด Resend ได้ใน dashboard)'],
    'body': None,  # filled below: the v1 result example
    'verify': {
        'python': ('import hashlib, hmac\n\n'
                   'def verify(secret: bytes, body: bytes, header: str) -> bool:\n'
                   '    expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()\n'
                   '    return hmac.compare_digest(expected, header)\n'),
        'javascript': ("import crypto from 'node:crypto';\n\n"
                       'function verify(secret, rawBody, header) {\n'
                       "  const expected = 'sha256=' + crypto.createHmac('sha256', secret).update(rawBody).digest('hex');\n"
                       '  return header.length === expected.length &&\n'
                       '    crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(header));\n'
                       '}\n'),
    },
}
CALLBACK['body'] = next(r['body'] for e in ENDPOINTS if e['id'] == 'v1-result' for r in e['responses']
                        if r['code'] == 200)

# A list, not a dict: jsonify sorts keys, and v1 must come first.
GROUPS = [
    {'id': 'v1', 'title': 'Worker API v1', 'note': 'ใช้งานได้ — สำหรับ Service หลัก ใช้ API key'},
    {'id': 'user', 'title': 'User portal API', 'note': 'หน้า /user เรียกใช้ (หลัง firewall)'},
]

# Values a snippet puts in place of a path parameter.
_PATH_EXAMPLES = {'key_id': EXAMPLE_KEY_ID, 'job_id': EXAMPLE_JOB, 'batch_id': EXAMPLE_BATCH, 'set_id': EXAMPLE_SET,
                  'preview_id': '0f1e2d3c4b5a69788796a5b4c3d2e1f0', 'page': '1'}


_FORM_EXAMPLES = {'client_ref': 'ERP-000123', 'callback_url': 'https://main.example.com/ocr/callback',
                  'batch_id': EXAMPLE_BATCH, 'upload_id': 'file_0001'}


def _form_value(name):
    return _FORM_EXAMPLES.get(name, 'example')


def example_path(entry):
    path = entry['path']
    for name, value in _PATH_EXAMPLES.items():
        path = path.replace('{' + name + '}', value)
    query = [p for p in entry.get('params', []) if p['in'] == 'query' and p['required']]
    return path + ('?' + '&'.join(p['name'] + '=1' for p in query) if query else '')


def _headers(entry):
    """Headers a snippet sends: the entry's own required ones, and the page
    token on a write to /api/user (needed from another origin)."""
    out = [(h['name'], h['value']) for h in entry.get('headers', []) if h.get('required')]
    if entry['group'] == 'user' and entry['method'] not in ('GET', 'HEAD'):
        out.append(('X-OCR-User-Token', '{USER_TOKEN}'))
    if entry.get('body') is not None:
        out.append(('Content-Type', 'application/json'))
    return out


def snippets(entry):
    """curl, Python (requests) and JavaScript (fetch) for one entry."""
    url = '{BASE}' + example_path(entry)
    method, headers = entry['method'], _headers(entry)
    files = [p for p in entry.get('params', []) if p['in'] == 'form'] if entry.get('multipart') else []
    body = json.dumps(entry['body'], ensure_ascii=False) if entry.get('body') is not None else None

    curl = ['curl -X ' + method + " '" + url + "'"]
    curl += ["  -H '" + name + ': ' + value + "'" for name, value in headers]
    for field in files:
        curl.append("  -F '" + field['name'] + ('=@invoice.pdf' if field['type'] == 'file'
                                                 else '=' + _form_value(field['name'])) + "'")
    if body:
        curl.append("  -d '" + body + "'")

    py = ['import requests', '']
    call = 'requests.' + method.lower() + "('" + url + "'"
    if headers:
        call += ',\n    headers={' + ', '.join(repr(n) + ': ' + repr(v) for n, v in headers
                                       if n != 'Content-Type') + '}'
    if files:
        uploads = [f for f in files if f['type'] == 'file']
        fields = [f for f in files if f['type'] != 'file']
        if uploads:
            call += ",\n    files={'" + uploads[0]['name'] + "': open('invoice.pdf', 'rb')}"
        if fields:
            call += ',\n    data={' + ', '.join(repr(f['name']) + ': ' + repr(_form_value(f['name']))
                                          for f in fields) + '}'
    if body:
        call += ',\n    json=' + body.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    py.append('response = ' + call + ')')
    py.append('print(response.status_code, response.json() if response.content else None)')

    js = []
    options = ["method: '" + method + "'"]
    if headers:
        options.append('headers: {' + ', '.join(repr(n) + ': ' + repr(v) for n, v in headers
                                                if not (files and n == 'Content-Type')) + '}')
    if files:
        js.append('const form = new FormData();')
        for field in files:
            js.append("form.append('" + field['name'] + "', " +
                      ('fileInput.files[0]' if field['type'] == 'file' else repr(_form_value(field['name']))) + ');')
        options.append('body: form')
    elif body:
        options.append('body: JSON.stringify(' + body + ')')
    js.append("const response = await fetch('" + url + "', {\n  " + ',\n  '.join(options) + '\n});')
    js.append('console.log(response.status, await response.text());')
    return {'curl': ' \\\n'.join(curl), 'python': '\n'.join(py), 'javascript': '\n'.join(js)}


def reference():
    """What GET /api/user/api-reference answers."""
    return {'groups': GROUPS, 'callback': CALLBACK,
            'endpoints': [{**entry, 'snippets': snippets(entry)} for entry in ENDPOINTS]}
