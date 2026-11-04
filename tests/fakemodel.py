"""A local OpenAI-compatible model server for tests that run the REAL engine.

Built 2026-11-04 for the worker-service plan: before `app.py` is split into an
engine and entrypoints, the whole pipeline (image prep, OCR stream, grouping,
Hybrid extraction, image rechecks, User JSON) must be run end to end with only
the model HTTP replaced, so every refactor step can prove it sends the same
requests and produces the same results.

It looks like vLLM to `backends` (`/v1/models` owned_by "vllm", `/tokenize`,
`/version`; no `/props` or `/api/tags`). Every reply is a pure function of the
request body, so a run is repeatable and any change in what the engine sends
changes what it gets back:

* a request with no `response_format` and an image is a page read: the reply is
  one of a few realistic Thai document pages, chosen by the image's hash, so
  classification and grouping see real headings;
* a request with `response_format` (or a JSON skeleton in its prompt) gets a
  JSON object filled from the schema or skeleton, with values derived from the
  key name and the request hash.

It records each chat request's canonical SHA-256 and the highest number of
requests in flight at once (for admission tests).
"""
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OCR_MODEL = 'typhoon-ocr-fake'
TEXT_MODEL = 'qwen3-fake'
WINDOW = 32768

PAGES = (
    "# ใบแจ้งหนี้ / INVOICE\n\nเลขที่ INV-{h}\nวันที่ 01/10/2569\n\n"
    "ผู้ขาย บริษัท ตัวอย่าง จำกัด เลขประจำตัวผู้เสียภาษี 0105548000011 สำนักงานใหญ่\n"
    "ผู้ซื้อ บริษัท ลูกค้า จำกัด เลขประจำตัวผู้เสียภาษี 0105559000001 สาขาที่ 00001\n\n"
    "| ลำดับ | รายการ | จำนวน | จำนวนเงิน |\n|---|---|---|---|\n| 1 | สินค้า A | 2 | 1,000.00 |\n\n"
    "รวมเงิน 1,000.00\nภาษีมูลค่าเพิ่ม 7% 70.00\nรวมทั้งสิ้น 1,070.00",
    "# ใบเสร็จรับเงิน/ใบกำกับภาษี\n\nเลขที่ RC-{h}\nวันที่ 02/10/2569\n\n"
    "ผู้ขาย บริษัท รับเงิน จำกัด เลขประจำตัวผู้เสียภาษี 0105556000022 สำนักงานใหญ่\n"
    "ผู้ซื้อ บริษัท จ่ายเงิน จำกัด เลขประจำตัวผู้เสียภาษี 0105557000033 สำนักงานใหญ่\n\n"
    "อ้างอิงใบแจ้งหนี้ INV-{h}\nชำระโดยเช็คเลขที่ 18631287 วันที่ 02/10/2569\n\n"
    "มูลค่าสินค้า 2,000.00\nภาษีมูลค่าเพิ่ม 140.00\nจำนวนเงินรวม 2,140.00",
    "# ใบลดหนี้ / CREDIT NOTE\n\nเลขที่ CN-{h}\nวันที่ 03/10/2569\nอ้างอิงใบกำกับภาษี IV-{h}\n\n"
    "ผู้ขาย บริษัท ตัวอย่าง จำกัด เลขประจำตัวผู้เสียภาษี 0105548000011 สำนักงานใหญ่\n"
    "ผู้ซื้อ บริษัท ลูกค้า จำกัด เลขประจำตัวผู้เสียภาษี 0105559000001 สำนักงานใหญ่\n\n"
    "มูลค่าที่ลด 500.00\nภาษีมูลค่าเพิ่ม 35.00\nรวม 535.00",
    "# หนังสือรับรองการหักภาษี ณ ที่จ่าย ตามมาตรา 50 ทวิ\n\nเล่มที่ 12 เลขที่ WT-{h}\n\n"
    "ผู้มีหน้าที่หักภาษี ณ ที่จ่าย บริษัท ผู้จ่าย จำกัด เลขประจำตัวผู้เสียภาษี 0105555555555\n"
    "ที่อยู่ 1 ถนนสีลม กรุงเทพมหานคร 10500\n"
    "ผู้ถูกหักภาษี ณ ที่จ่าย บริษัท ผู้รับ จำกัด เลขประจำตัวผู้เสียภาษี 0105556090377\n\n"
    "| ประเภทเงินได้ | วันที่จ่าย | จำนวนเงินที่จ่าย | ภาษีที่หัก |\n|---|---|---|---|\n"
    "| ค่าบริการ | 01/12/2568 | 10,000.00 | 300.00 |\n\nรวมเงินที่จ่าย 10,000.00 ภาษีที่หักนำส่ง 300.00",
    "# ใบวางบิล / BILLING NOTE\n\nเลขที่ BL-{h}\nวันที่ 04/10/2569\n\n"
    "ผู้วางบิล บริษัท ตัวอย่าง จำกัด เลขประจำตัวผู้เสียภาษี 0105548000011\n"
    "ลูกค้า บริษัท ลูกค้า จำกัด\n\n| เลขที่ใบแจ้งหนี้ | จำนวนเงิน |\n|---|---|\n| INV-{h} | 1,070.00 |\n\n"
    "รวมเงินทั้งสิ้น 1,070.00",
    "# ใบกำกับภาษี / TAX INVOICE\n\nเลขที่ TX-{h}\nวันที่ 05/10/2569\n\n"
    "ผู้ขาย บริษัท ภาษี จำกัด เลขประจำตัวผู้เสียภาษี 0107536110412 สำนักงานใหญ่\n"
    "ผู้ซื้อ บริษัท ซื้อของ จำกัด เลขประจำตัวผู้เสียภาษี 0105559000044 สาขาที่ 00002\n\n"
    "มูลค่าสินค้า 3,000.00\nภาษีมูลค่าเพิ่ม 210.00\nรวมทั้งสิ้น 3,210.00",
)


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _images(body):
    out = []
    for message in body.get('messages') or []:
        content = message.get('content')
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    out.append(str((part.get('image_url') or {}).get('url', '')))
    return out


def _text(body):
    parts = []
    for message in body.get('messages') or []:
        content = message.get('content')
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(p.get('text', '')) for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return '\n'.join(parts)


def page_text(image_url):
    h = hashlib.sha256(image_url.encode('utf-8')).hexdigest()
    return PAGES[int(h[:8], 16) % len(PAGES)].replace('{h}', h[:6].upper())


def value_for(key, seed):
    lowered = key.lower()
    h = hashlib.sha256((seed + '|' + key).encode('utf-8')).hexdigest()
    if 'tax_id' in lowered:
        return '01055' + str(int(h[:10], 16))[:8].rjust(8, '0')
    if 'date' in lowered:
        return '01/10/2569'
    if 'location' in lowered:
        return 'สำนักงานใหญ่'
    if 'branch' in lowered:
        return '00000'
    if any(word in lowered for word in ('subtotal', 'vat', 'amount', 'total', 'remaining', 'paid', 'wht')):
        return '1,0' + str(int(h[:2], 16) % 10) + '0.00'
    if lowered in ('document_type',):
        return 'ใบแจ้งหนี้'
    if 'number' in lowered or lowered.endswith('_no') or 'cheque' in lowered:
        return 'NO-' + h[:6].upper()
    if 'name' in lowered:
        return 'บริษัท ตัวอย่าง จำกัด'
    if 'address' in lowered:
        return '1 ถนนสีลม กรุงเทพมหานคร 10500'
    return key + ' ' + h[:6]


def fill(schema, key, seed):
    if not isinstance(schema, dict):
        return value_for(key, seed)
    for option in ('anyOf', 'oneOf'):
        if isinstance(schema.get(option), list) and schema[option]:
            return fill(schema[option][0], key, seed)
    if isinstance(schema.get('enum'), list) and schema['enum']:
        return schema['enum'][0]
    kind = schema.get('type')
    if isinstance(kind, list):
        kind = next((k for k in kind if k != 'null'), 'string')
    if kind == 'object' or 'properties' in schema:
        return {k: fill(v, k, seed) for k, v in (schema.get('properties') or {}).items()}
    if kind == 'array':
        item = schema.get('items') or {}
        return [fill(item, key, seed)]
    if kind == 'boolean':
        return int(hashlib.sha256((seed + key).encode()).hexdigest()[0], 16) % 2 == 0
    if kind in ('integer', 'number'):
        return 1
    return value_for(key, seed)


def _skeleton(text):
    for match in reversed(list(re.finditer(r'\{[^{}]*\}', text))):
        try:
            value = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(value, dict) and value:
            return value
    return None


def reply_for(body):
    """(text, is_json) for one chat request."""
    seed = digest(body)
    response_format = body.get('response_format') or {}
    images = _images(body)
    if not response_format and images:
        return '\n\n'.join(page_text(url) for url in images), False
    schema = ((response_format.get('json_schema') or {}).get('schema')
              if response_format.get('type') == 'json_schema' else None)
    if schema:
        return json.dumps(fill(schema, 'root', seed), ensure_ascii=False), True
    skeleton = _skeleton(_text(body))
    if skeleton is not None:
        return json.dumps({k: (fill({}, k, seed) if not isinstance(v, (list, dict)) else v)
                           for k, v in skeleton.items()}, ensure_ascii=False), True
    return '{}', True


class FakeModelServer:
    """Start with `with FakeModelServer() as server:`; `server.url` is its base URL."""

    def __init__(self, model=TEXT_MODEL, delay=0.0):
        self.model, self.delay = model, delay
        self.requests, self.lock = [], threading.Lock()
        self.in_flight = self.max_in_flight = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def _json(self, code, value):
                data = json.dumps(value, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/v1/models':
                    return self._json(200, {'object': 'list', 'data': [
                        {'id': server.model, 'object': 'model', 'owned_by': 'vllm', 'max_model_len': WINDOW}]})
                if path == '/version':
                    return self._json(200, {'version': '0.0.0-fake'})
                if path == '/health':
                    return self._json(200, {})
                return self._json(404, {'error': 'not found'})

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                path = self.path.split('?')[0]
                if path == '/tokenize':
                    count = max(1, len(json.dumps(body.get('messages') or [], ensure_ascii=False)) // 4)
                    return self._json(200, {'count': count, 'max_model_len': WINDOW})
                if path != '/v1/chat/completions':
                    return self._json(404, {'error': 'not found'})
                with server.lock:
                    server.requests.append(digest(body))
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                try:
                    if server.delay:
                        time.sleep(server.delay)
                    text, _ = reply_for(body)
                    usage = {'prompt_tokens': max(1, len(_text(body)) // 4),
                             'completion_tokens': max(1, len(text) // 4)}
                    usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
                    if not body.get('stream'):
                        return self._json(200, {'id': 'cmpl-fake', 'object': 'chat.completion',
                            'model': body.get('model') or server.model,
                            'choices': [{'index': 0, 'finish_reason': 'stop',
                                         'message': {'role': 'assistant', 'content': text}}],
                            'usage': usage})
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    chunks = [text[i:i + 40] for i in range(0, len(text), 40)] or ['']
                    for piece in chunks:
                        event = {'id': 'cmpl-fake', 'object': 'chat.completion.chunk',
                                 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]}
                        self.wfile.write(b'data: ' + json.dumps(event, ensure_ascii=False).encode('utf-8') + b'\n\n')
                    final = {'id': 'cmpl-fake', 'object': 'chat.completion.chunk',
                             'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage}
                    self.wfile.write(b'data: ' + json.dumps(final).encode('utf-8') + b'\n\ndata: [DONE]\n\n')
                    self.wfile.flush()
                    self.close_connection = True
                finally:
                    with server.lock:
                        server.in_flight -= 1

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name='fake-model', daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
