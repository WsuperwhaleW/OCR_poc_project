"""Durable user uploads, settings and results, separate from benchmark history."""
import json
import copy
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import userprompts


def now():
    return datetime.now(timezone.utc).isoformat()


def defaults(max_upload_mb):
    return dict(ocr_concurrency=50, detail='medium', loop_guard=True,
                extraction_concurrency=64, shape='hybrid', extras=False,
                schema_first=False, fill_missing=True, validation_retry=False,
                ocr_url='http://127.0.0.1:8000',
                extraction_url='http://127.0.0.1:8001', no_cache=False,
                user_file_max=0, max_upload_mb=max_upload_mb)


def validate_settings(body, initial):
    if not isinstance(body, dict) or set(body) - set(initial):
        raise ValueError('Unknown settings or invalid settings object.')
    out = {**initial, **body}
    for key, low, high in (('ocr_concurrency', 1, 256), ('extraction_concurrency', 1, 256),
                           ('user_file_max', 0, 1000000), ('max_upload_mb', 1, initial['max_upload_mb'])):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{key} must be a whole number from {low} to {high}.')
    for key in ('loop_guard', 'extras', 'schema_first', 'no_cache', 'fill_missing', 'validation_retry'):
        if not isinstance(out[key], bool):
            raise ValueError(f'{key} must be true or false.')
    if out['detail'] not in ('original', 'medium', 'low'):
        raise ValueError('Unknown OCR detail.')
    if out['shape'] not in ('single', 'agentic', 'hybrid'):
        raise ValueError('Unknown extraction shape.')
    for key in ('ocr_url', 'extraction_url'):
        if not isinstance(out[key], str):
            raise ValueError(f'{key} must be an HTTP URL.')
        parsed = urlsplit(out[key])
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError(f'{key} must be an HTTP URL without credentials, query or fragment.')
        try:
            parsed.port
        except ValueError as err:
            raise ValueError(f'Invalid port in {key}.') from err
        out[key] = out[key].strip().rstrip('/')
    return out


class UserStore:
    def __init__(self, directory, max_upload_mb):
        self.directory = Path(directory)
        self.uploads = self.directory / 'uploads'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'history.sqlite3'
        self.initial = defaults(max_upload_mb)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, file_count INTEGER NOT NULL, configuration TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, upload_id TEXT NOT NULL,
                    filename TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    status TEXT NOT NULL, progress TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                    configuration TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(batch_id, upload_id));
                CREATE INDEX IF NOT EXISTS jobs_status_date ON jobs(status, created_at, id);
                CREATE TABLE IF NOT EXISTS prompt_profiles (
                    code TEXT PRIMARY KEY, revision INTEGER NOT NULL, value TEXT NOT NULL);
            ''')
            if 'timeline' not in {r['name'] for r in db.execute('PRAGMA table_info(jobs)')}:
                db.execute("ALTER TABLE jobs ADD COLUMN timeline TEXT NOT NULL DEFAULT '{}'")
            columns = {r['name'] for r in db.execute('PRAGMA table_info(jobs)')}
            for column in ('retry_of', 'original_id'):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            if 'prompt_changes' not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN prompt_changes TEXT NOT NULL DEFAULT '{}'")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS jobs_extraction_retry ON jobs(retry_of,upload_id) WHERE retry_of<>''")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def settings(self):
        with self.connect() as db:
            row = db.execute('SELECT value FROM settings WHERE id=1').fetchone()
        return {**self.initial, **(json.loads(row[0]) if row else {})}

    def save_settings(self, body):
        if not isinstance(body, dict):
            raise ValueError('Invalid settings object.')
        # Validate against the deployment ceiling, not the previously saved lower limit.
        value = validate_settings({**self.settings(), **body}, self.initial)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (1,?)', (json.dumps(value),))
        return value

    def create_batch(self, file_count, configuration):
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
            raise ValueError('Select at least one file.')
        maximum = configuration['user_file_max']
        if maximum and file_count > maximum:
            raise ValueError(f'เลือกได้สูงสุด {maximum} ไฟล์ต่อครั้ง')
        batch_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO batches VALUES (?,?,?)',
                       (batch_id, file_count, json.dumps(configuration, ensure_ascii=False)))
        return batch_id

    def prompt_profiles(self):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM prompt_profiles').fetchall()
        return {row['code']: {'revision': row['revision'], 'value': json.loads(row['value'])} for row in rows}

    def original(self, job):
        key = job.get('original_id') or job['id']
        if not re.fullmatch('[0-9a-f]{32}', key):
            raise ValueError('รหัสไฟล์ต้นฉบับไม่ถูกต้อง')
        return self.uploads / key

    def extraction_retry(self, parent_id, request_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE retry_of=? AND upload_id=?',
                             (parent_id, request_id)).fetchone()
        return self.decode(row)

    def retry_extraction(self, parent_id, request_id, configuration):
        if not isinstance(request_id, str) or not re.fullmatch('[a-zA-Z0-9_-]{8,64}', request_id):
            raise ValueError('รหัสคำขอ retry ไม่ถูกต้อง')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM jobs WHERE retry_of=? AND upload_id=?',
                                  (parent_id, request_id)).fetchone()
            if existing:
                return self.decode(existing)
            source = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (parent_id,)).fetchone())
            if not source or source['status'] not in ('completed', 'partial', 'failed', 'interrupted'):
                raise ValueError('รอให้งานเดิมหยุดประมวลผลก่อน retry')
            previous = source['result']
            if (not str(previous.get('text') or '').strip() or not isinstance(previous.get('pages'), list)
                    or not previous['pages'] or not all(isinstance(page, str) for page in previous['pages'])):
                raise ValueError('งานนี้ยังไม่มีข้อความ OCR ที่บันทึกไว้เพียงพอสำหรับ retry')
            if not self.original(source).is_file():
                raise ValueError('ไม่พบไฟล์ต้นฉบับของงานนี้')
            # Copy only OCR evidence. Extraction/classification warnings and
            # timings belong to the previous attempt, which remains untouched.
            result = {key: copy.deepcopy(previous[key]) for key in
                      ('text', 'pages', 'page_count', 'pages_total', 'page_stats') if key in previous}
            result['warnings'] = copy.deepcopy(previous.get('ocr_warnings', [warning for warning in
                previous.get('warnings', []) if str(warning).startswith(('อ่านได้ ', 'ข้อความหน้า '))]))
            result['ocr_warnings'] = copy.deepcopy(result['warnings'])
            at, batch_id, job_id = now(), uuid.uuid4().hex, uuid.uuid4().hex
            audit = {'version': 1, 'scope': 'job', 'parent_job_id': source['id'], 'created_at': at,
                     'changes': userprompts.changes_between(source['configuration'].get('prompt_profiles', {}),
                                                            configuration.get('prompt_profiles', {}))}
            db.execute('INSERT INTO batches VALUES (?,?,?)', (batch_id, 1, json.dumps(configuration, ensure_ascii=False)))
            timeline = {'queued_extraction': {'started_at': at, 'progress': 'ใช้ OCR เดิม · กำลังรอคิว Extraction'}}
            db.execute('''INSERT INTO jobs(id,batch_id,upload_id,filename,created_at,updated_at,status,progress,
                          configuration,result,timeline,retry_of,original_id,prompt_changes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (job_id, batch_id, request_id, source['filename'], at, at, 'queued_extraction',
                        'ใช้ OCR เดิม · กำลังรอคิว Extraction', json.dumps(configuration, ensure_ascii=False),
                        json.dumps(result, ensure_ascii=False), json.dumps(timeline, ensure_ascii=False),
                        source['id'], source.get('original_id') or source['id'], json.dumps(audit, ensure_ascii=False)))
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    def retry_history(self, job):
        root = job.get('original_id') or job['id']
        with self.connect() as db:
            rows = db.execute('SELECT id,retry_of,created_at,status,prompt_changes FROM jobs '
                              'WHERE id=? OR original_id=? ORDER BY rowid', (root, root)).fetchall()
        return [{**dict(row), 'prompt_changes': json.loads(row['prompt_changes'])} for row in rows]

    def save_prompt_profile(self, code, value, revision):
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError('เวอร์ชัน prompt ไม่ถูกต้อง')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM prompt_profiles WHERE code=?', (code,)).fetchone()
            if (row['revision'] if row else 0) != revision:
                return None
            next_revision = revision + 1
            db.execute('INSERT OR REPLACE INTO prompt_profiles VALUES (?,?,?)',
                       (code, next_revision, json.dumps(value, ensure_ascii=False)))
        return {'revision': next_revision, 'value': value}

    def batch(self, batch_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
        return {**dict(row), 'configuration': json.loads(row['configuration'])} if row else None

    def add_job(self, batch_id, upload_id, filename, content):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{8,64}', upload_id or ''):
            raise ValueError('Invalid upload ID.')
        job_id, at = uuid.uuid4().hex, now()
        path = self.uploads / job_id
        # Persist the original before making this job visible to the scheduler.
        path.write_bytes(content)
        try:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                batch = db.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
                if not batch:
                    raise ValueError('Upload batch not found. Select the files again.')
                old = db.execute('SELECT id FROM jobs WHERE batch_id=? AND upload_id=?',
                                 (batch_id, upload_id)).fetchone()
                if old:
                    path.unlink()
                    return self.get(old['id'])
                received = db.execute('SELECT COUNT(*) FROM jobs WHERE batch_id=?', (batch_id,)).fetchone()[0]
                if received >= batch['file_count']:
                    raise ValueError('All files in this batch have already been received.')
                db.execute('INSERT INTO jobs(id,batch_id,upload_id,filename,created_at,updated_at,status,progress,configuration,timeline) '
                           'VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (job_id, batch_id, upload_id, filename, at, at, 'queued_ocr',
                            'กำลังรอคิว OCR', batch['configuration'], json.dumps({
                                'queued_ocr': {'started_at': at, 'progress': 'กำลังรอคิว OCR'}})))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return self.get(job_id)

    @staticmethod
    def decode(row):
        if row is None:
            return None
        out = dict(row)
        for key in ('configuration', 'result', 'timeline', 'prompt_changes'):
            if key in out:
                out[key] = json.loads(out[key])
        out.pop('upload_id', None)
        return out

    def get(self, job_id):
        if not re.fullmatch('[0-9a-f]{32}', job_id or ''):
            return None
        with self.connect() as db:
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    def list(self, offset=0, limit=50):
        offset = max(0, int(offset))
        with self.connect() as db:
            rows = db.execute('SELECT id,batch_id,filename,created_at,updated_at,status,progress,error,retry_of '
                              'FROM jobs ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?',
                              (limit + 1, offset)).fetchall()
            counts = {r['status']: r['n'] for r in db.execute(
                'SELECT status,COUNT(*) AS n FROM jobs GROUP BY status')}
        return dict(jobs=[dict(r) for r in rows[:limit]], more=len(rows) > limit, counts=counts)

    def update(self, job_id, **values):
        if not values or set(values) - {'status', 'progress', 'error', 'result'}:
            raise ValueError('Invalid job update.')
        if 'result' in values:
            values['result'] = json.dumps(values['result'], ensure_ascii=False)
        values['updated_at'] = now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                return
            values['timeline'] = self._timeline(row, values.get('status', row['status']),
                                                values.get('progress', row['progress']), values['updated_at'])
            db.execute('UPDATE jobs SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE id=?',
                       (*values.values(), job_id))

    @staticmethod
    def _timeline(row, status, progress, at):
        timeline = json.loads(row['timeline'])
        previous = row['status']
        # Legacy history has no measured start time. Do not invent its durations.
        if previous in timeline and previous != status:
            timeline[previous]['finished_at'] = at
            if status in ('failed', 'interrupted'):
                timeline[previous]['outcome'] = status
        if status not in timeline:
            timeline[status] = {'started_at': at}
        timeline[status]['progress'] = progress
        return json.dumps(timeline, ensure_ascii=False)

    def claim(self, stage):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE status=? ORDER BY created_at,id LIMIT 1',
                             ('queued_' + stage,)).fetchone()
            if row:
                at = now()
                progress = 'กำลังเตรียมเอกสารสำหรับ OCR' if stage == 'ocr' else 'กำลังดึงข้อมูล'
                db.execute('UPDATE jobs SET status=?,progress=?,updated_at=?,timeline=? WHERE id=?',
                           (stage, progress, at, self._timeline(row, stage, progress, at), row['id']))
                row = db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone()
        return self.decode(row)

    def recover(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT * FROM jobs WHERE status IN ('ocr','extraction')").fetchall():
                at, progress = now(), 'งานถูกขัดจังหวะ'
                db.execute("UPDATE jobs SET status='interrupted',progress=?,error=?,updated_at=?,timeline=? WHERE id=?",
                           (progress, 'แอปรีสตาร์ตระหว่างประมวลผล กรุณาส่งไฟล์เพื่อเริ่มงานใหม่', at,
                            self._timeline(row, 'interrupted', progress, at), row['id']))
