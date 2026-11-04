"""API keys and Idempotency-Key records for the Worker API (/api/v1).

Built 2026-11-04 at the user's request: "ระบบมี firewall อยู่แล้ว ทำแต่ api key พอ".
So there is no sign-in, session or role: `/user` sits behind the firewall, and
the only credential is an API key the main service sends as
`Authorization: Bearer <key>`. One key is one client, and a client sees only the
jobs it submitted.

Keys are random (`ocr_` + 32 bytes), shown once when created and stored only as
SHA-256: a stolen database gives no working key, and a key that random needs no
slow hash. Revoking takes effect on the next request.

The tables live in the User store's SQLite (`history.sqlite3`), created by
`initialize` from `UserStore.__init__`.
"""
import hashlib
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

KEY_PREFIX = 'ocr_'
IDEMPOTENCY_DAYS = 7
# `last_used_at` is written at most this often per key, not on every request.
LAST_USED_EVERY = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, hint TEXT NOT NULL, hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL, last_used_at TEXT NOT NULL DEFAULT '', revoked_at TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS api_idempotency (
            client_id TEXT NOT NULL, key TEXT NOT NULL, request_hash TEXT NOT NULL,
            job_id TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(client_id, key));
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(jobs)')}
    for column in ('client_id', 'client_ref', 'acknowledged_at'):
        if column not in columns:
            db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
    db.execute("CREATE INDEX IF NOT EXISTS jobs_client ON jobs(client_id) WHERE client_id<>''")


def hash_key(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def _public(row):
    return {k: row[k] for k in ('id', 'name', 'hint', 'created_at', 'last_used_at', 'revoked_at')} if row else None


def create_key(store, name):
    """A new key: (the key itself, shown once; its record)."""
    name = (name or '').strip() if isinstance(name, str) else ''
    if not 1 <= len(name) <= 80:
        raise ValueError('ตั้งชื่อ key 1–80 ตัวอักษร เช่น "ERP หลัก"')
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    record_id = uuid.uuid4().hex
    with store.connect() as db:
        db.execute('INSERT INTO api_keys(id,name,hint,hash,created_at) VALUES (?,?,?,?,?)',
                   (record_id, name, key[:8] + '…' + key[-4:], hash_key(key), now()))
        row = db.execute('SELECT * FROM api_keys WHERE id=?', (record_id,)).fetchone()
    return key, _public(row)


def list_keys(store):
    with store.connect() as db:
        rows = db.execute(
            "SELECT k.*, (SELECT COUNT(*) FROM jobs j WHERE j.client_id=k.id) AS jobs "
            "FROM api_keys k ORDER BY k.revoked_at<>'', k.created_at DESC").fetchall()
    return [{**_public(row), 'jobs': row['jobs']} for row in rows]


def revoke_key(store, key_id):
    with store.connect() as db:
        db.execute("UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at=''", (now(), key_id))
        row = db.execute('SELECT * FROM api_keys WHERE id=?', (key_id,)).fetchone()
    return _public(row)


def authenticate(store, header):
    """The client an `Authorization: Bearer <key>` header names, or None."""
    match = re.fullmatch(r'\s*Bearer\s+(\S+)\s*', header or '')
    if not match or not match.group(1).startswith(KEY_PREFIX):
        return None
    with store.connect() as db:
        row = db.execute("SELECT * FROM api_keys WHERE hash=? AND revoked_at=''",
                         (hash_key(match.group(1)),)).fetchone()
        if row is None:
            return None
        stamp = now()
        last = row['last_used_at']
        try:
            stale = not last or (datetime.fromisoformat(stamp) - datetime.fromisoformat(last)).total_seconds() > LAST_USED_EVERY
        except ValueError:
            stale = True
        if stale:
            db.execute('UPDATE api_keys SET last_used_at=? WHERE id=?', (stamp, row['id']))
    return _public(row)


def reserve(store, client_id, key, request_hash):
    """Claim an Idempotency-Key before creating the job.

    Returns None when this request now holds the key (create the job, then
    `complete`), the job id when the same request already made one,
    'conflict' when the key was used for a different request, or 'pending' when
    the same request is being accepted right now by another call.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=IDEMPOTENCY_DAYS)).isoformat()
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DELETE FROM api_idempotency WHERE created_at < ?', (cutoff,))
        row = db.execute('SELECT * FROM api_idempotency WHERE client_id=? AND key=?', (client_id, key)).fetchone()
        if row is None:
            try:
                db.execute('INSERT INTO api_idempotency VALUES (?,?,?,?,?)', (client_id, key, request_hash, '', now()))
            except sqlite3.IntegrityError:
                return 'pending'
            return None
    if row['request_hash'] != request_hash:
        return 'conflict'
    return row['job_id'] or 'pending'


def complete(store, client_id, key, job_id):
    with store.connect() as db:
        db.execute('UPDATE api_idempotency SET job_id=? WHERE client_id=? AND key=?', (job_id, client_id, key))


def release(store, client_id, key):
    """Give the key back when the job could not be created, so a retry may use it."""
    with store.connect() as db:
        db.execute("DELETE FROM api_idempotency WHERE client_id=? AND key=? AND job_id=''", (client_id, key))
