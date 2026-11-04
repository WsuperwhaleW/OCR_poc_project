"""What the Service dashboard of /user shows: one JSON, worked out from local data.

Built 2026-11-04 at the user's request: "อยากให้เพิ่มหน้า api ทั้งหมดก่อน เพื่อใช้ดู
แล้วก็ตัว dashboard ด้วย" (worker-service plan, step 2b). `GET
/api/user/service/overview` answers with `Overview.get()`.

Rules it keeps:

* **No model server is asked anything.** Model state is the last probe and its
  age (`backends.last_probe`); a dashboard left open must not poll vLLM.
* **The OCR text is never read in bulk.** Queries select `json_extract(...)`
  columns, never `result`, except the User JSON check, which converts at most
  `QUALITY_SAMPLE` recent jobs and remembers each one by (id, updated_at).
* Cached for `TTL` seconds, so several open tabs cost one computation.

`PLANNED` lists the cards whose data comes with later steps of the plan
(admission limits, callbacks, retention); the page shows them greyed out.
"""
import json
import shutil
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import config
import userjson

TTL = 2.0
QUALITY_SAMPLE = 200
FINISHED = ('completed', 'partial', 'failed', 'interrupted')
FAILED = ('failed', 'interrupted')
WAITING = ('queued_ocr', 'queued_extraction')
RUNNING = ('ocr', 'extraction')
# Alert thresholds.
OLDEST_WAIT_ALERT = 600
FAILURE_RATE_ALERT = 0.05
FREE_DISK_ALERT = 0.10

PLANNED = [
    {'id': 'capacity', 'step': 5, 'title': 'Capacity & admission',
     'shows': 'งานค้างเทียบเพดาน (เช่น 412 / 500), คำขอที่ส่งถึง model ต่อ endpoint เทียบเพดาน 150, '
              'จำนวนครั้งที่ปฏิเสธด้วย 429 queue_full และ 507 spool_full ในชั่วโมงล่าสุด'},
    {'id': 'callbacks', 'step': 6, 'title': 'Callbacks',
     'shows': 'callback ที่รอส่ง / ส่งแล้ว / ล้มเหลว, ครั้งที่ลองซ้ำถัดไป และปุ่ม Resend'},
    {'id': 'retention', 'step': 7, 'title': 'Retention & storage',
     'shows': 'ระยะเก็บประวัติ (300 งาน / 7 วัน, error 7 วัน), พื้นที่ spool เทียบเพดาน, งานที่ยกเว้นการล้าง '
              '(กำลังรัน / callback ค้าง), เวลาล้างครั้งถัดไป และปุ่ม Delete history'},
]


def _parse(value):
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _seconds(start, end):
    a, b = _parse(start), _parse(end)
    return max(0.0, (b - a).total_seconds()) if a and b else None


def percentiles(values):
    """p50 / p95 / max by nearest rank, or None when there is nothing to rank."""
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None

    def rank(p):
        return values[min(len(values) - 1, max(0, -(-len(values) * p // 100) - 1))]
    return {'count': len(values), 'p50': round(rank(50), 2), 'p95': round(rank(95), 2), 'max': round(values[-1], 2)}


def _directory_bytes(directory):
    total = 0
    try:
        for path in directory.iterdir():
            if path.is_file():
                total += path.stat().st_size
            elif path.is_dir():
                total += _directory_bytes(path)
    except OSError:
        pass
    return total


class Overview:
    """The dashboard's numbers for one UserService, cached for TTL seconds."""

    def __init__(self, service, engine, started=None):
        self.service, self.engine = service, engine
        self.started = started or time.time()
        self.lock = threading.Lock()
        self.cached = (0.0, None)
        self.quality_cache = {}

    def get(self, now=None):
        with self.lock:
            at, value = self.cached
            if value is not None and now is None and time.monotonic() - at < TTL:
                return value
            value = self.compute(now)
            if now is None:
                self.cached = (time.monotonic(), value)
            return value

    # ------------------------------------------------------------------ parts
    def compute(self, now=None):
        now = now or datetime.now(timezone.utc)
        store = self.service.store
        cfg = store.settings()
        with store.connect() as db:
            queue = self._queue(db, cfg, now)
            rows = [dict(r) for r in db.execute(
                "SELECT id, filename, status, created_at, updated_at, retry_of, "
                "json_extract(timeline,'$.queued_ocr.started_at') AS queued_at, "
                "json_extract(timeline,'$.ocr.started_at') AS ocr_at, "
                "json_extract(result,'$.timing.ocr') AS ocr_seconds, "
                "json_extract(result,'$.timing.extraction') AS extraction_seconds, "
                "json_extract(result,'$.extracted.doc_types') AS doc_types "
                "FROM jobs WHERE status IN (?,?,?,?) AND julianday(updated_at) >= julianday(?)",
                (*FINISHED, (now - timedelta(hours=24)).isoformat()))]
            week = {r['status']: r['n'] for r in db.execute(
                'SELECT status, COUNT(*) AS n FROM jobs WHERE status IN (?,?,?,?) '
                'AND julianday(updated_at) >= julianday(?) GROUP BY status',
                (*FINISHED, (now - timedelta(days=7)).isoformat()))}
            failures = [dict(r) for r in db.execute(
                'SELECT id, filename, status, updated_at, error, timeline FROM jobs WHERE status IN (?,?) '
                'ORDER BY updated_at DESC LIMIT 10', FAILED)]
        health = self._health(cfg, now)
        throughput = self._throughput(rows, week, now)
        quality = self._quality(rows)
        out = {
            'generated_at': now.isoformat(),
            'health': health,
            'queue': queue,
            'requests': self._requests(cfg),
            'throughput': throughput,
            'speed': self._speed(rows),
            'quality': quality,
            'failures': [self._failure(row) for row in failures],
            'planned': PLANNED,
        }
        out['alerts'] = self._alerts(out)
        return out

    def _queue(self, db, cfg, now):
        counts = {r['status']: r['n'] for r in db.execute(
            'SELECT status, COUNT(*) AS n FROM jobs WHERE status IN (?,?,?,?) GROUP BY status', (*WAITING, *RUNNING))}
        with self.service.lock:
            live = {stage: sum(1 for f in futures if not f.done()) for stage, futures in self.service.active.items()}
        stages = {}
        for stage, key in (('ocr', 'ocr_concurrency'), ('extraction', 'extraction_concurrency')):
            oldest = db.execute(
                "SELECT MIN(COALESCE(json_extract(timeline,'$.queued_{0}.started_at'), created_at)) "
                "FROM jobs WHERE status=?".format(stage), ('queued_' + stage,)).fetchone()[0]
            stages[stage] = {'waiting': counts.get('queued_' + stage, 0),
                             'running': live.get(stage, 0), 'running_recorded': counts.get(stage, 0),
                             'concurrency': cfg[key], 'oldest_wait_seconds': _seconds(oldest, now.isoformat())}
        return {'stages': stages,
                'open_jobs': sum(counts.values()),
                'oldest_wait_seconds': max((s['oldest_wait_seconds'] or 0 for s in stages.values()), default=0)}

    def _health(self, cfg, now):
        models = {}
        for stage, key in (('ocr', 'ocr_url'), ('extraction', 'extraction_url')):
            # The raw probe (backends.probe): `reachable` and the served `models`.
            # `available` is worked out later by backends.status, per workspace.
            seen = self.engine.backends.last_probe(cfg[key])
            info = seen[1] if seen else {}
            served = [m.get('name') for m in info.get('models') or [] if isinstance(m, dict)]
            model = info.get('model') or (served[0] if served else None)
            models[stage] = {'url': cfg[key], 'kind': info.get('kind'), 'model': model,
                             'available': bool(info.get('reachable') and model) if seen else None,
                             'reason': info.get('reason') or ('' if model else 'ไม่มีโมเดลที่ server ให้บริการ'),
                             'probe_age_seconds': round(seen[0], 1) if seen else None}
        store = self.service.store
        try:
            disk = shutil.disk_usage(store.directory)
            free = {'free_bytes': disk.free, 'total_bytes': disk.total}
        except OSError:
            free = {'free_bytes': None, 'total_bytes': None}
        return {'app_version': config.app_version(), 'started_at': datetime.fromtimestamp(
                    self.started, timezone.utc).isoformat(), 'uptime_seconds': round(time.time() - self.started),
                'models': models, 'data_dir': str(store.directory),
                'data_bytes': _directory_bytes(store.directory), **free}

    def _requests(self, cfg):
        gates = self.engine._EXTRACTION_LIMITS.snapshot()
        return {'extraction': gates, 'extraction_limit': self.engine.EXTRACT_REQUEST_CONCURRENCY,
                # OCR requests have no admission gate yet; step 5 adds one.
                'ocr': None, 'ocr_url': cfg['ocr_url']}

    def _throughput(self, rows, week, now):
        hours = []
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=23)
        for index in range(24):
            hours.append({'hour': (start + timedelta(hours=index)).isoformat(),
                          'completed': 0, 'partial': 0, 'failed': 0})
        last_hour = Counter()
        for row in rows:
            finished = _parse(row['updated_at'])
            if not finished:
                continue
            kind = 'failed' if row['status'] in FAILED else row['status']
            slot = int((finished - start).total_seconds() // 3600)
            if 0 <= slot < 24:
                hours[slot][kind] += 1
            if (now - finished).total_seconds() <= 3600:
                last_hour[kind] += 1

        def totals(counter):
            return {'completed': counter.get('completed', 0), 'partial': counter.get('partial', 0),
                    'failed': counter.get('failed', 0) + counter.get('interrupted', 0),
                    'total': sum(counter.values())}
        return {'hours': hours, 'last_hour': totals(last_hour),
                'last_24h': totals(Counter(row['status'] for row in rows)), 'last_7d': totals(Counter(week)),
                'retries_24h': sum(1 for row in rows if row['retry_of'])}

    def _speed(self, rows):
        done = [row for row in rows if row['status'] not in FAILED]
        return {
            'queue': percentiles([_seconds(row['queued_at'], row['ocr_at']) for row in rows]),
            'ocr': percentiles([row['ocr_seconds'] for row in rows]),
            'extraction': percentiles([row['extraction_seconds'] for row in done]),
            'total': percentiles([_seconds(row['created_at'], row['updated_at']) for row in done]),
        }

    def _quality(self, rows):
        finished = len(rows)
        counts = Counter('failed' if row['status'] in FAILED else row['status'] for row in rows)
        types, unknown, typed = Counter(), 0, 0
        for row in rows:
            if row['status'] in FAILED:
                continue
            try:
                codes = json.loads(row['doc_types']) if row['doc_types'] else []
            except ValueError:
                codes = []
            typed += 1
            if codes:
                types.update(codes)
            else:
                unknown += 1
        sample = sorted((row for row in rows if row['status'] not in FAILED),
                        key=lambda row: row['updated_at'], reverse=True)[:QUALITY_SAMPLE]
        checks = [check for check in (self._user_json(row) for row in sample) if check]
        known = [check for check in checks if check['schema_valid'] is not None]

        def rate(n, d):
            return round(n / d, 4) if d else None
        return {'finished': finished,
                'completed_rate': rate(counts['completed'], finished), 'partial_rate': rate(counts['partial'], finished),
                'failed_rate': rate(counts['failed'], finished),
                'unknown_type_rate': rate(unknown, typed),
                'top_doc_types': [{'code': code, 'count': n} for code, n in types.most_common(8)],
                'user_json': {'checked': len(checks),
                              'schema_valid_rate': rate(sum(1 for c in known if c['schema_valid']), len(known)),
                              'avg_warnings': round(sum(c['warnings'] for c in checks) / len(checks), 2) if checks else None,
                              'avg_missing_fields': (round(sum(c['missing_fields'] for c in checks) / len(checks), 2)
                                                     if checks else None)}}

    def _user_json(self, row):
        key = (row['id'], row['updated_at'])
        if key not in self.quality_cache:
            job = self.service.store.get(row['id'])
            if job is None:
                return None
            try:
                report = userjson.convert(job)
                self.quality_cache[key] = {'schema_valid': report.get('schema_valid'),
                                           'warnings': len(report.get('warnings') or []),
                                           'missing_fields': len(report.get('missing_fields') or [])}
            except Exception:
                self.quality_cache[key] = {'schema_valid': False, 'warnings': 0, 'missing_fields': 0}
            if len(self.quality_cache) > QUALITY_SAMPLE * 3:
                for old in list(self.quality_cache)[:QUALITY_SAMPLE]:
                    self.quality_cache.pop(old, None)
        return self.quality_cache[key]

    @staticmethod
    def _failure(row):
        try:
            timeline = json.loads(row['timeline'] or '{}')
        except ValueError:
            timeline = {}
        stage = next((name for name, entry in timeline.items()
                      if isinstance(entry, dict) and entry.get('outcome')), '')
        return {'id': row['id'], 'filename': row['filename'], 'status': row['status'],
                'finished_at': row['updated_at'], 'stage': stage,
                'error': (row['error'] or '').strip().splitlines()[0][:300] if (row['error'] or '').strip() else ''}

    @staticmethod
    def _alerts(out):
        alerts = []
        for stage, model in out['health']['models'].items():
            label = 'OCR' if stage == 'ocr' else 'Extraction'
            if model['available'] is None:
                alerts.append({'level': 'info', 'text': f'{label} server ยังไม่ถูกตรวจตั้งแต่เปิด service'})
            elif not model['available']:
                alerts.append({'level': 'error', 'text': f"{label} server ไม่พร้อม: {model['reason'] or 'offline'}"})
        wait = out['queue']['oldest_wait_seconds'] or 0
        if wait > OLDEST_WAIT_ALERT:
            alerts.append({'level': 'warn', 'text': f'มีงานรอคิวนานกว่า {int(wait // 60)} นาที'})
        hour = out['throughput']['last_hour']
        if hour['total'] and hour['failed'] / hour['total'] > FAILURE_RATE_ALERT:
            alerts.append({'level': 'error',
                           'text': f"งานล้มเหลว {hour['failed']} จาก {hour['total']} งานในชั่วโมงล่าสุด"})
        free, total = out['health']['free_bytes'], out['health']['total_bytes']
        if free is not None and total and free / total < FREE_DISK_ALERT:
            alerts.append({'level': 'error', 'text': f'พื้นที่ดิสก์เหลือ {free / total:.0%}'})
        return alerts
