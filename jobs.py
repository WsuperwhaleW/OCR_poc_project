"""A small in-process job queue with a resizable worker pool.

Documents are queued and worked by N threads. N should match llama-server's
`-np` slot count: more workers than slots does not add throughput, it just moves
the waiting from this queue into llama.cpp's, where it is invisible and cannot be
cancelled.

Deliberately in-process and in-memory. Jobs do not survive a restart, which is the
right trade for a local tool -- persistence would mean reconciling half-finished
work against a model server that may itself have restarted.
"""

import itertools
import threading
import time
import uuid
from collections import deque


# A ceiling on threads, not on concurrency policy: whether N in flight is a good
# idea is the model server's problem to report, not this queue's to prevent.
MAX_WORKERS = 64


class Cancelled(Exception):
    """Raised inside a worker when the job it is running was cancelled."""


class Job:
    __slots__ = ("id", "name", "kind", "detail", "status", "created", "started",
                 "finished", "pages_total", "pages_done", "stage", "result",
                 "error", "_cancel", "payload", "seq")

    _seq = itertools.count(1)

    def __init__(self, name, kind, detail, payload):
        self.id = uuid.uuid4().hex[:10]
        self.seq = next(Job._seq)
        self.name = name
        self.kind = kind              # "upload" | "case"
        self.detail = detail
        self.payload = payload        # bytes for upload, case id for case
        self.status = "queued"        # queued|running|done|failed|cancelled
        self.created = time.time()
        self.started = None
        self.finished = None
        self.pages_total = None
        self.pages_done = 0
        self.stage = ""               # human-readable current step
        self.result = None
        self.error = None
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise Cancelled()

    def to_dict(self, include_result=False):
        now = time.time()
        elapsed = None
        if self.started:
            elapsed = round((self.finished or now) - self.started, 1)
        data = {
            "id": self.id, "seq": self.seq, "name": self.name, "kind": self.kind,
            "detail": self.detail, "status": self.status, "stage": self.stage,
            "pages_total": self.pages_total, "pages_done": self.pages_done,
            "queued_for": round((self.started or now) - self.created, 1),
            "elapsed": elapsed, "error": self.error,
        }
        if include_result:
            data["result"] = self.result
        elif self.result:
            # Enough for a queue row without shipping the whole transcript.
            data["summary"] = {
                k: self.result.get(k)
                for k in ("tokens", "seconds", "page_count", "detail",
                          "truncated", "looped")
            }
            truth = self.result.get("truth")
            if truth and not truth.get("error"):
                data["summary"]["char_accuracy"] = truth.get("char_accuracy")
        return data


class JobQueue:
    def __init__(self, runner, workers=1):
        self._runner = runner
        self._lock = threading.Lock()
        self._pending = deque()
        self._jobs = {}
        self._wake = threading.Condition(self._lock)
        self._workers = []
        self._target_workers = 0
        self._stopping = False
        # Concurrent mode grows the pool to fit the batch: queue five documents
        # and five requests go out, rather than five queueing behind two workers.
        self._auto = False
        self.set_workers(workers)

    # -- pool ------------------------------------------------------------
    def set_auto_scale(self, on):
        with self._lock:
            self._auto = bool(on)

    @property
    def auto_scale(self):
        with self._lock:
            return self._auto

    def set_workers(self, count):
        """Grow or shrink the pool. Shrinking lets surplus threads finish and exit."""
        count = max(1, min(int(count), MAX_WORKERS))
        with self._lock:
            self._target_workers = count
            missing = count - len(self._workers)
            self._wake.notify_all()
        for _ in range(max(0, missing)):
            thread = threading.Thread(target=self._work, daemon=True)
            with self._lock:
                self._workers.append(thread)
            thread.start()
        return count

    @property
    def worker_count(self):
        with self._lock:
            return self._target_workers

    def _should_exit(self):
        # Called with the lock held.
        return self._stopping or len(self._workers) > self._target_workers

    def _work(self):
        me = threading.current_thread()
        while True:
            with self._lock:
                while not self._pending and not self._should_exit():
                    self._wake.wait(0.5)
                if self._should_exit():
                    if me in self._workers:
                        self._workers.remove(me)
                    return
                job = self._pending.popleft()

            if job.cancelled:
                job.status = "cancelled"
                job.finished = time.time()
                continue

            job.status = "running"
            job.started = time.time()
            job.stage = "starting"
            try:
                job.result = self._runner(job)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
                job.stage = "cancelled"
            except Exception as err:  # a failed job must not kill the worker
                job.status = "failed"
                job.error = f"{type(err).__name__}: {err}"
            finally:
                job.finished = time.time()

    # -- api -------------------------------------------------------------
    def submit(self, name, kind, detail, payload):
        return self.submit_many([(name, kind, detail, payload)])[0]

    def submit_many(self, specs):
        """Queue a batch so that none of it starts until all of it is queued.

        Submitting one at a time would let worker 1 begin document 1 while the
        rest of the batch is still being read off the request, which makes a
        concurrency test measure the upload as much as the model. Building the
        jobs first and admitting them under a single lock hold means the pool
        sees the whole batch at once and picks up N of them together.

        In concurrent mode the pool is grown to cover the batch *before* it is
        admitted, so the threads are already parked on the condition when the
        work lands and every document leaves for the model server together.
        """
        batch = [Job(name, kind, detail, payload) for name, kind, detail, payload in specs]

        with self._lock:
            auto = self._auto
            target = self._target_workers
            waiting = len(self._pending)
            running = sum(1 for j in self._jobs.values() if j.status == "running")
        if auto:
            need = waiting + running + len(batch)
            if need > target:
                self.set_workers(need)

        with self._lock:
            for job in batch:
                self._jobs[job.id] = job
                self._pending.append(job)
            self._wake.notify_all()
        return batch

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self):
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.seq)
        return jobs

    def cancel(self, job_id):
        job = self.get(job_id)
        if not job:
            return None
        if job.status == "queued":
            with self._lock:
                if job in self._pending:
                    self._pending.remove(job)
            job.status = "cancelled"
            job.finished = time.time()
        elif job.status == "running":
            job.cancel()          # worker notices at the next page boundary
        return job

    def clear_finished(self):
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status in ("done", "failed", "cancelled"):
                    del self._jobs[job_id]
                    removed += 1
        return removed

    def stats(self):
        counts = {"queued": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
        for job in self.list():
            counts[job.status] = counts.get(job.status, 0) + 1
        return {"counts": counts, "workers": self.worker_count,
                "auto_scale": self.auto_scale, "max_workers": MAX_WORKERS}
