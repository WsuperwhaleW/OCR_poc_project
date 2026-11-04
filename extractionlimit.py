"""A process-wide request ceiling per extraction endpoint, shared by all tabs.

Only admission is queued here; slots cover requests, never whole documents or
Hybrid workers. User requests retain their priority over background stress work.
"""
import heapq
import itertools
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlsplit


class Cancelled(RuntimeError):
    pass


def server_key(url):
    parts = urlsplit(url if '://' in url else 'http://' + url)
    path = parts.path.rstrip('/')
    for suffix in ('/v1/chat/completions/batch', '/v1/chat/completions', '/api/chat'):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    port = parts.port or (443 if parts.scheme.lower() == 'https' else 80)
    return parts.scheme.lower(), (parts.hostname or '').lower(), port, path


class _Gate:
    def __init__(self, limit):
        self.limit = limit
        self.condition = threading.Condition()
        self.waiters = []
        self.sequence = itertools.count()
        self.active = self.peak = 0

    @contextmanager
    def request(self, limit, priority, cancel, on_wait=None):
        with self.condition:
            self.limit = limit
            ticket = (priority, next(self.sequence))
            heapq.heappush(self.waiters, ticket)
            next_progress = time.monotonic() + .5
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled('extraction closed while waiting for a server slot')
                    if self.active < self.limit and self.waiters[0] == ticket:
                        heapq.heappop(self.waiters)
                        self.active += 1
                        self.peak = max(self.peak, self.active)
                        self.condition.notify_all()
                        break
                    self.condition.wait(.1)
                    if on_wait is not None and time.monotonic() >= next_progress:
                        on_wait()
                        next_progress = time.monotonic() + .5
            except BaseException:
                self.waiters.remove(ticket)
                heapq.heapify(self.waiters)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


class ServerLimits:
    def __init__(self):
        self.lock = threading.Lock()
        self.servers = {}

    def gate(self, url, limit):
        key = server_key(url)
        with self.lock:
            if key not in self.servers:
                self.servers[key] = _Gate(limit)
            return self.servers[key]

    @contextmanager
    def request(self, url, limit, priority=10, cancel=None, on_wait=None):
        with self.gate(url, limit).request(limit, priority, cancel, on_wait):
            yield

    def snapshot(self):
        """Each endpoint's gate as it stands, for the Service dashboard (2026-11-04).

        Read-only: in flight, waiting by priority (User = 0, background = 10, see
        priority.py) and the highest in-flight count since the process started.
        """
        with self.lock:
            gates = list(self.servers.items())
        out = []
        for (scheme, host, port, path), gate in gates:
            with gate.condition:
                waiting = [ticket[0] for ticket in gate.waiters]
                out.append({'url': f'{scheme}://{host}:{port}{path}', 'limit': gate.limit,
                            'active': gate.active, 'peak': gate.peak,
                            'waiting_user': sum(1 for p in waiting if p <= 0),
                            'waiting_background': sum(1 for p in waiting if p > 0)})
        return out
