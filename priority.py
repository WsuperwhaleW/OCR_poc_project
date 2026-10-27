"""User portal jobs go first -- scheduled by the vLLM server, not by the app.

The server runs `--scheduling-policy=priority` (vllm/podman-ocr-server/
podman-compose.yml), which orders its waiting queue by each request's
`priority` field, lower first, and preempts a running request of a later
priority when a sooner one needs the room. So the app holds nothing back: every
request is sent as it comes, and only the field says who goes first.

A User request keeps vLLM's default, 0. Every other chat request -- a stress
test, a random test, a read on the benchmark page -- is sent with `BACKGROUND`.
User requests are told apart by `user_pipeline` in their workspace, which
`workspace.bind` carries onto every worker thread they use.

A vLLM started without the priority policy refuses any priority but 0 (HTTP 400,
"Priority scheduling is not enabled"). That server is remembered and the request
sent again without the field: there everything is first come, first served, as
it was before.

The stress tests running now are kept here as well, so the User page can say
its jobs are sharing the server with one (`stress_runs`).
"""
import re
import threading
import time
from urllib.parse import urlsplit

import workspace

USER = 0            # vLLM's default; the field is not sent for it
BACKGROUND = 10     # everything else; any value above USER would do

_NOT_ENABLED = re.compile(r"priority scheduling is not enabled", re.I)
_fcfs_lock = threading.Lock()
_fcfs = set()           # servers (scheme://host:port) that refused a priority

_stress_lock = threading.Lock()
_stress = {}            # thread -> what that stress run is


def _server(url: str) -> str:
    parts = urlsplit(url or "")
    return f"{parts.scheme}://{parts.netloc}"


def request_priority() -> int:
    """The priority a request from this thread is sent with."""
    return USER if workspace.get('user_pipeline', False) else BACKGROUND


def tagged(url: str, body):
    """`body` with this thread's priority, or None where nothing is to be added:
    a User request, a body that is not a chat request (/tokenize), one that
    already says, or a server that has no priority scheduling."""
    if not isinstance(body, dict) or "messages" not in body or "priority" in body:
        return None
    value = request_priority()
    if value == USER:
        return None
    with _fcfs_lock:
        if _server(url) in _fcfs:
            return None
    return {**body, "priority": value}


def refused(url: str, res) -> bool:
    """Did this server refuse the priority because its policy is FCFS? Then it
    is remembered, and the caller sends the request again without one."""
    if getattr(res, "status_code", None) != 400 or not _NOT_ENABLED.search(res.text or ""):
        return False
    with _fcfs_lock:
        _fcfs.add(_server(url))
    return True


def stress_begin(**about):
    """This thread is running a stress test, described by `about`."""
    with _stress_lock:
        _stress[threading.current_thread()] = {**about, 'started_at': time.time()}


def stress_end():
    with _stress_lock:
        _stress.pop(threading.current_thread(), None)


def stress_runs() -> list:
    """The stress tests running now. A dead thread is not running, however it
    ended -- the same rule as `app.sweeps_running`."""
    with _stress_lock:
        for thread in [t for t in _stress if not t.is_alive()]:
            del _stress[thread]
        return [dict(run) for run in _stress.values()]
