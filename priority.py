"""User portal jobs go first: every other model request waits for them.

While a User job is queued or running, a request from anything else -- a stress
test, a random test, a read on the benchmark page -- waits here before it is
sent. A request already on the server finishes; only new ones wait, so a User
job waits behind at most one request per busy worker, never behind a queue.

The User pipeline says it is busy (`set_user_busy`, from its dispatcher) and its
own requests skip the gate: they run with `user_pipeline` set in their
workspace, which `workspace.bind` carries onto every worker thread they use.

Process-wide, not per workspace: the model server is shared by every tab, so a
User job in any of them holds back benchmark work in all the others.

The stress tests running now are kept here as well, so the User page can say
its jobs are sharing the server with one (`stress_runs`).
"""
import threading
import time

import workspace

_cond = threading.Condition()
_user_busy = False
_released_at = None     # perf_counter when the gate last opened; None if never shut
_waiting = 0            # requests held at the gate right now
_held_seconds = 0.0     # total seconds requests have spent held, since start

_stress_lock = threading.Lock()
_stress = {}            # thread -> what that stress run is


def set_user_busy(busy: bool):
    """The User pipeline has jobs queued or running (True), or none (False)."""
    global _user_busy, _released_at
    with _cond:
        if _user_busy and not busy:
            _released_at = time.perf_counter()
        _user_busy = bool(busy)
        if not busy:
            _cond.notify_all()


def user_busy() -> bool:
    with _cond:
        return _user_busy


def wait_turn(cancel=None, progress=None) -> float:
    """Block this request while User jobs have the server. Returns seconds held.

    A User request never waits. `cancel` (anything with `is_set`) lets a Stop
    reach a request that is held; the caller sees the flag set and behaves as it
    would for a Stop mid-request. `progress`, if given, is called about once a
    second while held, so a stall alert does not read yielding as hanging.
    """
    global _waiting, _held_seconds
    if workspace.get('user_pipeline', False):
        return 0.0
    with _cond:
        if not _user_busy:
            return 0.0
        began = time.perf_counter()
        _waiting += 1
        try:
            while _user_busy and not (cancel is not None and cancel.is_set()):
                _cond.wait(1.0)
                if progress is not None and _user_busy:
                    progress()
        finally:
            _waiting -= 1
            held = time.perf_counter() - began
            _held_seconds += held
    return held


def quiet_for() -> float:
    """Seconds since the gate last opened: 0 while it is shut, inf if it never was.

    A stress job held here hears nothing back, so its stall clock should only
    count from when the gate opened again.
    """
    with _cond:
        if _user_busy:
            return 0.0
        if _released_at is None:
            return float('inf')
        return time.perf_counter() - _released_at


def state() -> dict:
    with _cond:
        return {'user_busy': _user_busy, 'waiting': _waiting,
                'held_seconds': round(_held_seconds, 1)}


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
