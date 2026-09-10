"""Optional, persistent PaddleOCR subprocess management.

PaddleOCR is installed into this application's own virtual environment, so the
worker runs on ``sys.executable`` and there is no second environment to create.

**The worker subprocess survives that consolidation, and is not an accident of
it.** One venv is where the packages live; it is not permission for the Flask
process to import them.  Three things depend on the read happening elsewhere:
the standing rule that nothing is inferred in this process (no torch, no
numpy), Stop -- which cancels a read by killing the worker, and has no
in-process equivalent because ``predict()`` cannot be interrupted -- and crash
isolation, since a native fault in Paddle should cost a worker and not the
server.  Calls are serialized through one long-lived worker: the models load
once, and concurrent callers wait visibly rather than loading duplicate copies.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import config


class PaddleError(RuntimeError):
    pass


class PaddleCancelled(PaddleError):
    pass


def _bool(name: str, default=False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def override() -> str:
    """An interpreter named explicitly, or "" for this app's own."""
    return (os.environ.get("PADDLE_PYTHON") or "").strip()


def interpreter() -> Path | None:
    """The interpreter the worker runs on: this app's own, or an override.

    `PADDLE_PYTHON` is the escape hatch for an environment this one cannot be --
    a CUDA build, or a Python version Paddle supports and this app is not on.
    It is no longer how the ordinary setup is found.
    """
    configured = override()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() else None
    return Path(sys.executable).resolve() if sys.executable else None


def installed() -> bool:
    """Is PaddleOCR importable HERE?

    `find_spec` searches the path and does not execute the package, so asking
    this question does not drag Paddle into the Flask process -- which is the
    whole reason the worker exists. It answers for this interpreter only, so an
    overridden one is not tested by it; see `configured_status`.
    """
    try:
        return importlib.util.find_spec("paddleocr") is not None
    except Exception:
        return False


def configured_status() -> dict:
    """What this reader is, and whether it can run, without starting anything.

    Availability is now *the library is importable*, not *a directory exists*.
    Under the separate-venv layout those were nearly the same claim; sharing one
    environment makes the interpreter always present, so testing for it would
    report every machine as ready and fail at the first read instead.
    """
    path = interpreter()
    external = bool(override())
    # An overridden interpreter cannot be import-tested from here without paying
    # for a subprocess on a status call, so it is taken at its word and a broken
    # one surfaces on Re-check, which does start the worker.
    available = path is not None and (external or installed())
    if path is None and not external:
        # No interpreter and nothing named one: an embedded build with no
        # sys.executable. Nothing here can run at all.
        reason = "No Python interpreter to run the Paddle worker with."
    elif path is None:
        reason = (f"PADDLE_PYTHON={override()} is not a file. "
                  "Point it at a python executable, or unset it to use this "
                  "application's own environment.")
    elif available:
        reason = ""
    else:
        reason = ("PaddleOCR is optional and is not installed. "
                  "python -m pip install -r requirements-paddle.txt "
                  "into this application's environment, or set PADDLE_PYTHON "
                  "to an interpreter that has it.")
    return {
        "id": "paddle",
        "label": "Local PaddleOCR",
        "available": available,
        "python": str(path) if path else None,
        "device": os.environ.get("PADDLE_DEVICE") or "cpu",
        "detector": os.environ.get("PADDLE_DETECTOR") or "PP-OCRv5_mobile_det",
        "recognizer": os.environ.get("PADDLE_RECOGNIZER") or "th_PP-OCRv5_mobile_rec",
        "mkldnn": _bool("PADDLE_MKLDNN", False),
        "reason": reason,
    }


class PaddleWorker:
    def __init__(self):
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process = None
        self._output = None
        self._stderr = deque(maxlen=30)
        self._active = None

    def _reader(self, process, output):
        try:
            for line in process.stdout:
                output.put(line)
        finally:
            output.put(None)

    def _stderr_reader(self, process):
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def _start_locked(self):
        if self._process is not None and self._process.poll() is None:
            return
        state = configured_status()
        if not state["available"]:
            raise PaddleError(state["reason"])
        path = state["python"]
        worker = config.BASE_DIR / "paddle_worker.py"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [path, "-u", str(worker)],
                cwd=str(config.BASE_DIR), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )
        except OSError as error:
            raise PaddleError(f"Could not start PaddleOCR: {error}") from error
        output = queue.Queue()
        self._process, self._output = process, output
        self._stderr.clear()
        threading.Thread(target=self._reader, args=(process, output), daemon=True).start()
        threading.Thread(target=self._stderr_reader, args=(process,), daemon=True).start()

    def _stop_locked(self):
        process = self._process
        self._process = self._output = None
        self._active = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def cancel_active(self):
        with self._process_lock:
            self._stop_locked()

    def _events(self, command: dict, cancel=None):
        request_id = uuid.uuid4().hex
        command = {**command, "requestId": request_id}
        timeout = max(30, int(os.environ.get("PADDLE_TIMEOUT") or 1800))
        with self._request_lock:
            with self._process_lock:
                self._start_locked()
                process, output = self._process, self._output
                self._active = request_id
                try:
                    process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
                    process.stdin.flush()
                except Exception as error:
                    self._stop_locked()
                    raise PaddleError(f"PaddleOCR worker could not accept the job: {error}")

            started = time.monotonic()
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        self.cancel_active()
                        raise PaddleCancelled("PaddleOCR run cancelled")
                    if time.monotonic() - started > timeout:
                        self.cancel_active()
                        raise PaddleError(f"PaddleOCR exceeded its {timeout}s timeout")
                    try:
                        raw = output.get(timeout=1.0)
                    except queue.Empty:
                        yield {"event": "heartbeat", "requestId": request_id}
                        continue
                    if raw is None:
                        detail = " | ".join(self._stderr)[-1200:]
                        raise PaddleError("PaddleOCR worker stopped"
                                          + (f": {detail}" if detail else ""))
                    try:
                        event = json.loads(raw)
                    except ValueError:
                        continue
                    if event.get("requestId") != request_id:
                        continue
                    if event.get("event") == "error":
                        raise PaddleError(event.get("error") or "PaddleOCR failed")
                    yield event
                    if event.get("event") in {"done", "ready", "stopped"}:
                        return
            finally:
                with self._process_lock:
                    if self._active == request_id:
                        self._active = None

    def run_pages(self, paths: list[Path], cancel=None):
        yield from self._events(
            {"command": "ocr", "pages": [str(path) for path in paths]}, cancel)

    def probe(self) -> dict:
        # Importing Paddle can take longer than the one-second queue poll, so a
        # heartbeat may precede `ready`. Do not mistake that for a successful
        # probe with empty version information.
        for event in self._events({"command": "probe"}):
            if event.get("event") == "ready":
                return event.get("modelInfo") or {}
        raise PaddleError("PaddleOCR probe ended without a ready event")


WORKER = PaddleWorker()


def status(probe=False) -> dict:
    result = configured_status()
    if not result["available"] or not probe:
        return result
    try:
        info = WORKER.probe()
        return {**result, "available": True, "modelInfo": info, "reason": ""}
    except Exception as error:
        return {**result, "available": False, "reason": str(error)}
