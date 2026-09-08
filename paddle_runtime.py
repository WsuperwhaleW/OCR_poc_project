"""Optional, persistent PaddleOCR subprocess management.

PaddleOCR lives in a separate virtual environment so its large dependency tree
cannot destabilise the small Flask application's environment.  Calls are
serialized through one long-lived worker: the models load once, and concurrent
queue threads wait visibly rather than loading duplicate copies.
"""

from __future__ import annotations

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


def interpreter() -> Path | None:
    configured = (os.environ.get("PADDLE_PYTHON") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() else None
    candidates = (
        config.BASE_DIR / ".venv-paddle" / "Scripts" / "python.exe",
        config.BASE_DIR / ".venv-paddle" / "bin" / "python",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def configured_status() -> dict:
    path = interpreter()
    return {
        "id": "paddle",
        "label": "Local PaddleOCR",
        "available": path is not None,
        "python": str(path) if path else None,
        "device": os.environ.get("PADDLE_DEVICE") or "cpu",
        "detector": os.environ.get("PADDLE_DETECTOR") or "PP-OCRv5_mobile_det",
        "recognizer": os.environ.get("PADDLE_RECOGNIZER") or "th_PP-OCRv5_mobile_rec",
        "mkldnn": _bool("PADDLE_MKLDNN", False),
        "reason": "" if path else (
            "PaddleOCR is optional. Create .venv-paddle and install "
            "requirements-paddle.txt, or set PADDLE_PYTHON."
        ),
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
        path = interpreter()
        if path is None:
            raise PaddleError(configured_status()["reason"])
        worker = config.BASE_DIR / "paddle_worker.py"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [str(path), "-u", str(worker)],
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
