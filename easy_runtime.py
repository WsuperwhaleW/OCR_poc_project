"""Optional persistent EasyOCR subprocess management.

EasyOCR and Torch are installed into this application's own virtual environment,
so the worker runs on ``sys.executable`` and there is no second environment to
create.  **The worker subprocess is not an artefact of that separation and does
not go with it**: one venv is where the packages live, not permission for the
Flask process to import Torch.  It is what keeps this process free of ML
dependencies, what makes Stop able to cancel a read at all (a `readtext()` call
cannot be interrupted in-process), and what confines a native crash to a worker.
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


class EasyError(RuntimeError):
    pass


class EasyCancelled(EasyError):
    pass


def override() -> str:
    """An interpreter named explicitly, or "" for this app's own."""
    return (os.environ.get("EASYOCR_PYTHON") or "").strip()


def interpreter() -> Path | None:
    """The interpreter the worker runs on: this app's own, or an override.

    `EASYOCR_PYTHON` is the escape hatch for an environment this one cannot be
    -- a CUDA Torch build, most obviously. It is no longer how the ordinary
    setup is found.
    """
    configured = override()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() else None
    return Path(sys.executable).resolve() if sys.executable else None


def installed() -> bool:
    """Is EasyOCR importable HERE?

    `find_spec` searches the path without executing the package, so this does
    not pull Torch into the Flask process -- which is the whole reason the
    worker exists. It answers for this interpreter only; see `configured_status`.
    """
    try:
        return importlib.util.find_spec("easyocr") is not None
    except Exception:
        return False


def configured_status() -> dict:
    """What this reader is, and whether it can run, without starting anything.

    Availability is *the library is importable*, not *a directory exists*: with
    one shared environment the interpreter is always there, so the old test
    would call every machine ready and fail at the first read instead.
    """
    path = interpreter()
    external = bool(override())
    # An overridden interpreter is taken at its word rather than import-tested,
    # which would cost a subprocess on a status call; a broken one surfaces on
    # Re-check, which does start the worker.
    available = path is not None and (external or installed())
    languages = [value.strip() for value in
                 (os.environ.get("EASYOCR_LANGUAGES") or "th,en").split(",")
                 if value.strip()]
    if path is None and not external:
        # No interpreter and nothing named one: an embedded build with no
        # sys.executable. Nothing here can run at all.
        reason = "No Python interpreter to run the EasyOCR worker with."
    elif path is None:
        reason = (f"EASYOCR_PYTHON={override()} is not a file. Point it at a "
                  "python executable, or unset it to use this application's "
                  "own environment.")
    elif available:
        reason = ""
    else:
        reason = ("EasyOCR is optional and is not installed. "
                  "python -m pip install -r requirements-easyocr.txt into this "
                  "application's environment, or set EASYOCR_PYTHON to an "
                  "interpreter that has it.")
    return {
        "id": "easyocr", "label": "Local EasyOCR",
        "available": available, "python": str(path) if path else None,
        "device": os.environ.get("EASYOCR_DEVICE") or "cpu",
        "detector": "CRAFT", "recognizer": "+".join(languages),
        "languages": languages,
        "reason": reason,
    }


class EasyWorker:
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
            raise EasyError(state["reason"])
        path = state["python"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [path, "-u", str(config.BASE_DIR / "easy_worker.py")],
                cwd=str(config.BASE_DIR), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )
        except OSError as error:
            raise EasyError(f"Could not start EasyOCR: {error}") from error
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
        timeout = max(30, int(os.environ.get("EASYOCR_TIMEOUT") or 1800))
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
                    raise EasyError(f"EasyOCR worker could not accept the job: {error}")
            started = time.monotonic()
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        self.cancel_active()
                        raise EasyCancelled("EasyOCR run cancelled")
                    if time.monotonic() - started > timeout:
                        self.cancel_active()
                        raise EasyError(f"EasyOCR exceeded its {timeout}s timeout")
                    try:
                        raw = output.get(timeout=1.0)
                    except queue.Empty:
                        yield {"event": "heartbeat", "requestId": request_id}
                        continue
                    if raw is None:
                        detail = " | ".join(self._stderr)[-1200:]
                        raise EasyError("EasyOCR worker stopped"
                                        + (f": {detail}" if detail else ""))
                    try:
                        event = json.loads(raw)
                    except ValueError:
                        continue
                    if event.get("requestId") != request_id:
                        continue
                    if event.get("event") == "error":
                        raise EasyError(event.get("error") or "EasyOCR failed")
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
        for event in self._events({"command": "probe"}):
            if event.get("event") == "ready":
                return event.get("modelInfo") or {}
        raise EasyError("EasyOCR probe ended without a ready event")


WORKER = EasyWorker()


def status(probe=False) -> dict:
    result = configured_status()
    if not result["available"] or not probe:
        return result
    try:
        return {**result, "available": True,
                "modelInfo": WORKER.probe(), "reason": ""}
    except Exception as error:
        return {**result, "available": False, "reason": str(error)}
