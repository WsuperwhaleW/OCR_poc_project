"""Long-lived EasyOCR subprocess used by :mod:`easy_runtime`.

EasyOCR is installed alongside the web application now, and the Flask process
still does not import Torch, NumPy or EasyOCR -- **sharing an environment is not
sharing a process.** Commands and results are newline-delimited JSON; library
output is kept on stderr so stdout remains a machine-readable protocol.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path


LANGUAGES = [value.strip() for value in
             (os.environ.get("EASYOCR_LANGUAGES") or "th,en").split(",")
             if value.strip()]
DEVICE = (os.environ.get("EASYOCR_DEVICE") or "cpu").strip().lower()
MODEL_DIR = (os.environ.get("EASYOCR_MODEL_DIR") or "").strip()
DOWNLOAD = (os.environ.get("EASYOCR_DOWNLOAD") or ("0" if MODEL_DIR else "1")) \
    .strip().lower() in {"1", "true", "yes", "on"}

_reader = None
_runtime = None


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _imports():
    global _runtime
    if _runtime is None:
        with contextlib.redirect_stdout(sys.stderr):
            import easyocr
            import numpy as np
            import torch
            from PIL import Image
        _runtime = (easyocr, np, torch, Image)
    return _runtime


def runtime_info() -> dict:
    easyocr, _np, torch, _image = _imports()
    return {
        "engine": "EasyOCR", "backend": "easyocr",
        "easyOcrVersion": str(easyocr.__version__),
        "torchVersion": str(torch.__version__),
        "device": DEVICE, "languages": LANGUAGES,
        "detector": "CRAFT", "recognizer": "+".join(LANGUAGES),
        "modelDir": MODEL_DIR or None, "downloads": DOWNLOAD,
        "cudaAvailable": bool(torch.cuda.is_available()),
    }


def reader():
    global _reader
    if _reader is not None:
        return _reader
    easyocr, _np, _torch, _image = _imports()
    options = {
        "gpu": DEVICE not in {"", "cpu", "false", "0"},
        "download_enabled": DOWNLOAD,
        "verbose": False,
    }
    if MODEL_DIR:
        root = Path(MODEL_DIR).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        options["model_storage_directory"] = str(root)
    with contextlib.redirect_stdout(sys.stderr):
        _reader = easyocr.Reader(LANGUAGES, **options)
    return _reader


def decode_result(result, width: int, height: int) -> list[dict]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid page dimensions")
    lines = []
    for raw_polygon, text, raw_confidence in result:
        confidence = float(raw_confidence)
        pixels = [[float(x), float(y)] for x, y in raw_polygon]
        points = [[x / width, y / height] for x, y in pixels]
        values = [value for point in points for value in point]
        if (len(points) < 4 or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or not all(math.isfinite(value) and 0 <= value <= 1
                           for value in values)):
            raise ValueError("Invalid EasyOCR confidence or polygon")
        lines.append({
            "text": str(text), "confidence": confidence,
            "polygon": points,
            "rect": [min(p[0] for p in points), min(p[1] for p in points),
                     max(p[0] for p in points), max(p[1] for p in points)],
            "bbox": [min(p[0] for p in pixels), min(p[1] for p in pixels),
                     max(p[0] for p in pixels), max(p[1] for p in pixels)],
        })
    return lines


def read_page(path: Path) -> tuple[list[dict], int, int]:
    _easyocr, np, _torch, Image = _imports()
    with Image.open(path) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Page exceeds 40 megapixels")
        rgb = np.asarray(source.convert("RGB"))
        height, width = rgb.shape[:2]
    with contextlib.redirect_stdout(sys.stderr):
        result = reader().readtext(rgb, detail=1, paragraph=False)
    return decode_result(result, width, height), width, height


def handle_ocr(message: dict) -> None:
    request_id = str(message.get("requestId") or "")
    paths = [Path(str(value)).resolve() for value in message.get("pages") or []]
    if not request_id or not paths:
        raise ValueError("OCR command needs requestId and pages")
    if any(not path.is_file() for path in paths):
        raise ValueError("A prepared page file is missing")
    emit({"event": "loading", "requestId": request_id,
          "message": "Loading EasyOCR models" if _reader is None
                     else "Using loaded EasyOCR models"})
    started = time.perf_counter()
    for number, path in enumerate(paths, 1):
        emit({"event": "page_start", "requestId": request_id,
              "page": number, "total": len(paths)})
        page_started = time.perf_counter()
        lines, width, height = read_page(path)
        emit({"event": "page_result", "requestId": request_id,
              "page": number, "width": width, "height": height,
              "seconds": round(time.perf_counter() - page_started, 2),
              "lines": lines})
    emit({"event": "done", "requestId": request_id,
          "seconds": round(time.perf_counter() - started, 2),
          "pages": len(paths), "modelInfo": runtime_info()})


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", line_buffering=True)
    for raw in sys.stdin:
        request_id = ""
        try:
            message = json.loads(raw)
            request_id = str(message.get("requestId") or "")
            command = message.get("command")
            if command == "probe":
                emit({"event": "ready", "requestId": request_id,
                      "modelInfo": runtime_info()})
            elif command == "ocr":
                handle_ocr(message)
            elif command == "shutdown":
                emit({"event": "stopped", "requestId": request_id})
                return 0
            else:
                raise ValueError(f"Unknown worker command: {command}")
        except Exception as error:
            emit({"event": "error", "requestId": request_id,
                  "error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
