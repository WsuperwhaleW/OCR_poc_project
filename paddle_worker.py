"""Long-lived PaddleOCR subprocess used by :mod:`paddle_runtime`.

Paddle is installed alongside the web application now, and the main Flask
process still deliberately does not import Paddle, NumPy, PaddleX or their
transitive dependencies -- **sharing an environment is not sharing a process.**
It starts this module with its own interpreter (or the one `PADDLE_PYTHON`
names) and exchanges one JSON object per line.  Paddle's own console output is
redirected to stderr so stdout remains a machine-readable protocol.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path


DETECTOR = os.environ.get("PADDLE_DETECTOR") or "PP-OCRv5_mobile_det"
RECOGNIZER = os.environ.get("PADDLE_RECOGNIZER") or "th_PP-OCRv5_mobile_rec"
DEVICE = os.environ.get("PADDLE_DEVICE") or "cpu"
MKLDNN = (os.environ.get("PADDLE_MKLDNN") or "0").strip().lower() in {
    "1", "true", "yes", "on",
}
MODEL_ROOT = (os.environ.get("PADDLE_MODEL_DIR") or "").strip()

_engine = None
_runtime = None


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _torch_first():
    """Import Torch BEFORE Paddle, when both are installed. Order is load-bearing.

    Sharing one environment put them in one process for the first time, and on
    Windows `import paddle` then `import torch` dies with WinError 127 loading
    torch's `shm.dll` -- Paddle has already claimed the native runtime torch is
    reaching for. The reverse order is clean. It is not avoidable by not wanting
    torch here: `paddleocr` imports `paddlex`, which imports `modelscope`
    unconditionally, whose logger imports torch. So torch is loaded either way
    and this only decides when, which is why it costs nothing.

    A torch that will not import at all is left to fail where it is actually
    used -- reporting it from here would blame Paddle for someone else's break.
    """
    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # noqa: F401
    except Exception:
        pass


def _imports():
    """Import the optional runtime once, keeping its banners off stdout."""
    global _runtime
    if _runtime is None:
        with contextlib.redirect_stdout(sys.stderr):
            _torch_first()
            import paddle
            import paddleocr
            from PIL import Image
            from paddleocr import PaddleOCR
        _runtime = (paddle, paddleocr, Image, PaddleOCR)
    return _runtime


def runtime_info() -> dict:
    paddle, paddleocr, _image, _ocr = _imports()
    return {
        "engine": "PaddleOCR",
        "backend": "paddleocr",
        "paddlePaddleVersion": str(paddle.__version__),
        "paddleOcrVersion": str(paddleocr.__version__),
        "device": DEVICE,
        "detector": DETECTOR,
        "recognizer": RECOGNIZER,
        "mkldnn": MKLDNN,
        "modelDir": MODEL_ROOT or None,
    }


def engine():
    """Build the OCR pipeline lazily; later commands reuse the loaded models."""
    global _engine
    if _engine is not None:
        return _engine
    _paddle, _paddleocr, _image, PaddleOCR = _imports()
    options = {
        "device": DEVICE,
        "text_detection_model_name": DETECTOR,
        "text_recognition_model_name": RECOGNIZER,
        "enable_mkldnn": MKLDNN,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if MODEL_ROOT:
        root = Path(MODEL_ROOT).expanduser().resolve()
        detector = root / DETECTOR
        recognizer = root / RECOGNIZER
        missing = [str(path) for path in (detector, recognizer) if not path.is_dir()]
        if missing:
            raise ValueError("PADDLE_MODEL_DIR is missing: " + ", ".join(missing))
        options.update({
            "text_detection_model_dir": str(detector),
            "text_recognition_model_dir": str(recognizer),
        })
    with contextlib.redirect_stdout(sys.stderr):
        _engine = PaddleOCR(**options)
    return _engine


def decode_result(result, width: int, height: int) -> list[dict]:
    """Convert one Paddle result to JSON-safe, normalized text-line boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid page dimensions")
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    polygons = result["rec_polys"]
    lines = []
    for text, raw_score, raw_polygon in zip(texts, scores, polygons, strict=True):
        score = float(raw_score)
        pixels = [[float(x), float(y)] for x, y in raw_polygon]
        if len(pixels) < 4 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid PaddleOCR confidence or polygon")
        points = [[x / width, y / height] for x, y in pixels]
        if not all(math.isfinite(v) and 0 <= v <= 1 for point in points for v in point):
            raise ValueError("PaddleOCR returned coordinates outside the page")
        x1, y1 = min(p[0] for p in pixels), min(p[1] for p in pixels)
        x2, y2 = max(p[0] for p in pixels), max(p[1] for p in pixels)
        lines.append({
            "text": str(text),
            "confidence": score,
            "polygon": points,
            "rect": [min(p[0] for p in points), min(p[1] for p in points),
                     max(p[0] for p in points), max(p[1] for p in points)],
            "bbox": [x1, y1, x2, y2],
        })
    return lines


def read_page(path: Path) -> tuple[list[dict], int, int]:
    _paddle, _paddleocr, Image, _ocr = _imports()
    with Image.open(path) as source:
        width, height = source.size
        if width * height > 40_000_000:
            raise ValueError("Page exceeds 40 megapixels")
    with contextlib.redirect_stdout(sys.stderr):
        result = next(iter(engine().predict(str(path))))
    return decode_result(result, width, height), width, height


def handle_ocr(message: dict) -> None:
    request_id = str(message.get("requestId") or "")
    paths = [Path(str(value)).resolve() for value in message.get("pages") or []]
    if not request_id or not paths:
        raise ValueError("OCR command needs requestId and pages")
    if any(not path.is_file() for path in paths):
        raise ValueError("A prepared page file is missing")

    emit({"event": "loading", "requestId": request_id,
          "message": "Loading PaddleOCR models" if _engine is None
                     else "Using loaded PaddleOCR models"})
    started = time.perf_counter()
    pages = []
    for number, path in enumerate(paths, 1):
        emit({"event": "page_start", "requestId": request_id,
              "page": number, "total": len(paths)})
        page_started = time.perf_counter()
        lines, width, height = read_page(path)
        page = {
            "page": number,
            "width": width,
            "height": height,
            "seconds": round(time.perf_counter() - page_started, 2),
            "lines": lines,
        }
        pages.append(page)
        emit({"event": "page_result", "requestId": request_id, **page})
    emit({"event": "done", "requestId": request_id,
          "seconds": round(time.perf_counter() - started, 2),
          "pages": len(pages), "modelInfo": runtime_info()})


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
