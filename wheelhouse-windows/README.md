# Windows offline wheelhouse

Target: **Windows x64 (AMD64), standard CPython 3.13**. Built on 2026-10-05.

This folder contains the application dependencies from `requirements.txt`, including
PDF support (PyMuPDF) and HEIC/HEIF support (pillow-heif). Package versions match
the existing Linux wheelhouse where applicable, with Windows wheels for native packages.

Optional PaddleOCR/EasyOCR packages and OCR model weights are not included.
The external model server remains a separate installation.

## Install offline

Copy this folder beside `app.py` on the Windows target. Run PowerShell from the
project directory with Python 3.13 x64 installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse-windows -r .\wheelhouse-windows\requirements-lock.txt
.\.venv\Scripts\python.exe -c "import flask, requests, PIL, pymupdf, pillow_heif; print('dependencies OK')"
.\.venv\Scripts\python.exe app.py
```

Activation is optional because these commands use the virtual environment's
Python directly. `--no-index` prevents pip from downloading packages.

`requirements-lock.txt` pins every bundled package. The project-level
`requirements.txt` is also satisfied by this set:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse-windows -r .\requirements.txt
```

Native `cp313` wheels require Python 3.13. This bundle is not intended for
Windows ARM64, 32-bit Python, or a free-threaded Python build. Other Python
versions need their own wheelhouse.

## Rebuild online

Download into a new, empty folder to avoid mixing Python/platform targets:

```powershell
python -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 -r .\wheelhouse-windows\requirements-lock.txt -d .\wheelhouse-windows-rebuilt
```

`manifest.json` records each wheel's package name, version, byte size, and SHA-256
checksum. It also identifies the intended platform and Python version.
