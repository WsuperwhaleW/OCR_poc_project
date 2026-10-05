# Windows offline wheelhouse for Python 3.14.8

Target: **Windows x64 (AMD64), standard CPython 3.14.8**.
Built on 2026-10-05. Includes 16 wheels (31.0 MiB).

Includes application dependencies, PDF support (PyMuPDF), and HEIC/HEIF support
(pillow-heif). Optional PaddleOCR/EasyOCR dependencies and model weights are
separate installations.

## Compatibility

This bundle uses pillow-heif 1.1.0 for cp314 support. The root requirements.txt
caps pillow-heif below 1.0 and conflicts with this bundle. Use this folder's
requirements-lock.txt for installation, or requirements-target.txt for resolution.
All other package versions match the existing Python 3.13 Windows bundle.

Native wheels use the `cp314` ABI and are compatible across standard CPython
3.14.x patch releases. `3.14.8` is the requested target recorded
in manifest.json. Windows ARM64, 32-bit Python, and free-threaded Python require
different bundles.

## Install offline

Copy this folder beside app.py. From the project directory in PowerShell,
use a Windows x64 Python 3.14.8 installation:

```powershell
py -3.14 -c "import sys; print(sys.version)"
py -3.14 -m venv .venv-py314
.\.venv-py314\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse-windows-python-3.14.8 -r .\wheelhouse-windows-python-3.14.8\requirements-lock.txt
.\.venv-py314\Scripts\python.exe -m pip check
.\.venv-py314\Scripts\python.exe -c "import flask, requests, PIL, pymupdf, pillow_heif; print('dependencies OK')"
.\.venv-py314\Scripts\python.exe app.py
```

The launcher selects the 3.14 installation; check the first command's
output if you need the exact 3.14.8 patch. Direct virtual-environment
paths avoid requiring PowerShell activation. --no-index prevents network downloads.

## Rebuild online

Download into a new, empty folder:

```powershell
python -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --platform win_amd64 --python-version 3.14.8 --implementation cp --abi cp314 -r .\wheelhouse-windows-python-3.14.8\requirements-lock.txt -d .\wheelhouse-windows-python-3.14.8-rebuilt
```

## Build verification

Checked wheel ZIP integrity, compatible target tags, Requires-Python, complete
dependency closure using the target's Python/Windows markers, and target
requirements. An offline pip dry run resolved all 16 packages from local files
with --no-index. manifest.json records package versions and SHA-256 hashes.
The build machine has Python 3.13 only, so imports and execution on Python
3.14.8 have not been tested. Run the offline install/check/import commands
above on the target machine before using the application.
