# Linux offline wheelhouse

Target: **Linux x86_64, standard CPython 3.9** (manylinux2014 / glibc 2.17+).
Updated on 2026-10-28. Includes 24 wheels (39.55 MiB).

Includes the application dependencies, PDF/HEIC input support, and
`jsonschema==4.25.1` with all dependencies for User JSON v7 validation.
The Python 3.9 bundle uses `rpds-py==0.27.1`; the newer version in the
Windows bundles requires Python 3.10 or newer.

## Install offline

Copy this folder beside `app.py`, then run on the Linux target:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse -r wheelhouse/requirements-lock.txt
python -m pip check
python -c "import flask, requests, PIL, pymupdf, pillow_heif, jsonschema; print('dependencies OK')"
python app.py
```

The root `requirements.txt` is also satisfied by this bundle. Native wheels
require Linux x86_64 and Python 3.9; they cannot be used on Windows, ARM64,
or another Python ABI.

## Verification and rebuild

`manifest.json` records every wheel's SHA-256, package version, target tags
and byte size. Wheel ZIP integrity, Requires-Python, and the complete dependency
closure were checked using Linux/Python 3.9 environment markers. An offline
pip dry run resolved all 24 pinned packages from local files. Linux execution
was not tested on the Windows build machine.

To rebuild online, use a new empty destination:

```bash
python -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.9 --implementation cp --abi cp39 -r wheelhouse/requirements-lock.txt -d wheelhouse-rebuilt
```
