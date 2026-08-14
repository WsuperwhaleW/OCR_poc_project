"""Every setting the deployment can change, and the helpers that read them.

Two jobs, both about being installable somewhere other than the machine it was
written on:

* **Paths.** Everything is resolved from this file's own directory, so the app
  runs from any location under any account without an edit. Each directory can
  still be pointed elsewhere by environment variable, which is what a server
  deployment wants -- read-only application directory, writable data directory
  somewhere else entirely.
* **Typed environment reads.** `int(os.environ[...])` throws on a typo in a
  service unit or a stray space in a `.env`, and it throws at import, before
  there is any log to explain it. These helpers fall back to the default and
  say so on stderr instead, so a mistyped knob degrades to the default rather
  than refusing to boot.

Modules keep owning their own settings -- this only supplies the readers and the
handful of paths more than one module needs. See `.env.example` for the full list
with its defaults.
"""

import os
import sys
from pathlib import Path

# resolve() so that a relative launch (`python app.py` from elsewhere, a service
# manager with a different working directory) still yields absolute paths.
BASE_DIR = Path(__file__).resolve().parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def say(line: str, stream=None):
    """Print a line without letting the console's encoding kill the process.

    A Windows console is cp1252 by default, and the paths printed at startup can
    contain characters it has no mapping for -- an installation directory with a
    non-Latin name, a model path, a document filename. Unprintable characters
    become '?' rather than a UnicodeEncodeError, which would otherwise take down
    the app while it was doing nothing more dangerous than reporting where its
    log file is.
    """
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    print(line.encode(encoding, "replace").decode(encoding, "replace"), file=stream)


def _warn(name, raw, err, fallback):
    say(f"[ocr] ignoring {name}={raw!r} ({err}); using {fallback}", sys.stderr)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    _warn(name, raw, "expected true/false", default)
    return default


def env_int(name: str, default: int, minimum: int = None, maximum: int = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        _warn(name, raw, "expected a whole number", default)
        return default
    # Clamped rather than rejected: a deployment asking for 200 pages gets the
    # ceiling and a line saying so, which is more useful than a refusal.
    if minimum is not None and value < minimum:
        _warn(name, raw, f"below minimum {minimum}", minimum)
        return minimum
    if maximum is not None and value > maximum:
        _warn(name, raw, f"above maximum {maximum}", maximum)
        return maximum
    return value


def env_float(name: str, default: float, minimum: float = None,
              maximum: float = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        _warn(name, raw, "expected a number", default)
        return default
    if minimum is not None and value < minimum:
        _warn(name, raw, f"below minimum {minimum}", minimum)
        return minimum
    if maximum is not None and value > maximum:
        _warn(name, raw, f"above maximum {maximum}", maximum)
        return maximum
    return value


def env_str(name: str, default: str, allow_empty: bool = False) -> str:
    """A plain text setting, stripped of surrounding whitespace.

    `allow_empty` is the whole reason this differs from its siblings above. They
    treat an empty value as unset and fall back, which is right for a number; for
    a string that is *sent to the model*, "set to nothing" and "not set" are
    different requests, and a setting whose off switch is an empty value needs to
    be able to say so. With it False an empty value still falls back, matching
    env_bool/env_int.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value and not allow_empty:
        return default
    return value


def env_path(name: str, default: Path) -> Path:
    """A directory setting, expanded and made absolute.

    `~` is expanded so a service file can say `~/ocr-data`, and a relative value
    is taken against BASE_DIR rather than the working directory -- the working
    directory of a service is whatever started it, which is not something a
    setting should silently depend on.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    path = Path(raw.strip()).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

# Benchmark fixtures. Both are OPTIONAL: without them the app is a plain
# upload-and-read tool, and the page hides the folder picker, the case picker and
# the accuracy columns on its own. Kept separate from each other because they are
# separable -- source documents with no ground truth are still worth having in
# the folder picker.
MOCK_DIR = env_path("OCR_MOCK_DIR", BASE_DIR / "mockOcr")
SOLUTION_DIR = env_path("OCR_SOLUTION_DIR", BASE_DIR / "solution")

# The one directory the app must be able to write to. Split out so the
# application directory can be mounted read-only.
LOG_DIR = env_path("OCR_LOG_DIR", BASE_DIR / "logs")

# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

# Loopback by default, deliberately. This app has no authentication, no rate
# limiting and no upload quota beyond a size cap, so binding every interface is a
# decision the deployment has to make on purpose -- set HOST=0.0.0.0 behind a
# reverse proxy or on a trusted network, never straight onto the internet.
HOST = os.environ.get("OCR_HOST") or os.environ.get("HOST") or "127.0.0.1"
PORT = env_int("PORT", 5000, minimum=1, maximum=65535)
