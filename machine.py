"""What this instance is running ON -- the machine, not the model server.

**Built 2026-08-25 at the user's request**, for the Summary tab's environment
card: *have a banner card for my hardware and this env*. Every number in
`logs/runs.csv` is a wall clock, and a wall clock means nothing without the
machine that produced it -- CLAUDE.md quotes "~18-21 tok/s on an RTX 3060 Laptop"
in prose because nothing in this project has ever recorded it.

**Everything here is optional and nothing here raises.** A missing `nvidia-smi`,
a non-NVIDIA card, a locked-down host: each returns `None` for that field and the
card says *not detected* rather than inventing a value. That is the same rule
`preflight` follows for PDF support and the model server -- a deployment missing
an optional piece says so and keeps working.

**It is deliberately NOT logged.** `runlog.COLUMNS` is a file format and this is
a property of the process, not of a run; a machine column would be identical on
every row this instance ever writes and wrong on every row it inherits from
another. The Summary card says what THIS process is running on and, separately,
what the RUNS in view were inferred to have used -- two different claims, and
conflating them is exactly what `run_hardware`'s "inferred" label exists to
prevent.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

from config import say

# Probed once. Hardware does not change while the process runs, and `nvidia-smi`
# costs tens of milliseconds -- paid on the first summary rather than on import,
# so a machine without it never slows startup down.
_cache = None

# Long enough for a busy driver, short enough that a hung binary cannot hold a
# request open. A GPU that does not answer in three seconds is reported as not
# detected, which is the honest reading: nothing here is worth waiting for.
_SMI_TIMEOUT = 3.0


def _run(args: list) -> str:
    """One short read-only command, or `""` if anything at all goes wrong.

    Broad `except` on purpose. The failures are a missing binary, a driver that
    will not answer, a sandbox that refuses to fork and a timeout -- four
    different exceptions for one outcome, which is that this field is unknown.
    """
    try:
        # No console window on Windows: this runs behind a web request, and a
        # flashing terminal on every summary refresh would be a visible bug.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(args, capture_output=True, text=True,
                             timeout=_SMI_TIMEOUT, creationflags=flags)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _gpus() -> list:
    """NVIDIA cards by name, memory and driver, or `[]`.

    `nvidia-smi` only. AMD and Apple silicon report through entirely different
    tools, and a half-supported probe that names one vendor and silently omits
    another is worse than one that says it found nothing -- the card prints
    *not detected*, which is true on this machine and on those.
    """
    text = _run(["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"])
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            memory = round(float(parts[1]) / 1024.0, 1)
        except ValueError:
            memory = None
        gpus.append({"name": parts[0], "memory_gb": memory, "driver": parts[2]})
    return gpus


def _memory_gb():
    """Total system RAM in GB, or None where this platform will not say.

    Two implementations rather than a dependency: `psutil` would do it on every
    platform and `requirements.txt` says in as many words that this project does
    not add a dependency to solve a problem here. Windows and Linux cover every
    machine this has run on; anything else reports None.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024 ** 3), 1)
            return None
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / (1024 ** 3), 1)
    except Exception:
        return None


def _cpu_name() -> str:
    """A readable CPU name, falling back to whatever `platform` will give.

    `platform.processor()` returns a model string on Windows and frequently the
    bare architecture on Linux, so the registry/`/proc` reads below are what make
    this useful rather than decorative. Every one of them is best-effort.
    """
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return (platform.processor() or platform.machine() or "").strip()


def describe(force: bool = False) -> dict:
    """The machine, probed once and cached.

    `force` re-probes, which is only useful after a driver change -- nothing in
    the app calls it, and it exists so a reader does not have to restart to check
    a fix.
    """
    global _cache
    if _cache is not None and not force:
        return _cache
    gpus = _gpus()
    _cache = {
        "os": " ".join(x for x in (platform.system(), platform.release()) if x),
        "arch": platform.machine(),
        "cpu": _cpu_name(),
        # Logical processors, which is what llama.cpp's `-t` is counted in. It is
        # NOT the P-core count the performance notes recommend pinning to, and
        # the card does not pretend otherwise.
        "cpu_threads": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpus": gpus,
        # Said out loud rather than left as an empty list, because "no GPU" and
        # "could not ask" are different states and only the second is worth
        # anyone's time. `nvidia-smi` is the only probe here.
        "gpu_probe": "nvidia-smi" if gpus else "nvidia-smi (no answer)",
        "python": platform.python_version(),
    }
    return _cache


def summary() -> str:
    """One line for `preflight`, so the machine is in the log from boot.

    The same facts the card shows. Printed through `config.say` like everything
    else in that block -- a CPU name can carry characters a cp1252 console cannot
    encode, and a UnicodeEncodeError during startup would take the server with it.
    """
    info = describe()
    bits = [info["os"], info["cpu"]]
    if info["cpu_threads"]:
        bits.append(f"{info['cpu_threads']} threads")
    if info["memory_gb"]:
        bits.append(f"{info['memory_gb']} GB RAM")
    for gpu in info["gpus"]:
        bits.append(gpu["name"] + (f" {gpu['memory_gb']} GB" if gpu["memory_gb"] else ""))
    return ", ".join(b for b in bits if b)


def report():
    say(f"[ocr] machine: {summary() or 'not detected'}")
