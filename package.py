"""Build the deployable zip.

    python package.py                  # dist/thai-ocr-<date>.zip
    python package.py --no-fixtures    # app only, no mockOcr/ or solution/
    python package.py --out build.zip  # somewhere specific

Allow-list, not deny-list. A deny-list ships whatever was added since it was last
updated -- a scratch PDF, a database dump, a `.env` with a real endpoint in it --
because nobody remembers to add a new pattern before cutting a release. Listing
what goes IN means a new file is absent from the zip until someone says otherwise,
which is the failure that gets noticed immediately rather than the one that
doesn't.

Every path written into the archive is relative and under a single top-level
directory, so the zip cannot scatter files over whatever it is unpacked into.
"""

import argparse
import datetime as dt
import sys
import zipfile
from pathlib import Path

import config

# Paths are printed, and this project's own directory has a Thai name that a
# cp1252 console cannot encode -- see config.say.
say = config.say
BASE = config.BASE_DIR

# Individual files, in the order a reader should meet them.
FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "app.py",
    "config.py",
    "settings.py",
    "prompts.py",
    "backends.py",
    "grounding.py",
    "jobs.py",
    "runlog.py",
    "scoring.py",
    "verify.py",
    "compare.py",
    "package.py",
]

# (directory, glob). Only matching files are taken, so a stray artefact dropped
# in one of these does not travel.
TREES = [
    ("templates", "*.html"),
]

# Benchmark fixtures: the source documents and the ground truth to score against.
# Optional at runtime (the page hides the controls when they are absent), and
# optional here via --no-fixtures.
FIXTURES = [
    ("mockOcr", "*.pdf"),
    ("solution", "*.md"),
    ("solution", "manifest.json"),
]


def collect(no_fixtures=False):
    """Every (source path, archive path) pair, or exit if something is missing."""
    picked, missing = [], []

    for name in FILES:
        path = BASE / name
        (picked if path.is_file() else missing).append(
            (path, Path(name)) if path.is_file() else name)

    trees = list(TREES) + ([] if no_fixtures else FIXTURES)
    for folder, pattern in trees:
        root = BASE / folder
        if not root.is_dir():
            # Fixtures are genuinely optional; a missing templates/ is not.
            if (folder, pattern) in FIXTURES:
                say(f"  note: {folder}/ not present, skipping", sys.stderr)
                continue
            missing.append(f"{folder}/")
            continue
        found = sorted(root.glob(pattern))
        if not found and (folder, pattern) not in FIXTURES:
            missing.append(f"{folder}/{pattern}")
        for path in found:
            picked.append((path, path.relative_to(BASE)))

    if missing:
        say("missing, refusing to build:", sys.stderr)
        for name in missing:
            say(f"  {name}", sys.stderr)
        sys.exit(1)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output zip path")
    ap.add_argument("--no-fixtures", action="store_true",
                    help="omit mockOcr/ and solution/")
    ap.add_argument("--name", default="thai-ocr",
                    help="top-level directory inside the zip")
    args = ap.parse_args()

    stamp = dt.date.today().isoformat()
    root = f"{args.name}-{stamp}"
    out = Path(args.out) if args.out else BASE / "dist" / f"{root}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    picked = collect(args.no_fixtures)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for source, arcname in picked:
            zf.write(source, f"{root}/{arcname.as_posix()}")
        # An empty directory carries no file, but the app wants somewhere to
        # write before it has written anything -- so the placeholder ships and
        # the run log lands in the right place on first run.
        zf.writestr(f"{root}/logs/.gitkeep", "")

    size_mb = out.stat().st_size / 1024 ** 2
    say(f"{out}")
    say(f"  {len(picked) + 1} files, {size_mb:.1f} MB, unpacks to {root}/")
    if args.no_fixtures:
        say("  fixtures omitted -- upload-only, no accuracy scoring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
