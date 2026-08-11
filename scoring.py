"""Ground-truth lookup and accuracy scoring.

Shared by `compare.py` (CLI) and `app.py` (web page) so the terminal and the
browser can never report different numbers for the same run.
"""

import difflib
import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import config

ROOT = config.BASE_DIR
# Both are optional and relocatable; see config.py. Every read below checks
# existence first, so a deployment that ships neither still imports and runs --
# it simply has no case to score against.
SOLUTION = config.SOLUTION_DIR
MOCK = config.MOCK_DIR
OUT = SOLUTION / "out"

# Thai combining marks: tone marks plus above/below vowels. Scoring with these
# stripped separates "read the wrong word" from "read the word, lost the marks".
THAI_MARKS = re.compile(r"[ัิ-ฺ็-๎]")


def load_manifest():
    path = SOLUTION / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8")).get("cases", [])


def ground_truth_path(case_id: str):
    """Ground truth is Markdown (`solution/<id>.md`) so the tables render.

    `.txt` is still honoured if one exists, but nothing ships as .txt any more.
    Table markup is normalised away before scoring, so pipe tables here and HTML
    tables from the model compare equal.
    """
    for suffix in (".md", ".txt"):
        path = SOLUTION / f"{case_id}{suffix}"
        if path.exists():
            return path
    return None


def cases_index():
    """Cases that actually have a ground-truth file, keyed by id."""
    index = {}
    for case in load_manifest():
        gt = ground_truth_path(case["id"])
        if gt:
            index[case["id"]] = {**case, "ground_truth": gt, "pdf_path": MOCK / case["pdf"]}
    return index


def _slug(name: str) -> str:
    """Filename reduced to comparable form: no extension, letters/digits only."""
    return re.sub(r"[^a-z0-9]+", "", Path(name).stem.lower())


# Document reference numbers embedded in the filenames (BS…, F…TINV). These
# survive renaming far better than the full filename does.
_TOKEN = re.compile(r"(bs\d{6,}|f\d{6,}tinv)", re.I)


def _tokens(name: str) -> set:
    return {m.lower() for m in _TOKEN.findall(name)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def _hash_index():
    """sha256 -> case id for every source PDF. Cached; cheap to rebuild."""
    index = {}
    for case in cases_index().values():
        path = case["pdf_path"]
        if path.exists():
            index[sha256_bytes(path.read_bytes())] = case["id"]
    return index


def case_for_upload(filename: str = "", sha256: str = "", data: bytes = None):
    """Map an uploaded file to its ground-truth case.

    Tried in order of confidence, so a file that has been renamed or moved still
    resolves as long as its contents are unchanged:

      1. exact filename          2. filename ignoring case/punctuation
      3. explicit alias in the manifest
      4. sha256 of the contents  5. embedded document reference number

    Returns (case, how) or (None, None).
    """
    index = cases_index()

    if filename:
        name = Path(filename).name.lower()
        for case in index.values():
            if case["pdf"].lower() == name:
                return case, "filename"

        slug = _slug(filename)
        for case in index.values():
            if _slug(case["pdf"]) == slug:
                return case, "filename (normalised)"

        for case in index.values():
            if any(_slug(a) == slug for a in case.get("aliases", [])):
                return case, "manifest alias"

    if data is not None and not sha256:
        sha256 = sha256_bytes(data)
    if sha256:
        cid = _hash_index().get(sha256.lower())
        if cid:
            return index[cid], "file contents"

    if filename:
        found = _tokens(filename)
        if found:
            # An invoice and its receipt share a document number, so this is only
            # safe when exactly one case matches. Guessing between two would score
            # against the wrong ground truth and report a confidently wrong number.
            hits = [c for c in index.values() if found & _tokens(c["pdf"])]
            if len(hits) == 1:
                return hits[0], "document number"

    return None, None


def case_for_filename(filename: str):
    """Back-compat helper: filename-only lookup."""
    case, _ = case_for_upload(filename=filename)
    return case


def normalise(text: str, ignore_tables: bool = True) -> str:
    """Canonical form for comparison.

    Collapses whitespace and unifies table markup, so an HTML table and a Markdown
    pipe table with the same cells score identically -- the model flip-flops
    between them and that is formatting, not recognition.
    """
    text = unicodedata.normalize("NFC", text)
    if ignore_tables:
        # <br> is a line break however it is spelled -- an HTML table can hold a
        # real newline inside a cell, a Markdown pipe table has to encode it as
        # <br>, and the two must score the same.
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</?(table|tr|thead|tbody)>", "\n", text, flags=re.I)
        text = re.sub(r"</?(td|th)>", " ", text, flags=re.I)
        text = re.sub(r"^\s*\|?\s*-{2,}.*$", "", text, flags=re.M)  # pipe rules
        text = text.replace("\\|", "|")   # unescape Markdown-escaped pipes
        text = text.replace("|", " ")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def levenshtein(a: str, b: str) -> int:
    """Edit distance, O(min(len)) memory."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(expected: str, actual: str) -> dict:
    exp_c = expected.replace("\n", "").replace(" ", "")
    act_c = actual.replace("\n", "").replace(" ", "")
    cer = levenshtein(exp_c, act_c) / max(len(exp_c), 1)

    exp_w, act_w = expected.split(), actual.split()
    matcher = difflib.SequenceMatcher(None, exp_w, act_w, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    wer = 1 - matched / max(len(exp_w), 1)

    exp_nm = THAI_MARKS.sub("", exp_c)
    act_nm = THAI_MARKS.sub("", act_c)
    cer_nm = levenshtein(exp_nm, act_nm) / max(len(exp_nm), 1)

    return {
        "char_accuracy": round(max(0.0, 1 - cer), 4),
        "word_accuracy": round(max(0.0, 1 - wer), 4),
        "char_accuracy_no_marks": round(max(0.0, 1 - cer_nm), 4),
        "expected_chars": len(exp_c),
        "actual_chars": len(act_c),
        "expected_words": len(exp_w),
        "actual_words": len(act_w),
    }


def diff_lines(expected: str, actual: str, context: int = 1):
    return list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm="",
            n=context,
        )
    )


def evaluate(case, actual_text: str, ignore_tables: bool = True) -> dict:
    """Score one OCR result against its case's ground truth."""
    expected = normalise(case["ground_truth"].read_text("utf-8"), ignore_tables)
    actual = normalise(actual_text, ignore_tables)
    result = score(expected, actual)
    result.update(
        case=case["id"],
        pdf=case["pdf"],
        kind=case.get("kind", ""),
        diff=diff_lines(expected, actual),
    )
    return result
