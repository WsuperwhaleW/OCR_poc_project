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

# Characters a reader cannot see. Whitespace of every flavour -- `\s` is
# Unicode-aware here, so it covers NBSP, the U+2000 block and the ideographic
# space -- plus zero-width, directional and soft-hyphen marks that arrive from
# PDF text layers and from Thai word-break conventions.
#
# Character accuracy is the headline number and it scores *content*: where a
# line wraps, how a cell is padded and whether a word carries a zero-width break
# are layout, not recognition. Stripping them from both sides is what stops a
# correct read from losing points for whitespace it had no way to reproduce.
#
# ZERO_WIDTH is the same set without the whitespace. `normalise` drops those
# outright while collapsing real whitespace to one space: a zero-width mark
# joins text, it does not separate it, so turning one into a space would invent
# a word boundary that is not on the page.
ZERO_WIDTH = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]+")
INVISIBLE = re.compile(r"[\s\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]+")

# Table markup, and nothing else. Every pattern here tolerates attributes: the
# model emits `<td colspan="5">` and `<td colspan="2" rowspan="2">` freely, and
# a bare-tag pattern does not match those at all -- it leaves the whole string
# in the text, where sixteen characters of markup score as if a reader could see
# them on the page. A cell that spans two columns is a layout decision about the
# same content, so it must not move the number in either direction.
TABLE_BLOCK = re.compile(
    r"</?(?:table|thead|tbody|tfoot|caption|colgroup|tr)\b[^>]*>", re.I)
TABLE_CELL = re.compile(r"</?(?:td|th|col)\b[^>]*>", re.I)
LINE_BREAK = re.compile(r"<br\b[^>]*>", re.I)

# A Markdown separator row: pipes, dashes, alignment colons and spaces, nothing
# else. Anchored on a real run of dashes so a line of prose that opens with a
# dash survives, and colon-tolerant so `|:---:|` is markup here too.
PIPE_RULE = re.compile(r"^[|\-: \t]*-{2,}[|\-: \t]*$", re.M)

# `--- page 2 ---` is this app's own separator between the pages of one document
# (`app.py` joins page transcripts with it), not something a model read off the
# paper. It comes out of both sides unconditionally, including under
# `--keep-tables`: the truth files only carry it where a page actually breaks,
# so leaving it in charges a multi-page read for a marker it did not write.
PAGE_MARKER = re.compile(r"^\s*-{2,}\s*page\s+\d+\s*-{2,}\s*$", re.I | re.M)


def content_only(text: str) -> str:
    """Text reduced to the characters that carry content."""
    return INVISIBLE.sub("", text)


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
    text = PAGE_MARKER.sub("", text)
    if ignore_tables:
        # <br> is a line break however it is spelled -- an HTML table can hold a
        # real newline inside a cell, a Markdown pipe table has to encode it as
        # <br>, and the two must score the same.
        text = LINE_BREAK.sub("\n", text)
        text = TABLE_BLOCK.sub("\n", text)
        text = TABLE_CELL.sub(" ", text)
        text = PIPE_RULE.sub("", text)
        text = text.replace("\\|", "|")   # unescape Markdown-escaped pipes
        text = text.replace("|", " ")
    # Real whitespace collapses to one space; zero-width marks go entirely.
    lines = [
        re.sub(r"\s+", " ", ZERO_WIDTH.sub("", ln)).strip()
        for ln in text.splitlines()
    ]
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
    # Character accuracy is content only: every invisible character comes out of
    # both sides first, so line breaks, indentation and cell padding cannot move
    # the number in either direction. See INVISIBLE.
    exp_c = content_only(expected)
    act_c = content_only(actual)
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
