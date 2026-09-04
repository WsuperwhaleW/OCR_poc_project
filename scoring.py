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

# ---------------------------------------------------------------------------
# The three scripts a page like this is written in, scored apart
#
# Added 2026-09-04 at the user's request -- *on scoring add thai char accuracy,
# eng char accuracy, number char accuracy*. One headline hides which of the
# three a model is bad at, and they fail for different reasons and are worth
# different amounts:
#
#   digits  a misread figure is a wrong amount, a wrong tax ID or a wrong date,
#           and every Mandatory field in the requirement is one of those. This
#           is the column to read first.
#   thai    tone marks and upper/lower vowels stack, so this is where a small
#           model degrades first when the page is downscaled.
#   latin   the easy third on most of these pages, and a model that is weak
#           HERE is weak at something other than the alphabet.
#
# **Digits are asked first, and Thai numerals are digits.** U+0E50-0E59 sit
# inside the Thai block, and a figure is a figure whichever numerals it is
# printed in -- sol003 prints its own in Thai. Classifying them as Thai would
# put the project's most important characters in the column least read.
#
# `latin` is the LATIN ALPHABET, which is what "English" means on these pages.
# It is not a claim that the text is English -- a Thai company's romanised name
# is Latin script and is not.
_DIGIT_CHARS = re.compile(r"[0-9٠-٩۰-۹๐-๙]")
_THAI_CHARS = re.compile(r"[฀-๿]")
_LATIN_CHARS = re.compile(r"[A-Za-zÀ-ɏ]")

# Order matters: the first test that matches wins, so digits are asked before
# the Thai block they partly sit inside.
SCRIPTS = ("thai", "latin", "digit")
_SCRIPT_TESTS = (("digit", _DIGIT_CHARS), ("thai", _THAI_CHARS),
                 ("latin", _LATIN_CHARS))

# Below this many characters of a script on a page, its rate is not reported at
# all. Three Latin characters score 0% or 100% and nothing in between, and a
# column of those would read as a model that cannot read English. The COUNT is
# still reported, so a blank rate says *there was not enough of this to measure*
# rather than *nobody looked* -- the same distinction the run log's blank cells
# make everywhere else.
SCRIPT_MIN_CHARS = 20


def script_of(ch: str):
    """Which of the three scripts a character belongs to, or None.

    Punctuation, symbols and spacing belong to none of them, deliberately: a
    comma is not evidence about any alphabet. That is also why the three rates
    do not add up to `char_accuracy` -- it is recall over every content
    character, and these are recall over three subsets of it.
    """
    for name, test in _SCRIPT_TESTS:
        if test.match(ch):
            return name
    return None


def script_counts(text: str) -> dict:
    """How many characters of each script `text` holds."""
    counts = dict.fromkeys(SCRIPTS, 0)
    for ch in text:
        name = script_of(ch)
        if name:
            counts[name] += 1
    return counts

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

# A printed blank to be filled in -- the ruled line after `วันที่/Date`, the run
# after `เช็ค หมายเลข`. It is a RULING, not text: the page draws a rule, the model
# spells it as dots or underscores, and a hand transcription writes nothing at
# all, so scoring it charges a correct read for the one thing on the page that
# carries no content. Same class as the table markup above, and it was costing
# more than that ever did -- 560 characters on sol001, 17 points of its score.
#
# Four is the threshold so an ellipsis in prose stays content. Verified against
# all ten ground-truth files: not one contains a run this long, so nothing a
# human transcribed is at risk.
#
# It goes with the table markup under `--keep-tables` rather than beside the page
# marker, for two reasons: a Markdown separator row is a run of dashes, so out
# here it would eat half of one and leave `| :|`; and that flag means compare
# literally -- it is the only way to reproduce a number taken before this, which
# it cannot be if it still drops the rules.
FILL_RULE = re.compile(r"[.…_–—-]{4,}")

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


def _case_tokens(case) -> set:
    """Reference numbers this case can be recognised by, from its name and aliases.

    The shipped fixtures were renamed to <doc_type>_<id>.pdf on 2026-08-31, which
    took the BS…/F…TINV numbers out of `pdf` and left them only in `aliases`. This
    rule is the last resort for a file that has been renamed AND re-exported, so
    its contents no longer hash to ours -- reading the aliases too is what keeps
    it working at all now that no current filename carries a number.
    """
    names = [case["pdf"], *case.get("aliases", [])]
    return set().union(*(_tokens(n) for n in names))


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
            hits = [c for c in index.values() if found & _case_tokens(c)]
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
    between them and that is formatting, not recognition. A printed fill rule
    goes the same way, for the same reason: see FILL_RULE.
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
        # After the separator row, so a Markdown rule is markup here rather than
        # a run of dashes this would eat half of.
        text = FILL_RULE.sub(" ", text)
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


# A block has to look like the truth block before it is filed under it. Below
# this, two blocks share a few incidental characters and nothing more, and
# filing them together would move real content to the wrong place -- so an
# unrecognised block goes to the tail instead, where it costs what extra content
# has always cost.
BLOCK_MATCH_MIN = 0.45


def _matched_chars(expected: str, actual: str) -> int:
    """Content characters of `expected` the transcript actually produced.

    **Extra text costs nothing; missing and misread text costs everything.**
    Changed 2026-09-04 at the user's request -- *if the char is more than need
    dont take the point, only take point when the char that is suppose to ocr is
    wrong or missing, since it is important to the extraction part.* The score is
    now RECALL of the ground truth rather than an edit distance: a character the
    page prints and the model did not produce is a loss, and a character the
    model produced that the page does not print is not a gain and not a loss.

    The reasoning is pass 2. A value the read dropped or garbled cannot then be
    extracted -- that is the failure that propagates -- while a line of text the
    page does not print leaves every real value exactly where it was, so the
    extractor is no worse off. An edit distance charges both the same, so a model
    that reads the whole page and adds a sentence scored like one that lost a
    sentence, and the two are not the same problem.

    **`word_accuracy` has always been computed this way**, so this makes the two
    consistent rather than introducing a new idea: the character score is now the
    character-level version of what the word score already measured.

    Matched runs come from `difflib` rather than from a longest common
    subsequence, and that is the load-bearing choice. An LCS would credit
    characters that line up by coincidence, one here and one there, and on Thai
    text scattered coincidences are plentiful -- a page of garbage would score
    well. `SequenceMatcher` credits only CONTIGUOUS runs, recursively, which is
    what makes a match mean the model read that text rather than happened to emit
    those letters in that order. `autojunk=False` for the reason it is off
    everywhere else here: the heuristic drops any element appearing in more than
    1% of a long sequence, which for characters is most of the alphabet.

    **What it stops charging for is also what it stops CATCHING**, and that is
    the cost to know: invented text is now free in this number. It is returned
    separately as `invented_chars` and printed beside the score for exactly that
    reason -- see `score`.
    """
    return sum(block.size for block in match_blocks(expected, actual))


def match_blocks(expected: str, actual: str) -> list:
    """The contiguous runs of `expected` the transcript produced.

    Split out of `_matched_chars` so the per-script rates are taken from the
    SAME blocks the headline is taken from -- two independent alignments of one
    pair of texts would eventually disagree about which characters were read,
    and the three script rates would then not be a breakdown of anything.

    Each block is difflib's `(a, b, size)`: `a` indexes `expected`, which is
    what lets a matched run be classified by the ground truth's own characters
    rather than by the model's.
    """
    if not expected:
        return []
    matcher = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    return [b for b in matcher.get_matching_blocks() if b.size]


def matched_by_script(expected: str, blocks: list) -> dict:
    """Matched characters of each script, counted on the truth's side.

    Counted on `expected` rather than on the transcript on purpose: the question
    is *how much of the Thai on the page came back*, and a model that answers a
    Thai word with Latin letters must lose the Thai, not gain the Latin.
    """
    counts = dict.fromkeys(SCRIPTS, 0)
    for block in blocks:
        for ch in expected[block.a:block.a + block.size]:
            name = script_of(ch)
            if name:
                counts[name] += 1
    return counts


def _similar(a: str, b: str) -> float:
    """How alike two blocks are, 0..1, cheaply and symmetrically."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def align_blocks(expected: str, actual: str) -> str:
    """`actual` with its blocks put back into the truth's order.

    **Why pass 1 needs this.** Character accuracy matches CONTIGUOUS runs of the
    ground truth, and a paragraph read perfectly but emitted at the top instead
    of the bottom breaks the run it belongs to: it can only be credited against
    the truth text it sits beside, which is not the text it is. Two models can
    produce the same content and score far apart because one of them walked the
    page in a different order -- which is a layout decision, not a recognition
    failure, and pass 1 is scored on recognition. (Until 2026-09-04 the score was
    an edit distance and the same reordering was charged TWICE, once to delete
    the paragraph from where it is and once to insert it where it is not; the
    alignment mattered even more then, and still matters now.)

    This is the same reasoning `fieldscore._pair_rows` already applies to line
    items -- match by content, never by position, because one dropped row
    otherwise mis-scores every row after it -- lifted to whole blocks of a page.

    **It is a pre-alignment step, not a new metric.** The blocks are reordered
    and then handed to exactly the comparison that runs without it, so everything
    the score charges for it still charges for: a missing line is still missing
    and a misread character is still wrong. The only thing that stops costing
    anything here is *where on the page the model chose to put it*. (An invented
    line stopped costing anything on 2026-09-04, for a different reason and in a
    different place -- `_matched_chars`.)

    Inserted and removed line breaks cost nothing either, and did not before:
    every block is squashed to its content, and `score` strips whitespace from
    both sides regardless. What changes here is that a line the model split in
    two now files BOTH halves under the truth line they came from, instead of
    leaving the second half stranded at the end of the document.

    Blocks the truth has no home for keep their own relative order and go last,
    which is where unmatched content has always effectively been charged.
    """
    exp_blocks = [b for b in expected.split("\n") if b.strip()]
    act_blocks = [b for b in actual.split("\n") if b.strip()]
    if not exp_blocks or not act_blocks:
        return actual

    exp_keys = [content_only(b) for b in exp_blocks]
    # Each actual block goes to the truth block it looks most like. Many-to-one
    # on purpose: a truth line the model split across two lines gets both halves
    # back, in the order it emitted them.
    filed = {index: [] for index in range(len(exp_blocks))}
    tail = []
    for block in act_blocks:
        key = content_only(block)
        best, score_ = -1, 0.0
        for index, exp_key in enumerate(exp_keys):
            ratio = _similar(key, exp_key)
            if ratio > score_:
                best, score_ = index, ratio
        if best >= 0 and score_ >= BLOCK_MATCH_MIN:
            filed[best].append(block)
        else:
            tail.append(block)

    ordered = [block for index in range(len(exp_blocks)) for block in filed[index]]
    return "\n".join(ordered + tail)


def score(expected: str, actual: str, align: bool = True) -> dict:
    """Compare a transcript with its ground truth.

    `align` tries the actual's blocks in the truth's order as well as in their
    own, and keeps whichever scores better -- see `align_blocks`. On by default
    because reading order is a layout decision and pass 1 is scored on
    recognition; `align=False` gives the strictly positional score this returned
    before 2026-08-21, which is what `word_accuracy` still measures either way.

    **`char_accuracy` is RECALL of the ground truth since 2026-09-04**, not an
    edit distance: text the transcript has and the page does not costs nothing.
    See `_matched_chars` for the reasoning and for what it stops catching --
    `invented_chars` comes back beside the score so that half is still visible.
    Every pass-1 figure recorded before that date is therefore a LOWER bound on
    what the same transcript scores now, and the two must not be averaged
    together.
    """
    # Kept for word accuracy below, which stays order-sensitive on purpose.
    positional = actual
    # Character accuracy is content only: every invisible character comes out of
    # both sides first, so line breaks, indentation and cell padding cannot move
    # the number in either direction. See INVISIBLE.
    exp_c = content_only(expected)
    act_c = content_only(actual)
    # `matched_c`, not `matched`: the word-accuracy block below computes its own
    # match count and used to take that name, so the two silently shared it and
    # `matched_chars` came back as a word count.
    blocks = match_blocks(exp_c, act_c)
    matched_c = sum(b.size for b in blocks)

    # **The alignment can only ever help, never hurt.** It is a search for a
    # better correspondence between the two texts, and like any heuristic search
    # it sometimes finds a worse one: on a long table of near-identical rows the
    # per-block best match is noisy, so rows that were already in the right order
    # get filed under the wrong truth row and scrambled. Measured on the saved
    # outputs, that cost sol004 28 points and sol001 19 before this guard.
    #
    # Taking the better of the two is what makes the change monotone: no
    # transcript can score lower than it did before 2026-08-21, and one whose
    # content is right but reordered now scores what its content deserves.
    best_act = act_c
    if align:
        aligned = content_only(align_blocks(expected, actual))
        aligned_blocks = match_blocks(exp_c, aligned)
        aligned_matched = sum(b.size for b in aligned_blocks)
        if aligned_matched > matched_c:
            matched_c, best_act, blocks = aligned_matched, aligned, aligned_blocks
    cer = 1 - matched_c / max(len(exp_c), 1)

    # Word accuracy is deliberately left ORDER-SENSITIVE and is taken on the
    # unaligned text, so the two numbers still say different things: character
    # accuracy now answers "is the content there", word accuracy "is it in the
    # order the page prints it". The page shows only the first; the run log keeps
    # both, and the gap between them is what a reordering looks like.
    exp_w, act_w = expected.split(), positional.split()
    matcher = difflib.SequenceMatcher(None, exp_w, act_w, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    wer = 1 - matched / max(len(exp_w), 1)

    # The same rule with Thai vowels and tone marks off both sides, so the gap
    # between the two says whether the losses are marks or whole characters.
    exp_nm = THAI_MARKS.sub("", exp_c)
    act_nm = THAI_MARKS.sub("", best_act)
    cer_nm = 1 - _matched_chars(exp_nm, act_nm) / max(len(exp_nm), 1)

    # The same recall, taken over three subsets of the ground truth's own
    # characters and off the blocks the headline was taken from. **They do not
    # add up to `char_accuracy`**: punctuation, symbols and currency signs
    # belong to no script and are in the headline only.
    #
    # A script with too little of it on the page reports no rate and its count
    # anyway -- see SCRIPT_MIN_CHARS. Blank there means *not enough to measure*,
    # never *scored zero*.
    exp_by_script = script_counts(exp_c)
    hit_by_script = matched_by_script(exp_c, blocks)
    scripts = {}
    for name in SCRIPTS:
        total = exp_by_script[name]
        scripts[f"{name}_chars"] = total
        scripts[f"{name}_matched"] = hit_by_script[name]
        scripts[f"{name}_accuracy"] = (round(hit_by_script[name] / total, 4)
                                       if total >= SCRIPT_MIN_CHARS else None)

    return {
        "char_accuracy": round(max(0.0, 1 - cer), 4),
        # Thai, Latin ("English") and numerals apart. Digits are the column to
        # read first: every Mandatory field in the requirement is a number, a
        # date or an ID, and a page can score 95% overall while losing the one
        # digit that makes an amount wrong.
        **scripts,
        "word_accuracy": round(max(0.0, 1 - wer), 4),
        "char_accuracy_no_marks": round(max(0.0, 1 - cer_nm), 4),
        "expected_chars": len(exp_c),
        "actual_chars": len(act_c),
        # What the score is made of, and the half of it the score no longer
        # charges for. `expected_chars - matched_chars` is what was missed --
        # dropped or misread -- and `invented_chars` is transcript that answers
        # to nothing on the page. The second costs no points and is REPORTED,
        # because a model that invents is the failure this harness was built to
        # catch and a number nobody prints is a number nobody sees.
        "matched_chars": matched_c,
        "invented_chars": max(0, len(best_act) - matched_c),
        "expected_words": len(exp_w),
        "actual_words": len(act_w),
    }


# How many lines a rewrap may span on either side of a fold. A line the model
# split, or a run of lines it ran together, is one or two either way in every
# transcript here; the bound is what keeps the search below cheap and stops it
# pairing two distant regions that happen to concatenate alike.
FOLD_MAX_LINES = 4

# Above this many line pairs a group is not searched at all, only tested whole.
# A group this size is a rewritten page rather than a rewrapped line, and the
# search is quadratic in the group.
FOLD_MAX_CELLS = 4000


def _fold_group(exp, act):
    """The folding of one diff group that hides the most spacing.

    Both sides are cut into consecutive blocks; a block pair whose CONTENT is
    equal is folded to the truth's wording, and everything else is left as it
    was. Consecutive on both sides and left to right, so a block can only ever
    fold against the text opposite it -- moved text never folds.

    This is a search rather than a single test because the two shapes turn up
    together: `TAX ID … / ใบกำกับสินค้า …` run onto one line, in the same group as
    a word misread one line above. Testing the group whole keeps both, and a
    reader then cannot tell which of the two is the real finding.
    """
    n, m = len(exp), len(act)
    exp_keys = [content_only(x) for x in exp]
    act_keys = [content_only(x) for x in act]

    # best[i][j]: the most content characters foldable from exp[i:] against
    # act[j:]. Walked backwards so each cell reads only cells already settled.
    best = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            top, mv = -1, None
            # Folds are tried first and skips have to beat them strictly, so a
            # tie folds: hiding a spacing difference is the whole point.
            for k in range(i + 1, min(n, i + FOLD_MAX_LINES) + 1):
                key = "".join(exp_keys[i:k])
                for l in range(j + 1, min(m, j + FOLD_MAX_LINES) + 1):
                    if key == "".join(act_keys[j:l]) and len(key) + best[k][l] > top:
                        top, mv = len(key) + best[k][l], ("fold", k, l)
            if i < n and best[i + 1][j] > top:
                top, mv = best[i + 1][j], ("exp", i + 1, j)
            if j < m and best[i][j + 1] > top:
                top, mv = best[i][j + 1], ("act", i, j + 1)
            best[i][j], move[i][j] = top, mv

    out, i, j = [], 0, 0
    while move[i][j]:
        what, k, l = move[i][j]
        if what == "fold":
            out.extend(exp[i:k])          # folded: reads as the truth, so equal
        elif what == "act":
            out.append(act[j])            # kept: reported as invented text
        # "exp" drops an expected line from the output, which reports it missing.
        i, j = k, l
    return out


def fold_spacing(expected_lines, actual_lines):
    """`actual` rewritten to the truth's wording wherever ONLY the spacing moved.

    The diff is the one place this project still charged for layout. Character
    accuracy strips every invisible character from both sides before it measures
    anything (see INVISIBLE), so a truth line the model split over two lines, two
    lines it ran together, or a space it put inside a number cost the score
    nothing -- and then showed up in the differences list as a `-`/`+` pair
    anyway, which reads as a misread. A reader comparing the two cannot tell that
    from a real one without squashing both by eye.

    So text whose CONTENT the truth already holds is folded away: those actual
    lines are replaced by the expected ones, and `unified_diff` then sees them as
    equal and drops them along with their context. Nothing else is touched --
    text differing by so much as one character is left exactly as it was, so
    missing, invented and misread content still shows.

    **The lines are paired on their content, not on their text**, so a line the
    model spaced out inside a token -- `* A 0 3 $ C N *` for `*A03$CN*` -- pairs
    with its truth line rather than being lumped into a `replace` with whatever
    happens to sit beside it. Where that is not enough, `_fold_group` searches
    within the group for the blocks that do line up.

    **A reordering never folds**, because a folded pair has to be consecutive on
    both sides and in the same order: moved text still shows, which is what
    `word_accuracy` measures and what `align_blocks` forgives in the score.
    """
    matcher = difflib.SequenceMatcher(
        None,
        [content_only(ln) for ln in expected_lines],
        [content_only(ln) for ln in actual_lines],
        autojunk=False,
    )
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        exp, act = expected_lines[i1:i2], actual_lines[j1:j2]
        # `equal` is equal CONTENT, so the two may still be spelled differently;
        # taking the truth's wording is what folds that away.
        if tag == "equal" or content_only("".join(exp)) == content_only("".join(act)):
            out.extend(exp)
        elif len(exp) * len(act) <= FOLD_MAX_CELLS:
            out.extend(_fold_group(exp, act))
        else:
            out.extend(act)
    return out


def diff_lines(expected: str, actual: str, context: int = 1):
    """Unified diff of the two transcripts, with spacing-only groups folded out.

    What is left is the set of differences that actually move `char_accuracy`.
    An empty list therefore means "the same content", not "byte for byte the
    same page" -- which is the claim the score makes too.
    """
    return list(
        difflib.unified_diff(
            expected.splitlines(),
            fold_spacing(expected.splitlines(), actual.splitlines()),
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
