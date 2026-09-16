"""Which pass is at fault for a field the extraction did not get right.

The field score says a value is wrong. It does not say WHOSE fault that is, and
on a full-pipeline run there are two candidates: the read that produced the
transcript, and the extraction that mapped it onto the form. Every measurement in
CLAUDE.md that separates the two does it by REMOVING one of them -- every pass-2
baseline was taken `--from-truth` precisely because a wrong value is otherwise
never attributable. That works for a benchmark and not for a real run, which has
to be judged as it happened.

**The test is one question, asked in Python, of the transcript the extraction
actually read:** is the value the ground truth wants findable in it?

    it is not     -> pass 1 lost it. Whatever pass 2 did, it had nothing to
                     copy: `blame.OCR`.
    it is there   -> pass 1 produced it and pass 2 did not put it in the key:
                     `blame.EXTRACTION`.

`grounding.Source` is the matcher, deliberately -- the same one that decides
whether an EXTRACTED value is on the page -- so a value this calls lost is
exactly a value `grounding` could not have grounded, and the audit and the
attribution cannot disagree about whether two spellings are one value. It
inherits that matcher's known limits with it, and both err the same way -- they
find a value that is arguably not there, so they attribute to the extraction
what may have been the read's. A one- or two-character value is found in any page
of text; a figure is compared BY VALUE against every number the transcript
prints, so an amount the read produced for a different cell answers for this one.
Every Mandatory field in the requirement is a name, an identifier, a date or an
amount, which is the case it is good at, and the direction of the error is the
one to know: this understates the read's share rather than overstating it.

**It changes no score.** `fieldscore` computes the rate, this attributes what the
rate already charged for, and the two never touch: nothing here can move a
verdict, a denominator or a run's accuracy. What it produces is a record -- one
attribution per value, counts on the run-log row, and a sentence on the page.

**The counted population is the headline's own**, so the counts reconcile with
the row they sit on rather than needing an argument:

    blame_ocr + blame_extract + blame_unknown  ==  p1_scored - p1_correct

`spurious` and the Optional fields are attributed and reported per value, and
counted nowhere -- the same standing this project gives them everywhere else: a
value invented where the page prints nothing has no printed value to be judged
against, and nobody is held to an Optional field. `counted` on each verdict says
which side of that line it fell, decided here rather than re-derived by each
reader.

**A truth-fed run is not attributed at all.** Its transcript IS the ground truth,
so every expected value is findable in it by construction -- the truth
self-check asserts exactly that -- and every verdict would read `extraction`
whether or not the extraction was at fault for anything. A tautology printed as a
finding is worse than no finding.

**The page is consulted where it can be, and only to refuse a claim.** Given the
case's own `solution/<id>.md`, two attributions are held back rather than made:

  * an expected value that is not in the ground-truth transcript either is
    `UNKNOWN`, not `OCR`. The two ground truths disagree about that key, and
    blaming a read for failing to produce text the page does not print would put
    a permanent, confident accusation against every model that ever reads it.
  * a value invented under a key the page leaves blank, whose text IS in the
    transcript and is NOT on the page, is `OCR`: pass 1 wrote it and pass 2
    copied it faithfully. That is the 1 MP failure this project already
    documents -- at that budget the model stops misreading and starts inventing
    -- and it is the one case where an invented field is not the extraction's.

Without the page, both fall back to the attribution that claims less.
"""

import grounding

# Pass 1 produced no such text: the extraction had nothing to copy.
OCR = "ocr"
# The text was there and did not reach the key.
EXTRACTION = "extraction"
# Neither can be blamed from here -- see the UNKNOWN note in the docstring.
UNKNOWN = "unknown"

ORDER = (OCR, EXTRACTION, UNKNOWN)

# What each attribution claims, in one line. Read by the page and the CLI, so the
# two cannot describe one verdict differently.
MEANING = {
    OCR: "the read did not produce this text, so the extraction had nothing to"
         " copy",
    EXTRACTION: "the text is in the transcript and did not reach this field",
    UNKNOWN: "neither pass can be blamed from here -- the two ground truths"
             " disagree about this key",
}

# The outcomes worth asking the question of. `correct` and `absent` are not
# failures; the other four are, and only the first three are in the rate.
_COUNTED = ("partial", "wrong", "missed")
_ASKED = _COUNTED + ("spurious",)


def _readings(row) -> list:
    """Every reading of the truth that would have counted as correct.

    `accepted` where the truth file gives more than one -- a name printed in Thai
    and again in English -- because a read that produced either of them handed
    the extraction something it could have answered with. `expected` alone
    otherwise, which is what `fieldscore` puts on every row.
    """
    readings = row.get("accepted") or [row.get("expected")]
    return [r for r in readings if str(r or "").strip()]


def _lookups(readings) -> list:
    """The readings there is anything to look for.

    A printed dash and an empty string are values the page does not state as
    text, so a transcript can neither hold nor lose them. They are not a loss
    pass 1 can be blamed for; see `_attribute`.
    """
    return [r for r in readings
            if not grounding.is_nil(r) and grounding.squash(r)]


def _attribute(row, source, page):
    """(attribution, why) for one judged value that did not come back right."""
    if row.get("status") == "spurious":
        # The page states nothing here and the extraction filled it anyway.
        value = row.get("actual")
        if not source.holds(value):
            return EXTRACTION, ("nothing like it is in the transcript either"
                                " -- the extraction wrote it")
        if page is not None and not page.holds(value):
            return OCR, ("this text is in the transcript and not on the page"
                         " -- the read invented it and the extraction copied it")
        return EXTRACTION, ("the text is on the page, but not as this field's"
                            " value")

    lookups = _lookups(_readings(row))
    if not lookups:
        # The truth is a printed dash, which the extraction prompt allows to be
        # answered either way. A figure in its place is the extraction's doing,
        # and there is nothing here a read could have lost.
        return EXTRACTION, ("the page prints a dash here, so a value in its"
                            " place is the extraction's")

    if any(source.holds(reading) for reading in lookups):
        return EXTRACTION, ("the transcript holds this value and it did not"
                            " reach the field")
    if page is not None and not any(page.holds(reading) for reading in lookups):
        return UNKNOWN, ("the ground-truth transcript does not hold this value"
                         " either, so the two ground truths disagree about this"
                         " key")
    return OCR, "the transcript does not hold this value -- the read lost it"


def _judged(score) -> list:
    """Every value the score judged: the scalars and both tables' cells."""
    rows = list(((score or {}).get("scalars") or {}).get("rows") or [])
    for table in ("line_items", "income_items"):
        block = (score or {}).get(table)
        if isinstance(block, dict):
            rows.extend(block.get("rows") or [])
    return rows


def _mostly(counts):
    """The one-word headline, or None where nothing was attributed."""
    ranked = sorted(ORDER, key=lambda name: -counts.get(name, 0))
    top = counts.get(ranked[0], 0)
    if not top:
        return None
    if sum(1 for name in ORDER if counts.get(name, 0) == top) > 1:
        return "mixed"
    return ranked[0]


def check(score, transcript, page_text=None, truth_fed: bool = False) -> dict:
    """Attribute every value this score charged for to the pass that lost it.

    `score` is one document's `fieldscore.score` result, `transcript` the text it
    was extracted from, and `page_text` the case's own `solution/<id>.md` where
    the caller has it -- see the module docstring for what the page is consulted
    for, which is only ever to refuse an attribution.

    Returns `{"skipped": why}` where the question cannot honestly be asked, so a
    caller never has to decide that for itself.
    """
    if not isinstance(score, dict) or score.get("error"):
        return {"skipped": "there is no field score to attribute"}
    if truth_fed:
        return {"skipped": "this run was fed the ground-truth transcript, so"
                           " every value the extraction missed was in front of"
                           " it by construction"}
    if not (transcript or "").strip():
        return {"skipped": "there is no transcript to look the values up in"}

    source = grounding.Source(transcript)
    page = grounding.Source(page_text) if (page_text or "").strip() else None

    fields, counts = {}, {name: 0 for name in ORDER}
    uncounted = 0
    for row in _judged(score):
        if row.get("status") not in _ASKED:
            continue
        path = row.get("path")
        if not path:
            continue
        who, why = _attribute(row, source, page)
        # In the headline's denominator, or beside it. Decided here rather than
        # by each reader: `fieldscore.score` puts `required` on the row for the
        # same reason, and its comment is the one to read -- the page marks it,
        # the CLI prints it and the counts are taken over it, and three answers
        # to one question drift.
        counted = (row.get("status") in _COUNTED
                   and row.get("required") is not False)
        fields[path] = {"blame": who, "why": why, "status": row["status"],
                        "counted": counted}
        if counted:
            counts[who] += 1
        else:
            uncounted += 1

    return {
        "fields": fields,
        "counts": counts,
        # What the counts are over: the values in the rate that did not come back
        # right. Equal to `expected - correct` of the same score, which is what
        # makes the run-log columns readable beside `p1_scored` and `p1_correct`.
        "counted": sum(counts.values()),
        # Attributed and outside the rate -- an Optional field, or a value
        # invented where the page states none.
        "uncounted": uncounted,
        # Who carries most of it, in one word, or None where nothing went wrong.
        # `mixed` rather than a coin toss on a tie: the honest answer to a 2-2
        # split is that it is both.
        "mostly": _mostly(counts),
        # Whether the ground-truth transcript was available to refuse a claim
        # with. Reported because the two attributions that need it are weaker
        # without it, and a reader comparing two runs should see which had it.
        "page_checked": page is not None,
    }


def merge(blames) -> dict:
    """Every document's attribution in one file, as one for the file.

    `blames` is (document number, attribution) in file order.

    Paths are prefixed with the document they belong to, exactly as
    `app._merge_grounding` prefixes grounding's, and for the same reason: a bare
    `buyer_name` names a different field in each document of a pack, so a verdict
    nobody can trace back to a document is a verdict nobody can act on.

    Counts are summed rather than averaged -- they are counts -- so the invariant
    the run-log columns rest on survives a pack, whose `p1_scored`/`p1_correct`
    are pooled the same way.
    """
    kept = [(n, b) for n, b in blames if isinstance(b, dict) and b.get("fields")]
    if not kept:
        # Nothing to attribute. On a pack every document is skipped for the same
        # reason, so the first one given is the file's.
        skipped = next((b.get("skipped") for _, b in blames
                        if isinstance(b, dict) and b.get("skipped")), None)
        return {"skipped": skipped} if skipped else {}
    fields, counts = {}, {name: 0 for name in ORDER}
    counted = uncounted = 0
    page_checked = True
    for number, one in kept:
        for path, entry in one["fields"].items():
            fields["doc%d.%s" % (number, path)] = entry
        for name in ORDER:
            counts[name] += (one.get("counts") or {}).get(name, 0)
        counted += one.get("counted") or 0
        uncounted += one.get("uncounted") or 0
        page_checked = page_checked and bool(one.get("page_checked"))
    return {"fields": fields, "counts": counts, "counted": counted,
            "uncounted": uncounted, "mostly": _mostly(counts),
            "page_checked": page_checked}
