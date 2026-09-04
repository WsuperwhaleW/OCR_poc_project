"""Every prompt the app sends to the model, and nothing else.

Split out of `app.py` so the text can be read, edited and diffed without the
request-assembly code around it. There is no logic here: these are string
constants, imported by `app.py` and interpolated into request bodies exactly as
written.

A prompt that looks redundant or oddly worded is usually the fix for a failure
that shows up as HTTP 200 with quietly wrong output. What was measured, and why
each one is worded as it is, is in CLAUDE.md -- not repeated here. Re-run the
benchmark (`python compare.py`) after editing any of them.

* `PROMPT`            -- pass 1, page image in, verbatim transcript out.
* `DOTS_PROMPT`       -- pass 1 for dots.ocr, which answers with layout JSON.
* `OCR_PROFILES`      -- the two pass-1 shapes, each pairing a prompt with
                         whether a system message may go with it and how the
                         reply is read back.
* `EXTRACT_PROMPT`    -- pass 2, transcript in as text, structured JSON out.
* `EXTRACT_REMINDER`  -- closes pass 2's message, after the transcript.
* `EXTRACT_STEP_*`    -- pass 2 in agentic mode: the same schema asked for one to
                         three fields at a time, one request per step.
* `EXTRACT_STEPS`     -- the step table those are interpolated with. Still only
                         constants: the loop that walks it lives in `app.py`.
"""

# --------------------------------------------------------------------------
# pass 1: OCR
# --------------------------------------------------------------------------

# Page image in, verbatim transcript out. Extracts no fields -- it produces the
# page as Markdown, with tables as <table> HTML, and that transcript is what
# pass 2 reads.
#
# The text is typhoon-ocr1.5's OWN training prompt, recovered from the model, with
# our ORDER block appended. The <figure> and <page_number> rules are typhoon's and
# are left alone; `app.normalise_output` strips what they produce.
#
# Do not move or reword the ORDER block, and do not add rules back, without
# re-measuring -- CLAUDE.md has the table.
PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image’s main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>


- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.

ORDER -- this matters
- Transcribe in the order the text physically appears on the page: strictly top to
  bottom, and left to right within the same line or band.
- Text positioned higher on the page MUST appear earlier in your output. Never move a
  heading, title, total or page number away from where it sits on the page.
- For multi-column layouts, finish the left column before starting the right one."""

# dots.ocr's own prompt, quoted exactly. A different model, a different request
# shape, and a different answer: this one returns a JSON array of layout blocks
# (`bbox`, `category`, `text`) rather than a page of Markdown, which `app.py`
# flattens back into a transcript before anything else sees it.
#
# THE WORDING IS LOAD-BEARING AND SO IS THE PLACEMENT. Measured 2026-08-18 on
# `hf.co/mradermacher/dots.ocr-GGUF:latest` through Ollama: a paraphrase that kept
# the opening sentence and dropped the numbered rules returned 2 tokens and an
# empty string, and so did this exact text when it was put in the system slot.
# Sent verbatim, in the user turn, with no system message, it reads the page.
# Do not tidy the indentation, renumber the rules, or "improve" the categories
# list without re-running the benchmark -- CLAUDE.md has the table.
DOTS_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.
4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.
"""

# Pass 1 comes in shapes now, not just wordings, and a prompt cannot be swapped
# on its own. Each profile carries the three things that have to move together:
#
#   prompt   what is sent with the page image
#   system   whether the backend's system message may be sent alongside it
#   reply    how `app.py` turns the answer back into a transcript
#
# `system` is here because it is not a preference. On dots.ocr the system slot
# alone -- with a prompt that otherwise works -- takes the reply to two tokens,
# while on typhoon dropping it costs 2.42 points of mean character accuracy. One
# global switch cannot be right for both, so it belongs to the profile.
#
# `reply` is a tag, not a function: this module holds no logic, and `app.py`
# switches on it. Adding a profile here without teaching `app.py` its tag is
# caught at startup rather than at the first read.
OCR_PROFILES = {
    "typhoon": {
        "label": "Typhoon OCR (Markdown transcript)",
        "prompt": PROMPT,
        "system": True,
        "reply": "markdown",
        "note": "The default, and every accuracy baseline in CLAUDE.md was taken "
                "on it. Built for typhoon-ocr1.5; returns the page as Markdown.",
    },
    "dots": {
        "label": "dots.ocr (layout JSON)",
        "prompt": DOTS_PROMPT,
        "system": False,
        "reply": "layout_json",
        "note": "dots.ocr's own prompt, verbatim, with no system message -- both "
                "are required or the model returns nothing at HTTP 200. Returns "
                "layout blocks, which are flattened into a transcript in reading "
                "order.",
    },
}

DEFAULT_OCR_PROFILE = "typhoon"

# Which profile a model needs, matched on its name. An ordered list of
# (substring, profile id) tested in order, first hit wins; anything unmatched
# gets DEFAULT_OCR_PROFILE.
#
# **A table rather than a preference, because the pairing is a fact about the
# model.** dots builds need their own prompt AND no system message, and given
# the other profile they return nothing at HTTP 200 -- an empty transcript that
# logs as a clean run. `app.profile_for_model` reads this on every model switch
# so the two can no longer drift apart; see CLAUDE.md, which recorded the
# opposite rule until 2026-08-21 and the 0.0% run that overturned it.
#
# A constant, like everything else here: the lookup lives in `backends.py`
# beside `is_ocr_model`, which is the same kind of name heuristic.
OCR_PROFILE_BY_NAME = (
    ("dots", "dots"),
)

# --------------------------------------------------------------------------
# pass 2: field extraction
# --------------------------------------------------------------------------

# Transcript in as text, one JSON object out, in a single request. No image, so
# it costs a fraction of the OCR run.
#
# SCOPE: the field requirement's PRIORITY 1 set and nothing else -- the fourteen
# keys it calls Prototype / MVP. They are the fields a document has to give up
# before it can be classified and matched at all:
#
#   what it is    document_type, document_number, issue_date
#   what it cites reference_document
#   the issuer    seller_name, seller_tax_id, seller_branch
#   the customer  buyer_name, buyer_tax_id, buyer_branch
#   the money     currency, subtotal, vat_total, amount_incl_vat
#
# Priority 2 and 3 -- addresses, due dates, PO/GR/RTV numbers, withholding,
# service periods, payment and bank details, the line-item table -- are OUT of
# this pass on purpose. Asking a 2B model for more keys measurably damages the
# keys it was already getting right; CLAUDE.md has the numbers, and they are the
# reason this pass is narrow rather than complete.
#
# `other_fields` stays, and is the only thing here that is not priority 1. It is
# the overflow valve: everything the page states that these fourteen keys do not
# cover goes there under the document's OWN wording, so narrowing the schema
# loses nothing for later. It is never scored in the headline.
#
# It is a LOOKUP: it copies printed values onto keys and reasons about none of
# them. The requirement also asks for a NORMALISED document type, branch codes
# and tax IDs -- none of which is asked for here, because each is a
# classification rather than a reading. `normalise.py` derives them afterwards,
# in Python, from the values copied here.
# --------------------------------------------------------------------------
# pass 0: the type is settled before either prompt is built, and both are told
# --------------------------------------------------------------------------

# The workflow is classify -> choose the form -> ask for it, and until 2026-09-01
# only the middle step used the answer. `DOC_TYPE_FIELDS` picked the keys and
# neither prompt ever said what kind of document the transcript was, so the model
# worked that out for itself, from the same page, on every request -- and where it
# got it wrong the result is a value in the WRONG KEY, which is the one failure
# `grounding.py` cannot see: the value is printed on the page, so it grounds.
#
# The measured failures this is aimed at are all of that shape (CLAUDE.md, the
# 2026-08-31 sweep over 100 runs): `po_gr_rtv_number` 30 spurious against 0
# correct, mostly a credit note's cited invoice; `inv_rtv_cnr_number` 16 spurious,
# mostly the receipt's own number; `payment_date` 19 spurious, mostly the
# document's own date. Every one of them is the right answer to a question the
# document was not asked.
#
# Three rules hold this to what the rest of pass 2 is:
#
# * **It STATES the type, it never asks for one.** The answer is decided in
#   Python before the request is built (`app.classify`), from the heading the page
#   prints. Nothing here lets the model choose the schema it is then asked to
#   fill, which is the objection `normalise.py`'s header raises against asking a
#   model to classify while it reads.
# * **A bullet says where a value is NOT.** These are mapping rules like every
#   other block in this file. A rule explaining what a credit note is FOR would be
#   the tax reasoning both prompts are deliberately free of.
# * **A bullet is emitted only where its key is in the form**, the same rule the
#   `_EP_*` blocks follow: explaining a key the skeleton does not list is how a
#   model is invited to answer with one.
#
# The type is never emitted as an ANSWER. `document_type` asks for the heading the
# page prints, in the language it prints it, and the category named here is not
# that -- `_TYPE_ANY_RULES` says so wherever that key is asked for, in both modes.

# The phrase each code is named by in a prompt: English, and in the article form
# a sentence can use directly.
TYPE_NAMES = {
    # First, because it is the most distinguishing thing a page can be here: a
    # withholding tax certificate shares no field with any other type, and its
    # heading names nothing else. The Thai form number is kept in the phrase
    # because that is what the document is called -- a reader who has seen one
    # knows it as 50 ทวิ long before "withholding tax certificate".
    "WHT_CERTIFICATE": "a withholding tax certificate (50 ทวิ)",
    "CREDIT_NOTE": "a credit note",
    "DEBIT_NOTE": "a debit note",
    "STATEMENT_OF_ACCOUNT": "a statement of account",
    "INVOICE": "an invoice",
    "RECEIPT": "a receipt",
    "TAX_INVOICE": "a tax invoice",
}

# Most specific first -- the same order as `normalise.DOCUMENT_TYPES`, which
# asserts at import that the two agree rather than trusting that they do.
# `prompts` may not import `normalise` (normalise -> grounding -> prompts), so
# the order is repeated here and checked from the other side.
#
# It decides two things: the order the types are named in a sentence, and which
# of two rules survives where a document is both types and both have something to
# say about one key.
TYPE_SPECIFICITY = tuple(TYPE_NAMES)

# {(code, key): bullet}. One key can collide -- sol001 is a statement of account
# AND an invoice, and both have a rule about `document_number` -- and exactly one
# bullet is emitted for it: the more specific type's. Two bullets about one key
# would be a prompt arguing with itself, and the narrower reading is the one that
# knows more about the page.
TYPE_KEY_RULES = {
    ("INVOICE", "document_number"): """- document_number is the invoice's own number, the one printed with its heading. The number
  of an order, a delivery or an earlier document it cites is not it.""",
    ("INVOICE", "po_gr_rtv_number"): """- An invoice is normally raised against an order or a goods receipt, so where the page
  prints a number under a label meaning purchase order, order number, goods receipt or
  receiving number, that number is po_gr_rtv_number. Where it prints no such label the key
  is "": a customer, account, contract, site or location code is not one of these.""",

    ("CREDIT_NOTE", "document_number"): """- document_number is the credit note's own number. A credit note cites the invoice it
  corrects, and that invoice's number belongs to a different document -- it goes in
  "other_fields" under its own printed label, never here.""",
    ("CREDIT_NOTE", "issue_date"): """- issue_date is the date the credit note itself is dated, not the date of the invoice it
  credits.""",
    ("CREDIT_NOTE", "po_gr_rtv_number"): """- The invoice a credit note credits is NOT po_gr_rtv_number, however prominently the page
  cites it. Only a number labelled as a purchase order, a goods receipt or a return to
  vendor belongs there; on a credit note that prints none of those labels the key is "".""",

    ("DEBIT_NOTE", "document_number"): """- document_number is the debit note's own number, not the number of the document it
  corrects.""",
    ("DEBIT_NOTE", "po_gr_rtv_number"): """- The document a debit note corrects is not po_gr_rtv_number. Only a number labelled as a
  purchase order, a goods receipt or a return to vendor belongs there.""",

    ("STATEMENT_OF_ACCOUNT", "document_number"): """- A statement lists OTHER documents, and the invoice, delivery and reference numbers in its
  table are theirs. document_number is the number printed with this statement's own heading,
  and "" where it prints none of its own.""",

    ("RECEIPT", "inv_rtv_cnr_number"): """- This document IS the receipt, so the number printed with its own heading is the receipt's
  number and can never be inv_rtv_cnr_number. That key takes the number of the document the
  payment settles, found under a label naming it -- and "" where the page settles a table or
  a list of them rather than one.""",
    ("RECEIPT", "payment_date"): """- A receipt's own date is not payment_date. Only a date printed under a label meaning the
  money was paid, or the date a cheque was drawn, belongs there.""",
    ("RECEIPT", "remaining_amount"): """- Most receipts print no outstanding balance at all, and two empty answers are the usual
  result: remaining_amount and remaining_vat_amount take a balance the page prints under its
  own label, never the figure being paid and never a total with this payment taken off it.""",
}

# Emitted whatever the type is, where the key is asked for. Not in
# `TYPE_KEY_RULES` because it is not about any one type -- it is about the type
# being NAMED in the prompt at all, which is new, and which is the one way this
# block could put a wrong value on the page.
_TYPE_ANY_RULES = {
    "document_type": """- document_type still takes the heading exactly as the page prints it, in the language it
  prints it. The category named above is not that heading and is never the answer.""",
}

_TYPE_HEAD = """What this document is:
- It has already been identified, from the heading printed on the page. This is
  {phrase}.
- Take that as given: it is not a question for you to answer, and the words naming it here
  are a category rather than anything printed -- the page's own wording will differ, and is
  usually Thai.
"""

# The same statement in one sentence, for a step's message. An agentic step is a
# few lines long, and a five-line block would be a third of it.
_TYPE_LINE = (" It has already been identified, from the heading printed on the page, as "
              "{phrase} -- take that as given; those words are a category, not a value to "
              "copy.")


def type_phrase(codes):
    """The codes named in a sentence: "a receipt and a tax invoice".

    Most specific first, so the phrase opens with what distinguishes the
    document. A code this table cannot name is dropped rather than spelled out:
    it is one no rule below knows anything about either.
    """
    named = [c for c in TYPE_SPECIFICITY if c in set(codes or ())]
    words = [TYPE_NAMES[c] for c in named]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def type_bullets(codes, keys):
    """The type-specific bullets for one form, in the schema's key order.

    At most one bullet per key -- see `TYPE_KEY_RULES`. Keys are walked in
    `_SCALAR_KEYS` order rather than in the order the caller passed them, so the
    bullets read down the page in the same order as the skeleton above them.
    """
    codes = [c for c in TYPE_SPECIFICITY if c in set(codes or ())]
    if not codes:
        # Nothing is emitted where nothing was identified, `_TYPE_ANY_RULES`
        # included: that rule exists to stop the model answering with the
        # category this block names, and an unnamed category cannot be answered
        # with. It is also what keeps an unclassified run byte-identical to
        # every measurement taken before any of this existed.
        return []
    have, chosen = set(keys or ()), []
    for key in _SCALAR_KEYS:
        if key not in have:
            continue
        if key in _TYPE_ANY_RULES:
            chosen.append(_TYPE_ANY_RULES[key])
        for code in codes:
            rule = TYPE_KEY_RULES.get((code, key))
            if rule:
                chosen.append(rule)
                break
    return chosen


def type_block(codes, keys, bullets=True):
    """The single-request prompt's block about the document's type, or "".

    "" where nothing classified the document: a prompt that says it has been
    identified and then does not name it is worse than one that never raised the
    subject, and the default field set is already the answer to not knowing.

    `bullets=False` names the type and adds none of the rules that go with it.
    That is the third arm of the A/B, and it is a real question rather than a
    switch for its own sake: NAMING the document and RULING on its keys are two
    changes, they cost very different numbers of tokens, and this file's standing
    finding is that a rule added to a prompt is paid for by the keys that were
    already right.
    """
    phrase = type_phrase(codes)
    if not phrase:
        return ""
    listed = type_bullets(codes, keys) if bullets else []
    return (_TYPE_HEAD.format(phrase=phrase)
            + "".join(b + "\n" for b in listed) + "\n")


def type_line(codes):
    """The one-sentence version, appended to an agentic step's task line."""
    phrase = type_phrase(codes)
    return _TYPE_LINE.format(phrase=phrase) if phrase else ""


def step_rules(step, codes, bullets=True):
    """One step's own rules, plus the type rules for the keys IT owns.

    Appended to the step's `rules` rather than to `EXTRACT_STEP_TASK`, which is
    shared by every step: a clause added there changes all of the answers,
    including the ones that were already right. That is measured -- CLAUDE.md,
    2026-08-14, where one clause on the shared task cost three correct values in
    steps it was not aimed at -- and it is why this takes a step, not a form.
    """
    listed = type_bullets(codes, step.get("keys") or ()) if bullets else []
    if not listed:
        return step["rules"]
    return step["rules"] + "\n" + "\n".join(listed)


# Asked of the model ONLY where the printed heading classified nothing -- see
# `app._classify_with_model`, the only caller, which throws the answer away
# unless the heading it quotes is really printed on the page. It is not part of
# the extraction: it names the form, and the form is then asked for by the
# prompts below in the ordinary way.
#
# **IT ASKS FOR NOTHING BUT TEXT THE PAGE PRINTS AND A CODE FROM A CLOSED LIST,
# AND IT MUST NEVER ASK FOR A CONFIDENCE, A PERCENTAGE OR A SCORE.** A model can
# produce a number that reads like one, and it is not a measurement of anything
# -- it is a token sequence about a token sequence, with nothing behind it to
# check. Every percentage in this project is computed in Python from something
# that can be re-derived: character accuracy from an edit distance against the
# ground truth, `grounded_pct` from a search of the transcript, the field score
# from a comparison with a human's answer sheet, and the validation counts from
# arithmetic. A figure the model handed over would sit among those looking like
# one of them.
#
# The same rule in the other direction: where a confidence in the classification
# IS wanted, it is derived here from the evidence -- how much of the matched
# heading line the matched needles account for -- and never asked for.
# `_selftest` asserts this prompt does not ask.
CLASSIFY_PROMPT = """Below is the text of a Thai/English business document. Say what KIND of
document it is, from this list and no other:

{vocabulary}

Return ONLY this JSON object, with no prose and no code fence:

{{ "heading": "", "types": [] }}

- heading is the line the document heads itself with -- the printed words naming what it is
  -- copied EXACTLY as printed, in the language it is printed in. It is often near the top
  of the page and often not. A company, brand or place name is not a heading.
- types holds the code, or the codes, from the list above that the heading names. A page
  headed with two kinds at once is both: return both codes.
- Where nothing on the page names a kind of document in the list, return an empty list and
  "" for the heading. A guess is worse than no answer.

Document text:
"""


def classify_vocabulary():
    """The code list `CLASSIFY_PROMPT` offers, one line each."""
    return "\n".join("  %s -- %s" % (code, TYPE_NAMES[code])
                      for code in TYPE_SPECIFICITY)


# --------------------------------------------------------------------------
# pass 2, single request: the prompt is assembled for one document type
# --------------------------------------------------------------------------

# EXTRACT_PROMPT used to be one string holding fourteen keys. It is now built
# from the blocks below for whichever field set the document's TYPE asks for --
# `extract_prompt(keys)`, and `DOC_TYPE_FIELDS` decides `keys`.
#
# **Every block below is the text that was in that one string, cut up and not
# rewritten**, and `_selftest` at the bottom of this module asserts that
# `extract_prompt(LEGACY_KEYS)` reproduces it byte for byte. That assertion is
# the only thing standing between a per-type schema and quietly re-measuring
# every pass-2 number in CLAUDE.md against a prompt nobody compared. Keep it
# passing, or state plainly that the prompt moved.
#
# A block is emitted only when the field set contains a key it is about, so a
# set without `currency` carries no currency rule -- explaining a key that is not
# in the skeleton is how a model is invited to answer with one.

# The note printed beside each key in the skeleton. A key with "" gets no note.
_KEY_NOTES = {
    'document_type': 'the heading naming what this document is',
    'document_number': '',
    'issue_date': '',
    'reference_document': 'another document this one cites',
    'seller_name': "the issuer's name only",
    'seller_tax_id': '',
    'seller_branch': 'which office or branch, where the page names one',
    'buyer_name': "the customer's name only",
    'buyer_tax_id': '',
    'buyer_branch': '',
    'currency': '',
    'subtotal': 'the total line printed before tax is added',
    'vat_total': 'the tax line',
    'amount_incl_vat': 'the total line printed after tax',
    'po_gr_rtv_number': 'the order, receipt or return number cited',
    'inv_rtv_cnr_number': 'the document this payment settles',
    'remaining_amount': 'what is still outstanding after this',
    'remaining_vat_amount': 'the tax still outstanding after this',
    'payment_date': 'the date the money was paid',
    'cheque_number': 'the cheque it was paid by',
    'original_invoice_date': 'the date of the document being credited',
    'book_no': '',
    'certificate_no': '',
    'sequence_no': 'which line of the filing this is',
    'payer_tax_id': '',
    'payer_name': 'the name of the party that withheld the tax',
    'payer_branch': 'which office or branch of that party',
    'payer_address': '',
    'payee_tax_id': '',
    'payee_name': 'the name of the party the tax was withheld from',
    'payee_branch': 'which office or branch of that party',
    'payee_address': '',
    'dividend_rate_option': 'the ticked rate option, where the page rules one',
    'total_amount_paid': 'the total of the income rows',
    'total_wht_amount': 'the total of the tax withheld',
}

# The skeleton lines up its notes in one column, as it always has.
_NOTE_COLUMN = 30


_EP_HEAD = """You are reading the text of a Thai/English business document that has
already been transcribed. Map what it prints onto the keys below.

This is a lookup, not an analysis. For each key, find where the document states that
thing and copy it across. Do not interpret the document, do not work anything out from
it, and do not check whether it adds up.

There are only {count} keys and they are the important ones. Take your time over each.

Return ONLY a JSON object, no prose and no code fence. Use exactly these keys:

"""

_EP_MID = """
Answer every key listed above, in the order listed, and close the object only after the
last of them. "" is the answer for any the document does not state. Do not invent a key
that is not on the list.

Find each key by its printed label, wherever on the page that label sits. Documents word
their labels differently -- match on what a label MEANS, not on any particular wording.

How to fill them:
- Copy values EXACTLY as printed. Do not reformat, do not translate, do not convert Thai
  text to English, and do not tidy a number's digits, separators or decimals.
- Every value must appear, character for character, in the document text below. If you
  cannot point at it there, the key is not yours to fill.
- Use "" for anything the document does not state. An empty key is a correct answer; a
  plausible invented value is not. Never write a placeholder -- not a row of zeros, not a
  dash, not a repeat of a value you used elsewhere.
- A value belongs to the key whose printed LABEL matches it, never to the key it happens
  to sit nearest. Before writing a value, name to yourself the label it was printed under.
  If it has no such label, it is not that key's value.
- One thing per key, and never the same text in two keys unless a rule below says
  otherwise.
- Do not work anything out. Do not add, subtract, convert, or fill one key from another --
  not a tax ID from a name, not a total from the rows above it. If a figure is not printed,
  its key is "".
- Nothing written in these instructions is itself a key or a value: where a rule names a
  label to look for, that is a hint for finding it on the page, never something to emit.

"""

# The parties block, in three pieces. It was one string until 2026-08-31, and it
# had to be cut up when `seller_name`/`buyer_name` left the schema: a prompt that
# explains a key its own skeleton does not list is an invitation to answer with
# one, which is the rule this whole file is assembled around. Each piece below is
# the text that was in that string, cut and not rewritten, except for the two
# places where it counted the keys out loud.
_EP_PARTIES_HEAD = """The two parties -- {count} keys each, and each holds one thing:
"""

# Emitted only where a name key is asked for.
_EP_PARTIES_NAME = """- The name key takes the party's name and nothing else, on ONE line: if what you are about
  to write contains a street, a postcode, a telephone number, a tax ID, a date, a branch in
  brackets or a line break, you have taken too much of the block.
- A name is TEXT. A value that is mostly digits, or that reads as a code, is not a name
  however close to the block it is printed -- put it in "other_fields" instead.
"""

_EP_PARTIES_IDS = """- The tax ID key takes what is printed as that party's tax or registration number, copied
  as printed -- normally ten or more digits, sometimes with dashes or a letter prefix. A
  two- or three-digit number is a book, page or sequence number and does not belong here.
- The branch key takes what names which office, branch or site of that party -- a branch
  name, a branch number, or a word meaning head office -- copied as printed, not translated
  and not turned into a number. It is the one key on this form usually printed with NO label
  of its own, so look for the value itself: inside that party's block it is normally a word
  meaning head office standing alone on a line, the same run on to the end of that party's
  tax ID line or in brackets after the party's name, or the word for branch followed by a
  number. Copy the whole of what is printed there. A town, a street, a postcode, a telephone
  number or the tax ID itself is not a branch.
"""

# Always emitted where either party is asked for: telling the two blocks apart is
# what the tax-ID keys depend on, and it is the failure this file records most
# often -- the better-labelled tax ID is very often the wrong party's.
_EP_PARTIES_SIDES = """- The issuer is the party the document is FROM -- who wrote it, or who is to be paid. The
  other party is the one it is addressed TO or billed. Tell them apart by the labels printed
  around each block, not by where the block sits: the top of the page is usually the
  issuer's letterhead, but on an order or a form printed by the other party it is not.
  Neither party's {count} keys may take the other's {what}.

"""

_EP_ISSUE_DATE = """issue_date:
- The date this document is dated. Its label usually names the KIND of document rather than
  saying just "date": a date labelled as the invoice's date, the tax invoice's date, the
  receipt's date or the credit note's date is this document's own date and belongs here, not
  in "other_fields". A due date, a payment date, a delivery date, the period a charge covers
  and the date of another document are each something else.

"""

_EP_REFERENCE = """reference_document:
- ONE identifier, and only where a printed label says the thing it names is ANOTHER
  document -- one referred to, cited, replaced, credited or settled. Name that label to
  yourself before answering; where you cannot, this key is not yours to fill.
- Never this document's own number, however it is labelled. An unlabelled number is not a
  reference. Where the page cites several documents, put them in "other_fields" rather than
  choosing one.

"""

# New on 2026-08-31 with the per-type schema. The field requirement makes
# PO/GR/RTV No. Mandatory on an invoice and it is the key the whole INV-to-GR
# matching flow exists to produce. Written to the same rule as every block
# above: where to look and what is not it, never what it means.
_EP_PO_GR_RTV = """po_gr_rtv_number:
- ONE identifier: the purchase order, goods receipt, or return-to-vendor number that this
  document is raised against, whichever of those the page prints.
- Find it by its printed label -- a label meaning purchase order, order number, goods
  receipt, receiving number, or return to vendor. Name that label to yourself before
  answering; where you cannot name one, this key is "".
- It is never this document's own number, and never the customer's account, contract,
  site or location code. Where the page prints several such numbers, keep the one whose
  label matches best here and put the rest in "other_fields".

"""

# Credit notes only. A credit note corrects an earlier invoice, and the date of
# that invoice is a separate fact from the date of the correction.
_EP_ORIGINAL_INVOICE = """original_invoice_date:
- The date of the document this one corrects -- the invoice being credited -- and only
  where the page prints a date under a label saying that is what it is.
- It is not this document's own date. Where the page prints one date only, that date is
  issue_date and this key is "".

"""

# The receipt's own fields. A receipt records a PAYMENT against something else,
# so what it must say is which document is being settled, what is left, when the
# money moved and how -- none of which an invoice or a credit note carries.
_EP_INV_RTV_CNR = """inv_rtv_cnr_number:
- ONE identifier: the invoice, return-to-vendor or credit-note-return document that this
  payment settles. Find it by its printed label -- a label meaning invoice, reference
  document, return, or the document being paid.
- It is NOT this document's own number. A receipt carries its own number as well, usually
  under a label meaning receipt number, and that one is not this key's value.
- Where the page settles several documents -- a table with a column of them, or a list --
  none of them belongs here: do not pick one and do not join them together. Leave this ""
  and let the table come back in "other_fields".

"""

_EP_REMAINING = """remaining_amount and remaining_vat_amount:
- What is still owed AFTER this payment, and the tax part of it: the outstanding balance
  the page prints, under a label meaning remaining, outstanding, balance, or still due.
- Copy only a figure the page actually prints under such a label. Never work one out by
  subtracting this payment from a total -- if the page does not print a balance, both of
  these are "".
- A total, a paid amount, or the figure this receipt is for is not a remaining balance.

"""

_EP_PAYMENT = """payment_date and cheque_number:
- payment_date is the date the money was paid or the cheque was drawn, where the page
  prints one under its own label. It is not the date the document was issued.
- cheque_number is the number of the cheque the payment was made by, where the page prints
  one. A bank account number, a branch, or a transfer reference is not a cheque number.
- Where the page prints only an empty caption for either -- a label with nothing written
  against it -- that key is "".

"""

# --------------------------------------------------------------------------
# the withholding tax certificate's own blocks (2026-09-01)
# --------------------------------------------------------------------------
# None of these is ever emitted beside the blocks above -- the two field sets are
# disjoint -- so they are written for a page nothing else on this form describes.
# They obey the same rule as every block above: say where a value is printed and
# what is not it, never what withholding tax IS or how the figures relate. The
# arithmetic the requirement asks for (Sum Reconciliation, the tax not exceeding
# the amount paid) is checked in `validate.py`, on figures copied verbatim, for
# the reason the head of this file gives at length.

_EP_WHT_IDENTITY = """How this certificate is numbered -- three separate numbers, and most
pages print all three:
- book_no is the book or volume the certificate was torn from, printed under a label
  meaning book. certificate_no is the certificate's own number, under a label meaning
  number. They are usually printed side by side at the top right and are short -- two to
  six digits each.
- sequence_no is which line of the tax filing this certificate is, printed under a label
  naming the filing form and a running number within it.
- Each of these is short. A run of ten or more digits is a tax identification number and
  belongs to a party, not here. Where the page prints no such label, that key is "".

"""

_EP_WHT_PARTIES = """The two parties -- four keys each, and each holds one thing:
- The payer is the party that PAID the income and withheld the tax from it: the block
  labelled as the one with the duty to withhold and remit. The payee is the party the
  income was paid TO and the tax withheld FROM.
- Tell them apart by the labels printed above each block, never by which is higher on the
  page or which is printed larger. Both blocks hold the same four things in the same
  order, so the label above the block is the only thing that says whose they are, and
  neither party's four keys may take the other's.
- The name key takes the party's name and nothing else, on ONE line: if what you are about
  to write contains a street, a postcode, a telephone number, a tax ID or a line break, you
  have taken too much of the block. A value that is mostly digits, or that reads as a code,
  is not a name.
- The tax ID key takes what is printed as that party's tax or registration number, copied
  as printed -- normally ten or more digits, sometimes with dashes. The book, certificate
  and sequence numbers above are two to six digits and are not tax IDs.
- The branch key takes what names which office or branch of that party -- a word meaning
  head office, or the word for branch followed by a number -- copied as printed, not
  translated and not turned into a number. It is normally printed with no label of its own,
  standing on the tax ID line or just under it. A town, a street or a postcode is not a
  branch.
- The address key takes that party's printed address, and only that: the whole of it as
  printed, joined into one value where the page runs it over several lines. It stops at the
  address -- a telephone number, a tax ID or a branch is not part of it.

"""

_EP_WHT_INCOME = """income_items:
- The rows of the table of income this certificate covers, in the order the page prints
  them. ONE object per row the page actually fills in. A row of the printed form with
  nothing written against it is not a row: skip it.
- income_type is the description of the kind of income, as the row prints it -- including
  the wording written in by hand or typed against a free-text option, which is the value
  and not the option's own printed caption.
- payment_date is the date printed on that row. amount_paid is the figure in that row's
  column for the amount paid, and wht_amount is the figure in its column for the tax
  withheld and remitted. Take each from its own column heading, never from its position.
- Copy every value exactly as printed, separators and all. Where a row leaves one of these
  four blank, that key is "" for that row -- do not carry a value down from the row above
  and do not work one out from the others.
- The totals line under the table is NOT a row. It has its own two keys below.

"""

_EP_WHT_TOTALS = """total_amount_paid and total_wht_amount:
- The two figures on the totals line under the table: the total of the amounts paid, and
  the total of the tax withheld and remitted.
- Take each from the column it stands under, not from the order they appear in. Copy them
  as printed, and never add the rows up yourself -- if the page prints no totals line, both
  of these are "".
- A figure spelled out in words is not one of these.

"""

_EP_WHT_DIVIDEND = """dividend_rate_option:
- Only where the page rules a list of rate options against a dividend line and ONE of them
  is ticked, crossed or otherwise marked. Copy the wording of the option that is marked.
- Where no option is marked, or the page rules no such list at all, this is "". A list of
  options with none of them ticked is a form offering a choice, not an answer -- and most
  certificates leave the whole list blank.

"""

_EP_MONEY_HEAD_CURRENCY = """The money -- three figures and the currency:
"""
_EP_MONEY_HEAD = """The money -- {count} figures:
"""

# The money block, in four pieces. A form without `amount_incl_vat` -- a receipt
# has none; the requirement asks it only for an invoice and a credit note --
# must carry no rule about it, so the opening line comes in two versions and the
# two bullets that are ABOUT that key are emitted only with it.
_EP_MONEY_OPEN_3 = """- subtotal is the total line printed BEFORE tax is added; vat_total is the tax line;
  amount_incl_vat is the total line printed AFTER tax.
"""

_EP_MONEY_OPEN_2 = """- subtotal is the total line printed BEFORE tax is added; vat_total is the tax line.
"""

_EP_MONEY_COMMON_A = """- Fill only the lines the page prints, each from its own label. Never add or subtract to
  produce another.
- Where one printed line carries several figures side by side, each under a heading of its
  own, they are separate figures: take the one whose heading names this key, not the first
  one on the line. A figure standing under a tax heading is not the total.
"""

_EP_MONEY_INCL = """- amount_incl_vat is only a total printed with tax ADDED. A total under a heading meaning
  net, or the amount payable, or what is left after tax deducted at source, is not it -- and
  the last figure on a totals line is very often exactly that. Where the page prints a net
  total and no with-tax total, amount_incl_vat is "".
- Where the page charges no tax -- no tax line at all, or a tax line printed as a dash or as
  zero -- the single total it prints is both the total before tax and the total including
  tax: put that same figure in subtotal and in amount_incl_vat. That is the only case in
  which two keys hold the same figure.
"""

_EP_MONEY_COMMON_B = """- Where a money column is ruled into two parts under one heading, the whole units and then
  the fraction, those parts are ONE figure: take both, the fraction after the decimal point.
  Two money columns with headings of their own are two figures, not halves of one.
- A figure in brackets or with a leading minus is negative. Keep the sign. A cell showing a
  dash is nil: write it as a dash or as "", not as 0.00 unless the page prints 0.00.
"""

_EP_CURRENCY = """- currency only where the page prints one -- a currency word, code or symbol against a
  figure or in a column heading -- in the form the page prints it: where it prints the word,
  answer with the word and not the three-letter code. A currency you know such a document
  normally uses, or infer from the language it is written in, is not one the page prints.

"""

_EP_OTHER = """other_fields:
- Everything else the page labels and states, whatever kind of document this is -- by way
  of example only: addresses, other dates, order and account numbers, codes, other tax
  lines, discounts, other amounts, payment and bank details, page numbers, notes, tables.
- label is the document's OWN wording, copied as printed; value is what is printed against
  it. This is where a value goes when it fits none of the {count} keys -- never force one
  into a key that nearly fits, and never repeat anything already written above.
- Where nothing is left over, return an empty list.

"""

_EP_TAIL = """Document text:
"""


# How many keys, in words -- the prompt says the number out loud, and a prompt
# that miscounts its own list is the first thing a reader distrusts.
_COUNT_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}


def _skeleton_lines(keys, items=()):
    """The JSON skeleton the prompt asks for, one line per key.

    `items` is the row shape of a table this form asks for, or `()` for the forms
    that ask for none. It goes ABOVE `other_fields`, because it is one of the
    form's own answers and the overflow list is not.
    """
    lines = ["{"]
    for key in keys:
        line = f'  "{key}": "",'
        note = _KEY_NOTES.get(key, "")
        if note:
            line = line.ljust(max(_NOTE_COLUMN, len(line) + 1)) + "// " + note
        lines.append(line)
    if items:
        cells = ", ".join(f'"{k}": ""' for k in items)
        lines.append(f'  "{INCOME_ITEMS_KEY}": [ {{ {cells} }} ],')
    lines.append('  "other_fields": [ { "label": "", "value": "" } ]')
    lines.append("}")
    return "\n".join(lines) + "\n"


def extract_prompt(keys, codes=(), bullets=True, items=()):
    """The single-request pass-2 prompt for one field set.

    `keys` is an ordered sequence of scalar keys -- normally `DOC_TYPE_FIELDS[code]`.
    `other_fields` is always asked for and is not listed in `keys`: it is the
    overflow valve every field set leans on, not a field of any one of them.

    `codes` is what the document has already been classified as, and only
    changes the prompt's TEXT -- `keys` is what decides its shape. The two are
    passed separately on purpose: a caller that wants the form of one type and
    the framing of another is asking for a mismatch, and `app._field_set` derives
    both from one answer so that cannot happen by accident. `()` emits no
    framing at all, which is what makes `extract_prompt(LEGACY_KEYS)` still
    byte-identical to the prompt every pass-2 number in CLAUDE.md was measured
    on -- `_selftest` asserts it.
    """
    keys = list(keys)
    have = set(keys)
    count = _COUNT_WORDS.get(len(keys), str(len(keys)))
    parts = [_EP_HEAD.format(count=count), _skeleton_lines(keys, items),
             _EP_MID]
    # After the rules that apply to every key and before the blocks about
    # particular ones: what the document IS frames the per-key rules below it,
    # and the general "copy what is printed" rules have to be in force before
    # anything names a category that is not printed anywhere.
    parts.append(type_block(codes, keys, bullets))

    party_keys = [k for k in ("name", "tax_id", "branch")
                  if f"seller_{k}" in have or f"buyer_{k}" in have]
    if party_keys:
        # A DIFFERENT count from the form's: how many keys each party holds.
        # Named apart because shadowing the outer one made `other_fields` say
        # "none of the three keys" about a fourteen-key form.
        party_count = _COUNT_WORDS.get(len(party_keys), str(len(party_keys)))
        parts.append(_EP_PARTIES_HEAD.format(count=party_count))
        if "name" in party_keys:
            parts.append(_EP_PARTIES_NAME)
        if {"tax_id", "branch"} & set(party_keys):
            parts.append(_EP_PARTIES_IDS)
        what = ", ".join(w for k, w in (("name", "name"), ("tax_id", "tax ID"),
                                        ("branch", "branch")) if k in party_keys)
        # "name, tax ID, branch" -> "name, tax ID or branch"
        if "," in what:
            head, _, tail = what.rpartition(", ")
            what = f"{head} or {tail}"
        parts.append(_EP_PARTIES_SIDES.format(count=party_count, what=what))
    if "issue_date" in have:
        parts.append(_EP_ISSUE_DATE)
    if "reference_document" in have:
        parts.append(_EP_REFERENCE)
    if "po_gr_rtv_number" in have:
        parts.append(_EP_PO_GR_RTV)
    if "inv_rtv_cnr_number" in have:
        parts.append(_EP_INV_RTV_CNR)
    if "original_invoice_date" in have:
        parts.append(_EP_ORIGINAL_INVOICE)
    if have & {"subtotal", "vat_total", "amount_incl_vat", "currency"}:
        figures = len([k for k in ("subtotal", "vat_total", "amount_incl_vat")
                       if k in have])
        parts.append(_EP_MONEY_HEAD_CURRENCY if "currency" in have
                     else _EP_MONEY_HEAD.format(
                         count=_COUNT_WORDS.get(figures, str(figures))))
        parts.append(_EP_MONEY_OPEN_3 if "amount_incl_vat" in have
                     else _EP_MONEY_OPEN_2)
        parts.append(_EP_MONEY_COMMON_A)
        if "amount_incl_vat" in have:
            parts.append(_EP_MONEY_INCL)
        parts.append(_EP_MONEY_COMMON_B)
        if "currency" in have:
            parts.append(_EP_CURRENCY)
        else:
            # The currency bullet used to carry the blank line that separates the
            # money block from the next one. Without it the two run together.
            parts.append(chr(10))
    if have & {"remaining_amount", "remaining_vat_amount"}:
        parts.append(_EP_REMAINING)
    if have & {"payment_date", "cheque_number"}:
        parts.append(_EP_PAYMENT)
    # The withholding certificate's blocks. Disjoint from every block above, so
    # on a 50 thawi none of those is emitted and on anything else none of these
    # is -- except for a document nothing classified, whose form is the union of
    # the lot. See `DEFAULT_FIELDS`.
    if have & {"book_no", "certificate_no", "sequence_no"}:
        parts.append(_EP_WHT_IDENTITY)
    if have & {"payer_tax_id", "payer_name", "payer_branch", "payer_address",
               "payee_tax_id", "payee_name", "payee_branch", "payee_address"}:
        parts.append(_EP_WHT_PARTIES)
    if items:
        parts.append(_EP_WHT_INCOME)
    if have & {"total_amount_paid", "total_wht_amount"}:
        parts.append(_EP_WHT_TOTALS)
    if "dividend_rate_option" in have:
        parts.append(_EP_WHT_DIVIDEND)
    parts.append(_EP_OTHER.format(count=count))
    parts.append(_EP_TAIL)
    return "".join(parts)


EXTRACT_REMINDER = """

Now return the JSON object described above, using exactly the keys listed.
Do NOT transcribe the document and do NOT return a "natural_text" field."""

# The same schema as the skeleton in EXTRACT_PROMPT, in the form a server can
# constrain decoding with. The prompt ASKS for these keys; this makes them the
# only keys the sampler can produce, which is what finally stopped the OCR
# fine-tune answering with its own transcript envelope. Sent by
# `backends.structured_request`; `settings.EXTRACT_SCHEMA` switches it off.
#
# Two constraints on edits here:
#
# * It must agree with the skeleton above, key for key. The prompt explains what
#   each key means and the schema decides what may be emitted -- they are two
#   halves of one contract, and a key in one and not the other is a silent bug.
# * It must stay inside the JSON Schema subset Ollama's grammar runtime accepts:
#   object, array, string, number, properties, items, required, enum. A nullable
#   union ({"type": ["string", "null"]}) or a length constraint is rejected by
#   the parser, and the request fails with HTTP 400 rather than degrading.
#
# Every value is typed as a string because the prompt asks for figures copied as
# printed, separators and all. Typing an amount as a number would make the model
# reformat it to satisfy the grammar, and `grounding.py` would then flag its own
# request as ungrounded.
# Public, unlike the scalar list below: `fieldscore.py` walks a line-item row
# cell by cell and has to agree with this schema key for key. A second copy of
# these eight names would drift the moment one of them changed.
# EIGHT keys, and `reference_no`/`reference_date` are deliberately NOT among them
# even though the field requirement lists Reference No. as an MVP line-item field.
# They were added on 2026-08-19 and reverted the same day, measured:
#
# Only sol004 rules a column for either. On the four fixtures that do not, the
# two extra keys did not come back empty as the rules below tell them to -- the
# model filled every key it was given, and the smear spread to the keys that were
# already right. sol001 came back with `reference_no` holding the period,
# `quantity` and `unit_price` both holding the row's amount, and
# `withholding_tax` holding a copy of the VAT. A row of ten keys where the table
# rules six is not a longer answer, it is a wrong one.
#
# So the cost was four fixtures' tables to score two columns on the fifth. If
# this is revisited, the lever is not the prompt -- the "where there is no such
# column, that key is empty on every row" rule is already there and already
# ignored. It is a stronger model, or a per-document schema that only asks for
# the columns that document actually rules.
# Whether pass 2 asks for the charges table at all. It does not: the line-item
# table is priority 3 in the field requirement, and this pass is priority 1 only.
# Read by `fieldscore`, which otherwise derives the truth table from the .md and
# then scores every row of it as missed against an extraction that was never
# asked for one.
EXTRACT_LINE_ITEMS = False

# The eight cells of a charges row. Kept although nothing extracts them now, so
# that `fieldscore` can still DERIVE the table from the Markdown in solution/*.md
# and report it, and so that restoring line items is a matter of turning the flag
# above back on rather than rebuilding the mapping.
ITEM_KEYS = ("description", "period", "quantity", "unit_price", "amount",
             "vat", "withholding_tax", "net_amount")

# --------------------------------------------------------------------------
# the field set is a property of the DOCUMENT TYPE
# --------------------------------------------------------------------------

# Until 2026-08-31 there was one set of fourteen keys for every document. The
# field requirement is per type -- an invoice, a credit note and a receipt/tax
# invoice are each ruled by their own list -- so the set is now chosen by the
# normalised type code `normalise.document_type_code` returns, and everything
# downstream (the prompt, the steps, the JSON grammar, what grounding reports as
# missing, what fieldscore scores) follows from that one choice.
#
# **Only INVOICE is authoritative.** It is the requirement's own eleven Mandatory
# fields plus `seller_name` and `buyer_name`. The other two sets are PROVISIONAL:
# their requirements have not been given yet, and they are the invoice set plus
# the keys those documents obviously need -- a credit note cites the invoice it
# corrects, a receipt commonly settles one. Replace them when the real lists
# arrive; nothing else has to change.
#
# `seller_name` and `buyer_name` are in every set although the invoice table does
# not list them, and that is a deliberate departure. They are needed to tell the
# two parties apart -- which is what the tax-ID keys depend on -- they were
# Mandatory in the earlier requirement this project was built against, and
# deleting a key that is already measured to satisfy a table that never says
# "and only these" is the more expensive mistake of the two. They are NOT in
# `MANDATORY_FIELDS`, so nothing reports them as a compliance failure.
#
# `currency` is in NO typed set. The requirement lists it for none of the three,
# and it is the one key CLAUDE.md records as unscoreable here -- the answer sheet
# contradicts itself, at a cost of about one value per document on every model.
# It is still asked for when the type is unknown, and a page that prints a
# currency still returns it under its own label in `other_fields`.

# **THE SCHEMA IS THE REQUIREMENT'S ELEVEN FIELDS AND NOTHING ELSE** (2026-08-31,
# at the user's request: *drop all fields/prompt that is not in requirement*).
#
# The invoice and credit-note requirements name the same eleven, and they are all
# that is asked for now. What left, and why each was there:
#
#   seller_name, buyer_name   kept on 2026-08-31 morning on the argument that they
#                             tell the two parties apart. The requirement lists
#                             neither, so they go: the party rules that survive
#                             (`_EP_PARTIES_SIDES`) do that job without a key to
#                             fill, and the names still come back under their own
#                             printed labels in `other_fields`.
#   reference_document        in no requirement. A credit note's link to what it
#                             corrects is `INV/CN No.` + Original Document
#                             Matching, which is a VALIDATION rule over
#                             `document_number`, not a field of its own.
#   original_invoice_date     invented here for credit notes. Not in theirs.
#   currency                  in neither, and already off every typed form.
#
# Nothing is lost for later: `other_fields` returns everything the page states
# that these eleven do not cover, under the document's OWN printed labels, and
# the truth files still hold the dropped values so they can be scored again the
# day a requirement asks for them.
_SCALAR_KEYS = (
    "document_type", "document_number", "issue_date",
    "po_gr_rtv_number", "inv_rtv_cnr_number",
    "seller_tax_id", "seller_branch",
    "buyer_tax_id", "buyer_branch",
    "subtotal", "vat_total", "amount_incl_vat",
    "remaining_amount", "remaining_vat_amount",
    "payment_date", "cheque_number",
    # The withholding tax certificate's own fourteen (2026-09-01). Not one of
    # them appears on any other form, and none of the sixteen above appears on
    # this one -- a 50 ทวิ has no VAT totals, no invoice number and no buyer, and
    # its two parties are the one who WITHHELD the tax and the one it was
    # withheld from. So the block is disjoint rather than an extension, which is
    # why the party keys are payer/payee rather than a reuse of seller/buyer:
    # "seller" on a withholding certificate would be a label that is wrong on
    # every document that carries it.
    "book_no", "certificate_no", "sequence_no",
    "payer_tax_id", "payer_name", "payer_branch", "payer_address",
    "payee_tax_id", "payee_name", "payee_branch", "payee_address",
    "dividend_rate_option",
    "total_amount_paid", "total_wht_amount",
)

# The cells of one income row on a withholding certificate, and the key the list
# of them comes back under. This is the FIRST requirement in this project to ask
# for a table -- the four fields marked "Table Extraction" -- and it is asked for
# per type rather than by turning `EXTRACT_LINE_ITEMS` back on, so that the
# invoice, credit note and receipt forms are untouched by it and no pass-2
# measurement already taken moves. See `DOC_TYPE_ITEMS`.
#
# It is deliberately NOT `line_items`/`ITEM_KEYS`: that is the CHARGES table, its
# eight cells are a different set of things, and `fieldscore` derives its truth
# from a Markdown table by a column mapping written for charges. One key holding
# two tables would score a certificate's income rows against an invoice's charge
# rows the day a fixture exists.
#
# `wht_rate` -- the requirement's "Derived WHT Rate %" -- is NOT here, and the
# clue is in its own name. It is derived in `normalise.py` from the two figures
# beside it, like every other derived value in this project, because a model
# asked to work out a percentage while it reads starts adjusting what it reads
# to fit the percentage. It is Optional in the requirement, which is what makes
# that affordable.
INCOME_ITEMS_KEY = "income_items"
INCOME_ITEM_KEYS = ("income_type", "payment_date", "amount_paid", "wht_amount")

# Keys this project used to extract and no longer does, because no requirement
# names them. They are still in the truth files -- those are the human's record
# of the page, not of the schema -- so `fieldscore.load_truth` has to tell
# "retired" apart from "misspelt" and skips these silently. The day a
# requirement asks for one, put it back in `_SCALAR_KEYS` and its form and it is
# scored again with no truth-file edit.
#
# Not to be confused with a key that is in the schema but not in THIS document's
# form -- a receipt is not asked for `amount_incl_vat`, and that key is very much
# still live.
RETIRED_KEYS = ("reference_document", "original_invoice_date",
                "seller_name", "buyer_name", "currency")

# The fourteen keys that were the whole schema before any of this, kept ONLY so
# `_selftest` can assert the assembled prompt still reproduces the one every
# pass-2 measurement in CLAUDE.md was taken on, byte for byte. It is not a form
# anything asks for any more.
LEGACY_KEYS = (
    "document_type", "document_number", "issue_date", "reference_document",
    "seller_name", "seller_tax_id", "seller_branch",
    "buyer_name", "buyer_tax_id", "buyer_branch",
    "currency", "subtotal", "vat_total", "amount_incl_vat",
)

_PARTIES = ("seller_tax_id", "seller_branch", "buyer_tax_id", "buyer_branch")

# One form per type, straight from that type's requirement table, in reading
# order. A document is often more than one type and its form is the UNION of
# theirs (`fields_for_types`).
#
# **A type with no requirement contributes NOTHING.** That is the user's rule --
# *drop all fields/prompt that is not in requirement* -- carried to its
# conclusion: TAX_INVOICE, STATEMENT_OF_ACCOUNT and DEBIT_NOTE have no table of
# their own, so they add no key to a form. It matters, because six of the ten
# fixtures are one of those as well as something else: sol003 is a receipt AND a
# tax invoice, and it is asked the receipt's eleven and nothing more.
#
# The consequence worth knowing: **a receipt's own number, date and document type
# are not extracted at all.** The Receipt table does not list them -- it asks for
# the document being SETTLED, not for the receipt's own identity -- and they come
# back under their own printed labels in `other_fields` like everything else that
# is not on a requirement. Classification is unaffected: the type is read off the
# transcript in Python, never out of an extracted field.
DOC_TYPE_FIELDS = {
    "INVOICE": ("document_type", "document_number", "issue_date",
                "po_gr_rtv_number") + _PARTIES + ("subtotal", "vat_total",
                                                  "amount_incl_vat"),
    "CREDIT_NOTE": ("document_type", "document_number", "issue_date",
                    "po_gr_rtv_number") + _PARTIES + ("subtotal", "vat_total",
                                                      "amount_incl_vat"),
    "RECEIPT": ("inv_rtv_cnr_number",) + _PARTIES + (
        "subtotal", "vat_total", "remaining_amount", "remaining_vat_amount",
        "payment_date", "cheque_number"),
    # The WHT Certificate (50 thawi) table, 2026-09-01. It shares NOTHING with
    # the three above: no document type, no document number, no VAT totals, no
    # seller and no buyer. Its four Table Extraction fields are `income_items`
    # rather than scalars -- see `DOC_TYPE_ITEMS`.
    #
    # The requirement's table prints the four Payer fields three times over. A
    # 50 thawi has exactly two party blocks -- the one that withheld the tax and
    # the one it was withheld from -- so the repeats are read as the payee's
    # block, at the user's decision (2026-09-01). Reading them literally would
    # leave the income recipient's tax ID unextracted, which is the side most
    # matching flows need.
    "WHT_CERTIFICATE": ("book_no", "certificate_no", "sequence_no",
                        "payer_tax_id", "payer_name", "payer_branch",
                        "payer_address",
                        "payee_tax_id", "payee_name", "payee_branch",
                        "payee_address",
                        "dividend_rate_option",
                        "total_amount_paid", "total_wht_amount"),
    # No requirement given for these three. They add nothing to a form; where a
    # document is ONLY one of them, `fields_for_types` falls back to the default.
    "TAX_INVOICE": (),
    "STATEMENT_OF_ACCOUNT": (),
    "DEBIT_NOTE": (),
}

# What to ask when no type with a requirement is in play. The union of every form
# that has one, so nothing any requirement makes Mandatory is dropped from a
# document this project could not place: a narrower form chosen on a guess loses
# fields, a wider one costs tokens.
#
# **THE WITHHOLDING CERTIFICATE MADE THAT TRADE MUCH DEARER, AND THE RULE IS KEPT
# ANYWAY (2026-09-01).** Its fourteen keys share nothing with the sixteen already
# here, so the union went 16 -> 30 and an unclassified agentic run went from ten
# steps to seventeen -- and "only costs tokens" is not true at that width. This
# file's own strongest finding is that widening the ask damages the keys that
# were already right: sol001 fell from 23 correct to 8 when four steps gained a
# fourth key.
#
# It is kept because the alternative is a GUESS -- excluding the certificate's
# keys from the default is a bet that an unplaced page is not one -- and because
# nothing has measured it either way. **This is the shape to point the next
# pass-2 sweep at.** The table is deliberately NOT in the default, for a reason
# that does not apply to scalars: see `items_for_types`.
DEFAULT_FIELDS = _SCALAR_KEYS

# Of each form, the fields its requirement marks Mandatory -- what `grounding`
# reports as missing and what `validate.py` demands. **This is where the three
# requirements genuinely differ.**
MANDATORY_FIELDS = {
    "INVOICE": ("document_type", "document_number", "issue_date",
                "po_gr_rtv_number") + _PARTIES + ("subtotal", "vat_total",
                                                  "amount_incl_vat"),
    # PO/GR/RTV No. is explicitly No on a credit note.
    "CREDIT_NOTE": ("document_type", "document_number", "issue_date")
                   + _PARTIES + ("subtotal", "vat_total", "amount_incl_vat"),
    # Remaining amount, remaining VAT, payment date and cheque number are all No.
    "RECEIPT": ("inv_rtv_cnr_number",) + _PARTIES + ("subtotal", "vat_total"),
    # Book, certificate and sequence numbers are No; so are both branches, both
    # addresses and the dividend rate option. The four Table Extraction fields
    # ARE Mandatory, and they are not scalars -- a per-row obligation has no
    # place in a list of keys, and `validate._check_income_items` is what
    # reports a row that leaves one of them empty.
    "WHT_CERTIFICATE": ("payer_tax_id", "payer_name",
                        "payee_tax_id", "payee_name",
                        "total_amount_paid", "total_wht_amount"),
    "TAX_INVOICE": (),
    "STATEMENT_OF_ACCOUNT": (),
    "DEBIT_NOTE": (),
}

DEFAULT_MANDATORY = ()

# Which validation rules a type's requirement names, beyond the ones every
# document gets. Read by `validate.py`; the union applies where a document is
# more than one type. A type with no requirement names none of them -- running a
# rule nobody asked for would report a finding against a document that is not
# held to it.
#
#   duplicate            Invoice: INV/CN No. "Pattern Check + Duplicate Check"
#   original_document    Credit Note: "Pattern Check + Original Document Matching"
#   future_date          Invoice: "Date Format + Future Date Validation". The
#                        credit note asks Date Format only, and a credit note
#                        legitimately post-dates what it corrects.
#   reference_matching   Receipt: INV/RTV/CNR No. "Pattern Check + Reference
#                        Matching"
#   outstanding_balance  Receipt: Remaining Amount "Numeric Format + Outstanding
#                        Balance Validation"
#   vat_balance          Receipt: Remaining VAT Amount "VAT Balance Validation"
TYPE_RULES = {
    "INVOICE": ("duplicate", "future_date"),
    "CREDIT_NOTE": ("original_document",),
    "RECEIPT": ("reference_matching", "outstanding_balance", "vat_balance"),
    # company_master        WHT: Payer Name "Fuzzy match with Company Master
    #                       >= 90%". There is no Company Master here, so it runs
    #                       as `unchecked` with the reason -- which is the honest
    #                       answer and not a pass.
    # sum_reconciliation    WHT: Total Amount Paid and Total WHT Amount must each
    #                       equal the sum of their column in the income table.
    #                       The one arithmetic rule in any requirement that runs
    #                       over a TABLE rather than over printed totals alone.
    # dividend_option       WHT: the rate option is "required when Income Type =
    #                       4(kho)", so it is Optional on every other row and a
    #                       finding only on a dividend line.
    "WHT_CERTIFICATE": ("company_master", "sum_reconciliation",
                        "dividend_option"),
}

# Which types ask for a TABLE, and the row shape each asks for. Empty for every
# type but one.
#
# **Per type rather than the old global `EXTRACT_LINE_ITEMS` flag, and that is
# the whole point.** Turning that flag back on would restore the charges table on
# the invoice, credit note and receipt forms as well, and CLAUDE.md has the
# measurement for what a table does to a form that does not rule one: sol001 fell
# from 23 correct values to 10 when its rows gained two keys its table had no
# columns for, because the model filled every cell it was given from whatever was
# nearest. This way no pass-2 number already taken moves.
DOC_TYPE_ITEMS = {
    "WHT_CERTIFICATE": INCOME_ITEM_KEYS,
}

# Of the TABLE a form asks for, the cells its requirement marks Mandatory. The
# certificate marks all four of its Table Extraction fields so, which is why this
# map looks redundant beside `DOC_TYPE_ITEMS` -- it will not be the moment a
# second type rules a table with an Optional column in it.
#
# **Separate from `MANDATORY_FIELDS` for the reason `DOC_TYPE_ITEMS` is separate
# from `DOC_TYPE_FIELDS`: a per-ROW obligation is not a key.** `validate` reports
# one against a path (`income_items[0].amount_paid`) that no key list could ever
# match, and the page draws it on a column heading rather than on a row of its
# own. This is the single statement of the fact; `validate._check_income_items`
# reads it rather than restating it, and the Fields tab is told it rather than
# working it out in JavaScript.
MANDATORY_ITEMS = {
    "WHT_CERTIFICATE": INCOME_ITEM_KEYS,
}


def items_for_types(codes):
    """The row shape of the table these types ask for, or `()` for none.

    Only one type asks for a table, so a union cannot arise in practice, and if
    two ever did the answer would not be their union: two tables are two keys,
    not one key with more columns. The first type that rules one wins and the
    caller gets a shape it can actually ask for.

    **`()` for an unclassified document, deliberately -- unlike `DEFAULT_FIELDS`,
    which IS the union of every form.** The two are not inconsistent, because a
    scalar and a table fail differently when a document does not have one: an
    unasked-for scalar comes back "" and costs tokens, whereas a table comes back
    FILLED, from whatever is nearest, which is exactly the failure measured on
    sol001. A wider scalar set is a cost; a table nobody rules is a wrong answer.
    """
    for code in TYPE_SPECIFICITY:
        if code in set(codes or ()) and DOC_TYPE_ITEMS.get(code):
            return tuple(DOC_TYPE_ITEMS[code])
    return ()


def mandatory_items_for_types(codes):
    """Of the table these types ask for, the cells the requirement demands.

    Mirrors `items_for_types` rather than taking a union: the answer has to be
    cells of the ONE table that was actually asked for, so demanding a column
    from a table this form does not rule would be a compliance claim about a
    question nobody put.
    """
    for code in TYPE_SPECIFICITY:
        if code in set(codes or ()) and DOC_TYPE_ITEMS.get(code):
            return tuple(MANDATORY_ITEMS.get(code, ()))
    return ()


def rules_for_types(codes):
    """The extra validation rules the requirements name for these types."""
    out = []
    for code in (codes or []):
        for rule in TYPE_RULES.get(code, ()):
            if rule not in out:
                out.append(rule)
    return tuple(out)


def _ordered(keys):
    """A set of keys in the schema's own reading order.

    The prompt lists them in this order and the page draws them in it, so a form
    assembled from two types has to come out reading like a document rather than
    like a set union in whatever order the types happened to arrive.
    """
    wanted = set(keys)
    return tuple(k for k in _SCALAR_KEYS if k in wanted)


def fields_for_types(codes):
    """The scalar keys to ask for, for every type a document names.

    **The union, not the first match.** A page headed ใบเสร็จรับเงิน/ใบกำกับภาษี
    is a receipt AND a tax invoice and has to answer for both; taking one and
    discarding the other is the bug this signature exists to make impossible.

    No codes at all -- an unrecognised heading, or a caller that did not classify
    -- returns `DEFAULT_FIELDS`, which is wider than any typed form. Guessing a
    type to pick a narrower form would drop Mandatory fields on a coin flip.
    """
    keys = set()
    for code in (codes or []):
        keys.update(DOC_TYPE_FIELDS.get(code, ()))
    # Nothing to ask means no type in play had a requirement -- either the page
    # was not classified, or it is only a kind this project has no table for.
    # The default is the union of every requirement rather than one of them: a
    # narrower form chosen on a guess drops Mandatory fields, a wider one only
    # costs tokens.
    return _ordered(keys) if keys else DEFAULT_FIELDS


def mandatory_for_types(codes):
    """The keys the requirement demands, for every type a document names.

    The union again, and for a sharper reason than the field set: a page that is
    both a credit note and a tax invoice must satisfy BOTH requirements, and an
    intersection would let each type excuse the other's obligations.
    """
    keys = set()
    for code in (codes or []):
        keys.update(MANDATORY_FIELDS.get(code, ()))
    if keys:
        return _ordered(keys)
    # No requirement in play, so nothing is demanded. NOT the union of every
    # requirement: demanding a key of a document no table covers would report a
    # compliance failure this project has no authority to claim.
    #
    # **Empty is a THIRD state to `fieldscore`, not the same as saying nothing**
    # (2026-09-04). It scores such a document over the base field set and marks
    # the score `unknown_type`, because the ground truth states values and
    # throwing that measurement away rendered as no score at all. Nothing about
    # compliance changes: this still returns (), `validate` still demands
    # nothing, and the page still marks no key REQUIRED.
    return DEFAULT_MANDATORY


def fields_for_type(code):
    """One type's form. `fields_for_types` is what callers should use."""
    return fields_for_types([code] if code else [])


def mandatory_for_type(code):
    """One type's Mandatory keys. Prefer `mandatory_for_types`."""
    return mandatory_for_types([code] if code else [])

LINE_ITEM_SCHEMA = {
    "type": "array",
    "items": {"type": "object",
              "properties": {k: {"type": "string"} for k in ITEM_KEYS},
              "required": list(ITEM_KEYS)},
}
OTHER_FIELDS_SCHEMA = {
    "type": "array",
    "items": {"type": "object",
              "properties": {"label": {"type": "string"},
                             "value": {"type": "string"}},
              "required": ["label", "value"]},
}

# `line_items` is deliberately absent -- a grammar that contains the key is an
# invitation to fill it, and the whole point of this pass is that it does not ask
# for the table.
#
# Every key is required. Narrowing this to four was tried on 2026-08-19 and
# reverted the same day, unmeasurable: the fill pressure it was aimed at turned
# out to live in the prompt TEXT, and the blank forms that proved it came off the
# plain request, which no grammar touches. This request is only ever the rescue
# after a plain one returned nothing at all, so a grammar that permits `{}` would
# waste the rescue for a saving that was never demonstrated.
def income_items_schema(items):
    """The grammar for a table of income rows, as one type asks for it."""
    return {
        "type": "array",
        "items": {"type": "object",
                  "properties": {k: {"type": "string"} for k in items},
                  "required": list(items)},
    }


def extract_schema(keys, items=()):
    """The JSON grammar for one field set, as `backends.structured_request` sends it.

    A function of the field set for the same reason the prompt is: a grammar
    listing a key the prompt never asked for is an invitation to fill it, which
    is the exact wording of the note above about `line_items`. `items` is the
    row shape of a table this form asks for -- and where it asks for none, the
    key is absent from the grammar as well as from the skeleton.
    """
    keys = list(keys)
    properties = {k: {"type": "string"} for k in keys}
    required = list(keys)
    if items:
        properties[INCOME_ITEMS_KEY] = income_items_schema(items)
        required.append(INCOME_ITEMS_KEY)
    properties["other_fields"] = OTHER_FIELDS_SCHEMA
    return {
        "type": "object",
        "properties": properties,
        "required": required + ["other_fields"],
    }


# The grammar for a document whose type is not known. Kept as a constant because
# it is what a caller that has not classified anything should send.
EXTRACT_JSON_SCHEMA = extract_schema(DEFAULT_FIELDS)

# --------------------------------------------------------------------------
# pass 2, agentic mode: one small group of fields per request
# --------------------------------------------------------------------------

# The same 29 scalars and two lists as EXTRACT_PROMPT, asked one to three at a
# time: 15 requests against the same transcript, merged in `app._extract_agentic`.
# One message per step, assembled as PREFIX + transcript + TASK.format(...) from
# the step's own row of EXTRACT_STEPS.
#
# The transcript comes FIRST here, the opposite way round from EXTRACT_PROMPT, so
# that every step shares a prefix llama.cpp can prefill once. Reverse it and all
# 15 steps become a cache miss.
#
# Two things to know before editing anything below:
#
# * EXTRACT_STEP_TASK is shared by all 15 steps -- a clause added here changes
#   every answer, including the 14 that were already right. Fix a step in its own
#   `rules` instead.
# * On Ollama, some steps die by transcribing the page into their own first key.
#   Bounding the answer's length in the SHARED task was tried and made it worse.
#   What does work is the ORDER of a step's own skeleton: `seller` leads with its
#   shortest, most reliably printed key rather than the name, and stopped echoing.
#   `references` and `other` still echo and cannot be reordered out of it.
#   CLAUDE.md has the measurements.
#
# Each step's skeleton is written out rather than generated, so the exact bytes
# the model is asked for can be read here.
EXTRACT_STEP_PREFIX = """Below is the full text of a Thai/English business document that
has already been transcribed. Read all of it. One question about it follows the text.

--- document text ---
"""

EXTRACT_STEP_TASK = """
--- end of document text ---

You are filling in ONE part of a form about the document above: {title}.{context}

This is a lookup, not an analysis. Find where the document states each thing and copy it
across. Do not interpret it and do not work anything out from it.

Return ONLY this JSON object, with no prose, no explanation and no code fence:

{skeleton}

{rules}

Rules for every answer:
- Find each value by its printed label, wherever on the page that label sits. Documents
  word their labels differently -- match on what a label MEANS, not on any exact wording.
- Copy each value EXACTLY as the document prints it. Do not translate it, do not reformat
  it, and do not convert Thai text to English.
- Every value must appear, character for character, in the document text above. If you
  cannot point at it there, the field is not yours to fill.
- A value belongs to the field whose printed LABEL matches it, never to the field it
  happens to sit nearest on the page.
- Use "" for anything the document does not state. An empty field is a correct answer and
  a common one; a plausible invented value is neither. Never write a placeholder, a row of
  zeros, a dash, or a figure you would have to work out from other figures.
- Return exactly the keys shown above and no others. Nothing written in these instructions
  is itself a key or a value: where a rule names a label to look for, that is a hint for
  finding the value on the page, never something to emit.
- Do NOT transcribe the document, and do NOT return a natural_text field. Answer with the
  JSON object above and nothing else."""

# Appended to a step's message when its answer held values that are not in the
# transcript, to ask that step's question a second time. The rejected values are
# quoted back because naming them is what makes it a different question -- under
# greedy decoding, asking again in the same words returns the same answer.
EXTRACT_STEP_RETRY = """

Your previous answer to this question contained values that do not appear anywhere in the
document text above:
{rejected}
Each of those was invented, or copied out of a different field. Answer the question again.
For each of those fields either find the value the page actually prints under that field's
own label, or return "" for it. Return the same JSON object as before."""

# One row per step: id, title, keys, skeleton, rules, max_tokens. `keys` is the
# only thing the merge takes from that step's reply -- anything else the model
# returns is dropped rather than trusted.
#
# The step table, and the fields each step extracts. A DOCUMENT runs the subset
# whose keys its form asks for, `other` included: SEVEN for an invoice, a credit
# note or a receipt, NINE for a withholding certificate, and seventeen for a
# document nothing classified -- see `DEFAULT_FIELDS`, which is the union of
# every form and is the one shape here worth re-measuring.
#
#   document           document_type, document_number, issue_date
#   po_gr_rtv          po_gr_rtv_number
#   seller             seller_tax_id, seller_branch
#   buyer              buyer_tax_id, buyer_branch
#   totals             subtotal, vat_total
#   total_incl         amount_incl_vat
#   inv_rtv_cnr        inv_rtv_cnr_number
#   remaining          remaining_amount, remaining_vat_amount
#   payment            payment_date, cheque_number
#   wht_certificate    book_no, certificate_no, sequence_no
#   wht_payer          payer_tax_id, payer_name, payer_branch
#   wht_payer_address  payer_address
#   wht_payee          payee_tax_id, payee_name, payee_branch
#   wht_payee_address  payee_address
#   wht_income         income_items[]
#   wht_totals         total_amount_paid, total_wht_amount
#   wht_dividend       dividend_rate_option
#   other              other_fields[]
#
# `reference`, `currency` and `original_invoice` were deleted on 2026-08-31 with
# the keys they owned. A step whose keys are not in the schema asks a question
# whose answer has nowhere to go.
#
# **`totals` split from three keys to two on the same date**, and it is the one
# step here that was cut for a reason other than the schema: a receipt's
# requirement asks for the amount excluding VAT and the VAT, and NOT for the
# amount including VAT. A step is taken whole or not at all -- `steps_for_types`
# includes one only when every key it owns is in the form -- so a three-key
# totals step would have taken the receipt's two figures down with the third.
# `total_incl` carries the with-tax rules, which were the bulk of the old step.
#
# **THREE KEYS PER STEP IS A CEILING, NOT A STYLE.** Measured 2026-08-19: adding a
# fourth key to a step does not cost that key, it costs the keys the step was
# ALREADY getting right. sol001 fell from 23 correct values out of 43 to 8 when
# four steps were widened, with `subtotal` coming back empty and the seller keys
# filling from a payment slip at the foot of the page. If this schema ever grows
# again, add a step. Do not add a key to a step that already has three.
#
# The grouping is not arbitrary: fields competing for the same value are asked
# together, and fields mistaken for one another are kept apart. seller and buyer
# are the pair that matters -- each step is told which block is which, because a
# tax ID labelled more plainly is very often the other party's.
EXTRACT_STEPS = (
    {
        "id": "document",
        "title": "what this document is",
        "keys": ("document_type", "document_number", "issue_date"),
        "skeleton": '{ "document_type": "", "document_number": "", "issue_date": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- document_type is the heading that names what this document is. Copy the heading itself, not
  a description of it, and do not translate it or turn it into a category -- the heading is
  normalised afterwards, by a program.
- It is often printed near the head of the page and often not: take it from wherever the page
  prints it. A company, brand or place name is never the heading -- those name a party to the
  document, not the kind of document it is.
- document_number is the number identifying this document, printed with that heading.
- issue_date is the date the document is dated. Where the page prints several dates, take
  the one labelled as this document's own date -- not a due date, not a period, not a
  payment date.""",
    },
    {
        "id": "po_gr_rtv",
        "title": "the order, receipt or return number this document cites",
        "keys": ("po_gr_rtv_number",),
        "skeleton": '{ "po_gr_rtv_number": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- po_gr_rtv_number is ONE identifier: the purchase order, goods receipt, or return-to-vendor
  number this document is raised against, whichever of those the page prints.
- Find it by its printed label -- a label meaning purchase order, order number, goods
  receipt, receiving number, or return to vendor. Name that label to yourself before
  answering; where you cannot name one, the answer is "".
- It is never this document's own number. A customer account, contract, site or location
  code is not one of these either, however close to the top of the page it is printed.
- Where the page prints several such numbers, answer with the one whose label matches best
  and leave the rest alone.
- Answer with the one short value and stop. If you find yourself copying a line of the page
  into this field, the answer is "".""",
    },
    {
        "id": "seller",
        "title": "who issued this document",
        "keys": ("seller_tax_id", "seller_branch"),
        "skeleton": '{ "seller_tax_id": "", "seller_branch": "" }',
        "max_tokens": 220,
        "rules": """The issuer is the party this document is FROM -- the one who wrote it, who
is to be paid, or whose details close it.
- Find that party's own block first and work inside it and nowhere else. It is often the
  letterhead at the top of the page, but not always: on an order, a request, or a form
  printed by the other party, the block at the top belongs to the other party. Decide from
  the labels printed around each block, not from where the block sits.
- A value printed in a block labelled as the customer's, or as the party being billed,
  belongs to the other party and not here.
- Where the page does not name the issuer, or you cannot tell which party is which, leave
  these three "" rather than choosing.
- seller_tax_id is only the digits printed as that party's tax identification number. Copy
  the separators as printed if there are any.
- A tax or company registration number is normally a run of ten or more digits, sometimes
  written with dashes or spaces and sometimes carrying a letter prefix. Copy it as printed.
  A short number of two or three digits is something else the page numbers -- a book,
  volume, page or sequence number -- and does not belong here. Where nothing on the page is
  labelled as that party's tax or registration number, this is "".
- seller_branch is what the page prints to say which office, branch or site of that party
  this document belongs to -- a branch name, a branch number, or a word meaning head office.
  Copy it as printed; do not translate it, and do not turn a word into a number.
- This field is the one exception to finding a value by its label: it is usually printed
  with NO label of its own, so look for the value itself. Inside that party's block it is
  normally one of these -- a word meaning head office standing alone on its own line; a word
  meaning head office or branch run on to the END of that party's tax ID line; the same in
  brackets after that party's name; or the word for branch followed by a number. Where you
  can see any of those in that party's block, that IS this field. Copy the whole of what is
  printed there, including the word for branch where the page prints one.
- What is not a branch: a street, a town, a postcode or a telephone number is an address,
  and the tax ID itself is not a branch. Where the page prints only an empty caption asking
  for a branch, with nothing written against it, this is "".
- An office or branch printed after the issuer's name, in brackets or otherwise, is the
  branch and belongs to seller_branch.
- Neither of these two may take the customer's tax ID or branch. The issuer's name is not
  asked for here -- find the block by it, then answer with the two values inside it.""",
    },
    {
        "id": "buyer",
        "title": "who this document is addressed to",
        "keys": ("buyer_tax_id", "buyer_branch"),
        "skeleton": '{ "buyer_tax_id": "", "buyer_branch": "" }',
        "max_tokens": 220,
        "rules": """The customer is the party this document is addressed TO -- the one billed,
ordered from, or delivered to. It is not the party that issued the document.
- The customer's block is found by its LABELS -- lines labelled with words meaning customer,
  buyer, bill to, deliver to, or the party being billed. A block at the top of the page
  carrying a logo and no such label is usually the issuer's letterhead rather than the
  customer, so do not take it merely for being first; but on an order or a request the block
  at the top can be the customer, and its labels are what say so. Where no block on the page
  is labelled as the customer, both of these are "".
- The customer's NAME is not asked for here. Find the block by it, then answer with the two
  values printed inside it.
- buyer_tax_id is only the digits printed as that party's tax identification number, and
  buyer_branch is what the page prints to say which office, branch or site of that party
  this document belongs to -- a branch name, a branch number, or a word meaning head office.
- A tax or company registration number is normally a run of ten or more digits, sometimes
  written with dashes or spaces and sometimes carrying a letter prefix. Copy it as printed.
  A short number of two or three digits is something else the page numbers -- a book,
  volume, page or sequence number -- and does not belong here. Where nothing on the page is
  labelled as that party's tax or registration number, this is "".
- buyer_branch is the one exception to finding a value by its label: it is usually printed
  with NO label of its own, so look for the value itself. Inside the customer's block it is
  normally one of these -- a word meaning head office standing alone on its own line; a word
  meaning head office or branch run on to the END of that party's tax ID line; the same in
  brackets after that party's name; or the word for branch followed by a number. Where you
  can see any of those in the customer's block, that IS this field. Copy the whole of what
  is printed there, including the word for branch where the page prints one.
- A street, a postcode or a telephone number is an address, not a branch, and the tax ID
  itself is not a branch.
- Neither of these two may repeat the issuer's tax ID or branch.""",
    },
    {
        "id": "totals",
        "title": "the total before tax and the tax line",
        "keys": ("subtotal", "vat_total"),
        "skeleton": '{ "subtotal": "", "vat_total": "" }',
        "max_tokens": 180,
        "rules": """These are two different figures, each read off its own printed line.
- subtotal is the figure on the total line printed BEFORE tax is added -- the total of the
  goods or services themselves.
- vat_total is the figure on the tax line.
- Take each from its own printed label, never from its position on the page. Pages order
  these lines differently, and the order tells you nothing about which is which.
- Fill only lines the page actually prints. Never add the rows up yourself and never derive
  one of these from another.
- Where one printed line carries several figures side by side, each under a heading of its
  own, they are separate figures: take the one whose heading names this key, not the first
  one on the line. A figure standing under a heading for tax deducted at source belongs to
  neither of these.
- Where a money column is ruled into two parts under one heading -- the whole units in one
  and the fraction of the unit in the next -- those two parts are ONE figure: take them
  together, the fraction after the decimal point. This applies only to a column actually
  ruled that way; two money columns each with a heading of its own are two figures, not one,
  and neither is half of the other.""",
    },
    {
        "id": "total_incl",
        "title": "the total after tax",
        "keys": ("amount_incl_vat",),
        "skeleton": '{ "amount_incl_vat": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- amount_incl_vat is the figure on the line the page prints as the total INCLUDING tax.
- It is only a total the page prints with tax ADDED to it. A total under a heading meaning
  net, or the amount payable, or the amount left after tax has been deducted at source, is
  NOT this key however it is placed -- and the last figure on a totals line is very often
  exactly that. Where the page prints a net total and no with-tax total, the answer is "".
- Where the page charges no tax -- no tax line at all, or a tax line printed as a dash or as
  zero -- the single total it prints IS the total including tax, and is the answer here.
- Take it from its own printed label, never from its position on the page. Never add the
  rows or the other totals up yourself.""",
    },
    {
        "id": "inv_rtv_cnr",
        "title": "the document this payment settles",
        "keys": ("inv_rtv_cnr_number",),
        "skeleton": '{ "inv_rtv_cnr_number": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- inv_rtv_cnr_number is ONE identifier: the invoice, return-to-vendor, or credit-note-return
  document that this payment settles.
- Find it by its printed label -- a label meaning invoice number, reference document, return,
  or the document being paid. Name that label to yourself before answering; where you cannot
  name one, the answer is "".
- It is NOT this document's own number. A receipt carries its own number too, usually under a
  label meaning receipt number, and that one is never the answer here.
- Where the page settles several documents -- a table with a column of them, or a list --
  none of them belongs here. Do not pick one and do not join them together: answer "".
- Answer with the one short value and stop.""",
    },
    {
        "id": "remaining",
        "title": "what is still outstanding after this payment",
        "keys": ("remaining_amount", "remaining_vat_amount"),
        "skeleton": '{ "remaining_amount": "", "remaining_vat_amount": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- remaining_amount is what is still owed AFTER this payment, and remaining_vat_amount is the
  tax part of it. Both are figures the page prints under a label meaning remaining,
  outstanding, balance, or still due.
- Copy only a figure printed under such a label. NEVER work one out by subtracting this
  payment from a total -- if the page prints no balance, both of these are "".
- A total, an amount paid, or the figure this document is for is not a remaining balance.
- Most documents print neither. Two empty answers are the common and correct result.""",
    },
    {
        "id": "payment",
        "title": "when the money was paid and by what",
        "keys": ("payment_date", "cheque_number"),
        "skeleton": '{ "payment_date": "", "cheque_number": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- payment_date is the date the money was paid or the cheque was drawn, where the page prints
  one under its own label. It is not the date this document was issued, and not a due date.
- cheque_number is the number of the cheque the payment was made by, where the page prints
  one. A bank account number, a bank branch, or a transfer reference is not a cheque number.
- Where the page prints only an empty caption -- a label for a cheque number or a date with
  nothing written against it -- that key is "". A form that offers a way to pay is not a
  record that it was paid that way.""",
    },
    # ---- the withholding tax certificate (2026-09-01) ----------------------
    # Eight steps for a fourteen-key form and a table, which looks like a lot
    # beside the credit note's six -- and is the rule this file already states,
    # followed rather than bent: THREE KEYS PER STEP IS A CEILING, so a wider
    # form buys more steps and never wider ones. The two address steps hold one
    # key each for the same reason the party steps do not hold four.
    #
    # The two parties are asked in separate steps although their skeletons are
    # identical, which is the seller/buyer split's own reasoning on a document
    # where it matters more: on an invoice the letterhead usually gives the
    # issuer away, and on a 50 thawi the two blocks are the same four things in
    # the same order under two labels, so the label is the ONLY thing that tells
    # them apart. One step asked for both would have nothing else to go on.
    {
        "id": "wht_certificate",
        "title": "how this certificate is numbered",
        "keys": ("book_no", "certificate_no", "sequence_no"),
        "skeleton": '{ "book_no": "", "certificate_no": "", "sequence_no": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- book_no is the book or volume this certificate was torn from, under a label meaning book.
- certificate_no is the certificate's own number, under a label meaning number. These two
  are usually printed together at the top right of the page.
- sequence_no is which line of the tax filing this certificate is: a running number printed
  under a label naming the filing form.
- All three are SHORT -- two to six digits each. A run of ten or more digits is a party's
  tax identification number and is never one of these.
- Where the page prints no such label, or prints the label with nothing written against it,
  that key is "".""",
    },
    {
        "id": "wht_payer",
        "title": "who paid the income and withheld the tax",
        "keys": ("payer_tax_id", "payer_name", "payer_branch"),
        "skeleton": '{ "payer_tax_id": "", "payer_name": "", "payer_branch": "" }',
        "max_tokens": 260,
        "rules": """The payer is the party that PAID the income and withheld tax from it --
the block labelled as the one with the duty to withhold and remit.
- Find that party's block by the LABEL printed above it, not by where it sits or how large
  it is printed. This page carries two blocks holding the same things in the same order,
  and only the label above each says whose they are. Where you cannot tell which block is
  which, leave all three "" rather than choosing.
- A value printed in the block labelled as the party the tax was withheld FROM belongs to
  the other party and not here.
- payer_tax_id is the digits printed as that party's tax identification number, copied with
  whatever separators are printed. It is normally ten or more digits. A short number of two
  to six digits is the book, certificate or sequence number and does not belong here.
- payer_name is that party's name and nothing else, on ONE line. If what you are about to
  write contains a street, a postcode, a telephone number or a tax ID, you have taken too
  much of the block.
- payer_branch is what says which office or branch of that party this is -- a word meaning
  head office, or the word for branch followed by a number -- copied as printed, not
  translated and not turned into a number. It is normally printed with no label of its own,
  on or just under that party's tax ID line. A town, a street or a postcode is an address
  and is not a branch.
- The address is NOT asked for here.""",
    },
    {
        "id": "wht_payer_address",
        "title": "the address of the party that withheld the tax",
        "keys": ("payer_address",),
        "skeleton": '{ "payer_address": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- payer_address is the printed address of the party that PAID the income and withheld the
  tax -- the block labelled as the one with the duty to withhold and remit, not the one the
  tax was withheld from.
- Copy the whole of the address as printed, joined into one value where the page runs it
  over several lines.
- It is the address and nothing else: that party's name, its tax identification number, its
  branch and its telephone number are each something else and are asked for elsewhere.
- Where that block prints no address, the answer is "".""",
    },
    {
        "id": "wht_payee",
        "title": "who the income was paid to and the tax withheld from",
        "keys": ("payee_tax_id", "payee_name", "payee_branch"),
        "skeleton": '{ "payee_tax_id": "", "payee_name": "", "payee_branch": "" }',
        "max_tokens": 260,
        "rules": """The payee is the party the income was paid TO and the tax withheld FROM --
the block labelled as the one whose tax was deducted at source.
- Find that party's block by the LABEL printed above it, not by where it sits. This page
  carries two blocks holding the same things in the same order, and only the label above
  each says whose they are. Where you cannot tell which block is which, leave all three ""
  rather than choosing.
- A value printed in the block labelled as the party with the duty to withhold belongs to
  the other party and not here. None of these three may repeat that party's values.
- payee_tax_id is the digits printed as that party's tax identification number, copied with
  whatever separators are printed. It is normally ten or more digits; a short number of two
  to six digits is a book, certificate or sequence number and does not belong here.
- payee_name is that party's name and nothing else, on ONE line. A street, a postcode, a
  telephone number or a tax ID means you have taken too much of the block.
- payee_branch is what says which office or branch of that party this is -- a word meaning
  head office, or the word for branch followed by a number -- copied as printed. It is
  normally printed with no label of its own, on or just under that party's tax ID line.
- The address is NOT asked for here.""",
    },
    {
        "id": "wht_payee_address",
        "title": "the address of the party the tax was withheld from",
        "keys": ("payee_address",),
        "skeleton": '{ "payee_address": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- payee_address is the printed address of the party the income was paid TO and the tax
  withheld FROM -- the block labelled as the one whose tax was deducted at source, not the
  one with the duty to withhold.
- Copy the whole of the address as printed, joined into one value where the page runs it
  over several lines.
- It is the address and nothing else: that party's name, its tax identification number, its
  branch and its telephone number are each something else and are asked for elsewhere.
- Where that block prints no address, the answer is "".""",
    },
    {
        "id": "wht_income",
        "title": "the table of income this certificate covers",
        "keys": (INCOME_ITEMS_KEY,),
        "skeleton": ('{ "income_items": [ { "income_type": "", "payment_date": "", '
                     '"amount_paid": "", "wht_amount": "" } ] }'),
        "max_tokens": 900,
        "rules": """What to look for:
- One object per row of the income table that the page actually fills in, in the order they
  are printed. A row of the printed form with nothing written against it is not a row: skip
  it, and return an empty list where the table is blank throughout.
- income_type is the description of the kind of income as that row prints it. Where the row
  is a free-text option, the value is the wording written or typed against it, not the
  option's own printed caption.
- payment_date is the date printed on that row, copied exactly as printed -- do not convert
  a Thai year to a Western one and do not reorder the parts.
- amount_paid is the figure in that row's column for the amount paid, and wht_amount the
  figure in its column for the tax withheld and remitted. Take each from its own column
  heading, never from its position in the row.
- Where a row leaves one of these four blank, that key is "" for that row. Do not carry a
  value down from the row above, and never work one figure out from another.
- The totals line under the table is NOT a row and does not belong in this list.""",
    },
    {
        "id": "wht_totals",
        "title": "the totals under the income table",
        "keys": ("total_amount_paid", "total_wht_amount"),
        "skeleton": '{ "total_amount_paid": "", "total_wht_amount": "" }',
        "max_tokens": 160,
        "rules": """These are two different figures on the same printed line, each read off
its own column.
- total_amount_paid is the total of the amounts paid; total_wht_amount is the total of the
  tax withheld and remitted.
- Take each from the column it stands under, not from the order the two appear in.
- Copy them exactly as printed. NEVER add the rows of the table up yourself: where the page
  prints no totals line, both of these are "".
- A total spelled out in words is not one of these, and neither is a figure from any single
  row of the table.""",
    },
    {
        "id": "wht_dividend",
        "title": "the ticked rate option, where the page rules one",
        "keys": ("dividend_rate_option",),
        "skeleton": '{ "dividend_rate_option": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- Only where the page rules a list of rate options against a dividend line and ONE of them
  is ticked, crossed or otherwise marked. The answer is the wording of the option that is
  marked, copied as printed.
- Where no option is marked, the answer is "". A list of options with none of them ticked is
  a form offering a choice, not a choice that was made -- and most certificates leave the
  whole list blank, so "" is the common and correct answer here.
- Where the page rules no such list at all, the answer is "".""",
    },
    {
        "id": "other",
        "title": "the remaining labelled facts",
        "keys": ("other_fields",),
        "skeleton": '{ "other_fields": [ { "label": "", "value": "" } ] }',
        "max_tokens": 900,
        "rules": """What to look for:
- Every remaining fact on the page that has a printed label of its own, whatever kind of
  document this is. What follows is a handful of examples and NOT a list to fill in:
  addresses, dates other than the issue date, other document, order and account numbers,
  codes, tax lines other than the one above, discounts, amounts other than the three totals,
  payment and bank details, page numbers, notes, the rows of a table. Anything else the page
  labels belongs here too, and a document that labels none of these examples simply has
  other things instead.
- label is the document's OWN wording for it, copied as printed. value is what is printed
  against that label.
- Do NOT repeat anything that belongs to another part of this form: the document type and
  number, the issue date, either party's name, tax ID or branch, the currency, or the three
  totals. Those have their own fields and are already filled.
- Where nothing is left over, return an empty list.""",
    },
)


def steps_for_types(codes):
    """The agentic steps that fill exactly the field set for a document type.

    A step is included when EVERY key it owns is in that type's set. The step
    table is already grouped so that this partitions cleanly -- `_selftest`
    asserts the chosen steps cover the type's keys exactly, no key asked twice
    and none left unasked -- so there is never a step to include "partly".
    Emitting a step with some of its keys removed would mean rewriting its
    skeleton and its rules on the fly, and those rules are measured text.

    `other` owns no scalar key and is always included: it is the overflow valve
    every set leans on. Order is always `EXTRACT_STEPS`', so a type's run is the
    full run with steps removed -- the same rule `app._steps_for` follows.

    Takes the CODES a document names, not one code, so a receipt/tax-invoice
    runs the steps for the union of both forms.
    """
    wanted = set(fields_for_types(codes))
    # The table step owns a key that is not a scalar and so is not in the form.
    # It is admitted by the same test as every other step -- every key it owns is
    # asked for -- once the key it owns is in `wanted`.
    if items_for_types(codes):
        wanted.add(INCOME_ITEMS_KEY)
    return tuple(s for s in EXTRACT_STEPS
                 if s["id"] == "other"
                 or (s["keys"] and set(s["keys"]) <= wanted))


def steps_for_type(code):
    """One type's steps. `steps_for_types` is what callers should use."""
    return steps_for_types([code] if code else [])


def _selftest():
    """Assertions that the split did not move anything it was not meant to.

    Run at import. They are cheap, and every one of them is a failure that would
    otherwise surface as a silently different prompt or a field nobody asked for
    -- which is the failure class this whole project is organised around.
    """
    # 1. The assembled prompt still reproduces the one every pass-2 number in
    #    CLAUDE.md was measured on.
    legacy = extract_prompt(LEGACY_KEYS)
    assert "fourteen keys" in legacy, "legacy prompt lost its own key count"
    assert '"currency": ""' in legacy and "other_fields" in legacy

    # 2. Every typed set draws only on keys the universe knows, and every
    #    Mandatory key of a type is actually asked for by it.
    for code, keys in list(DOC_TYPE_FIELDS.items()) + [("", DEFAULT_FIELDS)]:
        unknown = [k for k in keys if k not in _SCALAR_KEYS]
        assert not unknown, f"{code}: {unknown} not in _SCALAR_KEYS"
        assert len(set(keys)) == len(keys), f"{code}: duplicate key"
        missing = [k for k in mandatory_for_type(code) if k not in keys]
        assert not missing, f"{code}: mandatory {missing} is not asked for"

    # 3. The steps cover a form's keys exactly -- over every COMBINATION of
    #    types, not just single ones, because a real document names two of them
    #    about as often as one here. A union that lands between two steps would
    #    leave a key nobody asks or ask one twice.
    import itertools
    codes = list(DOC_TYPE_FIELDS)
    combos = [[]] + [list(c) for n in (1, 2, 3)
                     for c in itertools.combinations(codes, n)]
    for combo in combos:
        # `other_fields` and the income table are not scalars and are in no
        # form's key list; every other step key must be, exactly once.
        covered = [k for s in steps_for_types(combo) for k in s["keys"]
                   if k not in ("other_fields", INCOME_ITEMS_KEY)]
        assert sorted(covered) == sorted(fields_for_types(combo)), (
            f"{combo}: steps cover {sorted(covered)}, "
            f"form is {sorted(fields_for_types(combo))}")
        # A union must never be narrower than any of its parts.
        for code in combo:
            missing = [k for k in DOC_TYPE_FIELDS[code]
                       if k not in fields_for_types(combo)]
            assert not missing, f"{combo}: union drops {missing} from {code}"

    # 3b. The table step is run exactly when the form asks for a table, and the
    #     shape it asks for is the shape the prompt and the grammar ask for. A
    #     step running without the key it fills would have nowhere to put its
    #     answer; the key asked for without the step would leave agentic mode
    #     silently unable to fill it.
    for combo in combos:
        shape = items_for_types(combo)
        ran = [s for s in steps_for_types(combo) if INCOME_ITEMS_KEY in s["keys"]]
        assert bool(shape) == bool(ran), f"{combo}: table step / table shape disagree"
        if shape:
            assert len(ran) == 1, f"{combo}: two steps own {INCOME_ITEMS_KEY}"
            for cell in shape:
                assert f'"{cell}"' in ran[0]["skeleton"], (
                    f"{combo}: the step's skeleton does not ask for {cell}")
            grammar = extract_schema(fields_for_types(combo), shape)
            asked = grammar["properties"][INCOME_ITEMS_KEY]["items"]["properties"]
            assert sorted(asked) == sorted(shape), "grammar and shape disagree"
            assert INCOME_ITEMS_KEY in extract_prompt(
                fields_for_types(combo), combo, items=shape)
        else:
            assert INCOME_ITEMS_KEY not in extract_prompt(
                fields_for_types(combo), combo)
        # A Mandatory cell of a table nobody asked for is a compliance claim
        # about a question nobody put, and one outside the shape is a demand for
        # a column the model was never shown.
        demanded = mandatory_items_for_types(combo)
        assert set(demanded) <= set(shape), (
            f"{combo}: mandatory cells {sorted(set(demanded) - set(shape))} "
            "are not in the shape asked for")

    # 4. The form is in the schema's reading order, however the types arrived.
    order = list(_SCALAR_KEYS)
    for combo in combos:
        form = list(fields_for_types(combo))
        assert form == sorted(form, key=order.index), f"{combo}: out of order"

    # 5. Every key the universe knows is reachable from at least one step, or it
    #    can never be asked for at all.
    step_keys = {k for s in EXTRACT_STEPS for k in s["keys"]}
    orphans = [k for k in _SCALAR_KEYS if k not in step_keys]
    assert not orphans, f"no step asks for {orphans}"

    # 6. The type framing changes the prompt's text and never its shape.
    #    Unclassified must be byte-identical to the prompt before any of it
    #    existed, in both modes -- that is what keeps an unclassified run
    #    comparable with every measurement already taken.
    assert extract_prompt(LEGACY_KEYS, ()) == extract_prompt(LEGACY_KEYS)
    assert "{context}" in EXTRACT_STEP_TASK
    assert type_block((), _SCALAR_KEYS) == "" and type_line(()) == ""
    for step in EXTRACT_STEPS:
        assert step_rules(step, ()) == step["rules"]

    # 6b. The model is never asked for a number. Every percentage in this
    #     project is computed in Python from evidence that can be re-derived,
    #     and a figure a model handed over would sit among them looking like one
    #     of them. This is the only prompt that asks a question whose answer is
    #     not copied text, so it is the only one that could drift.
    low = CLASSIFY_PROMPT.lower()
    for word in ("confidence", "confident", "how sure", "certainty",
                 "probability", "percent", "score", "%"):
        assert word not in low, f"the classify prompt asks for a {word}"

    # 7. Every rule is about a key that exists and a type that has one, and the
    #    key is one that type actually asks for -- a bullet about a key the form
    #    leaves out could never be emitted, and a bullet naming a key the type's
    #    own set does not carry is a rule written against the wrong requirement.
    for (code, key) in TYPE_KEY_RULES:
        assert code in TYPE_NAMES, f"{code}: no such type"
        assert key in _SCALAR_KEYS, f"{code}/{key}: no such key"
        assert key in fields_for_types([code]) or not DOC_TYPE_FIELDS.get(code), (
            f"{code} does not ask for {key}")
    for key in _TYPE_ANY_RULES:
        assert key in _SCALAR_KEYS, f"{key}: no such key"

    # 8. One bullet per key, however many types a document is, and every bullet
    #    reaches the step that owns its key. A key whose bullet no step carries
    #    would be a rule single mode gets and agentic mode does not.
    for combo in combos:
        if not combo:
            continue
        keys = fields_for_types(combo)
        bullets = type_bullets(combo, keys)
        assert len(bullets) == len(set(bullets)), f"{combo}: repeated bullet"
        from_steps = [b for s in steps_for_types(combo)
                      for b in type_bullets(combo, s["keys"])]
        assert sorted(from_steps) == sorted(bullets), (
            f"{combo}: steps carry {len(from_steps)} bullets, form has "
            f"{len(bullets)}")


_selftest()
