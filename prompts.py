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
EXTRACT_PROMPT = """You are reading the text of a Thai/English business document that has
already been transcribed. Map what it prints onto the keys below.

This is a lookup, not an analysis. For each key, find where the document states that
thing and copy it across. Do not interpret the document, do not work anything out from
it, and do not check whether it adds up.

There are only fourteen keys and they are the important ones. Take your time over each.

Return ONLY a JSON object, no prose and no code fence. Use exactly these keys:

{
  "document_type": "",        // the heading naming what this document is
  "document_number": "",
  "issue_date": "",
  "reference_document": "",   // another document this one cites
  "seller_name": "",          // the issuer's name only
  "seller_tax_id": "",
  "seller_branch": "",        // which office or branch, where the page names one
  "buyer_name": "",           // the customer's name only
  "buyer_tax_id": "",
  "buyer_branch": "",
  "currency": "",
  "subtotal": "",             // the total line printed before tax is added
  "vat_total": "",            // the tax line
  "amount_incl_vat": "",      // the total line printed after tax
  "other_fields": [ { "label": "", "value": "" } ]
}

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
- One thing per key, and never the same text in two keys.
- Do not work anything out. Do not add, subtract, convert, or fill one key from another --
  not a tax ID from a name, not a total from the rows above it. If a figure is not printed,
  its key is "".
- Nothing written in these instructions is itself a key or a value: where a rule names a
  label to look for, that is a hint for finding it on the page, never something to emit.

The two parties -- three keys each, and each holds one thing:
- The name key takes the party's name and nothing else, on ONE line: if what you are about
  to write contains a street, a postcode, a telephone number, a tax ID, a date, a branch in
  brackets or a line break, you have taken too much of the block.
- A name is TEXT. A value that is mostly digits, or that reads as a code, is not a name
  however close to the block it is printed -- put it in "other_fields" instead.
- The tax ID key takes what is printed as that party's tax or registration number, copied
  as printed -- normally ten or more digits, sometimes with dashes or a letter prefix. A
  two- or three-digit number is a book, page or sequence number and does not belong here.
- The branch key takes what names which office, branch or site of that party -- a branch
  name, a branch number, or a word meaning head office -- copied as printed, not translated
  and not turned into a number. It often sits beside that party's tax ID, but sitting there
  does not make a value a branch: a town, a street, a postcode, a telephone number or the
  tax ID itself is not one. Answer with the value, never with the words of a label.
- The issuer is the party the document is FROM -- who wrote it, or who is to be paid. The
  other party is the one it is addressed TO or billed. Tell them apart by the labels printed
  around each block, not by where the block sits: the top of the page is usually the
  issuer's letterhead, but on an order or a form printed by the other party it is not.
  Neither party's three keys may take the other's name, tax ID or branch.

reference_document:
- ONE identifier, and only where a printed label says the thing it names is ANOTHER
  document -- one referred to, cited, replaced, credited or settled. Name that label to
  yourself before answering; where you cannot, this key is not yours to fill.
- Never this document's own number, however it is labelled. An unlabelled number is not a
  reference. Where the page cites several documents, put them in "other_fields" rather than
  choosing one.

The money -- three figures and the currency:
- subtotal is the total line printed BEFORE tax is added; vat_total is the tax line;
  amount_incl_vat is the total line printed AFTER tax.
- Fill only the lines the page prints, each from its own label: a page printing one total
  fills the key its label names. Never copy one figure into two keys, and never add or
  subtract to produce another.
- Where one printed line carries several figures side by side, each under a heading of its
  own, they are separate figures: take the one whose heading names this key, not the first
  one on the line. A figure standing under a tax heading is not the total.
- Where a money column is ruled into two parts under one heading, the whole units and then
  the fraction, those parts are ONE figure: take both, the fraction after the decimal point.
  Two money columns with headings of their own are two figures, not halves of one.
- A figure in brackets or with a leading minus is negative. Keep the sign. A cell showing a
  dash is nil: write it as a dash or as "", not as 0.00 unless the page prints 0.00.
- currency only where the page prints one -- a currency word, code or symbol against a
  figure or in a column heading -- in the form the page prints it: where it prints the word,
  answer with the word and not the three-letter code. A currency you know such a document
  normally uses, or infer from the language it is written in, is not one the page prints.

other_fields:
- Everything else the page labels and states, whatever kind of document this is -- by way
  of example only: addresses, other dates, order and account numbers, codes, other tax
  lines, discounts, other amounts, payment and bank details, page numbers, notes, tables.
- label is the document's OWN wording, copied as printed; value is what is printed against
  it. This is where a value goes when it fits none of the fourteen keys -- never force one
  into a key that nearly fits, and never repeat anything already written above.
- Where nothing is left over, return an empty list.

Document text:
"""

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

# The requirement's priority-1 set, in reading order rather than in the order the
# requirement lists them. Fourteen keys. `grounding.SCALAR_FIELDS` is this list,
# and the two must not be allowed to drift.
_SCALAR_KEYS = (
    "document_type", "document_number", "issue_date", "reference_document",
    "seller_name", "seller_tax_id", "seller_branch",
    "buyer_name", "buyer_tax_id", "buyer_branch",
    "currency", "subtotal", "vat_total", "amount_incl_vat",
)

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
EXTRACT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "string"} for k in _SCALAR_KEYS},
        "other_fields": OTHER_FIELDS_SCHEMA,
    },
    "required": list(_SCALAR_KEYS) + ["other_fields"],
}

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

You are filling in ONE part of a form about the document above: {title}.

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
# The 7 steps, and the fields each extracts. Fourteen priority-1 scalars plus the
# overflow list:
#
#   document    document_type, document_number, issue_date
#   reference   reference_document
#   seller      seller_name, seller_tax_id, seller_branch
#   buyer       buyer_name, buyer_tax_id, buyer_branch
#   currency    currency
#   totals      subtotal, vat_total, amount_incl_vat
#   other       other_fields[]
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
        "id": "reference",
        "title": "the document this one refers to",
        "keys": ("reference_document",),
        "skeleton": '{ "reference_document": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- reference_document is ONE identifier, and only where the page labels it as another
  document this one cites or refers to.
- Before you answer, name to yourself the printed label the value sits under, and check that
  the label says the thing it names is ANOTHER document -- one referred to, cited, replaced,
  or being credited or settled. If you cannot name such a label, the answer is "".
- This document's OWN number is never the answer, however it is labelled. An unlabelled
  number is not a reference, and neither is a number that merely looks like one.
- Where the page cites several documents, none of them belongs here: do not pick one, and
  do not join them together.
- Answer with the one short value and stop. If you find yourself copying a line of the page
  into this field, the answer is "".""",
    },
    {
        "id": "seller",
        "title": "who issued this document",
        "keys": ("seller_tax_id", "seller_branch", "seller_name"),
        "skeleton": ('{ "seller_tax_id": "", "seller_branch": "", '
                     '"seller_name": "" }'),
        "max_tokens": 260,
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
- It is often printed beside that party's tax ID, but sitting next to the tax ID does not
  make a value a branch. A street, a town, a postcode or a telephone number is an address,
  not a branch, and the tax ID itself is not a branch either. A word meaning head office is
  a branch and belongs here.
- Answer with the value, never with the words of a label. If the only thing you can find is
  the caption asking for a branch, nothing was printed against it and this is "".
- None of these three may take the customer's name, tax ID or branch.
- seller_name is the issuer's name and nothing else, on ONE line. Stop at the end of the
  name. If what you are about to write contains a street, a postcode, a telephone number, a
  tax ID, a date or a line break, you have taken too much of the block -- cut it back.
- An office or branch printed after the name, in brackets or otherwise, is the branch and
  belongs to seller_branch. It is not part of the name.
- A name is TEXT. A value that is mostly digits, or that reads as a code, is not a name --
  leave seller_name "" rather than putting a code in it.""",
    },
    {
        "id": "buyer",
        "title": "who this document is addressed to",
        "keys": ("buyer_tax_id", "buyer_branch", "buyer_name"),
        "skeleton": '{ "buyer_tax_id": "", "buyer_branch": "", "buyer_name": "" }',
        "max_tokens": 260,
        "rules": """The customer is the party this document is addressed TO -- the one billed,
ordered from, or delivered to. It is not the party that issued the document.
- buyer_name is that party's name and nothing else, on ONE line.
- A name is TEXT. A value that is mostly digits, or that reads as a code -- a site
  reference, a meter number, a customer number -- is NOT a name, however close to this block
  it is printed. Where the block gives you a code and no name, buyer_name is "".
- The customer's block is found by its LABELS -- lines labelled with words meaning customer,
  buyer, bill to, deliver to, or the party being billed. A block at the top of the page
  carrying a logo and no such label is usually the issuer's letterhead rather than the
  customer, so do not take it merely for being first; but on an order or a request the block
  at the top can be the customer, and its labels are what say so. Where no block on the page
  is labelled as the customer, buyer_name is "".
- buyer_tax_id is only the digits printed as that party's tax identification number, and
  buyer_branch is what the page prints to say which office, branch or site of that party
  this document belongs to -- often beside that tax ID, but only where it names an office or
  branch rather than merely sitting next to one.
- A tax or company registration number is normally a run of ten or more digits, sometimes
  written with dashes or spaces and sometimes carrying a letter prefix. Copy it as printed.
  A short number of two or three digits is something else the page numbers -- a book,
  volume, page or sequence number -- and does not belong here. Where nothing on the page is
  labelled as that party's tax or registration number, this is "".
- buyer_branch says which office or branch, and nothing else. A street, a postcode or a
  telephone number is an address, not a branch.
- None of these three may repeat the issuer's name, tax ID or branch.""",
    },
    {
        "id": "currency",
        "title": "the currency",
        "keys": ("currency",),
        "skeleton": '{ "currency": "" }',
        "max_tokens": 120,
        "rules": """What to look for:
- currency only where the page prints one -- a currency word, code or symbol, in a
  money column heading or beside a figure. Copy it in the form the page prints it: a page
  that prints the word gives you the word, not the three-letter code for it.
- Point at it in the text above before you answer. A currency you know such a document
  normally uses, or infer from the language it is written in, is not a currency the page
  states: if you cannot find the word, code or symbol in the text, the answer is "".
- A currency word inside an amount spelled out in words is part of that amount, not a label
  for the column, and does not fill this field on its own.""",
    },
    {
        "id": "totals",
        "title": "the total before tax, the tax line, and the total after tax",
        "keys": ("subtotal", "vat_total", "amount_incl_vat"),
        "skeleton": ('{ "subtotal": "", "vat_total": "", '
                     '"amount_incl_vat": "" }'),
        "max_tokens": 200,
        "rules": """These are three different figures, each read off its own printed line.
- subtotal is the figure on the total line printed BEFORE tax is added -- the total of the
  goods or services themselves.
- vat_total is the figure on the tax line.
- amount_incl_vat is the figure on the line the page prints as the total INCLUDING tax.
- Take each from its own printed label, never from its position on the page. Pages order
  these lines differently, and the order tells you nothing about which is which.
- Fill only lines the page actually prints. Where it shows a single total, put it under the
  key its own label names and leave the others "". Never copy one figure into two of these
  keys, never add the rows up yourself, and never derive one of these from another.
- Where one printed line carries several figures side by side, each under a heading of its
  own, they are separate figures: take the one whose heading names this key, not the first
  one on the line. A figure standing under a tax heading is not the total, and a figure under
  a heading for tax deducted at source belongs to none of these three.
- Where a money column is ruled into two parts under one heading -- the whole units in one
  and the fraction of the unit in the next -- those two parts are ONE figure: take them
  together, the fraction after the decimal point. This applies only to a column actually
  ruled that way; two money columns each with a heading of its own are two figures, not one,
  and neither is half of the other.""",
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
