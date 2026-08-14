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
* `EXTRACT_PROMPT`    -- pass 2, transcript in as text, structured JSON out.
* `EXTRACT_REMINDER`  -- closes pass 2's message, after the transcript.
* `EXTRACT_STEP_*`    -- pass 2 in agentic mode: the same schema asked for one to
                         three fields at a time, one request per step.
* `EXTRACT_STEPS`     -- the step table those are interpolated with. Still only
                         constants: the loop that walks it lives in `app.py`.
* `VERIFY_PROMPT`     -- pass 3, the model's advisory review of the extracted
                         fields, once the arithmetic has already been checked.
"""

# --------------------------------------------------------------------------
# pass 1: OCR
# --------------------------------------------------------------------------

# Page image in, verbatim transcript out. Extracts no fields -- it produces the
# page as Markdown, with tables as <table> HTML, and that transcript is what
# passes 2 and 3 read.
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

# --------------------------------------------------------------------------
# pass 2: field extraction
# --------------------------------------------------------------------------

# Transcript in as text, one JSON object out, in a single request. No image, so
# it costs a fraction of the OCR run.
#
# Extracts 29 scalars and two lists:
#   what it is    document_type, document_number, issue_date, due_date
#   references    reference_document, po_number, original_invoice_number
#   codes         contract_number, customer_code, location_code, service_period
#   the issuer    seller_name, seller_tax_id, seller_branch, seller_address
#   the customer  buyer_name, buyer_tax_id, buyer_branch, buyer_address
#   basis         currency, vat_rate
#   totals        subtotal, vat_total, amount_incl_vat, withholding_tax_total,
#                 net_payable, amount_in_words
#   payment       payment_method, payment_reference
#   lists         line_items[], other_fields[]
#
# It is a LOOKUP: it copies printed values onto keys and reasons about none of
# them. Every judgement about what the figures mean belongs to `verify.py`.
#
# Three things here are load-bearing and look like style until removed -- no
# example value is written in quotes, the skeleton stays a JSON skeleton, and no
# Thai label wording is named. CLAUDE.md says what each one costs.
EXTRACT_PROMPT = """You are reading the text of a Thai/English business document that has
already been transcribed. Map what it prints onto the keys below.

This is a lookup, not an analysis. For each key, find where the document states that
thing and copy it across. Do not interpret the document, do not work anything out from
it, and do not check whether it adds up -- the numbers are checked afterwards by a
calculator, and nothing you write here is used to reason about them.

Return ONLY a JSON object, no prose and no code fence. Use exactly these keys:

{
  "document_type": "",        // the heading naming what this document is
  "document_number": "",
  "issue_date": "",
  "due_date": "",
  "reference_document": "",   // another document this one cites
  "po_number": "",
  "original_invoice_number": "",   // an earlier document this one corrects
  "contract_number": "",
  "customer_code": "",
  "location_code": "",        // code for the site, premises or meter charged
  "service_period": "",       // the period the whole document covers
  "seller_name": "",          // the issuer's name only
  "seller_tax_id": "",
  "seller_branch": "",        // whatever is printed beside that tax ID
  "seller_address": "",
  "buyer_name": "",           // the customer's name only
  "buyer_tax_id": "",
  "buyer_branch": "",
  "buyer_address": "",
  "currency": "",
  "vat_rate": "",             // just the number the document states
  "line_items": [
    { "description": "", "period": "", "quantity": "",
      "unit_price": "", "amount": "", "vat": "",
      "withholding_tax": "", "net_amount": "" }
  ],
  "subtotal": "",             // the total line printed before tax is added
  "vat_total": "",            // the tax line
  "amount_incl_vat": "",      // the total line printed after tax
  "withholding_tax_total": "",     // the tax-deducted-at-source line
  "net_payable": "",          // the line for what is left to pay
  "amount_in_words": "",
  "payment_method": "",
  "payment_reference": "",    // the reference identifying that payment
  "other_fields": [ { "label": "", "value": "" } ]
}

Every key above names something a document of this kind normally prints somewhere. Find
it by its printed label, wherever on the page that label happens to sit, and copy what is
printed against it. Documents word their labels differently and lay them out differently;
match on what the label MEANS, not on where it is or on any particular wording.

How to fill them:
- Copy values EXACTLY as printed. Do not reformat, do not translate, do not convert Thai
  text to English, and do not tidy a number's digits, separators or decimals.
- Every value must appear, character for character, in the document text below. If you
  cannot point at it there, the key is not yours to fill.
- Use "" for anything the document does not state. An empty key is a correct answer and a
  common one; a plausible invented value is neither. Never write a placeholder -- not a
  row of zeros, not a dash, not a repeat of a value you used elsewhere.
- A value belongs to the key whose printed LABEL matches it, never to the key it happens
  to sit nearest. Before writing a value, name to yourself the label it was printed under.
  If it has no such label, it is not that key's value.
- One thing per key, and never the same text in two keys.
- Do not work anything out. Do not add, subtract, convert, or fill one key from another --
  not a tax ID from a name, not a due date from an issue date, not a total from the rows
  above it. If a figure is not printed, its key is "".
- Output exactly the keys listed above and no others. Nothing written in these
  instructions is itself a key or a value: where a rule names a label to look for, that is
  a hint for finding it on the page, never something to emit.
- The skeleton above is the shape of the answer, not the answer. Returning it unchanged,
  with every value still "", is wrong for any document that states anything at all.
- Anything else the page labels and states -- meter readings, page numbers, notes,
  references these keys do not cover -- goes in "other_fields" under the document's own
  wording for it. That is the place for a value that does not fit a key; do not force it
  into one that nearly fits.

The two parties -- four keys each, and each holds one thing:
- The name key takes the party's name and nothing else, on ONE line. Stop at the end of
  the name. If what you are about to write contains a street, a postcode, a telephone
  number, a tax ID, a date or a line break, you have taken too much of the block.
- A name is TEXT. A value that is mostly digits, or that reads as a code -- a site
  reference, a meter number, an account or customer number -- is not a name, however close
  to the block it is printed. Leave the name key "" and put the code in whichever key the
  page labels it as, or in "other_fields".
- The tax ID key takes only the digits printed as the tax ID. The branch key takes what is
  printed beside it, copied as printed -- do not translate it, and do not turn a word into
  a number. The address key takes the street lines and postcode, without the name, the tax
  ID or the branch repeated inside it.
- The issuer is normally the letterhead at the top; the other party is the block addressed
  as the customer or buyer.

The references -- one identifier each, and only where the page labels it as that:
- These keys are empty on most documents, and that is the expected answer. Do not reuse
  the document's own number to fill them, and never write one value into several of them.
- An unlabelled number is not a reference. If a reference is labelled with something these
  keys do not cover, it belongs in "other_fields".
- service_period is a period covering the whole document, printed near its head. A period
  printed on one row of the table belongs to that row instead.

The table:
- One object per printed row of the charges table, in the order they are printed. Copy
  every row once, including a row whose amount is nil. Do not revisit a row.
- Rows that total the ones above them are not line items, whatever they are labelled.
  Their figures belong in the totals keys.
- Read across the row: a figure belongs to the column it sits under. Never shift a value
  into a neighbouring column to make a row look complete. Where the table rules no column
  for one of the keys, that key is "" on every row -- do not produce it from the other
  cells in the row.

Reading a figure:
- A cell showing a dash or left blank is nil: write it as a dash or as "", not as 0.00
  unless the page prints 0.00.
- A figure in brackets or with a leading minus is negative. Keep the sign.
- Some forms rule a money column into two parts, the whole units and the fraction. Two
  such parts under ONE column heading are one figure; two figures under DIFFERENT headings
  are two figures. Keep them apart.

The totals and the rate:
- Fill only the total lines the page actually prints, each from its own printed label. A
  page that prints one total fills the key its label names and leaves the others "". Never
  copy one figure into two keys, and never add or subtract to produce the other.
- vat_rate takes just the number the page states, written the way it writes it: a page
  showing 7 per cent gives 7, one showing 7.00 per cent gives 7.00. Not a fraction, not
  the per-cent sign. Where no rate is printed, "".
- currency only where the page prints one -- a currency word, code or symbol. A document
  written in Thai does not state its currency by being written in Thai. Where none appears
  anywhere, "".

Document text:
"""

# Closes pass 2's message, after the transcript. Extracts nothing itself: it
# restates the schema so the OCR fine-tune answers with the fields rather than
# with its own {"natural_text": "<the whole page>"} envelope. Load-bearing on
# long documents, and not belt and braces -- see CLAUDE.md.
EXTRACT_REMINDER = """

Now return the JSON object described above, using exactly the keys listed.
Do NOT transcribe the document and do NOT return a "natural_text" field."""

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
# * On Ollama, 3 of the 15 steps currently die by transcribing the page into their
#   own first key. Bounding the answer's length was tried and made it worse.
#   CLAUDE.md has the measurements and the one untested hypothesis left.
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
# The 15 steps, and the fields each extracts:
#
#   document        document_type, document_number, issue_date
#   dates           due_date, service_period
#   references      reference_document, po_number, original_invoice_number
#   codes           contract_number, customer_code, location_code
#   seller          seller_name, seller_tax_id, seller_branch
#   seller_address  seller_address
#   buyer           buyer_name, buyer_tax_id, buyer_branch
#   buyer_address   buyer_address
#   basis           currency, vat_rate
#   line_items      line_items[]
#   totals_goods    subtotal, vat_total
#   totals_pay      amount_incl_vat, withholding_tax_total, net_payable
#   words           amount_in_words
#   payment         payment_method, payment_reference
#   other           other_fields[]
#
# The grouping is not arbitrary: fields competing for the same value are asked
# together, and fields mistaken for one another are kept apart -- codes runs
# separately from buyer, which is where location_code was landing in buyer_name.
EXTRACT_STEPS = (
    {
        "id": "document",
        "title": "what this document is",
        "keys": ("document_type", "document_number", "issue_date"),
        "skeleton": '{ "document_type": "", "document_number": "", "issue_date": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- document_type is the heading that names what this document is, printed at the head of the
  page. Copy the heading itself, not a description of it.
- document_number is the number identifying this document, printed with that heading.
- issue_date is the date the document is dated. Where the page prints several dates, take
  the one labelled as this document's own date -- not a due date, not a period, not a
  payment date.""",
    },
    {
        "id": "dates",
        "title": "the dates and period this document covers",
        "keys": ("due_date", "service_period"),
        "skeleton": '{ "due_date": "", "service_period": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- due_date only where the page labels a date as the one payment falls due. A document for
  money already paid has none; leave it "".
- service_period is a period the document as a WHOLE covers, printed near the head of the
  page. A period printed on one row of the table belongs to that row, not here.
- Both of these are empty on many documents, and empty is the right answer for them.""",
    },
    {
        "id": "references",
        "title": "other documents this one refers to",
        "keys": ("reference_document", "po_number", "original_invoice_number"),
        "skeleton": ('{ "reference_document": "", "po_number": "", '
                     '"original_invoice_number": "" }'),
        "max_tokens": 200,
        "rules": """What to look for:
- Each key takes ONE identifier, and only where the page labels it as that kind of thing.
- reference_document is another document this one cites or refers to.
- po_number is a purchase order number the page names as such.
- original_invoice_number is the earlier invoice that a correcting document adjusts.
- These three are empty on most documents. Do NOT reuse this document's own number to fill
  any of them, and never write one value into more than one of them. An unlabelled number
  is not a reference.""",
    },
    {
        "id": "codes",
        "title": "the account, contract and site codes",
        "keys": ("contract_number", "customer_code", "location_code"),
        "skeleton": ('{ "contract_number": "", "customer_code": "", '
                     '"location_code": "" }'),
        "max_tokens": 200,
        "rules": """What to look for:
- contract_number is a contract or agreement number.
- customer_code is the issuer's own code for this customer.
- location_code is a code for the site, premises, meter or delivery point the charge
  belongs to. A short code mixing digits, dashes and a place name is this kind of thing --
  it is a code, not a name and not a document number.
- Only fill a key where the page labels the value as that. Anything labelled something else
  is not one of these, and all three are empty on many documents.""",
    },
    {
        "id": "seller",
        "title": "who issued this document",
        "keys": ("seller_name", "seller_tax_id", "seller_branch"),
        "skeleton": ('{ "seller_name": "", "seller_tax_id": "", '
                     '"seller_branch": "" }'),
        "max_tokens": 260,
        "rules": """The issuer's block is normally the letterhead at the top of the page.
- seller_name is the issuer's name and nothing else, on ONE line. Stop at the end of the
  name. If what you are about to write contains a street, a postcode, a telephone number, a
  tax ID, a date or a line break, you have taken too much of the block -- cut it back.
- seller_tax_id is only the digits printed as that party's tax identification number. Copy
  the separators as printed if there are any.
- seller_branch is what is printed beside that tax ID to say which office or branch it is,
  whether that is a word or a number. Copy it as printed; do not translate it, and do not
  turn a word into a number.
- A name is TEXT. A value that is mostly digits, or that reads as a code, is not a name --
  leave seller_name "" rather than putting a code in it.""",
    },
    {
        "id": "seller_address",
        "title": "where the issuer is",
        "keys": ("seller_address",),
        "skeleton": '{ "seller_address": "" }',
        "max_tokens": 300,
        "rules": """What to look for:
- seller_address is the street lines and postcode from the issuer's block -- the letterhead
  at the top of the page.
- Leave the name, the tax ID and the branch out of it: those are separate fields and must
  not be repeated inside this one.
- Keep the address itself exactly as printed, including its own line breaks.""",
    },
    {
        "id": "buyer",
        "title": "who this document is addressed to",
        "keys": ("buyer_name", "buyer_tax_id", "buyer_branch"),
        "skeleton": '{ "buyer_name": "", "buyer_tax_id": "", "buyer_branch": "" }',
        "max_tokens": 260,
        "rules": """The customer's block is the one addressed as the customer, the buyer or the
party being billed -- NOT the letterhead at the top of the page, which is the issuer.
- buyer_name is that party's name and nothing else, on ONE line.
- A name is TEXT. A value that is mostly digits, or that reads as a code -- a site
  reference, a meter number, a customer number -- is NOT a name, however close to this block
  it is printed. Where the block gives you a code and no name, buyer_name is "".
- The customer's block is found by its LABELS -- lines labelled with words meaning customer,
  name, buyer or the party billed. The block printed at the very top of the page, above this
  document's own heading, is the issuer's letterhead and is never the customer, however
  complete it looks. If the only name you can find is that one, buyer_name is "".
- buyer_tax_id is only the digits printed as that party's tax identification number, and
  buyer_branch is what is printed beside that tax ID to say which office or branch it is.
- buyer_branch says which office or branch, and nothing else. A street, a postcode or a
  telephone number is an address, not a branch, and does not belong in it.
- None of these three may repeat the issuer's name, tax ID or branch from the letterhead.""",
    },
    {
        "id": "buyer_address",
        "title": "where the customer is",
        "keys": ("buyer_address",),
        "skeleton": '{ "buyer_address": "" }',
        "max_tokens": 300,
        "rules": """What to look for:
- buyer_address is the street lines and postcode from the customer's block -- the one
  addressed as the customer or the party being billed, not the letterhead at the top.
- Leave that party's name, tax ID and branch out of it.
- Where the page gives the customer no address, this is "".""",
    },
    {
        "id": "basis",
        "title": "the currency and the tax rate",
        "keys": ("currency", "vat_rate"),
        "skeleton": '{ "currency": "", "vat_rate": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- currency only where the page actually prints one -- a currency word, code or symbol, in a
  money column heading, beside a total, or inside the amount written in words. A document
  written in Thai does not state its currency by being written in Thai; where no currency
  word, code or symbol appears anywhere, currency is "".
- vat_rate is the tax rate the page states. Put just the number, written the way the page
  writes it: a page showing 7 per cent gives 7, a page showing 7.00 per cent gives 7.00.
  Never a fraction such as 0.07, and never the per-cent sign.
- Where no rate is printed anywhere, vat_rate is "" -- do not assume one, and never work it
  out from the figures.""",
    },
    {
        "id": "line_items",
        "title": "the rows of the charges table",
        "keys": ("line_items",),
        "skeleton": """{ "line_items": [
    { "description": "", "period": "", "quantity": "", "unit_price": "",
      "amount": "", "vat": "", "withholding_tax": "", "net_amount": "" }
  ] }""",
        "max_tokens": 3000,
        "rules": """What to look for:
- One object per row of the charges table, in the order the rows are printed. Copy every row
  once, including a row whose amount is nil. A dropped row is the most damaging error here,
  and a row written twice is the second most damaging -- do not revisit a row you have
  already written.
- Rows that total the rows above them are NOT line items, whatever they are labelled. Their
  figures belong to the totals, not here. Stop at the last real charge.
- Read across the row: each figure belongs to the column it sits under. Never shift a value
  into a neighbouring column to make a row look complete.
- Each of these keys is filled only from a column the table actually rules for it. Where
  there is no such column, that key is "" on every row -- never produce it from the other
  cells in the row. A figure you worked out is wrong here even when the arithmetic is right.
- Keep digits exactly as printed, with their thousands separators and decimal places.
- Some forms rule a money column into two parts, the whole units and the fraction. Two such
  parts under ONE column heading are one figure -- a cell reading 9,741 then a gap then 60 is
  9,741.60. Two figures under DIFFERENT headings are two figures; keep them apart.
- A cell showing a dash or left blank is nil: write it as a dash or as "", never as 0.00 unless
  the page prints 0.00. A figure in brackets or with a leading minus is negative; keep the sign.
- Where the table has no rows at all, return an empty list.""",
    },
    {
        "id": "totals_goods",
        "title": "the total before tax, and the tax line",
        "keys": ("subtotal", "vat_total"),
        "skeleton": '{ "subtotal": "", "vat_total": "" }',
        "max_tokens": 160,
        "rules": """What to look for:
- subtotal is the figure on the total line printed before tax is added -- the total of the
  goods or services themselves.
- vat_total is the figure on the tax line.
- Take each from its own printed label, never from its position on the page. Pages order
  these lines differently, and the order tells you nothing about which is which.
- Fill only lines the page actually prints. Where it prints no such line, that key is "" --
  do not add the rows up yourself and do not derive one of these from the other.""",
    },
    {
        "id": "totals_pay",
        "title": "the total, any tax deducted, and what is left to pay",
        "keys": ("amount_incl_vat", "withholding_tax_total", "net_payable"),
        "skeleton": ('{ "amount_incl_vat": "", "withholding_tax_total": "", '
                     '"net_payable": "" }'),
        "max_tokens": 200,
        "rules": """These are three different figures, each read off its own printed line.
- amount_incl_vat is the figure on the line the page prints as the total including tax.
- withholding_tax_total is the figure on the line for tax deducted at source.
- net_payable is the figure on the line for what is left to pay after that deduction. A
  document with no such deduction usually prints no such line -- then net_payable is "".
- Fill only the lines the page prints. Where it shows a single grand total, put it under the
  key its own label names and leave the others "". Never copy one figure into two of these
  keys, and never add or subtract to produce another.""",
    },
    {
        "id": "words",
        "title": "the total written out in words",
        "keys": ("amount_in_words",),
        "skeleton": '{ "amount_in_words": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- amount_in_words is a total spelled out in words rather than digits, usually on a line of
  its own near the totals and often inside brackets.
- Copy the whole phrase exactly as printed, to the end of it. Do not convert it to digits,
  do not shorten it, and do not correct it.
- Where the page spells no amount out, this is "".""",
    },
    {
        "id": "payment",
        "title": "how the document is paid",
        "keys": ("payment_method", "payment_reference"),
        "skeleton": '{ "payment_method": "", "payment_reference": "" }',
        "max_tokens": 200,
        "rules": """What to look for:
- payment_method is how payment was or is to be made, as the page words it. A ticked box
  counts: copy the label beside the tick.
- payment_reference is the number or reference identifying that payment -- a cheque number, a
  transfer reference, a slip number.
- Both are empty on a document that says nothing about how it is paid, which is common.""",
    },
    {
        "id": "other",
        "title": "the remaining labelled facts",
        "keys": ("other_fields",),
        "skeleton": '{ "other_fields": [ { "label": "", "value": "" } ] }',
        "max_tokens": 700,
        "rules": """What to look for:
- Every remaining fact on the page that has a printed label of its own and is worth keeping:
  meter readings, page numbers, order numbers, delivery details, notes, anything this
  document prints that the rest of the form has no key for.
- label is the document's OWN wording for it, copied as printed. value is what is printed
  against that label.
- Do NOT repeat anything that belongs to another part of this form: the document type and
  number, the dates, either party's name, tax ID, branch or address, the charges table, the
  totals, or the payment lines. Those have their own fields and are already filled.
- Where nothing is left over, return an empty list.""",
    },
)

# --------------------------------------------------------------------------
# pass 3: advisory review
# --------------------------------------------------------------------------

# Extracted fields in, a list of issues out: {"issues": [{field, problem,
# severity}], "notes": ""}. Extracts no document values of its own.
#
# `verify.py` has already checked the arithmetic and its results go into the
# message, so this is asked only for the judgements arithmetic cannot make -- a
# value in the wrong field, an identifier of implausible shape. Advisory.
VERIFY_PROMPT = """You are auditing fields extracted from a Thai/English business document.
The arithmetic has ALREADY been checked by an exact calculator, and its results are given
to you below. Do not redo the sums and do not contradict them -- they are authoritative.

Your job is only what arithmetic cannot decide. Look for:
- values that landed in the wrong field (a date in a document number, an address in a name)
- identifiers with an implausible shape (a Thai tax ID is 13 digits, dates should be real dates)
- a line item whose description does not match its amount's magnitude
- a field left empty that the other fields contradict (a VAT total with no rate)
- anything that looks transcribed wrong rather than calculated wrong

An empty field is not itself a problem: it means the document does not state that
value, which is the correct answer. Report problems only -- never supply a value,
and never suggest what an empty field "should" contain. You cannot see the page.

Some documents price VAT-inclusive and some price VAT-exclusive; which one this is
appears below the fields, and it has already been taken into account by the
calculator. Do not raise an issue that assumes the other basis -- on a VAT-inclusive
document the line amounts are SUPPOSED to exceed the goods total and to add up to the
grand total instead, and that is not an error.

Return ONLY a JSON object, no prose and no code fence:

{
  "issues": [ { "field": "", "problem": "", "severity": "error" } ],
  "notes": ""
}

severity is "error" (certainly wrong), "warning" (suspicious) or "info".
Return an empty issues list if the fields look sound. Never invent a problem to
have something to report.

Extracted fields:
"""
