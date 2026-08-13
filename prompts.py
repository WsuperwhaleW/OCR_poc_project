"""Every prompt the app sends to the model, and nothing else.

Split out of `app.py` so the text can be read, edited and diffed without the
request-assembly code around it. There is no logic here: these are string
constants, imported by `app.py` and interpolated into request bodies exactly as
written.

Each one was arrived at by measurement rather than by taste, and the comments
say what was measured -- a prompt that looks redundant or oddly worded is
usually the fix for a failure that shows up as HTTP 200 with quietly wrong
output. Re-run the benchmark (`python compare.py`) after editing any of them.

* `PROMPT`            -- pass 1, page image in, verbatim transcript out.
* `EXTRACT_PROMPT`    -- pass 2, transcript in as text, structured JSON out.
* `EXTRACT_REMINDER`  -- closes pass 2's message, after the transcript.
* `VERIFY_PROMPT`     -- pass 3, the model's advisory review of the extracted
                         fields, once the arithmetic has already been checked.
"""

# --------------------------------------------------------------------------
# pass 1: OCR
# --------------------------------------------------------------------------

# Verbatim transcription. Deliberately NOT typhoon-ocr's prompt: that one asks the
# model to *describe* figures in Thai, which produces invented prose and translated
# content instead of a faithful transcript.
PROMPT = """Transcribe every piece of text in this document image, exactly as it appears.

LANGUAGE
- Do NOT translate anything. Reproduce each word in the language it is printed in.
- Thai text stays Thai. English text stays English. The document contains only Thai
  and English -- never output any other language or script.
- Do NOT transliterate, romanise, correct spelling, or modernise wording.

CONTENT
- Output ONLY text that is actually visible in the image. Invent nothing.
- Do NOT summarise, paraphrase, explain, comment, or add headings of your own.
- Do NOT describe logos, photos, charts, stamps, signatures or diagrams. Skip them.
- Copy numbers, dates, codes and amounts exactly as printed, keeping the original
  digits, separators and punctuation.
- If a character is genuinely illegible, transcribe your closest reading of it.
  Never guess at whole words that are not there.

ORDER -- this matters
- Transcribe in the order the text physically appears on the page: strictly top to
  bottom, and left to right within the same line or band.
- Text positioned higher on the page MUST appear earlier in your output. Never move a
  heading, title, total or page number away from where it sits on the page.
- For multi-column layouts, finish the left column before starting the right one.

FORMATTING -- keep it minimal
- Output plain text. Preserve the original line breaks, and the spacing or indentation
  that separates columns and aligns values.
- Do NOT use Markdown: no #, *, -, backticks, code fences, bold or italics.
- Do NOT wrap anything in <figure>, <page_number>, or any other tag.
- Tables are the ONE structure to preserve, and they must be HTML. Never use Markdown
  pipe tables.
- If the page has an itemised grid -- rows of entries under column headings, such as
  description / period / amount / VAT / total on an invoice or receipt -- you MUST
  output it as a <table>. Never flatten a grid into one value per line.
- Each row of the grid is ONE <tr>, with one <td> per column, in left-to-right order.
  Where a heading stacks Thai above English (จํานวนเงิน over Gross Amount), that is a
  single cell: put both in one <td>.
- Do NOT put headings, addresses, phone numbers, customer details, totals in words,
  signatures or free-standing paragraphs inside a table. Those are plain text.
- Never wrap the whole page in a table. Never use <caption>, <thead> or <tbody>.
  A page normally reads: plain text, then the table, then more plain text.
- Write tables in exactly this shape, with no styles, classes or attributes:
  <table>
  <tr><td>ไตรมาส</td><td>ยอดขาย</td></tr>
  <tr><td>Q1</td><td>1,200,000</td></tr>
  </table>
  Keep every cell in its original row and column. Preserve blank cells as <td></td> so
  the columns stay aligned. Use colspan/rowspan only where cells are genuinely merged.
- Checkboxes are text: ☐ for unchecked, ☑ for checked."""

# --------------------------------------------------------------------------
# pass 2: field extraction
# --------------------------------------------------------------------------

# Fed the finished transcript back as text, so there is no image to prefill and
# it costs a fraction of the OCR run. Two rules in here are load-bearing and
# look like style until they are removed:
#
# * No example value anywhere is written in quotes. A quoted illustration reads
#   as JSON to a small model -- with `// e.g. "INVOICE"` in the skeleton
#   comments, replies came back carrying a literal "INVOICE": "" key alongside
#   the real document_type. Every illustration is unquoted prose, and one rule
#   says outright that nothing in the instructions is itself a key or a value.
# * The skeleton stays a JSON skeleton. Listing the keys as prose instead was
#   tried and is worse -- the model answered {"keys": "document_type", ...} and
#   then looped.
EXTRACT_PROMPT = """You are reading the text of a Thai/English business document that has
already been transcribed. Extract its key business fields as JSON.

Return ONLY a JSON object, no prose and no code fence. Use exactly these keys:

{
  "document_type": "",        // as printed at the head of the page
  "document_number": "",
  "issue_date": "",
  "due_date": "",
  "reference_document": "",   // another document this one cites
  "po_number": "",
  "original_invoice_number": "",   // the invoice a credit/debit note corrects
  "contract_number": "",
  "customer_code": "",
  "location_code": "",        // site / branch / meter code the charge belongs to
  "service_period": "",       // the period the whole document covers
  "seller_name": "",          // the company name only
  "seller_tax_id": "",
  "seller_branch": "",        // the branch shown beside the seller's tax ID
  "seller_address": "",
  "buyer_name": "",           // the company name only
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
  "subtotal": "",             // the total BEFORE VAT
  "vat_total": "",
  "amount_incl_vat": "",      // the total INCLUDING VAT, before withholding
  "withholding_tax_total": "",
  "net_payable": "",          // what is left to pay after withholding is deducted
  "amount_in_words": "",
  "payment_method": "",
  "payment_reference": "",    // cheque or transfer number for the payment
  "other_fields": [ { "label": "", "value": "" } ]
}

Work through the document once, in this order, and fill the keys as you reach them:
its heading and number, the block naming who issued it, the block naming who receives
it, the charges table, then the totals and payment lines at the foot.

General rules:
- Copy values EXACTLY as they appear. Do not reformat, do not translate, and do not
  convert Thai text to English.
- Every value you write must appear, character for character, in the document text
  below. If you cannot point at it there, the field is not yours to fill.
- Use "" for anything the document does not state. Never guess or invent a value.
  An empty field is correct and useful; a plausible invented one is not.
- Never fill a field with a placeholder -- not a row of zeros, not a dash, not a repeat
  of a value you already used elsewhere. If the page does not state it, the answer is "".
- Output exactly the keys listed above and no others. Nothing written in these
  instructions is itself a key or a value: where a rule below names a label to look for,
  that is a hint for finding the figure on the page, never something to emit.
- The skeleton above is the shape of the answer, not the answer. Returning it unchanged,
  with every value still "", is wrong for any document that states anything at all.
- Do not fill a field by inference from the others: not the buyer's tax ID from the
  buyer's name, not a due date from an issue date, not a currency the page never
  names. Leave it "".
- Put every remaining fact that matters -- meter readings, page numbers, order numbers
  the keys above do not cover -- into "other_fields" with the document's own label.
- "line_items" holds only the rows of the charges table. Summary rows such as Total: VAT,
  Total: Non VAT or รวมเงิน are NOT line items; their figures belong in the totals fields
  below.

Buyer and seller -- four separate fields per party, and each holds one thing only:
- The name field takes the company name and nothing else, on ONE line. Stop at the end of
  the name. If what you are about to write contains a street, a postcode, a telephone
  number, a tax ID, a date or a line break, you have taken too much of the block -- cut
  it back to the name alone.
- The tax ID field takes only the digits printed after เลขประจำตัวผู้เสียภาษี or Tax ID.
- The branch field takes what is printed beside that tax ID -- the Thai word for head
  office, or a branch number. Copy it as printed; do not translate it and do not turn a
  word into a number.
- The address field takes the street lines and postcode, without the name, the tax ID or
  the branch repeated inside it.
- Never write the same text into two of these fields. The seller's block is usually the
  letterhead at the top; the buyer's is the block labelled ลูกค้า or ผู้ซื้อ.

References -- each key takes ONE identifier, and only when the page labels it as that:
- These keys are empty on most documents, and that is the expected answer. Do not reuse
  the document's own number to fill them, and never write one value into several of them.
  A page that prints only an invoice number gives document_number that number and leaves
  po_number, contract_number, customer_code and the rest "".
- A bare number with no label is not a PO number. If a reference is labelled with
  something these keys do not cover, it belongs in "other_fields", not forced into one
  of these.
- service_period is the period the document as a whole covers, printed once near the head
  of the page. A period printed on one row of the charges table is that row's period and
  belongs in that row instead.

Reading numbers -- take particular care here:
- Keep the digits exactly as printed, including thousands separators and both decimal
  places. Do not round, re-space or "tidy" them.
- Read across the row. A figure belongs to the column it sits under; never shift a
  value into a neighbouring column to make a row look complete.
- A cell showing "-", "–" or blank means nil. Write it as "-" or "", not as "0",
  unless the document actually prints 0.00.
- Some Thai forms rule the money column into baht and satang. "9,741 60" under such a
  column is ONE amount, 9,741.60 -- write it as "9,741.60". Two figures separated by a
  wide gap in DIFFERENT columns are two separate amounts; keep them apart.
- A figure in brackets, or with a leading "-", is negative. Preserve the sign.
- Some pages price VAT-inclusive: the money column is headed จำนวนเงิน (รวมภาษี) or
  ราคารวมภาษีมูลค่าเพิ่ม or Amount (incl. VAT), and every figure under it already
  contains the tax. Others price VAT-exclusive and add the tax below. Either way,
  copy each figure exactly as printed. Never take VAT out of an inclusive figure and
  never add it to an exclusive one, and do not adjust any figure to make a column
  add up. Where the page states which basis it uses -- in that column heading or in a
  note such as ราคานี้รวมภาษีมูลค่าเพิ่มแล้ว -- copy that wording into "other_fields"
  under its own label, so the reader knows which figures carry tax.
- Do not compute anything. If the document does not print a subtotal, leave it "";
  never add the rows up yourself.
- Copy every line of the charges table once, including rows whose amount is nil. A
  missing row is the most damaging error you can make here -- and a row written twice
  is the second most damaging, so do not revisit a row you have already written.

The totals -- four separate figures, each read off its own printed line:
- subtotal is the figure before VAT, labelled รวมเป็นเงิน or มูลค่าสินค้า/บริการ or
  รวมราคาสินค้า or Sub Total.
- amount_incl_vat is the figure after VAT is added, labelled จำนวนเงินรวมทั้งสิ้น or
  รวมเงินทั้งสิ้น or Total Amount. It is the one the amount in words normally spells out.
- Take each of these from its own label, not from its position. A VAT-inclusive page
  often prints them in the opposite order -- the grand total first, then the VAT, then
  the goods total below both -- and the top line there is amount_incl_vat, not subtotal.
- net_payable is what remains after withholding is deducted, labelled ยอดชำระสุทธิ or
  คงเหลือ or Net Payable or Amount Due. On a document with no withholding this line is
  often absent -- then leave net_payable "".
- Fill only the lines the page actually prints. If it shows a single grand total, decide
  from its label and from whether withholding was deducted which of the two it is, and put
  it there alone. Do not copy one figure into both fields, and never subtract or add to
  produce the other.

VAT rate:
- Look for a stated rate, printed after อัตราภาษีมูลค่าเพิ่ม or อัตราร้อยละ or VAT.
- Put just the number in vat_rate, written the way the page writes it: a page showing 7%
  gives 7, and a page showing 7.00% gives 7.00. Never convert it to a fraction such as
  0.07, and never keep the per-cent sign.
- If the document is zero-rated, that is 0. If no rate is stated anywhere, leave it ""
  -- do not assume one, and do not derive it by dividing the totals.

Document text:
"""

# Repeated after the transcript, and this is load-bearing rather than belt and
# braces. typhoon-ocr is an OCR fine-tune first: once a long transcript sits
# between the instructions and the point of generation, its training wins and it
# answers with its own {"natural_text": "<the whole page>"} envelope instead of
# the requested fields -- which then overruns the token cap mid-string and
# arrives as "Unterminated string starting at ... (char 17)", char 17 being the
# opening quote of that envelope. Measured on a 3,694-token prompt: instructions
# first alone gave the envelope every time; closing with this reminder gave the
# right schema every time. Short documents never showed it, so this only ever
# reproduced on the longer multi-page ones.
EXTRACT_REMINDER = """

Now return the JSON object described above, using exactly the keys listed.
Do NOT transcribe the document and do NOT return a "natural_text" field."""

# --------------------------------------------------------------------------
# pass 3: advisory review
# --------------------------------------------------------------------------

# The arithmetic is done in Python before this is sent, and the results go into
# the message with it. A 2B model is poor at addition, so letting it 'check' the
# sums would add noise and occasionally contradict a correct calculation; it is
# asked only for the judgements arithmetic cannot make, and its answer is
# advisory.
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
