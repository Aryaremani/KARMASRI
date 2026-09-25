"""
llm_structuring.py

LLM structuring pipeline: using models available via Ollama for structuring into JSON.

Prerequisites (LOCAL GPU setup — Windows):
    1. Install Ollama for Windows:
        https://ollama.com/download/windows
    2. Pull a model that fits your GPU's VRAM (8GB RTX 5050 -> 7B class model):
        ollama pull qwen2.5:7b
    3. Install Python dependencies:
        pip install ollama
        pip install json_repair

Note: Ollama automatically uses your NVIDIA GPU if available (confirmed via
`nvidia-smi` / `ollama ps` showing llama-server.exe with GPU memory usage).
No extra config needed for GPU — it's automatic.
"""

import os
import re
import ollama

try:
    # Standard PyPI package (`pip install json_repair`) exposes `repair_json`.
    from json_repair import repair_json as repair_llm_json
except ImportError:
    # Fallback in case you're using a custom/forked module that already
    # provides `repair_llm_json` directly.
    from json_repair import repair_llm_json

# ── LLM Extraction ────────────────────────────────────────────────────────────

BASE_PROMPT_TEMPLATE = '''Your task is to analyze the OCR text and extract the information into STRICT JSON format.

=== GENERAL RULES ===
1. Return ONLY valid JSON. No markdown, no explanations, no comments outside the JSON object.
2. Do not guess or invent values. If a value cannot be identified, return null.
3. Correct obvious OCR formatting errors only when the intended value is clear.
4. Extract every available bill, invoice, receipt, estimate, pharmacy, hospital,
   clinic, laboratory, or dental line item.
5. Dates should preferably be returned as YYYY-MM-DD when clearly identifiable.
6. If multiple possible bill numbers or dates exist, select the one associated
   with the main invoice/bill.
7. Never infer medical conditions or diagnoses unless explicitly written in the bill.

=== PATIENT NAME RULES ===
8. Name of the patient. On a standard hospital/clinic BILL or INVOICE, the patient's
   name commonly appears as "Name <patient name>" on the same line as, or immediately
   next to, "Bill No <number>" (e.g. "Name SREEDEVI Bill No 3193") — extract ONLY the
   name portion as patient_name, and the number portion as bill_no; do not merge them,
   drop the name, or leave patient_name null when it is present in this form (see
   Example F below).
8a. When the document is an ESTIMATE rather than a BILL/INVOICE, the patient's name
   may instead appear directly below or beside the word "Estimate" and before
   "Prepared By". Treat that name as patient_name when the layout clearly indicates
   it is the person receiving the estimate (see Example E below).

=== ITEM NAME RULES ===
9. Keep item_name CLEAN — the product/service name only. Strip out HSN codes, batch
   numbers, expiry dates, and location/bin codes even when jammed together with the
   name in the OCR text (e.g. "30049035 EMESET4MG INJ2ML A-38 S620043 05/25" becomes
   item_name "Emeset 4mg Injection 2ml" — nothing else).
10. NEVER treat a tooth number, tooth/area code, surface code, or other location
    identifier as a quantity or leave it attached to item_name. Dental/medical bills
    often have a "Tooth/Area" column with values like "23" or "11,12" — these identify
    WHERE a procedure was performed, not HOW MANY units. Only populate "quantity" when
    the source explicitly states a count (e.g. "Qty: 2", "x3", "2 vials").
10a. NEVER default a blank/missing quantity to 1. Many pharmacy/hospital bill rows
    have their Qty column blank or illegible for that specific row, even while other
    rows in the same table DO show a quantity — treat each row independently. Use the
    sanity check unit_price × quantity ≈ amount: if the OCR line only shows a rate and
    a final amount with no explicit count between them, and amount is NOT equal to
    unit_price (e.g. rate 12.90, amount 51.60 — a hidden quantity of 4), set quantity
    to null rather than guessing 1. Guessing 1 when the true quantity was blank/missing
    is just as wrong as inventing any other number (see Example G below).
10b. WATCH FOR A DATE FUSED WITH A STRAY DIGIT STRING (OCR artifact): a date in the
    OCR text is sometimes immediately followed, with NO space, by extra digits that
    do not belong to it — e.g. "25/10/202237" is really the date "25/10/2022" plus a
    stray "37" fused onto the end by an OCR line-merge error. NEVER read that trailing
    fused digit string as quantity. Use the sanity check: if treating it as quantity
    would make unit_price × quantity wildly exceed the line's actual amount (e.g.
    quantity 37 × rate 250.00 = 9250, but amount shown is only 250.00), that mismatch
    itself is the sign the digits are a date artifact, not a real quantity — set
    quantity to null instead (see Example H below).

=== AMOUNT, DECIMAL & QUANTITY RULES ===
11. Convert numeric amounts to numbers without currency symbols or commas.
12. Do not confuse MRP, unit price, discount, tax, quantity, and final line amount.
    The "amount" for each item is whatever final figure appears in the bill's own
    Fee/Amount column — never recompute it by multiplying unit_price by a
    tooth/area number, and never substitute a different column's value for it.
    Apply the decimal-point corrections in Rule 13 and the blank-column handling
    in Rule 14 to this figure where they apply — "use the Fee/Amount column's
    figure" and "fix an OCR-dropped decimal in that figure" are the same
    instruction, not competing ones.
13. DECIMAL POINT ERRORS (common OCR issue) — currency amounts always have exactly
    2 decimal digits. Watch for two patterns where the decimal got lost:
    (a) DROPPED ENTIRELY: a large whole number with NO decimal point at all next to
        amounts that clearly use "X,XXX.00" format (e.g. "500000" near other
        properly-formatted amounts) → insert the decimal 2 digits from the right:
        "500000" -> 5000.00. Never copy a neighboring number's decimal position.
    (b) REPLACED BY A SPACE: a scanned column gap read as whitespace instead of ".",
        producing two space-separated tokens where the second is exactly 2 digits
        (e.g. "5000 00") → join them: "5000 00" -> 5000.00.
    Only apply either fix when the source token has NO decimal point already. If a
    number already shows "20000.00", leave it untouched even if it looks large.
14. Watch for pharmacy rows like "MRP | Dis. | GST | Total" where Dis./GST are blank,
    so the OCR line only has TWO numbers (e.g. "14.53 29.06"). The LAST number is
    always the line "amount" — never copy MRP/unit_price into amount just because
    fewer numbers than expected appear, and never repurpose that last number into
    "discount" either; leave discount/tax null if genuinely blank.

=== TOTAL_AMOUNT RULES (highest-risk field — read carefully) ===
15. total_amount must represent the final/net amount payable whenever available.
16. NEVER confuse bill_no, patient/OP/IP numbers, or batch numbers with total_amount,
    even when they sit visually close to a "TOTAL" label due to OCR reading order.
    Only use a number as total_amount if it appears directly in the TOTAL/grand
    total column, formatted as currency (decimals like "1,234.00", or clearly in a
    totals row).
17. If the totals area is blank, cut off, illegible, or the text ends with "Continue"
    (or an OCR-garbled variant — "Contnue", "Continu", "Cont1nue", or any word
    starting with "Cont" near TOTAL/GRAND TOTAL) — set total_amount to null. Do not
    substitute a nearby unrelated number just because the word wasn't spelled right.
18. The true grand total sometimes appears as a bare, unlabeled number on its own
    line right after the last item (label lost to OCR). If a standalone
    currency-formatted number sits there and isn't already an item amount, treat it
    as total_amount — this does not override rule 16 (still never use an ID number).

=== DUPLICATE-SECTION RULE ===
19. Some bills show the same charges TWICE: once as category subtotals ("Bill
    Summary" — e.g. "Room & Nursing Charges: 2350.00") and again as a fully itemized
    breakdown ("Detailed Breakup") that sums to those same subtotals. These describe
    the SAME money. When both sections exist, extract items ONLY from the
    detailed/itemized section. If a bill has ONLY a summary section with no further
    breakdown, extract the summary rows as items instead — they're the only detail
    available.

=== FEW-SHOT EXAMPLES ===

Example A — dropped decimal + tooth code:
OCR TEXT:
"Implant-MegaGen AnyRidge 23   500000
Tooth: 23"
CORRECT ITEM OUTPUT:
{
  "item_name": "Implant-MegaGen AnyRidge",
  "quantity": null, "unit_price": null, "discount": null, "tax": null,
  "amount": 5000.00
}
(The "23" is a tooth code — stripped from item_name, never used as quantity.
"500000" had no decimal point — corrected to 5000.00.)

Example B — unlabeled total after last item:
OCR TEXT:
"CBC Test          450.00
Lipid Profile      650.00
1,100.00"
CORRECT OUTPUT (relevant fields):
{ "items": [
    {"item_name": "CBC Test", "amount": 450.00, ...},
    {"item_name": "Lipid Profile", "amount": 650.00, ...}
  ],
  "total_amount": 1100.00 }
(The bare "1,100.00" line has no "Total:" label but is the sum right after the
last item — treated as total_amount per rule 18.)

Example C — "Continue" near TOTAL (garbled OCR):
OCR TEXT:
"Sub Total   45,200.00
TOTAL   Contnue"
CORRECT OUTPUT (relevant field):
{ "total_amount": null }
(Word starting with "Cont" sits next to TOTAL — this is a garbled "Continue",
meaning the real total is on a page not captured. Do not use 45,200.00 or any
other nearby number.)

Example D — summary AND detailed sections both present:
OCR TEXT:
"Bill Summary
Room & Nursing Charges   2350.00
Professional Fees        1200.00

Detailed Breakup
Room Rent (3 days)        900.00
Nursing Charges           1450.00
Consultant Visit Fee      1200.00"
CORRECT OUTPUT: items = [Room Rent 900.00, Nursing Charges 1450.00,
Consultant Visit Fee 1200.00] — the Bill Summary rows are NOT added as
separate items; they are the same charges already captured in Detailed Breakup.

Example E — Estimate document, patient_name between "Estimate" and "Prepared By":
OCR TEXT:
"Estimate
Pradeep Kumar E IFS (21118)
Prepared By: Dr. Jensy George"
CORRECT OUTPUT (relevant field):
{ "patient_name": "Pradeep Kumar E IFS" }
(This is an Estimate, not a Bill/Invoice. The name sits between "Estimate" and
"Prepared By" — that is the patient, per rule 8. The "(21118)" code is not
part of the name.)

Example F — standard bill header, "Name X ... Bill No Y" on the same line:
OCR TEXT:
"Name SREEDEVI Bill No 3193
Address VASUMANA,PULLIKANAKKUPO Date 27/10/2022"
CORRECT OUTPUT (relevant fields):
{ "patient_name": "SREEDEVI", "bill_no": "3193" }
(This is a standard Bill/Invoice, not an Estimate — rule 8 applies, not 8a. "SREEDEVI"
is the patient name and "3193" is the bill number; neither is merged into the other,
and patient_name is NOT left null just because the layout is compact.)

Example G — blank quantity should stay null, never default to 1:
OCR TEXT:
"CHYMOLEXTAB 99556 25/10/2022 12.90 51.60"
CORRECT ITEM OUTPUT:
{
  "item_name": "CHYMOLEXTAB",
  "quantity": null, "unit_price": 12.90, "discount": null, "tax": null,
  "amount": 51.60
}
(No explicit count appears on this line — only a rate (12.90) and a final amount
(51.60). Since 51.60 is NOT equal to 12.90, a quantity was clearly involved (here,
4), but it isn't stated in the OCR text, so quantity is left null per rule 10a
rather than guessed as 1. The "amount" is still copied directly as 51.60 — do not
recompute it as unit_price × 1.)

Example H — stray digits fused onto a date are NOT a quantity:
OCR TEXT:
"MOPPING PAD 30 X 30 99554 25/10/202237 250.00 250.00"
CORRECT ITEM OUTPUT:
{
  "item_name": "MOPPING PAD 30 X 30",
  "quantity": null, "unit_price": 250.00, "discount": null, "tax": null,
  "amount": 250.00
}
(The date reads "25/10/2022" with a stray "37" fused onto the end — an OCR
line-merge artifact, not a quantity. Treating 37 as quantity would give
37 × 250.00 = 9250.00, wildly more than the actual amount of 250.00 shown on
the line — that mismatch is the giveaway per rule 10b. quantity is left null,
and amount is copied directly as the 250.00 that actually appears in the
Amount column.)

=== OUTPUT SCHEMA ===
Return JSON using exactly this structure. The "_reasoning" field comes first:
briefly note (1-3 short sentences) which total_amount source you used, whether
this is a standard Bill or an Estimate (and where you found patient_name either
way), whether any Continue/duplicate-section pattern applied, any decimal
correction made, and whether any item's quantity was left null because it
wasn't explicitly stated (rather than defaulted to 1). Keep it short — it will
be stripped before the data is used; its only purpose is to make you check
yourself before filling the fields below it.

{
  "_reasoning": null,
  "bill_no": null,
  "patient_name": null,
  "bill_date": null,
  "hospital_name": null,
  "doctor_name": null,
  "items": [
    {
      "item_name": null,
      "quantity": null,
      "unit_price": null,
      "discount": null,
      "tax": null,
      "amount": null
    }
  ],
  "subtotal": null,
  "taxable_value": null,
  "discount_amount": null,
  "tax_amount": null,
  "cgst_amount": null,
  "sgst_amount": null,
  "igst_amount": null,
  "total_amount": null,
  "currency": null
}

FIELD DEFINITIONS:
bill_no: Invoice number, bill number, receipt number, or invoice ID.
patient_name: Name of the patient (see rule 8 for the Estimate-specific case).
bill_date: Primary invoice/bill date, as YYYY-MM-DD when clearly identifiable.
hospital_name: Hospital, clinic, laboratory, or pharmacy name.
doctor_name: Doctor/consultant name if explicitly available.
items: Every individual medicine, lab test, procedure, consultation, room charge,
  service, consumable, or other billed item.
quantity: ONLY when explicitly stated as a count in a Units/Qty column of its own —
  never a tooth/area/surface code. Null if that column is blank/missing/illegible.
unit_price: Price per individual unit.
discount / tax: Line-level discount/tax when explicitly shown.
amount: Final amount of this particular line item.
subtotal: Amount before overall discounts/taxes when available.
taxable_value: Only fill if the bill explicitly labels a "Taxable Value" field —
  do not infer it by subtracting tax from total yourself.
discount_amount / tax_amount: Bill-level totals.
cgst_amount / sgst_amount / igst_amount: Only if explicitly shown as separate
  labeled figures.
total_amount: Final/net/grand amount payable.
currency: INR if the ₹ symbol or Indian bill format is present, else as identified.

If an item contains only item name and amount, return:
{
  "item_name": "CBC Test",
  "quantity": null, "unit_price": null, "discount": null, "tax": null,
  "amount": 450.00
}

=== FINAL REMINDER (most important) ===
Before you output total_amount, double-check it is not a bill number, patient/OP/IP
number, or batch number. If the true total is missing, cut off, or blocked by a
"Continue" (even garbled), output null rather than guessing. This single field
causes the most downstream damage when wrong.

OCR TEXT:
{{OCR_TEXT}}'''


# Reminder of the required output shape, included in the retry prompt below.
# RETRY_PROMPT_TEMPLATE deliberately does NOT repeat the full BASE_PROMPT
# rule set (that would defeat the point of a smaller, targeted retry ask),
# but omitting the schema entirely meant the model had to reproduce a
# nested JSON structure from memory on retry -- with nothing to check
# against, it's more likely to drop fields or drift format, independent of
# whether it fixes the original mismatch. This keeps the retry prompt
# targeted while still anchoring the exact field names/shape expected.
JSON_SCHEMA_REMINDER = '''Required JSON shape (keep this exact set of fields,
even for values you leave null — "_reasoning" included):
{
  "_reasoning": null,
  "bill_no": null, "patient_name": null, "bill_date": null,
  "hospital_name": null, "doctor_name": null,
  "items": [
    {"item_name": null, "quantity": null, "unit_price": null,
     "discount": null, "tax": null, "amount": null}
  ],
  "subtotal": null, "taxable_value": null, "discount_amount": null,
  "tax_amount": null, "cgst_amount": null, "sgst_amount": null,
  "igst_amount": null, "total_amount": null, "currency": null
}'''


RETRY_PROMPT_TEMPLATE = """
Your previous extraction did not pass validation: the sum of item amounts
({computed}) does not match the stated total_amount ({stated}), a difference
of {diff}.

Re-check your previous JSON output against the original OCR text below.
Common causes of this mismatch, in order of likelihood:
0. total_amount may have been extracted from the WRONG number entirely —
   e.g. a bill number, OP/IP number, or batch number that happens to sit
   near a "TOTAL" label, rather than an actual amount in the totals column.
   If the actual total column is blank/illegible/cut off in the OCR text
   (or the text ends with "Continue" suggesting a missing second page),
   set total_amount to null instead of reusing a nearby ID-style number.
1. A DROPPED DECIMAL POINT on one or more items — check every item's
   "amount" for a value that is roughly 10x, 100x, or 1000x too large
   compared to what the sum requires. This is the single most common cause
   of large mismatches like this one. If you find such a value, correct it
   by inserting the decimal point rather than removing/nulling the item.
2. An item amount was misread from a neighboring line (fix or remove it)
3. An item was missed entirely
4. The total_amount itself was misread
5. The real total is an UNLABELED number sitting on its own line right
   after the last item (no "Total:" text in front of it) — if you see a
   standalone currency-formatted number there that isn't an item amount,
   that is very likely the true total_amount, even without a label.

The diff of {diff} may itself be a clue: check whether diff divided by any
single item's amount comes out close to a round multiple like 9, 90, 99, or
999 — that pattern points directly at which item lost a decimal point and
by how much.

Return a corrected JSON object in the same format as before. If you cannot
confidently fix a specific item after checking for the decimal-point issue
above, set it to null rather than guessing.

{schema_reminder}

Previous JSON:
{previous_json}

Original OCR text:
{ocr_text}
"""


# Ollama defaults to a 2048-token CONTEXT window (num_ctx) if it isn't set
# explicitly -- this is separate from num_predict, which only caps the
# OUTPUT length. Our instruction block alone (rules + JSON schema + field
# definitions) is already ~2,500 tokens BEFORE the OCR text is appended at
# the end of the prompt, so with the default num_ctx the model's input gets
# silently truncated and it never actually sees the bill's OCR text -- it
# then correctly (per rule 2, "if unknown return null") returns null for
# every field, producing a valid-but-empty JSON object instead of an error.
# We size num_ctx off the real prompt length (roughly 1 token per 3 chars,
# generously rounded up) so this scales automatically for long multi-page
# bills instead of silently breaking again at some new size.
_MIN_NUM_CTX = 8192

# num_predict was previously a flat 2048 -- fine for a handful of line items,
# but a 30+ item pharmacy bill needs the model to write a LOT of JSON (one
# object per item, plus the header and trailing total/tax fields at the very
# END of the schema). If num_predict runs out first, generation gets cut off
# mid-item -- the item list is left incomplete, and every trailing field
# (subtotal, tax_amount, total_amount, currency...) never gets written at
# all, since they come after "items" in the schema. json_repair then has no
# choice but to drop the dangling incomplete item and close the JSON early,
# silently producing an object with no total_amount -- indistinguishable
# from a bill that genuinely has no total, unless you look closely at
# whether the LAST item is also missing fields (a truncation tell).
_MIN_NUM_PREDICT = 3072
# Calibrated against a real truncated response: qwen2.5:7b's pretty-printed
# JSON runs ~150-160 chars (~50-55 tokens) per item once you account for
# the multi-line field-per-line style it tends to use, not the compact
# single-line estimate a naive char-count would suggest. We pad generously
# on top of that since BPE tokenization of numbers/punctuation is less
# efficient than plain text, and a too-small budget silently truncates the
# tail of the JSON (see the WARNING check after the first-pass call below).
_TOKENS_PER_LINE = 70
_TRAILING_FIELDS_BUDGET = 500  # header fields + subtotal/tax/total/currency
_SAFETY_MULTIPLIER = 1.3
_MAX_NUM_PREDICT = 8192


def _estimate_num_predict(ocr_text):
    # Each OCR line is a rough proxy for "one more thing the model has to
    # transcribe into JSON" (a line item, a header field, etc.) -- not
    # exact, but far better than a flat constant that silently breaks once
    # a bill has enough rows.
    line_count = ocr_text.count("\n") + 1
    needed = int(((line_count * _TOKENS_PER_LINE) + _TRAILING_FIELDS_BUDGET) * _SAFETY_MULTIPLIER)
    return max(_MIN_NUM_PREDICT, min(_MAX_NUM_PREDICT, needed))


def _estimate_num_ctx(prompt, num_predict):
    approx_input_tokens = len(prompt) // 3  # generous: ~3 chars/token
    # Leave headroom for the ACTUAL num_predict being used (not a guess),
    # plus the system message.
    needed = approx_input_tokens + num_predict + 128
    return max(_MIN_NUM_CTX, needed)


def _call_ollama(client, model, prompt, ocr_text_for_sizing=""):
    num_predict = _estimate_num_predict(ocr_text_for_sizing or prompt)
    num_ctx = _estimate_num_ctx(prompt, num_predict)
    print(f"  Calling Ollama: prompt is {len(prompt)} chars (~{len(prompt)//3} tokens), "
          f"num_ctx={num_ctx}, num_predict={num_predict}")

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Output only complete valid JSON. Double quotes only. No newlines in keys. No <think> tags."},
            {"role": "user", "content": prompt}
        ],
        options={
            "num_predict": num_predict,
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        format="json"
    )
    return response["message"]["content"]


# ── Reasoning-scratchpad cleanup ────────────────────────────────────────────
#
# "_reasoning" in the schema exists purely to make the model think before
# filling the real fields (structured-reasoning technique). It's not part
# of the bill data and must never reach validate_total(),
# validate_line_items(), calculate_extraction_reliability() in backend.py,
# or the final returned result.

def strip_reasoning_field(result):
    if result and "_reasoning" in result:
        result.pop("_reasoning")
    return result


# ── Deterministic dropped-decimal fix ──────────────────────────────────────
#
# Rule 13 in the prompt asks the model to reinsert a decimal point OCR
# dropped from a currency amount (e.g. "500000" in the raw text really means
# 5000.00). In practice the model doesn't always follow that instruction
# reliably. Rather than depending purely on prompt compliance, cross-check
# every extracted amount/unit_price against the raw OCR text directly: if a
# field's value exactly matches a standalone run of 5+ digits in the OCR
# text with no decimal point or comma touching it, that's almost certainly
# the dropped-decimal case, and we fix it in code.
#
# Two OCR patterns produce this bug, depending on how PaddleOCR grouped the
# bounding boxes for a visually-gapped rupees/paise amount (e.g. "5000 00"
# on the scanned bill):
#   1. Fully joined — the gap disappears entirely and OCR emits "500000"
#      as one solid digit run.
#   2. Space-separated — OCR keeps them as two tokens, "5000" and "00", but
#      the LLM then wrongly concatenates them into 500000 itself.
# Both end up as the same wrong value in the model's output, so we detect
# candidates for both patterns and treat them the same way once found.

_DROPPED_DECIMAL_RE = re.compile(r'(?<![\d.,])(\d{5,})(?![\d.])')
_SPACED_DECIMAL_RE = re.compile(r'(?<![\d.,])(\d{2,6})[ \t]+(\d{2})(?![\d.])')


def _find_dropped_decimal_candidates(ocr_text):
    """Values that are almost certainly a currency amount with a dropped
    decimal point — either fully joined in the OCR text ("500000") or split
    by a gap that the model might wrongly re-join ("5000" + "00" -> 500000)."""
    candidates = {int(m.group(1)) for m in _DROPPED_DECIMAL_RE.finditer(ocr_text)}
    for m in _SPACED_DECIMAL_RE.finditer(ocr_text):
        joined = int(m.group(1) + m.group(2))
        if joined >= 10000:  # only currency-scale slips are worth flagging
            candidates.add(joined)
    return candidates


def fix_dropped_decimals(result, ocr_text):
    """Safety net for rule 13. Divide any item amount/unit_price (or bill-
    level monetary field) by 100 if its value exactly matches a standalone,
    decimal-less digit run found in the raw OCR text — e.g. an extracted
    amount of 500000 that also appears verbatim as "500000" in the OCR text
    gets corrected to 5000.00. Only touches fields whose value matches a
    real OCR token, so it won't touch numbers the model already got right.
    """
    if not result or not ocr_text:
        return result

    candidates = _find_dropped_decimal_candidates(ocr_text)
    if not candidates:
        return result

    def maybe_fix(value):
        if value is None:
            return value
        try:
            if float(value).is_integer() and int(value) in candidates:
                return round(int(value) / 100, 2)
        except (TypeError, ValueError):
            pass
        return value

    for item in result.get("items") or []:
        if "amount" in item:
            item["amount"] = maybe_fix(item["amount"])
        if "unit_price" in item:
            item["unit_price"] = maybe_fix(item["unit_price"])

    for field in ("subtotal", "taxable_value", "discount_amount", "tax_amount",
                  "cgst_amount", "sgst_amount", "igst_amount", "total_amount"):
        if field in result:
            result[field] = maybe_fix(result[field])

    return result


# ── Deterministic amount/discount-swap fix (safety net for rule 14) ───────
#
# Even with rule 14 in the prompt, a small model doesn't always get this
# right: it sometimes still copies unit_price straight into amount, and
# stashes the REAL line total in "discount" instead (which is nonsensical --
# there's no discount column value on these bills at all). The telltale
# signature is specific enough to catch deterministically: amount is an
# exact copy of unit_price, and "discount" holds a value that's a clean
# near-integer multiple of unit_price (i.e. unit_price x quantity) rather
# than a plausible discount. When that signature shows up, swap them back.

def fix_amount_discount_swap(result):
    if not result:
        return result

    for item in result.get("items") or []:
        amount, unit_price, discount = item.get("amount"), item.get("unit_price"), item.get("discount")
        if amount is None or unit_price is None or discount is None:
            continue
        try:
            amount_f, unit_price_f, discount_f = float(amount), float(unit_price), float(discount)
        except (TypeError, ValueError):
            continue
        if unit_price_f <= 0 or discount_f <= unit_price_f:
            continue
        if amount_f != unit_price_f:
            continue  # amount doesn't match the "copied from unit_price" signature
        ratio = discount_f / unit_price_f
        if abs(ratio - round(ratio)) < 0.02:  # discount is ~an integer multiple of unit_price
            item["amount"] = round(discount_f, 2)
            item["discount"] = None

    return result


# ── Trailing tooth/area code cleanup (safety net for rule 10) ─────────────
#
# The prompt tells the model to keep item_name clean (product/service name
# only, no codes), but a small model doesn't always strip a trailing
# tooth/area/surface number even when it correctly leaves it OUT of
# quantity. The result is a technically-correct amount with a leftover
# digit stuck on the end of the name, e.g. "Implant-MegaGen AnyRidge 23"
# instead of "Implant-MegaGen AnyRidge". Strip it in code as a final pass,
# since this is a simple, low-risk text cleanup rather than a judgment call
# the model needs to make.
#
# Matches a trailing whitespace/hyphen followed by one or more short
# (1-2 digit) numbers, optionally comma/hyphen-separated (covers single
# teeth like "23", multi-tooth spans like "11,12" or "23-24"). Deliberately
# narrow: this must NOT match things like "2.5ML" or "5ml" that are part of
# a real product name, so it only fires on a BARE trailing number with no
# unit/letters attached.

_TRAILING_TOOTH_CODE_RE = re.compile(
    r'[\s\-]+(\d{1,2}(?:[,\-]\d{1,2})*)\s*$'
)


def strip_trailing_tooth_code(result):
    if not result:
        return result

    for item in result.get("items") or []:
        name = item.get("item_name")
        if not name:
            continue
        match = _TRAILING_TOOTH_CODE_RE.search(name)
        if match:
            cleaned = name[:match.start()].rstrip(" -")
            if cleaned:  # never blank out a name entirely
                item["item_name"] = cleaned

    return result


# ── Deterministic "Continue" total-suppression (safety net for rule 17) ───
#
# Rule 17 tells the model to null out total_amount when the bill's own text
# shows the totals continue onto another page -- but a small model doesn't
# reliably catch OCR-garbled spellings of "Continue" ("Contnue", "Continu",
# etc.), and worse, tends to invent a plausible-looking number instead of
# admitting it doesn't know (different fabricated values across repeated
# runs on the same bill are a giveaway of this). Back this up in code: scan
# the raw OCR text directly for "TOTAL" followed shortly by any word
# starting with "Cont", and force total_amount to null if found, regardless
# of what the model returned.

_CONTINUE_NEAR_TOTAL_RE = re.compile(r'\bTOTAL\b[^\n]{0,25}\bCont\w*', re.IGNORECASE)


def strip_total_if_continued(result, ocr_text):
    if not result or not ocr_text:
        return result
    if _CONTINUE_NEAR_TOTAL_RE.search(ocr_text) and result.get("total_amount") is not None:
        print("  'TOTAL ... Continue' pattern found in OCR text — forcing total_amount to null "
              "(the real total is on a page that wasn't extracted).")
        result["total_amount"] = None
    return result


# ── Deterministic multi-page-indicator total suppression ──────────────────
#
# Some multi-page bills don't say "Continue" near TOTAL at all -- instead
# the ONLY signal that more pages exist is a page-count marker printed
# elsewhere on the page, e.g. "Page 1 of 2". strip_total_if_continued()
# above can't catch this case since there's no "Continue" text anywhere in
# the OCR output. When a document is explicitly on an earlier page than its
# stated total, the grand total is very likely on a later page that wasn't
# captured -- but ONLY act on this after the retry-on-mismatch logic has
# already tried and failed to reconcile items with total_amount, since a
# bill can legitimately restate its running/page subtotal on every page
# without that being wrong. Nulling total_amount just because a bill says
# "Page 1 of 2" -- even when the numbers already add up cleanly -- would be
# an overcorrection.

_MULTI_PAGE_RE = re.compile(r'\bPage\s*(\d+)\s*of\s*(\d+)\b', re.IGNORECASE)


def strip_total_if_incomplete_page(result, ocr_text):
    if not result or not ocr_text:
        return result
    match = _MULTI_PAGE_RE.search(ocr_text)
    if match:
        current_page, total_pages = int(match.group(1)), int(match.group(2))
        if current_page < total_pages and result.get("total_amount") is not None:
            print(f"  'Page {current_page} of {total_pages}' found and items still don't "
                  f"reconcile with total_amount after retry — this isn't the last page, "
                  f"forcing total_amount to null (real total is likely on a later page "
                  f"that wasn't extracted).")
            result["total_amount"] = None
    return result


def extract_with_llm(ocr_text, model="qwen2.5:7b", ollama_host=None, retry_on_mismatch=True):
    """
    Defaults to localhost:11434 — i.e. Ollama running LOCALLY on this same
    machine (your Legion's GPU). Only pass ollama_host or set OLLAMA_HOST
    if you want to point at a remote server (e.g. a DGX) instead.

    If retry_on_mismatch is True, and the first pass's items don't sum to
    the stated total_amount, we give the model one more chance with the
    validation error shown to it — this catches a meaningful fraction of
    misread line-item amounts without a full re-architecture.
    """
    client = ollama.Client(host=ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    # Plain substitution — do NOT use str.format() here, since the JSON
    # skeleton embedded in the template contains unescaped { } braces that
    # .format() would try to parse as replacement fields.
    prompt = BASE_PROMPT_TEMPLATE.replace("{{OCR_TEXT}}", ocr_text)
    raw_output = _call_ollama(client, model, prompt, ocr_text_for_sizing=ocr_text)
    result = repair_llm_json(raw_output)

    if result is None:
        print("Could not parse JSON. Raw model output:\n", raw_output)
        return None

    # A truncated response (ran out of num_predict before the model finished
    # writing the JSON) is easy to mistake for "the bill just doesn't have a
    # total" -- both end up with total_amount missing. Distinguish them: if
    # the LAST item is missing fields that every other item has, generation
    # was cut off mid-item, not a legitimate absence of a total.
    if result and result.get("items"):
        last_item = result["items"][-1]
        if any(k not in last_item for k in ("quantity", "unit_price", "discount", "tax", "amount")):
            print(
                "  WARNING: the last item in the response is missing fields other items have "
                "-- this looks like the model ran out of output budget (num_predict) mid-item, "
                "not a bill that genuinely lacks a total. Consider raising _TOKENS_PER_LINE or "
                "_MIN_NUM_PREDICT if this keeps happening."
            )

    result = strip_reasoning_field(result)
    result = fix_dropped_decimals(result, ocr_text)
    result = fix_amount_discount_swap(result)
    result = strip_trailing_tooth_code(result)
    result = strip_total_if_continued(result, ocr_text)

    if retry_on_mismatch:
        validation = validate_total(result)
        if validation["checked"] and not validation["match"]:
            print(
                f"Totals mismatch (computed={validation['computed']}, "
                f"stated={validation['stated']}, diff={validation['diff']}) — retrying once..."
            )
            retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                computed=validation["computed"],
                stated=validation["stated"],
                diff=validation["diff"],
                schema_reminder=JSON_SCHEMA_REMINDER,
                previous_json=result,
                ocr_text=ocr_text,
            )
            retry_output = _call_ollama(client, model, retry_prompt, ocr_text_for_sizing=ocr_text)
            retry_result = repair_llm_json(retry_output)
            if retry_result is not None:
                retry_result = strip_reasoning_field(retry_result)
                retry_result = fix_dropped_decimals(retry_result, ocr_text)
                retry_result = fix_amount_discount_swap(retry_result)
                retry_result = strip_trailing_tooth_code(retry_result)
                retry_result = strip_total_if_continued(retry_result, ocr_text)
                # Re-validate the retry — only accept it if it actually
                # improved the mismatch, otherwise a bad retry could silently
                # replace a "close enough" first answer with a worse one.
                retry_validation = validate_total(retry_result)
                if not retry_validation["checked"] or retry_validation["match"] or \
                   retry_validation["diff"] < validation["diff"]:
                    result = retry_result
                else:
                    print("Retry did not improve the mismatch, keeping first-pass result.")
            else:
                print("Retry produced unparseable JSON, keeping first-pass result.")

            # After the retry attempt (successful or not), check whether a
            # mismatch still remains. If so, and the document itself says
            # this isn't the last page, the honest answer is that we don't
            # know the real total -- null it out rather than keep a number
            # that's already been shown not to reconcile.
            final_validation = validate_total(result)
            if final_validation["checked"] and not final_validation["match"]:
                result = strip_total_if_incomplete_page(result, ocr_text)

    result = validate_line_items(result)
    result = add_computed_subtotal(result)

    return result

# ── Validation ────────────────────────────────────────────────────────────────

def validate_total(result):
    """Return a structured validation dict instead of just printing.

    { "checked": bool, "computed": float|None, "stated": float|None,
      "match": bool|None, "diff": float|None }
    """
    out = {"checked": False, "computed": None, "stated": None, "match": None, "diff": None}
    if not result or not result.get("items") or not result.get("total_amount"):
        return out
    try:
        computed = sum(
            float(item["amount"])
            for item in result["items"]
            if item.get("amount") is not None
        )
        stated = float(result["total_amount"])
        diff = abs(computed - stated)
        out.update({
            "checked": True,
            "computed": round(computed, 2),
            "stated": round(stated, 2),
            "match": diff <= 0.10,
            "diff": round(diff, 2),
        })
    except (ValueError, TypeError):
        pass
    return out


# ── Per-line-item validation (validate_total only checks the grand total) ──
#
# validate_total() catches when the SUM of items doesn't match total_amount,
# but a bill can pass that check while still having individual rows that are
# wrong in ways that happen to cancel out, or that simply have no bearing on
# the total at all if total_amount is null. This does NOT try to guess which
# field is wrong or silently correct anything (unlike fix_dropped_decimals/
# fix_amount_discount_swap, which only act on narrow, well-understood
# signatures) -- it just surfaces likely errors for a human reviewer, the
# same spirit as validate_total().

_LINE_ITEM_FLAT_TOLERANCE = 2.00  # rupees


def validate_line_items(result):
    """Adds result["line_item_warnings"]: a list of rows where quantity x
    unit_price doesn't reasonably reconcile with amount. Empty list if every
    checkable row reconciles (or no rows have enough fields to check).
    """
    if not result:
        return result

    warnings = []
    for idx, item in enumerate(result.get("items") or []):
        qty, price, amount = item.get("quantity"), item.get("unit_price"), item.get("amount")
        if qty is None or price is None or amount is None:
            continue  # nothing to cross-check without all three
        try:
            qty_f, price_f, amount_f = float(qty), float(price), float(amount)
        except (TypeError, ValueError):
            continue
        if qty_f <= 0 or price_f <= 0:
            continue
        expected = qty_f * price_f
        diff = abs(expected - amount_f)
        # Allow whichever is larger of a flat Rs.2 or 3% of the expected
        # value, since small legitimate per-line rounding happens.
        tolerance = max(_LINE_ITEM_FLAT_TOLERANCE, expected * 0.03)
        if diff > tolerance:
            warnings.append({
                "index": idx,
                "item_name": item.get("item_name"),
                "quantity": qty_f,
                "unit_price": price_f,
                "expected": round(expected, 2),
                "actual": round(amount_f, 2),
                "diff": round(diff, 2),
            })

    if warnings:
        print(f"  {len(warnings)} line item(s) don't reconcile (quantity x unit_price vs amount):")
        for w in warnings:
            print(f"    [{w['index']}] {w['item_name']!r}: {w['quantity']} x {w['unit_price']} "
                  f"= {w['expected']} but amount is {w['actual']} (diff {w['diff']})")

    result["line_item_warnings"] = warnings
    return result


# ── Computed subtotal fallback ─────────────────────────────────────────────
#
# When total_amount ends up null (e.g. strip_total_if_continued /
# strip_total_if_incomplete_page correctly refuse to guess at a total that
# isn't on the page), a downstream consumer is left with no total at all.
# This attaches the sum of whatever items WERE successfully extracted as a
# clearly-separate "computed_subtotal" field -- never written into
# total_amount itself, since it's not a substitute for the bill's real
# total, just the best available fallback figure for review/display. This
# also gives backend.py's calculate_extraction_reliability() a fallback
# signal to use when total_amount is null.

def add_computed_subtotal(result):
    if not result:
        return result
    try:
        computed = sum(
            float(item["amount"])
            for item in (result.get("items") or [])
            if item.get("amount") is not None
        )
        result["computed_subtotal"] = round(computed, 2)
    except (TypeError, ValueError):
        result["computed_subtotal"] = None
    return result
