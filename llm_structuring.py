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

IMPORTANT RULES:

1. Return ONLY valid JSON.

2. Do not include markdown, explanations, or comments.

3. Do not guess or invent values.

4. If a value cannot be identified, return null.

5. Correct obvious OCR formatting errors only when the intended value is clear.

6. Preserve medicine/test/service names as accurately as possible, but keep
item_name CLEAN — the product/service name only. Strip out HSN codes, batch
numbers, expiry dates, and location/bin codes even when they're jammed
together with the name in the OCR text (e.g. "30049035 EMESET4MG INJ2ML
A-38 S620043 05/25" should become item_name "Emeset 4mg Injection 2ml" —
nothing else).

7. Convert numeric amounts to numbers without currency symbols or commas.

8. Dates should preferably be returned as YYYY-MM-DD when clearly identifiable.

9. Extract every available bill line item.

10. Do not confuse MRP, unit price, discount, tax, quantity, and final line amount.

10a. NEVER treat a tooth number, tooth/area code, surface code, or other
location identifier as a quantity. Dental and medical bills often have a
"Tooth/Area", "Tooth No.", or "Surface" column containing values like "23",
"11,12", or "23-24" — these identify WHERE the procedure was performed, not
HOW MANY units were billed. Do not multiply the fee by this number. The
"amount" for each item is whatever final figure appears in the bill's own
Fee/Amount column for that line — copy it directly, do not recompute it by
multiplying unit_price by a tooth/area number. Only populate "quantity" when
the source explicitly states a count of units (e.g. "Qty: 2", "x3", "2
vials") — if a number near an item is a tooth/area/surface reference
instead, leave quantity null.

11. The total_amount must represent the final/net amount payable whenever available.

12. If multiple possible bill numbers or dates exist, select the one associated with the main invoice/bill.

13. Never infer medical conditions or diagnoses unless they are explicitly written in the bill.

14. NEVER confuse bill_no, patient/OP/IP numbers, batch numbers, or any other
ID-style number with total_amount. These numbers often sit visually close to
a "TOTAL" label due to OCR reading order, but a bill number (e.g.
"2223/167720") is NOT a monetary total, even if part of it (e.g. "2223")
looks like a plausible amount on its own. Only use a number as total_amount
if it appears directly in the TOTAL/grand total column of the bill itself,
formatted as a currency amount (with decimals like "1,234.00" or clearly
in a totals row/column). If the actual TOTAL column appears blank, cut off,
illegible, or the bill text ends with words like "Continue" (indicating the
totals are on a page you don't have), set total_amount to null rather than
substituting a nearby unrelated number. Watch for OCR-GARBLED spellings of
"Continue" near the TOTAL row too, not just the exact word -- OCR frequently
mangles it into things like "Contnue", "Continu", "Cont1nue", "Contlnue", or
similar near-matches. Any word starting with "Cont" sitting next to or below
the TOTAL/GRAND TOTAL label is almost certainly a garbled "Continue" and
should be treated the same as the exact word: set total_amount to null,
do not use a stray nearby number as the total just because "Continue" itself
wasn't spelled correctly.

15. WATCH FOR MISSING DECIMAL POINTS (a common OCR error): standard currency
amounts always have exactly 2 decimal digits. If an amount appears as a
large whole number with NO decimal point anywhere in it (e.g. "500000" or
"600000"), while other amounts nearby use a clear "X,XXX.00" format, the
decimal point was almost certainly dropped by OCR. Fix it by inserting the
decimal point exactly 2 digits from the right — e.g. "500000" -> 5000.00,
"600000" -> 6000.00 — NEVER by copying the decimal position of a
neighboring number, and never leaving it as a larger value like 50000.00.
ONLY apply this fix to amounts that have NO decimal point at all in the
source text. If a number already contains a decimal point (e.g.
"20000.00"), it is already correctly formatted — do NOT alter it, even if
it looks unusually large next to other items. Cross-check your
reinterpretation against any explicit total/subtotal shown elsewhere in the
document if one exists — the corrected line items should sum to that total.

15a. WATCH FOR A DECIMAL POINT REPLACED BY A SPACE: sometimes the source is
a scanned image where a currency amount's decimal point sits in a column
gap, and OCR reads that gap as whitespace instead of a "." — so an amount
that should be "5000.00" appears in the OCR text as two space-separated
number tokens: "5000 00" (or with extra spaces, "5000  00"). Do NOT treat
these as two separate numbers, and do NOT drop the trailing token or treat
it as a quantity. If a number token sitting where an amount is expected is
immediately followed by a short 2-digit token (e.g. "00", "25", "50"),
recognize this as the decimal/paise portion and join them with a "." — e.g.
"5000 00" -> 5000.00, "6000 00" -> 6000.00. This differs from rule 15
above (digits OCR'd with NO separator at all, like "500000"); this rule
covers the case where the separator became a space instead of being
dropped entirely.

16. WATCH FOR UNLABELED TOTALS: the true grand total sometimes appears as a
bare number on its own line immediately after the last line item, with NO
"Total:"/"Grand Total:" label in front of it (the label may have been on a
different line/column that OCR failed to associate with the number). If you
see a standalone number formatted as currency (e.g. "71,300.00" or
"71300.00") sitting right after the last item, and it is not already one of
the item amounts, treat it as total_amount. This does not override rule 14:
still never use a bill/patient/batch ID number as the total. And if the
totals area is genuinely blank, cut off, or illegible, still set
total_amount to null per rule 14 — do not invent a number that isn't there.

17. DO NOT DOUBLE-COUNT A SUMMARY SECTION AND A DETAILED SECTION: some bills
show the same charges TWICE in two different tables — once as broad
category subtotals (often titled "Bill Summary", "Summary", or similar,
with rows like "Room & Nursing Charges: 2350.00", "Professional Fees:
1200.00") and again as fully itemized individual charges in a separate
section (often titled "Detailed Breakup", "Itemized Bill", or similar) that
breaks those same categories down into their individual components. These
two sections describe the SAME money, not additional charges — the
itemized rows are what sum up to each category subtotal. When a bill has
both a category-summary section and a detailed/itemized section, extract
items ONLY from the detailed/itemized section into the items array. Do NOT
also add the category-summary rows as additional items — doing so counts
the same charges twice and inflates the sum of items well past the actual
total_amount. If a bill has ONLY a summary-style section with no further
itemized breakdown anywhere, then extract those summary rows as the items
instead, since in that case they're the only line-item detail available.

18. WATCH FOR SKIPPED BLANK COLUMNS CAUSING MRP/TOTAL CONFUSION: pharmacy
bills often have a row of columns like "MRP | Dis. | GST | Total" (or
similar: MRP, Discount, Tax, then a final line Total). When Dis. and/or GST
are BLANK for a row, the OCR text for that line will only contain TWO
numbers even though the header implies four columns — e.g. "14.53 29.06"
where 14.53 is MRP and 29.06 is the actual line Total, with no
discount/GST value present at all. In this situation:
- The LAST number on the line (rightmost, under the "Total" header) is
  always the line's "amount" — NEVER copy the MRP/unit_price value into
  amount just because fewer numbers than expected appear on the line.
- Do NOT treat that last number as "discount" either. If the Dis. and GST
  columns are genuinely blank in the source, leave "discount" and "tax"
  null for that row — don't repurpose the Total value to fill them.
- A quick sanity check: unit_price × quantity should equal (or be very
  close to) amount when quantity is known. If your extracted amount
  instead exactly equals unit_price, that's a strong signal you've
  fallen into this trap and should re-map: amount = the true line Total,
  not a copy of MRP.

Return JSON using exactly this structure:

{

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

bill_no:

Invoice number, bill number, receipt number, or invoice ID.

patient_name:

Name of the patient.

bill_date:

Primary invoice/bill date.

hospital_name:

Hospital, clinic, laboratory, or pharmacy name.

doctor_name:

Doctor/consultant name if explicitly available.

items:

Every individual medicine, laboratory test, procedure, consultation, room charge, service, consumable, or other billed item.

item_name:

Just the clean, human-readable product/service/medicine name — e.g.
"Emeset 4mg Injection 2ml", "Disposable Syringe 5ml", "CBCT Full Mouth".
Do NOT include HSN/item codes, batch numbers, expiry dates, location/bin
codes, or manufacturer codes in item_name — strip all of that out even if
it appears directly next to the name in the OCR text. Those code numbers
are not part of the item's name.

quantity:

Number of units/items — ONLY when explicitly stated as a count (e.g. "Qty:
2", "x3") in a Units/Qty column of its own. A tooth number, tooth/area
code, or surface code (e.g. "23", "11,12") is NOT a quantity — leave
quantity null for those and do not use that number to multiply the fee. If
the Units/Qty column is blank, missing, or illegible for a given row, set
quantity to null — do NOT reuse that row's amount or rate as the quantity
just because a number happens to be nearby.

unit_price:

Price per individual unit.

discount:

Discount applicable to this line item.

tax:

Tax/GST applicable to this line item.

amount:

Final amount of this particular line item.

subtotal:

Amount before overall discounts/taxes when available.

taxable_value:

The amount subject to tax, when the bill explicitly shows a "Taxable
Value" column or field separate from CGST/SGST/IGST — this is usually the
figure GST percentages get applied to. Only fill this if the bill actually
labels a value as "Taxable Value" (or clear equivalent); do not infer it by
subtracting tax from the total yourself.

discount_amount:

Total bill-level discount.

tax_amount:

Total GST/tax amount (combined, if the bill only shows one overall tax
figure rather than splitting CGST/SGST/IGST separately).

cgst_amount:

Central GST amount, only if explicitly shown as a separate labeled figure
(not just a column header with no value filled in).

sgst_amount:

State GST amount, only if explicitly shown as a separate labeled figure.

igst_amount:

Integrated GST amount, only if explicitly shown as a separate labeled
figure (used for inter-state bills instead of CGST+SGST).

total_amount:

Final/net/grand amount payable.

currency:

Currency such as INR, USD, AED, etc. If the ₹ symbol or Indian bill format is clearly present, return "INR".

If an item contains only item name and amount, return:

{

  "item_name": "CBC Test",

  "quantity": null,

  "unit_price": null,

  "discount": null,

  "tax": null,

  "amount": 450.00

}

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
even for values you leave null):
{
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
# then correctly (per rule 4, "if unknown return null") returns null for
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


# ── Deterministic dropped-decimal fix ──────────────────────────────────────
#
# Rule 15 in the prompt asks the model to reinsert a decimal point OCR
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
    """Safety net for rule 15. Divide any item amount/unit_price (or bill-
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


# ── Deterministic amount/discount-swap fix (safety net for rule 18) ───────
#
# Even with rule 18 in the prompt, a small model doesn't always get this
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


# ── Trailing tooth/area code cleanup (safety net for the item_name rule) ──
#
# The prompt tells the model to keep item_name clean (product/service name
# only, no codes), but a small model doesn't always strip a trailing
# tooth/area/surface number even when it correctly leaves it OUT of
# quantity (rule 10a). The result is a technically-correct amount with a
# leftover digit stuck on the end of the name, e.g. "Implant-MegaGen
# AnyRidge 23" instead of "Implant-MegaGen AnyRidge". Strip it in code as a
# final pass, since this is a simple, low-risk text cleanup rather than a
# judgment call the model needs to make.
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


# ── Deterministic "Continue" total-suppression (safety net for rule 14) ───
#
# Rule 14 tells the model to null out total_amount when the bill's own text
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
# the total at all if total_amount is null. Two real examples that motivated
# this: a row where OCR fused a date and a stray digit ("25/10/202237"),
# which can get misread as a quantity of 37, giving qty x unit_price way out
# of line with the row's actual amount; and a row where qty x unit_price
# (2 x 196.00 = 392.00) doesn't match the stated amount (352.80) at all --
# a genuine misread that a total-only check can't see if other rows offset
# it. This does NOT try to guess which field is wrong or silently correct
# anything (unlike fix_dropped_decimals/fix_amount_discount_swap, which only
# act on narrow, well-understood signatures) -- it just surfaces likely
# errors for a human reviewer, the same spirit as validate_total().

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
# total, just the best available fallback figure for review/display.

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
