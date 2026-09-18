"""
json_repair.py

Best-effort repair of JSON coming out of an LLM — handles markdown fences,
<think> blocks, leading/trailing prose around the JSON, truncated output
(missing closers), and trailing commas.

Usage:
    from json_repair import repair_llm_json

    result = repair_llm_json(llm_output_text)
    if result is None:
        # still unparseable after every repair attempt
        ...
"""

import re
import json


def _strip_llm_wrapping(text):
    """Markdown fences, <think> blocks, and common LLM JSON-string escaping quirks."""
    text = text.strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("\\'", "'")
    text = re.sub(r'\\n', ' ', text)
    return text.strip()


def _extract_balanced_json(text):
    """
    Find the first '{' or '[' and scan forward tracking a bracket stack and
    string state, so nesting order and in-string brackets are respected
    (unlike simple text.count("{") vs text.count("}")).

    Returns (json_substring, truncated, open_stack, unterminated_string,
    string_start_rel):
      - truncated: ran out of text before the bracket stack emptied
      - open_stack: brackets still open, in order opened (used to know what
        to append, and in what order, if truncated)
      - unterminated_string: True if truncation happened mid-string,
        meaning a closing quote is needed before the bracket closers (or,
        if it's a dangling KEY rather than a value, the entry needs to be
        dropped instead -- see repair_llm_json)
      - string_start_rel: index, relative to the returned json_substring,
        of the opening quote of the currently-open (possibly truncated)
        string. None if no string is open at the point of truncation.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None, False, [], False, None

    stack = []
    in_string = False
    escape = False
    end = None
    string_start_rel = None

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                string_start_rel = None
            continue

        if ch == '"':
            in_string = True
            string_start_rel = i - start
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                end = i
                break

    if end is not None:
        # Balanced — trim anything after the matching close (extra trailing
        # brackets/garbage/prose the model appended after the real JSON).
        return text[start:end + 1], False, [], False, None

    # Ran out of text before the stack emptied — truncated mid-object.
    return text[start:], True, stack, in_string, string_start_rel


def _remove_trailing_commas(text):
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r",\s*$", "", text)
    return text


def _drop_dangling_key(text):
    """Remove a trailing property key that got its colon but was cut off
    before any value followed it at all -- e.g. text ending in
    '..., "total_amount":'. There's no value to recover here, so the safest
    repair is to drop the whole incomplete entry rather than invent one."""
    return re.sub(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$', '', text)


def repair_llm_json(raw_text):
    """
    Best-effort repair of JSON coming out of an LLM. Handles, in order:
      1. Markdown fences, <think> blocks, stray backslash-escapes
      2. Leading prose before the JSON ("Here is the JSON: {...}")
      3. Trailing prose/extra brackets after the JSON
      4. Truncated JSON (missing closers), closed in the correct nesting order
      5. Trailing commas before a closing bracket

    Returns a parsed dict/list, or None if it still can't be parsed after
    all of the above.
    """
    text = _strip_llm_wrapping(raw_text)

    extracted, truncated, open_stack, unterminated_string, string_start_rel = \
        _extract_balanced_json(text)
    if extracted is None:
        return None

    if truncated:
        if unterminated_string and string_start_rel is not None:
            # A string was left open at the point of truncation. Figure out
            # whether it's a property KEY (never reached its colon+value)
            # or a VALUE (just missing its closing quote), by checking the
            # last non-whitespace character right before it started:
            #   ':'  -> this is a value -> safe to close with a quote.
            #   '{' or ',' (or nothing) -> this is a key that never got a
            #     value -> we can't safely invent one, so drop the whole
            #     dangling entry instead of producing a key with no value.
            prefix = extracted[:string_start_rel].rstrip()
            preceding_char = prefix[-1] if prefix else ""

            if preceding_char == ":":
                extracted += '"'  # close the dangling string value
            else:
                extracted = re.sub(r',\s*$', '', prefix)

        # Independently of the above, we may still be left with a trailing
        # key that DID get its closing quote and colon, but was cut off
        # before any value followed (e.g. '..., "total_amount":'). That's
        # equally unrecoverable, so drop it too.
        extracted = _drop_dangling_key(extracted)

        extracted = _remove_trailing_commas(extracted)
        # Close in reverse order of what's still open — e.g. stack ['{', '[']
        # means we're inside a list inside an object, so close ']' then '}'.
        closers = {"{": "}", "[": "]"}
        extracted += "".join(closers[b] for b in reversed(open_stack))
    else:
        extracted = _remove_trailing_commas(extracted)

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


# ── quick tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # truncated mid-way
        '{"bill_number": "123", "charge_items": {"room rent": "500", "medicine": "20',
        # extra trailing garbage / brackets
        '{"bill_number": "123", "total_amount": "500"}}} \n\nLet me know if you need anything else!',
        # leading prose + markdown fence + trailing comma
        '```json\nHere is the JSON:\n{"bill_number": "123", "total_amount": "500",}\n```',
        # think tags + nested truncation
        '<think>reasoning...</think>{"charge_items": {"a": "1", "b": "2"',
        # already valid
        '{"bill_number": "123", "total_amount": "500"}',
        # truncated mid-KEY name, no colon reached yet (the real bug report:
        # model hit num_predict before finishing the "total_amount" key)
        '{"bill_no": "3193", "items": [{"item_name": "SYRINGE", "amount": 33.0}], "total',
        # truncated right after a key's colon, before any value chars
        '{"bill_no": "3193", "items": [], "total_amount":',
    ]
    for c in cases:
        print(repr(c[:60]), "->", repair_llm_json(c))