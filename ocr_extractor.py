"""
ocr_extractor.py

OCR extraction: PaddleOCR (CPU) for text extraction, PyMuPDF
for PDF page rendering.

Install Python dependencies:
    pip install paddleocr paddlepaddle pymupdf pillow numpy deskew opencv-python pytesseract

Note: paddlepaddle has no separate GPU build requirement here — device="cpu"
below is deliberate and correct for a no-GPU machine.
"""
import os
import tempfile
import numpy as np
from image_processing import load_images, preprocess_image


os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

from paddleocr import PaddleOCR


def init_ocr_engine():
    """Try new 3.x API first, fall back to 2.x API if needed.

    Constructing PaddleOCR(...) can succeed even on a 2.x install that
    silently accepts 3.x-style kwargs — so we can't rely on the constructor
    alone to tell us which API we actually have. Check for a real 3.x-only
    method (predict) before trusting that detection.
    """
    try:
        engine = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            device="cpu"
        )
        if hasattr(engine, "predict"):
            print("PaddleOCR 3.x initialized")
            return engine, "v3"
        else:
            print("Constructor accepted 3.x kwargs but no predict() method found — this is actually a 2.x install")
    except Exception as e:
        print(f"3.x init failed ({e}), trying 2.x API...")

    try:
        engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            device="cpu"
        )
        print("PaddleOCR 2.x initialized")
        return engine, "v2"
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed to initialize: {e}")


# Initialize once at import time (this is the slow part — keep the backend
# process warm rather than restarting it per request).
ocr_engine, ocr_version = init_ocr_engine()

# ── OCR ───────────────────────────────────────────────────────────────────────

def run_ocr_v3(image_np):
    """PaddleOCR 3.x predict() API."""
    result = ocr_engine.predict(image_np)
    ocr_result = result[0]
    return ocr_result  # return raw result for further processing


def run_ocr_v3_via_path(image):
    """Fallback: save to temp file and pass path instead of numpy array.

    On Windows, a NamedTemporaryFile can't be reopened by another process
    while still open (unlike on Linux), so we close it explicitly before
    handing the path to PaddleOCR, and clean up in a finally block.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        image.save(tmp_path)
        result = ocr_engine.predict(tmp_path)
        return result[0]  # return raw result for further processing
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_ocr_v2(image_np):
    """PaddleOCR 2.x ocr() API.

    Returns raw 2.x nested-list format:
        [[ [bbox_poly], (text, score) ], ...]
    Callers must run this through parse_paddle_result_v2() before it
    reaches merge_lines(), since that expects the common word-dict shape.
    """
    result = ocr_engine.ocr(image_np, cls=True)
    return result[0] if result else []


def ocr_image(image):
    image_np = np.array(image)
    processed_image = preprocess_image(image_np)

    if ocr_version == "v3":
        try:
            ocr_result = run_ocr_v3(processed_image)
        except NotImplementedError:
            print("numpy input failed, trying file path fallback...")
            ocr_result = run_ocr_v3_via_path(image)
        except Exception as e:
            print(f"v3 predict failed ({e}), trying file path fallback...")
            ocr_result = run_ocr_v3_via_path(image)

        words = parse_paddle_result(ocr_result)

    else:
        raw_result = run_ocr_v2(processed_image)
        words = parse_paddle_result_v2(raw_result)  # FIX: was skipping parsing entirely

    # ── Confidence score logging ────────────────────────────────────────
    # Printed to terminal so low-confidence words (likely OCR errors) are
    # visible right away instead of only showing up as downstream
    # extraction mistakes with no obvious cause.
    if words:
        confidences = [w["confidence"] for w in words]
        avg_conf = sum(confidences) / len(confidences)
        low_conf_words = [w for w in words if w["confidence"] < 0.7]
        print(f"  OCR confidence: {len(words)} words, avg={avg_conf:.3f}, "
              f"min={min(confidences):.3f}, max={max(confidences):.3f}")
        if low_conf_words:
            print(f"  {len(low_conf_words)} low-confidence word(s) (<0.70):")
            for w in low_conf_words:
                print(f"    '{w['text']}' — {w['confidence']:.3f}")
    else:
        print("  OCR confidence: no words detected")

    lines = merge_lines(words)
    table = build_table(lines)
    text = lines_to_text(lines)

    return {
        "text": text,
        "lines": lines,
        "words": words,
        "table": table,
    }

def prepare_llm_input(ocr_result):
    return "\n".join(
        " ".join(word["text"] for word in line)
        for line in ocr_result["lines"]
    )

def ocr_file(file_path, progress_cb=None):
    """OCR all pages of a file, return concatenated text.

    progress_cb(page_num, total_pages) is called after each page, if given —
    used by the backend to stream progress to the frontend.
    """
    images = load_images(file_path)
    pages_text = []

    for i, image in enumerate(images, start=1):
        print(f"  Processing page {i}/{len(images)}...")
        ocr_result = ocr_image(image)
        text = prepare_llm_input(ocr_result)
        pages_text.append(f"--- Page {i} ---\n{text}")
        if progress_cb:
            progress_cb(i, len(images))

    return "\n\n".join(pages_text)

def parse_paddle_result(ocr_result):
    """Parse PaddleOCR 3.x predict() output into the common word-dict shape."""
    words = []

    for text, score, poly in zip(
        ocr_result["rec_texts"],
        ocr_result["rec_scores"],
        ocr_result["rec_polys"],
    ):
        poly = poly.tolist() if hasattr(poly, "tolist") else poly

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x = min(xs)
        y_top, y_bottom = min(ys), max(ys)

        words.append({
            "text": text,
            "confidence": float(score),
            "bbox": poly,
            "x": x,
            # Vertical CENTER of the box, not the top -- less sensitive to
            # ascenders/descenders varying between words on the same row,
            # which otherwise makes rows look more "staggered" than they
            # really are.
            "y": (y_top + y_bottom) / 2,
            "height": y_bottom - y_top,
        })

    return words


def parse_paddle_result_v2(raw_result):
    """Parse PaddleOCR 2.x ocr() output into the SAME word-dict shape as
    parse_paddle_result(), so downstream code (merge_lines, build_table,
    etc.) doesn't need to know which API version produced the data.

    2.x format per line: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score) ]
    """
    words = []
    if not raw_result:
        return words

    for line in raw_result:
        if not line or len(line) < 2:
            continue
        poly, (text, score) = line[0], line[1]

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x = min(xs)
        y_top, y_bottom = min(ys), max(ys)

        words.append({
            "text": text,
            "confidence": float(score),
            "bbox": poly,
            "x": x,
            "y": (y_top + y_bottom) / 2,
            "height": y_bottom - y_top,
        })

    return words

def sort_words(words):
    return sorted(words, key=lambda w: (w["y"], w["x"]))

def merge_lines(words, y_threshold=None):
    """Group OCR words into table/text rows by vertical position.

    A fixed pixel threshold doesn't generalize well: dense tables with
    tightly-packed rows (small font, high DPI) can have genuinely distinct
    rows whose word centers sit closer together than a fixed guess, causing
    two separate rows to be merged into one garbled line. To avoid that,
    the threshold is derived from this document's own measured word height
    (unless the caller passes an explicit y_threshold) -- it adapts to
    whatever font size/DPI this particular file was rendered at.
    """
    words = sort_words(words)
    if not words:
        return []

    if y_threshold is None:
        heights = sorted(w["height"] for w in words if w.get("height"))
        median_height = heights[len(heights) // 2] if heights else 20
        # A bit over half the median glyph height is normally enough to
        # keep words on the same row together while still separating
        # adjacent rows whose baselines are only a small gap apart.
        y_threshold = max(6, min(20, median_height * 0.6))

    lines = []
    current = []
    current_y = None
    for word in words:
        y = word["y"]
        if current_y is None:
            current = [word]
            current_y = y
            continue
        if abs(y - current_y) <= y_threshold:
            current.append(word)
            # Track the running average y of the line so far, rather than
            # comparing every new word only against the first word seen --
            # otherwise the comparison point never updates as the line
            # grows, letting later words drift further from where the row
            # actually is.
            current_y = sum(w["y"] for w in current) / len(current)
        else:
            current.sort(
                key=lambda w: w["x"]
            )
            lines.append(current)
            current = [word]
            current_y = y
    if current:
        current.sort(
            key=lambda w: w["x"]
        )
        lines.append(current)
    return lines

def lines_to_text(lines):
    output = []
    for line in lines:
        text = " ".join(
            w["text"]
            for w in line
        )
        output.append(text)
    return "\n".join(output)

def build_table(lines):
    rows = []
    for line in lines:
        entries = []
        for word in line:
            entries.append({
                "text": word["text"],
                "x": word["x"],
                "y": word["y"]
            })
        rows.append(entries)
    return rows
