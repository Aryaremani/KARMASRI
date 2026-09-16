import os
import time
import cv2
import pytesseract
from pytesseract import Output
from deskew import determine_skew
import pymupdf
from PIL import Image

# Windows: point pytesseract at the Tesseract binary explicitly, since it's
# often not added to PATH by the installer. Adjust this path if you installed
# to a different location. Harmless no-op on Linux/Mac if the path just
# doesn't exist there — pytesseract falls back to searching PATH normally.
_WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_WINDOWS_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT_PATH

# Where preprocessed images get saved for visual inspection (deskewed +
# rotated + CLAHE-enhanced, exactly what PaddleOCR actually receives).
_DEBUG_DIR = "debug_preprocessed"


def load_images(file_path, dpi=300):
    """Return list of PIL Images — all pages for PDF, single image otherwise."""
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        images = []
        doc = pymupdf.open(file_path)
        zoom = dpi / 72  # PyMuPDF's default render is 72 dpi
        matrix = pymupdf.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        doc.close()
        print(f"PDF has {len(images)} page(s)")
        return images
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return [Image.open(file_path)]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def fully_correct_image(image):
    # 1. Load image
    print("  Preprocessing image for OCR...")
    print(type(image))

    if hasattr(image, "shape"):
        print(image.shape)

    # Normalize to 3-channel BGR up front. Depending on the source file
    # (a grayscale scan, a PNG with an alpha channel, etc.), PIL/numpy can
    # hand us a 1-channel (H, W), 1-channel (H, W, 1), or 4-channel
    # (H, W, 4) array instead of the 3-channel (H, W, 3) BGR image every
    # cv2 call below assumes. Doing this conversion once here means we
    # don't have to guess later exactly where a channel-count mismatch
    # crept in.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. STEP ONE: Fix large 90/180/270 degree flips using Tesseract OSD
    try:
        osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotate_angle = osd["rotate"]

        if rotate_angle != 0:
            # Rotate safely to keep borders intact
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, -rotate_angle, 1.0)

            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]

            image = cv2.warpAffine(image, matrix, (new_w, new_h))
    except Exception as e:
        # Pass if OSD fails due to lack of text lines on a blank image,
        # or if Tesseract still isn't found — print so it's visible in
        # logs instead of silently degrading OCR accuracy.
        print(f"  OSD rotation check skipped ({e})")

    # 3. STEP TWO: Fix small tilts (0.5 to 15 degrees) using deskew library
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate fine skew angle based on text lines
    skew_angle = determine_skew(grayscale)

    # Sanity-check the angle before applying it. determine_skew is tuned for
    # small tilts (0.5-15 degrees) on scanned/photographed pages. On a
    # clean, mostly-white digital bill with lots of perfectly straight
    # table lines, it can misfire and return a large, spurious angle.
    # Unlike the OSD 90/180/270 correction above (which expands the canvas
    # to fit the rotated image), this step warps into a SAME-SIZE canvas --
    # so a large spurious angle rotates most of the actual content out of
    # frame entirely, leaving OCR with almost nothing. Only apply the
    # correction within the range this step is actually designed for;
    # treat anything outside it as a misdetection and skip it.
    MAX_FINE_SKEW_DEGREES = 15
    if skew_angle and abs(skew_angle) <= MAX_FINE_SKEW_DEGREES:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # deskew returns angle in counter-clockwise direction
        matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elif skew_angle:
        print(
            f"  Skipping deskew: detected angle {skew_angle:.1f} deg looks "
            f"like a misdetection (outside +/-{MAX_FINE_SKEW_DEGREES} deg)"
        )

    return image

def apply_clahe(image):
    # 1. Verify the image has 3 color channels (BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 2. Convert from BGR color space to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 3. Split the LAB image into its individual channels (L, A, B)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 4. Initialize CLAHE and apply it ONLY to the Lightness channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # 5. Merge the enhanced Lightness channel back with the original colors
        updated_lab = cv2.merge((enhanced_l, a_channel, b_channel))

        # 6. Convert back to standard BGR color format
        enhanced_color_image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

        return enhanced_color_image
    else:
        # Fallback if a grayscale image accidentally gets passed in
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

def preprocess_image(image):
    """Preprocess for better OCR accuracy — keep RGB for PaddleOCR."""
    image = fully_correct_image(image)
    enhanced_image = apply_clahe(image)

    # ── Save preprocessed image for visual inspection ──────────────────
    # enhanced_image is BGR at this point (all the cv2 work above operates
    # in BGR), which is exactly what cv2.imwrite expects — no color
    # conversion needed before saving.
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    debug_path = os.path.join(_DEBUG_DIR, f"preprocessed_{int(time.time() * 1000)}.png")
    cv2.imwrite(debug_path, enhanced_image)
    print(f"  Saved preprocessed image: {debug_path}")

    return enhanced_image
