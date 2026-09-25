"""
backend.py — local HTTP API in front of the OCR + LLM pipeline
 
Run (Windows PowerShell):
    .\\venv\\Scripts\\Activate.ps1
    uvicorn backend:app --host 0.0.0.0 --port 8000
 
Then open http://localhost:8000 in a browser (serves the frontend/ folder
if present, and exposes the API underneath it).
 
This process loads PaddleOCR once at startup (slow) and keeps it warm,
so requests after the first are much faster than running the script fresh
each time.
 
LOCAL GPU SETUP:
    Ollama should be installed and running locally (Windows service starts
    it automatically after install: https://ollama.com/download/windows).
    No OLLAMA_HOST env var needed — llm_structuring.py defaults to
    http://localhost:11434, which is where local Ollama listens.
"""
 
import os
import shutil
import tempfile
import ollama as ollama_pkg
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import ocr_extractor as ocr
import llm_structuring as llm

def calculate_extraction_reliability(fields, validation):
    """
    Estimate reliability of the structured extraction.

    Evidence used:
    - presence of important identifying fields
    - presence of charge items
    - validity of extracted amounts
    - total validation when a stated total exists

    This does NOT decide accept/reject.
    It only reports extraction reliability.
    """

    evidence = []

    # ---------------------------------------------------------
    # Important fields
    # ---------------------------------------------------------

    important_fields = [
        "patient_name",
        "bill_date",
        "hospital_name",
        "currency",
    ]

    present_fields = 0

    for field in important_fields:
        value = fields.get(field)

        if value is not None and str(value).strip():
            present_fields += 1

    field_completeness = (
        present_fields / len(important_fields)
    )

    evidence.append({
        "name": "field_completeness",
        "score": round(field_completeness * 100, 2),
    })

    # ---------------------------------------------------------
    # Charge items
    # ---------------------------------------------------------

    items = fields.get("items") or []

    if items:
        valid_items = 0

        for item in items:
            name = item.get("item_name")
            amount = item.get("amount")

            if (
                name
                and str(name).strip()
                and amount is not None
            ):
                try:
                    float(amount)
                    valid_items += 1
                except (TypeError, ValueError):
                    pass

        item_quality = valid_items / len(items)

    else:
        item_quality = 0.0

    evidence.append({
        "name": "item_quality",
        "score": round(item_quality * 100, 2),
    })

    # ---------------------------------------------------------
    # Total validation
    # ---------------------------------------------------------

    validation_score = None

    if validation.get("checked"):

        if validation.get("match") is True:
            validation_score = 1.0

        else:
            validation_score = 0.0

    evidence.append({
        "name": "total_validation",
        "score": (
            round(validation_score * 100, 2)
            if validation_score is not None
            else None
        ),
    })

    # ---------------------------------------------------------
    # Combine evidence
    # ---------------------------------------------------------

    scores = [
        field_completeness,
        item_quality,
    ]

    # Only include total validation when an actual stated total
    # was found and therefore a comparison was possible.
    if validation_score is not None:
        scores.append(validation_score)

    if scores:
        reliability = sum(scores) / len(scores)
    else:
        reliability = 0.0

    return {
        "score": round(reliability * 100, 2),
        "field_completeness": round(
            field_completeness * 100, 2
        ),
        "item_quality": round(
            item_quality * 100, 2
        ),
        "total_validation": (
            round(validation_score * 100, 2)
            if validation_score is not None
            else None
        ),
        "evidence": evidence,
    }
 
app = FastAPI(title="Bill Extractor API")
 
# Local tool — allow the frontend to call this from anywhere on localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Below this legibility score, the document is rejected before the LLM is
# ever called -- there's no point spending the (slow) Ollama call on a
# page the OCR engine plainly couldn't read confidently.
LEGIBILITY_REJECT_THRESHOLD = 75.0
 
 
@app.get("/api/health")
def health():
    """Reports OCR engine status AND whether Ollama is actually reachable,
    so failures show up here instead of as a raw connection traceback
    the first time someone uploads a file.
    """
    ollama_status = "unknown"
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        client = ollama_pkg.Client(host=ollama_host)
        client.list()
        ollama_status = "reachable"
    except Exception as e:
        ollama_status = f"unreachable ({e})"
 
    return {
        "status": "ok",
        "ocr_version": ocr.ocr_version,
        "ollama_host": ollama_host,
        "ollama_status": ollama_status,
    }
 
@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}")
 
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
 
    try:
        ocr_result = ocr.ocr_file(tmp_path)
        full_text = ocr_result["full_text"]
        print(f"OCR found {len(full_text)} characters in {file.filename}")
        print(f"OCR full text:\n{full_text}")
 
        if not full_text.strip():
            raise HTTPException(422, "OCR found no text in this file.")

        # ---- Legibility gate: reject BEFORE calling the LLM at all ----
        # A multi-page bill is only as trustworthy as its worst page, so
        # the whole document is rejected if ANY single page falls below
        # the threshold -- checking only an average could let one
        # unreadable page's bad fields slip through hidden behind good
        # pages elsewhere in the same document.
        pages_legibility = [
            {"page_number": p["page_number"], "legibility": p["legibility"]}
            for p in ocr_result["pages"]
        ]
        worst_page = min(pages_legibility, key=lambda p: p["legibility"]["score"])

        if worst_page["legibility"]["score"] < LEGIBILITY_REJECT_THRESHOLD:
            raise HTTPException(
                422,
                detail={
                    "rejected": True,
                    "message": (
                        f"Document rejected: page {worst_page['page_number']} scored "
                        f"{worst_page['legibility']['score']}/100 legibility "
                        f"(\"{worst_page['legibility']['label']}\"), below the "
                        f"{LEGIBILITY_REJECT_THRESHOLD} threshold required to run "
                        f"extraction. Re-scan or re-photograph the document and try again."
                    ),
                    "legibility_threshold": LEGIBILITY_REJECT_THRESHOLD,
                    "pages": pages_legibility,
                },
            )

        print("Calling LLM for structuring...")
        try:
            llm_out = llm.extract_with_llm(full_text, ollama_host=os.environ.get("OLLAMA_HOST"))
        except Exception as e:
            raise HTTPException(
                502,
                f"Could not reach Ollama at "
                f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}. "
                f"Is 'ollama serve' running? Original error: {e}",
            )
 
        if llm_out is None:
            raise HTTPException(422, "The model's output could not be parsed as JSON. Try again or use a clearer scan.")
 
        validation = llm.validate_total(llm_out)
 
        return {
            "fields": llm_out,
            "validation": validation,
            "ocr_char_count": len(full_text),
            "ocr_text": full_text,
            "pages": [
                {
                    "page_number": page["page_number"],
                    "legibility": page["legibility"],
                    "text": page["text"],
                }
                for page in ocr_result["pages"]
            ],
        }
    finally:
        os.unlink(tmp_path)
 
 
# Optional: serve the frontend from this same server at http://localhost:8000/
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
