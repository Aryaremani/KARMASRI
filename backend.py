
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
 
app = FastAPI(title="Bill Extractor API")
 
# Local tool — allow the frontend to call this from anywhere on localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
 
 
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
        full_text = ocr.ocr_file(tmp_path)
        print(f"OCR found {len(full_text)} characters in {file.filename}")
        print(f"OCR full text:\n{full_text}")
 
        if not full_text.strip():
            raise HTTPException(422, "OCR found no text in this file.")
 
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
        }
    finally:
        os.unlink(tmp_path)
 
 
# Optional: serve the frontend from this same server at http://localhost:8000/
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
 
