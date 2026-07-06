"""
Document Chunking & Storage Tool — Main FastAPI Application
Serves HTML pages via Jinja2 + all API routes
"""
import os
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from routers import chunking
import PyPDF2
import io

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Chunking & Storage Tool",
    description="Split documents into chunks and store in PostgreSQL database",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static files & templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ─── API Routers ──────────────────────────────────────────────────────────────
app.include_router(chunking.router)

# ─── Page Routes ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/step/1", response_class=HTMLResponse)
async def step1(request: Request):
    return templates.TemplateResponse(request, "step1_chunk.html", {
        "request": request, "step": 1, "title": "Create & Save Chunks"
    })


# ─── File Upload Endpoint ─────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF or TXT file and return extracted text."""
    content = await file.read()
    text = ""

    if file.filename.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parse error: {str(e)}")
    elif file.filename.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from file")

    return {
        "success": True,
        "filename": file.filename,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text)
    }


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "RAG Learning Simulator"}
