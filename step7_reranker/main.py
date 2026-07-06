"""Step 7 — Chunks Re-ranker App (FastAPI)"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from routers import rerank

app = FastAPI(title="Step 7 — Chunks Re-ranker", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.include_router(rerank.router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/api/health")
async def health():
    return {"status": "ok", "step": 7, "app": "Chunks Re-ranker"}
