"""
main.py — لایه API با FastAPI

Endpointها:
  POST /api/session          → ساخت Session جدید
  POST /api/ingest/pdf       → آپلود PDF
  POST /api/ingest/text      → ورودی متنی
  POST /api/chat             → پرسش و پاسخ RAG
  GET  /                     → رابط کاربری (index.html)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from processor import rag_processor

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
INDEX_HTML = APP_DIR / "index.html"

app = FastAPI(
    title="RAG Bot — LangChain + FAISS + Gemini",
    description="سیستم RAG ماژولار با Session ID برای جداسازی داده هر کاربر",
    version="1.0.0",
)

# CORS برای توسعه محلی (فرانت و API روی همان origin هستند؛ برای جدا بودن دامنه مفید است)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- مدل‌های درخواست/پاسخ ---


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="متن خام برای ingest")
    session_id: str | None = Field(None, description="شناسه Session؛ اگر نباشد از هدر استفاده می‌شود")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    message: str


def _resolve_session_id(header_session: str | None, body_session: str | None) -> str:
    """Session را از بدنه JSON یا هدر X-Session-ID می‌گیرد."""
    session_id = (body_session or header_session or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id الزامی است (هدر X-Session-ID یا فیلد session_id در body).",
        )
    return session_id


def _handle_processor_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"خطای سرور: {exc}")


# --- API Routes ---


@app.post("/api/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    """
    یک Session جدید می‌سازد.
    فرانت‌اند باید session_id را در localStorage ذخیره و در هر درخواست بعدی بفرستد.
    """
    session_id = rag_processor.create_session()
    return SessionResponse(
        session_id=session_id,
        message="Session ایجاد شد. این شناسه را در تمام درخواست‌های بعدی ارسال کنید.",
    )


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    try:
        return rag_processor.session_info(session_id)
    except Exception as exc:
        raise _handle_processor_error(exc) from exc


@app.post("/api/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """دریافت فایل PDF و افزودن chunkها به Vector Store همان Session."""
    session_id = _resolve_session_id(x_session_id, None)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فقط فایل PDF پذیرفته می‌شود.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="فایل خالی است.")

    try:
        result = rag_processor.ingest_pdf(session_id, content, file.filename)
        return result
    except Exception as exc:
        raise _handle_processor_error(exc) from exc


@app.post("/api/ingest/text")
async def ingest_text(
    payload: TextIngestRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """دریافت متن خام و chunk کردن آن در FAISS."""
    session_id = _resolve_session_id(x_session_id, payload.session_id)
    try:
        return rag_processor.ingest_text(session_id, payload.text)
    except Exception as exc:
        raise _handle_processor_error(exc) from exc


@app.post("/api/chat")
async def chat(
    payload: ChatRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """پرسش سوال و دریافت پاسخ مبتنی بر RAG (بدون رفرش — از سمت فرانت با fetch)."""
    session_id = _resolve_session_id(x_session_id, payload.session_id)
    try:
        return rag_processor.ask(session_id, payload.question)
    except Exception as exc:
        raise _handle_processor_error(exc) from exc


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    try:
        rag_processor.delete_session(session_id)
        return {"message": "Session حذف شد."}
    except Exception as exc:
        raise _handle_processor_error(exc) from exc


@app.get("/")
async def serve_index():
    """صفحه اصلی رابط کاربری."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html یافت نشد.")
    return FileResponse(INDEX_HTML)


# فایل‌های استاتیک اضافی (در صورت نیاز)
static_dir = APP_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
