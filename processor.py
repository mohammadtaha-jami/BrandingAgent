"""
لایه پردازش RAG: بارگذاری داده، Chunking، Embedding، FAISS و زنجیره پاسخ‌دهی.

نکته مهم: FAISS در این پروژه به‌صورت پیش‌فرض در RAM نگه‌داری می‌شود.
با ری‌استارت سرور، داده‌ها از بین می‌روند مگر اینکه persist را فعال کنید
(متغیر محیطی PERSIST_FAISS=true و پوشه FAISS_DATA_DIR).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
# در LangChain 1.x زنجیره‌های کلاسیک به پکیج langchain-classic منتقل شده‌اند
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- تنظیمات Chunking (طبق درخواست: ۱۰۰۰ کاراکتر، ۲۰۰ هم‌پوشانی) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# مدل‌های Google (Gemini Developer API)
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

# ذخیره‌سازی اختیاری FAISS روی دیسک (برای نزدیک شدن به Production)
PERSIST_FAISS = os.getenv("PERSIST_FAISS", "false").lower() in ("1", "true", "yes")
FAISS_DATA_DIR = Path(os.getenv("FAISS_DATA_DIR", "data/faiss_sessions"))


def _get_api_key() -> str:
    """کلید API گوگل را از محیط می‌خواند."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "کلید GOOGLE_API_KEY تنظیم نشده است. "
            "از Google AI Studio یک API Key بگیرید و در فایل .env قرار دهید."
        )
    return key


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    متن‌های بلند را به قطعات کوچک‌تر تقسیم می‌کند تا Embedding و بازیابی دقیق‌تر شود.
    هم‌پوشانی (overlap) باعث می‌شود جمله‌ای که بین دو chunk قطع شده، در هر دو دیده شود.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """مدل Embedding گوگل — هر chunk به یک بردار عددی تبدیل می‌شود."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=_get_api_key(),
    )


def _build_chat_model() -> ChatGoogleGenerativeAI:
    """مدل زبانی Gemini برای تولید پاسخ نهایی بر اساس chunkهای بازیابی‌شده."""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=_get_api_key(),
        temperature=0.2,
    )


@dataclass
class SessionState:
    """وضعیت هر کاربر (Session): Vector Store و تعداد سندهای ingest شده."""

    session_id: str
    vectorstore: FAISS | None = None
    document_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGProcessor:
    """
    مدیریت Sessionها و منطق RAG.

    هر session_id یک FAISS جدا دارد؛ کاربر A به داده‌های کاربر B دسترسی ندارد.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None
        self._text_splitter = _build_text_splitter()
        if PERSIST_FAISS:
            FAISS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Embedding مدل را فقط هنگام نیاز می‌سازد تا import بدون .env ممکن باشد."""
        if self._embeddings is None:
            self._embeddings = _build_embeddings()
        return self._embeddings

    def create_session(self) -> str:
        """یک شناسه Session جدید می‌سازد."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionState(session_id=session_id)
        return session_id

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Session نامعتبر است: {session_id}")
        return self._sessions[session_id]

    def _session_storage_path(self, session_id: str) -> Path:
        return FAISS_DATA_DIR / session_id

    def _save_vectorstore(self, session: SessionState) -> None:
        """ذخیره FAISS روی دیسک — معادل faiss.save_local در LangChain."""
        if not PERSIST_FAISS or session.vectorstore is None:
            return
        path = self._session_storage_path(session.session_id)
        path.mkdir(parents=True, exist_ok=True)
        session.vectorstore.save_local(str(path))

    def _load_vectorstore_if_exists(self, session_id: str) -> FAISS | None:
        """در صورت وجود فایل قبلی، Vector Store را از دیسک بارگذاری می‌کند."""
        if not PERSIST_FAISS:
            return None
        path = self._session_storage_path(session_id)
        index_file = path / "index.faiss"
        if not index_file.exists():
            return None
        return FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _add_chunks_to_session(self, session: SessionState, chunks: list[Document]) -> int:
        """Chunkها را به FAISS همان Session اضافه می‌کند (ایجاد یا append)."""
        if not chunks:
            return 0

        if session.vectorstore is None:
            # اولین بار: ساخت index در حافظه (یا بعداً save_local)
            session.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            # ingest بعدی: افزودن به index موجود
            session.vectorstore.add_documents(chunks)

        session.document_count += len(chunks)
        self._save_vectorstore(session)
        return len(chunks)

    def ingest_pdf(self, session_id: str, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        PDF را با PyPDFLoader می‌خواند، صفحات را chunk می‌کند و در FAISS ذخیره می‌کند.
        """
        session = self._get_session(session_id)
        suffix = Path(filename).suffix or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        chunks = self._text_splitter.split_documents(pages)
        added = self._add_chunks_to_session(session, chunks)

        return {
            "message": "PDF با موفقیت پردازش شد.",
            "pages_loaded": len(pages),
            "chunks_added": added,
            "total_chunks_in_session": session.document_count,
        }

    def ingest_text(self, session_id: str, text: str, source_label: str = "user_text") -> dict[str, Any]:
        """متن خام را به Document تبدیل، chunk و در FAISS همان Session ذخیره می‌کند."""
        session = self._get_session(session_id)
        text = (text or "").strip()
        if not text:
            raise ValueError("متن ورودی خالی است.")

        doc = Document(page_content=text, metadata={"source": source_label})
        chunks = self._text_splitter.split_documents([doc])
        added = self._add_chunks_to_session(session, chunks)

        return {
            "message": "متن با موفقیت پردازش شد.",
            "chunks_added": added,
            "total_chunks_in_session": session.document_count,
        }

    def _build_retrieval_chain(self, vectorstore: FAISS):
        """
        Retrieval Chain:
        1) سوال کاربر → embedding → جستجو در FAISS
        2) chunkهای مرتبط + سوال → پرامپت → Gemini → پاسخ
        """
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_template(
            """شما یک دستیار هوشمند هستید که فقط بر اساس متن زمینه (context) زیر پاسخ می‌دهید.
اگر پاسخ در context نبود، صادقانه بگویید «در منابع بارگذاری‌شده پاسخی برای این سوال پیدا نکردم».

متن زمینه:
{context}

سوال کاربر:
{input}

پاسخ (به زبان همان سوال):"""
        )

        llm = _build_chat_model()
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        """سوال کاربر را با RAG پاسخ می‌دهد."""
        session = self._get_session(session_id)
        question = (question or "").strip()
        if not question:
            raise ValueError("سوال خالی است.")

        if session.vectorstore is None:
            # تلاش برای بارگذاری از دیسک (اگر persist فعال باشد)
            loaded = self._load_vectorstore_if_exists(session_id)
            if loaded is None:
                raise ValueError(
                    "هنوز هیچ PDF یا متنی برای این Session بارگذاری نشده است."
                )
            session.vectorstore = loaded

        chain = self._build_retrieval_chain(session.vectorstore)
        result = chain.invoke({"input": question})

        answer = result.get("answer", "")
        source_docs = result.get("context", [])

        sources = []
        for doc in source_docs[:4]:
            meta = getattr(doc, "metadata", {}) or {}
            sources.append(
                {
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "preview": (doc.page_content or "")[:200],
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }

    def delete_session(self, session_id: str) -> None:
        """Session را از حافظه (و در صورت persist از دیسک) حذف می‌کند."""
        self._sessions.pop(session_id, None)
        if PERSIST_FAISS:
            path = self._session_storage_path(session_id)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    def session_info(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session_id,
            "chunks_indexed": session.document_count,
            "has_vectorstore": session.vectorstore is not None,
            "persist_enabled": PERSIST_FAISS,
        }


# نمونه singleton برای استفاده در FastAPI
rag_processor = RAGProcessor()
