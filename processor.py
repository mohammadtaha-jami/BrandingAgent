"""
لایه پردازش RAG با مسیریاب Intent (Router-Based Chatbot).

جریان پاسخ‌دهی:
  1) Intent Router — دسته‌بندی پیام (GREETING / KNOWLEDGE / CHITCHAT)
  2) منطق شرطی — RAG فقط برای KNOWLEDGE
  3) پرامپت ترکیبی — Context + تاریخچه گفتگو

نکته: FAISS به‌صورت پیش‌فرض در RAM نگه‌داری می‌شود.
با ری‌استارت سرور داده‌ها از بین می‌روند مگر PERSIST_FAISS=true باشد.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- تنظیمات Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHAT_TEMPERATURE = float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.3"))
ROUTER_TEMPERATURE = float(os.getenv("OLLAMA_ROUTER_TEMPERATURE", "0"))

# --- حافظه گفتگو (تعداد پیام‌های اخیر در هر Session) ---
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# --- FAISS persist ---
PERSIST_FAISS = os.getenv("PERSIST_FAISS", "false").lower() in ("1", "true", "yes")
FAISS_DATA_DIR = Path(os.getenv("FAISS_DATA_DIR", "data/faiss_sessions"))

IntentLabel = Literal["GREETING", "KNOWLEDGE", "CHITCHAT"]
VALID_INTENTS: frozenset[str] = frozenset({"GREETING", "KNOWLEDGE", "CHITCHAT"})

# --- پرامپت مسیریاب Intent ---
INTENT_ROUTER_PROMPT = """تو یک مسیریاب intent هستی. پیام کاربر را دقیقاً به یکی از سه دسته زیر تقسیم کن:

GREETING — سلام، احوالپرسی، تشکر، خداحافظی، خوش‌آمدگویی
KNOWLEDGE — سوالات مربوط به کسب‌وکار، قیمت‌ها، خدمات، محصولات، سیاست‌ها، یا محتوای PDF/متن بارگذاری‌شده
CHITCHAT — سوالات عمومی، متفرقه یا شخصی درباره ربات (مثل «تو کی هستی؟»، «هوا چطوره؟»، «چطوری؟»)

قوانین سخت:
- فقط و فقط یکی از این سه کلمه را برگردان: GREETING یا KNOWLEDGE یا CHITCHAT
- هیچ توضیح، جمله یا علامت اضافه‌ای ننویس

پیام کاربر:
{question}

خروجی:"""

# --- پرامپت عمومی (GREETING / CHITCHAT) ---
GENERAL_CHAT_PROMPT = """تو یک دستیار هوشمند و صمیمی هستی. به احوالپرسی و گفتگوی عمومی به زبان طبیعی و دوستانه پاسخ بده.
اگر کاربر درباره جزئیات تخصصی کسب‌وکار پرسید، مودبانه بگو برای آن سوالات از اطلاعات شرکت کمک می‌گیری.

{history}

سوال کاربر:
{question}

پاسخ (به زبان همان سوال):"""

# --- پرامپت تخصصی RAG (KNOWLEDGE) ---
KNOWLEDGE_RAG_PROMPT = """تو دستیار هوشمند شرکت هستی. با توجه به اطلاعات دیتابیس [Context] به سوال کاربر پاسخ بده.
اگر پاسخ در متن نبود، با تکیه بر دانش خودت راهنمایی کن اما اشاره کن که اطلاعات دقیق در سند ثبت نشده است.

{history}

اطلاعات دیتابیس (Context):
{context}

سوال کاربر:
{question}

پاسخ (به زبان همان سوال):"""


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    """متن بلند را به chunkهای کوچک‌تر با هم‌پوشانی تقسیم می‌کند."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def _build_embeddings() -> OllamaEmbeddings:
    """مدل Embedding محلی Ollama."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _build_chat_model(temperature: float | None = None) -> Ollama:
    """مدل زبانی Ollama — temperature قابل تنظیم برای Router یا پاسخ‌دهی."""
    return Ollama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature if temperature is not None else CHAT_TEMPERATURE,
    )


def _format_chat_history(history: list[dict[str, str]]) -> str:
    """تاریخچه گفتگو را برای درج در پرامپت قالب‌بندی می‌کند."""
    if not history:
        return "تاریخچه گفتگو: (شروع مکالمه)"

    lines = ["تاریخچه گفتگو:"]
    for msg in history:
        role_label = "کاربر" if msg["role"] == "user" else "دستیار"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def _parse_intent(raw_output: str, has_knowledge_base: bool) -> IntentLabel:
    """
    خروجی خام مدل را به یکی از برچسب‌های معتبر تبدیل می‌کند.
    در صورت نامشخص بودن، با توجه به وجود دیتابیس fallback می‌زند.
    """
    cleaned = (raw_output or "").strip().upper()
    match = re.search(r"\b(GREETING|KNOWLEDGE|CHITCHAT)\b", cleaned)
    if match:
        return match.group(1)  # type: ignore[return-value]

    # fallback برای عیب‌یابی: اگر مدل کلمه کلیدی برنگرداند
    if has_knowledge_base:
        return "KNOWLEDGE"
    return "CHITCHAT"


@dataclass
class ChatMessage:
    """یک پیام در تاریخچه گفتگو."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class SessionState:
    """وضعیت هر Session: Vector Store، تاریخچه چت و متادیتا."""

    session_id: str
    vectorstore: FAISS | None = None
    document_count: int = 0
    chat_history: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGProcessor:
    """
    مدیریت Sessionها، مسیریاب Intent و منطق RAG شرطی.

    هر session_id یک FAISS و یک تاریخچه گفتگوی جدا دارد.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._embeddings: OllamaEmbeddings | None = None
        self._text_splitter = _build_text_splitter()
        self._chat_llm: Ollama | None = None
        self._router_llm: Ollama | None = None
        if PERSIST_FAISS:
            FAISS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = _build_embeddings()
        return self._embeddings

    @property
    def chat_llm(self) -> Ollama:
        """LLM اصلی برای تولید پاسخ (با temperature معمولی)."""
        if self._chat_llm is None:
            self._chat_llm = _build_chat_model()
        return self._chat_llm

    @property
    def router_llm(self) -> Ollama:
        """LLM سبک با temperature پایین — فقط برای Intent Router."""
        if self._router_llm is None:
            self._router_llm = _build_chat_model(temperature=ROUTER_TEMPERATURE)
        return self._router_llm

    def create_session(self) -> str:
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
        if not PERSIST_FAISS or session.vectorstore is None:
            return
        path = self._session_storage_path(session.session_id)
        path.mkdir(parents=True, exist_ok=True)
        session.vectorstore.save_local(str(path))

    def _load_vectorstore_if_exists(self, session_id: str) -> FAISS | None:
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

    def _ensure_vectorstore(self, session: SessionState) -> FAISS | None:
        """Vector Store را از RAM یا دیسک بارگذاری می‌کند (در صورت وجود)."""
        if session.vectorstore is not None:
            return session.vectorstore
        loaded = self._load_vectorstore_if_exists(session.session_id)
        if loaded is not None:
            session.vectorstore = loaded
        return session.vectorstore

    def _append_history(self, session: SessionState, role: Literal["user", "assistant"], content: str) -> None:
        """پیام جدید را به تاریخچه Session اضافه و اندازه حافظه را محدود می‌کند."""
        session.chat_history.append(ChatMessage(role=role, content=content))
        if len(session.chat_history) > MAX_HISTORY_MESSAGES:
            session.chat_history = session.chat_history[-MAX_HISTORY_MESSAGES:]

    def _history_as_dicts(self, session: SessionState) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in session.chat_history]

    def _add_chunks_to_session(self, session: SessionState, chunks: list[Document]) -> int:
        if not chunks:
            return 0

        if session.vectorstore is None:
            session.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            session.vectorstore.add_documents(chunks)

        session.document_count += len(chunks)
        self._save_vectorstore(session)
        return len(chunks)

    # --- Intent Router ---

    def _classify_intent(self, question: str, has_knowledge_base: bool) -> IntentLabel:
        """
        لایه مسیریاب: پیام کاربر را به GREETING / KNOWLEDGE / CHITCHAT تقسیم می‌کند.
        """
        prompt = INTENT_ROUTER_PROMPT.format(question=question)
        raw = self.router_llm.invoke(prompt)
        intent = _parse_intent(str(raw), has_knowledge_base=has_knowledge_base)
        return intent

    # --- پاسخ‌دهی عمومی (بدون FAISS) ---

    def _answer_general(self, session: SessionState, question: str) -> str:
        """برای GREETING و CHITCHAT — بدون جستجو در FAISS."""
        history_text = _format_chat_history(self._history_as_dicts(session))
        prompt = GENERAL_CHAT_PROMPT.format(history=history_text, question=question)
        return str(self.chat_llm.invoke(prompt)).strip()

    # --- پاسخ‌دهی تخصصی RAG ---

    def _retrieve_context(self, vectorstore: FAISS, question: str, k: int = 4) -> tuple[str, list[Document]]:
        """chunkهای مرتبط را از FAISS بازیابی و به یک رشته Context تبدیل می‌کند."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs: list[Document] = retriever.invoke(question)
        if not docs:
            context = "(هیچ متنی در دیتابیس یافت نشد.)"
        else:
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return context, docs

    def _answer_knowledge(self, session: SessionState, question: str) -> tuple[str, list[Document]]:
        """
        برای KNOWLEDGE — جستجو در FAISS + پرامپت ترکیبی Context + تاریخچه.
        """
        vectorstore = self._ensure_vectorstore(session)
        history_text = _format_chat_history(self._history_as_dicts(session))

        if vectorstore is None:
            # دیتابیس خالی: بدون RAG ولی با پرامپت تخصصی
            context = "(هنوز هیچ PDF یا متنی برای این Session بارگذاری نشده است.)"
            source_docs: list[Document] = []
        else:
            context, source_docs = self._retrieve_context(vectorstore, question)

        prompt = KNOWLEDGE_RAG_PROMPT.format(
            history=history_text,
            context=context,
            question=question,
        )
        answer = str(self.chat_llm.invoke(prompt)).strip()
        return answer, source_docs

    def _docs_to_sources(self, docs: list[Document]) -> list[dict[str, Any]]:
        """متادیتای chunkها را برای نمایش در API آماده می‌کند."""
        sources = []
        for doc in docs[:4]:
            meta = getattr(doc, "metadata", {}) or {}
            sources.append(
                {
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "preview": (doc.page_content or "")[:200],
                }
            )
        return sources

    # --- Ingest ---

    def ingest_pdf(self, session_id: str, file_bytes: bytes, filename: str) -> dict[str, Any]:
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

    # --- نقطه ورود اصلی چت ---

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        """
        جریان Router-Based:
          1) Intent Router
          2) GREETING/CHITCHAT → پاسخ عمومی (بدون FAISS)
          3) KNOWLEDGE → RAG + پرامپت ترکیبی
        """
        session = self._get_session(session_id)
        question = (question or "").strip()
        if not question:
            raise ValueError("سوال خالی است.")

        has_knowledge_base = self._ensure_vectorstore(session) is not None

        # مرحله ۱: مسیریاب Intent
        intent = self._classify_intent(question, has_knowledge_base=has_knowledge_base)

        # مرحله ۲: منطق شرطی
        sources: list[dict[str, Any]] = []
        if intent in ("GREETING", "CHITCHAT"):
            answer = self._answer_general(session, question)
        else:
            answer, source_docs = self._answer_knowledge(session, question)
            sources = self._docs_to_sources(source_docs)

        # مرحله ۳: به‌روزرسانی حافظه گفتگو
        self._append_history(session, "user", question)
        self._append_history(session, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "intent": intent,
        }

    def delete_session(self, session_id: str) -> None:
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
            "chat_history_length": len(session.chat_history),
        }


rag_processor = RAGProcessor()
