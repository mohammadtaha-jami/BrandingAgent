# Branding Bot

<p align="center">
  <img src="docs/banner.png" alt="Branding Bot banner" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build" />
  <img src="https://img.shields.io/badge/version-2.0.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
</p>

<p align="center">
  <em>Replace <code>docs/banner.png</code> with your project logo or screenshot.</em>
</p>

Session-based RAG chatbot with an LLM intent router. Ingest PDFs or raw text, then chat against that knowledge — locally, via Ollama.

---

## About The Project

Generic chatbots either always hit a vector store (slow, noisy) or never do (no grounding). Branding Bot sits in between.

Each user gets an isolated **session** with its own FAISS index and conversation memory. An **intent router** classifies every message as `GREETING`, `KNOWLEDGE`, or `CHITCHAT`. Retrieval runs only when the question is actually about ingested content (products, prices, policies, uploaded documents). Small talk is answered without a FAISS lookup.

The API is FastAPI; a single-page RTL UI (`index.html`) is served at `/` for local demos.

---

## Key Features

- **Intent router** — classifies each turn (`GREETING` / `KNOWLEDGE` / `CHITCHAT`) before generation
- **Conditional RAG** — FAISS retrieval only on `KNOWLEDGE` intents
- **Session isolation** — UUID sessions, per-session vector store and chat history
- **PDF + text ingest** — `PyPDFLoader`, recursive chunking (1000 / 200 overlap)
- **Local LLM & embeddings** — Ollama chat model + `nomic-embed-text`
- **Conversation memory** — last N messages injected into prompts (configurable)
- **Optional FAISS persistence** — RAM by default; disk when `PERSIST_FAISS=true`
- **REST API + UI** — JSON endpoints plus a bundled chat page
- **Source previews** — knowledge answers can return chunk metadata (source, page, preview)

---

## Built With

| Layer | Stack |
| --- | --- |
| API | [![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) · [Uvicorn](https://www.uvicorn.org/) `0.32+` |
| RAG | [LangChain](https://www.langchain.com/) `0.3+` · [FAISS](https://github.com/facebookresearch/faiss) `faiss-cpu 1.9+` |
| LLM | [Ollama](https://ollama.com/) (default chat: `llama3.2:1b`) |
| Embeddings | Ollama `nomic-embed-text` |
| Documents | `pypdf` `5+` |
| Config | `python-dotenv` |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Ollama** running locally (default `http://localhost:11434`)
- Pulled models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Installation

```bash
git clone https://github.com/<org>/Branding-Bot.git
cd Branding-Bot

python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
# If execution policy blocks scripts:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Environment Setup

Copy the example below into `.env` in the project root (the file is gitignored).

```env
# Server
HOST=127.0.0.1
PORT=8000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_TEMPERATURE=0.3
OLLAMA_ROUTER_TEMPERATURE=0

# Memory
MAX_HISTORY_MESSAGES=10

# FAISS (default: in-memory; lost on restart)
PERSIST_FAISS=false
FAISS_DATA_DIR=data/faiss_sessions
```

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address for Uvicorn |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP API |
| `OLLAMA_CHAT_MODEL` | `llama3.2:1b` | Chat + router LLM |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_CHAT_TEMPERATURE` | `0.3` | Generation temperature |
| `OLLAMA_ROUTER_TEMPERATURE` | `0` | Router temperature (deterministic) |
| `MAX_HISTORY_MESSAGES` | `10` | Messages kept per session |
| `PERSIST_FAISS` | `false` | Persist indexes to disk |
| `FAISS_DATA_DIR` | `data/faiss_sessions` | Directory for persisted indexes |

---

## Usage

Start the API (reload enabled when run as `__main__`):

```bash
python main.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the UI, or call the API directly.

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### API flow

```bash
# 1. Create a session
curl -s -X POST http://127.0.0.1:8000/api/session

# 2. Ingest a PDF (header required)
curl -s -X POST http://127.0.0.1:8000/api/ingest/pdf \
  -H "X-Session-ID: <session_id>" \
  -F "file=@./brand-guide.pdf"

# 3. Or ingest raw text
curl -s -X POST http://127.0.0.1:8000/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","text":"Our support hours are 9–18."}'

# 4. Chat
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","question":"What are your support hours?"}'
```

`session_id` may be sent in the JSON body or as header `X-Session-ID`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/session` | Create session |
| `GET` | `/api/session/{id}` | Session stats |
| `POST` | `/api/ingest/pdf` | Upload PDF |
| `POST` | `/api/ingest/text` | Ingest text |
| `POST` | `/api/chat` | Ask (router + optional RAG) |
| `DELETE` | `/api/session/{id}` | Drop session (and disk index if persisted) |

Chat response shape:

```json
{
  "answer": "...",
  "sources": [],
  "intent": "KNOWLEDGE"
}
```

### Demo / screenshots

Add links or images here when available:

```markdown
![Chat UI](docs/screenshot-chat.png)
```

Live demo: _TBD_

---

## Contributing

1. Fork the repo and create a branch (`feat/...` or `fix/...`).
2. Keep changes focused; match existing Python style.
3. Open a pull request with a short description of **why** the change exists.

Bug reports should include OS, Python version, Ollama model names, and the request/response that failed.

---

## License

Distributed under the **MIT** License. See [`LICENSE`](LICENSE) (add the license file if it is not yet in the repo).

```
Copyright (c) 2026 Branding Bot contributors
```
