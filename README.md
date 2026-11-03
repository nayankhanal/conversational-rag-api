# RAG Backend (FastAPI)

Two REST APIs: **document ingestion** and **conversational RAG with interview booking**.
No FAISS/Chroma, no `RetrievalQAChain`, no UI — the RAG pipeline is hand-written.

| Concern | Choice |
|---|---|
| API | FastAPI (async) |
| Vector DB | Qdrant |
| Metadata + bookings | SQLite via async SQLAlchemy (swap `DATABASE_URL` for Postgres) |
| Chat memory | Redis (capped list per session, TTL) |
| LLM + embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |

## Run

```bash
docker compose up -d                 # Qdrant + Redis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                 # set OPENAI_API_KEY
.venv/bin/uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## APIs

### 1. Ingestion — `POST /api/v1/documents` (multipart)

| Field | Notes |
|---|---|
| `file` | `.pdf` or `.txt` (max 10 MB) |
| `strategy` | `fixed` (sliding character window) or `recursive` (paragraph → line → sentence → word, greedy packing). Default `recursive` |
| `chunk_size` | 100–8000, default 1000 |
| `chunk_overlap` | default 150, must be < `chunk_size` |

Flow: extract text → chunk → embed → upsert into Qdrant → save metadata row (filename, strategy,
chunk counts, …). Also: `GET /api/v1/documents`, `GET /api/v1/documents/{id}`.

### 2. Conversational RAG — `POST /api/v1/chat`

```json
{"session_id": "optional", "message": "What does the doc say about refunds?"}
```

Each turn:
1. Load history + any in-progress booking draft from Redis.
2. One LLM call (JSON mode) classifies the turn (question vs. booking), rewrites the question as a
   standalone query (multi-turn), and extracts booking slots (name, email, date, time).
3. **Question** → embed → Qdrant top-k → grounded answer from context + history, with `sources`.
4. **Booking** → slots are merged into the draft across turns; the bot asks for what's missing.
   When complete, details are validated (email format, date not in the past) and stored in the
   `bookings` table; the draft is cleared.
5. Both messages are appended to Redis history.

Also: `GET /api/v1/chat/{session_id}/history`, `DELETE /api/v1/chat/{session_id}`.

## Layout

```
app/
  api/         routes + FastAPI dependencies
  core/        settings, DI container (lifespan), exceptions
  db/          SQLAlchemy models + engine
  repositories/ data access (documents, bookings)
  schemas/     Pydantic request/response models
  services/    chunking, extraction, embeddings, vector store, memory, llm, ingestion, chat
tests/
```

Services depend on `Protocol`s (`services/protocols.py`), so the vector store, memory or LLM can be
swapped without touching the pipeline; tests use in-memory fakes.

## Quality

```bash
.venv/bin/pytest && .venv/bin/mypy app && .venv/bin/ruff check app tests
```
