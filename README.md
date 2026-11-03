# RAG Backend (FastAPI)

Two REST APIs: **document ingestion** and **conversational RAG with interview booking**.
No FAISS/Chroma, no `RetrievalQAChain`, no UI — the RAG pipeline is hand-written.

| Concern | Choice |
|---|---|
| API | FastAPI (async) |
| Vector DB | Qdrant (embedded local mode by default; server/Cloud via `QDRANT_URL`) |
| Metadata + bookings | SQLite via async SQLAlchemy (swap `DATABASE_URL` for Postgres) |
| Chat memory | Redis (capped list per session, TTL) |
| LLM + embeddings | Google Gemini free tier via its OpenAI-compatible API (`gemini-3.5-flash`, `gemini-embedding-001`). Any OpenAI-compatible provider works by changing `LLM_BASE_URL` and the model settings |

## Run locally

1. Redis: `brew install redis && brew services start redis` (check: `redis-cli ping` -> `PONG`)
2. Free Gemini key: https://aistudio.google.com/apikey
3. ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env          # paste your key as LLM_API_KEY
   uvicorn app.main:app --reload
   ```

Docs: http://localhost:8000/docs. Qdrant data is stored in `./qdrant_data` (no server needed).
`docker compose up -d` still starts Qdrant + Redis servers if you prefer; then set
`QDRANT_URL=http://localhost:6333`. If you change the embedding model, set
`EMBEDDING_DIMENSIONS` to its output size and use a fresh `QDRANT_PATH`/collection.

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
