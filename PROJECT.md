# ConfluenceAssist — Project Blueprint & Living Reference

> **This document is the single source of truth for the ConfluenceAssist project.**
> Any AI coding agent working on this codebase MUST read this document first, follow
> the established patterns, and update the relevant sections after completing a task.

---

## 1. Project Goal

Build a **production-ready, agentic Retrieval-Augmented Generation (RAG) Q&A system**
for enterprise use. The agent ingests content from **Atlassian Confluence** pages,
stores semantically chunked embeddings in **ChromaDB**, and answers user questions
using a **LangGraph-orchestrated CRAG (Corrective-RAG) agent** powered by
**Groq's LLaMA-3.3-70b-versatile** model.

The system must be:
- **Accurate**: answers derived ONLY from retrieved Confluence context.
- **Self-correcting**: automatically rewrites poor queries and retries retrieval.
- **Observable**: every response includes performance metrics and a confidence score.
- **Resilient**: gracefully handles stale collections, missing data, and API errors.

---

## 2. System Architecture (High-Level Design)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE (Offline)                  │
│                                                                      │
│  Confluence REST API v2                                              │
│       ↓                                                              │
│  ConfluenceLoader  →  HTML Parser (BS4)  →  Recursive Chunker       │
│       ↓                                        ↓                     │
│  Raw HTML pages         Clean markdown       1000-char chunks        │
│                                               (10% overlap)          │
│                                                  ↓                   │
│                                          HuggingFace Embeddings      │
│                                          (all-MiniLM-L6-v2, local)   │
│                                                  ↓                   │
│                                          ChromaDB (persistent)       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    CRAG AGENT (Online, per query)                     │
│                                                                      │
│  User Question                                                       │
│       ↓                                                              │
│  ┌─────────────────┐                                                 │
│  │ query_analyzer   │ → [Greeting?] → Canned response → END         │
│  └────────┬────────┘                                                 │
│           ↓                                                          │
│  ┌─────────────────┐                                                 │
│  │   retriever      │ → MMR search in ChromaDB (k=6)                │
│  └────────┬────────┘                                                 │
│           ↓                                                          │
│  ┌─────────────────┐                                                 │
│  │ relevance_grader │ → LLM grades each doc (relevant/not)          │
│  └────────┬────────┘                                                 │
│      ┌────┴────┐                                                     │
│      ↓         ↓                                                     │
│  [relevant]  [not relevant]                                          │
│      ↓         ↓                                                     │
│  answer_gen  query_rewriter → retriever (retry, max 2x)             │
│      ↓                                                               │
│  Final Answer + Sources + Confidence Score + Performance Stats       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         API & UI LAYER                                │
│                                                                      │
│  FastAPI Server (:8000)                                              │
│    ├─ GET  /chat              → Chat UI (single-page HTML/JS)        │
│    ├─ POST /api/v1/chat       → Chat API (returns answer + metadata) │
│    ├─ POST /api/v1/ingest     → Trigger Confluence ingestion         │
│    ├─ GET  /api/v1/health     → Health check                         │
│    ├─ GET  /api/v1/collections/stats → ChromaDB stats                │
│    └─ GET  /docs              → OpenAPI Swagger docs                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Layer               | Technology                              | Notes                                          |
| ------------------- | --------------------------------------- | ---------------------------------------------- |
| Agent Orchestration | **LangGraph** ≥0.2                      | StateGraph with conditional edges (CRAG)       |
| LLM (Chat + Grade)  | **ChatGroq** `llama-3.3-70b-versatile` | Temperature=0.1                                |
| Embeddings          | **HuggingFace** `all-MiniLM-L6-v2`     | Local, free, no API key needed                 |
| Vector Database     | **ChromaDB** (persistent)               | `langchain-chroma` wrapper                     |
| Data Source         | **Confluence REST API v2**              | `requests` + HTTPBasicAuth                     |
| HTML Parsing        | **BeautifulSoup4** + `markdownify`      | Strip XHTML → clean markdown                   |
| API Server          | **FastAPI** + **Uvicorn**               | Async, auto-generated OpenAPI docs             |
| Chat UI             | **Vanilla HTML/CSS/JS**                 | Single file, dark theme, Inter font            |
| Configuration       | **pydantic-settings**                   | `.env` loaded, type-validated singleton         |
| Logging             | **loguru**                              | Structured, colored, file + stdout              |
| Testing             | **pytest** + **pytest-asyncio**         | Unit + integration tests                       |

---

## 4. Directory Structure

```
Q&A_Agent/
├── .env                           ← Secrets (gitignored)
├── .env.example                   ← Template (committed to git)
├── .gitignore
├── requirements.txt               ← Python dependencies
├── pyproject.toml                 ← Project metadata
├── Makefile                       ← Common commands (run, test, ingest)
├── README.md                      ← User-facing README
├── PROJECT.md                     ← THIS FILE — living project blueprint
├── DEVELOPMENT_PROMPT.md          ← Original build prompt (reference only)
│
├── config/
│   ├── __init__.py
│   └── settings.py                ← pydantic-settings BaseSettings singleton
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── confluence_loader.py   ← Confluence REST API v2 client
│   │   ├── html_parser.py         ← BS4 + markdownify XHTML → markdown
│   │   ├── chunker.py             ← RecursiveCharacterTextSplitter
│   │   └── vector_store.py        ← ChromaDB Singleton wrapper
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               ← AgentState TypedDict
│   │   ├── prompts.py             ← All LLM prompts (system + node)
│   │   ├── nodes.py               ← LangGraph node functions (5 nodes)
│   │   └── graph.py               ← StateGraph builder + compile
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                ← FastAPI app + lifespan + CORS
│   │   ├── schemas.py             ← Pydantic request/response models
│   │   ├── static/
│   │   │   └── index.html         ← Chat UI (single file)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py            ← POST /api/v1/chat
│   │       └── admin.py           ← POST /ingest, GET /health, GET /stats
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              ← loguru setup
│       └── text_utils.py          ← HTML stripping helpers
│
├── scripts/
│   └── ingest.py                  ← CLI: python scripts/ingest.py [--force]
│
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_agent.py
│   └── test_api.py
│
└── data/
    └── chroma_db/                 ← Persisted vectors (gitignored)
```

---

## 5. Core Design Patterns & Decisions

### 5.1 CRAG (Corrective-RAG) Pattern
The agent uses a self-correcting retrieval loop:
1. Retrieve documents using MMR.
2. Grade each document for relevance using the LLM.
3. If relevant docs found → generate answer.
4. If no relevant docs AND retries remain → rewrite query → go to step 1.
5. If max retries (2) exceeded → generate best-effort answer from whatever was retrieved.

### 5.2 Singleton Pattern
- **VectorStoreManager**: Uses `__new__` to guarantee a single shared instance across ingestion and chat modules. This prevents the "stale collection" error that occurs when one module deletes a ChromaDB collection and another still holds the old reference.
- **LLM**: Module-level singleton in `nodes.py` (`_get_llm()`).
- **Graph**: Module-level singleton in `graph.py` (`get_graph()`).
- **Settings**: `lru_cache` singleton in `settings.py`.

### 5.3 Performance Observability
Every chat response includes a `metadata` object:
```json
{
  "retrieval_count": 12,
  "retrieval_time": 2.46,
  "grading_time": 1.84,
  "generation_time": 1.37,
  "total_time": 5.68,
  "confidence_score": 90
}
```
These are tracked across nodes using `time.time()` and accumulated in the `AgentState.metadata` dict.

### 5.4 Confidence Score
The LLM self-assesses its confidence (0-100%) based on context support. This is extracted via regex from a `[CONFIDENCE: XX%]` tag appended by the LLM, stripped from the displayed answer, and shown in the UI with color coding:
- 🟢 Green (≥80%): High confidence
- 🟠 Orange (50-79%): Moderate confidence
- 🔴 Red (<50%): Low confidence

### 5.5 Error Resilience
- **ChromaDB auto-recovery**: If retriever or stats queries encounter a "Collection does not exist" error, the system automatically calls `vsm.refresh()` and retries once.
- **Grading fail-safe**: If the LLM grader returns unparseable JSON, the document is included as relevant (fail-open).
- **Greeting detection**: Uses regex word-boundary matching + length guard (≤6 words) to prevent false positives (e.g., "which" matching "hi").

### 5.6 Session Memory
In-memory `SESSION_STORE` (dict: `session_id → list[BaseMessage]`), capped at 20 messages (10 turns). Resets on server restart.

### 5.7 Chunking Strategy
- **Splitter**: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.
- **Chunk size**: 1000 characters.
- **Overlap**: 100 characters (10%).
- **Idempotent upserts**: Each chunk ID = `{page_id}_{chunk_index}`.

---

## 6. Configuration Reference

All settings are loaded from `.env` via `config/settings.py`:

| Variable                 | Default                      | Description                        |
| ------------------------ | ---------------------------- | ---------------------------------- |
| `CONFLUENCE_URL`         | (required)                   | Atlassian instance URL             |
| `CONFLUENCE_EMAIL`       | (required)                   | Atlassian account email            |
| `CONFLUENCE_API_TOKEN`   | (required)                   | Atlassian API token                |
| `CONFLUENCE_SPACE_KEY`   | (required)                   | Confluence space key or ID         |
| `GROQ_API_KEY`           | (required)                   | Groq API key                       |
| `LLM_MODEL`             | `llama-3.3-70b-versatile`   | Groq model name                    |
| `LLM_TEMPERATURE`       | `0.1`                        | LLM temperature                    |
| `EMBEDDING_MODEL`       | `all-MiniLM-L6-v2`          | HuggingFace embedding model        |
| `CHROMA_PERSIST_DIR`    | `./data/chroma_db`           | ChromaDB persistence directory     |
| `CHROMA_COLLECTION_NAME`| `confluence_kb`              | ChromaDB collection name           |
| `API_HOST`              | `0.0.0.0`                    | FastAPI host                       |
| `API_PORT`              | `8000`                       | FastAPI port                       |
| `RETRIEVER_K`           | `6`                          | Number of docs to retrieve         |
| `MAX_RETRY_COUNT`       | `2`                          | Max query rewrite retries          |
| `CHUNK_SIZE`            | `1000`                       | Chunk size in characters           |
| `CHUNK_OVERLAP`         | `100`                        | Chunk overlap in characters (10%)  |

---

## 7. Prompt Engineering Rules

The system prompt (`SYSTEM_PROMPT`) in `src/agent/prompts.py` enforces these behaviors:

1. **Context-first**: All answers derived from retrieved Confluence documents only.
2. **Cite sources**: Every answer ends with `📄 Source: [Page Title] — [URL]`.
3. **Acknowledge uncertainty**: Never guess — state what's missing.
4. **Preserve accuracy**: Quote directly when precision matters.
5. **Be structured**: Use markdown formatting.
6. **Respect scope**: Never reveal internals.
7. **Multi-turn memory**: Reference prior turns when relevant.
8. **Cross-policy synthesis**: Synthesize across overlapping policy documents.
9. **Temporal accuracy**: Don't say obligations "still apply" during suspension periods.
10. **Enforcement precision**: Scope corrective actions to their exact trigger conditions.
11. **Decision clarity**: State priority order before explanation in multi-step scenarios.

The `ANSWER_PROMPT` additionally requires:
- A `[CONFIDENCE: XX%]` tag at the end of every response (extracted and stripped by the backend).

---

## 8. API Contract

### POST /api/v1/chat

**Request:**
```json
{
  "query": "What is our deployment process?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "Based on the Confluence documentation...",
  "sources": [
    { "title": "Deployment Guide", "url": "https://...", "page_id": "12345" }
  ],
  "session_id": "uuid",
  "query": "What is our deployment process?",
  "metadata": {
    "retrieval_count": 6,
    "retrieval_time": 0.52,
    "grading_time": 1.84,
    "generation_time": 1.37,
    "total_time": 3.74,
    "confidence_score": 90
  }
}
```

### POST /api/v1/ingest
**Request:** `{ "force_reload": true }`
**Response:** `{ "status": "success", "pages_processed": 6, "chunks_created": 32, "message": "..." }`

### GET /api/v1/health
**Response:** `{ "status": "healthy", "version": "1.0.0", "vector_store_count": 32 }`

---

## 9. Critical Rules for AI Agents

> **ANY AI CODING AGENT MUST FOLLOW THESE RULES when modifying this codebase.**

1. **NEVER hardcode API keys** — always read from `settings.*`.
2. **NEVER use `langchain_community.vectorstores.Chroma`** — use `langchain_chroma.Chroma`.
3. **NEVER call `vector_store.persist()`** — langchain-chroma auto-persists.
4. **ALWAYS add `chunk_index` to chunk metadata** for idempotent upserts.
5. **ALWAYS use `VectorStoreManager()` singleton** — never create raw `Chroma()` instances outside the manager.
6. **Use `with_structured_output` or JSON parsing** for structured LLM responses — not string parsing.
7. **All FastAPI endpoints must use `async def`**.
8. **Add `__init__.py` to every new package directory**.
9. **Run `pytest tests/ -v` before considering any task complete**.
10. **Update this PROJECT.md** after completing any significant feature or architectural change.

---

## 10. Achievements Log

This section tracks the history of completed milestones. **Update this after every completed task.**

| Date       | Milestone                                             | Details                                                                                                        |
| ---------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 2026-04-07 | **V1 MVP Released**                                   | Full CRAG pipeline, Confluence ingestion, FastAPI API, Chat UI, 26 tests passing.                              |
| 2026-04-07 | **Prompt: Cross-Policy Synthesis**                    | Added Rule #8 to force synthesis across overlapping policy documents.                                          |
| 2026-04-07 | **Prompt: Temporal Accuracy**                         | Added Rule #9 to prevent hallucinating obligations during leave/suspension periods.                             |
| 2026-04-07 | **Prompt: Enforcement Precision**                     | Added Rule #10 to scope corrective actions to their exact trigger conditions.                                  |
| 2026-04-10 | **Prompt: Decision Clarity**                          | Added Rule #11 to force explicit priority ordering in multi-step scenarios.                                    |
| 2026-04-10 | **Greeting Detection Fix**                            | Fixed false positive where "Which" triggered greeting detector. Added regex word-boundary + length guard.      |
| 2026-04-10 | **Performance Observability**                         | Added `metadata` dict to AgentState tracking retrieval count, retrieval/grading/generation times.              |
| 2026-04-10 | **Retrieval Time Tracking**                           | Added explicit `retrieval_time` metric to explain the gap between component times and total time.              |
| 2026-04-10 | **Confidence Score**                                  | LLM self-assesses confidence (0-100%). Extracted via regex, displayed with green/orange/red color coding.      |
| 2026-04-10 | **ChromaDB Singleton & Auto-Recovery**                | Implemented `__new__` singleton for VectorStoreManager. Added auto-refresh on "Collection does not exist".     |
| 2026-04-10 | **Chunk Overlap Reduced**                             | Changed from 200 (20%) to 100 (10%) overlap. Re-ingested all documents.                                       |

---

## 11. Known Limitations & Future Roadmap

### Current Limitations
- **Session memory is in-memory** — resets on server restart. Not suitable for multi-instance deployments.
- **No scheduled ingestion** — Confluence sync is manual (API call or CLI).
- **No page deletion reconciliation** — Deleted Confluence pages remain in the vector store until `force_reload`.
- **CORS is wide open** — `allow_origins=["*"]` must be restricted for production.
- **No authentication** — API endpoints are publicly accessible.

### Planned Improvements (Backlog)

| Priority | Feature                                  | Description                                                                     |
| -------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| P1       | **Redis Session Store**                  | Migrate `SESSION_STORE` from in-memory dict to Redis for persistence & scaling. |
| P1       | **Automated Daily Sync**                 | Background scheduler (APScheduler) for nightly Confluence re-ingestion.         |
| P1       | **Page Deletion Reconciliation**         | Compare Confluence page IDs with ChromaDB IDs, delete orphans.                  |
| P2       | **Docker Deployment**                    | `Dockerfile` + `docker-compose.yml` for containerized deployment.               |
| P2       | **CORS Restriction**                     | Whitelist specific origins instead of `*`.                                      |
| P2       | **API Authentication**                   | API key or OAuth2 for endpoint protection.                                      |
| P3       | **LangSmith Tracing**                    | Re-enable LangSmith observability for production debugging.                     |
| P3       | **Cross-Encoder Re-ranking**             | Replace LLM-based grader with a lightweight Cross-Encoder for faster grading.   |
| P3       | **Streaming Responses**                  | SSE-based token streaming for real-time answer rendering.                       |

---

## 12. Quick Reference — Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest Confluence data
python scripts/ingest.py          # incremental
python scripts/ingest.py --force  # full re-ingest (clears existing data)

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/ -v

# Or use the Makefile
make install
make ingest
make run
make test
```

---

*Last updated: 2026-04-14 | Version: 1.1.0 | Maintainer: Ramya Velaga*
