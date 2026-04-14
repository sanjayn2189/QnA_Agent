# ConfluenceAssist — Enterprise RAG Q&A Agent (V1 MVP)

> **V1.0.0** — AI-powered Q&A over your Atlassian Confluence knowledge base using LangGraph, Groq (LLaMA-3.3), and ChromaDB.

> 📘 **For developers & AI agents**: See [`PROJECT.md`](PROJECT.md) for the full system design, architecture, design decisions, and living changelog.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────┐
│                 INGESTION PIPELINE                   │
│  Confluence API → HTML Parser → Chunker → ChromaDB  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               LANGGRAPH CRAG AGENT                   │
│                                                      │
│  query_analyzer → retriever → relevance_grader       │
│       │               │               │              │
│    [greeting?]    [ChromaDB]    [relevant?]           │
│       ↓               │           ↙     ↘            │
│      END              │    answer_gen  query_rewriter │
│                       │        ↓           ↓         │
│                       │       END     → retriever     │
└─────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| Agent | **LangGraph** (CRAG pattern) |
| LLM | **Groq** — llama-3.3-70b-versatile |
| Embeddings | **HuggingFace** — all-MiniLM-L6-v2 (local, free) |
| Vector DB | **ChromaDB** (persistent) |
| API | **FastAPI** |
| Data Source | Atlassian Confluence REST API v2 |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/confluence-assist.git
cd confluence-assist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

**Required credentials:**

| Variable | How to Get |
|---|---|
| `CONFLUENCE_API_TOKEN` | [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens) → Create API Token |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) → Create API Key |

### 3. Ingest Confluence Data

```bash
python scripts/ingest.py
# or
make ingest

# Force re-ingest (clears existing data):
python scripts/ingest.py --force
```

### 4. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# or
make run
```

### 5. Query the Agent

```bash
# Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What topics are covered in the Confluence space?", "session_id": "demo"}'

# Follow-up question (same session)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more about the first topic", "session_id": "demo"}'
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/docs` | Interactive OpenAPI docs |
| `POST` | `/api/v1/chat` | Ask a question |
| `POST` | `/api/v1/ingest` | Trigger ingestion |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/collections/stats` | ChromaDB stats |

### Chat Request / Response

**Request:**
```json
{
  "query": "What is our deployment process?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "Based on the Confluence documentation...",
  "sources": [
    {
      "title": "Deployment Guide",
      "url": "https://your-domain.atlassian.net/wiki/...",
      "page_id": "12345"
    }
  ],
  "session_id": "abc-123",
  "query": "What is our deployment process?"
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# or
make test
```

---

## 📁 Project Structure

```
├── config/settings.py          # Pydantic-settings configuration
├── src/
│   ├── ingestion/
│   │   ├── confluence_loader.py  # Confluence REST API v2 client
│   │   ├── html_parser.py        # XHTML → clean text
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter
│   │   └── vector_store.py       # ChromaDB wrapper
│   ├── agent/
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── prompts.py            # System + node prompts
│   │   ├── nodes.py              # LangGraph node functions
│   │   └── graph.py              # StateGraph builder
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Request/response models
│   │   └── routes/               # chat.py, admin.py
│   └── utils/                    # logger, text helpers
├── scripts/ingest.py             # CLI ingestion
├── tests/                        # pytest test suite
├── .env.example                  # Environment template
└── requirements.txt              # Python dependencies
```

---

## ⚙️ How It Works

1. **Ingestion**: Fetches all pages from your Confluence space via REST API v2, parses the XHTML content, splits into ~1000-char chunks with overlap, embeds using HuggingFace's all-MiniLM-L6-v2, and stores in ChromaDB.

2. **Query Processing (CRAG)**: When a user asks a question:
   - **Query Analyzer** detects greetings vs. real queries
   - **Retriever** searches ChromaDB using MMR for diverse results
   - **Relevance Grader** uses the LLM to grade each retrieved document
   - If docs are relevant → **Answer Generator** produces a cited answer
   - If docs are irrelevant → **Query Rewriter** rewrites and retries (max 2x)

3. **Conversation Memory**: In-memory session store maintains chat history per `session_id` for multi-turn conversations.

---

## 📄 License

MIT
