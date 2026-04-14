"""
FastAPI application — ConfluenceAssist Enterprise RAG Q&A API.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.utils.logger import logger
from src.api.routes import chat, admin

# Path to static assets
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    logger.info("=" * 60)
    logger.info("🚀 Starting ConfluenceAssist RAG API")
    logger.info(f"   LLM Model    : {settings.llm_model}")
    logger.info(f"   Embedding    : {settings.embedding_model}")
    logger.info(f"   ChromaDB     : {settings.chroma_persist_dir}")
    logger.info(f"   Confluence   : {settings.confluence_url}")
    logger.info(f"   Chat UI      : http://localhost:{settings.api_port}/chat")
    logger.info("=" * 60)

    # Warm up: pre-load the graph + vector store on startup
    try:
        from src.agent.graph import get_graph
        get_graph()
        logger.info("✅ CRAG agent graph compiled and ready")
    except Exception as e:
        logger.warning(f"⚠️  Graph warm-up failed (will retry on first request): {e}")

    yield

    logger.info("👋 Shutting down ConfluenceAssist RAG API")


# ── Create FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(
    title="ConfluenceAssist — Enterprise RAG Q&A API (V1 MVP)",
    description=(
        "V1 MVP — AI-powered Q&A over Atlassian Confluence knowledge base. "
        "Uses LangGraph CRAG agent with Groq LLaMA-3.3, HuggingFace embeddings, "
        "and ChromaDB for retrieval. No evaluation or observability in this version."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ─────────────────────────────────────────────────────────

app.include_router(chat.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")


# ── Chat UI ───────────────────────────────────────────────────────────────────


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


# ── Root Endpoint ─────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "ConfluenceAssist",
        "description": "Enterprise RAG Q&A API over Confluence",
        "version": "1.0.0",
        "chat_ui": "/chat",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

