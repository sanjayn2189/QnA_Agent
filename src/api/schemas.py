"""
Pydantic request/response schemas for the API.
"""

from uuid import uuid4
from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Chat endpoint request body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question to answer from Confluence knowledge base",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session ID for multi-turn conversation memory",
    )


class IngestRequest(BaseModel):
    """Ingestion endpoint request body."""

    force_reload: bool = Field(
        default=False,
        description="If True, clear existing embeddings and re-ingest from scratch",
    )


# ─── Response Models ─────────────────────────────────────────────────────────


class SourceDoc(BaseModel):
    """A source Confluence page referenced in the answer."""

    title: str
    url: str
    page_id: str


class ChatResponse(BaseModel):
    """Chat endpoint response."""

    answer: str
    sources: list[SourceDoc]
    session_id: str
    query: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Ingestion endpoint response."""

    status: str
    pages_processed: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_store_count: int


class StatsResponse(BaseModel):
    """Vector store statistics response."""

    collection_name: str
    document_count: int
    persist_directory: str
    embedding_model: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
