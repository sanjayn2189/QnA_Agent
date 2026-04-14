"""
Admin API routes — ingestion, health check, stats.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    IngestRequest,
    IngestResponse,
    HealthResponse,
    StatsResponse,
)
from src.ingestion.confluence_loader import ConfluenceLoader
from src.ingestion.chunker import chunk_documents
from src.ingestion.vector_store import VectorStoreManager
from src.utils.logger import logger

router = APIRouter(tags=["admin"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_confluence(request: IngestRequest):
    """
    Trigger Confluence ingestion pipeline.
    Fetches all pages, chunks them, and upserts into ChromaDB.
    """
    logger.info(f"[/ingest] Starting ingestion (force_reload={request.force_reload})")

    try:
        vsm = VectorStoreManager()

        # Optionally clear existing data
        if request.force_reload:
            logger.warning("[/ingest] Force reload — clearing vector store")
            vsm.clear()

        # Step 1: Load from Confluence
        loader = ConfluenceLoader()
        documents = loader.load()

        if not documents:
            return IngestResponse(
                status="warning",
                pages_processed=0,
                chunks_created=0,
                message="No pages found in Confluence space",
            )

        # Step 2: Chunk documents
        chunks = chunk_documents(documents)

        # Step 3: Upsert into ChromaDB
        count = vsm.upsert_documents(chunks)

        msg = (
            f"Successfully ingested {len(documents)} pages "
            f"→ {count} chunks into ChromaDB"
        )
        logger.info(f"[/ingest] {msg}")

        return IngestResponse(
            status="success",
            pages_processed=len(documents),
            chunks_created=count,
            message=msg,
        )

    except Exception as e:
        logger.error(f"[/ingest] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vsm = VectorStoreManager()
        stats = vsm.get_stats()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_count=stats["document_count"],
        )
    except Exception as e:
        logger.error(f"[/health] Error: {e}")
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            vector_store_count=0,
        )


@router.get("/collections/stats", response_model=StatsResponse)
async def collection_stats():
    """Return ChromaDB collection statistics."""
    try:
        vsm = VectorStoreManager()
        stats = vsm.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"[/collections/stats] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
