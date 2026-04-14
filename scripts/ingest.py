#!/usr/bin/env python3
"""
CLI script to ingest Confluence pages into ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --force    # clear and re-ingest
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger
from src.ingestion.confluence_loader import ConfluenceLoader
from src.ingestion.chunker import chunk_documents
from src.ingestion.vector_store import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Confluence pages into ChromaDB vector store"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing embeddings and re-ingest from scratch",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("📥 Confluence Ingestion Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    # Initialize components
    vsm = VectorStoreManager()

    if args.force:
        logger.warning("🗑️  Force mode — clearing existing vector store")
        try:
            vsm.clear()
        except Exception:
            logger.info("No existing collection to clear")

    # Step 1: Load pages from Confluence
    logger.info("\n📄 Step 1: Fetching pages from Confluence...")
    loader = ConfluenceLoader()

    try:
        documents = loader.load()
    except Exception as e:
        logger.error(f"❌ Failed to load from Confluence: {e}")
        sys.exit(1)

    if not documents:
        logger.warning("⚠️  No pages found in Confluence space")
        sys.exit(0)

    logger.info(f"   Loaded {len(documents)} pages")

    # Step 2: Chunk documents
    logger.info("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"   Created {len(chunks)} chunks")

    # Step 3: Embed and upsert into ChromaDB
    logger.info("\n💾 Step 3: Embedding and upserting into ChromaDB...")
    count = vsm.upsert_documents(chunks)

    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Ingestion Complete!")
    logger.info(f"   Pages processed : {len(documents)}")
    logger.info(f"   Chunks created  : {count}")
    logger.info(f"   Time elapsed    : {elapsed:.1f}s")

    stats = vsm.get_stats()
    logger.info(f"   Total in DB     : {stats['document_count']} documents")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
