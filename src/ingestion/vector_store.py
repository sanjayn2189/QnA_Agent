"""
ChromaDB vector store manager using langchain-chroma.
Handles upsert, retrieval, and collection management.
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from src.utils.logger import logger


class VectorStoreManager:
    """Manages ChromaDB persistent vector store operations."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._embeddings = None
            cls._instance._store = None
        return cls._instance

    def refresh(self):
        """Clear internal cache to force reconnection (e.g., after disk changes)."""
        self._store = None
        self._embeddings = None
        logger.info("VectorStoreManager refreshed — internal cache cleared")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load HuggingFace embedding model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def get_store(self) -> Chroma:
        """Get or create the ChromaDB vector store instance."""
        if self._store is None:
            self._store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            logger.info(
                f"ChromaDB store initialized: collection='{settings.chroma_collection_name}', "
                f"persist_dir='{settings.chroma_persist_dir}'"
            )
        return self._store

    def upsert_documents(self, documents: list[Document]) -> int:
        """
        Upsert documents into ChromaDB.
        Uses page_id + chunk_index as unique ID for idempotent upserts.
        """
        store = self.get_store()

        # Generate deterministic IDs for idempotent upserts
        ids = []
        for doc in documents:
            page_id = doc.metadata.get("page_id", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"{page_id}_{chunk_idx}"
            ids.append(doc_id)

        # Upsert in batches to avoid memory issues
        batch_size = 100
        total_upserted = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            store.add_documents(
                documents=batch_docs,
                ids=batch_ids,
            )
            total_upserted += len(batch_docs)
            logger.debug(
                f"Upserted batch {i // batch_size + 1}: "
                f"{len(batch_docs)} documents"
            )

        logger.info(f"Upserted {total_upserted} documents into ChromaDB")
        return total_upserted

    def as_retriever(self, k: int | None = None, search_type: str = "mmr"):
        """
        Return a LangChain retriever backed by this vector store.
        Uses MMR (Maximal Marginal Relevance) for diverse results.
        """
        _k = k or settings.retriever_k
        store = self.get_store()

        retriever = store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": _k,
                "fetch_k": _k * 3,  # fetch more candidates for MMR re-ranking
            },
        )
        logger.debug(f"Created retriever: type={search_type}, k={_k}")
        return retriever

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Direct similarity search (bypass retriever abstraction)."""
        _k = k or settings.retriever_k
        store = self.get_store()
        return store.similarity_search(query, k=_k)

    def get_stats(self) -> dict:
        """Return collection statistics."""
        try:
            store = self.get_store()
            collection = store._collection
            count = collection.count()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning("Collection not found in stats, refreshing and retrying...")
                self.refresh()
                store = self.get_store()
                collection = store._collection
                count = collection.count()
            else:
                raise e

        return {
            "collection_name": settings.chroma_collection_name,
            "document_count": count,
            "persist_directory": settings.chroma_persist_dir,
            "embedding_model": settings.embedding_model,
        }

    def clear(self) -> None:
        """Delete the entire collection (destructive!)."""
        store = self.get_store()
        store.delete_collection()
        self._store = None
        logger.warning("ChromaDB collection deleted!")
