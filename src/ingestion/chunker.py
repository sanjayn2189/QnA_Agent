"""
Document chunker using LangChain's RecursiveCharacterTextSplitter.
Preserves metadata and adds chunk_index for idempotent upserts.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import logger


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.

    Each chunk preserves original metadata and gains a `chunk_index` field
    for idempotent vector store upserts.
    """
    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)

        for idx, chunk_text in enumerate(splits):
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": idx,
                "total_chunks": len(splits),
            }
            chunk = Document(
                page_content=chunk_text,
                metadata=chunk_metadata,
            )
            all_chunks.append(chunk)

        logger.debug(
            f"Chunked '{doc.metadata.get('title', 'unknown')}' → {len(splits)} chunks"
        )

    logger.info(
        f"Chunked {len(documents)} documents → {len(all_chunks)} total chunks "
        f"(size={_chunk_size}, overlap={_chunk_overlap})"
    )
    return all_chunks
