"""
Tests for the ingestion pipeline components.
"""

import pytest
from langchain_core.documents import Document

from src.ingestion.html_parser import parse_confluence_html
from src.ingestion.chunker import chunk_documents


class TestHTMLParser:
    """Tests for Confluence HTML parser."""

    def test_parses_basic_html(self, sample_confluence_html):
        """Should extract clean text from Confluence XHTML."""
        result = parse_confluence_html(sample_confluence_html)
        assert result  # not empty
        assert "deployment process" in result.lower()
        assert "Docker" in result or "docker" in result.lower()

    def test_removes_confluence_macros(self, sample_confluence_html):
        """Should strip ac:structured-macro tags."""
        result = parse_confluence_html(sample_confluence_html)
        assert "ac:structured-macro" not in result
        assert "ac:parameter" not in result

    def test_handles_empty_input(self):
        """Should return empty string for empty input."""
        assert parse_confluence_html("") == ""
        assert parse_confluence_html("   ") == ""
        assert parse_confluence_html(None) == ""

    def test_preserves_headings(self, sample_confluence_html):
        """Should preserve heading structure."""
        result = parse_confluence_html(sample_confluence_html)
        assert "Project Overview" in result
        assert "Steps" in result


class TestChunker:
    """Tests for document chunker."""

    def test_chunks_documents(self, sample_documents):
        """Should split documents into chunks."""
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= len(sample_documents)

    def test_preserves_metadata(self, sample_documents):
        """Should preserve original metadata in chunks."""
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert "page_id" in chunk.metadata
            assert "title" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["source"] == "confluence"

    def test_adds_chunk_index(self, sample_documents):
        """Should add chunk_index to each chunk's metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        for chunk in chunks:
            assert isinstance(chunk.metadata["chunk_index"], int)
            assert chunk.metadata["chunk_index"] >= 0

    def test_empty_input(self):
        """Should handle empty document list."""
        chunks = chunk_documents([])
        assert chunks == []

    def test_small_document_single_chunk(self):
        """Small document should produce single chunk."""
        doc = Document(
            page_content="Short text.",
            metadata={"page_id": "1", "title": "Test"},
        )
        chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1
