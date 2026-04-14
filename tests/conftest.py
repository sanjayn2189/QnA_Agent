"""
Pytest configuration and shared fixtures.
"""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_confluence_html():
    """Sample Confluence XHTML storage format content."""
    return """
    <h1>Project Overview</h1>
    <p>This is a test page about our <strong>deployment process</strong>.</p>
    <h2>Steps</h2>
    <ol>
        <li>Build the Docker image</li>
        <li>Push to registry</li>
        <li>Deploy to Kubernetes</li>
    </ol>
    <ac:structured-macro ac:name="code">
        <ac:parameter ac:name="language">bash</ac:parameter>
        <ac:plain-text-body>docker build -t myapp .</ac:plain-text-body>
    </ac:structured-macro>
    <p>For more details, contact the DevOps team.</p>
    """


@pytest.fixture
def sample_documents():
    """Sample LangChain Documents for testing."""
    return [
        Document(
            page_content=(
                "The deployment process involves building a Docker image, "
                "pushing it to our container registry, and deploying to Kubernetes. "
                "We use GitHub Actions for CI/CD automation."
            ),
            metadata={
                "page_id": "12345",
                "title": "Deployment Guide",
                "url": "https://example.atlassian.net/wiki/page/12345",
                "space_key": "ENG",
                "author": "user123",
                "last_modified": "2024-01-15T10:30:00Z",
                "source": "confluence",
            },
        ),
        Document(
            page_content=(
                "Our coding standards require all Python code to follow PEP 8. "
                "We use ruff for linting and black for formatting. "
                "All PRs must pass CI checks before merging."
            ),
            metadata={
                "page_id": "67890",
                "title": "Coding Standards",
                "url": "https://example.atlassian.net/wiki/page/67890",
                "space_key": "ENG",
                "author": "user456",
                "last_modified": "2024-02-20T14:00:00Z",
                "source": "confluence",
            },
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents):
    """Sample chunked documents with chunk_index metadata."""
    chunks = []
    for doc in sample_documents:
        chunk = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "chunk_index": 0,
                "total_chunks": 1,
            },
        )
        chunks.append(chunk)
    return chunks
