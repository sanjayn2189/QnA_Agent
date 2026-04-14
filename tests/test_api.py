"""
Tests for the FastAPI API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        """Root should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ConfluenceAssist"
        assert "version" in data

    def test_docs_available(self, client):
        """OpenAPI docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    def test_chat_rejects_empty_query(self, client):
        """Should reject empty query."""
        response = client.post(
            "/api/v1/chat",
            json={"query": ""},
        )
        assert response.status_code == 422  # validation error

    def test_chat_accepts_valid_query(self, client):
        """Should accept a valid query and return a response."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "Hello!", "session_id": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["session_id"] == "test-session"
        assert data["query"] == "Hello!"
