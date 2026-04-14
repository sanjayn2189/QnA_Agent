"""
Centralized application settings loaded from environment variables.
Uses pydantic-settings for type-safe validation.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration — all values loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Confluence ────────────────────────────────────────────────────────
    confluence_url: str
    confluence_email: str
    confluence_api_token: str
    confluence_space_key: str

    # ── Groq ─────────────────────────────────────────────────────────────
    groq_api_key: str

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "confluence_kb"

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # ── Retrieval ─────────────────────────────────────────────────────────
    retriever_k: int = 6
    max_retry_count: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def confluence_wiki_url(self) -> str:
        """Full URL to Confluence wiki base."""
        return f"{self.confluence_url}/wiki"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()


settings = get_settings()
