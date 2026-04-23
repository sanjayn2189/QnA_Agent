"""
Agent state definition for the LangGraph CRAG workflow.
"""

from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state flowing through the LangGraph CRAG agent.

    Each node reads from and writes to this state dict.
    """

    # User's original question
    query: str

    # Rewritten query (set by query_rewriter_node if grading fails)
    rewritten_query: str

    # Raw retrieved documents from ChromaDB
    retrieved_docs: list[Document]

    # Documents graded as relevant by the relevance_grader
    relevant_docs: list[Document]

    # Final generated answer text
    answer: str

    # Source citations extracted from relevant_docs metadata
    sources: list[dict]

    # Number of query-rewrite retries performed (guard: max 2)
    retry_count: int

    # Multi-turn conversation history
    chat_history: list[BaseMessage]

    # Performance metadata (times, counts, etc.)
    metadata: dict

    # Security flag — set True by security_check node if prompt injection detected
    is_malicious: bool

    # Query category set by query_analyzer: GREETING | INTERNAL_QUERY | OFF_TOPIC
    query_category: str

    # Number of validation retries for structured LLM output (guard: max 1)
    validation_retry_count: int

    # Parsing or schema validation errors for the LLM to learn from
    errors: list[str]
