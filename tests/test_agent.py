"""
Tests for the LangGraph CRAG agent components.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.nodes import (
    query_analyzer_node,
    route_after_grading,
    route_after_analyzer,
)
from src.agent.prompts import GREETING_PATTERNS, GREETING_RESPONSE


class TestQueryAnalyzer:
    """Tests for the query analyzer node."""

    def _make_state(self, query: str) -> AgentState:
        return {
            "query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "answer": "",
            "sources": [],
            "retry_count": 0,
            "chat_history": [],
        }

    def test_detects_greeting(self):
        """Should detect 'hello' as greeting and return canned response."""
        state = self._make_state("Hello there!")
        result = query_analyzer_node(state)
        assert result["answer"] == GREETING_RESPONSE
        assert result["sources"] == []

    def test_detects_thanks(self):
        """Should detect 'thank you' as small talk."""
        state = self._make_state("Thank you for your help!")
        result = query_analyzer_node(state)
        assert result["answer"] == GREETING_RESPONSE

    def test_passes_real_query(self):
        """Should pass normal questions through without an answer."""
        state = self._make_state("What is our deployment process?")
        result = query_analyzer_node(state)
        assert "answer" not in result or result.get("answer") == ""
        assert result["query"] == "What is our deployment process?"

    def test_normalizes_whitespace(self):
        """Should strip leading/trailing whitespace from query."""
        state = self._make_state("   What is CI/CD?   ")
        result = query_analyzer_node(state)
        assert result["query"] == "What is CI/CD?"

    def test_no_false_positive_on_which(self):
        """Regression: 'Which' should NOT match greeting 'hi'."""
        state = self._make_state(
            "New hire on medical leave. Which policy takes priority?"
        )
        result = query_analyzer_node(state)
        assert "answer" not in result or result.get("answer") == ""

    def test_long_query_with_greeting_word_not_detected(self):
        """Long queries with incidental greeting words should not be greetings."""
        state = self._make_state(
            "Hi, can you tell me about the remote work policy, "
            "onboarding steps, and mandatory training requirements?"
        )
        result = query_analyzer_node(state)
        # >6 words → should NOT be treated as greeting
        assert "answer" not in result or result.get("answer") == ""


class TestRouting:
    """Tests for graph routing functions."""

    def test_route_generate_with_relevant_docs(self, sample_chunks):
        """Should route to 'generate' when relevant docs exist."""
        state = {
            "relevant_docs": sample_chunks,
            "retry_count": 0,
        }
        assert route_after_grading(state) == "generate"

    def test_route_rewrite_with_no_docs(self):
        """Should route to 'rewrite' when no relevant docs and retries remain."""
        state = {
            "relevant_docs": [],
            "retry_count": 0,
        }
        assert route_after_grading(state) == "rewrite"

    def test_route_generate_on_max_retries(self):
        """Should force 'generate' when max retries exceeded."""
        state = {
            "relevant_docs": [],
            "retry_count": 2,  # max_retry_count default = 2
        }
        assert route_after_grading(state) == "generate"

    def test_route_analyzer_greeting_done(self):
        """Should route to 'done' when answer is already set (greeting)."""
        state = {"answer": "Hello! I'm ConfluenceAssist..."}
        assert route_after_analyzer(state) == "done"

    def test_route_analyzer_query_retrieve(self):
        """Should route to 'retrieve' for normal queries."""
        state = {"answer": ""}
        assert route_after_analyzer(state) == "retrieve"

    def test_route_analyzer_no_answer_key(self):
        """Should route to 'retrieve' when answer key is missing."""
        state = {}
        assert route_after_analyzer(state) == "retrieve"
