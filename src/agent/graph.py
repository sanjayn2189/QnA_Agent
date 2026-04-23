"""
LangGraph StateGraph builder for the CRAG (Corrective-RAG) agent.

Graph flow:
  security_check → [blocked? → END] → query_analyzer
    → [greeting? → END] → retriever → relevance_grader
    → [relevant? → answer_generator → END]
    → [not relevant? → query_rewriter → retriever → ...]
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    security_check_node,
    route_after_security,
    query_analyzer_node,
    retriever_node,
    relevance_grader_node,
    query_rewriter_node,
    answer_generator_node,
    route_after_grading,
    route_after_analyzer,
)
from src.utils.logger import logger


def build_graph():
    """
    Build and compile the CRAG agent graph.
    Returns a compiled LangGraph StateGraph.
    """
    logger.info("Building CRAG agent graph...")

    workflow = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    workflow.add_node("security_check", security_check_node)
    workflow.add_node("query_analyzer", query_analyzer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("relevance_grader", relevance_grader_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("answer_generator", answer_generator_node)

    # ── Define edges ──────────────────────────────────────────────────────

    # Entry point: security check runs first on every query
    workflow.set_entry_point("security_check")

    # After security: blocked queries stop, safe queries continue
    workflow.add_conditional_edges(
        "security_check",
        route_after_security,
        {
            "blocked": END,
            "safe": "query_analyzer",
        },
    )

    # After analyzer: either greeting (done) or proceed to retriever
    workflow.add_conditional_edges(
        "query_analyzer",
        route_after_analyzer,
        {
            "retry": "query_analyzer",
            "done": END,
            "retrieve": "retriever",
        },
    )

    # Retriever → Grader
    workflow.add_edge("retriever", "relevance_grader")

    # Grader → Generate or Rewrite (conditional)
    workflow.add_conditional_edges(
        "relevance_grader",
        route_after_grading,
        {
            "retry": "relevance_grader",
            "generate": "answer_generator",
            "rewrite": "query_rewriter",
        },
    )

    # Rewriter → back to Retriever (retry loop)
    workflow.add_edge("query_rewriter", "retriever")

    # Answer Generator → END
    workflow.add_edge("answer_generator", END)

    # ── Compile ───────────────────────────────────────────────────────────
    graph = workflow.compile()
    logger.info("CRAG agent graph compiled successfully")
    return graph


# ── Cached singleton ──────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return cached compiled CRAG agent graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
