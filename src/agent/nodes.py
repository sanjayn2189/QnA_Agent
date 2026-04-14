"""
LangGraph node functions for the CRAG (Corrective-RAG) agent.
Each node reads from AgentState and returns a partial state update dict.
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from config.settings import settings
from src.utils.logger import logger
from src.ingestion.vector_store import VectorStoreManager
from src.agent.prompts import (
    SYSTEM_PROMPT,
    ANSWER_PROMPT,
    GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    GREETING_PATTERNS,
    GREETING_RESPONSE,
)
from src.agent.state import AgentState


# ─── Shared LLM instances (module-level singletons) ─────────────────────────

_llm = None
_vector_store_manager = None


def _get_llm() -> ChatGroq:
    """Get or create the ChatGroq LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            max_retries=2,
        )
    return _llm


def _get_vector_store() -> VectorStoreManager:
    """Get or create the VectorStoreManager singleton."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


# ─── Node 1: Query Analyzer ─────────────────────────────────────────────────


def query_analyzer_node(state: AgentState) -> dict[str, Any]:
    """
    Analyze the incoming query:
    - Detect greetings/small-talk → return canned response
    - Otherwise normalize and pass through
    """
    import re

    query = state["query"].strip()
    logger.info(f"[query_analyzer] Processing: '{query}'")

    # Check for greeting patterns using word boundaries (not substring!)
    query_lower = query.lower()
    is_greeting = False
    for pattern in GREETING_PATTERNS:
        # Match whole words/phrases only — prevents "hi" matching "which"
        if re.search(rf'\b{re.escape(pattern)}\b', query_lower):
            # Extra guard: if query is long (>6 words), it's likely a real question
            if len(query_lower.split()) <= 6:
                is_greeting = True
                break

    if is_greeting:
        logger.info("[query_analyzer] Detected greeting/small-talk")
        return {
            "query": query,
            "answer": GREETING_RESPONSE,
            "sources": [],
            "retrieved_docs": [],
            "relevant_docs": [],
        }

    return {
        "query": query,
        "rewritten_query": "",
        "retry_count": state.get("retry_count", 0),
        "metadata": state.get("metadata", {
            "retrieval_count": 0,
            "retrieval_time": 0.0,
            "grading_time": 0.0,
            "generation_time": 0.0,
        }),
    }


# ─── Node 2: Retriever ──────────────────────────────────────────────────────


def retriever_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve relevant documents from ChromaDB using MMR search.
    Uses rewritten_query if available, otherwise original query.
    """
    # Use rewritten query if a rewrite happened
    search_query = state.get("rewritten_query") or state["query"]
    logger.info(f"[retriever] Searching for: '{search_query}'")

    start_time = time.time()

    vsm = _get_vector_store()
    retriever = vsm.as_retriever(k=settings.retriever_k, search_type="mmr")

    try:
        docs = retriever.invoke(search_query)
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning("[retriever] Collection missing, refreshing and retrying...")
            vsm.refresh()
            retriever = vsm.as_retriever(k=settings.retriever_k, search_type="mmr")
            docs = retriever.invoke(search_query)
        else:
            raise e

    metadata = state.get("metadata", {})
    metadata["retrieval_time"] = metadata.get("retrieval_time", 0.0) + (time.time() - start_time)
    metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + len(docs)

    logger.info(f"[retriever] Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "unknown")
        logger.debug(
            f"  [{i + 1}] '{title}' — {len(doc.page_content)} chars"
        )

    metadata = state.get("metadata", {})
    metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + len(docs)

    return {"retrieved_docs": docs, "metadata": metadata}


# ─── Node 3: Relevance Grader ───────────────────────────────────────────────


def relevance_grader_node(state: AgentState) -> dict[str, Any]:
    """
    Grade each retrieved document for relevance to the query.
    Uses LLM with structured JSON output.
    """
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        logger.warning("[relevance_grader] No documents to grade")
        return {"relevant_docs": []}

    start_time = time.time()

    llm = _get_llm()
    relevant_docs: list[Document] = []

    logger.info(f"[relevance_grader] Grading {len(docs)} documents")

    for doc in docs:
        title = doc.metadata.get("title", "unknown")
        prompt = GRADER_PROMPT.format(
            question=query,
            document=doc.page_content[:1500],  # truncate for grading
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse JSON response — handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            grade = json.loads(response_text)
            is_relevant = grade.get("relevant", False)
            reason = grade.get("reason", "No reason given")

            if is_relevant:
                relevant_docs.append(doc)
                logger.debug(f"  ✅ '{title}': {reason}")
            else:
                logger.debug(f"  ❌ '{title}': {reason}")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"  ⚠️ Failed to grade '{title}': {e}. Including as relevant."
            )
            relevant_docs.append(doc)  # fail-safe: include on error

    logger.info(
        f"[relevance_grader] {len(relevant_docs)}/{len(docs)} documents are relevant"
    )

    metadata = state.get("metadata", {})
    metadata["grading_time"] = metadata.get("grading_time", 0.0) + (time.time() - start_time)

    return {"relevant_docs": relevant_docs, "metadata": metadata}


# ─── Node 4: Query Rewriter ─────────────────────────────────────────────────


def query_rewriter_node(state: AgentState) -> dict[str, Any]:
    """
    Rewrite the query to be more specific and retrieval-optimized.
    Called when relevance grading finds insufficient relevant docs.
    """
    original_query = state["query"]
    retry_count = state.get("retry_count", 0)

    logger.info(
        f"[query_rewriter] Rewriting query (attempt {retry_count + 1})"
    )

    llm = _get_llm()
    prompt = QUERY_REWRITER_PROMPT.format(question=original_query)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()

    logger.info(f"[query_rewriter] Original: '{original_query}'")
    logger.info(f"[query_rewriter] Rewritten: '{rewritten}'")

    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count + 1,
    }


# ─── Node 5: Answer Generator ───────────────────────────────────────────────


def answer_generator_node(state: AgentState) -> dict[str, Any]:
    """
    Generate the final answer using relevant docs as context.
    Falls back to retrieved_docs if no relevant_docs (after max retries).
    """
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])
    retrieved_docs = state.get("retrieved_docs", [])
    chat_history = state.get("chat_history", [])
    metadata = state.get("metadata", {})

    start_time = time.time()

    # Use relevant docs, fall back to all retrieved if empty
    context_docs = relevant_docs if relevant_docs else retrieved_docs

    if not context_docs:
        logger.warning("[answer_generator] No context documents available")
        return {
            "answer": (
                "I searched the Confluence knowledge base but couldn't find "
                "relevant information for your question. Please try rephrasing, "
                "or check the Confluence space directly."
            ),
            "sources": [],
        }

    # Build context string from documents
    context_parts = []
    for i, doc in enumerate(context_docs):
        title = doc.metadata.get("title", "Untitled")
        context_parts.append(
            f"--- Document {i + 1}: {title} ---\n{doc.page_content}"
        )
    context_str = "\n\n".join(context_parts)

    # Format chat history
    history_str = ""
    if chat_history:
        history_parts = []
        for msg in chat_history[-6:]:  # last 3 turns (6 messages)
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_parts)
    else:
        history_str = "No previous conversation."

    # Build the answer prompt
    prompt = ANSWER_PROMPT.format(
        chat_history=history_str,
        context=context_str,
        question=query,
    )

    logger.info(
        f"[answer_generator] Generating answer from {len(context_docs)} docs"
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    answer = response.content.strip()

    # Extract unique sources from document metadata
    seen_pages = set()
    sources = []
    for doc in context_docs:
        page_id = doc.metadata.get("page_id", "")
        if page_id and page_id not in seen_pages:
            seen_pages.add(page_id)
            sources.append(
                {
                    "title": doc.metadata.get("title", "Untitled"),
                    "url": doc.metadata.get("url", ""),
                    "page_id": page_id,
                }
            )

    logger.info(
        f"[answer_generator] Generated answer ({len(answer)} chars, "
        f"{len(sources)} sources)"
    )

    import re

    metadata["generation_time"] = time.time() - start_time

    # Extract confidence score from the response e.g. [CONFIDENCE: 95%]
    confidence_match = re.search(r"\[CONFIDENCE:\s*(\d+)%\]", answer)
    if confidence_match:
        metadata["confidence_score"] = int(confidence_match.group(1))
        # Remove the confidence tag from the displayed answer
        answer = re.sub(r"\n*\[CONFIDENCE:\s*\d+%\]", "", answer).strip()
    else:
        metadata["confidence_score"] = None

    return {"answer": answer, "sources": sources, "metadata": metadata}


# ─── Routing Function ───────────────────────────────────────────────────────


def route_after_grading(state: AgentState) -> str:
    """
    Conditional edge: decide whether to generate an answer or rewrite the query.
    - If we have relevant docs → generate
    - If no relevant docs but retries remain → rewrite
    - If max retries exceeded → generate anyway (best-effort)
    """
    relevant_docs = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if relevant_docs:
        logger.info("[router] Relevant docs found → generating answer")
        return "generate"

    if retry_count >= settings.max_retry_count:
        logger.warning(
            f"[router] Max retries ({settings.max_retry_count}) reached → "
            "generating best-effort answer"
        )
        return "generate"

    logger.info(f"[router] No relevant docs, retry {retry_count + 1} → rewriting query")
    return "rewrite"


def route_after_analyzer(state: AgentState) -> str:
    """
    Conditional edge after query_analyzer:
    - If answer is already set (greeting), skip to END
    - Otherwise, proceed to retriever
    """
    if state.get("answer"):
        return "done"
    return "retrieve"
