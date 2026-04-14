"""
Chat API route — POST /chat
Runs the LangGraph CRAG agent with session-based conversation memory.
"""

import time
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.api.schemas import ChatRequest, ChatResponse, SourceDoc
from src.agent.graph import get_graph
from src.utils.logger import logger

router = APIRouter(tags=["chat"])

# ─── In-Memory Session Store ─────────────────────────────────────────────────
# Key: session_id, Value: list of BaseMessage
# NOTE: resets on server restart. Use Redis for production persistence.
SESSION_STORE: dict[str, list] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using the Confluence knowledge base.

    Supports multi-turn conversations via session_id.
    """
    start_time = time.time()
    query = request.query.strip()
    session_id = request.session_id

    logger.info(f"[/chat] query='{query}', session_id='{session_id}'")

    # Retrieve or initialize session history
    chat_history = SESSION_STORE.get(session_id, [])

    # Build initial agent state
    initial_state = {
        "query": query,
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "sources": [],
        "retry_count": 0,
        "chat_history": chat_history,
        "metadata": {
            "retrieval_count": 0,
            "grading_time": 0.0,
            "generation_time": 0.0,
        },
    }

    try:
        # Run the CRAG graph
        graph = get_graph()
        result = graph.invoke(initial_state)

        answer = result.get("answer", "I was unable to generate an answer.")
        sources_raw = result.get("sources", [])
        metadata = result.get("metadata", {})

        # Convert source dicts to SourceDoc models
        sources = [
            SourceDoc(
                title=s.get("title", "Untitled"),
                url=s.get("url", ""),
                page_id=s.get("page_id", ""),
            )
            for s in sources_raw
        ]

        # Update session history (keep last 10 turns = 20 messages)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        SESSION_STORE[session_id] = chat_history

        # Record total time
        metadata["total_time"] = time.time() - start_time

        logger.info(
            f"[/chat] Responded with {len(answer)} chars, "
            f"{len(sources)} sources, total_time={metadata['total_time']:.2f}s"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            query=query,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
