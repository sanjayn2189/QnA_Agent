"""
Context Quality Evaluation for the ConfluenceAssist RAG pipeline.

Measures whether retrieved chunks are actually useful BEFORE the LLM generates
an answer. This sits between Recall@K (did we find the right doc?) and answer
correctness (did we produce the right answer?).

Three signals per query:
  - Relevance    (0-1): How relevant is the context to the query?
  - Completeness (0-1): Does the context contain enough to fully answer?
  - Noise        (0-1): How much irrelevant content is mixed in? (lower = better)

Usage:
    python eval/context_quality.py              # default K=6
    python eval/context_quality.py --k 3        # smaller context window
    python eval/context_quality.py --k 6 --top-n 5   # show 5 worst queries
"""

import json
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from src.ingestion.vector_store import VectorStoreManager
from config.settings import settings


CONTEXT_QUALITY_PROMPT = """\
You are evaluating the quality of retrieved context for a question-answering system.

Query:
{query}

Retrieved Context:
{context}

Evaluate the context on the following:

1. Relevance (0 to 1):
   How relevant is the retrieved context to the query?

2. Completeness (0 to 1):
   Does the context contain enough information to fully answer the query?

3. Noise (0 to 1):
   How much irrelevant or distracting information is present?
   (0 = no noise, 1 = very noisy)

Return ONLY a JSON object with no additional text:
{{
  "relevance": <float between 0 and 1>,
  "completeness": <float between 0 and 1>,
  "noise": <float between 0 and 1>,
  "explanation": "<one sentence summarizing context quality>"
}}"""


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[dict]:
    """Load evaluation dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def build_context_block(chunks: list[Document]) -> str:
    """Combine retrieved chunks into a single numbered context block."""
    parts = []
    for i, doc in enumerate(chunks, 1):
        title = doc.metadata.get("title", "Unknown")
        parts.append(f"[Chunk {i} — {title}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def evaluate_context_quality(
    query: str,
    retrieved_chunks: list[Document],
    llm: ChatGroq,
) -> dict:
    """
    Use an LLM to evaluate the quality of retrieved context for a given query.

    Returns a dict with: relevance, completeness, noise, explanation.
    Returns None scores if LLM response cannot be parsed.
    """
    context = build_context_block(retrieved_chunks)
    prompt = CONTEXT_QUALITY_PROMPT.format(query=query, context=context)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        return {
            "relevance":    float(result.get("relevance", 0.0)),
            "completeness": float(result.get("completeness", 0.0)),
            "noise":        float(result.get("noise", 0.0)),
            "explanation":  result.get("explanation", ""),
        }

    except Exception as e:
        print(f"    ⚠️  Failed to parse LLM response: {e}")
        return {
            "relevance":    None,
            "completeness": None,
            "noise":        None,
            "explanation":  f"Parse error: {e}",
        }


def run_context_quality_eval(
    k: int = 6,
    dataset_path: str = "eval/eval_dataset.json",
    top_n_worst: int = 3,
):
    """
    Run context quality evaluation across the full dataset.
    """
    dataset = load_eval_dataset(dataset_path)
    total = len(dataset)

    # Shared singletons
    vsm = VectorStoreManager()
    retriever = vsm.as_retriever(k=k, search_type="mmr")
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.0,        # deterministic for evaluation
        api_key=settings.groq_api_key,
    )

    results = []

    print(f"\n{'='*70}")
    print(f"  Context Quality Evaluation — {total} queries  (K={k})")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        print(f"  [{i:2d}/{total}] {query}")

        # Retrieve top-K chunks
        chunks = retriever.invoke(query)

        # Evaluate context quality via LLM
        scores = evaluate_context_quality(query, chunks, llm)

        # Determine quality label for display
        if scores["relevance"] is not None:
            rel   = scores["relevance"]
            comp  = scores["completeness"]
            noise = scores["noise"]
            label = "🟢" if (rel >= 0.7 and noise <= 0.3) else ("🟡" if rel >= 0.5 else "🔴")
            print(f"          {label} Relevance={rel:.2f}  Completeness={comp:.2f}  Noise={noise:.2f}")
            print(f"          💬 {scores['explanation']}")
        else:
            print(f"          ⚠️  Evaluation failed")
        print()

        results.append({
            "query":          query,
            "expected_source": item.get("expected_source", ""),
            "retrieved_sources": [d.metadata.get("title", "?") for d in chunks],
            **scores,
        })

    # Compute averages (skip None)
    valid = [r for r in results if r["relevance"] is not None]
    avg_relevance    = sum(r["relevance"]    for r in valid) / len(valid) if valid else 0.0
    avg_completeness = sum(r["completeness"] for r in valid) / len(valid) if valid else 0.0
    avg_noise        = sum(r["noise"]        for r in valid) / len(valid) if valid else 0.0

    print(f"{'='*70}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"  Avg Relevance:    {avg_relevance:.4f}  (higher = better)")
    print(f"  Avg Completeness: {avg_completeness:.4f}  (higher = better)")
    print(f"  Avg Noise:        {avg_noise:.4f}  (lower = better)")
    print(f"  Evaluated:        {len(valid)}/{total} queries")
    print(f"{'='*70}\n")

    # Worst-performing queries (lowest relevance or highest noise)
    if valid and top_n_worst > 0:
        # Score = relevance - noise (higher is better)
        scored = sorted(valid, key=lambda r: r["relevance"] - r["noise"])
        worst = scored[:top_n_worst]
        print(f"  ⚠️  Bottom {top_n_worst} queries (low relevance or high noise):")
        for r in worst:
            print(f"    • [{r['relevance']:.2f} rel / {r['noise']:.2f} noise] {r['query']}")
            print(f"      → {r['explanation']}")
        print()

    # Save full results
    output = {
        "k": k,
        "total": total,
        "evaluated": len(valid),
        "avg_relevance":    round(avg_relevance, 4),
        "avg_completeness": round(avg_completeness, 4),
        "avg_noise":        round(avg_noise, 4),
        "details": results,
    }
    output_path = f"eval/context_quality_k{k}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  📄 Results saved to: {output_path}\n")

    return avg_relevance, avg_completeness, avg_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-based context quality evaluation for RAG retrieval"
    )
    parser.add_argument(
        "--k", type=int, default=settings.retriever_k,
        help="Number of chunks to retrieve (default: from settings)"
    )
    parser.add_argument(
        "--dataset", type=str, default="eval/eval_dataset.json",
        help="Path to evaluation dataset JSON"
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Number of worst-performing queries to highlight (default: 3)"
    )
    args = parser.parse_args()

    run_context_quality_eval(k=args.k, dataset_path=args.dataset, top_n_worst=args.top_n)
