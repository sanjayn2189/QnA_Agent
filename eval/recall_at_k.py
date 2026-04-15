"""
Retrieval Evaluation for the ConfluenceAssist RAG pipeline.

Computes two complementary metrics:
  - Recall@K : Did the expected source appear anywhere in the top-K chunks?
  - MRR      : How high did the expected source rank? (Mean Reciprocal Rank)

Why both?
  Recall@K tells you *coverage* — did we find the right document?
  MRR tells you *ranking quality* — did we find it early?
  A system with Recall@K=100% but MRR=0.3 is retrieving the right doc
  but only at rank 5-6, leaving the LLM to wade through irrelevant chunks first.

Usage:
    python eval/recall_at_k.py              # default K=6
    python eval/recall_at_k.py --k 3        # custom K
    python eval/recall_at_k.py --k 10       # larger K
"""

import json
import argparse
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.vector_store import VectorStoreManager
from config.settings import settings


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[dict]:
    """Load the evaluation dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def get_rank(expected_source: str, retrieved_sources: list[str]) -> int | None:
    """
    Return the 1-based rank of the first occurrence of expected_source
    in the deduplicated list of retrieved document titles.
    Returns None if not found.
    """
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return rank
    return None


def evaluate(k: int = 6, dataset_path: str = "eval/eval_dataset.json"):
    """
    Run Recall@K + MRR evaluation.

    For each query:
    1. Retrieve top-K chunks from ChromaDB
    2. Deduplicate titles preserving order → ranked source list
    3. Recall@K: hit if expected_source in ranked list
    4. MRR: reciprocal rank of expected_source (0 if not found)
    """
    dataset = load_eval_dataset(dataset_path)
    vsm = VectorStoreManager()
    retriever = vsm.as_retriever(k=k, search_type="mmr")

    hits = 0
    reciprocal_rank_sum = 0.0
    total = len(dataset)
    results = []

    print(f"\n{'='*70}")
    print(f"  Retrieval Evaluation (Recall@{k} + MRR) — {total} queries")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        expected = item["expected_source"]

        # Retrieve top-K chunks
        docs = retriever.invoke(query)

        # Deduplicate titles, preserving retrieval order
        # This gives us the ranked list at the *document* level
        retrieved_sources = list(dict.fromkeys(
            doc.metadata.get("title", "unknown") for doc in docs
        ))

        # Recall@K
        is_hit = expected in retrieved_sources
        if is_hit:
            hits += 1

        # MRR: find rank of expected source in the deduplicated list
        rank = get_rank(expected, retrieved_sources)
        rr = (1.0 / rank) if rank is not None else 0.0
        reciprocal_rank_sum += rr

        # Status label
        if rank == 1:
            status = "✅ RANK 1"
        elif is_hit:
            status = f"🟡 RANK {rank}"
        else:
            status = "❌ MISS"

        results.append({
            "query": query,
            "expected_source": expected,
            "retrieved_sources": retrieved_sources,
            "hit": is_hit,
            "rank": rank,
            "reciprocal_rank": round(rr, 4),
        })

        # Print per-query result
        print(f"  [{i:2d}/{total}] {status}")
        print(f"         Query:    {query}")
        print(f"         Expected: {expected}")
        print(f"         Got:      {', '.join(retrieved_sources)}")
        print(f"         RR:       {rr:.4f}  (rank={rank})")
        print()

    # Final scores
    recall = hits / total if total > 0 else 0.0
    mrr = reciprocal_rank_sum / total if total > 0 else 0.0

    print(f"{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Recall@{k}: {recall:.2%}  ({hits}/{total} hits)")
    print(f"  MRR:       {mrr:.4f}  (higher = expected doc ranks earlier)")
    print(f"  K value:   {k}")
    print(f"  Queries:   {total}")
    print(f"{'='*70}")
    print()
    print(f"  Interpretation:")
    print(f"    MRR = 1.0  → expected doc always ranked #1")
    print(f"    MRR = 0.5  → expected doc typically ranked #2")
    print(f"    MRR = 0.33 → expected doc typically ranked #3")
    print(f"{'='*70}\n")

    # Save results
    output = {
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "k": k,
        "hits": hits,
        "total": total,
        "details": results,
    }
    output_path = f"eval/results_k{k}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  📄 Detailed results saved to: {output_path}\n")

    return recall, mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recall@K + MRR evaluation for RAG retrieval quality"
    )
    parser.add_argument(
        "--k", type=int, default=settings.retriever_k,
        help="Number of chunks to retrieve (default: from settings)"
    )
    parser.add_argument(
        "--dataset", type=str, default="eval/eval_dataset.json",
        help="Path to evaluation dataset JSON"
    )
    args = parser.parse_args()

    evaluate(k=args.k, dataset_path=args.dataset)
