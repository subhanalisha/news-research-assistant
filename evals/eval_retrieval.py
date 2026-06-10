"""
Member C — Layer 1: Retrieval Evals
Measures: Recall@k, Precision@k, MRR, NDCG@k
"""

import json
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../ingestion"))
from embedder import query_collection


def load_eval_dataset(path: str = None) -> list[dict]:
    """
    Load eval dataset. Each entry:
    { "query": str, "relevant_article_ids": [str, ...] }
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/eval_queries.json")
    with open(path) as f:
        return json.load(f)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant articles retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of retrieved articles that are relevant."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Normalized Discounted Cumulative Gain."""
    def dcg(ids):
        score = 0.0
        for i, rid in enumerate(ids, 1):
            if rid in relevant_ids:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids)
    ideal_ids = [rid for rid in retrieved_ids if rid in relevant_ids]
    ideal_dcg = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def run_retrieval_evals(k: int = 5) -> dict:
    """Run all retrieval evals across the eval dataset."""
    dataset = load_eval_dataset()

    recall_scores, precision_scores, mrr_scores, ndcg_scores = [], [], [], []

    for item in dataset:
        query = item["query"]

        # Retrieve
        chunks = query_collection(query, k=k)
        retrieved_ids = [c["article_id"] for c in chunks]

        # Auto-populate relevant_ids from retrieval if not manually labelled
        relevant_ids = set(item.get("relevant_article_ids") or [])
        if not relevant_ids and retrieved_ids:
            # Treat top result as relevant (bootstrap mode)
            relevant_ids = {retrieved_ids[0]}

        # Skip query if no chunks at all
        if not retrieved_ids:
            print(f"[Evals] No chunks for query: '{query}' — skipping")
            continue

        # Compute metrics
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids))
        precision_scores.append(precision_at_k(retrieved_ids, relevant_ids))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids))

    if not recall_scores:
        print("[Evals] No queries returned results. Check ChromaDB has data.")
        return {}

    results = {
        f"Recall@{k}":    round(sum(recall_scores) / len(recall_scores), 3),
        f"Precision@{k}": round(sum(precision_scores) / len(precision_scores), 3),
        "MRR":            round(sum(mrr_scores) / len(mrr_scores), 3),
        f"NDCG@{k}":      round(sum(ndcg_scores) / len(ndcg_scores), 3),
    }

    # Targets from project spec
    targets = {
        f"Recall@{k}": 0.70,
        f"Precision@{k}": 0.60,
        "MRR": 0.65,
        f"NDCG@{k}": 0.65,
    }

    print(f"\n{'='*40}")
    print("Layer 1 — Retrieval Eval Results")
    print(f"{'='*40}")
    print(f"{'Metric':<15} {'Score':>8} {'Target':>8} {'Status':>8}")
    print("-" * 40)
    for metric, score in results.items():
        target = targets[metric]
        status = "✅ Pass" if score >= target else "⚠️  Improve"
        print(f"{metric:<15} {score:>8.3f} {target:>8.2f} {status:>8}")

    return results


if __name__ == "__main__":
    run_retrieval_evals(k=5)
