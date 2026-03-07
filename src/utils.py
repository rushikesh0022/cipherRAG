# src/utils.py
# ═══════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════

import numpy as np
from typing import List, Dict


def compute_score_gaps(doc_embeddings: np.ndarray,
                       query_embeddings: np.ndarray,
                       k: int = 3) -> Dict:
    """
    Compute the 'retrieval gap' δₖ = s_(k) - s_(k+1) for each query.

    This gap determines the maximum encryption error tolerable
    while still getting correct top-k ranking.

    THEOREM:
      If CKKS error ε < δₖ/2, then encrypted top-k = plaintext top-k.
      For BFV (exact integers), ε = 0, so this is trivially satisfied.
    """
    gaps = []

    for qv in query_embeddings:
        scores = doc_embeddings @ qv
        sorted_scores = np.sort(scores)[::-1]

        if len(sorted_scores) > k:
            gap = sorted_scores[k - 1] - sorted_scores[k]
            gaps.append(gap)

    gaps = np.array(gaps)

    return {
        "gaps": gaps,
        "min_gap": float(np.min(gaps)) if len(gaps) > 0 else 0,
        "max_gap": float(np.max(gaps)) if len(gaps) > 0 else 0,
        "mean_gap": float(np.mean(gaps)) if len(gaps) > 0 else 0,
        "median_gap": float(np.median(gaps)) if len(gaps) > 0 else 0,
        "max_tolerable_error": float(np.min(gaps) / 2) if len(gaps) > 0 else 0,
    }


def compare_rankings(ground_truth: List[List[int]],
                     predicted: List[List[int]],
                     top_k: int = 3) -> Dict:
    """
    Compare two sets of rankings.
    Returns accuracy metrics.
    """
    n = len(ground_truth)
    top1_correct = 0
    topk_correct = 0

    for i in range(n):
        gt = ground_truth[i]
        pred = predicted[i]

        if len(gt) > 0 and len(pred) > 0 and gt[0] == pred[0]:
            top1_correct += 1

        if set(gt[:top_k]) == set(pred[:top_k]):
            topk_correct += 1

    return {
        "top1_accuracy": top1_correct / n if n > 0 else 0,
        "topk_accuracy": topk_correct / n if n > 0 else 0,
        "n_queries": n,
        "top_k": top_k,
    }


def format_bytes(n_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes/1024**2:.1f} MB"
    else:
        return f"{n_bytes/1024**3:.1f} GB"
