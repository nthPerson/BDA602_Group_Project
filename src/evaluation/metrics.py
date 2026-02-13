"""Standard information retrieval evaluation metrics.

Implements Recall@K, MRR (Mean Reciprocal Rank), and MAP (Mean Average
Precision) for evaluating the citation recommendation pipeline.

All metrics operate on lists of paper IDs:
  - recommended: ordered list of paper IDs produced by the system
  - ground_truth: unordered set of paper IDs that are known correct answers
"""


def recall_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    """Compute Recall@K: fraction of ground truth found in top K recommendations.

    Args:
        recommended: Ordered list of recommended paper IDs.
        ground_truth: List of ground truth paper IDs (known relevant).
        k: Number of top recommendations to consider.

    Returns:
        Recall@K value in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0

    top_k = set(recommended[:k])
    truth = set(ground_truth)
    return len(top_k & truth) / len(truth)


def mrr(recommended: list[str], ground_truth: list[str]) -> float:
    """Compute MRR (Mean Reciprocal Rank): 1 / rank of first relevant result.

    Args:
        recommended: Ordered list of recommended paper IDs.
        ground_truth: List of ground truth paper IDs (known relevant).

    Returns:
        Reciprocal rank in (0.0, 1.0], or 0.0 if no relevant result is found.
    """
    truth = set(ground_truth)
    for i, paper_id in enumerate(recommended):
        if paper_id in truth:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(recommended: list[str], ground_truth: list[str]) -> float:
    """Compute Average Precision for a single query.

    AP = (1 / |relevant|) * sum(precision@k * rel(k)) for all k

    where rel(k) = 1 if item at rank k is relevant, 0 otherwise, and
    precision@k = (relevant items in top k) / k.

    Args:
        recommended: Ordered list of recommended paper IDs.
        ground_truth: List of ground truth paper IDs (known relevant).

    Returns:
        Average precision in [0.0, 1.0]. Returns 0.0 if ground_truth is empty
        or no relevant items are found.
    """
    if not ground_truth or not recommended:
        return 0.0

    truth = set(ground_truth)
    hits = 0
    sum_precision = 0.0

    for i, paper_id in enumerate(recommended):
        if paper_id in truth:
            hits += 1
            sum_precision += hits / (i + 1)

    if hits == 0:
        return 0.0

    return sum_precision / len(truth)
