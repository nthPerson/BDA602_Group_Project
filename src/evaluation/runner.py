"""Evaluation runner: runs the pipeline on evaluation samples and computes metrics.

Supports multiple evaluation modes:
- baseline: BM25 keyword retrieval
- full: Complete 5-agent pipeline
- ablation: Pipeline with one or more components disabled

Results are aggregated across all samples and returned as a dict of
metric name → average value.
"""

import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from src.data.models import EvalSample
from src.evaluation.baselines import BM25Baseline
from src.evaluation.metrics import average_precision, mrr, recall_at_k

logger = logging.getLogger(__name__)


def evaluate_bm25(
    bm25: BM25Baseline,
    samples: list[EvalSample],
    k_values: list[int] | None = None,
    top_k: int = 20,
) -> dict[str, float]:
    """Evaluate BM25 baseline on the evaluation dataset.

    Args:
        bm25: BM25Baseline instance (pre-built index).
        samples: List of EvalSample instances.
        k_values: List of K values for Recall@K. Defaults to [5, 10, 20].
        top_k: Number of results to retrieve per query.

    Returns:
        Dict of metric name → average value across all samples.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    results["map"] = []

    for sample in tqdm(samples, desc="BM25 eval", disable=len(samples) < 10):
        recommended = bm25.search(
            sample.query_text,
            top_k=top_k,
            exclude_ids={sample.query_paper_id},
        )

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recommended, sample.ground_truth_ids, k))
        results["mrr"].append(mrr(recommended, sample.ground_truth_ids))
        results["map"].append(average_precision(recommended, sample.ground_truth_ids))

    return _aggregate(results)


def evaluate_pipeline(
    pipeline: object,
    samples: list[EvalSample],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate the full pipeline on the evaluation dataset.

    Args:
        pipeline: Compiled LangGraph pipeline (with invoke() method).
        samples: List of EvalSample instances.
        k_values: List of K values for Recall@K. Defaults to [5, 10, 20].

    Returns:
        Dict of metric name → average value across all samples.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    results["map"] = []
    results["latency"] = []

    errors = 0
    for sample in tqdm(samples, desc="Pipeline eval"):
        start = time.time()
        try:
            state = pipeline.invoke({"user_text": sample.query_text, "metadata": {}, "errors": []})
        except Exception as e:
            logger.warning(f"Pipeline failed for {sample.query_paper_id}: {e}")
            errors += 1
            continue

        elapsed = time.time() - start
        results["latency"].append(elapsed)

        recommendations = state.get("final_recommendations", [])
        recommended_ids = [
            r.paper_id for r in recommendations if r.paper_id != sample.query_paper_id
        ]

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recommended_ids, sample.ground_truth_ids, k))
        results["mrr"].append(mrr(recommended_ids, sample.ground_truth_ids))
        results["map"].append(average_precision(recommended_ids, sample.ground_truth_ids))

    if errors > 0:
        logger.warning(f"{errors}/{len(samples)} samples failed during evaluation")

    return _aggregate(results)


def _aggregate(results: dict[str, list[float]]) -> dict[str, float]:
    """Average per-sample results into aggregate metrics.

    Args:
        results: Dict of metric name → list of per-sample values.

    Returns:
        Dict of metric name → average value. Empty lists become 0.0.
    """
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in results.items()}


def format_results_table(
    results: dict[str, dict[str, float]],
) -> str:
    """Format multiple evaluation results into a comparison table.

    Args:
        results: Dict of method name → metric dict.
            E.g., {"BM25": {"recall@5": 0.1, ...}, "Full pipeline": {...}}

    Returns:
        Formatted table string ready for printing.
    """
    # Collect all metric names (excluding latency for the header)
    all_metrics: list[str] = []
    for method_results in results.values():
        for metric in method_results:
            if metric not in all_metrics:
                all_metrics.append(metric)

    # Build header
    col_width = 12
    header = f"{'Method':<25}"
    for metric in all_metrics:
        header += f"{metric:>{col_width}}"

    lines = [
        "=== Evaluation Results ===",
        "",
        header,
        "-" * len(header),
    ]

    # Build rows
    for method, method_results in results.items():
        row = f"{method:<25}"
        for metric in all_metrics:
            val = method_results.get(metric, 0.0)
            if metric == "latency":
                row += f"{val:>{col_width}.1f}s"
            else:
                row += f"{val:>{col_width}.4f}"
        lines.append(row)

    return "\n".join(lines)


def save_results(results: dict[str, dict[str, float]], path: Path) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Dict of method name → metric dict.
        path: Path to the output JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")
