"""Evaluation metrics, datasets, and baselines."""

from src.evaluation.baselines import BM25Baseline
from src.evaluation.dataset import build_eval_dataset, load_eval_dataset, save_eval_dataset
from src.evaluation.metrics import average_precision, mrr, recall_at_k
from src.evaluation.runner import evaluate_bm25, evaluate_pipeline

__all__ = [
    "BM25Baseline",
    "average_precision",
    "build_eval_dataset",
    "evaluate_bm25",
    "evaluate_pipeline",
    "load_eval_dataset",
    "mrr",
    "recall_at_k",
    "save_eval_dataset",
]
