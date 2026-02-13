"""CLI script to run the evaluation suite.

Usage:
    # Run BM25 baseline only (free, fast)
    python scripts/run_evaluation.py --mode baseline

    # Run full pipeline evaluation (costs ~$1-2 for 200 samples)
    python scripts/run_evaluation.py --mode full

    # Run both baseline and pipeline
    python scripts/run_evaluation.py --mode all

    # Limit number of samples
    python scripts/run_evaluation.py --mode all --max-samples 50
"""

import argparse
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

# Load .env file BEFORE importing from src (to set OPENAI_API_KEY in os.environ)
load_dotenv()

from src.config import Settings  # noqa: E402
from src.evaluation.baselines import BM25Baseline  # noqa: E402
from src.evaluation.dataset import load_eval_dataset  # noqa: E402
from src.evaluation.runner import (  # noqa: E402
    evaluate_bm25,
    evaluate_pipeline,
    format_results_table,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument(
        "--mode",
        choices=["baseline", "full", "all"],
        default="baseline",
        help="Evaluation mode: baseline (BM25 only), full (pipeline), all (both)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to eval dataset JSON (default: data/eval_dataset.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: data/evaluation_results.json)",
    )
    args = parser.parse_args()

    settings = Settings()
    dataset_path = Path(args.dataset) if args.dataset else settings.data_dir / "eval_dataset.json"
    output_path = (
        Path(args.output) if args.output else settings.data_dir / "evaluation_results.json"
    )

    # Load evaluation dataset
    print(f"Loading evaluation dataset from {dataset_path}")
    samples = load_eval_dataset(dataset_path)
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples")

    all_results: dict[str, dict[str, float]] = {}

    # BM25 baseline
    if args.mode in ("baseline", "all"):
        print("\n--- Running BM25 baseline ---")
        db = sqlite3.connect(settings.db_path)
        try:
            bm25 = BM25Baseline.from_db(db)
            bm25_results = evaluate_bm25(bm25, samples)
            all_results["BM25 baseline"] = bm25_results
        finally:
            db.close()

    # Full pipeline
    if args.mode in ("full", "all"):
        print("\n--- Running full pipeline evaluation ---")
        print("(This will make OpenAI API calls and may take several minutes)")

        from src.orchestration.graph import build_pipeline

        pipeline = build_pipeline()
        pipeline_results = evaluate_pipeline(pipeline, samples)
        all_results["Full pipeline"] = pipeline_results

    # Print results table
    print("\n")
    print(format_results_table(all_results))

    # Save results
    save_results(all_results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
