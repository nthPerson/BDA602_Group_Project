"""CLI script to build the evaluation dataset from the corpus.

Usage:
    python scripts/build_eval_dataset.py
    python scripts/build_eval_dataset.py --min-refs 3 --max-samples 500
"""

import argparse
import sqlite3

from dotenv import load_dotenv

# Load .env file BEFORE importing from src (to set OPENAI_API_KEY in os.environ)
load_dotenv()

from src.config import Settings  # noqa: E402
from src.evaluation.dataset import build_eval_dataset, save_eval_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation dataset from the corpus")
    parser.add_argument(
        "--min-refs",
        type=int,
        default=5,
        help="Minimum number of in-corpus references to include a paper (default: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (default: no limit)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/eval_dataset.json)",
    )
    args = parser.parse_args()

    settings = Settings()
    output_path = settings.data_dir / "eval_dataset.json" if args.output is None else args.output

    print(f"Connecting to database: {settings.db_path}")
    db = sqlite3.connect(settings.db_path)

    try:
        print(f"Scanning corpus for papers with >= {args.min_refs} in-corpus references...")
        samples = build_eval_dataset(
            db,
            min_in_corpus_refs=args.min_refs,
            max_samples=args.max_samples,
        )

        print(f"Found {len(samples)} eligible papers")

        if samples:
            avg_refs = sum(s.ground_truth_count for s in samples) / len(samples)
            print(f"Average ground truth references per sample: {avg_refs:.1f}")

        save_eval_dataset(samples, output_path)
        print(f"Saved to {output_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
