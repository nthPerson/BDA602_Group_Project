"""Evaluation dataset construction from the corpus.

Builds EvalSample instances by selecting papers that have a sufficient number
of in-corpus references. Each sample uses the paper's abstract as the query
text and its in-corpus references as the ground truth citation set.

This implements the retrospective citation prediction evaluation paradigm:
given a paper's abstract, can the system recover the actual citations the
original authors used?
"""

import json
import logging
import sqlite3
from pathlib import Path

from src.data.db import get_all_papers, get_reference_ids
from src.data.models import EvalSample

logger = logging.getLogger(__name__)


def build_eval_dataset(
    db: sqlite3.Connection,
    min_in_corpus_refs: int = 5,
    max_samples: int | None = None,
) -> list[EvalSample]:
    """Build an evaluation dataset from the corpus.

    Selects papers that have at least `min_in_corpus_refs` references present
    in the corpus. For each selected paper, constructs an EvalSample with:
    - query_text: the paper's abstract
    - ground_truth_ids: its references that exist in the corpus

    Args:
        db: SQLite database connection.
        min_in_corpus_refs: Minimum number of in-corpus references required
            for a paper to be included as an eval sample.
        max_samples: Optional limit on number of samples. If None, includes
            all eligible papers.

    Returns:
        List of EvalSample instances.
    """
    logger.info(f"Scanning corpus for papers with >= {min_in_corpus_refs} in-corpus references...")
    papers = get_all_papers(db)
    logger.info(f"Total papers in corpus: {len(papers)}")

    # Get the set of all paper IDs in the corpus for fast membership checks
    corpus_ids = {p.paper_id for p in papers}

    samples: list[EvalSample] = []
    for paper in papers:
        # Skip papers without abstracts
        if not paper.abstract or not paper.abstract.strip():
            continue

        # Get this paper's references
        ref_ids = get_reference_ids(db, paper.paper_id)

        # Filter to only references present in our corpus
        in_corpus_refs = [rid for rid in ref_ids if rid in corpus_ids]

        if len(in_corpus_refs) >= min_in_corpus_refs:
            samples.append(
                EvalSample(
                    query_paper_id=paper.paper_id,
                    query_text=paper.abstract,
                    ground_truth_ids=in_corpus_refs,
                    ground_truth_count=len(in_corpus_refs),
                )
            )

    logger.info(f"Found {len(samples)} eligible papers with >= {min_in_corpus_refs} in-corpus refs")

    if max_samples is not None and len(samples) > max_samples:
        # Take the papers with the most in-corpus references first (richest eval signal)
        samples.sort(key=lambda s: s.ground_truth_count, reverse=True)
        samples = samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    return samples


def save_eval_dataset(samples: list[EvalSample], path: Path) -> None:
    """Save evaluation dataset to a JSON file.

    Args:
        samples: List of EvalSample instances to save.
        path: Path to the output JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "query_paper_id": s.query_paper_id,
            "query_text": s.query_text,
            "ground_truth_ids": s.ground_truth_ids,
            "ground_truth_count": s.ground_truth_count,
        }
        for s in samples
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(samples)} eval samples to {path}")


def load_eval_dataset(path: Path) -> list[EvalSample]:
    """Load evaluation dataset from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of EvalSample instances.
    """
    with open(path) as f:
        data = json.load(f)

    return [
        EvalSample(
            query_paper_id=d["query_paper_id"],
            query_text=d["query_text"],
            ground_truth_ids=d["ground_truth_ids"],
            ground_truth_count=d["ground_truth_count"],
        )
        for d in data
    ]
