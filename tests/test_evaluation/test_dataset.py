"""Tests for evaluation dataset construction.

These tests use a temporary SQLite database with synthetic papers and
citation edges to verify that dataset construction logic is correct.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

from src.data.models import EvalSample
from src.evaluation.dataset import build_eval_dataset, load_eval_dataset, save_eval_dataset

# ==================== Helpers ====================


def _create_test_db() -> sqlite3.Connection:
    """Create a temporary in-memory database with test papers and citations.

    Graph structure:
        P001 references P002, P003, P004, P005, P006, P007 (6 in-corpus refs)
        P002 references P003, P004 (2 in-corpus refs — below threshold)
        P003 references P001, P002, P004, P005, P006 (5 in-corpus refs)
        P004-P007 have no references
        P008 references P001, P002, P003, P004, P005, P006, P007 (7 refs, no abstract)
    """
    db = sqlite3.connect(":memory:")

    db.execute(
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            year INTEGER,
            citation_count INTEGER,
            doi TEXT,
            arxiv_id TEXT,
            authors TEXT,
            concepts TEXT,
            source TEXT,
            "references" TEXT,
            cited_by_count INTEGER,
            chunk_texts TEXT
        )
        """
    )

    db.execute(
        """
        CREATE TABLE citation_edges (
            source_id TEXT,
            target_id TEXT,
            PRIMARY KEY (source_id, target_id)
        )
        """
    )

    # Insert papers
    for i in range(1, 8):
        paper_id = f"P{i:03d}"
        abstract = f"Abstract for paper {i} about machine learning and NLP."
        db.execute(
            """INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id,
                f"Paper {i}",
                abstract,
                2020 + i,
                100 * i,
                None,
                None,
                json.dumps([f"Author {i}"]),
                json.dumps(["ML"]),
                None,
                json.dumps([]),
                100 * i,
                json.dumps([abstract]),
            ),
        )

    # P008: paper with no abstract
    db.execute(
        """INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "P008",
            "Paper 8",
            "",  # Empty abstract
            2028,
            800,
            None,
            None,
            json.dumps(["Author 8"]),
            json.dumps(["ML"]),
            None,
            json.dumps([]),
            800,
            json.dumps([]),
        ),
    )

    # Citation edges
    # P001 → P002, P003, P004, P005, P006, P007
    for target in ["P002", "P003", "P004", "P005", "P006", "P007"]:
        db.execute("INSERT INTO citation_edges VALUES (?, ?)", ("P001", target))

    # P002 → P003, P004
    for target in ["P003", "P004"]:
        db.execute("INSERT INTO citation_edges VALUES (?, ?)", ("P002", target))

    # P003 → P001, P002, P004, P005, P006
    for target in ["P001", "P002", "P004", "P005", "P006"]:
        db.execute("INSERT INTO citation_edges VALUES (?, ?)", ("P003", target))

    # P008 → P001-P007 (7 refs, but P008 has no abstract)
    for target in [f"P{i:03d}" for i in range(1, 8)]:
        db.execute("INSERT INTO citation_edges VALUES (?, ?)", ("P008", target))

    db.commit()
    return db


# ==================== Test Classes ====================


class TestSelectsPapersWithEnoughRefs:
    """Only papers with >= min_in_corpus_refs are selected."""

    def test_selects_papers_with_enough_refs(self) -> None:
        db = _create_test_db()

        # min_in_corpus_refs=5 → P001 (6 refs) and P003 (5 refs) qualify
        samples = build_eval_dataset(db, min_in_corpus_refs=5)
        sample_ids = {s.query_paper_id for s in samples}

        assert "P001" in sample_ids  # 6 in-corpus refs
        assert "P003" in sample_ids  # 5 in-corpus refs
        assert "P002" not in sample_ids  # Only 2 refs
        assert "P004" not in sample_ids  # 0 refs

        db.close()

    def test_lower_threshold_includes_more(self) -> None:
        db = _create_test_db()

        samples = build_eval_dataset(db, min_in_corpus_refs=2)
        sample_ids = {s.query_paper_id for s in samples}

        # P002 has 2 refs, should now be included
        assert "P002" in sample_ids

        db.close()


class TestQueryPaperNotInGroundTruth:
    """The query paper's own ID is excluded from its ground truth."""

    def test_query_paper_not_in_ground_truth(self) -> None:
        db = _create_test_db()

        samples = build_eval_dataset(db, min_in_corpus_refs=5)

        for sample in samples:
            assert sample.query_paper_id not in sample.ground_truth_ids

        db.close()


class TestGroundTruthIdsInCorpus:
    """All ground truth IDs actually exist in the corpus."""

    def test_ground_truth_ids_in_corpus(self) -> None:
        db = _create_test_db()

        # Get all paper IDs in the corpus
        cursor = db.cursor()
        cursor.execute("SELECT paper_id FROM papers")
        corpus_ids = {row[0] for row in cursor.fetchall()}

        samples = build_eval_dataset(db, min_in_corpus_refs=5)

        for sample in samples:
            for gt_id in sample.ground_truth_ids:
                assert gt_id in corpus_ids, f"{gt_id} not in corpus"

        db.close()


class TestSampleHasRequiredFields:
    """Each EvalSample has all fields populated."""

    def test_sample_has_required_fields(self) -> None:
        db = _create_test_db()

        samples = build_eval_dataset(db, min_in_corpus_refs=5)
        assert len(samples) > 0

        for sample in samples:
            assert isinstance(sample, EvalSample)
            assert len(sample.query_paper_id) > 0
            assert len(sample.query_text) > 0
            assert len(sample.ground_truth_ids) > 0
            assert sample.ground_truth_count == len(sample.ground_truth_ids)
            assert sample.ground_truth_count >= 5

        db.close()


class TestPapersWithoutAbstractExcluded:
    """Papers without abstracts are not included as eval samples."""

    def test_papers_without_abstract_excluded(self) -> None:
        db = _create_test_db()

        # P008 has 7 refs but empty abstract — should be excluded
        samples = build_eval_dataset(db, min_in_corpus_refs=1)
        sample_ids = {s.query_paper_id for s in samples}

        assert "P008" not in sample_ids

        db.close()


class TestMaxSamplesLimit:
    """max_samples parameter limits the number of samples."""

    def test_max_samples_limit(self) -> None:
        db = _create_test_db()

        samples = build_eval_dataset(db, min_in_corpus_refs=5, max_samples=1)
        assert len(samples) == 1
        # Should keep the paper with the most refs (P001 with 6)
        assert samples[0].query_paper_id == "P001"

        db.close()


class TestSaveAndLoadRoundTrip:
    """Save and load produce identical datasets."""

    def test_save_and_load_round_trip(self) -> None:
        db = _create_test_db()
        samples = build_eval_dataset(db, min_in_corpus_refs=5)
        db.close()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)

        try:
            save_eval_dataset(samples, path)
            loaded = load_eval_dataset(path)

            assert len(loaded) == len(samples)
            for orig, load in zip(loaded, samples, strict=True):
                assert orig.query_paper_id == load.query_paper_id
                assert orig.query_text == load.query_text
                assert orig.ground_truth_ids == load.ground_truth_ids
                assert orig.ground_truth_count == load.ground_truth_count
        finally:
            path.unlink()
