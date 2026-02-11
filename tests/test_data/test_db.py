"""Unit tests for database operations."""

import sqlite3
import time

import pytest

from src.data.db import (
    create_tables,
    get_cited_by,
    get_edge_count,
    get_paper_by_id,
    get_paper_count,
    get_papers_by_filter,
    get_papers_by_ids,
    get_references,
    insert_citation_edge,
    insert_citation_edges_batch,
    insert_paper,
    insert_papers_batch,
    paper_exists,
)
from src.data.models import CitationEdge, Paper


@pytest.fixture
def db() -> sqlite3.Connection:
    """Create an in-memory SQLite database for testing.

    Yields:
        In-memory database connection.
    """
    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample Paper instance for testing.

    Returns:
        Sample Paper with all fields populated.
    """
    return Paper(
        paper_id="W123",
        title="Attention Is All You Need",
        abstract="We propose a new architecture based solely on attention mechanisms.",
        year=2017,
        citation_count=90000,
        doi="10.1234/arxiv.1706.03762",
        arxiv_id="1706.03762",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        concepts=["Transformer", "Attention"],
        source="NeurIPS 2017",
        references=["W111", "W222"],
        cited_by_count=90000,
        chunk_texts=["We propose a new architecture..."],
    )


def test_create_tables(db: sqlite3.Connection) -> None:
    """Test that tables are created without errors."""
    cursor = db.cursor()

    # Check that papers table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
    assert cursor.fetchone() is not None

    # Check that citation_edges table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='citation_edges'"
    )
    assert cursor.fetchone() is not None


def test_insert_and_query_paper(db: sqlite3.Connection, sample_paper: Paper) -> None:
    """Test inserting a paper and querying it back."""
    insert_paper(db, sample_paper)

    retrieved = get_paper_by_id(db, "W123")
    assert retrieved is not None
    assert retrieved.paper_id == "W123"
    assert retrieved.title == "Attention Is All You Need"
    assert retrieved.year == 2017
    assert len(retrieved.authors) == 2
    assert retrieved.authors[0] == "Ashish Vaswani"
    assert "Transformer" in retrieved.concepts
    assert len(retrieved.references) == 2


def test_insert_papers_batch(db: sqlite3.Connection) -> None:
    """Test batch inserting 50 papers and verify count."""
    papers = []
    for i in range(50):
        papers.append(
            Paper(
                paper_id=f"W{i}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                year=2020 + (i % 5),
                citation_count=i * 10,
                doi=None,
                arxiv_id=None,
                authors=[f"Author {i}"],
                concepts=["ML"],
                source="Test Conference",
                references=[],
                cited_by_count=i * 10,
                chunk_texts=[f"Abstract {i}"],
            )
        )

    insert_papers_batch(db, papers)
    count = get_paper_count(db)
    assert count == 50


def test_insert_duplicate_paper(db: sqlite3.Connection, sample_paper: Paper) -> None:
    """Test that upserting an existing paper updates it."""
    insert_paper(db, sample_paper)

    # Modify and reinsert
    sample_paper.citation_count = 95000
    insert_paper(db, sample_paper)

    # Should have only one record, with updated count
    assert get_paper_count(db) == 1
    retrieved = get_paper_by_id(db, "W123")
    assert retrieved is not None
    assert retrieved.citation_count == 95000


def test_insert_citation_edge(db: sqlite3.Connection, sample_paper: Paper) -> None:
    """Test inserting a citation edge."""
    # Insert papers first
    insert_paper(db, sample_paper)
    paper2 = Paper(
        paper_id="W456",
        title="Another Paper",
        abstract="Another abstract",
        year=2018,
        citation_count=100,
        doi=None,
        arxiv_id=None,
        authors=["Author"],
        concepts=["ML"],
        source=None,
        references=[],
        cited_by_count=100,
        chunk_texts=["Another abstract"],
    )
    insert_paper(db, paper2)

    # Insert citation edge
    edge = CitationEdge(source_id="W123", target_id="W456")
    insert_citation_edge(db, edge)

    # Verify edge exists
    assert get_edge_count(db) == 1


def test_get_references(db: sqlite3.Connection) -> None:
    """Test fetching references (forward traversal)."""
    # Create papers
    p1 = Paper(
        paper_id="W1",
        title="Paper 1",
        abstract="Abstract 1",
        year=2020,
        citation_count=10,
        doi=None,
        arxiv_id=None,
        authors=["Author 1"],
        concepts=["ML"],
        source=None,
        references=["W2", "W3"],
        cited_by_count=10,
        chunk_texts=["Abstract 1"],
    )
    p2 = Paper(
        paper_id="W2",
        title="Paper 2",
        abstract="Abstract 2",
        year=2019,
        citation_count=20,
        doi=None,
        arxiv_id=None,
        authors=["Author 2"],
        concepts=["ML"],
        source=None,
        references=[],
        cited_by_count=20,
        chunk_texts=["Abstract 2"],
    )
    p3 = Paper(
        paper_id="W3",
        title="Paper 3",
        abstract="Abstract 3",
        year=2018,
        citation_count=30,
        doi=None,
        arxiv_id=None,
        authors=["Author 3"],
        concepts=["ML"],
        source=None,
        references=[],
        cited_by_count=30,
        chunk_texts=["Abstract 3"],
    )

    insert_papers_batch(db, [p1, p2, p3])

    # Insert edges: W1 cites W2 and W3
    edges = [CitationEdge(source_id="W1", target_id="W2"), CitationEdge(source_id="W1", target_id="W3")]
    insert_citation_edges_batch(db, edges)

    # Get references of W1
    refs = get_references(db, "W1")
    assert len(refs) == 2
    ref_ids = {p.paper_id for p in refs}
    assert "W2" in ref_ids
    assert "W3" in ref_ids


def test_get_cited_by(db: sqlite3.Connection) -> None:
    """Test fetching citations (backward traversal)."""
    # Create papers
    p1 = Paper(
        paper_id="W1",
        title="Paper 1",
        abstract="Abstract 1",
        year=2020,
        citation_count=10,
        doi=None,
        arxiv_id=None,
        authors=["Author 1"],
        concepts=["ML"],
        source=None,
        references=["W3"],
        cited_by_count=10,
        chunk_texts=["Abstract 1"],
    )
    p2 = Paper(
        paper_id="W2",
        title="Paper 2",
        abstract="Abstract 2",
        year=2019,
        citation_count=20,
        doi=None,
        arxiv_id=None,
        authors=["Author 2"],
        concepts=["ML"],
        source=None,
        references=["W3"],
        cited_by_count=20,
        chunk_texts=["Abstract 2"],
    )
    p3 = Paper(
        paper_id="W3",
        title="Paper 3",
        abstract="Abstract 3",
        year=2018,
        citation_count=30,
        doi=None,
        arxiv_id=None,
        authors=["Author 3"],
        concepts=["ML"],
        source=None,
        references=[],
        cited_by_count=30,
        chunk_texts=["Abstract 3"],
    )

    insert_papers_batch(db, [p1, p2, p3])

    # Insert edges: W1 and W2 both cite W3
    edges = [CitationEdge(source_id="W1", target_id="W3"), CitationEdge(source_id="W2", target_id="W3")]
    insert_citation_edges_batch(db, edges)

    # Get papers that cite W3
    cited_by = get_cited_by(db, "W3")
    assert len(cited_by) == 2
    cited_ids = {p.paper_id for p in cited_by}
    assert "W1" in cited_ids
    assert "W2" in cited_ids


def test_get_papers_by_filter(db: sqlite3.Connection) -> None:
    """Test filtering papers by year range."""
    papers = []
    for i in range(10):
        papers.append(
            Paper(
                paper_id=f"W{i}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                year=2015 + i,
                citation_count=i * 10,
                doi=None,
                arxiv_id=None,
                authors=[f"Author {i}"],
                concepts=["ML"],
                source=None,
                references=[],
                cited_by_count=i * 10,
                chunk_texts=[f"Abstract {i}"],
            )
        )

    insert_papers_batch(db, papers)

    # Filter by year range
    filtered = get_papers_by_filter(db, year_min=2018, year_max=2020)
    assert len(filtered) == 3  # Years 2018, 2019, 2020
    years = {p.year for p in filtered}
    assert years == {2018, 2019, 2020}


def test_paper_not_found(db: sqlite3.Connection) -> None:
    """Test querying a non-existent paper ID returns None."""
    result = get_paper_by_id(db, "NONEXISTENT")
    assert result is None


def test_get_paper_count(db: sqlite3.Connection) -> None:
    """Test that paper count matches number of inserted papers."""
    assert get_paper_count(db) == 0

    papers = [
        Paper(
            paper_id=f"W{i}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            year=2020,
            citation_count=10,
            doi=None,
            arxiv_id=None,
            authors=["Author"],
            concepts=["ML"],
            source=None,
            references=[],
            cited_by_count=10,
            chunk_texts=[f"Abstract {i}"],
        )
        for i in range(5)
    ]
    insert_papers_batch(db, papers)

    assert get_paper_count(db) == 5


def test_paper_exists(db: sqlite3.Connection, sample_paper: Paper) -> None:
    """Test paper_exists returns True for existing, False for missing."""
    assert paper_exists(db, "W123") is False

    insert_paper(db, sample_paper)

    assert paper_exists(db, "W123") is True
    assert paper_exists(db, "W999") is False


def test_get_papers_by_ids(db: sqlite3.Connection) -> None:
    """Test fetching multiple papers by ID list."""
    papers = [
        Paper(
            paper_id=f"W{i}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            year=2020,
            citation_count=10,
            doi=None,
            arxiv_id=None,
            authors=["Author"],
            concepts=["ML"],
            source=None,
            references=[],
            cited_by_count=10,
            chunk_texts=[f"Abstract {i}"],
        )
        for i in range(10)
    ]
    insert_papers_batch(db, papers)

    # Fetch specific papers
    results = get_papers_by_ids(db, ["W2", "W5", "W7"])
    assert len(results) == 3
    ids = {p.paper_id for p in results}
    assert ids == {"W2", "W5", "W7"}


@pytest.mark.slow
def test_batch_insert_performance(db: sqlite3.Connection) -> None:
    """Test that batch insert of 1000 papers completes in <1 second."""
    papers = [
        Paper(
            paper_id=f"W{i}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}" * 50,
            year=2020,
            citation_count=10,
            doi=None,
            arxiv_id=None,
            authors=["Author"],
            concepts=["ML", "NLP"],
            source=None,
            references=[],
            cited_by_count=10,
            chunk_texts=[f"Abstract {i}"],
        )
        for i in range(1000)
    ]

    start = time.time()
    insert_papers_batch(db, papers)
    elapsed = time.time() - start

    assert elapsed < 1.0
    assert get_paper_count(db) == 1000
