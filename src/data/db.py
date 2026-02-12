"""SQLite database operations for paper metadata and citation graph.

This module provides all database operations for storing and querying papers
and citation relationships. The database schema consists of two tables:
- papers: Stores paper metadata
- citation_edges: Stores citation relationships (source_id, target_id)
"""

import json
import sqlite3
from typing import Any

from src.data.models import CitationEdge, Paper


def create_tables(db: sqlite3.Connection) -> None:
    """Create the papers and citation_edges tables with indexes.

    Args:
        db: SQLite database connection.
    """
    cursor = db.cursor()

    # Create papers table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
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

    # Create citation_edges table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS citation_edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            PRIMARY KEY (source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES papers(paper_id),
            FOREIGN KEY (target_id) REFERENCES papers(paper_id)
        )
        """
    )

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_source ON citation_edges(source_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_target ON citation_edges(target_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_papers_citation_count ON papers(citation_count)"
    )

    db.commit()


def _paper_to_row(paper: Paper) -> tuple[Any, ...]:
    """Convert a Paper dataclass to a database row tuple.

    Args:
        paper: Paper instance to convert.

    Returns:
        Tuple of values matching the papers table schema.
    """
    return (
        paper.paper_id,
        paper.title,
        paper.abstract,
        paper.year,
        paper.citation_count,
        paper.doi,
        paper.arxiv_id,
        json.dumps(paper.authors),
        json.dumps(paper.concepts),
        paper.source,
        json.dumps(paper.references),
        paper.cited_by_count,
        json.dumps(paper.chunk_texts),
    )


def _row_to_paper(row: tuple[Any, ...]) -> Paper:
    """Convert a database row tuple to a Paper dataclass.

    Args:
        row: Database row tuple from papers table.

    Returns:
        Paper instance populated from the row.
    """
    return Paper(
        paper_id=row[0],
        title=row[1],
        abstract=row[2],
        year=row[3],
        citation_count=row[4],
        doi=row[5],
        arxiv_id=row[6],
        authors=json.loads(row[7]),
        concepts=json.loads(row[8]),
        source=row[9],
        references=json.loads(row[10]),
        cited_by_count=row[11],
        chunk_texts=json.loads(row[12]),
    )


def insert_paper(db: sqlite3.Connection, paper: Paper) -> None:
    """Insert or update a paper record.

    Uses REPLACE to handle duplicate paper_id (upsert behavior).

    Args:
        db: SQLite database connection.
        paper: Paper instance to insert.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        REPLACE INTO papers (
            paper_id, title, abstract, year, citation_count,
            doi, arxiv_id, authors, concepts, source,
            "references", cited_by_count, chunk_texts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _paper_to_row(paper),
    )
    db.commit()


def insert_papers_batch(db: sqlite3.Connection, papers: list[Paper]) -> None:
    """Batch insert papers for performance.

    Args:
        db: SQLite database connection.
        papers: List of Paper instances to insert.
    """
    cursor = db.cursor()
    rows = [_paper_to_row(p) for p in papers]
    cursor.executemany(
        """
        REPLACE INTO papers (
            paper_id, title, abstract, year, citation_count,
            doi, arxiv_id, authors, concepts, source,
            "references", cited_by_count, chunk_texts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    db.commit()


def insert_citation_edge(db: sqlite3.Connection, edge: CitationEdge) -> None:
    """Insert a citation edge.

    Args:
        db: SQLite database connection.
        edge: CitationEdge instance to insert.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO citation_edges (source_id, target_id)
        VALUES (?, ?)
        """,
        (edge.source_id, edge.target_id),
    )
    db.commit()


def insert_citation_edges_batch(db: sqlite3.Connection, edges: list[CitationEdge]) -> None:
    """Batch insert citation edges for performance.

    Args:
        db: SQLite database connection.
        edges: List of CitationEdge instances to insert.
    """
    cursor = db.cursor()
    rows = [(e.source_id, e.target_id) for e in edges]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO citation_edges (source_id, target_id)
        VALUES (?, ?)
        """,
        rows,
    )
    db.commit()


def get_paper_by_id(db: sqlite3.Connection, paper_id: str) -> Paper | None:
    """Fetch a single paper by ID.

    Args:
        db: SQLite database connection.
        paper_id: Paper ID to query.

    Returns:
        Paper instance if found, None otherwise.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT * FROM papers WHERE paper_id = ?
        """,
        (paper_id,),
    )
    row = cursor.fetchone()
    return _row_to_paper(row) if row else None


def get_papers_by_ids(db: sqlite3.Connection, paper_ids: list[str]) -> list[Paper]:
    """Fetch multiple papers by ID list.

    Args:
        db: SQLite database connection.
        paper_ids: List of paper IDs to query.

    Returns:
        List of Paper instances found.
    """
    if not paper_ids:
        return []

    cursor = db.cursor()
    placeholders = ",".join("?" * len(paper_ids))
    cursor.execute(
        f"""
        SELECT * FROM papers WHERE paper_id IN ({placeholders})
        """,
        paper_ids,
    )
    rows = cursor.fetchall()
    return [_row_to_paper(row) for row in rows]


def get_all_papers(db: sqlite3.Connection) -> list[Paper]:
    """Fetch all papers from the database.

    Args:
        db: SQLite database connection.

    Returns:
        List of all Paper instances in the database.
    """
    cursor = db.cursor()
    cursor.execute("SELECT * FROM papers")
    rows = cursor.fetchall()
    return [_row_to_paper(row) for row in rows]


def get_papers_by_filter(
    db: sqlite3.Connection,
    year_min: int | None = None,
    year_max: int | None = None,
    min_citations: int | None = None,
) -> list[Paper]:
    """Fetch papers matching filter criteria.

    Args:
        db: SQLite database connection.
        year_min: Minimum year (inclusive), or None.
        year_max: Maximum year (inclusive), or None.
        min_citations: Minimum citation count, or None.

    Returns:
        List of Paper instances matching the filters.
    """
    cursor = db.cursor()
    conditions = []
    params = []

    if year_min is not None:
        conditions.append("year >= ?")
        params.append(year_min)
    if year_max is not None:
        conditions.append("year <= ?")
        params.append(year_max)
    if min_citations is not None:
        conditions.append("citation_count >= ?")
        params.append(min_citations)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM papers WHERE {where_clause}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    return [_row_to_paper(row) for row in rows]


def get_references(db: sqlite3.Connection, paper_id: str) -> list[Paper]:
    """Get all papers cited BY this paper (forward traversal).

    Args:
        db: SQLite database connection.
        paper_id: Paper ID to query references for.

    Returns:
        List of Paper instances that are cited by the given paper.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT p.* FROM papers p
        JOIN citation_edges ce ON p.paper_id = ce.target_id
        WHERE ce.source_id = ?
        """,
        (paper_id,),
    )
    rows = cursor.fetchall()
    return [_row_to_paper(row) for row in rows]


def get_cited_by(db: sqlite3.Connection, paper_id: str) -> list[Paper]:
    """Get all papers that CITE this paper (backward traversal).

    Args:
        db: SQLite database connection.
        paper_id: Paper ID to query citations for.

    Returns:
        List of Paper instances that cite the given paper.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT p.* FROM papers p
        JOIN citation_edges ce ON p.paper_id = ce.source_id
        WHERE ce.target_id = ?
        """,
        (paper_id,),
    )
    rows = cursor.fetchall()
    return [_row_to_paper(row) for row in rows]


def get_paper_count(db: sqlite3.Connection) -> int:
    """Get total paper count.

    Args:
        db: SQLite database connection.

    Returns:
        Total number of papers in the database.
    """
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    return cursor.fetchone()[0]


def get_edge_count(db: sqlite3.Connection) -> int:
    """Get total citation edge count.

    Args:
        db: SQLite database connection.

    Returns:
        Total number of citation edges in the database.
    """
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM citation_edges")
    return cursor.fetchone()[0]


def paper_exists(db: sqlite3.Connection, paper_id: str) -> bool:
    """Check if a paper ID exists in the corpus.

    Args:
        db: SQLite database connection.
        paper_id: Paper ID to check.

    Returns:
        True if the paper exists, False otherwise.
    """
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1
        """,
        (paper_id,),
    )
    return cursor.fetchone() is not None
