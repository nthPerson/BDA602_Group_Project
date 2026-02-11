#!/usr/bin/env python3
"""CLI script to validate the built corpus and print statistics.

This script connects to the SQLite database, prints corpus statistics, and runs
integrity checks to ensure data quality.

Usage:
    python scripts/validate_corpus.py

Example output:
    === Corpus Statistics ===
    Total papers:         18,542
    Total citation edges: 127,891
    Papers with ≥1 edge:  15,230 (82.1%)
    Year range:           2018–2025
    ...
"""

import sqlite3
import sys
from collections import Counter

from src.config import get_settings
from src.data.db import get_edge_count, get_paper_count


def print_statistics(db: sqlite3.Connection) -> None:
    """Print corpus statistics.

    Args:
        db: SQLite database connection.
    """
    cursor = db.cursor()

    print("=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)
    print()

    # Total counts
    total_papers = get_paper_count(db)
    total_edges = get_edge_count(db)
    print(f"Total papers:         {total_papers:,}")
    print(f"Total citation edges: {total_edges:,}")
    print()

    # Papers with at least one citation edge
    cursor.execute("""
        SELECT COUNT(DISTINCT source_id)
        FROM citation_edges
    """)
    papers_with_edges = cursor.fetchone()[0]
    edge_coverage = (papers_with_edges / total_papers * 100) if total_papers > 0 else 0
    print(f"Papers with ≥1 edge:  {papers_with_edges:,} ({edge_coverage:.1f}%)")
    print()

    # Year range
    cursor.execute("SELECT MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL")
    min_year, max_year = cursor.fetchone()
    print(f"Year range:           {min_year}–{max_year}")
    print()

    # Papers by year
    cursor.execute("""
        SELECT year, COUNT(*) as count
        FROM papers
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year DESC
    """)
    year_counts = cursor.fetchall()
    print("Papers by year:")
    for year, count in year_counts[:10]:  # Show top 10 years
        print(f"  {year}: {count:,}")
    print()

    # Unique concepts
    cursor.execute("SELECT concepts FROM papers")
    all_concepts = []
    for (concepts_json,) in cursor.fetchall():
        import json

        concepts = json.loads(concepts_json)
        all_concepts.extend(concepts)
    unique_concepts = len(set(all_concepts))
    print(f"Unique concepts:      {unique_concepts:,}")
    print()

    # Top concepts
    concept_counts = Counter(all_concepts)
    print("Top 10 concepts:")
    for concept, count in concept_counts.most_common(10):
        print(f"  {concept}: {count:,}")
    print()

    # Top cited papers
    cursor.execute("""
        SELECT title, year, citation_count
        FROM papers
        ORDER BY citation_count DESC
        LIMIT 10
    """)
    top_papers = cursor.fetchall()
    print("Top 10 cited papers:")
    for i, (title, year, count) in enumerate(top_papers, start=1):
        # Truncate long titles
        title_short = title[:60] + "..." if len(title) > 60 else title
        print(f"  {i}. {title_short} ({year or 'N/A'}) — {count:,} citations")
    print()


def run_integrity_checks(db: sqlite3.Connection) -> bool:
    """Run integrity checks on the corpus.

    Args:
        db: SQLite database connection.

    Returns:
        True if all checks pass, False otherwise.
    """
    cursor = db.cursor()
    all_passed = True

    print("=" * 70)
    print("INTEGRITY CHECKS")
    print("=" * 70)
    print()

    # Check 1: No papers with empty titles
    cursor.execute("SELECT COUNT(*) FROM papers WHERE title IS NULL OR title = ''")
    empty_titles = cursor.fetchone()[0]
    if empty_titles == 0:
        print("✓ No papers with empty titles")
    else:
        print(f"✗ Found {empty_titles} papers with empty titles")
        all_passed = False

    # Check 2: No papers with empty abstracts
    cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NULL OR abstract = ''")
    empty_abstracts = cursor.fetchone()[0]
    if empty_abstracts == 0:
        print("✓ No papers with empty abstracts")
    else:
        print(f"✗ Found {empty_abstracts} papers with empty abstracts")
        all_passed = False

    # Check 3: All citation edges reference valid paper IDs
    cursor.execute("""
        SELECT COUNT(*)
        FROM citation_edges
        WHERE source_id NOT IN (SELECT paper_id FROM papers)
           OR target_id NOT IN (SELECT paper_id FROM papers)
    """)
    invalid_edges = cursor.fetchone()[0]
    if invalid_edges == 0:
        print("✓ All citation edges reference valid paper IDs")
    else:
        print(f"✗ Found {invalid_edges} citation edges with invalid paper IDs")
        all_passed = False

    # Check 4: No duplicate paper IDs
    cursor.execute("""
        SELECT paper_id, COUNT(*) as count
        FROM papers
        GROUP BY paper_id
        HAVING count > 1
    """)
    duplicates = cursor.fetchall()
    if len(duplicates) == 0:
        print("✓ No duplicate paper IDs")
    else:
        print(f"✗ Found {len(duplicates)} duplicate paper IDs")
        all_passed = False

    # Check 5: Year values are reasonable
    cursor.execute("""
        SELECT COUNT(*)
        FROM papers
        WHERE year < 1900 OR year > 2030
    """)
    invalid_years = cursor.fetchone()[0]
    if invalid_years == 0:
        print("✓ All year values are reasonable (1900-2030)")
    else:
        print(f"✗ Found {invalid_years} papers with invalid years")
        all_passed = False

    print()
    return all_passed


def main() -> int:
    """Main entry point for corpus validation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        return 1

    # Connect to database
    try:
        db = sqlite3.connect(settings.db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}", file=sys.stderr)
        print(f"Database path: {settings.db_path}", file=sys.stderr)
        print("Have you run scripts/build_corpus.py yet?", file=sys.stderr)
        return 1

    # Print statistics
    try:
        print_statistics(db)
    except Exception as e:
        print(f"Error printing statistics: {e}", file=sys.stderr)
        db.close()
        return 1

    # Run integrity checks
    try:
        all_passed = run_integrity_checks(db)
    except Exception as e:
        print(f"Error running integrity checks: {e}", file=sys.stderr)
        db.close()
        return 1
    finally:
        db.close()

    # Return success if all checks passed
    if all_passed:
        print("=" * 70)
        print("All integrity checks passed! ✓")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("Some integrity checks failed. Please review the corpus.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
