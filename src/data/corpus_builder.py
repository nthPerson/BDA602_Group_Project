"""Corpus builder orchestrates the full paper collection process.

This module coordinates:
- Fetching papers from OpenAlex via the client
- Inserting papers into the SQLite database
- Extracting and storing citation edges
- Progress tracking and error handling
"""

import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from src.data.db import (
    create_tables,
    insert_citation_edges_batch,
    insert_papers_batch,
)
from src.data.openalex_client import OpenAlexClient, OpenAlexConfig


@dataclass
class CorpusBuilderConfig:
    """Configuration for corpus building.

    Attributes:
        db_path: Path to SQLite database file.
        max_papers: Maximum number of papers to collect. None for unlimited.
        batch_size: Number of papers to batch before inserting to database.
        cache_dir: Directory to cache raw OpenAlex responses.
        use_year_chunks: If True, fetch papers year-by-year to bypass pagination limits.
        max_papers_per_year: When using year chunks, max papers to fetch per year.
    """

    db_path: str | Path
    max_papers: int | None = None
    batch_size: int = 100
    cache_dir: Path | None = None
    use_year_chunks: bool = True
    max_papers_per_year: int | None = None


class CorpusBuilder:
    """Orchestrates the corpus building process.

    This class coordinates fetching papers from OpenAlex, inserting them into
    the database, and extracting citation edges. It handles batching, progress
    tracking, and error recovery.
    """

    def __init__(
        self,
        config: CorpusBuilderConfig,
        openalex_config: OpenAlexConfig,
    ) -> None:
        """Initialize the corpus builder.

        Args:
            config: Configuration for corpus building.
            openalex_config: Configuration for OpenAlex client.
        """
        self.config = config
        self.client = OpenAlexClient(openalex_config)
        self.logger = logging.getLogger(__name__)

        # Ensure database directory exists
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self) -> dict[str, int]:
        """Build the corpus by fetching papers and inserting into database.

        This method:
        1. Creates database tables if they don't exist
        2. Fetches papers from OpenAlex (year-by-year if use_year_chunks=True)
        3. Inserts papers into the database
        4. Filters and inserts only valid citation edges (where both endpoints exist)
        5. Tracks and reports progress

        Returns:
            Dictionary with statistics: {"papers": count, "edges": count}.
        """
        self.logger.info("Starting corpus build...")

        # Connect to database and create tables
        db = sqlite3.connect(str(self.config.db_path))
        create_tables(db)

        # Track statistics and collected data
        total_papers = 0
        batch_papers = []
        all_paper_ids = set()  # Track all paper IDs for edge validation
        all_edges = []  # Collect all edges for later filtering

        # Phase 1: Fetch and insert papers
        try:
            if self.config.use_year_chunks:
                self.logger.info("Phase 1: Fetching and inserting papers (year-by-year mode)...")
                total_papers = self._build_with_year_chunks(
                    db, batch_papers, all_paper_ids, all_edges
                )
            else:
                self.logger.info("Phase 1: Fetching and inserting papers (standard mode)...")
                total_papers = self._build_standard(
                    db, batch_papers, all_paper_ids, all_edges
                )

            # Phase 2: Filter and insert valid citation edges
            self.logger.info(
                f"Phase 2: Filtering and inserting citation edges "
                f"(collected {len(all_paper_ids)} paper IDs)..."
            )

            # Filter edges to only include those where both endpoints exist
            valid_edges = [
                edge
                for edge in all_edges
                if edge.source_id in all_paper_ids and edge.target_id in all_paper_ids
            ]

            self.logger.info(
                f"Filtered {len(valid_edges)} valid edges out of {len(all_edges)} total edges "
                f"({len(valid_edges) / len(all_edges) * 100:.1f}% valid)."
            )

            # Insert valid edges in batches
            total_edges = 0
            batch_size = 1000  # Use larger batch size for edges
            for i in range(0, len(valid_edges), batch_size):
                edge_batch = valid_edges[i : i + batch_size]
                insert_citation_edges_batch(db, edge_batch)
                total_edges += len(edge_batch)

                if (i // batch_size) % 10 == 0:  # Log every 10 batches
                    self.logger.info(
                        f"Inserted {total_edges}/{len(valid_edges)} citation edges..."
                    )

            self.logger.info(f"Inserted {total_edges} valid citation edges.")

        except Exception as e:
            self.logger.error(f"Error during corpus build: {e}")
            raise
        finally:
            db.close()

        self.logger.info(
            f"Corpus build complete! {total_papers} papers and {total_edges} edges stored."
        )

        return {"papers": total_papers, "edges": total_edges}

    def _build_standard(
        self,
        db: sqlite3.Connection,
        batch_papers: list,
        all_paper_ids: set,
        all_edges: list,
    ) -> int:
        """Standard build mode - fetch all papers in one query (hits pagination limit).

        Args:
            db: SQLite database connection.
            batch_papers: List to accumulate papers before batch insert.
            all_paper_ids: Set to track all collected paper IDs.
            all_edges: List to accumulate all citation edges.

        Returns:
            Total number of papers collected.
        """
        total_papers = 0
        try:
            for i, paper in enumerate(
                self.client.fetch_papers(max_results=self.config.max_papers),
                start=1,
            ):
                batch_papers.append(paper)
                all_paper_ids.add(paper.paper_id)

                # Extract citation edges (but don't insert yet)
                edges = self.client.extract_citation_edges(paper)
                all_edges.extend(edges)

                # Insert paper batch when it reaches batch_size
                if len(batch_papers) >= self.config.batch_size:
                    insert_papers_batch(db, batch_papers)
                    total_papers += len(batch_papers)

                    self.logger.info(
                        f"[{i}/{self.config.max_papers or '?'}] "
                        f"Inserted {len(batch_papers)} papers. "
                        f"Total: {total_papers} papers."
                    )
                    sys.stdout.flush()  # Force immediate output

                    # Clear batch
                    batch_papers = []

            # Insert remaining papers
            if batch_papers:
                insert_papers_batch(db, batch_papers)
                total_papers += len(batch_papers)

                self.logger.info(
                    f"Final batch: {len(batch_papers)} papers. "
                    f"Total: {total_papers} papers."
                )

        except Exception as e:
            self.logger.error(f"Error during standard corpus build: {e}")
            raise

        return total_papers

    def _build_with_year_chunks(
        self,
        db: sqlite3.Connection,
        batch_papers: list,
        all_paper_ids: set,
        all_edges: list,
    ) -> int:
        """Year-chunked build mode - fetch papers year-by-year to bypass pagination limits.

        This mode queries OpenAlex separately for each year in the configured range.
        Each year typically stays under the ~10k pagination limit, allowing us to
        collect a much larger corpus.

        Args:
            db: SQLite database connection.
            batch_papers: List to accumulate papers before batch insert.
            all_paper_ids: Set to track all collected paper IDs.
            all_edges: List to accumulate all citation edges.

        Returns:
            Total number of papers collected across all years.
        """
        total_papers = 0
        overall_count = 0

        # Get year range from client config
        from_year = self.client.config.from_year
        to_year = self.client.config.to_year

        self.logger.info(
            f"Fetching papers year-by-year from {from_year} to {to_year} "
            f"(max {self.config.max_papers_per_year or 'unlimited'} per year)"
        )

        try:
            for year in range(from_year, to_year + 1):
                year_papers = 0
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Fetching papers from year {year}...")
                self.logger.info(f"{'='*60}")

                # Fetch papers for this specific year
                for i, paper in enumerate(
                    self.client.fetch_papers(
                        max_results=self.config.max_papers_per_year,
                        year_range=(year, year),
                    ),
                    start=1,
                ):
                    batch_papers.append(paper)
                    all_paper_ids.add(paper.paper_id)

                    # Extract citation edges (but don't insert yet)
                    edges = self.client.extract_citation_edges(paper)
                    all_edges.extend(edges)

                    overall_count += 1
                    year_papers += 1

                    # Insert paper batch when it reaches batch_size
                    if len(batch_papers) >= self.config.batch_size:
                        insert_papers_batch(db, batch_papers)
                        total_papers += len(batch_papers)

                        # Log progress with flush
                        self.logger.info(
                            f"  Year {year}: Processed {year_papers} papers | "
                            f"Global total: {total_papers} papers"
                        )
                        sys.stdout.flush()  # Force immediate output

                        # Clear batch
                        batch_papers = []

                    # Check if we've hit the global max_papers limit
                    if self.config.max_papers and total_papers >= self.config.max_papers:
                        self.logger.info(
                            f"Reached global max_papers limit ({self.config.max_papers}). Stopping."
                        )
                        break

                # Insert remaining papers from this year
                if batch_papers:
                    insert_papers_batch(db, batch_papers)
                    total_papers += len(batch_papers)
                    batch_papers = []

                self.logger.info(
                    f"Year {year} complete: {year_papers} papers. "
                    f"Total so far: {total_papers} papers."
                )

                # Check if we've hit the global max_papers limit
                if self.config.max_papers and total_papers >= self.config.max_papers:
                    break

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"All years processed. Total: {total_papers} papers.")
            self.logger.info(f"{'='*60}\n")

        except Exception as e:
            self.logger.error(f"Error during year-chunked corpus build: {e}")
            raise

        return total_papers
