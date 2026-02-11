"""Corpus builder orchestrates the full paper collection process.

This module coordinates:
- Fetching papers from OpenAlex via the client
- Inserting papers into the SQLite database
- Extracting and storing citation edges
- Progress tracking and error handling
"""

import logging
import sqlite3
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
    """

    db_path: str | Path
    max_papers: int | None = None
    batch_size: int = 100
    cache_dir: Path | None = None


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
        2. Fetches papers from OpenAlex in batches
        3. Inserts papers and citation edges into the database
        4. Tracks and reports progress

        Returns:
            Dictionary with statistics: {"papers": count, "edges": count}.
        """
        self.logger.info("Starting corpus build...")

        # Connect to database and create tables
        db = sqlite3.connect(str(self.config.db_path))
        create_tables(db)

        # Track statistics
        total_papers = 0
        total_edges = 0
        batch_papers = []
        batch_edges = []

        # Fetch papers from OpenAlex
        try:
            for i, paper in enumerate(
                self.client.fetch_papers(max_results=self.config.max_papers),
                start=1,
            ):
                batch_papers.append(paper)

                # Extract citation edges
                edges = self.client.extract_citation_edges(paper)
                batch_edges.extend(edges)

                # Insert batch when it reaches batch_size
                if len(batch_papers) >= self.config.batch_size:
                    insert_papers_batch(db, batch_papers)
                    insert_citation_edges_batch(db, batch_edges)

                    total_papers += len(batch_papers)
                    total_edges += len(batch_edges)

                    self.logger.info(
                        f"[{i}/{self.config.max_papers or '?'}] "
                        f"Inserted {len(batch_papers)} papers, {len(batch_edges)} edges. "
                        f"Total: {total_papers} papers, {total_edges} edges."
                    )

                    # Clear batches
                    batch_papers = []
                    batch_edges = []

            # Insert remaining papers
            if batch_papers:
                insert_papers_batch(db, batch_papers)
                insert_citation_edges_batch(db, batch_edges)
                total_papers += len(batch_papers)
                total_edges += len(batch_edges)

                self.logger.info(
                    f"Final batch: {len(batch_papers)} papers, {len(batch_edges)} edges. "
                    f"Total: {total_papers} papers, {total_edges} edges."
                )

        except Exception as e:
            self.logger.error(f"Error during corpus build: {e}")
            raise
        finally:
            db.close()

        self.logger.info(
            f"Corpus build complete! {total_papers} papers and {total_edges} edges stored."
        )

        return {"papers": total_papers, "edges": total_edges}
