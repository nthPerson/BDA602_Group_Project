#!/usr/bin/env python3
"""CLI script to embed and index the paper corpus into Qdrant.

This script:
1. Loads papers from SQLite
2. Embeds them using BGE-base-en-v1.5 (768-dim vectors)
3. Uploads embeddings + metadata to Qdrant for similarity search

Usage:
    python scripts/index_corpus.py [--batch-size N] [--recreate]

Examples:
    # Index full corpus (82k papers, takes 60-120 min on GPU)
    python scripts/index_corpus.py

    # Recreate collection from scratch
    python scripts/index_corpus.py --recreate

    # Use smaller batch size (if running out of memory)
    python scripts/index_corpus.py --batch-size 32
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from src.config import get_settings
from src.data.db import get_all_papers
from src.indexing import Embedder, EmbedderConfig, QdrantConfig, QdrantStore


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the indexing script.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("index_corpus.log"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Embed and index paper corpus into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64)",
    )

    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=100,
        help="Batch size for upserting to Qdrant (default: 100)",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate Qdrant collection (default: False)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for corpus indexing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting corpus indexing")
    logger.info("=" * 60)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Make sure your .env file is configured correctly.")
        return 1

    # Verify database exists
    db_path = Path(settings.db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run scripts/build_corpus.py first to build the corpus.")
        return 1

    # Log configuration
    logger.info(f"Database path: {settings.db_path}")
    logger.info(f"Embedding batch size: {args.batch_size}")
    logger.info(f"Upsert batch size: {args.upsert_batch_size}")
    logger.info(f"Recreate collection: {args.recreate}")
    logger.info("")

    try:
        # Step 1: Load papers from SQLite
        logger.info("Step 1: Loading papers from SQLite...")
        db = sqlite3.connect(str(settings.db_path))
        papers = get_all_papers(db)
        db.close()

        logger.info(f"Loaded {len(papers)} papers from database.")
        logger.info("")

        # Step 2: Initialize embedding model
        logger.info("Step 2: Initializing embedding model...")
        embedder_config = EmbedderConfig(
            batch_size=args.batch_size,
            show_progress=True,
        )
        embedder = Embedder(embedder_config)
        logger.info("")

        # Step 3: Embed papers
        logger.info("Step 3: Embedding papers...")
        logger.info(
            f"This will take approximately "
            f"{len(papers) * 0.7 / 60:.0f}-{len(papers) * 1.5 / 60:.0f} minutes "
            f"(depending on GPU availability)."
        )
        embeddings = embedder.embed_papers(papers)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        logger.info("")

        # Step 4: Initialize Qdrant client
        logger.info("Step 4: Connecting to Qdrant...")
        qdrant_config = QdrantConfig()
        qdrant_store = QdrantStore(qdrant_config)
        logger.info("")

        # Step 5: Create collection
        logger.info("Step 5: Creating Qdrant collection...")
        qdrant_store.create_collection(recreate=args.recreate)
        logger.info("")

        # Step 6: Upsert vectors
        logger.info("Step 6: Upserting vectors into Qdrant...")
        qdrant_store.upsert_papers(
            papers, embeddings, batch_size=args.upsert_batch_size
        )
        logger.info("")

        # Step 7: Verify
        logger.info("Step 7: Verifying indexing...")
        point_count = qdrant_store.get_point_count()
        logger.info(f"Qdrant collection point count: {point_count}")

        if point_count != len(papers):
            logger.warning(
                f"Point count mismatch! Expected {len(papers)}, got {point_count}"
            )
            return 1

        logger.info("=" * 60)
        logger.info("Indexing completed successfully!")
        logger.info(f"Indexed {point_count} papers into Qdrant.")
        logger.info(
            "You can now run 'python scripts/test_search.py' to test search."
        )
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nIndexing interrupted by user.")
        return 1

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
