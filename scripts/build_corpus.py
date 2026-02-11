#!/usr/bin/env python3
"""CLI script to build the paper corpus from OpenAlex.

This script fetches AI/ML papers from OpenAlex and populates the SQLite database
with paper metadata and citation edges. It should be run once to build the corpus.

Usage:
    python scripts/build_corpus.py [--max-papers N] [--cache-dir PATH]

Examples:
    # Build full corpus (15k-25k papers, takes 30-60 minutes)
    python scripts/build_corpus.py

    # Build small corpus for testing (500 papers, takes ~2 minutes)
    python scripts/build_corpus.py --max-papers 500

    # Build with caching enabled (stores raw API responses)
    python scripts/build_corpus.py --cache-dir data/cache
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import get_settings
from src.data.corpus_builder import CorpusBuilder, CorpusBuilderConfig
from src.data.openalex_client import OpenAlexConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the corpus builder.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("corpus_build.log"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build the paper corpus from OpenAlex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to fetch (default: unlimited)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of papers to batch before database insert (default: 100)",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to cache raw OpenAlex responses (default: no caching)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for corpus building.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting corpus build from OpenAlex")
    logger.info("=" * 60)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Make sure your .env file is configured correctly.")
        return 1

    # Create configurations
    openalex_config = OpenAlexConfig(
        email=settings.openalex_email,
        from_year=2018,
        to_year=2025,
        per_page=200,
        rate_limit_delay=settings.openalex_rate_limit_delay,
        cache_dir=args.cache_dir,
    )

    corpus_config = CorpusBuilderConfig(
        db_path=settings.db_path,
        max_papers=args.max_papers,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    # Log configuration
    logger.info(f"Database path: {settings.db_path}")
    logger.info(f"Max papers: {args.max_papers or 'unlimited'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Cache dir: {args.cache_dir or 'disabled'}")
    logger.info(f"OpenAlex email: {settings.openalex_email}")
    logger.info(f"Rate limit delay: {settings.openalex_rate_limit_delay}s")
    logger.info("")

    # Build corpus
    try:
        builder = CorpusBuilder(corpus_config, openalex_config)
        stats = builder.build()

        logger.info("=" * 60)
        logger.info("Corpus build completed successfully!")
        logger.info(f"Papers collected: {stats['papers']}")
        logger.info(f"Citation edges: {stats['edges']}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nCorpus build interrupted by user.")
        logger.warning("You can re-run this script to resume (upsert is safe).")
        return 1

    except Exception as e:
        logger.error(f"Corpus build failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
