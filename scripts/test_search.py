#!/usr/bin/env python3
"""Interactive script to test similarity search in Qdrant.

This script loads the embedding model and Qdrant client, then provides
an interactive prompt where you can enter queries and see the top results.

Usage:
    python scripts/test_search.py [--top-k N]

Examples:
    # Search with default top-5 results
    python scripts/test_search.py

    # Search with top-10 results
    python scripts/test_search.py --top-k 10
"""

import argparse
import logging
import sys

from src.indexing import Embedder, EmbedderConfig, QdrantConfig, QdrantStore


def setup_logging() -> None:
    """Configure minimal logging for the search test."""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings/errors
        format="%(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Interactive similarity search testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Minimum year filter (optional)",
    )

    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Maximum year filter (optional)",
    )

    return parser.parse_args()


def format_result(result: dict, rank: int) -> str:
    """Format a search result for display.

    Args:
        result: Search result dict from Qdrant.
        rank: Result rank (1-indexed).

    Returns:
        Formatted result string.
    """
    score = result["score"]
    title = result["title"]
    year = result["year"]
    citations = result["citation_count"]
    abstract_preview = result["abstract"][:150] + "..."

    return (
        f"\n{rank}. [{score:.3f}] {title} ({year})\n"
        f"   Citations: {citations:,}\n"
        f"   {abstract_preview}"
    )


def main() -> int:
    """Main entry point for interactive search testing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    setup_logging()

    print("=" * 70)
    print("Interactive Similarity Search Test")
    print("=" * 70)
    print()

    try:
        # Initialize embedder
        print("Loading embedding model (this may take a moment)...")
        embedder_config = EmbedderConfig(show_progress=False)
        embedder = Embedder(embedder_config)
        print("✓ Embedding model loaded.")
        print()

        # Initialize Qdrant
        print("Connecting to Qdrant...")
        qdrant_config = QdrantConfig()
        qdrant_store = QdrantStore(qdrant_config)

        # Verify collection exists
        if not qdrant_store.collection_exists():
            print()
            print("ERROR: Qdrant collection 'papers' does not exist.")
            print("Run 'python scripts/index_corpus.py' first to build the index.")
            return 1

        point_count = qdrant_store.get_point_count()
        print(f"✓ Connected to Qdrant ({point_count:,} papers indexed).")
        print()

        # Interactive search loop
        print("Enter your search query (or 'quit' to exit):")
        print("-" * 70)

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                if not query:
                    continue

                # Embed query
                query_vector = embedder.embed_query(query)

                # Search
                year_filter = None
                if args.year_min or args.year_max:
                    year_min = args.year_min or 1900
                    year_max = args.year_max or 2030
                    year_filter = (year_min, year_max)

                results = qdrant_store.search(
                    query_vector=query_vector,
                    limit=args.top_k,
                    year_filter=year_filter,
                )

                # Display results
                print(f"\nTop {len(results)} results:")
                print("=" * 70)
                for i, result in enumerate(results, 1):
                    print(format_result(result, i))

                print()
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

            except Exception as e:
                print(f"\nError during search: {e}")
                continue

        return 0

    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
