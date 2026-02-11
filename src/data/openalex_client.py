"""OpenAlex API client for fetching academic paper metadata.

This module provides a wrapper around the pyalex library with support for:
- Querying AI/ML papers by concept and date range
- Pagination through large result sets
- Rate limiting to respect API guidelines
- Parsing API responses into Paper and CitationEdge objects
- Local caching of raw API responses
"""

import json
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyalex
from pyalex import Works

from src.data.models import CitationEdge, Paper


@dataclass
class OpenAlexConfig:
    """Configuration for OpenAlex API client.

    Attributes:
        email: Polite pool email for faster rate limits.
        concepts: OpenAlex concept IDs to filter by (AI, ML, NLP, CV).
        from_year: Minimum publication year (inclusive).
        to_year: Maximum publication year (inclusive).
        per_page: Number of results per API page.
        rate_limit_delay: Seconds to wait between API calls.
        cache_dir: Directory to cache raw API responses.
    """

    email: str = "your-email@example.com"
    concepts: list[str] = field(
        default_factory=lambda: [
            "C154945302",  # Artificial Intelligence
            "C119857082",  # Machine Learning
            "C204321447",  # Natural Language Processing
            "C154945302",  # Computer Vision (duplicate ID, will be deduplicated)
        ]
    )
    from_year: int = 2018
    to_year: int = 2025
    per_page: int = 200
    rate_limit_delay: float = 0.1
    cache_dir: Path | None = None


class OpenAlexClient:
    """Client for fetching papers from OpenAlex API.

    This client wraps the pyalex library and provides methods for:
    - Querying papers by concept, date range, and citation count
    - Paginating through result sets
    - Parsing API responses into Paper objects
    - Extracting citation edges
    - Caching raw responses to avoid re-querying
    """

    def __init__(self, config: OpenAlexConfig) -> None:
        """Initialize the OpenAlex client.

        Args:
            config: Configuration for API queries and rate limiting.
        """
        self.config = config

        # Configure pyalex with polite pool email for faster rate limits
        pyalex.config.email = config.email

        # Create cache directory if specified
        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_papers(
        self,
        max_results: int | None = None,
        filter_no_abstract: bool = True,
    ) -> Generator[Paper, None, None]:
        """Fetch AI/ML papers from OpenAlex.

        This method queries OpenAlex for papers matching the configured concept
        and date filters, sorts by citation count descending, and yields Paper
        objects one at a time.

        Args:
            max_results: Maximum number of papers to fetch. None for all.
            filter_no_abstract: If True, skip papers without abstracts.

        Yields:
            Paper objects parsed from OpenAlex API responses.
        """
        # Build query with filters
        query = (
            Works()
            .filter(
                concepts={"id": "|".join(self.config.concepts)},
                from_publication_date=f"{self.config.from_year}-01-01",
                to_publication_date=f"{self.config.to_year}-12-31",
                has_abstract=filter_no_abstract,
                type="article",
            )
            .sort(cited_by_count="desc")
        )

        # Paginate through results
        count = 0
        for page_num, page in enumerate(query.paginate(per_page=self.config.per_page), start=1):
            # Rate limiting
            if page_num > 1:
                time.sleep(self.config.rate_limit_delay)

            # Parse each work in the page
            for work in page:
                # Skip papers without abstracts if filtering enabled
                if filter_no_abstract and not self._has_abstract(work):
                    continue

                # Parse into Paper object
                paper = self._parse_work_to_paper(work)

                # Cache the raw response if cache directory is configured
                if self.config.cache_dir:
                    self._cache_work(work, paper.paper_id)

                yield paper

                # Check if we've reached the max results
                count += 1
                if max_results and count >= max_results:
                    return

    def _has_abstract(self, work: dict[str, Any]) -> bool:
        """Check if a work has a non-empty abstract.

        Args:
            work: Raw OpenAlex work dictionary.

        Returns:
            True if the work has an abstract, False otherwise.
        """
        abstract = work.get("abstract_inverted_index")
        if not abstract:
            return False
        # Check if abstract is non-empty
        return len(abstract) > 0

    def _parse_work_to_paper(self, work: dict[str, Any]) -> Paper:
        """Parse an OpenAlex work dictionary into a Paper object.

        Args:
            work: Raw OpenAlex work dictionary.

        Returns:
            Paper object with all fields populated.
        """
        # Extract paper ID (remove OpenAlex URL prefix if present)
        paper_id = work.get("id", "")
        if paper_id.startswith("https://openalex.org/"):
            paper_id = paper_id.replace("https://openalex.org/", "")

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index", {}))

        # Extract authors
        authors = [
            authorship.get("author", {}).get("display_name", "Unknown")
            for authorship in work.get("authorships", [])
        ]

        # Extract concepts (topic labels)
        concepts = [
            concept.get("display_name", "")
            for concept in work.get("concepts", [])
            if concept.get("score", 0) > 0.3  # Only include high-confidence concepts
        ]

        # Extract references (list of OpenAlex work IDs this paper cites)
        references = []
        for ref in work.get("referenced_works", []):
            if ref.startswith("https://openalex.org/"):
                ref_id = ref.replace("https://openalex.org/", "")
                references.append(ref_id)
            else:
                references.append(ref)

        # Extract metadata
        title = work.get("title", "")
        year = work.get("publication_year")
        citation_count = work.get("cited_by_count", 0)
        doi = work.get("doi")

        # Extract arXiv ID if present
        arxiv_id = None
        for ext_id in work.get("ids", {}).items():
            if ext_id[0] == "arxiv":
                arxiv_id = ext_id[1]
                if arxiv_id and arxiv_id.startswith("https://arxiv.org/abs/"):
                    arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")

        # Extract source/venue
        source = None
        if work.get("primary_location"):
            source_obj = work["primary_location"].get("source")
            if source_obj:
                source = source_obj.get("display_name")

        # Create Paper object
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=year or 0,
            citation_count=citation_count,
            doi=doi,
            arxiv_id=arxiv_id,
            authors=authors,
            concepts=concepts,
            source=source,
            references=references,
            cited_by_count=citation_count,  # Same as citation_count in OpenAlex
            chunk_texts=[abstract] if abstract else [],
        )

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format.

        OpenAlex stores abstracts as inverted indexes: {word: [positions]}.
        This function reconstructs the original text.

        Args:
            inverted_index: Dictionary mapping words to position lists.

        Returns:
            Reconstructed abstract text.
        """
        if not inverted_index:
            return ""

        # Find the maximum position to determine text length
        max_pos = max(max(positions) for positions in inverted_index.values())

        # Create array to hold words at each position
        words = [""] * (max_pos + 1)

        # Place each word at its positions
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        # Join words with spaces
        return " ".join(words)

    def extract_citation_edges(self, paper: Paper) -> list[CitationEdge]:
        """Extract citation edges from a paper's references.

        Args:
            paper: Paper object with populated references field.

        Returns:
            List of CitationEdge objects representing citations.
        """
        edges = []
        for ref_id in paper.references:
            edges.append(CitationEdge(source_id=paper.paper_id, target_id=ref_id))
        return edges

    def _cache_work(self, work: dict[str, Any], paper_id: str) -> None:
        """Cache a raw OpenAlex work to disk.

        Args:
            work: Raw OpenAlex work dictionary.
            paper_id: Paper ID to use as filename.
        """
        if not self.config.cache_dir:
            return

        # Sanitize paper ID for filename (remove slashes)
        safe_id = paper_id.replace("/", "_")
        cache_file = self.config.cache_dir / f"{safe_id}.json"

        # Write to cache
        with open(cache_file, "w") as f:
            json.dump(work, f, indent=2)
