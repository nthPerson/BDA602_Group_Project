"""OpenAlex API client for fetching academic paper metadata.

This module provides a wrapper around the pyalex library with support for:
- Querying AI/ML papers by topics, subfields, or concepts (deprecated) and date range
- Pagination through large result sets
- Rate limiting to respect API guidelines
- Retry logic with exponential backoff for transient errors
- Parsing API responses into Paper and CitationEdge objects
- Local caching of raw API responses
"""

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyalex
from pyalex import Works
from requests.exceptions import HTTPError, RequestException

from src.data.models import CitationEdge, Paper


@dataclass
class OpenAlexConfig:
    """Configuration for OpenAlex API client.

    Attributes:
        email: Polite pool email for faster rate limits.
        filter_mode: Subject filter mode - "topics" (recommended), "subfields", or "concepts" (deprecated).
        topics: OpenAlex topic IDs to filter by (T-prefixed). Used when filter_mode="topics".
        subfields: OpenAlex subfield IDs to filter by. Used when filter_mode="subfields".
        concepts: DEPRECATED - OpenAlex concept IDs (C-prefixed). Used when filter_mode="concepts".
        from_year: Minimum publication year (inclusive).
        to_year: Maximum publication year (inclusive).
        per_page: Number of results per API page.
        rate_limit_delay: Seconds to wait between API calls.
        max_retries: Maximum number of retries for failed requests.
        retry_backoff: Exponential backoff multiplier for retries.
        cache_dir: Directory to cache raw API responses.
    """

    email: str = "your-email@example.com"
    filter_mode: str = "topics"  # "topics", "subfields", or "concepts"

    # Topics (recommended): Specific, high-quality AI/ML research areas
    topics: list[str] = field(
        default_factory=lambda: [
            # Core Neural Networks & Deep Learning
            "T10320",  # Neural Networks and Applications (247k works)
            "T10036",  # Advanced Neural Network Applications/Computer Vision (92k works)
            "T11273",  # Advanced Graph Neural Networks (47k works)
            # Machine Learning & Algorithms
            "T12072",  # Machine Learning and Algorithms (34k works)
            "T12535",  # Machine Learning and Data Classification (37k works)
            "T11689",  # Adversarial Robustness in Machine Learning (50k works)
            "T12026",  # Explainable Artificial Intelligence (40k works)
            "T11714",  # Multimodal Machine Learning Applications (48k works)
            "T11512",  # Anomaly Detection Techniques
            "T11307",  # Domain Adaptation and Few-Shot Learning
            "T10028",  # Topic Modeling (151k works)
            "T10100",  # Metaheuristic Optimization Algorithms
            # Natural Language Processing
            "T10181",  # Natural Language Processing Techniques (284k works)
            "T12031",  # Speech and Dialogue Systems
            "T10201",  # Speech Recognition and Synthesis
            "T10215",  # Semantic Web and Ontologies (169k works)
            # Computer Vision & Image Processing
            "T14339",  # Image Processing and 3D Reconstruction (233k works)
            "T10057",  # Face and Expression Recognition (73k works)
            "T10052",  # Medical Image Segmentation (82k works)
            "T10331",  # Video Surveillance and Tracking (80k works)
            "T10627",  # Advanced Image and Video Retrieval (116k works)
            "T10531",  # Advanced Vision and Imaging (103k works)
            # Robotics & Control
            "T10462",  # Reinforcement Learning in Robotics
            "T10586",  # Robotic Path Planning Algorithms (110k works)
            "T10820",  # Fuzzy Logic and Control Systems
        ]
    )

    # Subfields (broader coverage): Captures all topics within these subfields
    subfields: list[str] = field(
        default_factory=lambda: [
            "1702",  # Artificial Intelligence
            "1707",  # Computer Vision and Pattern Recognition
        ]
    )

    # Concepts (deprecated): Legacy filtering method with poor recent coverage
    concepts: list[str] = field(
        default_factory=lambda: [
            "C154945302",  # Artificial Intelligence
            "C119857082",  # Machine Learning
            "C204321447",  # Natural Language Processing
            "C41008148",  # Computer Vision
            "C50644808",  # Deep Learning
            "C2776649807",  # Reinforcement Learning
        ]
    )

    from_year: int = 2015
    to_year: int = 2026
    per_page: int = 200
    rate_limit_delay: float = 0.1
    max_retries: int = 5
    retry_backoff: float = 2.0
    cache_dir: Path | None = None


class OpenAlexClient:
    """Client for fetching papers from OpenAlex API.

    This client wraps the pyalex library and provides methods for:
    - Querying papers by topics, subfields, or concepts (deprecated), date range, and citation count
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
        self.logger = logging.getLogger(__name__)

        # Configure pyalex with polite pool email for faster rate limits
        pyalex.config.email = config.email

        # Create cache directory if specified
        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_papers(
        self,
        max_results: int | None = None,
        filter_no_abstract: bool = True,
        year_range: tuple[int, int] | None = None,
    ) -> Generator[Paper, None, None]:
        """Fetch AI/ML papers from OpenAlex.

        This method queries OpenAlex for papers matching the configured subject
        filters (topics, subfields, or concepts) and date filters, sorts by
        citation count descending, and yields Paper objects one at a time.
        Includes retry logic for transient API errors.

        Args:
            max_results: Maximum number of papers to fetch. None for all.
            filter_no_abstract: If True, skip papers without abstracts.
            year_range: Optional (from_year, to_year) tuple to override config years.
                       Useful for bypassing pagination limits by fetching year-by-year.

        Yields:
            Paper objects parsed from OpenAlex API responses.
        """
        # Build base query with common filters
        query = Works()

        # Add subject filter based on filter_mode
        if self.config.filter_mode == "topics":
            # Filter by specific topic IDs (most granular)
            query = query.filter(topics={"id": "|".join(self.config.topics)})
            self.logger.info(f"Using topic filter with {len(self.config.topics)} topics")
        elif self.config.filter_mode == "subfields":
            # Filter by subfield IDs (broader coverage - matches ANY topic, not just primary)
            query = query.filter(topics={"subfield": {"id": "|".join(self.config.subfields)}})
            self.logger.info(f"Using subfield filter with {len(self.config.subfields)} subfields")
        elif self.config.filter_mode == "concepts":
            # Legacy concept filtering (deprecated)
            query = query.filter(concepts={"id": "|".join(self.config.concepts)})
            self.logger.warning(
                "Using deprecated concept filtering. Consider switching to 'topics' or 'subfields' mode."
            )
        else:
            raise ValueError(
                f"Invalid filter_mode: {self.config.filter_mode}. "
                "Must be 'topics', 'subfields', or 'concepts'."
            )

        # Add common filters and sorting
        # Note: Include article, proceedings-article (conferences), and preprint (arXiv)
        # because AI/ML research is published primarily in conferences and preprints
        # Use year_range override if provided, otherwise use config years
        from_year, to_year = year_range if year_range else (self.config.from_year, self.config.to_year)
        query = (
            query.filter(
                from_publication_date=f"{from_year}-01-01",
                to_publication_date=f"{to_year}-12-31",
                has_abstract=filter_no_abstract,
                type="article|proceedings-article|preprint",
            )
            .sort(cited_by_count="desc")
        )

        # Paginate through results with retry logic
        count = 0
        page_num = 0
        page_iterator = query.paginate(per_page=self.config.per_page)

        while True:
            # Fetch next page with retry logic
            page = self._fetch_page_with_retry(page_iterator, page_num + 1)
            if page is None:
                # No more pages
                break

            page_num += 1

            # Rate limiting (skip for first page)
            if page_num > 1:
                time.sleep(self.config.rate_limit_delay)

            # Parse each work in the page
            for work in page:
                # Skip papers without abstracts if filtering enabled
                if filter_no_abstract and not self._has_abstract(work):
                    continue

                # Skip papers without titles
                if not self._has_title(work):
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

    def _fetch_page_with_retry(
        self,
        page_iterator: Any,
        page_num: int,
    ) -> list[dict[str, Any]] | None:
        """Fetch a page from the iterator with retry logic.

        Args:
            page_iterator: OpenAlex paginator iterator.
            page_num: Current page number (for logging).

        Returns:
            List of works from the page, or None if iteration is complete.
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                # Try to get the next page
                page = next(page_iterator)
                return page

            except StopIteration:
                # Normal end of pagination
                return None

            except (HTTPError, RequestException) as e:
                if attempt < self.config.max_retries:
                    # Calculate backoff delay
                    delay = self.config.retry_backoff ** attempt
                    self.logger.warning(
                        f"API error on page {page_num} (attempt {attempt + 1}/{self.config.max_retries + 1}): "
                        f"{e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Max retries exceeded
                    self.logger.error(
                        f"Failed to fetch page {page_num} after {self.config.max_retries + 1} attempts: {e}"
                    )
                    raise

        return None

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

    def _has_title(self, work: dict[str, Any]) -> bool:
        """Check if a work has a non-empty title.

        Args:
            work: Raw OpenAlex work dictionary.

        Returns:
            True if the work has a title, False otherwise.
        """
        title = work.get("title")
        if not title:
            return False
        # Check if title is non-empty (after stripping whitespace)
        return len(title.strip()) > 0

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
