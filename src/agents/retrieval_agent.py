"""Agent 2: Primary Retrieval.

This agent performs dense retrieval from the Qdrant vector store using the
expanded query from Agent 1. It embeds the query, searches for similar paper
vectors, and returns ranked ScoredPaper results.

No LLM is involved — this agent is pure embedding + vector search, making it
fast (<100ms typical) and deterministic.
"""

import logging
import time
from dataclasses import dataclass

from src.data.models import QueryAnalysis, ScoredPaper
from src.indexing.embedder import Embedder
from src.indexing.qdrant_store import QdrantStore
from src.orchestration.state import PipelineState


@dataclass
class RetrievalAgentConfig:
    """Configuration for the Primary Retrieval agent.

    Attributes:
        top_n: Maximum number of candidates to retrieve.
        year_min: Optional minimum publication year filter.
        year_max: Optional maximum publication year filter.
        min_citation_count: Optional minimum citation count filter.
    """

    top_n: int = 30
    year_min: int | None = None
    year_max: int | None = None
    min_citation_count: int | None = None


class RetrievalAgent:
    """Agent 2: Primary Retrieval from Qdrant.

    Takes a QueryAnalysis from Agent 1, embeds the expanded query,
    and searches the vector store for the most similar papers.
    """

    def __init__(
        self,
        config: RetrievalAgentConfig,
        embedder: Embedder,
        qdrant_store: QdrantStore,
    ) -> None:
        """Initialize the retrieval agent.

        Args:
            config: Agent configuration.
            embedder: Embedding model for encoding queries.
            qdrant_store: Qdrant client for vector search.
        """
        self.config = config
        self.embedder = embedder
        self.qdrant_store = qdrant_store
        self.logger = logging.getLogger(__name__)

    def run(self, state: PipelineState) -> dict:
        """Execute the retrieval step.

        Reads `query_analysis` from state, produces `retrieval_candidates`.

        Args:
            state: Current pipeline state with `query_analysis` populated.

        Returns:
            Dict with `retrieval_candidates` and updated `metadata`.
        """
        query_analysis = state.get("query_analysis")
        metadata = dict(state.get("metadata", {}))
        errors = list(state.get("errors", []))

        if query_analysis is None:
            self.logger.error("No query_analysis in state — cannot retrieve")
            errors.append("Agent 2: No query_analysis available")
            return {
                "retrieval_candidates": [],
                "metadata": metadata,
                "errors": errors,
            }

        start_time = time.time()

        try:
            candidates = self.retrieve(query_analysis)
            self.logger.info(
                f"Retrieved {len(candidates)} candidates "
                f"(top score: {candidates[0].similarity_score:.3f})"
                if candidates
                else "Retrieved 0 candidates"
            )
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            errors.append(f"Agent 2 retrieval failed: {e}")
            candidates = []

        elapsed = time.time() - start_time
        metadata["agent2_latency_s"] = round(elapsed, 3)
        metadata["agent2_candidates"] = len(candidates)

        return {
            "retrieval_candidates": candidates,
            "metadata": metadata,
            "errors": errors,
        }

    def retrieve(
        self,
        query_analysis: QueryAnalysis,
        top_n: int | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        min_citation_count: int | None = None,
    ) -> list[ScoredPaper]:
        """Retrieve papers from Qdrant based on the query analysis.

        Embeds the expanded query using the BGE instruction prefix,
        then searches Qdrant for the most similar papers.

        Args:
            query_analysis: Structured query from Agent 1.
            top_n: Override config top_n. If None, uses config value.
            year_min: Override config year filter minimum.
            year_max: Override config year filter maximum.
            min_citation_count: Override config minimum citation count.

        Returns:
            List of ScoredPaper objects sorted by similarity (highest first).
        """
        top_n = top_n or self.config.top_n
        year_min = year_min if year_min is not None else self.config.year_min
        year_max = year_max if year_max is not None else self.config.year_max
        min_citation_count = (
            min_citation_count
            if min_citation_count is not None
            else self.config.min_citation_count
        )

        # Embed the expanded query (instruction prefix applied inside embed_query)
        query_vector = self.embedder.embed_query(query_analysis.expanded_query)

        # Build year filter tuple if either bound is set
        year_filter = None
        if year_min is not None or year_max is not None:
            year_filter = (
                year_min if year_min is not None else 0,
                year_max if year_max is not None else 9999,
            )

        # Search Qdrant
        results = self.qdrant_store.search(
            query_vector=query_vector,
            limit=top_n,
            year_filter=year_filter,
            min_citation_count=min_citation_count,
        )

        # Map results to ScoredPaper objects
        scored_papers = [self._to_scored_paper(result) for result in results]

        return scored_papers

    @staticmethod
    def _to_scored_paper(result: dict) -> ScoredPaper:
        """Convert a Qdrant search result dict to a ScoredPaper.

        Args:
            result: Dict from QdrantStore.search() with score and payload fields.

        Returns:
            ScoredPaper instance.
        """
        return ScoredPaper(
            paper_id=result["paper_id"],
            title=result["title"],
            abstract=result.get("abstract", ""),
            year=result.get("year", 0),
            citation_count=result.get("citation_count", 0),
            similarity_score=result["score"],
            concepts=result.get("concepts", []),
        )
