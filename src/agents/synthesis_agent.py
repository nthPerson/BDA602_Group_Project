"""Agent 5: Synthesis.

This agent is the final step in the pipeline. It assembles the output by:

1. Filtering out low-confidence grounded papers (below configurable threshold)
2. Scoring remaining papers by a composite metric (rerank_score * confidence)
3. Looking up paper metadata (authors, year, citations) from the reranked
   candidates or an optional database connection
4. Assessing citation intent match quality
5. Formatting everything into Recommendation objects with sequential ranks

This is the thinnest agent — mostly formatting and quality control.
No LLM calls are involved.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass

from src.data.db import get_papers_by_ids
from src.data.models import (
    GroundedPaper,
    QueryAnalysis,
    RankedPaper,
    Recommendation,
)
from src.orchestration.state import PipelineState


@dataclass
class SynthesisAgentConfig:
    """Configuration for the Synthesis agent.

    Attributes:
        confidence_threshold: Minimum confidence to include a recommendation.
        max_recommendations: Maximum number of final recommendations.
    """

    confidence_threshold: float = 0.4
    max_recommendations: int = 10


class SynthesisAgent:
    """Agent 5: Synthesis — confidence filtering, scoring, and formatting.

    Reads grounded and reranked candidates from state and produces
    a ranked list of Recommendation objects.
    """

    def __init__(
        self,
        config: SynthesisAgentConfig,
        db: sqlite3.Connection | None = None,
    ) -> None:
        """Initialize the synthesis agent.

        Args:
            config: Agent configuration.
            db: Optional SQLite connection for looking up paper authors.
                If not provided, authors will be empty lists.
        """
        self.config = config
        self.db = db
        self.logger = logging.getLogger(__name__)

    def run(self, state: PipelineState) -> dict:
        """Execute the synthesis step.

        Reads `grounded_candidates`, `reranked_candidates`, and `query_analysis`
        from state. Produces `final_recommendations`.

        Args:
            state: Current pipeline state.

        Returns:
            Dict with final_recommendations and updated metadata.
        """
        grounded = state.get("grounded_candidates", [])
        reranked = state.get("reranked_candidates", [])
        query_analysis = state.get("query_analysis")
        metadata = dict(state.get("metadata", {}))
        errors = list(state.get("errors", []))

        start_time = time.time()

        recommendations = self.synthesize(grounded, reranked, query_analysis)

        elapsed = time.time() - start_time
        metadata["agent5_latency_s"] = round(elapsed, 3)
        metadata["agent5_recommendation_count"] = len(recommendations)

        self.logger.info(f"Synthesized {len(recommendations)} recommendations")

        return {
            "final_recommendations": recommendations,
            "metadata": metadata,
            "errors": errors,
        }

    def synthesize(
        self,
        grounded_papers: list[GroundedPaper],
        reranked_papers: list[RankedPaper],
        query_analysis: QueryAnalysis | None,
    ) -> list[Recommendation]:
        """Filter, score, and format grounded papers into recommendations.

        Args:
            grounded_papers: Papers with justifications from Agent 4.
            reranked_papers: Reranked papers from Agent 4 (used for metadata lookup).
            query_analysis: Query analysis from Agent 1 (used for intent matching).

        Returns:
            Ranked list of Recommendation objects.
        """
        if not grounded_papers:
            return []

        # Build metadata lookup from reranked papers
        meta_lookup: dict[str, RankedPaper] = {p.paper_id: p for p in reranked_papers}

        # Look up authors from database if available
        author_lookup: dict[str, list[str]] = {}
        if self.db is not None:
            paper_ids = [p.paper_id for p in grounded_papers]
            try:
                papers = get_papers_by_ids(self.db, paper_ids)
                author_lookup = {p.paper_id: p.authors for p in papers}
            except Exception as e:
                self.logger.warning(f"Author lookup failed: {e}")

        # Filter by confidence threshold
        filtered = [p for p in grounded_papers if p.confidence >= self.config.confidence_threshold]

        # Sort by composite score (rerank_score * confidence), descending
        filtered.sort(key=lambda p: p.rerank_score * p.confidence, reverse=True)

        # Limit to max_recommendations
        filtered = filtered[: self.config.max_recommendations]

        # Build final Recommendation objects
        recommendations: list[Recommendation] = []
        for i, paper in enumerate(filtered):
            meta = meta_lookup.get(paper.paper_id)
            recommendations.append(
                Recommendation(
                    rank=i + 1,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=author_lookup.get(paper.paper_id, []),
                    year=meta.year if meta else 0,
                    citation_count=meta.citation_count if meta else 0,
                    justification=paper.justification,
                    supporting_snippet=paper.supporting_snippet,
                    confidence=paper.confidence,
                    citation_intent_match=self._assess_intent_match(paper, query_analysis),
                )
            )

        return recommendations

    def _assess_intent_match(
        self,
        paper: GroundedPaper,
        query_analysis: QueryAnalysis | None,
    ) -> str:
        """Assess how well a paper matches the detected citation intent.

        Uses confidence and the detected intent to produce a descriptive label.

        Args:
            paper: Grounded paper to assess.
            query_analysis: Query analysis with detected intent.

        Returns:
            Descriptive string like "Strong match for BACKGROUND citation".
        """
        if query_analysis is None:
            return "unknown"

        intent = query_analysis.citation_intent.value
        confidence = paper.confidence

        if confidence >= 0.8:
            strength = "Strong"
        elif confidence >= 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"

        return f"{strength} match for {intent.upper()} citation"
