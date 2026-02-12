"""Agent 4: Reranking & Grounding.

This agent has two stages:

Stage A — Cross-encoder reranking:
    Uses BAAI/bge-reranker-base to re-score all expanded candidates by jointly
    encoding the query and each paper. This is far more accurate than the
    bi-encoder similarity from Agent 2 because it can attend to fine-grained
    interactions between query and document tokens.

Stage B — LLM grounding:
    For each top-K reranked paper, uses gpt-4o-mini (via instructor) to extract
    a supporting snippet from the abstract and generate a natural-language
    justification. Grounding calls run in parallel via ThreadPoolExecutor.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.data.models import GroundedPaper, RankedPaper, ScoredPaper
from src.orchestration.state import PipelineState

# ==================== Pydantic model for grounding ====================


class GroundingResponse(BaseModel):
    """Pydantic model for structured grounding output via instructor."""

    supporting_snippet: str = Field(
        description=(
            "The most relevant excerpt from the candidate paper's abstract "
            "that supports its relevance to the input text. Must be a direct "
            "quote or close paraphrase from the abstract — do not fabricate content."
        ),
    )
    justification: str = Field(
        description=(
            "A 1-3 sentence explanation of why this paper is a good citation "
            "for the input text. Reference specific concepts, methods, or findings "
            "from both the input text and the candidate paper."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that this paper is a relevant citation for the input text "
            "(0.0 = not relevant, 1.0 = highly relevant). Consider topical overlap, "
            "methodological alignment, and citation intent."
        ),
    )


# ==================== Agent Configuration ====================

GROUNDING_SYSTEM_PROMPT = (
    "You are a research citation analyst. Given an input text from an AI/ML "
    "research paper and a candidate paper, your task is to:\n\n"
    "1. Extract the most relevant snippet from the candidate paper's abstract "
    "that supports its relevance to the input text. This must be a direct quote "
    "or close paraphrase — never fabricate content not present in the abstract.\n\n"
    "2. Explain why this paper is a good citation for the input text. Be specific "
    "and grounded — reference concrete concepts, methods, or findings from both "
    "the input and the candidate paper.\n\n"
    "3. Assess your confidence that this is a relevant citation.\n\n"
    "Be concise and precise. Only reference information actually present in the "
    "candidate paper's abstract."
)


@dataclass
class RerankingAgentConfig:
    """Configuration for the Reranking & Grounding agent.

    Attributes:
        rerank_top_k: Number of top candidates to keep after reranking.
        ground_top_k: Number of top reranked papers to ground with LLM.
        llm_model: OpenAI model for grounding.
        max_retries: Max retries for LLM grounding calls.
        temperature: LLM temperature for grounding.
        max_workers: Max parallel workers for grounding calls.
    """

    rerank_top_k: int = 10
    ground_top_k: int = 5
    llm_model: str = "gpt-4o-mini"
    max_retries: int = 2
    temperature: float = 0.0
    max_workers: int = 5


# ==================== Agent Implementation ====================


class RerankingAgent:
    """Agent 4: Cross-encoder reranking + LLM grounding.

    Stage A re-scores candidates using a cross-encoder for accurate ranking.
    Stage B generates justifications and supporting snippets for the top papers.
    """

    def __init__(
        self,
        config: RerankingAgentConfig,
        cross_encoder: CrossEncoder,
        openai_client: OpenAI,
    ) -> None:
        """Initialize the reranking agent.

        Args:
            config: Agent configuration.
            cross_encoder: Cross-encoder model for reranking.
            openai_client: OpenAI client for grounding (wrapped with instructor).
        """
        self.config = config
        self.cross_encoder = cross_encoder
        self.instructor_client = instructor.from_openai(openai_client)
        self.logger = logging.getLogger(__name__)

    def run(self, state: PipelineState) -> dict:
        """Execute the reranking and grounding steps.

        Reads `expanded_candidates` and `user_text` from state.
        Produces `reranked_candidates` and `grounded_candidates`.

        Args:
            state: Current pipeline state.

        Returns:
            Dict with reranked_candidates, grounded_candidates, and updated metadata.
        """
        expanded_candidates = state.get("expanded_candidates", [])
        user_text = state.get("user_text", "")
        query_analysis = state.get("query_analysis")
        metadata = dict(state.get("metadata", {}))
        errors = list(state.get("errors", []))

        if not expanded_candidates:
            self.logger.warning("No expanded candidates — skipping reranking")
            return {
                "reranked_candidates": [],
                "grounded_candidates": [],
                "metadata": metadata,
                "errors": errors,
            }

        # Use expanded_query for reranking if available, otherwise user_text
        rerank_query = query_analysis.expanded_query if query_analysis is not None else user_text

        start_time = time.time()

        # Stage A: Cross-encoder reranking
        try:
            reranked = self.rerank(rerank_query, expanded_candidates)
            self.logger.info(
                f"Reranked {len(expanded_candidates)} candidates to top {len(reranked)}"
            )
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            errors.append(f"Agent 4 reranking failed: {e}")
            # Graceful degradation: convert top-k expanded to RankedPaper
            reranked = [
                RankedPaper(
                    paper_id=c.paper_id,
                    title=c.title,
                    abstract=c.abstract,
                    year=c.year,
                    citation_count=c.citation_count,
                    similarity_score=c.similarity_score,
                    concepts=c.concepts,
                    rerank_score=0.0,
                )
                for c in expanded_candidates[: self.config.rerank_top_k]
            ]

        rerank_elapsed = time.time() - start_time
        metadata["agent4a_latency_s"] = round(rerank_elapsed, 3)
        metadata["agent4a_reranked_count"] = len(reranked)

        # Stage B: LLM grounding (top ground_top_k papers)
        grounding_start = time.time()
        try:
            grounded = self.ground_batch(user_text, reranked[: self.config.ground_top_k])
            self.logger.info(f"Grounded {len(grounded)} papers with justifications")
        except Exception as e:
            self.logger.error(f"Grounding failed: {e}")
            errors.append(f"Agent 4 grounding failed: {e}")
            grounded = []

        grounding_elapsed = time.time() - grounding_start
        metadata["agent4b_latency_s"] = round(grounding_elapsed, 3)
        metadata["agent4b_grounded_count"] = len(grounded)
        metadata["agent4_total_latency_s"] = round(time.time() - start_time, 3)

        return {
            "reranked_candidates": reranked,
            "grounded_candidates": grounded,
            "metadata": metadata,
            "errors": errors,
        }

    def rerank(
        self,
        query: str,
        candidates: list[ScoredPaper],
        top_k: int | None = None,
    ) -> list[RankedPaper]:
        """Rerank candidates using the cross-encoder.

        Prepares (query, "title\\nabstract") pairs, scores them with the
        cross-encoder in a single batch, and returns the top-k sorted by score.

        Args:
            query: The query text (typically expanded_query from Agent 1).
            candidates: Candidates from Agent 3 expansion.
            top_k: Override config rerank_top_k. If None, uses config value.

        Returns:
            Top-k candidates reranked by cross-encoder score.
        """
        top_k = top_k if top_k is not None else self.config.rerank_top_k

        if not candidates:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, f"{c.title}\n{c.abstract}") for c in candidates]

        # Score all pairs in one batch
        scores = self.cross_encoder.predict(pairs, batch_size=32)

        # Ensure scores is a plain list of floats
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # Sort by cross-encoder score descending
        scored_candidates = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top-k and convert to RankedPaper
        return [
            RankedPaper(
                paper_id=candidate.paper_id,
                title=candidate.title,
                abstract=candidate.abstract,
                year=candidate.year,
                citation_count=candidate.citation_count,
                similarity_score=candidate.similarity_score,
                concepts=candidate.concepts,
                rerank_score=float(score),
            )
            for candidate, score in scored_candidates[:top_k]
        ]

    def ground(self, user_text: str, paper: RankedPaper) -> GroundedPaper:
        """Ground a single paper with an LLM-generated justification.

        Args:
            user_text: Original user input text.
            paper: Reranked paper to ground.

        Returns:
            GroundedPaper with snippet, justification, and confidence.

        Raises:
            Exception: If the LLM fails after all retries.
        """
        response = self.instructor_client.chat.completions.create(
            model=self.config.llm_model,
            response_model=GroundingResponse,
            messages=[
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Input text:\n{user_text}\n\n"
                        f"Candidate paper:\n"
                        f"Title: {paper.title}\n"
                        f"Abstract: {paper.abstract}"
                    ),
                },
            ],
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
        )

        return GroundedPaper(
            paper_id=paper.paper_id,
            title=paper.title,
            rerank_score=paper.rerank_score,
            supporting_snippet=response.supporting_snippet,
            justification=response.justification,
            confidence=response.confidence,
        )

    def ground_batch(
        self,
        user_text: str,
        papers: list[RankedPaper],
        max_workers: int | None = None,
    ) -> list[GroundedPaper]:
        """Ground multiple papers in parallel using ThreadPoolExecutor.

        Args:
            user_text: Original user input text.
            papers: Papers to ground.
            max_workers: Override config max_workers. If None, uses config value.

        Returns:
            List of GroundedPaper objects in original input order.
            Papers that fail grounding are skipped with a warning.
        """
        max_workers = max_workers if max_workers is not None else self.config.max_workers

        if not papers:
            return []

        results: list[tuple[int, GroundedPaper | None]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.ground, user_text, paper): i for i, paper in enumerate(papers)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    grounded = future.result()
                    results.append((index, grounded))
                except Exception as e:
                    self.logger.warning(f"Grounding failed for paper {papers[index].paper_id}: {e}")
                    results.append((index, None))

        # Sort by original order and filter out failures
        results.sort(key=lambda x: x[0])
        return [grounded for _, grounded in results if grounded is not None]
