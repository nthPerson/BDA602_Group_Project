"""Pipeline state definitions for the LangGraph orchestration layer.

This module defines the TypedDict state schema used by LangGraph to pass
data between agents. Each agent reads from and writes to specific keys
in this shared state.

Note: The dataclass version of AgentState in src/data/models.py is used
for type annotations in non-LangGraph contexts. This TypedDict version
is required by LangGraph's StateGraph.
"""

from typing import Any, TypedDict

from src.data.models import (
    GroundedPaper,
    QueryAnalysis,
    RankedPaper,
    Recommendation,
    ScoredPaper,
)


class PipelineState(TypedDict, total=False):
    """Shared state passed between all agents in the LangGraph pipeline.

    Each agent reads from and writes to specific keys. LangGraph manages
    state transitions and ensures type consistency.

    Required keys:
        user_text: The raw input text from the user.

    Optional keys (populated by agents):
        query_analysis: Output from Agent 1 (intent + keywords + expanded query).
        retrieval_candidates: Output from Agent 2 (dense retrieval results).
        expanded_candidates: Output from Agent 3 (citation graph expansion).
        reranked_candidates: Output from Agent 4a (cross-encoder reranking).
        grounded_candidates: Output from Agent 4b (LLM-grounded justifications).
        final_recommendations: Output from Agent 5 (final ranked recommendations).
        errors: List of error messages from any agent.
        metadata: Timing, token usage, and other run metadata.
    """

    # Input (required)
    user_text: str

    # Agent 1 output
    query_analysis: QueryAnalysis | None

    # Agent 2 output
    retrieval_candidates: list[ScoredPaper]

    # Agent 3 output
    expanded_candidates: list[ScoredPaper]

    # Agent 4 output
    reranked_candidates: list[RankedPaper]
    grounded_candidates: list[GroundedPaper]

    # Agent 5 output
    final_recommendations: list[Recommendation]

    # Metadata
    errors: list[str]
    metadata: dict[str, Any]
