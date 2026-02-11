"""Data models for the citation recommendation system.

All domain objects are defined here as dataclasses. These models are used
throughout the system for type safety and clear contracts between components.
"""

from dataclasses import dataclass, field
from enum import StrEnum

# ==================== Core Data Models ====================


@dataclass
class Paper:
    """Represents a research paper with metadata and citation information.

    This is the primary data structure for storing paper records in the database
    and passing paper information between agents.
    """

    paper_id: str  # OpenAlex work ID (e.g., "W2100837269")
    title: str
    abstract: str
    year: int
    citation_count: int
    doi: str | None
    arxiv_id: str | None
    authors: list[str]  # List of author display names
    concepts: list[str]  # OpenAlex concept labels
    source: str | None  # Journal/venue name
    references: list[str]  # List of OpenAlex work IDs this paper cites
    cited_by_count: int  # Total citation count (for filtering/ranking)
    chunk_texts: list[str]  # Chunked text segments (abstract + optional sections)


@dataclass
class CitationEdge:
    """Represents a citation relationship between two papers.

    source_id: The paper that cites
    target_id: The paper that is cited

    This enables both "references of" (forward) and "cited by" (backward) traversal.
    """

    source_id: str
    target_id: str


# ==================== Agent Output Models ====================


class CitationIntent(StrEnum):
    """Classification of why a paper is being cited."""

    BACKGROUND = "background"  # Citing for general context/motivation
    METHOD = "method"  # Citing a specific technique being used/extended
    COMPARISON = "comparison"  # Citing work being compared against
    BENCHMARK = "benchmark"  # Citing datasets/benchmarks/evaluation standards


@dataclass
class QueryAnalysis:
    """Output from Agent 1: Intent classification and query expansion.

    This structured output guides retrieval and helps the synthesis agent
    determine citation appropriateness.
    """

    topic_keywords: list[str]  # Key technical terms extracted
    citation_intent: CitationIntent  # Enum: BACKGROUND | METHOD | COMPARISON | BENCHMARK
    expanded_query: str  # Retrieval-optimized reformulation
    confidence: float  # Agent's confidence in its analysis (0-1)


@dataclass
class ScoredPaper:
    """A paper with similarity score from dense retrieval.

    Used by Agent 2 (retrieval) and Agent 3 (expansion) output.
    """

    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    similarity_score: float  # Cosine similarity from Qdrant
    concepts: list[str]


@dataclass
class RankedPaper:
    """A paper with reranking score from cross-encoder.

    Output from Agent 4 Stage A (reranking).
    """

    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    similarity_score: float
    concepts: list[str]
    rerank_score: float  # Cross-encoder score


@dataclass
class GroundedPaper:
    """A paper with LLM-generated justification and supporting snippet.

    Output from Agent 4 Stage B (grounding).
    """

    paper_id: str
    title: str
    rerank_score: float
    supporting_snippet: str  # Extracted from the paper
    justification: str  # LLM-generated explanation
    confidence: float  # LLM's self-assessed confidence (0-1)


@dataclass
class Recommendation:
    """Final recommendation output from Agent 5.

    This is what gets displayed to the user in the UI.
    """

    rank: int
    paper_id: str
    title: str
    authors: list[str]
    year: int
    citation_count: int
    justification: str
    supporting_snippet: str
    confidence: float
    citation_intent_match: str  # How well this matches the detected intent


# ==================== Evaluation Models ====================


@dataclass
class EvalSample:
    """A single evaluation sample for testing the recommendation system.

    Constructed from papers in the corpus where we know the ground truth citations.
    """

    query_paper_id: str
    query_text: str  # Abstract of the query paper
    ground_truth_ids: list[str]  # Known references in corpus
    ground_truth_count: int


# ==================== Agent State ====================


@dataclass
class AgentState:
    """Shared state passed between all agents in the LangGraph pipeline.

    Each agent reads from and writes to specific fields in this state.
    """

    # Input
    user_text: str

    # Agent 1 output
    query_analysis: QueryAnalysis | None = None

    # Agent 2 output
    retrieval_candidates: list[ScoredPaper] = field(default_factory=list)

    # Agent 3 output
    expanded_candidates: list[ScoredPaper] = field(default_factory=list)

    # Agent 4 output
    reranked_candidates: list[RankedPaper] = field(default_factory=list)
    grounded_candidates: list[GroundedPaper] = field(default_factory=list)

    # Agent 5 output
    final_recommendations: list[Recommendation] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    timestamps: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
