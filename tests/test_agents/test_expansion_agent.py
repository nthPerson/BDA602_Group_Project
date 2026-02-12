"""Tests for Agent 3: Citation Expansion.

All unit tests use a synthetic citation graph — a small, hand-crafted set
of papers and edges where we know exactly what the expansion should produce.

Synthetic test graph:
    Papers: A, B, C, D, E, F, G, H, I, J, K, L
    Edges:
      A → D, A → E, A → F     (A cites D, E, F)
      B → D, B → G             (B cites D, G)
      C → E, C → H             (C cites E, H)
      I → A, J → A             (I and J cite A — "cited by" for A)
      K → B                    (K cites B)
      L → (nothing — isolated) (L has no edges)

Tests cover:
- Expansion finds references (forward traversal)
- Expansion finds cited-by (backward traversal)
- Deduplication across seed papers
- Originals preserved in output
- max_expansion limit respected
- Multi-citation prioritization
- Isolated paper (no edges) handled gracefully
- Seed papers not duplicated as expansion results
- Only corpus papers included in expansion
- Integration test: Agents 1→2→3 end-to-end
"""

import sqlite3
from collections.abc import Generator

import pytest

from src.agents.expansion_agent import ExpansionAgent, ExpansionAgentConfig
from src.data.db import (
    create_tables,
    insert_citation_edges_batch,
    insert_papers_batch,
)
from src.data.models import CitationEdge, Paper, ScoredPaper

# ==================== Helpers ====================


def _make_paper(paper_id: str, citation_count: int = 100) -> Paper:
    """Create a minimal Paper for testing."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract=f"Abstract for paper {paper_id}.",
        year=2020,
        citation_count=citation_count,
        doi=None,
        arxiv_id=None,
        authors=["Author"],
        concepts=["AI"],
        source="Conference",
        references=[],
        cited_by_count=citation_count,
        chunk_texts=[],
    )


def _make_scored_paper(paper_id: str, score: float = 0.9) -> ScoredPaper:
    """Create a minimal ScoredPaper for testing."""
    return ScoredPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract=f"Abstract for paper {paper_id}.",
        year=2020,
        citation_count=100,
        similarity_score=score,
        concepts=["AI"],
    )


# ==================== Fixtures ====================


@pytest.fixture
def synthetic_db() -> Generator[sqlite3.Connection, None, None]:
    """Create an in-memory SQLite database with the synthetic test graph.

    Papers: A, B, C, D, E, F, G, H, I, J, K, L
    Edges:
      A → D, A → E, A → F
      B → D, B → G
      C → E, C → H
      I → A, J → A
      K → B
      L has no edges
    """
    db = sqlite3.connect(":memory:")
    create_tables(db)

    # Insert all papers with varying citation counts for prioritization testing
    papers = [
        _make_paper("A", citation_count=500),
        _make_paper("B", citation_count=300),
        _make_paper("C", citation_count=200),
        _make_paper("D", citation_count=1000),  # D is highly cited
        _make_paper("E", citation_count=800),
        _make_paper("F", citation_count=50),
        _make_paper("G", citation_count=150),
        _make_paper("H", citation_count=100),
        _make_paper("I", citation_count=75),
        _make_paper("J", citation_count=60),
        _make_paper("K", citation_count=40),
        _make_paper("L", citation_count=10),  # Isolated paper
    ]
    insert_papers_batch(db, papers)

    # Insert citation edges
    edges = [
        CitationEdge(source_id="A", target_id="D"),
        CitationEdge(source_id="A", target_id="E"),
        CitationEdge(source_id="A", target_id="F"),
        CitationEdge(source_id="B", target_id="D"),
        CitationEdge(source_id="B", target_id="G"),
        CitationEdge(source_id="C", target_id="E"),
        CitationEdge(source_id="C", target_id="H"),
        CitationEdge(source_id="I", target_id="A"),
        CitationEdge(source_id="J", target_id="A"),
        CitationEdge(source_id="K", target_id="B"),
    ]
    insert_citation_edges_batch(db, edges)

    yield db
    db.close()


@pytest.fixture
def agent(synthetic_db: sqlite3.Connection) -> ExpansionAgent:
    """Create an ExpansionAgent with the synthetic database."""
    config = ExpansionAgentConfig(seed_top_k=10, max_expansion=40)
    return ExpansionAgent(config, synthetic_db)


# ==================== Unit Tests ====================


class TestExpansionFindsReferences:
    """Test forward citation traversal (papers the seed cites)."""

    def test_expansion_finds_references(self, agent: ExpansionAgent) -> None:
        """Seed = [A] -> expanded includes D, E, F (A cites D, E, F)."""
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed)

        result_ids = {p.paper_id for p in result}
        assert "D" in result_ids
        assert "E" in result_ids
        assert "F" in result_ids


class TestExpansionFindsCitedBy:
    """Test backward citation traversal (papers that cite the seed)."""

    def test_expansion_finds_cited_by(self, agent: ExpansionAgent) -> None:
        """Seed = [A] -> expanded includes I, J (I and J cite A)."""
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed)

        result_ids = {p.paper_id for p in result}
        assert "I" in result_ids
        assert "J" in result_ids


class TestExpansionDeduplication:
    """Test that expansion deduplicates across seed papers."""

    def test_expansion_deduplication(self, agent: ExpansionAgent) -> None:
        """Seed = [A, B] -> D appears only once (both A and B cite D)."""
        seed = [_make_scored_paper("A"), _make_scored_paper("B")]
        result = agent.expand(seed)

        paper_ids = [p.paper_id for p in result]
        assert paper_ids.count("D") == 1


class TestExpansionPreservesOriginals:
    """Test that all original seed papers remain in the output."""

    def test_expansion_preserves_originals(self, agent: ExpansionAgent) -> None:
        """Seed = [A, B, C] -> all of A, B, C are in the output."""
        seed = [
            _make_scored_paper("A"),
            _make_scored_paper("B"),
            _make_scored_paper("C"),
        ]
        result = agent.expand(seed)

        result_ids = {p.paper_id for p in result}
        assert "A" in result_ids
        assert "B" in result_ids
        assert "C" in result_ids

    def test_originals_appear_first(self, agent: ExpansionAgent) -> None:
        """Original candidates come before expanded ones in the output."""
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed)

        # First paper should be the original seed
        assert result[0].paper_id == "A"
        assert result[0].similarity_score == 0.9  # Original score preserved


class TestMaxExpansionLimit:
    """Test that max_expansion parameter is respected."""

    def test_max_expansion_limit(self, agent: ExpansionAgent) -> None:
        """Seed = [A, B, C], max_expansion=3 -> at most 3 new papers added."""
        seed = [
            _make_scored_paper("A"),
            _make_scored_paper("B"),
            _make_scored_paper("C"),
        ]
        result = agent.expand(seed, max_expansion=3)

        # Total should be at most seed_count + max_expansion
        new_papers = [p for p in result if p.paper_id not in {"A", "B", "C"}]
        assert len(new_papers) <= 3

    def test_max_expansion_zero(self, agent: ExpansionAgent) -> None:
        """max_expansion=0 -> no expansion, only originals returned."""
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed, max_expansion=0)

        assert len(result) == 1
        assert result[0].paper_id == "A"


class TestPrioritizationMultiCitation:
    """Test that papers cited by multiple seeds are prioritized."""

    def test_prioritization_multi_citation(self, agent: ExpansionAgent) -> None:
        """Seed = [A, B] -> D ranked before G.

        D is cited by both A and B (multi-citation count = 2).
        G is cited only by B (multi-citation count = 1).
        """
        seed = [_make_scored_paper("A"), _make_scored_paper("B")]
        result = agent.expand(seed)

        # Get only the expanded papers (exclude seeds)
        expanded = [p for p in result if p.paper_id not in {"A", "B"}]
        expanded_ids = [p.paper_id for p in expanded]

        d_index = expanded_ids.index("D")
        g_index = expanded_ids.index("G")
        assert d_index < g_index, (
            f"D (cited by 2 seeds) should appear before G (cited by 1 seed), "
            f"but D is at index {d_index} and G at {g_index}"
        )

    def test_secondary_prioritization_by_citation_count(self, agent: ExpansionAgent) -> None:
        """Among papers with equal multi-citation count, higher citation_count wins.

        Seed = [A]:
        - D (citation_count=1000), E (citation_count=800), F (citation_count=50)
          all have multi-citation count = 1.
        - D should rank before E, E before F.
        """
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed)

        expanded = [p for p in result if p.paper_id != "A"]
        expanded_ids = [p.paper_id for p in expanded]

        # D, E, F should all appear; among references, D has highest citation_count
        assert "D" in expanded_ids
        assert "E" in expanded_ids
        assert "F" in expanded_ids

        d_index = expanded_ids.index("D")
        f_index = expanded_ids.index("F")
        assert d_index < f_index, (
            "D (citation_count=1000) should appear before F (citation_count=50)"
        )


class TestIsolatedPaper:
    """Test papers with no citation edges."""

    def test_isolated_paper_no_crash(self, agent: ExpansionAgent) -> None:
        """Seed = [L] -> output = [L] (no expansion, no error)."""
        seed = [_make_scored_paper("L")]
        result = agent.expand(seed)

        assert len(result) == 1
        assert result[0].paper_id == "L"


class TestExpansionExcludesSeedPapers:
    """Test that seed papers are not duplicated as expansion results."""

    def test_expansion_excludes_seed_papers(self, agent: ExpansionAgent) -> None:
        """Seed = [A, I] -> A appears once (not duplicated via I's reference to A)."""
        seed = [_make_scored_paper("A"), _make_scored_paper("I")]
        result = agent.expand(seed)

        paper_ids = [p.paper_id for p in result]
        assert paper_ids.count("A") == 1
        assert paper_ids.count("I") == 1


class TestOnlyCorpusPapersIncluded:
    """Test that expansion papers not in the corpus are filtered out."""

    def test_only_corpus_papers_included(self) -> None:
        """Expansion paper IDs that aren't in the corpus are filtered out."""
        db = sqlite3.connect(":memory:")
        create_tables(db)

        # Insert only papers A and D in the corpus
        insert_papers_batch(db, [_make_paper("A"), _make_paper("D")])

        # Create edge A -> D, A -> Z (Z is NOT in the corpus)
        insert_citation_edges_batch(
            db,
            [
                CitationEdge(source_id="A", target_id="D"),
                CitationEdge(source_id="A", target_id="Z"),
            ],
        )

        agent = ExpansionAgent(ExpansionAgentConfig(), db)
        seed = [_make_scored_paper("A")]
        result = agent.expand(seed)

        result_ids = {p.paper_id for p in result}
        assert "D" in result_ids
        assert "Z" not in result_ids

        db.close()


class TestExpansionRunMethod:
    """Test the run() method (state dict interface)."""

    def test_run_populates_metadata(self, agent: ExpansionAgent) -> None:
        """run() populates metadata with latency and counts."""
        state = {
            "user_text": "test",
            "retrieval_candidates": [
                _make_scored_paper("A"),
                _make_scored_paper("B"),
            ],
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert "agent3_latency_s" in result["metadata"]
        assert result["metadata"]["agent3_latency_s"] >= 0
        assert result["metadata"]["agent3_expanded_count"] > 0
        assert result["metadata"]["agent3_new_papers"] >= 0

    def test_run_with_empty_candidates(self, agent: ExpansionAgent) -> None:
        """run() with empty retrieval_candidates returns empty list."""
        state = {
            "user_text": "test",
            "retrieval_candidates": [],
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        assert result["expanded_candidates"] == []

    def test_run_returns_scored_papers(self, agent: ExpansionAgent) -> None:
        """run() returns ScoredPaper objects in expanded_candidates."""
        state = {
            "user_text": "test",
            "retrieval_candidates": [_make_scored_paper("A")],
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        candidates = result["expanded_candidates"]
        assert len(candidates) > 1  # At least seed + some expanded
        assert all(isinstance(c, ScoredPaper) for c in candidates)

    def test_run_expanded_papers_have_zero_similarity(self, agent: ExpansionAgent) -> None:
        """Expanded papers (not from retrieval) have similarity_score=0.0."""
        state = {
            "user_text": "test",
            "retrieval_candidates": [_make_scored_paper("A", score=0.9)],
            "metadata": {},
            "errors": [],
        }
        result = agent.run(state)

        candidates = result["expanded_candidates"]
        # Original should keep its score
        assert candidates[0].similarity_score == 0.9
        # Expanded papers should have score 0.0
        for p in candidates[1:]:
            assert p.similarity_score == 0.0


class TestExpansionNoDuplicatesInOutput:
    """Test that the combined output never has duplicate paper IDs."""

    def test_no_duplicates_in_full_output(self, agent: ExpansionAgent) -> None:
        """Full expansion from multiple seeds has no duplicate paper IDs."""
        seed = [
            _make_scored_paper("A"),
            _make_scored_paper("B"),
            _make_scored_paper("C"),
        ]
        result = agent.expand(seed)

        paper_ids = [p.paper_id for p in result]
        assert len(paper_ids) == len(set(paper_ids)), f"Duplicate paper IDs found: {paper_ids}"


# ==================== Integration Test ====================


@pytest.mark.integration
class TestExpansionIntegration:
    """Integration test: run Agents 1→2→3 in sequence."""

    def test_integration_agents_1_2_3(self) -> None:
        """Run Agents 1→2→3 end-to-end with a real query.

        Expanded set should be larger than retrieval-only set.
        Requires Qdrant running, corpus indexed, and OPENAI_API_KEY set.
        Set PYTEST_ALLOW_REAL_API=1 to run.
        """
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        from openai import OpenAI

        from src.agents.query_agent import QueryAgent, QueryAgentConfig
        from src.agents.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
        from src.config import Settings
        from src.indexing.embedder import Embedder, EmbedderConfig
        from src.indexing.qdrant_store import QdrantConfig, QdrantStore

        settings = Settings()

        # Set up dependencies
        qdrant_store = QdrantStore(QdrantConfig())
        if not qdrant_store.collection_exists():
            pytest.skip("Qdrant collection 'papers' not found — run index_corpus.py first")

        embedder = Embedder(EmbedderConfig(show_progress=False))
        openai_client = OpenAI(api_key=api_key)

        db = sqlite3.connect(settings.db_path)

        # Agent 1: Query Analysis
        query_agent = QueryAgent(QueryAgentConfig(), openai_client)
        state: dict = {
            "user_text": (
                "Attention mechanisms have become an integral part of compelling "
                "sequence modeling and transduction models, allowing modeling of "
                "dependencies without regard to their distance in the input or "
                "output sequences."
            ),
            "metadata": {},
            "errors": [],
        }
        state.update(query_agent.run(state))

        # Agent 2: Retrieval
        retrieval_agent = RetrievalAgent(RetrievalAgentConfig(top_n=10), embedder, qdrant_store)
        state.update(retrieval_agent.run(state))
        retrieval_count = len(state["retrieval_candidates"])

        # Agent 3: Expansion
        expansion_agent = ExpansionAgent(ExpansionAgentConfig(), db)
        state.update(expansion_agent.run(state))
        expanded_count = len(state["expanded_candidates"])

        # The expanded set should be at least as large as retrieval
        assert expanded_count >= retrieval_count
        # All expanded candidates should be ScoredPaper
        assert all(isinstance(c, ScoredPaper) for c in state["expanded_candidates"])

        db.close()
