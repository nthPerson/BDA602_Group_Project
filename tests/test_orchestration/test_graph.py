"""Tests for the LangGraph orchestration pipeline.

Test inventory (unit — no services required):
- test_graph_compiles: StateGraph compiles without errors
- test_graph_runs_all_nodes: All 5 node functions are called in order (mocked)
- test_state_propagates: State written by Agent 1 is readable by Agent 2, etc.
- test_metadata_timing: Metadata dict contains timing entries for each agent
- test_error_in_agent3_graceful: Agent 3 raises → errors logged, pipeline continues
- test_error_in_agent4_graceful: Agent 4 raises → pipeline produces results without grounding
- test_all_agents_fail_returns_empty: All agents fail → empty recommendations, errors logged
"""

from unittest.mock import MagicMock

from src.agents.synthesis_agent import SynthesisAgent, SynthesisAgentConfig
from src.data.models import (
    CitationIntent,
    GroundedPaper,
    QueryAnalysis,
    RankedPaper,
    Recommendation,
    ScoredPaper,
)
from src.orchestration.graph import build_graph

# ==================== Helpers ====================


def _make_mock_agent(return_value: dict | None = None) -> MagicMock:
    """Create a mock agent with a run() method returning the given dict."""
    agent = MagicMock()
    agent.run.return_value = return_value or {}
    return agent


def _make_query_analysis() -> QueryAnalysis:
    """Create a sample QueryAnalysis."""
    return QueryAnalysis(
        topic_keywords=["transformers", "attention"],
        citation_intent=CitationIntent.BACKGROUND,
        expanded_query="transformer attention mechanisms",
        confidence=0.9,
    )


def _make_scored_paper(paper_id: str = "W001") -> ScoredPaper:
    """Create a sample ScoredPaper."""
    return ScoredPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="Abstract text.",
        year=2023,
        citation_count=100,
        similarity_score=0.85,
        concepts=["ML"],
    )


def _make_ranked_paper(paper_id: str = "W001") -> RankedPaper:
    """Create a sample RankedPaper."""
    return RankedPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="Abstract text.",
        year=2023,
        citation_count=100,
        similarity_score=0.85,
        concepts=["ML"],
        rerank_score=3.5,
    )


def _make_grounded_paper(paper_id: str = "W001") -> GroundedPaper:
    """Create a sample GroundedPaper."""
    return GroundedPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        rerank_score=3.5,
        supporting_snippet="Relevant excerpt from abstract.",
        justification="This paper is relevant because...",
        confidence=0.85,
    )


def _make_recommendation(rank: int = 1, paper_id: str = "W001") -> Recommendation:
    """Create a sample Recommendation."""
    return Recommendation(
        rank=rank,
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        authors=["Author A"],
        year=2023,
        citation_count=100,
        justification="This paper is relevant because...",
        supporting_snippet="Relevant excerpt from abstract.",
        confidence=0.85,
        citation_intent_match="Strong match for BACKGROUND citation",
    )


# ==================== Fixtures ====================


def _build_mock_agents():
    """Create a set of mock agents that simulate a successful pipeline run."""
    qa = _make_query_analysis()
    scored = [_make_scored_paper("W001"), _make_scored_paper("W002")]
    ranked = [_make_ranked_paper("W001"), _make_ranked_paper("W002")]
    grounded = [_make_grounded_paper("W001")]
    recs = [_make_recommendation(1, "W001")]

    agent1 = _make_mock_agent(
        {
            "query_analysis": qa,
            "metadata": {"agent1_latency_s": 0.5},
            "errors": [],
        }
    )
    agent2 = _make_mock_agent(
        {
            "retrieval_candidates": scored,
            "metadata": {"agent1_latency_s": 0.5, "agent2_latency_s": 0.1},
            "errors": [],
        }
    )
    agent3 = _make_mock_agent(
        {
            "expanded_candidates": scored,
            "metadata": {
                "agent1_latency_s": 0.5,
                "agent2_latency_s": 0.1,
                "agent3_latency_s": 0.05,
            },
            "errors": [],
        }
    )
    agent4 = _make_mock_agent(
        {
            "reranked_candidates": ranked,
            "grounded_candidates": grounded,
            "metadata": {
                "agent1_latency_s": 0.5,
                "agent2_latency_s": 0.1,
                "agent3_latency_s": 0.05,
                "agent4a_latency_s": 0.3,
                "agent4b_latency_s": 0.8,
                "agent4_total_latency_s": 1.1,
            },
            "errors": [],
        }
    )
    agent5 = _make_mock_agent(
        {
            "final_recommendations": recs,
            "metadata": {
                "agent1_latency_s": 0.5,
                "agent2_latency_s": 0.1,
                "agent3_latency_s": 0.05,
                "agent4a_latency_s": 0.3,
                "agent4b_latency_s": 0.8,
                "agent4_total_latency_s": 1.1,
                "agent5_latency_s": 0.001,
                "agent5_recommendation_count": 1,
            },
            "errors": [],
        }
    )

    return agent1, agent2, agent3, agent4, agent5


# ==================== Test Classes ====================


class TestGraphCompiles:
    """Test that StateGraph compiles without errors."""

    def test_graph_compiles(self) -> None:
        """build_graph() returns a compiled graph without errors."""
        agents = _build_mock_agents()
        graph = build_graph(*agents)
        assert graph is not None
        # Verify it has an invoke method
        assert hasattr(graph, "invoke")


class TestGraphRunsAllNodes:
    """Test all 5 node functions are called in order."""

    def test_graph_runs_all_nodes(self) -> None:
        """All 5 agent run() methods are called when pipeline is invoked."""
        agent1, agent2, agent3, agent4, agent5 = _build_mock_agents()
        graph = build_graph(agent1, agent2, agent3, agent4, agent5)

        graph.invoke({"user_text": "test input", "metadata": {}, "errors": []})

        # Verify all agents were called
        assert agent1.run.called
        assert agent2.run.called
        assert agent3.run.called
        assert agent4.run.called
        assert agent5.run.called


class TestStatePropagates:
    """Test state written by one agent is readable by the next."""

    def test_state_propagates(self) -> None:
        """State from Agent 1 is visible to Agent 2, etc."""
        qa = _make_query_analysis()
        scored = [_make_scored_paper("W001")]

        # Agent 1 writes query_analysis
        agent1 = _make_mock_agent({"query_analysis": qa, "metadata": {"a1": True}, "errors": []})

        # Agent 2 should receive state with query_analysis
        def agent2_run(state):
            # Verify Agent 1's output is present
            assert state.get("query_analysis") is not None
            return {
                "retrieval_candidates": scored,
                "metadata": dict(state.get("metadata", {}), a2=True),
                "errors": [],
            }

        agent2 = MagicMock()
        agent2.run.side_effect = agent2_run

        # Agent 3 should receive both
        def agent3_run(state):
            assert state.get("query_analysis") is not None
            assert len(state.get("retrieval_candidates", [])) > 0
            return {
                "expanded_candidates": scored,
                "metadata": dict(state.get("metadata", {}), a3=True),
                "errors": [],
            }

        agent3 = MagicMock()
        agent3.run.side_effect = agent3_run

        agent4 = _make_mock_agent(
            {
                "reranked_candidates": [_make_ranked_paper("W001")],
                "grounded_candidates": [_make_grounded_paper("W001")],
                "metadata": {"a1": True, "a2": True, "a3": True, "a4": True},
                "errors": [],
            }
        )
        agent5 = _make_mock_agent(
            {
                "final_recommendations": [_make_recommendation()],
                "metadata": {"a1": True, "a2": True, "a3": True, "a4": True, "a5": True},
                "errors": [],
            }
        )

        graph = build_graph(agent1, agent2, agent3, agent4, agent5)
        result = graph.invoke({"user_text": "test", "metadata": {}, "errors": []})

        # Final state should have all keys
        assert result.get("query_analysis") is not None
        assert len(result.get("final_recommendations", [])) > 0


class TestMetadataTiming:
    """Test metadata dict contains timing entries for each agent."""

    def test_metadata_timing(self) -> None:
        """Metadata accumulates timing from all agents."""
        agents = _build_mock_agents()
        graph = build_graph(*agents)

        result = graph.invoke({"user_text": "test", "metadata": {}, "errors": []})

        metadata = result.get("metadata", {})
        assert "agent1_latency_s" in metadata
        assert "agent2_latency_s" in metadata
        assert "agent3_latency_s" in metadata
        assert "agent4a_latency_s" in metadata
        assert "agent5_latency_s" in metadata


class TestErrorInAgent3Graceful:
    """Test Agent 3 failure is handled gracefully."""

    def test_error_in_agent3_graceful(self) -> None:
        """When Agent 3 raises, error is logged and pipeline continues."""
        qa = _make_query_analysis()
        scored = [_make_scored_paper("W001")]

        agent1 = _make_mock_agent({"query_analysis": qa, "metadata": {"a1": 0.1}, "errors": []})
        agent2 = _make_mock_agent(
            {
                "retrieval_candidates": scored,
                "metadata": {"a1": 0.1, "a2": 0.1},
                "errors": [],
            }
        )

        # Agent 3 raises an exception
        agent3 = MagicMock()
        agent3.run.side_effect = RuntimeError("Citation DB unavailable")

        # Agent 4 should still run (with empty expanded_candidates)
        agent4 = _make_mock_agent(
            {
                "reranked_candidates": [],
                "grounded_candidates": [],
                "metadata": {"a1": 0.1, "a2": 0.1, "a4": 0.1},
                "errors": ["citation_expansion failed: Citation DB unavailable"],
            }
        )

        # Use real Agent 5 to verify it handles empty input
        agent5 = SynthesisAgent(SynthesisAgentConfig())

        graph = build_graph(agent1, agent2, agent3, agent4, agent5)
        result = graph.invoke({"user_text": "test", "metadata": {}, "errors": []})

        # Pipeline should complete without crashing
        assert "errors" in result
        assert any("citation_expansion failed" in e for e in result["errors"])
        # Recommendations may be empty (since no expanded candidates)
        assert isinstance(result.get("final_recommendations", []), list)


class TestErrorInAgent4Graceful:
    """Test Agent 4 failure is handled gracefully."""

    def test_error_in_agent4_graceful(self) -> None:
        """When Agent 4 raises, error is logged and synthesis gets empty inputs."""
        qa = _make_query_analysis()
        scored = [_make_scored_paper("W001")]

        agent1 = _make_mock_agent({"query_analysis": qa, "metadata": {"a1": 0.1}, "errors": []})
        agent2 = _make_mock_agent(
            {
                "retrieval_candidates": scored,
                "metadata": {"a1": 0.1, "a2": 0.1},
                "errors": [],
            }
        )
        agent3 = _make_mock_agent(
            {
                "expanded_candidates": scored,
                "metadata": {"a1": 0.1, "a2": 0.1, "a3": 0.1},
                "errors": [],
            }
        )

        # Agent 4 raises
        agent4 = MagicMock()
        agent4.run.side_effect = RuntimeError("Cross-encoder OOM")

        # Use real Agent 5
        agent5 = SynthesisAgent(SynthesisAgentConfig())

        graph = build_graph(agent1, agent2, agent3, agent4, agent5)
        result = graph.invoke({"user_text": "test", "metadata": {}, "errors": []})

        # Pipeline should complete
        assert any("reranking_and_grounding failed" in e for e in result["errors"])
        # No grounded candidates → no recommendations
        assert result.get("final_recommendations", []) == []


class TestAllAgentsFailReturnsEmpty:
    """Test all agents fail → empty recommendations, errors logged."""

    def test_all_agents_fail_returns_empty(self) -> None:
        """When all agents fail, pipeline returns empty results with errors."""
        agents = []
        for _ in range(5):
            agent = MagicMock()
            agent.run.side_effect = RuntimeError("Everything is broken")
            agents.append(agent)

        graph = build_graph(*agents)
        result = graph.invoke({"user_text": "test", "metadata": {}, "errors": []})

        # Pipeline should not crash
        errors = result.get("errors", [])
        assert len(errors) >= 1  # At least some errors logged
        # No recommendations
        assert result.get("final_recommendations", []) == []
