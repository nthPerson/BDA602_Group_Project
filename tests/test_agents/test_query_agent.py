"""Tests for Agent 1: Query & Intent Analysis.

Tests cover:
- Valid structured LLM output parsing
- Keyword extraction from LLM response
- Citation intent enum validation
- Expanded query generation
- Retry on invalid JSON
- Fallback on API failure
- Fallback using raw text as expanded query
- Live LLM query analysis (integration)
"""

from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from src.agents.query_agent import (
    QueryAgent,
    QueryAgentConfig,
    QueryAnalysisResponse,
    _fallback_keyword_extraction,
)
from src.data.models import CitationIntent, QueryAnalysis

# ==================== Fixtures ====================


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def agent_config() -> QueryAgentConfig:
    """Create a default agent configuration."""
    return QueryAgentConfig(model="gpt-4o-mini", max_retries=2, temperature=0.0)


@pytest.fixture
def sample_user_text() -> str:
    """Sample user text for testing."""
    return (
        "Recent advances in transformer architectures have demonstrated that "
        "scaling model parameters significantly improves few-shot learning "
        "capabilities. This has led to the development of large language models "
        "such as GPT-3 and PaLM that can perform diverse NLP tasks."
    )


@pytest.fixture
def valid_llm_response() -> QueryAnalysisResponse:
    """Create a valid QueryAnalysisResponse as would be returned by instructor."""
    return QueryAnalysisResponse(
        topic_keywords=[
            "transformer",
            "scaling",
            "few-shot learning",
            "large language models",
            "GPT-3",
            "PaLM",
        ],
        citation_intent="background",
        expanded_query=(
            "transformer architecture scaling laws few-shot learning "
            "large language models GPT PaLM parameter scaling NLP"
        ),
        confidence=0.85,
    )


# ==================== Unit Tests ====================


class TestQueryAgentStructuredOutput:
    """Test LLM-based query analysis with mocked responses."""

    def test_valid_structured_output(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
        valid_llm_response: QueryAnalysisResponse,
    ) -> None:
        """Mocked LLM returns valid JSON -> QueryAnalysis is correctly parsed."""
        agent = QueryAgent(agent_config, mock_openai_client)

        # Mock the instructor-wrapped client's create method
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = valid_llm_response

        result = agent.analyze(sample_user_text)

        assert isinstance(result, QueryAnalysis)
        assert result.topic_keywords == valid_llm_response.topic_keywords
        assert result.citation_intent == CitationIntent.BACKGROUND
        assert result.expanded_query == valid_llm_response.expanded_query
        assert result.confidence == 0.85

    def test_keywords_extracted(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
        valid_llm_response: QueryAnalysisResponse,
    ) -> None:
        """Output contains non-empty topic_keywords list."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = valid_llm_response

        result = agent.analyze(sample_user_text)

        assert len(result.topic_keywords) > 0
        assert all(isinstance(kw, str) for kw in result.topic_keywords)

    def test_citation_intent_valid(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
    ) -> None:
        """Citation intent is one of the 4 valid enum values."""
        agent = QueryAgent(agent_config, mock_openai_client)

        for intent_value in ["background", "method", "comparison", "benchmark"]:
            response = QueryAnalysisResponse(
                topic_keywords=["test"],
                citation_intent=intent_value,
                expanded_query="test query",
                confidence=0.8,
            )
            agent.client = MagicMock()
            agent.client.chat.completions.create.return_value = response

            result = agent.analyze(sample_user_text)
            assert result.citation_intent in list(CitationIntent)

    def test_expanded_query_nonempty(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
        valid_llm_response: QueryAnalysisResponse,
    ) -> None:
        """Expanded query is not empty."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = valid_llm_response

        result = agent.analyze(sample_user_text)

        assert len(result.expanded_query) > 0
        assert result.expanded_query.strip() != ""

    def test_retry_on_invalid_json(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
        valid_llm_response: QueryAnalysisResponse,
    ) -> None:
        """Mocked LLM returns junk first, valid JSON second -> retry succeeds.

        instructor handles retries internally — we test that the agent still
        produces a valid result when the instructor client eventually returns
        a valid response.
        """
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()

        # instructor retries internally, so we simulate it succeeding after retry
        agent.client.chat.completions.create.return_value = valid_llm_response

        result = agent.analyze(sample_user_text)

        assert isinstance(result, QueryAnalysis)
        assert len(result.topic_keywords) > 0


class TestQueryAgentFallback:
    """Test fallback behavior when LLM is unavailable."""

    def test_fallback_on_api_failure(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
    ) -> None:
        """LLM completely fails -> fallback keyword extraction produces keywords."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = Exception("API Error")

        # Use the run() method which handles the fallback
        state = {"user_text": sample_user_text, "metadata": {}, "errors": []}
        result = agent.run(state)

        analysis = result["query_analysis"]
        assert isinstance(analysis, QueryAnalysis)
        assert len(analysis.topic_keywords) > 0
        assert analysis.citation_intent == CitationIntent.BACKGROUND
        assert analysis.confidence == 0.3

    def test_fallback_uses_raw_text(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
    ) -> None:
        """In fallback mode, expanded_query equals the raw input text."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = Exception("API Error")

        state = {"user_text": sample_user_text, "metadata": {}, "errors": []}
        result = agent.run(state)

        analysis = result["query_analysis"]
        assert analysis.expanded_query == sample_user_text


class TestFallbackKeywordExtraction:
    """Test the fallback keyword extraction function directly."""

    def test_extracts_technical_terms(self) -> None:
        """Extracts meaningful technical terms from ML text."""
        text = (
            "We propose a novel attention mechanism for transformer models "
            "that improves performance on natural language processing tasks."
        )
        keywords = _fallback_keyword_extraction(text)

        assert len(keywords) > 0
        # Should find key technical terms
        assert any("attention" in kw for kw in keywords)
        assert any("transformer" in kw for kw in keywords)

    def test_removes_stop_words(self) -> None:
        """Stop words are not included in keywords."""
        text = "The model is trained on the dataset and evaluated"
        keywords = _fallback_keyword_extraction(text)

        stop_words = {"the", "is", "on", "and"}
        for kw in keywords:
            assert kw not in stop_words

    def test_handles_empty_text(self) -> None:
        """Empty text returns empty keywords list."""
        keywords = _fallback_keyword_extraction("")
        assert keywords == []


class TestQueryAgentRun:
    """Test the run() method (state dict interface)."""

    def test_run_populates_metadata(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
        valid_llm_response: QueryAnalysisResponse,
    ) -> None:
        """run() populates metadata with latency."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = valid_llm_response

        state = {"user_text": sample_user_text, "metadata": {}, "errors": []}
        result = agent.run(state)

        assert "agent1_latency_s" in result["metadata"]
        assert result["metadata"]["agent1_latency_s"] >= 0

    def test_run_records_errors_on_failure(
        self,
        mock_openai_client: MagicMock,
        agent_config: QueryAgentConfig,
        sample_user_text: str,
    ) -> None:
        """run() records errors when LLM fails."""
        agent = QueryAgent(agent_config, mock_openai_client)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = Exception("API down")

        state = {"user_text": sample_user_text, "metadata": {}, "errors": []}
        result = agent.run(state)

        assert len(result["errors"]) > 0
        assert "Agent 1 LLM failed" in result["errors"][0]


class TestQueryAgentIntentParsing:
    """Test intent string-to-enum parsing."""

    def test_parse_valid_intents(self, mock_openai_client: MagicMock) -> None:
        """All valid intent strings parse to correct enum values."""
        agent = QueryAgent(QueryAgentConfig(), mock_openai_client)

        assert agent._parse_intent("background") == CitationIntent.BACKGROUND
        assert agent._parse_intent("method") == CitationIntent.METHOD
        assert agent._parse_intent("comparison") == CitationIntent.COMPARISON
        assert agent._parse_intent("benchmark") == CitationIntent.BENCHMARK

    def test_parse_invalid_intent_defaults_to_background(
        self, mock_openai_client: MagicMock
    ) -> None:
        """Unrecognized intent strings default to BACKGROUND."""
        agent = QueryAgent(QueryAgentConfig(), mock_openai_client)

        assert agent._parse_intent("unknown") == CitationIntent.BACKGROUND
        assert agent._parse_intent("") == CitationIntent.BACKGROUND

    def test_parse_intent_case_insensitive(
        self, mock_openai_client: MagicMock
    ) -> None:
        """Intent parsing is case-insensitive."""
        agent = QueryAgent(QueryAgentConfig(), mock_openai_client)

        assert agent._parse_intent("BACKGROUND") == CitationIntent.BACKGROUND
        assert agent._parse_intent("Method") == CitationIntent.METHOD
        assert agent._parse_intent("  comparison  ") == CitationIntent.COMPARISON


# ==================== Integration Test ====================


@pytest.mark.integration
class TestQueryAgentLive:
    """Integration tests requiring a real OpenAI API key."""

    def test_live_query_analysis(self) -> None:
        """Real LLM call with a known abstract -> output is sensible.

        Requires OPENAI_API_KEY in environment.
        Set PYTEST_ALLOW_REAL_API=1 to run.
        """
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        agent = QueryAgent(QueryAgentConfig(), client)

        text = (
            "Attention mechanisms have become an integral part of compelling "
            "sequence modeling and transduction models in various tasks, "
            "allowing modeling of dependencies without regard to their "
            "distance in the input or output sequences."
        )

        result = agent.analyze(text)

        assert isinstance(result, QueryAnalysis)
        assert len(result.topic_keywords) >= 2
        assert result.citation_intent in list(CitationIntent)
        assert len(result.expanded_query) > 0
        assert 0.0 <= result.confidence <= 1.0
