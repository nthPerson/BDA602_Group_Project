"""Agent 1: Query & Intent Analysis.

This agent transforms raw user text (a paragraph, claim, or draft section)
into a structured query representation. It uses an LLM (gpt-4o-mini via
instructor) to extract keywords, classify citation intent, and reformulate
the text into an effective retrieval query.

If the LLM is unavailable, the agent falls back to simple keyword extraction
to ensure the pipeline can still produce results.
"""

import logging
import re
import time
from dataclasses import dataclass

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from src.data.models import CitationIntent, QueryAnalysis
from src.orchestration.state import PipelineState

# ==================== Pydantic model for instructor ====================


class QueryAnalysisResponse(BaseModel):
    """Pydantic model for structured LLM output via instructor.

    This mirrors the QueryAnalysis dataclass but uses Pydantic for
    instructor's structured output validation.
    """

    topic_keywords: list[str] = Field(
        description=(
            "Key technical terms and concepts extracted from the text. "
            "Include specific methods, models, datasets, and domain terms. "
            "3-8 keywords, ordered by relevance."
        ),
    )
    citation_intent: str = Field(
        description=(
            "The type of citation the text is looking for. Must be one of: "
            "'background' (general context/motivation), "
            "'method' (a specific technique being used or extended), "
            "'comparison' (work being compared against), "
            "'benchmark' (datasets, benchmarks, or evaluation standards)."
        ),
    )
    expanded_query: str = Field(
        description=(
            "A retrieval-optimized reformulation of the input text. "
            "This should be a concise, keyword-rich query suitable for "
            "semantic search over academic paper abstracts. Include synonyms "
            "and related terms that the original text implies but doesn't state."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis (0-1). Lower if the text is ambiguous.",
    )


# ==================== Agent Configuration ====================

SYSTEM_PROMPT = (
    "You are an AI research librarian specializing in machine learning, "
    "artificial intelligence, and computer science. Analyze the given text "
    "from an AI/ML research paper and extract structured information to help "
    "find relevant citation candidates.\n\n"
    "Your task:\n"
    "1. Extract key technical terms and concepts (methods, models, datasets, "
    "domain-specific terms).\n"
    "2. Classify the citation intent — what kind of paper is being referenced:\n"
    "   - 'background': General context, motivation, or prior work overview\n"
    "   - 'method': A specific technique, algorithm, or approach being used/extended\n"
    "   - 'comparison': Work being compared against or contrasted with\n"
    "   - 'benchmark': Datasets, benchmarks, metrics, or evaluation standards\n"
    "3. Reformulate the text into an effective search query for finding "
    "relevant papers in a semantic search engine over academic abstracts.\n\n"
    "Be precise and specific. Focus on technical concepts rather than "
    "generic language."
)


@dataclass
class QueryAgentConfig:
    """Configuration for the Query & Intent Analysis agent.

    Attributes:
        model: OpenAI model to use for query analysis.
        max_retries: Maximum number of retries for structured output.
        temperature: LLM temperature (lower = more deterministic).
        timeout: Maximum seconds to wait for LLM response.
    """

    model: str = "gpt-4o-mini"
    max_retries: int = 2
    temperature: float = 0.0
    timeout: float = 30.0


# ==================== Fallback keyword extraction ====================

# Common English stop words to filter out
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it that this with by from as are was "
    "were be been being have has had do does did will would could should may might "
    "shall can not no nor so if then than too very also just about above after again "
    "all any because before between both but each few how into more most much other "
    "out over own same she some such there these they those through under until up "
    "we what when where which while who why our their its his her he him them me my "
    "your i you we us".split()
)


def _fallback_keyword_extraction(text: str) -> list[str]:
    """Extract keywords from text without using an LLM.

    Uses simple heuristics: tokenize, remove stop words, and keep
    longer words that are likely technical terms.

    Args:
        text: Input text to extract keywords from.

    Returns:
        List of extracted keywords (up to 10).
    """
    # Tokenize: extract words and multi-word terms
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())

    # Filter out stop words and very short words
    candidates = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    # Count frequency to find important terms
    freq: dict[str, int] = {}
    for word in candidates:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency, then by length (longer = more specific)
    sorted_words = sorted(freq.keys(), key=lambda w: (freq[w], len(w)), reverse=True)

    # Deduplicate and take top 10
    seen: set[str] = set()
    keywords: list[str] = []
    for word in sorted_words:
        if word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) >= 10:
            break

    return keywords


# ==================== Agent Implementation ====================


class QueryAgent:
    """Agent 1: Query & Intent Analysis.

    Transforms raw user text into a structured QueryAnalysis using an LLM.
    Falls back to simple keyword extraction if the LLM is unavailable.
    """

    def __init__(self, config: QueryAgentConfig, openai_client: OpenAI) -> None:
        """Initialize the query agent.

        Args:
            config: Agent configuration.
            openai_client: OpenAI client instance (will be wrapped with instructor).
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = instructor.from_openai(openai_client)

    def run(self, state: PipelineState) -> dict:
        """Execute the query analysis step.

        Reads `user_text` from state, produces `query_analysis`.

        Args:
            state: Current pipeline state with `user_text` populated.

        Returns:
            Dict with `query_analysis` and updated `metadata`.
        """
        user_text = state["user_text"]
        metadata = dict(state.get("metadata", {}))
        errors = list(state.get("errors", []))

        start_time = time.time()

        try:
            analysis = self.analyze(user_text)
            self.logger.info(
                f"Query analysis complete: intent={analysis.citation_intent}, "
                f"keywords={analysis.topic_keywords}"
            )
        except Exception as e:
            self.logger.warning(f"LLM analysis failed ({e}), using fallback")
            errors.append(f"Agent 1 LLM failed: {e}")
            analysis = self._fallback_analyze(user_text)

        elapsed = time.time() - start_time
        metadata["agent1_latency_s"] = round(elapsed, 3)

        return {
            "query_analysis": analysis,
            "metadata": metadata,
            "errors": errors,
        }

    def analyze(self, user_text: str) -> QueryAnalysis:
        """Analyze user text using the LLM to produce a structured query.

        Args:
            user_text: Raw text from the user (paragraph, claim, or draft section).

        Returns:
            QueryAnalysis with keywords, intent, expanded query, and confidence.

        Raises:
            Exception: If the LLM fails after all retries.
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            response_model=QueryAnalysisResponse,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
        )

        # Map the string intent to the CitationIntent enum
        intent = self._parse_intent(response.citation_intent)

        return QueryAnalysis(
            topic_keywords=response.topic_keywords,
            citation_intent=intent,
            expanded_query=response.expanded_query,
            confidence=response.confidence,
        )

    def _fallback_analyze(self, user_text: str) -> QueryAnalysis:
        """Produce a QueryAnalysis without using an LLM.

        Uses simple keyword extraction and defaults to BACKGROUND intent.

        Args:
            user_text: Raw text from the user.

        Returns:
            QueryAnalysis with extracted keywords, BACKGROUND intent,
            the raw text as expanded_query, and low confidence.
        """
        keywords = _fallback_keyword_extraction(user_text)

        return QueryAnalysis(
            topic_keywords=keywords,
            citation_intent=CitationIntent.BACKGROUND,
            expanded_query=user_text,
            confidence=0.3,
        )

    @staticmethod
    def _parse_intent(intent_str: str) -> CitationIntent:
        """Parse a string intent value into a CitationIntent enum.

        Args:
            intent_str: String intent from the LLM response.

        Returns:
            CitationIntent enum value. Defaults to BACKGROUND if unrecognized.
        """
        intent_lower = intent_str.lower().strip()
        try:
            return CitationIntent(intent_lower)
        except ValueError:
            return CitationIntent.BACKGROUND
