"""Tests for evaluation metrics: Recall@K, MRR, MAP.

These are pure math tests — no external services required. The test cases
use hand-verified inputs and expected outputs from the Stage 8 development plan.
"""

import pytest

from src.evaluation.metrics import average_precision, mrr, recall_at_k

# ==================== Recall@K ====================


class TestRecallAtKPerfect:
    """All ground truth found in top K."""

    def test_recall_at_k_perfect(self) -> None:
        recommended = ["A", "B", "C"]
        ground_truth = ["A", "B", "C"]
        assert recall_at_k(recommended, ground_truth, k=3) == 1.0


class TestRecallAtKPartial:
    """Some ground truth found in top K."""

    def test_recall_at_k_partial(self) -> None:
        recommended = ["A", "B", "C"]
        ground_truth = ["A", "C", "D", "E"]
        assert recall_at_k(recommended, ground_truth, k=3) == 0.5


class TestRecallAtKMiss:
    """No ground truth in top K."""

    def test_recall_at_k_miss(self) -> None:
        recommended = ["A", "B", "C"]
        ground_truth = ["D", "E", "F"]
        assert recall_at_k(recommended, ground_truth, k=3) == 0.0


class TestRecallAtKTruncated:
    """Only top K are checked, ignoring remainder."""

    def test_recall_at_k_truncated(self) -> None:
        recommended = ["A", "B", "C", "D", "E"]
        ground_truth = ["A", "E"]
        # Only top 3 checked: ["A", "B", "C"] → only "A" found → 1/2 = 0.5
        assert recall_at_k(recommended, ground_truth, k=3) == 0.5


class TestRecallAtKEmptyGroundTruth:
    """Edge case: empty ground truth returns 0.0."""

    def test_recall_at_k_empty_ground_truth(self) -> None:
        assert recall_at_k(["A", "B"], [], k=3) == 0.0


class TestRecallAtKEmptyRecommendations:
    """Edge case: no recommendations returns 0.0."""

    def test_recall_at_k_empty_recommendations(self) -> None:
        assert recall_at_k([], ["A", "B"], k=3) == 0.0


# ==================== MRR ====================


class TestMRRFirstPosition:
    """First result is relevant → MRR = 1.0."""

    def test_mrr_first_position(self) -> None:
        recommended = ["A", "B", "C"]
        ground_truth = ["A"]
        assert mrr(recommended, ground_truth) == 1.0


class TestMRRThirdPosition:
    """First relevant at rank 3 → MRR = 1/3."""

    def test_mrr_third_position(self) -> None:
        recommended = ["X", "Y", "A", "B"]
        ground_truth = ["A"]
        assert mrr(recommended, ground_truth) == pytest.approx(1 / 3)


class TestMRRNotFound:
    """No relevant result in recommendations → MRR = 0.0."""

    def test_mrr_not_found(self) -> None:
        recommended = ["X", "Y", "Z"]
        ground_truth = ["A"]
        assert mrr(recommended, ground_truth) == 0.0


class TestMRRMultipleRelevant:
    """Multiple relevant results — only first one counts."""

    def test_mrr_multiple_relevant(self) -> None:
        recommended = ["X", "A", "B", "C"]
        ground_truth = ["A", "B", "C"]
        # First relevant at rank 2 → 1/2
        assert mrr(recommended, ground_truth) == 0.5


class TestMRREmpty:
    """Edge cases for empty inputs."""

    def test_mrr_empty_recommendations(self) -> None:
        assert mrr([], ["A"]) == 0.0

    def test_mrr_empty_ground_truth(self) -> None:
        assert mrr(["A", "B"], []) == 0.0


# ==================== MAP (Average Precision) ====================


class TestMAPPerfectRanking:
    """All relevant at top positions → AP = 1.0."""

    def test_map_perfect_ranking(self) -> None:
        recommended = ["A", "B", "C", "X", "Y"]
        ground_truth = ["A", "B", "C"]
        # P@1=1/1, P@2=2/2, P@3=3/3 → AP = (1+1+1)/3 = 1.0
        assert average_precision(recommended, ground_truth) == 1.0


class TestMAPMixedRanking:
    """Relevant at positions 2, 5, 8 → hand-calculated AP."""

    def test_map_mixed_ranking(self) -> None:
        recommended = ["X", "A", "Y", "Z", "B", "W", "V", "C", "U", "T"]
        ground_truth = ["A", "B", "C"]
        # Relevant at ranks 2, 5, 8
        # P@2 = 1/2, P@5 = 2/5, P@8 = 3/8
        # AP = (0.5 + 0.4 + 0.375) / 3 = 1.275 / 3 = 0.425
        assert average_precision(recommended, ground_truth) == pytest.approx(0.425)


class TestMAPEmptyGroundTruth:
    """No ground truth → AP = 0.0."""

    def test_map_empty_ground_truth(self) -> None:
        assert average_precision(["A", "B"], []) == 0.0


class TestMAPEmptyRecommendations:
    """No recommendations → AP = 0.0."""

    def test_map_empty_recommendations(self) -> None:
        assert average_precision([], ["A", "B"]) == 0.0


class TestMAPNoRelevantFound:
    """Recommendations exist but none are relevant → AP = 0.0."""

    def test_map_no_relevant_found(self) -> None:
        recommended = ["X", "Y", "Z"]
        ground_truth = ["A", "B"]
        assert average_precision(recommended, ground_truth) == 0.0


class TestMAPSingleRelevantAtTop:
    """Single relevant at rank 1 out of 2 ground truth."""

    def test_map_single_relevant_at_top(self) -> None:
        recommended = ["A", "X", "Y"]
        ground_truth = ["A", "B"]
        # P@1 = 1/1, only 1 relevant found out of 2 total
        # AP = 1.0 / 2 = 0.5
        assert average_precision(recommended, ground_truth) == 0.5
