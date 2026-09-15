"""Tests for the human-validation agreement metrics."""

from backend.app.services.validation_metrics import (
    aggregate_agreement_metrics,
    compute_agreement_metrics,
)


def _validation(verdict, emotion="happy", relevance=85, contradictory=False):
    return {
        "timestamp": 1.0,
        "emotion": emotion,
        "relevance_score": relevance,
        "is_contradictory": contradictory,
        "verdict": verdict,
    }


def test_no_validations_returns_none_rate_not_zero():
    """A session nobody reviewed must not read as 0% agreement."""
    metrics = compute_agreement_metrics([])

    assert metrics["agreement_rate"] is None
    assert metrics["confirmed"] == 0
    assert metrics["total_moments"] == 0


def test_unanswered_moments_are_excluded_from_the_rate():
    metrics = compute_agreement_metrics([
        _validation("correct"),
        _validation(None),
        _validation(None),
    ])

    assert metrics["agreement_rate"] == 1.0
    assert metrics["unanswered"] == 2
    assert metrics["total_moments"] == 3


def test_unsure_counted_but_excluded_from_the_rate():
    metrics = compute_agreement_metrics([
        _validation("correct"),
        _validation("incorrect"),
        _validation("unsure"),
    ])

    assert metrics["unsure"] == 1
    assert metrics["agreement_rate"] == 0.5  # 1 confirmed of 2 decided


def test_agreement_rate_counts_confirmed_over_decided():
    metrics = compute_agreement_metrics([
        _validation("correct"),
        _validation("correct"),
        _validation("correct"),
        _validation("incorrect"),
    ])

    assert metrics["agreement_rate"] == 0.75


def test_breakdown_by_relevance_band():
    metrics = compute_agreement_metrics([
        _validation("correct", relevance=95),
        _validation("correct", relevance=82),
        _validation("incorrect", relevance=65),
        _validation("correct", relevance=70),
    ])

    bands = metrics["by_relevance_band"]
    assert bands["high"]["agreement_rate"] == 1.0
    assert bands["medium"]["agreement_rate"] == 0.5


def test_missing_relevance_score_lands_in_unknown_band():
    """Records written before relevance_score was captured must not crash."""
    metrics = compute_agreement_metrics([
        {"timestamp": 1.0, "emotion": "sad", "verdict": "correct"},
    ])

    assert metrics["by_relevance_band"]["unknown"]["confirmed"] == 1


def test_breakdown_by_emotion_and_contradiction():
    metrics = compute_agreement_metrics([
        _validation("correct", emotion="angry", contradictory=True),
        _validation("incorrect", emotion="angry", contradictory=True),
        _validation("correct", emotion="happy", contradictory=False),
    ])

    assert metrics["by_emotion"]["angry"]["agreement_rate"] == 0.5
    assert metrics["by_emotion"]["happy"]["agreement_rate"] == 1.0
    assert metrics["by_contradiction"]["contradictory"]["agreement_rate"] == 0.5
    assert metrics["by_contradiction"]["congruent"]["agreement_rate"] == 1.0


def test_missing_emotion_does_not_crash():
    metrics = compute_agreement_metrics([
        {"timestamp": 1.0, "emotion": None, "verdict": "correct"},
    ])

    assert metrics["by_emotion"]["unknown"]["confirmed"] == 1


class _FakeFeedback:
    def __init__(self, moment_validations):
        self.moment_validations = moment_validations


def test_aggregate_pools_across_sessions():
    metrics = aggregate_agreement_metrics([
        _FakeFeedback([_validation("correct"), _validation("incorrect")]),
        _FakeFeedback([_validation("correct"), _validation("correct")]),
    ])

    assert metrics["confirmed"] == 3
    assert metrics["rejected"] == 1
    assert metrics["agreement_rate"] == 0.75
    assert metrics["sessions_with_validations"] == 2


def test_aggregate_ignores_sessions_with_no_decided_verdicts():
    """A skipped survey shouldn't count as a session that produced evidence."""
    metrics = aggregate_agreement_metrics([
        _FakeFeedback([_validation("correct")]),
        _FakeFeedback([_validation(None)]),
        _FakeFeedback(None),
    ])

    assert metrics["sessions_with_validations"] == 1
    assert metrics["agreement_rate"] == 1.0
