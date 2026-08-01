"""Lightweight verification to prevent confidence regressions."""
from __future__ import annotations

from confidence_scorer import ConfidenceScorer


def test_confidence_bounds():
    # Simulate a variety of retrieval scores and ensure final score falls inside 30..95
    for raw in [0.01, 0.2, 0.5, 0.9, 1.0]:
        score = ConfidenceScorer.calculate(raw, conflict_penalty=0.0)
        assert 30 <= score <= 95, f"Confidence {score} out of expected range for raw={raw}"


if __name__ == "__main__":
    test_confidence_bounds()
    print("verify_orchestration: confidence bounds OK")
