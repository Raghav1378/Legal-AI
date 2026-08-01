"""Deterministic confidence scoring"""
from __future__ import annotations

from typing import Any


class ConfidenceScorer:
    FLOOR = 50
    CEILING = 95

    @classmethod
    def calculate(cls, retrieval_score: float, conflict_penalty: float = 0.0) -> int:
        """Calculate a confidence score in the range [FLOOR, CEILING].

        `retrieval_score` is expected 0.0-1.0; result returned is integer percent.
        """
        # base normalized score
        final_score = max(0.0, min(1.0, retrieval_score * (1.0 - conflict_penalty)))

        # scale to 0-100 and clamp to floor/ceiling
        scaled = int(final_score * 100)
        return int(max(cls.FLOOR, min(cls.CEILING, scaled)))
