class ConfidenceScorer:
    # Deterministic penalty constants
    PENALTY_CONFLICT = 15
    PENALTY_JSON_REPAIR = 10
    PENALTY_HALLUCINATION = 15
    PENALTY_METADATA_FAILURE = 5
    PENALTY_EMPTY_RETRIEVAL = 20
    PENALTY_TOOL_CORRUPTION = 10
    PENALTY_PRECEDENT_CONTAMINATION = 10

    # Quality Bonuses
    BONUS_CASE_CITATION = 5
    BONUS_STATUTE_CITATION = 5

    # Hard floor and ceiling
    FLOOR = 30
    CEILING = 95

    @staticmethod
    def calculate(
        num_agreeing_sources: int,
        conflicts_detected: bool,
        json_repairs: int = 0,
        hallucinations_removed: int = 0,
        metadata_failures: int = 0,
        empty_retrieval: bool = False,
        tool_corruptions: int = 0,
        precedent_contaminations: int = 0,
        has_case_citation: bool = False,
        has_statute_citation: bool = False,
    ) -> int:
        """
        Deterministic confidence scoring.

        Base: 85 (high-quality legal AI baseline)
        Each penalty is deducted for specific failure conditions.
        Result is clamped to [FLOOR, CEILING].
        """
        score = 85

        # Source quality bonus
        if num_agreeing_sources > 1:
            score += 5

        # Citation bonuses
        if has_case_citation:
            score += ConfidenceScorer.BONUS_CASE_CITATION
        if has_statute_citation:
            score += ConfidenceScorer.BONUS_STATUTE_CITATION

        # Deterministic penalties
        if conflicts_detected:
            score -= ConfidenceScorer.PENALTY_CONFLICT

        score -= json_repairs * ConfidenceScorer.PENALTY_JSON_REPAIR
        score -= hallucinations_removed * ConfidenceScorer.PENALTY_HALLUCINATION
        score -= metadata_failures * ConfidenceScorer.PENALTY_METADATA_FAILURE
        score -= tool_corruptions * ConfidenceScorer.PENALTY_TOOL_CORRUPTION
        score -= precedent_contaminations * ConfidenceScorer.PENALTY_PRECEDENT_CONTAMINATION

        if empty_retrieval:
            score -= ConfidenceScorer.PENALTY_EMPTY_RETRIEVAL

        # Hard clamp: never above 95, never below 30
        return max(ConfidenceScorer.FLOOR, min(ConfidenceScorer.CEILING, score))
