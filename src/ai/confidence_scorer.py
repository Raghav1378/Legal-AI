class ConfidenceScorer:
    PENALTY_CONFLICT = 15
    PENALTY_JSON_REPAIR = 10
    PENALTY_HALLUCINATION = 15
    PENALTY_METADATA_FAILURE = 5
    PENALTY_EMPTY_RETRIEVAL = 20
    PENALTY_PRECEDENT_CONTAMINATION = 10
    PENALTY_EMPTY_CITATIONS = 5

    BONUS_STATUTORY_CITATION = 5
    BONUS_CASE_CITATION = 5
    BONUS_WEB_SOURCES = 5
    BONUS_AGREEING_SOURCES = 5

    FLOOR = 50
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
        has_web_sources: bool = False,
        is_citations_empty: bool = False
    ) -> int:
        score = 70

        if num_agreeing_sources > 1:
            score += ConfidenceScorer.BONUS_AGREEING_SOURCES
        if has_statute_citation:
            score += ConfidenceScorer.BONUS_STATUTORY_CITATION
        if has_case_citation:
            score += ConfidenceScorer.BONUS_CASE_CITATION
        if has_web_sources:
            score += ConfidenceScorer.BONUS_WEB_SOURCES

        if conflicts_detected:
            score -= ConfidenceScorer.PENALTY_CONFLICT

        # If web sources were found, we don't treat retrieval as "empty" even if DB was 0
        is_truly_empty = empty_retrieval and not has_web_sources
        if is_truly_empty:
            score -= ConfidenceScorer.PENALTY_EMPTY_RETRIEVAL

        score -= json_repairs * ConfidenceScorer.PENALTY_JSON_REPAIR
        score -= hallucinations_removed * ConfidenceScorer.PENALTY_HALLUCINATION
        score -= metadata_failures * ConfidenceScorer.PENALTY_METADATA_FAILURE
        score -= precedent_contaminations * ConfidenceScorer.PENALTY_PRECEDENT_CONTAMINATION

        if is_citations_empty:
            score -= ConfidenceScorer.PENALTY_EMPTY_CITATIONS

        return max(ConfidenceScorer.FLOOR, min(ConfidenceScorer.CEILING, score))
