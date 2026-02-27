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
        score = 1.0

        if not has_statute_citation:
            score -= 0.4
            
        if not has_case_citation:
            score -= 0.2
            
        if is_citations_empty:
            score -= 0.2
            
        # Interpretation is generic if it triggered json repairs or empty retrieval
        interpretation_is_generic = (json_repairs > 0) or empty_retrieval
        if interpretation_is_generic:
            score -= 0.2

        final_score = max(score, 0.1)
        
        # Return as an integer 0-100 because mcp_orchestrator does: `float(final_confidence) / 100.0`
        return int(final_score * 100)
