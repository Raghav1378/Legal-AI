import re
from typing import List, Dict, Any, Tuple

class HallucinationGuard:
    MIN_KEYWORD_LEN = 5
    STOP_WORDS = {
        'the', 'and', 'that', 'this', 'with', 'from', 'under', 'when',
        'have', 'shall', 'must', 'will', 'been', 'such', 'also', 'any',
        'may', 'not', 'are', 'for', 'can', 'its', 'all', 'but', 'into',
    }
    CASE_NAME_PATTERN = re.compile(r".+ v\. .+", re.IGNORECASE)

    def __init__(self, retrieved_docs: List[Dict[str, Any]]):
        parts = []
        for doc in retrieved_docs:
            parts.append((doc.get('content') or ''))
            parts.append((doc.get('summary') or ''))
            parts.append((doc.get('title') or ''))
            if isinstance(doc.get('key_findings'), list):
                parts.extend(doc['key_findings'])
        self.corpus = ' '.join(parts).lower()
        self._has_sources = bool(retrieved_docs)

    def validate(
        self, response: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int, int, List[str]]:
        warnings = []
        hallucination_count = 0
        precedent_contamination = 0

        original_precedents = response.get('precedents', [])
        valid_precedents = []
        for p in original_precedents:
            if self.CASE_NAME_PATTERN.match(str(p)):
                valid_precedents.append(p)
            else:
                precedent_contamination += 1
                warnings.append(f"Invalid precedent format removed: \"{p}\"")
        response['precedents'] = valid_precedents

        if not self._has_sources:
            return response, 0, precedent_contamination, warnings

        original_observations = response.get('key_observations', [])
        validated_observations = []
        for obs in original_observations:
            if self._is_supported(obs):
                validated_observations.append(obs)
            else:
                hallucination_count += 1
        response['key_observations'] = validated_observations

        interpretation = response.get('legal_interpretation', '')
        if interpretation and interpretation != "N/A" and not self._is_supported(interpretation):
            hallucination_count += 1
            response['legal_interpretation'] = "The interpretation cannot be fully validated against retrieved statutory text."

        return response, hallucination_count, precedent_contamination, warnings

    def _is_supported(self, claim: str) -> bool:
        if not self.corpus:
            return True

        words = re.findall(r'[a-zA-Z]+', claim.lower())
        keywords = [
            w for w in words
            if len(w) >= self.MIN_KEYWORD_LEN and w not in self.STOP_WORDS
        ]

        if not keywords:
            return True

        return any(kw in self.corpus for kw in keywords)
