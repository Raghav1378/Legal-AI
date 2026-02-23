import re
from typing import List, Dict, Any, Tuple

class HallucinationGuard:
    """
    Validates AI-generated legal claims against retrieved source documents.

    A claim is considered 'unsupported' if none of the retrieved documents
    contain any significant keyword from that claim.
    """

    # Minimum keyword length to be considered significant
    MIN_KEYWORD_LEN = 5

    # Short words to ignore in matching
    STOP_WORDS = {
        'the', 'and', 'that', 'this', 'with', 'from', 'under', 'when',
        'have', 'shall', 'must', 'will', 'been', 'such', 'also', 'any',
        'may', 'not', 'are', 'for', 'can', 'its', 'all', 'but', 'into',
    }

    # Valid case name pattern: "Party v. Party"
    CASE_NAME_PATTERN = re.compile(r".+ v\. .+", re.IGNORECASE)

    def __init__(self, retrieved_docs: List[Dict[str, Any]]):
        # Build a single lowercased corpus from all retrieved doc text
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
        """
        Validate key_observations, legal_interpretation, and precedents.

        Returns:
            (cleaned_response, hallucination_count, precedent_contamination_count, warning_messages)
        """
        warnings = []
        hallucination_count = 0
        precedent_contamination = 0

        # If no sources were retrieved at all, we can't validate grounding —
        # however, we CAN still validate precedent pattern purity.
        if not self._has_sources:
            warnings.append("No source documents retrieved — grounding check skipped.")
        
        # --- Validate precedents (Purity check) ---
        original_precedents = response.get('precedents', [])
        valid_precedents = []
        for p in original_precedents:
            if self.CASE_NAME_PATTERN.match(str(p)):
                valid_precedents.append(p)
            else:
                precedent_contamination += 1
                warnings.append(f"Invalid precedent format removed: \"{p}\" (Expected 'Party v. Party')")
        response['precedents'] = valid_precedents

        if not self._has_sources:
            return response, 0, precedent_contamination, warnings

        # --- Validate key_observations (Grounding check) ---
        original_observations = response.get('key_observations', [])
        validated_observations = []
        for obs in original_observations:
            if self._is_supported(obs):
                validated_observations.append(obs)
            else:
                hallucination_count += 1
                warnings.append(f"Unsupported observation removed: \"{obs[:80]}...\"" if len(obs) > 80 else f"Unsupported observation removed: \"{obs}\"")
        response['key_observations'] = validated_observations

        # --- Validate legal_interpretation ---
        interpretation = response.get('legal_interpretation', '')
        if interpretation and interpretation != "N/A" and not self._is_supported(interpretation):
            hallucination_count += 1
            warnings.append(f"Unsupported legal_interpretation neutralized.")
            response['legal_interpretation'] = "Legal interpretation could not be independently verified from retrieved sources."

        return response, hallucination_count, precedent_contamination, warnings

    def _is_supported(self, claim: str) -> bool:
        """
        Returns True if at least one significant keyword from the claim
        is found in the source corpus.
        """
        if not self.corpus:
            return True  # No corpus to validate against — give benefit of doubt

        # Extract significant words from the claim
        words = re.findall(r'[a-zA-Z]+', claim.lower())
        keywords = [
            w for w in words
            if len(w) >= self.MIN_KEYWORD_LEN and w not in self.STOP_WORDS
        ]

        if not keywords:
            return True  # Too short to validate meaningfully

        # Claim is supported if ANY keyword matches in the corpus
        return any(kw in self.corpus for kw in keywords)
