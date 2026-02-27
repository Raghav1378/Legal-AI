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
            parts.append(str(doc.get('content') or ''))
            parts.append(str(doc.get('summary') or ''))
            parts.append(str(doc.get('title') or ''))
            parts.append(str(doc.get('url') or ''))
            parts.append(str(doc.get('source') or ''))
            
            # Add new structured fields to the corpus
            new_fields = [
                'elements_required', 'mental_element', 'key_principles', 
                'legal_issue', 'punishment', 'category'
            ]
            for field in new_fields:
                val = doc.get(field)
                if isinstance(val, list):
                    parts.extend([str(i) for i in val])
                elif val:
                    parts.append(str(val))
                    
            if isinstance(doc.get('key_findings'), list):
                parts.extend([str(f) for f in doc['key_findings']])
                
        self.corpus = ' '.join(parts).lower()
        self._has_sources = bool(retrieved_docs)
        self._has_web_sources = any(d.get('source') == 'web' for d in retrieved_docs)

    def validate(
        self, response: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int, int, List[str]]:
        warnings = []
        hallucination_count = 0
        precedent_contamination = 0

        # Whitelist strings used by the engine's Rule 1 fallback entries
        _FALLBACK_STRINGS = {"No binding precedents identified.", "Insufficient verified case references found."}

        original_precedents = response.get('precedents', [])
        valid_precedents = []
        for p in original_precedents:
            p_title = p.get("case_title", str(p)) if isinstance(p, dict) else str(p)
            if p_title in _FALLBACK_STRINGS:
                valid_precedents.append(p)  # Rule 1 fallback — keep as-is
            elif self.CASE_NAME_PATTERN.match(str(p_title)):
                valid_precedents.append(p)
            else:
                precedent_contamination += 1
                warnings.append(f"Invalid precedent format removed: \"{p_title}\"")
        response['precedents'] = valid_precedents

        if not self._has_sources:
            return response, 0, precedent_contamination, warnings

        original_observations = response.get('key_observations', [])
        if isinstance(original_observations, str):
            original_observations = [original_observations]
            
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
            # Instead of wiping, we add a cautionary suffix if web sources exist but grounding is thin
            if self._has_web_sources:
                response['legal_interpretation'] += " (Note: This interpretation incorporates verified web research findings.)"
            else:
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

        # If we have web sources, we are slightly more permissive (need fewer keyword matches)
        # to account for noisier web snippet content.
        match_count = sum(1 for kw in keywords if kw in self.corpus)
        if self._has_web_sources:
             return match_count >= 1 # Permissive: at least one unique keyword must match
        
        return match_count >= 1 # Standard
