import re
import json
import os
from typing import List, Dict, Any, Optional, Tuple

class RetrieverService:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
        
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword-based RAG retrieval with structured metadata enrichment.
        Returns docs enriched with parsed act_name and section.
        """
        keywords = query.lower().split()
        results = []
        
        for doc in self.dataset:
            score = 0
            search_text = (
                (doc.get('title') or "") + " " +
                (doc.get('content') or "") + " " +
                (doc.get('summary') or "") + " " +
                (doc.get('section') or "")
            ).lower()
            
            for kw in keywords:
                if kw in search_text:
                    score += 1
            
            if score > 0:
                results.append({"doc": doc, "score": score})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        docs = [r['doc'] for r in results[:limit]]
        
        # Enrich each doc with structured metadata
        return [self._enrich_doc(doc) for doc in docs]

    def _enrich_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds structured `_act_name` and `_section` fields to a document.
        Uses dataset fields first, then attempts to parse from content/summary.
        Returns None fields if not extractable. Never returns "Unknown".
        """
        enriched = dict(doc)

        # 1. Prefer explicit dataset fields
        if doc.get('act') and doc.get('section'):
            enriched['_act_name'] = doc['act']
            enriched['_section'] = f"Section {doc['section']}"
            return enriched

        # 2. Attempt regex extraction from content + title
        text_to_parse = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('summary', '')}"
        act_name, section = self._parse_provision_metadata(text_to_parse)
        enriched['_act_name'] = act_name  # None if not found
        enriched['_section'] = section    # None if not found
        return enriched

    def _parse_provision_metadata(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract act_name and section from a legal explanation string.
        
        Examples handled:
          "Section 154 of the Code of Criminal Procedure, 1973"
          "u/s 302 IPC"
          "Article 21 of the Constitution of India"
        """
        act_name = None
        section = None

        # Pattern: "Section X of [Act Name]" or "Section X of the [Act Name]"
        pattern_full = re.search(
            r'(Section\s+[\dA-Za-z]+(?:\([^)]*\))?)\s+of\s+(?:the\s+)?([A-Z][A-Za-z\s,]+(?:\d{4})?)',
            text
        )
        if pattern_full:
            section = pattern_full.group(1).strip()
            act_name = pattern_full.group(2).strip().rstrip(',')
            return act_name, section

        # Pattern: "u/s X IPC" or "under Section X CrPC"
        pattern_short = re.search(
            r'(?:u/s|under\s+[Ss]ection)\s+([\dA-Za-z]+)\s+([A-Z]{2,})',
            text
        )
        if pattern_short:
            section = f"Section {pattern_short.group(1).strip()}"
            act_name = pattern_short.group(2).strip()
            return act_name, section

        # Pattern: Bare "Section X" with an act acronym nearby (IPC/CrPC/etc.)
        pattern_bare = re.search(r'(Section\s+[\dA-Za-z]+)', text)
        act_acronym = re.search(r'\b(IPC|CrPC|CPC|IEA|NDPS|POCSO)\b', text)
        if pattern_bare:
            section = pattern_bare.group(1).strip()
            act_name = act_acronym.group(1) if act_acronym else None
            return act_name, section

        return None, None

    def get_all_content(self) -> str:
        """Returns all dataset content as a single string for claim validation."""
        parts = []
        for doc in self.dataset:
            parts.append((doc.get('content') or '') + ' ' + (doc.get('summary') or ''))
        return ' '.join(parts).lower()
