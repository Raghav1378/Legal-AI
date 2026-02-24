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
        
        return [self._enrich_doc(doc) for doc in docs]

    def _enrich_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(doc)
        act = doc.get('act')
        section = str(doc.get('section')) if doc.get('section') is not None else None

        if act and section:
            enriched['_act_name'] = act
            enriched['_section'] = self._normalize_section(section, act)
            return enriched

        text_to_parse = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('summary', '')}"
        act_name, raw_section = self._parse_provision_metadata(text_to_parse)
        
        enriched['_act_name'] = act_name
        enriched['_section'] = self._normalize_section(raw_section, act_name) if raw_section else None
        return enriched

    def _normalize_section(self, section: Optional[str], act: Optional[str]) -> Optional[str]:
        if not section:
            return None
        
        if section.lower() == "unknown":
            return None

        clean_section = str(section).strip()
        
        if clean_section.isdigit():
            is_const = act and ("constitution" in act.lower() or "art" in act.lower())
            prefix = "Article" if is_const else "Section"
            return f"{prefix} {clean_section}"
            
        if not (clean_section.startswith("Section") or clean_section.startswith("Article")):
            is_const = act and ("constitution" in act.lower() or "art" in act.lower())
            prefix = "Article" if is_const else "Section"
            return f"{prefix} {clean_section}"

        return clean_section

    def _parse_provision_metadata(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        act_name = None
        section = None

        pattern_full = re.search(
            r'([Ss]ection|[Aa]rticle)\s+([\dA-Za-z]+(?:\([^)]*\))?)\s+of\s+(?:the\s+)?([A-Z][A-Za-z\s,]+(?:\d{4})?)',
            text
        )
        if pattern_full:
            section = pattern_full.group(2).strip()
            act_name = pattern_full.group(3).strip().rstrip(',')
            return act_name, section

        pattern_short = re.search(
            r'(?:u/s|under\s+[Ss]ection|Art\.)\s+([\dA-Za-z]+)\s+([A-Z]{2,})',
            text
        )
        if pattern_short:
            section = pattern_short.group(1).strip()
            act_name = pattern_short.group(2).strip()
            return act_name, section

        pattern_bare = re.search(r'([Ss]ection|[Aa]rticle)\s+([\dA-Za-z]+)', text)
        act_acronym = re.search(r'\b(IPC|CrPC|CPC|IEA|NDPS|POCSO)\b', text)
        if pattern_bare:
            section = pattern_bare.group(2).strip()
            act_name = act_acronym.group(1) if act_acronym else None
            return act_name, section

        return None, None

    def get_all_content(self) -> str:
        parts = []
        for doc in self.dataset:
            parts.append((doc.get('content') or '') + ' ' + (doc.get('summary') or ''))
        return ' '.join(parts).lower()
