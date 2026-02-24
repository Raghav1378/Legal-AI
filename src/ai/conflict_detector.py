from typing import List, Dict, Any

class ConflictDetector:
    @staticmethod
    def detect(documents: List[Dict[str, Any]]) -> bool:
        conflicts_found = False
        
        critical_keywords = ['overruled', 'notwithstanding', 'distinguished', 'contradicts']
        
        for doc in documents:
            # check the new explicit conflict_keywords field
            explicit_keywords = doc.get('conflict_keywords', [])
            if any(kw in explicit_keywords for kw in critical_keywords):
                conflicts_found = True
                break

            # Fallback to existing logic
            text = (
                str(doc.get('content') or "") + " " + 
                str(doc.get('summary') or "") + " " +
                " ".join(doc.get('key_principles', []))
            ).lower()
            
            for kw in critical_keywords:
                if kw in text:
                    conflicts_found = True
                    break
            if conflicts_found:
                break
                
        return conflicts_found
