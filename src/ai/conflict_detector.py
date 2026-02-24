from typing import List, Dict, Any

class ConflictDetector:
    @staticmethod
    def detect(documents: List[Dict[str, Any]]) -> bool:
        conflicts_found = False
        
        critical_keywords = ['overruled', 'notwithstanding', 'distinguished', 'contradicts']
        
        for doc in documents:
            text = (doc.get('content') or doc.get('summary') or "").lower()
            for kw in critical_keywords:
                if kw in text:
                    conflicts_found = True
                    break
            if conflicts_found:
                break
                
        return conflicts_found
