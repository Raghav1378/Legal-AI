from typing import TypedDict, List, Any, Optional


class Citation(TypedDict, total=False):
    title: str
    court: Optional[str]
    year: Optional[int]
    source: Optional[str]
    url: Optional[str]


class LegalResponse(TypedDict, total=False):
    issue_summary: str
    legal_interpretation: str
    conclusion: str
    citations: List[Citation]
    case_references: List[str]
    conflicts_detected: bool
    confidence_score: int
