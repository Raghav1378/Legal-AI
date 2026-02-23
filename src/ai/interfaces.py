from typing import List, Optional, TypedDict
from datetime import datetime

class LegalProvision(TypedDict):
    act_name: str
    section: str
    explanation: str

class CaseReference(TypedDict):
    case_name: str
    court: Optional[str]
    year: Optional[int]
    citation_reference: Optional[str]

class Citation(TypedDict):
    title: str
    court: Optional[str]
    year: Optional[int]
    source: str
    url: Optional[str]

class LegalResponse(TypedDict):
    issue_summary: str
    relevant_legal_provisions: List[LegalProvision]
    applicable_sections: List[str]
    case_references: List[CaseReference]
    key_observations: List[str]
    legal_interpretation: str
    precedents: List[str]
    conclusion: str
    citations: List[Citation]
    conflicts_detected: bool
    confidence_score: int

class AgentExecutionLog(TypedDict):
    chat_id: str
    agentName: str
    executionTimeMs: int
    status: str  # 'SUCCESS' | 'FAILED'
    error_message: Optional[str]
    confidence_score: Optional[int]
    conflicts_detected: Optional[bool]
    created_at: datetime

class AIActionResult(TypedDict):
    structuredResponse: LegalResponse
    agentLogs: List[AgentExecutionLog]
    totalExecutionTimeMs: int
