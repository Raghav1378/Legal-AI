from typing import List, Optional, TypedDict
from datetime import datetime

class LegalProvision(TypedDict):
    act_name: str
    description: str

class ApplicableSection(TypedDict):
    section_number: str
    section_title: str
    section_summary: str

class CaseReference(TypedDict):
    case_title: str
    court: Optional[str]
    year: Optional[int]
    holding_summary: str

class Precedent(TypedDict):
    case_title: str
    principle_established: str

class Citation(TypedDict):
    citation_reference: str
    source_url: Optional[str]

class LegalResponse(TypedDict):
    response_id: str
    issue_summary: str
    relevant_legal_provisions: List[LegalProvision]
    applicable_sections: List[ApplicableSection]
    case_references: List[CaseReference]
    key_observations: List[str]
    legal_interpretation: str
    precedents: List[Precedent]
    conclusion: str
    citations: List[Citation]
    confidence_score: float
    generated_at: str
    jurisdiction: str

class AgentExecutionLog(TypedDict):
    agentName: str
    executionTimeMs: int
    status: str  # 'SUCCESS' | 'FAILED'

class AIActionResult(TypedDict):
    structuredResponse: LegalResponse
    agentLogs: List[AgentExecutionLog]
    totalExecutionTimeMs: int
