from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from datetime import datetime

class LegalProvision(BaseModel):
    act_name: Optional[str] = None
    description: str

class ApplicableSection(BaseModel):
    section_number: str
    section_title: str
    section_summary: str

class CaseReference(BaseModel):
    case_title: str
    court: Optional[str] = None
    year: Optional[int] = None
    holding_summary: str

    @field_validator('year', mode='before')
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

class Precedent(BaseModel):
    case_title: str
    principle_established: str

class Citation(BaseModel):
    citation_reference: str
    source_url: Optional[str] = None

class LegalResponseSchema(BaseModel):
    response_id: str
    issue_summary: str
    relevant_legal_provisions: List[LegalProvision] = Field(default_factory=list)
    applicable_sections: List[ApplicableSection] = Field(default_factory=list)
    case_references: List[CaseReference] = Field(default_factory=list)
    key_observations: List[str] = Field(default_factory=list)
    legal_interpretation: str
    precedents: List[Precedent] = Field(default_factory=list)
    conclusion: str
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = 0.0
    analysis_limitations: Optional[str] = None
    generated_at: str
    jurisdiction: str = "India"

class QueryRequest(BaseModel):
    chat_id: str
    query: str

class AgentExecutionLogSchema(BaseModel):
    agentName: str
    executionTimeMs: int
    status: str

class QueryResponse(BaseModel):
    structuredResponse: LegalResponseSchema
    agentLogs: List[AgentExecutionLogSchema]
    totalExecutionTimeMs: int
