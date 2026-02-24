from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from datetime import datetime

class LegalProvision(BaseModel):
    act_name: Optional[str] = None
    section: Optional[str] = None
    explanation: str

class CaseReference(BaseModel):
    case_name: str
    court: Optional[str] = None
    year: Optional[int] = None
    citation_reference: Optional[str] = None

    @field_validator('year', mode='before')
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

class Citation(BaseModel):
    title: str
    court: Optional[str] = None
    year: Optional[int] = None
    source: str = "General Law"
    url: Optional[str] = None

    @field_validator('year', mode='before')
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

class LegalResponseSchema(BaseModel):
    issue_summary: str
    relevant_legal_provisions: List[LegalProvision] = Field(default_factory=list)
    applicable_sections: List[str] = Field(default_factory=list)
    case_references: List[CaseReference] = Field(default_factory=list)
    key_observations: List[str] = Field(default_factory=list)
    legal_interpretation: str
    precedents: List[str] = Field(default_factory=list)
    conclusion: str
    citations: List[Citation] = Field(default_factory=list)
    conflicts_detected: bool = False
    confidence_score: int = 30

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
