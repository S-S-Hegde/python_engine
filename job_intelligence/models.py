import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "gemini-3.5-flash"
    pipeline_module: str = "Job Intelligence Service"

class JobComplexity(BaseModel):
    overall: str = "Intermediate"
    technical: int = Field(default=3, ge=1, le=5)
    architecture: int = Field(default=3, ge=1, le=5)
    communication: int = Field(default=3, ge=1, le=5)
    domain: int = Field(default=3, ge=1, le=5)

class CandidateLevelExpected(BaseModel):
    level: str = "Intermediate"
    experience_range: str = "1-3 years"
    minimum_proficiency: int = Field(default=3, ge=1, le=5)

class CapabilityNode(BaseModel):
    id: str = Field(..., description="Stable identifier in cap_<domain>_<subdomain> format")
    name: str
    confidence: float = Field(default=90.0, ge=0.0, le=100.0)
    classification: str = "Verified_Requirement"
    importance: str = "Critical"  # Critical, Important, Optional
    weight: float = Field(..., ge=0.0, le=100.0)
    expected_proficiency: int = Field(default=3, ge=1, le=5)
    generated_from: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    expected_evidence: List[str] = Field(default_factory=list)
    sub_capabilities: List[str] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)

class CompetencyNode(BaseModel):
    name: str
    weight: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(default=90.0, ge=0.0, le=100.0)
    depends_on: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)

class JobAnalysisResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    role: str
    business_objectives: List[str] = Field(default_factory=list)
    engineering_objectives: List[str] = Field(default_factory=list)
    job_complexity: JobComplexity = Field(default_factory=JobComplexity)
    candidate_level_expected: CandidateLevelExpected = Field(default_factory=CandidateLevelExpected)
    capability_graph: List[CapabilityNode] = Field(default_factory=list)
    competency_graph: List[CompetencyNode] = Field(default_factory=list)
    risk_areas: List[str] = Field(default_factory=list)
    positive_hiring_signals: List[str] = Field(default_factory=list)
    negative_hiring_signals: List[str] = Field(default_factory=list)
