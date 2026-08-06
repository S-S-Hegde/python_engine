import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "candidate-profile-v2"
    pipeline_module: str = "Candidate Profile Engine"

class SeniorityDetail(BaseModel):
    seniority_level: str  # Student, Fresher, Junior, Mid-Level, Senior, Lead, Architect
    seniority_score: float = Field(..., ge=0.0, le=100.0)
    reasoning: str = ""

class SpecializationDetail(BaseModel):
    primary_domain: str
    secondary_domains: List[str] = Field(default_factory=list)
    archetype: str  # e.g. Full Stack Architect, Backend Specialist, AI/ML Practitioner
    specialization_confidence: float = Field(..., ge=0.0, le=100.0)

class RoleFitItem(BaseModel):
    role_title: str
    fit_category: str  # Best-Fit, Alternative, Unsuitable
    fit_score: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(..., ge=0.0, le=100.0)
    reasoning: str = ""

class LearningRoadmapStep(BaseModel):
    step_number: int
    topic: str
    target_competency: str
    duration_weeks: int = 4
    action_item: str
    key_outcomes: List[str] = Field(default_factory=list)

class RiskArea(BaseModel):
    risk_id: str
    category: str  # Verification, TechnicalGap, SingleDomainDependency, SeniorityMismatch
    severity: str  # Critical, High, Medium, Low
    title: str
    description: str
    mitigation_strategy: str

class EngineeringProfileDetail(BaseModel):
    overall_maturity: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    growth_areas: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)

class CandidateSummary(BaseModel):
    overall_profile_score: float = Field(..., ge=0.0, le=100.0)
    archetype: str
    seniority_level: str
    primary_specialization: str
    total_competencies_count: int = 0
    total_capabilities_count: int = 0
    verification_summary: str = ""

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_sections_validated: int = 0
    warnings: List[str] = Field(default_factory=list)

class CandidateProfileResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    candidate_summary: CandidateSummary
    engineering_profile: EngineeringProfileDetail
    seniority: SeniorityDetail
    specialization: SpecializationDetail
    best_fit_roles: List[RoleFitItem] = Field(default_factory=list)
    growth_roadmap: List[LearningRoadmapStep] = Field(default_factory=list)
    risks: List[RiskArea] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class CandidateProfileRequestPayload(BaseModel):
    capability_scoring_result: Dict[str, Any]
    competency_intelligence_result: Dict[str, Any]
