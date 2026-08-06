import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "trust-score-engine-v2"
    pipeline_module: str = "Final Trust Score Engine & Verification Report"

class TrustSummary(BaseModel):
    overall_trust_score: float = Field(..., ge=0.0, le=100.0)
    verification_level: str  # Strongly Verified, Verified, Partially Verified, Weakly Verified, Unverified, Contradicted
    hiring_confidence: str  # High, Moderate, Low, Do Not Hire
    candidate_readiness: str  # Production Ready, Nearly Ready, Learning, Not Ready
    final_recommendation: str

class VerificationSummary(BaseModel):
    total_capabilities_evaluated: int = 0
    strongly_verified_count: int = 0
    verified_count: int = 0
    partially_verified_count: int = 0
    weakly_verified_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    verified_capabilities: List[str] = Field(default_factory=list)
    unverified_capabilities: List[str] = Field(default_factory=list)
    contradictions_list: List[str] = Field(default_factory=list)

class RiskSummary(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str  # Low, Medium, High, Critical
    risk_factors: List[str] = Field(default_factory=list)
    mitigation_recommendations: List[str] = Field(default_factory=list)

class ConfidenceSummary(BaseModel):
    engineering_confidence: float = Field(..., ge=0.0, le=100.0)
    evidence_confidence: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=100.0)
    explanation: str = ""

class ReportDetail(BaseModel):
    executive_summary: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    recommended_interview_topics: List[str] = Field(default_factory=list)
    recommended_learning_path: List[str] = Field(default_factory=list)
    final_hiring_recommendation: str

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_modules_integrated: int = 0
    warnings: List[str] = Field(default_factory=list)

class TrustScoreResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    trust_summary: TrustSummary
    verification_summary: VerificationSummary
    risk_summary: RiskSummary
    confidence_summary: ConfidenceSummary
    recommendations: List[str] = Field(default_factory=list)
    report: ReportDetail
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class TrustScoreRequestPayload(BaseModel):
    job_analysis: Optional[Dict[str, Any]] = None
    resume_analysis: Optional[Dict[str, Any]] = None
    repository_analysis: Optional[Dict[str, Any]] = None
    technical_assessment: Optional[Dict[str, Any]] = None
    behavioral_assessment: Optional[Dict[str, Any]] = None
    evidence_fusion_result: Optional[Dict[str, Any]] = None
    capability_scoring_result: Optional[Dict[str, Any]] = None
    competency_intelligence_result: Optional[Dict[str, Any]] = None
    candidate_profile_result: Optional[Dict[str, Any]] = None
