import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "fusion-reasoning-v2"
    pipeline_module: str = "Evidence Fusion Engine"

class ContradictionItem(BaseModel):
    contradiction_id: str
    capability_id: str
    severity: str = "Medium"  # Low, Medium, High, Critical
    type: str  # e.g. ClaimWithoutRepoEvidence, MetricMismatch, LevelContradiction
    description: str
    source_claims: List[str] = Field(default_factory=list)
    confidence_penalty: float = 15.0

class UnifiedCapabilityProfile(BaseModel):
    capability_id: str
    capability_name: str = ""
    status: str = "Unverified"  # Verified, Partially Verified, Unverified, Contradicted
    merged_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    reliability: float = Field(default=0.0, ge=0.0, le=100.0)
    resume_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    repository_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    assessment_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    behavioral_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    professional_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    overall_reasoning: str = ""

class ReliabilitySummary(BaseModel):
    overall_reliability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    repository_weight: float = 0.40
    assessment_weight: float = 0.30
    professional_weight: float = 0.15
    resume_weight: float = 0.15
    highest_reliability_source: str = "Repository"

class ConfidenceSummary(BaseModel):
    average_merged_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    verified_capabilities_count: int = 0
    partially_verified_count: int = 0
    unverified_capabilities_count: int = 0
    contradicted_capabilities_count: int = 0

class ContradictionReport(BaseModel):
    total_contradictions_found: int = 0
    critical_contradictions_count: int = 0
    contradictions: List[ContradictionItem] = Field(default_factory=list)

class MissingEvidenceReport(BaseModel):
    total_missing_gaps: int = 0
    missing_capabilities: List[str] = Field(default_factory=list)

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_profiles_validated: int = 0
    warnings: List[str] = Field(default_factory=list)

class EvidenceFusionResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    capability_profiles: List[UnifiedCapabilityProfile] = Field(default_factory=list)
    reliability_summary: ReliabilitySummary = Field(default_factory=ReliabilitySummary)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    contradiction_report: ContradictionReport = Field(default_factory=ContradictionReport)
    missing_evidence_report: MissingEvidenceReport = Field(default_factory=MissingEvidenceReport)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class FusionRequestPayload(BaseModel):
    job_analysis: Dict[str, Any]
    resume_analysis: Optional[Dict[str, Any]] = None
    repository_analysis: Optional[Dict[str, Any]] = None
    technical_assessment: Optional[Dict[str, Any]] = None
    behavioral_assessment: Optional[Dict[str, Any]] = None
    professional_experience: Optional[List[Dict[str, Any]]] = None
