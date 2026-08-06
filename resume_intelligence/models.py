import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "gemini-3.5-flash"
    pipeline_module: str = "Resume Intelligence Service"

class EvidenceObject(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence ID e.g. ev_resume_0001")
    capability_id: str = Field(..., description="Must reference Module 1 cap_<domain>_<subdomain> ID")
    source: str = "Resume"
    section: str = "Work Experience"  # Work Experience, Projects, Skills, Education, Achievements
    location: str = "Project 1"
    quote: str = Field(..., description="Verbatim quote from resume")
    engineering_decision: str = "Used standard implementation pattern"
    ownership: str = "Individual"  # Individual, Primary Contributor, Team Contributor, Unknown
    complexity: str = "Medium"     # Very Low, Low, Medium, High, Very High
    impact: str = "Demonstrated capability evidence"
    confidence: float = Field(default=80.0, ge=0.0, le=100.0)
    verification_status: str = "Resume Claim" # Resume Claim, Quantified Claim, Architecture Claim, Leadership Claim, Project Claim
    generated_from: List[str] = Field(default_factory=list)

class ResumeMetric(BaseModel):
    metric: str
    context: str
    capability_id: str

class CandidateSummary(BaseModel):
    candidate_name: str = "Candidate"
    detected_level: str = "Intermediate"  # Student, Fresher, Intermediate, Senior, Lead
    total_claims: int = 0
    total_quantified_claims: int = 0

class OwnershipSummary(BaseModel):
    individual_count: int = 0
    primary_contributor_count: int = 0
    team_contributor_count: int = 0
    unknown_count: int = 0

class CapabilityMappingSummary(BaseModel):
    capability_id: str
    capability_name: str
    evidence_count: int = 0
    highest_confidence: float = 0.0

class ConfidenceSummary(BaseModel):
    average_confidence: float = 0.0
    high_confidence_claims_count: int = 0

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_evidence_validated: int = 0
    warnings: List[str] = Field(default_factory=list)

class ResumeAnalysisResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    candidate_summary: CandidateSummary = Field(default_factory=CandidateSummary)
    evidence_objects: List[EvidenceObject] = Field(default_factory=list)
    resume_metrics: List[ResumeMetric] = Field(default_factory=list)
    ownership_summary: OwnershipSummary = Field(default_factory=OwnershipSummary)
    capability_mapping: List[CapabilityMappingSummary] = Field(default_factory=list)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)
