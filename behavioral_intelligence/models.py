import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "behavioral-intelligence-v2"
    pipeline_module: str = "Behavioral & Soft Skills Intelligence Engine"

class StarDetail(BaseModel):
    question_id: str
    has_situation: bool = False
    has_task: bool = False
    has_action: bool = False
    has_result: bool = False
    star_score: float = Field(..., ge=0.0, le=100.0)
    explanation: str = ""

class CommunicationDetail(BaseModel):
    clarity_score: float = Field(..., ge=0.0, le=100.0)
    structure_score: float = Field(..., ge=0.0, le=100.0)
    conciseness_score: float = Field(..., ge=0.0, le=100.0)
    overall_communication_score: float = Field(..., ge=0.0, le=100.0)

class LeadershipDetail(BaseModel):
    collaboration_score: float = Field(..., ge=0.0, le=100.0)
    conflict_resolution_score: float = Field(..., ge=0.0, le=100.0)
    mentorship_score: float = Field(..., ge=0.0, le=100.0)
    overall_leadership_score: float = Field(..., ge=0.0, le=100.0)

class OwnershipDetail(BaseModel):
    accountability_score: float = Field(..., ge=0.0, le=100.0)
    blame_shifting_detected: bool = False
    learning_mindset_score: float = Field(..., ge=0.0, le=100.0)
    overall_ownership_score: float = Field(..., ge=0.0, le=100.0)

class BehavioralEvidenceObject(BaseModel):
    evidence_id: str
    capability_id: str
    quote: str  # Direct quotation or verified transcript excerpt
    source: str = "behavioral_assessment"
    confidence: float = Field(..., ge=0.0, le=100.0)
    ownership: str = "Candidate Response"
    verified: bool = True
    status: str = "Strongly Verified"  # Strongly Verified, Verified, Partially Verified, Weakly Verified, Unsupported, Contradicted
    details: Dict[str, Any] = Field(default_factory=dict)

class CapabilityBehavioralScore(BaseModel):
    capability_id: str
    capability_name: str
    star_score: float = Field(..., ge=0.0, le=100.0)
    ownership_score: float = Field(..., ge=0.0, le=100.0)
    communication_score: float = Field(..., ge=0.0, le=100.0)
    final_capability_score: float = Field(..., ge=0.0, le=100.0)

class BehavioralSummary(BaseModel):
    overall_behavioral_score: float = Field(..., ge=0.0, le=100.0)
    total_responses_analyzed: int = 0
    primary_strengths: List[str] = Field(default_factory=list)
    areas_for_growth: List[str] = Field(default_factory=list)
    recommendation: str = ""

class ConfidenceSummary(BaseModel):
    average_behavioral_confidence: float = Field(..., ge=0.0, le=100.0)
    verification_level: str = "High"

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_responses_validated: int = 0
    warnings: List[str] = Field(default_factory=list)

class BehavioralAnalysisResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    behavioral_summary: BehavioralSummary
    capability_scores: List[CapabilityBehavioralScore] = Field(default_factory=list)
    evidence_objects: List[BehavioralEvidenceObject] = Field(default_factory=list)
    star_analysis: List[StarDetail] = Field(default_factory=list)
    communication_analysis: CommunicationDetail
    leadership_analysis: LeadershipDetail
    ownership_analysis: OwnershipDetail
    confidence_summary: ConfidenceSummary
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class BehavioralQuestionResponse(BaseModel):
    question_id: str
    question_text: str = ""
    target_capability_id: Optional[str] = None  # Explicit capability ID or auto-mapped
    target_capability_name: Optional[str] = None
    response_text: str
    audio_transcript_confidence: float = 100.0

class BehavioralSubmissionPayload(BaseModel):
    assessment_id: str
    candidate_id: str
    responses: List[BehavioralQuestionResponse] = Field(default_factory=list)
    job_analysis: Optional[Dict[str, Any]] = None  # Module 1 output for capability ID validation
