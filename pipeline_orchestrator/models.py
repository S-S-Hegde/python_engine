from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class ExecutionMode(str, Enum):
    SYNC = "SYNC"
    ASYNC = "ASYNC"

class PipelineStatus(str, Enum):
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    PARTIAL = "Partial"

class ExecutionStatus(str, Enum):
    COMPLETED = "Completed"
    SKIPPED = "Skipped"
    FAILED = "Failed"

class PipelineConfig(BaseModel):
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SYNC)
    run_resume_intelligence: bool = True
    run_repository_intelligence: bool = True
    run_technical_assessment: bool = True
    run_behavioral_assessment: bool = True
    run_candidate_profile: bool = True
    run_trust_score: bool = True
    run_recruiter_decision: bool = True

class PipelineRequest(BaseModel):
    job_requirements: List[str]
    resume_text: Optional[str] = None
    repositories: Optional[List[Dict[str, Any]]] = None  # [{ "name": "app", "fork": False, "files": [...] }]
    technical_assessment: Optional[Dict[str, Any]] = None # Match TechnicalAssessmentSubmission
    behavioral_assessment: Optional[Dict[str, Any]] = None # Match BehavioralSubmissionPayload
    professional_experience_years: Optional[int] = None
    config: PipelineConfig = Field(default_factory=PipelineConfig)

class ExecutionRecord(BaseModel):
    module: str
    status: ExecutionStatus
    execution_time_ms: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[str] = None

class PipelineResponse(BaseModel):
    request_id: str
    execution_mode: ExecutionMode
    pipeline_status: PipelineStatus
    metadata: Dict[str, Any]
    
    # Unified output payload
    job_analysis: Optional[Dict[str, Any]] = None
    resume_analysis: Optional[Dict[str, Any]] = None
    repository_analysis: Optional[Dict[str, Any]] = None
    assessment_analysis: Optional[Dict[str, Any]] = None
    behavioral_analysis: Optional[Dict[str, Any]] = None
    evidence_fusion: Optional[Dict[str, Any]] = None
    capability_scores: Optional[Dict[str, Any]] = None
    competency_scores: Optional[Dict[str, Any]] = None
    candidate_profile: Optional[Dict[str, Any]] = None
    trust_score: Optional[Dict[str, Any]] = None
    recruiter_decision: Optional[Dict[str, Any]] = None
    
    pipeline_execution: List[ExecutionRecord]
    audit_trail: Dict[str, Any]
    schema_versions: Dict[str, str]
