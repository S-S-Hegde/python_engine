import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "gemini-3.5-flash"
    pipeline_module: str = "Repository Intelligence Service"

class RepositoryEvidenceObject(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence ID e.g. ev_repo_0001")
    capability_id: str = Field(..., description="Must reference Module 1 cap_<domain>_<subdomain> ID")
    source: str = "Repository"
    repository: str = Field(..., description="Repository name or URL")
    location: str = Field(..., description="File path within repo e.g. backend/controllers/authController.js")
    quote: str = Field(..., description="Code snippet or architectural pattern detail")
    engineering_decision: str = "Stateless component or controller architecture"
    complexity: str = "Medium"  # Very Low, Low, Medium, High, Very High
    ownership: str = "Verified"  # Verified, Contributed, Unverified
    confidence: float = Field(default=85.0, ge=0.0, le=100.0)
    verification_status: str = "Repository Verified"  # Repository Verified, Code Evidence, Pattern Match
    generated_from: List[str] = Field(default_factory=list)

class RepositorySummary(BaseModel):
    github_username: str = "Unknown"
    repositories_analyzed: List[str] = Field(default_factory=list)
    total_files_scanned: int = 0
    total_commits_analyzed: int = 0
    primary_language: str = "Unknown"

class ArchitectureSummary(BaseModel):
    pattern: str = "Layered Architecture"  # MVC, Layered, Clean Architecture, Microservices, Monolith, Feature-based, Component-based
    explanation: str = "Separation of controllers, routes, and model schemas."
    detected_folders: List[str] = Field(default_factory=list)
    has_tests: bool = False
    has_docker: bool = False
    has_ci_cd: bool = False

class FrameworkSummary(BaseModel):
    detected_frameworks: List[str] = Field(default_factory=list)
    primary_backend_stack: Optional[str] = None
    primary_frontend_stack: Optional[str] = None
    database_technologies: List[str] = Field(default_factory=list)
    devops_tools: List[str] = Field(default_factory=list)

class OriginalityReport(BaseModel):
    is_fork: bool = False
    is_single_day_dump: bool = False
    commit_count: int = 0
    unique_commit_days: int = 0
    quality_commit_ratio: float = 0.0
    originality_score: float = Field(default=100.0, ge=0.0, le=100.0)
    verdict: str = "Organic Development"  # Organic Development, Single-Day Dump, Forked Repo

class CapabilityMappingSummary(BaseModel):
    capability_id: str
    capability_name: str
    evidence_count: int = 0
    highest_confidence: float = 0.0

class ConfidenceSummary(BaseModel):
    average_confidence: float = 0.0
    high_confidence_evidence_count: int = 0

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_evidence_validated: int = 0
    warnings: List[str] = Field(default_factory=list)

class RepositoryAnalysisResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    repository_summary: RepositorySummary = Field(default_factory=RepositorySummary)
    evidence_objects: List[RepositoryEvidenceObject] = Field(default_factory=list)
    architecture_summary: ArchitectureSummary = Field(default_factory=ArchitectureSummary)
    framework_summary: FrameworkSummary = Field(default_factory=FrameworkSummary)
    originality_report: OriginalityReport = Field(default_factory=OriginalityReport)
    capability_mapping: List[CapabilityMappingSummary] = Field(default_factory=list)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)
