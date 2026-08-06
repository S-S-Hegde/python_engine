import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "technical-assessment-v2"
    pipeline_module: str = "Technical Assessment Intelligence Engine"

class TestCaseResult(BaseModel):
    test_id: str
    test_type: str = "public"  # public, hidden, edge_case
    passed: bool
    input_data: str = ""
    expected_output: str = ""
    actual_output: str = ""
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

class ExecutionResult(BaseModel):
    submission_id: str
    question_id: str
    status: str  # Executed, CompileError, RuntimeError, TimeLimitExceeded, Passed, Failed
    passed_tests_count: int = 0
    total_tests_count: int = 0
    pass_rate: float = Field(0.0, ge=0.0, le=100.0)
    public_tests_passed: bool = True
    hidden_tests_passed: bool = True
    edge_cases_passed: bool = True
    test_cases: List[TestCaseResult] = Field(default_factory=list)
    compilation_error: Optional[str] = None
    runtime_error: Optional[str] = None

class ComplexityDetail(BaseModel):
    time_complexity: str = "O(n)"  # O(1), O(n), O(n log n), O(n^2), O(2^n)
    space_complexity: str = "O(1)"  # O(1), O(n)
    complexity_score: float = Field(..., ge=0.0, le=100.0)
    explanation: str = ""

class CodeQualityDetail(BaseModel):
    readability_score: float = Field(..., ge=0.0, le=100.0)
    modular_design_score: float = Field(..., ge=0.0, le=100.0)
    naming_conventions_score: float = Field(..., ge=0.0, le=100.0)
    error_handling_score: float = Field(..., ge=0.0, le=100.0)
    overall_quality_score: float = Field(..., ge=0.0, le=100.0)

class PlagiarismDetail(BaseModel):
    is_plagiarized: bool = False
    similarity_percentage: float = Field(0.0, ge=0.0, le=100.0)
    matched_source: Optional[str] = None
    anomaly_flags: List[str] = Field(default_factory=list)

class TechnicalAssessmentEvidenceObject(BaseModel):
    evidence_id: str
    capability_id: str
    quote: str  # Specific code snippet or assertion summary
    source: str = "technical_assessment"
    confidence: float = Field(..., ge=0.0, le=100.0)
    ownership: str = "Candidate Submission"
    verified: bool = True
    status: str = "Strongly Verified"  # Strongly Verified, Verified, Partially Verified, Weakly Verified, Unsupported, Contradicted
    details: Dict[str, Any] = Field(default_factory=dict)

class CapabilityAssessmentScore(BaseModel):
    capability_id: str
    capability_name: str
    correctness_score: float = Field(..., ge=0.0, le=100.0)
    quality_score: float = Field(..., ge=0.0, le=100.0)
    complexity_score: float = Field(..., ge=0.0, le=100.0)
    final_capability_score: float = Field(..., ge=0.0, le=100.0)

class AssessmentSummary(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0)
    total_questions: int = 0
    passed_questions: int = 0
    overall_pass_rate: float = Field(0.0, ge=0.0, le=100.0)
    recommendation: str = ""

class ConfidenceSummary(BaseModel):
    average_assessment_confidence: float = Field(..., ge=0.0, le=100.0)
    verification_level: str = "High"

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_submissions_evaluated: int = 0
    warnings: List[str] = Field(default_factory=list)

class AssessmentAnalysisResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    assessment_summary: AssessmentSummary
    capability_scores: List[CapabilityAssessmentScore] = Field(default_factory=list)
    evidence_objects: List[TechnicalAssessmentEvidenceObject] = Field(default_factory=list)
    execution_results: List[ExecutionResult] = Field(default_factory=list)
    complexity_analysis: List[ComplexityDetail] = Field(default_factory=list)
    code_quality: CodeQualityDetail
    plagiarism_report: PlagiarismDetail
    confidence_summary: ConfidenceSummary
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class SubmissionQuestionItem(BaseModel):
    question_id: str
    target_capability_id: Optional[str] = None  # Explicit capability ID or auto-mapped
    target_capability_name: Optional[str] = None
    submitted_code: str
    language: str = "javascript"
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    compilation_error: Optional[str] = None
    runtime_error: Optional[str] = None
    copy_paste_events_count: int = 0
    time_spent_seconds: int = 300

class TechnicalAssessmentSubmission(BaseModel):
    assessment_id: str
    candidate_id: str
    questions: List[SubmissionQuestionItem] = Field(default_factory=list)
    job_analysis: Optional[Dict[str, Any]] = None  # Module 1 output for capability ID validation
