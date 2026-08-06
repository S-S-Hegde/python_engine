from typing import Optional, Dict, Any
from job_intelligence.models import JobAnalysisResult
from .models import (
    TechnicalAssessmentSubmission,
    SubmissionQuestionItem,
    AssessmentAnalysisResult,
    CapabilityAssessmentScore,
    TechnicalAssessmentEvidenceObject,
    ExecutionResult,
    ComplexityDetail,
    CodeQualityDetail,
    PlagiarismDetail,
    AssessmentSummary,
    ConfidenceSummary,
    ValidationReport
)
from .assessment_engine import AssessmentEngine
from .confidence_engine import SchemaValidator

class TechnicalAssessmentService:
    def __init__(self):
        self.engine = AssessmentEngine()

    def analyze_assessment(
        self,
        submission: TechnicalAssessmentSubmission,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> AssessmentAnalysisResult:
        """
        Primary entry point for Module 8 (Technical Assessment Intelligence Engine).
        Analyzes technical assessment submissions and maps findings to Module 1 capability IDs.
        """
        return self.engine.analyze_assessment(submission, job_analysis=job_analysis)

__all__ = [
    "TechnicalAssessmentService",
    "TechnicalAssessmentSubmission",
    "SubmissionQuestionItem",
    "AssessmentAnalysisResult",
    "CapabilityAssessmentScore",
    "TechnicalAssessmentEvidenceObject",
    "ExecutionResult",
    "ComplexityDetail",
    "CodeQualityDetail",
    "PlagiarismDetail",
    "AssessmentSummary",
    "ConfidenceSummary",
    "ValidationReport",
    "SchemaValidator"
]
