from typing import Optional
from job_intelligence.models import JobAnalysisResult
from .models import (
    ResumeAnalysisResult,
    EvidenceObject,
    ResumeMetric,
    CandidateSummary,
    ValidationReport
)
from .parser import ResumeIntelligenceParser
from .validators import SchemaValidator

class ResumeIntelligenceService:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.parser = ResumeIntelligenceParser(model_name=model_name)

    def analyze_resume(
        self,
        resume_text: str,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> ResumeAnalysisResult:
        """
        Primary entry point for Module 2 (Resume Intelligence Service).
        Parses raw candidate resume text and maps evidence directly onto Module 1 capability IDs.
        """
        return self.parser.parse_resume(resume_text=resume_text, job_analysis=job_analysis)

__all__ = [
    "ResumeIntelligenceService",
    "ResumeAnalysisResult",
    "EvidenceObject",
    "ResumeMetric",
    "CandidateSummary",
    "ValidationReport",
    "SchemaValidator"
]
