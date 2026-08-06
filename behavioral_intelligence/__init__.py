from typing import Optional, Dict, Any
from job_intelligence.models import JobAnalysisResult
from .models import (
    BehavioralSubmissionPayload,
    BehavioralQuestionResponse,
    BehavioralAnalysisResult,
    CapabilityBehavioralScore,
    BehavioralEvidenceObject,
    StarDetail,
    CommunicationDetail,
    LeadershipDetail,
    OwnershipDetail,
    BehavioralSummary,
    ConfidenceSummary,
    ValidationReport
)
from .behavioral_engine import BehavioralEngine
from .confidence_engine import SchemaValidator

class BehavioralIntelligenceService:
    def __init__(self):
        self.engine = BehavioralEngine()

    def analyze_behavior(
        self,
        submission: BehavioralSubmissionPayload,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> BehavioralAnalysisResult:
        """
        Primary entry point for Module 9 (Behavioral & Soft Skills Intelligence Engine).
        Analyzes behavioral interview responses and maps findings to Module 1 capability IDs.
        """
        return self.engine.analyze_behavior(submission, job_analysis=job_analysis)

__all__ = [
    "BehavioralIntelligenceService",
    "BehavioralSubmissionPayload",
    "BehavioralQuestionResponse",
    "BehavioralAnalysisResult",
    "CapabilityBehavioralScore",
    "BehavioralEvidenceObject",
    "StarDetail",
    "CommunicationDetail",
    "LeadershipDetail",
    "OwnershipDetail",
    "BehavioralSummary",
    "ConfidenceSummary",
    "ValidationReport",
    "SchemaValidator"
]
