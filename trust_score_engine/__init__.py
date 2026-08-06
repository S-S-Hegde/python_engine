from typing import Optional, Dict, Any
from job_intelligence.models import JobAnalysisResult
from resume_intelligence.models import ResumeAnalysisResult
from repository_intelligence.models import RepositoryAnalysisResult
from technical_assessment.models import AssessmentAnalysisResult
from behavioral_intelligence.models import BehavioralAnalysisResult
from evidence_fusion.models import EvidenceFusionResult
from capability_scoring.models import CapabilityScoringResult
from competency_intelligence.models import CompetencyIntelligenceResult
from candidate_profile.models import CandidateProfileResult

from .models import (
    TrustScoreResult,
    TrustSummary,
    VerificationSummary,
    RiskSummary,
    ConfidenceSummary,
    ReportDetail,
    ValidationReport,
    TrustScoreRequestPayload
)
from .verification_engine import FinalTrustEngine
from .validators import SchemaValidator

class TrustScoreService:
    def __init__(self):
        self.engine = FinalTrustEngine()

    def generate_trust_score(
        self,
        job_analysis: Optional[JobAnalysisResult] = None,
        resume_analysis: Optional[ResumeAnalysisResult] = None,
        repository_analysis: Optional[RepositoryAnalysisResult] = None,
        technical_assessment: Optional[AssessmentAnalysisResult] = None,
        behavioral_assessment: Optional[BehavioralAnalysisResult] = None,
        evidence_fusion_result: Optional[EvidenceFusionResult] = None,
        capability_scoring_result: Optional[CapabilityScoringResult] = None,
        competency_intelligence_result: Optional[CompetencyIntelligenceResult] = None,
        candidate_profile_result: Optional[CandidateProfileResult] = None
    ) -> TrustScoreResult:
        """
        Primary entry point for Module 10 (Final Trust Score Engine & Verification Report).
        Consumes outputs from Modules 1–9 and calculates ONE final Trust Score.
        """
        return self.engine.generate_trust_score(
            job_analysis=job_analysis,
            resume_analysis=resume_analysis,
            repository_analysis=repository_analysis,
            technical_assessment=technical_assessment,
            behavioral_assessment=behavioral_assessment,
            evidence_fusion_result=evidence_fusion_result,
            capability_scoring_result=capability_scoring_result,
            competency_intelligence_result=competency_intelligence_result,
            candidate_profile_result=candidate_profile_result
        )

__all__ = [
    "TrustScoreService",
    "TrustScoreResult",
    "TrustSummary",
    "VerificationSummary",
    "RiskSummary",
    "ConfidenceSummary",
    "ReportDetail",
    "ValidationReport",
    "SchemaValidator",
    "TrustScoreRequestPayload"
]
