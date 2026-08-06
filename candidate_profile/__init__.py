from typing import Optional, Dict, Any
from capability_scoring.models import CapabilityScoringResult
from competency_intelligence.models import CompetencyIntelligenceResult
from .models import (
    CandidateProfileResult,
    CandidateSummary,
    SeniorityDetail,
    SpecializationDetail,
    RoleFitItem,
    LearningRoadmapStep,
    RiskArea,
    ValidationReport
)
from .profile_engine import ProfileEngine, SchemaValidator

class CandidateProfileService:
    def __init__(self):
        self.engine = ProfileEngine()

    def generate_candidate_profile(
        self,
        scoring_result: CapabilityScoringResult,
        competency_result: CompetencyIntelligenceResult
    ) -> CandidateProfileResult:
        """
        Primary entry point for Module 7 (Candidate Profile Engine).
        Synthesizes capability and competency intelligence into one complete engineering profile.
        """
        return self.engine.generate_candidate_profile(scoring_result, competency_result)

__all__ = [
    "CandidateProfileService",
    "CandidateProfileResult",
    "CandidateSummary",
    "SeniorityDetail",
    "SpecializationDetail",
    "RoleFitItem",
    "LearningRoadmapStep",
    "RiskArea",
    "ValidationReport",
    "SchemaValidator"
]
