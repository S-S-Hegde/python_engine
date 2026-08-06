from typing import Optional, Dict, Any, List
from job_intelligence.models import JobAnalysisResult
from resume_intelligence.models import ResumeAnalysisResult
from repository_intelligence.models import RepositoryAnalysisResult
from .models import (
    EvidenceFusionResult,
    UnifiedCapabilityProfile,
    ContradictionItem,
    ReliabilitySummary,
    ConfidenceSummary,
    ContradictionReport,
    MissingEvidenceReport,
    ValidationReport
)
from .fusion_engine import FusionEngine, SchemaValidator

class EvidenceFusionService:
    def __init__(self, custom_reliability_weights: Dict[str, float] = None):
        self.fusion_engine = FusionEngine(custom_reliability_weights=custom_reliability_weights)

    def fuse_evidence(
        self,
        job_analysis: JobAnalysisResult,
        resume_analysis: Optional[ResumeAnalysisResult] = None,
        repository_analysis: Optional[RepositoryAnalysisResult] = None,
        technical_assessment: Optional[Dict[str, Any]] = None,
        behavioral_assessment: Optional[Dict[str, Any]] = None,
        professional_experience: Optional[List[Dict[str, Any]]] = None
    ) -> EvidenceFusionResult:
        """
        Primary entry point for Module 4 (Evidence Fusion Engine).
        Combines evidence from Modules 1, 2, 3, and assessments into unified capability profiles.
        """
        return self.fusion_engine.fuse_evidence(
            job_analysis=job_analysis,
            resume_analysis=resume_analysis,
            repository_analysis=repository_analysis,
            technical_assessment=technical_assessment,
            behavioral_assessment=behavioral_assessment,
            professional_experience=professional_experience
        )

__all__ = [
    "EvidenceFusionService",
    "EvidenceFusionResult",
    "UnifiedCapabilityProfile",
    "ContradictionItem",
    "ReliabilitySummary",
    "ConfidenceSummary",
    "ContradictionReport",
    "MissingEvidenceReport",
    "ValidationReport",
    "SchemaValidator"
]
