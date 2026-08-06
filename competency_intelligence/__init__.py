import time
from typing import Optional, Dict, Any, List
from capability_scoring.models import CapabilityScoringResult
from .models import (
    CompetencyIntelligenceResult,
    CompetencyProfile,
    CompetencySummary,
    GrowthRecommendation,
    ValidationReport,
    Metadata
)
from .competency_builder import CompetencyBuilder
from .aggregation_engine import AggregationEngine, SchemaValidator
from .maturity_engine import MaturityEngine
from .recommendation_engine import RecommendationEngine

class CompetencyIntelligenceService:
    def __init__(self):
        pass

    def evaluate_competencies(
        self,
        scoring_result: CapabilityScoringResult
    ) -> CompetencyIntelligenceResult:
        """
        Primary entry point for Module 6 (Competency Intelligence Engine).
        Converts Module 5 CapabilityScoringResult into high-level engineering competencies.
        """
        start_time = time.perf_counter()

        scores = scoring_result.capability_scores if scoring_result else []
        valid_capability_ids = [s.capability_id for s in scores]

        grouped = CompetencyBuilder.group_capabilities_by_competency(scores)

        competency_profiles: List[CompetencyProfile] = []
        for comp_id, cap_scores in grouped.items():
            profile = AggregationEngine.aggregate_competency(comp_id, cap_scores)
            competency_profiles.append(profile)

        summary = MaturityEngine.compute_summary(competency_profiles)
        feedback = RecommendationEngine.generate_feedback_and_recommendations(competency_profiles)
        validation_report = SchemaValidator.validate_competency_result(competency_profiles, valid_capability_ids)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return CompetencyIntelligenceResult(
            metadata=Metadata(processing_time_ms=elapsed_ms),
            competencies=competency_profiles,
            competency_summary=summary,
            strengths=feedback.get("strengths", []),
            weaknesses=feedback.get("weaknesses", []),
            missing_competencies=feedback.get("missing_competencies", []),
            growth_recommendations=feedback.get("growth_recommendations", []),
            validation_report=validation_report
        )

__all__ = [
    "CompetencyIntelligenceService",
    "CompetencyIntelligenceResult",
    "CompetencyProfile",
    "CompetencySummary",
    "GrowthRecommendation",
    "ValidationReport",
    "SchemaValidator"
]
