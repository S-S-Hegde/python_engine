import time
from typing import Dict, Any, List, Optional
from capability_scoring.models import CapabilityScoringResult
from competency_intelligence.models import CompetencyIntelligenceResult
from .models import (
    CandidateProfileResult,
    CandidateSummary,
    Metadata
)
from .seniority_engine import SeniorityEngine
from .specialization_engine import SpecializationEngine
from .role_mapper import RoleMapper
from .strengths_engine import StrengthsEngine
from .risk_engine import RiskEngine
from .recommendation_engine import RecommendationEngine
from .models import ValidationReport

class SchemaValidator:
    @staticmethod
    def validate_candidate_profile(
        result: CandidateProfileResult,
        valid_competency_ids: List[str]
    ) -> ValidationReport:
        warnings: List[str] = []
        sections_validated = 0

        # Validate Candidate Summary
        s = result.candidate_summary
        sections_validated += 1
        if not (0.0 <= s.overall_profile_score <= 100.0):
            warnings.append(f"Overall profile score {s.overall_profile_score} out of bounds [0, 100]. Clamping.")
            s.overall_profile_score = max(0.0, min(100.0, s.overall_profile_score))

        # Validate Seniority
        sections_validated += 1
        if not (0.0 <= result.seniority.seniority_score <= 100.0):
            warnings.append(f"Seniority score {result.seniority.seniority_score} out of bounds [0, 100]. Clamping.")
            result.seniority.seniority_score = max(0.0, min(100.0, result.seniority.seniority_score))

        # Validate Specialization
        sections_validated += 1
        if not (0.0 <= result.specialization.specialization_confidence <= 100.0):
            warnings.append(f"Specialization confidence {result.specialization.specialization_confidence} out of bounds [0, 100]. Clamping.")
            result.specialization.specialization_confidence = max(0.0, min(100.0, result.specialization.specialization_confidence))

        # Validate Best Fit Roles
        sections_validated += 1
        for role in result.best_fit_roles:
            if not (0.0 <= role.fit_score <= 100.0):
                warnings.append(f"Role '{role.role_title}' fit_score {role.fit_score} out of bounds [0, 100]. Clamping.")
                role.fit_score = max(0.0, min(100.0, role.fit_score))

        # Validate Growth Roadmap competency references
        sections_validated += 1
        for step in result.growth_roadmap:
            if valid_competency_ids and step.target_competency not in valid_competency_ids:
                warnings.append(f"Roadmap step references unknown competency_id '{step.target_competency}'.")

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_sections_validated=sections_validated,
            warnings=warnings
        )


class ProfileEngine:
    def __init__(self):
        pass

    def generate_candidate_profile(
        self,
        scoring_result: CapabilityScoringResult,
        competency_result: CompetencyIntelligenceResult
    ) -> CandidateProfileResult:
        start_time = time.perf_counter()

        valid_comp_ids = [c.competency_id for c in competency_result.competencies] if competency_result and competency_result.competencies else []

        seniority = SeniorityEngine.evaluate_seniority(scoring_result, competency_result)
        specialization = SpecializationEngine.evaluate_specialization(competency_result)
        best_fit_roles = RoleMapper.evaluate_role_fits(competency_result, specialization)
        eng_profile = StrengthsEngine.compile_engineering_profile(scoring_result, competency_result)
        risks = RiskEngine.evaluate_risk_areas(scoring_result, competency_result)
        roadmap = RecommendationEngine.build_growth_roadmap(competency_result)
        recommendations = RecommendationEngine.generate_recommendations(competency_result)

        overall_score = round(
            ((scoring_result.readiness_summary.overall_capability_score if scoring_result else 0.0) +
             (competency_result.competency_summary.overall_competency_score if competency_result else 0.0)) / 2.0,
            2
        )

        num_capabilities = len(scoring_result.capability_scores) if scoring_result and scoring_result.capability_scores else 0
        num_competencies = len(competency_result.competencies) if competency_result and competency_result.competencies else 0

        verification_summary = (
            f"Candidate profile synthesized across {num_capabilities} capabilities and {num_competencies} competencies. "
            f"Verified seniority: {seniority.seniority_level} ({seniority.seniority_score}% score)."
        )

        summary = CandidateSummary(
            overall_profile_score=overall_score,
            archetype=specialization.archetype,
            seniority_level=seniority.seniority_level,
            primary_specialization=specialization.primary_domain,
            total_competencies_count=num_competencies,
            total_capabilities_count=num_capabilities,
            verification_summary=verification_summary
        )

        res = CandidateProfileResult(
            metadata=Metadata(processing_time_ms=0.0),
            candidate_summary=summary,
            engineering_profile=eng_profile,
            seniority=seniority,
            specialization=specialization,
            best_fit_roles=best_fit_roles,
            growth_roadmap=roadmap,
            risks=risks,
            recommendations=recommendations
        )

        val_report = SchemaValidator.validate_candidate_profile(res, valid_comp_ids)
        res.validation_report = val_report

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res.metadata.processing_time_ms = elapsed_ms

        return res
