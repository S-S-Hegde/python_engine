from typing import List
from capability_scoring.models import CapabilityScoringResult
from competency_intelligence.models import CompetencyIntelligenceResult
from .models import EngineeringProfileDetail

class StrengthsEngine:
    @classmethod
    def compile_engineering_profile(
        cls,
        scoring: CapabilityScoringResult,
        competency: CompetencyIntelligenceResult
    ) -> EngineeringProfileDetail:
        strengths: List[str] = []
        weaknesses: List[str] = []
        growth_areas: List[str] = []
        knowledge_gaps: List[str] = []

        if scoring:
            strengths.extend(scoring.strengths)
            weaknesses.extend(scoring.weaknesses)

        if competency:
            for c in competency.competencies:
                if c.competency_score >= 75.0:
                    strengths.append(f"Production-grade mastery in {c.competency_name}.")
                elif c.competency_score < 50.0:
                    growth_areas.append(f"Requires advancement in {c.competency_name}.")
                    knowledge_gaps.append(f"Unverified evidence gap in {c.competency_name}.")

        overall_maturity = (
            f"Candidate demonstrates {scoring.readiness_summary.readiness_level} execution maturity "
            f"with {competency.competency_summary.overall_competency_score}% overall competency score."
        )

        return EngineeringProfileDetail(
            overall_maturity=overall_maturity,
            strengths=list(set(strengths)),
            weaknesses=list(set(weaknesses)),
            growth_areas=list(set(growth_areas)),
            knowledge_gaps=list(set(knowledge_gaps))
        )
