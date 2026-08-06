from typing import List, Dict, Any
from competency_intelligence.models import CompetencyIntelligenceResult
from .models import LearningRoadmapStep

class RecommendationEngine:
    @classmethod
    def build_growth_roadmap(
        cls,
        competency: CompetencyIntelligenceResult
    ) -> List[LearningRoadmapStep]:
        steps: List[LearningRoadmapStep] = []
        step_num = 1

        if competency and competency.competencies:
            sorted_comp = sorted(competency.competencies, key=lambda c: c.competency_score)
            for comp in sorted_comp[:3]:  # Top 3 growth targets
                steps.append(
                    LearningRoadmapStep(
                        step_number=step_num,
                        topic=f"Mastery in {comp.competency_name}",
                        target_competency=comp.competency_id,
                        duration_weeks=4,
                        action_item=f"Build verified repository projects incorporating production-level patterns for {comp.competency_name}.",
                        key_outcomes=[
                            f"Achieve verified status in {comp.competency_name}",
                            "Publish clean, tested repository code"
                        ]
                    )
                )
                step_num += 1

        return steps

    @classmethod
    def generate_recommendations(
        cls,
        competency: CompetencyIntelligenceResult
    ) -> List[str]:
        recs: List[str] = []
        if competency and competency.growth_recommendations:
            for g in competency.growth_recommendations:
                recs.append(g.action_item)
        if not recs:
            recs.append("Candidate demonstrates solid engineering profile. Recommend technical interview phase.")
        return recs
