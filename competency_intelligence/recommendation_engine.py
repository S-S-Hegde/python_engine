from typing import List, Dict, Any
from .models import GrowthRecommendation, CompetencyProfile

class RecommendationEngine:
    @classmethod
    def generate_feedback_and_recommendations(
        cls,
        profiles: List[CompetencyProfile]
    ) -> Dict[str, Any]:
        strengths: List[str] = []
        weaknesses: List[str] = []
        missing_comp: List[str] = []
        recommendations: List[GrowthRecommendation] = []
        r_idx = 1

        for p in profiles:
            cid = p.competency_id
            name = p.competency_name

            if p.competency_score >= 80.0:
                strengths.append(f"High proficiency in {name} ({p.competency_score}% score - {p.maturity_level}).")
            elif p.competency_score < 50.0:
                weaknesses.append(f"Low competency score in {name} ({p.competency_score}% score - {p.maturity_level}).")

            if p.maturity_level in ["Developing", "Beginner"]:
                missing_comp.append(f"{name} lacks production-level verification.")
                recommendations.append(
                    GrowthRecommendation(
                        recommendation_id=f"rec_comp_{cid}_{r_idx:02d}",
                        competency_id=cid,
                        target_maturity="Advanced" if p.maturity_level == "Intermediate" else "Intermediate",
                        current_score=p.competency_score,
                        action_item=f"Build verified repository projects incorporating production-level patterns for {name}.",
                        suggested_resources=[f"Production Guidelines for {name}", f"{name} Verification Blueprint"]
                    )
                )
                r_idx += 1

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "missing_competencies": missing_comp,
            "growth_recommendations": recommendations
        }
