from typing import List, Dict, Any
from .models import RecommendationItem, CapabilityScoreDetail

class RecommendationEngine:
    @classmethod
    def generate_feedback_and_recommendations(
        cls,
        scores: List[CapabilityScoreDetail]
    ) -> Dict[str, Any]:
        strengths: List[str] = []
        weaknesses: List[str] = []
        missing_ev: List[str] = []
        recommendations: List[RecommendationItem] = []
        r_idx = 1

        for s in scores:
            cap_id = s.capability_id
            name = s.capability_name

            if s.final_capability_score >= 80.0:
                strengths.append(f"Strong performance in '{name}' ({s.final_capability_score}% score).")
            elif s.final_capability_score < 50.0:
                weaknesses.append(f"Low verification score for '{name}' ({s.final_capability_score}% score).")

            if s.status == "Contradicted":
                recommendations.append(
                    RecommendationItem(
                        recommendation_id=f"rec_{cap_id}_{r_idx:02d}",
                        capability_id=cap_id,
                        category="CodeVerification",
                        priority="High",
                        title=f"Resolve Contradictions for {name}",
                        recommendation=f"Candidate resume claims for '{name}' contradict repository code scan. Request live code walkthrough."
                    )
                )
                r_idx += 1
            elif s.status == "Unsupported" or s.final_capability_score < 45.0:
                missing_ev.append(f"Missing repository execution evidence for '{name}'.")
                recommendations.append(
                    RecommendationItem(
                        recommendation_id=f"rec_{cap_id}_{r_idx:02d}",
                        capability_id=cap_id,
                        category="Assessment",
                        priority="Medium",
                        title=f"Administer Technical Assessment for {name}",
                        recommendation=f"Capability '{name}' lacks verified code. Schedule targeted technical assessment."
                    )
                )
                r_idx += 1

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "missing_evidence": missing_ev,
            "recommendations": recommendations
        }
