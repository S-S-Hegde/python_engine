from typing import List
from .models import TrustSummary, RiskSummary

class RecommendationEngine:
    @classmethod
    def generate_recommendations(
        cls,
        trust_summary: TrustSummary,
        risk_summary: RiskSummary
    ) -> List[str]:
        recs: List[str] = [trust_summary.final_recommendation]
        recs.extend(risk_summary.mitigation_recommendations)
        return recs
