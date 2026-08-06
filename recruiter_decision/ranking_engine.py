from typing import Dict, Any
from .models import RankingMetrics

class RankingEngine:
    def calculate_ranking(self, trust_result: Dict[str, Any], capability_result: Dict[str, Any], competency_result: Dict[str, Any]) -> RankingMetrics:
        # Trust score provides the base ranking score
        trust_score = trust_result.get("trust_summary", {}).get("overall_trust_score", 0.0)
        
        # Capability and competency aggregations
        capabilities = capability_result.get("capability_scores", [])
        cap_score_total = sum(c.get("score", 0.0) for c in capabilities)
        cap_avg = cap_score_total / len(capabilities) if capabilities else 0.0
        
        competencies = competency_result.get("competencies", [])
        comp_score_total = sum(c.get("maturity_score", 0.0) for c in competencies)
        comp_avg = comp_score_total / len(competencies) if competencies else 0.0
        
        # Calculate derived ranking metrics
        strength_ranking = (trust_score * 0.4) + (cap_avg * 0.4) + (comp_avg * 0.2)
        
        # Inverse mapping for weakness ranking
        weakness_ranking = 100.0 - strength_ranking
        
        # Calculate final combined ranking score
        candidate_ranking_score = min(100.0, max(0.0, strength_ranking))

        return RankingMetrics(
            candidate_ranking_score=round(candidate_ranking_score, 2),
            engineering_strength_ranking=round(strength_ranking, 2),
            engineering_weakness_ranking=round(weakness_ranking, 2),
            capability_ranking=round(cap_avg, 2),
            competency_ranking=round(comp_avg, 2)
        )
