from typing import Dict, Any, List
from .models import DecisionPolicyType, HireDecision, ComparisonSummary
from .decision_policy import DecisionPolicyEngine

class ComparisonEngine:
    def __init__(self):
        self.policy_engine = DecisionPolicyEngine()

    def evaluate_ai_decision(self, score: float, policy_type: DecisionPolicyType) -> HireDecision:
        policy = self.policy_engine.get_policy(policy_type)
        thresholds = policy["thresholds"]
        
        if score >= thresholds[HireDecision.STRONG_HIRE]:
            return HireDecision.STRONG_HIRE
        elif score >= thresholds[HireDecision.HIRE]:
            return HireDecision.HIRE
        elif score >= thresholds[HireDecision.BORDERLINE]:
            return HireDecision.BORDERLINE
        elif score >= thresholds[HireDecision.HOLD]:
            return HireDecision.HOLD
        else:
            return HireDecision.REJECT

    def evaluate_cohort(self, candidate_score: float, candidate_id: str, cohort_results: List[Dict[str, Any]]) -> ComparisonSummary:
        if not cohort_results:
            return ComparisonSummary(
                cohort_size=1,
                percentile=100.0,
                ranking_position=1,
                relative_strengths=["Sole candidate evaluated in this cohort."],
                relative_weaknesses=["No cohort comparison available."]
            )
        
        all_scores = [c.get("trust_summary", {}).get("overall_trust_score", 0.0) for c in cohort_results]
        all_scores.append(candidate_score)
        all_scores.sort(reverse=True)
        
        cohort_size = len(all_scores)
        ranking_position = all_scores.index(candidate_score) + 1
        
        # Percentile calculation
        if cohort_size == 1:
            percentile = 100.0
        else:
            percentile = ((cohort_size - ranking_position) / (cohort_size - 1)) * 100.0
            
        relative_strengths = []
        if percentile >= 75:
            relative_strengths.append("Candidate is in the top quartile of the evaluated cohort.")
            
        relative_weaknesses = []
        if percentile <= 25:
            relative_weaknesses.append("Candidate falls in the bottom quartile of the evaluated cohort.")

        return ComparisonSummary(
            cohort_size=cohort_size,
            percentile=round(percentile, 1),
            ranking_position=ranking_position,
            relative_strengths=relative_strengths,
            relative_weaknesses=relative_weaknesses
        )
