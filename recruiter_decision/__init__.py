from .models import (
    RecruiterDecisionResult,
    RecruiterDecisionRequestPayload,
    HireDecision,
    DecisionPolicyType
)
from .decision_engine import DecisionEngine

class RecruiterDecisionService:
    def __init__(self):
        self.engine = DecisionEngine()
        
    def generate_decision(self, payload: RecruiterDecisionRequestPayload) -> RecruiterDecisionResult:
        return self.engine.generate_recruiter_decision(
            trust_result=payload.trust_score_result,
            capability_result=payload.capability_scoring_result,
            competency_result=payload.competency_intelligence_result,
            profile_result=payload.candidate_profile_result,
            policy_type=payload.policy_type,
            cohort_results=payload.cohort_results,
            human_override=payload.human_override
        )

__all__ = [
    "RecruiterDecisionService",
    "RecruiterDecisionResult",
    "RecruiterDecisionRequestPayload",
    "HireDecision",
    "DecisionPolicyType"
]
