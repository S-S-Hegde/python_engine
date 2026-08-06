import re
from typing import List, Dict, Any
from .models import BehavioralQuestionResponse

class DecisionEngine:
    @classmethod
    def evaluate_decision_making(cls, responses: List[BehavioralQuestionResponse]) -> Dict[str, Any]:
        has_ethical_awareness = False
        problem_solving_scores: List[float] = []

        for r in responses:
            text_lower = (r.response_text or "").lower()

            if re.search(r"\b(ethical|privacy|security|compliance|user trust|data protection|integrity|best practice)\b", text_lower):
                has_ethical_awareness = True

            if re.search(r"\b(analyzed|evaluated options|trade-offs|benchmarked|root cause|structured approach|decided)\b", text_lower):
                problem_solving_scores.append(90.0)
            else:
                problem_solving_scores.append(70.0)

        avg_ps = sum(problem_solving_scores) / max(1, len(problem_solving_scores))

        return {
            "problem_solving_score": round(avg_ps, 2),
            "has_ethical_awareness": has_ethical_awareness
        }
