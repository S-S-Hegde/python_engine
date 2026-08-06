from typing import List
from capability_scoring.models import CapabilityScoreDetail

class DependencyEngine:
    @staticmethod
    def calculate_dependency_penalty(scores: List[CapabilityScoreDetail]) -> float:
        """
        Calculates penalty if prerequisite or foundational capabilities are weak or contradicted.
        """
        if not scores:
            return 0.0

        penalty = 0.0
        min_score = min(s.final_capability_score for s in scores)
        has_contradiction = any(s.status == "Contradicted" for s in scores)

        if min_score < 40.0:
            penalty += 15.0
        elif min_score < 55.0:
            penalty += 10.0

        if has_contradiction:
            penalty += 20.0

        return round(min(50.0, max(0.0, penalty)), 2)
