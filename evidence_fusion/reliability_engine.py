from typing import List, Dict, Any
from .models import ReliabilitySummary, UnifiedCapabilityProfile

class ReliabilityEngine:
    DEFAULT_WEIGHTS = {
        "repository": 0.40,
        "assessment": 0.30,
        "professional": 0.15,
        "resume": 0.10,
        "behavioral": 0.05
    }

    @classmethod
    def calculate_capability_reliability(
        cls,
        profile: UnifiedCapabilityProfile,
        custom_weights: Dict[str, float] = None
    ) -> float:
        weights = custom_weights or cls.DEFAULT_WEIGHTS

        score = 0.0
        active_weights_sum = 0.0

        if profile.repository_evidence:
            score += 95.0 * weights.get("repository", 0.40)
            active_weights_sum += weights.get("repository", 0.40)

        if profile.assessment_evidence:
            score += 85.0 * weights.get("assessment", 0.30)
            active_weights_sum += weights.get("assessment", 0.30)

        if profile.professional_evidence:
            score += 80.0 * weights.get("professional", 0.15)
            active_weights_sum += weights.get("professional", 0.15)

        if profile.resume_evidence:
            score += 60.0 * weights.get("resume", 0.10)
            active_weights_sum += weights.get("resume", 0.10)

        if profile.behavioral_evidence:
            score += 70.0 * weights.get("behavioral", 0.05)
            active_weights_sum += weights.get("behavioral", 0.05)

        if active_weights_sum <= 0.0:
            return 0.0

        reliability = round(score / active_weights_sum, 2)
        return min(100.0, max(0.0, reliability))

    @classmethod
    def compute_summary(
        cls,
        profiles: List[UnifiedCapabilityProfile],
        custom_weights: Dict[str, float] = None
    ) -> ReliabilitySummary:
        weights = custom_weights or cls.DEFAULT_WEIGHTS

        if not profiles:
            return ReliabilitySummary()

        total_rel = sum(cls.calculate_capability_reliability(p, weights) for p in profiles)
        avg_rel = round(total_rel / len(profiles), 2)

        return ReliabilitySummary(
            overall_reliability_score=avg_rel,
            repository_weight=weights.get("repository", 0.40),
            assessment_weight=weights.get("assessment", 0.30),
            professional_weight=weights.get("professional", 0.15),
            resume_weight=weights.get("resume", 0.10),
            highest_reliability_source="Repository Verified Code"
        )
