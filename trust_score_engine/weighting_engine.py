from typing import Dict, Any

class WeightingEngine:
    DEFAULT_WEIGHTS = {
        "capability_score": 0.25,
        "competency_score": 0.20,
        "evidence_reliability": 0.15,
        "repo_reliability": 0.15,
        "technical_assessment": 0.15,
        "behavioral_assessment": 0.10
    }

    @classmethod
    def get_normalized_weights(cls, custom_weights: Dict[str, float] = None) -> Dict[str, float]:
        weights = custom_weights or cls.DEFAULT_WEIGHTS
        total = sum(weights.values())
        if total <= 0:
            return cls.DEFAULT_WEIGHTS
        return {k: v / total for k, v in weights.items()}
