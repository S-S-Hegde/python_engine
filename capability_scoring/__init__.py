from typing import Optional, Dict, Any
from evidence_fusion.models import EvidenceFusionResult
from .models import (
    CapabilityScoringResult,
    CapabilityScoreDetail,
    ReadinessSummary,
    FormulaBreakdown,
    FormulaWeightConfig,
    RecommendationItem,
    ValidationReport
)
from .scoring_engine import ScoringEngine, SchemaValidator

class CapabilityScoringService:
    def __init__(self, weight_config: Optional[FormulaWeightConfig] = None):
        self.engine = ScoringEngine(weight_config=weight_config)

    def evaluate_capabilities(
        self,
        fusion_result: EvidenceFusionResult
    ) -> CapabilityScoringResult:
        """
        Primary entry point for Module 5 (Capability Scoring Engine).
        Converts Module 4 EvidenceFusionResult into deterministic, explainable capability scores.
        """
        return self.engine.evaluate_capabilities(fusion_result)

__all__ = [
    "CapabilityScoringService",
    "CapabilityScoringResult",
    "CapabilityScoreDetail",
    "ReadinessSummary",
    "FormulaBreakdown",
    "FormulaWeightConfig",
    "RecommendationItem",
    "ValidationReport",
    "SchemaValidator"
]
