import time
from typing import Dict, Any, List, Optional
from evidence_fusion.models import EvidenceFusionResult
from .models import (
    CapabilityScoringResult,
    CapabilityScoreDetail,
    FormulaBreakdown,
    FormulaWeightConfig,
    Metadata
)
from .capability_evaluator import CapabilityEvaluator
from .penalty_engine import PenaltyEngine
from .readiness_engine import ReadinessEngine
from .recommendation_engine import RecommendationEngine
from .models import ValidationReport

class SchemaValidator:
    @staticmethod
    def validate_scoring_result(
        scores: List[CapabilityScoreDetail],
        valid_capability_ids: List[str]
    ) -> ValidationReport:
        warnings: List[str] = []
        seen_ids: set = set()

        for score in scores:
            cap_id = score.capability_id
            if not cap_id or not isinstance(cap_id, str):
                warnings.append("Capability score detail contains empty or invalid capability_id.")

            if cap_id in seen_ids:
                warnings.append(f"Duplicate capability score ID detected: '{cap_id}'.")
            seen_ids.add(cap_id)

            if valid_capability_ids and cap_id not in valid_capability_ids:
                warnings.append(f"Capability ID '{cap_id}' not found in input Module 4 profiles.")

            if not (0.0 <= score.final_capability_score <= 100.0):
                warnings.append(f"Final capability score {score.final_capability_score} out of bounds [0, 100]. Clamping.")
                score.final_capability_score = max(0.0, min(100.0, score.final_capability_score))

            b = score.formula_breakdown
            raw_sum = b.raw_weighted_sum if hasattr(b, "raw_weighted_sum") else b.get("raw_weighted_sum", 0.0)
            if not (0.0 <= raw_sum <= 100.0):
                warnings.append(f"Raw weighted sum {raw_sum} out of bounds [0, 100]. Clamping.")
                if hasattr(b, "raw_weighted_sum"):
                    b.raw_weighted_sum = max(0.0, min(100.0, b.raw_weighted_sum))

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_capabilities_evaluated=len(scores),
            warnings=warnings
        )


class ScoringEngine:
    def __init__(self, weight_config: Optional[FormulaWeightConfig] = None):
        self.weights = weight_config or FormulaWeightConfig()

    def evaluate_capabilities(
        self,
        fusion_result: EvidenceFusionResult
    ) -> CapabilityScoringResult:
        start_time = time.perf_counter()

        scores: List[CapabilityScoreDetail] = []
        valid_cap_ids: List[str] = []

        if fusion_result and fusion_result.capability_profiles:
            for p in fusion_result.capability_profiles:
                valid_cap_ids.append(p.capability_id)
                p_dict = p.model_dump()

                cov = CapabilityEvaluator.calculate_coverage_score(p_dict)
                dep = CapabilityEvaluator.calculate_engineering_depth_score(p_dict)
                cx = CapabilityEvaluator.calculate_complexity_score(p_dict)
                rel = p.reliability
                con = CapabilityEvaluator.calculate_consistency_score(p_dict)
                cnf = p.merged_confidence

                raw_sum = (
                    (cov * self.weights.w_coverage) +
                    (dep * self.weights.w_depth) +
                    (cx * self.weights.w_complexity) +
                    (rel * self.weights.w_reliability) +
                    (con * self.weights.w_consistency) +
                    (cnf * self.weights.w_confidence)
                )

                missing_pen = PenaltyEngine.calculate_missing_evidence_penalty(p.missing_evidence)
                contradiction_dicts = [c.model_dump() for c in p.contradictions]
                cnt_pen = PenaltyEngine.calculate_contradiction_penalty(contradiction_dicts)

                final_score = round(max(0.0, min(100.0, raw_sum - missing_pen - cnt_pen)), 2)

                # Determine Status
                has_cnt = len(p.contradictions) > 0
                if has_cnt and any(c.severity in ["High", "Critical"] for c in p.contradictions):
                    status = "Contradicted"
                elif final_score >= 85.0:
                    status = "Strongly Verified"
                elif final_score >= 70.0:
                    status = "Verified"
                elif final_score >= 55.0:
                    status = "Partially Verified"
                elif final_score >= 40.0:
                    status = "Weakly Verified"
                elif has_cnt:
                    status = "Contradicted"
                else:
                    status = "Unsupported"

                expr = (
                    f"({cov:.1f}*{self.weights.w_coverage:.2f}) + "
                    f"({dep:.1f}*{self.weights.w_depth:.2f}) + "
                    f"({cx:.1f}*{self.weights.w_complexity:.2f}) + "
                    f"({rel:.1f}*{self.weights.w_reliability:.2f}) + "
                    f"({con:.1f}*{self.weights.w_consistency:.2f}) + "
                    f"({cnf:.1f}*{self.weights.w_confidence:.2f}) - "
                    f"{missing_pen:.1f}(Missing) - {cnt_pen:.1f}(Contradiction) = {final_score:.1f}"
                )

                breakdown = FormulaBreakdown(
                    coverage_score=cov,
                    depth_score=dep,
                    complexity_score=cx,
                    reliability_score=rel,
                    consistency_score=con,
                    confidence_score=cnf,
                    raw_weighted_sum=round(raw_sum, 2),
                    missing_evidence_penalty=missing_pen,
                    contradiction_penalty=cnt_pen,
                    final_capability_score=final_score,
                    weights_used=self.weights,
                    formula_expression=expr
                )

                scores.append(
                    CapabilityScoreDetail(
                        capability_id=p.capability_id,
                        capability_name=p.capability_name,
                        status=status,
                        final_capability_score=final_score,
                        formula_breakdown=breakdown,
                        reasoning=p.overall_reasoning or f"Evaluated capability with score {final_score}% ({status}).",
                        evidence_counts={
                            "resume": len(p.resume_evidence),
                            "repository": len(p.repository_evidence),
                            "assessment": len(p.assessment_evidence)
                        },
                        contradiction_count=len(p.contradictions)
                    )
                )

        readiness_summary = ReadinessEngine.evaluate_readiness(scores)
        feedback = RecommendationEngine.generate_feedback_and_recommendations(scores)
        validation_report = SchemaValidator.validate_scoring_result(scores, valid_cap_ids)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return CapabilityScoringResult(
            metadata=Metadata(processing_time_ms=elapsed_ms),
            capability_scores=scores,
            readiness_summary=readiness_summary,
            strengths=feedback.get("strengths", []),
            weaknesses=feedback.get("weaknesses", []),
            missing_evidence=feedback.get("missing_evidence", []),
            recommendations=feedback.get("recommendations", []),
            validation_report=validation_report
        )
