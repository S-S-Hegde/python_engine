from typing import List
from .models import ExecutionResult, ConfidenceSummary, ValidationReport, AssessmentAnalysisResult

class ConfidenceEngine:
    @classmethod
    def compute_summary(
        cls,
        execution_results: List[ExecutionResult],
        is_plagiarized: bool
    ) -> ConfidenceSummary:
        if not execution_results:
            return ConfidenceSummary(average_assessment_confidence=50.0, verification_level="Low")

        avg_pass = sum(r.pass_rate for r in execution_results) / len(execution_results)

        if is_plagiarized:
            conf = min(30.0, avg_pass * 0.40)
            level = "Low (Flagged Plagiarism)"
        elif avg_pass >= 85.0:
            conf = avg_pass
            level = "High"
        elif avg_pass >= 50.0:
            conf = avg_pass
            level = "Medium"
        else:
            conf = avg_pass
            level = "Low"

        return ConfidenceSummary(
            average_assessment_confidence=round(conf, 2),
            verification_level=level
        )

class SchemaValidator:
    @staticmethod
    def validate_assessment_result(result: AssessmentAnalysisResult) -> ValidationReport:
        warnings: List[str] = []

        s = result.assessment_summary
        if not (0.0 <= s.overall_score <= 100.0):
            warnings.append(f"Overall assessment score {s.overall_score} out of bounds [0, 100]. Clamping.")
            s.overall_score = max(0.0, min(100.0, s.overall_score))

        for c in result.capability_scores:
            if not (0.0 <= c.final_capability_score <= 100.0):
                warnings.append(f"Capability '{c.capability_id}' score {c.final_capability_score} out of bounds [0, 100]. Clamping.")
                c.final_capability_score = max(0.0, min(100.0, c.final_capability_score))

        for ev in result.evidence_objects:
            if not (0.0 <= ev.confidence <= 100.0):
                warnings.append(f"Evidence '{ev.evidence_id}' confidence {ev.confidence} out of bounds [0, 100]. Clamping.")
                ev.confidence = max(0.0, min(100.0, ev.confidence))

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_submissions_evaluated=len(result.execution_results),
            warnings=warnings
        )
