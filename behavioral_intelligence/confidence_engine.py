from typing import List
from .models import ConfidenceSummary, ValidationReport, BehavioralAnalysisResult, BehavioralQuestionResponse

class ConfidenceEngine:
    @classmethod
    def compute_summary(
        cls,
        responses: List[BehavioralQuestionResponse],
        blame_shifting_detected: bool
    ) -> ConfidenceSummary:
        if not responses:
            return ConfidenceSummary(average_behavioral_confidence=50.0, verification_level="Low")

        avg_transcript_conf = sum(r.audio_transcript_confidence for r in responses) / len(responses)

        if blame_shifting_detected:
            conf = min(40.0, avg_transcript_conf * 0.50)
            level = "Low (Flagged Blame Shifting)"
        elif avg_transcript_conf >= 85.0:
            conf = avg_transcript_conf
            level = "High"
        elif avg_transcript_conf >= 50.0:
            conf = avg_transcript_conf
            level = "Medium"
        else:
            conf = avg_transcript_conf
            level = "Low"

        return ConfidenceSummary(
            average_behavioral_confidence=round(conf, 2),
            verification_level=level
        )

class SchemaValidator:
    @staticmethod
    def validate_behavioral_result(result: BehavioralAnalysisResult) -> ValidationReport:
        warnings: List[str] = []

        s = result.behavioral_summary
        if not (0.0 <= s.overall_behavioral_score <= 100.0):
            warnings.append(f"Overall behavioral score {s.overall_behavioral_score} out of bounds [0, 100]. Clamping.")
            s.overall_behavioral_score = max(0.0, min(100.0, s.overall_behavioral_score))

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
            total_responses_validated=len(result.star_analysis),
            warnings=warnings
        )
