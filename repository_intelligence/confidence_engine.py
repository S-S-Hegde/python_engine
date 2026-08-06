from typing import List
from .models import RepositoryEvidenceObject, ConfidenceSummary, OriginalityReport

class ConfidenceEngine:
    @staticmethod
    def calculate_evidence_confidence(
        evidence: RepositoryEvidenceObject,
        originality: OriginalityReport
    ) -> float:
        """
        Calculates a deterministic confidence score (0.0 - 100.0) based on:
        - Originality score (penalizes forked/dumped repos)
        - Code snippet / pattern detail specificity
        - Location depth
        """
        base = 70.0

        # Adjust for originality
        if originality.is_fork:
            base -= 30.0
        elif originality.is_single_day_dump:
            base -= 20.0

        # Location depth bonus
        if "/" in evidence.location:
            base += 10.0

        # Detail specificity bonus
        if len(evidence.quote) > 30:
            base += 10.0

        # Engineering decision clarity
        if evidence.engineering_decision and len(evidence.engineering_decision) > 15:
            base += 10.0

        final_conf = round(min(100.0, max(0.0, base)), 2)
        evidence.confidence = final_conf
        return final_conf

    @classmethod
    def compute_summary(
        cls,
        evidence_list: List[RepositoryEvidenceObject],
        originality: OriginalityReport
    ) -> ConfidenceSummary:
        if not evidence_list:
            return ConfidenceSummary(average_confidence=0.0, high_confidence_evidence_count=0)

        for ev in evidence_list:
            cls.calculate_evidence_confidence(ev, originality)

        avg_conf = round(sum(e.confidence for e in evidence_list) / len(evidence_list), 2)
        high_conf_count = sum(1 for e in evidence_list if e.confidence >= 80.0)

        return ConfidenceSummary(
            average_confidence=avg_conf,
            high_confidence_evidence_count=high_conf_count
        )
