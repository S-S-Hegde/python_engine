import re
from typing import List
from .models import EvidenceObject, ConfidenceSummary

class ConfidenceEngine:
    @staticmethod
    def calculate_evidence_confidence(evidence: EvidenceObject) -> float:
        """
        Calculates a deterministic confidence score (0.0 - 100.0) based on:
        - Specificity & quote length
        - Quantified numbers/metrics
        - Engineering decision depth
        - Ownership clarity
        - Complexity rating
        """
        base_confidence = 60.0

        # Specificity check
        if len(evidence.quote) > 30:
            base_confidence += 10.0

        # Quantified metrics check
        if re.search(r'\d+%', evidence.quote) or re.search(r'\d+', evidence.quote):
            base_confidence += 15.0
            evidence.verification_status = "Quantified Claim"

        # Engineering decision depth check
        if evidence.engineering_decision and len(evidence.engineering_decision) > 15:
            base_confidence += 10.0
            if "architecture" in evidence.engineering_decision.lower() or "designed" in evidence.quote.lower():
                evidence.verification_status = "Architecture Claim"

        # Ownership bonus
        ownership_map = {
            "Individual": 5.0,
            "Primary Contributor": 3.0,
            "Team Contributor": 1.0,
            "Unknown": 0.0
        }
        base_confidence += ownership_map.get(evidence.ownership, 0.0)

        # Complexity bonus
        complexity_map = {
            "Very High": 5.0,
            "High": 4.0,
            "Medium": 2.0,
            "Low": 0.0,
            "Very Low": -2.0
        }
        base_confidence += complexity_map.get(evidence.complexity, 0.0)

        # Bound strictly between 0.0 and 100.0
        final_confidence = round(min(100.0, max(0.0, base_confidence)), 2)
        evidence.confidence = final_confidence
        return final_confidence

    @classmethod
    def compute_summary(cls, evidence_list: List[EvidenceObject]) -> ConfidenceSummary:
        if not evidence_list:
            return ConfidenceSummary(average_confidence=0.0, high_confidence_claims_count=0)

        for ev in evidence_list:
            cls.calculate_evidence_confidence(ev)

        avg_conf = round(sum(e.confidence for e in evidence_list) / len(evidence_list), 2)
        high_conf_count = sum(1 for e in evidence_list if e.confidence >= 80.0)

        return ConfidenceSummary(
            average_confidence=avg_conf,
            high_confidence_claims_count=high_conf_count
        )
