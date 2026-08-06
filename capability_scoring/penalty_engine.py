from typing import Dict, Any, List

class PenaltyEngine:
    @staticmethod
    def calculate_missing_evidence_penalty(missing_sources: List[str]) -> float:
        """
        Calculates penalty for missing evidence streams.
        - Missing Repository Verification: 15.0 penalty
        - Missing Technical Assessment: 10.0 penalty
        - Missing Resume Claim: 5.0 penalty
        """
        penalty = 0.0
        for src in missing_sources:
            if "Repository" in src:
                penalty += 15.0
            elif "Technical" in src:
                penalty += 10.0
            elif "Resume" in src:
                penalty += 5.0

        return round(min(50.0, max(0.0, penalty)), 2)

    @staticmethod
    def calculate_contradiction_penalty(contradictions: List[Dict[str, Any]]) -> float:
        """
        Calculates penalty for flagged contradictions based on severity.
        - Low: 5.0
        - Medium: 15.0
        - High: 25.0
        - Critical: 40.0
        """
        if not contradictions:
            return 0.0

        penalty = 0.0
        for c in contradictions:
            severity = str(c.get("severity", "Medium")).capitalize()
            if severity == "Critical":
                penalty += 40.0
            elif severity == "High":
                penalty += 25.0
            elif severity == "Medium":
                penalty += 15.0
            else:
                penalty += 5.0

        return round(min(70.0, max(0.0, penalty)), 2)
