from typing import List, Dict, Any, Optional
from .models import RiskSummary

class RiskEngine:
    @classmethod
    def evaluate_risk_summary(
        cls,
        contradictions_count: int,
        unsupported_count: int,
        is_forked_repo: bool,
        is_plagiarized: bool
    ) -> RiskSummary:
        risk_factors: List[str] = []
        mitigations: List[str] = []
        raw_risk = 0.0

        if is_plagiarized:
            raw_risk += 75.0
            risk_factors.append("Plagiarism / Copy-Paste Anomaly Flagged.")
            mitigations.append("Require live invigilated coding session.")

        if contradictions_count > 0:
            raw_risk += contradictions_count * 25.0
            risk_factors.append(f"{contradictions_count} verification contradiction(s) between resume and code.")
            mitigations.append("Conduct technical interview deep-dive on contradicted capabilities.")

        if is_forked_repo:
            raw_risk += 30.0
            risk_factors.append("Repository code is forked from another author.")
            mitigations.append("Verify original commit authorship and PR contributions.")

        if unsupported_count > 2:
            raw_risk += 20.0
            risk_factors.append(f"{unsupported_count} capabilities lack code or assessment proof.")
            mitigations.append("Assign target technical assessment for unverified capabilities.")

        risk_score = round(max(0.0, min(100.0, raw_risk)), 2)

        if risk_score >= 70.0:
            level = "Critical"
        elif risk_score >= 45.0:
            level = "High"
        elif risk_score >= 20.0:
            level = "Medium"
        else:
            level = "Low"

        return RiskSummary(
            risk_score=risk_score,
            risk_level=level,
            risk_factors=risk_factors,
            mitigation_recommendations=mitigations
        )
