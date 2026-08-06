from typing import Dict, Any, List
from .models import RiskAnalysis, SupportedStatement

class RiskEngine:
    def evaluate_risk(self, trust_result: Dict[str, Any], profile_result: Dict[str, Any]) -> RiskAnalysis:
        trust_risk = trust_result.get("risk_summary", {})
        
        base_risk_score = trust_risk.get("risk_score", 0.0)
        risk_factors = trust_risk.get("risk_factors", [])
        
        critical_vulnerabilities: List[SupportedStatement] = []
        for factor in risk_factors:
            critical_vulnerabilities.append(SupportedStatement(
                statement=f"Risk Factor Identified: {factor}",
                supported_by=["trust_risk.risk_factors"]
            ))
            
        mitigations = trust_risk.get("mitigation_recommendations", [])
        if not mitigations:
            mitigations.append("Conduct a deep-dive technical interview on architecture and design principles.")
            
        risk_level = "Low"
        if base_risk_score >= 80.0:
            risk_level = "Critical"
        elif base_risk_score >= 50.0:
            risk_level = "High"
        elif base_risk_score >= 20.0:
            risk_level = "Medium"

        return RiskAnalysis(
            risk_score=base_risk_score,
            risk_level=risk_level,
            critical_vulnerabilities=critical_vulnerabilities,
            mitigations=mitigations
        )
