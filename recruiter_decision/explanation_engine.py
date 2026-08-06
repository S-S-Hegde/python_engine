from typing import Dict, Any, List
from .models import Explanation, SupportedStatement

class ExplanationEngine:
    def generate_explanation(self, trust_result: Dict[str, Any], profile_result: Dict[str, Any]) -> Explanation:
        why_hire = []
        why_not_hire = []
        unsupported_count = 0
        
        # Parse strengths and verified capabilities from trust report
        report = trust_result.get("report", {})
        verif_summary = trust_result.get("verification_summary", {})
        
        strengths = report.get("strengths", [])
        verified_caps = verif_summary.get("verified_capabilities", [])
        
        if strengths:
            why_hire.append(SupportedStatement(
                statement="Candidate possesses strong verified capabilities.",
                supported_by=["trust_report.strengths", *verified_caps]
            ))
            
        # Parse weaknesses and missing evidence
        weaknesses = report.get("weaknesses", [])
        contradictions = verif_summary.get("contradictions_list", [])
        missing_evidence = report.get("missing_evidence", [])
        
        if weaknesses or missing_evidence:
            why_not_hire.append(SupportedStatement(
                statement="Candidate lacks sufficient evidence in key capability areas.",
                supported_by=["trust_report.weaknesses", "trust_report.missing_evidence"]
            ))
            
        if contradictions:
            why_not_hire.append(SupportedStatement(
                statement="Evidence sources contain conflicting information.",
                supported_by=["trust_report.contradictions"]
            ))
            
        # Calculate explainability score based on evidence presence
        total_statements = len(why_hire) + len(why_not_hire)
        if total_statements == 0:
            unsupported_count += 1
            why_not_hire.append(SupportedStatement(
                statement="No substantial evidence available to support a hiring decision.",
                supported_by=["missing_data"]
            ))
            
        explainability_score = 100.0 - (unsupported_count * 10)
        explainability_score = max(0.0, explainability_score)
            
        return Explanation(
            why_hire=why_hire,
            why_not_hire=why_not_hire,
            unsupported_reasoning_count=unsupported_count,
            explainability_score=explainability_score
        )
