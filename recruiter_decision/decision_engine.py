from typing import Dict, Any, List
import datetime
import time

from .models import (
    RecruiterDecisionResult,
    DecisionPolicyType,
    AuditTrail,
    DecisionSummary,
    HireDecision,
    ValidationReport
)
from .comparison_engine import ComparisonEngine
from .explanation_engine import ExplanationEngine
from .ranking_engine import RankingEngine
from .interview_engine import InterviewEngine
from .risk_engine import RiskEngine
from .validators import SchemaValidator

class DecisionEngine:
    def __init__(self):
        self.comparison_engine = ComparisonEngine()
        self.explanation_engine = ExplanationEngine()
        self.ranking_engine = RankingEngine()
        self.interview_engine = InterviewEngine()
        self.risk_engine = RiskEngine()
        self.validator = SchemaValidator()

    def generate_recruiter_decision(
        self,
        trust_result: Dict[str, Any],
        capability_result: Dict[str, Any],
        competency_result: Dict[str, Any],
        profile_result: Dict[str, Any],
        policy_type: DecisionPolicyType = DecisionPolicyType.ENTERPRISE,
        cohort_results: List[Dict[str, Any]] = None,
        human_override: Dict[str, str] = None
    ) -> RecruiterDecisionResult:
        
        start_time = time.time()
        
        # 1. Calculate Ranking
        ranking = self.ranking_engine.calculate_ranking(trust_result, capability_result, competency_result)
        
        # 2. Derive AI Recommendation based on policy and strength ranking
        ai_recommendation = self.comparison_engine.evaluate_ai_decision(ranking.candidate_ranking_score, policy_type)
        
        # 3. Handle Human Override
        final_decision = ai_recommendation
        decision_override = False
        override_reason = None
        override_by = None
        override_timestamp = None
        
        if human_override:
            override_val = human_override.get("final_decision")
            if override_val and override_val != ai_recommendation.value:
                final_decision = HireDecision(override_val)
                decision_override = True
                override_reason = human_override.get("override_reason")
                override_by = human_override.get("override_by")
                override_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                
        # 4. Generate Explanations
        explanation = self.explanation_engine.generate_explanation(trust_result, profile_result)
        
        # 5. Generate Interview Plan
        interview_plan = self.interview_engine.generate_interview_plan(trust_result, profile_result)
        
        # 6. Evaluate Risk
        risk_analysis = self.risk_engine.evaluate_risk(trust_result, profile_result)
        
        # 7. Evaluate Cohort Comparison
        comparison_summary = self.comparison_engine.evaluate_cohort(ranking.candidate_ranking_score, "current_candidate", cohort_results)

        # 8. Create Decision Summary
        decision_confidence = trust_result.get("confidence_summary", {}).get("overall_confidence", 0.0)
        
        decision_summary = DecisionSummary(
            ai_recommendation=ai_recommendation,
            final_decision=final_decision,
            decision_override=decision_override,
            override_reason=override_reason,
            override_by=override_by,
            override_timestamp=override_timestamp,
            decision_confidence=decision_confidence,
            decision_policy_used=policy_type
        )
        
        # 9. Create Audit Trail
        audit_trail = AuditTrail(
            pipeline_version="2.0",
            decision_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            modules_used=[
                "Job Intelligence",
                "Resume Intelligence",
                "Repository Intelligence",
                "Evidence Fusion",
                "Capability Scoring",
                "Competency Intelligence",
                "Candidate Profile",
                "Technical Assessment",
                "Behavioral Intelligence",
                "Trust Score Engine"
            ],
            policy_version=policy_type.value,
            schema_versions={
                "trust_score_engine": trust_result.get("metadata", {}).get("schema_version", "2.0"),
                "capability_scoring": capability_result.get("metadata", {}).get("schema_version", "2.0"),
                "competency_intelligence": competency_result.get("metadata", {}).get("schema_version", "2.0"),
                "candidate_profile": profile_result.get("metadata", {}).get("schema_version", "2.0")
            }
        )
        
        # 10. Generate Executive Summary
        exec_summary = f"AI Recommendation: {ai_recommendation.value}. Final Decision: {final_decision.value}. Ranking Score: {ranking.candidate_ranking_score}. Explainability: {explanation.explainability_score}."
        if decision_override:
            exec_summary += f" [HUMAN OVERRIDE APPLIED: {override_reason}]"

        # 11. Compile Result
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        result = RecruiterDecisionResult(
            metadata={
                "schema_version": "2.0",
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "processing_time_ms": processing_time_ms,
                "model": "recruiter-decision-engine-v2",
                "pipeline_module": "Recruiter Decision & Explainability Engine"
            },
            audit_trail=audit_trail,
            decision_summary=decision_summary,
            explanation=explanation,
            ranking=ranking,
            comparison_summary=comparison_summary,
            interview_plan=interview_plan,
            risk_analysis=risk_analysis,
            executive_summary=exec_summary,
            validation_report=ValidationReport(is_valid=True, warnings=[])
        )
        
        # 12. Validate Result
        validation_report = self.validator.validate_recruiter_decision(result)
        result.validation_report = validation_report
        
        return result
