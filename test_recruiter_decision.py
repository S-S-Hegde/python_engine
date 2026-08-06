import unittest
from typing import Dict, Any

from recruiter_decision.models import (
    HireDecision,
    DecisionPolicyType,
    RecruiterDecisionRequestPayload
)
from recruiter_decision.decision_engine import DecisionEngine
from recruiter_decision import RecruiterDecisionService

def mock_result(score: float, caps: list = None, risk: float = 10.0, contradictions: list = None, overrides: dict = None) -> RecruiterDecisionRequestPayload:
    trust = {
        "metadata": {"schema_version": "2.0"},
        "trust_summary": {"overall_trust_score": score},
        "risk_summary": {"risk_score": risk, "risk_factors": ["High turnover rate in previous roles"] if risk > 50 else [], "mitigation_recommendations": []},
        "verification_summary": {"verified_capabilities": caps or [], "contradictions_list": contradictions or []},
        "report": {
            "strengths": ["Strong backend skills"] if score > 70 else [],
            "weaknesses": ["System design"] if score < 60 else [],
            "missing_evidence": ["Cloud deployment"],
            "recommended_interview_topics": ["System design", "Cloud architecture"]
        },
        "confidence_summary": {"overall_confidence": 85.0}
    }
    
    cap = {
        "metadata": {"schema_version": "2.0"},
        "capability_scores": [{"capability_id": "c1", "score": score}]
    }
    
    comp = {
        "metadata": {"schema_version": "2.0"},
        "competencies": [{"competency_id": "c1", "maturity_score": score}]
    }
    
    prof = {
        "metadata": {"schema_version": "2.0"},
        "profile": {"seniority_level": "Senior"}
    }
    
    return RecruiterDecisionRequestPayload(
        trust_score_result=trust,
        capability_scoring_result=cap,
        competency_intelligence_result=comp,
        candidate_profile_result=prof,
        policy_type=DecisionPolicyType.ENTERPRISE,
        cohort_results=None,
        human_override=overrides
    )

class TestRecruiterDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.service = RecruiterDecisionService()
        
    def test_01_excellent_candidate_enterprise(self):
        payload = mock_result(95.0, caps=["Python", "Docker"])
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.STRONG_HIRE)
        
    def test_02_excellent_candidate_ranking(self):
        payload = mock_result(95.0)
        res = self.service.generate_decision(payload)
        self.assertGreaterEqual(res.ranking.candidate_ranking_score, 90.0)
        
    def test_03_weak_candidate_enterprise(self):
        payload = mock_result(45.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.REJECT)
        
    def test_04_weak_candidate_ranking(self):
        payload = mock_result(45.0)
        res = self.service.generate_decision(payload)
        self.assertLessEqual(res.ranking.candidate_ranking_score, 50.0)
        
    def test_05_startup_policy_lenient(self):
        payload = mock_result(68.0)
        payload.policy_type = DecisionPolicyType.STARTUP
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.HIRE)
        
    def test_06_startup_policy_borderline(self):
        payload = mock_result(55.0)
        payload.policy_type = DecisionPolicyType.STARTUP
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.BORDERLINE)
        
    def test_07_enterprise_policy_strict(self):
        payload = mock_result(68.0)
        payload.policy_type = DecisionPolicyType.ENTERPRISE
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.BORDERLINE)
        
    def test_08_intern_policy_lenient(self):
        payload = mock_result(48.0)
        payload.policy_type = DecisionPolicyType.INTERN
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.HIRE)
        
    def test_09_senior_policy_strict(self):
        payload = mock_result(80.0)
        payload.policy_type = DecisionPolicyType.SENIOR
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.BORDERLINE)
        
    def test_10_explainability_high_score(self):
        payload = mock_result(90.0, caps=["A", "B"])
        res = self.service.generate_decision(payload)
        self.assertEqual(res.explanation.explainability_score, 100.0)
        
    def test_11_explainability_low_score(self):
        payload = mock_result(10.0)
        payload.trust_score_result["report"]["strengths"] = []
        payload.trust_score_result["report"]["weaknesses"] = []
        payload.trust_score_result["report"]["missing_evidence"] = []
        res = self.service.generate_decision(payload)
        self.assertEqual(res.explanation.explainability_score, 90.0) # 1 unsupported reason
        
    def test_12_evidence_traceability_hire(self):
        payload = mock_result(90.0, caps=["A", "B"])
        res = self.service.generate_decision(payload)
        self.assertGreater(len(res.explanation.why_hire), 0)
        self.assertIn("trust_report.strengths", res.explanation.why_hire[0].supported_by)
        
    def test_13_evidence_traceability_not_hire(self):
        payload = mock_result(40.0)
        res = self.service.generate_decision(payload)
        self.assertGreater(len(res.explanation.why_not_hire), 0)
        self.assertIn("trust_report.weaknesses", res.explanation.why_not_hire[0].supported_by)
        
    def test_14_contradicted_candidate_explanation(self):
        payload = mock_result(50.0, contradictions=["Resume claims AWS, Repo shows no AWS"])
        res = self.service.generate_decision(payload)
        statements = [s.statement for s in res.explanation.why_not_hire]
        self.assertTrue(any("conflicting information" in s for s in statements))
        
    def test_15_audit_trail_presence(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertIsNotNone(res.audit_trail)
        self.assertEqual(res.audit_trail.pipeline_version, "2.0")
        
    def test_16_audit_trail_modules_used(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertIn("Trust Score Engine", res.audit_trail.modules_used)
        self.assertIn("Evidence Fusion", res.audit_trail.modules_used)
        
    def test_17_audit_trail_policy_version(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.audit_trail.policy_version, "Enterprise")
        
    def test_18_audit_trail_schema_versions(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.audit_trail.schema_versions["trust_score_engine"], "2.0")
        self.assertEqual(res.audit_trail.schema_versions["capability_scoring"], "2.0")
        
    def test_19_human_override_applies(self):
        payload = mock_result(90.0, overrides={"final_decision": "Reject", "override_reason": "Culture fit", "override_by": "Recruiter1"})
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.STRONG_HIRE)
        self.assertEqual(res.decision_summary.final_decision, HireDecision.REJECT)
        self.assertTrue(res.decision_summary.decision_override)
        
    def test_20_human_override_metadata(self):
        payload = mock_result(90.0, overrides={"final_decision": "Reject", "override_reason": "Culture fit", "override_by": "Recruiter1"})
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.override_reason, "Culture fit")
        self.assertEqual(res.decision_summary.override_by, "Recruiter1")
        self.assertIsNotNone(res.decision_summary.override_timestamp)
        
    def test_21_human_override_immutable_ai(self):
        payload = mock_result(45.0, overrides={"final_decision": "Hire", "override_reason": "Urgent", "override_by": "Manager"})
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.REJECT)
        self.assertEqual(res.decision_summary.final_decision, HireDecision.HIRE)
        
    def test_22_human_override_executive_summary(self):
        payload = mock_result(45.0, overrides={"final_decision": "Hire", "override_reason": "Urgent", "override_by": "Manager"})
        res = self.service.generate_decision(payload)
        self.assertIn("[HUMAN OVERRIDE APPLIED: Urgent]", res.executive_summary)
        
    def test_23_validation_report_override_missing_reason(self):
        payload = mock_result(90.0, overrides={"final_decision": "Reject", "override_by": "Manager"})
        res = self.service.generate_decision(payload)
        self.assertFalse(res.validation_report.is_valid)
        self.assertTrue(any("override_reason is missing" in w for w in res.validation_report.warnings))
        
    def test_24_multi_candidate_cohort_size(self):
        payload = mock_result(80.0)
        payload.cohort_results = [{"trust_summary": {"overall_trust_score": 70.0}}, {"trust_summary": {"overall_trust_score": 90.0}}]
        res = self.service.generate_decision(payload)
        self.assertEqual(res.comparison_summary.cohort_size, 3)
        
    def test_25_multi_candidate_ranking_position(self):
        payload = mock_result(80.0)
        payload.cohort_results = [{"trust_summary": {"overall_trust_score": 70.0}}, {"trust_summary": {"overall_trust_score": 90.0}}]
        res = self.service.generate_decision(payload)
        self.assertEqual(res.comparison_summary.ranking_position, 2)
        
    def test_26_multi_candidate_percentile(self):
        payload = mock_result(80.0)
        payload.cohort_results = [{"trust_summary": {"overall_trust_score": 70.0}}, {"trust_summary": {"overall_trust_score": 90.0}}]
        res = self.service.generate_decision(payload)
        self.assertEqual(res.comparison_summary.percentile, 50.0)
        
    def test_27_multi_candidate_top_quartile(self):
        payload = mock_result(95.0)
        payload.cohort_results = [{"trust_summary": {"overall_trust_score": 70.0}}, {"trust_summary": {"overall_trust_score": 60.0}}, {"trust_summary": {"overall_trust_score": 50.0}}]
        res = self.service.generate_decision(payload)
        self.assertEqual(res.comparison_summary.percentile, 100.0)
        self.assertIn("Candidate is in the top quartile", res.comparison_summary.relative_strengths[0])
        
    def test_28_multi_candidate_bottom_quartile(self):
        payload = mock_result(45.0)
        payload.cohort_results = [{"trust_summary": {"overall_trust_score": 70.0}}, {"trust_summary": {"overall_trust_score": 60.0}}, {"trust_summary": {"overall_trust_score": 80.0}}]
        res = self.service.generate_decision(payload)
        self.assertEqual(res.comparison_summary.percentile, 0.0)
        self.assertIn("Candidate falls in the bottom quartile", res.comparison_summary.relative_weaknesses[0])
        
    def test_29_interview_engine_focus_areas(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertGreaterEqual(len(res.interview_plan.focus_areas), 1)
        self.assertEqual(res.interview_plan.focus_areas[0].topic, "System design")
        
    def test_30_interview_engine_missing_evidence(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        topics = [f.topic for f in res.interview_plan.focus_areas]
        self.assertIn("Cloud deployment", topics)
        
    def test_31_interview_engine_questions(self):
        payload = mock_result(70.0)
        res = self.service.generate_decision(payload)
        self.assertGreater(len(res.interview_plan.technical_questions_to_ask), 0)
        self.assertGreater(len(res.interview_plan.behavioral_questions_to_ask), 0)
        
    def test_32_risk_engine_critical(self):
        payload = mock_result(30.0, risk=85.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.risk_analysis.risk_level, "Critical")
        
    def test_33_risk_engine_high(self):
        payload = mock_result(50.0, risk=65.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.risk_analysis.risk_level, "High")
        
    def test_34_risk_engine_medium(self):
        payload = mock_result(70.0, risk=35.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.risk_analysis.risk_level, "Medium")
        
    def test_35_risk_engine_low(self):
        payload = mock_result(90.0, risk=5.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.risk_analysis.risk_level, "Low")
        
    def test_36_risk_factors_traceability(self):
        payload = mock_result(50.0, risk=65.0)
        res = self.service.generate_decision(payload)
        self.assertGreater(len(res.risk_analysis.critical_vulnerabilities), 0)
        self.assertEqual(res.risk_analysis.critical_vulnerabilities[0].supported_by[0], "trust_risk.risk_factors")
        
    def test_37_schema_validation_success(self):
        payload = mock_result(90.0)
        res = self.service.generate_decision(payload)
        self.assertTrue(res.validation_report.is_valid)
        self.assertEqual(len(res.validation_report.warnings), 0)
        
    def test_38_ranking_metrics_bounds(self):
        payload = mock_result(90.0)
        res = self.service.generate_decision(payload)
        self.assertTrue(0.0 <= res.ranking.candidate_ranking_score <= 100.0)
        self.assertTrue(0.0 <= res.ranking.engineering_strength_ranking <= 100.0)
        self.assertTrue(0.0 <= res.ranking.engineering_weakness_ranking <= 100.0)
        
    def test_39_custom_policy_support(self):
        payload = mock_result(86.0)
        payload.policy_type = DecisionPolicyType.CUSTOM
        res = self.service.generate_decision(payload)
        self.assertEqual(res.decision_summary.ai_recommendation, HireDecision.STRONG_HIRE)
        
    def test_40_processing_time_metrics(self):
        payload = mock_result(90.0)
        res = self.service.generate_decision(payload)
        self.assertIn("processing_time_ms", res.metadata)
        self.assertGreaterEqual(res.metadata["processing_time_ms"], 0.0)
        
    def test_41_metadata_schema_version(self):
        payload = mock_result(90.0)
        res = self.service.generate_decision(payload)
        self.assertEqual(res.metadata["schema_version"], "2.0")

if __name__ == '__main__':
    unittest.main()
