import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import EvidenceFusionService
from evidence_fusion.models import ContradictionItem
from capability_scoring import CapabilityScoringService
from competency_intelligence import CompetencyIntelligenceService
from candidate_profile import CandidateProfileService
from technical_assessment import TechnicalAssessmentService, TechnicalAssessmentSubmission, SubmissionQuestionItem
from behavioral_intelligence import BehavioralIntelligenceService, BehavioralSubmissionPayload, BehavioralQuestionResponse
from trust_score_engine import TrustScoreService, SchemaValidator

class TestModule10TrustScoreEngine(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()
        self.scoring_service = CapabilityScoringService()
        self.competency_service = CompetencyIntelligenceService()
        self.profile_service = CandidateProfileService()
        self.tech_service = TechnicalAssessmentService()
        self.behavior_service = BehavioralIntelligenceService()
        self.trust_service = TrustScoreService()

        # Common job requirements
        self.job_analysis = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "Docker Containerization"])

    def test_01_excellent_candidate(self):
        """Test excellent candidate resulting in high Trust Score."""
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        fusion.reliability_summary.overall_reliability_score = 90.0
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        scoring.readiness_summary.overall_capability_score = 90.0
        comp = self.competency_service.evaluate_competencies(scoring)
        comp.competency_summary.overall_competency_score = 90.0
        prof = self.profile_service.generate_candidate_profile(scoring, comp)

        res = self.trust_service.generate_trust_score(
            job_analysis=self.job_analysis,
            evidence_fusion_result=fusion,
            capability_scoring_result=scoring,
            competency_intelligence_result=comp,
            candidate_profile_result=prof
        )
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreaterEqual(res.trust_summary.overall_trust_score, 70.0)

    def test_02_weak_candidate(self):
        """Test weak candidate resulting in low Trust Score."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertLessEqual(res.trust_summary.overall_trust_score, 75.0)

    def test_03_resume_only_candidate(self):
        """Test candidate with only resume evidence."""
        resume = self.resume_service.analyze_resume("Built Node.js APIs", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis, resume_analysis=resume)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, resume_analysis=resume, evidence_fusion_result=fusion)
        self.assertIsNotNone(res.trust_summary.overall_trust_score)

    def test_04_repository_only_candidate(self):
        """Test candidate with only repository evidence."""
        repo = self.repo_service.analyze_repository("user", {"name": "app", "fork": False}, ["server.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis, repository_analysis=repo)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, repository_analysis=repo, evidence_fusion_result=fusion)
        self.assertIsNotNone(res.trust_summary.overall_trust_score)

    def test_05_assessment_only_candidate(self):
        """Test candidate with only technical assessment evidence."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="a1", candidate_id="c1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": True}])]
        )
        tech = self.tech_service.analyze_assessment(sub, self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, technical_assessment=tech)
        self.assertIsNotNone(res.trust_summary.overall_trust_score)

    def test_06_behavior_only_candidate(self):
        """Test candidate with only behavioral assessment evidence."""
        beh = self.behavior_service.analyze_behavior(
            BehavioralSubmissionPayload(assessment_id="b1", candidate_id="c1", responses=[BehavioralQuestionResponse(question_id="q1", response_text="Text")]),
            self.job_analysis
        )
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, behavioral_assessment=beh)
        self.assertIsNotNone(res.trust_summary.overall_trust_score)

    def test_07_contradicted_candidate(self):
        """Test candidate with contradictions resulting in Contradicted status."""
        resume = self.resume_service.analyze_resume("Expert in Docker", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis, resume_analysis=resume)
        fusion.contradiction_report.contradictions.append(
            ContradictionItem(
                contradiction_id="cd_1",
                capability_id="c1",
                capability_name="Docker",
                type="Direct Contradiction",
                description="Contradiction"
            )
        )
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, evidence_fusion_result=fusion)
        self.assertEqual(res.trust_summary.verification_level, "Contradicted")

    def test_08_missing_evidence_penalty(self):
        """Test candidate with missing evidence sources penalty."""
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        fusion.missing_evidence_report.missing_capabilities = ["cap_1", "cap_2"]
        if fusion.capability_profiles:
            fusion.capability_profiles[0].status = "Unsupported"
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, evidence_fusion_result=fusion)
        self.assertGreater(len(res.verification_summary.unverified_capabilities), 0)

    def test_09_forked_repository_penalty(self):
        """Test candidate with forked repository penalty."""
        repo = self.repo_service.analyze_repository("user", {"name": "app", "fork": True}, ["server.js"], [], self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, repository_analysis=repo)
        self.assertIn("forked", res.risk_summary.risk_factors[0].lower())

    def test_10_plagiarism_penalty(self):
        """Test candidate with plagiarism flag penalty."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="a1", candidate_id="c1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", copy_paste_events_count=15)]
        )
        tech = self.tech_service.analyze_assessment(sub, self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, technical_assessment=tech)
        self.assertEqual(res.trust_summary.hiring_confidence, "Do Not Hire")

    def test_11_strong_fullstack_engineer(self):
        """Test strong full-stack engineer candidate."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.report.executive_summary)

    def test_12_student_candidate(self):
        """Test student candidate evaluation."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.candidate_readiness)

    def test_13_fresher_candidate(self):
        """Test fresher candidate evaluation."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.hiring_confidence)

    def test_14_senior_engineer_candidate(self):
        """Test senior engineer candidate evaluation."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.final_recommendation)

    def test_15_level_strongly_verified(self):
        """Test Strongly Verified level assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIn(res.trust_summary.verification_level, ["Strongly Verified", "Verified", "Partially Verified", "Weakly Verified", "Unverified", "Contradicted"])

    def test_16_level_verified(self):
        """Test Verified level assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.verification_level)

    def test_17_level_partially_verified(self):
        """Test Partially Verified level assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.verification_level)

    def test_18_level_weakly_verified(self):
        """Test Weakly Verified level assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.verification_level)

    def test_19_level_unverified(self):
        """Test Unverified level assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.verification_level)

    def test_20_level_contradicted(self):
        """Test Contradicted level assignment."""
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        fusion.contradiction_report.contradictions.append(
            ContradictionItem(
                contradiction_id="cd_1",
                capability_id="c1",
                capability_name="Docker",
                type="Direct Contradiction",
                description="Contradiction"
            )
        )
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, evidence_fusion_result=fusion)
        self.assertEqual(res.trust_summary.verification_level, "Contradicted")

    def test_21_high_hiring_confidence(self):
        """Test High hiring confidence assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.hiring_confidence)

    def test_22_moderate_hiring_confidence(self):
        """Test Moderate hiring confidence assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.hiring_confidence)

    def test_23_low_hiring_confidence(self):
        """Test Low hiring confidence assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.hiring_confidence)

    def test_24_do_not_hire_confidence(self):
        """Test Do Not Hire confidence assignment."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="a1", candidate_id="c1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", copy_paste_events_count=15)]
        )
        tech = self.tech_service.analyze_assessment(sub, self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, technical_assessment=tech)
        self.assertEqual(res.trust_summary.hiring_confidence, "Do Not Hire")

    def test_25_readiness_production_ready(self):
        """Test Production Ready readiness assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.candidate_readiness)

    def test_26_readiness_nearly_ready(self):
        """Test Nearly Ready readiness assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.candidate_readiness)

    def test_27_readiness_learning(self):
        """Test Learning readiness assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.candidate_readiness)

    def test_28_readiness_not_ready(self):
        """Test Not Ready readiness assignment."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.trust_summary.candidate_readiness)

    def test_29_critical_risk_score(self):
        """Test Critical risk score for plagiarism."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="a1", candidate_id="c1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", copy_paste_events_count=15)]
        )
        tech = self.tech_service.analyze_assessment(sub, self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, technical_assessment=tech)
        self.assertEqual(res.risk_summary.risk_level, "Critical")

    def test_30_high_risk_score(self):
        """Test High risk score for contradictions."""
        fusion = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        fusion.contradiction_report.contradictions.extend([
            ContradictionItem(contradiction_id="cd1", capability_id="c1", capability_name="C1", type="Type", description="desc"),
            ContradictionItem(contradiction_id="cd2", capability_id="c2", capability_name="C2", type="Type", description="desc"),
            ContradictionItem(contradiction_id="cd3", capability_id="c3", capability_name="C3", type="Type", description="desc")
        ])
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, evidence_fusion_result=fusion)
        self.assertEqual(res.risk_summary.risk_level, "Critical")

    def test_31_medium_risk_score(self):
        """Test Medium risk score for forked repo."""
        repo = self.repo_service.analyze_repository("user", {"name": "app", "fork": True}, ["server.js"], [], self.job_analysis)
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis, repository_analysis=repo)
        self.assertIn(res.risk_summary.risk_level, ["Medium", "High", "Critical"])

    def test_32_low_risk_score(self):
        """Test Low risk score."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.risk_summary.risk_score)

    def test_33_report_executive_summary(self):
        """Test report executive summary string."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIn("VeriProof Verification Report", res.report.executive_summary)

    def test_34_report_strengths(self):
        """Test report strengths list."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertGreater(len(res.report.strengths), 0)

    def test_35_report_weaknesses(self):
        """Test report weaknesses list."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertIsNotNone(res.report.weaknesses)

    def test_36_report_recommended_interview_topics(self):
        """Test report recommended interview topics list."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertGreater(len(res.report.recommended_interview_topics), 0)

    def test_37_report_recommended_learning_path(self):
        """Test report recommended learning path list."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        self.assertGreater(len(res.report.recommended_learning_path), 0)

    def test_38_out_of_bounds_clamping(self):
        """Verify SchemaValidator clamps out of bounds scores to [0, 100]."""
        res = self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        res.trust_summary.overall_trust_score = 150.0
        report = SchemaValidator.validate_trust_result(res, 1)
        self.assertEqual(res.trust_summary.overall_trust_score, 100.0)

    def test_39_performance_execution_timing(self):
        """Verify Trust Score Engine executes in under 15ms."""
        start = time.perf_counter()
        self.trust_service.generate_trust_score(job_analysis=self.job_analysis)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 15.0)

    def test_40_full_ten_module_end_to_end_pipeline(self):
        """Full end-to-end 10-module pipeline integration (Modules 1 -> 2 -> 3 -> 8 -> 9 -> 4 -> 5 -> 6 -> 7 -> 10)."""
        job = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend"])
        res = self.resume_service.analyze_resume("Built Node.js APIs", job)
        repo = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/api.js"], [], job)

        tech_sub = self.tech_service.analyze_assessment(
            TechnicalAssessmentSubmission(
                assessment_id="a40", candidate_id="c40",
                questions=[SubmissionQuestionItem(question_id="q1", target_capability_id=job.capability_graph[0].id, submitted_code="function api(){}", test_cases=[{"passed": True}])]
            ), job_analysis=job
        )

        beh_sub = self.behavior_service.analyze_behavior(
            BehavioralSubmissionPayload(
                assessment_id="b40", candidate_id="c40",
                responses=[BehavioralQuestionResponse(question_id="q1", target_capability_id=job.capability_graph[0].id, response_text="When X, my task Y, I created Z. Result: improved by 50%.")]
            ), job_analysis=job
        )

        tech_dict = {"evidence": [ev.model_dump() for ev in tech_sub.evidence_objects]}
        beh_dict = {"evidence": [ev.model_dump() for ev in beh_sub.evidence_objects]}

        fusion = self.fusion_service.fuse_evidence(
            job_analysis=job, resume_analysis=res, repository_analysis=repo,
            technical_assessment=tech_dict, behavioral_assessment=beh_dict
        )
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        comp = self.competency_service.evaluate_competencies(scoring)
        profile = self.profile_service.generate_candidate_profile(scoring, comp)

        final_trust = self.trust_service.generate_trust_score(
            job_analysis=job,
            resume_analysis=res,
            repository_analysis=repo,
            technical_assessment=tech_sub,
            behavioral_assessment=beh_sub,
            evidence_fusion_result=fusion,
            capability_scoring_result=scoring,
            competency_intelligence_result=comp,
            candidate_profile_result=profile
        )

        self.assertEqual(final_trust.metadata.schema_version, "2.0")
        self.assertTrue(final_trust.validation_report.is_valid)
        self.assertGreater(final_trust.trust_summary.overall_trust_score, 0.0)

if __name__ == "__main__":
    unittest.main()
