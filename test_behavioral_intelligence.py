import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import EvidenceFusionService
from capability_scoring import CapabilityScoringService
from competency_intelligence import CompetencyIntelligenceService
from candidate_profile import CandidateProfileService
from technical_assessment import (
    TechnicalAssessmentService,
    TechnicalAssessmentSubmission,
    SubmissionQuestionItem
)
from behavioral_intelligence import (
    BehavioralIntelligenceService,
    BehavioralSubmissionPayload,
    BehavioralQuestionResponse,
    SchemaValidator
)

class TestModule9BehavioralIntelligence(unittest.TestCase):
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

        # Target job requirement
        self.job_analysis = self.job_service.analyze_job(raw_requirements=["Node.js REST API Architecture", "Docker Containerization"])

    def test_01_excellent_star_response(self):
        """Test response with complete STAR methodology structure."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_01",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    target_capability_id=self.job_analysis.capability_graph[0].id,
                    response_text="When our API traffic spiked by 300% in a project, my task was to optimize the route handler. I created an in-memory cache and refactored DB queries. As a result, we reduced latency by 45% and saved $10k in server costs."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertEqual(res.star_analysis[0].star_score, 100.0)
        self.assertEqual(res.evidence_objects[0].status, "Strongly Verified")

    def test_02_weak_star_response(self):
        """Test response missing STAR components."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_02",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="I usually write code and fix bugs whenever something goes wrong in production."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertLess(res.star_analysis[0].star_score, 75.0)

    def test_03_missing_star_structure(self):
        """Test unstructured response with 0 STAR components."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_03",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="Yes."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.star_analysis[0].star_score, 10.0)
        self.assertEqual(res.evidence_objects[0].status, "Unsupported")

    def test_04_ownership_and_accountability(self):
        """Test detection of strong ownership and accountability."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_04",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="When a deployment failed, I took responsibility for the configuration error, stepped up to roll back, and fixed the script."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.ownership_analysis.accountability_score, 95.0)
        self.assertFalse(res.ownership_analysis.blame_shifting_detected)

    def test_05_blame_shifting_detection(self):
        """Test detection of blame-shifting language."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_05",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="The build failed because they failed to test, it was not my fault and management forced the release."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertTrue(res.ownership_analysis.blame_shifting_detected)
        self.assertLessEqual(res.ownership_analysis.accountability_score, 30.0)

    def test_06_leadership_and_collaboration(self):
        """Test evaluation of collaboration and teamwork."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_06",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="I collaborated with cross-functional teams and partnered with product design to achieve a shared goal."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.leadership_analysis.collaboration_score, 90.0)

    def test_07_conflict_resolution(self):
        """Test evaluation of conflict resolution."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_07",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="When there was a disagreement on architecture, I listened to feedback and found a consensus."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.leadership_analysis.conflict_resolution_score, 90.0)

    def test_08_mentorship_evaluation(self):
        """Test evaluation of mentorship."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_08",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="I mentored junior developers and shared knowledge during team retrospectives."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.leadership_analysis.mentorship_score, 90.0)

    def test_09_ethical_decision_making(self):
        """Test detection of ethical awareness and data privacy integrity."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_09",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(
                    question_id="q1",
                    response_text="I evaluated privacy and security compliance to maintain user trust and data protection."
                )
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertGreaterEqual(res.capability_scores[0].final_capability_score, 40.0)

    def test_10_communication_conciseness_optimal(self):
        """Test communication evaluation with optimal word count."""
        words = ["word"] * 80
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_10",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text=" ".join(words))]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.communication_analysis.conciseness_score, 90.0)

    def test_11_communication_rambled_penalty(self):
        """Test communication penalty for rambled responses."""
        words = ["word"] * 300
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_11",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text=" ".join(words))]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.communication_analysis.conciseness_score, 65.0)

    def test_12_explicit_capability_mapping(self):
        """Verify response mapped to explicit Module 1 capability ID."""
        cap_id = self.job_analysis.capability_graph[0].id
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_12",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", target_capability_id=cap_id, response_text="Text")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.capability_scores[0].capability_id, cap_id)

    def test_13_capability_mapping_fallback(self):
        """Verify capability mapping fallback when no target ID provided."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_13",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Text")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertIn("cap_", res.capability_scores[0].capability_id)

    def test_14_evidence_object_generation_no_hallucination(self):
        """Verify evidence quote contains exact candidate response excerpt."""
        sample_text = "When I designed the auth microservice, I ensured high availability."
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_14",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text=sample_text)]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertIn(sample_text, res.evidence_objects[0].quote)

    def test_15_source_tagging_and_ownership(self):
        """Verify evidence object source and ownership fields."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_15",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Response")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        ev = res.evidence_objects[0]
        self.assertEqual(ev.source, "behavioral_assessment")
        self.assertEqual(ev.ownership, "Candidate Response")

    def test_16_status_strongly_verified(self):
        """Verify Strongly Verified status for 100% STAR score."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_16",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="When X, my task was Y, I created Z. Result: improved by 50%.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Strongly Verified")

    def test_17_status_verified(self):
        """Verify Verified status for 75% STAR score."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_17",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="When X, my task was Y, I created Z.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Verified")

    def test_18_status_partially_verified(self):
        """Verify Partially Verified status for 50% STAR score."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_18",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="When X, I created Z.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Partially Verified")

    def test_19_status_weakly_verified(self):
        """Verify Weakly Verified status for 25% STAR score."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_19",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="When X happened.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Weakly Verified")

    def test_20_status_unsupported(self):
        """Verify Unsupported status for non-STAR response."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_20",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Hello.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Unsupported")

    def test_21_out_of_bounds_clamping(self):
        """Verify SchemaValidator clamps out of bounds scores to [0, 100]."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_21",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Text")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        res.behavioral_summary.overall_behavioral_score = 150.0
        report = SchemaValidator.validate_behavioral_result(res)
        self.assertEqual(res.behavioral_summary.overall_behavioral_score, 100.0)

    def test_22_validation_report_completeness(self):
        """Test SchemaValidator returns valid validation report."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_22",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Text")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertTrue(res.validation_report.is_valid)

    def test_23_evidence_fusion_compatibility(self):
        """Verify behavioral evidence objects convert into Evidence Fusion format."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_23",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", target_capability_id=self.job_analysis.capability_graph[0].id, response_text="When X, my task Y, I created Z. Result: improved by 50%.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        beh_dict = {
            "evidence": [ev.model_dump() for ev in res.evidence_objects]
        }
        fusion = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            behavioral_assessment=beh_dict
        )
        self.assertGreater(len(fusion.capability_profiles), 0)

    def test_24_learning_mindset_detection(self):
        """Test learning mindset detection."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_24",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="I learned valuable lessons and improved our process.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.ownership_analysis.learning_mindset_score, 90.0)

    def test_25_multi_response_submission(self):
        """Test submission with multiple behavioral responses."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_25",
            candidate_id="cand_1",
            responses=[
                BehavioralQuestionResponse(question_id="q1", response_text="Resp 1"),
                BehavioralQuestionResponse(question_id="q2", response_text="Resp 2")
            ]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(len(res.star_analysis), 2)

    def test_26_primary_strengths_compilation(self):
        """Test compilation of primary strengths."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_26",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="When X, my task Y, I took responsibility and collaborated with cross-functional teams. Result: 50% improvement.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertGreater(len(res.behavioral_summary.primary_strengths), 0)

    def test_27_areas_for_growth_compilation(self):
        """Test compilation of areas for growth."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_27",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Brief response.")]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertGreater(len(res.behavioral_summary.areas_for_growth), 0)

    def test_28_confidence_summary_verification_level(self):
        """Test confidence summary calculation."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_28",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Response text", audio_transcript_confidence=95.0)]
        )
        res = self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.confidence_summary.verification_level, "High")

    def test_29_performance_execution_timing(self):
        """Verify Behavioral Intelligence Engine executes in under 15ms."""
        sub = BehavioralSubmissionPayload(
            assessment_id="beh_29",
            candidate_id="cand_1",
            responses=[BehavioralQuestionResponse(question_id="q1", response_text="Response text")]
        )
        start = time.perf_counter()
        self.behavior_service.analyze_behavior(sub, job_analysis=self.job_analysis)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 15.0)

    def test_30_full_nine_module_end_to_end_pipeline(self):
        """Full end-to-end 9-module pipeline integration (Modules 1 -> 2 -> 3 -> 8 -> 9 -> 4 -> 5 -> 6 -> 7)."""
        job = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend"])
        res = self.resume_service.analyze_resume("Built Node.js APIs", job)
        repo = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/api.js"], [], job)

        tech_sub = TechnicalAssessmentService().analyze_assessment(
            TechnicalAssessmentSubmission(
                assessment_id="a30", candidate_id="c30",
                questions=[SubmissionQuestionItem(question_id="q1", target_capability_id=job.capability_graph[0].id, submitted_code="function api(){}", test_cases=[{"passed": True}])]
            ), job_analysis=job
        )

        beh_sub = self.behavior_service.analyze_behavior(
            BehavioralSubmissionPayload(
                assessment_id="b30", candidate_id="c30",
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

        self.assertEqual(beh_sub.metadata.schema_version, "2.0")
        self.assertEqual(profile.metadata.schema_version, "2.0")
        self.assertTrue(beh_sub.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
