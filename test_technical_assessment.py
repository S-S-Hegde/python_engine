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
    SubmissionQuestionItem,
    SchemaValidator
)

class TestModule8TechnicalAssessment(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()
        self.scoring_service = CapabilityScoringService()
        self.competency_service = CompetencyIntelligenceService()
        self.profile_service = CandidateProfileService()
        self.tech_service = TechnicalAssessmentService()

        # Target job requirement
        self.job_analysis = self.job_service.analyze_job(raw_requirements=["Node.js REST API Architecture", "Docker Containerization"])

    def test_01_correct_solution(self):
        """Test submission with 100% correct test cases."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_01",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    target_capability_id=self.job_analysis.capability_graph[0].id,
                    submitted_code="function buildApi() { try { return 'ok'; } catch(e) {} }",
                    test_cases=[{"test_id": "tc1", "passed": True}, {"test_id": "tc2", "passed": True}]
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertEqual(res.assessment_summary.overall_pass_rate, 100.0)
        self.assertEqual(res.evidence_objects[0].status, "Strongly Verified")

    def test_02_wrong_answers(self):
        """Test submission with failed test cases."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_02",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve() { return false; }",
                    test_cases=[{"test_id": "tc1", "passed": False}, {"test_id": "tc2", "passed": False}]
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.assessment_summary.overall_pass_rate, 0.0)
        self.assertEqual(res.evidence_objects[0].status, "Unsupported")

    def test_03_runtime_errors(self):
        """Test submission with runtime error."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_03",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve() { throw new Error('Null pointer'); }",
                    runtime_error="TypeError: Cannot read property of undefined"
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.execution_results[0].status, "RuntimeError")
        self.assertEqual(res.evidence_objects[0].status, "Contradicted")

    def test_04_compile_errors(self):
        """Test submission with compile error."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_04",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve( {",
                    compilation_error="SyntaxError: Unexpected token {"
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.execution_results[0].status, "CompileError")

    def test_05_edge_case_failures(self):
        """Test submission where edge case tests fail."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_05",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve() {}",
                    test_cases=[
                        {"test_id": "tc1", "test_type": "public", "passed": True},
                        {"test_id": "tc2", "test_type": "edge_case", "passed": False}
                    ]
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertFalse(res.execution_results[0].edge_cases_passed)

    def test_06_hidden_tests_failures(self):
        """Test submission where hidden tests fail."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_06",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve() {}",
                    test_cases=[
                        {"test_id": "tc1", "test_type": "public", "passed": True},
                        {"test_id": "tc2", "test_type": "hidden", "passed": False}
                    ]
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertFalse(res.execution_results[0].hidden_tests_passed)

    def test_07_plagiarism_copy_paste_anomaly(self):
        """Test plagiarism detection via copy-paste count anomaly."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_07",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="function solve() { return true; }",
                    copy_paste_events_count=15
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertTrue(res.plagiarism_report.is_plagiarized)

    def test_08_plagiarism_rapid_speed(self):
        """Test plagiarism detection via rapid typing speed anomaly."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_08",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(
                    question_id="q1",
                    submitted_code="\n".join([f"let var_{i} = {i};" for i in range(20)]),
                    time_spent_seconds=5
                )
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertTrue(res.plagiarism_report.is_plagiarized)

    def test_09_complexity_constant_o1(self):
        """Test complexity evaluation for O(1) constant code."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_09",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="let a = 1 + 2; return a;")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.complexity_analysis[0].time_complexity, "O(1)")

    def test_10_complexity_linear_on(self):
        """Test complexity evaluation for O(n) linear loop."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_10",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="for (let i=0; i<n; i++) { sum += i; }")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.complexity_analysis[0].time_complexity, "O(n)")

    def test_11_complexity_logarithmic_onlogn(self):
        """Test complexity evaluation for O(n log n) code."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_11",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="function binarySearch() { for(let i=0; i<n; i++) {} }")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.complexity_analysis[0].time_complexity, "O(n log n)")

    def test_12_complexity_quadratic_on2(self):
        """Test complexity evaluation for O(n^2) nested loop."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_12",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="for(let i=0; i<n; i++) { for(let j=0; j<n; j++) {} }")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.complexity_analysis[0].time_complexity, "O(n^2)")

    def test_13_complexity_exponential_o2n(self):
        """Test complexity evaluation for O(2^n) recursive code."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_13",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="function recursion(n) { return recursion(n-1) + recursion(n-2); }")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.complexity_analysis[0].time_complexity, "O(2^n)")

    def test_14_explicit_module1_capability_mapping(self):
        """Verify question mapped to explicit Module 1 capability ID."""
        cap_id = self.job_analysis.capability_graph[0].id
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_14",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", target_capability_id=cap_id, submitted_code="code")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.capability_scores[0].capability_id, cap_id)

    def test_15_capability_mapping_fallback(self):
        """Verify capability mapping fallback when no target ID provided."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_15",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertIn("cap_", res.capability_scores[0].capability_id)

    def test_16_code_quality_readability(self):
        """Test readability score calculation with comments."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_16",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="// Helpful comment\nlet x = 10;\nlet y = 20;")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertGreaterEqual(res.code_quality.readability_score, 80.0)

    def test_17_code_quality_modular_design(self):
        """Test modular design score calculation with functions."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_17",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="function main() { return 1; }")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.code_quality.modular_design_score, 90.0)

    def test_18_code_quality_error_handling(self):
        """Test error handling score calculation with try/catch."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_18",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="try { run(); } catch(e) {}")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.code_quality.error_handling_score, 95.0)

    def test_19_source_tagging_and_ownership(self):
        """Verify evidence object source and ownership fields."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_19",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        ev = res.evidence_objects[0]
        self.assertEqual(ev.source, "technical_assessment")
        self.assertEqual(ev.ownership, "Candidate Submission")

    def test_20_status_strongly_verified(self):
        """Verify Strongly Verified status for 100% pass rate."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_20",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": True}])]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Strongly Verified")

    def test_21_status_verified(self):
        """Verify Verified status for 75% pass rate."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_21",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": True}, {"passed": True}, {"passed": True}, {"passed": False}])]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Verified")

    def test_22_status_partially_verified(self):
        """Verify Partially Verified status for 50% pass rate."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_22",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": True}, {"passed": False}])]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Partially Verified")

    def test_23_status_contradicted(self):
        """Verify Contradicted status for runtime/compile error."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_23",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", runtime_error="Error")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Contradicted")

    def test_24_status_unsupported(self):
        """Verify Unsupported status for 0% pass rate."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_24",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": False}])]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(res.evidence_objects[0].status, "Unsupported")

    def test_25_out_of_bounds_clamping(self):
        """Verify SchemaValidator clamps out of bounds scores to [0, 100]."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_25",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        res.assessment_summary.overall_score = 150.0
        report = SchemaValidator.validate_assessment_result(res)
        self.assertEqual(res.assessment_summary.overall_score, 100.0)

    def test_26_validation_report_warning_tracking(self):
        """Test SchemaValidator returns validation report."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_26",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code")]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertTrue(res.validation_report.is_valid)

    def test_27_evidence_fusion_compatibility(self):
        """Verify technical assessment evidence objects convert into Evidence Fusion format."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_27",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", target_capability_id=self.job_analysis.capability_graph[0].id, submitted_code="code", test_cases=[{"passed": True}])]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        tech_assessment_dict = {
            "evidence": [ev.model_dump() for ev in res.evidence_objects]
        }
        fusion = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            technical_assessment=tech_assessment_dict
        )
        self.assertGreater(len(fusion.capability_profiles), 0)

    def test_28_multi_question_submission(self):
        """Test submission with multiple questions."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_28",
            candidate_id="cand_1",
            questions=[
                SubmissionQuestionItem(question_id="q1", submitted_code="code 1", test_cases=[{"passed": True}]),
                SubmissionQuestionItem(question_id="q2", submitted_code="code 2", test_cases=[{"passed": False}])
            ]
        )
        res = self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        self.assertEqual(len(res.execution_results), 2)

    def test_29_performance_execution_timing(self):
        """Verify Technical Assessment Engine executes in under 15ms."""
        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_29",
            candidate_id="cand_1",
            questions=[SubmissionQuestionItem(question_id="q1", submitted_code="code", test_cases=[{"passed": True}])]
        )
        start = time.perf_counter()
        self.tech_service.analyze_assessment(sub, job_analysis=self.job_analysis)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 15.0)

    def test_30_full_eight_module_end_to_end_pipeline(self):
        """Full end-to-end 8-module pipeline integration (Modules 1 -> 2 -> 3 -> 8 -> 4 -> 5 -> 6 -> 7)."""
        job = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend"])
        res = self.resume_service.analyze_resume("Built Node.js APIs", job)
        repo = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/api.js"], [], job)

        sub = TechnicalAssessmentSubmission(
            assessment_id="asm_30",
            candidate_id="cand_30",
            questions=[SubmissionQuestionItem(question_id="q1", target_capability_id=job.capability_graph[0].id, submitted_code="function api() { try{}catch(e){} }", test_cases=[{"passed": True}])]
        )
        tech_res = self.tech_service.analyze_assessment(sub, job_analysis=job)

        tech_dict = {"evidence": [ev.model_dump() for ev in tech_res.evidence_objects]}
        fusion = self.fusion_service.fuse_evidence(job_analysis=job, resume_analysis=res, repository_analysis=repo, technical_assessment=tech_dict)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        comp = self.competency_service.evaluate_competencies(scoring)
        profile = self.profile_service.generate_candidate_profile(scoring, comp)

        self.assertEqual(tech_res.metadata.schema_version, "2.0")
        self.assertEqual(profile.metadata.schema_version, "2.0")
        self.assertTrue(tech_res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
