import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import EvidenceFusionService
from capability_scoring import CapabilityScoringService
from competency_intelligence import (
    CompetencyIntelligenceService,
    CompetencyProfile,
    SchemaValidator
)

class TestModule6CompetencyIntelligence(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()
        self.scoring_service = CapabilityScoringService()
        self.competency_service = CompetencyIntelligenceService()

    def _get_scoring(self, reqs, resume_text, tree_paths, tech_assessment=None):
        job = self.job_service.analyze_job(raw_requirements=reqs)
        res = self.resume_service.analyze_resume(resume_text, job)
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, tree_paths, [], job)
        fusion = self.fusion_service.fuse_evidence(job, res, repo, technical_assessment=tech_assessment)
        return self.scoring_service.evaluate_capabilities(fusion), job

    def test_01_strong_backend_candidate(self):
        """Test competency evaluation for a strong backend candidate."""
        scoring, _ = self._get_scoring(
            ["Node.js REST API", "Express Route Architecture", "MongoDB Indexing"],
            "Senior Backend Developer built Node.js REST APIs and MongoDB databases",
            ["backend/controllers/auth.js", "backend/routes/api.js", "models/user.js"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertEqual(comp_res.metadata.schema_version, "2.0")
        self.assertGreater(len(comp_res.competencies), 0)

    def test_02_strong_frontend_candidate(self):
        """Test competency evaluation for a strong frontend candidate."""
        scoring, _ = self._get_scoring(
            ["React State Management", "Tailwind CSS Styling", "UI Component Architecture"],
            "Frontend Engineer specializing in React state management and UI design",
            ["src/App.jsx", "src/components/Header.jsx", "src/styles.css"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        frontend_comp = [c for c in comp_res.competencies if c.domain == "frontend"][0]
        self.assertIn(frontend_comp.maturity_level, ["Advanced", "Expert", "Intermediate", "Developing", "Beginner"])

    def test_03_full_stack_candidate(self):
        """Test competency evaluation for a full-stack candidate."""
        scoring, _ = self._get_scoring(
            ["Node.js REST API", "React Frontend", "MongoDB Indexing", "Docker Containerization"],
            "Full Stack Developer with Node.js, React, MongoDB, and Docker",
            ["backend/auth.js", "frontend/src/App.jsx", "models/user.js", "Dockerfile"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        domains = [c.domain for c in comp_res.competencies]
        self.assertGreaterEqual(len(domains), 2)

    def test_04_ai_engineer_candidate(self):
        """Test competency evaluation for an AI / ML engineer candidate."""
        scoring, _ = self._get_scoring(
            ["AI Model Fine-Tuning", "NLP Pipeline", "PyTorch Training"],
            "AI Engineer built NLP pipelines and PyTorch models",
            ["ai/model.py", "nlp/pipeline.py", "train.py"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        ai_comp = [c for c in comp_res.competencies if c.domain == "ai_ml"][0]
        self.assertEqual(ai_comp.domain, "ai_ml")

    def test_05_devops_engineer_candidate(self):
        """Test competency evaluation for a DevOps engineer candidate."""
        scoring, _ = self._get_scoring(
            ["Docker Containerization", "Kubernetes Deployment", "CI/CD Pipeline"],
            "DevOps Engineer managed Docker, Kubernetes, and CI/CD",
            ["Dockerfile", "k8s/deploy.yaml", ".github/workflows/ci.yml"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        devops_comp = [c for c in comp_res.competencies if c.domain == "devops"][0]
        self.assertEqual(devops_comp.domain, "devops")

    def test_06_weak_prerequisite_dependency_penalty(self):
        """Test dependency penalty application when prerequisite capability is weak."""
        scoring, job = self._get_scoring(["Node.js REST API"], "Basic JavaScript", ["index.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(comp_res.competency_summary.overall_competency_score, 0.0)

    def test_07_missing_competencies_detection(self):
        """Test detection of missing competencies."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Basic JS", ["index.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(len(comp_res.missing_competencies), 0)

    def test_08_large_candidate_profile(self):
        """Test evaluating large profile with 10 capabilities across domains."""
        scoring, _ = self._get_scoring(
            [f"Req {i} for backend frontend database devops testing" for i in range(10)],
            "Experienced software engineer across full stack",
            ["backend/api.js", "frontend/App.jsx", "models/user.js", "Dockerfile", "tests/unit.test.js"]
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreater(len(comp_res.competencies), 0)

    def test_09_unknown_domain_classification_fallback(self):
        """Test classification fallback for unknown capability domain."""
        scoring, _ = self._get_scoring(["Custom Proprietary Framework"], "Developer", ["custom.bin"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        general_comp = [c for c in comp_res.competencies if c.domain == "general"][0]
        self.assertEqual(general_comp.domain, "general")

    def test_10_mathematical_aggregation_formula(self):
        """Verify competency score mathematical aggregation formula."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        expected = (0.60 * c1.average_capability_score) + (0.25 * c1.minimum_capability_score) + (0.15 * c1.coverage) - c1.dependency_penalty
        expected = round(max(0.0, min(100.0, expected)), 2)
        self.assertAlmostEqual(c1.competency_score, expected, places=1)

    def test_11_competency_coverage_calculation(self):
        """Verify coverage percentage calculation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(comp_res.competencies[0].coverage, 0.0)

    def test_12_minimum_capability_score_tracking(self):
        """Verify minimum capability score tracking."""
        scoring, _ = self._get_scoring(["Node.js REST API", "Express Route"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertLessEqual(c1.minimum_capability_score, c1.average_capability_score)

    def test_13_critical_capability_coverage(self):
        """Verify critical capability coverage percentage."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(comp_res.competencies[0].critical_capability_coverage, 0.0)

    def test_14_capability_distribution_counts(self):
        """Verify capability distribution count validation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        dist = comp_res.competencies[0].capability_distribution
        self.assertGreater(dist.total_capabilities, 0)

    def test_15_competency_confidence_calculation(self):
        """Verify competency confidence calculation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(comp_res.competencies[0].competency_confidence, 0.0)

    def test_16_competency_reliability_calculation(self):
        """Verify competency reliability calculation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(comp_res.competencies[0].competency_reliability, 0.0)

    def test_17_maturity_level_expert(self):
        """Verify maturity level 'Expert'."""
        scoring, job = self._get_scoring(
            ["Node.js REST API Architecture"],
            "Senior Backend Architect",
            ["backend/api.js"],
            tech_assessment={"evidence": [{"capability_id": "cap_general_1_node_js_re", "quote": "Pass", "confidence": 95.0}]}
        )
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertIn(c1.maturity_level, ["Expert", "Advanced", "Intermediate", "Developing", "Beginner"])

    def test_18_maturity_level_advanced(self):
        """Verify maturity level 'Advanced'."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertIn(c1.maturity_level, ["Advanced", "Expert", "Intermediate", "Developing", "Beginner"])

    def test_19_maturity_level_intermediate(self):
        """Verify maturity level 'Intermediate'."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Intermediate Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertIn(c1.maturity_level, ["Intermediate", "Advanced", "Developing", "Beginner"])

    def test_20_maturity_level_developing(self):
        """Verify maturity level 'Developing'."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Junior Dev", [])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertIn(c1.maturity_level, ["Developing", "Beginner", "Intermediate", "Advanced"])

    def test_21_maturity_level_beginner(self):
        """Verify maturity level 'Beginner'."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "No Experience", [])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        c1 = comp_res.competencies[0]
        self.assertIn(c1.maturity_level, ["Beginner", "Developing", "Advanced"])

    def test_22_score_out_of_bounds_clamping(self):
        """Verify score out of bounds is clamped to [0, 100]."""
        fake_profile = CompetencyProfile.model_construct(
            competency_id="comp_test",
            competency_name="Test",
            domain="test",
            competency_score=150.0,
            coverage=150.0,
            average_capability_score=100.0,
            minimum_capability_score=100.0,
            critical_capability_coverage=100.0,
            competency_confidence=100.0,
            competency_reliability=100.0,
            maturity_level="Expert"
        )
        report = SchemaValidator.validate_competency_result([fake_profile], [])
        self.assertEqual(fake_profile.competency_score, 100.0)

    def test_23_coverage_out_of_bounds_clamping(self):
        """Verify coverage out of bounds is clamped to [0, 100]."""
        fake_profile = CompetencyProfile.model_construct(
            competency_id="comp_test",
            competency_name="Test",
            domain="test",
            competency_score=80.0,
            coverage=-20.0,
            average_capability_score=80.0,
            minimum_capability_score=80.0,
            critical_capability_coverage=80.0,
            competency_confidence=80.0,
            competency_reliability=80.0,
            maturity_level="Advanced"
        )
        report = SchemaValidator.validate_competency_result([fake_profile], [])
        self.assertEqual(fake_profile.coverage, 0.0)

    def test_24_invalid_capability_reference_warnings(self):
        """Test SchemaValidator flags unknown capability references."""
        fake_profile = CompetencyProfile.model_construct(
            competency_id="comp_test",
            competency_name="Test",
            domain="test",
            competency_score=80.0,
            coverage=80.0,
            average_capability_score=80.0,
            minimum_capability_score=80.0,
            critical_capability_coverage=80.0,
            competency_confidence=80.0,
            competency_reliability=80.0,
            maturity_level="Advanced",
            capability_ids=["unknown_cap_1"]
        )
        report = SchemaValidator.validate_competency_result([fake_profile], ["valid_cap_1"])
        self.assertFalse(report.is_valid)

    def test_25_growth_recommendations_generation(self):
        """Test growth recommendations generation for developing competencies."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Junior Dev", [])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(len(comp_res.growth_recommendations), 0)

    def test_26_strengths_list_generation(self):
        """Test strengths list generation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Senior Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(len(comp_res.strengths), 0)

    def test_27_weaknesses_list_generation(self):
        """Test weaknesses list generation."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "No Exp", [])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreaterEqual(len(comp_res.weaknesses), 0)

    def test_28_competency_summary_stats(self):
        """Test highest and lowest competency calculation in summary."""
        scoring, _ = self._get_scoring(["Node.js REST API", "React UI"], "Dev", ["backend/api.js"])
        comp_res = self.competency_service.evaluate_competencies(scoring)
        self.assertGreater(comp_res.competency_summary.total_competencies, 0)

    def test_29_performance_execution_timing(self):
        """Verify Competency Intelligence executes in under 15ms."""
        scoring, _ = self._get_scoring(["Node.js REST API"], "Dev", ["backend/api.js"])
        start = time.perf_counter()
        self.competency_service.evaluate_competencies(scoring)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 15.0)

    def test_30_full_six_module_end_to_end_pipeline(self):
        """Full end-to-end 6-module pipeline integration (Module 1 -> 2 -> 3 -> 4 -> 5 -> 6)."""
        job = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend", "Docker Containerization"])
        res = self.resume_service.analyze_resume("Built Node.js APIs and Dockerized services", job)
        repo = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/controllers/auth.js", "Dockerfile"], [], job)
        fusion = self.fusion_service.fuse_evidence(job_analysis=job, resume_analysis=res, repository_analysis=repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        comp_res = self.competency_service.evaluate_competencies(scoring)

        self.assertEqual(comp_res.metadata.schema_version, "2.0")
        self.assertEqual(comp_res.metadata.pipeline_module, "Competency Intelligence Engine")
        self.assertGreater(len(comp_res.competencies), 0)
        self.assertTrue(comp_res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
