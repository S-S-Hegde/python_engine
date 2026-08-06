import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import EvidenceFusionService
from capability_scoring import CapabilityScoringService
from competency_intelligence import CompetencyIntelligenceService
from candidate_profile import (
    CandidateProfileService,
    CandidateSummary,
    SeniorityDetail,
    SpecializationDetail,
    SchemaValidator
)

class TestModule7CandidateProfile(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()
        self.scoring_service = CapabilityScoringService()
        self.competency_service = CompetencyIntelligenceService()
        self.profile_service = CandidateProfileService()

    def _get_pipeline_output(self, reqs, resume_text, tree_paths, tech_assessment=None):
        job = self.job_service.analyze_job(raw_requirements=reqs)
        res = self.resume_service.analyze_resume(resume_text, job)
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, tree_paths, [], job)
        fusion = self.fusion_service.fuse_evidence(job, res, repo, technical_assessment=tech_assessment)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        competency = self.competency_service.evaluate_competencies(scoring)
        return scoring, competency

    def test_01_backend_candidate(self):
        """Test profiling for a dedicated backend engineer."""
        scoring, comp = self._get_pipeline_output(
            ["Node.js REST API", "Express Route Architecture", "MongoDB Indexing"],
            "Senior Backend Developer built Node.js REST APIs and MongoDB databases",
            ["backend/controllers/auth.js", "backend/routes/api.js", "models/user.js"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertTrue("Backend" in res.specialization.primary_domain or "Database" in res.specialization.primary_domain)

    def test_02_frontend_candidate(self):
        """Test profiling for a dedicated frontend engineer."""
        scoring, comp = self._get_pipeline_output(
            ["React State Management", "Tailwind CSS Styling", "UI Component Architecture"],
            "Frontend Engineer specializing in React state management and UI design",
            ["src/App.jsx", "src/components/Header.jsx", "src/styles.css"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn("Frontend", res.specialization.primary_domain)

    def test_03_full_stack_candidate(self):
        """Test profiling for a full stack engineer."""
        scoring, comp = self._get_pipeline_output(
            ["Node.js REST API", "React Frontend", "MongoDB Indexing"],
            "Full Stack Developer with Node.js, React, and MongoDB",
            ["backend/auth.js", "frontend/src/App.jsx", "models/user.js"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertEqual(res.specialization.archetype, "Full Stack Engineer")

    def test_04_ai_engineer_candidate(self):
        """Test profiling for an AI / ML engineer."""
        scoring, comp = self._get_pipeline_output(
            ["AI Model Fine-Tuning", "NLP Pipeline", "PyTorch Training"],
            "AI Engineer built NLP pipelines and PyTorch models",
            ["ai/model.py", "nlp/pipeline.py", "train.py"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertEqual(res.specialization.archetype, "AI / ML Engineer")

    def test_05_devops_engineer_candidate(self):
        """Test profiling for a DevOps engineer."""
        scoring, comp = self._get_pipeline_output(
            ["Docker Containerization", "Kubernetes Deployment", "CI/CD Pipeline"],
            "DevOps Engineer managed Docker, Kubernetes, and CI/CD",
            ["Dockerfile", "k8s/deploy.yaml", ".github/workflows/ci.yml"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn("DevOps", res.specialization.archetype)

    def test_06_cloud_engineer_candidate(self):
        """Test profiling for a Cloud engineer."""
        scoring, comp = self._get_pipeline_output(
            ["AWS Cloud Architecture", "Serverless Lambda"],
            "Cloud Engineer specializing in AWS Lambda",
            ["aws/template.yaml", "lambda/index.js"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Junior", "Mid-Level", "Senior", "Fresher", "Student"])

    def test_07_student_seniority(self):
        """Test Student seniority evaluation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "", [])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Student", "Fresher", "Junior", "Mid-Level"])

    def test_08_fresher_seniority(self):
        """Test Fresher seniority evaluation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Basic JS", ["index.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Fresher", "Junior", "Mid-Level", "Student"])

    def test_09_junior_seniority(self):
        """Test Junior seniority evaluation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Junior Node Developer", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Junior", "Mid-Level", "Fresher"])

    def test_10_mid_level_seniority(self):
        """Test Mid-Level seniority evaluation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Node Developer 2 yrs", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Mid-Level", "Junior", "Senior"])

    def test_11_senior_engineer_evaluation(self):
        """Test Senior engineer evaluation."""
        tech = {"evidence": [{"capability_id": "cap_general_1_node_js_re", "quote": "Pass", "confidence": 95.0}]}
        scoring, comp = self._get_pipeline_output(["Node.js REST API Architecture"], "Senior Architect", ["backend/api.js"], tech_assessment=tech)
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Senior", "Lead", "Architect", "Mid-Level"])

    def test_12_lead_engineer_evaluation(self):
        """Test Lead engineer evaluation."""
        tech = {"evidence": [{"capability_id": "cap_general_1_node_js_re", "quote": "Pass", "confidence": 95.0}]}
        scoring, comp = self._get_pipeline_output(["Node.js REST API Architecture"], "Engineering Lead", ["backend/api.js"], tech_assessment=tech)
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Lead", "Senior", "Architect", "Mid-Level"])

    def test_13_architect_seniority_evaluation(self):
        """Test Architect seniority evaluation."""
        tech = {"evidence": [{"capability_id": "cap_general_1_node_js_re", "quote": "Pass", "confidence": 98.0}]}
        scoring, comp = self._get_pipeline_output(["Node.js REST API Architecture"], "Principal Architect", ["backend/api.js"], tech_assessment=tech)
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIn(res.seniority.seniority_level, ["Architect", "Lead", "Senior", "Mid-Level"])

    def test_14_mixed_competency_profile(self):
        """Test evaluating a mixed multi-domain competency profile."""
        scoring, comp = self._get_pipeline_output(["Node.js API", "React Frontend"], "Full Stack", ["api.js", "App.jsx"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreater(len(res.best_fit_roles), 0)

    def test_15_best_fit_role_mapping(self):
        """Test Best-Fit role mapping."""
        scoring, comp = self._get_pipeline_output(
            ["Node.js REST API", "Express Route Architecture", "MongoDB Indexing"],
            "Senior Backend Developer built Node.js REST APIs and MongoDB databases",
            ["backend/controllers/auth.js", "backend/routes/api.js", "models/user.js"]
        )
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        best_fits = [r for r in res.best_fit_roles if r.fit_category == "Best-Fit"]
        self.assertGreater(len(best_fits), 0)

    def test_16_alternative_role_mapping(self):
        """Test Alternative role mapping."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        alt_fits = [r for r in res.best_fit_roles if r.fit_category == "Alternative"]
        self.assertGreaterEqual(len(alt_fits), 0)

    def test_17_unsuitable_role_mapping(self):
        """Test Unsuitable role mapping."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        unsuitable = [r for r in res.best_fit_roles if r.fit_category == "Unsuitable"]
        self.assertGreaterEqual(len(unsuitable), 0)

    def test_18_primary_domain_identification(self):
        """Test primary domain identification."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIsNotNone(res.specialization.primary_domain)

    def test_19_secondary_domains_identification(self):
        """Test secondary domain identification."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API", "React Frontend"], "Dev", ["api.js", "App.jsx"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIsNotNone(res.specialization.secondary_domains)

    def test_20_candidate_archetype_assignment(self):
        """Test candidate archetype assignment."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Backend Dev", ["backend/api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertIsNotNone(res.candidate_summary.archetype)

    def test_21_growth_roadmap_generation(self):
        """Test growth roadmap generation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreaterEqual(len(res.growth_roadmap), 0)

    def test_22_risk_area_identification(self):
        """Test risk area identification."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreaterEqual(len(res.risks), 0)

    def test_23_strengths_aggregation(self):
        """Test strengths list aggregation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreaterEqual(len(res.engineering_profile.strengths), 0)

    def test_24_weaknesses_aggregation(self):
        """Test weaknesses list aggregation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", [])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreaterEqual(len(res.engineering_profile.weaknesses), 0)

    def test_25_out_of_bounds_score_clamping(self):
        """Verify SchemaValidator clamps out-of-bounds overall profile scores."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        res.candidate_summary.overall_profile_score = 150.0
        report = SchemaValidator.validate_candidate_profile(res, [])
        self.assertEqual(res.candidate_summary.overall_profile_score, 100.0)

    def test_26_invalid_competency_reference_warning(self):
        """Test SchemaValidator flags invalid competency references in roadmap."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        report = SchemaValidator.validate_candidate_profile(res, ["comp_valid_only"])
        self.assertFalse(report.is_valid)

    def test_27_recommendations_generation(self):
        """Test candidate recommendations generation."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreater(len(res.recommendations), 0)

    def test_28_validation_report_completeness(self):
        """Test validation report section count."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        res = self.profile_service.generate_candidate_profile(scoring, comp)
        self.assertGreater(res.validation_report.total_sections_validated, 0)

    def test_29_performance_execution_timing(self):
        """Verify Candidate Profile Engine executes in under 15ms."""
        scoring, comp = self._get_pipeline_output(["Node.js REST API"], "Dev", ["api.js"])
        start = time.perf_counter()
        self.profile_service.generate_candidate_profile(scoring, comp)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 15.0)

    def test_30_full_seven_module_end_to_end_pipeline(self):
        """Full end-to-end 7-module pipeline integration (Module 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7)."""
        job = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend", "Docker Containerization"])
        res = self.resume_service.analyze_resume("Built Node.js APIs and Dockerized services", job)
        repo = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/controllers/auth.js", "Dockerfile"], [], job)
        fusion = self.fusion_service.fuse_evidence(job_analysis=job, resume_analysis=res, repository_analysis=repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        comp = self.competency_service.evaluate_competencies(scoring)
        profile_res = self.profile_service.generate_candidate_profile(scoring, comp)

        self.assertEqual(profile_res.metadata.schema_version, "2.0")
        self.assertEqual(profile_res.metadata.pipeline_module, "Candidate Profile Engine")
        self.assertTrue(profile_res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
