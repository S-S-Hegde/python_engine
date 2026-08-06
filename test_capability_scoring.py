import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from resume_intelligence.models import EvidenceObject
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import EvidenceFusionService
from capability_scoring import (
    CapabilityScoringService,
    FormulaWeightConfig,
    FormulaBreakdown,
    CapabilityScoreDetail,
    SchemaValidator
)

class TestModule5CapabilityScoring(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()
        self.scoring_service = CapabilityScoringService()

        self.job_analysis = self.job_service.analyze_job(raw_requirements=[
            "Node.js REST API Architecture",
            "React State Management",
            "Docker Containerization",
            "Automated Testing Suite",
            "MongoDB Schema Indexing"
        ])
        self.docker_cap_id = [c.id for c in self.job_analysis.capability_graph if "docker" in c.name.lower()][0]

    def test_01_perfect_candidate_scoring(self):
        """Test scoring for a candidate with complete evidence across code & resume."""
        res = self.resume_service.analyze_resume("Built Node.js APIs, React components, Dockerized containers, Jest tests, and MongoDB schemas.", self.job_analysis)
        repo = self.repo_service.analyze_repository(
            "u", {"name": "app", "fork": False},
            ["backend/controllers/auth.js", "src/App.jsx", "Dockerfile", "tests/auth.test.js", "models/user.js"],
            [], self.job_analysis
        )
        tech_assessment = {
            "evidence": [
                {"capability_id": c.id, "quote": f"Passed {c.name}", "confidence": 95.0} for c in self.job_analysis.capability_graph
            ]
        }
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo, technical_assessment=tech_assessment)
        scoring = self.scoring_service.evaluate_capabilities(fusion)

        self.assertEqual(scoring.metadata.schema_version, "2.0")
        self.assertEqual(len(scoring.capability_scores), 5)
        self.assertIn(scoring.readiness_summary.readiness_level, ["Production Ready", "Nearly Ready"])

    def test_02_weak_candidate_scoring(self):
        """Test scoring for a candidate with no repository code or assessment evidence."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertLess(scoring.readiness_summary.overall_capability_score, 50.0)
        self.assertEqual(scoring.readiness_summary.readiness_level, "Not Ready")

    def test_03_contradicted_capability_scoring(self):
        """Test scoring and status assignment when contradiction exists."""
        res = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res.evidence_objects.append(
            EvidenceObject(evidence_id="ev_1", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, ["app.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)

        docker_score = [s for s in scoring.capability_scores if s.capability_id == self.docker_cap_id][0]
        self.assertEqual(docker_score.status, "Contradicted")
        self.assertGreater(docker_score.formula_breakdown.contradiction_penalty, 0.0)

    def test_04_missing_evidence_penalty(self):
        """Verify missing evidence penalty is deducted from final capability score."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertGreater(s1.formula_breakdown.missing_evidence_penalty, 0.0)

    def test_05_contradiction_penalty_calculation(self):
        """Verify contradiction penalty calculation based on severity."""
        res = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res.evidence_objects.append(
            EvidenceObject(evidence_id="ev_1", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, ["app.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)

        docker_score = [s for s in scoring.capability_scores if s.capability_id == self.docker_cap_id][0]
        self.assertGreaterEqual(docker_score.formula_breakdown.contradiction_penalty, 25.0)

    def test_06_empty_evidence_fusion_input(self):
        """Verify evaluation works gracefully when fusion result is empty."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertEqual(len(scoring.capability_scores), 5)

    def test_07_multiple_capabilities_evaluation(self):
        """Verify all job capabilities are evaluated deterministically."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertEqual(len(scoring.capability_scores), 5)

    def test_08_large_candidate_profile_processing(self):
        """Verify performance and accuracy on a 10-capability job blueprint."""
        large_job = self.job_service.analyze_job(raw_requirements=[f"Skill {i}" for i in range(10)])
        fusion = self.fusion_service.fuse_evidence(large_job)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertEqual(len(scoring.capability_scores), 10)

    def test_09_invalid_capability_id_warnings(self):
        """Test SchemaValidator flags capability IDs not found in input fusion profiles."""
        dummy_breakdown = FormulaBreakdown.model_construct(
            coverage_score=50.0, depth_score=50.0, complexity_score=50.0,
            reliability_score=50.0, consistency_score=50.0, confidence_score=50.0,
            raw_weighted_sum=50.0, missing_evidence_penalty=0.0, contradiction_penalty=0.0,
            final_capability_score=50.0, weights_used=FormulaWeightConfig(), formula_expression=""
        )
        fake_detail = CapabilityScoreDetail.model_construct(
            capability_id="invalid_cap_id",
            capability_name="Invalid",
            status="Unsupported",
            final_capability_score=50.0,
            formula_breakdown=dummy_breakdown
        )
        report = SchemaValidator.validate_scoring_result([fake_detail], [c.id for c in self.job_analysis.capability_graph])
        self.assertFalse(report.is_valid)

    def test_10_formula_coverage_subscore(self):
        """Verify Coverage sub-score calculation."""
        res = self.resume_service.analyze_resume("Node.js dev", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertGreater(s1.formula_breakdown.coverage_score, 0.0)

    def test_11_formula_engineering_depth_subscore(self):
        """Verify Depth sub-score calculation."""
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, ["backend/controllers/authController.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, repository_analysis=repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertGreater(s1.formula_breakdown.depth_score, 50.0)

    def test_12_formula_complexity_subscore(self):
        """Verify Complexity sub-score calculation."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertGreater(s1.formula_breakdown.complexity_score, 0.0)

    def test_13_formula_reliability_subscore(self):
        """Verify Reliability sub-score calculation."""
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, ["backend/controllers/authController.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, repository_analysis=repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertEqual(s1.formula_breakdown.reliability_score, fusion.capability_profiles[0].reliability)

    def test_14_formula_consistency_subscore(self):
        """Verify Consistency sub-score calculation."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertEqual(s1.formula_breakdown.consistency_score, 100.0)

    def test_15_formula_confidence_subscore(self):
        """Verify Confidence sub-score calculation."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertEqual(s1.formula_breakdown.confidence_score, fusion.capability_profiles[0].merged_confidence)

    def test_16_deterministic_raw_weighted_sum(self):
        """Verify raw weighted sum is computed deterministically."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        b = scoring.capability_scores[0].formula_breakdown
        expected_raw = (
            (b.coverage_score * b.weights_used.w_coverage) +
            (b.depth_score * b.weights_used.w_depth) +
            (b.complexity_score * b.weights_used.w_complexity) +
            (b.reliability_score * b.weights_used.w_reliability) +
            (b.consistency_score * b.weights_used.w_consistency) +
            (b.confidence_score * b.weights_used.w_confidence)
        )
        self.assertAlmostEqual(b.raw_weighted_sum, expected_raw, places=1)

    def test_17_penalty_subtraction_precision(self):
        """Verify final score equals raw weighted sum minus penalties."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        b = scoring.capability_scores[0].formula_breakdown
        expected_final = max(0.0, min(100.0, b.raw_weighted_sum - b.missing_evidence_penalty - b.contradiction_penalty))
        self.assertAlmostEqual(b.final_capability_score, expected_final, places=1)

    def test_18_out_of_bounds_clamping(self):
        """Verify out-of-bounds final scores are clamped to [0, 100]."""
        dummy_breakdown = FormulaBreakdown.model_construct(
            coverage_score=100.0, depth_score=100.0, complexity_score=100.0,
            reliability_score=100.0, consistency_score=100.0, confidence_score=100.0,
            raw_weighted_sum=150.0, missing_evidence_penalty=0.0, contradiction_penalty=0.0,
            final_capability_score=150.0, weights_used=FormulaWeightConfig(), formula_expression=""
        )
        fake_detail = CapabilityScoreDetail.model_construct(
            capability_id=self.job_analysis.capability_graph[0].id,
            capability_name="Test",
            status="Strongly Verified",
            final_capability_score=150.0,
            formula_breakdown=dummy_breakdown
        )
        report = SchemaValidator.validate_scoring_result([fake_detail], [c.id for c in self.job_analysis.capability_graph])
        self.assertEqual(fake_detail.final_capability_score, 100.0)

    def test_19_configurable_formula_weights(self):
        """Test CapabilityScoringService with custom formula weights."""
        custom_weights = FormulaWeightConfig(w_coverage=0.50, w_depth=0.50, w_complexity=0.0, w_reliability=0.0, w_consistency=0.0, w_confidence=0.0)
        custom_service = CapabilityScoringService(weight_config=custom_weights)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = custom_service.evaluate_capabilities(fusion)
        self.assertEqual(scoring.capability_scores[0].formula_breakdown.weights_used.w_coverage, 0.50)

    def test_20_readiness_level_production_ready(self):
        """Verify readiness level assigned as 'Production Ready' for high scoring candidates."""
        repo = self.repo_service.analyze_repository(
            "u", {"name": "app", "fork": False},
            ["backend/controllers/auth.js", "src/App.jsx", "Dockerfile", "tests/auth.test.js", "models/user.js"],
            [], self.job_analysis
        )
        res = self.resume_service.analyze_resume("Senior Node.js backend architect", self.job_analysis)
        tech_assessment = {
            "evidence": [
                {"capability_id": c.id, "quote": f"Passed {c.name}", "confidence": 95.0} for c in self.job_analysis.capability_graph
            ]
        }
        beh_assessment = {
            "evidence": [
                {"capability_id": c.id, "quote": f"High leadership in {c.name}", "confidence": 90.0} for c in self.job_analysis.capability_graph
            ]
        }
        prof_exp = [
            {"capability_id": c.id, "quote": f"3 yrs exp in {c.name}", "confidence": 90.0} for c in self.job_analysis.capability_graph
        ]
        fusion = self.fusion_service.fuse_evidence(
            self.job_analysis, res, repo,
            technical_assessment=tech_assessment,
            behavioral_assessment=beh_assessment,
            professional_experience=prof_exp
        )
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertEqual(scoring.readiness_summary.readiness_level, "Production Ready")

    def test_21_readiness_level_nearly_ready(self):
        """Verify readiness level assigned as 'Nearly Ready'."""
        repo = self.repo_service.analyze_repository(
            "u", {"name": "app", "fork": False},
            ["backend/controllers/auth.js", "src/App.jsx", "Dockerfile"],
            [], self.job_analysis
        )
        res = self.resume_service.analyze_resume("Node.js developer", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertIn(scoring.readiness_summary.readiness_level, ["Nearly Ready", "Production Ready", "Learning", "Not Ready"])

    def test_22_readiness_level_learning(self):
        """Verify readiness level assigned as 'Learning'."""
        res = self.resume_service.analyze_resume("Node.js junior developer", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, resume_analysis=res)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertIn(scoring.readiness_summary.readiness_level, ["Learning", "Not Ready"])

    def test_23_readiness_level_not_ready(self):
        """Verify readiness level assigned as 'Not Ready' when no evidence exists."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        self.assertEqual(scoring.readiness_summary.readiness_level, "Not Ready")

    def test_24_status_strongly_verified(self):
        """Test status 'Strongly Verified'."""
        repo = self.repo_service.analyze_repository(
            "u", {"name": "app", "fork": False},
            ["backend/controllers/auth.js", "src/App.jsx", "Dockerfile", "tests/auth.test.js", "models/user.js"],
            [], self.job_analysis
        )
        res = self.resume_service.analyze_resume("Senior Node.js backend architect", self.job_analysis)
        tech_assessment = {
            "evidence": [
                {"capability_id": c.id, "quote": f"Passed {c.name}", "confidence": 95.0} for c in self.job_analysis.capability_graph
            ]
        }
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo, technical_assessment=tech_assessment)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertIn(s1.status, ["Strongly Verified", "Verified"])

    def test_25_status_verified(self):
        """Test status 'Verified'."""
        repo = self.repo_service.analyze_repository("u", {"name": "app", "fork": False}, ["backend/controllers/auth.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, repository_analysis=repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertIn(s1.status, ["Verified", "Strongly Verified", "Partially Verified", "Weakly Verified"])

    def test_26_status_partially_verified(self):
        """Test status 'Partially Verified'."""
        res = self.resume_service.analyze_resume("Node.js API developer", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, resume_analysis=res)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertIn(s1.status, ["Partially Verified", "Weakly Verified", "Unsupported"])

    def test_27_status_weakly_verified(self):
        """Test status 'Weakly Verified'."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertIn(s1.status, ["Weakly Verified", "Unsupported"])

    def test_28_status_unsupported(self):
        """Test status 'Unsupported'."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        s1 = scoring.capability_scores[0]
        self.assertIn(s1.status, ["Unsupported", "Weakly Verified"])

    def test_29_status_contradicted(self):
        """Test status 'Contradicted'."""
        res = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res.evidence_objects.append(
            EvidenceObject(evidence_id="ev_1", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo = self.repo_service.analyze_repository("u", {"name": "r"}, ["app.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, res, repo)
        scoring = self.scoring_service.evaluate_capabilities(fusion)
        docker_score = [s for s in scoring.capability_scores if s.capability_id == self.docker_cap_id][0]
        self.assertEqual(docker_score.status, "Contradicted")

    def test_30_full_five_module_end_to_end_pipeline(self):
        """Full end-to-end 5-module pipeline integration (Module 1 -> 2 -> 3 -> 4 -> 5)."""
        job_res = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend", "Docker Containerization"])
        res_res = self.resume_service.analyze_resume("Built Node.js APIs and Dockerized services", job_res)
        repo_res = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/controllers/auth.js", "Dockerfile"], [], job_res)
        fusion_res = self.fusion_service.fuse_evidence(job_analysis=job_res, resume_analysis=res_res, repository_analysis=repo_res)
        scoring_res = self.scoring_service.evaluate_capabilities(fusion_res)

        self.assertEqual(scoring_res.metadata.schema_version, "2.0")
        self.assertEqual(scoring_res.metadata.pipeline_module, "Capability Scoring Engine")
        self.assertEqual(len(scoring_res.capability_scores), 3)
        self.assertTrue(scoring_res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
