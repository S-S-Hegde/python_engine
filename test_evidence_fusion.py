import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from resume_intelligence.models import EvidenceObject
from repository_intelligence import RepositoryIntelligenceService
from evidence_fusion import (
    EvidenceFusionService,
    SchemaValidator,
    UnifiedCapabilityProfile,
    ContradictionItem
)

class TestModule4EvidenceFusion(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        self.fusion_service = EvidenceFusionService()

        # Target job requirement blueprint
        self.job_analysis = self.job_service.analyze_job(raw_requirements=[
            "Node.js REST API Architecture",
            "React State Management",
            "Docker Containerization",
            "Automated Testing Suite",
            "MongoDB Schema Indexing"
        ])
        self.docker_cap_id = [c.id for c in self.job_analysis.capability_graph if "docker" in c.name.lower()][0]

    def test_01_resume_only_fusion(self):
        """Test evidence fusion with Resume evidence only."""
        res_analysis = self.resume_service.analyze_resume(
            "Designed Node.js REST APIs and React frontend.",
            self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis
        )
        self.assertEqual(fusion_result.metadata.schema_version, "2.0")
        self.assertEqual(len(fusion_result.capability_profiles), 5)
        self.assertGreaterEqual(fusion_result.confidence_summary.average_merged_confidence, 0.0)

    def test_02_repository_only_fusion(self):
        """Test evidence fusion with Repository evidence only."""
        repo_analysis = self.repo_service.analyze_repository(
            github_username="devuser",
            repo_data={"name": "repo1", "fork": False},
            tree_paths=["backend/controllers/authController.js", "Dockerfile"],
            commits=[],
            job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            repository_analysis=repo_analysis
        )
        self.assertEqual(fusion_result.metadata.schema_version, "2.0")
        self.assertGreater(fusion_result.reliability_summary.overall_reliability_score, 0.0)

    def test_03_resume_plus_repository_fusion(self):
        """Test fusion combining Resume and Repository evidence."""
        res_analysis = self.resume_service.analyze_resume(
            "Built Node.js APIs and Dockerized services.",
            self.job_analysis
        )
        repo_analysis = self.repo_service.analyze_repository(
            github_username="devuser",
            repo_data={"name": "repo1", "fork": False},
            tree_paths=["backend/controllers/authController.js", "Dockerfile"],
            commits=[],
            job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis,
            repository_analysis=repo_analysis
        )
        self.assertGreater(fusion_result.confidence_summary.verified_capabilities_count, 0)

    def test_04_technical_assessment_evidence_fusion(self):
        """Test fusing technical assessment evidence."""
        tech_assessment = {
            "evidence": [
                {
                    "capability_id": self.job_analysis.capability_graph[0].id,
                    "quote": "Score 95% on REST API Architecture scenario.",
                    "confidence": 95.0
                }
            ]
        }
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            technical_assessment=tech_assessment
        )
        p1 = fusion_result.capability_profiles[0]
        self.assertEqual(len(p1.assessment_evidence), 1)

    def test_05_behavioral_assessment_evidence_fusion(self):
        """Test fusing behavioral assessment evidence."""
        beh_assessment = {
            "evidence": [
                {
                    "capability_id": self.job_analysis.capability_graph[0].id,
                    "quote": "Demonstrated high ownership and architectural leadership.",
                    "confidence": 80.0
                }
            ]
        }
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            behavioral_assessment=beh_assessment
        )
        p1 = fusion_result.capability_profiles[0]
        self.assertEqual(len(p1.behavioral_evidence), 1)

    def test_06_professional_experience_fusion(self):
        """Test fusing professional experience records."""
        prof_exp = [
            {
                "capability_id": self.job_analysis.capability_graph[0].id,
                "quote": "Senior Backend Engineer at Tech Corp (2 years).",
                "confidence": 85.0
            }
        ]
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            professional_experience=prof_exp
        )
        p1 = fusion_result.capability_profiles[0]
        self.assertEqual(len(p1.professional_evidence), 1)

    def test_07_all_five_sources_combined(self):
        """Test evidence fusion combining all 5 sources."""
        res_analysis = self.resume_service.analyze_resume("Node.js dev", self.job_analysis)
        repo_analysis = self.repo_service.analyze_repository(
            github_username="u", repo_data={"name": "r"}, tree_paths=["app.js"], commits=[], job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis,
            repository_analysis=repo_analysis,
            technical_assessment={"evidence": [{"capability_id": self.job_analysis.capability_graph[0].id, "quote": "Exam pass"}]},
            behavioral_assessment={"evidence": [{"capability_id": self.job_analysis.capability_graph[0].id, "quote": "Leadership score"}]},
            professional_experience=[{"capability_id": self.job_analysis.capability_graph[0].id, "quote": "3 yrs exp"}]
        )
        p1 = fusion_result.capability_profiles[0]
        self.assertEqual(len(p1.resume_evidence), 1)
        self.assertEqual(len(p1.repository_evidence), 1)

    def test_08_conflicting_evidence_detection(self):
        """Test contradiction detection when Resume claims Docker but Repo has no Docker."""
        res_analysis = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res_analysis.evidence_objects.append(
            EvidenceObject(evidence_id="ev_r_001", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo_analysis = self.repo_service.analyze_repository(
            github_username="u", repo_data={"name": "r"}, tree_paths=["app.js"], commits=[], job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis,
            repository_analysis=repo_analysis
        )
        self.assertGreater(fusion_result.contradiction_report.total_contradictions_found, 0)

    def test_09_conflicting_evidence_penalty_application(self):
        """Verify contradiction penalty reduces merged confidence without deleting evidence."""
        res_analysis = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res_analysis.evidence_objects.append(
            EvidenceObject(evidence_id="ev_r_001", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo_analysis = self.repo_service.analyze_repository(
            github_username="u", repo_data={"name": "r"}, tree_paths=["app.js"], commits=[], job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis,
            repository_analysis=repo_analysis
        )
        docker_profile = [p for p in fusion_result.capability_profiles if p.capability_id == self.docker_cap_id][0]
        self.assertGreater(len(docker_profile.resume_evidence), 0)  # Evidence NOT deleted!
        self.assertGreater(len(docker_profile.contradictions), 0)

    def test_10_unverified_metric_contradiction(self):
        """Test flagging of unverified metric claims."""
        res_analysis = self.resume_service.analyze_resume("Achieved 40% latency reduction", self.job_analysis)
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis
        )
        self.assertGreater(fusion_result.contradiction_report.total_contradictions_found, 0)

    def test_11_single_day_dump_originality_penalty(self):
        """Test penalty applied for single-day dump repository."""
        commits = [{"commit": {"author": {"date": "2026-08-01T10:00:00Z"}, "message": f"Dump {i}"}} for i in range(20)]
        repo_analysis = self.repo_service.analyze_repository(
            github_username="u", repo_data={"name": "r"}, tree_paths=["backend/controllers/authController.js"], commits=commits, job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            repository_analysis=repo_analysis
        )
        self.assertGreater(fusion_result.contradiction_report.total_contradictions_found, 0)

    def test_12_forked_repo_originality_penalty(self):
        """Test penalty applied for forked repository."""
        repo_analysis = self.repo_service.analyze_repository(
            github_username="u", repo_data={"name": "r", "fork": True}, tree_paths=["backend/controllers/authController.js"], commits=[], job_analysis=self.job_analysis
        )
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            repository_analysis=repo_analysis
        )
        self.assertGreater(fusion_result.contradiction_report.total_contradictions_found, 0)

    def test_13_duplicate_evidence_deduplication(self):
        """Verify identical evidence quotes are deduplicated."""
        res_analysis = self.resume_service.analyze_resume("Node.js API Node.js API", self.job_analysis)
        fusion_result = self.fusion_service.fuse_evidence(
            job_analysis=self.job_analysis,
            resume_analysis=res_analysis
        )
        p1 = fusion_result.capability_profiles[0]
        quotes = [e.get("quote") for e in p1.resume_evidence]
        self.assertEqual(len(quotes), len(set(quotes)))

    def test_14_missing_evidence_detection(self):
        """Verify missing evidence report identifies gaps across all capabilities."""
        fusion_result = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        self.assertEqual(fusion_result.missing_evidence_report.total_missing_gaps, 5)

    def test_15_empty_evidence_handling(self):
        """Verify handling when all input analyses are empty."""
        fusion_result = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        self.assertEqual(len(fusion_result.capability_profiles), 5)
        self.assertEqual(fusion_result.confidence_summary.verified_capabilities_count, 0)

    def test_16_multiple_repositories_fusion(self):
        """Test evidence fusion when candidate has multiple repositories."""
        repo1 = self.repo_service.analyze_repository("u", {"name": "r1"}, ["backend/controllers/a.js"], [], self.job_analysis)
        fusion_result = self.fusion_service.fuse_evidence(self.job_analysis, repository_analysis=repo1)
        self.assertEqual(fusion_result.metadata.schema_version, "2.0")

    def test_17_multiple_capabilities_fusion(self):
        """Verify fusion correctly handles multi-capability graphs."""
        fusion_result = self.fusion_service.fuse_evidence(job_analysis=self.job_analysis)
        self.assertEqual(len(fusion_result.capability_profiles), 5)

    def test_18_large_candidate_profile_processing(self):
        """Test performance and correctness on large 10-requirement job blueprint."""
        large_job = self.job_service.analyze_job(raw_requirements=[f"Req {i}" for i in range(10)])
        fusion_result = self.fusion_service.fuse_evidence(job_analysis=large_job)
        self.assertEqual(len(fusion_result.capability_profiles), 10)

    def test_19_invalid_capability_id_warnings(self):
        """Test SchemaValidator flags capability IDs not present in Module 1."""
        p = UnifiedCapabilityProfile(capability_id="invalid_cap_id", capability_name="Invalid")
        report = SchemaValidator.validate_fusion_result([p], [c.id for c in self.job_analysis.capability_graph])
        self.assertFalse(report.is_valid)
        self.assertGreater(len(report.warnings), 0)

    def test_20_out_of_bounds_confidence_clamping(self):
        """Verify merged_confidence out-of-bounds is clamped to [0, 100]."""
        p = UnifiedCapabilityProfile.model_construct(capability_id=self.job_analysis.capability_graph[0].id, merged_confidence=150.0, reliability=80.0)
        report = SchemaValidator.validate_fusion_result([p], [c.id for c in self.job_analysis.capability_graph])
        self.assertEqual(p.merged_confidence, 100.0)

    def test_21_out_of_bounds_reliability_clamping(self):
        """Verify reliability out-of-bounds is clamped to [0, 100]."""
        p = UnifiedCapabilityProfile.model_construct(capability_id=self.job_analysis.capability_graph[0].id, merged_confidence=50.0, reliability=-20.0)
        report = SchemaValidator.validate_fusion_result([p], [c.id for c in self.job_analysis.capability_graph])
        self.assertEqual(p.reliability, 0.0)

    def test_22_configurable_reliability_weights(self):
        """Test EvidenceFusionService with custom reliability weights."""
        custom_svc = EvidenceFusionService(custom_reliability_weights={"repository": 0.80, "resume": 0.20})
        res = custom_svc.fuse_evidence(job_analysis=self.job_analysis)
        self.assertEqual(res.reliability_summary.repository_weight, 0.80)

    def test_23_verification_status_verified(self):
        """Verify status assigned as 'Verified' when high repo confidence is present."""
        repo_analysis = self.repo_service.analyze_repository("u", {"name": "r"}, ["backend/controllers/a.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, repository_analysis=repo_analysis)
        p1 = fusion.capability_profiles[0]
        self.assertEqual(p1.status, "Verified")

    def test_24_verification_status_partially_verified(self):
        """Verify status assigned as 'Partially Verified' for resume claims without repo."""
        res_analysis = self.resume_service.analyze_resume("Node.js developer", self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, resume_analysis=res_analysis)
        p1 = fusion.capability_profiles[0]
        self.assertEqual(p1.status, "Partially Verified")

    def test_25_verification_status_contradicted(self):
        """Verify status assigned as 'Contradicted' when critical flags exist."""
        res_analysis = self.resume_service.analyze_resume("Docker master", self.job_analysis)
        res_analysis.evidence_objects.append(
            EvidenceObject(evidence_id="ev_r_001", capability_id=self.docker_cap_id, quote="Mastered Docker Containerization", confidence=85.0)
        )
        repo_analysis = self.repo_service.analyze_repository("u", {"name": "r"}, ["app.js"], [], self.job_analysis)
        fusion = self.fusion_service.fuse_evidence(self.job_analysis, resume_analysis=res_analysis, repository_analysis=repo_analysis)
        docker_profile = [p for p in fusion.capability_profiles if p.capability_id == self.docker_cap_id][0]
        self.assertEqual(docker_profile.status, "Contradicted")

    def test_26_verification_status_unverified(self):
        """Verify status assigned as 'Unverified' when no evidence is supplied."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        p1 = fusion.capability_profiles[0]
        self.assertEqual(p1.status, "Unverified")

    def test_27_reliability_summary_calculation(self):
        """Verify overall reliability score calculation in summary."""
        fusion = self.fusion_service.fuse_evidence(self.job_analysis)
        self.assertGreaterEqual(fusion.reliability_summary.overall_reliability_score, 0.0)

    def test_28_performance_timing(self):
        """Verify Evidence Fusion executes in under 50ms."""
        start = time.perf_counter()
        self.fusion_service.fuse_evidence(self.job_analysis)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 50.0)

    def test_29_schema_validator_broken_mapping_detection(self):
        """Test SchemaValidator detects missing capability references."""
        p1 = UnifiedCapabilityProfile(capability_id=self.job_analysis.capability_graph[0].id)
        p2 = UnifiedCapabilityProfile(capability_id=self.job_analysis.capability_graph[0].id)
        report = SchemaValidator.validate_fusion_result([p1, p2], [c.id for c in self.job_analysis.capability_graph])
        self.assertFalse(report.is_valid)

    def test_30_full_end_to_end_pipeline_execution(self):
        """Full multi-module pipeline integration (Module 1 -> Module 2 -> Module 3 -> Module 4)."""
        job_res = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend", "Docker Containerization"])
        res_res = self.resume_service.analyze_resume("Built Node.js APIs and Dockerized services", job_res)
        repo_res = self.repo_service.analyze_repository("devuser", {"name": "app", "fork": False}, ["backend/controllers/auth.js", "Dockerfile"], [], job_res)

        fusion_res = self.fusion_service.fuse_evidence(
            job_analysis=job_res,
            resume_analysis=res_res,
            repository_analysis=repo_res
        )

        self.assertEqual(fusion_res.metadata.schema_version, "2.0")
        self.assertEqual(fusion_res.metadata.pipeline_module, "Evidence Fusion Engine")
        self.assertEqual(len(fusion_res.capability_profiles), 3)
        self.assertTrue(fusion_res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
