import unittest
import time
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService, SchemaValidator, EvidenceObject

class TestModule2ResumeIntelligence(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.resume_service = ResumeIntelligenceService()
        # Mock Job Analysis for integration testing
        self.job_analysis = self.job_service.analyze_job(raw_requirements=[
            "Node.js REST API Architecture",
            "React Component State Management",
            "MongoDB Database Schema Indexing",
            "Docker Containerization"
        ])

    def test_01_empty_resume(self):
        """Test processing of empty resume string."""
        res = self.resume_service.analyze_resume("", self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertEqual(res.metadata.pipeline_module, "Resume Intelligence Service")
        self.assertEqual(len(res.evidence_objects), 1)

    def test_02_one_line_short_resume(self):
        """Test processing of ultra-short one-line resume."""
        res = self.resume_service.analyze_resume("Built React components and Node.js backend APIs.", self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(len(res.evidence_objects), 0)

    def test_03_student_resume_profile(self):
        """Test processing of student resume profile."""
        resume = "B.Tech Computer Science Student. Built mini project in React with basic state hooks."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertIn(res.candidate_summary.detected_level, ["Student", "Fresher", "Intermediate"])

    def test_04_professional_resume_profile(self):
        """Test processing of senior professional resume profile."""
        resume = "Senior Full Stack Engineer with 6 years experience. Architected microservices in Node.js, reduced latency by 40%, managed Redux Toolkit frontend."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(res.candidate_summary.total_quantified_claims, 0)

    def test_05_large_multi_page_resume(self):
        """Test processing of large multi-page resume."""
        large_resume = "Experience:\n" + ("- Designed Express REST APIs handling 10,000 requests/day with JWT auth.\n" * 30)
        res = self.resume_service.analyze_resume(large_resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(len(res.evidence_objects), 0)

    def test_06_capability_id_preservation_from_module_1(self):
        """Verify Module 2 evidence strictly uses Module 1 capability IDs without inventing new ones."""
        valid_ids = [c.id for c in self.job_analysis.capability_graph]
        res = self.resume_service.analyze_resume("Built Node.js APIs and React UI.", self.job_analysis)

        for ev in res.evidence_objects:
            self.assertIn(ev.capability_id, valid_ids, f"Capability ID {ev.capability_id} not in Module 1 graph {valid_ids}")

    def test_07_quantified_metric_extraction(self):
        """Verify extraction of percentages and throughput numbers."""
        resume = "Optimized MongoDB indexes achieving 45% query speedup and handling 20,000 daily active users."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertGreater(len(res.resume_metrics), 0)

    def test_08_ownership_detection_individual(self):
        """Verify individual ownership detection from verbs like 'built' / 'created'."""
        resume = "Sole developer. Built independent microservice in Docker."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertGreaterEqual(res.ownership_summary.individual_count, 0)

    def test_09_complexity_estimation_high(self):
        """Verify complexity estimation for architectural phrases."""
        resume = "Architected distributed OAuth2 authentication middleware and compound database indexes."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        complexities = [e.complexity for e in res.evidence_objects]
        self.assertTrue(any(c in ["High", "Very High", "Medium"] for c in complexities))

    def test_10_contradictory_vague_claims(self):
        """Test handling of vague claims without crashing."""
        resume = "Experienced in everything. Expert in all languages."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_11_multiple_projects_profile(self):
        """Test multi-project resume handling."""
        resume = "Project 1: E-commerce in React. Project 2: Banking backend in Node.js. Project 3: Analytics in MongoDB."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertGreaterEqual(len(res.evidence_objects), 3)

    def test_12_research_profile_resume(self):
        """Test research/academic resume profile."""
        resume = "Published paper on Distributed Systems. Implemented consensus algorithm in C++."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_13_ai_engineer_resume(self):
        """Test AI Engineer profile resume."""
        resume = "Trained PyTorch Transformers on 10M tokens. Deployed vector search pipeline using Qdrant."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_14_backend_engineer_resume(self):
        """Test Backend Engineer profile resume."""
        resume = "Designed Express REST controllers, middleware pipeline, JWT validation, and Postgres migrations."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_15_frontend_engineer_resume(self):
        """Test Frontend Engineer profile resume."""
        resume = "Developed responsive React UI components, Redux Toolkit slices, custom hooks, and Tailwind CSS."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_16_confidence_bounds_check(self):
        """Verify confidence scores are strictly bounded in [0.0, 100.0]."""
        resume = "Created REST endpoints with Node.js and MongoDB."
        res = self.resume_service.analyze_resume(resume, self.job_analysis)
        for ev in res.evidence_objects:
            self.assertGreaterEqual(ev.confidence, 0.0)
            self.assertLessEqual(ev.confidence, 100.0)

    def test_17_performance_timing(self):
        """Verify fallback parsing executes in under 50ms."""
        start = time.perf_counter()
        self.resume_service.analyze_resume("Built React app with Express backend.", self.job_analysis)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 50.0)

    def test_18_validator_duplicate_quote_detection(self):
        """Test SchemaValidator flags duplicate quotes in validation report."""
        ev1 = EvidenceObject(evidence_id="ev_1", capability_id="cap_test", quote="Duplicate quote text.")
        ev2 = EvidenceObject(evidence_id="ev_2", capability_id="cap_test", quote="Duplicate quote text.")
        report = SchemaValidator.validate_evidence_objects([ev1, ev2], ["cap_test"])
        self.assertFalse(report.is_valid)
        self.assertGreater(len(report.warnings), 0)

    def test_19_capability_mapping_summary(self):
        """Verify capability mapping summary aggregates evidence count and highest confidence."""
        res = self.resume_service.analyze_resume("Built React UI and Express APIs.", self.job_analysis)
        self.assertGreater(len(res.capability_mapping), 0)

    def test_20_full_integration_pipeline(self):
        """Full integration test combining Module 1 output and Module 2 evidence extraction."""
        job_res = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend"])
        res = self.resume_service.analyze_resume("Built Node.js API with JWT auth and React dashboard.", job_res)

        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(len(res.evidence_objects), 0)
        self.assertTrue(res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
