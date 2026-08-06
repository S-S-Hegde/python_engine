import unittest
import time
from job_intelligence import JobIntelligenceService
from repository_intelligence import (
    RepositoryIntelligenceService,
    SchemaValidator,
    RepositoryEvidenceObject,
    OriginalityReport,
    ArchitectureSummary,
    FrameworkSummary
)

class TestModule3RepositoryIntelligence(unittest.TestCase):
    def setUp(self):
        self.job_service = JobIntelligenceService()
        self.repo_service = RepositoryIntelligenceService()
        # Module 1 Job Analysis contract for integration mapping
        self.job_analysis = self.job_service.analyze_job(raw_requirements=[
            "Node.js REST API Architecture",
            "React Component State Management",
            "MongoDB Database Schema Indexing",
            "Docker Containerization",
            "Automated Testing Suite"
        ])

    def test_01_empty_repository(self):
        """Test processing of empty file tree and no commits."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "empty-repo", "fork": False},
            tree_paths=[],
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertEqual(res.metadata.pipeline_module, "Repository Intelligence Service")
        self.assertEqual(len(res.evidence_objects), 1)

    def test_02_small_repository(self):
        """Test small repository with index.js and package.json."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "small-repo", "fork": False},
            tree_paths=["package.json", "index.js", "README.md"],
            commits=[{"commit": {"author": {"date": "2026-08-01T10:00:00Z"}, "message": "Initial commit"}}],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(len(res.evidence_objects), 0)

    def test_03_large_repository(self):
        """Test large multi-folder repository tree."""
        large_tree = [f"src/components/Comp{i}.jsx" for i in range(20)] + [f"backend/controllers/ctrl{i}.js" for i in range(10)]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "large-repo", "fork": False},
            tree_paths=large_tree,
            commits=[{"commit": {"author": {"date": f"2026-08-0{i%9+1}T10:00:00Z"}, "message": f"Feature commit {i}"}} for i in range(25)],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(res.repository_summary.total_files_scanned, 20)

    def test_04_forked_repository_detection(self):
        """Verify forked repo flag and originality score penalty."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "forked-repo", "fork": True},
            tree_paths=["src/app.js"],
            commits=[{"commit": {"author": {"date": "2026-08-01T10:00:00Z"}, "message": "Fork commit"}}],
            job_analysis=self.job_analysis
        )
        self.assertTrue(res.originality_report.is_fork)
        self.assertEqual(res.originality_report.verdict, "Forked Repo")
        self.assertLess(res.originality_report.originality_score, 60.0)

    def test_05_single_day_dump_detection(self):
        """Verify bulk single-day commit dump detection."""
        commits = [{"commit": {"author": {"date": "2026-08-01T10:00:00Z"}, "message": f"Dump commit {i}"}} for i in range(20)]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "dump-repo", "fork": False},
            tree_paths=["src/index.js"],
            commits=commits,
            job_analysis=self.job_analysis
        )
        self.assertTrue(res.originality_report.is_single_day_dump)
        self.assertEqual(res.originality_report.verdict, "Single-Day Dump")

    def test_06_docker_devops_detection(self):
        """Verify Dockerfile and CI/CD workflow detection."""
        tree = ["Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "docker-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertTrue(res.architecture_summary.has_docker)
        self.assertTrue(res.architecture_summary.has_ci_cd)
        self.assertIn("Docker", res.framework_summary.devops_tools)

    def test_07_mern_stack_detection(self):
        """Verify MERN stack framework detection (MongoDB, Express, React, Node)."""
        tree = ["src/components/Header.jsx", "backend/controllers/authController.js", "backend/models/User.js", "package.json"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "mern-repo", "fork": False, "language": "JavaScript"},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertIn("React", res.framework_summary.detected_frameworks)
        self.assertIn("MongoDB", res.framework_summary.database_technologies)

    def test_08_java_spring_boot_detection(self):
        """Verify Java Spring Boot detection."""
        tree = ["src/main/java/com/app/Application.java", "src/main/java/com/app/controllers/UserController.java", "pom.xml"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "java-repo", "fork": False, "language": "Java"},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertIn("Spring Boot", res.framework_summary.detected_frameworks)

    def test_09_python_fastapi_detection(self):
        """Verify Python FastAPI detection."""
        tree = ["main.py", "api/routes.py", "requirements.txt"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "python-repo", "fork": False, "language": "Python"},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertIn("FastAPI", res.framework_summary.detected_frameworks)

    def test_10_ai_ml_project_detection(self):
        """Verify AI/ML project path detection."""
        tree = ["models/transformer.py", "train.py", "requirements.txt"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "ai-repo", "fork": False, "language": "Python"},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_11_missing_readme_repository(self):
        """Verify handling of repo with no README."""
        tree = ["src/app.js", "package.json"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "no-readme", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.metadata.schema_version, "2.0")

    def test_12_no_commits_repository(self):
        """Verify handling of repo with zero commit history."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "no-commits", "fork": False},
            tree_paths=["index.js"],
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.originality_report.commit_count, 0)

    def test_13_large_commit_history(self):
        """Verify handling of large organic commit history across multiple days."""
        commits = [{"commit": {"author": {"date": f"2026-08-{i%25+1:02d}T10:00:00Z"}, "message": f"Meaningful feature update commit {i}"}} for i in range(50)]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "large-history", "fork": False},
            tree_paths=["src/app.js"],
            commits=commits,
            job_analysis=self.job_analysis
        )
        self.assertFalse(res.originality_report.is_single_day_dump)
        self.assertGreater(res.originality_report.unique_commit_days, 5)

    def test_14_repository_with_tests(self):
        """Verify testing suite detection (has_tests: True)."""
        tree = ["src/app.js", "tests/app.test.js"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "test-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertTrue(res.architecture_summary.has_tests)

    def test_15_repository_without_tests(self):
        """Verify testing suite detection (has_tests: False)."""
        tree = ["src/app.js", "package.json"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "no-test-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertFalse(res.architecture_summary.has_tests)

    def test_16_architecture_pattern_mvc(self):
        """Verify MVC Architecture pattern detection."""
        tree = ["controllers/userController.js", "models/User.js", "views/index.html"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "mvc-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.architecture_summary.pattern, "MVC Architecture")

    def test_17_architecture_pattern_component_based(self):
        """Verify Component-Based Architecture pattern detection."""
        tree = ["src/components/Button.jsx", "src/components/Card.jsx"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "comp-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.architecture_summary.pattern, "Component-Based Architecture")

    def test_18_architecture_pattern_microservices(self):
        """Verify Microservices pattern detection."""
        tree = ["auth-service/package.json", "user-service/package.json", "docker-compose.yml"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "ms-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.architecture_summary.pattern, "Microservices")

    def test_19_architecture_pattern_clean_architecture(self):
        """Verify Clean Architecture pattern detection."""
        tree = ["core/entities.py", "domain/usecases.py", "adapters/repository.py"]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "clean-repo", "fork": False},
            tree_paths=tree,
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertEqual(res.architecture_summary.pattern, "Clean Architecture")

    def test_20_capability_id_preservation_from_module_1(self):
        """Verify Module 3 evidence strictly maps to Module 1 capability IDs."""
        valid_ids = [c.id for c in self.job_analysis.capability_graph]
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "app", "fork": False},
            tree_paths=["backend/controllers/authController.js"],
            commits=[],
            job_analysis=self.job_analysis
        )
        for ev in res.evidence_objects:
            self.assertIn(ev.capability_id, valid_ids)

    def test_21_confidence_bounds_check(self):
        """Verify confidence scores are strictly bounded in [0.0, 100.0]."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "app", "fork": False},
            tree_paths=["src/app.js"],
            commits=[],
            job_analysis=self.job_analysis
        )
        for ev in res.evidence_objects:
            self.assertGreaterEqual(ev.confidence, 0.0)
            self.assertLessEqual(ev.confidence, 100.0)

    def test_22_validator_duplicate_quote_detection(self):
        """Test SchemaValidator flags duplicate repository evidence quotes."""
        ev1 = RepositoryEvidenceObject(evidence_id="ev_1", capability_id="cap_test", repository="r", location="l1", quote="Duplicate code snippet.")
        ev2 = RepositoryEvidenceObject(evidence_id="ev_2", capability_id="cap_test", repository="r", location="l2", quote="Duplicate code snippet.")
        report = SchemaValidator.validate_repository_evidence([ev1, ev2], ["cap_test"])
        self.assertFalse(report.is_valid)
        self.assertGreater(len(report.warnings), 0)

    def test_23_performance_timing(self):
        """Verify static repository analysis executes in under 50ms."""
        start = time.perf_counter()
        self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "perf-repo", "fork": False},
            tree_paths=["src/index.js", "package.json", "Dockerfile"],
            commits=[],
            job_analysis=self.job_analysis
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 50.0)

    def test_24_originality_score_range(self):
        """Verify originality score is strictly bounded in [0.0, 100.0]."""
        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "orig-repo", "fork": False},
            tree_paths=["src/app.js"],
            commits=[],
            job_analysis=self.job_analysis
        )
        self.assertGreaterEqual(res.originality_report.originality_score, 0.0)
        self.assertLessEqual(res.originality_report.originality_score, 100.0)

    def test_25_full_pipeline_integration(self):
        """Full integration test combining Module 1 output and Module 3 repository evidence extraction."""
        job_res = self.job_service.analyze_job(raw_requirements=["Node.js REST API", "React Frontend", "Docker Containerization"])
        tree = ["backend/controllers/authController.js", "src/components/Header.jsx", "Dockerfile"]
        commits = [{"commit": {"author": {"date": f"2026-08-{i+1:02d}T10:00:00Z"}, "message": f"Feature update {i}"}} for i in range(10)]

        res = self.repo_service.analyze_repository(
            github_username="testuser",
            repo_data={"name": "full-stack-app", "fork": False, "language": "JavaScript"},
            tree_paths=tree,
            commits=commits,
            job_analysis=job_res
        )

        self.assertEqual(res.metadata.schema_version, "2.0")
        self.assertGreater(len(res.evidence_objects), 0)
        self.assertTrue(res.validation_report.is_valid)

if __name__ == "__main__":
    unittest.main()
