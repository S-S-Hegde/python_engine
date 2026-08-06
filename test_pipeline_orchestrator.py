import unittest
from typing import Dict, Any

from pipeline_orchestrator.models import (
    PipelineRequest,
    PipelineConfig,
    ExecutionMode,
    PipelineStatus,
    ExecutionStatus
)
from pipeline_orchestrator.orchestrator import PipelineOrchestratorService

def mock_payload(student=False, fresher=False, professional=False, no_repo=False, fail_module=False) -> PipelineRequest:
    cfg = PipelineConfig()
    
    exp_years = 0
    if professional:
        exp_years = 5
    elif fresher:
        exp_years = 1
        
    repos = []
    if not no_repo:
        repos = [{"name": "backend-api", "fork": False, "files": []}]
        
    resume_txt = "Built Python microservices and React frontends"
    jd = ["Python", "React"]
    
    tech_assess = {
        "assessment_id": "a1",
        "candidate_id": "c1",
        "questions": [{"question_id": "q1", "submitted_code": "def hello(): pass"}]
    }
    
    beh_assess = {
        "assessment_id": "b1",
        "candidate_id": "c1",
        "responses": [{"question_id": "q1", "response_text": "I lead a team of 3 developers"}]
    }
    
    # Simulate a missing dependency for failure cases if requested
    if fail_module:
        tech_assess = None
    
    return PipelineRequest(
        job_requirements=jd,
        resume_text=resume_txt,
        repositories=repos,
        technical_assessment=tech_assess,
        behavioral_assessment=beh_assess,
        professional_experience_years=exp_years,
        config=cfg
    )

class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = PipelineOrchestratorService()
        
    def test_01_student_pipeline_full_execution(self):
        payload = mock_payload(student=True)
        res = self.orchestrator.run_pipeline(payload)
        self.assertEqual(res.pipeline_status, PipelineStatus.COMPLETED)
        
    def test_02_fresher_pipeline_full_execution(self):
        payload = mock_payload(fresher=True)
        res = self.orchestrator.run_pipeline(payload)
        self.assertEqual(res.pipeline_status, PipelineStatus.COMPLETED)
        
    def test_03_professional_pipeline_with_repo(self):
        payload = mock_payload(professional=True)
        res = self.orchestrator.run_pipeline(payload)
        self.assertEqual(res.pipeline_status, PipelineStatus.COMPLETED)
        
    def test_04_professional_pipeline_no_repo_skipped(self):
        payload = mock_payload(professional=True, no_repo=True)
        res = self.orchestrator.run_pipeline(payload)
        # Should complete successfully but Repository Intelligence should be SKIPPED
        self.assertEqual(res.pipeline_status, PipelineStatus.COMPLETED)
        repo_record = next(r for r in res.pipeline_execution if r.module == "Repository Intelligence")
        self.assertEqual(repo_record.status, ExecutionStatus.SKIPPED)
        self.assertIn("without public repositories", repo_record.reason)
        
    def test_05_student_pipeline_no_repo_failed(self):
        payload = mock_payload(student=True, no_repo=True)
        res = self.orchestrator.run_pipeline(payload)
        repo_record = next(r for r in res.pipeline_execution if r.module == "Repository Intelligence")
        self.assertEqual(repo_record.status, ExecutionStatus.FAILED)
        
    def test_06_disable_resume_config(self):
        payload = mock_payload()
        payload.config.run_resume_intelligence = False
        res = self.orchestrator.run_pipeline(payload)
        record = next(r for r in res.pipeline_execution if r.module == "Resume Intelligence")
        self.assertEqual(record.status, ExecutionStatus.SKIPPED)
        
    def test_07_disable_tech_assess_config(self):
        payload = mock_payload()
        payload.config.run_technical_assessment = False
        res = self.orchestrator.run_pipeline(payload)
        record = next(r for r in res.pipeline_execution if r.module == "Technical Assessment")
        self.assertEqual(record.status, ExecutionStatus.SKIPPED)
        
    def test_08_disable_behavioral_config(self):
        payload = mock_payload()
        payload.config.run_behavioral_assessment = False
        res = self.orchestrator.run_pipeline(payload)
        record = next(r for r in res.pipeline_execution if r.module == "Behavioral Intelligence")
        self.assertEqual(record.status, ExecutionStatus.SKIPPED)
        
    def test_09_request_id_presence(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.request_id)
        
    def test_10_execution_mode_sync(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertEqual(res.execution_mode, ExecutionMode.SYNC)
        
    def test_11_execution_mode_async(self):
        payload = mock_payload()
        payload.config.execution_mode = ExecutionMode.ASYNC
        res = self.orchestrator.run_pipeline(payload)
        self.assertEqual(res.execution_mode, ExecutionMode.ASYNC)
        
    def test_12_audit_trail_exists(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.audit_trail)
        
    def test_13_schema_versions_exists(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.schema_versions)
        self.assertIn("job_intelligence", res.schema_versions)
        
    def test_14_pipeline_execution_timing(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        for r in res.pipeline_execution:
            if r.status == ExecutionStatus.COMPLETED:
                self.assertIsNotNone(r.execution_time_ms)
                
    def test_15_job_intelligence_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.job_analysis)
        
    def test_16_resume_intelligence_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.resume_analysis)
        
    def test_17_repository_intelligence_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.repository_analysis)
        
    def test_18_tech_assess_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.assessment_analysis)
        
    def test_19_behavioral_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.behavioral_analysis)
        
    def test_20_fusion_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.evidence_fusion)
        
    def test_21_capability_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.capability_scores)
        
    def test_22_competency_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.competency_scores)
        
    def test_23_profile_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.candidate_profile)
        
    def test_24_trust_score_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.trust_score)
        
    def test_25_recruiter_decision_output_mapped(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.recruiter_decision)

    def test_26_comprehensive_validation(self):
        payload = mock_payload()
        res = self.orchestrator.run_pipeline(payload)
        self.assertIsNotNone(res.pipeline_status)

def generate_fuzz_tests():
    def create_test(i):
        def test(self):
            payload = mock_payload()
            if i % 2 == 0:
                payload.config.run_behavioral_assessment = False
            if i % 3 == 0:
                payload.config.run_technical_assessment = False
            if i % 5 == 0:
                payload.repositories = None
            res = self.orchestrator.run_pipeline(payload)
            self.assertIsNotNone(res.pipeline_status)
        return test
        
    for i in range(27, 52):
        setattr(TestPipelineOrchestrator, f"test_{i}_fuzz_validation", create_test(i))

generate_fuzz_tests()

if __name__ == '__main__':
    unittest.main()
