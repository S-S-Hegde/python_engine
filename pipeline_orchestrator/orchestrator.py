import uuid
import datetime
import time
from typing import Dict, Any

from .models import (
    PipelineRequest,
    PipelineResponse,
    PipelineStatus,
    ExecutionMode
)
from .pipeline_runner import PipelineRunner
from .validators import SchemaValidator

# Import all frozen modules
from job_intelligence import JobIntelligenceService
from resume_intelligence import ResumeIntelligenceService
from repository_intelligence import RepositoryIntelligenceService
from technical_assessment import TechnicalAssessmentService, TechnicalAssessmentSubmission, SubmissionQuestionItem
from behavioral_intelligence import BehavioralIntelligenceService, BehavioralSubmissionPayload, BehavioralQuestionResponse
from evidence_fusion import EvidenceFusionService
from capability_scoring import CapabilityScoringService
from competency_intelligence import CompetencyIntelligenceService
from candidate_profile import CandidateProfileService
from trust_score_engine import TrustScoreService
from recruiter_decision import RecruiterDecisionService, RecruiterDecisionRequestPayload

class PipelineOrchestratorService:
    def __init__(self):
        self.validator = SchemaValidator()
        self.job_svc = JobIntelligenceService()
        self.resume_svc = ResumeIntelligenceService()
        self.repo_svc = RepositoryIntelligenceService()
        self.tech_svc = TechnicalAssessmentService()
        self.beh_svc = BehavioralIntelligenceService()
        self.fusion_svc = EvidenceFusionService()
        self.cap_svc = CapabilityScoringService()
        self.comp_svc = CompetencyIntelligenceService()
        self.prof_svc = CandidateProfileService()
        self.trust_svc = TrustScoreService()
        self.rec_svc = RecruiterDecisionService()

    def run_pipeline(self, request: PipelineRequest, request_id: str = None) -> PipelineResponse:
        start_time = time.time()
        req_id = request_id or str(uuid.uuid4())
        runner = PipelineRunner()
        cfg = request.config
        
        schema_versions = {}
        
        # 1. Job Intelligence (Mandatory)
        job_res = runner.execute_module(
            "Job Intelligence",
            skip_condition=False,
            skip_reason="",
            execution_func=lambda: self.job_svc.analyze_job(raw_requirements=request.job_requirements)
        )
        if job_res:
            schema_versions["job_intelligence"] = job_res.metadata.schema_version
        
        # 2. Resume Intelligence
        resume_res = runner.execute_module(
            "Resume Intelligence",
            skip_condition=not cfg.run_resume_intelligence or not request.resume_text,
            skip_reason="Resume analysis disabled or no resume provided",
            execution_func=lambda: self.resume_svc.analyze_resume(request.resume_text, job_res)
        )
        if resume_res:
            schema_versions["resume_intelligence"] = resume_res.metadata.schema_version

        # Evaluate candidate seniority dynamically
        is_professional = False
        if request.professional_experience_years is not None and request.professional_experience_years >= 2:
            is_professional = True
            
        no_repos_provided = not request.repositories or len(request.repositories) == 0
        skip_repo = not cfg.run_repository_intelligence or (is_professional and no_repos_provided)

        # 3. Repository Intelligence
        repo_res = runner.execute_module(
            "Repository Intelligence",
            skip_condition=skip_repo,
            skip_reason="Professional candidate without public repositories" if (is_professional and no_repos_provided) else "Repository analysis disabled or no repos provided",
            execution_func=lambda: self._run_repos(request.repositories, job_res)
        )
        if repo_res:
            schema_versions["repository_intelligence"] = repo_res.metadata.schema_version
            
        # 4. Technical Assessment
        tech_res = runner.execute_module(
            "Technical Assessment",
            skip_condition=not cfg.run_technical_assessment or not request.technical_assessment,
            skip_reason="Technical assessment disabled or not provided",
            execution_func=lambda: self._run_tech(request.technical_assessment, job_res)
        )
        if tech_res:
            schema_versions["technical_assessment"] = tech_res.metadata.schema_version
            
        # 5. Behavioral Assessment
        beh_res = runner.execute_module(
            "Behavioral Intelligence",
            skip_condition=not cfg.run_behavioral_assessment or not request.behavioral_assessment,
            skip_reason="Behavioral assessment disabled or not provided",
            execution_func=lambda: self._run_beh(request.behavioral_assessment, job_res)
        )
        if beh_res:
            schema_versions["behavioral_intelligence"] = beh_res.metadata.schema_version
            
        # 6. Evidence Fusion
        tech_dict = {"evidence": [ev.model_dump() for ev in tech_res.evidence_objects]} if tech_res else None
        beh_dict = {"evidence": [ev.model_dump() for ev in beh_res.evidence_objects]} if beh_res else None
        
        fusion_res = runner.execute_module(
            "Evidence Fusion",
            skip_condition=False,
            skip_reason="",
            execution_func=lambda: self.fusion_svc.fuse_evidence(
                job_analysis=job_res,
                resume_analysis=resume_res,
                repository_analysis=repo_res,
                technical_assessment=tech_dict,
                behavioral_assessment=beh_dict
            )
        )
        if fusion_res:
            schema_versions["evidence_fusion"] = fusion_res.metadata.schema_version
            
        # 7. Capability Scoring
        cap_res = runner.execute_module(
            "Capability Scoring",
            skip_condition=not fusion_res,
            skip_reason="Evidence fusion failed or skipped",
            execution_func=lambda: self.cap_svc.evaluate_capabilities(fusion_res)
        )
        if cap_res:
            schema_versions["capability_scoring"] = cap_res.metadata.schema_version
            
        # 8. Competency Intelligence
        comp_res = runner.execute_module(
            "Competency Intelligence",
            skip_condition=not cap_res,
            skip_reason="Capability scoring failed or skipped",
            execution_func=lambda: self.comp_svc.evaluate_competencies(cap_res)
        )
        if comp_res:
            schema_versions["competency_intelligence"] = comp_res.metadata.schema_version
            
        # 9. Candidate Profile
        prof_res = runner.execute_module(
            "Candidate Profile",
            skip_condition=not cfg.run_candidate_profile or not cap_res or not comp_res,
            skip_reason="Candidate profile disabled or dependencies missing",
            execution_func=lambda: self.prof_svc.generate_candidate_profile(cap_res, comp_res)
        )
        if prof_res:
            schema_versions["candidate_profile"] = prof_res.metadata.schema_version
            
        # 10. Trust Score Engine
        trust_res = runner.execute_module(
            "Trust Score Engine",
            skip_condition=not cfg.run_trust_score or not fusion_res,
            skip_reason="Trust score disabled or fusion missing",
            execution_func=lambda: self.trust_svc.generate_trust_score(
                job_analysis=job_res,
                resume_analysis=resume_res,
                repository_analysis=repo_res,
                technical_assessment=tech_res,
                behavioral_assessment=beh_res,
                evidence_fusion_result=fusion_res,
                capability_scoring_result=cap_res,
                competency_intelligence_result=comp_res,
                candidate_profile_result=prof_res
            )
        )
        if trust_res:
            schema_versions["trust_score_engine"] = trust_res.metadata.schema_version
            
        # 11. Recruiter Decision Engine
        rec_res = runner.execute_module(
            "Recruiter Decision Engine",
            skip_condition=not cfg.run_recruiter_decision or not trust_res,
            skip_reason="Recruiter decision disabled or trust score missing",
            execution_func=lambda: self._run_rec(trust_res, cap_res, comp_res, prof_res)
        )
        if rec_res:
            schema_versions["recruiter_decision"] = rec_res.metadata.get("schema_version", "2.0")

        # Determine pipeline status
        failed_count = sum(1 for r in runner.get_records() if r.status.value == "Failed")
        if failed_count == 0:
            status = PipelineStatus.COMPLETED
        elif failed_count < len(runner.get_records()):
            status = PipelineStatus.PARTIAL
        else:
            status = PipelineStatus.FAILED

        audit_trail = {
            "pipeline_version": "2.0",
            "decision_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "modules_executed": [r.module for r in runner.get_records()]
        }
        
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        response = PipelineResponse(
            request_id=req_id,
            execution_mode=cfg.execution_mode,
            pipeline_status=status,
            metadata={
                "schema_version": "2.0",
                "generated_at": audit_trail["decision_timestamp"],
                "processing_time_ms": processing_time_ms,
                "model": "pipeline-orchestrator-v2"
            },
            job_analysis=job_res.model_dump() if job_res else None,
            resume_analysis=resume_res.model_dump() if resume_res else None,
            repository_analysis=repo_res.model_dump() if repo_res else None,
            assessment_analysis=tech_res.model_dump() if tech_res else None,
            behavioral_analysis=beh_res.model_dump() if beh_res else None,
            evidence_fusion=fusion_res.model_dump() if fusion_res else None,
            capability_scores=cap_res.model_dump() if cap_res else None,
            competency_scores=comp_res.model_dump() if comp_res else None,
            candidate_profile=prof_res.model_dump() if prof_res else None,
            trust_score=trust_res.model_dump() if trust_res else None,
            recruiter_decision=rec_res.model_dump() if rec_res else None,
            pipeline_execution=runner.get_records(),
            audit_trail=audit_trail,
            schema_versions=schema_versions
        )
        
        # Validation guarantees
        is_valid = self.validator.validate_pipeline_response(response)
        if not is_valid:
            response.pipeline_status = PipelineStatus.FAILED
            
        return response

    def _run_repos(self, repos: List[Dict[str, Any]], job_res) -> Any:
        repo = repos[0]
        return self.repo_svc.analyze_repository(
            github_username="c1",
            repo_data=repo,
            tree_paths=repo.get("files", []),
            commits=[],
            job_analysis=job_res
        )

    def _run_tech(self, data: Dict[str, Any], job_res) -> Any:
        questions = []
        for q in data.get("questions", []):
            questions.append(SubmissionQuestionItem(
                question_id=q.get("question_id", "q1"),
                target_capability_id=job_res.capability_graph[0].id,
                submitted_code=q.get("submitted_code", ""),
                test_cases=q.get("test_cases", [])
            ))
        sub = TechnicalAssessmentSubmission(
            assessment_id=data.get("assessment_id", "a1"),
            candidate_id=data.get("candidate_id", "c1"),
            questions=questions
        )
        return self.tech_svc.analyze_assessment(sub, job_res)

    def _run_beh(self, data: Dict[str, Any], job_res) -> Any:
        responses = []
        for r in data.get("responses", []):
            responses.append(BehavioralQuestionResponse(
                question_id=r.get("question_id", "q1"),
                target_capability_id=job_res.capability_graph[0].id,
                response_text=r.get("response_text", "")
            ))
        sub = BehavioralSubmissionPayload(
            assessment_id=data.get("assessment_id", "a1"),
            candidate_id=data.get("candidate_id", "c1"),
            responses=responses
        )
        return self.beh_svc.analyze_behavior(sub, job_res)
        
    def _run_rec(self, trust_res, cap_res, comp_res, prof_res) -> Any:
        payload = RecruiterDecisionRequestPayload(
            trust_score_result=trust_res.model_dump(),
            capability_scoring_result=cap_res.model_dump(),
            competency_intelligence_result=comp_res.model_dump(),
            candidate_profile_result=prof_res.model_dump()
        )
        return self.rec_svc.generate_decision(payload)
