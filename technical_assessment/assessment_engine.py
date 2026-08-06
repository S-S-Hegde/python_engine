import time
from typing import List, Dict, Any, Optional
from job_intelligence.models import JobAnalysisResult
from .models import (
    TechnicalAssessmentSubmission,
    AssessmentAnalysisResult,
    CapabilityAssessmentScore,
    TechnicalAssessmentEvidenceObject,
    ExecutionResult,
    ComplexityDetail,
    AssessmentSummary,
    Metadata
)
from .question_engine import QuestionEngine
from .code_execution_analyzer import CodeExecutionAnalyzer
from .complexity_engine import ComplexityEngine
from .answer_evaluator import AnswerEvaluator
from .plagiarism_checker import PlagiarismChecker
from .confidence_engine import ConfidenceEngine, SchemaValidator

class AssessmentEngine:
    def __init__(self):
        pass

    def analyze_assessment(
        self,
        submission: TechnicalAssessmentSubmission,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> AssessmentAnalysisResult:
        start_time = time.perf_counter()

        # If job_analysis dict passed inside submission payload
        if not job_analysis and submission.job_analysis:
            try:
                job_analysis = JobAnalysisResult.model_validate(submission.job_analysis)
            except Exception:
                job_analysis = None

        execution_results: List[ExecutionResult] = []
        complexity_details: List[ComplexityDetail] = []
        evidence_objects: List[TechnicalAssessmentEvidenceObject] = []
        cap_scores_map: Dict[str, Dict[str, Any]] = {}

        for idx, q in enumerate(submission.questions):
            cap_id, cap_name = QuestionEngine.resolve_target_capability(q, job_analysis)

            exec_res = CodeExecutionAnalyzer.analyze_submission_execution(submission.assessment_id, q)
            execution_results.append(exec_res)

            comp_detail = ComplexityEngine.analyze_complexity(q.submitted_code)
            complexity_details.append(comp_detail)

            # Verification status based on pass rate
            if exec_res.pass_rate >= 90.0:
                status = "Strongly Verified"
            elif exec_res.pass_rate >= 70.0:
                status = "Verified"
            elif exec_res.pass_rate >= 40.0:
                status = "Partially Verified"
            elif exec_res.pass_rate > 0.0:
                status = "Weakly Verified"
            elif exec_res.compilation_error or exec_res.runtime_error:
                status = "Contradicted"
            else:
                status = "Unsupported"

            ev_id = f"ev_tech_{idx+1:04d}"
            quote = f"Submission for {q.question_id}: {exec_res.passed_tests_count}/{exec_res.total_tests_count} test cases passed ({exec_res.pass_rate}%)."

            evidence_objects.append(
                TechnicalAssessmentEvidenceObject(
                    evidence_id=ev_id,
                    capability_id=cap_id,
                    quote=quote,
                    source="technical_assessment",
                    confidence=exec_res.pass_rate,
                    ownership="Candidate Submission",
                    verified=exec_res.pass_rate >= 50.0,
                    status=status,
                    details={
                        "question_id": q.question_id,
                        "time_complexity": comp_detail.time_complexity,
                        "space_complexity": comp_detail.space_complexity,
                        "status": exec_res.status
                    }
                )
            )

            if cap_id not in cap_scores_map:
                cap_scores_map[cap_id] = {
                    "capability_name": cap_name,
                    "correctness_scores": [],
                    "complexity_scores": []
                }

            cap_scores_map[cap_id]["correctness_scores"].append(exec_res.pass_rate)
            cap_scores_map[cap_id]["complexity_scores"].append(comp_detail.complexity_score)

        code_quality = AnswerEvaluator.evaluate_code_quality(submission.questions)
        plagiarism_report = PlagiarismChecker.check_plagiarism(submission.questions)

        capability_scores: List[CapabilityAssessmentScore] = []
        for cid, info in cap_scores_map.items():
            avg_corr = sum(info["correctness_scores"]) / len(info["correctness_scores"])
            avg_comp = sum(info["complexity_scores"]) / len(info["complexity_scores"])
            qual = code_quality.overall_quality_score

            # Plagiarism penalty
            pen = 30.0 if plagiarism_report.is_plagiarized else 0.0

            raw_final = (avg_corr * 0.50) + (qual * 0.30) + (avg_comp * 0.20) - pen
            final_score = round(max(0.0, min(100.0, raw_final)), 2)

            capability_scores.append(
                CapabilityAssessmentScore(
                    capability_id=cid,
                    capability_name=info["capability_name"],
                    correctness_score=round(avg_corr, 2),
                    quality_score=round(qual, 2),
                    complexity_score=round(avg_comp, 2),
                    final_capability_score=final_score
                )
            )

        passed_q_count = sum(1 for r in execution_results if r.pass_rate == 100.0)
        overall_pass = round((passed_q_count / max(1, len(submission.questions))) * 100.0, 2) if submission.questions else 0.0

        overall_score = round(sum(c.final_capability_score for c in capability_scores) / max(1, len(capability_scores)), 2) if capability_scores else 0.0

        if plagiarism_report.is_plagiarized:
            rec = "Flagged for Plagiarism / Copy-Paste Anomaly. Manual review required."
        elif overall_score >= 80.0:
            rec = "Strong Technical Performance. Candidate demonstrated production-grade coding skills."
        elif overall_score >= 50.0:
            rec = "Satisfactory Technical Performance. Moderate proficiency demonstrated."
        else:
            rec = "Unsatisfactory Technical Performance. Low correctness or code quality."

        summary = AssessmentSummary(
            overall_score=overall_score,
            total_questions=len(submission.questions),
            passed_questions=passed_q_count,
            overall_pass_rate=overall_pass,
            recommendation=rec
        )

        conf_summary = ConfidenceEngine.compute_summary(execution_results, plagiarism_report.is_plagiarized)

        res = AssessmentAnalysisResult(
            metadata=Metadata(processing_time_ms=0.0),
            assessment_summary=summary,
            capability_scores=capability_scores,
            evidence_objects=evidence_objects,
            execution_results=execution_results,
            complexity_analysis=complexity_details,
            code_quality=code_quality,
            plagiarism_report=plagiarism_report,
            confidence_summary=conf_summary
        )

        val_report = SchemaValidator.validate_assessment_result(res)
        res.validation_report = val_report

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res.metadata.processing_time_ms = elapsed_ms

        return res
