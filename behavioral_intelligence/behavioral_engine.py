import time
from typing import List, Dict, Any, Optional
from job_intelligence.models import JobAnalysisResult
from .models import (
    BehavioralSubmissionPayload,
    BehavioralAnalysisResult,
    CapabilityBehavioralScore,
    BehavioralEvidenceObject,
    StarDetail,
    BehavioralSummary,
    Metadata
)
from .question_engine import QuestionEngine
from .star_analyzer import StarAnalyzer
from .communication_engine import CommunicationEngine
from .ownership_engine import OwnershipEngine
from .leadership_engine import LeadershipEngine
from .decision_engine import DecisionEngine
from .confidence_engine import ConfidenceEngine, SchemaValidator

class BehavioralEngine:
    def __init__(self):
        pass

    def analyze_behavior(
        self,
        submission: BehavioralSubmissionPayload,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> BehavioralAnalysisResult:
        start_time = time.perf_counter()

        if not job_analysis and submission.job_analysis:
            try:
                job_analysis = JobAnalysisResult.model_validate(submission.job_analysis)
            except Exception:
                job_analysis = None

        star_details: List[StarDetail] = []
        evidence_objects: List[BehavioralEvidenceObject] = []
        cap_scores_map: Dict[str, Dict[str, Any]] = {}

        for idx, resp in enumerate(submission.responses):
            cap_id, cap_name = QuestionEngine.resolve_target_capability(resp, job_analysis)

            star = StarAnalyzer.analyze_star(resp)
            star_details.append(star)

            # Verification status based on STAR completeness
            if star.star_score == 100.0:
                status = "Strongly Verified"
            elif star.star_score >= 75.0:
                status = "Verified"
            elif star.star_score >= 50.0:
                status = "Partially Verified"
            elif star.star_score >= 25.0:
                status = "Weakly Verified"
            else:
                status = "Unsupported"

            ev_id = f"ev_beh_{idx+1:04d}"
            # Reference exact excerpt from candidate response (no hallucination)
            quote = (resp.response_text[:140] + "...") if len(resp.response_text) > 140 else resp.response_text

            evidence_objects.append(
                BehavioralEvidenceObject(
                    evidence_id=ev_id,
                    capability_id=cap_id,
                    quote=f"Response for {resp.question_id}: '{quote}' (STAR Score: {star.star_score}%).",
                    source="behavioral_assessment",
                    confidence=star.star_score,
                    ownership="Candidate Response",
                    verified=star.star_score >= 50.0,
                    status=status,
                    details={
                        "question_id": resp.question_id,
                        "star_score": star.star_score,
                        "has_situation": star.has_situation,
                        "has_task": star.has_task,
                        "has_action": star.has_action,
                        "has_result": star.has_result
                    }
                )
            )

            if cap_id not in cap_scores_map:
                cap_scores_map[cap_id] = {
                    "capability_name": cap_name,
                    "star_scores": []
                }
            cap_scores_map[cap_id]["star_scores"].append(star.star_score)

        comm_analysis = CommunicationEngine.evaluate_communication(submission.responses)
        own_analysis = OwnershipEngine.evaluate_ownership(submission.responses)
        lead_analysis = LeadershipEngine.evaluate_leadership(submission.responses)
        dec_analysis = DecisionEngine.evaluate_decision_making(submission.responses)

        capability_scores: List[CapabilityBehavioralScore] = []
        for cid, info in cap_scores_map.items():
            avg_star = sum(info["star_scores"]) / len(info["star_scores"])
            own_score = own_analysis.overall_ownership_score
            comm_score = comm_analysis.overall_communication_score

            raw_final = (avg_star * 0.40) + (own_score * 0.35) + (comm_score * 0.25)
            final_score = round(max(0.0, min(100.0, raw_final)), 2)

            capability_scores.append(
                CapabilityBehavioralScore(
                    capability_id=cid,
                    capability_name=info["capability_name"],
                    star_score=round(avg_star, 2),
                    ownership_score=round(own_score, 2),
                    communication_score=round(comm_score, 2),
                    final_capability_score=final_score
                )
            )

        overall_score = round(sum(c.final_capability_score for c in capability_scores) / max(1, len(capability_scores)), 2) if capability_scores else 0.0

        strengths: List[str] = []
        growth: List[str] = []

        if comm_analysis.overall_communication_score >= 80.0:
            strengths.append("High Communication & Response Structure Clarity.")
        if own_analysis.overall_ownership_score >= 80.0:
            strengths.append("Strong Individual Accountability and Ownership Mindset.")
        if lead_analysis.overall_leadership_score >= 80.0:
            strengths.append("Demonstrated Cross-Functional Collaboration & Team Alignment.")

        if own_analysis.blame_shifting_detected:
            growth.append("Eliminate blame-shifting language; focus on personal agency and accountability.")
        if any(s.star_score < 75.0 for s in star_details):
            growth.append("Structure behavioral interview answers using complete STAR methodology (Situation, Task, Action, Result).")

        if overall_score >= 80.0:
            rec = "Strong Behavioral Performance. Candidate demonstrated clear ownership, structured communication, and team leadership."
        elif overall_score >= 50.0:
            rec = "Satisfactory Behavioral Performance. Adequate communication and accountability demonstrated."
        else:
            rec = "Unsatisfactory Behavioral Performance. Unstructured answers or poor accountability demonstrated."

        summary = BehavioralSummary(
            overall_behavioral_score=overall_score,
            total_responses_analyzed=len(submission.responses),
            primary_strengths=strengths,
            areas_for_growth=growth,
            recommendation=rec
        )

        conf_summary = ConfidenceEngine.compute_summary(submission.responses, own_analysis.blame_shifting_detected)

        res = BehavioralAnalysisResult(
            metadata=Metadata(processing_time_ms=0.0),
            behavioral_summary=summary,
            capability_scores=capability_scores,
            evidence_objects=evidence_objects,
            star_analysis=star_details,
            communication_analysis=comm_analysis,
            leadership_analysis=lead_analysis,
            ownership_analysis=own_analysis,
            confidence_summary=conf_summary
        )

        val_report = SchemaValidator.validate_behavioral_result(res)
        res.validation_report = val_report

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res.metadata.processing_time_ms = elapsed_ms

        return res
