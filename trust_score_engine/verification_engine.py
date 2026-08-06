import time
from typing import List, Dict, Any, Optional
from evidence_fusion.models import EvidenceFusionResult
from capability_scoring.models import CapabilityScoringResult
from job_intelligence.models import JobAnalysisResult
from resume_intelligence.models import ResumeAnalysisResult
from repository_intelligence.models import RepositoryAnalysisResult
from technical_assessment.models import AssessmentAnalysisResult
from behavioral_intelligence.models import BehavioralAnalysisResult
from competency_intelligence.models import CompetencyIntelligenceResult
from candidate_profile.models import CandidateProfileResult

from .models import (
    VerificationSummary,
    TrustScoreResult,
    TrustSummary,
    RiskSummary,
    ConfidenceSummary,
    ReportDetail,
    Metadata
)
from .trust_calculator import TrustCalculator
from .risk_engine import RiskEngine
from .report_generator import ReportGenerator
from .recommendation_engine import RecommendationEngine
from .validators import SchemaValidator

class VerificationEngine:
    @classmethod
    def compile_verification_summary(
        cls,
        fusion_res: Optional[EvidenceFusionResult],
        scoring_res: Optional[CapabilityScoringResult]
    ) -> VerificationSummary:
        if not fusion_res and not scoring_res:
            return VerificationSummary()

        profiles = fusion_res.capability_profiles if fusion_res else []

        sv_count = sum(1 for p in profiles if p.status == "Strongly Verified")
        v_count = sum(1 for p in profiles if p.status == "Verified")
        pv_count = sum(1 for p in profiles if p.status == "Partially Verified")
        wv_count = sum(1 for p in profiles if p.status == "Weakly Verified")
        u_count = sum(1 for p in profiles if p.status == "Unsupported")
        c_count = sum(1 for p in profiles if p.status == "Contradicted")

        verified_caps: List[str] = [p.capability_name for p in profiles if p.status in ["Strongly Verified", "Verified", "Partially Verified"]]
        unverified_caps: List[str] = [p.capability_name for p in profiles if p.status in ["Weakly Verified", "Unsupported"]]
        contradictions: List[str] = []

        if fusion_res and fusion_res.contradiction_report:
            for item in fusion_res.contradiction_report.contradictions:
                c_name = getattr(item, "capability_name", getattr(item, "capability_id", "Capability"))
                contradictions.append(f"Contradiction in {c_name}: {item.description}")

        return VerificationSummary(
            total_capabilities_evaluated=len(profiles),
            strongly_verified_count=sv_count,
            verified_count=v_count,
            partially_verified_count=pv_count,
            weakly_verified_count=wv_count,
            unsupported_count=u_count,
            contradicted_count=c_count,
            verified_capabilities=verified_caps,
            unverified_capabilities=unverified_caps,
            contradictions_list=contradictions
        )

class FinalTrustEngine:
    def __init__(self):
        pass

    def generate_trust_score(
        self,
        job_analysis: Optional[JobAnalysisResult] = None,
        resume_analysis: Optional[ResumeAnalysisResult] = None,
        repository_analysis: Optional[RepositoryAnalysisResult] = None,
        technical_assessment: Optional[AssessmentAnalysisResult] = None,
        behavioral_assessment: Optional[BehavioralAnalysisResult] = None,
        evidence_fusion_result: Optional[EvidenceFusionResult] = None,
        capability_scoring_result: Optional[CapabilityScoringResult] = None,
        competency_intelligence_result: Optional[CompetencyIntelligenceResult] = None,
        candidate_profile_result: Optional[CandidateProfileResult] = None
    ) -> TrustScoreResult:
        start_time = time.perf_counter()

        modules_integrated = sum(1 for m in [
            job_analysis, resume_analysis, repository_analysis,
            technical_assessment, behavioral_assessment, evidence_fusion_result,
            capability_scoring_result, competency_intelligence_result, candidate_profile_result
        ] if m is not None)

        cap_score = capability_scoring_result.readiness_summary.overall_capability_score if capability_scoring_result else 75.0
        comp_score = competency_intelligence_result.competency_summary.overall_competency_score if competency_intelligence_result else 75.0

        ev_reliability = evidence_fusion_result.reliability_summary.overall_reliability_score if (evidence_fusion_result and hasattr(evidence_fusion_result.reliability_summary, "overall_reliability_score")) else 75.0

        repo_reliability = repository_analysis.confidence_summary.average_confidence if (repository_analysis and hasattr(repository_analysis.confidence_summary, "average_confidence")) else 75.0

        tech_score = technical_assessment.assessment_summary.overall_score if technical_assessment else 75.0
        beh_score = behavioral_assessment.behavioral_summary.overall_behavioral_score if behavioral_assessment else 75.0

        contradictions_count = len(evidence_fusion_result.contradiction_report.contradictions) if (evidence_fusion_result and evidence_fusion_result.contradiction_report) else 0
        missing_sources_count = len(evidence_fusion_result.missing_evidence_report.missing_capabilities) if (evidence_fusion_result and evidence_fusion_result.missing_evidence_report) else 0

        is_forked = repository_analysis.originality_report.is_fork if (repository_analysis and hasattr(repository_analysis, "originality_report")) else False
        is_plagiarized = technical_assessment.plagiarism_report.is_plagiarized if (technical_assessment and hasattr(technical_assessment, "plagiarism_report")) else False

        trust_summary = TrustCalculator.calculate_trust(
            capability_score=cap_score,
            competency_score=comp_score,
            evidence_reliability=ev_reliability,
            repo_reliability=repo_reliability,
            tech_assessment_score=tech_score,
            behavioral_assessment_score=beh_score,
            contradictions_count=contradictions_count,
            missing_sources_count=missing_sources_count,
            is_forked_repo=is_forked,
            is_plagiarized=is_plagiarized
        )

        verification_summary = VerificationEngine.compile_verification_summary(evidence_fusion_result, capability_scoring_result)

        unsupported_count = verification_summary.unsupported_count
        risk_summary = RiskEngine.evaluate_risk_summary(contradictions_count, unsupported_count, is_forked, is_plagiarized)

        eng_conf = round((cap_score + comp_score) / 2.0, 2)
        ev_conf = round((ev_reliability + repo_reliability) / 2.0, 2)
        overall_conf = round((eng_conf * 0.50) + (ev_conf * 0.50), 2)

        conf_summary = ConfidenceSummary(
            engineering_confidence=eng_conf,
            evidence_confidence=ev_conf,
            overall_confidence=overall_conf,
            explanation=f"Overall confidence computed at {overall_conf}% based on {modules_integrated} integrated modules."
        )

        archetype = candidate_profile_result.candidate_summary.archetype if candidate_profile_result else "Software Engineer"
        seniority = candidate_profile_result.seniority.seniority_level if candidate_profile_result else "Mid-Level"

        report = ReportGenerator.generate_report(
            trust_summary=trust_summary,
            verification_summary=verification_summary,
            risk_summary=risk_summary,
            candidate_archetype=archetype,
            seniority_level=seniority
        )

        recommendations = RecommendationEngine.generate_recommendations(trust_summary, risk_summary)

        res = TrustScoreResult(
            metadata=Metadata(processing_time_ms=0.0),
            trust_summary=trust_summary,
            verification_summary=verification_summary,
            risk_summary=risk_summary,
            confidence_summary=conf_summary,
            recommendations=recommendations,
            report=report
        )

        val_report = SchemaValidator.validate_trust_result(res, modules_integrated)
        res.validation_report = val_report

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res.metadata.processing_time_ms = elapsed_ms

        return res
