from typing import List, Dict, Any, Optional
from .models import ReportDetail, TrustSummary, VerificationSummary, RiskSummary

class ReportGenerator:
    @classmethod
    def generate_report(
        cls,
        trust_summary: TrustSummary,
        verification_summary: VerificationSummary,
        risk_summary: RiskSummary,
        candidate_archetype: str = "Software Engineer",
        seniority_level: str = "Mid-Level"
    ) -> ReportDetail:
        exec_summary = (
            f"VeriProof Verification Report for {seniority_level} {candidate_archetype}. "
            f"Overall Trust Score: {trust_summary.overall_trust_score}% ({trust_summary.verification_level}). "
            f"Candidate is evaluated as {trust_summary.candidate_readiness} with {trust_summary.hiring_confidence} hiring confidence."
        )

        strengths: List[str] = [
            f"Verified proficiency in: {', '.join(verification_summary.verified_capabilities[:4])}"
            if verification_summary.verified_capabilities else "Foundational engineering background."
        ]

        weaknesses: List[str] = []
        if verification_summary.unverified_capabilities:
            weaknesses.append(f"Unverified evidence for: {', '.join(verification_summary.unverified_capabilities[:3])}")

        missing_ev: List[str] = verification_summary.unverified_capabilities

        interview_topics: List[str] = [
            f"Deep-dive into {cap} implementation patterns" for cap in verification_summary.unverified_capabilities[:3]
        ]
        if not interview_topics:
            interview_topics = ["System Architecture Design", "Code Refactoring & Scalability"]

        learning_path: List[str] = [
            f"Build production-grade repository project targeting {cap}" for cap in verification_summary.unverified_capabilities[:3]
        ]
        if not learning_path:
            learning_path = ["Advanced Cloud Architecture Guidelines", "CI/CD Production Deployment Practice"]

        return ReportDetail(
            executive_summary=exec_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_evidence=missing_ev,
            recommended_interview_topics=interview_topics,
            recommended_learning_path=learning_path,
            final_hiring_recommendation=trust_summary.final_recommendation
        )
