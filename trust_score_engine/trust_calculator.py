from typing import Dict, Any, Optional
from .weighting_engine import WeightingEngine
from .models import TrustSummary

class TrustCalculator:
    @classmethod
    def calculate_trust(
        cls,
        capability_score: float,
        competency_score: float,
        evidence_reliability: float,
        repo_reliability: float,
        tech_assessment_score: float,
        behavioral_assessment_score: float,
        contradictions_count: int,
        missing_sources_count: int,
        is_forked_repo: bool = False,
        is_plagiarized: bool = False,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> TrustSummary:
        weights = WeightingEngine.get_normalized_weights(custom_weights)

        weighted_sum = (
            (capability_score * weights["capability_score"]) +
            (competency_score * weights["competency_score"]) +
            (evidence_reliability * weights["evidence_reliability"]) +
            (repo_reliability * weights["repo_reliability"]) +
            (tech_assessment_score * weights["technical_assessment"]) +
            (behavioral_assessment_score * weights["behavioral_assessment"])
        )

        # Penalties calculation
        contradiction_penalty = contradictions_count * 15.0
        missing_penalty = missing_sources_count * 5.0
        fork_penalty = 15.0 if is_forked_repo else 0.0
        plagiarism_penalty = 30.0 if is_plagiarized else 0.0

        total_penalty = contradiction_penalty + missing_penalty + fork_penalty + plagiarism_penalty

        raw_trust = weighted_sum - total_penalty
        final_trust = round(max(0.0, min(100.0, raw_trust)), 2)

        # Determine Verification Level
        if contradictions_count > 0:
            v_level = "Contradicted"
        elif final_trust >= 85.0:
            v_level = "Strongly Verified"
        elif final_trust >= 70.0:
            v_level = "Verified"
        elif final_trust >= 50.0:
            v_level = "Partially Verified"
        elif final_trust >= 30.0:
            v_level = "Weakly Verified"
        else:
            v_level = "Unverified"

        # Determine Hiring Confidence & Candidate Readiness
        if v_level == "Contradicted" or is_plagiarized:
            h_conf = "Do Not Hire"
            readiness = "Not Ready"
            rec = "Reject Candidate: Flagged for contradiction or plagiarism."
        elif final_trust >= 80.0:
            h_conf = "High"
            readiness = "Production Ready"
            rec = "Strong Hire: Highly recommended for production engineering roles."
        elif final_trust >= 60.0:
            h_conf = "Moderate"
            readiness = "Nearly Ready"
            rec = "Hire with Targeted Onboarding: Recommended for mid-level roles."
        elif final_trust >= 40.0:
            h_conf = "Low"
            readiness = "Learning"
            rec = "Consider for Junior / Internship Role: Requires active mentorship."
        else:
            h_conf = "Do Not Hire"
            readiness = "Not Ready"
            rec = "Unsatisfactory Verification: Insufficient evidence provided."

        return TrustSummary(
            overall_trust_score=final_trust,
            verification_level=v_level,
            hiring_confidence=h_conf,
            candidate_readiness=readiness,
            final_recommendation=rec
        )
