from typing import List
from .models import ReadinessSummary, CapabilityScoreDetail

class ReadinessEngine:
    @classmethod
    def evaluate_readiness(cls, scores: List[CapabilityScoreDetail]) -> ReadinessSummary:
        if not scores:
            return ReadinessSummary(
                overall_capability_score=0.0,
                readiness_level="Not Ready",
                readiness_percentage=0.0,
                readiness_reasoning="No capability scores evaluated."
            )

        avg_score = round(sum(s.final_capability_score for s in scores) / len(scores), 2)

        strongly_v = sum(1 for s in scores if s.status == "Strongly Verified")
        v_count = sum(1 for s in scores if s.status == "Verified")
        pv_count = sum(1 for s in scores if s.status == "Partially Verified")
        wv_count = sum(1 for s in scores if s.status == "Weakly Verified")
        u_count = sum(1 for s in scores if s.status == "Unsupported")
        c_count = sum(1 for s in scores if s.status == "Contradicted")

        if avg_score >= 85.0:
            level = "Production Ready"
            reasoning = f"Candidate demonstrates production-grade engineering readiness ({avg_score}% overall score) with {strongly_v} strongly verified capabilities."
        elif avg_score >= 70.0:
            level = "Nearly Ready"
            reasoning = f"Candidate is nearly ready ({avg_score}% overall score) with {v_count + strongly_v} verified capabilities."
        elif avg_score >= 45.0:
            level = "Learning"
            reasoning = f"Candidate demonstrates learning level capabilities ({avg_score}% overall score) needing mentorship."
        else:
            level = "Not Ready"
            reasoning = f"Candidate does not meet core verification thresholds ({avg_score}% overall score)."

        return ReadinessSummary(
            overall_capability_score=avg_score,
            readiness_level=level,
            readiness_percentage=avg_score,
            readiness_reasoning=reasoning,
            strongly_verified_count=strongly_v,
            verified_count=v_count,
            partially_verified_count=pv_count,
            weakly_verified_count=wv_count,
            unsupported_count=u_count,
            contradicted_count=c_count
        )
