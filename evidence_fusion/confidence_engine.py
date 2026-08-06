from typing import List, Dict, Any
from .models import UnifiedCapabilityProfile, ConfidenceSummary, ContradictionItem

class ConfidenceEngine:
    @classmethod
    def compute_profile_confidence_and_status(
        cls,
        profile: UnifiedCapabilityProfile
    ) -> UnifiedCapabilityProfile:
        evidence_confidences: List[float] = []

        for e in profile.repository_evidence:
            evidence_confidences.append(float(e.get("confidence", 90.0)))
        for e in profile.assessment_evidence:
            evidence_confidences.append(float(e.get("confidence", 85.0)))
        for e in profile.professional_evidence:
            evidence_confidences.append(float(e.get("confidence", 80.0)))
        for e in profile.resume_evidence:
            evidence_confidences.append(float(e.get("confidence", 70.0)))
        for e in profile.behavioral_evidence:
            evidence_confidences.append(float(e.get("confidence", 70.0)))

        if not evidence_confidences:
            profile.merged_confidence = 0.0
            profile.status = "Unverified"
            profile.overall_reasoning = f"No evidence objects detected for capability '{profile.capability_name}'."
            return profile

        base_conf = sum(evidence_confidences) / len(evidence_confidences)

        # Apply contradiction penalties
        total_penalty = sum(c.confidence_penalty for c in profile.contradictions)
        final_conf = max(0.0, min(100.0, base_conf - total_penalty))
        profile.merged_confidence = round(final_conf, 2)

        has_repo = len(profile.repository_evidence) > 0
        has_assessment = len(profile.assessment_evidence) > 0
        has_contradictions = len(profile.contradictions) > 0

        # Determine Unified Status
        if has_contradictions and any(c.severity in ["High", "Critical"] for c in profile.contradictions):
            profile.status = "Contradicted"
            profile.overall_reasoning = f"Contradictions detected between claims and repository code: {profile.contradictions[0].description}"
        elif (has_repo or has_assessment) and profile.merged_confidence >= 70.0:
            profile.status = "Verified"
            profile.overall_reasoning = f"Capability verified with high confidence ({profile.merged_confidence}%) using code & execution evidence."
        elif profile.merged_confidence >= 45.0:
            profile.status = "Partially Verified"
            profile.overall_reasoning = f"Capability partially supported by resume claims and partial repository evidence ({profile.merged_confidence}% confidence)."
        else:
            profile.status = "Unverified"
            profile.overall_reasoning = f"Insufficient verifiable code or assessment evidence ({profile.merged_confidence}% confidence)."

        return profile

    @classmethod
    def compute_summary(cls, profiles: List[UnifiedCapabilityProfile]) -> ConfidenceSummary:
        if not profiles:
            return ConfidenceSummary()

        avg_conf = round(sum(p.merged_confidence for p in profiles) / len(profiles), 2)
        v_count = sum(1 for p in profiles if p.status == "Verified")
        pv_count = sum(1 for p in profiles if p.status == "Partially Verified")
        uv_count = sum(1 for p in profiles if p.status == "Unverified")
        c_count = sum(1 for p in profiles if p.status == "Contradicted")

        return ConfidenceSummary(
            average_merged_confidence=avg_conf,
            verified_capabilities_count=v_count,
            partially_verified_count=pv_count,
            unverified_capabilities_count=uv_count,
            contradicted_capabilities_count=c_count
        )
