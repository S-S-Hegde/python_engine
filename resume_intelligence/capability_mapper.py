from typing import List, Dict, Any
from .models import EvidenceObject, OwnershipSummary, CapabilityMappingSummary

class CapabilityMapper:
    @staticmethod
    def map_evidence_to_capabilities(
        evidence_list: List[EvidenceObject],
        valid_capability_ids: List[str],
        capability_name_map: Dict[str, str]
    ) -> List[CapabilityMappingSummary]:
        """
        Maps evidence objects strictly to valid Module 1 capability IDs.
        """
        summary_map: Dict[str, CapabilityMappingSummary] = {}

        # Initialize map for all Module 1 capabilities
        for cap_id in valid_capability_ids:
            cap_name = capability_name_map.get(cap_id, cap_id)
            summary_map[cap_id] = CapabilityMappingSummary(
                capability_id=cap_id,
                capability_name=cap_name,
                evidence_count=0,
                highest_confidence=0.0
            )

        for ev in evidence_list:
            if ev.capability_id in summary_map:
                summary = summary_map[ev.capability_id]
                summary.evidence_count += 1
                summary.highest_confidence = max(summary.highest_confidence, ev.confidence)
            else:
                # If capability_id is not in valid list, assign to nearest or fallback
                if valid_capability_ids:
                    fallback_id = valid_capability_ids[0]
                    ev.capability_id = fallback_id
                    summary = summary_map[fallback_id]
                    summary.evidence_count += 1
                    summary.highest_confidence = max(summary.highest_confidence, ev.confidence)

        return list(summary_map.values())

    @staticmethod
    def compute_ownership_summary(evidence_list: List[EvidenceObject]) -> OwnershipSummary:
        individual = sum(1 for e in evidence_list if e.ownership == "Individual")
        primary = sum(1 for e in evidence_list if e.ownership == "Primary Contributor")
        team = sum(1 for e in evidence_list if e.ownership == "Team Contributor")
        unknown = sum(1 for e in evidence_list if e.ownership not in ["Individual", "Primary Contributor", "Team Contributor"])

        return OwnershipSummary(
            individual_count=individual,
            primary_contributor_count=primary,
            team_contributor_count=team,
            unknown_count=unknown
        )
