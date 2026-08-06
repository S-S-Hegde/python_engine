from typing import List, Dict, Any
from .models import RepositoryEvidenceObject, CapabilityMappingSummary

class CapabilityMapper:
    @staticmethod
    def map_evidence_to_capabilities(
        evidence_list: List[RepositoryEvidenceObject],
        valid_capability_ids: List[str],
        capability_name_map: Dict[str, str]
    ) -> List[CapabilityMappingSummary]:
        """
        Maps repository evidence strictly to valid Module 1 capability IDs.
        """
        summary_map: Dict[str, CapabilityMappingSummary] = {}

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
                if valid_capability_ids:
                    fallback_id = valid_capability_ids[0]
                    ev.capability_id = fallback_id
                    summary = summary_map[fallback_id]
                    summary.evidence_count += 1
                    summary.highest_confidence = max(summary.highest_confidence, ev.confidence)

        return list(summary_map.values())
