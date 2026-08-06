import re
from typing import List, Dict, Any
from .models import CapabilityNode

class CapabilityGraphBuilder:
    @staticmethod
    def sanitize_id(raw_id: str, name: str, domain: str = "general") -> str:
        """Ensures a stable cap_<domain>_<subdomain> identifier format."""
        if raw_id and re.match(r"^cap_[a-z0-9]+_[a-z0-9_]+$", raw_id):
            return raw_id

        clean_name = re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")
        clean_domain = re.sub(r"[^a-z0-9]", "_", domain.lower()).strip("_")
        return f"cap_{clean_domain}_{clean_name}"[:40]

    @classmethod
    def build_from_raw_nodes(cls, raw_nodes: List[Dict[str, Any]]) -> List[CapabilityNode]:
        nodes: List[CapabilityNode] = []
        for raw in raw_nodes:
            raw_id = str(raw.get("id", ""))
            name = str(raw.get("name", "Unspecified Capability"))
            domain = raw_id.split("_")[1] if raw_id.startswith("cap_") and "_" in raw_id[4:] else "general"

            stable_id = cls.sanitize_id(raw_id, name, domain)

            node = CapabilityNode(
                id=stable_id,
                name=name,
                confidence=float(raw.get("confidence", 90.0)),
                classification=str(raw.get("classification", "Verified_Requirement")),
                importance=str(raw.get("importance", "Critical")),
                weight=float(raw.get("weight", 10.0)),
                expected_proficiency=int(raw.get("expected_proficiency", 3)),
                generated_from=list(raw.get("generated_from", [])),
                dependencies=list(raw.get("dependencies", [])),
                expected_evidence=list(raw.get("expected_evidence", ["Repository", "Technical Assessment", "Resume"])),
                sub_capabilities=list(raw.get("sub_capabilities", [])),
                validation_rules=list(raw.get("validation_rules", [])),
                negative_evidence=list(raw.get("negative_evidence", []))
            )
            nodes.append(node)
        return nodes
