from typing import List, Dict, Any
from .models import CompetencyNode

class CompetencyGraphBuilder:
    @classmethod
    def build_from_raw_nodes(cls, raw_nodes: List[Dict[str, Any]]) -> List[CompetencyNode]:
        nodes: List[CompetencyNode] = []
        for raw in raw_nodes:
            node = CompetencyNode(
                name=str(raw.get("name", "Core Competency")),
                weight=float(raw.get("weight", 20.0)),
                confidence=float(raw.get("confidence", 90.0)),
                depends_on=list(raw.get("depends_on", [])),
                capabilities=list(raw.get("capabilities", []))
            )
            nodes.append(node)
        return nodes
