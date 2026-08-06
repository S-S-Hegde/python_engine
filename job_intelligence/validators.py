import re
from typing import List, Dict, Set
from .models import JobAnalysisResult, CapabilityNode, CompetencyNode

class ValidationError(Exception):
    pass

class SchemaValidator:
    @staticmethod
    def validate_capability_ids(capabilities: List[CapabilityNode]) -> None:
        pattern = re.compile(r"^cap_[a-z0-9]+_[a-z0-9_]+$")
        seen_ids: Set[str] = set()

        for cap in capabilities:
            if not cap.id or not isinstance(cap.id, str):
                raise ValidationError("Capability node contains empty or non-string ID.")

            # Sanitize to lower_snake_case if needed
            sanitized = cap.id.lower().strip()
            sanitized = re.sub(r"[^a-z0-9_]", "_", sanitized)
            if not sanitized.startswith("cap_"):
                sanitized = "cap_gen_" + sanitized

            if not pattern.match(sanitized):
                raise ValidationError(f"Capability ID '{cap.id}' (sanitized: '{sanitized}') does not match required format 'cap_<domain>_<subdomain>'")

            cap.id = sanitized

            if cap.id in seen_ids:
                raise ValidationError(f"Duplicate Capability ID detected: '{cap.id}'")
            seen_ids.add(cap.id)

    @staticmethod
    def detect_dependency_cycles(capabilities: List[CapabilityNode]) -> None:
        """Detects dependency cycles in capability graph using DFS."""
        graph: Dict[str, List[str]] = {cap.id: cap.dependencies for cap in capabilities}
        visited: Dict[str, int] = {cap.id: 0 for cap in capabilities}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node_id: str) -> None:
            if node_id not in visited:
                # Dependency to node outside graph is ignored or handled
                return
            if visited[node_id] == 1:
                raise ValidationError(f"Dependency cycle detected in Capability Graph involving node '{node_id}'")
            if visited[node_id] == 2:
                return

            visited[node_id] = 1
            for neighbor in graph.get(node_id, []):
                dfs(neighbor)
            visited[node_id] = 2

        for cap in capabilities:
            if visited[cap.id] == 0:
                dfs(cap.id)

    @staticmethod
    def validate_weights(capabilities: List[CapabilityNode], competencies: List[CompetencyNode]) -> None:
        for cap in capabilities:
            if cap.weight < 0:
                raise ValidationError(f"Capability '{cap.id}' has negative weight: {cap.weight}")

        for comp in competencies:
            if comp.weight < 0:
                raise ValidationError(f"Competency '{comp.name}' has negative weight: {comp.weight}")

        if capabilities:
            cap_sum = sum(c.weight for c in capabilities)
            if abs(cap_sum - 100.0) > 0.05:
                raise ValidationError(f"Capability weights sum to {cap_sum:.2f}, expected 100.0")

        if competencies:
            comp_sum = sum(c.weight for c in competencies)
            if abs(comp_sum - 100.0) > 0.05:
                raise ValidationError(f"Competency weights sum to {comp_sum:.2f}, expected 100.0")

    @staticmethod
    def validate_bounds_and_fields(capabilities: List[CapabilityNode], result: JobAnalysisResult) -> None:
        """Validates proficiency levels [1-5] and confidence scores [0-100]."""
        for cap in capabilities:
            if not (0.0 <= cap.confidence <= 100.0):
                raise ValidationError(f"Capability '{cap.id}' confidence {cap.confidence} out of bounds [0, 100]")
            if not (1 <= cap.expected_proficiency <= 5):
                raise ValidationError(f"Capability '{cap.id}' proficiency {cap.expected_proficiency} out of bounds [1, 5]")

        if not (1 <= result.job_complexity.technical <= 5):
            raise ValidationError(f"Job complexity technical score {result.job_complexity.technical} out of bounds [1, 5]")
        if not (1 <= result.candidate_level_expected.minimum_proficiency <= 5):
            raise ValidationError(f"Candidate expected proficiency {result.candidate_level_expected.minimum_proficiency} out of bounds [1, 5]")

    @classmethod
    def validate_full_result(cls, result: JobAnalysisResult) -> JobAnalysisResult:
        """Runs complete pre-flight validation on JobAnalysisResult object."""
        cls.validate_capability_ids(result.capability_graph)
        cls.detect_dependency_cycles(result.capability_graph)
        cls.validate_weights(result.capability_graph, result.competency_graph)
        cls.validate_bounds_and_fields(result.capability_graph, result)
        return result
