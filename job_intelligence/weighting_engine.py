from typing import List
from .models import CapabilityNode, CompetencyNode

class WeightingEngine:
    @staticmethod
    def normalize_capability_weights(capabilities: List[CapabilityNode]) -> List[CapabilityNode]:
        """Normalizes capability weights so that the sum across all nodes equals 100.0 exactly."""
        if not capabilities:
            return []

        # Clamp any negative weights to 0.0
        for cap in capabilities:
            if cap.weight < 0:
                cap.weight = 0.0

        total_weight = sum(cap.weight for cap in capabilities)
        if total_weight <= 0:
            equal_weight = round(100.0 / len(capabilities), 2)
            for cap in capabilities:
                cap.weight = equal_weight
            capabilities[-1].weight = round(capabilities[-1].weight + (100.0 - sum(c.weight for c in capabilities)), 2)
            return capabilities

        # Proportional scaling
        scaled_sum = 0.0
        for cap in capabilities:
            cap.weight = round((cap.weight / total_weight) * 100.0, 2)
            scaled_sum += cap.weight

        # Adjust residual rounding error onto the highest weighted capability
        residual = round(100.0 - scaled_sum, 2)
        if abs(residual) > 0.001 and len(capabilities) > 0:
            highest_cap = max(capabilities, key=lambda c: c.weight)
            highest_cap.weight = round(highest_cap.weight + residual, 2)

        return capabilities

    @staticmethod
    def normalize_competency_weights(competencies: List[CompetencyNode]) -> List[CompetencyNode]:
        """Normalizes competency weights so that the sum equals 100.0 exactly."""
        if not competencies:
            return []

        for comp in competencies:
            if comp.weight < 0:
                comp.weight = 0.0

        total_weight = sum(comp.weight for comp in competencies)
        if total_weight <= 0:
            equal_weight = round(100.0 / len(competencies), 2)
            for comp in competencies:
                comp.weight = equal_weight
            competencies[-1].weight = round(competencies[-1].weight + (100.0 - sum(c.weight for c in competencies)), 2)
            return competencies

        scaled_sum = 0.0
        for comp in competencies:
            comp.weight = round((comp.weight / total_weight) * 100.0, 2)
            scaled_sum += comp.weight

        residual = round(100.0 - scaled_sum, 2)
        if abs(residual) > 0.001 and len(competencies) > 0:
            highest_comp = max(competencies, key=lambda c: c.weight)
            highest_comp.weight = round(highest_comp.weight + residual, 2)

        return competencies
