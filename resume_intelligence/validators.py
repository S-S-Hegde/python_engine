from typing import List, Set
from .models import EvidenceObject, ValidationReport

class ValidationError(Exception):
    pass

class SchemaValidator:
    @staticmethod
    def validate_evidence_objects(
        evidence_list: List[EvidenceObject],
        valid_capability_ids: List[str]
    ) -> ValidationReport:
        warnings: List[str] = []
        seen_quotes: Set[str] = set()

        for idx, ev in enumerate(evidence_list):
            if not ev.evidence_id:
                ev.evidence_id = f"ev_resume_{idx+1:04d}"

            if not ev.quote or not ev.quote.strip():
                warnings.append(f"Evidence '{ev.evidence_id}' has empty quote.")

            if ev.quote in seen_quotes:
                warnings.append(f"Duplicate evidence quote detected for '{ev.evidence_id}'.")
            seen_quotes.add(ev.quote)

            if valid_capability_ids and ev.capability_id not in valid_capability_ids:
                warnings.append(f"Evidence '{ev.evidence_id}' uses capability_id '{ev.capability_id}' not present in Module 1 Job Analysis.")
                ev.capability_id = valid_capability_ids[0]

            if not (0.0 <= ev.confidence <= 100.0):
                warnings.append(f"Evidence '{ev.evidence_id}' confidence {ev.confidence} out of bounds [0, 100]. Clamping.")
                ev.confidence = max(0.0, min(100.0, ev.confidence))

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_evidence_validated=len(evidence_list),
            warnings=warnings
        )
