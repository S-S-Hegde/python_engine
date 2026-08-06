from typing import List
from .models import TrustScoreResult, ValidationReport

class SchemaValidator:
    @staticmethod
    def validate_trust_result(
        result: TrustScoreResult,
        modules_count: int
    ) -> ValidationReport:
        warnings: List[str] = []

        s = result.trust_summary
        if not (0.0 <= s.overall_trust_score <= 100.0):
            warnings.append(f"Overall trust score {s.overall_trust_score} out of bounds [0, 100]. Clamping.")
            s.overall_trust_score = max(0.0, min(100.0, s.overall_trust_score))

        c = result.confidence_summary
        if not (0.0 <= c.overall_confidence <= 100.0):
            warnings.append(f"Overall confidence {c.overall_confidence} out of bounds [0, 100]. Clamping.")
            c.overall_confidence = max(0.0, min(100.0, c.overall_confidence))

        r = result.risk_summary
        if not (0.0 <= r.risk_score <= 100.0):
            warnings.append(f"Risk score {r.risk_score} out of bounds [0, 100]. Clamping.")
            r.risk_score = max(0.0, min(100.0, r.risk_score))

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_modules_integrated=modules_count,
            warnings=warnings
        )
