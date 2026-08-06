from typing import List, Dict, Any
from capability_scoring.models import CapabilityScoreDetail
from .models import CompetencyProfile, CapabilityDistribution, ValidationReport
from .competency_builder import DOMAIN_NAME_MAP
from .dependency_engine import DependencyEngine
from .maturity_engine import MaturityEngine

class SchemaValidator:
    @staticmethod
    def validate_competency_result(
        profiles: List[CompetencyProfile],
        valid_capability_ids: List[str]
    ) -> ValidationReport:
        warnings: List[str] = []
        seen_ids: set = set()

        for p in profiles:
            cid = p.competency_id
            if not cid or not isinstance(cid, str):
                warnings.append("Competency profile contains empty or invalid competency_id.")

            if cid in seen_ids:
                warnings.append(f"Duplicate competency ID detected: '{cid}'.")
            seen_ids.add(cid)

            if not (0.0 <= p.competency_score <= 100.0):
                warnings.append(f"Competency '{cid}' score {p.competency_score} out of bounds [0, 100]. Clamping.")
                p.competency_score = max(0.0, min(100.0, p.competency_score))

            if not (0.0 <= p.coverage <= 100.0):
                warnings.append(f"Competency '{cid}' coverage {p.coverage} out of bounds [0, 100]. Clamping.")
                p.coverage = max(0.0, min(100.0, p.coverage))

            for cap_ref in p.capability_ids:
                if valid_capability_ids and cap_ref not in valid_capability_ids:
                    warnings.append(f"Competency '{cid}' references unknown capability_id '{cap_ref}'.")

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_competencies_evaluated=len(profiles),
            warnings=warnings
        )


class AggregationEngine:
    @classmethod
    def aggregate_competency(
        cls,
        comp_id: str,
        scores: List[CapabilityScoreDetail]
    ) -> CompetencyProfile:
        domain = comp_id.replace("comp_", "")
        comp_name = DOMAIN_NAME_MAP.get(domain, domain.replace("_", " ").title())

        cap_ids = [s.capability_id for s in scores]
        cap_scores = [s.final_capability_score for s in scores]

        avg_score = sum(cap_scores) / len(cap_scores) if cap_scores else 0.0
        min_score = min(cap_scores) if cap_scores else 0.0

        # Distribution
        sv_count = sum(1 for s in scores if s.status == "Strongly Verified")
        v_count = sum(1 for s in scores if s.status == "Verified")
        pv_count = sum(1 for s in scores if s.status == "Partially Verified")
        wv_count = sum(1 for s in scores if s.status == "Weakly Verified")
        u_count = sum(1 for s in scores if s.status == "Unsupported")
        c_count = sum(1 for s in scores if s.status == "Contradicted")

        dist = CapabilityDistribution(
            total_capabilities=len(scores),
            strongly_verified_count=sv_count,
            verified_count=v_count,
            partially_verified_count=pv_count,
            weakly_verified_count=wv_count,
            unsupported_count=u_count,
            contradicted_count=c_count
        )

        coverage = round(((sv_count + v_count + pv_count) / max(1, len(scores))) * 100.0, 2)
        critical_cov = round(((sv_count + v_count) / max(1, len(scores))) * 100.0, 2)

        dep_penalty = DependencyEngine.calculate_dependency_penalty(scores)

        raw_score = (0.60 * avg_score) + (0.25 * min_score) + (0.15 * coverage) - dep_penalty
        final_comp_score = round(max(0.0, min(100.0, raw_score)), 2)

        conf = round(max(0.0, min(100.0, avg_score - dep_penalty)), 2)
        rel = round(max(0.0, min(100.0, (coverage * 0.70) + (critical_cov * 0.30))), 2)

        maturity = MaturityEngine.determine_maturity_level(final_comp_score)

        reasoning = (
            f"Competency '{comp_name}' evaluated at {final_comp_score}% ({maturity}). "
            f"Average capability score: {avg_score:.1f}%, Coverage: {coverage:.1f}%."
        )

        return CompetencyProfile(
            competency_id=comp_id,
            competency_name=comp_name,
            domain=domain,
            competency_score=final_comp_score,
            coverage=coverage,
            average_capability_score=round(avg_score, 2),
            minimum_capability_score=round(min_score, 2),
            critical_capability_coverage=critical_cov,
            competency_confidence=conf,
            competency_reliability=rel,
            maturity_level=maturity,
            capability_ids=cap_ids,
            capability_distribution=dist,
            dependency_penalty=dep_penalty,
            reasoning=reasoning
        )
