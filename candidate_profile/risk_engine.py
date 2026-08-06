from typing import List
from capability_scoring.models import CapabilityScoringResult
from competency_intelligence.models import CompetencyIntelligenceResult
from .models import RiskArea

class RiskEngine:
    @classmethod
    def evaluate_risk_areas(
        cls,
        scoring: CapabilityScoringResult,
        competency: CompetencyIntelligenceResult
    ) -> List[RiskArea]:
        risks: List[RiskArea] = []
        r_idx = 1

        if scoring and scoring.readiness_summary.contradicted_count > 0:
            risks.append(
                RiskArea(
                    risk_id=f"rsk_{r_idx:02d}",
                    category="Verification",
                    severity="High",
                    title="Code Verification Contradiction",
                    description=f"Found {scoring.readiness_summary.contradicted_count} capability contradiction(s) between resume claims and repository code.",
                    mitigation_strategy="Conduct live code walkthrough and technical pair programming exercise."
                )
            )
            r_idx += 1

        if competency and competency.competency_summary.total_competencies == 1:
            risks.append(
                RiskArea(
                    risk_id=f"rsk_{r_idx:02d}",
                    category="SingleDomainDependency",
                    severity="Medium",
                    title="Single Domain Concentration",
                    description="Candidate capabilities are concentrated strictly within one engineering domain.",
                    mitigation_strategy="Assign cross-domain engineering tasks to broaden technical breadth."
                )
            )
            r_idx += 1

        if scoring and scoring.readiness_summary.unsupported_count > 2:
            risks.append(
                RiskArea(
                    risk_id=f"rsk_{r_idx:02d}",
                    category="TechnicalGap",
                    severity="Medium",
                    title="Multiple Unverified Capabilities",
                    description=f"{scoring.readiness_summary.unsupported_count} capabilities lack code or assessment verification.",
                    mitigation_strategy="Administer targeted technical assessments for missing capabilities."
                )
            )
            r_idx += 1

        return risks
