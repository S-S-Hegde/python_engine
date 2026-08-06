from typing import List, Dict, Any
from competency_intelligence.models import CompetencyIntelligenceResult
from capability_scoring.models import CapabilityScoringResult
from .models import SeniorityDetail

class SeniorityEngine:
    @classmethod
    def evaluate_seniority(
        cls,
        scoring: CapabilityScoringResult,
        competency: CompetencyIntelligenceResult
    ) -> SeniorityDetail:
        comp_score = competency.competency_summary.overall_competency_score if competency else 0.0
        readiness_score = scoring.readiness_summary.overall_capability_score if scoring else 0.0

        combined_score = round((comp_score * 0.50) + (readiness_score * 0.50), 2)

        has_high_depth = False
        if scoring and scoring.capability_scores:
            avg_depth = sum(s.formula_breakdown.depth_score for s in scoring.capability_scores) / len(scoring.capability_scores)
            has_high_depth = avg_depth >= 80.0

        num_competencies = len(competency.competencies) if competency else 0

        if combined_score >= 94.0 or (combined_score >= 88.0 and num_competencies >= 3 and has_high_depth):
            level = "Architect"
            reasoning = f"Candidate exhibits architectural leadership across multiple engineering domains ({combined_score}% score)."
        elif combined_score >= 88.0:
            level = "Lead"
            reasoning = f"Candidate demonstrates lead-level technical mastery and high depth ({combined_score}% score)."
        elif combined_score >= 78.0:
            level = "Senior"
            reasoning = f"Candidate possesses senior-level engineering execution capabilities ({combined_score}% score)."
        elif combined_score >= 65.0:
            level = "Mid-Level"
            reasoning = f"Candidate demonstrates autonomous mid-level engineering competence ({combined_score}% score)."
        elif combined_score >= 50.0:
            level = "Junior"
            reasoning = f"Candidate demonstrates foundational junior-level engineering skills ({combined_score}% score)."
        elif combined_score >= 35.0:
            level = "Fresher"
            reasoning = f"Candidate is at fresher/entry-level engineering maturity ({combined_score}% score)."
        else:
            level = "Student"
            reasoning = f"Candidate profile indicates student or early learning stage ({combined_score}% score)."

        return SeniorityDetail(
            seniority_level=level,
            seniority_score=combined_score,
            reasoning=reasoning
        )
