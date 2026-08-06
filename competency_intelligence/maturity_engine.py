from typing import List, Dict, Any
from .models import CompetencySummary

class MaturityEngine:
    @staticmethod
    def determine_maturity_level(score: float) -> str:
        if score >= 85.0:
            return "Expert"
        elif score >= 70.0:
            return "Advanced"
        elif score >= 55.0:
            return "Intermediate"
        elif score >= 40.0:
            return "Developing"
        else:
            return "Beginner"

    @classmethod
    def compute_summary(cls, competency_profiles: List[Any]) -> CompetencySummary:
        if not competency_profiles:
            return CompetencySummary(
                overall_competency_score=0.0,
                highest_competency="",
                lowest_competency="",
                total_competencies=0
            )

        avg_score = round(sum(p.competency_score for p in competency_profiles) / len(competency_profiles), 2)
        sorted_profiles = sorted(competency_profiles, key=lambda p: p.competency_score, reverse=True)

        highest = sorted_profiles[0].competency_name if sorted_profiles else ""
        lowest = sorted_profiles[-1].competency_name if sorted_profiles else ""

        exp = sum(1 for p in competency_profiles if p.maturity_level == "Expert")
        adv = sum(1 for p in competency_profiles if p.maturity_level == "Advanced")
        inter = sum(1 for p in competency_profiles if p.maturity_level == "Intermediate")
        dev = sum(1 for p in competency_profiles if p.maturity_level == "Developing")
        beg = sum(1 for p in competency_profiles if p.maturity_level == "Beginner")

        return CompetencySummary(
            overall_competency_score=avg_score,
            highest_competency=highest,
            lowest_competency=lowest,
            total_competencies=len(competency_profiles),
            expert_count=exp,
            advanced_count=adv,
            intermediate_count=inter,
            developing_count=dev,
            beginner_count=beg
        )
