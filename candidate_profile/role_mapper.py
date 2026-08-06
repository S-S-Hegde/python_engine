from typing import List
from competency_intelligence.models import CompetencyIntelligenceResult, CompetencyProfile
from .models import RoleFitItem, SpecializationDetail

ROLE_DOMAIN_REQUIREMENTS = [
    ("Backend Engineer", ["backend", "database", "general"]),
    ("Frontend Engineer", ["frontend", "general"]),
    ("Full Stack Engineer", ["backend", "frontend", "database"]),
    ("AI / ML Engineer", ["ai_ml", "backend", "general"]),
    ("DevOps Engineer", ["devops", "cloud", "testing"]),
    ("Cloud Engineer", ["cloud", "devops", "backend"]),
    ("Data Engineer", ["database", "backend", "general"]),
    ("Cybersecurity Engineer", ["security", "backend", "devops"])
]

class RoleMapper:
    @classmethod
    def evaluate_role_fits(
        cls,
        competency_res: CompetencyIntelligenceResult,
        specialization: SpecializationDetail
    ) -> List[RoleFitItem]:
        profiles: List[CompetencyProfile] = competency_res.competencies if competency_res else []
        comp_map = {p.domain: p for p in profiles}

        role_items: List[RoleFitItem] = []

        for role_title, req_domains in ROLE_DOMAIN_REQUIREMENTS:
            scores: List[float] = []
            for d in req_domains:
                if d in comp_map:
                    scores.append(comp_map[d].competency_score)
                else:
                    scores.append(40.0)

            fit_score = round(sum(scores) / len(scores), 2)
            conf = round(min(100.0, max(0.0, fit_score * 0.95)), 2)

            if fit_score >= 45.0:
                cat = "Best-Fit"
                reasoning = f"Strong alignment ({fit_score}% fit) for {role_title} based on verified competencies."
            elif fit_score >= 35.0:
                cat = "Alternative"
                reasoning = f"Moderate capability alignment ({fit_score}% fit) for {role_title} requiring minor growth."
            else:
                cat = "Unsuitable"
                reasoning = f"Insufficient verified evidence ({fit_score}% fit) for {role_title} requirements."

            role_items.append(
                RoleFitItem(
                    role_title=role_title,
                    fit_category=cat,
                    fit_score=fit_score,
                    confidence=conf,
                    reasoning=reasoning
                )
            )

        return sorted(role_items, key=lambda r: r.fit_score, reverse=True)
