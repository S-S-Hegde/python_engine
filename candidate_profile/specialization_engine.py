from typing import List
from competency_intelligence.models import CompetencyIntelligenceResult, CompetencyProfile
from .models import SpecializationDetail

DOMAIN_LABEL_MAP = {
    "backend": "Backend Engineer",
    "frontend": "Frontend Engineer",
    "database": "Database Specialist",
    "devops": "DevOps Engineer",
    "testing": "Software Quality / QA Engineer",
    "security": "Cybersecurity Engineer",
    "cloud": "Cloud Engineer",
    "ai_ml": "AI / ML Engineer",
    "software_design": "Software Architect",
    "general": "Software Engineer"
}

class SpecializationEngine:
    @classmethod
    def evaluate_specialization(
        cls,
        competency_res: CompetencyIntelligenceResult
    ) -> SpecializationDetail:
        if not competency_res or not competency_res.competencies:
            return SpecializationDetail(
                primary_domain="General Software Engineering",
                secondary_domains=[],
                archetype="General Software Engineer",
                specialization_confidence=50.0
            )

        profiles = sorted(competency_res.competencies, key=lambda c: c.competency_score, reverse=True)
        top = profiles[0]

        primary_domain = top.competency_name
        secondary_domains: List[str] = [
            p.competency_name for p in profiles[1:]
            if p.competency_score >= 50.0 or (top.competency_score > 0 and p.competency_score / top.competency_score >= 0.75)
        ]

        # Archetype logic
        has_backend = any(p.domain == "backend" and p.competency_score >= 50.0 for p in profiles)
        has_frontend = any(p.domain == "frontend" and p.competency_score >= 50.0 for p in profiles)
        has_devops = any(p.domain == "devops" and p.competency_score >= 50.0 for p in profiles)
        has_ai = any(p.domain == "ai_ml" and p.competency_score >= 50.0 for p in profiles)

        if has_backend and has_frontend:
            archetype = "Full Stack Engineer"
        elif has_ai:
            archetype = "AI / ML Engineer"
        elif has_devops:
            archetype = "DevOps / Infrastructure Engineer"
        else:
            archetype = DOMAIN_LABEL_MAP.get(top.domain, "Software Engineer")

        conf = round(min(100.0, max(0.0, top.competency_confidence)), 2)

        return SpecializationDetail(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            archetype=archetype,
            specialization_confidence=conf
        )
