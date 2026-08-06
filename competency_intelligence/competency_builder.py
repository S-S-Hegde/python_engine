from typing import Dict, Any, List
from capability_scoring.models import CapabilityScoreDetail

DOMAIN_RULES = [
    ("backend", ["backend", "api", "rest", "graphql", "node", "express", "fastapi", "spring", "microservice", "server"]),
    ("frontend", ["frontend", "react", "next", "vue", "angular", "css", "tailwind", "ui", "state"]),
    ("database", ["database", "mongo", "sql", "postgres", "redis", "orm", "indexing", "schema"]),
    ("devops", ["devops", "docker", "container", "kubernetes", "ci", "cd", "pipeline", "deploy"]),
    ("testing", ["testing", "test", "jest", "pytest", "unit", "e2e", "qa", "suite"]),
    ("security", ["security", "auth", "jwt", "oauth", "crypto", "encryption", "cors"]),
    ("cloud", ["cloud", "aws", "gcp", "azure", "serverless", "lambda", "s3"]),
    ("ai_ml", ["ai", "ml", "machine_learning", "nlp", "llm", "tensorflow", "pytorch", "model"]),
    ("software_design", ["design", "architecture", "pattern", "clean", "solid", "refactor"])
]

DOMAIN_NAME_MAP = {
    "backend": "Backend Engineering",
    "frontend": "Frontend Engineering",
    "database": "Database Engineering",
    "devops": "DevOps & Infrastructure",
    "testing": "Automated Testing & Quality",
    "security": "Application Security",
    "cloud": "Cloud Engineering",
    "ai_ml": "AI & Machine Learning",
    "software_design": "Software Architecture & Design",
    "general": "Software Engineering Fundamentals"
}

class CompetencyBuilder:
    @classmethod
    def classify_capability_domain(cls, cap_id: str, cap_name: str) -> str:
        text = f"{cap_id} {cap_name}".lower()
        for domain, keywords in DOMAIN_RULES:
            if any(kw in text for kw in keywords):
                return domain
        return "general"

    @classmethod
    def group_capabilities_by_competency(
        cls,
        scores: List[CapabilityScoreDetail]
    ) -> Dict[str, List[CapabilityScoreDetail]]:
        grouped: Dict[str, List[CapabilityScoreDetail]] = {}

        for score in scores:
            domain = cls.classify_capability_domain(score.capability_id, score.capability_name)
            comp_id = f"comp_{domain}"
            if comp_id not in grouped:
                grouped[comp_id] = []
            grouped[comp_id].append(score)

        return grouped
