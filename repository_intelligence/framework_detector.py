import re
from typing import List, Dict, Any, Set, Optional
from .models import FrameworkSummary

class FrameworkDetector:
    @staticmethod
    def detect_frameworks_and_stack(
        tree_paths: List[str],
        languages: List[str],
        repo_data: Optional[Dict[str, Any]] = None
    ) -> FrameworkSummary:
        detected: Set[str] = set()
        databases: Set[str] = set()
        devops: Set[str] = set()
        backend: Optional[str] = None
        frontend: Optional[str] = None

        paths_lower = [p.lower() for p in tree_paths]

        # 1. Frontend Framework Detection
        if any(p.endswith((".jsx", ".tsx")) for p in paths_lower) or any("react" in p for p in paths_lower):
            detected.add("React")
            frontend = "React"
        if any("vue" in p for p in paths_lower) or any(p.endswith(".vue") for p in paths_lower):
            detected.add("Vue.js")
            frontend = frontend or "Vue.js"
        if any("angular" in p for p in paths_lower):
            detected.add("Angular")
            frontend = frontend or "Angular"
        if any("next" in p for p in paths_lower) or any("pages/api" in p or "app/api" in p for p in paths_lower):
            detected.add("Next.js")
            frontend = frontend or "Next.js"

        # 2. Backend Framework Detection
        if any("express" in p for p in paths_lower) or any(p.endswith(("server.js", "app.js", "index.js")) for p in paths_lower) or any("backend/" in p or "controllers/" in p for p in paths_lower):
            if "JavaScript" in languages or "TypeScript" in languages or any(p.endswith((".js", ".ts")) for p in paths_lower):
                detected.add("Express / Node.js")
                backend = "Node.js / Express"
        if any("fastapi" in p for p in paths_lower) or any("main.py" in p for p in paths_lower):
            if "Python" in languages or any(p.endswith(".py") for p in paths_lower):
                detected.add("FastAPI")
                backend = "Python / FastAPI"
        if any("django" in p for p in paths_lower) or any("manage.py" in p for p in paths_lower):
            detected.add("Django")
            backend = "Python / Django"
        if any("spring" in p for p in paths_lower) or any(p.endswith(".java") for p in paths_lower):
            detected.add("Spring Boot")
            backend = "Java / Spring Boot"

        # 3. Database Technology Detection
        if any("mongoose" in p or "mongo" in p or "models/" in p or "model/" in p or "schema" in p for p in paths_lower):
            databases.add("MongoDB")
        if any("pg" in p or "postgres" in p or "psycopg" in p for p in paths_lower):
            databases.add("PostgreSQL")
        if any("mysql" in p for p in paths_lower):
            databases.add("MySQL")
        if any("redis" in p for p in paths_lower):
            databases.add("Redis")
        if any("prisma" in p or "schema.prisma" in p for p in paths_lower):
            databases.add("Prisma ORM")

        # 4. DevOps & Infrastructure Detection
        if any("dockerfile" in p or "docker-compose" in p for p in paths_lower):
            devops.add("Docker")
        if any("kubernetes" in p or "k8s" in p or p.endswith((".k8s.yaml", ".k8s.yml")) for p in paths_lower):
            devops.add("Kubernetes")
        if any(".github/workflows" in p or "ci.yml" in p or "ci.yaml" in p for p in paths_lower):
            devops.add("GitHub Actions")
        if any("terraform" in p or p.endswith(".tf") for p in paths_lower):
            devops.add("Terraform")

        return FrameworkSummary(
            detected_frameworks=sorted(list(detected)),
            primary_backend_stack=backend,
            primary_frontend_stack=frontend,
            database_technologies=sorted(list(databases)),
            devops_tools=sorted(list(devops))
        )
