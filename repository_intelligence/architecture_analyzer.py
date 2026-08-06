from typing import List, Dict, Any
from .models import ArchitectureSummary

class ArchitectureAnalyzer:
    @staticmethod
    def analyze_architecture(tree_paths: List[str]) -> ArchitectureSummary:
        paths_lower = [p.lower() for p in tree_paths]

        folders: List[str] = []
        for path in paths_lower:
            parts = path.split("/")
            if len(parts) > 1:
                folder = parts[0]
                if folder not in folders and folder in ["src", "app", "controllers", "models", "routes", "services", "components", "lib", "core", "pkg", "api", "utils"]:
                    folders.append(folder)

        # Check for testing files
        has_tests = any(
            "test" in p or "spec" in p or p.endswith((".test.js", ".spec.js", "_test.py", "test_.py"))
            for p in paths_lower
        )

        # Check for Docker & CI/CD
        has_docker = any("dockerfile" in p or "docker-compose" in p for p in paths_lower)
        has_ci_cd = any(".github/workflows" in p or "gitlab-ci" in p for p in paths_lower)

        # Infer Architecture Pattern
        pattern = "Layered Architecture"
        explanation = "Project separates concerns across controllers, routes, models, and utility services."

        has_mvc = any("controllers" in p for p in paths_lower) and any("models" in p for p in paths_lower)
        has_microservices = any("services/" in p or "microservices/" in p for p in paths_lower) or (any("docker-compose" in p for p in paths_lower) and len([p for p in paths_lower if "package.json" in p or "requirements.txt" in p]) > 1)
        has_clean_arch = any("core" in p or "domain" in p for p in paths_lower) and any("usecases" in p or "adapters" in p for p in paths_lower)
        has_components = any("components" in p for p in paths_lower) or any(p.endswith((".jsx", ".tsx")) for p in paths_lower)

        if has_clean_arch:
            pattern = "Clean Architecture"
            explanation = "Domain entities and usecases are decoupled from infrastructure adapters."
        elif has_microservices:
            pattern = "Microservices"
            explanation = "Distributed service components configured with docker-compose or container isolation."
        elif has_mvc:
            pattern = "MVC Architecture"
            explanation = "Clear separation between Models (schemas), Views (UI), and Controllers (request handlers)."
        elif has_components:
            pattern = "Component-Based Architecture"
            explanation = "UI and application logic composed of reusable, isolated component modules."

        return ArchitectureSummary(
            pattern=pattern,
            explanation=explanation,
            detected_folders=folders,
            has_tests=has_tests,
            has_docker=has_docker,
            has_ci_cd=has_ci_cd
        )
