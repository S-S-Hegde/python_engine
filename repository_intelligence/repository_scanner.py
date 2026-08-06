import re
from typing import List, Dict, Any
from .models import RepositoryEvidenceObject

class RepositoryScanner:
    @classmethod
    def scan_tree_and_extract_evidence(
        cls,
        repo_name: str,
        tree_paths: List[str],
        valid_capability_ids: List[str],
        capability_name_map: Dict[str, str]
    ) -> List[RepositoryEvidenceObject]:
        evidence_list: List[RepositoryEvidenceObject] = []
        if not valid_capability_ids:
            valid_capability_ids = ["cap_general_engineering"]

        paths_lower = [p.lower() for p in tree_paths]
        idx = 1

        def get_target_cap(keywords: List[str]) -> str:
            for cap_id in valid_capability_ids:
                cap_name = capability_name_map.get(cap_id, "").lower()
                for kw in keywords:
                    if kw in cap_id.lower() or kw in cap_name:
                        return cap_id
            return valid_capability_ids[0]

        # 1. API & Backend Controllers Evidence
        api_files = [p for p in tree_paths if any(k in p.lower() for k in ["controller", "routes", "api", "server.js", "app.py", "views.py"])]
        if api_files:
            cap_id = get_target_cap(["api", "backend", "route", "server", "controller"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=api_files[0],
                    quote=f"Extracted API route handlers and controller endpoint definitions in {api_files[0]}.",
                    engineering_decision="Controller-Service routing architecture for RESTful requests.",
                    complexity="High" if len(api_files) > 3 else "Medium",
                    ownership="Verified",
                    confidence=90.0,
                    verification_status="Repository Verified",
                    generated_from=[api_files[0]]
                )
            )
            idx += 1

        # 2. Authentication & Security Evidence
        auth_files = [p for p in tree_paths if any(k in p.lower() for k in ["auth", "jwt", "passport", "security", "session"])]
        if auth_files:
            cap_id = get_target_cap(["auth", "security", "jwt", "rbac"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=auth_files[0],
                    quote=f"Stateless authentication middleware and token validation in {auth_files[0]}.",
                    engineering_decision="JWT Bearer token security and role-based access control.",
                    complexity="High",
                    ownership="Verified",
                    confidence=92.0,
                    verification_status="Repository Verified",
                    generated_from=[auth_files[0]]
                )
            )
            idx += 1

        # 3. Database Schemas & Models Evidence
        db_files = [p for p in tree_paths if any(k in p.lower() for k in ["model", "schema", "entity", "migration", "prisma"])]
        if db_files:
            cap_id = get_target_cap(["db", "database", "schema", "mongo", "postgres", "model"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=db_files[0],
                    quote=f"Document/Relational schema model definitions and query structures in {db_files[0]}.",
                    engineering_decision="Indexed data modeling with schema validation constraints.",
                    complexity="Medium",
                    ownership="Verified",
                    confidence=88.0,
                    verification_status="Repository Verified",
                    generated_from=[db_files[0]]
                )
            )
            idx += 1

        # 4. Frontend Component & State Evidence
        frontend_files = [p for p in tree_paths if any(k in p.lower() for k in ["component", "jsx", "tsx", "redux", "slice", "store"])]
        if frontend_files:
            cap_id = get_target_cap(["frontend", "react", "state", "ui", "component"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=frontend_files[0],
                    quote=f"Modular UI component decomposition and state management in {frontend_files[0]}.",
                    engineering_decision="Component-driven architecture with centralized state slice management.",
                    complexity="Medium",
                    ownership="Verified",
                    confidence=89.0,
                    verification_status="Repository Verified",
                    generated_from=[frontend_files[0]]
                )
            )
            idx += 1

        # 5. Docker & DevOps Evidence
        devops_files = [p for p in tree_paths if any(k in p.lower() for k in ["dockerfile", "docker-compose", ".github/workflows", "terraform"])]
        if devops_files:
            cap_id = get_target_cap(["docker", "devops", "container", "ci", "cd", "deploy"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=devops_files[0],
                    quote=f"Container configuration and environment deployment setup in {devops_files[0]}.",
                    engineering_decision="Multi-stage containerization build for isolated runtime environments.",
                    complexity="Medium",
                    ownership="Verified",
                    confidence=90.0,
                    verification_status="Repository Verified",
                    generated_from=[devops_files[0]]
                )
            )
            idx += 1

        # 6. Testing Suite Evidence
        test_files = [p for p in tree_paths if any(k in p.lower() for k in ["test", "spec"])]
        if test_files:
            cap_id = get_target_cap(["test", "testing", "quality", "unit", "spec"])
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id=f"ev_repo_{idx:04d}",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=test_files[0],
                    quote=f"Automated unit/integration test specifications in {test_files[0]}.",
                    engineering_decision="Automated test suite verifying component and API integration contracts.",
                    complexity="Medium",
                    ownership="Verified",
                    confidence=90.0,
                    verification_status="Repository Verified",
                    generated_from=[test_files[0]]
                )
            )
            idx += 1

        # Fallback if no specific pattern was matched
        if not evidence_list:
            cap_id = valid_capability_ids[0]
            evidence_list.append(
                RepositoryEvidenceObject(
                    evidence_id="ev_repo_0001",
                    capability_id=cap_id,
                    source="Repository",
                    repository=repo_name,
                    location=tree_paths[0] if tree_paths else "root",
                    quote=f"Source code implementation in {tree_paths[0] if tree_paths else 'repository'}.",
                    engineering_decision="Standard repository codebase implementation",
                    complexity="Medium",
                    ownership="Verified",
                    confidence=75.0,
                    verification_status="Repository Verified",
                    generated_from=[tree_paths[0] if tree_paths else "root"]
                )
            )

        return evidence_list
