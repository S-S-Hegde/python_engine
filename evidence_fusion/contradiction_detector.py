from typing import List, Dict, Any
from .models import ContradictionItem, UnifiedCapabilityProfile

class ContradictionDetector:
    @classmethod
    def detect_contradictions(
        cls,
        profile: UnifiedCapabilityProfile,
        repo_summary: Dict[str, Any],
        originality_verdict: str
    ) -> List[ContradictionItem]:
        contradictions: List[ContradictionItem] = []
        c_idx = 1

        cap_id = profile.capability_id
        cap_name = profile.capability_name.lower()

        has_resume = len(profile.resume_evidence) > 0
        has_repo = len(profile.repository_evidence) > 0
        has_tests = repo_summary.get("has_tests", False)
        has_docker = repo_summary.get("has_docker", False)

        # 1. Contradiction: Specific Tech claimed on Resume but missing from Repo code
        if has_resume and not has_repo:
            if "docker" in cap_name or "container" in cap_name:
                if not has_docker:
                    contradictions.append(
                        ContradictionItem(
                            contradiction_id=f"cnt_{cap_id}_{c_idx:02d}",
                            capability_id=cap_id,
                            severity="High",
                            type="ClaimWithoutRepoEvidence",
                            description=f"Resume explicitly claims containerization ({profile.capability_name}), but repository scan found no Dockerfile or container configuration.",
                            source_claims=[e.get("quote", "") for e in profile.resume_evidence],
                            confidence_penalty=25.0
                        )
                    )
                    c_idx += 1
            elif "test" in cap_name or "quality" in cap_name:
                if not has_tests:
                    contradictions.append(
                        ContradictionItem(
                            contradiction_id=f"cnt_{cap_id}_{c_idx:02d}",
                            capability_id=cap_id,
                            severity="Medium",
                            type="ClaimWithoutRepoEvidence",
                            description=f"Resume claims testing suite ({profile.capability_name}), but repository scan found no test files or spec suites.",
                            source_claims=[e.get("quote", "") for e in profile.resume_evidence],
                            confidence_penalty=20.0
                        )
                    )
                    c_idx += 1

        # 2. Contradiction: Single-Day Dump or Forked Repository
        if has_repo and originality_verdict in ["Single-Day Dump", "Forked Repo"]:
            contradictions.append(
                ContradictionItem(
                    contradiction_id=f"cnt_{cap_id}_{c_idx:02d}",
                    capability_id=cap_id,
                    severity="High" if originality_verdict == "Forked Repo" else "Medium",
                    type="RepositoryOriginalityFlag",
                    description=f"Repository evidence for '{profile.capability_name}' flagged with verdict '{originality_verdict}'. Ownership confidence reduced.",
                    source_claims=[e.get("quote", "") for e in profile.repository_evidence],
                    confidence_penalty=30.0 if originality_verdict == "Forked Repo" else 15.0
                )
            )
            c_idx += 1

        # 3. Contradiction: Unverified Quantified Metrics
        for rev in profile.resume_evidence:
            quote = str(rev.get("quote", "")).lower()
            if any(k in quote for k in ["reduction", "increase", "boost", "improvement", "%", "latency", "throughput"]):
                if not has_repo and not profile.assessment_evidence:
                    contradictions.append(
                        ContradictionItem(
                            contradiction_id=f"cnt_{cap_id}_{c_idx:02d}",
                            capability_id=cap_id,
                            severity="Low",
                            type="UnverifiedMetricClaim",
                            description=f"Resume contains quantified metric claim ('{rev.get('quote')}'), but no repository benchmark or assessment evidence supports it.",
                            source_claims=[rev.get("quote", "")],
                            confidence_penalty=10.0
                        )
                    )
                    c_idx += 1

        return contradictions
