from typing import Optional, Dict, Any, List
from job_intelligence.models import JobAnalysisResult
from .models import (
    RepositoryAnalysisResult,
    RepositoryEvidenceObject,
    OriginalityReport,
    ArchitectureSummary,
    FrameworkSummary,
    RepositorySummary,
    ValidationReport
)
from .parser import RepositoryIntelligenceParser
from .validators import SchemaValidator

class RepositoryIntelligenceService:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.parser = RepositoryIntelligenceParser(model_name=model_name)

    def analyze_repository(
        self,
        github_username: str,
        repo_data: Dict[str, Any],
        tree_paths: List[str],
        commits: List[Dict[str, Any]],
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> RepositoryAnalysisResult:
        """
        Primary entry point for Module 3 (Repository Intelligence Service).
        Extracts verifiable engineering evidence from GitHub repositories and maps directly to Module 1 capability IDs.
        """
        return self.parser.parse_repository(
            github_username=github_username,
            repo_data=repo_data,
            tree_paths=tree_paths,
            commits=commits,
            job_analysis=job_analysis
        )

__all__ = [
    "RepositoryIntelligenceService",
    "RepositoryAnalysisResult",
    "RepositoryEvidenceObject",
    "OriginalityReport",
    "ArchitectureSummary",
    "FrameworkSummary",
    "RepositorySummary",
    "ValidationReport",
    "SchemaValidator"
]
