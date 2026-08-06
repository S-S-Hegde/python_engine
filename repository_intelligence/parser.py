import json
import logging
import time
import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from job_intelligence.models import JobAnalysisResult
from .models import (
    RepositoryAnalysisResult,
    RepositorySummary,
    ArchitectureSummary,
    FrameworkSummary,
    OriginalityReport,
    RepositoryEvidenceObject,
    Metadata
)
from .prompts import REPOSITORY_INTELLIGENCE_PROMPT
from .framework_detector import FrameworkDetector
from .architecture_analyzer import ArchitectureAnalyzer
from .originality_checker import OriginalityChecker
from .repository_scanner import RepositoryScanner
from .capability_mapper import CapabilityMapper
from .confidence_engine import ConfidenceEngine
from .validators import SchemaValidator

logger = logging.getLogger("VeriProof.RepositoryIntelligence")

class RepositoryIntelligenceParser:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.model_name = model_name
        self.model = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

    def parse_repository(
        self,
        github_username: str,
        repo_data: Dict[str, Any],
        tree_paths: List[str],
        commits: List[Dict[str, Any]],
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> RepositoryAnalysisResult:
        start_time = time.perf_counter()

        # Extract Module 1 valid capability IDs
        valid_capability_ids: List[str] = []
        capability_name_map: Dict[str, str] = {}

        if job_analysis and job_analysis.capability_graph:
            for cap in job_analysis.capability_graph:
                valid_capability_ids.append(cap.id)
                capability_name_map[cap.id] = cap.name
        else:
            valid_capability_ids = ["cap_general_engineering"]
            capability_name_map = {"cap_general_engineering": "General Software Engineering"}

        repo_name = repo_data.get("name", "Repository")
        languages = list(repo_data.get("languages", ["JavaScript"])) if isinstance(repo_data.get("languages"), list) else [str(repo_data.get("language", "JavaScript"))]

        # 1. Structural & Static Analysis
        framework_summary = FrameworkDetector.detect_frameworks_and_stack(tree_paths, languages, repo_data)
        architecture_summary = ArchitectureAnalyzer.analyze_architecture(tree_paths)
        originality_report = OriginalityChecker.evaluate_originality(repo_data, commits)

        # 2. Extract Evidence Objects from file tree & code
        evidence_objects = RepositoryScanner.scan_tree_and_extract_evidence(
            repo_name=repo_name,
            tree_paths=tree_paths,
            valid_capability_ids=valid_capability_ids,
            capability_name_map=capability_name_map
        )

        # 3. Summaries & Calculations
        confidence_summary = ConfidenceEngine.compute_summary(evidence_objects, originality_report)
        capability_mapping = CapabilityMapper.map_evidence_to_capabilities(evidence_objects, valid_capability_ids, capability_name_map)
        validation_report = SchemaValidator.validate_repository_evidence(evidence_objects, valid_capability_ids)

        repo_summary = RepositorySummary(
            github_username=github_username or "Unknown",
            repositories_analyzed=[repo_name],
            total_files_scanned=len(tree_paths),
            total_commits_analyzed=len(commits),
            primary_language=languages[0] if languages else "JavaScript"
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RepositoryAnalysisResult(
            metadata=Metadata(
                processing_time_ms=elapsed_ms,
                model=self.model_name if self.model else "static-scanner"
            ),
            repository_summary=repo_summary,
            evidence_objects=evidence_objects,
            architecture_summary=architecture_summary,
            framework_summary=framework_summary,
            originality_report=originality_report,
            capability_mapping=capability_mapping,
            confidence_summary=confidence_summary,
            validation_report=validation_report
        )
