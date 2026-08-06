from typing import Optional, List
from .models import JobAnalysisResult, CapabilityNode, CompetencyNode
from .parser import JobIntelligenceParser
from .validators import SchemaValidator

class JobIntelligenceService:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.parser = JobIntelligenceParser(model_name=model_name)

    def analyze_job(self, jd_text: Optional[str] = None, raw_requirements: Optional[List[str]] = None) -> JobAnalysisResult:
        """
        Primary entry point for Module 1 (Job Intelligence Service).
        Parses raw JD text or raw requirement strings into a verified Schema 2.0 contract.
        """
        if jd_text and jd_text.strip():
            return self.parser.parse_job_description(jd_text)
        elif raw_requirements:
            return self.parser.parse_legacy_requirements(raw_requirements)
        else:
            return self.parser.parse_legacy_requirements(["Full Stack Software Engineering"])

__all__ = [
    "JobIntelligenceService",
    "JobAnalysisResult",
    "CapabilityNode",
    "CompetencyNode",
    "SchemaValidator"
]
