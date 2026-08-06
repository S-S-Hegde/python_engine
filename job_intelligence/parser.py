import json
import logging
import time
import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from .models import (
    JobAnalysisResult,
    JobComplexity,
    CandidateLevelExpected,
    CapabilityNode,
    CompetencyNode,
    Metadata
)
from .prompts import JOB_INTELLIGENCE_PROMPT
from .capability_graph import CapabilityGraphBuilder
from .competency_graph import CompetencyGraphBuilder
from .weighting_engine import WeightingEngine
from .validators import SchemaValidator

logger = logging.getLogger("VeriProof.JobIntelligence")

class JobIntelligenceParser:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.model_name = model_name
        self.model = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

    def _clean_json(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        for prefix in ["```json", "```JSON", "```"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def parse_job_description(self, jd_text: str) -> JobAnalysisResult:
        start_time = time.perf_counter()

        if not jd_text or not jd_text.strip():
            return self._build_fallback_result(["General Engineering"], start_time)

        if not self.model:
            logger.warning("Gemini model not initialized. Using fallback parsing.")
            return self._build_fallback_result([line.strip() for line in jd_text.splitlines() if line.strip()][:5], start_time)

        prompt = JOB_INTELLIGENCE_PROMPT.format(jd_text=jd_text)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                ),
            )
            raw_text = getattr(response, "text", "")
            cleaned = self._clean_json(raw_text)
            data = json.loads(cleaned)

            capabilities = CapabilityGraphBuilder.build_from_raw_nodes(data.get("capability_graph", []))
            capabilities = WeightingEngine.normalize_capability_weights(capabilities)

            competencies = CompetencyGraphBuilder.build_from_raw_nodes(data.get("competency_graph", []))
            competencies = WeightingEngine.normalize_competency_weights(competencies)

            complexity_data = data.get("job_complexity", {})
            complexity = JobComplexity(
                overall=complexity_data.get("overall", "Intermediate"),
                technical=int(complexity_data.get("technical", 3)),
                architecture=int(complexity_data.get("architecture", 3)),
                communication=int(complexity_data.get("communication", 3)),
                domain=int(complexity_data.get("domain", 3))
            )

            level_data = data.get("candidate_level_expected", {})
            candidate_level = CandidateLevelExpected(
                level=level_data.get("level", "Intermediate"),
                experience_range=level_data.get("experience_range", "1-3 years"),
                minimum_proficiency=int(level_data.get("minimum_proficiency", 3))
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            result = JobAnalysisResult(
                metadata=Metadata(
                    processing_time_ms=elapsed_ms,
                    model=self.model_name
                ),
                role=str(data.get("role", "Software Engineer")),
                business_objectives=list(data.get("business_objectives", [])),
                engineering_objectives=list(data.get("engineering_objectives", [])),
                job_complexity=complexity,
                candidate_level_expected=candidate_level,
                capability_graph=capabilities,
                competency_graph=competencies,
                risk_areas=list(data.get("risk_areas", [])),
                positive_hiring_signals=list(data.get("positive_hiring_signals", [])),
                negative_hiring_signals=list(data.get("negative_hiring_signals", []))
            )

            return SchemaValidator.validate_full_result(result)

        except Exception as e:
            logger.error(f"Error parsing job description via LLM: {str(e)}")
            return self._build_fallback_result([line.strip() for line in jd_text.splitlines() if line.strip()][:5], start_time)

    def parse_legacy_requirements(self, requirements: List[str]) -> JobAnalysisResult:
        """Fallback converter for legacy string array job_requirements."""
        start_time = time.perf_counter()
        return self._build_fallback_result(requirements, start_time)

    def _build_fallback_result(self, keywords: List[str], start_time: float) -> JobAnalysisResult:
        capabilities: List[CapabilityNode] = []
        if not keywords:
            keywords = ["Full Stack Development"]

        equal_weight = round(100.0 / len(keywords), 2)
        for idx, kw in enumerate(keywords):
            clean_kw = kw.strip().lower()
            cap_id = f"cap_general_{idx+1}_{clean_kw[:10]}"
            capabilities.append(
                CapabilityNode(
                    id=cap_id,
                    name=kw.strip().capitalize(),
                    confidence=80.0,
                    importance="Critical" if idx == 0 else "Important",
                    weight=equal_weight,
                    expected_proficiency=3,
                    generated_from=[kw],
                    dependencies=[],
                    expected_evidence=["Repository", "Technical Assessment", "Resume"],
                    sub_capabilities=[kw],
                    validation_rules=[f"Evidence of {kw} implementation"],
                    negative_evidence=[f"No implementation evidence of {kw}"]
                )
            )

        capabilities = WeightingEngine.normalize_capability_weights(capabilities)

        competencies = [
            CompetencyNode(
                name="Core Engineering Competency",
                weight=100.0,
                confidence=80.0,
                depends_on=[c.id for c in capabilities],
                capabilities=[c.id for c in capabilities]
            )
        ]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = JobAnalysisResult(
            metadata=Metadata(
                processing_time_ms=elapsed_ms,
                model="fallback-normalizer"
            ),
            role="Software Engineer",
            business_objectives=["Deliver reliable core software functionality"],
            engineering_objectives=["Implement target capabilities with clean code and tests"],
            job_complexity=JobComplexity(overall="Intermediate", technical=3, architecture=3, communication=3, domain=3),
            candidate_level_expected=CandidateLevelExpected(level="Intermediate", experience_range="1-3 years", minimum_proficiency=3),
            capability_graph=capabilities,
            competency_graph=competencies,
            risk_areas=["Lack of verifiable test coverage"],
            positive_hiring_signals=["Modular code structure"],
            negative_hiring_signals=["Monolithic single-file project layout"]
        )

        return SchemaValidator.validate_full_result(result)
