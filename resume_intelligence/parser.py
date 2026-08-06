import json
import logging
import time
import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from job_intelligence.models import JobAnalysisResult
from .models import (
    ResumeAnalysisResult,
    CandidateSummary,
    EvidenceObject,
    ResumeMetric,
    Metadata
)
from .prompts import RESUME_INTELLIGENCE_PROMPT
from .confidence_engine import ConfidenceEngine
from .capability_mapper import CapabilityMapper
from .evidence_extractor import EvidenceExtractor
from .validators import SchemaValidator

logger = logging.getLogger("VeriProof.ResumeIntelligence")

class ResumeIntelligenceParser:
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

    def parse_resume(
        self,
        resume_text: str,
        job_analysis: Optional[JobAnalysisResult] = None
    ) -> ResumeAnalysisResult:
        start_time = time.perf_counter()

        # Extract valid capability IDs and name mapping from Module 1 JobAnalysisResult
        valid_capability_ids: List[str] = []
        capability_name_map: Dict[str, str] = {}

        if job_analysis and job_analysis.capability_graph:
            for cap in job_analysis.capability_graph:
                valid_capability_ids.append(cap.id)
                capability_name_map[cap.id] = cap.name
        else:
            # Fallback capability ID if no JobAnalysisResult provided
            valid_capability_ids = ["cap_general_engineering"]
            capability_name_map = {"cap_general_engineering": "General Software Engineering"}

        if not resume_text or not resume_text.strip():
            return self._build_fallback_result("", valid_capability_ids, capability_name_map, start_time)

        if not self.model:
            logger.warning("Gemini model not initialized. Using fallback evidence extraction.")
            return self._build_fallback_result(resume_text, valid_capability_ids, capability_name_map, start_time)

        # Structure prompt with Module 1 capability IDs
        job_caps_json = json.dumps([{"id": c_id, "name": capability_name_map.get(c_id, c_id)} for c_id in valid_capability_ids], indent=2)
        prompt = RESUME_INTELLIGENCE_PROMPT.format(
            job_capabilities_json=job_caps_json,
            resume_text=resume_text
        )

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

            raw_evidence = data.get("evidence_objects", [])
            evidence_objects: List[EvidenceObject] = []

            for idx, item in enumerate(raw_evidence):
                cap_id = str(item.get("capability_id", ""))
                if cap_id not in valid_capability_ids and valid_capability_ids:
                    cap_id = valid_capability_ids[0]

                ev = EvidenceObject(
                    evidence_id=str(item.get("evidence_id", f"ev_resume_{idx+1:04d}")),
                    capability_id=cap_id,
                    source="Resume",
                    section=str(item.get("section", "Experience")),
                    location=str(item.get("location", "Project 1")),
                    quote=str(item.get("quote", "Demonstrated capability claim.")),
                    engineering_decision=str(item.get("engineering_decision", "Standard implementation pattern")),
                    ownership=str(item.get("ownership", "Individual")),
                    complexity=str(item.get("complexity", "Medium")),
                    impact=str(item.get("impact", "Demonstrated evidence")),
                    confidence=float(item.get("confidence", 80.0)),
                    verification_status=str(item.get("verification_status", "Resume Claim")),
                    generated_from=list(item.get("generated_from", []))
                )
                evidence_objects.append(ev)

            # Extract metrics
            raw_metrics = data.get("resume_metrics", [])
            resume_metrics: List[ResumeMetric] = []
            for m in raw_metrics:
                c_id = str(m.get("capability_id", valid_capability_ids[0]))
                if c_id not in valid_capability_ids:
                    c_id = valid_capability_ids[0]
                resume_metrics.append(
                    ResumeMetric(
                        metric=str(m.get("metric", "")),
                        context=str(m.get("context", "")),
                        capability_id=c_id
                    )
                )

            if not resume_metrics:
                resume_metrics = EvidenceExtractor.extract_metrics_from_text(resume_text, valid_capability_ids)

            # Compute summaries
            confidence_summary = ConfidenceEngine.compute_summary(evidence_objects)
            capability_mapping = CapabilityMapper.map_evidence_to_capabilities(evidence_objects, valid_capability_ids, capability_name_map)
            ownership_summary = CapabilityMapper.compute_ownership_summary(evidence_objects)
            validation_report = SchemaValidator.validate_evidence_objects(evidence_objects, valid_capability_ids)

            quantified_count = sum(1 for e in evidence_objects if e.verification_status == "Quantified Claim") + len(resume_metrics)

            candidate_summary = CandidateSummary(
                candidate_name=str(data.get("candidate_name", "Candidate")),
                detected_level=str(data.get("detected_level", "Intermediate")),
                total_claims=len(evidence_objects),
                total_quantified_claims=quantified_count
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return ResumeAnalysisResult(
                metadata=Metadata(
                    processing_time_ms=elapsed_ms,
                    model=self.model_name
                ),
                candidate_summary=candidate_summary,
                evidence_objects=evidence_objects,
                resume_metrics=resume_metrics,
                ownership_summary=ownership_summary,
                capability_mapping=capability_mapping,
                confidence_summary=confidence_summary,
                validation_report=validation_report
            )

        except Exception as e:
            logger.error(f"Error parsing resume via LLM: {str(e)}")
            return self._build_fallback_result(resume_text, valid_capability_ids, capability_name_map, start_time)

    def _build_fallback_result(
        self,
        resume_text: str,
        valid_capability_ids: List[str],
        capability_name_map: Dict[str, str],
        start_time: float
    ) -> ResumeAnalysisResult:
        evidence_objects = EvidenceExtractor.fallback_extract_evidence(resume_text, valid_capability_ids, capability_name_map)
        resume_metrics = EvidenceExtractor.extract_metrics_from_text(resume_text, valid_capability_ids)

        confidence_summary = ConfidenceEngine.compute_summary(evidence_objects)
        capability_mapping = CapabilityMapper.map_evidence_to_capabilities(evidence_objects, valid_capability_ids, capability_name_map)
        ownership_summary = CapabilityMapper.compute_ownership_summary(evidence_objects)
        validation_report = SchemaValidator.validate_evidence_objects(evidence_objects, valid_capability_ids)

        quantified_count = sum(1 for e in evidence_objects if e.verification_status == "Quantified Claim") + len(resume_metrics)

        candidate_summary = CandidateSummary(
            candidate_name="Candidate",
            detected_level="Intermediate" if len(evidence_objects) > 3 else "Student",
            total_claims=len(evidence_objects),
            total_quantified_claims=quantified_count
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ResumeAnalysisResult(
            metadata=Metadata(
                processing_time_ms=elapsed_ms,
                model="fallback-normalizer"
            ),
            candidate_summary=candidate_summary,
            evidence_objects=evidence_objects,
            resume_metrics=resume_metrics,
            ownership_summary=ownership_summary,
            capability_mapping=capability_mapping,
            confidence_summary=confidence_summary,
            validation_report=validation_report
        )
