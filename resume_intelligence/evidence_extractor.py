import re
from typing import List, Dict, Any
from .models import EvidenceObject, ResumeMetric

class EvidenceExtractor:
    @staticmethod
    def extract_metrics_from_text(resume_text: str, valid_capability_ids: List[str]) -> List[ResumeMetric]:
        """Extracts quantifiable metrics (percentages, numbers, throughputs) from resume text."""
        metrics: List[ResumeMetric] = []
        if not resume_text or not valid_capability_ids:
            return metrics

        cap_id = valid_capability_ids[0]

        # Match numbers with %, years, daily, requests, users, latency, microservices, etc.
        pattern = re.compile(r'(\d+%\s*[\w\s]{2,30}|\d+\+?\s*(?:years?|daily|monthly|requests|users|microservices|ms|seconds?|percent| speedup)[\w\s]{0,30})', re.IGNORECASE)

        for match in pattern.finditer(resume_text):
            metric_str = match.group(1).strip()
            if len(metric_str) > 2:
                metrics.append(
                    ResumeMetric(
                        metric=metric_str,
                        context=resume_text[max(0, match.start()-20):min(len(resume_text), match.end()+30)].strip(),
                        capability_id=cap_id
                    )
                )

        return metrics

    @classmethod
    def fallback_extract_evidence(
        cls,
        resume_text: str,
        valid_capability_ids: List[str],
        capability_name_map: Dict[str, str]
    ) -> List[EvidenceObject]:
        """Fallback rule-based evidence extractor for offline/unit-test execution."""
        evidence_list: List[EvidenceObject] = []
        if not valid_capability_ids:
            valid_capability_ids = ["cap_general_engineering"]

        if not resume_text or not resume_text.strip():
            # Emit single placeholder for empty text
            cap_id = valid_capability_ids[0]
            evidence_list.append(
                EvidenceObject(
                    evidence_id="ev_resume_0001",
                    capability_id=cap_id,
                    source="Resume",
                    section="General",
                    location="Resume Text",
                    quote="No resume text provided.",
                    engineering_decision="None",
                    ownership="Unknown",
                    complexity="Low",
                    impact="None",
                    confidence=50.0,
                    verification_status="Resume Claim",
                    generated_from=["No resume text provided."]
                )
            )
            return evidence_list

        # Split on newlines, periods, or project markers
        raw_chunks = re.split(r'[\n.]|(?=Project \d+:)', resume_text)
        lines = [c.strip() for c in raw_chunks if c.strip() and len(c.strip()) > 8]

        if not lines:
            lines = [resume_text.strip()]

        for idx, line in enumerate(lines[:10]):
            cap_id = valid_capability_ids[idx % len(valid_capability_ids)]
            cap_name = capability_name_map.get(cap_id, "Capability")

            has_metric = bool(re.search(r'\d+', line))
            verification_status = "Quantified Claim" if has_metric else "Resume Claim"
            complexity = "High" if "architect" in line.lower() or "designed" in line.lower() or has_metric else "Medium"
            ownership = "Individual" if any(verb in line.lower() for verb in ["built", "developed", "created", "sole", "architected"]) else "Primary Contributor"

            ev = EvidenceObject(
                evidence_id=f"ev_resume_{idx+1:04d}",
                capability_id=cap_id,
                source="Resume",
                section="Experience / Projects",
                location=f"Section {idx+1}",
                quote=line[:200],
                engineering_decision=f"Implemented {cap_name} solution using standard patterns",
                ownership=ownership,
                complexity=complexity,
                impact=f"Demonstrated evidence for {cap_name}",
                confidence=85.0 if has_metric else 75.0,
                verification_status=verification_status,
                generated_from=[line[:100]]
            )
            evidence_list.append(ev)

        return evidence_list
