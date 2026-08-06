import time
from typing import Dict, Any, List, Optional
from job_intelligence.models import JobAnalysisResult
from resume_intelligence.models import ResumeAnalysisResult
from repository_intelligence.models import RepositoryAnalysisResult
from .models import (
    EvidenceFusionResult,
    UnifiedCapabilityProfile,
    ReliabilitySummary,
    ConfidenceSummary,
    ContradictionReport,
    MissingEvidenceReport,
    ContradictionItem,
    Metadata
)
from .evidence_merger import EvidenceMerger
from .contradiction_detector import ContradictionDetector
from .reliability_engine import ReliabilityEngine
from .confidence_engine import ConfidenceEngine
from .models import ValidationReport

class SchemaValidator:
    @staticmethod
    def validate_fusion_result(
        profiles: List[UnifiedCapabilityProfile],
        valid_job_capability_ids: List[str]
    ) -> ValidationReport:
        warnings: List[str] = []
        seen_profile_ids: set = set()

        for profile in profiles:
            cap_id = profile.capability_id
            if not cap_id or not isinstance(cap_id, str):
                warnings.append("Profile contains empty or non-string capability_id.")

            if cap_id in seen_profile_ids:
                warnings.append(f"Duplicate capability profile ID detected: '{cap_id}'.")
            seen_profile_ids.add(cap_id)

            if valid_job_capability_ids and cap_id not in valid_job_capability_ids:
                warnings.append(f"Capability ID '{cap_id}' does not exist in Module 1 Job Analysis.")

            if not (0.0 <= profile.merged_confidence <= 100.0):
                warnings.append(f"Profile '{cap_id}' merged_confidence {profile.merged_confidence} out of bounds [0, 100]. Clamping.")
                profile.merged_confidence = max(0.0, min(100.0, profile.merged_confidence))

            if not (0.0 <= profile.reliability <= 100.0):
                warnings.append(f"Profile '{cap_id}' reliability {profile.reliability} out of bounds [0, 100]. Clamping.")
                profile.reliability = max(0.0, min(100.0, profile.reliability))

        return ValidationReport(
            is_valid=len(warnings) == 0,
            total_profiles_validated=len(profiles),
            warnings=warnings
        )


class FusionEngine:
    def __init__(self, custom_reliability_weights: Dict[str, float] = None):
        self.custom_weights = custom_reliability_weights

    def fuse_evidence(
        self,
        job_analysis: JobAnalysisResult,
        resume_analysis: Optional[ResumeAnalysisResult] = None,
        repository_analysis: Optional[RepositoryAnalysisResult] = None,
        technical_assessment: Optional[Dict[str, Any]] = None,
        behavioral_assessment: Optional[Dict[str, Any]] = None,
        professional_experience: Optional[List[Dict[str, Any]]] = None
    ) -> EvidenceFusionResult:
        start_time = time.perf_counter()

        valid_capability_ids: List[str] = []
        capability_name_map: Dict[str, str] = {}

        if job_analysis and job_analysis.capability_graph:
            for cap in job_analysis.capability_graph:
                valid_capability_ids.append(cap.id)
                capability_name_map[cap.id] = cap.name
        else:
            valid_capability_ids = ["cap_general_engineering"]
            capability_name_map = {"cap_general_engineering": "General Software Engineering"}

        # Extract evidence maps per capability ID
        resume_ev_map: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in valid_capability_ids}
        repo_ev_map: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in valid_capability_ids}
        tech_ev_map: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in valid_capability_ids}
        beh_ev_map: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in valid_capability_ids}
        prof_ev_map: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in valid_capability_ids}

        # 1. Collect Resume Evidence
        if resume_analysis and resume_analysis.evidence_objects:
            for ev in resume_analysis.evidence_objects:
                cid = ev.capability_id if ev.capability_id in resume_ev_map else valid_capability_ids[0]
                resume_ev_map[cid].append(ev.model_dump())

        # 2. Collect Repository Evidence
        repo_summary_info = {}
        originality_verdict = "Organic Development"
        if repository_analysis:
            repo_summary_info = repository_analysis.architecture_summary.model_dump()
            originality_verdict = repository_analysis.originality_report.verdict
            if repository_analysis.evidence_objects:
                for ev in repository_analysis.evidence_objects:
                    cid = ev.capability_id if ev.capability_id in repo_ev_map else valid_capability_ids[0]
                    repo_ev_map[cid].append(ev.model_dump())

        # 3. Collect Technical Assessment Evidence
        if technical_assessment:
            for item in technical_assessment.get("evidence", []):
                cid = item.get("capability_id", valid_capability_ids[0])
                if cid in tech_ev_map:
                    tech_ev_map[cid].append(item)

        # 4. Collect Behavioral Assessment Evidence
        if behavioral_assessment:
            for item in behavioral_assessment.get("evidence", []):
                cid = item.get("capability_id", valid_capability_ids[0])
                if cid in beh_ev_map:
                    beh_ev_map[cid].append(item)

        # 5. Collect Professional Experience
        if professional_experience:
            for item in professional_experience:
                cid = item.get("capability_id", valid_capability_ids[0])
                if cid in prof_ev_map:
                    prof_ev_map[cid].append(item)

        # Build Unified Capability Profiles
        profiles: List[UnifiedCapabilityProfile] = []
        all_contradictions: List[ContradictionItem] = []
        missing_capabilities: List[str] = []

        for cid in valid_capability_ids:
            cap_name = capability_name_map.get(cid, cid)

            # Deduplicate evidence objects per source
            merged_res = EvidenceMerger.deduplicate_and_group_evidence(resume_ev_map[cid])
            merged_repo = EvidenceMerger.deduplicate_and_group_evidence(repo_ev_map[cid])
            merged_tech = EvidenceMerger.deduplicate_and_group_evidence(tech_ev_map[cid])
            merged_beh = EvidenceMerger.deduplicate_and_group_evidence(beh_ev_map[cid])
            merged_prof = EvidenceMerger.deduplicate_and_group_evidence(prof_ev_map[cid])

            missing_sources: List[str] = []
            if not merged_res:
                missing_sources.append("Resume Claim")
            if not merged_repo:
                missing_sources.append("Repository Verification")
            if not merged_tech:
                missing_sources.append("Technical Assessment")

            if len(missing_sources) == 3:
                missing_capabilities.append(cid)

            profile = UnifiedCapabilityProfile(
                capability_id=cid,
                capability_name=cap_name,
                resume_evidence=merged_res,
                repository_evidence=merged_repo,
                assessment_evidence=merged_tech,
                behavioral_evidence=merged_beh,
                professional_evidence=merged_prof,
                missing_evidence=missing_sources
            )

            # Detect Contradictions
            contradictions = ContradictionDetector.detect_contradictions(
                profile=profile,
                repo_summary=repo_summary_info,
                originality_verdict=originality_verdict
            )
            profile.contradictions = contradictions
            all_contradictions.extend(contradictions)

            # Calculate Reliability & Confidence
            profile.reliability = ReliabilityEngine.calculate_capability_reliability(profile, self.custom_weights)
            ConfidenceEngine.compute_profile_confidence_and_status(profile)

            profiles.append(profile)

        # Compute Summaries
        reliability_summary = ReliabilityEngine.compute_summary(profiles, self.custom_weights)
        confidence_summary = ConfidenceEngine.compute_summary(profiles)

        critical_count = sum(1 for c in all_contradictions if c.severity in ["High", "Critical"])
        contradiction_report = ContradictionReport(
            total_contradictions_found=len(all_contradictions),
            critical_contradictions_count=critical_count,
            contradictions=all_contradictions
        )

        missing_report = MissingEvidenceReport(
            total_missing_gaps=len(missing_capabilities),
            missing_capabilities=missing_capabilities
        )

        validation_report = SchemaValidator.validate_fusion_result(profiles, valid_capability_ids)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return EvidenceFusionResult(
            metadata=Metadata(processing_time_ms=elapsed_ms),
            capability_profiles=profiles,
            reliability_summary=reliability_summary,
            confidence_summary=confidence_summary,
            contradiction_report=contradiction_report,
            missing_evidence_report=missing_report,
            validation_report=validation_report
        )
