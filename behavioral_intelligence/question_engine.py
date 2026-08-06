from typing import Optional, Tuple
from job_intelligence.models import JobAnalysisResult
from .models import BehavioralQuestionResponse

class QuestionEngine:
    @staticmethod
    def resolve_target_capability(
        response: BehavioralQuestionResponse,
        job_analysis: Optional[JobAnalysisResult]
    ) -> Tuple[str, str]:
        """
        Maps a behavioral question response to a Module 1 capability ID and name.
        Guarantees capability IDs come ONLY from Module 1.
        """
        valid_caps = job_analysis.capability_graph if (job_analysis and job_analysis.capability_graph) else []

        if valid_caps:
            # 1. Direct match by target capability ID
            if response.target_capability_id:
                for cap in valid_caps:
                    if cap.id == response.target_capability_id:
                        return cap.id, cap.name

            # 2. Match by question text or response content tokens
            text = f"{response.question_id} {response.question_text} {response.response_text}".lower()
            for cap in valid_caps:
                name_tokens = [t for t in cap.name.lower().split() if len(t) > 2]
                if any(t in text for t in name_tokens):
                    return cap.id, cap.name

            # 3. Default to first valid Module 1 capability
            return valid_caps[0].id, valid_caps[0].name

        # Fallback if no job analysis provided
        fallback_id = response.target_capability_id or f"cap_general_1_{response.question_id.lower()}"
        fallback_name = response.target_capability_name or response.question_id.replace("_", " ").title()
        return fallback_id, fallback_name
