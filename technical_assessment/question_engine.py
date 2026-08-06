from typing import List, Dict, Any, Optional
from job_intelligence.models import JobAnalysisResult
from .models import SubmissionQuestionItem

class QuestionEngine:
    @staticmethod
    def resolve_target_capability(
        question: SubmissionQuestionItem,
        job_analysis: Optional[JobAnalysisResult]
    ) -> tuple[str, str]:
        """
        Maps a submitted question to a Module 1 capability ID and name.
        Guarantees capability IDs come ONLY from Module 1.
        """
        valid_caps = job_analysis.capability_graph if (job_analysis and job_analysis.capability_graph) else []

        if valid_caps:
            # 1. Direct match by capability ID
            if question.target_capability_id:
                for cap in valid_caps:
                    if cap.id == question.target_capability_id:
                        return cap.id, cap.name

            # 2. Fuzzy match by name or keywords
            code_text = f"{question.question_id} {question.submitted_code}".lower()
            for cap in valid_caps:
                name_tokens = [t for t in cap.name.lower().split() if len(t) > 2]
                if any(t in code_text for t in name_tokens):
                    return cap.id, cap.name

            # 3. Default to first valid Module 1 capability
            return valid_caps[0].id, valid_caps[0].name

        # Fallback if no job analysis provided
        fallback_id = question.target_capability_id or f"cap_general_1_{question.question_id.lower()}"
        fallback_name = question.target_capability_name or question.question_id.replace("_", " ").title()
        return fallback_id, fallback_name
