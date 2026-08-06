from typing import Dict, Any, List
from .models import InterviewPlan, InterviewRecommendation

class InterviewEngine:
    def generate_interview_plan(self, trust_result: Dict[str, Any], profile_result: Dict[str, Any]) -> InterviewPlan:
        report = trust_result.get("report", {})
        weaknesses = report.get("weaknesses", [])
        missing_evidence = report.get("missing_evidence", [])
        interview_topics = report.get("recommended_interview_topics", [])
        
        focus_areas: List[InterviewRecommendation] = []
        
        # Target weaknesses and missing evidence
        for topic in interview_topics:
            focus_areas.append(InterviewRecommendation(
                priority="High",
                topic=topic,
                rationale="Identified as a critical area requiring deeper evaluation.",
                supported_by=["trust_report.recommended_interview_topics"]
            ))
            
        for missing in missing_evidence:
            focus_areas.append(InterviewRecommendation(
                priority="Medium",
                topic=missing,
                rationale="Missing verified evidence for this capability.",
                supported_by=["trust_report.missing_evidence"]
            ))
            
        # Compile standard questions based on role mapping (from Candidate Profile)
        role_metadata = profile_result.get("metadata", {})
        
        tech_questions = [
            "Walk me through the most complex architectural decision you made.",
            "How do you approach scaling systems under heavy load?"
        ]
        
        beh_questions = [
            "Tell me about a time you had to push back on a product requirement.",
            "Describe a situation where you had to lead a project with ambiguous requirements."
        ]
        
        return InterviewPlan(
            recommended_duration_minutes=60,
            focus_areas=focus_areas,
            technical_questions_to_ask=tech_questions,
            behavioral_questions_to_ask=beh_questions
        )
