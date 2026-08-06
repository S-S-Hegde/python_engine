import re
from typing import List
from .models import LeadershipDetail, BehavioralQuestionResponse

class LeadershipEngine:
    @classmethod
    def evaluate_leadership(cls, responses: List[BehavioralQuestionResponse]) -> LeadershipDetail:
        if not responses:
            return LeadershipDetail(
                collaboration_score=70.0,
                conflict_resolution_score=70.0,
                mentorship_score=70.0,
                overall_leadership_score=70.0
            )

        total_collab = 0.0
        total_conflict = 0.0
        total_mentor = 0.0

        for r in responses:
            text_lower = (r.response_text or "").lower()

            # Collaboration
            if re.search(r"\b(collaborated|partnered|teamwork|cross-functional|aligned with team|together|shared goal)\b", text_lower):
                collab = 90.0
            else:
                collab = 70.0

            # Conflict resolution
            if re.search(r"\b(disagreement|conflict|differing opinions|resolved|consensus|compromise|listened to feedback|aligned)\b", text_lower):
                conflict = 90.0
            else:
                conflict = 65.0

            # Mentorship
            if re.search(r"\b(mentored|guided|coached|helped junior|shared knowledge|onboarded|supported team)\b", text_lower):
                mentor = 90.0
            else:
                mentor = 60.0

            total_collab += collab
            total_conflict += conflict
            total_mentor += mentor

        n = len(responses)
        avg_collab = round(total_collab / n, 2)
        avg_conflict = round(total_conflict / n, 2)
        avg_mentor = round(total_mentor / n, 2)

        overall = round((avg_collab * 0.40) + (avg_conflict * 0.35) + (avg_mentor * 0.25), 2)

        return LeadershipDetail(
            collaboration_score=avg_collab,
            conflict_resolution_score=avg_conflict,
            mentorship_score=avg_mentor,
            overall_leadership_score=overall
        )
