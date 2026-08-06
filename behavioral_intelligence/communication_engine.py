from typing import List
from .models import CommunicationDetail, BehavioralQuestionResponse

class CommunicationEngine:
    @classmethod
    def evaluate_communication(cls, responses: List[BehavioralQuestionResponse]) -> CommunicationDetail:
        if not responses:
            return CommunicationDetail(
                clarity_score=70.0,
                structure_score=70.0,
                conciseness_score=70.0,
                overall_communication_score=70.0
            )

        total_clarity = 0.0
        total_structure = 0.0
        total_conciseness = 0.0

        for r in responses:
            text = r.response_text or ""
            words = text.split()
            word_count = len(words)

            # Conciseness: 40 - 250 words is optimal
            if 40 <= word_count <= 250:
                conciseness = 90.0
            elif word_count > 250:
                conciseness = 65.0
            else:
                conciseness = 45.0

            # Clarity: complete sentences & length balance
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            clarity = 85.0 if len(sentences) >= 3 else (70.0 if len(sentences) >= 1 else 40.0)

            # Structure: transitions and formatting
            structure = 85.0 if len(sentences) >= 4 else 75.0

            total_clarity += clarity
            total_structure += structure
            total_conciseness += conciseness

        n = len(responses)
        avg_c = round(total_clarity / n, 2)
        avg_s = round(total_structure / n, 2)
        avg_cn = round(total_conciseness / n, 2)

        overall = round((avg_c * 0.40) + (avg_s * 0.35) + (avg_cn * 0.25), 2)

        return CommunicationDetail(
            clarity_score=avg_c,
            structure_score=avg_s,
            conciseness_score=avg_cn,
            overall_communication_score=overall
        )
