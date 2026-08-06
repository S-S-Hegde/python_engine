import re
from typing import List
from .models import OwnershipDetail, BehavioralQuestionResponse

class OwnershipEngine:
    @classmethod
    def evaluate_ownership(cls, responses: List[BehavioralQuestionResponse]) -> OwnershipDetail:
        if not responses:
            return OwnershipDetail(
                accountability_score=70.0,
                blame_shifting_detected=False,
                learning_mindset_score=70.0,
                overall_ownership_score=70.0
            )

        blame_shifting_flag = False
        total_accountability = 0.0
        total_learning = 0.0

        for r in responses:
            text_lower = (r.response_text or "").lower()

            # Check blame shifting
            if re.search(r"\b(not my fault|they failed|others broke|management forced|it was their mistake|blame|wasn't my responsibility)\b", text_lower):
                blame_shifting_flag = True
                acc = 30.0
            elif re.search(r"\b(i took responsibility|my mistake|i owned|i stepped up|i fixed|i ensured|i held myself accountable)\b", text_lower):
                acc = 95.0
            elif re.search(r"\b(i|my|me)\b", text_lower):
                acc = 75.0
            else:
                acc = 50.0

            # Check learning mindset
            if re.search(r"\b(learned|improved|retrospective|feedback|growth|prevent in the future|next time|takeaway)\b", text_lower):
                learn = 90.0
            else:
                learn = 65.0

            total_accountability += acc
            total_learning += learn

        n = len(responses)
        avg_acc = round(total_accountability / n, 2)
        avg_learn = round(total_learning / n, 2)

        pen = 25.0 if blame_shifting_flag else 0.0
        overall = round(max(0.0, min(100.0, ((avg_acc * 0.60) + (avg_learn * 0.40)) - pen)), 2)

        return OwnershipDetail(
            accountability_score=avg_acc,
            blame_shifting_detected=blame_shifting_flag,
            learning_mindset_score=avg_learn,
            overall_ownership_score=overall
        )
