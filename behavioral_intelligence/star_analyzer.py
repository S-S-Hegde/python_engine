import re
from .models import StarDetail, BehavioralQuestionResponse

class StarAnalyzer:
    @classmethod
    def analyze_star(cls, response: BehavioralQuestionResponse) -> StarDetail:
        """
        Evaluates STAR structure (Situation, Task, Action, Result) deterministically.
        """
        text = response.response_text or ""
        text_lower = text.lower()

        # Situation indicators
        has_s = bool(re.search(r"\b(when|at my previous|during|in a project|we were|the system had|the challenge was|context|background)\b", text_lower))

        # Task indicators
        has_t = bool(re.search(r"\b(my role|my task|i needed to|responsible for|objective|goal|assigned to|required to)\b", text_lower))

        # Action indicators
        has_a = bool(re.search(r"\b(i created|i implemented|i designed|i led|i decided|i refactored|i built|i communicated|i initiated|i resolved)\b", text_lower))

        # Result indicators
        has_r = bool(re.search(r"\b(result|outcome|increased|reduced|improved|saved|achieved|percent|%|successfully|delivered|impact)\b", text_lower))

        # Scoring
        score_components = [has_s, has_t, has_a, has_r]
        star_count = sum(1 for c in score_components if c)

        if star_count == 4:
            score = 100.0
            explanation = "Complete STAR structure present (Situation, Task, Action, Result)."
        elif star_count == 3:
            score = 75.0
            explanation = "Partial STAR structure present (3 of 4 elements detected)."
        elif star_count == 2:
            score = 50.0
            explanation = "Weak STAR structure present (2 of 4 elements detected)."
        elif star_count == 1:
            score = 25.0
            explanation = "Minimal STAR structure (only 1 element detected)."
        else:
            score = 10.0
            explanation = "Non-STAR or unstructured response."

        return StarDetail(
            question_id=response.question_id,
            has_situation=has_s,
            has_task=has_t,
            has_action=has_a,
            has_result=has_r,
            star_score=score,
            explanation=explanation
        )
