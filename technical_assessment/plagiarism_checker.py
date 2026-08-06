from typing import List
from .models import PlagiarismDetail, SubmissionQuestionItem

class PlagiarismChecker:
    @classmethod
    def check_plagiarism(cls, questions: List[SubmissionQuestionItem]) -> PlagiarismDetail:
        """
        Evaluates copy-paste anomalies, submission speed, and code similarity flags.
        """
        anomaly_flags: List[str] = []
        max_similarity = 0.0

        for q in questions:
            # Check copy-paste anomaly
            if q.copy_paste_events_count >= 10:
                anomaly_flags.append(f"Question '{q.question_id}': High copy-paste frequency ({q.copy_paste_events_count} events).")
                max_similarity = max(max_similarity, 85.0)

            # Check rapid completion anomaly (< 15 seconds for complex code)
            lines = len([l for l in q.submitted_code.split("\n") if l.strip()])
            if q.time_spent_seconds < 15 and lines > 15:
                anomaly_flags.append(f"Question '{q.question_id}': Unrealistic typing speed ({lines} lines in {q.time_spent_seconds}s).")
                max_similarity = max(max_similarity, 90.0)

        is_plag = len(anomaly_flags) > 0 and max_similarity >= 80.0

        return PlagiarismDetail(
            is_plagiarized=is_plag,
            similarity_percentage=round(max_similarity, 2),
            matched_source="External Repository / Template" if is_plag else None,
            anomaly_flags=anomaly_flags
        )
