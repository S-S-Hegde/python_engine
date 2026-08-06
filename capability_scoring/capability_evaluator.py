from typing import Dict, Any, List

class CapabilityEvaluator:
    COMPLEXITY_MAP = {
        "very high": 100.0,
        "high": 85.0,
        "medium": 65.0,
        "low": 40.0,
        "very low": 20.0
    }

    @classmethod
    def calculate_coverage_score(cls, profile_dict: Dict[str, Any]) -> float:
        """Calculates evidence coverage across the 5 possible evidence channels."""
        channels = 0
        if profile_dict.get("repository_evidence"):
            channels += 1
        if profile_dict.get("assessment_evidence"):
            channels += 1
        if profile_dict.get("resume_evidence"):
            channels += 1
        if profile_dict.get("professional_evidence"):
            channels += 1
        if profile_dict.get("behavioral_evidence"):
            channels += 1

        score = (channels / 5.0) * 100.0
        return round(min(100.0, max(0.0, score)), 2)

    @classmethod
    def calculate_engineering_depth_score(cls, profile_dict: Dict[str, Any]) -> float:
        """Calculates engineering depth based on code quote length and decision details."""
        repo_ev = profile_dict.get("repository_evidence", [])
        resume_ev = profile_dict.get("resume_evidence", [])

        if not repo_ev and not resume_ev:
            return 0.0

        depth = 50.0
        if repo_ev:
            depth += 30.0
            for item in repo_ev:
                if len(str(item.get("quote", ""))) > 40:
                    depth += 10.0
                    break

        if resume_ev:
            depth += 10.0

        return round(min(100.0, max(0.0, depth)), 2)

    @classmethod
    def calculate_complexity_score(cls, profile_dict: Dict[str, Any]) -> float:
        """Calculates average complexity score from evidence objects."""
        all_ev = (
            profile_dict.get("repository_evidence", []) +
            profile_dict.get("assessment_evidence", []) +
            profile_dict.get("resume_evidence", [])
        )

        if not all_ev:
            return 30.0

        scores: List[float] = []
        for ev in all_ev:
            cx = str(ev.get("complexity", "Medium")).lower()
            scores.append(cls.COMPLEXITY_MAP.get(cx, 65.0))

        avg_cx = sum(scores) / len(scores)
        return round(min(100.0, max(0.0, avg_cx)), 2)

    @classmethod
    def calculate_consistency_score(cls, profile_dict: Dict[str, Any]) -> float:
        """Calculates consistency score penalizing for contradictions."""
        contradictions = profile_dict.get("contradictions", [])
        if not contradictions:
            return 100.0

        penalty = sum(15.0 for _ in contradictions)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)
