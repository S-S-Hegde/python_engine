from typing import Dict, Any, List
from .models import OriginalityReport
from .git_analyzer import GitAnalyzer

class OriginalityChecker:
    @staticmethod
    def evaluate_originality(
        repo_data: Dict[str, Any],
        commits: List[Dict[str, Any]]
    ) -> OriginalityReport:
        is_fork = bool(repo_data.get("fork", False))
        git_stats = GitAnalyzer.analyze_commit_history(commits)

        is_single_day = git_stats["is_single_day_dump"]
        commit_count = git_stats["commit_count"]
        unique_days = git_stats["unique_commit_days"]
        quality_ratio = git_stats["quality_commit_ratio"]

        # Base score
        score = 100.0
        verdict = "Organic Development"

        if is_fork:
            score -= 60.0
            verdict = "Forked Repo"
        elif is_single_day:
            score -= 35.0
            verdict = "Single-Day Dump"
        elif commit_count < 3:
            score -= 15.0

        if quality_ratio > 0.5:
            score += 10.0

        score = round(min(100.0, max(0.0, score)), 2)

        return OriginalityReport(
            is_fork=is_fork,
            is_single_day_dump=is_single_day,
            commit_count=commit_count,
            unique_commit_days=unique_days,
            quality_commit_ratio=quality_ratio,
            originality_score=score,
            verdict=verdict
        )
