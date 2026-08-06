from typing import List, Dict, Any, Set

class GitAnalyzer:
    @staticmethod
    def analyze_commit_history(commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes commit frequency, date distribution, and message length quality."""
        if not commits:
            return {
                "commit_count": 0,
                "unique_commit_days": 0,
                "is_single_day_dump": False,
                "quality_commit_ratio": 0.0
            }

        commit_dates: Set[str] = set()
        meaningful_count = 0

        for c in commits:
            commit_obj = c.get("commit", {})
            author_date = commit_obj.get("author", {}).get("date", "")[:10]
            if author_date:
                commit_dates.add(author_date)

            msg = commit_obj.get("message", "").strip()
            if len(msg) > 15 and not msg.lower().startswith(("initial commit", "update", "fix")):
                meaningful_count += 1

        total_commits = len(commits)
        unique_days = len(commit_dates)

        # Single-day dump detection: >= 15 commits occurring on <= 1 single day
        is_single_day_dump = (total_commits >= 15 and unique_days <= 1)
        quality_ratio = round(meaningful_count / max(total_commits, 1), 2)

        return {
            "commit_count": total_commits,
            "unique_commit_days": unique_days,
            "is_single_day_dump": is_single_day_dump,
            "quality_commit_ratio": quality_ratio
        }
