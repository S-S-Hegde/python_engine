import re
from typing import List
from .models import CodeQualityDetail, SubmissionQuestionItem

class AnswerEvaluator:
    @classmethod
    def evaluate_code_quality(cls, questions: List[SubmissionQuestionItem]) -> CodeQualityDetail:
        if not questions:
            return CodeQualityDetail(
                readability_score=70.0,
                modular_design_score=70.0,
                naming_conventions_score=70.0,
                error_handling_score=70.0,
                overall_quality_score=70.0
            )

        total_readability = 0.0
        total_modular = 0.0
        total_naming = 0.0
        total_error_handling = 0.0

        for q in questions:
            code = q.submitted_code or ""

            # Readability
            lines = [l.strip() for l in code.split("\n") if l.strip()]
            has_comments = any(l.startswith("//") or l.startswith("#") or "/*" in l for l in lines)
            readability = 85.0 if has_comments else (75.0 if len(lines) > 5 else 70.0)

            # Modular design
            has_functions = bool(re.search(r"\b(function|def|class|const\s+\w+\s*=\s*\()", code))
            modular = 90.0 if has_functions else 60.0

            # Naming conventions
            camel_or_snake = bool(re.search(r"\b[a-z]+[A-Z]\w*|\b[a-z]+_[a-z]+\b", code))
            naming = 85.0 if camel_or_snake else 70.0

            # Error handling
            has_try_catch = bool(re.search(r"\b(try|catch|throw|except|finally)\b", code))
            error_h = 95.0 if has_try_catch else 65.0

            total_readability += readability
            total_modular += modular
            total_naming += naming
            total_error_handling += error_h

        n = len(questions)
        avg_r = round(total_readability / n, 2)
        avg_m = round(total_modular / n, 2)
        avg_n = round(total_naming / n, 2)
        avg_e = round(total_error_handling / n, 2)

        overall = round((avg_r * 0.25) + (avg_m * 0.30) + (avg_n * 0.20) + (avg_e * 0.25), 2)

        return CodeQualityDetail(
            readability_score=avg_r,
            modular_design_score=avg_m,
            naming_conventions_score=avg_n,
            error_handling_score=avg_e,
            overall_quality_score=overall
        )
