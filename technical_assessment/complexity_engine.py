import re
from .models import ComplexityDetail

class ComplexityEngine:
    @classmethod
    def analyze_complexity(cls, code: str) -> ComplexityDetail:
        """
        Analyzes time and space complexity based on AST / structural loop detection.
        """
        clean_code = re.sub(r"//.*|/\*[\s\S]*?\*/", "", code)
        clean_code = re.sub(r"#.*", "", clean_code)

        # Detect nested loops
        for_count = len(re.findall(r"\b(for|while)\b", clean_code))
        has_nested = bool(re.search(r"\b(for|while)\b[\s\S]*?\b(for|while)\b", clean_code))
        has_log = bool(re.search(r"\b(binarySearch|log|pivot|partition)\b", clean_code, re.IGNORECASE))
        has_recursion = bool(re.search(r"\b(recursion|return\s+\w+\s*\()", clean_code))

        if has_nested or for_count >= 2:
            time_comp = "O(n^2)"
            score = 60.0
            explanation = "Nested loops detected resulting in quadratic O(n^2) time complexity."
        elif has_log:
            time_comp = "O(n log n)"
            score = 85.0
            explanation = "Logarithmic partitioning detected resulting in optimal O(n log n) time complexity."
        elif for_count == 1:
            time_comp = "O(n)"
            score = 80.0
            explanation = "Single linear loop iteration detected resulting in O(n) time complexity."
        elif has_recursion:
            time_comp = "O(2^n)"
            score = 50.0
            explanation = "Recursive branching detected resulting in exponential O(2^n) time complexity."
        else:
            time_comp = "O(1)"
            score = 95.0
            explanation = "Constant time operations detected resulting in O(1) time complexity."

        # Space complexity
        has_structures = bool(re.search(r"\b(map|set|array|list|dict|new\s+Array|new\s+Map|\[\]|\{\})\b", clean_code, re.IGNORECASE))
        space_comp = "O(n)" if has_structures else "O(1)"

        return ComplexityDetail(
            time_complexity=time_comp,
            space_complexity=space_comp,
            complexity_score=score,
            explanation=explanation
        )
