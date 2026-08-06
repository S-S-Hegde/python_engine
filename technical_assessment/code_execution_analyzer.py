from typing import List, Dict, Any
from .models import SubmissionQuestionItem, ExecutionResult, TestCaseResult

class CodeExecutionAnalyzer:
    @classmethod
    def analyze_submission_execution(
        cls,
        submission_id: str,
        question: SubmissionQuestionItem
    ) -> ExecutionResult:
        """
        Analyzes test case outcomes, compilation errors, and runtime errors.
        """
        if question.compilation_error:
            return ExecutionResult(
                submission_id=submission_id,
                question_id=question.question_id,
                status="CompileError",
                passed_tests_count=0,
                total_tests_count=len(question.test_cases) or 1,
                pass_rate=0.0,
                public_tests_passed=False,
                hidden_tests_passed=False,
                edge_cases_passed=False,
                compilation_error=question.compilation_error
            )

        if question.runtime_error:
            return ExecutionResult(
                submission_id=submission_id,
                question_id=question.question_id,
                status="RuntimeError",
                passed_tests_count=0,
                total_tests_count=len(question.test_cases) or 1,
                pass_rate=0.0,
                public_tests_passed=False,
                hidden_tests_passed=False,
                edge_cases_passed=False,
                runtime_error=question.runtime_error
            )

        test_results: List[TestCaseResult] = []
        passed_count = 0
        total_count = len(question.test_cases)

        pub_passed = True
        hid_passed = True
        edge_passed = True

        for idx, tc in enumerate(question.test_cases):
            tid = tc.get("test_id", f"tc_{idx+1:02d}")
            ttype = tc.get("test_type", "public")
            passed = bool(tc.get("passed", False))

            if passed:
                passed_count += 1
            else:
                if ttype == "public":
                    pub_passed = False
                elif ttype == "hidden":
                    hid_passed = False
                elif ttype == "edge_case":
                    edge_passed = False

            test_results.append(
                TestCaseResult(
                    test_id=tid,
                    test_type=ttype,
                    passed=passed,
                    input_data=str(tc.get("input", "")),
                    expected_output=str(tc.get("expected", "")),
                    actual_output=str(tc.get("actual", "")),
                    error_message=tc.get("error_message"),
                    execution_time_ms=float(tc.get("execution_time_ms", 10.0))
                )
            )

        pass_rate = round((passed_count / max(1, total_count)) * 100.0, 2) if total_count > 0 else (100.0 if not question.runtime_error else 0.0)
        status = "Passed" if pass_rate == 100.0 else ("Executed" if pass_rate > 0 else "Failed")

        return ExecutionResult(
            submission_id=submission_id,
            question_id=question.question_id,
            status=status,
            passed_tests_count=passed_count,
            total_tests_count=total_count or (1 if pass_rate > 0 else 0),
            pass_rate=pass_rate,
            public_tests_passed=pub_passed,
            hidden_tests_passed=hid_passed,
            edge_cases_passed=edge_passed,
            test_cases=test_results
        )
