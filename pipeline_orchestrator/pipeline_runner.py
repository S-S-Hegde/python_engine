import time
from typing import Callable, Any, Dict, List, Tuple
from .models import ExecutionRecord, ExecutionStatus
from .error_handler import PipelineErrorHandler

class PipelineRunner:
    def __init__(self):
        self.error_handler = PipelineErrorHandler()
        self.execution_records: List[ExecutionRecord] = []

    def execute_module(
        self,
        module_name: str,
        skip_condition: bool,
        skip_reason: str,
        execution_func: Callable[[], Any]
    ) -> Any:
        """
        Wraps module execution, handles skipping, errors, and timing.
        """
        if skip_condition:
            self.execution_records.append(
                ExecutionRecord(
                    module=module_name,
                    status=ExecutionStatus.SKIPPED,
                    reason=skip_reason
                )
            )
            return None
            
        start_time = time.time()
        try:
            result = execution_func()
            exec_time = round((time.time() - start_time) * 1000, 2)
            
            self.execution_records.append(
                ExecutionRecord(
                    module=module_name,
                    status=ExecutionStatus.COMPLETED,
                    execution_time_ms=exec_time
                )
            )
            return result
        except Exception as e:
            error_msg = self.error_handler.handle_module_failure(module_name, e)
            self.execution_records.append(
                ExecutionRecord(
                    module=module_name,
                    status=ExecutionStatus.FAILED,
                    error=error_msg
                )
            )
            return None

    def get_records(self) -> List[ExecutionRecord]:
        return self.execution_records
