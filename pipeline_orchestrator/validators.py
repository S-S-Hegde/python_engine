from typing import List, Dict, Any
from .models import PipelineResponse, PipelineStatus, ExecutionStatus

class SchemaValidator:
    def validate_pipeline_response(self, response: PipelineResponse) -> bool:
        # Validate Request ID
        if not response.request_id:
            return False
            
        # Validate Execution Mode and Status
        if not response.execution_mode or not response.pipeline_status:
            return False
            
        # Verify Pipeline Execution Records
        if not response.pipeline_execution:
            return False
            
        # Ensure critical modules have an execution record
        expected_modules = [
            "Job Intelligence",
            "Evidence Fusion",
            "Capability Scoring",
            "Competency Intelligence",
            "Candidate Profile",
            "Trust Score Engine",
            "Recruiter Decision Engine"
        ]
        
        executed_module_names = [record.module for record in response.pipeline_execution]
        for mod in expected_modules:
            if mod not in executed_module_names:
                return False
                
        # Validate that skipped modules have a reason
        for record in response.pipeline_execution:
            if record.status == ExecutionStatus.SKIPPED and not record.reason:
                return False
            if record.status == ExecutionStatus.FAILED and not record.error:
                return False
                
        return True
