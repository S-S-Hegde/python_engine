from typing import Dict, Any, List
from .models import RecruiterDecisionResult, ValidationReport

class SchemaValidator:
    def validate_recruiter_decision(self, data: RecruiterDecisionResult) -> ValidationReport:
        warnings = []
        is_valid = True
        
        # Validate Override Consistency
        ds = data.decision_summary
        if ds.decision_override:
            if not ds.override_reason:
                warnings.append("Decision override flag is True but override_reason is missing.")
                is_valid = False
            if not ds.override_by:
                warnings.append("Decision override flag is True but override_by is missing.")
                is_valid = False
                
        # Validate Explainability Scoring bounds
        if not (0.0 <= data.explanation.explainability_score <= 100.0):
            warnings.append("Explainability score out of bounds.")
            is_valid = False
            
        # Validate Evidence Traceability
        if not data.explanation.why_hire and not data.explanation.why_not_hire:
            warnings.append("No explanations provided for why_hire or why_not_hire.")
            
        for statement in data.explanation.why_hire:
            if not statement.supported_by:
                warnings.append(f"Statement '{statement.statement}' in why_hire lacks evidence references.")
                is_valid = False
                
        for statement in data.explanation.why_not_hire:
            if not statement.supported_by:
                warnings.append(f"Statement '{statement.statement}' in why_not_hire lacks evidence references.")
                is_valid = False
                
        # Validate Audit Trail
        at = data.audit_trail
        if not at.pipeline_version:
            warnings.append("Audit trail missing pipeline_version.")
            is_valid = False
        if not at.policy_version:
            warnings.append("Audit trail missing policy_version.")
            is_valid = False
        if not at.modules_used:
            warnings.append("Audit trail missing modules_used.")
            is_valid = False
            
        return ValidationReport(
            is_valid=is_valid,
            warnings=warnings
        )
