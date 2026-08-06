from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class HireDecision(str, Enum):
    STRONG_HIRE = "Strong Hire"
    HIRE = "Hire"
    BORDERLINE = "Borderline"
    HOLD = "Hold"
    REJECT = "Reject"

class DecisionPolicyType(str, Enum):
    STARTUP = "Startup"
    ENTERPRISE = "Enterprise"
    INTERN = "Intern Hiring"
    SENIOR = "Senior Hiring"
    CUSTOM = "Custom"

class AuditTrail(BaseModel):
    pipeline_version: str = Field(default="2.0")
    decision_timestamp: str
    modules_used: List[str]
    policy_version: str
    schema_versions: Dict[str, str]

class SupportedStatement(BaseModel):
    statement: str
    supported_by: List[str]  # References to upstream evidence IDs or module summaries

class Explanation(BaseModel):
    why_hire: List[SupportedStatement]
    why_not_hire: List[SupportedStatement]
    unsupported_reasoning_count: int
    explainability_score: float = Field(..., ge=0.0, le=100.0)

class InterviewRecommendation(BaseModel):
    priority: str
    topic: str
    rationale: str
    supported_by: List[str]

class InterviewPlan(BaseModel):
    recommended_duration_minutes: int
    focus_areas: List[InterviewRecommendation]
    technical_questions_to_ask: List[str]
    behavioral_questions_to_ask: List[str]

class RiskAnalysis(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str
    critical_vulnerabilities: List[SupportedStatement]
    mitigations: List[str]

class ComparisonSummary(BaseModel):
    cohort_size: int
    percentile: float = Field(..., ge=0.0, le=100.0)
    ranking_position: int
    relative_strengths: List[str]
    relative_weaknesses: List[str]

class RankingMetrics(BaseModel):
    candidate_ranking_score: float = Field(..., ge=0.0, le=100.0)
    engineering_strength_ranking: float = Field(..., ge=0.0, le=100.0)
    engineering_weakness_ranking: float = Field(..., ge=0.0, le=100.0)
    capability_ranking: float = Field(..., ge=0.0, le=100.0)
    competency_ranking: float = Field(..., ge=0.0, le=100.0)

class DecisionSummary(BaseModel):
    ai_recommendation: HireDecision
    final_decision: HireDecision
    decision_override: bool
    override_reason: Optional[str] = None
    override_by: Optional[str] = None
    override_timestamp: Optional[str] = None
    decision_confidence: float = Field(..., ge=0.0, le=100.0)
    decision_policy_used: DecisionPolicyType

class ValidationReport(BaseModel):
    is_valid: bool
    warnings: List[str]

class RecruiterDecisionResult(BaseModel):
    metadata: Dict[str, Any]
    audit_trail: AuditTrail
    decision_summary: DecisionSummary
    explanation: Explanation
    ranking: RankingMetrics
    comparison_summary: Optional[ComparisonSummary] = None
    interview_plan: InterviewPlan
    risk_analysis: RiskAnalysis
    executive_summary: str
    validation_report: ValidationReport

class RecruiterDecisionRequestPayload(BaseModel):
    trust_score_result: Dict[str, Any]
    capability_scoring_result: Dict[str, Any]
    competency_intelligence_result: Dict[str, Any]
    candidate_profile_result: Dict[str, Any]
    policy_type: DecisionPolicyType = DecisionPolicyType.ENTERPRISE
    cohort_results: Optional[List[Dict[str, Any]]] = None # For multi-candidate ranking
    human_override: Optional[Dict[str, str]] = None # { "final_decision": "Hire", "override_reason": "...", "override_by": "..." }
