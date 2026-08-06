import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "capability-scoring-v2"
    pipeline_module: str = "Capability Scoring Engine"

class FormulaWeightConfig(BaseModel):
    w_coverage: float = 0.20
    w_depth: float = 0.20
    w_complexity: float = 0.15
    w_reliability: float = 0.20
    w_consistency: float = 0.15
    w_confidence: float = 0.10

class FormulaBreakdown(BaseModel):
    coverage_score: float = Field(..., ge=0.0, le=100.0)
    depth_score: float = Field(..., ge=0.0, le=100.0)
    complexity_score: float = Field(..., ge=0.0, le=100.0)
    reliability_score: float = Field(..., ge=0.0, le=100.0)
    consistency_score: float = Field(..., ge=0.0, le=100.0)
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    raw_weighted_sum: float = Field(..., ge=0.0, le=100.0)
    missing_evidence_penalty: float = Field(..., ge=0.0, le=100.0)
    contradiction_penalty: float = Field(..., ge=0.0, le=100.0)
    final_capability_score: float = Field(..., ge=0.0, le=100.0)
    weights_used: FormulaWeightConfig = Field(default_factory=FormulaWeightConfig)
    formula_expression: str = ""

class CapabilityScoreDetail(BaseModel):
    capability_id: str
    capability_name: str
    status: str  # Strongly Verified, Verified, Partially Verified, Weakly Verified, Unsupported, Contradicted
    final_capability_score: float = Field(..., ge=0.0, le=100.0)
    formula_breakdown: FormulaBreakdown
    reasoning: str = ""
    evidence_counts: Dict[str, int] = Field(default_factory=dict)
    contradiction_count: int = 0

class ReadinessSummary(BaseModel):
    overall_capability_score: float = Field(..., ge=0.0, le=100.0)
    readiness_level: str  # Production Ready, Expert, Nearly Ready, Learning, Not Ready
    readiness_percentage: float = Field(..., ge=0.0, le=100.0)
    readiness_reasoning: str = ""
    strongly_verified_count: int = 0
    verified_count: int = 0
    partially_verified_count: int = 0
    weakly_verified_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0

class RecommendationItem(BaseModel):
    recommendation_id: str
    capability_id: str
    category: str  # Action, Assessment, CodeVerification, Documentation
    priority: str  # High, Medium, Low
    title: str
    recommendation: str

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_capabilities_evaluated: int = 0
    warnings: List[str] = Field(default_factory=list)

class CapabilityScoringResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    capability_scores: List[CapabilityScoreDetail] = Field(default_factory=list)
    readiness_summary: ReadinessSummary
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class CapabilityScoringRequestPayload(BaseModel):
    evidence_fusion_result: Dict[str, Any]
    weight_config: Optional[FormulaWeightConfig] = None
