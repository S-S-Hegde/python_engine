import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    processing_time_ms: float = 0.0
    model: str = "competency-intelligence-v2"
    pipeline_module: str = "Competency Intelligence Engine"

class CapabilityDistribution(BaseModel):
    total_capabilities: int = 0
    strongly_verified_count: int = 0
    verified_count: int = 0
    partially_verified_count: int = 0
    weakly_verified_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0

class CompetencyProfile(BaseModel):
    competency_id: str
    competency_name: str
    domain: str  # backend, frontend, devops, database, testing, security, cloud, ai_ml, software_design, general
    competency_score: float = Field(..., ge=0.0, le=100.0)
    coverage: float = Field(..., ge=0.0, le=100.0)
    average_capability_score: float = Field(..., ge=0.0, le=100.0)
    minimum_capability_score: float = Field(..., ge=0.0, le=100.0)
    critical_capability_coverage: float = Field(..., ge=0.0, le=100.0)
    competency_confidence: float = Field(..., ge=0.0, le=100.0)
    competency_reliability: float = Field(..., ge=0.0, le=100.0)
    maturity_level: str  # Expert, Advanced, Intermediate, Developing, Beginner
    capability_ids: List[str] = Field(default_factory=list)
    capability_distribution: CapabilityDistribution = Field(default_factory=CapabilityDistribution)
    dependency_penalty: float = Field(default=0.0, ge=0.0, le=100.0)
    reasoning: str = ""

class CompetencySummary(BaseModel):
    overall_competency_score: float = Field(..., ge=0.0, le=100.0)
    highest_competency: str = ""
    lowest_competency: str = ""
    total_competencies: int = 0
    expert_count: int = 0
    advanced_count: int = 0
    intermediate_count: int = 0
    developing_count: int = 0
    beginner_count: int = 0

class GrowthRecommendation(BaseModel):
    recommendation_id: str
    competency_id: str
    target_maturity: str
    current_score: float
    action_item: str
    suggested_resources: List[str] = Field(default_factory=list)

class ValidationReport(BaseModel):
    is_valid: bool = True
    total_competencies_evaluated: int = 0
    warnings: List[str] = Field(default_factory=list)

class CompetencyIntelligenceResult(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    competencies: List[CompetencyProfile] = Field(default_factory=list)
    competency_summary: CompetencySummary
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_competencies: List[str] = Field(default_factory=list)
    growth_recommendations: List[GrowthRecommendation] = Field(default_factory=list)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

class CompetencyRequestPayload(BaseModel):
    capability_scoring_result: Dict[str, Any]
