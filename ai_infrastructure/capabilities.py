"""
AI Capabilities and Model Alias Configuration
VeriProof AI Infrastructure Foundation
"""
import enum
import os

class AICapability(str, enum.Enum):
    JSON_EXTRACTION = "JSON_EXTRACTION"
    STRUCTURED_VALIDATION = "STRUCTURED_VALIDATION"
    LONG_FORM_GENERATION = "LONG_FORM_GENERATION"
    CODE_ANALYSIS = "CODE_ANALYSIS"
    CODE_GRADING = "CODE_GRADING"
    REPOSITORY_UNDERSTANDING = "REPOSITORY_UNDERSTANDING"
    CLASSIFICATION = "CLASSIFICATION"
    RANKING = "RANKING"
    BEHAVIORAL_REASONING = "BEHAVIORAL_REASONING"
    EXECUTIVE_SUMMARIZATION = "EXECUTIVE_SUMMARIZATION"
    EVIDENCE_FUSION = "EVIDENCE_FUSION"
    TRUST_SCORING = "TRUST_SCORING"

# Model Alias Mapping to Provider & Concrete Models
MODEL_ALIASES = {
    "json_fast": {
        "primary": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "fallback_1": {"provider": "openai", "model": "gpt-4o-mini"},
        "fallback_2": {"provider": "grok", "model": "grok-beta"},
    },
    "code_reasoning": {
        "primary": {"provider": "openai", "model": "gpt-4o-mini"},
        "fallback_1": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "fallback_2": {"provider": "local", "model": "heuristic_static"},
    },
    "doc_writer": {
        "primary": {"provider": "openai", "model": "gpt-4o"},
        "fallback_1": {"provider": "gemini", "model": "gemini-1.5-pro"},
        "fallback_2": {"provider": "grok", "model": "grok-2"},
    },
    "behavior": {
        "primary": {"provider": "grok", "model": "grok-2"},
        "fallback_1": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "fallback_2": {"provider": "openai", "model": "gpt-4o-mini"},
    },
    "executive_synthesis": {
        "primary": {"provider": "openai", "model": "gpt-4o"},
        "fallback_1": {"provider": "gemini", "model": "gemini-1.5-pro"},
        "fallback_2": {"provider": "grok", "model": "grok-2"},
    },
    "trust_engine": {
        "primary": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "fallback_1": {"provider": "openai", "model": "gpt-4o-mini"},
        "fallback_2": {"provider": "local", "model": "heuristic_trust"},
    },
}

# Capability to Model Alias Mapping
CAPABILITY_ROUTING = {
    AICapability.JSON_EXTRACTION: "json_fast",
    AICapability.STRUCTURED_VALIDATION: "json_fast",
    AICapability.LONG_FORM_GENERATION: "doc_writer",
    AICapability.CODE_ANALYSIS: "code_reasoning",
    AICapability.CODE_GRADING: "code_reasoning",
    AICapability.REPOSITORY_UNDERSTANDING: "code_reasoning",
    AICapability.CLASSIFICATION: "json_fast",
    AICapability.RANKING: "json_fast",
    AICapability.BEHAVIORAL_REASONING: "behavior",
    AICapability.EXECUTIVE_SUMMARIZATION: "executive_synthesis",
    AICapability.EVIDENCE_FUSION: "trust_engine",
    AICapability.TRUST_SCORING: "trust_engine",
}
