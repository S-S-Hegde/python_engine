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

# Model Alias Mapping to Provider & Concrete Models across Groq, OpenRouter, Mistral, Cohere, Gemini, OpenAI, NVIDIA
MODEL_ALIASES = {
    "json_fast": {
        "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_1": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_2": {"provider": "mistral", "model": "mistral-small-latest"},
        "fallback_3": {"provider": "cohere", "model": "command-r-plus"},
        "fallback_4": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "fallback_5": {"provider": "openai", "model": "gpt-4o-mini"},
        "fallback_6": {"provider": "nvidia", "model": "meta/llama-3.3-70b-instruct"},
    },
    "code_reasoning": {
        "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_1": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_2": {"provider": "mistral", "model": "mistral-small-latest"},
        "fallback_3": {"provider": "cohere", "model": "command-r-plus"},
        "fallback_4": {"provider": "gemini", "model": "gemini-2.0-flash"},
    },
    "doc_writer": {
        "primary": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_1": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_2": {"provider": "mistral", "model": "mistral-small-latest"},
        "fallback_3": {"provider": "cohere", "model": "command-r-plus"},
    },
    "behavior": {
        "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_1": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_2": {"provider": "mistral", "model": "mistral-small-latest"},
        "fallback_3": {"provider": "cohere", "model": "command-r-plus"},
    },
    "executive_synthesis": {
        "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_1": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_2": {"provider": "cohere", "model": "command-r-plus"},
    },
    "trust_engine": {
        "primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "fallback_1": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        "fallback_2": {"provider": "mistral", "model": "mistral-small-latest"},
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
