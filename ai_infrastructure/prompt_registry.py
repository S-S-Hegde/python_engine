"""
Centralized Prompt Registry
VeriProof AI Infrastructure Foundation
"""
import logging
from typing import Dict, Any, Optional
from .capabilities import AICapability

logger = logging.getLogger("ai_infrastructure.prompt_registry")

class PromptDefinition:
    def __init__(
        self,
        prompt_id: str,
        version: str,
        capability: AICapability,
        system_prompt: str,
        user_template: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        deterministic: bool = True
    ):
        self.prompt_id = prompt_id
        self.version = version
        self.capability = capability
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.expected_schema = expected_schema
        self.deterministic = deterministic

class PromptRegistry:
    _prompts: Dict[str, PromptDefinition] = {}

    @classmethod
    def register(cls, prompt_def: PromptDefinition):
        key = f"{prompt_def.prompt_id}:v{prompt_def.version}"
        cls._prompts[key] = prompt_def
        # Also store under default latest alias
        cls._prompts[prompt_def.prompt_id] = prompt_def
        logger.debug(f"Registered prompt {key}")

    @classmethod
    def get(cls, prompt_id: str, version: Optional[str] = None) -> PromptDefinition:
        key = f"{prompt_id}:v{version}" if version else prompt_id
        if key not in cls._prompts:
            raise KeyError(f"Prompt '{key}' not found in PromptRegistry")
        return cls._prompts[key]

# ── INITIALIZE CORE PROMPTS ───────────────────────────────────────────────────

PromptRegistry.register(
    PromptDefinition(
        prompt_id="resume_claims_extraction",
        version="1.0",
        capability=AICapability.JSON_EXTRACTION,
        system_prompt="You are an expert AI resume parser. Extract candidate claims into valid JSON matching the specified schema.",
        user_template="Extract all candidate claims from the following resume text:\n\n{resume_text}",
        deterministic=True
    )
)

PromptRegistry.register(
    PromptDefinition(
        prompt_id="repo_docs_generator",
        version="1.0",
        capability=AICapability.LONG_FORM_GENERATION,
        system_prompt="You are a senior technical writer. Generate clean, professional Markdown documentation for the given software repository.",
        user_template="Generate comprehensive technical documentation for repository '{repo_name}':\n\nLanguages: {languages}\nFiles: {file_list}\nContext: {readme_excerpt}",
        deterministic=True
    )
)

PromptRegistry.register(
    PromptDefinition(
        prompt_id="assessment_mcq_generator",
        version="1.0",
        capability=AICapability.JSON_EXTRACTION,
        system_prompt="You are a senior technical interviewer. Generate multiple choice assessment questions in valid JSON.",
        user_template="Generate {num_questions} multiple choice questions for difficulty '{difficulty}' on skills: {skills_text}",
        deterministic=False
    )
)

PromptRegistry.register(
    PromptDefinition(
        prompt_id="code_snippet_evaluator",
        version="1.0",
        capability=AICapability.CODE_GRADING,
        system_prompt="You are a automated code evaluator. Grade the code snippet for correctness, algorithmic efficiency, and quality.",
        user_template="Evaluate code snippet ({language}):\nContext: {context}\nCode:\n{code}",
        deterministic=True
    )
)

PromptRegistry.register(
    PromptDefinition(
        prompt_id="behavioral_evaluator",
        version="1.0",
        capability=AICapability.BEHAVIORAL_REASONING,
        system_prompt="You are a behavioral assessment expert. Evaluate candidate response for professional integrity and problem-solving tone.",
        user_template="Question Context: {question_context}\nCandidate Answer: {response_text}",
        deterministic=False
    )
)
