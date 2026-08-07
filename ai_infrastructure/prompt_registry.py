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
        system_prompt=(
            "You are an expert AI resume parser. You MUST extract all technical skills, programming languages, frameworks, "
            "tools, databases, and engineering claims into a JSON object with a single top-level key 'claims' containing an array of claim objects.\n\n"
            "Required JSON Schema:\n"
            "{\n"
            '  "claims": [\n'
            '    {\n'
            '      "claim_id": "claim_1",\n'
            '      "skill": "Exact Skill/Tool Name (e.g. React.js, Python, PostgreSQL)",\n'
            '      "context": "Brief explanation of candidate experience with this skill",\n'
            '      "source_quote": "Direct snippet or mention from resume text",\n'
            '      "category": "Skill",\n'
            '      "confidence": 95\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Do NOT include markdown fences or any text outside the valid JSON object."
        ),
        user_template="Extract all candidate claims, skills, tools, and technical experience from the following resume text:\n\n{resume_text}",
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
        system_prompt="You are a senior technical interviewer for VeriProof forensic credential platform. Generate high-precision, role-tailored multiple choice assessment questions in valid JSON.",
        user_template=(
            "Generate {num_questions} technical multiple choice questions (MCQs) for difficulty '{difficulty}'.\n"
            "Job Title: {job_title}\n"
            "Target Skills: {skills_text}\n\n"
            "Candidate Resume Description & Evidence:\n{resume_description}\n\n"
            "Job Description & Requirements:\n{job_description}\n\n"
            "Formulate challenging questions directly evaluating the candidate's declared claims against the job role requirements.\n"
            "Return JSON object with 'questions': list of dicts containing 'question_text', 'options' (list of 4 strings), 'correct_answer', and 'skill'."
        ),
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
