"""
Centralized AI Orchestrator Service
VeriProof AI Infrastructure Foundation
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional

from .capabilities import AICapability, CAPABILITY_ROUTING, MODEL_ALIASES
from .prompt_registry import PromptRegistry
from .circuit_breaker import ProviderHealthMonitor
from .cache_manager import CacheManager
from .schema_validator import ResponseValidator
from .provider_adapters import GeminiAdapter, OpenAIAdapter, GrokAdapter, LocalFallbackAdapter

logger = logging.getLogger("ai_infrastructure.orchestrator")

class AIOrchestratorService:
    @classmethod
    def get_adapter(cls, provider_info: Dict[str, str]):
        provider = provider_info.get("provider", "gemini")
        model = provider_info.get("model", "gemini-2.0-flash")

        if provider == "gemini":
            return GeminiAdapter(model)
        elif provider == "openai":
            return OpenAIAdapter(model)
        elif provider == "grok":
            return GrokAdapter(model)
        else:
            return LocalFallbackAdapter(model)

    @classmethod
    def execute_task(
        cls,
        prompt_id: str,
        payload_inputs: Dict[str, Any],
        prompt_version: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        corr_id = correlation_id or f"AI-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        logger.info(f"[{corr_id}] Executing AI Task: '{prompt_id}'")

        # 1. Fetch Prompt Definition from Registry
        prompt_def = PromptRegistry.get(prompt_id, prompt_version)
        capability = prompt_def.capability

        # 2. Check Deterministic Cache
        cache_key = None
        if prompt_def.deterministic:
            cache_key = CacheManager.generate_key(prompt_id, prompt_def.version, payload_inputs)
            cached_data = CacheManager.get(cache_key)
            if cached_data is not None:
                return {
                    "correlation_id": corr_id,
                    "prompt_id": prompt_id,
                    "version": prompt_def.version,
                    "cached": True,
                    "result": cached_data,
                    "provider": "cache",
                    "latencyMs": 1
                }

        # 3. Format Prompt Templates
        user_prompt = prompt_def.user_template.format(**payload_inputs)
        system_prompt = prompt_def.system_prompt

        # 4. Resolve Provider Fallback Chain based on Capability
        alias_name = CAPABILITY_ROUTING.get(capability, "json_fast")
        routing_tiers = MODEL_ALIASES.get(alias_name, MODEL_ALIASES["json_fast"])

        providers_to_try = [
            routing_tiers["primary"],
            routing_tiers["fallback_1"],
            routing_tiers["fallback_2"],
            {"provider": "local", "model": "heuristic_fallback"}
        ]

        # 5. Execute Fallback Chain
        for tier_idx, prov_info in enumerate(providers_to_try):
            prov_name = prov_info["provider"]
            breaker = ProviderHealthMonitor.get_breaker(prov_name)

            if not breaker.can_execute():
                logger.warning(f"[{corr_id}] Circuit breaker for provider '{prov_name}' is OPEN. Bypassing tier {tier_idx + 1}.")
                continue

            adapter = cls.get_adapter(prov_info)
            logger.info(f"[{corr_id}] Attempting Tier {tier_idx + 1} Provider: '{prov_name}' (Model: '{prov_info['model']}')")

            success, raw_output, meta = adapter.generate(system_prompt, user_prompt)

            if success:
                # 6. Structured Schema Validation & Parsing
                is_json_task = capability in [AICapability.JSON_EXTRACTION, AICapability.STRUCTURED_VALIDATION, AICapability.CODE_GRADING]
                
                if is_json_task:
                    val_ok, parsed_json, val_err = ResponseValidator.validate_and_parse_json(raw_output)
                    if val_ok:
                        breaker.record_success()
                        if cache_key:
                            CacheManager.set(cache_key, parsed_json)

                        return {
                            "correlation_id": corr_id,
                            "prompt_id": prompt_id,
                            "version": prompt_def.version,
                            "cached": False,
                            "result": parsed_json,
                            "provider": prov_name,
                            "model": prov_info["model"],
                            "latencyMs": meta.get("durationMs", 0)
                        }
                    else:
                        breaker.record_failure(f"Schema validation failed: {val_err}")
                else:
                    breaker.record_success()
                    if cache_key:
                        CacheManager.set(cache_key, raw_output)

                    return {
                        "correlation_id": corr_id,
                        "prompt_id": prompt_id,
                        "version": prompt_def.version,
                        "cached": False,
                        "result": raw_output,
                        "provider": prov_name,
                        "model": prov_info["model"],
                        "latencyMs": meta.get("durationMs", 0)
                    }
            else:
                breaker.record_failure(meta.get("error", "Unknown error"))

        # Final Local Fallback if all API attempts failed
        logger.error(f"[{corr_id}] All cloud providers failed for task '{prompt_id}'. Executing local deterministic fallback.")
        local_adapter = LocalFallbackAdapter()
        _, local_out, _ = local_adapter.generate(system_prompt, user_prompt)
        _, parsed_local, _ = ResponseValidator.validate_and_parse_json(local_out)

        return {
            "correlation_id": corr_id,
            "prompt_id": prompt_id,
            "version": prompt_def.version,
            "cached": False,
            "result": parsed_local or local_out,
            "provider": "local_fallback",
            "model": "heuristic",
            "latencyMs": 1
        }
