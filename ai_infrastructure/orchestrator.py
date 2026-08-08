"""
Centralized AI Orchestrator Service
VeriProof AI Infrastructure Foundation
"""
import logging
import time
import uuid
import os
from typing import Dict, Any, Optional

from .capabilities import AICapability, CAPABILITY_ROUTING, MODEL_ALIASES
from .prompt_registry import PromptRegistry
from .circuit_breaker import ProviderHealthMonitor
from .cache_manager import CacheManager
from .schema_validator import ResponseValidator
from .provider_adapters import (
    GeminiAdapter,
    OpenAIAdapter,
    GroqAdapter,
    OpenRouterAdapter,
    MistralAdapter,
    CohereAdapter,
    NvidiaAdapter,
    LocalFallbackAdapter,
    get_env_secret
)

from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("ai_infrastructure.orchestrator")

class AIOrchestratorService:
    @classmethod
    def print_provider_status_matrix(cls) -> Dict[str, str]:
        """Audit environment variables and return readiness matrix for all configured AI providers."""
        providers_check = [
            ("Gemini", ["GEMINI_API_KEY"]),
            ("Groq", ["GROQ_API_KEY", "GROK_API_KEY"]),
            ("OpenRouter", ["OPENROUTER_API_KEY", "OPenRouter_API_Key"]),
            ("Mistral", ["MISTRAL_API_KEY", "Mistal_API_Key"]),
            ("Cohere", ["COHERE_API_KEY", "Cohere_API_Key"]),
            ("NVIDIA", ["NVIDIA_API_KEY", "NVIDIA_NIM_API_Key"]),
            ("OpenAI", ["OPENAI_API_KEY"]),
        ]
        
        status_matrix = {}
        print("\n=========================================================================")
        print("   VERIPROOF UNIFIED AI PROVIDER ORCHESTRATION MATRIX                   ")
        print("=========================================================================")
        
        for name, env_keys in providers_check:
            secret = get_env_secret(*env_keys)
            if secret:
                masked = f"{secret[:4]}...{secret[-4:]}" if len(secret) > 8 else "***"
                status_matrix[name] = "READY"
                dots = "." * (25 - len(name))
                print(f"   {name} {dots} READY ({masked})")
            else:
                status_matrix[name] = "NOT_CONFIGURED"
                dots = "." * (25 - len(name))
                print(f"   {name} {dots} NOT_CONFIGURED")
                
        print("=========================================================================\n")
        return status_matrix

    @classmethod
    def get_adapter(cls, provider_info: Dict[str, str]):
        provider = provider_info.get("provider", "groq").lower()
        model = provider_info.get("model", "llama-3.3-70b-versatile")

        if provider == "gemini":
            return GeminiAdapter(model)
        elif provider == "openai":
            return OpenAIAdapter(model)
        elif provider in ["groq", "grok"]:
            return GroqAdapter(model)
        elif provider == "openrouter":
            return OpenRouterAdapter(model)
        elif provider == "mistral":
            return MistralAdapter(model)
        elif provider == "cohere":
            return CohereAdapter(model)
        elif provider == "nvidia":
            return NvidiaAdapter(model)
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

        providers_to_try = []
        for key in ["primary", "fallback_1", "fallback_2", "fallback_3", "fallback_4", "fallback_5", "fallback_6"]:
            if key in routing_tiers:
                providers_to_try.append(routing_tiers[key])
        providers_to_try.append({"provider": "local", "model": "heuristic_fallback"})

        # 5. Parallel Provider Competition Execution (Ultra-Fast Response)
        valid_providers = [p for p in providers_to_try if p["provider"] != "local" and ProviderHealthMonitor.get_breaker(p["provider"]).can_execute()]
        
        if valid_providers and len(valid_providers) > 1:
            def _try_provider(prov_info):
                prov_name = prov_info["provider"]
                adapter = cls.get_adapter(prov_info)
                success, raw_output, meta = adapter.generate(system_prompt, user_prompt)
                if success:
                    is_json_task = capability in [AICapability.JSON_EXTRACTION, AICapability.STRUCTURED_VALIDATION, AICapability.CODE_GRADING]
                    if is_json_task:
                        val_ok, parsed_json, _ = ResponseValidator.validate_and_parse_json(raw_output)
                        if val_ok:
                            return (prov_name, prov_info["model"], parsed_json, meta.get("durationMs", 0))
                    else:
                        return (prov_name, prov_info["model"], raw_output, meta.get("durationMs", 0))
                return None

            try:
                with ThreadPoolExecutor(max_workers=min(len(valid_providers), 5)) as executor:
                    futures = {executor.submit(_try_provider, p): p for p in valid_providers}
                    for future in as_completed(futures):
                        res = future.result()
                        if res:
                            prov_name, model_name, result_data, latency_ms = res
                            logger.info(f"[{corr_id}] Ultra-Fast Parallel AI Win from '{prov_name}' ({latency_ms} ms)!")
                            if cache_key:
                                CacheManager.set(cache_key, result_data)
                            return {
                                "correlation_id": corr_id,
                                "prompt_id": prompt_id,
                                "version": prompt_def.version,
                                "cached": False,
                                "result": result_data,
                                "provider": prov_name,
                                "model": model_name,
                                "latencyMs": latency_ms,
                                "retries": 0
                            }
            except Exception as par_err:
                logger.warning(f"[{corr_id}] Parallel AI execution notice: {par_err}. Falling back to sequential chain.")

        # 5. Execute Fallback Chain
        retries = 0
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
                logger.info(f"[{corr_id}] RAW AI RESPONSE from '{prov_name}' ({len(raw_output)} bytes):\n================ RAW AI RESPONSE ================\n{raw_output}\n================================================")
                
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
                            "latencyMs": meta.get("durationMs", 0),
                            "retries": retries
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
                        "latencyMs": meta.get("durationMs", 0),
                        "retries": retries
                    }
            else:
                retries += 1
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
            "result": parsed_local,
            "provider": "local",
            "model": "heuristic_fallback",
            "latencyMs": 1,
            "retries": retries,
            "fallback": True
        }
