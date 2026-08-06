"""
Input-Hashed Response Cache Manager
VeriProof AI Infrastructure Foundation
"""
import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict

logger = logging.getLogger("ai_infrastructure.cache_manager")

class CacheManager:
    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def generate_key(cls, prompt_id: str, prompt_version: str, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(f"{prompt_id}:v{prompt_version}:{serialized}".encode("utf-8")).hexdigest()
        return f"ai_cache:{prompt_id}:{hash_digest}"

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        if key in cls._cache:
            entry = cls._cache[key]
            if entry["expires_at"] > time.time():
                logger.debug(f"[Cache HIT] Key: {key[:24]}...")
                return entry["data"]
            else:
                logger.debug(f"[Cache EXPIRED] Key: {key[:24]}...")
                del cls._cache[key]
        return None

    @classmethod
    def set(cls, key: str, data: Any, ttl_seconds: int = 86400):
        cls._cache[key] = {
            "data": data,
            "expires_at": time.time() + ttl_seconds,
            "created_at": time.time(),
        }
        logger.debug(f"[Cache SET] Key: {key[:24]}... TTL: {ttl_seconds}s")
