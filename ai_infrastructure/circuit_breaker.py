"""
Circuit Breaker and Provider Health Monitor
VeriProof AI Infrastructure Foundation
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("ai_infrastructure.circuit_breaker")

class CircuitState:
    CLOSED = "CLOSED"       # Healthy: Requests flow normally
    OPEN = "OPEN"           # Unhealthy: Requests bypassed to fallback
    HALF_OPEN = "HALF_OPEN" # Testing: Probing provider recovery

class CircuitBreaker:
    def __init__(self, provider_name: str, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        now = time.time()
        if self.state == CircuitState.OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                logger.info(f"[{self.provider_name}] Circuit switching from OPEN to HALF_OPEN (Probing recovery)")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.provider_name}] Circuit recovered! Switching to CLOSED.")
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()

    def record_failure(self, error_msg: str = ""):
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(f"[{self.provider_name}] Failure recorded ({self.failure_count}/{self.failure_threshold}): {error_msg}")

        if self.failure_count >= self.failure_threshold and self.state != CircuitState.OPEN:
            logger.error(f"[{self.provider_name}] Circuit tripped to OPEN! Tripped for {self.recovery_timeout}s.")
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()

class ProviderHealthMonitor:
    _breakers: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_breaker(cls, provider_name: str) -> CircuitBreaker:
        if provider_name not in cls._breakers:
            cls._breakers[provider_name] = CircuitBreaker(provider_name)
        return cls._breakers[provider_name]

    @classmethod
    def get_health_stats(cls) -> Dict[str, Any]:
        return {
            provider: {
                "state": breaker.state,
                "failures": breaker.failure_count,
                "successes": breaker.success_count,
            }
            for provider, breaker in cls._breakers.items()
        }
