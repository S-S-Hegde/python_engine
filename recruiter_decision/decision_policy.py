from typing import Dict, Any
from .models import DecisionPolicyType, HireDecision

class DecisionPolicyEngine:
    """
    Defines configurable hiring policies that govern thresholds.
    """
    
    POLICIES = {
        DecisionPolicyType.STARTUP: {
            "thresholds": {
                HireDecision.STRONG_HIRE: 80.0,
                HireDecision.HIRE: 65.0,
                HireDecision.BORDERLINE: 50.0,
                HireDecision.HOLD: 40.0
            },
            "description": "High agility, bias for action. Lower threshold for acceptable hires, accepts higher risk."
        },
        DecisionPolicyType.ENTERPRISE: {
            "thresholds": {
                HireDecision.STRONG_HIRE: 90.0,
                HireDecision.HIRE: 75.0,
                HireDecision.BORDERLINE: 65.0,
                HireDecision.HOLD: 55.0
            },
            "description": "Strict compliance and high baseline requirements."
        },
        DecisionPolicyType.INTERN: {
            "thresholds": {
                HireDecision.STRONG_HIRE: 60.0,
                HireDecision.HIRE: 45.0,
                HireDecision.BORDERLINE: 35.0,
                HireDecision.HOLD: 25.0
            },
            "description": "Evaluates potential over experience."
        },
        DecisionPolicyType.SENIOR: {
            "thresholds": {
                HireDecision.STRONG_HIRE: 95.0,
                HireDecision.HIRE: 85.0,
                HireDecision.BORDERLINE: 75.0,
                HireDecision.HOLD: 65.0
            },
            "description": "Very strict requirements for senior hires. Zero tolerance for low capability scores."
        },
        DecisionPolicyType.CUSTOM: {
            "thresholds": {
                HireDecision.STRONG_HIRE: 85.0,
                HireDecision.HIRE: 70.0,
                HireDecision.BORDERLINE: 60.0,
                HireDecision.HOLD: 50.0
            },
            "description": "Customizable baseline policy."
        }
    }

    def get_policy(self, policy_type: DecisionPolicyType) -> Dict[str, Any]:
        return self.POLICIES.get(policy_type, self.POLICIES[DecisionPolicyType.ENTERPRISE])
