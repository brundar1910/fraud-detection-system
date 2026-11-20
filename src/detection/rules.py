"""
Fraud detection heuristics and heuristic rule definitions.
"""

from typing import Dict, Any, Optional


class BaseFraudRule:
    """Abstract base class for fraud rules."""

    def __init__(self, name: str, description: str, severity: str, base_risk: float):
        self.name = name
        self.description = description
        self.severity = severity
        self.base_risk = base_risk

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class HighAmountRule(BaseFraudRule):
    """Flags single transactions exceeding high dollar thresholds."""

    def __init__(self, threshold: float = 10000.00):
        super().__init__(
            name="HIGH_AMOUNT_THRESHOLD",
            description=f"Transaction amount exceeds ${threshold:,.2f}",
            severity="HIGH",
            base_risk=80.0,
        )
        self.threshold = threshold

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        amount = transaction.get("amount", 0.0)
        if amount > self.threshold:
            return {
                "rule_name": self.name,
                "rule_description": self.description,
                "risk_score": min(100.0, self.base_risk + ((amount - self.threshold) / 1000.0)),
                "severity": self.severity,
                "metadata": {"amount": amount, "threshold": self.threshold},
            }
        return None


class VelocitySpikeRule(BaseFraudRule):
    """Flags accounts exceeding maximum allowable transactions in a sliding window."""

    def __init__(self, max_count: int = 5):
        super().__init__(
            name="VELOCITY_SPIKE",
            description=f"More than {max_count} transactions within the 10-minute window",
            severity="CRITICAL",
            base_risk=85.0,
        )
        self.max_count = max_count

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not context:
            return None
        recent_count = context.get("recent_transaction_count", 0)
        if recent_count > self.max_count:
            return {
                "rule_name": self.name,
                "rule_description": self.description,
                "risk_score": min(100.0, self.base_risk + (recent_count * 2)),
                "severity": self.severity,
                "metadata": {"recent_count": recent_count, "max_allowed": self.max_count},
            }
        return None
