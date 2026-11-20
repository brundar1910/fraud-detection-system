"""
Fraud Detector engine evaluating incoming transactions against rules and generating alerts.
"""

from typing import Dict, Any, List, Optional
from src.detection.rules import BaseFraudRule, HighAmountRule, VelocitySpikeRule
from src.utils.logger import setup_logger

logger = setup_logger("fraud_detector")


class RealTimeFraudDetector:
    """Evaluates transactions in real time against configured rules."""

    def __init__(self, rules: Optional[List[BaseFraudRule]] = None):
        self.rules: List[BaseFraudRule] = rules or [
            HighAmountRule(threshold=10000.00),
            VelocitySpikeRule(max_count=5),
        ]

    def evaluate_transaction(
        self,
        transaction: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a single transaction and returns any triggered fraud alerts.
        """
        triggered_alerts = []
        for rule in self.rules:
            result = rule.evaluate(transaction, context)
            if result:
                alert = {
                    "transaction_id": transaction.get("transaction_id"),
                    "account_id": transaction.get("account_id"),
                    "rule_name": result["rule_name"],
                    "rule_description": result["rule_description"],
                    "risk_score": round(result["risk_score"], 2),
                    "severity": result["severity"],
                    "status": "OPEN",
                    "metadata": result.get("metadata", {}),
                }
                triggered_alerts.append(alert)
                logger.warning(
                    f"Fraud alert triggered for account {transaction.get('account_id')}: "
                    f"Rule={result['rule_name']} Score={result['risk_score']}"
                )
        return triggered_alerts
