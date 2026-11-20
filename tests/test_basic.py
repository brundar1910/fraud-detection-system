"""
Basic sanity tests for project modules: configuration, rules, and detector.
"""

from src.config import settings
from src.detection.rules import HighAmountRule, VelocitySpikeRule
from src.detection.detector import RealTimeFraudDetector


def test_config_defaults():
    assert settings.environment in ("development", "production", "testing")
    assert settings.db.port == 5432
    assert "postgresql://" in settings.db.url


def test_high_amount_rule():
    rule = HighAmountRule(threshold=5000.0)
    normal_txn = {"amount": 100.0}
    fraud_txn = {"amount": 7500.0}

    assert rule.evaluate(normal_txn) is None
    alert = rule.evaluate(fraud_txn)
    assert alert is not None
    assert alert["rule_name"] == "HIGH_AMOUNT_THRESHOLD"
    assert alert["risk_score"] >= 80.0


def test_velocity_rule():
    rule = VelocitySpikeRule(max_count=3)
    normal_context = {"recent_transaction_count": 2}
    spike_context = {"recent_transaction_count": 5}

    assert rule.evaluate({}, normal_context) is None
    alert = rule.evaluate({}, spike_context)
    assert alert is not None
    assert alert["severity"] == "CRITICAL"


def test_fraud_detector():
    detector = RealTimeFraudDetector()
    txn = {
        "transaction_id": "test-txn-123",
        "account_id": "test-acc-456",
        "amount": 15000.0,
    }
    alerts = detector.evaluate_transaction(txn)
    assert len(alerts) >= 1
    assert alerts[0]["transaction_id"] == "test-txn-123"
