"""
FraudShield AI - Machine Learning & SHAP Test Suite
Tests feature engineering, XGBoost model predictions, and SHAP explainability.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.train_models import engineer_features, MODELS_DIR
from ml.predict import FraudPredictor


def test_model_artifact_exists():
    """Validates that best_model.joblib exists and is non-empty."""
    artifact = MODELS_DIR / "best_model.joblib"
    assert artifact.exists()
    assert artifact.stat().st_size > 10000


def test_feature_engineering_pipeline():
    """Tests feature generation on sample transactions."""
    df_sample = pd.DataFrame([
        {
            "transaction_id": "tx_test_1",
            "account_id": "acc_test_1",
            "merchant_id": "mer_test_1",
            "amount": 250.0,
            "transaction_type": "PAYMENT",
            "channel": "ONLINE",
            "step": 12,
            "oldbalance_orig": 1000.0,
            "newbalance_orig": 750.0,
            "oldbalance_dest": 0.0,
            "newbalance_dest": 0.0,
            "is_fraud": 0,
            "timestamp": "2026-01-01 12:00:00",
            "merchant_risk_level": "LOW",
            "merchant_mcc": "5411",
        }
    ])
    X, y, feature_names = engineer_features(df_sample)
    assert X.shape[0] == 1
    assert "log_amount" in feature_names
    assert "orig_drain_ratio" in feature_names
    assert "error_balance_orig" in feature_names
    assert not X.isnull().values.any()


def test_fraud_predictor_inference_and_shap():
    """Validates batch scoring and Tree SHAP feature attributions."""
    predictor = FraudPredictor()
    assert predictor.model is not None
    assert predictor.explainer is not None

    df_sample = pd.DataFrame([
        {
            "transaction_id": "tx_normal",
            "account_id": "acc_1",
            "merchant_id": "mer_1",
            "amount": 50.0,
            "transaction_type": "PAYMENT",
            "channel": "IN_STORE",
            "step": 50,
            "oldbalance_orig": 500.0,
            "newbalance_orig": 450.0,
            "oldbalance_dest": 0.0,
            "newbalance_dest": 0.0,
            "is_fraud": 0,
            "timestamp": "2026-01-05 14:00:00",
            "merchant_risk_level": "LOW",
            "merchant_mcc": "5411",
        },
        {
            "transaction_id": "tx_fraud_spike",
            "account_id": "acc_1",
            "merchant_id": "mer_shell",
            "amount": 15000.0,
            "transaction_type": "TRANSFER",
            "channel": "ONLINE",
            "step": 52,
            "oldbalance_orig": 15000.0,
            "newbalance_orig": 0.0,
            "oldbalance_dest": 100.0,
            "newbalance_dest": 15100.0,
            "is_fraud": 1,
            "timestamp": "2026-01-05 16:00:00",
            "merchant_risk_level": "CRITICAL",
            "merchant_mcc": "4829",
        },
    ])

    predictions = predictor.predict_batch(df_sample, compute_shap=True, top_k_features=3)
    assert len(predictions) == 2

    # Normal transaction validation
    p_norm = predictions[0]
    assert 0.0 <= p_norm["fraud_probability"] <= 1.0
    assert p_norm["risk_tier"] in ("LOW", "MEDIUM")

    # Fraud spike validation
    p_fraud = predictions[1]
    assert p_fraud["fraud_probability"] > p_norm["fraud_probability"]
    assert p_fraud["risk_tier"] in ("HIGH", "CRITICAL")
    assert len(p_fraud["top_features"]) > 0
    assert "amount" in p_fraud["top_features"] or "log_amount" in p_fraud["top_features"] or "orig_drain_ratio" in p_fraud["top_features"]
