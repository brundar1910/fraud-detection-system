"""
FraudShield AI - SHAP Explainability & Real-Time Inference Module
Scores transactions, generates local feature-level SHAP attributions,
and logs inference results to fraud_predictions and fraud_alerts.
"""

import os
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib
import shap
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

from ml.train_models import engineer_features, get_db_connection

load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "ml" / "models" / "best_model.joblib"


class FraudPredictor:
    """Production inference engine with SHAP explainability."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {model_path}. Please run ml/train_models.py first."
            )

        print(f"[*] Loading model bundle from {model_path}...")
        bundle = joblib.load(model_path)
        self.model_name = bundle["model_name"]
        self.version = bundle["version"]
        self.algorithm = bundle["algorithm"]
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.feature_names = bundle["feature_names"]
        self.metrics = bundle["metrics"]

        print(f"[+] Loaded {self.model_name} ({self.version}) with {len(self.feature_names)} features.")
        # Initialize SHAP explainer
        print("[*] Initializing SHAP Explainer...")
        if "Tree" in self.algorithm or "XGB" in self.algorithm or "Forest" in self.algorithm:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model)
        print("[+] SHAP Explainer ready.")

    def predict_batch(
        self,
        df_raw: pd.DataFrame,
        compute_shap: bool = True,
        top_k_features: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Takes raw transaction DataFrame, computes behavioral features,
        scores predictions, calculates SHAP explanations, and returns structured records.
        """
        X, _, _ = engineer_features(df_raw)

        # Align columns with training feature set
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]

        # Apply scaling if model requires it
        X_eval = self.scaler.transform(X) if self.scaler is not None else X

        t0 = time.time()
        # Compute probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_eval)[:, 1]
        elif hasattr(self.model, "decision_function"):
            scores = -self.model.decision_function(X_eval)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            probs = self.model.predict(X_eval).astype(float)

        batch_latency_ms = (time.time() - t0) * 1000.0
        per_tx_latency_ms = round(batch_latency_ms / max(1, len(X)), 2)

        # Compute SHAP values if requested
        shap_values = None
        if compute_shap:
            try:
                # TreeExplainer returns raw values (for binary classifier, may be list or array)
                sv = self.explainer.shap_values(X)
                if isinstance(sv, list) and len(sv) == 2:
                    shap_values = sv[1]  # positive class
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    shap_values = sv[:, :, 1]
                else:
                    shap_values = sv
            except Exception as e:
                print(f"[!] Warning: SHAP calculation failed with error: {e}")
                shap_values = None

        predictions: List[Dict[str, Any]] = []

        for idx in range(len(df_raw)):
            prob = float(probs[idx])
            tx_id = str(df_raw.iloc[idx]["transaction_id"])
            acc_id = str(df_raw.iloc[idx]["account_id"])

            # Classification labels & Risk tiers
            if prob >= 0.75:
                label = "FRAUD"
                tier = "CRITICAL" if prob >= 0.88 else "HIGH"
            elif prob >= 0.40:
                label = "SUSPICIOUS"
                tier = "MEDIUM"
            else:
                label = "LEGITIMATE"
                tier = "LOW"

            # Top SHAP feature drivers
            top_features_dict = {}
            if shap_values is not None:
                tx_shap = shap_values[idx]
                # Filter positive contributors to fraud risk
                feature_contribs = [
                    (self.feature_names[j], float(tx_shap[j]))
                    for j in range(len(self.feature_names))
                ]
                # Sort descending by positive contribution
                feature_contribs.sort(key=lambda x: x[1], reverse=True)
                for f_name, f_val in feature_contribs[:top_k_features]:
                    top_features_dict[f_name] = round(f_val, 4)

            predictions.append({
                "prediction_id": str(uuid.uuid4()),
                "transaction_id": tx_id,
                "account_id": acc_id,
                "model_version": self.version,
                "fraud_probability": round(prob, 4),
                "prediction_label": label,
                "risk_tier": tier,
                "top_features": top_features_dict,
                "inference_latency_ms": per_tx_latency_ms,
                "amount": float(df_raw.iloc[idx]["amount"]),
            })

        return predictions

    def save_predictions_to_db(self, predictions: List[Dict[str, Any]]):
        """Persists ML predictions and triggers high-severity alerts into PostgreSQL."""
        conn = get_db_connection()
        cur = conn.cursor()

        pred_sql = """
        INSERT INTO fraud_predictions (
            prediction_id, transaction_id, model_version, fraud_probability,
            prediction_label, risk_tier, top_features, inference_latency_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (prediction_id) DO NOTHING;
        """

        alert_sql = """
        INSERT INTO fraud_alerts (
            alert_id, transaction_id, account_id, rule_name, rule_description,
            risk_score, severity, status, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (alert_id) DO NOTHING;
        """

        pred_batch = []
        alert_batch = []

        for p in predictions:
            pred_batch.append((
                p["prediction_id"],
                p["transaction_id"],
                p["model_version"],
                p["fraud_probability"],
                p["prediction_label"],
                p["risk_tier"],
                Json(p["top_features"]),
                p["inference_latency_ms"],
            ))

            # Automatically trigger alerts for HIGH and CRITICAL risk tiers
            if p["risk_tier"] in ("HIGH", "CRITICAL"):
                top_str = ", ".join([f"{k} (+{v:.2f})" for k, v in p["top_features"].items()])
                alert_batch.append((
                    str(uuid.uuid4()),
                    p["transaction_id"],
                    p["account_id"],
                    "ML_MODEL_HIGH_CONFIDENCE_FRAUD",
                    f"Machine Learning scored fraud probability {p['fraud_probability']*100:.1f}%. Top drivers: {top_str}",
                    round(p["fraud_probability"] * 100.0, 2),
                    p["risk_tier"],
                    "OPEN",
                    Json({
                        "model_version": p["model_version"],
                        "top_shap_features": p["top_features"],
                        "fraud_probability": p["fraud_probability"],
                    }),
                ))

        # Execute inserts
        cur.executemany(pred_sql, pred_batch)
        if alert_batch:
            cur.executemany(alert_sql, alert_batch)

        conn.commit()
        cur.close()
        conn.close()
        print(f"[+] Saved {len(pred_batch):,} predictions to fraud_predictions.")
        if alert_batch:
            print(f"[+] Raised {len(alert_batch):,} high-priority alerts into fraud_alerts.")


def run_sample_inference(batch_size: int = 1000):
    """Executes sample batch scoring on actual transactions from fraud_db."""
    print(f"\n[*] Running SHAP inference pipeline on sample of {batch_size} transactions...")
    conn = get_db_connection()
    query = f"""
    SELECT 
        t.transaction_id,
        t.account_id,
        t.merchant_id,
        t.amount,
        t.transaction_type,
        t.channel,
        t.step,
        t.oldbalance_orig,
        t.newbalance_orig,
        t.oldbalance_dest,
        t.newbalance_dest,
        t.is_fraud,
        t.timestamp,
        COALESCE(m.risk_level, 'LOW') AS merchant_risk_level,
        COALESCE(m.category_code, '0000') AS merchant_mcc
    FROM transactions t
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    ORDER BY t.is_fraud DESC, t.timestamp DESC
    LIMIT {batch_size};
    """
    df_sample = pd.read_sql_query(query, conn)
    conn.close()

    predictor = FraudPredictor()
    predictions = predictor.predict_batch(df_sample, compute_shap=True)
    predictor.save_predictions_to_db(predictions)

    # Print sample top predictions with SHAP
    print("\n" + "=" * 80)
    print("      SAMPLE SHAP EXPLAINABILITY INFERENCE (TOP FLAGGED EVENTS)")
    print("=" * 80)
    flagged = [p for p in predictions if p["risk_tier"] in ("HIGH", "CRITICAL")][:5]
    for p in flagged:
        print(f"Transaction ID : {p['transaction_id']}")
        print(f"Risk Tier      : {p['risk_tier']} (Probability: {p['fraud_probability']:.4f})")
        print(f"Label          : {p['prediction_label']}")
        print(f"Latency        : {p['inference_latency_ms']} ms")
        print("Top SHAP Drivers:")
        for feat, val in p["top_features"].items():
            print(f"  • {feat:<24} : +{val:.4f} impact")
        print("-" * 80)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_sample_inference(batch_size=1000)
