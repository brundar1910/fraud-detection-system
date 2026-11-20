"""
FraudShield AI - Machine Learning Training & Evaluation Pipeline
Trains and compares four fraud detection models:
  1. Logistic Regression (Baseline Linear Classifier)
  2. Random Forest (Ensemble Bagging)
  3. XGBoost (Gradient Boosted Trees - Industry Benchmark)
  4. Isolation Forest (Unsupervised Anomaly Detection)

Features engineered:
  - Temporal & Velocity: Step, hour of day, day of week, account rolling transaction velocity
  - Financial Behavior: Amount deviation from account mean, spending spike ratio
  - PaySim Balance Deltas: Error in orig/dest balance, account drain ratio
  - Merchant Risk: MCC tier encoding

Evaluates Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.
Logs runs to PostgreSQL model_versions table and persists the best model artifact to ml/models/best_model.joblib.
"""

import os
import sys
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

MODELS_DIR = BASE_DIR / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def load_raw_data() -> pd.DataFrame:
    """Loads transaction records joined with account and merchant context."""
    print("[*] Querying 50,000 transactions and features from PostgreSQL fraud_db...")
    conn = get_db_connection()
    query = """
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
    ORDER BY t.timestamp ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"[+] Loaded {len(df):,} transactions from database.")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Engineers temporal, velocity, balance discrepancy, and behavioral deviation features."""
    print("[*] Engineering behavioral, balance delta, and velocity features...")
    df = df.copy()

    # 1. Temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week
    df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

    # 2. Financial Amount transformations
    df["log_amount"] = np.log1p(df["amount"])

    # 3. PaySim Balance Delta & Discrepancy Errors
    # Theoretical error in sender balance (oldbalance_orig - amount should equal newbalance_orig)
    df["error_balance_orig"] = (df["oldbalance_orig"] - df["amount"]) - df["newbalance_orig"]
    # Theoretical error in dest balance (oldbalance_dest + amount should equal newbalance_dest)
    df["error_balance_dest"] = (df["oldbalance_dest"] + df["amount"]) - df["newbalance_dest"]
    # Account drain ratio: what fraction of the sender's balance was liquidated
    df["orig_drain_ratio"] = df["amount"] / (df["oldbalance_orig"] + 1.0)
    # Binary indicator: Did the transaction completely wipe out the sender's account to zero?
    df["is_complete_drain"] = ((df["oldbalance_orig"] > 0) & (df["newbalance_orig"] == 0.0)).astype(int)

    # 4. Behavioral Deviation: Deviation of transaction amount from account historical mean
    account_stats = df.groupby("account_id")["amount"].agg(["mean", "std"]).reset_index()
    account_stats.columns = ["account_id", "acc_mean_amount", "acc_std_amount"]
    account_stats["acc_std_amount"] = account_stats["acc_std_amount"].fillna(1.0)
    df = df.merge(account_stats, on="account_id", how="left")
    df["amount_zscore_acc"] = (df["amount"] - df["acc_mean_amount"]) / (df["acc_std_amount"] + 1e-5)
    df["spending_spike_ratio"] = df["amount"] / (df["acc_mean_amount"] + 1.0)

    # 5. Account Velocity (rolling transaction count in historical sequence)
    df["account_velocity_count"] = df.groupby("account_id").cumcount() + 1

    # 6. Merchant Risk Encoding
    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    df["merchant_risk_score"] = df["merchant_risk_level"].map(risk_map).fillna(0)

    # 7. Categorical One-Hot Encoding
    type_dummies = pd.get_dummies(df["transaction_type"], prefix="type", drop_first=False)
    channel_dummies = pd.get_dummies(df["channel"], prefix="channel", drop_first=False)

    feature_dfs = [
        df[[
            "amount",
            "log_amount",
            "step",
            "hour_of_day",
            "day_of_week",
            "is_night",
            "oldbalance_orig",
            "newbalance_orig",
            "oldbalance_dest",
            "newbalance_dest",
            "error_balance_orig",
            "error_balance_dest",
            "orig_drain_ratio",
            "is_complete_drain",
            "spending_spike_ratio",
            "amount_zscore_acc",
            "account_velocity_count",
            "merchant_risk_score",
        ]],
        type_dummies,
        channel_dummies,
    ]

    X = pd.concat(feature_dfs, axis=1)
    # Ensure all columns are numeric
    X = X.astype(float)
    y = df["is_fraud"].astype(int)
    feature_names = list(X.columns)

    print(f"[+] Feature engineering complete. Extracted {X.shape[1]} behavioral features.")
    return X, y, feature_names


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
    """Trains 4 models and computes authentic evaluation metrics."""
    # Chronological or stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / max(1, pos_count)
    print(f"[*] Dataset Split: Train={len(X_train):,} (Fraud={pos_count:,}), Test={len(X_test):,} (Fraud={y_test.sum():,})")
    print(f"[*] Class imbalance scale_pos_weight: {scale_weight:.2f}")

    # Standard Scaler for Linear Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[1/4] Training Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    t0 = time.time()
    lr.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0
    y_pred_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = (y_pred_prob_lr >= 0.5).astype(int)

    results["Logistic Regression"] = {
        "model": lr,
        "scaler": scaler,
        "algorithm": "LogisticRegression",
        "version": "v1.0.0-lr",
        "hyperparameters": {"class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"},
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall": recall_score(y_test, y_pred_lr, zero_division=0),
        "f1": f1_score(y_test, y_pred_lr, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob_lr),
        "pr_auc": average_precision_score(y_test, y_pred_prob_lr),
        "train_time_sec": lr_train_time,
    }

    # -------------------------------------------------------------------------
    # 2. Random Forest Classifier
    # -------------------------------------------------------------------------
    print("[2/4] Training Random Forest (class_weight='balanced', n_estimators=100)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_pred_prob_rf >= 0.5).astype(int)

    results["Random Forest"] = {
        "model": rf,
        "scaler": None,
        "algorithm": "RandomForestClassifier",
        "version": "v1.0.0-rf",
        "hyperparameters": {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced"},
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob_rf),
        "pr_auc": average_precision_score(y_test, y_pred_prob_rf),
        "train_time_sec": rf_train_time,
    }

    # -------------------------------------------------------------------------
    # 3. XGBoost Classifier
    # -------------------------------------------------------------------------
    print(f"[3/4] Training XGBoost (scale_pos_weight={scale_weight:.2f}, n_estimators=150)...")
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    t0 = time.time()
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - t0
    y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)

    results["XGBoost"] = {
        "model": xgb,
        "scaler": None,
        "algorithm": "XGBClassifier",
        "version": "v1.0.0-xgb",
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.08,
            "scale_pos_weight": float(round(scale_weight, 2)),
        },
        "precision": precision_score(y_test, y_pred_xgb, zero_division=0),
        "recall": recall_score(y_test, y_pred_xgb, zero_division=0),
        "f1": f1_score(y_test, y_pred_xgb, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob_xgb),
        "pr_auc": average_precision_score(y_test, y_pred_prob_xgb),
        "train_time_sec": xgb_train_time,
    }

    # -------------------------------------------------------------------------
    # 4. Isolation Forest (Unsupervised Anomaly Detection)
    # -------------------------------------------------------------------------
    print("[4/4] Training Isolation Forest (unsupervised anomaly detection)...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.035,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    iso.fit(X_train)
    iso_train_time = time.time() - t0
    # Isolation forest returns -1 for outliers, 1 for inliers
    # Convert anomaly score to probability proxy: decision_function (lower = more abnormal)
    iso_scores = -iso.decision_function(X_test)
    # Min-max scale to [0, 1]
    min_s, max_s = iso_scores.min(), iso_scores.max()
    y_pred_prob_iso = (iso_scores - min_s) / (max_s - min_s + 1e-8)
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)

    results["Isolation Forest"] = {
        "model": iso,
        "scaler": None,
        "algorithm": "IsolationForest",
        "version": "v1.0.0-iforest",
        "hyperparameters": {"n_estimators": 100, "contamination": 0.035},
        "precision": precision_score(y_test, y_pred_iso, zero_division=0),
        "recall": recall_score(y_test, y_pred_iso, zero_division=0),
        "f1": f1_score(y_test, y_pred_iso, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob_iso),
        "pr_auc": average_precision_score(y_test, y_pred_prob_iso),
        "train_time_sec": iso_train_time,
    }

    # -------------------------------------------------------------------------
    # Print Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("             FRAUDSHIELD AI - MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Model Name':<22} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Train (s)':<9}"
    print(header)
    print("-" * 90)
    for name, r in results.items():
        row = (
            f"{name:<22} | "
            f"{r['precision']:<9.4f} | "
            f"{r['recall']:<8.4f} | "
            f"{r['f1']:<8.4f} | "
            f"{r['pr_auc']:<8.4f} | "
            f"{r['roc_auc']:<8.4f} | "
            f"{r['train_time_sec']:<9.2f}"
        )
        print(row)
    print("=" * 90 + "\n")

    # Select Best Model based on PR-AUC (best metric for extreme class imbalance)
    best_model_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_entry = results[best_model_name]
    print(f"[BEST MODEL] Best Performing Model: {best_model_name} (PR-AUC: {best_entry['pr_auc']:.4f}, F1: {best_entry['f1']:.4f})")

    # -------------------------------------------------------------------------
    # Save Best Model Artifact
    # -------------------------------------------------------------------------
    best_artifact_path = MODELS_DIR / "best_model.joblib"
    artifact_payload = {
        "model_name": best_model_name,
        "version": best_entry["version"],
        "algorithm": best_entry["algorithm"],
        "model": best_entry["model"],
        "scaler": best_entry["scaler"],
        "feature_names": feature_names,
        "metrics": {
            "precision": float(round(best_entry["precision"], 4)),
            "recall": float(round(best_entry["recall"], 4)),
            "f1": float(round(best_entry["f1"], 4)),
            "pr_auc": float(round(best_entry["pr_auc"], 4)),
            "roc_auc": float(round(best_entry["roc_auc"], 4)),
        },
        "hyperparameters": best_entry["hyperparameters"],
    }
    joblib.dump(artifact_payload, best_artifact_path)
    print(f"[+] Persisted winning model bundle to: {best_artifact_path}")

    # -------------------------------------------------------------------------
    # Log Model Versions to PostgreSQL Database
    # -------------------------------------------------------------------------
    print("[*] Logging all model runs and metrics into PostgreSQL model_versions table...")
    conn = get_db_connection()
    cur = conn.cursor()

    # Clear existing runs
    cur.execute("DELETE FROM model_versions;")
    conn.commit()

    for name, r in results.items():
        is_active = (name == best_model_name)
        metrics_dict = {
            "precision": float(round(r["precision"], 4)),
            "recall": float(round(r["recall"], 4)),
            "f1": float(round(r["f1"], 4)),
            "pr_auc": float(round(r["pr_auc"], 4)),
            "roc_auc": float(round(r["roc_auc"], 4)),
            "train_time_sec": float(round(r["train_time_sec"], 2)),
        }
        cur.execute(
            """
            INSERT INTO model_versions (
                model_name, version, algorithm, hyperparameters, metrics,
                feature_names, artifact_path, is_active, training_samples_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version) DO UPDATE SET
                metrics = EXCLUDED.metrics,
                is_active = EXCLUDED.is_active;
            """,
            (
                name,
                r["version"],
                r["algorithm"],
                Json(r["hyperparameters"]),
                Json(metrics_dict),
                Json(feature_names),
                str(best_artifact_path) if is_active else None,
                is_active,
                len(X_train),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
    print("[+] All model runs and evaluation metrics successfully logged to model_versions table.")


def main():
    df = load_raw_data()
    X, y, feature_names = engineer_features(df)
    train_and_evaluate_models(X, y, feature_names)


if __name__ == "__main__":
    main()
