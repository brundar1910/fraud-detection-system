"""
FraudShield AI - FastAPI Production Backend Server
REST API for Real-Time Financial Fraud Detection & Autonomous Forensic Investigation.
"""

import os
import sys
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from backend.schemas import (
    TransactionScoreRequest,
    TransactionScoreResponse,
    MetricsSummaryResponse,
    TransactionListItem,
    AlertListItem,
    CustomerRiskProfileResponse,
    NetworkGraphResponse,
    InvestigationReportResponse,
)
from ml.predict import FraudPredictor
from investigation.investigator import FraudInvestigationAgent
from graph_engine.fraud_graph import OptimizedFraudGraphEngine

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

app = FastAPI(
    title="FraudShield AI - API",
    description="Real-Time Financial Fraud Detection, SHAP Explainability & Investigation Platform",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_predictor: Optional[FraudPredictor] = None
_investigator: Optional[FraudInvestigationAgent] = None


def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


def get_investigator() -> FraudInvestigationAgent:
    global _investigator
    if _investigator is None:
        _investigator = FraudInvestigationAgent()
    return _investigator


def get_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        cursor_factory=RealDictCursor,
    )


# -----------------------------------------------------------------------------
# 1. System Health Check
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Validates PostgreSQL connection and ML model readiness."""
    db_status = "unhealthy"
    model_status = "unhealthy"
    model_info = {}

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    try:
        predictor = get_predictor()
        model_status = "ready"
        model_info = {
            "model_name": predictor.model_name,
            "version": predictor.version,
            "features_count": len(predictor.feature_names),
            "metrics": predictor.metrics,
        }
    except Exception as e:
        model_status = f"error: {e}"

    return {
        "status": "healthy" if db_status == "healthy" and model_status == "ready" else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "ml_engine": model_status,
        "model_details": model_info,
        "environment": "native-windows-postgresql17",
    }


# -----------------------------------------------------------------------------
# 2. Executive KPIs & Metrics Summary
# -----------------------------------------------------------------------------
@app.get("/metrics/summary", response_model=MetricsSummaryResponse, tags=["Analytics"])
def get_metrics_summary():
    """Returns top-level financial fraud KPIs and active model performance."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            COUNT(*) AS total_tx,
            COALESCE(SUM(amount), 0.0) AS total_vol,
            COUNT(CASE WHEN is_fraud = TRUE THEN 1 END) AS confirmed_fraud,
            COALESCE(SUM(CASE WHEN is_fraud = TRUE THEN amount ELSE 0 END), 0.0) AS fraud_vol
        FROM transactions;
    """)
    tx_kpi = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM fraud_alerts;")
    total_alerts = cur.fetchone()["count"]

    cur.execute("SELECT COUNT(*) FROM vw_customer_risk_scores WHERE calculated_composite_risk >= 50.0;")
    high_risk_customers = cur.fetchone()["count"]

    cur.execute("""
        SELECT model_name, version, metrics 
        FROM model_versions 
        WHERE is_active = TRUE 
        LIMIT 1;
    """)
    model_row = cur.fetchone()

    cur.close()
    conn.close()

    total_tx = tx_kpi["total_tx"] or 0
    confirmed = tx_kpi["confirmed_fraud"] or 0
    detection_rate = round((confirmed / total_tx * 100), 2) if total_tx > 0 else 0.0

    model_metrics = model_row["metrics"] if model_row and "metrics" in model_row else {}

    return MetricsSummaryResponse(
        total_transactions=total_tx,
        total_fraud_alerts=total_alerts,
        high_risk_customers_count=high_risk_customers,
        confirmed_fraud_count=confirmed,
        fraud_detection_rate_pct=detection_rate,
        active_model_name=model_row["model_name"] if model_row else "XGBoost",
        active_model_version=model_row["version"] if model_row else "v1.0.0-xgb",
        model_pr_auc=float(model_metrics.get("pr_auc", 0.9259)),
        model_roc_auc=float(model_metrics.get("roc_auc", 0.9856)),
        total_settled_volume_usd=float(round(tx_kpi["total_vol"], 2)),
        total_fraud_volume_usd=float(round(tx_kpi["fraud_vol"], 2)),
    )


# -----------------------------------------------------------------------------
# 3. Paginated Transactions Feed
# -----------------------------------------------------------------------------
@app.get("/transactions", response_model=List[TransactionListItem], tags=["Transactions"])
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    is_fraud: Optional[bool] = None,
    account_id: Optional[str] = None,
):
    """Retrieves paginated transactions with enrichment data."""
    conn = get_db()
    cur = conn.cursor()

    conditions = ["1=1"]
    params = []

    if status:
        conditions.append("t.status = %s")
        params.append(status)
    if is_fraud is not None:
        conditions.append("t.is_fraud = %s")
        params.append(is_fraud)
    if account_id:
        conditions.append("t.account_id = %s")
        params.append(account_id)

    where_clause = " AND ".join(conditions)
    query = f"""
    SELECT 
        t.transaction_id, t.account_id, t.amount, t.transaction_type,
        t.channel, t.status, t.is_fraud, t.timestamp,
        c.first_name || ' ' || c.last_name AS customer_name,
        m.merchant_name
    FROM transactions t
    JOIN accounts a ON t.account_id = a.account_id
    JOIN customers c ON a.customer_id = c.customer_id
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    WHERE {where_clause}
    ORDER BY t.timestamp DESC
    LIMIT %s OFFSET %s;
    """
    params.extend([limit, offset])
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        results.append(
            TransactionListItem(
                transaction_id=str(r["transaction_id"]),
                account_id=str(r["account_id"]),
                customer_name=r["customer_name"],
                merchant_name=r["merchant_name"],
                amount=float(r["amount"]),
                transaction_type=r["transaction_type"],
                channel=r["channel"],
                status=r["status"],
                is_fraud=bool(r["is_fraud"]),
                timestamp=str(r["timestamp"]),
                surveillance_verdict="FLAGGED" if r["is_fraud"] or r["status"] == "FLAGGED" else "NORMAL",
            )
        )
    return results


# -----------------------------------------------------------------------------
# 4. Paginated Fraud Alerts Feed
# -----------------------------------------------------------------------------
@app.get("/alerts", response_model=List[AlertListItem], tags=["Alerts"])
def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    severity: Optional[str] = None,
    status: Optional[str] = None,
):
    """Retrieves paginated fraud alerts with metadata."""
    conn = get_db()
    cur = conn.cursor()

    conditions = ["1=1"]
    params = []
    if severity:
        conditions.append("severity = %s")
        params.append(severity)
    if status:
        conditions.append("status = %s")
        params.append(status)

    where_clause = " AND ".join(conditions)
    query = f"""
    SELECT alert_id, transaction_id, account_id, rule_name, rule_description,
           risk_score, severity, status, metadata, created_at
    FROM fraud_alerts
    WHERE {where_clause}
    ORDER BY created_at DESC
    LIMIT %s OFFSET %s;
    """
    params.extend([limit, offset])
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        meta = r["metadata"] if isinstance(r["metadata"], dict) else {}
        results.append(
            AlertListItem(
                alert_id=str(r["alert_id"]),
                transaction_id=str(r["transaction_id"]),
                account_id=str(r["account_id"]),
                rule_name=r["rule_name"],
                rule_description=r["rule_description"],
                risk_score=float(r["risk_score"]),
                severity=r["severity"],
                status=r["status"],
                metadata=meta,
                created_at=str(r["created_at"]),
            )
        )
    return results


# -----------------------------------------------------------------------------
# 5. Customer 360 Risk Profile
# -----------------------------------------------------------------------------
@app.get("/customers/{customer_id}/risk", response_model=CustomerRiskProfileResponse, tags=["Customers"])
def get_customer_risk_profile(customer_id: str):
    """Returns composite risk score, recent activity, alerts, and shared device linkages."""
    conn = get_db()
    cur = conn.cursor()

    # 1. Risk View Profile
    cur.execute("SELECT * FROM vw_customer_risk_scores WHERE customer_id = %s;", (customer_id,))
    profile = cur.fetchone()
    if not profile:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Customer ID not found.")

    # 2. Recent Transactions
    cur.execute("""
        SELECT t.transaction_id, t.amount, t.transaction_type, t.channel,
               t.status, t.is_fraud, t.timestamp, m.merchant_name
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE a.customer_id = %s
        ORDER BY t.timestamp DESC
        LIMIT 10;
    """, (customer_id,))
    recent_txs = [dict(r) for r in cur.fetchall()]

    # 3. Customer Alerts
    cur.execute("""
        SELECT fa.alert_id, fa.rule_name, fa.risk_score, fa.severity, fa.status, fa.created_at
        FROM fraud_alerts fa
        JOIN accounts a ON fa.account_id = a.account_id
        WHERE a.customer_id = %s
        ORDER BY fa.created_at DESC
        LIMIT 10;
    """, (customer_id,))
    alerts = [dict(r) for r in cur.fetchall()]

    # 4. Shared Device / IP links
    cur.execute("""
        SELECT DISTINCT t.device_id, t.ip_address, COUNT(DISTINCT t2.account_id) AS shared_with_accounts
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        JOIN transactions t2 ON (t.device_id = t2.device_id OR t.ip_address = t2.ip_address)
        WHERE a.customer_id = %s AND t2.account_id != t.account_id
        GROUP BY t.device_id, t.ip_address
        LIMIT 5;
    """, (customer_id,))
    shared_infra = [dict(r) for r in cur.fetchall()]

    cur.close()
    conn.close()

    return CustomerRiskProfileResponse(
        customer_id=str(profile["customer_id"]),
        first_name=profile["first_name"],
        last_name=profile["last_name"],
        email=profile["email"],
        customer_status=profile["customer_status"],
        base_kyc_risk=float(profile["base_kyc_risk"]),
        calculated_composite_risk=float(profile["calculated_composite_risk"]),
        risk_tier=profile["risk_tier"],
        risk_rank=profile["risk_rank"],
        risk_percentile=float(profile["risk_percentile"]),
        total_transactions=profile["total_transactions"],
        total_spent=float(profile["total_spent"]),
        active_accounts_count=profile["active_accounts_count"],
        total_balance=float(profile["total_balance"]),
        anomalous_transactions_count=profile["anomalous_transactions_count"],
        verified_fraud_count=profile["verified_fraud_count"],
        total_alerts_raised=profile["total_alerts_raised"],
        critical_alerts_count=profile["critical_alerts_count"],
        recent_transactions=recent_txs,
        associated_alerts=alerts,
        shared_infrastructure=shared_infra,
    )


# -----------------------------------------------------------------------------
# 6. Network Graph Analytics Endpoint
# -----------------------------------------------------------------------------
@app.get("/graph/network", response_model=NetworkGraphResponse, tags=["Graph Intelligence"])
def get_graph_network(max_nodes: int = Query(80, ge=10, le=200)):
    """Returns subnetwork nodes and edges for visualizing rings and shared infrastructure."""
    engine = OptimizedFraudGraphEngine(recent_limit=5000)
    summary = engine.export_summary()
    subgraph = engine.export_subgraph_json(max_nodes=max_nodes)

    return NetworkGraphResponse(
        total_nodes=summary["total_nodes"],
        total_edges=summary["total_edges"],
        shared_infrastructure_count=summary["shared_infrastructure_count"],
        merchant_fraud_rings_count=summary["merchant_fraud_rings_count"],
        nodes=subgraph["nodes"],
        edges=subgraph["edges"],
    )


# -----------------------------------------------------------------------------
# 7. Real-Time Transaction Scoring Endpoint
# -----------------------------------------------------------------------------
@app.post("/transactions/score", response_model=TransactionScoreResponse, tags=["Scoring"])
def score_transaction(req: TransactionScoreRequest):
    """
    Evaluates incoming transaction against XGBoost ML model + rules + SHAP,
    stores results to PostgreSQL, and triggers alerts if high-risk.
    """
    predictor = get_predictor()
    tx_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Convert request to single-row DataFrame
    event = {
        "transaction_id": tx_id,
        "account_id": req.account_id,
        "merchant_id": req.merchant_id,
        "amount": req.amount,
        "currency": "USD",
        "transaction_type": req.transaction_type,
        "channel": req.channel,
        "status": "COMPLETED",
        "step": 744,
        "oldbalance_orig": req.oldbalance_orig if req.oldbalance_orig is not None else req.amount * 1.5,
        "newbalance_orig": req.newbalance_orig if req.newbalance_orig is not None else max(0.0, req.amount * 0.5),
        "oldbalance_dest": req.oldbalance_dest,
        "newbalance_dest": req.newbalance_dest,
        "name_orig": req.name_orig or f"ACC_{req.account_id[:6]}",
        "name_dest": req.name_dest or "MERC_DEST",
        "is_fraud": False,
        "is_flagged_fraud": False,
        "ip_address": req.ip_address or "127.0.0.1",
        "device_id": req.device_id or "dev_direct_api",
        "timestamp": now,
        "merchant_risk_level": "LOW",
        "merchant_mcc": "0000",
    }

    df_single = pd.DataFrame([event])
    predictions = predictor.predict_batch(df_single, compute_shap=True, top_k_features=4)
    pred = predictions[0]

    prob = pred["fraud_probability"]
    risk_tier = pred["risk_tier"]
    label = pred["prediction_label"]

    # Status Determination
    status = "FLAGGED" if risk_tier in ("HIGH", "CRITICAL") else "COMPLETED"
    is_flagged = (status == "FLAGGED")

    # Rule checks
    triggered_rules = []
    if req.amount > 10000.0:
        triggered_rules.append("RULE_HIGH_AMOUNT_EXCEEDED")
    if is_flagged:
        triggered_rules.append("ML_XGBOOST_RISK_SPIKE")

    # Persist to database
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO transactions (
            transaction_id, account_id, merchant_id, amount, currency,
            transaction_type, channel, status, step, oldbalance_orig,
            newbalance_orig, oldbalance_dest, newbalance_dest, name_orig,
            name_dest, is_fraud, is_flagged_fraud, ip_address, device_id, timestamp
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
    """, (
        tx_id, req.account_id, req.merchant_id, req.amount, "USD",
        req.transaction_type, req.channel, status, event["step"],
        event["oldbalance_orig"], event["newbalance_orig"],
        req.oldbalance_dest, req.newbalance_dest, event["name_orig"],
        event["name_dest"], is_flagged, is_flagged and req.amount > 200000,
        event["ip_address"], event["device_id"], now
    ))

    # Save to fraud_predictions
    cur.execute("""
        INSERT INTO fraud_predictions (
            prediction_id, transaction_id, model_version, fraud_probability,
            prediction_label, risk_tier, top_features, inference_latency_ms, predicted_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        pred["prediction_id"], tx_id, pred["model_version"], prob,
        label, risk_tier, Json(pred["top_features"]),
        pred["inference_latency_ms"], now
    ))

    # If flagged, save alert
    if is_flagged:
        cur.execute("""
            INSERT INTO fraud_alerts (
                alert_id, transaction_id, account_id, rule_name, rule_description,
                risk_score, severity, status, metadata, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            str(uuid.uuid4()), tx_id, req.account_id,
            "REALTIME_API_FLAGGED_TRANSACTION",
            f"Transaction evaluated with {prob*100:.1f}% fraud probability.",
            round(prob * 100.0, 2), risk_tier, "OPEN",
            Json(pred["top_features"]), now
        ))

    conn.commit()
    cur.close()
    conn.close()

    return TransactionScoreResponse(
        transaction_id=tx_id,
        account_id=req.account_id,
        amount=req.amount,
        fraud_probability=prob,
        prediction_label=label,
        risk_tier=risk_tier,
        is_flagged=is_flagged,
        status=status,
        rule_alerts_triggered=triggered_rules,
        top_shap_features=pred["top_features"],
        inference_latency_ms=pred["inference_latency_ms"],
        surveillance_verdict="FLAGGED" if is_flagged else "NORMAL",
    )


# -----------------------------------------------------------------------------
# 8. Autonomous AI Investigation Endpoint
# -----------------------------------------------------------------------------
@app.post("/investigate/{transaction_id}", response_model=InvestigationReportResponse, tags=["Forensics"])
def investigate_transaction(transaction_id: str):
    """Invokes the AI Investigation Agent to generate and persist a forensic dossier."""
    investigator = get_investigator()
    try:
        report = investigator.generate_report(transaction_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic investigation failed: {e}")

    return InvestigationReportResponse(
        report_id=report["report_id"],
        transaction_id=report["transaction_id"],
        account_id=report.get("account_id"),
        investigator_id=report["investigator_id"],
        report_title=report["report_title"],
        executive_summary=report["executive_summary"],
        recommended_action=report["recommended_action"],
        status=report["status"],
        genai_narrative=report["genai_narrative"],
        evidence_summary=report["evidence_summary"],
        created_at=str(report["created_at"]),
    )
