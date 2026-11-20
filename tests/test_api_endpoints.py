"""
FraudShield AI - FastAPI Endpoints Integration Test Suite
Validates all REST endpoints against the live PostgreSQL database and XGBoost/SHAP engine.
"""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app, get_db

client = TestClient(app)


def test_health_endpoint():
    """Validates /health returns healthy status for DB and ML model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["database"] == "healthy"
    assert data["ml_engine"] == "ready"
    assert "model_details" in data
    print("\n[+] /health endpoint: PASS")


def test_metrics_summary():
    """Validates /metrics/summary returns accurate financial KPIs."""
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] >= 50000
    assert data["total_fraud_alerts"] > 0
    assert data["active_model_name"] != ""
    assert data["model_pr_auc"] > 0.80
    print(f"[+] /metrics/summary endpoint: PASS (Total TX: {data['total_transactions']:,}, Model: {data['active_model_name']})")


def test_transactions_list():
    """Validates /transactions with pagination and filters."""
    response = client.get("/transactions?limit=10&offset=0")
    assert response.status_code == 200
    txs = response.json()
    assert len(txs) == 10
    assert "transaction_id" in txs[0]
    assert "amount" in txs[0]
    print("[+] /transactions endpoint: PASS")


def test_alerts_list():
    """Validates /alerts returns paginated fraud alerts."""
    response = client.get("/alerts?limit=5")
    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) == 5
    assert "alert_id" in alerts[0]
    assert "severity" in alerts[0]
    print("[+] /alerts endpoint: PASS")


def test_customer_risk_profile():
    """Validates /customers/{customer_id}/risk returns complete 360 profile."""
    # Fetch a valid customer_id
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT customer_id FROM customers LIMIT 1;")
    cust_id = str(cur.fetchone()["customer_id"])
    cur.close()
    conn.close()

    response = client.get(f"/customers/{cust_id}/risk")
    assert response.status_code == 200
    profile = response.json()
    assert profile["customer_id"] == cust_id
    assert "calculated_composite_risk" in profile
    assert "risk_tier" in profile
    print(f"[+] /customers/{{customer_id}}/risk: PASS (Customer: {profile['first_name']} {profile['last_name']}, Tier: {profile['risk_tier']})")


def test_graph_network():
    """Validates /graph/network returns nodes and edges for visualization."""
    response = client.get("/graph/network?max_nodes=30")
    assert response.status_code == 200
    graph_data = response.json()
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert len(graph_data["nodes"]) > 0
    print(f"[+] /graph/network: PASS (Returned {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges)")


def test_transaction_scoring_api():
    """Validates POST /transactions/score runs live XGBoost + SHAP inference and saves to DB."""
    # Fetch a valid account_id
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT account_id FROM accounts LIMIT 1;")
    acc_id = str(cur.fetchone()["account_id"])
    cur.close()
    conn.close()

    # Normal transaction
    payload_normal = {
        "account_id": acc_id,
        "amount": 42.50,
        "transaction_type": "PAYMENT",
        "channel": "IN_STORE",
        "ip_address": "127.0.0.1",
        "device_id": "dev_test_normal",
    }
    res_normal = client.post("/transactions/score", json=payload_normal)
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert data_normal["status"] == "COMPLETED"
    assert data_normal["risk_tier"] in ("LOW", "MEDIUM")

    # High-risk anomalous transaction
    payload_fraud = {
        "account_id": acc_id,
        "amount": 18500.00,
        "transaction_type": "TRANSFER",
        "channel": "ONLINE",
        "oldbalance_orig": 18500.00,
        "newbalance_orig": 0.00,
        "ip_address": "75.4.103.151",
        "device_id": "473f6928eeccb38da5152341879606f1",
    }
    res_fraud = client.post("/transactions/score", json=payload_fraud)
    assert res_fraud.status_code == 200
    data_fraud = res_fraud.json()
    assert data_fraud["status"] == "FLAGGED"
    assert data_fraud["risk_tier"] in ("HIGH", "CRITICAL")
    assert data_fraud["fraud_probability"] > 0.80
    assert "top_shap_features" in data_fraud
    print(f"[+] POST /transactions/score: PASS (Flagged Score: {data_fraud['fraud_probability']*100:.1f}%, Tier: {data_fraud['risk_tier']})")


def test_investigate_api():
    """Validates POST /investigate/{transaction_id} generates and saves forensic report."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT transaction_id FROM transactions WHERE is_fraud = TRUE LIMIT 1;")
    tx_id = str(cur.fetchone()["transaction_id"])
    cur.close()
    conn.close()

    response = client.post(f"/investigate/{tx_id}")
    assert response.status_code == 200
    report = response.json()
    assert report["transaction_id"] == tx_id
    assert "recommended_action" in report
    assert "genai_narrative" in report
    assert len(report["genai_narrative"]) > 100
    print(f"[+] POST /investigate/{{transaction_id}}: PASS (Action: {report['recommended_action']}, Report ID: {report['report_id']})")


if __name__ == "__main__":
    test_health_endpoint()
    test_metrics_summary()
    test_transactions_list()
    test_alerts_list()
    test_customer_risk_profile()
    test_graph_network()
    test_transaction_scoring_api()
    test_investigate_api()
    print("\n[SUCCESS] ALL 8 FASTAPI ENDPOINTS PASSED VALIDATION!\n")
