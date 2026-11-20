"""
FraudShield AI - Database & Analytical Views Test Suite
Tests connection, table row counts, and SQL view integrity.
"""

import sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from ml.train_models import get_db_connection


def test_database_connection():
    """Verify active connection to local PostgreSQL 17 fraud_db."""
    conn = get_db_connection()
    assert conn is not None
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    res = cur.fetchone()[0]
    assert res == 1
    cur.close()
    conn.close()


def test_table_row_counts():
    """Ensure all core tables have ingested data."""
    conn = get_db_connection()
    cur = conn.cursor()
    for table in ["customers", "accounts", "merchants", "transactions", "fraud_alerts", "model_versions"]:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        assert count > 0, f"Table {table} has 0 rows"
    cur.close()
    conn.close()


def test_analytical_views():
    """Validates that all 3 SQL views return rows with window-function columns populated."""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Flagged transactions view
    cur.execute("SELECT transaction_id, spending_spike_ratio, surveillance_verdict FROM vw_flagged_transactions LIMIT 5;")
    rows = cur.fetchall()
    assert len(rows) == 5

    # 2. Customer risk scores view
    cur.execute("SELECT customer_id, calculated_composite_risk, risk_tier, risk_rank FROM vw_customer_risk_scores LIMIT 5;")
    cust_rows = cur.fetchall()
    assert len(cust_rows) == 5
    assert cust_rows[0][1] >= 0.0

    # 3. Merchant risk summary view
    cur.execute("SELECT merchant_id, alert_rate_pct, category_avg_ticket_size FROM vw_merchant_risk_summary LIMIT 5;")
    merch_rows = cur.fetchall()
    assert len(merch_rows) == 5

    cur.close()
    conn.close()
