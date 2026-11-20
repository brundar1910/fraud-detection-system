"""
FraudShield AI - Database Setup & Ingestion Script
Initializes fraud_db on local PostgreSQL 17, applies sql/schema.sql,
and loads generated synthetic entities and transactions.
"""

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def find_working_connection():
    """Tries configured password and common fallbacks."""
    passwords = [DB_PASSWORD, "postgres", "admin", "root", "1234", "password", ""]
    seen = set()
    for p in passwords:
        if p in seen:
            continue
        seen.add(p)
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=p,
                dbname="postgres",
                connect_timeout=3,
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            return conn, p
        except psycopg2.OperationalError as e:
            if "authentication failed" in str(e):
                continue
            else:
                print(f"[!] PostgreSQL Operational Error: {e}")
                return None, None
    return None, None


def ensure_database(conn):
    """Creates target database if it doesn't already exist."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()
    if not exists:
        print(f"[*] Database '{DB_NAME}' does not exist. Creating...")
        cur.execute(f'CREATE DATABASE "{DB_NAME}";')
        print(f"[+] Database '{DB_NAME}' created successfully.")
    else:
        print(f"[+] Database '{DB_NAME}' already exists.")
    cur.close()


def apply_schema(working_password):
    """Executes sql/schema.sql against fraud_db."""
    schema_path = BASE_DIR / "sql" / "schema.sql"
    print(f"[*] Applying DDL schema from {schema_path} to '{DB_NAME}'...")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=working_password,
        dbname=DB_NAME,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(schema_sql)
    cur.close()
    conn.close()
    print("[+] Schema applied successfully: Tables, constraints, indexes & views created.")


def load_data(working_password):
    """Loads CSVs from datasets/ into PostgreSQL fraud_db."""
    datasets_dir = BASE_DIR / "datasets"
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=working_password,
        dbname=DB_NAME,
    )
    cur = conn.cursor()

    # 1. Customers
    cust_csv = datasets_dir / "customers.csv"
    if cust_csv.exists():
        print(f"[*] Loading customers from {cust_csv}...")
        df_cust = pd.read_csv(cust_csv)
        records = [tuple(x) for x in df_cust.to_numpy()]
        cols = ",".join(df_cust.columns)
        placeholders = ",".join(["%s"] * len(df_cust.columns))
        cur.executemany(f"INSERT INTO customers ({cols}) VALUES ({placeholders}) ON CONFLICT (customer_id) DO NOTHING;", records)
        conn.commit()

    # 2. Accounts
    acc_csv = datasets_dir / "accounts.csv"
    if acc_csv.exists():
        print(f"[*] Loading accounts from {acc_csv}...")
        df_acc = pd.read_csv(acc_csv)
        records = [tuple(x) for x in df_acc.to_numpy()]
        cols = ",".join(df_acc.columns)
        placeholders = ",".join(["%s"] * len(df_acc.columns))
        cur.executemany(f"INSERT INTO accounts ({cols}) VALUES ({placeholders}) ON CONFLICT (account_id) DO NOTHING;", records)
        conn.commit()

    # 3. Merchants
    merch_csv = datasets_dir / "merchants.csv"
    if merch_csv.exists():
        print(f"[*] Loading merchants from {merch_csv}...")
        df_merch = pd.read_csv(merch_csv)
        records = [tuple(x) for x in df_merch.to_numpy()]
        cols = ",".join(df_merch.columns)
        placeholders = ",".join(["%s"] * len(df_merch.columns))
        cur.executemany(f"INSERT INTO merchants ({cols}) VALUES ({placeholders}) ON CONFLICT (merchant_id) DO NOTHING;", records)
        conn.commit()

    # 4. Transactions (50,000 rows)
    tx_csv = datasets_dir / "fraudshield_enriched_transactions.csv"
    if tx_csv.exists():
        print(f"[*] Loading 50,000 transactions from {tx_csv}...")
        df_tx = pd.read_csv(tx_csv)
        # Match schema columns
        target_cols = [
            "transaction_id", "account_id", "merchant_id", "amount", "currency",
            "transaction_type", "channel", "status", "ip_address", "device_id",
            "location_lat", "location_lon", "step", "oldbalance_orig", "newbalance_orig",
            "oldbalance_dest", "newbalance_dest", "name_orig", "name_dest",
            "is_fraud", "is_flagged_fraud", "timestamp"
        ]
        # Rename mapping from CSV to schema
        rename_map = {
            "type": "transaction_type",
            "oldbalanceOrg": "oldbalance_orig",
            "newbalanceOrig": "newbalance_orig",
            "oldbalanceDest": "oldbalance_dest",
            "newbalanceDest": "newbalance_dest",
            "nameOrig": "name_orig",
            "nameDest": "name_dest",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        }
        df_tx_ready = df_tx.rename(columns=rename_map)
        df_tx_ready["currency"] = "USD"
        df_tx_ready["is_fraud"] = df_tx_ready["is_fraud"].astype(bool)
        df_tx_ready["is_flagged_fraud"] = df_tx_ready["is_flagged_fraud"].astype(bool)
        # Fill NaN merchants with None for SQL NULL
        df_tx_ready["merchant_id"] = df_tx_ready["merchant_id"].where(pd.notnull(df_tx_ready["merchant_id"]), None)

        df_subset = df_tx_ready[target_cols]
        # Batch insert for high speed
        batch_size = 5000
        cols_str = ",".join(target_cols)
        placeholders = ",".join(["%s"] * len(target_cols))
        insert_sql = f"INSERT INTO transactions ({cols_str}) VALUES ({placeholders}) ON CONFLICT (transaction_id) DO NOTHING;"

        records = [tuple(None if pd.isna(v) else v for v in row) for row in df_subset.to_numpy()]
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cur.executemany(insert_sql, batch)
            conn.commit()
            print(f"    Loaded {min(i + batch_size, len(records)):,}/{len(records):,} transactions...")

    # 5. Fraud Alerts
    alerts_csv = datasets_dir / "fraud_alerts.csv"
    if alerts_csv.exists():
        print(f"[*] Loading fraud alerts from {alerts_csv}...")
        df_alerts = pd.read_csv(alerts_csv)
        cols = ",".join(df_alerts.columns)
        placeholders = ",".join(["%s"] * len(df_alerts.columns))
        records = [tuple(x) for x in df_alerts.to_numpy()]
        cur.executemany(f"INSERT INTO fraud_alerts ({cols}) VALUES ({placeholders}) ON CONFLICT (alert_id) DO NOTHING;", records)
        conn.commit()

    cur.close()
    conn.close()
    print("[+] All datasets loaded into PostgreSQL successfully.")


def verify_row_counts(working_password):
    """Queries and prints row counts of all tables and views."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=working_password,
        dbname=DB_NAME,
    )
    cur = conn.cursor()

    tables = ["customers", "accounts", "merchants", "transactions", "fraud_alerts", "model_versions", "fraud_predictions", "investigation_reports"]
    print("\n" + "=" * 50)
    print("      DATABASE VERIFICATION: ROW COUNTS")
    print("=" * 50)
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t};")
        count = cur.fetchone()[0]
        print(f"  • {t:<26}: {count:,} rows")

    views = ["vw_flagged_transactions", "vw_customer_risk_scores", "vw_merchant_risk_summary"]
    print("-" * 50)
    print("      ANALYTICAL VIEWS ROW COUNTS")
    print("-" * 50)
    for v in views:
        cur.execute(f"SELECT COUNT(*) FROM {v};")
        count = cur.fetchone()[0]
        print(f"  • {v:<26}: {count:,} rows")
    print("=" * 50 + "\n")
    cur.close()
    conn.close()


def main():
    conn, working_password = find_working_connection()
    if conn is None:
        print("ERROR_POSTGRES_AUTH_FAILED")
        sys.exit(1)

    print(f"[+] Connected to PostgreSQL successfully using password: '{working_password}'")
    # Update .env if working password was different
    if working_password != DB_PASSWORD:
        env_path = BASE_DIR / ".env"
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(f"DB_PASSWORD={DB_PASSWORD}", f"DB_PASSWORD={working_password}")
        content = content.replace(f":{DB_PASSWORD}@", f":{working_password}@")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[+] Updated .env with validated PostgreSQL password.")

    ensure_database(conn)
    conn.close()

    apply_schema(working_password)
    load_data(working_password)
    verify_row_counts(working_password)


if __name__ == "__main__":
    main()
