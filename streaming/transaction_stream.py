"""
FraudShield AI - Real-Time Financial Transaction Streaming Engine
Simulates a live event producer/consumer pipeline running natively on Windows:
  1. Generates realistic live financial transaction events every 2-3 seconds.
  2. Injects periodic high-risk fraud anomalies (bursts, spikes, rings).
  3. Evaluates transactions in real time through the trained XGBoost model + SHAP.
  4. Persists scored events, predictions, and alerts into PostgreSQL 17.
"""

import os
import sys
import time
import uuid
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from ml.predict import FraudPredictor
from ml.train_models import get_db_connection


class RealTimeTransactionStreamer:
    """Live streaming producer/consumer pipeline."""

    def __init__(self, interval_seconds: float = 2.0):
        self.interval = interval_seconds
        self.predictor = FraudPredictor()
        self.accounts: List[Dict[str, Any]] = []
        self.merchants: List[Dict[str, Any]] = []
        self.load_reference_entities()

    def load_reference_entities(self):
        """Loads active accounts and merchants from PostgreSQL for stream sampling."""
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT account_id, account_number, balance FROM accounts WHERE status = 'ACTIVE' LIMIT 200;")
        self.accounts = [
            {"account_id": str(r[0]), "account_number": r[1], "balance": float(r[2])}
            for r in cur.fetchall()
        ]

        cur.execute("SELECT merchant_id, merchant_name, category_code, risk_level FROM merchants WHERE status = 'ACTIVE';")
        self.merchants = [
            {"merchant_id": str(r[0]), "merchant_name": r[1], "category_code": r[2], "risk_level": r[3]}
            for r in cur.fetchall()
        ]

        cur.close()
        conn.close()
        print(f"[+] Streaming engine initialized with {len(self.accounts)} accounts and {len(self.merchants)} merchants.")

    def generate_single_event(self, force_fraud: bool = False) -> Dict[str, Any]:
        """Synthesizes a realistic live transaction event."""
        acc = random.choice(self.accounts)
        acc_id = acc["account_id"]
        bal = acc["balance"]

        # Anomaly / Fraud injection
        if force_fraud or random.random() < 0.15:  # 15% streaming anomaly probability
            is_fraud = True
            pattern = random.choice(["SPENDING_SPIKE", "SHELL_TRANSFER", "CARD_TESTING"])
            if pattern == "SPENDING_SPIKE":
                amount = round(random.uniform(3500.0, 18000.0), 2)
                txn_type = "TRANSFER"
                channel = "ONLINE"
            elif pattern == "SHELL_TRANSFER":
                amount = round(random.uniform(1500.0, 7500.0), 2)
                txn_type = "PAYMENT"
                channel = "API"
            else:
                amount = round(random.uniform(45.0, 350.0), 2)
                txn_type = "PAYMENT"
                channel = "ONLINE"
        else:
            is_fraud = False
            amount = round(float(random.uniform(10.0, 320.0)), 2)
            txn_type = random.choice(["PAYMENT", "CASH_OUT", "DEBIT", "TRANSFER"])
            channel = random.choice(["ONLINE", "MOBILE_APP", "IN_STORE", "ATM"])

        new_bal = round(max(0.0, bal - amount), 2)

        if txn_type in ("PAYMENT", "DEBIT"):
            merch = random.choice(self.merchants)
            merchant_id = merch["merchant_id"]
            name_dest = f"M{merch['category_code']}_{merch['merchant_name'][:8]}"
            merchant_risk = merch["risk_level"]
            mcc = merch["category_code"]
        else:
            merchant_id = None
            dest_acc = random.choice(self.accounts)
            name_dest = dest_acc["account_number"]
            merchant_risk = "LOW"
            mcc = "0000"

        return {
            "transaction_id": str(uuid.uuid4()),
            "account_id": acc_id,
            "merchant_id": merchant_id,
            "amount": amount,
            "currency": "USD",
            "transaction_type": txn_type,
            "channel": channel,
            "status": "COMPLETED",
            "step": random.randint(700, 744),
            "oldbalance_orig": bal,
            "newbalance_orig": new_bal,
            "oldbalance_dest": 0.0,
            "newbalance_dest": 0.0,
            "name_orig": acc["account_number"],
            "name_dest": name_dest,
            "is_fraud": is_fraud,
            "is_flagged_fraud": False,
            "ip_address": f"192.168.1.{random.randint(10, 250)}",
            "device_id": f"dev_{uuid.uuid4().hex[:12]}",
            "location_lat": 37.7749 + random.uniform(-0.05, 0.05),
            "location_lon": -122.4194 + random.uniform(-0.05, 0.05),
            "timestamp": datetime.now(timezone.utc),
            "merchant_risk_level": merchant_risk,
            "merchant_mcc": mcc,
        }

    def process_and_persist_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Scores transaction with XGBoost/SHAP and records to PostgreSQL."""
        df_single = pd.DataFrame([event])
        predictions = self.predictor.predict_batch(df_single, compute_shap=True, top_k_features=3)
        pred = predictions[0]

        # Update status if model flags it as high-risk
        status = "FLAGGED" if pred["risk_tier"] in ("HIGH", "CRITICAL") else "COMPLETED"
        event["status"] = status

        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Insert into transactions
        tx_sql = """
        INSERT INTO transactions (
            transaction_id, account_id, merchant_id, amount, currency,
            transaction_type, channel, status, step, oldbalance_orig,
            newbalance_orig, oldbalance_dest, newbalance_dest, name_orig,
            name_dest, is_fraud, is_flagged_fraud, ip_address, device_id,
            location_lat, location_lon, timestamp
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        cur.execute(
            tx_sql,
            (
                event["transaction_id"],
                event["account_id"],
                event["merchant_id"],
                event["amount"],
                event["currency"],
                event["transaction_type"],
                event["channel"],
                status,
                event["step"],
                event["oldbalance_orig"],
                event["newbalance_orig"],
                event["oldbalance_dest"],
                event["newbalance_dest"],
                event["name_orig"],
                event["name_dest"],
                event["is_fraud"],
                event["is_flagged_fraud"],
                event["ip_address"],
                event["device_id"],
                event["location_lat"],
                event["location_lon"],
                event["timestamp"],
            ),
        )

        # 2. Insert into fraud_predictions
        pred_sql = """
        INSERT INTO fraud_predictions (
            prediction_id, transaction_id, model_version, fraud_probability,
            prediction_label, risk_tier, top_features, inference_latency_ms, predicted_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            pred_sql,
            (
                pred["prediction_id"],
                pred["transaction_id"],
                pred["model_version"],
                pred["fraud_probability"],
                pred["prediction_label"],
                pred["risk_tier"],
                Json(pred["top_features"]),
                pred["inference_latency_ms"],
                event["timestamp"],
            ),
        )

        # 3. Insert into fraud_alerts if high/critical
        alert_raised = False
        if pred["risk_tier"] in ("HIGH", "CRITICAL"):
            alert_sql = """
            INSERT INTO fraud_alerts (
                alert_id, transaction_id, account_id, rule_name, rule_description,
                risk_score, severity, status, metadata, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            cur.execute(
                alert_sql,
                (
                    str(uuid.uuid4()),
                    event["transaction_id"],
                    event["account_id"],
                    "REALTIME_STREAM_ML_ALERT",
                    f"Live streamed transaction flagged with {pred['fraud_probability']*100:.1f}% risk.",
                    round(pred["fraud_probability"] * 100.0, 2),
                    pred["risk_tier"],
                    "OPEN",
                    Json({"top_features": pred["top_features"]}),
                    event["timestamp"],
                ),
            )
            alert_raised = True

        conn.commit()
        cur.close()
        conn.close()

        return {
            "transaction_id": event["transaction_id"],
            "account_number": event["name_orig"],
            "amount": event["amount"],
            "type": event["transaction_type"],
            "risk_tier": pred["risk_tier"],
            "probability": pred["fraud_probability"],
            "status": status,
            "alert_raised": alert_raised,
            "top_driver": list(pred["top_features"].keys())[0] if pred["top_features"] else "N/A",
        }

    def start_streaming(self, max_count: int = 0, duration_seconds: int = 0):
        """Continuously streams events."""
        print(f"[*] Starting live transaction stream (interval: {self.interval}s)...")
        print("    [Press Ctrl+C to stop]")
        t_start = time.time()
        count = 0

        try:
            while True:
                # Stop conditions
                if max_count > 0 and count >= max_count:
                    print(f"[*] Completed target stream count of {max_count} events.")
                    break
                if duration_seconds > 0 and (time.time() - t_start) >= duration_seconds:
                    print(f"[*] Completed streaming duration of {duration_seconds} seconds.")
                    break

                count += 1
                event = self.generate_single_event()
                res = self.process_and_persist_event(event)

                tier_symbol = "[ALERT] " if res["risk_tier"] in ("HIGH", "CRITICAL") else "[NORMAL]"
                print(
                    f"[{count:04d}] {tier_symbol} TX: {res['transaction_id'][:8]}... | "
                    f"Acc: {res['account_number']} | "
                    f"${res['amount']:>8.2f} {res['type']:<8} | "
                    f"Risk: {res['risk_tier']:<8} ({res['probability']*100:>5.1f}%) | "
                    f"Driver: {res['top_driver']}"
                )
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n[*] Live streaming halted gracefully.")


def main():
    parser = argparse.ArgumentParser(description="FraudShield AI Real-Time Transaction Streamer")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between events in seconds")
    parser.add_argument("--count", type=int, default=5, help="Number of events to generate (0 for infinite)")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds to stream (0 for infinite)")
    args = parser.parse_args()

    streamer = RealTimeTransactionStreamer(interval_seconds=args.interval)
    streamer.start_streaming(max_count=args.count, duration_seconds=args.duration)


if __name__ == "__main__":
    main()
