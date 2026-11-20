"""
FraudShield AI - Synthetic Financial Data & Transaction Pipeline Generator
Generates PaySim-style transactions with injected real-world fraud patterns:
  1. Exact duplicates within 2 minutes (Replay / Duplicate charge attack)
  2. >5x spending spikes over historical account baseline (Account Takeover / Carding)
  3. Rapid velocity bursts (5-10 rapid transactions in <5 minutes)
  4. Coordinated fraud rings (Shared shell merchants and collusive mule accounts)

Outputs clean PaySim CSV and enriched relational CSVs ready for PostgreSQL 17 ingestion.
"""

import os
import sys
import uuid
import random
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


class FraudShieldDataPipeline:
    """End-to-end data generator for FraudShield AI."""

    PAYSIM_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

    MERCHANT_CATEGORIES = [
        ("5411", "Grocery Stores / Supermarkets", "LOW"),
        ("5812", "Restaurants and Fast Food", "LOW"),
        ("5732", "Electronics & Computer Stores", "MEDIUM"),
        ("5944", "Jewelry & Luxury Goods", "HIGH"),
        ("4829", "Wire Transfer / Money Remittance", "HIGH"),
        ("7995", "Betting & Casino Gambling", "CRITICAL"),
        ("5311", "Department Stores & Retail", "LOW"),
        ("5999", "Shell Merchant / Uncategorized", "CRITICAL"),
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.customers: List[Dict[str, Any]] = []
        self.accounts: List[Dict[str, Any]] = []
        self.merchants: List[Dict[str, Any]] = []
        self.fraud_ring_mules: List[Dict[str, Any]] = []
        self.fraud_ring_merchants: List[Dict[str, Any]] = []

    def generate_entities(self, num_customers: int = 1500, num_merchants: int = 120):
        """Generate normalized master customers, accounts, and merchants."""
        print(f"[*] Generating {num_customers} customers and {num_merchants} merchants...")
        start_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # 1. Merchants
        for i in range(num_merchants):
            mcc, cat_name, risk = random.choice(self.MERCHANT_CATEGORIES)
            is_shell = (risk == "CRITICAL" and i < 5)
            merchant = {
                "merchant_id": str(uuid.uuid4()),
                "merchant_name": f"Shell Corp {i+1}" if is_shell else f"{self.fake.company()} {self.fake.company_suffix()}",
                "category_code": mcc,
                "category_name": cat_name,
                "risk_level": "CRITICAL" if is_shell else risk,
                "country": "USA",
                "city": self.fake.city(),
                "status": "ACTIVE",
                "created_at": start_date - timedelta(days=random.randint(60, 365)),
                "updated_at": start_date,
            }
            self.merchants.append(merchant)
            if is_shell or risk in ("CRITICAL", "HIGH"):
                self.fraud_ring_merchants.append(merchant)

        # 2. Customers & Accounts
        for i in range(num_customers):
            c_id = str(uuid.uuid4())
            profile = self.fake.simple_profile()
            parts = profile["name"].split(" ", 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else "Smith"
            base_risk = round(random.uniform(0.0, 25.0), 2)
            created_at = start_date - timedelta(days=random.randint(90, 730))

            customer = {
                "customer_id": c_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}.{i}@{self.fake.free_email_domain()}",
                "phone_number": self.fake.phone_number()[:30],
                "date_of_birth": profile["birthdate"],
                "ssn_hash": self.fake.sha256(),
                "address_line1": self.fake.street_address(),
                "address_line2": None,
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "postal_code": self.fake.postcode(),
                "country": "USA",
                "risk_score": base_risk,
                "status": "ACTIVE",
                "created_at": created_at,
                "updated_at": start_date,
            }
            self.customers.append(customer)

            # Assign 1 to 2 accounts per customer
            num_accs = 2 if random.random() < 0.3 else 1
            for j in range(num_accs):
                acc_type = "CHECKING" if j == 0 else "SAVINGS"
                init_balance = round(random.uniform(1000.0, 25000.0), 2)
                account = {
                    "account_id": str(uuid.uuid4()),
                    "customer_id": c_id,
                    "account_number": f"C{i:07d}{j}",  # PaySim style account string
                    "account_type": acc_type,
                    "balance": init_balance,
                    "baseline_avg_spend": round(random.uniform(30.0, 150.0), 2),
                    "currency": "USD",
                    "status": "ACTIVE",
                    "daily_limit": 10000.00,
                    "created_at": created_at,
                    "updated_at": start_date,
                }
                self.accounts.append(account)
                if i < 30:  # Reserve first 30 accounts as potential mule / compromised ring accounts
                    self.fraud_ring_mules.append(account)

        print(f"[+] Created {len(self.customers)} customers, {len(self.accounts)} accounts, {len(self.merchants)} merchants.")

    def generate_transactions(self, target_count: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates PaySim-style and enriched transactions with injected fraud patterns:
          - Pattern 1: Exact Duplicates (<2 mins) (~350 txns)
          - Pattern 2: >5x Spending Spikes (~500 txns)
          - Pattern 3: Rapid Velocity Bursts (~400 txns)
          - Pattern 4: Coordinated Fraud Rings (~500 txns)
          - Normal Transactions: (~48,250 txns)
        """
        print(f"[*] Generating {target_count} transactions across 744 simulation steps (31 days)...")
        start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Track live balances per account
        account_balances = {acc["account_id"]: acc["balance"] for acc in self.accounts}
        account_map = {acc["account_id"]: acc for acc in self.accounts}

        transactions_list: List[Dict[str, Any]] = []

        # Target fraud allocation (~3.5% overall fraud rate)
        num_spikes = 450
        num_burst_series = 60  # ~6 txns each -> 360 txns
        num_duplicates = 350
        num_ring_txns = 450
        num_fraud_total = num_spikes + (num_burst_series * 6) + num_duplicates + num_ring_txns
        num_normal = target_count - num_fraud_total

        # ---------------------------------------------------------------------
        # 1. GENERATE BASELINE NORMAL TRANSACTIONS
        # ---------------------------------------------------------------------
        print(f"[*] Generating {num_normal} normal baseline transactions...")
        for _ in range(num_normal):
            acc = random.choice(self.accounts)
            acc_id = acc["account_id"]
            bal_orig = account_balances[acc_id]

            # Distribution of normal ticket sizes (log-normal, mean ~$65)
            raw_amount = float(np.random.lognormal(mean=3.8, sigma=0.85))
            amount = round(max(2.50, min(raw_amount, bal_orig * 0.40 if bal_orig > 10 else 1500.0)), 2)

            step = random.randint(1, 744)
            minute_offset = random.randint(0, 59)
            second_offset = random.randint(0, 59)
            txn_time = start_time + timedelta(hours=step, minutes=minute_offset, seconds=second_offset)

            txn_type = random.choices(
                ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT", "CASH_IN"],
                weights=[0.45, 0.25, 0.15, 0.10, 0.05],
            )[0]

            if txn_type == "CASH_IN":
                new_bal_orig = round(bal_orig + amount, 2)
            else:
                new_bal_orig = round(max(0.0, bal_orig - amount), 2)
            account_balances[acc_id] = new_bal_orig

            # Destination
            if txn_type in ("PAYMENT", "DEBIT"):
                dest_merchant = random.choice(self.merchants)
                merchant_id = dest_merchant["merchant_id"]
                name_dest = f"M{dest_merchant['category_code']}_{dest_merchant['merchant_name'][:8].replace(' ', '')}"
                old_bal_dest = 0.00
                new_bal_dest = 0.00
            else:
                dest_acc = random.choice(self.accounts)
                merchant_id = None
                name_dest = dest_acc["account_number"]
                old_bal_dest = account_balances[dest_acc["account_id"]]
                new_bal_dest = round(old_bal_dest + amount, 2)
                account_balances[dest_acc["account_id"]] = new_bal_dest

            transactions_list.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": acc_id,
                "merchant_id": merchant_id,
                "step": step,
                "type": txn_type,
                "amount": amount,
                "nameOrig": acc["account_number"],
                "oldbalanceOrg": bal_orig,
                "newbalanceOrig": new_bal_orig,
                "nameDest": name_dest,
                "oldbalanceDest": old_bal_dest,
                "newbalanceDest": new_bal_dest,
                "isFraud": 0,
                "isFlaggedFraud": 0,
                "channel": random.choice(["ONLINE", "MOBILE_APP", "IN_STORE", "ATM"]),
                "status": "COMPLETED",
                "ip_address": self.fake.ipv4(),
                "device_id": self.fake.md5(),
                "location_lat": float(self.fake.latitude()),
                "location_lon": float(self.fake.longitude()),
                "timestamp": txn_time,
                "fraud_pattern": "NORMAL",
            })

        # ---------------------------------------------------------------------
        # 2. INJECT PATTERN 1: EXACT DUPLICATES WITHIN 2 MINUTES
        # ---------------------------------------------------------------------
        print(f"[*] Injecting {num_duplicates} exact duplicate burst transactions (<2 mins)...")
        candidates = random.sample(transactions_list, num_duplicates)
        for base_tx in candidates:
            dup_gap_seconds = random.randint(15, 115)
            dup_time = base_tx["timestamp"] + timedelta(seconds=dup_gap_seconds)
            acc_id = base_tx["account_id"]
            bal = account_balances[acc_id]
            amount = base_tx["amount"]
            new_bal = round(max(0.0, bal - amount), 2)
            account_balances[acc_id] = new_bal

            transactions_list.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": acc_id,
                "merchant_id": base_tx["merchant_id"],
                "step": base_tx["step"],
                "type": base_tx["type"],
                "amount": amount,
                "nameOrig": base_tx["nameOrig"],
                "oldbalanceOrg": bal,
                "newbalanceOrig": new_bal,
                "nameDest": base_tx["nameDest"],
                "oldbalanceDest": base_tx["oldbalanceDest"],
                "newbalanceDest": round(base_tx["oldbalanceDest"] + amount, 2),
                "isFraud": 1,
                "isFlaggedFraud": 1 if amount > 200000 else 0,
                "channel": base_tx["channel"],
                "status": "FLAGGED",
                "ip_address": base_tx["ip_address"],
                "device_id": base_tx["device_id"],
                "location_lat": base_tx["location_lat"],
                "location_lon": base_tx["location_lon"],
                "timestamp": dup_time,
                "fraud_pattern": "EXACT_DUPLICATE_BURST",
            })

        # ---------------------------------------------------------------------
        # 3. INJECT PATTERN 2: >5x SPENDING SPIKES (Account Takeover / Card Drain)
        # ---------------------------------------------------------------------
        print(f"[*] Injecting {num_spikes} >5x spending spike fraud events...")
        for _ in range(num_spikes):
            acc = random.choice(self.accounts)
            acc_id = acc["account_id"]
            baseline = acc["baseline_avg_spend"]
            spike_multiplier = round(random.uniform(5.5, 20.0), 2)
            amount = round(baseline * spike_multiplier, 2)
            if amount < 500.0:
                amount = round(random.uniform(2500.0, 18500.0), 2)

            bal_orig = account_balances[acc_id]
            # Fraud draining account to zero or near zero
            new_bal_orig = 0.00 if random.random() < 0.7 else round(max(0.0, bal_orig - amount), 2)
            account_balances[acc_id] = new_bal_orig

            step = random.randint(10, 740)
            txn_time = start_time + timedelta(hours=step, minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            txn_type = random.choice(["TRANSFER", "CASH_OUT"])

            dest_mule = random.choice(self.fraud_ring_mules)
            dest_acc_number = dest_mule["account_number"]
            old_dest = account_balances[dest_mule["account_id"]]
            new_dest = round(old_dest + amount, 2)
            account_balances[dest_mule["account_id"]] = new_dest

            transactions_list.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": acc_id,
                "merchant_id": None,
                "step": step,
                "type": txn_type,
                "amount": amount,
                "nameOrig": acc["account_number"],
                "oldbalanceOrg": bal_orig,
                "newbalanceOrig": new_bal_orig,
                "nameDest": dest_acc_number,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": 1,
                "isFlaggedFraud": 1 if amount > 200000 else 0,
                "channel": "ONLINE",
                "status": "FLAGGED",
                "ip_address": self.fake.ipv4(),
                "device_id": self.fake.md5(),
                "location_lat": float(self.fake.latitude()),
                "location_lon": float(self.fake.longitude()),
                "timestamp": txn_time,
                "fraud_pattern": "SPENDING_SPIKE_5X",
            })

        # ---------------------------------------------------------------------
        # 4. INJECT PATTERN 3: RAPID VELOCITY BURSTS (Card Testing / Rapid Fire)
        # ---------------------------------------------------------------------
        print(f"[*] Injecting {num_burst_series} rapid velocity burst series (6 txns each)...")
        for _ in range(num_burst_series):
            acc = random.choice(self.accounts)
            acc_id = acc["account_id"]
            step = random.randint(20, 720)
            burst_start = start_time + timedelta(hours=step, minutes=random.randint(0, 45))
            shared_ip = self.fake.ipv4()
            shared_device = self.fake.md5()
            merchant = random.choice(self.merchants)

            current_time = burst_start
            for seq in range(6):
                gap = random.randint(8, 35)  # rapid consecutive seconds
                current_time += timedelta(seconds=gap)
                amount = round(random.uniform(75.0, 450.0), 2)
                bal = account_balances[acc_id]
                new_bal = round(max(0.0, bal - amount), 2)
                account_balances[acc_id] = new_bal

                transactions_list.append({
                    "transaction_id": str(uuid.uuid4()),
                    "account_id": acc_id,
                    "merchant_id": merchant["merchant_id"],
                    "step": step,
                    "type": "PAYMENT",
                    "amount": amount,
                    "nameOrig": acc["account_number"],
                    "oldbalanceOrg": bal,
                    "newbalanceOrig": new_bal,
                    "nameDest": f"M{merchant['category_code']}_{merchant['merchant_name'][:8].replace(' ', '')}",
                    "oldbalanceDest": 0.00,
                    "newbalanceDest": 0.00,
                    "isFraud": 1,
                    "isFlaggedFraud": 0,
                    "channel": "ONLINE",
                    "status": "DECLINED" if seq >= 4 else "FLAGGED",
                    "ip_address": shared_ip,
                    "device_id": shared_device,
                    "location_lat": float(self.fake.latitude()),
                    "location_lon": float(self.fake.longitude()),
                    "timestamp": current_time,
                    "fraud_pattern": "RAPID_VELOCITY_BURST",
                })

        # ---------------------------------------------------------------------
        # 5. INJECT PATTERN 4: SHARED SUSPICIOUS MERCHANT / ACCOUNT FRAUD RINGS
        # ---------------------------------------------------------------------
        print(f"[*] Injecting {num_ring_txns} coordinated fraud ring transactions...")
        ring_shell_merchants = self.fraud_ring_merchants[:3] if self.fraud_ring_merchants else self.merchants[:3]
        ring_mule_accs = self.fraud_ring_mules[:8]
        shared_ring_ips = [self.fake.ipv4() for _ in range(3)]
        shared_ring_devices = [self.fake.md5() for _ in range(4)]

        for _ in range(num_ring_txns):
            # Compromised sender transferring to mule or paying shell merchant
            sender_acc = random.choice(self.accounts)
            sender_id = sender_acc["account_id"]
            is_merchant_target = random.random() < 0.55

            amount = round(random.uniform(1200.0, 9500.0), 2)
            bal = account_balances[sender_id]
            new_bal = round(max(0.0, bal - amount), 2)
            account_balances[sender_id] = new_bal

            step = random.randint(15, 735)
            txn_time = start_time + timedelta(hours=step, minutes=random.randint(0, 59), seconds=random.randint(0, 59))

            if is_merchant_target:
                target_merchant = random.choice(ring_shell_merchants)
                merchant_id = target_merchant["merchant_id"]
                name_dest = f"M{target_merchant['category_code']}_RING_{target_merchant['merchant_name'][:6].replace(' ', '')}"
                old_dest = 0.00
                new_dest = 0.00
                txn_type = "PAYMENT"
            else:
                mule_dest = random.choice(ring_mule_accs)
                merchant_id = None
                name_dest = mule_dest["account_number"]
                old_dest = account_balances[mule_dest["account_id"]]
                new_dest = round(old_dest + amount, 2)
                account_balances[mule_dest["account_id"]] = new_dest
                txn_type = "TRANSFER"

            transactions_list.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": sender_id,
                "merchant_id": merchant_id,
                "step": step,
                "type": txn_type,
                "amount": amount,
                "nameOrig": sender_acc["account_number"],
                "oldbalanceOrg": bal,
                "newbalanceOrig": new_bal,
                "nameDest": name_dest,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": 1,
                "isFlaggedFraud": 1 if amount > 200000 else 0,
                "channel": "API" if random.random() < 0.4 else "ONLINE",
                "status": "FLAGGED",
                "ip_address": random.choice(shared_ring_ips),
                "device_id": random.choice(shared_ring_devices),
                "location_lat": float(self.fake.latitude()),
                "location_lon": float(self.fake.longitude()),
                "timestamp": txn_time,
                "fraud_pattern": "SUSPICIOUS_FRAUD_RING",
            })

        # Convert to DataFrame and sort chronologically
        df_enriched = pd.DataFrame(transactions_list)
        df_enriched = df_enriched.sort_values(by="timestamp").reset_index(drop=True)

        # Standard PaySim benchmark format subset
        paysim_cols = [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ]
        df_paysim = df_enriched[paysim_cols].copy()

        return df_paysim, df_enriched

    def save_datasets(self, df_paysim: pd.DataFrame, df_enriched: pd.DataFrame):
        """Save generated datasets and relational entities to datasets/ directory."""
        print(f"[*] Saving datasets to {DATASETS_DIR}...")

        # 1. PaySim benchmark format
        paysim_path = DATASETS_DIR / "paysim_synthetic_transactions.csv"
        df_paysim.to_csv(paysim_path, index=False)
        print(f"[+] Saved PaySim dataset: {paysim_path} ({len(df_paysim):,} rows)")

        # 2. Enriched platform dataset
        enriched_path = DATASETS_DIR / "fraudshield_enriched_transactions.csv"
        df_enriched.to_csv(enriched_path, index=False)
        print(f"[+] Saved Enriched dataset: {enriched_path} ({len(df_enriched):,} rows)")

        # 3. Master relational entity tables
        pd.DataFrame(self.customers).to_csv(DATASETS_DIR / "customers.csv", index=False)
        pd.DataFrame(self.accounts).drop(columns=["baseline_avg_spend"], errors="ignore").to_csv(DATASETS_DIR / "accounts.csv", index=False)
        pd.DataFrame(self.merchants).to_csv(DATASETS_DIR / "merchants.csv", index=False)
        print(f"[+] Saved master entity CSVs: customers.csv, accounts.csv, merchants.csv")

        # 4. Generate initial fraud alerts matching the injected patterns
        fraud_alerts = []
        fraud_txns = df_enriched[df_enriched["isFraud"] == 1]
        for _, row in fraud_txns.iterrows():
            pat = row["fraud_pattern"]
            rule_map = {
                "EXACT_DUPLICATE_BURST": ("RULE_DUPLICATE_TRANSACTION", "Exact duplicate transaction within 2 minutes", 90.0, "HIGH"),
                "SPENDING_SPIKE_5X": ("RULE_SPENDING_SPIKE", "Transaction amount exceeds 5x rolling baseline", 85.0, "HIGH"),
                "RAPID_VELOCITY_BURST": ("RULE_RAPID_FIRE_VELOCITY", "Excessive transaction velocity in short time window", 95.0, "CRITICAL"),
                "SUSPICIOUS_FRAUD_RING": ("RULE_FRAUD_RING_INTERACTION", "Transaction linked to coordinated mule/shell entity ring", 98.0, "CRITICAL"),
            }
            r_name, r_desc, score, sev = rule_map.get(pat, ("RULE_ANOMALOUS_EVENT", "Suspicious transaction event", 75.0, "MEDIUM"))
            fraud_alerts.append({
                "alert_id": str(uuid.uuid4()),
                "transaction_id": row["transaction_id"],
                "account_id": row["account_id"],
                "rule_name": r_name,
                "rule_description": r_desc,
                "risk_score": score,
                "severity": sev,
                "status": "OPEN",
                "metadata": f'{{"pattern": "{pat}", "amount": {row["amount"]}}}',
                "created_at": row["timestamp"],
            })
        pd.DataFrame(fraud_alerts).to_csv(DATASETS_DIR / "fraud_alerts.csv", index=False)
        print(f"[+] Saved {len(fraud_alerts):,} fraud alerts: fraud_alerts.csv")

    def print_summary(self, df_enriched: pd.DataFrame):
        """Displays data summary and fraud breakdown."""
        total = len(df_enriched)
        fraud = df_enriched["isFraud"].sum()
        normal = total - fraud
        rate = (fraud / total) * 100

        print("\n" + "=" * 70)
        print("          FRAUDSHIELD AI - DATA GENERATION SUMMARY")
        print("=" * 70)
        print(f"Total Transactions Generated: {total:,}")
        print(f"Legitimate Transactions:     {normal:,} ({100-rate:.2f}%)")
        print(f"Fraudulent Transactions:     {fraud:,} ({rate:.2f}%)")
        print("-" * 70)
        print("Injected Fraud Breakdown by Pattern:")
        patterns = df_enriched[df_enriched["isFraud"] == 1]["fraud_pattern"].value_counts()
        for pat, count in patterns.items():
            print(f"  • {pat:<28}: {count:,} ({count/fraud*100:.1f}%)")
        print("-" * 70)
        print("PaySim Transaction Type Breakdown:")
        types = df_enriched["type"].value_counts()
        for t, count in types.items():
            print(f"  • {t:<12}: {count:,} ({count/total*100:.1f}%)")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FraudShield AI Synthetic Data Generator")
    parser.add_argument("--num-transactions", type=int, default=50000, help="Number of transactions to generate")
    parser.add_argument("--num-customers", type=int, default=1500, help="Number of customer profiles")
    parser.add_argument("--num-merchants", type=int, default=120, help="Number of merchant profiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    pipeline = FraudShieldDataPipeline(seed=args.seed)
    pipeline.generate_entities(num_customers=args.num_customers, num_merchants=args.num_merchants)
    df_paysim, df_enriched = pipeline.generate_transactions(target_count=args.num_transactions)
    pipeline.save_datasets(df_paysim, df_enriched)
    pipeline.print_summary(df_enriched)


if __name__ == "__main__":
    main()
