"""
Synthetic data generator for financial entities and transactions using Faker.
Generates realistic records conforming to sql/schema.sql.
"""

import uuid
import random
from datetime import datetime, timezone
from typing import Dict, Any, List
from faker import Faker

fake = Faker()


class SyntheticDataGenerator:
    """Generates synthetic financial data for testing and streaming ingestion."""

    MERCHANT_CATEGORIES = [
        ("5411", "Grocery Stores / Supermarkets", "LOW"),
        ("5812", "Restaurants and Dining Places", "LOW"),
        ("5732", "Electronic Sales & Equipment", "MEDIUM"),
        ("4829", "Wire Transfer Money Orders", "HIGH"),
        ("7995", "Betting & Casino Gambling", "CRITICAL"),
        ("5944", "Jewelry, Watch, and Silverware Stores", "HIGH"),
        ("5311", "Department Stores", "LOW"),
    ]

    def __init__(self, seed: int = 42):
        Faker.seed(seed)
        random.seed(seed)

    def generate_customer(self) -> Dict[str, Any]:
        """Generate a single customer profile."""
        profile = fake.simple_profile()
        first_name, last_name = profile["name"].split(" ", 1) if " " in profile["name"] else (profile["name"], "Doe")
        return {
            "customer_id": str(uuid.uuid4()),
            "first_name": first_name,
            "last_name": last_name,
            "email": fake.unique.ascii_safe_email(),
            "phone_number": fake.phone_number()[:30],
            "date_of_birth": profile["birthdate"],
            "ssn_hash": fake.sha256(),
            "address_line1": fake.street_address(),
            "address_line2": fake.secondary_address() if random.random() < 0.2 else None,
            "city": fake.city(),
            "state": fake.state_abbr(),
            "postal_code": fake.postcode(),
            "country": "USA",
            "risk_score": round(random.uniform(0.0, 30.0), 2),
            "status": "ACTIVE",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

    def generate_account(self, customer_id: str) -> Dict[str, Any]:
        """Generate a bank account linked to a customer."""
        return {
            "account_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "account_number": fake.unique.bban(),
            "account_type": random.choice(["CHECKING", "SAVINGS", "CREDIT"]),
            "balance": round(random.uniform(500.0, 50000.0), 2),
            "currency": "USD",
            "status": "ACTIVE",
            "daily_limit": round(random.choice([2500.0, 5000.0, 10000.0, 25000.0]), 2),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

    def generate_merchants(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate a set of merchant profiles."""
        merchants = []
        for _ in range(count):
            mcc, cat_name, risk = random.choice(self.MERCHANT_CATEGORIES)
            merchants.append({
                "merchant_id": str(uuid.uuid4()),
                "merchant_name": f"{fake.company()} {fake.company_suffix()}",
                "category_code": mcc,
                "category_name": cat_name,
                "risk_level": risk,
                "country": "USA",
                "city": fake.city(),
                "status": "ACTIVE",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
        return merchants

    def generate_payment_method(self, account_id: str) -> Dict[str, Any]:
        """Generate a tokenized payment card or method."""
        brand = random.choice(["Visa", "Mastercard", "American Express"])
        return {
            "payment_method_id": str(uuid.uuid4()),
            "account_id": account_id,
            "payment_type": random.choice(["DEBIT_CARD", "CREDIT_CARD", "ACH"]),
            "card_last_four": f"{random.randint(1000, 9999)}",
            "card_brand": brand,
            "token_reference": f"tok_{uuid.uuid4().hex[:16]}",
            "expiry_month": random.randint(1, 12),
            "expiry_year": random.randint(2026, 2030),
            "is_active": True,
            "billing_postal_code": fake.postcode(),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

    def generate_transaction(
        self,
        account_id: str,
        merchant_id: str = None,
        payment_method_id: str = None,
        is_suspicious: bool = False,
    ) -> Dict[str, Any]:
        """Generate a transaction record with optional anomaly injection."""
        if is_suspicious:
            amount = round(random.uniform(5000.0, 25000.0), 2)
            channel = "ONLINE"
            status = "PENDING"
        else:
            amount = round(random.uniform(5.0, 450.0), 2)
            channel = random.choice(["ONLINE", "IN_STORE", "ATM", "MOBILE_APP"])
            status = "COMPLETED"

        return {
            "transaction_id": str(uuid.uuid4()),
            "account_id": account_id,
            "merchant_id": merchant_id,
            "payment_method_id": payment_method_id,
            "amount": amount,
            "currency": "USD",
            "transaction_type": random.choice(["PURCHASE", "TRANSFER", "WITHDRAWAL"]),
            "channel": channel,
            "status": status,
            "ip_address": fake.ipv4(),
            "device_id": fake.md5(),
            "location_lat": float(fake.latitude()),
            "location_lon": float(fake.longitude()),
            "timestamp": datetime.now(timezone.utc),
        }
