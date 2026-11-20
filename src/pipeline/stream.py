"""
Simulated real-time streaming ingestion pipeline for transactions.
"""

import time
from typing import Iterator, Dict, Any
from src.data_generation.generator import SyntheticDataGenerator
from src.utils.logger import setup_logger

logger = setup_logger("stream_pipeline")


class TransactionStreamPipeline:
    """Simulates real-time transaction ingestion stream."""

    def __init__(self, generator: SyntheticDataGenerator = None):
        self.generator = generator or SyntheticDataGenerator()

    def stream_transactions(
        self,
        account_ids: list,
        batch_size: int = 10,
        interval_seconds: float = 1.0,
    ) -> Iterator[list[Dict[str, Any]]]:
        """
        Yields batches of transactions simulating a live event stream (e.g., Kafka / Kinesis).
        """
        logger.info("Starting simulated transaction stream...")
        while True:
            batch = []
            for _ in range(batch_size):
                import random
                account_id = random.choice(account_ids)
                is_suspicious = random.random() < 0.05  # 5% anomaly rate
                txn = self.generator.generate_transaction(
                    account_id=account_id,
                    is_suspicious=is_suspicious,
                )
                batch.append(txn)
            yield batch
            time.sleep(interval_seconds)
