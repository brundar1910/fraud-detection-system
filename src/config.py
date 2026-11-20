"""
Configuration module for Fraud Detection System.
Loads environment variables and provides centralized application settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file if present
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "fraud_detection_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")

    @property
    def url(self) -> str:
        """Returns standard PostgreSQL connection string."""
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class AppSettings:
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    stream_batch_size: int = int(os.getenv("STREAM_BATCH_SIZE", "500"))
    anomaly_detection_threshold: float = float(os.getenv("ANOMALY_DETECTION_THRESHOLD", "75.0"))
    db: DatabaseSettings = DatabaseSettings()


settings = AppSettings()
