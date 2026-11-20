"""
Logging configuration for the fraud detection system.
"""

import logging
import sys
from src.config import settings


def setup_logger(name: str = "fraud_detection") -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(settings.log_level.upper())
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
