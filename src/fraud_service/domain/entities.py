"""Domain entities for fraud scoring."""

import math
import re
from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Raw transaction from CSV."""
    transaction_id: str
    amount_sar: float
    channel: str
    merchant_category: str
    customer_id: str
    timestamp: str
    ip_country: str
    device_type: str
    is_fraud: int | None = None  # Training-only, not part of serving contract


class EnrichedTransaction(BaseModel):
    """Transaction with computed features."""
    transaction_id: str
    amount_log: float
    channel: str
    mcc: str
    hour_of_day: int
    is_night: int


class FraudScore(BaseModel):
    """Fraud score result for a transaction."""
    transaction_id: str
    score: float = Field(ge=0.0, le=1.0)
    decision: str = Field(pattern="^(allow|review|block)$")


def enrich_transaction(raw: Transaction) -> EnrichedTransaction:
    """Extract and compute features from a raw transaction.

    Uses log1p transform (not log10) to match the model's training data.
    Normalises channel and merchant-category strings so feature values remain
    stable across casing and whitespace differences at the serving boundary.
    """
    amount_log = math.log1p(raw.amount_sar)
    channel = raw.channel.strip().lower()
    mcc = re.sub(r"\s+", "_", raw.merchant_category.strip().upper())

    # Parse ISO 8601 timestamp (may have Z suffix or +00:00)
    timestamp_str = raw.timestamp
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(timestamp_str)
    hour_of_day = timestamp.hour
    is_night = 1 if hour_of_day < 6 else 0

    return EnrichedTransaction(
        transaction_id=raw.transaction_id,
        amount_log=amount_log,
        channel=channel,
        mcc=mcc,
        hour_of_day=hour_of_day,
        is_night=is_night,
    )
