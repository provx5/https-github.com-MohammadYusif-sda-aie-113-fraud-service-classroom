from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for fraud prediction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., min_length=3, max_length=64)
    amount_sar: float = Field(..., gt=0.0, le=1_000_000.0)
    channel: Literal["atm", "pos", "ecom", "transfer"]
    merchant_category: str = Field(..., min_length=3, max_length=64)
    customer_id: str = Field(..., min_length=3, max_length=64)
    timestamp: datetime | str
    ip_country: str = Field(..., min_length=2, max_length=3)
    device_type: str = Field(..., min_length=2, max_length=32)

    @field_validator("transaction_id", "customer_id", "merchant_category", "ip_country", "device_type")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned

    @field_validator("timestamp")
    @classmethod
    def parse_timestamp(cls, value: datetime | str) -> str:
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("timestamp is required")
        return cleaned


class PredictResponse(BaseModel):
    """Success response for a fraud prediction."""

    transaction_id: str
    score: float = Field(ge=0.0, le=1.0)
    decision: Literal["allow", "review", "block"]
    model_version: str
