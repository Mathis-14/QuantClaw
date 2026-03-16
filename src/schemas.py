"""
Pydantic schemas for Deribit data validation.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DeribitInstrument(BaseModel):
    """Deribit instrument schema."""
    instrument_name: str
    strike: float
    expiry_date: datetime
    option_type: str
    bid: float = Field(alias="best_bid_price")
    ask: float = Field(alias="best_ask_price")
    underlying_price: float
    implied_volatility: float = Field(default=0.0, alias="mark_iv")
    funding_rate: Optional[float] = None
    timestamp: datetime

    @field_validator("bid", "ask")
    @classmethod
    def validate_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return v

    @field_validator("implied_volatility", mode="before")
    @classmethod
    def validate_iv(cls, v: Optional[float]) -> float:
        if v is not None:
            return v / 100.0  # Convert % to decimal
        return 0.0


class DeribitOption(DeribitInstrument):
    """Deribit option with additional validation."""
    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        if v not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        return v

    @field_validator("expiry_date")
    @classmethod
    def validate_expiry(cls, v: datetime) -> datetime:
        days_to_expiry = (v - datetime.now(timezone.utc)).days
        if days_to_expiry < 7:
            raise ValueError("Expiry must be at least 7 days from now")
        return v