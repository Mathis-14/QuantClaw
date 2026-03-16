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
    bid: float
    ask: float
    underlying_price: float
    implied_volatility: Optional[float] = None
    funding_rate: Optional[float] = None
    timestamp: datetime

    @field_validator("bid", "ask")
    @classmethod
    def validate_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return v




class DeribitOption(DeribitInstrument):
    """Deribit option with additional validation."""
    @field_validator("expiry_date")
    @classmethod
    def validate_expiry(cls, v: datetime) -> datetime:
        days_to_expiry = (v - datetime.now(timezone.utc)).days
        if not (7 <= days_to_expiry <= 180):
            raise ValueError("Expiry must be 7-180 days from now")
        return v