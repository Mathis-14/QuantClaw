"""
Tests for Deribit data pipeline.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from src.pipeline import (
    fetch_instruments,
    fetch_ticker,
    process_instrument,
    fetch_underlying_price,
)
from src.schemas import DeribitOption


@pytest.mark.asyncio
async def test_fetch_instruments():
    """Test fetching instruments."""
    mock_response = {
        "result": [
            {
                "instrument_name": "BTC-26JUN26-50000-C",
                "strike": 50000.0,
                "expiration_timestamp": 1782489600000,
                "option_type": "call",
            }
        ]
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.json = AsyncMock(return_value=mock_response)
        mock_get.return_value.raise_for_status = AsyncMock()
        
        result = await fetch_instruments("BTC")
        assert len(result) == 1
        assert result[0]["instrument_name"] == "BTC-26JUN26-50000-C"


@pytest.mark.asyncio
async def test_fetch_ticker():
    """Test fetching ticker data."""
    mock_response = {
        "result": {
            "bid_price": 100.0,
            "ask_price": 102.0,
            "greeks": {"iv": 0.5},
            "funding_8h": 0.0001,
        }
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.json = AsyncMock(return_value=mock_response)
        mock_get.return_value.raise_for_status = AsyncMock()
        
        result = await fetch_ticker("BTC-26JUN26-50000-C")
        assert result["bid_price"] == 100.0
        assert result["greeks"]["iv"] == 0.5


@pytest.mark.asyncio
async def test_process_instrument():
    """Test processing a single instrument."""
    instrument = {
        "instrument_name": "BTC-26JUN26-50000-C",
        "strike": 50000.0,
        "expiration_timestamp": 1782489600000,
        "option_type": "call",
    }
    ticker = {
        "bid_price": 100.0,
        "ask_price": 102.0,
        "greeks": {"iv": 0.5},
        "funding_8h": 0.0001,
    }
    
    with patch("src.pipeline.fetch_ticker", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = ticker
        
        result = await process_instrument(instrument, "BTC", 50000.0)
        assert isinstance(result, DeribitOption)
        assert result.strike == 50000.0
        assert result.implied_volatility == 0.5


@pytest.mark.asyncio
async def test_fetch_underlying_price():
    """Test fetching underlying price."""
    mock_response = {"result": {"last_price": 50000.0}}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.json = AsyncMock(return_value=mock_response)
        mock_get.return_value.raise_for_status = AsyncMock()
        
        result = await fetch_underlying_price("BTC")
        assert result == 50000.0