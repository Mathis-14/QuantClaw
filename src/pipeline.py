"""
Deribit data pipeline with rate limiting and validation.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from pydantic import ValidationError

from .schemas import DeribitOption

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.deribit.com/api/v2/public"
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_SPREAD_PERCENT = 20.0
MIN_DAYS_TO_EXPIRY = 7
MAX_DAYS_TO_EXPIRY = 180


async def fetch_instruments(currency: str) -> list[dict]:
    """Fetch all instruments for a currency."""
    url = f"{BASE_URL}/get_instruments"
    params = {"currency": currency, "kind": "option"}
    
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                await response.aread()
                response.raise_for_status()
                return (await response.json())["result"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to fetch instruments: {e}")
                return []
    return []


async def fetch_ticker(instrument_name: str) -> Optional[dict]:
    """Fetch ticker data for a single instrument."""
    url = f"{BASE_URL}/ticker"
    params = {"instrument_name": instrument_name}
    
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                await response.aread()
                response.raise_for_status()
                return (await response.json())["result"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to fetch ticker for {instrument_name}: {e}")
                return None
    return None


async def process_instrument(
    instrument: dict, currency: str, underlying_price: float
) -> Optional[DeribitOption]:
    """Process a single instrument and fetch its ticker data."""
    ticker = await fetch_ticker(instrument["instrument_name"])
    if not ticker:
        logger.warning(f"No ticker data for {instrument['instrument_name']}")
        return None
    
    if "bid_price" not in ticker or "ask_price" not in ticker:
        logger.warning(f"Missing bid/ask for {instrument['instrument_name']}")
        return None
    
    try:
        option = DeribitOption(
            instrument_name=instrument["instrument_name"],
            strike=instrument["strike"],
            expiry_date=datetime.fromtimestamp(instrument["expiration_timestamp"] / 1000, tz=timezone.utc),
            option_type=instrument["option_type"],
            bid=ticker["bid_price"],
            ask=ticker["ask_price"],
            underlying_price=underlying_price,
            implied_volatility=ticker.get("greeks", {}).get("iv"),
            funding_rate=ticker.get("funding_8h"),
            timestamp=datetime.now(timezone.utc),
        )
        return option
    except ValidationError as e:
        logger.warning(f"Validation failed for {instrument['instrument_name']}: {e}")
        return None


async def fetch_underlying_price(currency: str) -> float:
    """Fetch the current underlying price."""
    url = f"{BASE_URL}/ticker"
    params = {"instrument_name": f"{currency.upper()}-PERPETUAL"}
    
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                await response.aread()
                response.raise_for_status()
                return (await response.json())["result"]["last_price"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to fetch underlying price for {currency}: {e}")
                return 0.0
    return 0.0


async def fetch_and_save_data(currency: str) -> None:
    """Fetch and save Deribit options data."""
    instruments = await fetch_instruments(currency)
    if not instruments:
        logger.error(f"No instruments fetched for {currency}")
        return
    
    underlying_price = await fetch_underlying_price(currency)
    if underlying_price <= 0:
        logger.error(f"Invalid underlying price for {currency}")
        return
    
    options = []
    for instrument in instruments:
        option = await process_instrument(instrument, currency, underlying_price)
        if option:
            options.append(option)
    
    if not options:
        logger.error(f"No valid options for {currency}")
        return
    
    # Save to CSV
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/raw")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{currency.lower()}_options_{timestamp}.csv"
    
    df = [option.model_dump() for option in options]
    import pandas as pd
    pd.DataFrame(df).to_csv(output_path, index=False)
    logger.info(f"Saved {len(options)} {currency} options to {output_path}")


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(fetch_and_save_data("BTC"))
    asyncio.run(fetch_and_save_data("ETH"))