"""
Fetch options data from Deribit and save to CSV.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deribit HTTP API
DERIBIT_API = "https://www.deribit.com/api/v2/public"
INSTRUMENTS = ["BTC", "ETH"]
MIN_DAYS_TO_EXPIRY = 7
MAX_DAYS_TO_EXPIRY = 180
MAX_SPREAD_PERCENT = 20.0


async def fetch_instruments(client: httpx.AsyncClient, currency: str) -> List[Dict]:
    """Fetch all options instruments for a given currency."""
    url = f"{DERIBIT_API}/get_instruments"
    params = {"currency": currency, "kind": "option"}
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()["result"]


async def fetch_bid_ask(client: httpx.AsyncClient, instrument_name: str) -> tuple[float, float]:
    """Fetch bid/ask for a single option via public/ticker."""
    url = f"{DERIBIT_API}/ticker"
    params = {"instrument_name": instrument_name}
    
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()["result"]
        
        bid = result.get("best_bid_price", result.get("bid_price", 0.0))
        ask = result.get("best_ask_price", result.get("ask_price", 0.0))
        return bid, ask
    except Exception as e:
        logger.error(f"Error fetching bid/ask for {instrument_name}: {e}")
        return 0.0, 0.0


async def fetch_implied_volatility(client: httpx.AsyncClient, instrument_name: str) -> float:
    """Fetch greeks.iv for a single option via public/ticker (optional)."""
    url = f"{DERIBIT_API}/ticker"
    params = {"instrument_name": instrument_name}
    
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()["result"]
        
        if "greeks" in result and "iv" in result["greeks"]:
            return result["greeks"]["iv"]
        else:
            return np.nan
    except Exception as e:
        logger.error(f"Error fetching greeks.iv for {instrument_name}: {e}")
        return np.nan


async def fetch_volatility_surface(client: httpx.AsyncClient, currency: str) -> pd.DataFrame:
    """Fetch the full volatility surface for a given currency."""
    # Fetch all instruments
    instruments = await fetch_instruments(client, currency)
    logger.info(f"Fetched {len(instruments)} {currency} options from get_instruments")
    
    # Create lookup for expiry_date and underlying_price
    expiry_lookup = {}
    underlying_lookup = {}
    for instr in instruments:
        expiry_lookup[instr["instrument_name"]] = instr["expiration_timestamp"]
        underlying_lookup[instr["instrument_name"]] = instr.get("underlying_price", 0.0)
    
    # Convert to DataFrame
    records = []
    for instrument in instruments:
        try:
            instrument_name = instrument["instrument_name"]
            expiry_date = expiry_lookup.get(instrument_name)
            if not expiry_date:
                logger.warning(f"Missing expiry_date for {instrument_name}")
                continue
            
            # Fetch bid/ask
            bid, ask = await fetch_bid_ask(client, instrument_name)
            if bid <= 0 or ask <= 0:
                logger.warning(f"Invalid bid/ask for {instrument_name}")
                continue
            
            # Compute spread percent
            spread_percent = (ask - bid) / bid * 100.0
            if spread_percent > MAX_SPREAD_PERCENT:
                logger.warning(f"Spread too high for {instrument_name}: {spread_percent:.2f}%")
                continue
            
            # Days to expiry
            days_to_expiry = (pd.Timestamp(expiry_date, unit='ms') - pd.Timestamp.now()).days
            if days_to_expiry < MIN_DAYS_TO_EXPIRY or days_to_expiry > MAX_DAYS_TO_EXPIRY:
                logger.warning(f"Days to expiry out of range for {instrument_name}: {days_to_expiry}")
                continue
            
            # Fetch greeks.iv (optional)
            iv = await fetch_implied_volatility(client, instrument_name)
            
            record = {
                "instrument_name": instrument_name,
                "strike": instrument.get("strike", 0.0),
                "expiry_date": expiry_date,
                "option_type": "call" if "-C-" in instrument_name else "put",
                "bid": bid,
                "ask": ask,
                "underlying_price": underlying_lookup.get(instrument_name, 0.0),
                "implied_volatility": iv,
                "funding_rate": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
        except Exception as e:
            logger.warning(f"Error processing {instrument.get('instrument_name', 'unknown')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning(f"No valid {currency} options fetched.")
    else:
        logger.info(f"Fetched {len(df)} valid {currency} options")
    
    return df


async def save_data(df: pd.DataFrame, currency: str) -> Path:
    """Save cleaned data to CSV with timestamped filename."""
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/raw")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{currency.lower()}_options_{timestamp_str}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {currency} options to {output_path}")
    return output_path


async def run_pipeline() -> None:
    """Run the full Deribit data pipeline for BTC and ETH."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        for currency in INSTRUMENTS:
            logger.info(f"Fetching {currency} options...")
            df = await fetch_volatility_surface(client, currency)
            if not df.empty:
                await save_data(df, currency)


if __name__ == "__main__":
    asyncio.run(run_pipeline())