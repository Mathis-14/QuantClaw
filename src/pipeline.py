"""
Deribit Options Data Pipeline

Fetches BTC and ETH options from Deribit public API, validates schema,
filters instruments, computes moneyness and forward price, and saves to CSV.

Usage:
    python -m src.pipeline
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DERIBIT_API = "https://www.deribit.com/api/v2/public"
INSTRUMENTS = ["BTC", "ETH"]
MIN_DAYS_TO_EXPIRY = 7
MAX_DAYS_TO_EXPIRY = 180
MAX_SPREAD_PERCENT = 20.0  # 20%
ACT_365_DAY_COUNT = 365.0

# Pandera Schema for Raw Data
# Pandera Schemas
RawOptionSchema = pa.DataFrameSchema({
    "instrument_name": pa.Column(str),
    "strike": pa.Column(float),
    "expiry_date": pa.Column(str),  # ISO 8601
    "option_type": pa.Column(str, checks=pa.Check.isin(["call", "put"])),
    "bid": pa.Column(float, checks=pa.Check.ge(0)),
    "ask": pa.Column(float, checks=pa.Check.ge(0)),
    "underlying_price": pa.Column(float, checks=pa.Check.ge(0)),
    "implied_volatility": pa.Column(float, checks=pa.Check.ge(0)),
    "funding_rate": pa.Column(float),
    "timestamp": pa.Column(str),  # ISO 8601
})

CleanOptionSchema = pa.DataFrameSchema({
    "instrument_name": pa.Column(str),
    "strike": pa.Column(float),
    "expiry_date": pa.Column(pd.Timestamp),
    "option_type": pa.Column(str, checks=pa.Check.isin(["call", "put"])),
    "mid_price": pa.Column(float, checks=pa.Check.ge(0)),
    "underlying_price": pa.Column(float, checks=pa.Check.ge(0)),
    "implied_volatility": pa.Column(float, checks=pa.Check.ge(0)),
    "days_to_expiry": pa.Column(float, checks=pa.Check.ge(0)),
    "moneyness": pa.Column(float, checks=pa.Check.ge(0)),
    "forward_price": pa.Column(float, checks=pa.Check.ge(0)),
    "timestamp": pa.Column(pd.Timestamp),
})


def day_count_act_365(expiry_date: pd.Timestamp, timestamp: pd.Timestamp) -> float:
    """Compute days to expiry using ACT/365 day count convention."""
    delta = expiry_date - timestamp
    return delta.days + delta.seconds / (24 * 3600)


def compute_forward_price(
    underlying_price: float, funding_rate: float, days_to_expiry: float
) -> float:
    """Compute forward price using funding rate and days to expiry."""
    return underlying_price * np.exp(funding_rate * days_to_expiry / ACT_365_DAY_COUNT)


def filter_instruments(df: DataFrame[RawOptionSchema]) -> DataFrame[RawOptionSchema]:
    """Filter instruments based on bid/ask spread, days to expiry, and liquidity."""
    df = df.copy()
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["spread_percent"] = (df["ask"] - df["bid"]) / df["mid_price"] * 100
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["days_to_expiry"] = df.apply(
        lambda row: day_count_act_365(row["expiry_date"], row["timestamp"]), axis=1
    )
    
    # Apply filters
    mask = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["spread_percent"] <= MAX_SPREAD_PERCENT)
        & (df["days_to_expiry"] >= MIN_DAYS_TO_EXPIRY)
        & (df["days_to_expiry"] <= MAX_DAYS_TO_EXPIRY)
    )
    return df[mask]


def clean_and_enrich(df: DataFrame[RawOptionSchema]) -> DataFrame[CleanOptionSchema]:
    """Clean and enrich raw data with moneyness, forward price, and day count."""
    df = df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["days_to_expiry"] = df.apply(
        lambda row: day_count_act_365(row["expiry_date"], row["timestamp"]), axis=1
    )
    df["moneyness"] = df["strike"] / df["underlying_price"]
    df["forward_price"] = df.apply(
        lambda row: compute_forward_price(
            row["underlying_price"], row["funding_rate"], row["days_to_expiry"]
        ),
        axis=1,
    )
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    return df


async def fetch_instruments(client: httpx.AsyncClient, currency: str) -> List[Dict]:
    """Fetch all options instruments for a given currency."""
    url = f"{DERIBIT_API}/get_instruments"
    params = {"currency": currency, "kind": "option"}
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()["result"]


async def fetch_option_book(
    client: httpx.AsyncClient, instrument_name: str, max_retries: int = 3
) -> Optional[Dict]:
    """Fetch order book and implied volatility for a single option with retries."""
    url = f"{DERIBIT_API}/get_order_book"
    params = {"instrument_name": instrument_name}
    
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()["result"]
            
            # Skip if required keys are missing
            if not all(key in result for key in ["strike", "bid_price", "ask_price", "underlying_price"]):
                logger.warning(f"Missing keys in {instrument_name}, skipping")
                return None
            
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = (attempt + 1) * 0.1  # 100ms, 200ms, 300ms
                logger.warning(f"Rate limited for {instrument_name}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"HTTP error for {instrument_name}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {instrument_name}: {e}")
            return None
    
    return None


async def fetch_all_options(
    client: httpx.AsyncClient, currency: str
) -> DataFrame[RawOptionSchema]:
    """Fetch all options data for a given currency with rate-limiting and retries."""
    instruments = await fetch_instruments(client, currency)
    tasks = []
    for instr in instruments:
        task = fetch_option_book(client, instr["instrument_name"])
        tasks.append(task)
        await asyncio.sleep(0.1)  # 100ms delay between requests
    
    books = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failed fetches and None results
    books = [book for book in books if not isinstance(book, Exception) and book is not None]
    
    # Convert to DataFrame
    records = []
    for book in books:
        try:
            record = {
                "instrument_name": book["instrument_name"],
                "strike": book["strike"],
                "expiry_date": book["expiration_timestamp"],
                "option_type": "call" if "-C-" in book["instrument_name"] else "put",
                "bid": book["bid_price"],
                "ask": book["ask_price"],
                "underlying_price": book["underlying_price"],
                "implied_volatility": book.get("greeks", {}).get("iv", np.nan),
                "funding_rate": book.get("funding_8h", 0.0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
        except KeyError as e:
            logger.warning(f"Missing key in {book.get('instrument_name', 'unknown')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning(f"No valid {currency} options fetched. Skipping.")
        return pd.DataFrame(columns=RawOptionSchema.columns.keys())
    return RawOptionSchema.validate(df, lazy=True)


def generate_quality_report(df: DataFrame[CleanOptionSchema], currency: str) -> Dict:
    """Generate a data quality report for the fetched options."""
    report = {
        "currency": currency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_instruments": len(df),
        "strikes_per_expiry": (
            df.groupby(["expiry_date", "option_type"])["strike"]
            .nunique()
            .reset_index()
            .groupby("expiry_date")
            .agg({"strike": "mean"})
            .to_dict()["strike"]
        ),
        "expiries": df["expiry_date"].dt.strftime("%Y-%m-%d").unique().tolist(),
        "moneyness_range": {
            "min": df["moneyness"].min(),
            "max": df["moneyness"].max(),
        },
        "implied_volatility_range": {
            "min": df["implied_volatility"].min(),
            "max": df["implied_volatility"].max(),
        },
    }
    return report


def save_data(df: DataFrame[CleanOptionSchema], currency: str) -> Path:
    """Save cleaned data to CSV."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_dir = Path("data/raw")
    output_dir.mkdir(exist_ok=True) 
    
    output_path = output_dir / f"{currency.lower()}_options_{date_str}.csv"
    df.to_csv(output_path, index=False)
    return output_path


async def run_pipeline() -> None:
    """Run the full Deribit data pipeline for BTC and ETH."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        for currency in INSTRUMENTS:
            logger.info(f"Fetching {currency} options...")
            raw_df = await fetch_all_options(client, currency)
            if raw_df.empty:
                logger.warning(f"No valid {currency} options fetched. Skipping.")
                continue
            filtered_df = filter_instruments(raw_df)
            if filtered_df.empty:
                logger.warning(f"No {currency} options passed filters. Skipping.")
                continue
            clean_df = clean_and_enrich(filtered_df)
            
            # Save data
            output_path = save_data(clean_df, currency)
            logger.info(f"Saved {currency} options to {output_path}")
            
            # Generate and save quality report
            report = generate_quality_report(clean_df, currency)
            report_path = Path("data") / f"{currency.lower()}_data_quality_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved {currency} quality report to {report_path}")
            
            # Send report to Telegram
            await send_report_to_telegram(report, report_path)


async def send_report_to_telegram(report: Dict, report_path: Path) -> None:
    """Send data quality report to Telegram."""
    caption = (
        f"[Deribit Pipeline] {report['currency']} Data Quality Report\n"
        f"Total Instruments: {report['total_instruments']}\n"
        f"Expiries: {len(report['expiries'])}\n"
        f"Moneyness Range: {report['moneyness_range']['min']:.2f} - {report['moneyness_range']['max']:.2f}"
    )
    
    # Send report file
    message(action="send", channel="telegram", media=str(report_path), caption=caption)


if __name__ == "__main__":
    asyncio.run(run_pipeline())