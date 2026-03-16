"""
Debug Deribit API response for a single instrument.
"""

import asyncio
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.deribit.com/api/v2/public"


async def fetch_ticker(instrument_name: str) -> None:
    """Fetch and log ticker data for a single instrument."""
    url = f"{BASE_URL}/ticker"
    params = {"instrument_name": instrument_name}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        ticker = response.json()["result"]
        
        logger.info(f"Ticker for {instrument_name}: {ticker}")
        
        # Check for required fields
        required_fields = ["bid_price", "ask_price", "greeks", "underlying_price"]
        for field in required_fields:
            if field not in ticker:
                logger.warning(f"Missing field: {field}")
            else:
                logger.info(f"{field}: {ticker[field]}")


if __name__ == "__main__":
    # Test with a liquid BTC option (e.g., ATM call)
    asyncio.run(fetch_ticker("BTC-27MAR26-60000-C"))