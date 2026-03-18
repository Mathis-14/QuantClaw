"""Minimal Deribit Data Importer for BTC options (production-ready)."""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any


class DeribitDataImporter:
    """Fetch BTC options data from Deribit API (production environment)."""

    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    def __init__(self):
        self.base_url = "https://www.deribit.com"
        self.raw_data = []
        self.df = None

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch BTC options instruments from Deribit API."""
        url = f"{self.base_url}/api/v2/public/get_book_summary_by_currency"
        params = {"currency": "BTC", "kind": "option"}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        self.raw_data = [inst for inst in response.json()["result"] if inst["instrument_name"].endswith("-C")]
        print(f"✓ Fetched {len(self.raw_data)} BTC options from Deribit.")
        return self.raw_data

    def build_dataframe(self) -> pd.DataFrame:
        """Transform raw API data into a clean DataFrame with implied volatilities."""
        if not self.raw_data:
            raise ValueError("No raw data. Call fetch_data() first.")

        rows = []
        for instrument in self.raw_data:
            name = instrument.get("instrument_name")
            parsed = self._parse_instrument_name(name)
            if not parsed:
                continue

            # Use mark_iv from Deribit directly
            iv = instrument.get("mark_iv")
            if iv is None or iv <= 0:
                print(f"Warning: Missing or invalid IV for {name}")
                continue

            rows.append({
                "instrument_name": name,
                "strike": parsed["strike"],
                "expiry_date": parsed["expiry_date"],
                "implied_volatility": iv,
                "underlying_price": instrument["underlying_price"],
            })

        self.df = pd.DataFrame(rows)
        self.df["expiry_date"] = pd.to_datetime(self.df["expiry_date"])
        self.df = self.df.sort_values(["expiry_date", "strike"]).reset_index(drop=True)
        print(f"✓ DataFrame built with {len(self.df)} instruments.")
        return self.df

    def save_to_csv(self, filepath: str = "data/processed/btc_options_deribit.csv") -> str:
        """Save DataFrame to CSV."""
        if self.df is None:
            raise ValueError("No data. Call build_dataframe() first.")
        self.df.to_csv(filepath, index=False)
        print(f"✓ Data saved to {filepath}")
        return filepath

    @staticmethod
    def _parse_expiry_date(expiry_str: str) -> Optional[datetime]:
        """Parse Deribit expiry format (e.g., '29MAR26' → 2026-03-29)."""
        try:
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5].upper()
            year = 2000 + int(expiry_str[5:7])
            month = DeribitDataImporter.MONTH_MAP.get(month_str)
            if month is None:
                return None
            return datetime(year, month, day)
        except (ValueError, KeyError, IndexError):
            return None

    @staticmethod
    def _parse_instrument_name(name: str) -> Optional[Dict[str, Any]]:
        """Parse instrument name (e.g., 'BTC-29MAR26-90000-C')."""
        parts = name.split("-")
        if len(parts) != 4:
            return None
        try:
            strike = float(parts[2])
            expiry_date = DeribitDataImporter._parse_expiry_date(parts[1])
            return {
                "strike": strike,
                "expiry_date": expiry_date,
                "type": parts[3]
            }
        except ValueError:
            return None


if __name__ == "__main__":
    importer = DeribitDataImporter()
    importer.fetch_data()
    importer.build_dataframe()
    importer.save_to_csv()