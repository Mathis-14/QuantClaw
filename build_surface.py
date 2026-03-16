"""Build and plot SPY implied volatility surface."""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
from implied_vol import implied_vol
from fetcher import resolve_tickers


# Config
TICKER = "SPY"
MIN_DAYS = 7
MAX_DAYS = 180
RISK_FREE_RATE = 0.0
OUTPUT_DIR = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/exports/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vol_surface.png")


# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_spy_option_chain() -> pd.DataFrame:
    """Fetch SPY option chain from yfinance."""
    _, options_ticker = resolve_tickers(TICKER)
    ticker = yf.Ticker(options_ticker)
    expirations = ticker.options
    
    chains = []
    for exp in expirations:
        expiry_date = pd.Timestamp(exp).date()
        days_to_expiry = (expiry_date - datetime.now().date()).days
        if days_to_expiry < MIN_DAYS or days_to_expiry > MAX_DAYS:
            continue
            
        try:
            chain = ticker.option_chain(exp)
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                df = df.copy()
                df["expiry"] = exp
                df["days_to_expiry"] = days_to_expiry
                df["option_type"] = side
                chains.append(df)
        except Exception as e:
            print(f"Failed to fetch {exp}: {e}")
    
    if not chains:
        raise ValueError("No option chains fetched.")
    
    return pd.concat(chains, ignore_index=True)


def filter_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Filter OTM options with positive bid."""
    spot = yf.Ticker(TICKER).fast_info.last_price
    
    # Filter OTM calls and puts
    otm_calls = df[(df["option_type"] == "call") & (df["strike"] > spot)]
    otm_puts = df[(df["option_type"] == "put") & (df["strike"] < spot)]
    otm = pd.concat([otm_calls, otm_puts])
    
    # Filter positive bid
    otm = otm[otm["bid"] > 0]
    
    return otm


def compute_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied volatilities for filtered chain."""
    spot = yf.Ticker(TICKER).fast_info.last_price
    
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(
            price=row["mid"],
            S=spot,
            K=row["strike"],
            T=row["days_to_expiry"] / 365.0,
            r=RISK_FREE_RATE,
            option_type=row["option_type"],
        )
        if iv is not None:
            ivs.append({
                "strike": row["strike"],
                "expiry": row["expiry"],
                "days_to_expiry": row["days_to_expiry"],
                "moneyness": row["strike"] / spot,
                "implied_vol": iv,
            })
    
    return pd.DataFrame(ivs)


def build_surface_grid(iv_df: pd.DataFrame) -> tuple:
    """Build grid for 3D surface plot."""
    # Unique strikes and expiries
    strikes = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), 50)
    expiries = np.linspace(iv_df["days_to_expiry"].min(), iv_df["days_to_expiry"].max(), 20)
    
    # Create grid
    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
    vol_grid = np.full_like(strike_grid, np.nan)
    
    # Fill grid (nearest neighbor)
    for i, expiry in enumerate(expiries):
        for j, strike in enumerate(strikes):
            subset = iv_df[
                (np.abs(iv_df["days_to_expiry"] - expiry) < 5) &
                (np.abs(iv_df["moneyness"] - strike) < 0.02)
            ]
            if not subset.empty:
                vol_grid[i, j] = subset["implied_vol"].mean()
    
    return strike_grid, expiry_grid, vol_grid


def plot_surface(strike_grid: np.ndarray, expiry_grid: np.ndarray, vol_grid: np.ndarray) -> None:
    """Plot 3D implied volatility surface."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(
        strike_grid,
        expiry_grid,
        vol_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.8,
    )
    
    ax.set_title("SPY Implied Volatility Surface")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    fig.colorbar(surf, label="IV")
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main workflow."""
    print("Fetching SPY option chain...")
    chain = fetch_spy_option_chain()
    
    print("Filtering OTM options...")
    filtered = filter_chain(chain)
    filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0
    
    print("Computing implied volatilities...")
    iv_df = compute_implied_vols(filtered)
    
    print("Building surface grid...")
    strike_grid, expiry_grid, vol_grid = build_surface_grid(iv_df)
    
    print("Plotting surface...")
    plot_surface(strike_grid, expiry_grid, vol_grid)
    
    print(f"Plot saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()