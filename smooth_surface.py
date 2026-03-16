"""Build smoothed SPY volatility surface with arbitrage checks."""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import Rbf
from implied_vol import implied_vol
from fetcher import resolve_tickers
from arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_spread_arbitrage,
    plot_butterfly_diagnostics,
    plot_calendar_diagnostics,
)


# Config
TICKER = "SPY"
MIN_DAYS = 7
MAX_DAYS = 180
RISK_FREE_RATE = 0.0
OUTPUT_DIR = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/exports/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vol_surface_smoothed.png")


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
    print(f"SPY spot price: {spot}")
    
    # Filter OTM calls and puts
    otm_calls = df[(df["option_type"] == "call") & (df["strike"] > spot)]
    otm_puts = df[(df["option_type"] == "put") & (df["strike"] < spot)]
    otm = pd.concat([otm_calls, otm_puts])
    
    # Filter positive bid
    otm = otm[otm["bid"] > 0]
    print(f"Filtered {len(otm)} OTM options with positive bid.")
    
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
        if iv is not None and not np.isnan(iv) and iv > 0:
            ivs.append({
                "strike": row["strike"],
                "expiry": row["expiry"],
                "days_to_expiry": row["days_to_expiry"],
                "moneyness": row["strike"] / spot,
                "implied_vol": iv,
            })
    
    iv_df = pd.DataFrame(ivs)
    print(f"Computed {len(iv_df)} implied volatilities.")
    print(f"IV stats: min={iv_df['implied_vol'].min():.4f}, max={iv_df['implied_vol'].max():.4f}, mean={iv_df['implied_vol'].mean():.4f}")
    
    return iv_df


def build_smoothed_grid(iv_df: pd.DataFrame) -> tuple:
    """Build smoothed grid using RBF interpolation."""
    # Unique strikes and expiries
    strikes = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), 50)
    expiries = np.linspace(iv_df["days_to_expiry"].min(), iv_df["days_to_expiry"].max(), 20)
    
    # Create grid
    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
    
    # RBF interpolation
    rbf = Rbf(
        iv_df["moneyness"],
        iv_df["days_to_expiry"],
        iv_df["implied_vol"],
        function="thin_plate",
    )
    vol_grid = rbf(strike_grid, expiry_grid)
    
    # Clip extreme values
    vol_grid = np.clip(vol_grid, 0.1, 1.0)
    
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
        rstride=1,
        cstride=1,
    )
    
    ax.set_title("SPY Smoothed Volatility Surface", fontsize=16, pad=20)
    ax.set_xlabel("Moneyness (K/S)", fontsize=12, labelpad=10)
    ax.set_ylabel("Days to Expiry", fontsize=12, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=12, labelpad=10)
    fig.colorbar(surf, label="IV", pad=0.1)
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main workflow."""
    print("Fetching SPY option chain...")
    chain = fetch_spy_option_chain()
    print(f"Raw chain size: {len(chain)}")
    
    print("Filtering OTM options...")
    filtered = filter_chain(chain)
    filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0
    print(f"Filtered chain size: {len(filtered)}")
    
    print("Computing implied volatilities...")
    iv_df = compute_implied_vols(filtered)
    
    print("Building smoothed surface grid...")
    strike_grid, expiry_grid, vol_grid = build_smoothed_grid(iv_df)
    
    # Convert days to expiry to years for arbitrage checks
    expiries_years = expiry_grid[:, 0] / 365.0
    
    print("Checking for butterfly arbitrage...")
    is_butterfly_free, butterfly_violations = check_butterfly_arbitrage(
        strikes=strike_grid[0, :],
        expiries=expiries_years,
        vol_grid=vol_grid,
    )
    print(f"Butterfly arbitrage-free: {is_butterfly_free}")
    if not is_butterfly_free:
        print("Butterfly violations (top 5):")
        print(butterfly_violations.head())
    
    print("Checking for calendar spread arbitrage...")
    is_calendar_free, calendar_violations = check_calendar_spread_arbitrage(
        strikes=strike_grid[0, :],
        expiries=expiries_years,
        vol_grid=vol_grid,
    )
    print(f"Calendar spread arbitrage-free: {is_calendar_free}")
    if not is_calendar_free:
        print("Calendar violations (top 5):")
        print(calendar_violations.head())
    
    print("Plotting surface...")
    plot_surface(strike_grid, expiry_grid, vol_grid)
    
    print("Plotting arbitrage diagnostics...")
    plot_butterfly_diagnostics(
        strikes=strike_grid[0, :],
        expiries=expiries_years,
        vol_grid=vol_grid,
        output_dir=OUTPUT_DIR,
    )
    plot_calendar_diagnostics(
        strikes=strike_grid[0, :],
        expiries=expiries_years,
        vol_grid=vol_grid,
        output_dir=OUTPUT_DIR,
    )
    
    print(f"Smoothed plot saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()