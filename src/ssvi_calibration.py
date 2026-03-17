"""
SSVI calibration pipeline for BTC and ETH options.

This script:
1. Loads BTC and ETH options data.
2. Cleans and prepares the data for calibration.
3. Calibrates SSVI for both assets.
4. Generates and saves volatility smile plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vol_surface.data.schema import VolSlice
from vol_surface.models.ssvi import (
    ssvi_initial_guess,
    ssvi_parameter_bounds,
    ssvi_total_variance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_options_data(file_path: Path) -> pd.DataFrame:
    """Load options data from CSV."""
    df = pd.read_csv(file_path)
    df = df[df["implied_volatility"].notna()]
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], unit="ms")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare_vol_slices(df: pd.DataFrame, underlying: str) -> List[VolSlice]:
    """Prepare VolSlice objects for each expiry."""
    slices = []
    spot = {
        "BTC": 52000.0,
        "ETH": 3000.0,
    }.get(underlying, 1.0)
    if spot <= 0:
        raise ValueError(f"Invalid spot price for {underlying}")

    for expiry, group in df.groupby("expiry_date"):
        T = (expiry - pd.Timestamp.now(tz=expiry.tz)).days / 365.0
        if T <= 0:
            continue

        group = group[group["implied_volatility"] > 0]
        if len(group) == 0:
            continue

        strikes = group["strike"].values
        log_moneyness = np.log(strikes / spot)
        total_variance = (group["implied_volatility"] ** 2) * T
        implied_vols = group["implied_volatility"].values

        weights = [1.0] * len(strikes)  # Uniform weights for simplicity
        slices.append(
            VolSlice(
                expiry=expiry.date(),
                T=T,
                forward=spot,
                strikes=strikes.tolist(),
                log_moneyness=log_moneyness.tolist(),
                total_variance=total_variance.tolist(),
                implied_vols=implied_vols.tolist(),
                weights=weights,
            )
        )
    return slices


def calibrate_ssvi_slice(
    vol_slice: VolSlice,
) -> Tuple[dict, float]:
    """Calibrate SSVI for a single maturity slice."""
    k = np.array(vol_slice.log_moneyness)
    w = np.array(vol_slice.total_variance)
    theta = np.mean(w)
    weights = np.ones_like(k)  # Uniform weights for simplicity

    def objective(params: np.ndarray) -> float:
        rho, eta, gamma = params
        w_pred = ssvi_total_variance(k, theta, rho, eta, gamma)
        return np.sqrt(np.mean(weights * (w - w_pred) ** 2))

    p0 = ssvi_initial_guess()
    lower, upper = ssvi_parameter_bounds()
    bounds = list(zip(lower, upper))
    result = minimize(
        objective, p0, bounds=bounds, method="L-BFGS-B"
    )
    if not result.success:
        logger.warning(f"SSVI calibration failed for T={vol_slice.T}: {result.message}")

    params = {
        "rho": result.x[0],
        "eta": result.x[1],
        "gamma": result.x[2],
        "theta": theta,
    }
    return params, result.fun


def calibrate_ssvi_surface(
    vol_slices: List[VolSlice],
) -> Tuple[dict, float]:
    """Calibrate SSVI for all slices (joint calibration)."""
    thetas = [np.mean(slice.total_variance) for slice in vol_slices]
    k_all = np.concatenate([np.array(slice.log_moneyness) for slice in vol_slices])
    w_all = np.concatenate([np.array(slice.total_variance) for slice in vol_slices])
    theta_all = np.concatenate([
        np.full_like(slice.log_moneyness, theta)
        for slice, theta in zip(vol_slices, thetas)
    ])
    weights = np.ones_like(k_all)

    def objective(params: np.ndarray) -> float:
        rho, eta, gamma = params
        w_pred = ssvi_total_variance(k_all, theta_all, rho, eta, gamma)
        return np.sqrt(np.mean(weights * (w_all - w_pred) ** 2))

    p0 = ssvi_initial_guess()
    lower, upper = ssvi_parameter_bounds()
    bounds = list(zip(lower, upper))
    result = minimize(
        objective, p0, bounds=bounds, method="L-BFGS-B"
    )
    if not result.success:
        logger.warning(f"SSVI surface calibration failed: {result.message}")

    params = {
        "rho": result.x[0],
        "eta": result.x[1],
        "gamma": result.x[2],
    }
    return params, result.fun


def plot_ssvi_smiles(
    vol_slices: List[VolSlice],
    params: dict,
    asset: str,
    output_dir: Path,
) -> None:
    """Plot SSVI smiles for all expiries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for slice in vol_slices:
        k = np.array(slice.log_moneyness)
        w_market = np.array(slice.total_variance)
        theta = np.mean(w_market)
        w_pred = ssvi_total_variance(k, theta, params["rho"], params["eta"], params["gamma"])
        iv_market = np.sqrt(w_market / slice.T)
        iv_pred = np.sqrt(w_pred / slice.T)

        plt.plot(
            k, iv_market, "o", label=f"Market T={slice.T:.2f}", alpha=0.5
        )
        plt.plot(
            k, iv_pred, "-", label=f"SSVI T={slice.T:.2f}", alpha=0.8
        )

    plt.title(f"SSVI Volatility Smiles for {asset}")
    plt.xlabel("Log-Moneyness (log(K/F))")
    plt.ylabel("Implied Volatility")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"ssvi_smiles_{asset.lower()}.png")
    plt.close()


def main() -> None:
    """Run SSVI calibration and plotting for BTC and ETH using demo data."""
    data_dir = Path("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/raw")
    output_dir = Path("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots")

    # Load and prepare data
    demo_file = data_dir / "demo_options.csv"
    demo_df = load_options_data(demo_file)

    # Filter for BTC and ETH (demo data contains BTC)
    btc_df = demo_df[demo_df["instrument_name"].str.contains("BTC")]
    eth_df = pd.DataFrame()  # No ETH in demo data, skip for now

    btc_slices = prepare_vol_slices(btc_df, "BTC")
    eth_slices = []

    if not btc_slices:
        raise ValueError("No valid slices found for BTC")

    # Calibrate SSVI
    btc_params, btc_rmse = calibrate_ssvi_surface(btc_slices)
    logger.info(f"BTC SSVI Params: {btc_params}, RMSE: {btc_rmse:.6f}")

    # Plot smiles
    plot_ssvi_smiles(btc_slices, btc_params, "BTC", output_dir)
    if eth_slices:
        eth_params, eth_rmse = calibrate_ssvi_surface(eth_slices)
        logger.info(f"ETH SSVI Params: {eth_params}, RMSE: {eth_rmse:.6f}")
        plot_ssvi_smiles(eth_slices, eth_params, "ETH", output_dir)

    logger.info(f"SSVI plots saved to {output_dir}")


if __name__ == "__main__":
    main()