"""
SVI and SSVI Calibration for Volatility Surfaces

Implements raw SVI per expiry slice and joint SSVI calibration.

Usage:
    from src.svi import calibrate_svi, calibrate_ssvi
    svi_params = calibrate_svi(clean_df)
    ssvi_params = calibrate_ssvi(clean_df)
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

# Suppress matplotlib warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Import message for Telegram
from src.arbitrage import message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SVIParams:
    """SVI parameters for a single expiry slice."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry_date: pd.Timestamp
    rmse: Optional[float] = None
    r_squared: Optional[float] = None


@dataclass
class SSVIParams:
    """SSVI parameters for the entire surface."""
    rho: float
    eta: float
    gamma: float
    rmse: Optional[float] = None
    r_squared: Optional[float] = None


def raw_svi(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    Raw SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    Args:
        k: Log-moneyness (log(K/F)).
        a, b, rho, m, sigma: SVI parameters.
    
    Returns:
        Total variance w(k).
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def ssvi(theta: np.ndarray, k: np.ndarray, rho: float, eta: float, gamma: float) -> np.ndarray:
    """
    SSVI formula: w(k, theta) = theta / 2 * (1 + rho * eta * k + sqrt((eta * k + rho)^2 + (1 - rho^2)))
    
    Args:
        theta: Total variance at ATM (w(0)).
        k: Log-moneyness (log(K/F)).
        rho, eta, gamma: SSVI parameters.
    
    Returns:
        Total variance w(k, theta).
    """
    return theta / 2 * (1 + rho * eta * k + np.sqrt((eta * k + rho) ** 2 + (1 - rho ** 2)))


def svi_objective(params: np.ndarray, k: np.ndarray, w: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted least squares objective for SVI calibration.
    
    Args:
        params: [a, b, rho, m, sigma].
        k: Log-moneyness.
        w: Total variance.
        weights: 1 / bid_ask_spread_pct.
    
    Returns:
        Weighted RMSE.
    """
    a, b, rho, m, sigma = params
    w_pred = raw_svi(k, a, b, rho, m, sigma)
    return np.sqrt(np.mean(weights * (w - w_pred) ** 2))


def ssvi_objective(params: np.ndarray, thetas: np.ndarray, k: np.ndarray, w: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted least squares objective for SSVI calibration.
    
    Args:
        params: [rho, eta, gamma].
        thetas: ATM total variance per expiry.
        k: Log-moneyness.
        w: Total variance.
        weights: 1 / bid_ask_spread_pct.
    
    Returns:
        Weighted RMSE.
    """
    rho, eta, gamma = params
    # Ensure eta * (1 + |rho|) <= 2 (no-arbitrage condition)
    if eta * (1 + abs(rho)) > 2:
        return np.inf
    
    # Compute SSVI for all expiries
    w_pred = np.array([
        ssvi(theta, k_i, rho, eta, gamma) for theta, k_i in zip(thetas, k)
    ])
    return np.sqrt(np.mean(weights * (w - w_pred) ** 2))


def calibrate_svi_slice(
    group: pd.DataFrame, 
    k_col: str = "log_moneyness",
    w_col: str = "total_variance",
    weight_col: str = "weight"
) -> SVIParams:
    """
    Calibrate raw SVI for a single expiry slice.
    
    Args:
        group: DataFrame with columns [k_col, w_col, weight_col].
        k_col: Column name for log-moneyness.
        w_col: Column name for total variance.
        weight_col: Column name for weights (1 / bid_ask_spread_pct).
    
    Returns:
        SVIParams for the slice.
    """
    k = group[k_col].values
    w = group[w_col].values
    weights = group[weight_col].values
    
    # Initial guess: [a, b, rho, m, sigma]
    p0 = [0.01, 0.1, -0.5, 0.0, 0.1]
    
    # Bounds: a >= 0, b >= 0, -1 <= rho <= 1, sigma > 0
    bounds = [
        (-np.inf, np.inf),  # a
        (0, np.inf),        # b
        (-1, 1),            # rho
        (-np.inf, np.inf),  # m
        (1e-6, np.inf),     # sigma
    ]
    
    # Calibrate
    result = minimize(
        svi_objective, 
        p0, 
        args=(k, w, weights), 
        bounds=bounds, 
        method="L-BFGS-B"
    )
    
    a, b, rho, m, sigma = result.x
    
    # Compute RMSE and R²
    w_pred = raw_svi(k, a, b, rho, m, sigma)
    rmse = np.sqrt(np.mean((w - w_pred) ** 2))
    ss_total = np.sum((w - np.mean(w)) ** 2)
    ss_res = np.sum((w - w_pred) ** 2)
    r_squared = 1 - (ss_res / ss_total) if ss_total > 0 else 1.0
    
    return SVIParams(
        a=a, b=b, rho=rho, m=m, sigma=sigma, 
        expiry_date=group["expiry_date"].iloc[0], 
        rmse=rmse, r_squared=r_squared
    )


def calibrate_svi(df: pd.DataFrame) -> List[SVIParams]:
    """
    Calibrate raw SVI for all expiry slices.
    
    Args:
        df: DataFrame with columns [expiry_date, log_moneyness, total_variance, weight].
    
    Returns:
        List of SVIParams per expiry.
    """
    # Ensure required columns exist
    required_columns = {"expiry_date", "log_moneyness", "total_variance", "weight"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Group by expiry_date and calibrate SVI per slice
    svi_params = []
    for expiry, group in df.groupby("expiry_date"):
        params = calibrate_svi_slice(group)
        svi_params.append(params)
        
        # Plot and send to Telegram
        plot_path = plot_svi_fit(group, params)
        caption = (
            f"SVI Fit for {expiry.strftime('%Y-%m-%d')}: RMSE={params.rmse:.6f}, R²={params.r_squared:.6f}\n"
            f"Params: a={params.a:.6f}, b={params.b:.6f}, rho={params.rho:.6f}, m={params.m:.6f}, sigma={params.sigma:.6f}"
        )
        message(
            action="send", 
            channel="telegram", 
            media=str(plot_path), 
            caption=caption
        )
    
    # Save parameters
    save_svi_params(svi_params)
    
    return svi_params


def calibrate_ssvi(df: pd.DataFrame) -> SSVIParams:
    """
    Calibrate joint SSVI for the entire surface.
    
    Args:
        df: DataFrame with columns [expiry_date, log_moneyness, total_variance, weight].
    
    Returns:
        SSVIParams for the surface.
    """
    # Ensure required columns exist
    required_columns = {"expiry_date", "log_moneyness", "total_variance", "weight"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract ATM total variance (theta) per expiry
    atm_df = df[df["log_moneyness"].abs() < 1e-4]  # Close to ATM
    thetas = atm_df.groupby("expiry_date")["total_variance"].mean().values
    
    # Flatten data for SSVI calibration
    k = df["log_moneyness"].values
    w = df["total_variance"].values
    weights = df["weight"].values
    
    # Initial guess: [rho, eta, gamma]
    p0 = [-0.5, 0.1, 0.5]
    
    # Bounds: -1 <= rho <= 1, eta >= 0, 0 <= gamma <= 1
    bounds = [
        (-1, 1),        # rho
        (0, np.inf),     # eta
        (0, 1),         # gamma
    ]
    
    # Calibrate
    result = minimize(
        ssvi_objective, 
        p0, 
        args=(thetas, k, w, weights), 
        bounds=bounds, 
        method="L-BFGS-B"
    )
    
    rho, eta, gamma = result.x
    
    # Compute RMSE and R²
    w_pred = np.array([
        ssvi(theta, k_i, rho, eta, gamma) for theta, k_i in zip(np.repeat(thetas, len(k) // len(thetas)), k)
    ])
    rmse = np.sqrt(np.mean((w - w_pred) ** 2))
    ss_total = np.sum((w - np.mean(w)) ** 2)
    ss_res = np.sum((w - w_pred) ** 2)
    r_squared = 1 - (ss_res / ss_total) if ss_total > 0 else 1.0
    
    params = SSVIParams(rho=rho, eta=eta, gamma=gamma, rmse=rmse, r_squared=r_squared)
    
    # Save parameters
    save_ssvi_params(params)
    
    return params


def plot_svi_fit(group: pd.DataFrame, params: SVIParams, output_dir: Path = Path("plots")) -> Path:
    """
    Plot market mid IV vs SVI fit for a single expiry slice.
    
    Args:
        group: DataFrame with columns [log_moneyness, total_variance].
        params: SVIParams for the slice.
        output_dir: Directory to save plot.
    
    Returns:
        Path to generated plot.
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_path = output_dir / f"svi_fit_{params.expiry_date.strftime('%Y%m%d')}.png"
    
    k = group["log_moneyness"].values
    w = group["total_variance"].values
    w_pred = raw_svi(k, params.a, params.b, params.rho, params.m, params.sigma)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(k, w, label="Market Mid IV", color="blue")
    plt.plot(k, w_pred, label="SVI Fit", color="red")
    plt.title(f"SVI Fit for {params.expiry_date.strftime('%Y-%m-%d')}\nRMSE={params.rmse:.6f}, R²={params.r_squared:.6f}")
    plt.xlabel("Log-Moneyness (log(K/F))")
    plt.ylabel("Total Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def save_svi_params(params: List[SVIParams], output_dir: Path = Path("results")) -> Path:
    """
    Save SVI parameters to JSON.
    
    Args:
        params: List of SVIParams.
        output_dir: Directory to save file.
    
    Returns:
        Path to generated file.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"svi_params_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    
    data = [
        {
            "expiry_date": p.expiry_date.isoformat(),
            "a": p.a,
            "b": p.b,
            "rho": p.rho,
            "m": p.m,
            "sigma": p.sigma,
            "rmse": p.rmse,
            "r_squared": p.r_squared,
        }
        for p in params
    ]
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_path


def save_ssvi_params(params: SSVIParams, output_dir: Path = Path("results")) -> Path:
    """
    Save SSVI parameters to JSON.
    
    Args:
        params: SSVIParams.
        output_dir: Directory to save file.
    
    Returns:
        Path to generated file.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"ssvi_params_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    
    data = {
        "rho": params.rho,
        "eta": params.eta,
        "gamma": params.gamma,
        "rmse": params.rmse,
        "r_squared": params.r_squared,
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_path