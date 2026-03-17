"""Visualization tools for volatility surfaces and diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_ssvi_smiles(
    data: pd.DataFrame,
    params: Dict[str, float],
    spot_price: float,
    output_dir: Path,
    asset: str = "BTC",
) -> None:
    """Plot SSVI volatility smiles for all expiries.

    Args:
        data: DataFrame with options data.
        params: SSVI parameters (rho, eta, gamma).
        spot_price: Underlying spot price.
        output_dir: Directory to save the plot.
        asset: Asset name (e.g., "BTC").
    """
    expiries = pd.to_datetime(data['expiry_date']).dt.date.unique()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for expiry in expiries:
        expiry_df = data[data['expiry_date'].str.contains(expiry.strftime('%Y-%m-%d'))]
        strikes = expiry_df['strike'].values
        k = np.log(strikes / spot_price)
        T = (pd.to_datetime(expiry) - pd.Timestamp.now()).days / 365.25
        
        # Market implied vols
        market_vols = expiry_df['mark_iv'].values
        
        # SSVI implied vols (placeholder: replace with actual SSVI function)
        from vol_surface.models.ssvi import ssvi_implied_vol
        theta = np.mean((market_vols ** 2) * T)
        ssvi_vols = ssvi_implied_vol(k, T, theta, params['rho'], params['eta'], params['gamma'])
        
        plt.plot(k, market_vols, 'o', label=f'Market {expiry}', alpha=0.5)
        plt.plot(k, ssvi_vols, '-', label=f'SSVI {expiry}', alpha=0.8)
    
    plt.title(f'SSVI Volatility Smiles for {asset}')
    plt.xlabel('Log-Moneyness (log(K/F))')
    plt.ylabel('Implied Volatility')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f'ssvi_smiles_{asset.lower()}_recalibrated.png')
    plt.close()
    logger.info(f"SSVI smiles plot saved to {output_dir}")


def plot_ssvi_surface_3d(
    data: pd.DataFrame,
    params: Dict[str, float],
    spot_price: float,
    output_dir: Path,
    asset: str = "BTC",
) -> None:
    """Plot 3D SSVI surface for total variance.

    Args:
        data: DataFrame with options data.
        params: SSVI parameters (rho, eta, gamma).
        spot_price: Underlying spot price.
        output_dir: Directory to save the plot.
        asset: Asset name (e.g., "BTC").
    """
    expiries = pd.to_datetime(data['expiry_date']).dt.date.unique()
    strikes = np.linspace(50000, 90000, 50)
    log_moneyness = np.log(strikes / spot_price)
    
    # Create meshgrid for 3D plot
    T_values = [(pd.to_datetime(expiry) - pd.Timestamp.now()).days / 365.25 for expiry in expiries]
    k_values = log_moneyness
    T_grid, k_grid = np.meshgrid(T_values, k_values, indexing='ij')
    
    # Calculate total variance for each (k, T)
    from vol_surface.models.ssvi import ssvi_total_variance
    w_grid = np.zeros_like(T_grid)
    for i, expiry in enumerate(expiries):
        theta = np.mean((data[data['expiry_date'].str.contains(expiry.strftime('%Y-%m-%d'))]['mark_iv'] ** 2) * T_values[i])
        for j, k in enumerate(k_values):
            w_grid[i, j] = ssvi_total_variance(k, theta, params['rho'], params['eta'], params['gamma'])
    
    # Plot 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k_grid, T_grid, w_grid, cmap='viridis', edgecolor='k', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Log-Moneyness (log(K/F))')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Total Variance')
    ax.set_title(f'3D SSVI Surface for {asset}')
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'ssvi_surface_{asset.lower()}_recalibrated.png')
    plt.close()
    logger.info(f"3D SSVI surface plot saved to {output_dir}")