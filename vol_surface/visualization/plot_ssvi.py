"""Visualization utilities for SSVI surfaces and diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from vol_surface.data.schema import SSVIParams
from vol_surface.models.ssvi import ssvi_implied_vol, ssvi_total_variance

# Constants
FIG_SIZE_DIAGNOSTICS = (12, 10)
FIG_SIZE_SMILES = (10, 6)
FIG_SIZE_SURFACE = (12, 8)
FONT_SIZE_AXES = 12
FONT_SIZE_TITLE = 14
COLORMAP_K = "viridis"
COLORS_EXPIRY = ["blue", "orange", "green"]
GRID_ALPHA = 0.3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_ssvi_smiles(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
    output_dir: Path,
    prefix: str = "BTC",
) -> list[Path]:
    """Generate 2D smile plots for each maturity."""
    output_paths = []
    for i, T in enumerate(T_grid):
        theta = theta_grid[i]
        iv = ssvi_implied_vol(k_grid, T, theta, params.rho, params.eta, params.gamma)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_grid, iv, label=f"SSVI Smile (T={T:.1f} days)", color="blue")
        ax.set_xlabel("Log-Moneyness (k = ln(K/F))")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{prefix} SSVI Smile at T={T:.1f} days")
        ax.grid(True)
        ax.legend()
        
        output_path = output_dir / f"{prefix}_SSVI_smiles_{int(T)}D_verification.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
        logger.info(f"Saved smile plot for T={T:.1f} days to {output_path}")
    
    return output_paths


def plot_ssvi_surface_3d(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
    output_dir: Path,
    prefix: str = "BTC",
) -> Path:
    """Generate 3D surface plot of SSVI implied volatilities."""
    K, T = np.meshgrid(k_grid, T_grid)
    IV = np.empty_like(K)
    
    for i, t in enumerate(T_grid):
        theta = theta_grid[i]
        IV[i, :] = ssvi_implied_vol(k_grid, t, theta, params.rho, params.eta, params.gamma)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K, T, IV, cmap="viridis", edgecolor="none")
    
    ax.set_xlabel("Log-Moneyness (k = ln(K/F))")
    ax.set_ylabel("Time to Maturity (days)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{prefix} SSVI Surface (3D)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    output_path = output_dir / f"{prefix}_SSVI_surface_3d_verification.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 3D surface plot to {output_path}")
    
    return output_path


def plot_total_variance_monotonicity(
    ax: Axes,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
) -> None:
    """Plot total variance monotonicity in T for each moneyness (k).

    Args:
        ax: Matplotlib axis.
        k_grid: Array of log-moneyness values.
        T_grid: Array of time-to-maturity values.
        params: SSVI parameters.
        theta_grid: Array of ATM total variances for each T.
    """
    norm = Normalize(vmin=k_grid.min(), vmax=k_grid.max())
    cmap = plt.get_cmap(COLORMAP_K)
    
    for i, k in enumerate(k_grid):
        w_T = [
            ssvi_total_variance(
                np.array([k]), theta, params.rho, params.eta, params.gamma
            )
            for theta in theta_grid
        ]
        ax.plot(T_grid, w_T, color=cmap(norm(k)), label=f"k={k:.2f}")
    
    ax.set_xlabel("Time to Maturity (days)", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Total Variance w(k,T)", fontsize=FONT_SIZE_AXES)
    ax.set_title("Total Variance Monotonicity in T", fontsize=FONT_SIZE_TITLE)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)


def plot_risk_neutral_density(
    ax: Axes,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
) -> None:
    """Plot risk-neutral density (second derivative of total variance w.r.t. k).

    Args:
        ax: Matplotlib axis.
        k_grid: Array of log-moneyness values.
        T_grid: Array of time-to-maturity values.
        params: SSVI parameters.
        theta_grid: Array of ATM total variances for each T.
    """
    k_fine = np.linspace(k_grid.min(), k_grid.max(), 200)
    
    for i, T in enumerate(T_grid):
        theta = theta_grid[i]
        w_fine = ssvi_total_variance(
            k_fine, theta, params.rho, params.eta, params.gamma
        )
        # Second derivative approximation
        d2w_dk2 = np.gradient(np.gradient(w_fine, k_fine), k_fine)
        ax.plot(k_fine, d2w_dk2, color=COLORS_EXPIRY[i], label=f"T={T:.1f} days")
        
        # Shade arbitrage regions (density < 0)
        arbitrage_mask = d2w_dk2 < 0
        if np.any(arbitrage_mask):
            ax.fill_between(
                k_fine, 0, d2w_dk2,
                where=arbitrage_mask,
                color="red",
                alpha=0.3,
                label="Arbitrage (density < 0)"
            )
    
    ax.set_xlabel("Log-Moneyness (k = ln(K/F))", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Density", fontsize=FONT_SIZE_AXES)
    ax.set_title("Risk-Neutral Density (∂²w/∂k²)", fontsize=FONT_SIZE_TITLE)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)


def plot_ssvi_diagnostics(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
    output_dir: Path,
    prefix: str = "BTC",
) -> Path:
    """Generate diagnostics: density and monotonicity checks.

    Args:
        k_grid: Array of log-moneyness values.
        T_grid: Array of time-to-maturity values.
        params: SSVI parameters.
        theta_grid: Array of ATM total variances for each T.
        output_dir: Directory to save the plot.
        prefix: Prefix for the filename (e.g., "BTC" or "ETH").

    Returns:
        Path to the saved diagnostics plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE_DIAGNOSTICS)
    
    # Plot total variance monotonicity
    plot_total_variance_monotonicity(axes[0], k_grid, T_grid, params, theta_grid)
    
    # Plot risk-neutral density
    plot_risk_neutral_density(axes[1], k_grid, T_grid, params, theta_grid)
    
    plt.tight_layout()
    output_path = output_dir / f"{prefix}_SSVI_diagnostics_fixed.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved fixed diagnostics plot to {output_path}")
    
    return output_path