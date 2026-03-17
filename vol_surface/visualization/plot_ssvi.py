"""Visualization utilities for SSVI surfaces and diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from vol_surface.data.schema import SSVIParams
from vol_surface.models.ssvi import ssvi_implied_vol, ssvi_total_variance

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


def plot_ssvi_diagnostics(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    params: SSVIParams,
    theta_grid: np.ndarray,
    output_dir: Path,
    prefix: str = "BTC",
) -> Path:
    """Generate diagnostics: density and monotonicity checks."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Total variance monotonicity in T
    for i, k in enumerate(k_grid):
        w_T = [
            ssvi_total_variance(
                np.array([k]), theta, params.rho, params.eta, params.gamma
            )
            for theta in theta_grid
        ]
        axes[0].plot(T_grid, w_T, label=f"k={k:.2f}")
    
    axes[0].set_xlabel("Time to Maturity (days)")
    axes[0].set_ylabel("Total Variance w(k,T)")
    axes[0].set_title("Total Variance Monotonicity in T")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: Risk-neutral density (second derivative of w w.r.t. k)
    k_fine = np.linspace(k_grid.min(), k_grid.max(), 200)
    for i, T in enumerate(T_grid):
        theta = theta_grid[i]
        w_fine = ssvi_total_variance(
            k_fine, theta, params.rho, params.eta, params.gamma
        )
        # Second derivative approximation
        d2w_dk2 = np.gradient(np.gradient(w_fine, k_fine), k_fine)
        axes[1].plot(k_fine, d2w_dk2, label=f"T={T:.1f} days")
    
    axes[1].set_xlabel("Log-Moneyness (k = ln(K/F))")
    axes[1].set_ylabel("Risk-Neutral Density")
    axes[1].set_title("Risk-Neutral Density (d²w/dk²)")
    axes[1].grid(True)
    axes[1].legend()
    
    output_path = output_dir / f"{prefix}_SSVI_diagnostics_verification.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved diagnostics plot to {output_path}")
    
    return output_path