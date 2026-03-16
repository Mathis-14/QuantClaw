"""
Arbitrage Checks for Volatility Surfaces

Implements calendar spread and butterfly arbitrage checks.
Raises ArbitrageViolationWarning for violations.

Usage:
    from src.arbitrage import check_arbitrage, ArbitrageViolationWarning
    violations = check_arbitrage(clean_df)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArbitrageViolationWarning:
    """Warning for arbitrage violations."""
    expiry_date: pd.Timestamp
    strike: Optional[float] = None
    violation_type: Optional[str] = None  # "calendar" or "butterfly"
    severity: Optional[float] = None  # Magnitude of violation
    message: Optional[str] = None


def calendar_spread_check(
    df: pd.DataFrame, total_variance_col: str = "total_variance"
) -> List[ArbitrageViolationWarning]:
    """
    Check for calendar spread arbitrage: total variance must be non-decreasing in T.
    
    Args:
        df: DataFrame with columns [expiry_date, strike, total_variance].
        total_variance_col: Column name for total variance.
    
    Returns:
        List of ArbitrageViolationWarning for violations.
    """
    violations = []
    
    # Sort by expiry_date and strike
    df = df.copy().reset_index(drop=True)
    df.columns = df.columns.astype(str)  # Ensure column names are strings
    df = df.sort_values(["expiry_date", "strike"])
    
    # Group by strike and check total_variance is non-decreasing in expiry_date
    for strike, group in df.groupby("strike"):
        group = group.sort_values("expiry_date")
        total_variance = group[total_variance_col].values
        
        # Check for decreases
        diffs = np.diff(total_variance)
        for i, diff in enumerate(diffs):
            if diff < -1e-6:  # Allow small numerical errors
                expiry = group.iloc[i + 1]["expiry_date"]
                severity = abs(diff)
                violations.append(
                    ArbitrageViolationWarning(
                        expiry_date=expiry,
                        strike=strike,
                        violation_type="calendar",
                        severity=severity,
                        message=f"Calendar spread arbitrage: total_variance decreased by {severity:.6f} at strike {strike}",
                    )
                )
    
    return violations


def butterfly_check(
    df: pd.DataFrame, 
    total_variance_col: str = "total_variance",
    density_col: str = "density"
) -> List[ArbitrageViolationWarning]:
    """
    Check for butterfly arbitrage: density g(K) must be non-negative.
    
    Args:
        df: DataFrame with columns [expiry_date, strike, total_variance, density].
        total_variance_col: Column name for total variance.
        density_col: Column name for density g(K).
    
    Returns:
        List of ArbitrageViolationWarning for violations.
    """
    violations = []
    
    # Sort by expiry_date and strike
    df = df.sort_values(["expiry_date", "strike"])
    
    # Group by expiry_date and check density is non-negative
    for expiry, group in df.groupby("expiry_date"):
        group = group.sort_values("strike")
        density = group[density_col].values
        
        # Check for negative density
        for i, val in enumerate(density):
            if val < -1e-6:  # Allow small numerical errors
                strike = group.iloc[i]["strike"]
                severity = abs(val)
                violations.append(
                    ArbitrageViolationWarning(
                        expiry_date=expiry,
                        strike=strike,
                        violation_type="butterfly",
                        severity=severity,
                        message=f"Butterfly arbitrage: density g(K) = {val:.6f} < 0 at strike {strike}",
                    )
                )
    
    return violations


def compute_density(
    df: pd.DataFrame, 
    total_variance_col: str = "total_variance",
    forward_col: str = "forward_price",
    density_col: str = "density"
) -> pd.DataFrame:
    """
    Compute density g(K) from total variance using finite differences.
    
    Args:
        df: DataFrame with columns [expiry_date, strike, total_variance, forward_price].
        total_variance_col: Column name for total variance.
        forward_col: Column name for forward price.
        density_col: Column name for output density.
    
    Returns:
        DataFrame with density column added.
    """
    df = df.copy()
    
    # Sort by expiry_date and strike
    df = df.sort_values(["expiry_date", "strike"])
    
    # Group by expiry_date and compute density
    def _compute_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("strike")
        strikes = group["strike"].values
        total_variance = group[total_variance_col].values
        forward = group[forward_col].iloc[0]
        
        # Compute first derivative (d w(K) / dK)
        dw_dk = np.gradient(total_variance, strikes)
        
        # Compute second derivative (d² w(K) / dK²)
        d2w_dk2 = np.gradient(dw_dk, strikes)
        
        # Compute density g(K) = 1 - K/w * dw_dk + K²/2 * d²w_dk²
        density = 1 - (strikes / total_variance) * dw_dk + (strikes**2 / 2) * d2w_dk2
        
        # Handle edge cases (forward price)
        mask = np.isclose(strikes, forward, rtol=1e-4)
        density[mask] = 1.0  # Density at forward is 1
        
        group[density_col] = density
        return group
    
    df = df.groupby("expiry_date", group_keys=False).apply(_compute_group)
    return df


def generate_arbitrage_report(
    violations: List[ArbitrageViolationWarning], output_dir: Path = Path("reports")
) -> Path:
    """
    Generate arbitrage report JSON.
    
    Args:
        violations: List of ArbitrageViolationWarning.
        output_dir: Directory to save report.
    
    Returns:
        Path to generated report.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / f"arbitrage_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    
    # Aggregate violations
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_violations": len(violations),
        "calendar_violations": sum(1 for v in violations if v.violation_type == "calendar"),
        "butterfly_violations": sum(1 for v in violations if v.violation_type == "butterfly"),
        "severity_distribution": {
            "calendar": {
                "min": min((v.severity for v in violations if v.violation_type == "calendar"), default=0),
                "max": max((v.severity for v in violations if v.violation_type == "calendar"), default=0),
                "mean": np.mean([v.severity for v in violations if v.violation_type == "calendar"]) if violations else 0,
            },
            "butterfly": {
                "min": min((v.severity for v in violations if v.violation_type == "butterfly"), default=0),
                "max": max((v.severity for v in violations if v.violation_type == "butterfly"), default=0),
                "mean": np.mean([v.severity for v in violations if v.violation_type == "butterfly"]) if violations else 0,
            },
        },
        "violations": [
            {
                "expiry_date": v.expiry_date.isoformat(),
                "strike": v.strike,
                "violation_type": v.violation_type,
                "severity": v.severity,
                "message": v.message,
            }
            for v in violations
        ],
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path


def plot_density(
    df: pd.DataFrame, 
    expiry_date: pd.Timestamp, 
    output_dir: Path = Path("plots")
) -> Path:
    """
    Plot density g(K) for a given expiry slice.
    
    Args:
        df: DataFrame with columns [expiry_date, strike, density].
        expiry_date: Expiry date to plot.
        output_dir: Directory to save plot.
    
    Returns:
        Path to generated plot.
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_path = output_dir / f"density_{expiry_date.strftime('%Y%m%d')}.png"
    
    # Filter for expiry_date
    group = df[df["expiry_date"] == expiry_date]
    group = group.sort_values("strike")
    
    plt.figure(figsize=(10, 6))
    plt.plot(group["strike"], group["density"], label="Density g(K)", color="blue")
    plt.axhline(0, color="red", linestyle="--", label="Arbitrage Boundary")
    plt.title(f"Density g(K) for Expiry {expiry_date.strftime('%Y-%m-%d')}")
    plt.xlabel("Strike")
    plt.ylabel("Density g(K)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def check_arbitrage(df: pd.DataFrame) -> List[ArbitrageViolationWarning]:
    """
    Run full arbitrage checks on a volatility surface.
    
    Args:
        df: DataFrame with columns [expiry_date, strike, total_variance, forward_price].
    
    Returns:
        List of ArbitrageViolationWarning.
    """
    # Ensure required columns exist
    required_columns = {"expiry_date", "strike", "total_variance", "forward_price"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Compute density g(K)
    df = compute_density(df)
    
    # Run checks
    calendar_violations = calendar_spread_check(df)
    butterfly_violations = butterfly_check(df)
    
    # Combine violations
    violations = calendar_violations + butterfly_violations
    
    # Generate report
    report_path = generate_arbitrage_report(violations)
    logger.info(f"Arbitrage report saved to {report_path}")
    
    # Send report to Telegram
    message(
        action="send", 
        channel="telegram", 
        media=str(report_path), 
        caption=f"🚨 Arbitrage Report: {len(violations)} violations found"
    )
    
    # Plot density for each expiry and send to Telegram
    for expiry in df["expiry_date"].unique():
        plot_path = plot_density(df, expiry)
        caption = f"Density g(K) for {expiry.strftime('%Y-%m-%d')}: {len([v for v in violations if v.expiry_date == expiry])} violations"
        message(
            action="send", 
            channel="telegram", 
            media=str(plot_path), 
            caption=caption
        )
    
    return violations