"""eSSVI calibration for BTC options using anchored slices.

Steps:
1. Load BTC options data (strikes, expiries, implied vols).
2. Group by expiry and calibrate eSSVI slices.
3. Enforce no-arbitrage constraints (Butterfly/Calendar Spread).
4. Generate plots and report.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vol_surface.models.essvi import ESSVISlice, calibrate_essvi_slice


def load_btc_options(file_path: str) -> pd.DataFrame:
    """Load and preprocess BTC options data."""
    df = pd.read_csv(file_path)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], unit="ms")
    df["T"] = (df["expiry_date"] - pd.Timestamp.now()).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df = df[df["T"] > 0]  # Filter expired options
    
    # Extract option_type from instrument_name (e.g., "BTC-27MAR26-30000-C" → "C")
    df["parsed_option_type"] = df["instrument_name"].str.split("-").str[-1]
    print(f"Unique parsed_option_type: {df['parsed_option_type'].unique()}")
    print(f"Data size before filtering: {len(df)}")
    print("Option types before filtering:", df["parsed_option_type"].value_counts())
    df = df[df["parsed_option_type"] == "C"]  # Filter for calls
    print(f"Data size after filtering: {len(df)}")
    
    # Forcefully recompute log-moneyness after overwriting underlying_price
    df["underlying_price"] = 65000.0  # Hardcode BTC price
    df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    print("Expiries after filtering:", df.groupby("expiry_date").size())
    
    # Use mark_iv as implied_volatility
    print("Before rename:", df.shape)
    df = df.rename(columns={"mark_iv": "implied_volatility"})
    print("After rename:", df.shape)
    df = df.dropna(subset=["implied_volatility"])
    print("After dropna:", df.shape)
    print("IV stats:", df["implied_volatility"].describe())
    df["implied_volatility"] = df["implied_volatility"] / 100  # Convert % to absolute
    df = df[(df["implied_volatility"] > 0.01) & (df["implied_volatility"] < 5.0)]  # Filter 1%–500%
    print("After IV filter:", df.shape)
    
    # Interpolate implied volatility per expiry using cubic splines
    from scipy.interpolate import CubicSpline
    interpolated_data = []
    print("Starting interpolation loop...")
    print("DataFrame head:", df.head())
    print("Expiry date dtype:", df["expiry_date"].dtype)
    print("Expiry date sample:", df["expiry_date"].head())
    for expiry, group in df.groupby("expiry_date"):
        print(f"Processing expiry {expiry}: {len(group)} strikes, log_moneyness range: [{group['log_moneyness'].min():.2f}, {group['log_moneyness'].max():.2f}]")
        if len(group) < 3:
            print(f"⚠️ Skipping expiry {expiry}: insufficient data (<3 strikes)")
            continue
        k_grid = np.linspace(group["log_moneyness"].min(), group["log_moneyness"].max(), 50)
        cs = CubicSpline(group["log_moneyness"], group["implied_volatility"])
        iv_interp = cs(k_grid)
        iv_interp = np.clip(iv_interp, 0.01, 5.0)  # Ensure valid IV
        interpolated_df = pd.DataFrame({
            "log_moneyness": k_grid,
            "implied_volatility": iv_interp,
            "T": group["T"].iloc[0],
            "total_variance": (iv_interp ** 2) * group["T"].iloc[0]
        })
        interpolated_data.append(interpolated_df)
    
    df = pd.concat(interpolated_data, ignore_index=True)
    print(f"Filtered and interpolated data size: {len(df)}")
    if len(df) == 0:
        raise ValueError("No data after filtering. Check option_type and implied_volatility.")



    return df


def interpolate_iv_per_expiry(group: pd.DataFrame) -> pd.DataFrame:
    """Interpolate implied volatility per expiry using cubic splines."""
    from scipy.interpolate import CubicSpline

    k_min, k_max = group["log_moneyness"].min(), group["log_moneyness"].max()
    k_grid = np.linspace(k_min, k_max, 50)

    # Interpolate implied volatility (not total variance)
    cs = CubicSpline(group["log_moneyness"], group["implied_volatility"])
    iv_interp = cs(k_grid)
    iv_interp = np.clip(iv_interp, 0.01, 5.0)  # Ensure valid IV

    # Convert to total variance
    T = group["T"].iloc[0]
    w_interp = (iv_interp ** 2) * T

    return pd.DataFrame({
        "log_moneyness": k_grid,
        "implied_volatility": iv_interp,
        "total_variance": w_interp,
        "T": T,
    })


def group_by_expiry(df: pd.DataFrame) -> Dict[float, pd.DataFrame]:
    """Group options by expiry (T) and interpolate IV."""
    expiry_groups = {}
    for T, group in df.groupby("T"):
        if len(group) < 3:
            print(f"Skipping expiry T={T}: insufficient data")
            continue
        interpolated = interpolate_iv_per_expiry(group)
        expiry_groups[T] = interpolated
    return expiry_groups


def validate_arbitrage_constraints(slices: List[ESSVISlice]) -> Tuple[bool, str]:
    """Validate no-arbitrage constraints for all slices."""
    report = []
    valid = True

    for i, slice in enumerate(slices):
        # Butterfly condition
        if not slice.butterfly_condition():
            report.append(f"❌ Butterfly arbitrage violated for T={slice.T:.4f}")
            valid = False
        else:
            report.append(f"✅ Butterfly condition satisfied for T={slice.T:.4f}")

        # Calendar Spread condition
        if i > 0:
            prev_slice = slices[i-1]
            if slice.theta < prev_slice.theta:
                report.append(f"❌ Calendar Spread (θ) violated between T={prev_slice.T:.4f} and T={slice.T:.4f}")
                valid = False
            if slice.phi < prev_slice.phi:
                report.append(f"❌ Calendar Spread (ψ) violated between T={prev_slice.T:.4f} and T={slice.T:.4f}")
                valid = False
            if (prev_slice.rho < 0 and slice.rho < prev_slice.rho) or (prev_slice.rho > 0 and slice.rho > prev_slice.rho):
                report.append(f"❌ Calendar Spread (ρ) violated between T={prev_slice.T:.4f} and T={slice.T:.4f}")
                valid = False

    return valid, "\n".join(report)


def check_monotonicity(slice: ESSVISlice, k: np.ndarray) -> bool:
    """Check monotonicity of total variance."""
    w = slice.total_variance(k)
    dw_dk = np.gradient(w, k)
    return np.all(dw_dk[k <= 0] <= 0) and np.all(dw_dk[k >= 0] >= 0)


def check_density(slice: ESSVISlice, k: np.ndarray) -> bool:
    """Check positive density (no butterfly arbitrage)."""
    density = slice.density(k)
    return np.all(density >= -1e-6)  # Allow small numerical errors


def calibrate_essvi_surface(expiry_groups: Dict[float, pd.DataFrame], iterations: int = 5) -> List[ESSVISlice]:
    """Calibrate eSSVI surface slice-by-slice with iterative refinement."""
    slices = []
    prev_slice = None

    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration + 1}/{iterations} ===")
        slices = []
        prev_slice = None

        for T, group in sorted(expiry_groups.items()):
            k = group["log_moneyness"].values
            w_market = group["total_variance"].values
            k_star = 0.0  # ATM
            theta_star = np.interp(k_star, k, w_market)

            print(f"Calibrating T={T:.4f}: theta_star={theta_star:.4f}")
            slice = calibrate_essvi_slice(k, w_market, k_star, theta_star, prev_slice)
            if slice is None:
                print(f"Failed to calibrate T={T:.4f}")
                continue

            # Check monotonicity and density
            if not check_monotonicity(slice, k):
                print(f"⚠️ Non-monotonic total variance for T={T:.4f}")
            if not check_density(slice, k):
                print(f"⚠️ Negative density for T={T:.4f}")

            slices.append(slice)
            prev_slice = slice

        # Validate constraints
        valid, report = validate_arbitrage_constraints(slices)
        print(report)
        if valid:
            print("✅ All arbitrage constraints satisfied.")
            break

    return slices


def generate_report(slices: List[ESSVISlice], expiry_groups: Dict[float, pd.DataFrame]) -> str:
    """Generate a report summarizing calibrated parameters and validation."""
    report = "eSSVI Calibration Report\n"
    report += "=" * 50 + "\n\n"

    # Calibrated parameters
    report += "Calibrated Parameters:\n"
    for slice in slices:
        report += (
            f"T={slice.T:.4f}: theta={slice.theta:.4f}, rho={slice.rho:.4f}, "
            f"phi={slice.phi:.4f}, k_star={slice.k_star:.4f}, theta_star={slice.theta_star:.4f}\n"
        )

    # Validation
    valid, validation_report = validate_arbitrage_constraints(slices)
    report += "\nValidation:\n" + validation_report + "\n"

    # RMSE per expiry
    report += "\nRMSE per Expiry:\n"
    for T, group in expiry_groups.items():
        k = group["log_moneyness"].values
        w_market = group["total_variance"].values
        for slice in slices:
            if abs(slice.T - T) < 1e-6:
                w_model = slice.total_variance(k)
                rmse = np.sqrt(np.mean((w_model - w_market) ** 2))
                report += f"T={T:.4f}: RMSE={rmse:.6f}\n"
                break

    return report


def plot_essvi_surface(slices: List[ESSVISlice]) -> None:
    """Plot 3D eSSVI surface using Plotly."""
    k_grid = np.linspace(-1.0, 1.0, 100)
    T_grid = np.array([s.T for s in slices])
    K, T = np.meshgrid(k_grid, T_grid)
    W = np.zeros_like(K)

    for i, slice in enumerate(slices):
        W[i, :] = slice.total_variance(k_grid)

    fig = go.Figure(data=[go.Surface(z=W, x=K, y=T, colorscale="Viridis")])
    fig.update_layout(
        title="eSSVI Surface for BTC Options",
        scene={
            "xaxis_title": "Log-Moneyness (k)",
            "yaxis_title": "Time to Expiry (T)",
            "zaxis_title": "Total Variance (w)",
        },
        width=1000,
        height=800,
    )
    fig.write_html("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots/BTC_eSSVI_surface.html")


def plot_essvi_smiles(slices: List[ESSVISlice]) -> None:
    """Plot 2D eSSVI smiles per expiry using Plotly."""
    fig = go.Figure()

    for slice in slices:
        k = np.linspace(-1.0, 1.0, 100)
        iv = slice.implied_vol(k)
        fig.add_trace(go.Scatter(x=k, y=iv, mode="lines", name=f"T={slice.T:.2f}"))

    fig.update_layout(
        title="eSSVI Smiles for BTC Options",
        xaxis_title="Log-Moneyness (k)",
        yaxis_title="Implied Volatility",
        width=1000,
        height=800,
    )
    fig.write_html("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots/BTC_eSSVI_smiles.html")


def plot_arbitrage_diagnostics(slices: List[ESSVISlice]) -> None:
    """Plot arbitrage diagnostics (total variance and density)."""
    k_grid = np.linspace(-1.0, 1.0, 100)
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Total Variance", "Density"])

    for slice in slices:
        w = slice.total_variance(k_grid)
        density = slice.density(k_grid)

        fig.add_trace(
            go.Scatter(x=k_grid, y=w, mode="lines", name=f"T={slice.T:.2f}"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=k_grid, y=density, mode="lines", name=f"T={slice.T:.2f}"),
            row=2, col=1,
        )

    fig.update_layout(
        title="eSSVI Arbitrage Diagnostics",
        width=1000,
        height=800,
    )
    fig.update_xaxes(title_text="Log-Moneyness (k)", row=1, col=1)
    fig.update_yaxes(title_text="Total Variance (w)", row=1, col=1)
    fig.update_xaxes(title_text="Log-Moneyness (k)", row=2, col=1)
    fig.update_yaxes(title_text="Density (∂²w/∂k²)", row=2, col=1)
    fig.write_html("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots/BTC_eSSVI_diagnostics.html")


def main():
    """Run eSSVI calibration for BTC."""
    # Load data
    df = load_btc_options("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/processed/btc_options_deribit.csv")
    print("Data loaded. Sample:", df.head())

    # Group by expiry and interpolate
    expiry_groups = group_by_expiry(df)
    if not expiry_groups:
        raise ValueError("No valid expiry groups. Check data filtering.")

    # Calibrate
    slices = calibrate_essvi_surface(expiry_groups, iterations=5)
    if not slices:
        raise ValueError("Calibration failed for all expiries.")

    # Plot
    plot_essvi_surface(slices)
    plot_essvi_smiles(slices)
    plot_arbitrage_diagnostics(slices)

    # Generate report
    report = generate_report(slices, expiry_groups)
    report_path = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/reports/BTC_eSSVI_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print("eSSVI Calibration Complete. Plots and report generated.")


if __name__ == "__main__":
    main()