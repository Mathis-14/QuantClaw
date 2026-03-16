"""
SVI calibration on BTC options (real data).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime

# Load BTC data
csv_path = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/raw/btc_options_20260316_140247.csv"
df = pd.read_csv(csv_path)

# Filter for valid IV and non-zero bid/ask
print(f"Total rows: {len(df)}")
df = df[df["mark_iv"].notna()]
print(f"Rows with IV: {len(df)}")
df = df[(df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)]
print(f"Rows with bid/ask > 0: {len(df)}")

# Map columns for calibration
df["bid"] = df["best_bid_price"]
df["ask"] = df["best_ask_price"]
df["implied_volatility"] = df["mark_iv"] / 100.0  # Convert % to decimal

if len(df) == 0:
    raise ValueError("No valid options with implied_volatility and bid/ask > 0")

# Debug: Show unique expiries
expiries = pd.to_datetime(df["expiry_date"], unit="ms").unique()
print(f"Unique expiries: {expiries}")

# Use the first expiry for SVI calibration (most liquid)
first_expiry = pd.to_datetime(df["expiry_date"].iloc[0], unit="ms")
df = df[pd.to_datetime(df["expiry_date"], unit="ms") == first_expiry]
print(f"Rows for first expiry ({first_expiry}): {len(df)}")

# Prepare data
k = np.log(df["strike"] / df["underlying_price"]).values
w = (df["implied_volatility"] ** 2) * (first_expiry - datetime.now()).days / 365.0
weights = 1.0 / ((df["best_ask_price"] - df["best_bid_price"]) / df["best_bid_price"] * 100.0)

# SVI model
def raw_svi(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

# Calibrate SVI
def svi_objective(params):
    a, b, rho, m, sigma = params
    w_pred = raw_svi(k, a, b, rho, m, sigma)
    return np.sqrt(np.mean(weights * (w - w_pred) ** 2))

p0 = [0.01, 0.1, -0.5, 0.0, 0.1]
bounds = [(-np.inf, np.inf), (0, np.inf), (-1, 1), (-np.inf, np.inf), (1e-6, np.inf)]
result = minimize(svi_objective, p0, bounds=bounds, method="L-BFGS-B")
a, b, rho, m, sigma = result.x

# Plot SVI fit
k_range = np.linspace(min(k) - 0.1, max(k) + 0.1, 100)
w_pred = raw_svi(k_range, a, b, rho, m, sigma)

plt.figure(figsize=(10, 6))
plt.scatter(k, w, label="Market", color="blue", alpha=0.6)
plt.plot(k_range, w_pred, label="SVI Fit", color="red", linewidth=2)
plt.title(f"SVI Fit for BTC Options (Expiry: {first_expiry.strftime('%Y-%m-%d')})\nRMSE={result.fun:.6f}")
plt.xlabel("Log-Moneyness (log(K/F))")
plt.ylabel("Total Variance")
plt.legend()
plt.grid(True)

# Save plot
output_dir = Path("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots")
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / f"svi_fit_btc_{first_expiry.strftime('%Y%m%d')}.png"
plt.savefig(plot_path)
plt.close()

print(f"SVI calibration complete. Plot saved to {plot_path}")
print(f"Parameters: a={a:.6f}, b={b:.6f}, rho={rho:.6f}, m={m:.6f}, sigma={sigma:.6f}")

# Send plot to Telegram
from openclaw.tool import message
message(
    action="send",
    channel="telegram",
    media=str(plot_path),
    caption=f"SVI Fit for BTC Options (Expiry: {first_expiry.strftime('%Y-%m-%d')})\nRMSE={result.fun:.6f}"
)