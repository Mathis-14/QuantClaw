"""
SVI calibration on BTC options (demo data).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# Load demo data
df = pd.read_csv("data/raw/demo_options.csv")

# Filter for BTC options with valid IV
df = df[df["implied_volatility"].notna()]
if len(df) == 0:
    raise ValueError("No valid options with implied_volatility")

# Prepare data
k = np.log(df["strike"] / df["underlying_price"]).values
w = (df["implied_volatility"] ** 2) * (pd.to_datetime(df["expiry_date"]) - pd.Timestamp.now()).dt.days / 365.0
weights = 1.0 / ((df["ask"] - df["bid"]) / df["bid"] * 100.0)

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
plt.scatter(k, w, label="Market", color="blue")
plt.plot(k_range, w_pred, label="SVI Fit", color="red")
plt.title(f"SVI Fit for BTC Options\nRMSE={result.fun:.6f}")
plt.xlabel("Log-Moneyness (log(K/F))")
plt.ylabel("Total Variance")
plt.legend()
plt.grid(True)

# Save plot
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "svi_fit_btc_demo.png")
plt.close()

print(f"SVI calibration complete. Plot saved to plots/svi_fit_btc_demo.png")
print(f"Parameters: a={a:.6f}, b={b:.6f}, rho={rho:.6f}, m={m:.6f}, sigma={sigma:.6f}")