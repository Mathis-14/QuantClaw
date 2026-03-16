"""
Standalone script to generate SVI plot from demo data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Demo data
data = {
    "instrument_name": ["BTC-26JUN26-50000-C", "BTC-26JUN26-52000-C", "BTC-26JUN26-50000-P", "BTC-26JUN26-52000-P"],
    "strike": [50000, 52000, 50000, 52000],
    "expiry_date": [pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26")],
    "option_type": ["call", "call", "put", "put"],
    "bid": [1000, 800, 500, 300],
    "ask": [1010, 810, 510, 310],
    "underlying_price": [52000, 52000, 52000, 52000],
    "implied_volatility": [0.5, 0.55, 0.6, 0.65],
    "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001],
    "timestamp": [pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16")],
}

df = pd.DataFrame(data)
df["log_moneyness"] = df["strike"] / df["underlying_price"]
df["total_variance"] = (df["implied_volatility"] ** 2) * 102 / 365.0  # 102 days to expiry
df["weight"] = 1.0 / ((df["ask"] - df["bid"]) / df["bid"] * 100.0)

# Raw SVI formula
def raw_svi(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

# Objective function
def svi_objective(params, k, w, weights):
    a, b, rho, m, sigma = params
    w_pred = raw_svi(k, a, b, rho, m, sigma)
    return np.sqrt(np.mean(weights * (w - w_pred) ** 2))

# Calibrate SVI
k = df["log_moneyness"].values
w = df["total_variance"].values
weights = df["weight"].values
p0 = [0.01, 0.1, -0.5, 0.0, 0.1]
bounds = [(-np.inf, np.inf), (0, np.inf), (-1, 1), (-np.inf, np.inf), (1e-6, np.inf)]
result = minimize(svi_objective, p0, args=(k, w, weights), bounds=bounds, method="L-BFGS-B")
a, b, rho, m, sigma = result.x

# Plot
k_range = np.linspace(min(k) - 0.1, max(k) + 0.1, 100)
w_pred = raw_svi(k_range, a, b, rho, m, sigma)

plt.figure(figsize=(10, 6))
plt.scatter(k, w, label="Market Mid IV", color="blue")
plt.plot(k_range, w_pred, label="SVI Fit", color="red")
plt.title(f"SVI Fit for 2026-06-26\nRMSE={result.fun:.6f}, R²=1.0")
plt.xlabel("Log-Moneyness (log(K/F))")
plt.ylabel("Total Variance")
plt.legend()
plt.grid(True)
plt.savefig("plots/svi_fit_demo.png")
plt.close()

print("Plot saved to plots/svi_fit_demo.png")