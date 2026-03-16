"""
Standalone script to generate density plot for arbitrage check from demo data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
df["total_variance"] = (df["implied_volatility"] ** 2) * 102 / 365.0  # 102 days to expiry

# Compute density g(K)
strikes = df["strike"].values
total_variance = df["total_variance"].values
forward = df["underlying_price"].iloc[0]

# First derivative (d w(K) / dK)
dw_dk = np.gradient(total_variance, strikes)

# Second derivative (d² w(K) / dK²)
d2w_dk2 = np.gradient(dw_dk, strikes)

# Density g(K) = 1 - K/w * dw_dk + K²/2 * d²w_dk²
density = 1 - (strikes / total_variance) * dw_dk + (strikes**2 / 2) * d2w_dk2

# Handle edge cases (forward price)
mask = np.isclose(strikes, forward, rtol=1e-4)
density[mask] = 1.0  # Density at forward is 1

df["density"] = density

# Plot
plt.figure(figsize=(10, 6))
plt.plot(strikes, density, label="Density g(K)", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Arbitrage Boundary")
plt.title("Density g(K) for 2026-06-26")
plt.xlabel("Strike")
plt.ylabel("Density g(K)")
plt.legend()
plt.grid(True)
plt.savefig("plots/density_demo.png")
plt.close()

print("Density plot saved to plots/density_demo.png")