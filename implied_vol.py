"""Black-Scholes implied volatility calculation."""

from typing import Optional
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


N = norm.cdf


def black_scholes(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    option_type: str = "call",
) -> float:
    """Black-Scholes price for European options."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * N(d1) - K * np.exp(-r * T) * N(d2)
    else:
        return K * np.exp(-r * T) * N(-d2) - S * N(-d1)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Optional[float]:
    """Calculate implied volatility using Brent's method."""
    try:
        def objective(sigma):
            return black_scholes(S, K, T, sigma, r, option_type) - price
        
        # Bounds for sigma
        a, b = 1e-4, 5.0
        return brentq(objective, a, b, xtol=tol, maxiter=max_iter)
    except ValueError:
        # Fallback to scalar minimization if brentq fails
        try:
            result = minimize_scalar(
                lambda sigma: abs(objective(sigma)),
                bounds=(a, b),
                method="bounded",
                options={"xatol": tol},
            )
            return result.x if result.success else None
        except Exception:
            return None