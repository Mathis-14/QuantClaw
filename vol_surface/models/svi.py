"""SVI (Stochastic Volatility Inspired) per-slice model.

Parametrization:
    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

where k = log(K/F) is log-moneyness and w is total implied variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vol_surface.data.schema import SVIParams


def svi_total_variance(
    k: NDArray[np.float64],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Evaluate the SVI total-variance function."""
    km = k - m
    return a + b * (rho * km + np.sqrt(km**2 + sigma**2))


def svi_implied_vol(
    k: NDArray[np.float64],
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Implied vol from SVI total variance: sigma_impl = sqrt(w/T)."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)


def svi_from_params(params: SVIParams) -> dict[str, float]:
    return dict(a=params.a, b=params.b, rho=params.rho, m=params.m, sigma=params.sigma)


def params_to_vector(p: SVIParams) -> NDArray[np.float64]:
    return np.array([p.a, p.b, p.rho, p.m, p.sigma])


def vector_to_params(x: NDArray[np.float64]) -> SVIParams:
    return SVIParams(a=float(x[0]), b=float(x[1]), rho=float(x[2]),
                     m=float(x[3]), sigma=float(x[4]))


def svi_initial_guess(
    k: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Heuristic initial guess for SVI parameters from data."""
    w_atm = float(np.interp(0.0, k, w))
    a = max(w_atm * 0.8, 1e-4)
    b = max(float(np.std(w) / (np.std(k) + 1e-8)), 1e-4)
    rho = 0.0
    m = 0.0
    sigma = max(float(np.std(k) * 0.5), 1e-3)
    return np.array([a, b, rho, m, sigma])


def svi_parameter_bounds() -> tuple[list[float], list[float]]:
    """Return (lower, upper) bounds for [a, b, rho, m, sigma]."""
    lower = [-0.5, 1e-8, -0.999, -2.0, 1e-6]
    upper = [2.0, 5.0, 0.999, 2.0, 5.0]
    return lower, upper
