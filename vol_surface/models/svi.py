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
    w = np.maximum(svi_total_variance(k, a, b, rho, m, sigma), 1e-10)
    return np.sqrt(w / T)


def svi_butterfly_g(
    k: NDArray[np.float64],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Gatheral (2006) butterfly density g(k).

    g(k) >= 0 everywhere is necessary and sufficient for the SVI smile to be
    free of butterfly arbitrage.

    Formula: g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    """
    km = k - m
    sqrt_disc = np.sqrt(km**2 + sigma**2)
    w = np.maximum(a + b * (rho * km + sqrt_disc), 1e-10)

    w_prime = b * (rho + km / sqrt_disc)
    w_double_prime = b * sigma**2 / sqrt_disc**3

    term1 = (1.0 - k * w_prime / (2.0 * w)) ** 2
    term2 = (w_prime**2 / 4.0) * (1.0 / w + 0.25)
    term3 = w_double_prime / 2.0

    return term1 - term2 + term3


# ── Conversion helpers ──────────────────────────────────────────────────────


def params_to_vector(p: SVIParams) -> NDArray[np.float64]:
    return np.array([p.a, p.b, p.rho, p.m, p.sigma])


def vector_to_params(x: NDArray[np.float64]) -> SVIParams:
    return SVIParams(a=float(x[0]), b=float(x[1]), rho=float(x[2]),
                     m=float(x[3]), sigma=float(x[4]))


# ── Initial guess & bounds ──────────────────────────────────────────────────


def svi_initial_guess(
    k: NDArray[np.float64],
    w: NDArray[np.float64],
    T: float = 1.0,
    prior: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Maturity-aware initial guess for SVI parameters.

    Uses tenor buckets (short < 6M, medium < 18M, long) that reflect typical
    equity smile shapes.  If *prior* is given (warm-start from a neighbouring
    slice), returns it directly.
    """
    if prior is not None:
        return prior.copy()

    w_atm = max(float(np.interp(0.0, k, w)), 1e-4)

    if T < 0.5:
        return np.array([w_atm * 0.9, 0.40, -0.70, 0.0, 0.10])
    elif T < 1.5:
        return np.array([w_atm * 0.9, 0.20, -0.50, 0.0, 0.20])
    else:
        return np.array([w_atm * 0.9, 0.10, -0.30, 0.0, 0.30])


def svi_parameter_bounds() -> tuple[list[float], list[float]]:
    """(lower, upper) bounds for [a, b, rho, m, sigma].

    b capped at 2.0 (no realistic equity smile needs more).
    sigma >= 1e-3 to prevent the V-shape degeneracy where sigma -> 0
    turns SVI into a kinked function that breaks butterfly density.
    """
    lower = [-0.5, 1e-8, -0.999, -2.0, 1e-3]
    upper = [2.0,  2.0,   0.999,  2.0, 5.0]
    return lower, upper
