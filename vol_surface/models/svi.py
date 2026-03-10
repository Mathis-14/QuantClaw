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
    free of butterfly arbitrage.  Negative g(k) means the implied risk-neutral
    density has gone negative (static arbitrage).

    Formula: g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2

    Adapted from the reference implementation in Stat_app_bofa/SVI-SSVI/svi.py
    (butterfly_constraint).  Derivatives are computed analytically for speed
    and numerical stability.
    """
    km = k - m
    sqrt_disc = np.sqrt(km**2 + sigma**2)
    w = a + b * (rho * km + sqrt_disc)
    w_safe = np.maximum(w, 1e-10)

    # Analytical first and second derivatives of SVI w.r.t. k
    w_prime = b * (rho + km / sqrt_disc)
    w_double_prime = b * sigma**2 / sqrt_disc**3

    term1 = (1.0 - k * w_prime / (2.0 * w_safe)) ** 2
    term2 = (w_prime**2 / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w_double_prime / 2.0

    return term1 - term2 + term3


def svi_from_params(params: SVIParams) -> dict[str, float]:
    return dict(a=params.a, b=params.b, rho=params.rho, m=params.m, sigma=params.sigma)


def params_to_vector(p: SVIParams) -> NDArray[np.float64]:
    return np.array([p.a, p.b, p.rho, p.m, p.sigma])


def vector_to_params(x: NDArray[np.float64]) -> SVIParams:
    return SVIParams(a=float(x[0]), b=float(x[1]), rho=float(x[2]),
                     m=float(x[3]), sigma=float(x[4]))


def svi_initial_guess(
    k: NDArray[np.float64],
    w: NDArray[np.float64],
    T: float = 1.0,
    prior: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Maturity-aware initial guess for SVI parameters.

    Fix 3: instead of a single generic starting point for every slice, use
    maturity buckets that reflect typical equity-smile shapes at different
    tenors.  If a prior calibration result is provided (warm start), use
    those params directly so sequential slices initialise from their neighbour.
    """
    if prior is not None:
        return prior.copy()

    # Scale the 'a' component to the actual ATM total variance so the starting
    # point is in the right order of magnitude regardless of maturity.
    w_atm = max(float(np.interp(0.0, k, w)), 1e-4)

    if T < 0.5:
        # Short-dated: steep skew, tight curvature
        return np.array([w_atm * 0.9, 0.40, -0.70, 0.0, 0.10])
    elif T < 1.5:
        # Medium-dated: moderate skew and curvature
        return np.array([w_atm * 0.9, 0.20, -0.50, 0.0, 0.20])
    else:
        # Long-dated: shallow skew, broader curvature — keep b small to
        # avoid the b→upper-bound explosion seen in stale long-tenor quotes.
        return np.array([w_atm * 0.9, 0.10, -0.30, 0.0, 0.30])


def svi_parameter_bounds() -> tuple[list[float], list[float]]:
    """Return (lower, upper) bounds for [a, b, rho, m, sigma].

    Fix 2 changes vs. original:
    - b upper bound: 5.0 → 2.0  (no realistic equity smile needs b > 2)
    - sigma lower bound: 1e-6 → 1e-3  (prevents the V-shape degeneration
      where sigma → 0 makes the SVI a kinked function, breaking butterfly)
    """
    lower = [-0.5, 1e-8, -0.999, -2.0, 1e-3]
    upper = [2.0,  2.0,   0.999,  2.0, 5.0]
    return lower, upper
