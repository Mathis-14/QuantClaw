"""Optimization wrappers for SVI and SSVI calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from vol_surface.data.schema import SVIParams, SSVIParams, VolSlice
from vol_surface.models.svi import (
    svi_butterfly_g,
    svi_initial_guess,
    svi_parameter_bounds,
    svi_total_variance,
    vector_to_params,
)
from vol_surface.models.ssvi import (
    ssvi_initial_guess,
    ssvi_parameter_bounds,
    ssvi_total_variance,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Fix 4: butterfly soft-penalty weight in the SVI residual vector.
# Large enough to steer away from negative-density solutions but not so large
# that it dominates the RMSE term and distorts the fit.
LAMBDA_ARB = 10.0


@dataclass
class OptResult:
    params: NDArray[np.float64]
    cost: float
    success: bool
    hessian_inv: NDArray[np.float64] | None = None
    message: str = ""


def calibrate_svi_slice(
    vol_slice: VolSlice,
    max_retries: int = MAX_RETRIES,
    prior: NDArray[np.float64] | None = None,
) -> tuple[SVIParams | None, OptResult]:
    """Calibrate SVI to a single maturity slice.

    Uses least_squares (TRF) which is far more robust than SLSQP for finding
    global minima on noisy market data.  Penalties are encoded as extra
    pseudo-residuals appended to the fit residuals so TRF can exploit its
    trust-region structure:

    - Fix 2: ATM-variance hard-like constraint as a heavy pseudo-residual
    - Fix 3: maturity-aware warm-start via T-bucket or prior
    - Fix 4: butterfly soft penalty as pseudo-residuals on a fine k-grid
    """
    k, w, iv, weights = vol_slice.as_arrays()
    n_data = len(k)
    lower, upper = svi_parameter_bounds()

    # Fine grid for butterfly penalty — span the fitted range plus a small buffer
    k_fine = np.linspace(float(k.min()) - 0.05, float(k.max()) + 0.05, 200)

    def residuals(x: NDArray) -> NDArray:
        # 1) Weighted total-variance fit residuals
        w_model = svi_total_variance(k, *x)
        res_fit = np.sqrt(weights) * (w_model - w)

        # Fix 4: butterfly pseudo-residuals on a fine grid.
        # max(0, -g(k)) is non-zero only where the risk-neutral density goes
        # negative; sqrt(LAMBDA_ARB) so that squaring recovers LAMBDA_ARB * sum(-g)^2.
        g = svi_butterfly_g(k_fine, *x)
        res_butterfly = np.sqrt(LAMBDA_ARB) * np.maximum(0.0, -g)

        # Fix 2: ATM total variance a + b*sigma must be >= 1e-4.
        # Encode as a single heavy pseudo-residual; 1e3 weight makes it near-hard.
        atm_viol = max(0.0, 1e-4 - (x[0] + x[1] * x[4]))
        res_atm = np.array([1e3 * atm_viol])

        # Fix 2: soft penalty to keep |rho| <= 0.95.
        rho_viol = max(0.0, abs(x[2]) - 0.95)
        res_rho = np.array([10.0 * rho_viol])

        return np.concatenate([res_fit, res_butterfly, res_atm, res_rho])

    # Fix 3: diverse starting points to maximise convergence robustness.
    # Attempt 0: warm-start prior or maturity-bucket guess
    # Attempt 1: data-driven guess (independent of maturity assumption)
    # Attempt 2: flat-smile failsafe
    x0_candidates: list[NDArray] = [
        svi_initial_guess(k, w, T=vol_slice.T, prior=prior),
        _data_driven_guess(k, w),
        np.array([max(float(np.mean(w)), 1e-4), 0.05, -0.3, 0.0, 0.30]),
    ]

    best: OptResult | None = None
    for attempt in range(max_retries):
        x0 = x0_candidates[attempt % len(x0_candidates)].copy()
        if attempt >= len(x0_candidates):
            rng = np.random.default_rng(attempt)
            x0 = x0 * (1.0 + 0.15 * rng.standard_normal(5))

        x0 = np.clip(x0, lower, upper)
        x0[2] = np.clip(x0[2], -0.99, 0.99)   # rho stays in open interval
        x0[4] = max(float(x0[4]), 1e-3)        # sigma >= 1e-3 (Fix 2)

        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=5000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            cost = float(result.cost)
            # Covariance only from the data residuals, not the penalty rows.
            hess_inv = _approximate_hessian_inv(result.jac[:n_data])

            opt = OptResult(
                params=result.x,
                cost=cost,
                success=bool(result.success),
                hessian_inv=hess_inv,
                message=result.message,
            )
            if best is None or cost < best.cost:
                best = opt
            if result.success:
                break
        except Exception as exc:
            logger.warning("SVI attempt %d failed: %s", attempt + 1, exc)
            continue

    if best is None:
        x0_fb = svi_initial_guess(k, w, T=vol_slice.T)
        return None, OptResult(
            params=x0_fb, cost=float("inf"), success=False,
            message="All optimization attempts failed",
        )

    try:
        svi_params = vector_to_params(best.params)
    except Exception as exc:
        logger.error("Invalid SVI params: %s", exc)
        return None, best

    if svi_params.no_arb_lower_bound() < -1e-8:
        logger.warning(
            "SVI no-arb lower bound violated: %.6f", svi_params.no_arb_lower_bound()
        )

    return svi_params, best


def calibrate_ssvi_surface(
    slices: list[VolSlice],
    atm_total_vars: list[float],
    max_retries: int = MAX_RETRIES,
) -> tuple[SSVIParams | None, OptResult]:
    """Calibrate SSVI jointly across all maturities.

    *atm_total_vars* are the ATM total variances theta_t for each slice,
    typically taken from per-slice SVI calibrations evaluated at k=0.
    """
    all_k = []
    all_w = []
    all_weights = []

    for vol_slice, theta in zip(slices, atm_total_vars):
        k, w, iv, weights = vol_slice.as_arrays()
        all_k.append(k)
        all_w.append(w)
        all_weights.append(weights)

    k_all = np.concatenate(all_k)
    w_all = np.concatenate(all_w)
    wt_all = np.concatenate(all_weights)

    x0 = ssvi_initial_guess()
    # Fix 6: use tighter equity-specific bounds (defined in ssvi.py).
    lower, upper = ssvi_parameter_bounds()

    def objective(x: NDArray) -> float:
        rho, eta, gamma = x
        w_model = np.empty_like(k_all)
        idx = 0
        for ki, theta_i in zip(all_k, atm_total_vars):
            n = len(ki)
            w_model[idx: idx + n] = ssvi_total_variance(ki, theta_i, rho, eta, gamma)
            idx += n
        return float(np.sum(wt_all * (w_model - w_all) ** 2))

    def constraint_no_arb(x: NDArray) -> float:
        return 4.0 - x[1] * (1 + abs(x[0]))

    best: OptResult | None = None
    for attempt in range(max_retries):
        x0_p = x0.copy()
        if attempt > 0:
            rng = np.random.default_rng(attempt + 42)
            x0_p += 0.1 * attempt * rng.standard_normal(3)
            x0_p = np.clip(x0_p, lower, upper)

        try:
            result = minimize(
                objective,
                x0_p,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints={"type": "ineq", "fun": constraint_no_arb},
                options={"maxiter": 2000, "ftol": 1e-14},
            )
            cost = float(result.fun)
            opt = OptResult(
                params=result.x,
                cost=cost,
                success=bool(result.success),
                message=str(result.message),
            )
            if best is None or cost < best.cost:
                best = opt
            if result.success:
                break
        except Exception as exc:
            logger.warning("SSVI attempt %d failed: %s", attempt + 1, exc)
            continue

    if best is None:
        return None, OptResult(
            params=x0, cost=float("inf"), success=False,
            message="All SSVI optimization attempts failed",
        )

    try:
        ssvi_params = SSVIParams(
            rho=float(best.params[0]),
            eta=float(best.params[1]),
            gamma=float(best.params[2]),
        )
    except Exception as exc:
        logger.error("Invalid SSVI params: %s", exc)
        return None, best

    return ssvi_params, best


def _data_driven_guess(
    k: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Data-driven SVI initial guess (independent of maturity assumption).

    Used as a fallback starting point alongside the T-bucket guess so the
    optimizer has a diverse set of initial conditions.
    """
    w_atm = max(float(np.interp(0.0, k, w)), 1e-4)
    b = max(float(np.std(w) / (np.std(k) + 1e-8)), 1e-4)
    sigma = max(float(np.std(k) * 0.5), 1e-3)
    return np.array([max(w_atm * 0.8, 1e-4), min(b, 1.0), 0.0, 0.0, sigma])


def _approximate_hessian_inv(
    jac: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Approximate covariance matrix from Jacobian: (J^T J)^{-1}."""
    try:
        jtj = jac.T @ jac
        return np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
