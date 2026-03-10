"""Optimization wrappers for SVI and SSVI calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from vol_surface.data.schema import SVIParams, SSVIParams, VolSlice
from vol_surface.models.svi import (
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
) -> tuple[SVIParams | None, OptResult]:
    """Calibrate SVI to a single maturity slice.

    Returns (SVIParams, OptResult). SVIParams is None if all attempts fail.
    """
    k, w, iv, weights = vol_slice.as_arrays()
    x0 = svi_initial_guess(k, w)
    lower, upper = svi_parameter_bounds()

    def residuals(x: NDArray) -> NDArray:
        w_model = svi_total_variance(k, *x)
        return np.sqrt(weights) * (w_model - w)

    best: OptResult | None = None
    for attempt in range(max_retries):
        x0_perturbed = x0 * (1 + 0.2 * attempt * np.random.default_rng(attempt).standard_normal(5))
        x0_perturbed = np.clip(x0_perturbed, lower, upper)
        x0_perturbed[2] = np.clip(x0_perturbed[2], -0.99, 0.99)  # rho
        x0_perturbed[4] = max(x0_perturbed[4], 1e-5)  # sigma > 0

        try:
            result = least_squares(
                residuals,
                x0_perturbed,
                bounds=(lower, upper),
                method="trf",
                max_nfev=5000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            cost = float(result.cost)
            hess_inv = _approximate_hessian_inv(result.jac)

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
        return None, OptResult(
            params=x0, cost=float("inf"), success=False,
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
    all_theta = []

    for vol_slice, theta in zip(slices, atm_total_vars):
        k, w, iv, weights = vol_slice.as_arrays()
        all_k.append(k)
        all_w.append(w)
        all_weights.append(weights)
        all_theta.append(np.full_like(k, theta))

    k_all = np.concatenate(all_k)
    w_all = np.concatenate(all_w)
    wt_all = np.concatenate(all_weights)
    theta_all = np.concatenate(all_theta)

    x0 = ssvi_initial_guess()
    lower, upper = ssvi_parameter_bounds()

    def objective(x: NDArray) -> float:
        rho, eta, gamma = x
        w_model = np.empty_like(k_all)
        idx = 0
        for ki, theta_i in zip(all_k, atm_total_vars):
            n = len(ki)
            w_model[idx : idx + n] = ssvi_total_variance(ki, theta_i, rho, eta, gamma)
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
            x0_p[0] = np.clip(x0_p[0], -0.99, 0.99)  # rho

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

            hess_inv = None
            if hasattr(result, "hess_inv"):
                hess_inv = np.asarray(result.hess_inv)

            opt = OptResult(
                params=result.x,
                cost=cost,
                success=bool(result.success),
                hessian_inv=hess_inv,
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


def _approximate_hessian_inv(
    jac: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Approximate covariance matrix from Jacobian: (J^T J)^{-1} * s^2."""
    try:
        jtj = jac.T @ jac
        return np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
