"""Fit quality diagnostics: RMSE, confidence intervals, violation summaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vol_surface.calibration.optimizer import OptResult
from vol_surface.data.schema import SVIParams, SSVIParams, VolSlice
from vol_surface.models.svi import svi_total_variance
from vol_surface.models.ssvi import ssvi_total_variance


def rmse(residuals: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def svi_slice_rmse(vol_slice: VolSlice, params: SVIParams) -> float:
    """RMSE of SVI fit for a single slice (in total-variance space)."""
    k, w, _, _ = vol_slice.as_arrays()
    w_model = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
    return rmse(w_model - w)


def svi_slice_iv_rmse(vol_slice: VolSlice, params: SVIParams) -> float:
    """RMSE of SVI fit in implied-vol space."""
    k, w, iv, _ = vol_slice.as_arrays()
    w_model = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
    iv_model = np.sqrt(np.maximum(w_model, 1e-10) / vol_slice.T)
    return rmse(iv_model - iv)


def ssvi_surface_rmse(
    slices: list[VolSlice],
    atm_total_vars: list[float],
    params: SSVIParams,
) -> float:
    """RMSE of SSVI fit across all slices (total-variance space)."""
    all_residuals = []
    for vol_slice, theta in zip(slices, atm_total_vars):
        k, w, _, _ = vol_slice.as_arrays()
        w_model = ssvi_total_variance(k, theta, params.rho, params.eta, params.gamma)
        all_residuals.append(w_model - w)
    return rmse(np.concatenate(all_residuals))


def confidence_intervals_95(
    opt_result: OptResult,
    param_names: list[str],
    n_data: int,
) -> dict[str, tuple[float, float]]:
    """95% confidence intervals from Hessian inverse (approximate).

    Uses t-distribution with (n_data - n_params) degrees of freedom.
    Falls back to +/-inf if Hessian is unavailable.
    """
    from scipy.stats import t as t_dist

    n_params = len(param_names)
    dof = max(n_data - n_params, 1)
    t_val = t_dist.ppf(0.975, dof)

    if opt_result.hessian_inv is None:
        return {name: (float("-inf"), float("inf")) for name in param_names}

    s2 = opt_result.cost / max(dof, 1)  # residual variance estimate
    cov = opt_result.hessian_inv * s2
    result = {}
    for i, name in enumerate(param_names):
        if i < cov.shape[0]:
            se = np.sqrt(max(cov[i, i], 0))
            lo = float(opt_result.params[i] - t_val * se)
            hi = float(opt_result.params[i] + t_val * se)
            result[name] = (lo, hi)
        else:
            result[name] = (float("-inf"), float("inf"))
    return result


def fit_quality_report(
    slices: list[VolSlice],
    svi_params_list: list[SVIParams | None],
    ssvi_params: SSVIParams | None,
    atm_total_vars: list[float],
) -> dict:
    """Generate a summary dict of fit quality metrics."""
    report: dict = {"per_slice": [], "surface": {}}

    for vol_slice, svi_p in zip(slices, svi_params_list):
        entry = {"expiry": str(vol_slice.expiry), "T": vol_slice.T, "n_strikes": len(vol_slice.strikes)}
        if svi_p is not None:
            entry["svi_rmse_tvar"] = svi_slice_rmse(vol_slice, svi_p)
            entry["svi_rmse_iv"] = svi_slice_iv_rmse(vol_slice, svi_p)
            entry["svi_params"] = svi_p.model_dump()
        else:
            entry["svi_rmse_tvar"] = None
            entry["svi_rmse_iv"] = None
            entry["status"] = "failed"
        report["per_slice"].append(entry)

    if ssvi_params is not None and atm_total_vars:
        valid_slices = [s for s, p in zip(slices, svi_params_list) if p is not None]
        valid_thetas = [t for t, p in zip(atm_total_vars, svi_params_list) if p is not None]
        report["surface"]["ssvi_rmse_tvar"] = ssvi_surface_rmse(
            valid_slices, valid_thetas, ssvi_params
        )
        report["surface"]["ssvi_params"] = ssvi_params.model_dump()

    return report
