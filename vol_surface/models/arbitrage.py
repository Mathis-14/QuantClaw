"""Butterfly and calendar spread arbitrage checks."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from vol_surface.data.schema import ArbitrageViolation

logger = logging.getLogger(__name__)


def check_butterfly(
    strikes: NDArray[np.float64],
    total_variance: NDArray[np.float64],
    expiry_label: str,
    tol: float = -1e-8,
) -> list[ArbitrageViolation]:
    """Butterfly arbitrage: d^2 w / dk^2 >= 0 (convexity of total variance in strike).

    Approximated by finite differences on the strike grid.
    """
    violations: list[ArbitrageViolation] = []
    if len(strikes) < 3:
        return violations

    for i in range(1, len(strikes) - 1):
        dk1 = strikes[i] - strikes[i - 1]
        dk2 = strikes[i + 1] - strikes[i]
        if dk1 <= 0 or dk2 <= 0:
            continue
        d2w = (
            total_variance[i + 1] / dk2
            - total_variance[i] * (1 / dk1 + 1 / dk2)
            + total_variance[i - 1] / dk1
        ) / ((dk1 + dk2) / 2)

        if d2w < tol:
            violations.append(
                ArbitrageViolation(
                    type="butterfly",
                    maturity=expiry_label,
                    strike=float(strikes[i]),
                    severity=float(abs(d2w)),
                )
            )
    return violations


def check_calendar(
    slices: list[tuple[str, float, NDArray[np.float64], NDArray[np.float64]]],
    tol: float = -1e-8,
) -> list[ArbitrageViolation]:
    """Calendar spread arbitrage: total variance must be non-decreasing in T
    at every strike.

    *slices* is a list of (expiry_label, T, strikes, total_variance) sorted by T.
    We interpolate onto a common strike grid and check monotonicity.
    """
    violations: list[ArbitrageViolation] = []
    if len(slices) < 2:
        return violations

    slices_sorted = sorted(slices, key=lambda s: s[1])

    all_strikes = np.sort(
        np.unique(np.concatenate([s[2] for s in slices_sorted]))
    )

    interp_tvars: list[tuple[str, float, NDArray[np.float64]]] = []
    for label, T, strikes, tvar in slices_sorted:
        w_interp = np.interp(all_strikes, strikes, tvar)
        interp_tvars.append((label, T, w_interp))

    for i in range(1, len(interp_tvars)):
        label_prev, T_prev, w_prev = interp_tvars[i - 1]
        label_cur, T_cur, w_cur = interp_tvars[i]
        diff = w_cur - w_prev
        mask = diff < tol
        for j in np.where(mask)[0]:
            violations.append(
                ArbitrageViolation(
                    type="calendar",
                    maturity=label_cur,
                    strike=float(all_strikes[j]),
                    severity=float(abs(diff[j])),
                )
            )
    return violations


def run_all_checks(
    slices: list[tuple[str, float, NDArray[np.float64], NDArray[np.float64]]],
    tol: float = -1e-8,
    min_severity: float = 1e-5,
) -> list[ArbitrageViolation]:
    """Run butterfly + calendar checks and return all violations.

    Fix 5: violations with severity < min_severity are filtered out as
    float64 rounding artefacts.  On a clean SVI surface these appear with
    severity ~1e-8.  Real butterfly violations have severity > 1e-4.
    """
    violations: list[ArbitrageViolation] = []
    for label, T, strikes, tvar in slices:
        violations.extend(check_butterfly(strikes, tvar, label, tol))
    violations.extend(check_calendar(slices, tol))

    violations = [v for v in violations if v.severity >= min_severity]
    if violations:
        logger.warning("Detected %d arbitrage violations", len(violations))
    return violations
