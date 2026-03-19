"""Butterfly and calendar spread arbitrage checks."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from vol_surface.data.schema import ArbitrageViolation

logger = logging.getLogger(__name__)

SliceSpec = tuple[str, float, NDArray[np.float64], NDArray[np.float64]]


def check_butterfly(
    strikes: NDArray[np.float64],
    total_variance: NDArray[np.float64],
    expiry_label: str,
    tol: float = -1e-8,
) -> list[ArbitrageViolation]:
    """Butterfly arbitrage: d^2 w / dK^2 >= 0 (convexity in strike).

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
    slices: list[SliceSpec],
    tol: float = -1e-8,
) -> list[ArbitrageViolation]:
    """Calendar spread arbitrage: total variance must be non-decreasing in T.

    Interpolates onto a common strike grid and checks monotonicity.
    """
    violations: list[ArbitrageViolation] = []
    if len(slices) < 2:
        return violations

    slices_sorted = sorted(slices, key=lambda s: s[1])
    all_strikes = np.sort(np.unique(np.concatenate([s[2] for s in slices_sorted])))

    interp: list[tuple[str, float, NDArray[np.float64]]] = []
    for label, T, strikes, tvar in slices_sorted:
        interp.append((label, T, np.interp(all_strikes, strikes, tvar)))

    for i in range(1, len(interp)):
        _, _, w_prev = interp[i - 1]
        label_cur, _, w_cur = interp[i]
        diff = w_cur - w_prev
        for j in np.where(diff < tol)[0]:
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
    slices: list[SliceSpec],
    tol: float = -1e-8,
    min_severity: float = 1e-5,
) -> list[ArbitrageViolation]:
    """Run butterfly + calendar checks and return all violations.

    Violations with severity < *min_severity* are filtered out as float64
    rounding artefacts (clean SVI surfaces produce ~1e-8 noise).
    """
    violations: list[ArbitrageViolation] = []
    for label, T, strikes, tvar in slices:
        violations.extend(check_butterfly(strikes, tvar, label, tol))
    violations.extend(check_calendar(slices, tol))

    violations = [v for v in violations if v.severity >= min_severity]
    if violations:
        logger.warning("Detected %d arbitrage violations", len(violations))
    return violations
