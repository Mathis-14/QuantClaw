"""Tests for butterfly and calendar arbitrage checks on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from vol_surface.models.arbitrage import check_butterfly, check_calendar, run_all_checks


class TestButterflyArbitrage:
    """Butterfly (convexity) checks."""

    def test_convex_surface_no_violations(self):
        """A convex total-variance curve should have zero violations."""
        strikes = np.linspace(90, 110, 21)
        # Parabolic (convex) total variance
        tvar = 0.04 + 0.001 * (strikes - 100) ** 2
        violations = check_butterfly(strikes, tvar, "2025-06-20")
        assert len(violations) == 0

    def test_concavity_detected(self):
        """Introduce a concave dip and verify detection."""
        strikes = np.linspace(90, 110, 21)
        tvar = 0.04 + 0.001 * (strikes - 100) ** 2
        # Create a concave region by pushing the middle down
        tvar[10] -= 0.05
        violations = check_butterfly(strikes, tvar, "2025-06-20")
        assert len(violations) > 0
        assert all(v.type == "butterfly" for v in violations)

    def test_too_few_strikes(self):
        """Less than 3 strikes: can't check, return empty."""
        strikes = np.array([100.0, 105.0])
        tvar = np.array([0.04, 0.045])
        violations = check_butterfly(strikes, tvar, "2025-06-20")
        assert len(violations) == 0


class TestCalendarArbitrage:
    """Calendar spread (monotonicity in T) checks."""

    def test_monotone_no_violations(self):
        """Total variance increasing in T should have zero violations."""
        strikes = np.linspace(90, 110, 11)
        slices = [
            ("2025-03-20", 0.1, strikes, 0.01 * np.ones_like(strikes)),
            ("2025-06-20", 0.25, strikes, 0.025 * np.ones_like(strikes)),
            ("2025-12-20", 0.75, strikes, 0.06 * np.ones_like(strikes)),
        ]
        violations = check_calendar(slices)
        assert len(violations) == 0

    def test_inversion_detected(self):
        """Later maturity with lower total variance should be flagged."""
        strikes = np.linspace(90, 110, 11)
        slices = [
            ("2025-03-20", 0.1, strikes, 0.04 * np.ones_like(strikes)),
            ("2025-06-20", 0.25, strikes, 0.02 * np.ones_like(strikes)),  # inverted!
        ]
        violations = check_calendar(slices)
        assert len(violations) > 0
        assert all(v.type == "calendar" for v in violations)

    def test_single_slice_no_check(self):
        strikes = np.array([100.0])
        slices = [("2025-06-20", 0.25, strikes, np.array([0.04]))]
        violations = check_calendar(slices)
        assert len(violations) == 0


class TestRunAllChecks:
    """Integration: combined butterfly + calendar."""

    def test_clean_surface(self):
        strikes = np.linspace(90, 110, 15)
        tvar1 = 0.02 + 0.001 * (strikes - 100) ** 2
        tvar2 = 0.05 + 0.001 * (strikes - 100) ** 2
        slices = [
            ("2025-03-20", 0.1, strikes, tvar1),
            ("2025-06-20", 0.25, strikes, tvar2),
        ]
        violations = run_all_checks(slices)
        assert len(violations) == 0

    def test_violations_from_both(self):
        strikes = np.linspace(90, 110, 15)
        tvar1 = 0.05 + 0.001 * (strikes - 100) ** 2
        tvar1[7] -= 0.1  # butterfly violation
        tvar2 = 0.02 + 0.001 * (strikes - 100) ** 2  # calendar inversion
        slices = [
            ("2025-03-20", 0.1, strikes, tvar1),
            ("2025-06-20", 0.25, strikes, tvar2),
        ]
        violations = run_all_checks(slices)
        types = {v.type for v in violations}
        assert "butterfly" in types
        assert "calendar" in types
