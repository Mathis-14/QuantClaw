"""CLI entry point: python -m vol_surface --ticker SPX --output ./output"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from vol_surface.calibration.diagnostics import (
    confidence_intervals_95,
    fit_quality_report,
    svi_slice_rmse,
    ssvi_surface_rmse,
)
from vol_surface.calibration.optimizer import calibrate_ssvi_surface, calibrate_svi_slice
from vol_surface.data.cleaner import clean_chain
from vol_surface.data.fetcher import YFinanceFetcher, resolve_tickers
from vol_surface.data.schema import (
    SSVIParams,
    SVIParams,
    SliceResult,
    VolSurface,
)
from vol_surface.models.arbitrage import run_all_checks
from vol_surface.models.svi import svi_total_variance
from vol_surface.output.report import save_report
from vol_surface.output.serializer import save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vol_surface",
        description="Calibrate equity index volatility surface (SVI + SSVI)",
    )
    p.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying ticker (SPX, SPY, NDX, QQQ; default: SPY)",
    )
    p.add_argument("--output", default="./output", help="Output directory")
    p.add_argument("--min-strikes", type=int, default=5, help="Minimum strikes per slice")
    p.add_argument("--moneyness-band", type=float, default=0.5, help="|log(K/F)| cutoff")
    p.add_argument("--skip-ssvi", action="store_true", help="Skip SSVI surface fit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Fetching option chain for %s...", args.ticker)
    fetcher = YFinanceFetcher()
    chain = fetcher.fetch(args.ticker)

    logger.info("Cleaning data...")
    slices = clean_chain(
        chain,
        min_strikes=args.min_strikes,
        moneyness_band=args.moneyness_band,
    )
    if not slices:
        logger.error("No valid slices after cleaning. Aborting.")
        return 1

    logger.info("Calibrating SVI per slice (%d slices)...", len(slices))
    svi_results: list[SVIParams | None] = []
    opt_results = []
    slice_results: list[SliceResult] = []

    for vol_slice in slices:
        svi_p, opt = calibrate_svi_slice(vol_slice)
        svi_results.append(svi_p)
        opt_results.append(opt)

        sr = SliceResult(
            expiry=str(vol_slice.expiry),
            T=vol_slice.T,
            svi_params=svi_p,
            slice_rmse=svi_slice_rmse(vol_slice, svi_p) if svi_p else None,
            status="ok" if svi_p else "failed",
            message=opt.message if not opt.success else "",
        )
        slice_results.append(sr)
        status = "OK" if svi_p else "FAILED"
        logger.info("  %s (T=%.3f): %s  cost=%.2e", vol_slice.expiry, vol_slice.T, status, opt.cost)

    # ATM total variances from SVI fits
    atm_total_vars: list[float] = []
    for svi_p, vol_slice in zip(svi_results, slices):
        if svi_p is not None:
            w_atm = float(svi_total_variance(np.array([0.0]), svi_p.a, svi_p.b, svi_p.rho, svi_p.m, svi_p.sigma)[0])
            atm_total_vars.append(max(w_atm, 1e-8))
        else:
            atm_total_vars.append(float(np.mean(vol_slice.total_variance)))

    # SSVI calibration
    ssvi_params: SSVIParams | None = None

    surface_rmse_val = None
    ssvi_ci: dict[str, tuple[float, float]] = {}

    if not args.skip_ssvi:
        valid_slices = [s for s, p in zip(slices, svi_results) if p is not None]
        valid_thetas = [t for t, p in zip(atm_total_vars, svi_results) if p is not None]
        if len(valid_slices) >= 2:
            logger.info("Calibrating SSVI surface...")
            ssvi_params, ssvi_opt_result = calibrate_ssvi_surface(valid_slices, valid_thetas)

            if ssvi_params:
                surface_rmse_val = ssvi_surface_rmse(valid_slices, valid_thetas, ssvi_params)
                logger.info("SSVI: rho=%.4f eta=%.4f gamma=%.4f  RMSE=%.6f",
                            ssvi_params.rho, ssvi_params.eta, ssvi_params.gamma, surface_rmse_val)
                n_data = sum(len(s.strikes) for s in valid_slices)
                ssvi_ci = confidence_intervals_95(ssvi_opt_result, ["rho", "eta", "gamma"], n_data)
            else:
                logger.warning("SSVI calibration failed")
        else:
            logger.warning("Need >= 2 valid slices for SSVI; skipping")

    # Arbitrage checks
    logger.info("Running arbitrage checks...")
    arb_input = []
    for vol_slice, svi_p in zip(slices, svi_results):
        if svi_p is not None:
            k_arr = np.array(vol_slice.log_moneyness)
            w_model = svi_total_variance(k_arr, svi_p.a, svi_p.b, svi_p.rho, svi_p.m, svi_p.sigma)
            arb_input.append((str(vol_slice.expiry), vol_slice.T, np.array(vol_slice.strikes), w_model))

    violations = run_all_checks(arb_input)
    if violations:
        logger.warning("%d arbitrage violations detected!", len(violations))
        for v in violations[:10]:
            logger.warning("  %s at %s K=%.2f severity=%.6f", v.type, v.maturity, v.strike, v.severity)

    # Build output
    spot_source, options_source = resolve_tickers(args.ticker)

    vol_surface = VolSurface(
        timestamp=datetime.utcnow().isoformat(),
        ticker=args.ticker,
        spot=chain.spot,
        spot_source=spot_source,
        options_source=options_source,
        maturities=slice_results,
        ssvi_params=ssvi_params,
        surface_rmse=surface_rmse_val,
        arbitrage_violations=violations,
    )

    diagnostics = fit_quality_report(slices, svi_results, ssvi_params, atm_total_vars)
    ci_dict = {"ssvi": ssvi_ci} if ssvi_ci else None

    out_dir = Path(args.output)
    json_path = save_json(vol_surface, out_dir / "vol_surface.json")
    report_path = save_report(vol_surface, diagnostics, out_dir / "calibration_report.md", ci_dict)

    logger.info("Saved %s", json_path)
    logger.info("Saved %s", report_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
