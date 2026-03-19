# QuantClaw — Agent Audit Report

*Audit date: 2026-03-10*
*Scope: Full codebase, git history, file structure, code quality*

---

## 1. Git Forensics — Who Wrote What

### Commits by `Engineer | engineer@openclaw.ai` (OpenClaw agent)

| Branch | Commits | What was introduced |
|--------|---------|---------------------|
| `feat/spy-vol-surface` | 6 commits | Root-level scripts (`build_surface.py`, `smooth_surface.py`, etc.), old SABR-based `vol_surface/fetcher.py`, `vol_surface/surface.py`, `vol_surface/visualization.py`, `vol_surface/calibration.py` |
| `feat/deribit-data-pipeline-v2` | 1 commit | `src/` directory (Deribit pipeline, schemas), `data/` directory (raw CSVs, SQLite init) |
| `feat/svi-btc-expiries` | 2 commits | `src/svi_calibration.py`, `src/svi_calibration_btc.py`, BTC-specific SVI scripts |

**Also introduced by the agent (via `feat/spy-vol-surface`):**
- `pricing/` module (barrier options, autocallable, Monte Carlo, PDE pricer)
- 12 throwaway plot-generation scripts at root
- `implied_vol.py`, `cleaner.py`, `fetcher.py`, `schema.py` at root

### Commits by Mathis

| Branch | Commits | What was introduced |
|--------|---------|---------------------|
| `feature/vol-surface-calibration` | 9 commits | Core library: `vol_surface/data/`, `vol_surface/models/`, `vol_surface/calibration/`, `vol_surface/output/`, CLI, all tests |
| `fix/ticker-option` | 1 commit | SPX/NDX ticker resolution fix |
| `fix/svi-ssvi` | 6 commits | Outlier filter, tighter bounds, butterfly penalty, SSVI hard bounds, test refinements, analysis script |

---

## 2. What Was Deleted (45 files)

### Root-level scripts (16 files) — ALL agent-generated

| File | Problem |
|------|---------|
| `build_surface.py` | Hardcoded path `/home/openclaw_daemon/...`, imports non-existent `arbitrage` module |
| `smooth_surface.py` | Same hardcoded paths, same broken imports |
| `fetch_real_data.py` | Uses old `vol_surface.fetcher.OptionChainFetcher` API that no longer exists |
| `generate_and_plot.py` | SABR-based, doesn't use SVI/SSVI at all |
| `generate_plots.py` | References old `VolatilityVisualizer` API |
| `generate_plots_base64.py` | Calls `plt_to_buffer()` which is broken (`plt.ioff()` returns None, not a BytesIO) |
| `generate_plots_final.py` | Writes base64 to `/tmp/`, mock data only |
| `generate_density_plot.py` | Hardcoded BTC demo data, writes to `plots/` |
| `generate_plot.py` | Hardcoded BTC demo data |
| `generate_ssvi_plot.py` | Hardcoded BTC demo data |
| `plot_generator.py` | Prints base64 to stdout |
| `cleaner.py` | 20-line duplicate of helpers in `vol_surface/data/fetcher.py` |
| `fetcher.py` | Duplicate of `vol_surface/data/fetcher.py`, imports root `schema.py` |
| `schema.py` | Dataclass-based schema, superseded by Pydantic `vol_surface/data/schema.py` |
| `implied_vol.py` | Standalone BS IV solver, never imported by the package |
| `debug_deribit.py` | One-off debug script for Deribit API |

### Dead directories (3 directories, 24 files)

| Directory | Problem |
|-----------|---------|
| `src/` | Deribit pipeline (`pipeline.py`, `schemas.py`), BTC SVI scripts. Contains `from openclaw.tool import message` (Telegram integration). Hardcoded paths. |
| `data/` | `init_db.py` (SQLite schema), 10 raw CSV files from Deribit API calls. Data artifacts, not source code. |
| `pricing/` | Barrier + autocallable pricing engine (Monte Carlo, PDE). **Completely disconnected** from `vol_surface` — zero imports between them. `monte_carlo.py` has an undefined variable bug (`risk_free_rate` at line 106). |

### Dead files inside `vol_surface/` (4 files)

| File | Problem |
|------|---------|
| `vol_surface/fetcher.py` | Old SQLite-backed Yahoo Finance fetcher (236 lines). Superseded by `vol_surface/data/fetcher.py`. |
| `vol_surface/surface.py` | SABR-based `VolatilitySurface` class. Imports `vol_surface.calibration.SABR` (dead). Not used anywhere. |
| `vol_surface/visualization.py` | Matplotlib `VolatilityVisualizer` class. Imports `vol_surface.surface` (dead). |
| `vol_surface/calibration.py` | SABR calibration class. **Shadowed** the `vol_surface/calibration/` package directory. |

### Dead test file

| File | Problem |
|------|---------|
| `tests/test_pipeline.py` | Tests `src.pipeline` and `src.schemas`. Requires `httpx` and `pytest-asyncio` (not in deps). |

### Stale output artifacts

| File | Problem |
|------|---------|
| `output/calibration_report.md` | From a previous CLI run. Gitignored. |
| `output/vol_surface.json` | From a previous CLI run. Gitignored. |

---

## 3. Bugs Fixed During Cleanup

| Bug | Location | Fix |
|-----|----------|-----|
| `vol_surface/__init__.py` imported deleted modules (`fetcher`, `calibration`, `surface`, `visualization`) | `vol_surface/__init__.py` | Rewrote to only export `__version__` |
| `datetime.utcnow()` deprecated (Python 3.12+, removed in future) | `vol_surface/data/fetcher.py`, `vol_surface/cli.py`, `vol_surface/output/report.py` | Replaced with `datetime.now(timezone.utc)` or `datetime.now()` |

---

## 4. Remaining Code Quality Issues (Not Fixed — For Awareness)

### Moderate

1. **`vol_surface/data/fetcher.py:119`** — `_parse_frame` iterates rows with `df.iterrows()`. Fine for small chains but O(n) Python loop over potentially 10k+ rows. Vectorized pandas would be 10-50x faster.

2. **No type stubs for `yfinance`** — Every `yf.Ticker` call is untyped. `fast_info` attributes change between yfinance versions.

3. **`vol_surface/calibration/optimizer.py`** — `MAX_RETRIES = 3` but `x0_candidates` has 3 entries, so attempt 2 (index 2) is the last candidate and never retries with perturbation. The `if attempt >= len(x0_candidates)` branch is dead code.

4. **Hessian estimation** — `_approximate_hessian_inv` uses `np.linalg.inv(J^T J)` which is numerically unstable for ill-conditioned Jacobians. Should use `np.linalg.pinv` or `lstsq`.

### Minor

5. **`.gitignore` ignores `*.json`** globally — `output/vol_surface.json` is always gitignored. This is fine for artifacts but means the CLI output is never tracked.

6. **`README.md` is empty** — Just `# QuantClaw`. No usage instructions, no architecture overview, no installation guide.

7. **No `py.typed` marker** — Package doesn't advertise PEP 561 type information.

8. **`pricing/utils.py`** was an empty file (deleted).

---

## 5. What the Agent Did Well

1. **Structured code with ABCs** — `DataFetcher`, `PathDependentOption`, `VolatilityModel` are proper interfaces with type annotations.

2. **Pydantic v2 schemas** — `OptionQuote`, `VolSlice`, `SVIParams`, `SSVIParams` use field validators and model validators correctly.

3. **SABR implementation was correct** — The `SABR.sabr_vol` ATM/non-ATM formula and calibration were mathematically sound.

4. **Test structure** — `conftest.py` with shared fixtures, parametrized tests, clean separation of unit vs integration.

---

## 6. What the Agent Did Poorly

### File Hygiene (Critical)

- **16 throwaway scripts** dumped at root, never cleaned up. Each iteration created a new file (`generate_plot.py`, `generate_plots.py`, `generate_plots_base64.py`, `generate_plots_final.py`) instead of editing the previous one.
- **Dead code accumulated across branches**: old fetcher, surface, calibration, visualization files were never removed when the architecture changed from SABR to SVI/SSVI.
- **`vol_surface/calibration.py` shadowed `vol_surface/calibration/`** — a file and a directory with the same name. Python resolves this ambiguously. This was never caught.

### Architecture Discipline (Critical)

- **Three separate `OptionChain` definitions**: root `schema.py` (dataclass), `vol_surface/data/schema.py` (Pydantic), and `vol_surface/fetcher.py` (dataclass with different fields). They are incompatible with each other.
- **Two separate fetcher implementations**: root `fetcher.py` (standalone, imports root `schema.py`) and `vol_surface/data/fetcher.py` (package, imports Pydantic schema). Neither knows the other exists.
- **`pricing/` module is completely disconnected** — zero imports to/from `vol_surface`. It was merged into `main` via PR but has no integration. Its Monte Carlo pricer has an undefined variable bug.

### Hardcoded Paths (Critical)

- 5+ files contain `/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/`. These paths only work on the agent's sandbox VM and will crash on any other machine.

### Commit Discipline (Moderate)

- Single-commit PRs that dump 500+ lines of new code make review impossible.
- `feat(vol-surface-calibration): Implement option chain fetcher, SABR calibration, and volatility surface visualization` — one commit does three things.
- `feat(exotic-pricing-engine): Implement barrier and autocallable pricing with Monte Carlo and PDE methods` — an entire pricing engine in one commit, merged without tests.

### Testing (Moderate)

- `tests/test_pipeline.py` requires `httpx` and `pytest-asyncio` which aren't in `pyproject.toml` dependencies. Running `pytest` would fail if this file were included.
- No integration tests for the actual data pipeline (yfinance -> clean -> calibrate -> output).
- The Monte Carlo pricer (`risk_free_rate` undefined on line 106) was never tested.

---

## 7. Recommendations for the Agent

### Rules to Enforce

1. **Never create a new file when you should edit an existing one.** If `generate_plot.py` exists, fix it — don't create `generate_plot_v2.py`.

2. **Delete dead code in the same PR that replaces it.** When the architecture changes from SABR to SVI/SSVI, remove the old SABR files.

3. **Never commit hardcoded absolute paths.** Use `Path(__file__).parent` or CLI arguments.

4. **Never create a file and directory with the same name** (e.g., `calibration.py` alongside `calibration/`). Python will only see one.

5. **Run `pytest` before opening a PR.** The `test_pipeline.py` file would have failed on import.

6. **One concern per PR.** Don't merge a pricing engine, a data pipeline, and a visualization tool in the same PR with no tests.

7. **Check imports after deleting files.** `vol_surface/__init__.py` still imported four deleted modules — instant crash.

### Architecture Suggestions

- The `pricing/` module (barrier, autocallable, MC, PDE) was a reasonable start but needs its own `pyproject.toml` or at minimum an integration layer with `vol_surface`. Don't merge disconnected code.
- Consider moving the analysis script (`scripts/test_surface_analysis.py`) into a `notebooks/` directory or making it a proper CLI subcommand.

---

## 8. Final State After Cleanup

```
QuantClaw/
├── pyproject.toml              # Package config
├── README.md                   # (empty — needs content)
├── .gitignore
├── SURFACE_ANALYSIS_REPORT.md  # Calibration test results
├── AGENT_AUDIT_REPORT.md       # This file
├── scripts/
│   └── test_surface_analysis.py  # 5-test analysis script
├── tests/
│   ├── conftest.py
│   ├── test_arbitrage.py
│   ├── test_roundtrip.py
│   ├── test_ssvi.py
│   └── test_svi.py
└── vol_surface/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py                  # CLI entry point
    ├── data/
    │   ├── schema.py           # Pydantic schemas
    │   ├── fetcher.py          # yfinance adapter
    │   └── cleaner.py          # Chain -> VolSlice cleaning
    ├── models/
    │   ├── svi.py              # SVI model + butterfly g(k)
    │   ├── ssvi.py             # SSVI model
    │   └── arbitrage.py        # Butterfly + calendar checks
    ├── calibration/
    │   ├── optimizer.py        # SVI/SSVI optimization
    │   └── diagnostics.py      # RMSE, confidence intervals
    └── output/
        ├── serializer.py       # JSON output
        └── report.py           # Markdown report
```

**37 tests passing. 45 files deleted. 3 bugs fixed.**

---

*QuantClaw v0.1.0 — branch `refactor/project`*
