# 03 Environment, benchmarks, grid and reward (working notes)

**TL;DR / In plain terms:** the simulator draws Gaussian daily excess returns from the fixed μ and Σ, the benchmarks are solved from the same μ and Σ, and the grid is built from the asset count and the step alone. I re-derived the benchmarks and the grid counts with separate code and they match. No methods doc covers Part II or the reward, and no committed output shows the budget-only closed-form portfolio.

Sources: `docs/methods/ENVIRONMENT.md`, `src/environment.py`, `src/benchmarks/mean_variance.py`, `src/action_space/grid.py`, `src/rewards.py`.

## Environment (`src/environment.py`, ENVIRONMENT.md)
- r = μ + Lz with L the Cholesky factor of Σ; excess returns; daily; asset order `TICKERS`.
- `SeedPlan`: SeedSequence spawn keys (master, 0, run, 0) training returns, (master, 0, run, 1) agent randomness, (master, 1) evaluation. Reported master seed 20261016 (environment default 20261002 is overridden by `src/config.py`).
- Leakage guard: tests fail if agent code imports the loader, the benchmarks or the data files.
- ENVIRONMENT.md header says "73 tests on 2026-11-06"; the file now has 77. Dated statement, left as is.

## Benchmarks (`src/benchmarks/mean_variance.py`)
- Objective U(w) = w'μ − (γ/2)w'Σw, daily.
- Budget-only closed form w = (1/γ)Σ⁻¹(μ − η1), η from 1'w = 1. The derivation lives in the module docstring only; `test_mean_variance.py` (29 tests) checks it.
- Long-only optimum solved numerically; best grid portfolio by enumeration.
- Independent check (2026-11-20, scratch script): long-only optimum by enumerating all 511 supports and solving each equality-constrained problem. Results in `10_reports_results_inventory.md`. All match `benchmarks.*.exact` in each `summary.json`.
- Budget-only portfolio at γ = 5 (my calculation, no committed output has it): (1.82, −0.88, −0.19, 0.27, −0.02, −0.27, 0.69, −0.13, −0.29), U = 925.9 bps/yr against 530.5 long-only.

## Grid (`src/action_space/grid.py`)
- All nonnegative multiples of the step summing to one. Counts C(u + 8, 8): 495 (25%), 1,287 (20%), 43,758 (10%), 3,108,105 (5%). Verified.
- Inputs are the asset count and the step. No μ, Σ or optimum.
- 5% grid is not trained (count verified only).

## Reward (`src/rewards.py`)
- Per episode of B days: realized mean of portfolio excess return − (γ/2) × realized variance (ddof = 1), daily units.
- E[reward] = U(w) exactly for ddof = 1 with iid draws, so reward and objective agree (Part III "reward matches objective").
- Reward SD at B = 20 about 1.55e-3 daily, about 3,909 bps/yr, 7.45 times U at the optimum (my calculation). This is why the 10,000-episode evaluation sample cannot resolve gaps of a few bps (RUN_ISSUES Finding 5), and why headline gaps use exact μ and Σ.

## Gaps
- No `docs/methods/` file for Part II (derivation, verification, constraint match) or for the reward. README pipeline rows for these have empty documentation cells.
- B = 20 has one line of justification ("a monthly holding period", `src/config.py` docstring).
