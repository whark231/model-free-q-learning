"""Budget-only closed-form portfolio (Part III), written to results/benchmarks/.

Run from the repo root:  python -m src.benchmarks.closed_form_table

Uses the Part I calibration (daily mu and Sigma) and
src.benchmarks.mean_variance.closed_form_optimal_weights, which imposes only
sum(w) = 1 and allows short positions. Each solution is checked against an
independent numerical solve (scipy SLSQP with the budget constraint only)
and against the first-order condition before anything is written.
Utilities are annualized for reporting only (x 252, in bps).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.benchmarks.mean_variance import closed_form_optimal_weights
from src.data.universe import TICKERS
from src.environment import load_calibration

GAMMAS = (3, 5, 8)
OUT_DIR = Path("results/benchmarks")
BPS_PER_YEAR = 252 * 1e4


def utility(w, mu, sigma, gamma):
    return float(w @ mu - 0.5 * gamma * w @ sigma @ w)


def numerical_budget_only(mu, sigma, gamma):
    n = mu.shape[0]
    res = minimize(
        lambda w: -utility(w, mu, sigma, gamma),
        np.full(n, 1.0 / n),
        jac=lambda w: -(mu - gamma * sigma @ w),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"numerical solve failed at gamma={gamma}: {res.message}")
    return res.x


def main():
    mu, sigma = load_calibration()
    rows, checks = [], {}
    for gamma in GAMMAS:
        w = closed_form_optimal_weights(mu, sigma, gamma)
        w_num = numerical_budget_only(mu, sigma, gamma)
        # First-order condition: mu - gamma Sigma w equals the same multiplier for every asset.
        grad = mu - gamma * sigma @ w
        checks[gamma] = {
            "budget_error": float(abs(w.sum() - 1.0)),
            "foc_spread": float(grad.max() - grad.min()),
            "max_abs_diff_vs_numerical": float(np.abs(w - w_num).max()),
        }
        if checks[gamma]["budget_error"] > 1e-10 or checks[gamma]["foc_spread"] > 1e-12:
            raise RuntimeError(f"closed form fails its own conditions at gamma={gamma}: {checks[gamma]}")
        if checks[gamma]["max_abs_diff_vs_numerical"] > 1e-5:
            raise RuntimeError(f"closed form disagrees with numerical solve at gamma={gamma}: {checks[gamma]}")
        rows.append(
            {"gamma": gamma, **dict(zip(TICKERS, w)), "utility_bps_per_year": utility(w, mu, sigma, gamma) * BPS_PER_YEAR}
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "closed_form_budget_only.csv", index=False, float_format="%.6f")
    (OUT_DIR / "closed_form_budget_only_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(table.round(4).to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
