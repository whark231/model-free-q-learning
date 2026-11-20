"""Oracle runs and checks 3 and 4 at every cost setting (William).

In plain terms: solve the oracle for the four portfolio sizes and the two $1B
mechanism illustrations, check that the answer does not depend on the number
of simulated paths (check 3) or on the lattice spacing (check 4), and compare
the oracle's policy with the simple reference policies inside the oracle's own
model. Writes results/tc-drift/oracle/ (docs/methods/TC_DRIFT_DESIGN.md §5, §8).

    python -m src.tc_drift.oracle_runs              # about an hour on William's laptop
    python -m src.tc_drift.oracle_runs --quick      # smoke test, M = 40, no fine lattice

How it works:
- Paths: M_CHECK = 4,000 blocks from np.random.default_rng(spec.ORACLE_PATH_SEED);
  the production oracle uses the first M = 2,000, which are exactly the
  2,000 that seed draws on its own (asserted).
- One scoring per (M, lattice) serves every cost setting and the myopic
  reference, because the scoring does not depend on the costs or the
  discount (oracle.score, oracle.solve_scored).
- Check 3 (restated on the policy, decided William 2026-11-17, after the
  V-based form failed at $1B, so post hoc): score the M = 2,000 and the
  M = 4,000 policy tables with value_of in an independent model, 4,000 paths
  from CHECK3_FRESH_SEED that neither policy was fitted on. The metric is the
  mean over the 1,326 2% starts of the annualized difference (M = 4,000
  policy minus M = 2,000 policy); it passes if its size is under 0.5 bps/yr,
  and it can fail in either direction. Also reported: the same comparison in
  the M = 4,000 model (which shares its first 2,000 paths with the M = 2,000
  policy, so its bias has no fixed sign), the V-based form, and the
  difference as a share of the oracle's lead over always rebalancing. Where
  that lead is itself under 0.5 bps/yr a pass says little, because even
  always rebalancing would pass ("informative" is False).
- Check 4: the 0.5% lattice against the 1% lattice, both at M = 2,000; the
  mean over the 2% starts of the annualized V; tolerance 0.5 bps/yr.

Everything here is in the oracle's own model (the fixed paths it was solved
on). Its in-sample V is biased upward, so its leads over the reference
policies are too; the evaluation replays (Albert's evaluate.py) are the
reported gaps. The in-model figures are for the checks, the expected
directions of §8, and the size of the oracle-minus-always-rebalance gap D10
needs.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from src.tc_drift import calibration, env, oracle, spec

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "tc-drift" / "oracle"
M_ORACLE = 2_000
M_CHECK = 4_000
FINE_INCREMENT = 0.005
TOLERANCE_BPS = 0.5          # checks 3 and 4, decided William 2026-11-15 before either ran
HORIZON = 90                 # D10: blocks per evaluation path
NEAR_OPTIMUM_L1 = 0.04       # the "first look" neighbourhood of the best target (§5)
CHECK3_FRESH_SEED = (20261016, 4)  # SeedSequence entropy for check 3's independent model; ORACLE_PATH_SEED is (20261016, 3)
LEVELS = ("zero", "aum_100m", "aum_1b", "aum_10b")
MECHANISMS = ("kappa_only", "lambda_only")


# ---------------------------------------------------------------------------
# Reference policies and in-model statistics
# ---------------------------------------------------------------------------

def best_target(market: spec.Market, gamma: float) -> int:
    """The 2% target with the highest constant-weight utility, as in the D10 pilot."""
    t = spec.target_grid()
    u = t @ market.mu - 0.5 * gamma * np.einsum("ai,ij,aj->a", t, market.sigma, t)
    return int(np.argmax(u))


def rebalance_table(sol: oracle.OracleSolution, best: int) -> np.ndarray:
    """Always trade to the best target; HOLD where already on it (the same move at no cost)."""
    table = np.full(len(sol.V), best, dtype=np.int64)
    table[sol.target_index[best]] = spec.HOLD
    return table


def never_trade_table(sol: oracle.OracleSolution) -> np.ndarray:
    return np.full(len(sol.V), spec.HOLD, dtype=np.int64)


def trade_once_value(sol: oracle.OracleSolution, best: int) -> np.ndarray:
    """Value of trading to the best target in block 1 and holding forever after,
    from every lattice state: the post-trade point's buy-and-hold value minus the
    one cost. It uses the block index, so it is not a stationary table (C10)."""
    hold_forever = sol.value_of(never_trade_table(sol))
    return hold_forever[sol.target_index[best]] - sol.cost[:, best] / sol.block_length


def path_statistics(sol: oracle.OracleSolution, table: np.ndarray, start_index: np.ndarray,
                    horizon: int = HORIZON) -> dict:
    """Exact expected trading statistics in the oracle's model, from starts drawn
    uniformly from start_index, over `horizon` blocks: the state distribution is
    pushed through the policy and the transition matrix block by block.

    hold_fraction: share of blocks with HOLD. turnover_per_block: mean one-way
    turnover, ||t - s||_1 / 2. trades_per_path: expected trades in `horizon`
    blocks. mean_trade_size: one-way turnover per trade."""
    n = len(sol.V)
    states = spec.lattice(sol.lattice_increment)
    hold = table == spec.HOLD
    trade = np.where(hold, 0, table)
    landing = np.where(hold, np.arange(n), sol.target_index[trade])
    one_way = np.where(hold, 0.0, 0.5 * np.abs(states[landing] - states).sum(axis=1))
    PT = sol.transition.T.tocsr()
    dist = np.bincount(start_index, minlength=n).astype(float)
    dist /= dist.sum()
    held = turnover = 0.0
    for _ in range(horizon):
        held += dist[hold].sum()
        turnover += dist @ one_way
        dist = PT @ np.bincount(landing, weights=dist, minlength=n)
    trades = horizon - held
    return {"hold_fraction": held / horizon, "turnover_per_block": turnover / horizon,
            "trades_per_path": trades, "mean_trade_size": turnover / trades if trades > 0 else 0.0}


def _mean_bps(x: np.ndarray) -> float:
    return float(spec.annualized_bps(x).mean())


def _max_abs_bps(x: np.ndarray) -> float:
    return float(np.abs(spec.annualized_bps(x)).max())


def summarize(sol: oracle.OracleSolution, myopic: oracle.OracleSolution, best: int) -> dict:
    """Values and trading statistics of the oracle, myopic and the reference
    policies, in the oracle's model, from the 2% starts."""
    starts = sol.target_index
    states = spec.lattice(sol.lattice_increment)
    rebal = rebalance_table(sol, best)
    myopic_value = sol.value_of(myopic.policy_table)          # chosen at rho = 0, scored at the solution's rho
    near = np.abs(states - states[starts[best]]).sum(axis=1) <= NEAR_OPTIMUM_L1 + 1e-12
    return {
        "oracle_value_bps": _mean_bps(sol.V[starts]),
        "lead_over_always_rebalance_bps": _mean_bps((sol.V - sol.value_of(rebal))[starts]),
        "lead_over_trade_once_bps": _mean_bps((sol.V - trade_once_value(sol, best))[starts]),
        "lead_over_never_trade_bps": _mean_bps((sol.V - sol.value_of(never_trade_table(sol)))[starts]),
        "lead_over_myopic_bps": _mean_bps((sol.V - myopic_value)[starts]),
        "hold_share_near_best_target": float((sol.policy_table[near] == spec.HOLD).mean()),
        "cv_correction_bps": {"mean": _mean_bps(sol.cv_correction / (1 - sol.discount)),
                              "max_abs": _max_abs_bps(sol.cv_correction / (1 - sol.discount))},
        "n_iterations": sol.n_iterations,
        "oracle": path_statistics(sol, sol.policy_table, starts),
        "myopic": path_statistics(sol, myopic.policy_table, starts),
        "always_rebalance": path_statistics(sol, rebal, starts),
    }


def check3(sol_m: oracle.OracleSolution, sol_2m: oracle.OracleSolution, fresh: oracle.OracleSolution) -> dict:
    """The M and 2M policy tables scored in an independent model `fresh`
    (primary), and in the 2M model; plus the V-based form."""
    starts = sol_2m.target_index
    independent = (fresh.value_of(sol_2m.policy_table) - fresh.value_of(sol_m.policy_table))[starts]
    shortfall = (sol_2m.V - sol_2m.value_of(sol_m.policy_table))[starts]
    reverse = (sol_m.V - sol_m.value_of(sol_2m.policy_table))[starts]
    v_change = (sol_2m.V - sol_m.V)[starts]
    return {
        "policy_difference_in_independent_model_bps": _mean_bps(independent),
        "policy_difference_max_bps": _max_abs_bps(independent),
        "passes": bool(abs(_mean_bps(independent)) < TOLERANCE_BPS),
        "policy_shortfall_in_larger_model_bps": _mean_bps(shortfall),
        "reverse_shortfall_in_smaller_model_bps": _mean_bps(reverse),
        "same_action_share_at_2pct_starts": float((sol_m.policy_table[starts] == sol_2m.policy_table[starts]).mean()),
        "v_change_bps": _mean_bps(v_change),
        "v_change_max_bps": _max_abs_bps(v_change),
    }


def check4(sol: oracle.OracleSolution, fine: oracle.OracleSolution) -> dict:
    change = fine.V[fine.target_index] - sol.V[sol.target_index]
    return {"v_change_bps": _mean_bps(change), "v_change_max_bps": _max_abs_bps(change),
            "passes": bool(abs(_mean_bps(change)) < TOLERANCE_BPS),
            "same_action_share_at_2pct_starts": float(
                (sol.policy_table[sol.target_index] == fine.policy_table[fine.target_index]).mean())}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _provenance() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    files = ["src/tc_drift/oracle_runs.py", "src/tc_drift/oracle.py", "src/tc_drift/env.py", "src/tc_drift/spec.py",
             "src/tc_drift/calibration.py"]
    return {"commit": run("rev-parse", "HEAD"), "src_dirty": bool(run("status", "--porcelain", "--", "src")),
            "sha256": {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in files}}


def main(out_dir: Path = OUT_DIR, m: int = M_ORACLE, fine: bool = True,
         increment: float = spec.LATTICE_INCREMENT, fine_increment: float = FINE_INCREMENT) -> dict:
    """increment and fine_increment exist for the tests' small runs; the
    reported run uses the 1% and 0.5% lattices."""
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    market = env.load_market()
    settings = {**calibration.cost_settings(), **calibration.mechanism_settings()}
    gamma = settings["aum_1b"].gamma
    assert all(s.gamma == gamma for s in settings.values())
    best = best_target(market, gamma)

    paths_big = env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), 2 * m)
    paths = env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), m)
    if not np.array_equal(paths, paths_big[:m]):
        raise RuntimeError("the first M of the 2M paths differ from the M paths drawn alone")
    paths_fresh = env.draw_returns(market, np.random.default_rng(CHECK3_FRESH_SEED), 2 * m)

    scored = {}
    for key, p, inc in [("m", paths, increment), ("2m", paths_big, increment), ("fresh", paths_fresh, increment)] + \
                       ([("fine", paths, fine_increment)] if fine else []):
        scored[key] = oracle.score(market, gamma, p, block_outcome=env.block_outcome, lattice_increment=inc)
        log(f"scored {key}: {p.shape[0]} paths, lattice {inc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, s in settings.items():
        sol = oracle.solve_scored(scored["m"], s)
        myopic = oracle.solve_scored(scored["m"], s, discount=0.0)
        entry = {"kappa": s.kappa, "lam": s.lam, "gamma": s.gamma, **summarize(sol, myopic, best)}
        entry["check3"] = check3(sol, oracle.solve_scored(scored["2m"], s), oracle.solve_scored(scored["fresh"], s))
        lead = entry["lead_over_always_rebalance_bps"]
        entry["check3"]["informative"] = bool(lead > TOLERANCE_BPS)   # else always rebalancing would pass too
        entry["check3"]["difference_share_of_lead"] = (
            abs(entry["check3"]["policy_difference_in_independent_model_bps"]) / lead if lead > 0 else None)
        if fine:
            entry["check4"] = check4(sol, oracle.solve_scored(scored["fine"], s))
        results[name] = entry
        np.savez_compressed(out_dir / f"{name}.npz", V=sol.V, G=sol.G, policy_table=sol.policy_table,
                            r_hat=sol.r_hat, cv_correction=sol.cv_correction, U=sol.U,
                            myopic_policy_table=myopic.policy_table)
        c3 = entry["check3"]
        log(f"{name}: V {entry['oracle_value_bps']:.3f}, lead over rebalance {entry['lead_over_always_rebalance_bps']:.3f}, "
            f"hold {entry['oracle']['hold_fraction']:.3f}; check 3 {c3['policy_difference_in_independent_model_bps']:+.3f} "
            f"(V-based {c3['v_change_bps']:+.3f})"
            + (f"; check 4 {entry['check4']['v_change_bps']:+.3f}" if fine else ""))

    summary = {
        "what": "TC drift oracle runs and checks 3 and 4, in the oracle's own model (TC_DRIFT_DESIGN.md §5, §8). "
                "Post hoc relative to baseline-v1. The oracle's values and its leads over other policies are "
                "in-sample, so biased upward; the reported gaps come from evaluate.py replays on fresh paths. "
                "never_trade here holds the start forever; trade_once trades to the best target in block 1, "
                "then holds (evaluate.never_trade_policy is the latter).",
        "units": "bps/yr: spec.annualized_bps of a mean over the 1,326 2% starts (infinite horizon); "
                 "statistics over a 90-block horizon from those starts",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": {"oracle_path_seed": spec.ORACLE_PATH_SEED, "check3_fresh_seed": CHECK3_FRESH_SEED},
        "paths": {"m": m, "check3_m": 2 * m, "block_length": spec.BLOCK_LENGTH},
        "lattice": {"increment": increment, "check4_increment": fine_increment if fine else None},
        "discount": spec.DISCOUNT, "tolerance_bps": TOLERANCE_BPS, "horizon_blocks": HORIZON,
        "best_target": spec.target_grid()[best].round(4).tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "provenance": _provenance(), "runtime_seconds": round(time.time() - t0),
        "settings": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    log(f"wrote {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: M = 40, no fine lattice, scratch output")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.quick:
        main(args.out or OUT_DIR.parent / "oracle-quick", m=40, fine=False)
    else:
        main(args.out or OUT_DIR)
