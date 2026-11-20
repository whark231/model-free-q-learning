"""Oracle check 5 (continuous replay) and the D10 effect size (Albert).

In plain terms: solve the oracle at each cost level, then play its policy and
the always-rebalance policy on fresh simulated paths, with holdings that are
never rounded. Compare the oracle's lead over always-rebalance in that replay
with the same lead inside the oracle's own model. If the two agree, the
oracle's lattice and its fixed paths do not distort the gaps the evaluation
reports. The same run measures the oracle's lead over always-rebalance, the
effect size the D10 rule (TC_DRIFT_DESIGN.md §13) compares the agent's gap SE
against.

    python -m src.tc_drift.oracle_replay              # all four levels, M = 2,000
    python -m src.tc_drift.oracle_replay --levels aum_1b --m-oracle 4000

What it reports per level (bps/yr, H = 90 blocks, m = 1,000 starts):
- lead_replay: oracle minus always-rebalance, paired, on continuous holdings;
- lead_model: the same lead in the oracle's model, both policies scored over
  the same H blocks with evaluate.model_value at the same start states;
- check5 = lead_replay - lead_model, with the replay's paired SE;
- rounding: the oracle replayed with holdings rounded to the lattice after
  every block, minus the continuous replay (paired; isolates the rounding);
- level_replay_minus_model: the unpaired comparison as worded in §5, for
  reference (its SE is about 12 bps/yr);
- the oracle's and always-rebalance's hold fraction and turnover in the replay.

Seeds: the oracle's fixed paths from spec.ORACLE_PATH_SEED; the replay from
SeedPlan(TUNING_MASTER_SEED, 1).evaluation_rng(), the pilot's stream, so the
reported evaluation stream stays unused until the final runs. The §9 tuning
criterion uses the same stream; that is not leakage (the agent never sees
these paths), and the docs say so.
"""

import argparse
import json
import time

import numpy as np

from src.environment import SeedPlan
from src.tc_drift import config, env, evaluate, oracle, spec
from src.tc_drift.calibration import cost_settings

H = config.H_EVAL
M_EVAL = config.M_EVAL_PILOTS   # check 5 was run at m = 1,000 (results/tc-drift/checks/); kept so it reproduces
M_ORACLE = config.M_ORACLE


def oracle_paths(market, m_oracle: int) -> np.ndarray:
    return env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), m_oracle)


def rounded_replay(solution, starts, paths, settings, discount=spec.DISCOUNT) -> np.ndarray:
    """The oracle's policy with holdings snapped to the nearest lattice point
    after every block, the oracle model's own state dynamics. Per-path value."""
    points = spec.lattice(solution.lattice_increment)
    state = points[spec.nearest_lattice_index(starts, solution.lattice_increment)]
    value = np.zeros(len(starts))
    for t, block in enumerate(paths):
        action = solution.policy(state)
        cost = spec.trade_cost(state, action, settings)
        reward, w_end = env.block_outcome(spec.post_trade_weights(state, action), block, settings.gamma, True)
        value += discount**t * spec.net_reward(reward, cost)
        state = points[spec.nearest_lattice_index(w_end, solution.lattice_increment)]
    return value


def check_level(market, settings, opaths, starts, paths, best: int, h: int) -> dict:
    t0 = time.perf_counter()
    sol = oracle.solve(market, settings, opaths, block_outcome=env.block_outcome)
    solve_seconds = time.perf_counter() - t0

    orc = evaluate.evaluate_policy(evaluate.oracle_policy(sol), starts, paths, settings)
    reb = evaluate.evaluate_policy(evaluate.always_rebalance_policy(best), starts, paths, settings)
    lead_replay = evaluate.paired_gap(orc["value"], reb["value"], h)

    start_idx = spec.nearest_lattice_index(starts, sol.lattice_increment)
    v_orc = evaluate.model_value(sol, sol.policy_table, h)[start_idx]
    v_reb = evaluate.model_value(sol, np.full(len(sol.V), best, dtype=np.int64), h)[start_idx]
    lead_model = float(spec.annualized_bps(v_orc - v_reb, horizon=h).mean())

    gap_replay = spec.annualized_bps(orc["value"] - reb["value"], horizon=h)
    gap_model = spec.annualized_bps(v_orc - v_reb, horizon=h)
    diff = gap_replay - gap_model
    rounding = evaluate.paired_gap(rounded_replay(sol, starts, paths, settings), orc["value"], h)
    level = spec.annualized_bps(orc["value"] - v_orc, horizon=h)

    return {
        "solve_seconds": round(solve_seconds, 1),
        "n_iterations": sol.n_iterations,
        "lead_replay": lead_replay,
        "lead_model": lead_model,
        "check5": {"mean": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))},
        "rounding": rounding,
        "level_replay_minus_model": {"mean": float(level.mean()), "se": float(level.std(ddof=1) / np.sqrt(len(level)))},
        "oracle_hold_fraction": orc["mean_hold_fraction"],
        "oracle_turnover_per_block": float(orc["turnover_per_block"].mean()),
        "rebalance_turnover_per_block": float(reb["turnover_per_block"].mean()),
        "oracle_value_bps": orc["mean_value_bps"],
        "rebalance_value_bps": reb["mean_value_bps"],
        # cv_correction is a per-block daily reward, not a discounted sum, so it is
        # annualized by 252 days and 1e4 bp, without annualized_bps's (1 - rho).
        "cv_correction_bps_yr_mean": float(sol.cv_correction.mean() * spec.TRADING_DAYS * spec.BPS),
    }, sol


def main(levels=None, m_oracle: int = M_ORACLE, h: int = H, m: int = M_EVAL) -> dict:
    market = env.load_market()
    settings_by_level = cost_settings()
    levels = list(settings_by_level) if levels is None else list(levels)
    best = evaluate.best_static_target(market, spec.GAMMA_HEADLINE)
    starts, paths = evaluate.draw_evaluation_set(market, SeedPlan(spec.TUNING_MASTER_SEED, 1).evaluation_rng(), h, m)
    opaths = oracle_paths(market, m_oracle)
    out = {"H": h, "m": m, "M_oracle": m_oracle, "best_target": spec.target_grid()[best].round(2).tolist(),
           "replay_seed": "TUNING_MASTER_SEED evaluation stream", "oracle_seed": list(spec.ORACLE_PATH_SEED),
           "levels": {}}
    for level in levels:
        out["levels"][level], _ = check_level(market, settings_by_level[level], opaths, starts, paths, best, h)
        print(json.dumps({level: out["levels"][level]}), flush=True)
    return out


def path_count_in_replay(level: str = "aum_1b", m_small: int = 2000, m_large: int = 4000,
                         h: int = H, m: int = M_EVAL) -> dict:
    """William's check-3 option 2, measured in the replay: the M = m_large
    oracle policy minus the M = m_small one, paired on the same evaluation
    paths. The fixed paths for m_small are the first m_small of m_large."""
    market = env.load_market()
    settings = cost_settings()[level]
    starts, paths = evaluate.draw_evaluation_set(market, SeedPlan(spec.TUNING_MASTER_SEED, 1).evaluation_rng(), h, m)
    big = oracle_paths(market, m_large)
    values, tables = {}, {}
    for n in (m_small, m_large):
        sol = oracle.solve(market, settings, big[:n], block_outcome=env.block_outcome)
        tables[n] = sol.policy_table
        values[n] = evaluate.evaluate_policy(evaluate.oracle_policy(sol), starts, paths, settings)["value"]
    starts_on_grid = spec.target_lattice_index()
    return {"level": level, "m_small": m_small, "m_large": m_large,
            "large_minus_small": evaluate.paired_gap(values[m_large], values[m_small], h),
            "policy_changed_share_2pct_starts": float((tables[m_large][starts_on_grid]
                                                       != tables[m_small][starts_on_grid]).mean())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--m-oracle", type=int, default=M_ORACLE)
    parser.add_argument("--path-count", action="store_true", help="also run the check-3 replay comparison at $1B")
    args = parser.parse_args()
    result = main(args.levels, args.m_oracle)
    if args.path_count:
        result["path_count_in_replay"] = path_count_in_replay()
    print(json.dumps(result, indent=2))
