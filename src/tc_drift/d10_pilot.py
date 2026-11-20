"""Pilot for decision D10: how noisy is a policy gap at H = 90 blocks and m = 1,000 starts?

In plain terms: before fixing the evaluation size, play simple policies
(always rebalance to the best 2% target, rebalance to it every 2, 4 or 8
blocks, trade to it once and then hold, never trade) from the same 1,000 random
2% starts on the same 90-block paths, at each cost level, and measure the
standard error of the gap between pairs. A paired gap's SE grows with how far
apart the two policies' holdings get, so these numbers bracket, but do not
predict, the SE of the agent-against-oracle gap. TC_DRIFT_DESIGN.md §7 records
the result.

    python -m src.tc_drift.d10_pilot

Uses the tuning seed (spec.TUNING_MASTER_SEED), never the reported seed. Picks
the best target with mu and Sigma. That is allowed because the pilot is an
evaluation reference, not an agent.
"""

import json

import numpy as np

from src.environment import SeedPlan
from src.tc_drift import env, spec
from src.tc_drift.calibration import cost_settings

H = 90
M = 1000


def discounted_net_reward(policy, starts, paths, settings, discount=spec.DISCOUNT):
    """Per-start discounted sum of net rewards, drift on. paths (H, m, B, 3)."""
    state, total = starts.copy(), np.zeros(len(starts))
    for k, block in enumerate(paths):
        action = policy(state, k)
        cost = spec.trade_cost(state, action, settings)
        reward, state = env.block_outcome(spec.post_trade_weights(state, action), block, settings.gamma, drift=True)
        total += discount**k * spec.net_reward(reward, cost)
    return total


def main(h: int = H, m: int = M) -> dict:
    market = env.load_market()
    targets = spec.target_grid()
    utility = targets @ market.mu - spec.GAMMA_HEADLINE / 2 * np.einsum("ai,ij,aj->a", targets, market.sigma, targets)
    best = int(np.argmax(utility))
    rng = SeedPlan(spec.TUNING_MASTER_SEED, 1).evaluation_rng()
    starts = targets[rng.integers(0, len(targets), m)]
    paths = np.stack([env.draw_returns(market, rng, m) for _ in range(h)])

    policies = {
        "rebalance": lambda s, k: np.full(len(s), best),
        "every_2": lambda s, k: np.full(len(s), best if k % 2 == 0 else spec.HOLD),
        "every_4": lambda s, k: np.full(len(s), best if k % 4 == 0 else spec.HOLD),
        "every_8": lambda s, k: np.full(len(s), best if k % 8 == 0 else spec.HOLD),
        "once": lambda s, k: np.full(len(s), best if k == 0 else spec.HOLD),
        "never": lambda s, k: np.full(len(s), spec.HOLD),
    }
    out = {"H": h, "m": m, "seed": "TUNING_MASTER_SEED evaluation stream", "paths_mb": paths.nbytes / 1e6,
           "weight_beyond_horizon": spec.DISCOUNT**h, "best_target": targets[best].round(2).tolist(), "levels": {}}
    for level, settings in cost_settings().items():
        value = {name: discounted_net_reward(p, starts, paths, settings) for name, p in policies.items()}
        row = {}
        for a, b in (("rebalance", "every_2"), ("rebalance", "every_4"), ("rebalance", "every_8"),
                     ("rebalance", "once"), ("rebalance", "never"), ("once", "never")):
            gap = spec.annualized_bps(value[a] - value[b], horizon=h)
            row[f"{a}_minus_{b}"] = {"mean": round(float(gap.mean()), 2),
                                     "se": round(float(gap.std(ddof=1) / np.sqrt(m)), 2)}
        level_value = spec.annualized_bps(value["rebalance"], horizon=h)
        row["rebalance_value_se"] = round(float(level_value.std(ddof=1) / np.sqrt(m)), 2)
        out["levels"][level] = row
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
