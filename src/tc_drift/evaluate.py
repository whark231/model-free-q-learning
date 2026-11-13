"""Scoring policies on common simulated paths (Albert). Phase 1: signatures only.

In plain terms: every policy (the agent's greedy policy, the oracle, and the
reference policies) is run from the same start holdings on the same simulated
returns, so differences between them come from the decisions, not the luck
of the draw (docs/methods/TC_DRIFT_DESIGN.md §7). mu and Sigma are used here
only to draw the evaluation paths, never handed to a policy.
"""

from typing import Callable

import numpy as np

from src.tc_drift.spec import DISCOUNT, TCAgent, TCSettings

# (holdings (m, 3), block index t) -> actions (m,). Stationary policies (the
# agent's greedy policy, the oracle) ignore t; "trade once, then hold" needs it.
Policy = Callable[[np.ndarray, int], np.ndarray]


def evaluate_policy(policy: Policy, starts: np.ndarray, excess_paths: np.ndarray, settings: TCSettings, *,
                    discount: float = DISCOUNT, drift: bool = True) -> dict:
    """Run `policy` from starts (m, 3) over excess_paths (H, m, B, 3).

    Returns per-path and mean: discounted net reward, hold fraction (a trade
    with zero size counts as a hold), trades per path, turnover per block,
    mean trade size, and the mean held weights. The oracle gap is the
    oracle's replayed discounted net reward minus this one on the same paths,
    in bps per year (spec.annualized_bps with the finite horizon H)."""
    raise NotImplementedError("Phase 2 (Albert)")


def agent_policy(agent: TCAgent) -> Policy:
    """Batch adapter: apply agent.greedy_action row by row."""
    raise NotImplementedError("Phase 2 (Albert)")


def oracle_policy(solution) -> Policy:
    """Batch adapter for oracle.OracleSolution.policy (already accepts (m, 3))."""
    raise NotImplementedError("Phase 2 (Albert)")


def always_rebalance_policy(best_target: int) -> Policy:
    """Trade back to one fixed target every block (the no-cost choice)."""
    raise NotImplementedError("Phase 2 (Albert)")


def never_trade_policy(first_target: int) -> Policy:
    """Trade to first_target in block 0, HOLD in every later block."""
    raise NotImplementedError("Phase 2 (Albert)")
