"""Scoring policies on common simulated paths (Albert).

In plain terms: every policy (the agent's greedy policy, the oracle, and the
reference policies) is run from the same start holdings on the same simulated
returns, so differences between them come from the decisions, not the luck
of the draw (docs/methods/TC_DRIFT_DESIGN.md §7). mu and Sigma are used here
only to draw the evaluation paths and to pick the reference target, never
handed to a policy.

Conventions:
- A policy maps (holdings (m, 3), block index t) to actions (m,). Stationary
  policies ignore t; "trade once, then hold" needs it (C10).
- Values are discounted sums of net rewards in daily units over the H blocks
  of the paths. Gaps are reported in bps per year with
  spec.annualized_bps(..., horizon=H), on the same paths and starts.
- A trade whose target equals the current holdings counts as a hold (§4).
- Turnover and trade size are one-way, 0.5 * ||u||_1.
"""

from typing import Callable

import numpy as np

from src.tc_drift import env
from src.tc_drift.spec import (BLOCK_LENGTH, DISCOUNT, HOLD, N_ACTIONS, N_ASSETS, N_TARGETS, Market, TCAgent,
                               TCSettings, annualized_bps, check_holdings, net_reward, post_trade_weights,
                               target_grid, trade_cost)

# (holdings (m, 3), block index t) -> actions (m,). Stationary policies (the
# agent's greedy policy, the oracle) ignore t; "trade once, then hold" needs it.
Policy = Callable[[np.ndarray, int], np.ndarray]

ZERO_TRADE_TOL = 1e-12  # ||u||_1 at or below this counts as a hold, as in train.py


def evaluate_policy(policy: Policy, starts: np.ndarray, excess_paths: np.ndarray, settings: TCSettings, *,
                    discount: float = DISCOUNT, drift: bool = True) -> dict:
    """Run `policy` from starts (m, 3) over excess_paths (H, m, B, 3).

    Returns per-path and mean: discounted net reward, hold fraction (a trade
    with zero size counts as a hold), trades per path, turnover per block,
    mean trade size, and the mean held weights. The oracle gap is the
    oracle's replayed discounted net reward minus this one on the same paths,
    in bps per year (spec.annualized_bps with the finite horizon H).

    excess_paths may also be any sequence of H arrays of shape (m, B, 3), so
    large evaluations can pass blocks generated one at a time.

    Keys: value (m,), value_bps (m,), hold_fraction (m,), trades (m,),
    turnover_per_block (m,), cost_bps_per_block (m,), mean_trade_size (scalar,
    over all trades; nan if none), mean_weights (3,) (post-trade weights over
    every block and path), mean_value_bps, mean_hold_fraction, horizon, m.
    """
    if not 0 <= discount < 1:
        raise ValueError(f"discount must be in [0, 1), got {discount!r}")
    if not isinstance(drift, (bool, np.bool_)):
        raise TypeError(f"drift must be True or False, got {type(drift).__name__}")
    state = check_holdings(starts).copy()
    m = len(state)
    horizon = len(excess_paths)
    if horizon < 1:
        raise ValueError("excess_paths must hold at least one block")

    value = np.zeros(m)
    holds = np.zeros(m)
    turnover = np.zeros(m)
    cost_total = np.zeros(m)
    weight_sum = np.zeros(N_ASSETS)
    trade_size_sum, n_trades = 0.0, 0
    for t in range(horizon):
        block = np.asarray(excess_paths[t], dtype=float)
        if block.shape != (m, BLOCK_LENGTH, N_ASSETS):
            raise ValueError(f"block {t} has shape {block.shape}, expected {(m, BLOCK_LENGTH, N_ASSETS)}")
        action = _checked_actions(policy(state.copy(), t), m)
        cost = trade_cost(state, action, settings)
        x = post_trade_weights(state, action)
        size = 0.5 * np.abs(x - state).sum(axis=1)
        is_trade = 2 * size > ZERO_TRADE_TOL
        reward, state = env.block_outcome(x, block, settings.gamma, drift)
        value += discount**t * net_reward(reward, cost)
        holds += ~is_trade
        turnover += size
        cost_total += cost
        weight_sum += x.sum(axis=0)
        trade_size_sum += float(size[is_trade].sum())
        n_trades += int(is_trade.sum())

    value_bps = annualized_bps(value, discount, horizon=horizon)
    return {
        "value": value,
        "value_bps": value_bps,
        "hold_fraction": holds / horizon,
        "trades": horizon - holds,
        "turnover_per_block": turnover / horizon,
        "cost_bps_per_block": cost_total / horizon * 1e4,
        "mean_trade_size": trade_size_sum / n_trades if n_trades else float("nan"),
        "mean_weights": weight_sum / (horizon * m),
        "mean_value_bps": float(value_bps.mean()),
        "mean_hold_fraction": float((holds / horizon).mean()),
        "horizon": horizon,
        "m": m,
    }


def agent_policy(agent: TCAgent) -> Policy:
    """Batch adapter: apply agent.greedy_action row by row."""
    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return np.array([agent.greedy_action(row) for row in holdings], dtype=np.int64)
    return policy


def oracle_policy(solution) -> Policy:
    """Batch adapter for oracle.OracleSolution.policy (already accepts (m, 3))."""
    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return np.asarray(solution.policy(holdings), dtype=np.int64)
    return policy


def always_rebalance_policy(best_target: int) -> Policy:
    """Trade back to one fixed target every block (the no-cost choice)."""
    target = _target_index(best_target, "best_target")

    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return np.full(len(holdings), target, dtype=np.int64)
    return policy


def own_target_rebalance_policy(agent) -> Policy:
    """Trade every block to the agent's own favourite target, the target with
    the highest learned theta . phi (AUDIT_2026-11-06.md TC-3). Built from the
    agent alone, so nothing leaks; with always_rebalance_policy it splits the
    agent's gap to the oracle into favourite-target error and the rest.
    Added 2026-11-20 by William for Albert's review."""
    return always_rebalance_policy(int(agent.favourite_target()))


def never_trade_policy(first_target: int) -> Policy:
    """Trade to first_target in block 0, HOLD in every later block."""
    target = _target_index(first_target, "first_target")

    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return np.full(len(holdings), target if t == 0 else HOLD, dtype=np.int64)
    return policy


# ---------------------------------------------------------------------------
# Helpers for the runs and the checks (evaluation side only)
# ---------------------------------------------------------------------------

def best_static_target(market: Market, gamma: float) -> int:
    """Index of the 2% target with the highest U(w) = w'mu - (gamma/2) w'Sigma w,
    the no-cost choice the reference policies trade to. Uses mu and Sigma, so
    it is a benchmark; it is never given to the agent."""
    targets = target_grid()
    U = targets @ market.mu - 0.5 * gamma * np.einsum("ai,ij,aj->a", targets, market.sigma, targets)
    return int(np.argmax(U))


def draw_evaluation_set(market: Market, rng: np.random.Generator, horizon: int, m: int):
    """Start holdings (m, 3), uniform on the 2% grid (D4), then paths (H, m, B, 3),
    one env.draw_returns call per block, in that order. The order matches
    d10_pilot.py, so the same generator gives the pilot's starts and paths."""
    targets = target_grid()
    starts = targets[rng.integers(0, len(targets), m)]
    paths = np.stack([env.draw_returns(market, rng, m) for _ in range(horizon)])
    return starts, paths


def paired_gap(value_a: np.ndarray, value_b: np.ndarray, horizon: int,
               discount: float = DISCOUNT) -> dict:
    """Mean and standard error, in bps/yr, of value_a - value_b on the same
    paths and starts (per-path discounted values from evaluate_policy)."""
    a, b = np.asarray(value_a, dtype=float), np.asarray(value_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("value_a and value_b must be 1-D, the same length and at least 2 long")
    gap = annualized_bps(a - b, discount, horizon=horizon)
    return {"mean": float(gap.mean()), "se": float(gap.std(ddof=1) / np.sqrt(len(gap)))}


def model_value(solution, policy_table: np.ndarray, horizon: int, discount: float = DISCOUNT) -> np.ndarray:
    """Finite-horizon value of a fixed lattice policy in the oracle's own
    model: H sweeps of V <- Q(s, pi(s)) from V = 0, with the solve's r_hat,
    next-state distribution and costs. The model-side counterpart of a
    replay over H blocks (check 5), so both sides use the same horizon and
    discounting. solution.value_of gives the infinite-horizon value instead."""
    table = np.asarray(policy_table)
    n = len(solution.V)
    if table.shape != (n,) or not np.issubdtype(table.dtype, np.integer) or ((table < 0) | (table > HOLD)).any():
        raise ValueError(f"policy_table must be integer actions in [0, {HOLD}], shape ({n},)")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    hold = table == HOLD
    trade = np.where(hold, 0, table)
    landing = np.where(hold, np.arange(n), solution.target_index[trade])
    step_cost = np.where(hold, 0.0, solution.cost[np.arange(n), trade]) / solution.block_length
    P = solution.transition
    V = np.zeros(n)
    for _ in range(horizon):
        V = (solution.r_hat + discount * (P @ V))[landing] - step_cost
    return V


def lattice_table(policy: Policy, lattice_points: np.ndarray, t: int = 0) -> np.ndarray:
    """A stationary policy evaluated on every lattice point, as an oracle-style table."""
    return _checked_actions(policy(check_holdings(lattice_points), t), len(lattice_points))


def probe_convergence(log: dict, final_fraction: float = 0.10) -> dict:
    """D12's primary rule: the greedy action on every probe state is unchanged
    at every checkpoint over the final `final_fraction` of training.

    The window starts at the last checkpoint at or before
    (1 - final_fraction) * n_blocks, so its first row is the policy in force
    when the final stretch began. Returns converged (bool), the share of
    probes stable through the window, and stable_from_block: the first
    checkpoint from which every probe's greedy action equals its final one
    (descriptive; None if only the last checkpoint qualifies trivially)."""
    if not 0 < final_fraction < 1:
        raise ValueError("final_fraction must be in (0, 1)")
    blocks = np.asarray(log["checkpoint_block"])
    greedy = np.asarray(log["probe_greedy"])
    cutoff = (1 - final_fraction) * log["n_blocks"]
    before = np.flatnonzero(blocks <= cutoff)
    first = int(before[-1]) if before.size else 0
    window = greedy[first:]
    stable = (window == window[-1]).all(axis=0)
    same_as_final = (greedy == greedy[-1]).all(axis=1)
    changed = np.flatnonzero(~same_as_final)
    start = int(changed[-1]) + 1 if changed.size else 0
    return {
        "converged": bool(stable.all() and len(window) > 1),
        "share_stable": float(stable.mean()),
        "window_first_block": int(blocks[first]),
        "stable_from_block": int(blocks[start]) if start < len(blocks) - 1 else None,
    }


def _checked_actions(actions, m: int) -> np.ndarray:
    a = np.asarray(actions)
    if a.shape != (m,) or a.dtype == bool or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"a policy must return {m} integer actions, got shape {a.shape} and dtype {a.dtype}")
    if ((a < 0) | (a >= N_ACTIONS)).any():
        raise ValueError(f"actions must lie in [0, {HOLD}]")
    return a.astype(np.int64)


def _target_index(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < N_TARGETS:
        raise ValueError(f"{name} must be a target index in [0, {N_TARGETS}), got {value!r}")
    return int(value)


__all__ = ["Policy", "evaluate_policy", "agent_policy", "oracle_policy", "always_rebalance_policy",
           "never_trade_policy", "own_target_rebalance_policy", "best_static_target", "draw_evaluation_set", "paired_gap", "model_value",
           "lattice_table", "probe_convergence"]
