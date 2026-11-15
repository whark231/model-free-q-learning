"""Numerical benchmark: the optimal policy with known mu and Sigma (William).

In plain terms: on the 1% lattice of holdings, work out the best action from
every state by value iteration, with the expected block reward and the
next-state distribution estimated once on fixed simulated paths. It is a
sample-average approximation (finitely many paths, rounding to the lattice),
so every reported gap uses its policy replayed on the evaluation paths, never
its in-sample V. Benchmark and evaluation only; the agent never sees anything
from this module (docs/methods/TC_DRIFT_DESIGN.md §5).

solve() takes the fixed paths and the block-scoring function as inputs, so it
can be built and tested with a reference loop before Albert's
env.block_outcome lands, and production runs pass env.block_outcome.

How it works (TC_DRIFT_DESIGN.md §3 and §5):
1. Score every lattice point x on the same M paths with block_outcome, twice:
   buy-and-hold (drift on) and constant weights (drift off). The control
   variate r_hat(x) = U(x) + mean_m[R_bh - R_const] estimates the expected
   block reward; the end-of-block holdings, rounded to the lattice, give the
   next-state distribution P(x, .).
2. Value iteration on the post-decision form:
       G(x) = r_hat(x) + rho * sum_y P(x, y) V(y)
       V(s) = max(G(s), max_a [G(target_a) - cost(s, a) / B])
   with HOLD winning ties, until max |V_new - V| <= 1e-12 * (1 - rho), the
   stopping rule of src.transaction_costs.optimal_policy.

Assumptions:
- The paths must have spec.BLOCK_LENGTH days, the B that spec.net_reward
  spreads the cost over in the environment and the evaluation.
- In the no-drift mode the buy-and-hold and constant-weight rewards are the
  same block_outcome call, so the correction is exactly 0 and r_hat = U
  exactly (check 1).
- Every 2% target must be a lattice point, so lattice_increment must divide
  0.02 (0.02, 0.01 and 0.005 do).
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
from scipy import sparse

from src.tc_drift.spec import (BLOCK_LENGTH, DISCOUNT, HOLD, LATTICE_INCREMENT, N_ASSETS, Market, TCSettings,
                               lattice, nearest_lattice_index, target_grid, target_lattice_index)

# (w0 (m, 3), excess_returns (m, B, 3), gamma, drift) -> (reward (m,), w_end (m, 3)); env.block_outcome's signature
BlockOutcome = Callable[[np.ndarray, np.ndarray, float, bool], tuple[np.ndarray, np.ndarray]]

STOP_FACTOR = 1e-12      # stop when max |V_new - V| <= STOP_FACTOR * (1 - rho), as the old grid solver does
MAX_ITERATIONS = 100_000  # rho = 0.95 needs about 600 from V = 0; this only guards against a bad input
_ROW_CHUNK = 1024         # lattice rows per chunk for the cost matrix, the sweeps and P (caps memory on the 0.5% lattice)


@dataclass
class OracleSolution:
    """Everything the checks and the report need, indexed by lattice state.

    V: optimal value per lattice state (daily units, discounted).
    G: post-decision value per lattice point (§3).
    policy_table: optimal action per lattice state (HOLD wins ties).
    r_hat: control-variate estimate of each lattice point's expected block reward.
    cv_correction: r_hat minus U, per lattice point (reported for every setting).
    next_index: (n_lattice, n_paths) lattice index of the end-of-block holdings.
    U: x'mu - (gamma / 2) x'Sigma x per lattice point, the exact expected
       constant-weight block reward.
    cost: (n_lattice, N_TARGETS) cost of trading from each lattice state to
       each target, a fraction of wealth (not yet divided by B).
    block_length: B, the days each cost is spread over.
    n_iterations: value-iteration sweeps until the stopping rule held.
    """

    V: np.ndarray
    G: np.ndarray
    policy_table: np.ndarray
    r_hat: np.ndarray
    cv_correction: np.ndarray
    next_index: np.ndarray
    settings: TCSettings
    discount: float
    drift: bool
    lattice_increment: float
    U: np.ndarray
    cost: np.ndarray
    block_length: int
    n_iterations: int

    @cached_property
    def transition(self) -> sparse.csr_matrix:
        """P[x, y] = share of the fixed paths on which holding x ends the block at lattice point y."""
        return _transition_matrix(self.next_index)

    @cached_property
    def target_index(self) -> np.ndarray:
        """Lattice index of each 2% target."""
        return target_lattice_index(self.lattice_increment)

    def policy(self, state: np.ndarray):
        """Optimal action for holdings state, (3,) -> int or (m, 3) -> (m,).
        Off-lattice holdings use their nearest lattice point
        (spec.nearest_lattice_index)."""
        idx = nearest_lattice_index(state, self.lattice_increment)
        if np.ndim(idx) == 0:
            return int(self.policy_table[idx])
        return self.policy_table[idx]

    def value_of(self, policy_table: np.ndarray, discount: float | None = None) -> np.ndarray:
        """Value on the lattice of a fixed policy (one action per lattice
        state) on the same fixed paths, at `discount` (default: this
        solution's). Scores the myopic table at rho = 0.95.

        Iterates V(s) = Q(s, pi(s)) with the same r_hat, next-state
        distribution and costs as the solve, and the same stopping rule."""
        rho = self.discount if discount is None else _check_discount(discount)
        table = np.asarray(policy_table)
        n = len(self.V)
        if table.shape != (n,) or table.dtype == bool or not np.issubdtype(table.dtype, np.integer):
            raise ValueError(f"policy_table must be an integer array of shape ({n},)")
        if ((table < 0) | (table > HOLD)).any():
            raise ValueError(f"actions must lie in [0, {HOLD}]")
        hold = table == HOLD
        trade = np.where(hold, 0, table)
        landing = np.where(hold, np.arange(n), self.target_index[trade])
        step_cost = np.where(hold, 0.0, self.cost[np.arange(n), trade]) / self.block_length
        P = self.transition
        V = np.zeros(n)
        for _ in range(MAX_ITERATIONS):
            G = self.r_hat + rho * (P @ V)
            updated = G[landing] - step_cost
            if np.max(np.abs(updated - V)) <= STOP_FACTOR * (1 - rho):
                return updated
            V = updated
        raise RuntimeError("policy evaluation did not converge")


def solve(market: Market, settings: TCSettings, excess_paths: np.ndarray, *, block_outcome: BlockOutcome,
          discount: float = DISCOUNT, drift: bool = True,
          lattice_increment: float = LATTICE_INCREMENT) -> OracleSolution:
    """Value iteration on Q(s, a) = r(a) - cost(s, a) / B + rho * E[V(next)].

    excess_paths (M, B, 3) are the fixed paths, drawn once from
    np.random.default_rng(spec.ORACLE_PATH_SEED) and reused for every setting.
    discount = 0 gives the myopic reference policy."""
    rho = _check_discount(discount)
    if not isinstance(drift, (bool, np.bool_)):
        raise TypeError(f"drift must be True or False, got {type(drift).__name__}")
    paths = np.asarray(excess_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != N_ASSETS or paths.shape[0] < 1 or paths.shape[1] != BLOCK_LENGTH:
        raise ValueError(f"excess_paths must have shape (M, {BLOCK_LENGTH}, {N_ASSETS}) with M >= 1, got {paths.shape}")
    if not np.isfinite(paths).all() or (paths <= -1.0).any():
        raise ValueError("excess_paths must be finite and above -100%")
    mu, sigma = _check_market(market)

    states = lattice(lattice_increment)
    targets = target_grid()
    target_idx = target_lattice_index(lattice_increment)
    if not np.allclose(states[target_idx], targets, rtol=0, atol=1e-12):
        raise ValueError(f"every 2% target must be a lattice point; increment {lattice_increment!r} does not divide 0.02")

    U = states @ mu - 0.5 * settings.gamma * np.einsum("ni,ij,nj->n", states, sigma, states)
    r_hat, correction, next_index = _score_lattice(states, paths, settings.gamma, drift, block_outcome,
                                                   lattice_increment, U)
    cost = _cost_matrix(states, targets, target_idx, settings)
    B = paths.shape[1]
    G, V, policy_table, n_iter = _value_iteration(r_hat, _transition_matrix(next_index), cost, B, target_idx, rho)

    return OracleSolution(V=V, G=G, policy_table=policy_table, r_hat=r_hat, cv_correction=correction,
                          next_index=next_index, settings=settings, discount=rho, drift=bool(drift),
                          lattice_increment=lattice_increment, U=U, cost=cost, block_length=B, n_iterations=n_iter)


# ---------------------------------------------------------------------------
# Pieces of the solve
# ---------------------------------------------------------------------------

def _check_discount(discount: float) -> float:
    if isinstance(discount, bool) or not np.isfinite(discount) or not 0 <= discount < 1:
        raise ValueError(f"discount must be in [0, 1), got {discount!r}")
    return float(discount)


def _check_market(market: Market) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(market.mu, dtype=float)
    sigma = np.asarray(market.sigma, dtype=float)
    if mu.shape != (N_ASSETS,) or sigma.shape != (N_ASSETS, N_ASSETS):
        raise ValueError(f"market must have mu ({N_ASSETS},) and Sigma ({N_ASSETS}, {N_ASSETS})")
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
        raise ValueError("mu and Sigma must be finite")
    if not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("Sigma must be symmetric positive definite")
    return mu, sigma


def _score_lattice(states, paths, gamma, drift, block_outcome, increment, U):
    """r_hat, the control-variate correction and the next-state indices for
    every lattice point, each scored on all M paths (common random numbers)."""
    n, M = len(states), paths.shape[0]
    correction = np.empty(n)
    next_index = np.empty((n, M), dtype=np.int32)
    for i, x in enumerate(states):
        w0 = x[None, :]
        reward_bh, w_end = _checked_outcome(block_outcome(w0, paths, gamma, drift), M)
        if drift:
            reward_const, _ = _checked_outcome(block_outcome(w0, paths, gamma, False), M)
            correction[i] = np.mean(reward_bh - reward_const)
        else:
            correction[i] = 0.0  # the same call scored both, so R_bh - R_const is exactly 0
        next_index[i] = nearest_lattice_index(w_end, increment)
    return U + correction, correction, next_index


def _checked_outcome(outcome, M):
    """block_outcome's (reward, w_end) as finite arrays of shape (M,) and (M, 3)."""
    reward, w_end = (np.asarray(x, dtype=float) for x in outcome)
    if reward.shape != (M,) or w_end.shape != (M, N_ASSETS):
        raise ValueError(f"block_outcome must return shapes ({M},) and ({M}, {N_ASSETS}), "
                         f"got {reward.shape} and {w_end.shape}")
    if not (np.isfinite(reward).all() and np.isfinite(w_end).all()):
        raise ValueError("block_outcome returned non-finite values")
    return reward, w_end


def _cost_matrix(states, targets, target_idx, settings: TCSettings) -> np.ndarray:
    """cost[s, a] = kappa ||t_a - s||_1 + (lam / 2) ||t_a - s||_2^2, built in
    chunks (the same formula as spec.trade_cost, checked against it in the
    tests). Trading to the target the state already sits on costs exactly 0."""
    cost = np.empty((len(states), len(targets)))
    for start in range(0, len(states), _ROW_CHUNK):
        u = targets[None, :, :] - states[start:start + _ROW_CHUNK, None, :]
        cost[start:start + _ROW_CHUNK] = settings.kappa * np.abs(u).sum(axis=2) + 0.5 * settings.lam * (u * u).sum(axis=2)
    cost[target_idx, np.arange(len(targets))] = 0.0
    return cost


def _transition_matrix(next_index: np.ndarray) -> sparse.csr_matrix:
    """Built a block of rows at a time, so the n * M temporaries never exist at once."""
    n, M = next_index.shape
    blocks = []
    for start in range(0, n, _ROW_CHUNK):
        idx = next_index[start:start + _ROW_CHUNK]
        rows = np.repeat(np.arange(len(idx), dtype=np.int32), M)
        block = sparse.coo_matrix((np.full(idx.size, 1.0 / M), (rows, idx.ravel())), shape=(len(idx), n)).tocsr()
        block.sum_duplicates()
        blocks.append(block)
    return sparse.vstack(blocks, format="csr")


def _best_trade(G, cost, B, target_idx):
    """For each state, the best trade a and its value G(target_a) - cost[s, a] / B,
    a block of rows at a time (the lowest action index wins ties among trades)."""
    n = len(G)
    best = np.empty(n, dtype=np.int64)
    value = np.empty(n)
    g_targets = G[target_idx]
    for start in range(0, n, _ROW_CHUNK):
        scores = g_targets[None, :] - cost[start:start + _ROW_CHUNK] / B
        chunk_best = scores.argmax(axis=1)
        best[start:start + _ROW_CHUNK] = chunk_best
        value[start:start + _ROW_CHUNK] = scores[np.arange(len(scores)), chunk_best]
    return best, value


def _value_iteration(r_hat, P, cost, B, target_idx, rho):
    """V(s) = max(G(s), max_a [G(target_a) - cost[s, a] / B]) with
    G = r_hat + rho * P V. HOLD wins ties (a trade must be strictly better)."""
    V = np.zeros(len(r_hat))
    for iteration in range(1, MAX_ITERATIONS + 1):
        G = r_hat + rho * (P @ V)
        updated = np.maximum(G, _best_trade(G, cost, B, target_idx)[1])
        if np.max(np.abs(updated - V)) <= STOP_FACTOR * (1 - rho):
            # One more sweep from the converged V, so the returned G, V and
            # policy are mutually consistent: V[s] = Q(s, policy[s]) exactly.
            G = r_hat + rho * (P @ updated)
            best_trade, best_trade_value = _best_trade(G, cost, B, target_idx)
            policy_table = np.where(best_trade_value > G, best_trade, HOLD).astype(np.int64)
            return G, np.maximum(G, best_trade_value), policy_table, iteration
        V = updated
    raise RuntimeError("value iteration did not converge")
