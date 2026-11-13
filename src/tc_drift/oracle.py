"""Numerical benchmark: the optimal policy with known mu and Sigma (William).
Phase 1: signatures only.

In plain terms: on the 1% lattice of holdings, work out the best action from
every state by value iteration, with the expected block reward and the
next-state distribution estimated once on fixed simulated paths. It is a
sample-average approximation (finitely many paths, rounding to the lattice),
so every reported gap uses its policy replayed on the evaluation paths, never
its in-sample V. Benchmark and evaluation only; the agent never sees anything
from this module (docs/methods/TC_DRIFT_DESIGN.md §5).

solve() takes the fixed paths and the block-scoring function as inputs, so
William can build and test it with his own reference loop before Albert's
env.block_outcome lands, and production runs pass env.block_outcome.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.tc_drift.spec import DISCOUNT, LATTICE_INCREMENT, Market, TCSettings

# (w0 (m, 3), excess_returns (m, B, 3), gamma, drift) -> (reward (m,), w_end (m, 3)); env.block_outcome's signature
BlockOutcome = Callable[[np.ndarray, np.ndarray, float, bool], tuple[np.ndarray, np.ndarray]]


@dataclass
class OracleSolution:
    """Everything the checks and the report need, indexed by lattice state.

    V: optimal value per lattice state (daily units, discounted).
    G: post-decision value per lattice point (§3).
    policy_table: optimal action per lattice state (HOLD wins ties).
    r_hat: control-variate estimate of each lattice point's expected block reward.
    cv_correction: r_hat minus U, per lattice point (reported for every setting).
    next_index: (n_lattice, n_paths) lattice index of the end-of-block holdings.
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

    def policy(self, state: np.ndarray):
        """Optimal action for holdings state, (3,) -> int or (m, 3) -> (m,).
        Off-lattice holdings use their nearest lattice point
        (spec.nearest_lattice_index)."""
        raise NotImplementedError("Phase 2 (William)")

    def value_of(self, policy_table: np.ndarray, discount: float | None = None) -> np.ndarray:
        """Value on the lattice of a fixed policy (one action per lattice
        state) on the same fixed paths, at `discount` (default: this
        solution's). Scores the myopic table at rho = 0.95."""
        raise NotImplementedError("Phase 2 (William)")


def solve(market: Market, settings: TCSettings, excess_paths: np.ndarray, *, block_outcome: BlockOutcome,
          discount: float = DISCOUNT, drift: bool = True,
          lattice_increment: float = LATTICE_INCREMENT) -> OracleSolution:
    """Value iteration on Q(s, a) = r(a) - cost(s, a) / B + rho * E[V(next)].

    excess_paths (M, B, 3) are the fixed paths, drawn once from
    np.random.default_rng(spec.ORACLE_PATH_SEED) and reused for every setting.
    discount = 0 gives the myopic reference policy."""
    raise NotImplementedError("Phase 2 (William)")
