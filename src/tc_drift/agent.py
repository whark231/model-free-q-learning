"""Model-free function-approximation agent with a HOLD action (William).
Phase 1: signatures only.

In plain terms: the agent estimates how good each action is from its current
holdings, using a smooth (spline) model of the value of the portfolio it ends
up holding (the same model for a target and for the current holdings under
HOLD), minus two learned cost terms in the size of the trade. It learns
only from the net rewards it receives (docs/methods/TC_DRIFT_DESIGN.md §6).

Leakage boundary: the agent is built from a random generator and its own
hyperparameters, and reads the target grid from spec.target_grid(). It never
receives mu, Sigma, kappa, lambda, gamma or anything from the oracle. The
constructor must reject any keyword not in HYPERPARAMETERS
(src/tests/test_tc_drift_spec.py and the leakage scan in
src/tests/test_environment.py).
"""

import numpy as np


class DriftAgent:
    """Implements spec.TCAgent. Hyperparameters are keyword-only, limited to
    HYPERPARAMETERS, and fixed by the tuning design
    (docs/methods/TC_DRIFT_DESIGN.md §9). William fills in the final list in
    Phase 2; none may name mu, Sigma, the costs, gamma or the oracle."""

    # One shared value model for trades and HOLD (TC_DRIFT_DESIGN.md D5), so
    # one step size for theta and one for the two cost coefficients.
    HYPERPARAMETERS: frozenset = frozenset({
        "alpha_value", "alpha_cost", "l2", "target_refresh_every", "beta", "n_basis", "degree",
    })

    def __init__(self, rng: np.random.Generator, **hyperparameters):
        raise NotImplementedError("Phase 2 (William)")

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("Phase 2 (William)")

    def update(self, state: np.ndarray, action: int, reward_net: float, next_state: np.ndarray) -> None:
        raise NotImplementedError("Phase 2 (William)")

    def greedy_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("Phase 2 (William)")
