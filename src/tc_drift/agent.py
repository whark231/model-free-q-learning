"""Model-free function-approximation agent with a HOLD action (William).

In plain terms: the agent estimates how good each action is from its current
holdings, using a smooth (spline) model of the value of the portfolio it ends
up holding (the same model for a target and for the current holdings under
HOLD), minus two learned cost terms in the size of the trade. It learns
only from the net rewards it receives (docs/methods/TC_DRIFT_DESIGN.md §6).

The value model (one shared model for trades and HOLD, decided D5):

    trade to target a:  Qhat(s, a)    = theta . phi(t_a) - c1 ||t_a - s||_1 - c2 ||t_a - s||_2^2
    HOLD:               Qhat(s, HOLD) = theta . phi(s)

phi is the three-asset spline map (src/tc_drift/features.py). theta . phi(x)
estimates the post-decision value G(x) of starting a block holding x, so every
step, trade or hold, teaches the value of holding some portfolio. c1 and c2
should approach kappa / B and lambda / (2 B) if theta . phi is unbiased; the
agent is never told kappa, lambda or B's role in the cost.

Learning, per update(s, a, r, s'):
- target y = r + rho * max_a' Qhat_frozen(s', a'), with a frozen copy of
  (theta, c1, c2) refreshed every `target_refresh_every` updates;
- semi-gradient step on (y - Qhat(s, a))^2, with step size alpha_value for
  theta (plus an l2 shrinkage) and alpha_cost for (c1, c2);
- then c1, c2 = max(0, c1), max(0, c2) (decided D11): costs cannot be
  negative, a fact about costs in general, not their size (disclosed).

Exploration (decided D6): epsilon_t = exp(-beta * t), t = select_action calls
so far. An exploratory step draws uniformly from all 1,327 actions, HOLD
included.

Ties (TC_DRIFT_DESIGN.md §4): HOLD wins when no trade is better by more than
TIE_TOLERANCE; among trades the lowest action index wins. Greedy choices use
no random numbers, so greedy_action never changes the agent's state.

Leakage boundary: the agent is built from a random generator and its own
hyperparameters, and reads the target grid from spec.target_grid(). It never
receives mu, Sigma, kappa, lambda, gamma or anything from the oracle. The
constructor rejects any keyword not in HYPERPARAMETERS
(src/tests/test_tc_drift_spec.py and the leakage scan in
src/tests/test_environment.py). The discount rho is spec.DISCOUNT, the
objective every policy is scored on, not a property of the market.

Post hoc: designed after baseline-v1 and the first transaction-cost sweep were
seen. Label every result that uses it that way.
"""

import numpy as np

from src.tc_drift.features import n_features, spline_features
from src.tc_drift.spec import DISCOUNT, HOLD, N_ACTIONS, check_holdings, target_grid

# A trade must beat HOLD by more than this (daily units) to be chosen. Values
# are about 4e-3 and the decisions that matter differ by about 1e-8, so this
# only absorbs rounding: the same weights scored through two matrix products
# can differ in the last bits.
TIE_TOLERANCE = 1e-14

# Squared-norm bounds of each parameter block's features: ||phi||^2 <= 6 for
# the spline map (each asset's basis sums to one), and ||u||_1^2 + ||u||_2^4
# <= 4 + 4 for the two distance features (TC_DRIFT_DESIGN.md §6, "Stability").
_MAX_SQ_VALUE = 6.0
_MAX_SQ_COST = 8.0


class DriftAgent:
    """Implements spec.TCAgent. Hyperparameters are keyword-only and limited
    to HYPERPARAMETERS (TC_DRIFT_DESIGN.md §9); none names mu, Sigma, the
    costs, gamma or the oracle.

    Required: alpha_value, alpha_cost, beta, target_refresh_every.
    Optional: l2 (default 0), n_basis (5), degree (3).
    """

    # One shared value model for trades and HOLD (TC_DRIFT_DESIGN.md D5), so
    # one step size for theta and one for the two cost coefficients.
    HYPERPARAMETERS: frozenset = frozenset({
        "alpha_value", "alpha_cost", "l2", "target_refresh_every", "beta", "n_basis", "degree",
    })
    REQUIRED: frozenset = frozenset({"alpha_value", "alpha_cost", "beta", "target_refresh_every"})
    DEFAULTS = {"l2": 0.0, "n_basis": 5, "degree": 3}

    def __init__(self, rng: np.random.Generator, **hyperparameters):
        unknown = set(hyperparameters) - self.HYPERPARAMETERS
        if unknown:
            raise TypeError(f"DriftAgent got unexpected hyperparameters {sorted(unknown)}; "
                            f"allowed: {sorted(self.HYPERPARAMETERS)}")
        missing = self.REQUIRED - set(hyperparameters)
        if missing:
            raise TypeError(f"DriftAgent is missing required hyperparameters {sorted(missing)}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator (from src.environment.SeedPlan)")
        hp = {**self.DEFAULTS, **hyperparameters}

        for name in ("alpha_value", "alpha_cost", "beta", "l2"):
            value = hp[name]
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if hp["alpha_value"] <= 0 or hp["alpha_cost"] < 0:
            raise ValueError("alpha_value must be positive and alpha_cost nonnegative")
        if hp["beta"] < 0 or hp["l2"] < 0:
            raise ValueError("beta and l2 must be nonnegative")
        bound = hp["alpha_value"] * (_MAX_SQ_VALUE + hp["l2"]) + hp["alpha_cost"] * _MAX_SQ_COST
        if bound >= 2.0:
            raise ValueError(f"step sizes break the stability bound: alpha_value * (6 + l2) + alpha_cost * 8 "
                             f"= {bound:.3g} must be below 2 (TC_DRIFT_DESIGN.md section 6)")
        for name in ("target_refresh_every", "n_basis", "degree"):
            value = hp[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if hp["target_refresh_every"] < 1:
            raise ValueError("target_refresh_every must be at least 1")

        self.hyperparameters = dict(hp)
        self.alpha_value = float(hp["alpha_value"])
        self.alpha_cost = float(hp["alpha_cost"])
        self.beta = float(hp["beta"])
        self.l2 = float(hp["l2"])
        self.target_refresh_every = int(hp["target_refresh_every"])
        self.n_basis = int(hp["n_basis"])
        self.degree = int(hp["degree"])
        self.discount = DISCOUNT
        self.rng = rng

        self._targets = target_grid()
        self._phi_targets = spline_features(self._targets, self.n_basis, self.degree)  # (N_TARGETS, k), fixed
        self._phi_targets.flags.writeable = False
        self._theta = np.zeros(n_features(self.n_basis, self.degree))
        self._cost = np.zeros(2)          # (c1, c2): per unit of ||u||_1 and of ||u||_2^2, daily units
        self._t = 0                       # select_action calls; the epsilon schedule reads this
        self._n_updates = 0
        self._g_online = None             # theta . phi(t_a) for every target, recomputed after each update
        self._state_cache = {}            # features and target distances of the last two states seen
        self._refresh_frozen()

    # -- Diagnostics (read-only copies; train.py and evaluate.py may log them) --

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def cost_linear(self) -> float:
        """c1, the learned cost per unit of ||u||_1, in daily reward units."""
        return float(self._cost[0])

    @property
    def cost_quadratic(self) -> float:
        """c2, the learned cost per unit of ||u||_2^2, in daily reward units."""
        return float(self._cost[1])

    @property
    def epsilon(self) -> float:
        """The exploration rate the next select_action call will use."""
        return float(np.exp(-self.beta * self._t))

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def holding_values(self, weights: np.ndarray) -> np.ndarray:
        """theta . phi(x) for holdings x, (m, 3) -> (m,): the agent's implied
        post-decision value G. For the divergence diagnostic (§6), compared
        with the oracle's G at evaluation only."""
        w = check_holdings(weights)
        return spline_features(w / w.sum(axis=1, keepdims=True), self.n_basis, self.degree) @ self._theta

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Qhat(state, a) for all N_ACTIONS actions with the current
        parameters; index HOLD is the hold value."""
        s = self._state(state)
        trades, hold = self._scores(s, self._g_online_values(), self._theta, self._cost)
        return np.append(trades, hold)

    # -- The TCAgent interface --

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy: draw u = rng.random(); explore with
        rng.integers(N_ACTIONS) (HOLD included) if u < epsilon, else take the
        greedy action. t increases once per call, on every branch."""
        s = self._state(state)
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(N_ACTIONS))
        else:
            action = self._greedy(s, self._g_online_values(), self._theta, self._cost)
        self._t += 1
        return action

    def update(self, state: np.ndarray, action: int, reward_net: float, next_state: np.ndarray) -> None:
        """One semi-gradient step on (y - Qhat(s, a))^2 with
        y = reward_net + rho * max_a' Qhat_frozen(next_state, a')."""
        s = self._state(state)
        s_next = self._state(next_state)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action <= HOLD:
            raise ValueError(f"action must be an integer in [0, {HOLD}], got {action!r}")
        if isinstance(reward_net, bool) or not np.isfinite(reward_net):
            raise ValueError(f"reward_net must be finite, got {reward_net!r}")

        frozen_trades, frozen_hold = self._scores(s_next, self._g_frozen, self._theta_frozen, self._cost_frozen)
        target = float(reward_net) + self.discount * max(frozen_trades.max(), frozen_hold)

        if action == HOLD:
            phi = self._features(s)
            distances = np.zeros(2)
        else:
            phi = self._phi_targets[action]
            u = self._targets[action] - s
            distances = np.array([np.abs(u).sum(), (u * u).sum()])
        error = target - (phi @ self._theta - distances @ self._cost)

        self._theta += self.alpha_value * (error * phi - self.l2 * self._theta)
        self._cost = np.maximum(self._cost - self.alpha_cost * error * distances, 0.0)   # dQ/dc = -distance; D11
        if not (np.all(np.isfinite(self._theta)) and np.all(np.isfinite(self._cost))):
            raise FloatingPointError("the parameters became non-finite; the step sizes are probably too large")

        self._g_online = None
        self._n_updates += 1
        if self._n_updates % self.target_refresh_every == 0:
            self._refresh_frozen()

    def greedy_action(self, state: np.ndarray) -> int:
        """The current greedy action (HOLD wins ties); never changes the agent's state."""
        return self._greedy(self._state(state), self._g_online_values(), self._theta, self._cost)

    # -- Internals --

    def _state(self, state) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        if s.shape != (3,):
            raise ValueError(f"state must have shape (3,), got {s.shape}")
        s = check_holdings(s)[0]
        return s / s.sum()  # check_holdings clips -1e-9 rounding without renormalizing; the feature map re-checks the sum

    def _cached(self, s: np.ndarray):
        """(phi(s), ||t_a - s||_1 for every target, ||t_a - s||_2^2 for every
        target). A step scores the next state in update() and again in the
        following select_action(), so the last two states are kept."""
        key = s.tobytes()
        entry = self._state_cache.get(key)
        if entry is None:
            u = self._targets - s
            entry = (spline_features(s, self.n_basis, self.degree)[0], np.abs(u).sum(axis=1), (u * u).sum(axis=1))
            if len(self._state_cache) >= 2:
                self._state_cache.pop(next(iter(self._state_cache)))
            self._state_cache[key] = entry
        return entry

    def _features(self, s: np.ndarray) -> np.ndarray:
        return self._cached(s)[0]

    def _g_online_values(self) -> np.ndarray:
        if self._g_online is None:
            self._g_online = self._phi_targets @ self._theta
        return self._g_online

    def _refresh_frozen(self) -> None:
        self._theta_frozen = self._theta.copy()
        self._cost_frozen = self._cost.copy()
        self._g_frozen = self._phi_targets @ self._theta_frozen

    def _scores(self, s, g_targets, theta, cost):
        """(Qhat for each trade (N_TARGETS,), Qhat for HOLD) at holdings s."""
        phi, d1, d2 = self._cached(s)
        return g_targets - cost[0] * d1 - cost[1] * d2, float(phi @ theta)

    def _greedy(self, s, g_targets, theta, cost) -> int:
        trades, hold = self._scores(s, g_targets, theta, cost)
        best = int(trades.argmax())
        return best if trades[best] > hold + TIE_TOLERANCE else HOLD
