"""The linear function-approximation Q-learning agent (post-hoc extension)

In plain terms: the tabular agent (src/agent.py) learns one Q-value per
portfolio. This agent learns a short parameter vector theta instead, and
scores portfolio a as

    Qhat(a) = theta . F[a]

where F is a feature matrix built once from the action grid by
src/features.py (one row per portfolio). A reward for one portfolio moves
theta, and so moves the Q-value of every portfolio with similar features.
That sharing is the point: on the 10% grid the table gets about 85 random
tries per portfolio (EXPERIMENTS.md §7.7), far too few to rank them.

One training episode, from train.py's side, is the same as for the tabular
agent:

    a = agent.select_action()
    r = episodic_mean_variance_reward(grid[a], batch, gamma)
    agent.update(a, r)

The agent sees only the action index, the scalar reward, and the feature
matrix F it was built with. It never sees portfolio weights, returns, mu,
Sigma or a benchmark; test_environment.py's leakage tests cover this file.

Differences from TabularQAgent, all deliberate:
- Learning rate: a small constant alpha (the course's suggestion for SGD).
  The tabular baseline that was run uses alpha = 1/N(a). With one-hot
  features and a constant alpha, both agents compute the same Q-values from
  the same (action, reward) sequence (src/tests/test_fa_agent.py). Their
  action sequences still differ, because they consume the tie-break stream
  differently. The reported comparison is constant-alpha FA against
  sample-average tabular, so they are not the same learning rule.
- No discount argument: rho = 0, the assignment's bandit baseline, so the
  target is the reward itself and nothing is bootstrapped.
- update() changes every portfolio's Q-value, not one, so it returns None.
  train.py treats a returned value as "only Q(action) changed" and would
  then keep a wrong copy of the Q-values.
- The greedy portfolio is cached, because finding it means scoring every
  portfolio (F @ theta), which is expensive on big grids.

Post hoc: designed after the baseline-v1 results were seen.
"""

import numpy as np


class LinearQAgent:
    """Stateless Q-learning with a linear Q-function, Qhat(a) = theta . F[a].

    Stateless for the same reason as the tabular agent: returns are i.i.d., so
    nothing observable before a decision helps predict its reward.
    """

    # Tells train.py that one update moves every portfolio's Qhat, so its
    # logging must not assume only Q(action) changed. train.py's separate
    # function-approximation path reads this flag (FA_INTEGRATION_NOTES.md).
    updates_all_actions = True

    def __init__(
        self,
        features: np.ndarray,
        *,
        alpha: float,
        epsilon0: float,
        beta: float,
        rng: np.random.Generator,
        l2: float = 0.0,
        greedy_refresh_every: int = 1,
    ) -> None:
        """
        features: F, shape (n_actions, d), from src/features.py. Row a describes
            portfolio a. Must be a floating dtype; kept as passed (float32
            saves memory on the 10% grid) and never modified.

        alpha: constant step size, > 0. SGD on this squared error is stable
            only for alpha < 2 / (max_a ||F[a]||^2 + l2).

        epsilon0, beta: the same exploration schedule as the tabular agent,
            epsilon_t = epsilon0 * exp(-beta * t).

        rng: a seeded generator from environment.py's SeedPlan, the same
            stream the tabular agent would get, so runs stay comparable.

        l2: ridge penalty, >= 0. Each update also shrinks theta by
            alpha * l2 * theta. Redundant features (the splines are knowingly
            redundant) are not by themselves a problem here: theta starts at 0
            and every step is a combination of feature rows, so theta never
            moves in directions the features cannot tell apart. What l2 does
            is damp directions the data barely informs (small eigenvalues of
            the feature covariance), where constant-alpha SGD would otherwise
            keep chasing reward noise. The price is bias toward 0.

        greedy_refresh_every: recompute the greedy portfolio after this many
            updates. 1 means the greedy is always exact. Larger values save
            time on big grids, at the cost of exploiting a greedy that is up
            to that many updates old.
        """
        features = np.asarray(features)
        if not np.issubdtype(features.dtype, np.floating):
            # An integer F would truncate theta when the greedy is scored.
            raise ValueError(f"features must have a floating dtype, got {features.dtype}")
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError(f"features must be a non-empty 2-D matrix, got shape {features.shape}")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite")
        for name, value in (("alpha", alpha), ("beta", beta), ("l2", l2), ("epsilon0", epsilon0)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta < 0 or l2 < 0:
            raise ValueError("beta and l2 must be nonnegative")
        if not 0.0 <= epsilon0 <= 1.0:
            raise ValueError("epsilon0 must be in [0, 1]")
        if isinstance(greedy_refresh_every, bool) or not isinstance(greedy_refresh_every, (int, np.integer)) \
                or greedy_refresh_every < 1:
            raise ValueError("greedy_refresh_every must be a positive integer")

        self.n_actions = features.shape[0]
        self.alpha = float(alpha)
        self.epsilon0 = float(epsilon0)
        self.beta = float(beta)
        self.l2 = float(l2)
        self.greedy_refresh_every = int(greedy_refresh_every)
        self.rng = rng
        # Same layout as TabularQAgent: ties use their own spawned stream, so
        # how often ties are broken never shifts the explore/exploit draws.
        self._tie_rng = rng.spawn(1)[0]

        self._features = features
        self._theta = np.zeros(features.shape[1])  # theta starts at 0, so every Qhat starts at 0
        self._visits = np.zeros(self.n_actions, dtype=int)
        self._t = 0  # select_action calls so far; the epsilon schedule reads this
        self._updates_since_refresh = 0
        self.last_abs_dq = 0.0  # |change in Qhat(action)| from the latest update, for train.py's log
        self._greedy = self._compute_greedy()

    @property
    def q_values(self) -> np.ndarray:
        """Qhat for every portfolio, F @ theta, as a new float64 array.

        Exact, and therefore slow on big grids: numpy upcasts a float32 F to
        float64 on every call (about 5.6 ms on the 10% grid). train.py's FA
        path reads it only at checkpoints. The greedy the agent exploits is
        scored in float32 instead, so on a near-tie it can differ from the
        lowest-index argmax of these values that evaluate.py reports."""
        return self._features @ self._theta

    @property
    def visit_counts(self) -> np.ndarray:
        """Read-only copy of how many times each action has been updated.
        Not used for learning (alpha is constant); train.py and evaluate.py
        report it."""
        return self._visits.copy()

    @property
    def theta(self) -> np.ndarray:
        """Read-only copy of the learned parameters."""
        return self._theta.copy()

    @property
    def epsilon(self) -> float:
        """The exploration rate the next select_action call will use."""
        return self._epsilon()

    @property
    def greedy_action(self) -> int:
        """The cached greedy portfolio: the highest Qhat at the last refresh,
        ties broken uniformly at random on _tie_rng.

        Reading it is free and never draws from any stream. The tie stream is
        consumed only when the greedy is recomputed, which happens inside
        update() on a fixed count, never because something read this property.
        So train.py can read it every episode without changing what the agent
        does. (TabularQAgent differs: its greedy_action draws a tie-break on
        every read when a tie exists.)"""
        return self._greedy

    # -- Internal helpers --

    def _epsilon(self) -> float:
        """epsilon_t = epsilon0 * exp(-beta * t), as in the tabular agent."""
        return self.epsilon0 * np.exp(-self.beta * self._t)

    def _compute_greedy(self) -> int:
        """Score every portfolio and return the best, ties broken at random.
        theta is cast to the feature dtype so a float32 F is not copied to
        float64 on every refresh; float32 resolves Q differences far smaller
        than the gaps between portfolios that matter here."""
        scores = self._features @ self._theta.astype(self._features.dtype, copy=False)
        tied = np.flatnonzero(scores == scores.max())
        return int(self._tie_rng.choice(tied))

    def select_action(self) -> int:
        """Epsilon-greedy, exactly as TabularQAgent.select_action: draw
        u = rng.random(); explore with rng.integers(n_actions) if u < epsilon,
        else take the cached greedy. t increases once per call, after epsilon
        was read, on every branch."""
        if self.rng.random() < self._epsilon():
            action = int(self.rng.integers(self.n_actions))  # explore
        else:
            action = self._greedy  # exploit
        self._t += 1
        return action

    def update(self, action: int, reward: float) -> None:
        """One semi-gradient step on the squared error (reward - Qhat(action))^2:

            theta <- theta + alpha * [(reward - theta . phi) * phi - l2 * theta]

        where phi = F[action]. With rho = 0 the target is the reward itself.
        In words: move theta in the direction of this portfolio's features, by
        an amount proportional to how wrong the prediction was.

        Returns None on purpose: every portfolio's Qhat changed, so train.py
        must not treat a return value as the new Q(action) alone.
        """
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} is outside 0..{self.n_actions - 1}")

        phi = self._features[action].astype(float)
        before = float(phi @ self._theta)
        error = reward - before
        if self.l2:
            self._theta += self.alpha * (error * phi - self.l2 * self._theta)
        else:
            self._theta += self.alpha * error * phi
        after = float(phi @ self._theta)
        if not np.isfinite(after):
            raise FloatingPointError(
                f"Qhat of action {action} became {after}; alpha is probably above the stability bound "
                "2 / max ||phi||^2"
            )
        self.last_abs_dq = abs(after - before)
        self._visits[action] += 1

        # Refresh on a fixed count of updates, never on reads, so the agent's
        # behaviour does not depend on who reads greedy_action and when.
        self._updates_since_refresh += 1
        if self._updates_since_refresh >= self.greedy_refresh_every:
            self._greedy = self._compute_greedy()
            self._updates_since_refresh = 0
