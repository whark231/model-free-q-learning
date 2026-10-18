"""The tabular Q-learning agent

In plain terms: one training episode, from train.py's side, looks like:

    a = agent.select_action()
    w = grid[a]
    batch = sim.sample_episode(B)                        # environment.py
    r = episodic_mean_variance_reward(w, batch, gamma)    # rewards.py
    agent.update(a, r)

The agent only ever sees an action index and a scalar reward. It never sees
portfolio weights, returns, mu, Sigma, or a benchmark value, since train.py
owns all of that. That is the leakage boundary the assignment requires, and
test_environment.py's leakage tests already check that this file does not
import the benchmarks, the data loader, or the raw data.
"""

from typing import Callable, Union

import numpy as np


class TabularQAgent:
    """Stateless tabular Q-learning agent. Stateless because the synthetic
    returns are i.i.d., so there is nothing to condition a choice on besides
    what the agent has already learned. No state argument appears anywhere
    below.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        alpha_rule: Union[float, Callable[[int], float]],
        epsilon0: float,
        beta: float,
        q_init: float,
        discount: float,
        rng: np.random.Generator,
    ) -> None:
        """
        n_actions: size of the fixed action grid (src/action_space/grid.py).

        alpha_rule: the learning rate. A float for a constant rate, or a
            callable that takes an action's visit count and returns alpha,
            for a rule like sample averaging (alpha = 1 / n). The assignment
            asks to report and justify "whether [the learning rate] changes
            with action visits"; this parameter is that choice, stated
            explicitly rather than hardcoded.

        epsilon0: initial exploration rate, 0 <= epsilon0 <= 1.

        beta: exploration-decay parameter. epsilon_t = epsilon0 * exp(-beta * t),
            one of the schedules the assignment permits.

        q_init: initial value for every action's Q-value.

        discount: rho in the Q-learning update. The assignment's "clean
            bandit-like benchmark" for this stateless problem is rho = 0,
            since max_a' Q(a') is the same number added to every action's
            target regardless of which action is being updated, so a
            positive rho changes the overall scale of the Q-values but not
            which action looks best. Kept as an explicit argument so a
            choice other than 0 is a stated decision, not an accident.

        rng: a seeded generator, not a raw seed. Seed management (which
            stream goes to the agent versus the environment) lives in
            environment.py's SeedPlan, not here.
        """
        self.n_actions = n_actions
        self.alpha_rule = alpha_rule
        self.epsilon0 = epsilon0
        self.beta = beta
        self.discount = discount
        self.rng = rng
        self._tie_rng = rng.spawn(1)[0]

        self._q = np.full(n_actions, q_init, dtype=float)
        self._visits = np.zeros(n_actions, dtype=int)
        self._t = 0  # number of select_action calls so far; the epsilon schedule reads this

    @property
    def q_values(self) -> np.ndarray:
        """Read-only copy of the current Q-table, shape (n_actions,)."""
        return self._q.copy()

    @property
    def visit_counts(self) -> np.ndarray:
        """Read-only copy of how many times each action has been updated,
        shape (n_actions,). Needed for a 1/N(a) learning rate and for
        per-action visit logging in train.py."""
        return self._visits.copy()

    # -- Internal helpers --

    def _epsilon(self) -> float:
        """Current exploration rate, from the schedule agreed with Albert:
        epsilon_t = epsilon0 * exp(-beta * t)."""
        return self.epsilon0 * np.exp(-self.beta * self._t)

    def _alpha(self, action: int) -> float:
        """Learning rate to use for updating `action` right now.
        """
        if callable(self.alpha_rule):
            # visits so far + 1 = the visit about to happen, so the first update uses n = 1
            return float(self.alpha_rule(int(self._visits[action]) + 1))
        return float(self.alpha_rule)


    def select_action(self) -> int:
        """Epsilon-greedy choice over self._q.

        1. draw u = self.rng.random()
        2. if u < self._epsilon(): return a uniformly random action index
           (self.rng.integers(self.n_actions))
        3. otherwise return self.greedy_action (don't reimplement the
           tie-break here; greedy_action owns that rule)
        4. increment self._t exactly once per call, on every branch,
           since that's what the epsilon schedule reads
        """
        if self.rng.random() < self._epsilon():
            action = int(self.rng.integers(self.n_actions))  # explore
        else:
            action = self.greedy_action  # exploit
        self._t += 1  # once per call, after epsilon was read, on every branch
        return action

    def update(self, action: int, reward: float) -> None:
        """The Q-learning update for the action that was actually taken:

            Q(action) <- Q(action) + alpha * (reward + discount * max(Q) - Q(action))

        Rejects a non-finite reward outright rather than letting it poison
        this action's Q-value. A NaN Q-value doesn't fail here: self._q.max()
        becomes NaN, every tie-break comparison in greedy_action becomes
        False, and flatnonzero returns empty, so the actual crash happens on
        some later, unrelated call with no indication it started here.
        """
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")

        alpha = self._alpha(action)
        td_target = reward + self.discount * self._q.max()
        self._q[action] += alpha * (td_target - self._q[action])
        self._visits[action] += 1

    @property
    def greedy_action(self) -> int:
        """The action with the highest current Q-value, ties broken
        uniformly at random rather than by numpy's silent first-index rule."""
        tied = np.flatnonzero(self._q == self._q.max())
        return int(self._tie_rng.choice(tied))