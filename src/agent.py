"""The agreed interface between src/agent.py (William) and src/train.py
(Albert), proposed and agreed in REVIEW_NOTES_FOR_WILLIAM.md section 2.

In plain terms: this file is a contract, not an implementation. It lets
train.py be written against TabularQAgent's method names without guessing
them, and lets agent.py's internals change freely as long as this shape
stays the same. One training episode, from train.py's side:

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

Every method and property below raises NotImplementedError. The actual
Q-learning logic is built on feature/agent, not here.
"""

from typing import Callable, Union

import numpy as np


class TabularQAgent:
    """Stateless tabular Q-learning agent: the required baseline, Part III
    of the assignment. Stateless because the synthetic returns are i.i.d.,
    so there is nothing to condition a choice on besides what the agent has
    already learned. No state argument appears anywhere below.
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
        raise NotImplementedError

    def select_action(self) -> int:
        """Return an action index in [0, n_actions), chosen epsilon-greedy
        over the agent's current Q-values. Advances whatever counter the
        epsilon-decay schedule reads, so repeated calls make the agent
        explore less over time."""
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        """Apply one Q-learning update for the action that was actually
        taken:

            Q(action) <- Q(action) + alpha * (reward + discount * max(Q) - Q(action))

        `reward` is computed entirely outside this class. This method's only
        job is the arithmetic for the one action index it is given."""
        raise NotImplementedError

    @property
    def greedy_action(self) -> int:
        """The action with the highest current Q-value. Ties must be broken
        by a stated, reproducible rule (for example, uniformly at random via
        `rng`), not by whatever numpy's argmax happens to do by default."""
        raise NotImplementedError

    @property
    def q_values(self) -> np.ndarray:
        """Read-only copy of the current Q-table, shape (n_actions,)."""
        raise NotImplementedError

    @property
    def visit_counts(self) -> np.ndarray:
        """Read-only copy of how many times each action has been updated,
        shape (n_actions,). Needed for a 1/N(a) learning rate and for
        per-action visit logging in train.py."""
        raise NotImplementedError
