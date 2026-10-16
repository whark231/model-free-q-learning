import numpy as np
import pytest

from src.agent import TabularQAgent


def make_agent(n_actions=5, alpha_rule=0.1, epsilon0=0.0, beta=0.0, q_init=0.0, discount=0.0, seed=0):
    return TabularQAgent(
        n_actions,
        alpha_rule=alpha_rule,
        epsilon0=epsilon0,
        beta=beta,
        q_init=q_init,
        discount=discount,
        rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Construction and read-only accessors
# ---------------------------------------------------------------------------

def test_initial_q_values_are_q_init():
    agent = make_agent(n_actions=4, q_init=2.5)
    np.testing.assert_array_equal(agent.q_values, np.full(4, 2.5))


def test_initial_visit_counts_are_zero():
    agent = make_agent(n_actions=4)
    np.testing.assert_array_equal(agent.visit_counts, np.zeros(4, dtype=int))


def test_q_values_returns_a_copy_not_a_view():
    """Same bug class fixed in best_discrete_weights (REVIEW_NOTES_FOR_WILLIAM.md
    issue B): a caller mutating a returned array must not corrupt agent state."""
    agent = make_agent(n_actions=3, q_init=0.0)
    w = agent.q_values
    w += 100.0
    np.testing.assert_array_equal(agent.q_values, np.zeros(3))


def test_visit_counts_returns_a_copy_not_a_view():
    agent = make_agent(n_actions=3)
    agent.update(0, 1.0)
    counts = agent.visit_counts
    counts += 100
    np.testing.assert_array_equal(agent.visit_counts, [1, 0, 0])


# ---------------------------------------------------------------------------
# Epsilon-decay schedule: epsilon_t = epsilon0 * exp(-beta * t)
# ---------------------------------------------------------------------------

def test_epsilon_at_t_zero_equals_epsilon0():
    agent = make_agent(epsilon0=0.7, beta=0.05)
    assert agent._epsilon() == pytest.approx(0.7)


def test_epsilon_decays_exactly_with_the_agreed_formula():
    """select_action must advance t exactly once per call, on every branch,
    since this is the only thing that makes the schedule actually decay."""
    epsilon0, beta = 0.9, 0.01
    agent = make_agent(epsilon0=epsilon0, beta=beta)
    n_calls = 37
    for _ in range(n_calls):
        agent.select_action()
    assert agent._epsilon() == pytest.approx(epsilon0 * np.exp(-beta * n_calls))


# ---------------------------------------------------------------------------
# select_action: epsilon-greedy
# ---------------------------------------------------------------------------

def test_pure_exploitation_always_returns_the_greedy_action():
    agent = make_agent(n_actions=5, epsilon0=0.0, beta=0.0)
    agent._q[:] = [0.1, 0.9, 0.3, 0.2, 0.0]  # action 1 is unambiguously best
    for _ in range(50):
        assert agent.select_action() == 1


def test_pure_exploration_samples_every_action_roughly_uniformly():
    """epsilon0=1, beta=0: never decays, always explores. Over many draws,
    each of n_actions should get picked close to 1/n_actions of the time."""
    n_actions = 5
    n_draws = 20_000
    agent = make_agent(n_actions=n_actions, epsilon0=1.0, beta=0.0, seed=1)
    counts = np.zeros(n_actions, dtype=int)
    for _ in range(n_draws):
        counts[agent.select_action()] += 1

    expected = n_draws / n_actions
    # Binomial standard error per action, times 6 for a negligible false-failure rate.
    se = np.sqrt(n_draws * (1 / n_actions) * (1 - 1 / n_actions))
    assert np.all(np.abs(counts - expected) < 6 * se)


def test_epsilon_zero_never_explores_even_when_random_draw_is_near_zero():
    """rng.random() returns [0, 1), so epsilon=0 must never satisfy
    `draw < epsilon`; this pins the comparison direction isn't flipped."""
    agent = make_agent(n_actions=3, epsilon0=0.0, beta=0.0)
    agent._q[:] = [5.0, 1.0, 1.0]
    for _ in range(200):
        assert agent.select_action() == 0


# ---------------------------------------------------------------------------
# update: TD arithmetic
# ---------------------------------------------------------------------------

def test_update_matches_hand_computed_value_with_constant_alpha_and_zero_discount():
    agent = make_agent(n_actions=3, alpha_rule=0.1, q_init=2.0, discount=0.0)
    agent.update(1, reward=5.0)
    # Q(1) <- 2.0 + 0.1 * (5.0 + 0 - 2.0) = 2.3
    assert agent.q_values[1] == pytest.approx(2.3)
    # untouched actions stay at q_init
    assert agent.q_values[0] == pytest.approx(2.0)
    assert agent.q_values[2] == pytest.approx(2.0)


def test_update_td_target_uses_the_max_q_value_when_discount_is_positive():
    agent = make_agent(n_actions=3, alpha_rule=0.5, q_init=0.0, discount=0.9)
    agent._q[:] = [0.0, 4.0, 1.0]  # max is 4.0, from an action not being updated
    agent.update(0, reward=1.0)
    # td_target = 1.0 + 0.9 * 4.0 = 4.6; Q(0) <- 0.0 + 0.5 * (4.6 - 0.0) = 2.3
    assert agent.q_values[0] == pytest.approx(2.3)


def test_update_increments_visit_count_only_for_the_updated_action():
    agent = make_agent(n_actions=3)
    agent.update(0, 1.0)
    agent.update(0, 1.0)
    agent.update(2, 1.0)
    np.testing.assert_array_equal(agent.visit_counts, [2, 0, 1])


# ---------------------------------------------------------------------------
# Sample-average learning rate (alpha_rule as a callable): the standard
# rule the prospectus/review notes propose as the baseline, alpha = 1/N(a).
# ---------------------------------------------------------------------------

def test_sample_average_first_update_overwrites_q_init_exactly():
    """With alpha = 1/N(a), the first visit's alpha is 1, so the initial
    value is replaced outright rather than blended in. Noted as an expected
    property in REVIEW_NOTES_FOR_WILLIAM.md section 4."""
    agent = make_agent(n_actions=2, alpha_rule=lambda n: 1 / n, q_init=999.0)
    agent.update(0, reward=3.0)
    assert agent.q_values[0] == pytest.approx(3.0)


def test_sample_average_converges_to_the_exact_running_mean():
    agent = make_agent(n_actions=1, alpha_rule=lambda n: 1 / n, q_init=0.0)
    rewards = [3.0, 7.0, -2.0, 5.0, 1.0]
    for r in rewards:
        agent.update(0, r)
    assert agent.q_values[0] == pytest.approx(np.mean(rewards))


# ---------------------------------------------------------------------------
# greedy_action: tie-breaking
# ---------------------------------------------------------------------------

def test_greedy_action_is_the_unique_max_when_there_is_no_tie():
    agent = make_agent(n_actions=4)
    agent._q[:] = [0.1, 0.1, 9.0, 0.1]
    assert agent.greedy_action == 2


def test_greedy_action_breaks_ties_randomly_not_always_the_first_index():
    """Regression for the bug class np.argmax has by default: silently
    always returning the first tied index, which would bias the agent
    toward low action indices whenever ties occur (notably at t=0, before
    any action has been visited, when every Q-value is still q_init)."""
    agent = make_agent(n_actions=5, q_init=0.0, seed=2)  # all five tied
    seen = {agent.greedy_action for _ in range(200)}
    assert len(seen) > 1


def test_greedy_action_tiebreak_uses_a_separate_rng_stream():
    """Tie-breaking must not perturb the explore/exploit stream (agent.rng),
    or how many ties happen to occur would change unrelated exploration
    draws downstream -- a reproducibility hazard for seeded training runs."""
    baseline = make_agent(n_actions=5, seed=3)
    next_draw_without_ties = baseline.rng.random()

    agent = make_agent(n_actions=5, seed=3)
    for _ in range(25):
        agent.greedy_action
    next_draw_with_ties = agent.rng.random()

    assert next_draw_without_ties == next_draw_with_ties


# ---------------------------------------------------------------------------
# Integration: does the whole loop actually find the best arm
# ---------------------------------------------------------------------------

def test_agent_finds_the_best_arm_in_a_noisy_bandit():
    """End-to-end check with the interface used exactly as train.py will use
    it: select_action, compute a reward externally, update. True means are
    unknown to the agent; only noisy per-episode rewards are observed."""
    true_means = [0.1, 0.3, 0.5, 0.2]
    n_episodes = 20_000

    rng = np.random.default_rng(0)
    reward_rng = np.random.default_rng(1000)
    beta = -np.log(0.01) / (n_episodes / 2)  # epsilon ~0.01 at the halfway point
    agent = TabularQAgent(
        len(true_means),
        alpha_rule=lambda n: 1 / n,
        epsilon0=1.0,
        beta=beta,
        q_init=0.0,
        discount=0.0,
        rng=rng,
    )

    for _ in range(n_episodes):
        a = agent.select_action()
        reward = true_means[a] + reward_rng.normal(0.0, 1.0)
        agent.update(a, reward)

    assert agent.greedy_action == int(np.argmax(true_means))
    np.testing.assert_allclose(agent.q_values, true_means, atol=0.1)
