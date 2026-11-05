import numpy as np
import pytest

from src.action_space.grid import generate_weight_grid
from src.agent import TabularQAgent
from src.fa_agent import LinearQAgent
from src.features import onehot_features, quadratic_features


def make_agent(features, alpha=0.1, epsilon0=0.0, beta=0.0, seed=0, **kw):
    return LinearQAgent(features, alpha=alpha, epsilon0=epsilon0, beta=beta,
                        rng=np.random.default_rng(seed), **kw)


# ---------------------------------------------------------------------------
# a. With one-hot features the linear agent learns the tabular agent's Q-values
# ---------------------------------------------------------------------------

def test_onehot_features_reproduce_the_tabular_q_table_exactly():
    """phi(a) = e_a turns theta . phi(a) into theta[a], and the SGD step into
    Q(a) <- Q(a) + alpha * (r - Q(a)). Feed both agents the same (action,
    reward) sequence through update() and compare bit for bit."""
    n, alpha = 40, 0.05
    data = np.random.default_rng(7)
    actions = data.integers(n, size=5_000)
    rewards = data.normal(2e-4, 1.5e-3, size=5_000)

    fa = make_agent(onehot_features(n), alpha=alpha)
    tab = TabularQAgent(n, alpha_rule=alpha, epsilon0=0.0, beta=0.0, q_init=0.0, discount=0.0,
                        rng=np.random.default_rng(0))
    for a, r in zip(actions, rewards):
        fa.update(int(a), float(r))
        tab.update(int(a), float(r))

    np.testing.assert_array_equal(fa.q_values, tab.q_values)
    np.testing.assert_array_equal(fa.visit_counts, tab.visit_counts)


# ---------------------------------------------------------------------------
# b. Noise-free recovery with the quadratic (reference) features
# ---------------------------------------------------------------------------

def test_noise_free_rewards_recover_theta_and_the_true_best_portfolio():
    """If every reward is exactly F[a] . theta*, SGD has a zero-error solution
    and must converge to it. The 45 quadratic features have full rank on the
    20% grid (condition number about 33), so theta* is the only solution."""
    F = quadratic_features(generate_weight_grid(9, 0.2))
    theta_star = np.random.default_rng(11).normal(size=F.shape[1])
    true_q = F.astype(float) @ theta_star
    best = int(np.argmax(true_q))
    assert np.sort(true_q)[-1] - np.sort(true_q)[-2] > 1e-3  # a clear best, so the greedy check is meaningful

    agent = make_agent(F, alpha=0.5)
    picks = np.random.default_rng(3).integers(F.shape[0], size=80_000)  # uniform tries, fed directly
    for a in picks:
        agent.update(int(a), float(true_q[a]))

    np.testing.assert_allclose(agent.theta, theta_star, rtol=0, atol=1e-6)
    assert agent.greedy_action == best


def test_l2_shrinks_theta_toward_zero():
    """With a ridge penalty the fixed point is pulled toward 0, so theta ends
    up strictly shorter than the unpenalized solution on the same data."""
    F = quadratic_features(generate_weight_grid(9, 0.25))
    theta_star = np.random.default_rng(1).normal(size=F.shape[1])
    picks = np.random.default_rng(2).integers(F.shape[0], size=20_000)
    plain, ridge = make_agent(F, alpha=0.5), make_agent(F, alpha=0.5, l2=0.05)
    for a in picks:
        r = float(F[a].astype(float) @ theta_star)
        plain.update(int(a), r)
        ridge.update(int(a), r)
    assert np.linalg.norm(ridge.theta) < np.linalg.norm(plain.theta)


# ---------------------------------------------------------------------------
# c. Epsilon schedule: the counter moves once per call on both branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("epsilon0", [0.0, 0.5, 1.0])
def test_epsilon_counter_increments_once_per_select_action_on_every_branch(epsilon0):
    """epsilon0 = 0 always exploits, 1 always explores, 0.5 mixes. In every
    case n calls must leave epsilon = epsilon0 * exp(-beta * n)."""
    beta, n = 0.01, 37
    agent = make_agent(onehot_features(5), epsilon0=epsilon0, beta=beta)
    assert agent.epsilon == epsilon0
    for k in range(1, n + 1):
        agent.select_action()
        assert agent.epsilon == pytest.approx(epsilon0 * np.exp(-beta * k), rel=1e-12)


def test_first_call_uses_epsilon0_not_the_decayed_rate():
    """epsilon is read before t moves: with epsilon0 = 1 and a huge beta, the
    first call must still explore (it consumes an rng.integers draw). Seed 4
    is chosen so that reading t first gives a different action (1 instead of
    4); for some seeds the two orders happen to agree."""
    agent = make_agent(onehot_features(5), epsilon0=1.0, beta=50.0, seed=4)
    ref = np.random.default_rng(4)
    ref.spawn(1)
    ref.random()
    assert agent.select_action() == int(ref.integers(5))


# ---------------------------------------------------------------------------
# d. Cached greedy: stale between refreshes, exact at refresh 1
# ---------------------------------------------------------------------------

def test_greedy_stays_stale_until_the_refresh_count_then_catches_up():
    F = onehot_features(4)
    agent = make_agent(F, alpha=1.0, greedy_refresh_every=3)
    start = agent.greedy_action
    target = (start + 1) % 4  # some action other than the initial pick
    agent.update(target, 1.0)
    agent.update(target, 1.0)
    assert agent.greedy_action == start  # two updates: still the old greedy
    agent.update(target, 1.0)
    assert agent.greedy_action == target  # third update triggers the refresh
    other = (target + 1) % 4
    for _ in range(2):
        agent.update(other, 5.0)
    assert agent.greedy_action == target  # the counter restarted: stale again for two updates
    agent.update(other, 5.0)
    assert agent.greedy_action == other  # and refreshes on the sixth update overall


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_refresh_every_one_is_always_the_exact_current_greedy(dtype):
    """After every update the greedy must be one of the current best
    portfolios. Ties are common early (portfolios sharing no features with
    the ones updated keep equal Qhat), and are broken at random, so check
    membership in the tied set rather than np.argmax's lowest index. With
    float32 features the agent scores in float32, so "best" holds up to
    float32 rounding of the scores."""
    F = quadratic_features(generate_weight_grid(9, 0.25), dtype=dtype)
    agent = make_agent(F, alpha=0.5)
    data = np.random.default_rng(5)
    for _ in range(300):
        agent.update(int(data.integers(F.shape[0])), float(data.normal()))
        q = agent.q_values
        tol = 0.0 if dtype == np.float64 else 1e-6 * np.abs(q).max()
        assert q[agent.greedy_action] >= q.max() - tol


def test_reading_greedy_action_never_changes_what_the_agent_does():
    """train.py reads greedy_action every episode for logging. Two agents on
    the same seed, one read constantly and one never, must act identically."""
    F = onehot_features(6)  # all-zero start: a 6-way tie, so tie-breaks do happen
    quiet, watched = (make_agent(F, epsilon0=0.3, seed=9, greedy_refresh_every=2) for _ in range(2))
    data = np.random.default_rng(1)
    for _ in range(500):
        r = float(data.normal())
        a_quiet = quiet.select_action()
        for _ in range(3):
            watched.greedy_action
        a_watched = watched.select_action()
        assert a_quiet == a_watched
        quiet.update(a_quiet, r)
        watched.update(a_watched, r)


# ---------------------------------------------------------------------------
# e. Seeds
# ---------------------------------------------------------------------------

def run_actions(seed, n=2_000):
    F = quadratic_features(generate_weight_grid(9, 0.25))
    agent = make_agent(F, alpha=0.2, epsilon0=1.0, beta=1e-3, seed=seed)
    theta_star = np.random.default_rng(99).normal(size=F.shape[1])
    noise = np.random.default_rng(100)
    actions = []
    for _ in range(n):
        a = agent.select_action()
        agent.update(a, float(F[a].astype(float) @ theta_star + noise.normal()))
        actions.append(a)
    return actions


def test_same_seed_gives_the_same_action_sequence():
    assert run_actions(12) == run_actions(12)
    assert run_actions(12) != run_actions(13)


# ---------------------------------------------------------------------------
# Interface contract with train.py
# ---------------------------------------------------------------------------

def test_update_returns_none_and_flags_that_every_q_value_moves():
    """train.py reads a non-None return as "only Q(action) changed"; for this
    agent every Qhat changes, so it must return None and set the flag."""
    agent = make_agent(quadratic_features(generate_weight_grid(9, 0.25)))
    assert agent.update(0, 1.0) is None
    assert LinearQAgent.updates_all_actions is True


def test_last_abs_dq_is_the_change_in_the_chosen_actions_q():
    F = quadratic_features(generate_weight_grid(9, 0.25))
    agent = make_agent(F, alpha=0.3)
    agent.update(10, 0.5)
    before = agent.q_values[10]
    agent.update(10, 2.0)
    assert agent.last_abs_dq == pytest.approx(abs(agent.q_values[10] - before), rel=1e-12)


def test_tie_break_uses_a_separate_stream_from_exploration():
    """Same seed, one agent refreshes the greedy (and so draws tie-breaks)
    after every update, the other almost never. Exploration draws must not
    shift: with epsilon0 = 1 both explore identically."""
    F = onehot_features(8)
    often = make_agent(F, epsilon0=1.0, seed=2, greedy_refresh_every=1)
    rarely = make_agent(F, epsilon0=1.0, seed=2, greedy_refresh_every=10**9)
    for _ in range(200):
        a, b = often.select_action(), rarely.select_action()
        assert a == b
        often.update(a, 0.0)
        rarely.update(b, 0.0)


def test_integer_features_and_out_of_range_actions_are_rejected():
    with pytest.raises(ValueError, match="floating"):
        LinearQAgent(np.eye(3, dtype=int), alpha=0.1, epsilon0=0.0, beta=0.0, rng=np.random.default_rng(0))
    agent = make_agent(onehot_features(3))
    for bad in (-1, 3):
        with pytest.raises(IndexError):
            agent.update(bad, 1.0)


def test_non_finite_reward_is_rejected_and_divergence_is_caught():
    agent = make_agent(quadratic_features(generate_weight_grid(9, 0.25)), alpha=1e300)
    with pytest.raises(ValueError):
        agent.update(0, float("nan"))
    with pytest.raises(FloatingPointError, match="stability bound"), np.errstate(all="ignore"):
        for _ in range(10):
            agent.update(0, 1e300)


@pytest.mark.parametrize("kwargs", [
    dict(alpha=0.0), dict(alpha=-1.0), dict(alpha=float("nan")), dict(epsilon0=1.5), dict(beta=-1.0),
    dict(l2=-0.1), dict(greedy_refresh_every=0), dict(greedy_refresh_every=2.5),
])
def test_bad_settings_are_rejected(kwargs):
    settings = dict(alpha=0.1, epsilon0=0.5, beta=0.0)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        LinearQAgent(onehot_features(3), rng=np.random.default_rng(0), **settings)
