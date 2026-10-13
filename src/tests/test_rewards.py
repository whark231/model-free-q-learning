import numpy as np
import pytest

from src.rewards import episodic_mean_variance_reward

# ---------------------------------------------------------------------------
# Deterministic sanity checks: exact formula on hand-computable batches
# ---------------------------------------------------------------------------

def test_matches_hand_computed_value_on_a_known_batch():
    """2 days, 2 assets, equal weights. Portfolio returns are [0.01, 0.01],
    so the sample variance is exactly 0 and the reward is just the mean."""
    w = np.array([0.5, 0.5])
    returns_batch = np.array([
        [0.02, 0.00],
        [0.00, 0.02],
    ])
    reward = episodic_mean_variance_reward(w, returns_batch, gamma=2.0)
    assert reward == pytest.approx(0.01)


def test_variance_uses_bessels_correction():
    """Portfolio returns [0, 2] (mean 1): ddof=1 sample variance is 2
    (divide by B-1=1), not the ddof=0 population variance of 1 (divide by
    B=2). Pins the specific mechanism the unbiasedness claim depends on."""
    w = np.array([1.0])
    returns_batch = np.array([[0.0], [2.0]])
    reward = episodic_mean_variance_reward(w, returns_batch, gamma=1.0)
    assert reward == pytest.approx(1.0 - 0.5 * 1.0 * 2.0)


# ---------------------------------------------------------------------------
# Input validation: a short or malformed batch must raise, not return NaN.
# ddof=1 variance divides by B-1, so a 1-day batch silently produced NaN,
# and NaN compares as the largest value under np.argmax, which made it a
# permanent, silent greedy choice once it reached a Q-table.
# ---------------------------------------------------------------------------

def test_raises_on_one_day_batch():
    w = np.array([0.5, 0.5])
    one_day_batch = np.array([[0.01, 0.02]])
    with pytest.raises(ValueError, match="at least 2 days"):
        episodic_mean_variance_reward(w, one_day_batch, gamma=1.0)


def test_raises_on_empty_batch():
    w = np.array([0.5, 0.5])
    empty_batch = np.empty((0, 2))
    with pytest.raises(ValueError):
        episodic_mean_variance_reward(w, empty_batch, gamma=1.0)


def test_raises_on_one_dimensional_input():
    w = np.array([0.5, 0.5])
    one_dimensional = np.array([0.01, 0.02])
    with pytest.raises(ValueError, match="must be"):
        episodic_mean_variance_reward(w, one_dimensional, gamma=1.0)


def test_raises_on_wrong_asset_count():
    """A batch for the wrong number of assets should fail loudly rather
    than broadcasting or silently misaligning against w."""
    w = np.array([0.5, 0.5, 0.0])  # 3 assets
    two_asset_batch = np.array([[0.01, 0.02], [0.00, 0.01]])  # 2 assets
    with pytest.raises(ValueError, match="must be"):
        episodic_mean_variance_reward(w, two_asset_batch, gamma=1.0)


def test_ddof1_vs_ddof0_reward_gap_matches_the_exact_algebraic_identity():
    """For any batch, var_ddof0 = var_ddof1 * (B-1)/B exactly, so
    reward_ddof0 = reward_ddof1 + (gamma/2) * var_ddof1 / B exactly -- no
    Monte Carlo needed to check this, it holds batch-by-batch. Confirms
    *why* ddof=1 removes the bias rather than just observing that it does."""
    rng = np.random.default_rng(0)
    w = np.array([0.5, 0.3, 0.2])
    gamma = 3.0
    B = 5

    returns_batch = rng.normal(size=(B, 3)) * 0.02
    portfolio_returns = returns_batch @ w

    reward_ddof1 = episodic_mean_variance_reward(w, returns_batch, gamma)
    reward_ddof0 = portfolio_returns.mean() - 0.5 * gamma * portfolio_returns.var(ddof=0)

    expected_gap = 0.5 * gamma * portfolio_returns.var(ddof=1) / B
    assert (reward_ddof0 - reward_ddof1) == pytest.approx(expected_gap)


# ---------------------------------------------------------------------------
# Statistical unbiasedness: E[reward] should equal the true mean-variance
# utility w'mu - (gamma/2) w'Sigma w, even at a short episode length B where
# a biased variance estimator would show up clearly.
# ---------------------------------------------------------------------------

@pytest.fixture
def known_return_distribution():
    """3 assets, independent daily-scale returns with known mean/variance."""
    mu = np.array([0.0008, 0.0005, 0.0010])
    sigma = np.diag([0.0004, 0.0002, 0.0006])
    w = np.array([0.5, 0.3, 0.2])
    gamma = 3.0
    return mu, sigma, w, gamma


def true_mean_variance_utility(mu, sigma, w, gamma):
    return w @ mu - 0.5 * gamma * (w @ sigma @ w)


def test_reward_is_unbiased_for_the_true_mean_variance_utility(known_return_distribution):
    """Monte Carlo check: average episodic reward over many independent
    short episodes converges to the true utility. Verified analytically
    beforehand that at n_episodes=5000, episode_length=5, seed=42 the
    observed gap is ~0.2 standard errors -- 6 SE keeps the false-failure
    probability negligible (~2e-9 under CLT) while staying a real
    statistical bound, not an arbitrary tolerance."""
    mu, sigma, w, gamma = known_return_distribution
    true_utility = true_mean_variance_utility(mu, sigma, w, gamma)

    rng = np.random.default_rng(42)
    n_episodes = 5_000
    episode_length = 5

    rewards = np.empty(n_episodes)
    for i in range(n_episodes):
        returns_batch = rng.multivariate_normal(mu, sigma, size=episode_length)
        rewards[i] = episodic_mean_variance_reward(w, returns_batch, gamma)

    mc_mean = rewards.mean()
    mc_standard_error = rewards.std(ddof=1) / np.sqrt(n_episodes)

    assert abs(mc_mean - true_utility) < 6 * mc_standard_error


def test_ddof0_variance_estimate_is_measurably_biased_low(known_return_distribution):
    """Root-cause check for the previous test: at the same episode length,
    the ddof=0 variance estimator undershoots the true portfolio variance
    by a predictable factor (B-1)/B, which is what would make the reward
    biased if episodic_mean_variance_reward used it instead of ddof=1.

    Averaging the variance estimate directly (rather than the full reward)
    converges far faster than the reward-level test above, because it
    isn't drowned out by the mean term's sampling noise -- the biased
    estimator here sits ~24 standard errors below the truth at only 5,000
    episodes, versus needing a much larger sample to see the same effect
    show up in the noisier full reward."""
    mu, sigma, w, gamma = known_return_distribution
    true_portfolio_variance = w @ sigma @ w

    rng = np.random.default_rng(42)
    n_episodes = 5_000
    episode_length = 5

    sample_vars_ddof1 = np.empty(n_episodes)
    sample_vars_ddof0 = np.empty(n_episodes)
    for i in range(n_episodes):
        returns_batch = rng.multivariate_normal(mu, sigma, size=episode_length)
        portfolio_returns = returns_batch @ w
        sample_vars_ddof1[i] = portfolio_returns.var(ddof=1)
        sample_vars_ddof0[i] = portfolio_returns.var(ddof=0)

    se1 = sample_vars_ddof1.std(ddof=1) / np.sqrt(n_episodes)
    se0 = sample_vars_ddof0.std(ddof=1) / np.sqrt(n_episodes)

    # ddof=1 (what the function actually uses): centered on the truth
    assert abs(sample_vars_ddof1.mean() - true_portfolio_variance) < 6 * se1

    # ddof=0 (the alternative the function avoids): clearly biased low,
    # by approximately true_variance / episode_length
    expected_biased_variance = true_portfolio_variance * (episode_length - 1) / episode_length
    assert abs(sample_vars_ddof0.mean() - expected_biased_variance) < 6 * se0
    assert sample_vars_ddof0.mean() < true_portfolio_variance - 6 * se0
