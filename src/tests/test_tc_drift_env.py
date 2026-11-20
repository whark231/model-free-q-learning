"""Tests for src/tc_drift/env.py (Albert).

In plain terms: check that the three-asset market is the right slice of the
calibration, that the returns are the baseline's draws with the right moments,
and that one block is scored correctly. The drift scoring is checked against a
loop that counts shares and dollars day by day, a different derivation from
the cumulative-product code, and against William's sizing loop in
design_numbers.py. With drift off, the reward must equal the baseline's
episodic_mean_variance_reward.
"""

import numpy as np
import pytest

from src.data.universe import TICKERS
from src.environment import ReturnSimulator, load_calibration
from src.rewards import episodic_mean_variance_reward
from src.tc_drift import env
from src.tc_drift.design_numbers import buy_and_hold_block
from src.tc_drift.spec import ASSETS, BLOCK_LENGTH, Market, target_grid

GAMMA = 5.0


@pytest.fixture(scope="module")
def market():
    return env.load_market()


def _paths(market, m, seed=0, B=BLOCK_LENGTH):
    return env.draw_returns(market, np.random.default_rng(seed), m, B)


def _share_loop(w0, e, gamma):
    """Hold $1 split as w0, buy that many shares at price 1, and walk the prices
    forward day by day. Each day's portfolio excess return is the dollar change
    over the dollar value at the start of the day."""
    shares = np.array(w0, dtype=float)
    prices = np.ones(len(w0))
    port = []
    for day in e:
        start = shares @ prices
        prices = prices * (1.0 + day)
        port.append((shares @ prices) / start - 1.0)
    port = np.array(port)
    end_value = shares * prices
    return port.mean() - gamma / 2 * port.var(ddof=1), end_value / end_value.sum()


# ---------------------------------------------------------------------------
# load_market
# ---------------------------------------------------------------------------

def test_market_is_the_spy_ief_gld_slice_of_the_calibration(market):
    mu9, sigma9 = load_calibration()
    pos = [TICKERS.index(t) for t in ASSETS]
    assert ASSETS == ("SPY", "IEF", "GLD")
    np.testing.assert_array_equal(market.mu, mu9[pos])
    for a, i in enumerate(pos):
        for b, j in enumerate(pos):
            assert market.sigma[a, b] == sigma9[i, j]
    assert market.mu.shape == (3,) and market.sigma.shape == (3, 3)
    np.testing.assert_array_equal(market.sigma, market.sigma.T)
    assert np.all(np.linalg.eigvalsh(market.sigma) > 0)


def test_market_arrays_are_read_only(market):
    with pytest.raises(ValueError):
        market.mu[0] = 1.0
    with pytest.raises(ValueError):
        market.sigma[0, 0] = 1.0


# ---------------------------------------------------------------------------
# draw_returns
# ---------------------------------------------------------------------------

def test_draws_are_the_baseline_simulators_draws(market):
    a = env.draw_returns(market, np.random.default_rng(7), 5, 63)
    b = ReturnSimulator(market.mu, market.sigma, np.random.default_rng(7)).sample_episodes(5, 63)
    assert a.shape == (5, 63, 3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, env.draw_returns(market, np.random.default_rng(7), 5, 63))


def test_draws_have_the_calibrated_mean_and_covariance(market):
    e = _paths(market, 4000, seed=1).reshape(-1, 3)    # 252,000 days
    n = len(e)
    se_mu = np.sqrt(np.diag(market.sigma) / n)
    assert np.all(np.abs(e.mean(axis=0) - market.mu) < 4 * se_mu)
    cov = np.cov(e, rowvar=False)
    # SE of a sample covariance entry under normality: sqrt((s_ij^2 + s_ii s_jj) / n)
    s = market.sigma
    se_cov = np.sqrt((s**2 + np.outer(np.diag(s), np.diag(s))) / n)
    assert np.all(np.abs(cov - s) < 4 * se_cov)


@pytest.mark.parametrize("kwargs, error", [
    ({"m": 0}, ValueError), ({"m": 2.0}, TypeError), ({"block_length": 1}, ValueError),
])
def test_draw_returns_rejects_bad_counts(market, kwargs, error):
    args = {"m": 2, "block_length": 63, **kwargs}
    with pytest.raises(error):
        env.draw_returns(market, np.random.default_rng(0), **args)


def test_draw_returns_rejects_an_unseeded_or_legacy_generator(market):
    with pytest.raises(TypeError):
        env.draw_returns(market, np.random.RandomState(0), 2, 63)


# ---------------------------------------------------------------------------
# block_outcome, no drift
# ---------------------------------------------------------------------------

def test_no_drift_reward_equals_the_baseline_reward_row_by_row(market):
    e = _paths(market, 200, seed=2)
    rng = np.random.default_rng(3)
    w0 = target_grid()[rng.integers(0, len(target_grid()), 200)]
    reward, w_end = env.block_outcome(w0, e, GAMMA, drift=False)
    expected = np.array([episodic_mean_variance_reward(w, b, GAMMA) for w, b in zip(w0, e)])
    np.testing.assert_allclose(reward, expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(w_end, w0)


def test_no_drift_end_weights_are_a_copy(market):
    w0 = np.array([[0.5, 0.2, 0.3]])
    _, w_end = env.block_outcome(w0, _paths(market, 1), GAMMA, drift=False)
    w_end[0, 0] = 9.0
    assert w0[0, 0] == 0.5


# ---------------------------------------------------------------------------
# block_outcome, drift
# ---------------------------------------------------------------------------

def test_drift_matches_a_share_counting_loop(market):
    e = _paths(market, 50, seed=4)
    rng = np.random.default_rng(5)
    w0 = rng.dirichlet(np.ones(3), size=50)
    reward, w_end = env.block_outcome(w0, e, GAMMA, drift=True)
    for k in range(50):
        r_k, w_k = _share_loop(w0[k], e[k], GAMMA)
        assert reward[k] == pytest.approx(r_k, rel=1e-10, abs=1e-16)
        np.testing.assert_allclose(w_end[k], w_k, rtol=1e-12, atol=1e-15)


def test_drift_matches_williams_reference_loop(market):
    """design_numbers.buy_and_hold_block with rf = 0 is the same model written
    as a loop; the cross-test the design asks for (§10)."""
    e = _paths(market, 300, seed=6)
    w0 = np.array([0.4688, 0.1363, 0.3949])
    w0 = w0 / w0.sum()
    reward, w_end = env.block_outcome(w0, e, GAMMA, drift=True)
    r_ref, w_ref = buy_and_hold_block(w0, e, GAMMA, 0.0)
    np.testing.assert_allclose(reward, r_ref, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(w_end, w_ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("corner", [0, 1, 2])
def test_a_single_asset_portfolio_does_not_drift(market, corner):
    w0 = np.eye(3)[corner]
    e = _paths(market, 20, seed=8)
    r_drift, w_drift = env.block_outcome(w0, e, GAMMA, drift=True)
    r_fixed, _ = env.block_outcome(w0, e, GAMMA, drift=False)
    np.testing.assert_allclose(r_drift, r_fixed, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(w_drift, np.repeat(w0[None, :], 20, axis=0))


def test_drifted_weights_are_valid_holdings_and_move_toward_the_winner(market):
    e = np.zeros((1, 2, 3))
    e[0, 0] = [0.10, 0.0, -0.10]                 # SPY up 10%, GLD down 10%, then a flat day
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    reward, w_end = env.block_outcome(w0, e, 0.0, drift=True)
    np.testing.assert_allclose(w_end[0], np.array([1.1, 1.0, 0.9]) / 3.0)
    # Day 1 return 0 (equal weights, +10% and -10%); day 2 return 0 (flat). Mean 0.
    assert reward[0] == pytest.approx(0.0, abs=1e-15)
    e = _paths(market, 500, seed=9)
    _, w_end = env.block_outcome(np.full(3, 1 / 3), e, GAMMA, drift=True)
    assert (w_end >= 0).all()
    np.testing.assert_allclose(w_end.sum(axis=1), 1.0, atol=1e-12)


def test_design_numbers_reproduce_with_this_environment(market):
    """§12: at the gamma = 5 optimum the buy-and-hold block reward SD is about
    2,390 bps/yr, the constant-weight SD about 2,385, their difference on the
    same paths about 103, and one block's one-way turnover a median 2.26%.
    Different random numbers from design_numbers, so check to about 3%."""
    from src.benchmarks.mean_variance import constrained_optimal_weights
    w_star = constrained_optimal_weights(market.mu, market.sigma, GAMMA)
    e = _paths(market, 20_000, seed=10)
    r_bh, w_end = env.block_outcome(w_star, e, GAMMA, drift=True)
    r_c, _ = env.block_outcome(w_star, e, GAMMA, drift=False)
    to_bps = 252 * 1e4
    assert r_bh.std() * to_bps == pytest.approx(2390, rel=0.03)
    assert r_c.std() * to_bps == pytest.approx(2385, rel=0.03)
    assert (r_bh - r_c).std() * to_bps == pytest.approx(103, rel=0.05)
    turnover = 0.5 * np.abs(w_end - w_star).sum(axis=1)
    assert np.median(turnover) * 100 == pytest.approx(2.26, rel=0.05)
    # The control variate's mean: E[R_const] = U(w*) exactly; the drift correction is under 3 bps/yr.
    utility = w_star @ market.mu - GAMMA / 2 * w_star @ market.sigma @ w_star
    se = r_c.std() / np.sqrt(len(r_c))
    assert abs(r_c.mean() - utility) < 4 * se
    assert abs((r_bh - r_c).mean()) * to_bps < 3


# ---------------------------------------------------------------------------
# Shapes, broadcasting and validation
# ---------------------------------------------------------------------------

def test_one_portfolio_is_scored_on_every_path(market):
    e = _paths(market, 7, seed=11)
    w0 = np.array([0.2, 0.5, 0.3])
    for drift in (True, False):
        r1, w1 = env.block_outcome(w0, e, GAMMA, drift)
        r2, w2 = env.block_outcome(np.repeat(w0[None, :], 7, axis=0), e, GAMMA, drift)
        assert r1.shape == (7,) and w1.shape == (7, 3)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize("w0, e_shape, kwargs, error", [
    (np.full((3, 3), 1 / 3), (2, 63, 3), {}, ValueError),            # rows do not match paths
    (np.array([0.5, 0.5]), (2, 63, 3), {}, ValueError),              # two assets
    (np.array([0.6, 0.6, -0.2]), (2, 63, 3), {}, ValueError),        # short position
    (np.array([0.5, 0.2, 0.2]), (2, 63, 3), {}, ValueError),         # sums to 0.9
    (np.full(3, 1 / 3), (2, 63), {}, ValueError),                    # returns not 3-D
    (np.full(3, 1 / 3), (2, 1, 3), {}, ValueError),                  # one day: no ddof=1 variance
    (np.full(3, 1 / 3), (2, 63, 3), {"gamma": -1.0}, ValueError),
    (np.full(3, 1 / 3), (2, 63, 3), {"gamma": np.nan}, ValueError),
    (np.full(3, 1 / 3), (2, 63, 3), {"drift": 1}, TypeError),
    (np.full(3, 1 / 3), (2, 63, 3), {"drift": None}, TypeError),
])
def test_block_outcome_rejects_bad_inputs(w0, e_shape, kwargs, error):
    args = {"gamma": GAMMA, "drift": True, **kwargs}
    with pytest.raises(error):
        env.block_outcome(w0, np.zeros(e_shape), **args)


def test_block_outcome_rejects_non_finite_and_wipeout_returns():
    e = np.zeros((1, 63, 3))
    e[0, 5, 1] = np.nan
    with pytest.raises(ValueError):
        env.block_outcome(np.full(3, 1 / 3), e, GAMMA, drift=False)
    e[0, 5, 1] = -1.0
    with pytest.raises(ValueError):
        env.block_outcome(np.full(3, 1 / 3), e, GAMMA, drift=True)


def test_simulate_block_is_draw_then_score(market):
    w0 = target_grid()[[0, 100, 700]]
    r, w = env.simulate_block(w0, np.random.default_rng(12), 63, True, market=market, gamma=GAMMA)
    e = env.draw_returns(market, np.random.default_rng(12), 3, 63)
    r_ref, w_ref = env.block_outcome(w0, e, GAMMA, drift=True)
    np.testing.assert_array_equal(r, r_ref)
    np.testing.assert_array_equal(w, w_ref)


def test_a_hand_built_market_works(market):
    """The functions take any Market, so tests and the oracle can use toy moments."""
    toy = Market(mu=np.array([1e-4, 2e-4, 3e-4]), sigma=np.diag([1e-4, 2e-4, 3e-4]))
    r, w = env.simulate_block(np.full(3, 1 / 3), np.random.default_rng(13), 10, True, market=toy, gamma=1.0)
    assert r.shape == (1,) and w.shape == (1, 3)


def test_hand_case_second_day_uses_drifted_weights():
    """Day 1 moves the assets +10%, 0, -10%; day 2 moves only the first by 1%.
    By hand: after day 1 the values are 1.1/3, 1/3, 0.9/3 out of 1, so day 2's
    portfolio return is (1.1/3) * 0.01, not the equal-weight 0.01/3."""
    w0 = np.full(3, 1 / 3)
    e = np.array([[[0.10, 0.0, -0.10], [0.01, 0.0, 0.0]]])
    reward, w_end = env.block_outcome(w0, e, 0.0, drift=True)
    day1, day2 = 0.0, (1.1 / 3) * 0.01
    assert reward[0] == pytest.approx((day1 + day2) / 2, rel=1e-12)
    end = np.array([1.1 * 1.01, 1.0, 0.9]) / 3
    np.testing.assert_allclose(w_end[0], end / end.sum(), rtol=1e-12)


def test_end_weights_are_valid_holdings_for_the_lattice(market):
    """The oracle maps w_end to its lattice, so w_end must pass spec's checks."""
    from src.tc_drift.spec import LATTICE_INCREMENT, nearest_lattice_index
    w0 = target_grid()[np.random.default_rng(3).integers(0, 1326, 500)]
    _, w_end = env.block_outcome(w0, _paths(market, 500, seed=4), GAMMA, drift=True)
    idx = nearest_lattice_index(w_end)
    n = round(1 / LATTICE_INCREMENT)
    assert idx.shape == (500,) and idx.min() >= 0 and idx.max() < (n + 1) * (n + 2) // 2


def test_no_drift_broadcast_returns_a_writeable_full_copy(market):
    w0 = np.array([0.5, 0.3, 0.2])
    _, w_end = env.block_outcome(w0, _paths(market, 7, seed=5), GAMMA, drift=False)
    assert w_end.shape == (7, 3) and w_end.flags.writeable and w_end.flags.owndata
    w_end[0, 0] = 9.0
    np.testing.assert_array_equal(w0, [0.5, 0.3, 0.2])
    assert w_end[1, 0] == 0.5
