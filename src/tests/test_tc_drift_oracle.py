"""Tests for the transaction-cost-with-drift oracle (src/tc_drift/oracle.py).

In plain terms: the oracle is checked against things derived a different way.
- Check 1 (TC_DRIFT_DESIGN.md §5): with drift off it must agree with the old
  grid solver, src.transaction_costs.optimal_policy, on the 2% grid.
- Check 2: with zero cost the answer is known in closed form (drift off) or
  bounded below (drift on).
- Every value, cost, reward estimate and next state is recomputed for a sample
  of states with independent code: spec.trade_cost for costs, a day-by-day
  loop for the block, brute-force nearest points for the rounding, and a
  gather instead of the sparse transition matrix.

The block simulator here is a plain day-by-day loop written for these tests,
so the oracle is tested without Albert's env.block_outcome (§10, "working
without blocking"). Checks 3 to 5 (path count, lattice size, continuous
replay) run at production size and are not unit tests.
"""

import numpy as np
import pytest

from src.environment import ReturnSimulator, load_calibration
from src.rewards import episodic_mean_variance_reward
from src.tc_drift import calibration, env, oracle, spec

B = spec.BLOCK_LENGTH
RHO = spec.DISCOUNT


# ---------------------------------------------------------------------------
# Reference block simulator and fixtures
# ---------------------------------------------------------------------------

def loop_block_outcome(w0, excess_returns, gamma, drift):
    """Day-by-day reference for env.block_outcome: hold fixed shares (drift)
    or fixed weights (no drift), score mean - (gamma / 2) var (ddof = 1)."""
    e = np.asarray(excess_returns, dtype=float)
    m, n_days, _ = e.shape
    w = np.array(np.broadcast_to(np.atleast_2d(w0), (m, 3)), dtype=float)
    daily = np.empty((m, n_days))
    for t in range(n_days):
        daily[:, t] = (w * e[:, t]).sum(axis=1)
        if drift:
            w = w * (1.0 + e[:, t])
            w = w / w.sum(axis=1, keepdims=True)
    return daily.mean(axis=1) - 0.5 * gamma * daily.var(axis=1, ddof=1), w


@pytest.fixture(scope="module")
def market():
    mu9, sigma9 = load_calibration()
    idx = list(spec.ASSET_INDEX)
    return spec.Market(mu=mu9[idx], sigma=sigma9[np.ix_(idx, idx)])


def fixed_paths(market, m):
    rng = np.random.default_rng(spec.ORACLE_PATH_SEED)
    return ReturnSimulator(market.mu, market.sigma, rng).sample_episodes(m, B)


@pytest.fixture(scope="module")
def paths(market):
    return fixed_paths(market, 24)


@pytest.fixture(scope="module")
def headline():
    return calibration.cost_settings()["aum_1b"]


@pytest.fixture(scope="module")
def drift_solution(market, paths, headline):
    """The headline setting with drift, on the 1% lattice and 24 paths."""
    return oracle.solve(market, headline, paths, block_outcome=loop_block_outcome)


def _brute_nearest(points, lattice_points):
    d = ((points[:, None, :] - lattice_points[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1), np.sort(d, axis=1)[:, :2]


# ---------------------------------------------------------------------------
# The reference simulator itself
# ---------------------------------------------------------------------------

def test_reference_loop_without_drift_is_the_baseline_reward(market, paths):
    w = spec.target_grid()[[0, 500, 1325]]
    for x in w:
        reward, w_end = loop_block_outcome(x, paths, 5.0, False)
        expected = [episodic_mean_variance_reward(x, paths[i], 5.0) for i in range(len(paths))]
        np.testing.assert_allclose(reward, expected, rtol=1e-12, atol=1e-18)
        np.testing.assert_array_equal(w_end, np.tile(x, (len(paths), 1)))


def test_reference_loop_drift_by_hand():
    e = np.array([[[0.10, 0.00, -0.05], [0.00, 0.02, 0.01]]])
    reward, w_end = loop_block_outcome(np.array([0.5, 0.3, 0.2]), e, 2.0, True)
    day1 = 0.5 * 0.10 + 0.2 * -0.05
    v = np.array([0.55, 0.3, 0.19])                 # values after day 1
    day2 = (v / v.sum()) @ e[0, 1]
    end = v * np.array([1.0, 1.02, 1.01])
    expected = np.mean([day1, day2]) - np.var([day1, day2], ddof=1)
    np.testing.assert_allclose(reward, [expected], rtol=1e-14)
    np.testing.assert_allclose(w_end[0], end / end.sum(), rtol=1e-14)


# ---------------------------------------------------------------------------
# Check 1: no drift equals the old grid oracle
# ---------------------------------------------------------------------------

# Without drift a grid state never leaves the grid, so the 2% lattice gives the
# same comparison faster; the headline level also runs on the 1% lattice.
@pytest.mark.parametrize("level, increment", [("aum_1b", 0.01), ("aum_10b", 0.02), ("steep", 0.02)])
def test_check1_no_drift_matches_the_old_grid_oracle(market, level, increment):
    from src.transaction_costs import optimal_policy  # read-only import of the old design's solver

    # "steep": a 20-point transfer costs 2.4% of wealth. Without drift a trade pays
    # off forever, so only a cost this large makes HOLD common enough to test.
    settings = (spec.TCSettings(kappa=0.05, lam=0.1) if level == "steep"
                else calibration.cost_settings()[level])
    sol = oracle.solve(market, settings, fixed_paths(market, 4), block_outcome=loop_block_outcome, drift=False,
                       lattice_increment=increment)
    n_states = len(spec.lattice(increment))

    # No drift: r_hat is U exactly (one call scored both rewards) and the state never moves.
    np.testing.assert_array_equal(sol.cv_correction, 0.0)
    np.testing.assert_array_equal(sol.r_hat, sol.U)
    np.testing.assert_array_equal(sol.next_index, np.repeat(np.arange(n_states)[:, None], 4, axis=1))

    grid = spec.target_grid()
    tidx = spec.target_lattice_index(increment)
    n = len(grid)
    u_grid = grid @ market.mu - 0.5 * settings.gamma * np.einsum("ni,ij,nj->n", grid, market.sigma, grid)
    np.testing.assert_allclose(sol.U[tidx], u_grid, rtol=0, atol=1e-18)
    actions = np.arange(n)
    daily_cost = np.stack([spec.trade_cost(np.tile(s, (n, 1)), actions, settings) for s in grid]) / B

    old_policy = optimal_policy(u_grid, daily_cost, RHO)
    # The old solver returns only a policy; evaluate it with the same stopping rule.
    v_old = np.zeros(n)
    step = u_grid[old_policy] - daily_cost[actions, old_policy]
    while True:
        updated = step + RHO * v_old[old_policy]
        if np.max(np.abs(updated - v_old)) <= 1e-12 * (1 - RHO):
            v_old = updated
            break
        v_old = updated

    np.testing.assert_allclose(sol.V[tidx], v_old, rtol=0, atol=1e-11)

    # Policies by post-trade weights (HOLD = stay), skipping states whose best two
    # distinct destinations are within the tolerance.
    q = (u_grid + RHO * v_old)[None, :] - daily_cost
    top2 = np.sort(q, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1e-11
    new_table = sol.policy_table[tidx]
    new_landing = np.where(new_table == spec.HOLD, tidx, tidx[np.where(new_table == spec.HOLD, 0, new_table)])
    assert np.array_equal(new_landing[clear], tidx[old_policy][clear])
    assert clear.mean() > 0.99
    if level == "steep":
        assert (new_table == spec.HOLD).sum() > 10   # the comparison exercises HOLD, not only trades


# ---------------------------------------------------------------------------
# Check 2: zero cost
# ---------------------------------------------------------------------------

def test_check2_zero_cost_without_drift_is_exact(market):
    sol = oracle.solve(market, spec.TCSettings(0.0, 0.0), fixed_paths(market, 4), block_outcome=loop_block_outcome,
                       drift=False)
    u_star = sol.U[sol.target_index].max()
    np.testing.assert_allclose(sol.V, np.maximum(sol.U, u_star) / (1 - RHO), rtol=0, atol=1e-11)
    # An off-grid lattice start better than every target is kept by HOLD.
    better = sol.U > u_star
    assert better.any()
    assert (sol.policy_table[better] == spec.HOLD).all()


def test_check2_zero_cost_with_drift_beats_rebalancing_to_the_best_target(market, paths):
    sol = oracle.solve(market, spec.TCSettings(0.0, 0.0), paths, block_outcome=loop_block_outcome)
    rebalance = sol.r_hat[sol.target_index].max() / (1 - RHO)
    assert sol.V.min() >= rebalance - 1e-12
    # How close: with free trading the oracle's lead over always rebalancing
    # comes only from drift landing on a better lattice point. At M = 2,000
    # (TC_DRIFT_DESIGN.md section 5) the median lead is 0.008 bps/yr.
    lead_bps = spec.annualized_bps(sol.V - rebalance)
    assert np.median(lead_bps) < 0.5


# ---------------------------------------------------------------------------
# The solve recomputed independently on a sample of states
# ---------------------------------------------------------------------------

def test_bellman_equation_holds_with_independent_costs(drift_solution):
    sol = drift_solution
    G_check = sol.r_hat + RHO * sol.V[sol.next_index].mean(axis=1)   # gather, not the sparse matrix
    # G comes from the last sweep's V, so the two differ by at most the stopping tolerance.
    np.testing.assert_allclose(sol.G, G_check, rtol=0, atol=oracle.STOP_FACTOR * (1 - RHO))

    states = spec.lattice()
    rng = np.random.default_rng(0)
    sample = np.concatenate([sol.target_index[rng.choice(spec.N_TARGETS, 100, replace=False)],
                             rng.choice(spec.N_LATTICE, 200, replace=False)])
    actions = np.arange(spec.N_ACTIONS)
    for s in sample:
        cost = spec.trade_cost(np.tile(states[s], (spec.N_ACTIONS, 1)), actions, sol.settings)
        q = np.append(sol.G[sol.target_index], sol.G[s]) - cost / B
        assert sol.V[s] == pytest.approx(q.max(), abs=1e-15)
        a = sol.policy_table[s]
        assert q[a] == pytest.approx(q.max(), abs=1e-15)
        if q[spec.HOLD] >= q[:spec.N_TARGETS].max():
            assert a == spec.HOLD   # HOLD wins ties


def test_cost_matrix_matches_spec_trade_cost(drift_solution):
    states = spec.lattice()
    for a in [0, 1, 37, 600, 1325]:
        expected = spec.trade_cost(states, np.full(spec.N_LATTICE, a), drift_solution.settings)
        np.testing.assert_allclose(drift_solution.cost[:, a], expected, rtol=1e-12, atol=1e-20)


def test_reward_estimate_and_next_states_recomputed(market, paths, drift_solution):
    sol = drift_solution
    states = spec.lattice()
    gamma = sol.settings.gamma
    for i in [0, 17, 1000, 2600, 4321, spec.N_LATTICE - 1]:
        x = states[i]
        r_bh, w_end = loop_block_outcome(x, paths, gamma, True)
        r_const, _ = loop_block_outcome(x, paths, gamma, False)
        u = x @ market.mu - 0.5 * gamma * x @ market.sigma @ x
        assert sol.U[i] == pytest.approx(u, rel=1e-12)
        assert sol.cv_correction[i] == pytest.approx(np.mean(r_bh - r_const), rel=1e-10, abs=1e-18)
        assert sol.r_hat[i] == pytest.approx(u + np.mean(r_bh - r_const), rel=1e-12)
        nearest, two = _brute_nearest(w_end, states)
        unambiguous = (two[:, 1] - two[:, 0]) > 1e-12
        assert np.array_equal(sol.next_index[i][unambiguous], nearest[unambiguous])


def test_control_variate_is_exact_for_a_single_asset(drift_solution):
    # A one-asset portfolio does not drift, so buy-and-hold equals constant weights.
    corners = [spec.nearest_lattice_index(np.eye(3)[k]) for k in range(3)]
    np.testing.assert_array_equal(drift_solution.cv_correction[corners], 0.0)
    assert all((drift_solution.next_index[c] == c).all() for c in corners)


def test_never_trades_to_the_point_it_already_holds(drift_solution):
    sol = drift_solution
    trades = sol.policy_table != spec.HOLD
    landing = sol.target_index[sol.policy_table[trades]]
    assert not np.any(landing == np.flatnonzero(trades))


# ---------------------------------------------------------------------------
# value_of, the myopic reference and policy()
# ---------------------------------------------------------------------------

def test_value_of_the_optimal_policy_is_v(drift_solution):
    np.testing.assert_allclose(drift_solution.value_of(drift_solution.policy_table), drift_solution.V,
                               rtol=0, atol=1e-12)


def test_value_of_matches_a_hand_rolled_evaluation(drift_solution):
    sol = drift_solution
    table = np.full(spec.N_LATTICE, spec.HOLD)
    table[::3] = 7                       # an arbitrary mixed policy
    n = spec.N_LATTICE
    landing = np.where(table == spec.HOLD, np.arange(n), sol.target_index[np.where(table == spec.HOLD, 0, table)])
    cost = spec.trade_cost(spec.lattice(), table, sol.settings) / B
    V = np.zeros(n)
    for _ in range(2000):
        V = sol.r_hat[landing] + RHO * V[sol.next_index[landing]].mean(axis=1) - cost
    np.testing.assert_allclose(sol.value_of(table), V, rtol=0, atol=1e-12)


def test_myopic_reference(market, paths, headline, drift_solution):
    myopic = oracle.solve(market, headline, paths, block_outcome=loop_block_outcome, discount=0.0)
    # rho = 0: V is the best immediate net reward, G is r_hat.
    np.testing.assert_array_equal(myopic.G, myopic.r_hat)
    trades = myopic.r_hat[myopic.target_index][None, :] - myopic.cost / B
    np.testing.assert_allclose(myopic.V, np.maximum(myopic.r_hat, trades.max(axis=1)), rtol=0, atol=1e-18)
    # Scored at rho = 0.95 on the same paths, it cannot beat the oracle.
    scored = drift_solution.value_of(myopic.policy_table, discount=RHO)
    assert (scored <= drift_solution.V + 1e-12).all()
    assert (scored < drift_solution.V - 1e-9).any()


def test_policy_lookup(drift_solution):
    sol = drift_solution
    states = spec.lattice()
    assert isinstance(sol.policy(states[10]), int)
    np.testing.assert_array_equal(sol.policy(states), sol.policy_table)
    off = np.array([0.4712, 0.1391, 0.3897])
    assert sol.policy(off) == sol.policy_table[spec.nearest_lattice_index(off)]


def test_common_random_numbers_across_settings(market, paths, drift_solution):
    other = oracle.solve(market, calibration.cost_settings()["aum_10b"], paths, block_outcome=loop_block_outcome,
                         lattice_increment=0.02)
    # The reward estimates and next states depend on the paths, not on the cost setting.
    coarse = spec.nearest_lattice_index(spec.lattice(0.02))
    np.testing.assert_array_equal(other.r_hat, drift_solution.r_hat[coarse])


# ---------------------------------------------------------------------------
# Cross-test with Albert's environment (§10: whoever merges second owns it)
# ---------------------------------------------------------------------------

def test_env_draws_the_same_market_and_paths_as_these_tests(market, paths):
    m = env.load_market()
    np.testing.assert_array_equal(m.mu, market.mu)
    np.testing.assert_array_equal(m.sigma, market.sigma)
    np.testing.assert_array_equal(env.draw_returns(m, np.random.default_rng(spec.ORACLE_PATH_SEED), len(paths)), paths)


def test_env_block_outcome_and_the_reference_loop_give_the_same_solution(market, paths, headline, drift_solution):
    """The production oracle (env.block_outcome) against the one these tests
    check (the day-by-day loop), on the same paths."""
    sol = oracle.solve(market, headline, paths, block_outcome=env.block_outcome)
    np.testing.assert_array_equal(sol.next_index, drift_solution.next_index)
    np.testing.assert_allclose(sol.r_hat, drift_solution.r_hat, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sol.V, drift_solution.V, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(sol.policy_table, drift_solution.policy_table)


# ---------------------------------------------------------------------------
# One scoring for several settings (oracle_runs.py)
# ---------------------------------------------------------------------------

def test_solve_scored_equals_solve(market, paths, headline, drift_solution):
    scores = oracle.score(market, headline.gamma, paths, block_outcome=loop_block_outcome)
    for settings, discount, reference in [(headline, RHO, drift_solution),
                                          (calibration.cost_settings()["aum_10b"], 0.0, None)]:
        got = oracle.solve_scored(scores, settings, discount=discount)
        if reference is None:
            reference = oracle.solve(market, settings, paths, block_outcome=loop_block_outcome, discount=discount)
        for field in ("V", "G", "policy_table", "r_hat", "cv_correction", "next_index", "U", "cost"):
            np.testing.assert_array_equal(getattr(got, field), getattr(reference, field))
        assert (got.settings, got.discount, got.n_iterations) == (reference.settings, reference.discount,
                                                                  reference.n_iterations)


def test_solve_scored_rejects_another_gamma(market, paths):
    scores = oracle.score(market, 5.0, paths, block_outcome=loop_block_outcome, lattice_increment=0.02)
    with pytest.raises(ValueError, match="gamma"):
        oracle.solve_scored(scores, spec.TCSettings(0.0, 0.0, gamma=3.0))


def test_coarse_and_fine_lattices_share_the_targets(market):
    for inc, n in [(0.02, spec.N_TARGETS), (0.005, 20301)]:
        assert len(spec.lattice(inc)) == n
    sol = oracle.solve(market, spec.TCSettings(0.0, 0.0), fixed_paths(market, 2), block_outcome=loop_block_outcome,
                       drift=False, lattice_increment=0.02)
    np.testing.assert_array_equal(sol.target_index, np.arange(spec.N_TARGETS))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, error", [
    ({"discount": 1.0}, ValueError),
    ({"discount": -0.1}, ValueError),
    ({"discount": True}, ValueError),
    ({"drift": 1}, TypeError),
    ({"lattice_increment": 0.04}, ValueError),     # the 2% targets are not lattice points
    ({"lattice_increment": 0.03}, ValueError),     # 1/0.03 is not a whole number
])
def test_solve_rejects_bad_arguments(market, kwargs, error):
    with pytest.raises(error):
        oracle.solve(market, spec.TCSettings(0.0, 0.0), fixed_paths(market, 2), block_outcome=loop_block_outcome,
                     **kwargs)


@pytest.mark.parametrize("shape", [(4, 63), (4, 63, 2), (0, 63, 3), (4, 1, 3), (4, 21, 3)])
def test_solve_rejects_bad_paths(market, shape):
    with pytest.raises(ValueError):
        oracle.solve(market, spec.TCSettings(0.0, 0.0), np.zeros(shape), block_outcome=loop_block_outcome)


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_solve_rejects_bad_returns_even_without_drift(market, bad):
    paths = fixed_paths(market, 2)
    paths[1, 5, 2] = bad
    with pytest.raises(ValueError):
        oracle.solve(market, spec.TCSettings(0.0, 0.0), paths, block_outcome=loop_block_outcome, drift=False)


def _short_reward(w, e, g, d):
    reward, w_end = loop_block_outcome(w, e, g, d)
    return reward[:1], w_end


def _flat_w_end(w, e, g, d):
    reward, w_end = loop_block_outcome(w, e, g, d)
    return reward, w_end[0]


def _nan_reward(w, e, g, d):
    reward, w_end = loop_block_outcome(w, e, g, d)
    return np.full_like(reward, np.nan), w_end


@pytest.mark.parametrize("broken", [_short_reward, _flat_w_end, _nan_reward])
def test_solve_rejects_a_malformed_block_outcome(market, broken):
    with pytest.raises(ValueError):
        oracle.solve(market, spec.TCSettings(0.0, 0.0), fixed_paths(market, 2), block_outcome=broken,
                     lattice_increment=0.02)


def test_solve_rejects_a_bad_market(market):
    bad = spec.Market(mu=market.mu, sigma=-market.sigma)
    with pytest.raises(ValueError):
        oracle.solve(bad, spec.TCSettings(0.0, 0.0), fixed_paths(market, 2), block_outcome=loop_block_outcome)


@pytest.mark.parametrize("table", [np.zeros(5, dtype=int), np.full(spec.N_LATTICE, 1327),
                                   np.full(spec.N_LATTICE, 1.0), np.zeros(spec.N_LATTICE, dtype=bool)])
def test_value_of_rejects_bad_tables(drift_solution, table):
    with pytest.raises(ValueError):
        drift_solution.value_of(table)
