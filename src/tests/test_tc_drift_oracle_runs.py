"""Tests for the oracle runs' in-model statistics (src/tc_drift/oracle_runs.py).

In plain terms: every value and trading statistic the run reports is
recomputed a different way, on a small oracle (the 2% lattice, 24 paths):
policy values by a dense linear solve instead of iteration, and the trading
statistics by simulating the Markov chain instead of pushing the
distribution through it.
"""

import json

import numpy as np
import pytest

from src.tc_drift import calibration, env, oracle, oracle_runs, spec

B = spec.BLOCK_LENGTH
RHO = spec.DISCOUNT
INC = 0.02
TOL = oracle_runs.TOLERANCE_BPS


@pytest.fixture(scope="module")
def setup():
    market = env.load_market()
    settings = calibration.cost_settings()["aum_10b"]   # the largest costs, so policies differ
    paths = env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), 24)
    scores = oracle.score(market, settings.gamma, paths, block_outcome=env.block_outcome, lattice_increment=INC)
    sol = oracle.solve_scored(scores, settings)
    myopic = oracle.solve_scored(scores, settings, discount=0.0)
    return market, sol, myopic, oracle_runs.best_target(market, settings.gamma)


def dense_value(sol, table, rho=RHO):
    """V = r_hat[landing] - cost / B + rho * P[landing] V, solved directly."""
    n = len(sol.V)
    P = sol.transition.toarray()
    hold = table == spec.HOLD
    trade = np.where(hold, 0, table)
    landing = np.where(hold, np.arange(n), sol.target_index[trade])
    c = np.where(hold, 0.0, sol.cost[np.arange(n), trade]) / B
    return np.linalg.solve(np.eye(n) - rho * P[landing], sol.r_hat[landing] - c)


def test_best_target_is_the_constant_weight_optimum(setup):
    market, sol, _, best = setup
    t = spec.target_grid()
    u = [w @ market.mu - 2.5 * w @ market.sigma @ w for w in t]
    assert best == int(np.argmax(u))
    np.testing.assert_allclose(t[best], [0.46, 0.14, 0.40], atol=1e-12)


def test_reference_policy_values_match_a_dense_solve(setup):
    _, sol, myopic, best = setup
    for table in (oracle_runs.rebalance_table(sol, best), oracle_runs.never_trade_table(sol), myopic.policy_table,
                  sol.policy_table):
        np.testing.assert_allclose(sol.value_of(table), dense_value(sol, table), rtol=0, atol=1e-12)
    np.testing.assert_allclose(dense_value(sol, sol.policy_table), sol.V, rtol=0, atol=1e-12)


def test_trade_once_value(setup):
    """Trade to the best target, then hold forever: one cost plus the buy-and-hold
    value from the target, by a dense solve of the hold-forever chain."""
    _, sol, _, best = setup
    n = len(sol.V)
    hold_forever = np.linalg.solve(np.eye(n) - RHO * sol.transition.toarray(), sol.r_hat)
    expected = hold_forever[sol.target_index[best]] - sol.cost[:, best] / B
    np.testing.assert_allclose(oracle_runs.trade_once_value(sol, best), expected, rtol=0, atol=1e-12)
    # From the best target itself it equals never trading.
    a = sol.target_index[best]
    assert oracle_runs.trade_once_value(sol, best)[a] == pytest.approx(hold_forever[a], abs=1e-12)


def test_rebalance_table_holds_only_on_the_best_target(setup):
    _, sol, _, best = setup
    table = oracle_runs.rebalance_table(sol, best)
    assert (table == spec.HOLD).sum() == 1 and table[sol.target_index[best]] == spec.HOLD
    assert (table[table != spec.HOLD] == best).all()


def test_path_statistics_match_a_simulated_chain(setup):
    """Simulate 40,000 chains: each block apply the policy, then draw the next
    state from the fixed paths' end points of the landing point."""
    _, sol, _, _ = setup
    horizon, n_chains = 12, 40_000
    rng = np.random.default_rng(5)
    states = spec.lattice(INC)
    table = sol.policy_table
    s = rng.choice(sol.target_index, n_chains)
    holds = np.zeros(n_chains)
    turnover = np.zeros(n_chains)
    for _ in range(horizon):
        a = table[s]
        hold = a == spec.HOLD
        landing = np.where(hold, s, sol.target_index[np.where(hold, 0, a)])
        holds += hold
        turnover += 0.5 * np.abs(states[landing] - states[s]).sum(axis=1)
        s = sol.next_index[landing, rng.integers(0, sol.next_index.shape[1], n_chains)]
    got = oracle_runs.path_statistics(sol, table, sol.target_index, horizon=horizon)
    assert 0.05 < got["hold_fraction"] < 0.95                          # both kinds of block occur
    se = (holds / horizon).std() / np.sqrt(n_chains)
    assert got["hold_fraction"] == pytest.approx((holds / horizon).mean(), abs=4 * se)
    se = (turnover / horizon).std() / np.sqrt(n_chains)
    assert got["turnover_per_block"] == pytest.approx((turnover / horizon).mean(), abs=4 * se)
    assert got["trades_per_path"] == pytest.approx(horizon * (1 - got["hold_fraction"]), rel=1e-12)
    assert got["mean_trade_size"] == pytest.approx(got["turnover_per_block"] * horizon / got["trades_per_path"],
                                                   rel=1e-12)


def test_path_statistics_of_simple_tables(setup):
    _, sol, _, best = setup
    never = oracle_runs.path_statistics(sol, oracle_runs.never_trade_table(sol), sol.target_index)
    assert never["hold_fraction"] == pytest.approx(1.0, abs=1e-12)         # a sum of 90 shares of 1
    assert never["turnover_per_block"] == 0.0 and never["mean_trade_size"] == 0.0
    # Always rebalancing from a start on the best target: the first block holds, then it trades
    # every block unless drift lands exactly on the target.
    start = np.array([sol.target_index[best]])
    stats = oracle_runs.path_statistics(sol, oracle_runs.rebalance_table(sol, best), start, horizon=1)
    assert stats["hold_fraction"] == pytest.approx(1.0, abs=1e-12)


def test_check3_scores_both_policies_in_the_independent_model(setup):
    _, sol, myopic, _ = setup
    same = oracle_runs.check3(sol, sol, sol)
    # value_of stops within 1e-12 (1 - rho) of V, about 6e-9 bps/yr
    assert same["policy_difference_in_independent_model_bps"] == pytest.approx(0.0, abs=1e-6) and same["passes"]
    assert same["same_action_share_at_2pct_starts"] == 1.0 and same["v_change_bps"] == 0.0
    starts = sol.target_index
    lead = float(spec.annualized_bps(dense_value(sol, sol.policy_table) - dense_value(sol, myopic.policy_table))[starts]
                 .mean())
    assert lead > TOL
    # "M" policy = myopic, "2M" policy = the oracle, both scored in `fresh` = the oracle's model: +lead.
    up = oracle_runs.check3(myopic, sol, sol)
    assert up["policy_difference_in_independent_model_bps"] == pytest.approx(lead, abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert not up["passes"]
    # The other way round the difference is negative, and it fails too (|difference| is tested).
    down = oracle_runs.check3(sol, myopic, sol)
    assert down["policy_difference_in_independent_model_bps"] == pytest.approx(-lead, abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert not down["passes"]
    # The 2M-model form and the reverse form, against the dense solve.
    # the myopic solution's own model has rho = 0
    shortfall = spec.annualized_bps(myopic.V - dense_value(myopic, sol.policy_table, rho=0.0))[starts].mean()
    assert down["policy_shortfall_in_larger_model_bps"] == pytest.approx(shortfall, abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    reverse = spec.annualized_bps(sol.V - dense_value(sol, myopic.policy_table))[starts].mean()
    assert down["reverse_shortfall_in_smaller_model_bps"] == pytest.approx(reverse, abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert down["v_change_bps"] == pytest.approx(spec.annualized_bps(myopic.V - sol.V)[starts].mean(), rel=1e-12)


def test_summarize_fields_match_independent_calculations(setup):
    """Each value is recomputed with dense solves, at the 2% starts, at rho = 0.95."""
    _, sol, myopic, best = setup
    out = oracle_runs.summarize(sol, myopic, best)
    starts = sol.target_index
    n = len(sol.V)

    def bps(x):
        return float(np.mean(np.asarray(x)[starts]) * spec.TRADING_DAYS * 1e4 * (1 - RHO))

    hold_table = np.full(n, spec.HOLD)
    rebal = np.full(n, best)
    rebal[sol.target_index[best]] = spec.HOLD
    hold_forever = dense_value(sol, hold_table)
    trade_once = hold_forever[sol.target_index[best]] - sol.cost[:, best] / B
    assert out["oracle_value_bps"] == pytest.approx(bps(sol.V), rel=1e-12)
    assert out["lead_over_always_rebalance_bps"] == pytest.approx(bps(sol.V - dense_value(sol, rebal)), abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert out["lead_over_never_trade_bps"] == pytest.approx(bps(sol.V - hold_forever), abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert out["lead_over_trade_once_bps"] == pytest.approx(bps(sol.V - trade_once), abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    assert out["lead_over_myopic_bps"] == pytest.approx(bps(sol.V - dense_value(sol, myopic.policy_table)), abs=1e-6)   # value iteration stops within ~1.3e-7 bps/yr
    for key in ("lead_over_always_rebalance_bps", "lead_over_trade_once_bps", "lead_over_never_trade_bps",
                "lead_over_myopic_bps"):
        assert out[key] > 0, key                                    # the oracle is optimal in its own model
    # The correction per block, as a constant daily reward: x * 252 * 1e4.
    assert out["cv_correction_bps"]["mean"] == pytest.approx(sol.cv_correction.mean() * 252e4, rel=1e-12)
    assert out["cv_correction_bps"]["max_abs"] == pytest.approx(np.abs(sol.cv_correction).max() * 252e4, rel=1e-12)
    states = spec.lattice(INC)
    near = [i for i in range(n) if np.abs(states[i] - spec.target_grid()[best]).sum() <= 0.04 + 1e-9]
    assert out["hold_share_near_best_target"] == pytest.approx(np.mean(sol.policy_table[near] == spec.HOLD))
    assert out["oracle"] == oracle_runs.path_statistics(sol, sol.policy_table, starts)
    assert out["myopic"] == oracle_runs.path_statistics(sol, myopic.policy_table, starts)


@pytest.fixture(scope="module")
def fine(setup):
    """The same setting on the 1% lattice (about a minute: the sweeps scale with the lattice)."""
    market, sol, _, _ = setup
    paths = env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), 24)
    return oracle.solve(market, sol.settings, paths, block_outcome=env.block_outcome, lattice_increment=0.01)


def test_check4_compares_the_same_targets_on_both_lattices(setup, fine):
    _, sol, _, _ = setup
    fine_states = spec.lattice(0.01)
    idx = [int(np.argmin(np.abs(fine_states - t).sum(axis=1))) for t in spec.target_grid()]
    change = spec.annualized_bps(fine.V[idx] - sol.V[sol.target_index])
    got = oracle_runs.check4(sol, fine)
    assert got["v_change_bps"] == pytest.approx(change.mean(), rel=1e-12)
    assert got["v_change_max_bps"] == pytest.approx(np.abs(change).max(), rel=1e-12)
    assert got["passes"] == (abs(change.mean()) < TOL)
    assert got["same_action_share_at_2pct_starts"] == pytest.approx(
        np.mean(fine.policy_table[idx] == sol.policy_table))
    assert oracle_runs.check4(sol, sol)["v_change_bps"] == 0.0


def test_main_writes_every_setting(tmp_path):
    """The whole run, small: 6 paths on the 2% lattice."""
    out = oracle_runs.main(tmp_path, m=6, fine=False, increment=INC)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    names = set(oracle_runs.LEVELS) | set(oracle_runs.MECHANISMS)
    assert set(saved["settings"]) == names and set(out["settings"]) == names
    assert saved["settings"]["zero"]["kappa"] == 0.0 and saved["settings"]["zero"]["lam"] == 0.0
    for name in names:
        entry = saved["settings"][name]
        assert {"policy_difference_in_independent_model_bps", "passes", "informative"} <= set(entry["check3"])
        assert "check4" not in entry
        with np.load(tmp_path / f"{name}.npz") as z:
            assert z["policy_table"].shape == (spec.N_TARGETS,)
    assert saved["seed"]["check3_fresh_seed"] != saved["seed"]["oracle_path_seed"]
    assert len(saved["provenance"]["sha256"]) == 5
    assert "biased upward" in saved["what"]


def test_main_refuses_paths_whose_first_m_differ(tmp_path, monkeypatch):
    real = env.draw_returns

    def shifted(market, rng, m, *args):
        x = real(market, rng, m, *args)
        return x * 1.01 if m == 6 else x

    monkeypatch.setattr(oracle_runs.env, "draw_returns", shifted)
    with pytest.raises(RuntimeError, match="first M"):
        oracle_runs.main(tmp_path, m=6, fine=False, increment=INC)
