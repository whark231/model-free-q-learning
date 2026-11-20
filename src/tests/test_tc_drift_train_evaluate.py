"""Tests for src/tc_drift/train.py, evaluate.py and config.py (Albert).

In plain terms: check the training loop and the evaluation against loops
written separately here, which redraw the same random numbers one block at a
time and work out drift by counting shares and dollars, rather than calling
env.block_outcome. Also the cross-test the design assigns to whoever merged
second (TC_DRIFT_DESIGN.md §10): the oracle gives the same solution with
env.block_outcome as with William's day-by-day test loop.
"""

import numpy as np
import pytest

from src.environment import ReturnSimulator, SeedPlan
from src.tc_drift import config, env, evaluate, oracle, spec, train
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings
from src.tests.test_tc_drift_oracle import loop_block_outcome

B = spec.BLOCK_LENGTH
SETTINGS = cost_settings()["aum_1b"]


@pytest.fixture(scope="module")
def market():
    return env.load_market()


@pytest.fixture(scope="module")
def best(market):
    return evaluate.best_static_target(market, spec.GAMMA_HEADLINE)


def _share_loop(w0, e, gamma):
    """Independent drift: fixed shares, prices walked forward day by day."""
    shares, prices, port = np.array(w0, dtype=float), np.ones(3), []
    for day in e:
        start = shares @ prices
        prices = prices * (1.0 + day)
        port.append((shares @ prices) / start - 1.0)
    port = np.array(port)
    end = shares * prices
    return port.mean() - gamma / 2 * port.var(ddof=1), end / end.sum()


def _hand_cost(s, a, settings):
    if a == spec.HOLD:
        return 0.0
    u = spec.target_grid()[a] - s
    return settings.kappa * np.abs(u).sum() + 0.5 * settings.lam * (u @ u)


# ---------------------------------------------------------------------------
# Stub agents (TC_DRIFT_DESIGN.md §10: train.py is tested with trivial agents)
# ---------------------------------------------------------------------------

class Recorder:
    """Takes a fixed action rule and records every (s, a, r, s') it is given."""

    def __init__(self, rule, rng=None):
        self.rule, self.rng, self.seen = rule, rng, []

    def select_action(self, state):
        return self.rule(state, self.rng)

    def update(self, state, action, reward_net, next_state):
        self.seen.append((np.array(state), action, reward_net, np.array(next_state)))

    def greedy_action(self, state):
        return spec.HOLD


def _random_rule(state, rng):
    return int(rng.integers(spec.N_ACTIONS))


def _train(agent, market, n_blocks=37, restart_every=10, checkpoint_every=8, seed=0, **kw):
    returns_rng, start_rng = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    return train.train(agent, market, kw.pop("settings", SETTINGS), n_blocks=n_blocks, returns_rng=returns_rng,
                       start_rng=start_rng, start_states=kw.pop("starts", spec.target_grid()),
                       restart_every=restart_every, checkpoint_every=checkpoint_every,
                       probe_states=kw.pop("probes", spec.target_grid()[:5]), **kw)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_training_transitions_match_an_independent_replay(market):
    """Redraw the returns one block at a time and the starts in order, and
    rebuild every (s, a, r, s') with the share loop and the cost formula."""
    agent = Recorder(_random_rule, np.random.default_rng(5))
    n, every = 37, 10
    log = _train(agent, market, n_blocks=n, restart_every=every)
    sim = ReturnSimulator(market.mu, market.sigma, np.random.default_rng([0, 1]))
    start_rng = np.random.default_rng([0, 2])
    targets = spec.target_grid()
    state = targets[start_rng.integers(len(targets))]
    for k, (s, a, r, s_next) in enumerate(agent.seen):
        if k > 0 and k % every == 0:
            state = targets[start_rng.integers(len(targets))]
        np.testing.assert_allclose(s, state, rtol=0, atol=1e-15)
        x = state if a == spec.HOLD else targets[a]
        R, w_end = _share_loop(x, sim.sample_episode(B), SETTINGS.gamma)
        assert r == pytest.approx(R - _hand_cost(state, a, SETTINGS) / B, rel=1e-10, abs=1e-15)
        np.testing.assert_allclose(s_next, w_end, rtol=1e-12)
        state = w_end
    assert len(agent.seen) == n and log["n_restarts"] == (n - 1) // every


def test_restart_still_passes_the_drifted_end_holdings(market):
    """The block before a restart is updated with its true end holdings, not the new start."""
    agent = Recorder(lambda s, rng: spec.HOLD)
    _train(agent, market, n_blocks=12, restart_every=4)
    for k in (3, 7):
        _, _, _, s_next = agent.seen[k]
        next_start = agent.seen[k + 1][0]
        assert not np.allclose(s_next, next_start)
        assert spec.check_holdings(s_next).shape == (1, 3)


def test_chunk_size_does_not_change_the_run(market, monkeypatch):
    a1 = Recorder(_random_rule, np.random.default_rng(1))
    log1 = _train(a1, market, n_blocks=30)
    monkeypatch.setattr(train, "_CHUNK", 7)
    a2 = Recorder(_random_rule, np.random.default_rng(1))
    log2 = _train(a2, market, n_blocks=30)
    assert [r for _, _, r, _ in a1.seen] == [r for _, _, r, _ in a2.seen]
    np.testing.assert_array_equal(log1["mean_net_reward"], log2["mean_net_reward"])


def test_log_windows(market, best):
    """Always rebalance from the best target: the first block is a zero-size
    trade (a hold), every later one undoes a block of drift."""
    agent = Recorder(lambda s, rng: best)
    log = _train(agent, market, n_blocks=20, restart_every=100, checkpoint_every=8, starts=spec.target_grid()[[best]])
    np.testing.assert_array_equal(log["checkpoint_block"], [8, 16, 20])
    assert log["hold_fraction"][0] == pytest.approx(1 / 8) and (log["hold_fraction"][1:] == 0).all()
    assert (log["explicit_hold_fraction"] == 0).all()
    costs = [_hand_cost(s, a, SETTINGS) for s, a, _, _ in agent.seen]
    turn = [0.5 * np.abs(spec.target_grid()[best] - s).sum() for s, _, _, _ in agent.seen]
    np.testing.assert_allclose(log["mean_cost_bps"], [np.mean(costs[:8]) * 1e4, np.mean(costs[8:16]) * 1e4,
                                                       np.mean(costs[16:]) * 1e4], rtol=1e-12)
    np.testing.assert_allclose(log["mean_turnover"][2], np.mean(turn[16:]), rtol=1e-12)
    assert 0.005 < np.mean(turn[1:]) < 0.06   # one block of drift, design §12: median 2.26%
    assert log["simulated_days"] == 20 * B and log["probe_greedy"].shape == (3, 5)


def test_always_hold_agent(market):
    log = _train(Recorder(lambda s, rng: spec.HOLD), market, n_blocks=16)
    assert (log["hold_fraction"] == 1).all() and (log["explicit_hold_fraction"] == 1).all()
    assert (log["mean_cost_bps"] == 0).all() and (log["mean_turnover"] == 0).all()


def test_no_drift_mode_keeps_the_post_trade_weights(market):
    agent = Recorder(_random_rule, np.random.default_rng(2))
    _train(agent, market, n_blocks=10, drift=False)
    for s, a, _, s_next in agent.seen:
        np.testing.assert_array_equal(s_next, spec.post_trade_weights(s, a))


def test_drift_agent_trains_and_logs_diagnostics(market):
    agent = DriftAgent(np.random.default_rng(3), alpha_value=0.01, alpha_cost=0.001, beta=1e-3,
                       target_refresh_every=50)
    log = _train(agent, market, n_blocks=200, restart_every=50, checkpoint_every=50)
    assert agent.n_updates == 200
    d = log["diagnostics"]
    assert set(d) == {"cost_linear", "cost_quadratic", "epsilon", "n_updates", "theta_norm"}
    np.testing.assert_array_equal(d["n_updates"], [50, 100, 150, 200])
    assert np.isfinite(d["theta_norm"]).all() and (np.diff(d["epsilon"]) < 0).all()
    assert log["probe_greedy"].min() >= 0 and log["probe_greedy"].max() <= spec.HOLD


@pytest.mark.parametrize("kwargs, error", [
    ({"n_blocks": 0}, ValueError), ({"n_blocks": 2.0}, ValueError), ({"restart_every": 0}, ValueError),
    ({"checkpoint_every": True}, ValueError), ({"drift": 1}, TypeError),
])
def test_train_rejects_bad_arguments(market, kwargs, error):
    with pytest.raises(error):
        _train(Recorder(lambda s, rng: spec.HOLD), market, **kwargs)


def test_train_rejects_shared_generators_and_bad_actions(market):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="different"):
        train.train(Recorder(lambda s, r: spec.HOLD), market, SETTINGS, n_blocks=2, returns_rng=rng, start_rng=rng,
                    start_states=spec.target_grid(), restart_every=1, checkpoint_every=1,
                    probe_states=spec.target_grid()[:1])
    for bad in (spec.N_ACTIONS, -1, 3.0, True):
        with pytest.raises(ValueError, match="invalid action"):
            _train(Recorder(lambda s, rng, bad=bad: bad), market, n_blocks=2)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_set(market):
    return evaluate.draw_evaluation_set(market, np.random.default_rng(11), 6, 5)


def _hand_value(policy, start, path_blocks, settings, rho=spec.DISCOUNT):
    """One start, block by block, with the share loop and the cost formula."""
    v, state = 0.0, np.array(start, dtype=float)
    for t, block in enumerate(path_blocks):
        a = int(policy(state[None, :], t)[0])
        x = state if a == spec.HOLD else spec.target_grid()[a]
        c = _hand_cost(state, a, settings)
        R, state = _share_loop(x, block, settings.gamma)
        v += rho**t * (R - c / B)
    return v


@pytest.mark.parametrize("name", ["rebalance", "once", "never", "random"])
def test_evaluate_policy_matches_a_hand_loop(market, best, small_set, name):
    starts, paths = small_set
    rng_actions = np.random.default_rng(4).integers(0, spec.N_ACTIONS, (len(paths), len(starts)))
    policies = {
        "rebalance": evaluate.always_rebalance_policy(best),
        "once": evaluate.never_trade_policy(best),
        "never": lambda s, t: np.full(len(s), spec.HOLD),
        "random": None,
    }
    if name == "random":
        out = evaluate.evaluate_policy(lambda s, t: rng_actions[t], starts, paths, SETTINGS)
        for i in range(len(starts)):
            hand = _hand_value(lambda s, t, i=i: rng_actions[t][[i]], starts[i], paths[:, i], SETTINGS)
            assert out["value"][i] == pytest.approx(hand, rel=1e-10)
        return
    out = evaluate.evaluate_policy(policies[name], starts, paths, SETTINGS)
    for i in range(len(starts)):
        assert out["value"][i] == pytest.approx(_hand_value(policies[name], starts[i], paths[:, i], SETTINGS),
                                                rel=1e-10)
    np.testing.assert_allclose(out["value_bps"], spec.annualized_bps(out["value"], horizon=len(paths)))


def test_evaluate_counts_and_weights(market, best, small_set):
    starts, paths = small_set
    H = len(paths)
    never = evaluate.evaluate_policy(lambda s, t: np.full(len(s), spec.HOLD), starts, paths, SETTINGS)
    assert (never["hold_fraction"] == 1).all() and (never["trades"] == 0).all()
    assert np.isnan(never["mean_trade_size"]) and (never["cost_bps_per_block"] == 0).all()
    once = evaluate.evaluate_policy(evaluate.never_trade_policy(best), starts, paths, SETTINGS)
    moved = ~np.all(np.isclose(starts, spec.target_grid()[best]), axis=1)
    np.testing.assert_array_equal(once["trades"], moved.astype(float))
    reb = evaluate.evaluate_policy(evaluate.always_rebalance_policy(best), np.tile(spec.target_grid()[best], (5, 1)),
                                   paths, SETTINGS)
    np.testing.assert_allclose(reb["hold_fraction"], 1 / H)   # block 0 is a zero-size trade
    np.testing.assert_allclose(reb["mean_weights"], spec.target_grid()[best], atol=1e-15)


def test_evaluate_accepts_a_list_of_blocks(best, small_set):
    starts, paths = small_set
    p = evaluate.always_rebalance_policy(best)
    a = evaluate.evaluate_policy(p, starts, paths, SETTINGS)["value"]
    b = evaluate.evaluate_policy(p, starts, list(paths), SETTINGS)["value"]
    np.testing.assert_array_equal(a, b)


def test_policy_sees_a_copy_of_the_holdings(best, small_set):
    starts, paths = small_set

    def vandal(s, t):
        s[:] = np.nan
        return np.full(len(s), spec.HOLD)
    out = evaluate.evaluate_policy(vandal, starts, paths, SETTINGS)
    assert np.isfinite(out["value"]).all()


@pytest.mark.parametrize("bad", [lambda s, t: np.zeros(len(s) + 1, dtype=int),
                                 lambda s, t: np.full(len(s), spec.N_ACTIONS),
                                 lambda s, t: np.zeros(len(s)),
                                 lambda s, t: np.zeros(len(s), dtype=bool)])
def test_evaluate_rejects_bad_policies(small_set, bad):
    starts, paths = small_set
    with pytest.raises(ValueError):
        evaluate.evaluate_policy(bad, starts, paths, SETTINGS)


def test_evaluate_rejects_bad_paths(small_set):
    starts, paths = small_set
    with pytest.raises(ValueError):
        evaluate.evaluate_policy(lambda s, t: np.full(len(s), spec.HOLD), starts[:3], paths, SETTINGS)
    with pytest.raises(ValueError):
        evaluate.evaluate_policy(lambda s, t: np.full(len(s), spec.HOLD), starts, paths[:0], SETTINGS)


@pytest.mark.parametrize("make, bad", [(evaluate.always_rebalance_policy, spec.HOLD),
                                       (evaluate.never_trade_policy, -1),
                                       (evaluate.always_rebalance_policy, 2.0)])
def test_reference_policies_reject_bad_targets(make, bad):
    with pytest.raises(ValueError):
        make(bad)


def test_agent_adapter_is_greedy_row_by_row(small_set):
    agent = DriftAgent(np.random.default_rng(0), alpha_value=0.01, alpha_cost=0.001, beta=0.0,
                       target_refresh_every=10)
    agent._theta[:] = np.random.default_rng(1).normal(size=agent._theta.shape)  # a non-trivial greedy policy
    agent._g_online = None
    starts, _ = small_set
    out = evaluate.agent_policy(agent)(starts, 0)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [agent.greedy_action(s) for s in starts])


def test_best_static_target(market):
    targets = spec.target_grid()
    U = targets @ market.mu - 2.5 * np.array([w @ market.sigma @ w for w in targets])
    best = evaluate.best_static_target(market, 5.0)
    assert best == int(np.argmax(U))
    np.testing.assert_allclose(targets[best], [0.46, 0.14, 0.40], atol=1e-12)


def test_evaluation_set_reproduces_the_pilot_order(market):
    starts, paths = evaluate.draw_evaluation_set(market, np.random.default_rng(9), 3, 4)
    rng = np.random.default_rng(9)
    s2 = spec.target_grid()[rng.integers(0, spec.N_TARGETS, 4)]
    p2 = np.stack([env.draw_returns(market, rng, 4) for _ in range(3)])
    np.testing.assert_array_equal(starts, s2)
    np.testing.assert_array_equal(paths, p2)


def test_paired_gap():
    a, b = np.array([1e-4, 2e-4, 4e-4]), np.array([0.0, 1e-4, 1e-4])
    g = evaluate.paired_gap(a, b, horizon=10)
    d = spec.annualized_bps(a - b, horizon=10)
    assert g["mean"] == pytest.approx(d.mean()) and g["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(3))
    with pytest.raises(ValueError):
        evaluate.paired_gap(a, b[:2], horizon=10)


def test_probe_convergence():
    def log(rows, blocks):
        return {"checkpoint_block": np.array(blocks), "probe_greedy": np.array(rows), "n_blocks": blocks[-1]}
    blocks = list(range(10, 110, 10))
    steady = log([[1, 2]] * 10, blocks)
    r = evaluate.probe_convergence(steady)
    assert r["converged"] and r["share_stable"] == 1 and r["window_first_block"] == 90 and r["stable_from_block"] == 10
    late = log([[1, 2]] * 9 + [[1, 3]], blocks)
    r = evaluate.probe_convergence(late)
    assert not r["converged"] and r["share_stable"] == 0.5 and r["stable_from_block"] is None
    early = log([[0, 0]] * 5 + [[1, 2]] * 5, blocks)
    r = evaluate.probe_convergence(early)
    assert r["converged"] and r["stable_from_block"] == 60
    with pytest.raises(ValueError):
        evaluate.probe_convergence(steady, final_fraction=0)


# ---------------------------------------------------------------------------
# The oracle with env.block_outcome (the §10 cross-test) and the model value
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oracle_paths(market):
    return env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), 8)


@pytest.fixture(scope="module")
def env_solution(market, oracle_paths):
    return oracle.solve(market, SETTINGS, oracle_paths, block_outcome=env.block_outcome)


def test_oracle_cross_test_env_against_williams_loop(market, oracle_paths, env_solution):
    loop = oracle.solve(market, SETTINGS, oracle_paths, block_outcome=loop_block_outcome)
    np.testing.assert_allclose(env_solution.r_hat, loop.r_hat, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(env_solution.next_index, loop.next_index)
    np.testing.assert_array_equal(env_solution.policy_table, loop.policy_table)
    np.testing.assert_allclose(env_solution.V, loop.V, rtol=1e-12)


def test_oracle_adapter(env_solution, small_set):
    starts, _ = small_set
    out = evaluate.oracle_policy(env_solution)(starts, 3)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [env_solution.policy(s) for s in starts])


def test_model_value_matches_value_of_and_one_step(env_solution, best):
    n = len(env_solution.V)
    table = env_solution.policy_table
    np.testing.assert_allclose(evaluate.model_value(env_solution, table, 2000), env_solution.value_of(table),
                               rtol=1e-9, atol=1e-14)
    reb = np.full(n, best, dtype=np.int64)
    one = evaluate.model_value(env_solution, reb, 1)
    expected = env_solution.r_hat[env_solution.target_index[best]] - env_solution.cost[:, best] / B
    np.testing.assert_allclose(one, expected, rtol=1e-12)
    two = evaluate.model_value(env_solution, reb, 2)
    P = env_solution.transition
    np.testing.assert_allclose(two, expected + spec.DISCOUNT * (P @ one)[env_solution.target_index[best]],
                               rtol=1e-12)
    with pytest.raises(ValueError):
        evaluate.model_value(env_solution, reb[:-1], 2)


def test_lattice_table_of_a_reference_policy(best):
    table = evaluate.lattice_table(evaluate.always_rebalance_policy(best), spec.lattice())
    assert table.shape == (spec.N_LATTICE,) and (table == best).all()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_probe_states_are_fixed_and_cover_the_grid_plus_off_grid_points():
    p = config.probe_states()
    assert p.shape == (spec.N_TARGETS + config.N_OFFGRID_PROBES, 3)
    np.testing.assert_array_equal(p[:spec.N_TARGETS], spec.target_grid())
    off = p[spec.N_TARGETS:]
    idx = spec.nearest_lattice_index(off)
    np.testing.assert_allclose(spec.lattice()[idx], off, atol=1e-15)
    assert len(np.unique(idx)) == config.N_OFFGRID_PROBES
    assert not np.isin(idx, spec.target_lattice_index()).any()
    np.testing.assert_array_equal(p, config.probe_states())


def test_run_streams_are_distinct_and_reproducible():
    a = [g.random() for g in config.run_rngs(0)]
    b = [g.random() for g in config.run_rngs(0)]
    c = [g.random() for g in config.run_rngs(1)]
    assert a == b and len(set(a)) == 3 and a != c
    returns_rng, _ = SeedPlan(spec.MASTER_SEED, config.N_SEEDS).run_streams(0)
    assert config.run_rngs(0)[0].random() == returns_rng.random()
    assert config.evaluation_rng().random() == SeedPlan(spec.MASTER_SEED, config.N_SEEDS).evaluation_rng().random()


def test_pending_settings_are_listed():
    assert config.pending() == []            # all set 2026-11-20 (test_tc_drift_run.py checks the values)
    assert config.RESTART_EVERY == config.H_EVAL == 90 and config.M_EVAL == 4000 and config.M_EVAL_PILOTS == 1000


class Counter(Recorder):
    """Greedy action depends on the updates seen and the probe, so the log's
    rows show when and in which order the probes were read."""

    def __init__(self):
        super().__init__(lambda s, rng: spec.HOLD)
        self.n = 0

    def update(self, state, action, reward_net, next_state):
        super().update(state, action, reward_net, next_state)
        self.n += 1

    def greedy_action(self, state):
        return (self.n * 10 + int(round(state[0] * 50))) % spec.N_ACTIONS


def test_probes_are_read_after_the_checkpoint_update_in_order(market):
    probes = spec.target_grid()[[0, 600, 1325]]
    log = _train(Counter(), market, n_blocks=20, checkpoint_every=8, probes=probes)
    first = np.round(probes[:, 0] * 50).astype(int)
    expected = [(n * 10 + first) % spec.N_ACTIONS for n in (8, 16, 20)]
    np.testing.assert_array_equal(log["probe_greedy"], expected)


def test_mean_net_reward_is_the_window_mean_of_the_rewards_given(market):
    agent = Recorder(_random_rule, np.random.default_rng(8))
    log = _train(agent, market, n_blocks=20, checkpoint_every=8)
    r = [x[2] for x in agent.seen]
    np.testing.assert_allclose(log["mean_net_reward"], [np.mean(r[:8]), np.mean(r[8:16]), np.mean(r[16:])],
                               rtol=1e-12)


def test_oracle_replay_wiring(market, monkeypatch):
    """Small sizes: check5 is lead_replay minus lead_model, every field is finite."""
    from src.tc_drift import oracle_replay
    out = oracle_replay.main(["aum_1b"], m_oracle=6, h=4, m=20)["levels"]["aum_1b"]
    assert out["check5"]["mean"] == pytest.approx(out["lead_replay"]["mean"] - out["lead_model"], abs=1e-9)
    flat = [v for v in out.values() if isinstance(v, float)] + \
           [x for v in out.values() if isinstance(v, dict) for x in v.values()]
    assert np.isfinite(flat).all()


def test_rounded_replay_snaps_the_state_after_every_block(market):
    """The rounded replay restated step by step: starts and every end-of-block
    holding are snapped to the 1% lattice before the policy sees them."""
    from src.tc_drift import oracle_replay
    starts, paths = evaluate.draw_evaluation_set(market, np.random.default_rng(2), 3, 6)
    sol = oracle.solve(market, SETTINGS, env.draw_returns(market, np.random.default_rng(1), 4),
                       block_outcome=env.block_outcome, drift=False)
    a = oracle_replay.rounded_replay(sol, starts, paths, SETTINGS)
    state, b = starts.copy(), np.zeros(len(starts))
    for t, block in enumerate(paths):
        act = sol.policy(state)
        cost = spec.trade_cost(state, act, SETTINGS)
        reward, w_end = env.block_outcome(spec.post_trade_weights(state, act), block, SETTINGS.gamma, True)
        b += spec.DISCOUNT**t * spec.net_reward(reward, cost)
        state = spec.lattice()[spec.nearest_lattice_index(w_end)]
    np.testing.assert_allclose(a, b, rtol=1e-12)
