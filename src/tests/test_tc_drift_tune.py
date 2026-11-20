"""Tests for the tuning pilot (src/tc_drift/tune.py).

In plain terms: the pilot runs training in segments so it can read theta in
between; these tests check that a segmented run is the same run as one long
train() call, that the choosing rule and the convergence rule do what the
design says, and that one tiny run writes its outputs.
"""

import json

import numpy as np
import pytest

from src.tc_drift import config, env, spec, train, tune
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

HP = {"alpha_value": 0.01, "alpha_cost": 0.01, "beta": 2e-3, "target_refresh_every": 50, "cost_rls_after": 300}


def test_a_segmented_run_is_the_same_run_as_one_call():
    market = env.load_market()
    settings = cost_settings()["aum_1b"]
    probes = config.probe_states()[::40]
    n, seg = 900, 180

    r1, a1, s1 = config.run_rngs(0, spec.TUNING_MASTER_SEED)
    one = DriftAgent(a1, **HP)
    log = train.train(one, market, settings, n_blocks=n, returns_rng=r1, start_rng=s1,
                      start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                      checkpoint_every=seg, probe_states=probes)

    r2, a2, s2 = config.run_rngs(0, spec.TUNING_MASTER_SEED)
    parts = DriftAgent(a2, **HP)
    seg_log = tune.train_segmented(parts, market, settings, n, seg, r2, s2, probes)

    assert one.cost_rls_active and parts.cost_rls_active
    np.testing.assert_array_equal(parts.theta, one.theta)
    assert (parts.cost_linear, parts.cost_quadratic) == (one.cost_linear, one.cost_quadratic)
    np.testing.assert_array_equal(seg_log["checkpoint_block"], log["checkpoint_block"])
    np.testing.assert_array_equal(seg_log["probe_greedy"], log["probe_greedy"])
    np.testing.assert_allclose(seg_log["c1"], log["diagnostics"]["cost_linear"], rtol=0, atol=0)
    np.testing.assert_allclose(seg_log["hold_fraction"], log["hold_fraction"], rtol=0, atol=0)
    assert seg_log["theta"].shape == (n // seg, 87)
    np.testing.assert_array_equal(seg_log["theta"][-1], one.theta)


def test_segments_must_respect_the_restart_interval():
    assert tune.segments(4_000_000, tune.SEGMENT) == [40_050] * 99 + [35_050]
    assert sum(tune.segments(4_000_000, tune.SEGMENT)) == 4_000_000
    with pytest.raises(ValueError):
        tune.segments(1000, 100)


def test_schedule_constants():
    assert np.exp(-tune.BETA * tune.N_BLOCKS / 2) == pytest.approx(0.01)
    assert np.exp(-tune.BETA * tune.RLS_AFTER) == pytest.approx(0.5, abs=1e-6)
    for c in tune.CANDIDATES:
        hp = tune.hyperparameters(c)
        assert set(hp) <= DriftAgent.HYPERPARAMETERS
        DriftAgent(np.random.default_rng(0), **hp)          # accepted, inside the stability bound


def _seeds(values):
    return dict(enumerate(values))


def test_choosing_rule():
    clear = tune.choose({"a": _seeds([10.0, 11.0, 12.0]), "b": _seeds([5.0, 5.5, 6.0])})
    assert clear["decision"] == "a" and clear["decided_on_seeds"] == 3 and not clear["close_call"]
    assert clear["paired_difference"] == pytest.approx(5.5)
    # Clear on the first three seeds: extra seeds, if present, are ignored.
    extra = tune.choose({"a": _seeds([10.0, 11.0, 12.0, -99, -99, -99]), "b": _seeds([5.0, 5.5, 6.0, 0, 0, 0])})
    assert extra["decision"] == "a" and extra["decided_on_seeds"] == 3
    # Leader ahead by less than one paired SE: pending until both have six seeds.
    close = {"a": _seeds([10.0, 0.0, 12.0]), "b": _seeds([9.0, 1.0, 11.5])}
    out = tune.choose(close)
    assert out["close_call"] and out["decision"] == "pending" and out["extra_seeds_for"] == {"a": [3, 4, 5], "b": [3, 4, 5]}
    close["a"].update({3: 3.0}); close["b"].update({3: 3.0})
    assert tune.choose(close)["decision"] == "pending"              # never decided on four seeds
    close["a"].update({4: 4.0, 5: 5.0}); close["b"].update({4: 6.0, 5: 7.0})
    six = tune.choose(close)
    assert six["decision"] == "b" and six["decided_on_seeds"] == 6   # the six-seed mean decides
    # A missing first-round run blocks the decision.
    assert tune.choose({"a": _seeds([1.0, 2.0]), "b": _seeds([0.0, 1.0, 2.0])})["decision"] == "pending"


def _log(favourites, greedy, step=10):
    n = len(favourites)
    return {"checkpoint_block": np.arange(1, n + 1) * step, "favourite_target": np.array(favourites),
            "probe_greedy": np.array(greedy)}


def test_convergence_rule_as_changed():
    t = spec.target_grid()
    best = int(np.flatnonzero(np.isclose(t, [0.46, 0.14, 0.40]).all(axis=1))[0])
    near = int(np.flatnonzero(np.isclose(t, [0.48, 0.12, 0.40]).all(axis=1))[0])    # L1 0.04 away
    far = int(np.flatnonzero(np.isclose(t, [0.50, 0.10, 0.40]).all(axis=1))[0])     # L1 0.08 away
    stable = [[1] * 20] * 20
    # 20 checkpoints of 10 blocks: the window starts at block 180 (the 18th checkpoint).
    out = tune.convergence(_log([far] * 17 + [near, best, best], stable), 200)
    assert out["window_first_block"] == 180 and out["n_checkpoints"] == 3
    assert out["primary_favourite_within_one_step"] and out["original_all_probes_unchanged"]
    assert not tune.convergence(_log([best] * 17 + [far, best, best], stable), 200)["primary_favourite_within_one_step"]
    # One probe of 20 changes in the window: 95% stable passes, the original rule fails.
    greedy = [[1] * 20 for _ in range(20)]
    greedy[18] = [2] + [1] * 19
    out = tune.convergence(_log([best] * 20, greedy), 200)
    assert out["secondary_exact_action_95pct_2026_11_19"] and not out["original_all_probes_unchanged"]
    greedy[18] = [2, 2] + [1] * 18
    assert not tune.convergence(_log([best] * 20, greedy), 200)["secondary_exact_action_95pct_2026_11_19"]
    # Different probes flipping at different checkpoints: 95% are stable at each
    # checkpoint, but only 90% never change, so the rule (the stricter reading) fails.
    greedy = [[1] * 20 for _ in range(20)]
    greedy[17] = [2] + [1] * 19
    greedy[18] = [1, 2] + [1] * 18
    out = tune.convergence(_log([best] * 20, greedy), 200)
    assert out["probes_unchanged_share"] == pytest.approx(0.9) and not out["secondary_exact_action_95pct_2026_11_19"]
    # A window of one checkpoint passes nothing; no checkpoint before 90% starts at the first.
    one = tune.convergence(_log([best], [[1] * 20]), 10)
    assert one["n_checkpoints"] == 1 and not one["primary_favourite_within_one_step"]
    late = tune.convergence(_log([best] * 3, stable[:3], step=100), 200)
    assert late["window_first_block"] == 100 and late["primary_favourite_within_one_step"]


def test_restated_secondary_rule_and_the_converged_from_block():
    t = spec.target_grid()
    idx = lambda w: int(np.flatnonzero(np.isclose(t, w).all(axis=1))[0])
    best, near, far = idx([0.46, 0.14, 0.40]), idx([0.48, 0.12, 0.40]), idx([0.50, 0.10, 0.40])
    probes = t[:20].copy()
    probes[19] = t[best]                                     # a probe sitting at the best target
    n, step, N = 20, 900, 18_000                             # window: checkpoints 18 to 20 (blocks 16,200 on)

    def run(fav, greedy):
        log = {"checkpoint_block": np.arange(1, n + 1) * step, "favourite_target": np.array(fav),
               "probe_greedy": np.array(greedy)}
        return tune.convergence(log, N, probes)

    base = [[best] * 20 for _ in range(n)]
    out = run([best] * n, base)
    assert out["converged"] and out["probes_within_one_step_share"] == 1.0
    assert out["converged_from_block"] == step and out["converged_from_episode"] == step // 90
    assert out["converged_from_daily_observations"] == step * spec.BLOCK_LENGTH
    # HOLD at the probe that sits on the best target lands on the same holdings.
    g = [row[:] for row in base]
    g[18][19] = spec.HOLD
    out = run([best] * n, g)
    assert out["probes_within_one_step_share"] == 1.0
    assert out["hold_probes_in_window"] == 1 and out["hold_decision_stable_share"] == 0.0 and out["hold_probes_final"] == 0
    # HOLD at an off-grid probe is judged by the probe itself: (0.47, 0.14, 0.39) is
    # 0.02 from the best target, but (0.40, 0.20, 0.40) is 0.12 from it.
    probes[18] = [0.47, 0.14, 0.39]
    g = [row[:] for row in base]
    g[18][18] = spec.HOLD
    assert run([best] * n, g)["probes_within_one_step_share"] == 1.0
    probes[18] = [0.40, 0.20, 0.40]
    assert run([best] * n, g)["probes_within_one_step_share"] == pytest.approx(0.95)
    probes[18] = t[18]
    # Moving to the neighbouring target is within one step: the exact-action rules fail, the restated one passes.
    g = [row[:] for row in base]
    g[18][:2] = [near, near]
    out = run([best] * n, g)
    assert out["secondary_95pct_probes_within_one_step"] and not out["secondary_exact_action_95pct_2026_11_19"]
    # Two steps away at 1 probe of 20 passes (95%); at 2 probes it fails.
    g[18][:2] = [far, near]
    assert run([best] * n, g)["secondary_95pct_probes_within_one_step"]
    g = [row[:] for row in base]
    g[17][:2] = [far, far]                                   # the window's first checkpoint (block 16,200)
    out = run([best] * n, g)
    assert out["window_first_block"] == 18 * step
    assert out["probes_within_one_step_share"] == pytest.approx(0.9) and not out["converged"]
    assert out["rules_hold_from_block"] == 19 * step      # from the next checkpoint on, both rules hold,
    assert out["converged_from_block"] is None and out["converged_from_episode"] is None   # but not converged
    g[17][:2] = [best, best]
    g[18][:2] = [far, far]                                   # only the last checkpoint is clean after it
    assert run([best] * n, g)["rules_hold_from_block"] is None
    # Different probes two steps away at different checkpoints: 95% are within one
    # step at each checkpoint, but only 90% throughout, so the rule fails.
    g = [row[:] for row in base]
    g[17][0] = far
    g[18][1] = far
    out = run([best] * n, g)
    assert out["probes_within_one_step_share"] == pytest.approx(0.9) and not out["secondary_95pct_probes_within_one_step"]
    # The favourite target two steps away until checkpoint 6: converged from checkpoint 6 (block 5,400).
    out = run([far] * 5 + [best] * 15, base)
    assert out["converged"] and out["converged_from_block"] == 6 * step
    # Only the last checkpoint differs: the rule never holds before it.
    out = run([best] * (n - 1) + [far], base)
    assert not out["converged"] and out["converged_from_block"] is None


def test_one_tiny_run_writes_its_outputs(tmp_path):
    record = tune.run_one("a1e-3", 0, out_dir=tmp_path, n_blocks=10 * 180, segment=180, m=8)
    saved = json.loads((tmp_path / "a1e-3" / "seed0.json").read_text(encoding="utf-8"))
    assert saved["criterion_mean_value_bps"] == pytest.approx(record["criterion_mean_value_bps"])
    assert saved["policy"] == "tail_average" and not saved["averaged_post_hoc"]
    assert "criterion_mean_value_bps" in saved["last_iterate"]
    z = np.load(tmp_path / "a1e-3" / "seed0.npz")
    assert z["theta"].shape == (10, 87) and z["value"].shape == (8,) and z["last_iterate_value"].shape == (8,)
    # 10 checkpoints of 180 blocks: the tail average is the last one only (block 1800 > 1620).
    np.testing.assert_array_equal(z["averaged_theta"], z["theta"][-1])
    assert b"\r\n" not in (tmp_path / "a1e-3" / "seed0.json").read_bytes()


def test_trailing_average_by_hand():
    blocks = np.array([100, 200, 300, 400, 500])
    theta = np.arange(5.0)[:, None] * np.ones((5, 87))
    # n = 500, 10% = 50 blocks: only the end checkpoint itself.
    np.testing.assert_array_equal(tune.trailing_average(blocks, theta, 500, 4), theta[4])
    # 10% of 2,500 = 250 blocks back from 500: checkpoints 300, 400, 500.
    np.testing.assert_allclose(tune.trailing_average(blocks, theta, 2500, 4), np.full(87, 3.0))
    np.testing.assert_allclose(tune.trailing_average(blocks, theta, 2500, 2), np.full(87, 1.0))   # 100, 200, 300


def _drive(agent, n):
    rng = np.random.default_rng(7)
    states = config.probe_states()
    for _ in range(n):
        s, s2 = states[rng.integers(len(states))], states[rng.integers(len(states))]
        agent.update(s, int(rng.integers(spec.HOLD + 1)), float(rng.normal(0, 1e-3)), s2)
    return agent


def test_with_parameters_is_an_independent_copy():
    agent = _drive(DriftAgent(np.random.default_rng(1), **{**HP, "cost_rls_after": None}), 300)
    same = agent.with_parameters(agent.theta)
    probes = config.probe_states()[::25]
    assert [same.greedy_action(x) for x in probes] == [agent.greedy_action(x) for x in probes]
    assert same.favourite_target() == agent.favourite_target()
    other = agent.with_parameters(np.zeros(87), (0.0, 0.0))
    assert other.cost_linear == 0.0 and np.all(other.theta == 0)
    assert np.any(agent.theta != 0) and agent.cost_quadratic == same.cost_quadratic
    with pytest.raises(RuntimeError):                               # copies are for evaluation only
        same.update(probes[0], 5, 0.01, probes[1])
    assert not same._phi_targets.flags.writeable
    before = agent.theta
    agent.update(probes[0], 5, 0.01, probes[1])                     # training the original leaves the copy alone
    assert np.any(agent.theta != before) and np.all(same.theta == before)
    with pytest.raises(ValueError):
        agent.with_parameters(np.zeros(86))
    with pytest.raises(ValueError):
        agent.with_parameters(np.zeros(87), (-1.0, 0.0))


def test_averaged_policy_of_constant_snapshots_is_the_last_iterate():
    agent = _drive(DriftAgent(np.random.default_rng(2), **{**HP, "cost_rls_after": None}), 300)
    n = 20
    log = {"checkpoint_block": np.arange(1, n + 1) * 10, "theta": np.tile(agent.theta, (n, 1)),
           "c1": np.full(n, agent.cost_linear), "c2": np.full(n, agent.cost_quadratic)}
    probes = config.probe_states()[::40]
    averaged, conv, favourites = tune.averaged_policy(agent, log, 200, probes)
    np.testing.assert_array_equal(averaged.theta, agent.theta)
    assert conv["primary_favourite_within_one_step"] and conv["original_all_probes_unchanged"]
    assert conv["split_half_favourite_l1"] == 0.0 and conv["split_half_probe_agreement"] == 1.0
    assert conv["n_snapshots_averaged"] == 2 and set(favourites.tolist()) == {agent.favourite_target()}


def test_rescore_matches_scoring_during_the_run(tmp_path):
    """A run rescored after the fact gets the same averaged results as a run
    that scored the averaged policy itself."""
    record = tune.run_one("a3e-4", 1, out_dir=tmp_path, n_blocks=12 * 180, segment=180, m=8)
    path = tmp_path / "a3e-4" / "seed1.json"
    z = dict(np.load(path.with_suffix(".npz")))
    old = {k: v for k, v in record.items() if k not in ("policy", "averaged_post_hoc", "last_iterate")}
    old.update(record["last_iterate"])                                 # the record as it was before the change
    path.write_text(json.dumps(old), encoding="utf-8")
    z["value"] = z.pop("last_iterate_value")
    z["favourite_rebalance_value"] = z.pop("last_iterate_favourite_rebalance_value")
    averaged_theta = z.pop("averaged_theta")
    np.savez_compressed(path.with_suffix(".npz"), **z)

    assert tune.rescore(tmp_path) == [str(path)]
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["averaged_post_hoc"] and again["policy"] == "tail_average"
    assert again["criterion_mean_value_bps"] == pytest.approx(record["criterion_mean_value_bps"], abs=1e-9)
    assert again["convergence"] == record["convergence"]
    assert again["last_iterate"]["criterion_mean_value_bps"] == record["last_iterate"]["criterion_mean_value_bps"]
    rescored = np.load(path.with_suffix(".npz"))
    np.testing.assert_array_equal(rescored["averaged_theta"], averaged_theta)
    np.testing.assert_array_equal(rescored["last_iterate_value"], z["value"])
    assert tune.rescore(tmp_path) == []                                # already rescored: left alone
    # A rescore that wrote the arrays but not the record refuses to run again.
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(RuntimeError):
        tune.rescore(tmp_path)


def test_a_rerun_of_the_pilot_scores_on_the_pilots_1000_paths():
    """config.M_EVAL is the final runs' 4,000; the pilot's records were scored at 1,000."""
    import inspect
    assert inspect.signature(tune.run_one).parameters["m"].default == config.M_EVAL_PILOTS == 1000
