"""Tests for the final-run driver (src/tc_drift/run.py).

In plain terms: the settings are the tuned ones; the theta that train() now
logs is the same theta the tuning pilot read between segments; one tiny
training run and one tiny scoring run write what the report needs, and the
batched agent policy scores exactly like the row-by-row one.
"""

import csv
import json
import math

import numpy as np
import pytest

from src.tc_drift import config, env, evaluate, run, spec, train, tune
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

N, CKPT = 1800, 90


def test_config_holds_the_tuned_settings():
    assert config.pending() == []
    assert config.HYPERPARAMETERS == {**tune.hyperparameters("a3e-4"), "cost_rls_prior": 1e-3}
    assert config.N_BLOCKS == tune.N_BLOCKS and config.CHECKPOINT_EVERY == tune.SEGMENT     # the pilot's grid
    assert config.HYPERPARAMETERS["beta"] == pytest.approx(math.log(100) / (config.N_BLOCKS / 2), rel=1e-15)
    assert config.M_EVAL == 4000


def test_logged_theta_is_the_theta_the_pilot_read():
    """train() logging theta in one call matches tune.train_segmented reading it between calls."""
    market = env.load_market()
    settings = cost_settings()["aum_1b"]
    probes = config.probe_states()[::50]
    hp = {**config.HYPERPARAMETERS, "cost_rls_after": 200, "target_refresh_every": 50}
    r1, a1, s1 = config.run_rngs(3)
    one = DriftAgent(a1, **hp)
    log = train.train(one, market, settings, n_blocks=N, returns_rng=r1, start_rng=s1,
                      start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                      checkpoint_every=CKPT * 2, probe_states=probes)
    r2, a2, s2 = config.run_rngs(3)
    seg = tune.train_segmented(DriftAgent(a2, **hp), market, settings, N, CKPT * 2, r2, s2, probes)
    np.testing.assert_array_equal(log["diagnostics"]["theta"], seg["theta"])
    flat = run.flat_log(log, one)
    np.testing.assert_array_equal(flat["favourite_target"], seg["favourite_target"])
    np.testing.assert_array_equal(flat["c1"], seg["c1"])


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    records = [run.train_one("aum_1b", s, out_dir=out, n_blocks=N, checkpoint_every=CKPT) for s in (0, 1)]
    return out, records


def test_train_one_writes_the_log_and_the_averaged_theta(trained):
    out, records = trained
    z = np.load(out / "aum_1b" / "seed0.npz")
    saved = json.loads((out / "aum_1b" / "seed0.json").read_text(encoding="utf-8"))
    assert saved["policy"] == "tail_average" and saved["n_blocks"] == N and saved["seed"] == 0
    assert z["theta"].shape == (N // CKPT, 87)
    np.testing.assert_array_equal(z["theta"][-1], z["final_theta"])
    tail = z["checkpoint_block"] > 0.9 * N
    np.testing.assert_allclose(z["averaged_theta"], z["theta"][tail].mean(axis=0), rtol=0, atol=0)
    assert set(saved["convergence"]) >= {"converged", "converged_from_block", "secondary_95pct_probes_within_one_step"}
    assert b"\r\n" not in (out / "aum_1b" / "seed0.json").read_bytes()
    # Different seeds, different runs.
    assert not np.array_equal(z["final_theta"], np.load(out / "aum_1b" / "seed1.npz")["final_theta"])


def test_evaluate_level_scores_every_policy_on_common_paths(trained, tmp_path):
    out, _ = trained
    n_lattice = len(spec.lattice())
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    # A stand-in oracle: never trade; myopic: always trade to target 0.
    np.savez(oracle_dir / "aum_1b.npz", policy_table=np.full(n_lattice, spec.HOLD),
             myopic_policy_table=np.zeros(n_lattice, dtype=np.int64))
    with pytest.raises(RuntimeError):
        run.evaluate_level("aum_1b", out_dir=out, m=6, oracle_dir=oracle_dir)          # 8 seeds missing
    s = run.evaluate_level("aum_1b", out_dir=out, m=6, oracle_dir=oracle_dir, require_all_seeds=False)
    assert s["seeds"] == [0, 1] and s["missing_seeds"] == list(range(2, 10)) and s["m_eval"] == 6
    assert set(s["across_seeds"]) == set(run.GAP_KEYS)
    row = s["per_seed"][0]
    # TC-3: the three gaps add up to the agent's gap to the oracle.
    lhs = row["agent_minus_oracle"]["mean"]
    rhs = (row["agent_minus_favourite_rebalance"]["mean"] + row["favourite_rebalance_minus_always_rebalance"]["mean"]
           - s["oracle_minus_always_rebalance"]["mean"])
    assert lhs == pytest.approx(rhs, abs=1e-9)
    assert set(row["verdicts"]) == {"d12_2026_11_12_original_primary", "d12_2026_11_19_last_iterate",
                                    "d12_2026_11_19_tail_average", "d12_2026_11_20"}
    # The stand-in oracle never trades, so its value is evaluate's never-trade value from the starts.
    v = np.load(out / "aum_1b" / "values.npz")
    market = env.load_market()
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(), config.H_EVAL, 6)
    np.testing.assert_array_equal(v["starts"], starts)
    hold = evaluate.evaluate_policy(lambda h, t: np.full(len(h), spec.HOLD), starts, paths, cost_settings()["aum_1b"])
    np.testing.assert_array_equal(v["ref_oracle"], hold["value"])
    with open(out / "aum_1b" / "per_seed.csv", encoding="utf-8") as fh:
        table = list(csv.reader(fh))
    assert len(table) == 3 and len(set(table[0])) == len(table[0]) == len(table[1])
    assert table[0][12:15] == ["favourite_spy", "favourite_ief", "favourite_gld"]
    # The seed-averaged path gap: the mean of the seeds' gaps, with a path SE.
    pooled = s["seed_averaged_path_gaps"]["agent_minus_oracle"]
    assert pooled["mean"] == pytest.approx(s["across_seeds"]["agent_minus_oracle"]["mean"], abs=1e-9)
    assert run.report(out)[0]["n_seeds"] == 2


def test_batched_agent_policy_scores_like_the_row_by_row_one(trained):
    out, _ = trained
    z = np.load(out / "aum_1b" / "seed1.npz")
    agent = DriftAgent(np.random.default_rng(0), **config.HYPERPARAMETERS).with_parameters(
        z["averaged_theta"], tuple(z["final_cost"]))
    market = env.load_market()
    starts, paths = evaluate.draw_evaluation_set(market, np.random.default_rng(5), 6, 40)
    settings = cost_settings()["aum_10b"]
    a = evaluate.evaluate_policy(run.batch_policy(agent), starts, paths, settings)
    b = evaluate.evaluate_policy(evaluate.agent_policy(agent), starts, paths, settings)
    np.testing.assert_array_equal(a["value"], b["value"])
    np.testing.assert_array_equal(a["hold_fraction"], b["hold_fraction"])


def test_own_target_rebalance_trades_to_the_favourite_target(trained):
    out, _ = trained
    z = np.load(out / "aum_1b" / "seed0.npz")
    agent = DriftAgent(np.random.default_rng(0), **config.HYPERPARAMETERS).with_parameters(z["averaged_theta"])
    actions = evaluate.own_target_rebalance_policy(agent)(spec.target_grid()[:5], 3)
    assert (actions == agent.favourite_target()).all()


def test_queue_and_scoring_refuse_records_from_other_settings(trained):
    """The tiny test runs are not final runs: the queue will not count them
    as done, and scoring with the final settings refuses them."""
    out, _ = trained
    with pytest.raises(ValueError):
        run.queue(2, out_dir=out, levels=["aum_1b"], seeds=[0, 1])
    with pytest.raises(ValueError):
        run._check_record(json.loads((out / "aum_1b" / "seed0.json").read_text(encoding="utf-8")),
                          config.N_BLOCKS, config.CHECKPOINT_EVERY)


def test_record_keeps_theta_and_writes_atomically(trained, tmp_path):
    out, _ = trained
    saved = json.loads((out / "aum_1b" / "seed0.json").read_text(encoding="utf-8"))
    z = np.load(out / "aum_1b" / "seed0.npz")
    np.testing.assert_array_equal(saved["averaged_theta"], z["averaged_theta"])
    np.testing.assert_array_equal(saved["final_theta"], z["final_theta"])
    run._write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1} and not (tmp_path / "x.json.tmp").exists()
