"""Tests for the final-run driver (src/tc_drift/run.py), theta logging in
train() (AUDIT_2026-11-06.md TC-6), the favourite-target reference policy
(TC-3), the oracle-table policy, and the decided config values."""

import json
import math

import numpy as np
import pytest

from src.tc_drift import config, env, evaluate, oracle_replay, run, spec, train, tune
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

PROBES = config.probe_states()[::40]          # a few probes keep the tests quick


def _agent(seed=0, **overrides):
    return DriftAgent(np.random.default_rng(seed), **{**config.HYPERPARAMETERS, **overrides})


# --- config --------------------------------------------------------------------

def test_decided_values():
    assert config.pending() == []
    assert config.N_BLOCKS == 4_000_000 and config.CHECKPOINT_EVERY == tune.SEGMENT == 40_050
    assert config.M_EVAL == 4000 and config.CHECKPOINT_EVERY % config.RESTART_EVERY == 0
    hp = config.HYPERPARAMETERS
    assert hp["beta"] == pytest.approx(math.log(100) / 2_000_000)
    assert math.exp(-hp["beta"] * config.N_BLOCKS / 2) == pytest.approx(0.01)
    assert math.exp(-hp["beta"] * hp["cost_rls_after"]) == pytest.approx(0.5, abs=1e-6)
    assert set(hp) <= DriftAgent.HYPERPARAMETERS


def test_hyperparameters_are_the_tuning_pilots_winner():
    pilot = tune.hyperparameters("a3e-4")
    assert {k: v for k, v in config.HYPERPARAMETERS.items() if k != "cost_rls_prior"} == pilot
    assert config.HYPERPARAMETERS["cost_rls_prior"] == DriftAgent.DEFAULTS["cost_rls_prior"]


def test_checkpoints_are_the_layout_design_6_registered():
    blocks = np.append(np.arange(config.CHECKPOINT_EVERY, config.N_BLOCKS + 1, config.CHECKPOINT_EVERY),
                       config.N_BLOCKS)                         # train() always ends on n_blocks
    np.testing.assert_array_equal(blocks, np.cumsum(tune.segments(config.N_BLOCKS, tune.SEGMENT)))
    assert (blocks > 0.9 * config.N_BLOCKS).sum() == 11
    first = tune.window_start(blocks, config.N_BLOCKS)
    assert blocks[first] == 3_564_450 and len(blocks) - first == 12


def test_check5_keeps_its_evaluation_size():
    assert oracle_replay.M_EVAL == config.M_EVAL_PILOTS == 1000


# --- theta in the training log (TC-6) -------------------------------------------

def _train(agent, n_blocks, checkpoint_every, seed=3, probes=PROBES):
    market = env.load_market()
    r, _, s = config.run_rngs(seed, spec.TUNING_MASTER_SEED)
    return train.train(agent, market, cost_settings()["aum_1b"], n_blocks=n_blocks, returns_rng=r, start_rng=s,
                       start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                       checkpoint_every=checkpoint_every, probe_states=probes)


def test_theta_log_matches_williams_segmented_run():
    """One train() call with theta logged gives exactly the run tune.py builds
    from consecutive calls: the same theta, favourite and probe actions at
    every checkpoint, so the pilot's method and the final runs' agree."""
    n, seg = 900, 180
    log = _train(_agent(1), n, seg)
    market = env.load_market()
    r, _, s = config.run_rngs(3, spec.TUNING_MASTER_SEED)
    seg_log = tune.train_segmented(_agent(1), market, cost_settings()["aum_1b"], n, seg, r, s, PROBES)
    np.testing.assert_array_equal(log["checkpoint_block"], seg_log["checkpoint_block"])
    np.testing.assert_array_equal(log["theta"], seg_log["theta"])
    np.testing.assert_array_equal(log["favourite_target"], seg_log["favourite_target"])
    np.testing.assert_array_equal(log["probe_greedy"], seg_log["probe_greedy"])
    np.testing.assert_array_equal(log["diagnostics"]["cost_linear"], seg_log["c1"])


def test_theta_log_shape_and_last_row():
    agent = _agent(2)
    log = _train(agent, 450, 90)
    assert log["theta"].shape == (5, len(agent.theta))
    np.testing.assert_array_equal(log["theta"][-1], agent.theta)
    assert log["favourite_target"][-1] == agent.favourite_target()
    np.testing.assert_allclose(np.linalg.norm(log["theta"], axis=1), log["diagnostics"]["theta_norm"])


def test_agents_without_theta_log_none():
    class Holder:
        def select_action(self, state):
            return spec.HOLD

        def update(self, *args):
            pass

        def greedy_action(self, state):
            return spec.HOLD
    log = _train(Holder(), 180, 90)
    assert "theta" not in log and "favourite_target" not in log


# --- policies ------------------------------------------------------------------

def test_own_target_policy_trades_to_the_favourite():
    agent = _agent(4).with_parameters(np.random.default_rng(9).normal(size=len(_agent().theta)))
    pol = evaluate.own_target_rebalance_policy(agent)
    holdings = spec.target_grid()[:7]
    assert (pol(holdings, 0) == agent.favourite_target()).all()
    assert agent.favourite_target() == int(np.argmax(agent.holding_values(spec.target_grid())))


def test_oracle_table_policy_is_a_nearest_lattice_lookup():
    rng = np.random.default_rng(0)
    points = spec.lattice()
    table = rng.integers(0, spec.HOLD + 1, len(points))
    w = rng.dirichlet(np.ones(3), 50)
    pol = evaluate.oracle_table_policy(table)
    expected = [table[spec.nearest_lattice_index(x)] for x in w]
    np.testing.assert_array_equal(pol(w, 3), expected)
    with pytest.raises(ValueError):
        evaluate.oracle_table_policy(table.astype(float))
    with pytest.raises(ValueError):
        evaluate.oracle_table_policy(np.full(len(points), spec.HOLD + 1))


# --- the driver ----------------------------------------------------------------

def test_d12_versions_mapping():
    avg = {"primary_favourite_within_one_step": True, "secondary_exact_action_95pct_2026_11_19": False,
           "converged": True}
    last = {"original_all_probes_unchanged": False, "primary_favourite_within_one_step": True,
            "secondary_exact_action_95pct_2026_11_19": True}
    assert run.d12_versions(avg, last) == {"2026_11_12_original_primary_last_iterate": False,
                                           "2026_11_19_exact_action_last_iterate": True,
                                           "2026_11_19_tail_average": False, "2026_11_20_operative": True}


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    for s in (0, 1):
        run.train_one("aum_1b", s, out, n_blocks=1800, checkpoint_every=180)
    tables = {"policy_table": np.full(len(spec.lattice()), spec.HOLD),
              "myopic_policy_table": np.zeros(len(spec.lattice()), dtype=np.int64)}
    summary = run.evaluate_level("aum_1b", out, m=40, seeds=[0, 1], oracle_tables=tables)
    return out, summary


def test_driver_writes_its_files(smoke):
    out, summary = smoke
    folder = out / "aum_1b"
    for name in ("seed0.json", "seed0.npz", "seed1.json", "seed1.npz", "summary.json", "per_seed.csv",
                 "values.npz"):
        assert (folder / name).exists(), name
    for name in ("seed0.json", "summary.json", "per_seed.csv"):
        assert b"\r\n" not in (folder / name).read_bytes()
    record = json.loads((folder / "seed0.json").read_text())
    assert record["smoke"] and record["n_blocks"] == 1800 and record["episodes"] == 20 == record["n_restarts"] + 1
    assert record["simulated_days"] == 1800 * spec.BLOCK_LENGTH
    lines = (folder / "per_seed.csv").read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("seed,agent_minus_oracle_mean")
    assert summary["smoke"] and summary["m"] == 40 and len(summary["per_seed"]) == 2
    assert set(summary["per_seed"][0]["d12"]) == set(run.d12_versions(
        summary["per_seed"][0]["convergence_tail_average"], summary["per_seed"][0]["convergence_last_iterate"]))


def test_driver_values_recompute(smoke):
    """The summary's gaps are paired gaps of the saved per-path values, and the
    references are the policies they claim to be (independent recomputation)."""
    out, summary = smoke
    v = np.load(out / "aum_1b" / "values.npz")
    row = summary["per_seed"][1]
    gap = evaluate.paired_gap(v["seed1_agent"], v["ref_oracle"], config.H_EVAL)
    assert row["gaps_bps"]["agent_minus_oracle"] == pytest.approx(gap)
    split = (row["gaps_bps"]["split_agent_minus_own_target"]["mean"]
             + row["gaps_bps"]["split_own_target_minus_always_rebalance"]["mean"]
             - summary["oracle_minus_always_rebalance"]["mean"])
    assert split == pytest.approx(row["gaps_bps"]["agent_minus_oracle"]["mean"], abs=1e-9)

    market = env.load_market()
    settings = cost_settings()["aum_1b"]
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(), config.H_EVAL, 40)
    hold = evaluate.evaluate_policy(lambda h, t: np.full(len(h), spec.HOLD), starts, paths, settings)
    np.testing.assert_allclose(v["ref_oracle"], hold["value"])       # the table was all HOLD
    best = evaluate.best_static_target(market, settings.gamma)
    reb = evaluate.evaluate_policy(evaluate.always_rebalance_policy(best), starts, paths, settings)
    np.testing.assert_allclose(v["ref_always_rebalance"], reb["value"])


def test_rebuilt_agent_is_the_trained_one(smoke):
    out, _ = smoke
    record, z, last = run.load_run(out, "aum_1b", 0)
    np.testing.assert_array_equal(last.theta, z["theta"][-1])
    assert last.favourite_target() == record["final_favourite_index"]


def test_load_run_rejects_a_changed_log(smoke, tmp_path):
    out, _ = smoke
    folder = tmp_path / "aum_1b"
    folder.mkdir()
    for name in ("seed0.json", "seed0.npz"):
        (folder / name).write_bytes((out / "aum_1b" / name).read_bytes())
    z = dict(np.load(folder / "seed0.npz"))
    z["theta"] = z["theta"] * 1.01
    np.savez_compressed(folder / "seed0.npz", **z)
    with pytest.raises(RuntimeError, match="hash"):
        run.load_run(tmp_path, "aum_1b", 0)


def test_driver_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        run.train_one("aum_2b", 0, tmp_path, n_blocks=90, checkpoint_every=90)
    with pytest.raises(ValueError):
        run.train_one("aum_1b", 10, tmp_path, n_blocks=90, checkpoint_every=90)
    with pytest.raises(FileNotFoundError):
        run.evaluate_level("aum_1b", tmp_path, m=10, seeds=[0])
    with pytest.raises(FileNotFoundError):
        run.load_oracle_tables("aum_1b", tmp_path)


def test_oracle_tables_are_checked_against_the_summary(tmp_path):
    """The git-ignored tables must reproduce the committed summary (skipped
    where they have not been regenerated)."""
    if not (run.ORACLE_DIR / "aum_1b.npz").exists():
        pytest.skip("oracle tables not regenerated on this machine")
    tables = run.load_oracle_tables("aum_1b")
    assert len(tables["policy_table"]) == len(spec.lattice())
    (tmp_path / "aum_1b.npz").write_bytes((run.ORACLE_DIR / "aum_1b.npz").read_bytes())
    summary = json.loads((run.ORACLE_DIR / "summary.json").read_text())
    summary["settings"]["aum_1b"]["oracle_value_bps"] += 0.01
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="reproduce"):
        run.load_oracle_tables("aum_1b", tmp_path)


def test_job_runner_pins_threads_runs_in_parallel_and_reports_failures(tmp_path):
    import sys
    show = "import os; print(os.environ['OMP_NUM_THREADS'], os.environ['VECLIB_MAXIMUM_THREADS'])"
    jobs = [("ok_a", [sys.executable, "-c", show]), ("ok_b", [sys.executable, "-c", show]),
            ("bad", [sys.executable, "-c", "raise SystemExit(3)"])]
    failed = run._run_jobs(jobs, workers=2, log_dir=tmp_path)
    assert failed == ["bad"]
    assert (tmp_path / "ok_a.log").read_text().split() == ["1", "1"]
    assert (tmp_path / "ok_b.log").exists()
