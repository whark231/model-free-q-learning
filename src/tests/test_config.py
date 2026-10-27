"""Checks on the experiment settings (src/config.py) and the batch runner
(src/run_experiments.py).

The first block pins the team's decisions as literal values, so changing one
breaks the build and forces the change to be recorded on purpose. The rest
checks that every experiment is runnable and that the batch runner trains,
evaluates, resumes and indexes correctly, using a fast stand-in agent.
"""

import json
import math

import numpy as np
import pytest

import src.config as cfg
import src.train as train
from src.action_space.grid import generate_weight_grid
from src.run_experiments import COMPARISON_NAME, run_all

# ---------------------------------------------------------------------------
# Decisions of 2026-10-16, pinned
# ---------------------------------------------------------------------------

def test_risk_aversion_decision():
    assert cfg.GAMMAS == (3.0, 5.0, 8.0)
    assert cfg.GAMMA_BASELINE == 5.0
    assert set(cfg.GAMMA_ROBUSTNESS) == {3.0, 8.0}
    assert set(cfg.GAMMAS) == {cfg.GAMMA_BASELINE, *cfg.GAMMA_ROBUSTNESS}


def test_episode_length_decision():
    assert cfg.EPISODE_LENGTH == 20


def test_convergence_threshold_is_ten_percent_of_training_episodes():
    assert cfg.CONVERGENCE_K_FRACTION == 0.10
    assert cfg.convergence_k(13_000_000) == 1_300_000
    assert cfg.convergence_k(2_001) == 201  # rounded up, never below 10%
    assert cfg.eval_config().converge_k == cfg.convergence_k(cfg.N_EPISODES)


def test_primary_convergence_rule_is_the_strict_reading():
    """The final greedy action held for K through the end of training; it must
    be one of the rules evaluate.py actually reports."""
    assert cfg.PRIMARY_CONVERGENCE_RULE == "greedy_stable_through_end"
    import inspect

    import src.evaluate as ev
    assert f'"{cfg.PRIMARY_CONVERGENCE_RULE}"' in inspect.getsource(ev._evaluate)


def test_proposed_defaults():
    assert (cfg.N_RUNS, cfg.N_EPISODES, cfg.DQ_TOL, cfg.NEAR_OPTIMAL_BPS) == (10, 13_000_000, 1e-5, 25.0)
    assert (cfg.INCREMENT_BASELINE, cfg.INCREMENT_COMPARISON) == (0.2, 0.25)
    assert (cfg.MASTER_SEED, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY) == (20261016, 10_000, 10_000)
    assert (cfg.EPSILON0, cfg.ALPHA_RULE, cfg.Q_INIT, cfg.DISCOUNT) == (1.0, "sample_average", 0.0, 0.0)


def test_train_and_eval_configs_pass_every_setting_through():
    experiment = cfg.experiments()[0]
    tc = cfg.train_config(experiment, n_episodes=4_000, n_runs=3, log_every=500)
    assert (tc.n_episodes, tc.n_runs, tc.log_every) == (4_000, 3, 500)
    assert tc.beta == cfg.beta_for(4_000)
    assert (tc.epsilon0, tc.alpha_rule, tc.q_init, tc.discount) == (1.0, "sample_average", 0.0, 0.0)
    assert (tc.master_seed, tc.episode_length) == (20261016, 20)
    ec = cfg.eval_config(n_episodes=4_000, n_eval_episodes=321)
    assert (ec.converge_k, ec.n_eval_episodes, ec.dq_tol, ec.near_optimal_bps) == (400, 321, 1e-5, 25.0)


def test_beta_makes_epsilon_one_percent_halfway_through_training():
    for n in (1_000, 13_000_000):
        assert math.exp(-cfg.beta_for(n) * n / 2) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# The experiment list covers what the report needs, and every entry runs
# ---------------------------------------------------------------------------

def test_experiment_list():
    experiments = {e.name: e for e in cfg.experiments()}
    assert len(experiments) == 4
    base = experiments["baseline_g5_grid20"]
    assert (base.gamma, base.increment) == (5.0, 0.2)
    assert (experiments["resolution_g5_grid25"].gamma, experiments["resolution_g5_grid25"].increment) == (5.0, 0.25)
    assert {(e.gamma, e.increment) for n, e in experiments.items() if n.startswith("robust")} == {(3.0, 0.2), (8.0, 0.2)}


def test_resolution_comparison_differs_from_the_baseline_only_in_the_grid():
    """evaluate.compare_experiments refuses anything else."""
    experiments = {e.name: e for e in cfg.experiments()}
    a = cfg.train_config(experiments["baseline_g5_grid20"]).__dict__
    b = cfg.train_config(experiments["resolution_g5_grid25"]).__dict__
    assert {k for k in a if a[k] != b[k]} == {"increment"}


def test_every_full_scale_experiment_fits_the_snapshot_memory_guard():
    for experiment in cfg.experiments():
        tc = cfg.train_config(experiment)
        assert tc.gamma in cfg.GAMMAS and tc.episode_length == cfg.EPISODE_LENGTH
        n_actions = generate_weight_grid(9, tc.increment).shape[0]
        n_checkpoints = -(-tc.n_episodes // tc.log_every)
        assert n_checkpoints * n_actions <= train.MAX_SNAPSHOT_VALUES, experiment.name


# ---------------------------------------------------------------------------
# Batch runner, at tiny scale with a stand-in agent
# ---------------------------------------------------------------------------

class AveragingExplorer:
    """Uniform exploration with sample-average Q-values: fast and Q-dependent."""

    epsilon = 1.0  # always explores

    def __init__(self, n_actions, rng):
        self._rng, self._q, self._n = rng, np.zeros(n_actions), np.zeros(n_actions, dtype=int)

    def select_action(self):
        return int(self._rng.integers(self._q.size))

    def update(self, action, reward):
        self._n[action] += 1
        self._q[action] += (reward - self._q[action]) / self._n[action]

    @property
    def q_values(self):
        return self._q.copy()

    @property
    def visit_counts(self):
        return self._n.copy()


def factory(config, n_actions, rng):
    return AveragingExplorer(n_actions, rng)


SMALL = dict(n_episodes=1_000, n_runs=2, n_eval_episodes=200, log_every=250, agent_factory=factory)


def test_dry_run_lists_every_experiment_without_writing(tmp_path, capsys):
    plan = run_all(tmp_path, dry_run=True)
    assert plan["planned"] == [e.name for e in cfg.experiments()]
    assert not any(tmp_path.iterdir())
    assert "K=1,300,000" in capsys.readouterr().out


def test_runs_everything_and_writes_an_index(tmp_path):
    index = run_all(tmp_path, **SMALL)
    names = [e.name for e in cfg.experiments()]
    assert [e["name"] for e in index["experiments"]] == names
    for name in names:
        assert (tmp_path / name / "manifest.json").exists()
        assert (tmp_path / name / "evaluation" / "summary.json").exists()
    assert (tmp_path / COMPARISON_NAME / "convergence_by_resolution.png").exists()
    on_disk = json.loads((tmp_path / "index.json").read_text())
    assert on_disk["settings"]["convergence_k"] == 100  # 10% of 1,000
    assert on_disk["full_scale"] is False
    assert json.loads((tmp_path / "settings.json").read_text()) == on_disk["settings"]
    markdown = (tmp_path / "INDEX.md").read_text()
    assert "Reduced settings" in markdown and "baseline_g5_grid20" in markdown
    assert "greedy_stable_through_end" in markdown
    gammas = {e["name"]: json.loads((tmp_path / e["name"] / "manifest.json").read_text())["config"]["gamma"]
              for e in index["experiments"]}
    assert gammas == {"baseline_g5_grid20": 5.0, "resolution_g5_grid25": 5.0, "robust_g3_grid20": 3.0, "robust_g8_grid20": 8.0}


def test_index_values_equal_the_evaluation_summaries(tmp_path):
    index = run_all(tmp_path, **SMALL)
    for entry in index["experiments"]:
        summary = json.loads((tmp_path / entry["name"] / "evaluation" / "summary.json").read_text())
        assert entry["discretization_gap_bps"] == summary["discretization_gap"]["ce_loss_bps"]
        assert entry["learning_gap_bps"] == summary["learned"]["ce_loss_vs_best_discrete_bps"]
        assert entry["convergence"]["greedy_stable_through_end"] == summary["convergence"]["greedy_stable_through_end"]
        assert (entry["gamma"], entry["increment"]) == (summary["training_config"]["gamma"], summary["training_config"]["increment"])
        assert entry["converge_k"] == summary["eval_config"]["converge_k"] == 100
    comparison = json.loads((tmp_path / COMPARISON_NAME / "comparison.json").read_text())
    assert index["comparison"]["results"] == comparison
    assert set(comparison) == {"25%", "20%"} and comparison["25%"]["n_actions"] == 495


def test_rerun_of_the_same_experiment_reuses_its_training(tmp_path):
    run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)
    npz = tmp_path / "baseline_g5_grid20" / "run_000.npz"
    before = npz.stat().st_mtime_ns
    run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)
    assert npz.stat().st_mtime_ns == before
    index = run_all(tmp_path, only=["robust_g3_grid20"], **SMALL)
    assert npz.stat().st_mtime_ns == before
    assert {e["name"] for e in index["experiments"]} == {"baseline_g5_grid20", "robust_g3_grid20"}


def test_a_folder_with_different_settings_is_refused_and_left_untouched(tmp_path):
    """A smoke-sized call into a folder of finished work must not relabel or
    overwrite it."""
    run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)
    summary_path = tmp_path / "baseline_g5_grid20" / "evaluation" / "summary.json"
    before = summary_path.read_bytes()
    for change in (dict(n_episodes=2_000), dict(n_runs=1), dict(n_eval_episodes=100), dict(log_every=500)):
        with pytest.raises(ValueError, match="different settings"):
            run_all(tmp_path, only=["baseline_g5_grid20"], **{**SMALL, **change})
    assert summary_path.read_bytes() == before


def test_an_experiment_trained_with_other_settings_is_refused(tmp_path):
    run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)
    manifest_path = tmp_path / "baseline_g5_grid20" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["beta"] = 0.5
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trained with different settings.*beta"):
        run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)


def test_a_half_finished_experiment_is_refused_with_instructions(tmp_path):
    run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)
    manifest_path = tmp_path / "baseline_g5_grid20" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"] = manifest["runs"][:1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partly trained .1 of 2 runs.*Delete that one folder"):
        run_all(tmp_path, only=["baseline_g5_grid20"], **SMALL)


def test_unknown_experiment_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment"):
        run_all(tmp_path, only=["baseline_g5_grid2"], **SMALL)


def test_full_scale_flag_reflects_every_setting():
    from src.run_experiments import batch_settings
    full = batch_settings(cfg.N_EPISODES, cfg.N_RUNS, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY)
    for args in ((cfg.N_EPISODES, cfg.N_RUNS, 500, cfg.LOG_EVERY), (cfg.N_EPISODES, cfg.N_RUNS, cfg.N_EVAL_EPISODES, 500),
                 (cfg.N_EPISODES, 2, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY)):
        assert batch_settings(*args) != full


# ---------------------------------------------------------------------------
# Post-hoc variants (EXPERIMENTS.md §7): the baseline stays exactly baseline-v1
# ---------------------------------------------------------------------------

REPO = train.REPO_ROOT
VARIANT = cfg.RunVariant("test-variant", "T", "2026-10-27", "a smaller learning gap",
                         episode_length=5, epsilon_at_halfway=0.1, increment=0.25)


def test_baseline_settings_still_match_the_committed_baseline_v1_run():
    """Re-derived from the files the 2026-10-22 run wrote, not from config.py."""
    from src.run_experiments import batch_settings
    committed = json.loads((REPO / "results" / "settings.json").read_text())
    assert batch_settings(cfg.N_EPISODES, cfg.N_RUNS, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY) == committed
    for experiment in cfg.experiments():
        manifest = json.loads((REPO / "results" / experiment.name / "manifest.json").read_text())
        assert json.loads(json.dumps(cfg.train_config(experiment).__dict__)) == manifest["config"], experiment.name


def test_a_variant_changes_only_its_own_settings():
    assert set(VARIANT.changes()) == {"episode_length", "epsilon_at_halfway", "increment"}
    experiments = cfg.experiments(VARIANT)
    assert [e.name for e in experiments] == ["baseline_g5_grid25_T", "robust_g3_grid25_T", "robust_g8_grid25_T"]
    assert [e.gamma for e in experiments] == [5.0, 3.0, 8.0]
    baseline_names = {e.name for e in cfg.experiments()}
    for experiment in experiments:
        assert experiment.name not in baseline_names
        tc = cfg.train_config(experiment, n_episodes=VARIANT.n_episodes)
        base = cfg.train_config(cfg.Experiment("b", experiment.gamma, cfg.INCREMENT_BASELINE, ""))
        differ = {k for k in tc.__dict__ if tc.__dict__[k] != base.__dict__[k]}
        assert differ == {"episode_length", "beta", "increment"}
        assert (tc.episode_length, tc.increment) == (5, 0.25)
        assert math.exp(-tc.beta * tc.n_episodes / 2) == pytest.approx(0.1)


def test_every_registered_variant_is_runnable_and_fits_the_memory_guard():
    for run_id, run in cfg.POST_HOC_RUNS.items():
        assert run.run_id == run_id and run.changes() and run.expected and run.registered
        names = [e.name for e in cfg.experiments(run)]
        assert len(set(names)) == 3 and all(n.endswith("_" + run.suffix) for n in names)
        n_actions = generate_weight_grid(9, run.increment).shape[0]
        n_checkpoints = -(-run.n_episodes // cfg.LOG_EVERY)
        assert n_checkpoints * n_actions <= train.MAX_SNAPSHOT_VALUES, run_id


def test_variant_batch_records_its_settings_and_is_labelled_post_hoc(tmp_path):
    index = run_all(tmp_path, run=VARIANT, **SMALL)
    assert [e["name"] for e in index["experiments"]] == [e.name for e in cfg.experiments(VARIANT)]
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["post_hoc_run"] == "test-variant" and settings["episode_length"] == 5
    assert settings["increments"] == [0.25] and settings["beta"] == cfg.beta_for(1_000, 0.1)
    for e in index["experiments"]:
        config = json.loads((tmp_path / e["name"] / "manifest.json").read_text())["config"]
        assert (config["episode_length"], config["increment"], config["n_episodes"]) == (5, 0.25, 1_000)
        assert math.exp(-config["beta"] * 500) == pytest.approx(0.1)
    assert index["comparison"] is None and index["post_hoc_run"]["run_id"] == "test-variant"
    assert "Post-hoc run `test-variant`" in (tmp_path / "INDEX.md").read_text()


def test_a_variant_is_refused_in_a_baseline_folder_and_vice_versa(tmp_path):
    run_all(tmp_path / "base", only=["baseline_g5_grid20"], **SMALL)
    with pytest.raises(ValueError, match="different settings"):
        run_all(tmp_path / "base", run=VARIANT, **SMALL)
    run_all(tmp_path / "variant", only=["baseline_g5_grid25_T"], run=VARIANT, **SMALL)
    with pytest.raises(ValueError, match="different settings"):
        run_all(tmp_path / "variant", only=["baseline_g5_grid20"], **SMALL)


def test_variant_dry_run_defaults_to_its_own_training_budget(tmp_path, capsys):
    run = cfg.RunVariant("budget", "N2", "2026-10-27", "x", n_episodes=26_000_000)
    run_all(tmp_path, run=run, dry_run=True)
    out = capsys.readouterr().out
    assert "26,000,000 episodes" in out and "K=2,600,000" in out and not any(tmp_path.iterdir())
