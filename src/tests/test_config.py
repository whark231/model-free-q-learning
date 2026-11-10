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
from src.train import config_record

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


def test_a_tabular_config_does_not_match_a_function_approximation_folder(tmp_path):
    """training_status compares every key on either side, so an FA folder
    (extra keys) is never taken as a finished tabular run, even with the same
    constant alpha."""
    from src.run_experiments import training_status
    from src.train import TrainConfig
    common = dict(gamma=5.0, episode_length=20, n_episodes=1_000, master_seed=1, n_runs=1, beta=0.0,
                  increment=0.25, alpha_rule=0.1)
    fa = TrainConfig(**common, agent="linear", feature_map="quadratic", greedy_refresh_every=10, log_every=500)
    run_dir = tmp_path / "fa"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"config": config_record(fa), "runs": [{}]}))
    (run_dir / "run_000.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="different settings"):
        training_status(run_dir, TrainConfig(**common, log_every=500))
    assert training_status(run_dir, fa) == "finished"


def test_baseline_settings_still_match_the_committed_baseline_v1_run():
    """Re-derived from the files the 2026-10-22 run wrote, not from config.py."""
    from src.run_experiments import batch_settings
    committed = json.loads((REPO / "results" / "settings.json").read_text())
    assert batch_settings(cfg.N_EPISODES, cfg.N_RUNS, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY) == committed
    for experiment in cfg.experiments():
        manifest = json.loads((REPO / "results" / experiment.name / "manifest.json").read_text())
        assert json.loads(json.dumps(config_record(cfg.train_config(experiment)))) == manifest["config"], experiment.name


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


def test_post_hoc_run_b63_is_registered_as_written_in_experiments_md():
    run = cfg.POST_HOC_RUNS["b63"]
    assert run.changes() == {"episode_length": (20, 63)}
    assert run.registered == "2026-10-27"
    assert [e.name for e in cfg.experiments(run)] == ["baseline_g5_grid20_B63", "robust_g3_grid20_B63", "robust_g8_grid20_B63"]


def test_post_hoc_run_n26m_is_registered_as_written_in_experiments_md():
    run = cfg.POST_HOC_RUNS["n26m"]
    assert run.changes() == {"n_episodes": (13_000_000, 26_000_000)}
    assert run.registered == "2026-10-27"
    assert [e.name for e in cfg.experiments(run)] == ["baseline_g5_grid20_N26M", "robust_g3_grid20_N26M", "robust_g8_grid20_N26M"]


def test_post_hoc_run_eps_decay_is_registered_as_written_in_experiments_md():
    run = cfg.POST_HOC_RUNS["eps-decay"]
    assert run.changes() == {"epsilon_at_halfway": (0.01, 0.03)}
    assert run.registered == "2026-10-27"
    assert [e.name for e in cfg.experiments(run)] == ["baseline_g5_grid20_eps3", "robust_g3_grid20_eps3", "robust_g8_grid20_eps3"]


def test_post_hoc_run_all_three_is_registered_as_written_in_experiments_md():
    run = cfg.POST_HOC_RUNS["all-three"]
    assert run.changes() == {"episode_length": (20, 63), "n_episodes": (13_000_000, 26_000_000), "epsilon_at_halfway": (0.01, 0.03)}
    assert run.registered == "2026-10-27"
    assert [e.name for e in cfg.experiments(run)] == ["baseline_g5_grid20_B63_N26M_eps3", "robust_g3_grid20_B63_N26M_eps3", "robust_g8_grid20_B63_N26M_eps3"]


def test_post_hoc_run_grid10_robustness_is_registered_as_written_in_experiments_md():
    run = cfg.POST_HOC_RUNS["grid10-robustness"]
    assert run.changes() == {"episode_length": (20, 63), "n_episodes": (13_000_000, 26_000_000),
                             "epsilon_at_halfway": (0.01, 0.03), "increment": (0.2, 0.1)}
    assert run.registered == "2026-10-27"
    assert [e.name for e in cfg.experiments(run)] == [
        "baseline_g5_grid10_B63_N26M_eps3", "robust_g3_grid10_B63_N26M_eps3", "robust_g8_grid10_B63_N26M_eps3"]
    assert generate_weight_grid(9, run.increment).shape[0] == 43_758  # the count the assignment states


def test_random_tries_per_portfolio_quoted_in_experiments_md():
    """(1 - exp(-beta N)) / beta / n_actions, re-derived by summing the schedule."""
    def tries(n, eps_half, n_actions):
        beta = cfg.beta_for(n, eps_half)
        return np.exp(-beta * np.arange(n, dtype=float)).sum() / n_actions
    assert round(tries(13_000_000, 0.01, 1_287)) == 1_097
    assert round(tries(13_000_000, 0.01, 43_758)) == 32
    assert round(tries(26_000_000, 0.01, 43_758)) == 65
    assert round(tries(26_000_000, 0.03, 43_758)) == 85
    assert round(tries(26_000_000, 0.10, 43_758)) == 128


def test_a_variant_experiment_carries_its_own_training_budget():
    """cfg.train_config(e) without n_episodes must not fall back to the baseline's 13M."""
    for run in cfg.POST_HOC_RUNS.values():
        for experiment in cfg.experiments(run):
            tc = cfg.train_config(experiment)
            assert tc.n_episodes == run.n_episodes
            assert tc.beta == cfg.beta_for(run.n_episodes, run.epsilon_at_halfway)
    assert cfg.train_config(cfg.experiments()[0]).n_episodes == 13_000_000


def test_grid10_near_ties_quoted_in_experiments_md():
    """The §7.7 table, re-derived from μ and Σ (analysis only; the agent never sees them)."""
    from src.environment import PROCESSED_DIR, load_calibration
    from src.evaluate import objective
    mu, sigma = load_calibration(PROCESSED_DIR)
    expected = {3.0: (0.04, 5), 5.0: (4.45, 25), 8.0: (1.25, 36)}
    for gamma, (second_gap, within25) in expected.items():
        loss = 252e4 * (lambda v: v.max() - v)(objective(generate_weight_grid(9, 0.1), mu, sigma, gamma))
        assert round(np.sort(loss)[1], 2) == second_gap
        assert int((loss <= 25).sum()) == within25


# ---------------------------------------------------------------------------
# Constant-learning-rate runs (EXPERIMENTS.md §8, registered 2026-11-05)
# ---------------------------------------------------------------------------

TUNING = json.loads((REPO / "results" / "tuning" / "fa_tuning.json").read_text())


def test_every_committed_post_hoc_settings_file_still_matches_its_variant():
    """Adding agent fields must not make finished post-hoc folders look different."""
    from src.run_experiments import batch_settings
    for run_id in ("b63", "n26m", "eps-decay", "all-three", "grid10-robustness"):
        run = cfg.POST_HOC_RUNS[run_id]
        committed = json.loads((REPO / "results" / run_id / "settings.json").read_text())
        assert batch_settings(run.n_episodes, cfg.N_RUNS, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY, run) == committed, run_id


def test_tabular_const_changes_only_the_learning_rate_to_the_tuned_value():
    run = cfg.POST_HOC_RUNS["tabular-const"]
    assert run.changes() == {"alpha_rule": ("sample_average", TUNING["chosen"]["tabular"]["alpha"])}
    assert run.registered == "2026-11-05"
    assert [e.name for e in cfg.experiments(run)] == [
        "baseline_g5_grid20_TABconst", "robust_g3_grid20_TABconst", "robust_g8_grid20_TABconst"]
    for experiment in cfg.experiments(run):
        record = config_record(cfg.train_config(experiment))
        assert record["alpha_rule"] == 0.00032 and "agent" not in record  # a tabular manifest
        assert record["master_seed"] == 20261016 and record["n_runs"] == 10  # the reported seeds, not tuning's


def test_fa_quadratic_uses_the_tuned_alpha_and_the_reported_settings():
    run = cfg.POST_HOC_RUNS["fa-quadratic"]
    chosen = TUNING["chosen"]["quadratic"]
    assert run.changes() == {"alpha_rule": ("sample_average", chosen["alpha"]), "agent": ("tabular", "linear"),
                             "feature_map": (None, "quadratic"), "greedy_refresh_every": (1, 100)}
    assert run.l2 == chosen["l2"] == 0.0
    for experiment, gamma in zip(cfg.experiments(run), (5.0, 3.0, 8.0)):
        tc = cfg.train_config(experiment)
        assert (tc.agent, tc.feature_map, tc.alpha_rule, tc.greedy_refresh_every) == ("linear", "quadratic", 0.00032, 100)
        assert (tc.gamma, tc.episode_length, tc.n_episodes, tc.increment) == (gamma, 20, 13_000_000, 0.2)
        assert tc.master_seed == 20261016 and tc.beta == cfg.beta_for(13_000_000)
        assert tc.n_episodes % tc.greedy_refresh_every == 0 and cfg.LOG_EVERY % tc.greedy_refresh_every == 0


def test_fa_spline_uses_the_alpha_and_l2_chosen_on_six_seeds():
    run = cfg.POST_HOC_RUNS["fa-spline"]
    chosen = TUNING["chosen"]["spline"]
    assert chosen["decided_on_seeds"] == 6 and chosen["close_call"]
    assert (run.alpha_rule, run.l2) == (chosen["alpha"], chosen["l2"]) == (cfg.TUNED_ALPHA_SPLINE, 0.0)
    assert run.alpha_rule == pytest.approx(1.6e-4 * 2 / 45)  # the fraction of the spline bound
    for experiment in cfg.experiments(run):
        tc = cfg.train_config(experiment)
        assert (tc.agent, tc.feature_map, tc.greedy_refresh_every) == ("linear", "spline", 100)
        assert tc.master_seed == 20261016 and tc.n_runs == 10 and tc.episode_length == 20


def test_an_fa_variant_runs_end_to_end_and_records_its_agent(tmp_path):
    run = cfg.POST_HOC_RUNS["fa-quadratic"]
    index = run_all(tmp_path, run=run, n_episodes=1_000, n_runs=2, n_eval_episodes=200, log_every=500)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert (settings["agent"], settings["feature_map"], settings["alpha_rule"]) == ("linear", "quadratic", 0.00032)
    assert settings["greedy_refresh_every"] == 100 and settings["l2"] == 0.0
    for e in index["experiments"]:
        manifest = json.loads((tmp_path / e["name"] / "manifest.json").read_text())
        assert manifest["config"]["agent"] == "linear" and manifest["n_features"] == 45
    # The tabular-const folder refuses FA settings and the other way round.
    with pytest.raises(ValueError, match="different settings"):
        run_all(tmp_path, run=cfg.POST_HOC_RUNS["tabular-const"], n_episodes=1_000, n_runs=2,
                n_eval_episodes=200, log_every=500)


def test_a_variant_passes_every_agent_setting_through_to_training(capsys, tmp_path):
    """Every registered l2 is 0, so a made-up variant checks l2 (and the rest) reach TrainConfig."""
    run = cfg.RunVariant("x", "X", "2026-11-05", "x", alpha_rule=1e-5, agent="linear", feature_map="spline",
                         l2=1e-3, greedy_refresh_every=50)
    for experiment in cfg.experiments(run):
        tc = cfg.train_config(experiment)
        assert (tc.alpha_rule, tc.agent, tc.feature_map, tc.l2, tc.greedy_refresh_every) == \
            (1e-5, "linear", "spline", 1e-3, 50)
    run_all(tmp_path, run=cfg.POST_HOC_RUNS["fa-spline"], dry_run=True)
    out = capsys.readouterr().out
    assert out.count("agent linear spline (l2=0), alpha 7.111111111111112e-06") == 3


# ---------------------------------------------------------------------------
# Audit fixes, 2026-11-09 (AUDIT_2026-11-06.md M1 and M10)
# ---------------------------------------------------------------------------

def test_index_shows_total_gap_settled_and_q_fit(tmp_path):
    index = run_all(tmp_path, **SMALL)
    markdown = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "Total gap, median (range)" in markdown and "Settled (post hoc), median (range)" in markdown
    assert "Near-optimal for good (checkpoints)" in markdown and "| Q fit |" in markdown
    for entry in index["experiments"]:
        summary = json.loads((tmp_path / entry["name"] / "evaluation" / "summary.json").read_text())
        assert entry["total_gap_bps"] == summary["learned"]["ce_loss_vs_continuous_bps"]
        assert entry["settled"] == summary["convergence"]["settled"]
        assert entry["q_spearman_all"] == summary["q_fit"]["all"]["spearman"]
    comparison = json.loads((tmp_path / COMPARISON_NAME / "comparison.json").read_text())
    assert "ce_loss_vs_continuous_bps" in comparison["20%"] and "settled" in comparison["20%"]


def test_index_reads_summaries_written_before_the_new_fields(tmp_path):
    from src.run_experiments import _index_entry, _index_markdown
    run_all(tmp_path, **SMALL)
    summary = json.loads((tmp_path / "baseline_g5_grid20" / "evaluation" / "summary.json").read_text())
    del summary["learned"]["ce_loss_vs_continuous_bps"], summary["convergence"]["settled"], summary["q_fit"]
    entry = _index_entry(cfg.experiments()[0], summary)
    index = json.loads((tmp_path / "index.json").read_text())
    index["experiments"] = [entry]
    row = [line for line in _index_markdown(index).splitlines() if line.startswith("| baseline_g5_grid20")][0]
    assert row.count("n/a") == 3


def test_dry_run_prints_gamma_on_a_cp1252_console():
    """Windows consoles default to cp1252, which has no γ (audit M1)."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    for module in ("src.run_experiments", "src.evaluate"):
        args = [sys.executable, "-m", module, "--help"] if module == "src.evaluate" else \
               [sys.executable, "-m", module, "--run", "tabular-const", "--dry-run"]
        done = subprocess.run(args, capture_output=True, env=env, cwd=train.REPO_ROOT)
        assert done.returncode == 0, done.stderr.decode(errors="replace")


def test_a_committed_manifest_without_its_logs_is_refused_clearly(tmp_path):
    """A fresh clone has committed manifests but no .npz logs (audit F-H1)."""
    from src.run_experiments import training_status
    experiment = cfg.experiments()[0]
    run_all(tmp_path, only=[experiment.name], **SMALL)
    run_dir = tmp_path / experiment.name
    (run_dir / "run_000.npz").unlink()
    train_cfg = cfg.train_config(experiment, SMALL["n_episodes"], SMALL["n_runs"], SMALL["log_every"])
    with pytest.raises(FileNotFoundError, match="--results-dir results/repro"):
        training_status(run_dir, train_cfg)
    assert (run_dir / "manifest.json").exists()  # nothing deleted
