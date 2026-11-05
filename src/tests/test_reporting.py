"""Tests for src/reporting: the per-run extractor and the multi-run report builder.

Where possible each number is re-derived a different way: the reward SD by
simulation, excursions and settling points by hand-built change logs, and the
extractor's per-seed results against src/evaluate.py on a small real run.
"""

import json

import numpy as np
import pytest

from src import config as cfg
from src.environment import load_calibration
from src.evaluate import evaluate_experiment
from src.reporting import build, extract
from src.run_experiments import batch_settings
from src.train import run_experiment


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [2, 20, 63])
def test_reward_sd_matches_simulation(B):
    mu, sigma = load_calibration()
    w = np.zeros(9)
    w[[0, 3, 6]] = [0.4, 0.2, 0.4]
    rng = np.random.default_rng(11)
    r = rng.multivariate_normal(mu, sigma, size=(40_000, B)) @ w
    sim = (r.mean(1) - 2.5 * r.var(1, ddof=1)).std()
    assert extract.reward_sd(w, mu, sigma, 5.0, B) == pytest.approx(sim, rel=0.02)


def test_tries_to_separate_gives_a_two_standard_error_gap():
    sd, gap = 1.5e-3, 4.2e-6
    n = extract.tries_to_separate(sd, gap)
    assert 2 * np.sqrt(2 * sd**2 / n) == pytest.approx(gap)


def test_settled_from_is_first_checkpoint_after_the_last_disagreement():
    ckpt = np.array([10, 20, 30, 40, 50])
    assert extract.settled_from(ckpt, np.array([1, 7, 7, 2, 7]), final=7) == 50
    assert extract.settled_from(ckpt, np.array([1, 7, 3, 7, 7]), final=7) == 40
    assert extract.settled_from(ckpt, np.array([7, 7, 7, 7, 7]), final=7) == 10


@pytest.mark.parametrize("greedy", [[1, 7, 2], [1, 2, 3], []])
def test_settled_from_is_missing_when_the_final_policy_did_not_settle(greedy):
    assert extract.settled_from(np.arange(len(greedy)), np.array(greedy), final=7) is None


def test_fa_tied_final_policy_agrees_across_training_evaluation_and_reporting(tmp_path):
    from src.train import TrainConfig
    from src.evaluate import EvalConfig
    tc = TrainConfig(gamma=5, episode_length=20, n_episodes=1, master_seed=2, n_runs=1,
                     beta=0.001, log_every=1, agent="linear", feature_map="quadratic", alpha_rule=0.1)
    manifest = run_experiment(tc, tmp_path)
    with np.load(tmp_path / "run_000.npz") as logs:
        final = int(np.argmax(logs["final_q"]))
        cached = int(logs["checkpoint_greedy"][-1])
        assert cached != final  # real calibration and seed reproduce the review finding
        assert logs["final_q"][cached] == logs["final_q"][final]
        assert int(logs["final_greedy"]) == final
    assert manifest["runs"][0]["final_greedy_action"] == final
    summary = evaluate_experiment(tmp_path, EvalConfig(100, 1, 1e-5, 25))
    mu, sigma = load_calibration()
    report = extract.extract_experiment(tmp_path, {"near_optimal_bps": 25, "dq_tol": 1e-5},
                                        mu, sigma, extract.PROCESSED_DIR)
    assert summary["per_seed"][0]["learned_action"] == report["seeds"][0]["action"] == final
    assert report["seeds"][0]["settled_obs"] is None
    assert report["summary"]["settled"]["n"] == 0
    assert report["seeds"][0]["primary_obs"] is None
    assert summary["per_seed"][0]["greedy_stable_through_end_episode"] is None


def test_excursions_count_a_hop_between_two_wrong_actions_once():
    loss = np.array([0.0, 50.0, 300.0, 20.0])
    ch_ep = np.array([0, 100, 200, 205, 210, 400, 403])
    ch_act = np.array([3, 0, 1, 2, 0, 3, 0])  # settle at 100; excursion 200-210 visits 1 then 2; then 400-403 visits 3
    out = extract.excursions(ch_ep, ch_act, final=0, after=100, n_episodes=1_000, loss_by_action=loss)
    assert out == [(200, 10, 300.0), (400, 3, 20.0)]


def test_near_optimal_every_episode_counts_brief_switches():
    loss = np.array([0.0, 10.0, 80.0])
    ch_ep, ch_act = np.array([0, 50, 90, 95]), np.array([2, 1, 2, 0])
    assert extract.near_optimal_every_episode(ch_ep, ch_act, loss, 25.0) == 95
    assert extract.near_optimal_every_episode(ch_ep[:2], ch_act[:2], loss, 25.0) == 50
    assert extract.near_optimal_every_episode(np.array([0, 9]), np.array([0, 2]), loss, 25.0) is None


def test_bin_series_keeps_spikes_and_passes_short_series_through():
    x = np.arange(1, 1001)
    loss = np.zeros(1000)
    loss[537] = 99.0
    xb, yb = extract.bin_series(x, {"loss": loss, "avg": np.arange(1000.0)}, {"loss": "max", "avg": "last"}, max_points=100)
    assert len(xb) == 100 and xb[-1] == 1000
    assert yb["loss"].max() == 99.0 and yb["avg"][-1] == 999.0
    xs, ys = extract.bin_series(x[:50], {"loss": loss[:50]}, {"loss": "max"}, max_points=100)
    assert len(xs) == 50


def test_family_key():
    assert extract.family_key(5.0, 0.2) == "gamma5_grid20"
    assert extract.family_key(3.0, 0.25) == "gamma3_grid25"


def _exp(**params):
    base = {k: 1 for k, _ in build.COMPARED_PARAMS}
    base.update(params)
    return {"params": base, "calibration_sha256": {"a": "x"}, "code": {"code_sha256": {"src/train.py": "1"}}}


def test_param_changes_lists_only_real_differences():
    old, new = _exp(episode_length=20, epsilon_at_half=0.01), _exp(episode_length=63, epsilon_at_half=0.01 + 1e-16)
    changes = build.param_changes(old, new)
    assert [c["param"] for c in changes] == ["episode_length"]
    assert changes[0]["from"] == 20 and changes[0]["to"] == 63


def test_param_changes_flags_new_calibration_and_code():
    new = _exp()
    new["calibration_sha256"] = {"a": "y"}
    new["code"]["code_sha256"] = {"src/train.py": "2"}
    assert {c["param"] for c in build.param_changes(_exp(), new)} == {"calibration", "code"}


def test_comparison_runs_default_to_every_earlier_run():
    reg = {"runs": [{"id": "a"}, {"id": "b"}, {"id": "c", "compare_with": ["a"]}]}
    assert build.comparison_runs(reg["runs"][1], reg) == ["a"]
    assert build.comparison_runs(reg["runs"][0], reg) == []
    assert build.comparison_runs(reg["runs"][2], reg) == ["a"]


# ---------------------------------------------------------------------------
# End to end on a small real run
# ---------------------------------------------------------------------------

N_EP, N_RUNS, LOG = 4_000, 2, 500


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    results = root / "results" / "tiny"
    results.mkdir(parents=True)
    settings = batch_settings(N_EP, N_RUNS, 500, LOG)
    (results / "settings.json").write_text(json.dumps(settings))
    exp = next(e for e in cfg.experiments() if e.name == "resolution_g5_grid25")
    run_experiment(cfg.train_config(exp, n_episodes=N_EP, n_runs=N_RUNS, log_every=LOG), results / exp.name)
    evaluate_experiment(results / exp.name, cfg.eval_config(N_EP, 500))
    return root, results, exp


def test_extract_matches_evaluate_on_a_real_run(small_run):
    root, results, exp = small_run
    entry = {"id": "tiny", "results_dir": "results/tiny", "scale": "reduced"}
    data = extract.extract_run(entry, root=root, processed_dir=extract.ROOT / "data/processed")
    (e,) = data["experiments"]
    summary = json.loads((results / exp.name / "evaluation" / "summary.json").read_text())
    for mine, theirs in zip(e["seeds"], summary["per_seed"]):
        assert mine["action"] == theirs["learned_action"]
        assert mine["loss_best"] == pytest.approx(theirs["ce_loss_vs_best_discrete_bps"])
        assert mine["loss_cont"] == pytest.approx(theirs["ce_loss_vs_continuous_bps"])
        for k_mine, k_theirs in [("primary_obs", "greedy_stable_through_end_observations"),
                                 ("dq_obs", "dq_below_tol_for_good_observations"),
                                 ("near_obs_ckpt", "near_optimal_for_good_observations")]:
            assert mine[k_mine] == theirs[k_theirs]
        assert mine["regret_avg_bps"] == pytest.approx(theirs["final_cumulative_regret"] / N_EP * 252e4)
    assert e["summary"]["disc_gap_bps"] == pytest.approx(summary["discretization_gap"]["ce_loss_bps"])
    assert e["params"]["n_actions"] == 495 and e["params"]["episode_length"] == cfg.EPISODE_LENGTH
    assert e["family"] == "gamma5_grid25"


def test_extract_refuses_a_changed_calibration(small_run, tmp_path):
    root, results, _ = small_run
    fake = tmp_path / "processed"
    fake.mkdir()
    for p in (extract.ROOT / "data/processed").iterdir():
        if p.is_file():
            (fake / p.name).write_bytes(p.read_bytes())
    (fake / "covariance_daily.csv").write_text((fake / "covariance_daily.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="calibration"):
        extract.extract_run({"id": "tiny", "results_dir": "results/tiny"}, root=root, processed_dir=fake)


def test_report_layers_in_earlier_runs_and_lists_what_changed(small_run, tmp_path):
    root, _, _ = small_run
    data = extract.extract_run({"id": "first", "results_dir": "results/tiny", "scale": "reduced"}, root=root,
                               processed_dir=extract.ROOT / "data/processed")
    second = json.loads(json.dumps(data))
    second["run"]["id"] = "second"
    second["experiments"][0]["params"]["episode_length"] = 63
    reports = tmp_path / "reports"
    for run in (data, second):
        (reports / run["run"]["id"]).mkdir(parents=True)
        (reports / run["run"]["id"] / "data.json").write_text(json.dumps(run))
    registry = {"reference_run": "first", "runs": [{"id": "first", "pre_registered": True},
                                                   {"id": "second", "pre_registered": False, "notes": {"tldr": ["<!--<script>x</script>"]}}]}

    model = build.build_report_data("second", registry, reports)
    assert model["run_order"] == ["first", "second"]
    (fam,) = model["families"]
    assert [c["param"] for c in fam["entries"][1]["vs_previous"]] == ["episode_length"]
    prior = next(r for r in model["runs"] if r["run"]["id"] == "first")
    assert "series" not in prior["experiments"][0]["seeds"][0]          # prior runs drop per-seed series
    assert "series" in model["runs"][-1]["experiments"][0]["seeds"][0]   # the current run keeps them

    out = build.write_report("second", registry, reports)
    html = out.read_text(encoding="utf-8")
    assert build.PLACEHOLDER not in html and '"current_id":"second"' in html
    # Registry text must not be able to open a comment or close the script inside the embedded data.
    payload = html.split("const R = ", 1)[1].split(";\n(function", 1)[0]
    assert "<" not in payload
    assert json.loads(payload)["registry"]["second"]["notes"]["tldr"] == ["<!--<script>x</script>"]

    first_only = build.build_report_data("first", registry, reports)
    assert first_only["run_order"] == ["first"]


def test_registry_entry_requires_pre_registered():
    with pytest.raises(ValueError, match="pre_registered"):
        extract.registry_entry("a", {"runs": [{"id": "a"}]})
    assert extract.registry_entry("a", {"runs": [{"id": "a", "pre_registered": False}]})["id"] == "a"


def test_bin_series_mean_and_last():
    x = np.arange(1, 9)
    xb, yb = extract.bin_series(x, {"m": np.arange(8.0), "l": np.arange(8.0)}, {"m": "mean", "l": "last"}, max_points=4)
    assert xb.tolist() == [2, 4, 6, 8] and yb["m"].tolist() == [0.5, 2.5, 4.5, 6.5] and yb["l"].tolist() == [1, 3, 5, 7]


@pytest.fixture(scope="module")
def long_run(tmp_path_factory):
    """B = 60, 1,400 checkpoints (more than MAX_POINTS) and a short final window."""
    from src.train import TrainConfig
    root = tmp_path_factory.mktemp("repo_long")
    results = root / "results" / "long"
    results.mkdir(parents=True)
    n_ep, log_every = 7_005, 5
    settings = batch_settings(n_ep, 2, 200, log_every)
    (results / "settings.json").write_text(json.dumps(settings))
    tc = TrainConfig(gamma=5.0, episode_length=60, n_episodes=n_ep, master_seed=7, n_runs=2,
                     beta=cfg.beta_for(n_ep), increment=0.25, log_every=log_every, block_size=1_000)
    run_experiment(tc, results / "g5_grid25_B60")
    return root


def test_long_run_extracts_with_binning_and_medians_taken_before_binning(long_run):
    data = extract.extract_run({"id": "long", "results_dir": "results/long"}, root=long_run,
                               processed_dir=extract.ROOT / "data/processed")
    (e,) = data["experiments"]
    assert e["params"]["episode_length"] == 60 and e["params"]["simulated_days_per_seed"] == 7_005 * 60
    assert len(e["series"]["x_episodes"]) == extract.MAX_POINTS and e["series"]["x_episodes"][-1] == 7_005
    # Recompute the median loss curve a different way: raw medians across seeds, then bin means.
    exp_dir = long_run / "results" / "long" / "g5_grid25_B60"
    mu, sigma = load_calibration()
    from src.action_space.grid import generate_weight_grid
    from src.evaluate import objective
    grid = generate_weight_grid(9, 0.25)
    u = objective(grid, mu, sigma, 5.0)
    loss = (u.max() - u) * 252e4
    curves = np.vstack([loss[np.load(exp_dir / f"run_00{r}.npz")["checkpoint_greedy"]] for r in (0, 1)])
    raw_median = np.median(curves, 0)
    edges = np.linspace(0, raw_median.size, extract.MAX_POINTS + 1).astype(int)
    expected = [raw_median[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
    np.testing.assert_allclose(e["series"]["loss_median"], np.round(expected, 2), atol=0.006)
    for s in e["seeds"]:
        assert len(s["series"]["loss"]) == extract.MAX_POINTS
        assert s["settled_obs"] % 60 == 0


def test_the_settings_table_records_the_agent(tmp_path):
    """FA manifests carry agent fields; tabular ones leave them out and read as tabular."""
    from src.train import TrainConfig
    common = dict(gamma=5, episode_length=20, n_episodes=200, master_seed=2, n_runs=1, beta=0.001,
                  log_every=100, increment=0.25)
    mu, sigma = load_calibration()
    run_experiment(TrainConfig(**common, agent="linear", feature_map="spline", alpha_rule=1e-3, l2=1e-4,
                               greedy_refresh_every=100), tmp_path / "fa")
    run_experiment(TrainConfig(**common), tmp_path / "tab")
    settings = {"near_optimal_bps": 25, "dq_tol": 1e-5}
    fa = extract.extract_experiment(tmp_path / "fa", settings, mu, sigma, extract.PROCESSED_DIR)["params"]
    tab = extract.extract_experiment(tmp_path / "tab", settings, mu, sigma, extract.PROCESSED_DIR)["params"]
    assert (fa["agent"], fa["feature_map"], fa["l2"], fa["greedy_refresh_every"], fa["alpha_rule"]) == \
        ("linear", "spline", 1e-4, 100, 1e-3)
    assert (tab["agent"], tab["feature_map"], tab["l2"], tab["greedy_refresh_every"], tab["alpha_rule"]) == \
        ("tabular", None, 0.0, 1, "sample_average")
    assert {"agent", "feature_map", "l2", "greedy_refresh_every"} <= {k for k, _ in build.COMPARED_PARAMS}


def test_older_report_data_read_as_tabular_so_comparisons_show_only_real_changes(tmp_path):
    """data.json files extracted before the agent fields existed must not
    make a constant-alpha tabular run look like it changed the agent."""
    old = _exp(alpha_rule="sample_average")
    for key in build.AGENT_PARAM_DEFAULTS:
        del old["params"][key]
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "data.json").write_text(json.dumps({"experiments": [old]}))
    loaded = build.load_run_data("old", tmp_path)["experiments"][0]
    assert {k: loaded["params"][k] for k in build.AGENT_PARAM_DEFAULTS} == build.AGENT_PARAM_DEFAULTS
    new = _exp(alpha_rule=0.00032, **build.AGENT_PARAM_DEFAULTS)
    assert [c["param"] for c in build.param_changes(loaded, new)] == ["alpha_rule"]
