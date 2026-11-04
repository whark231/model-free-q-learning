"""Tests for the step-size tuning script (src/tune_fa.py, post hoc)."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import src.tune_fa as tune
from src.action_space.grid import generate_weight_grid
from src.config import EPSILON_AT_HALFWAY, beta_for
from src.environment import SeedPlan, ReturnSimulator, load_calibration
from src.rewards import episodic_mean_variance_reward
from src.train import run_experiment

GRID = generate_weight_grid(9, 0.2)


def test_pass1_has_the_reference_and_every_model_at_every_fraction():
    cands = tune.pass1_candidates()
    assert len(cands) == 1 + 3 * 8
    assert len({c["id"] for c in cands}) == len(cands)
    assert cands[0] == {"id": "tabular_1overN", "model": "tabular", "fraction": None, "l2": 0.0}
    for model in tune.MODELS:
        assert sorted(c["fraction"] for c in cands if c["model"] == model and c["fraction"] is not None) \
            == sorted(tune.FRACTIONS)
    assert all(c["l2"] == 0.0 for c in cands)
    assert tune.TUNING_MASTER_SEED != 20261016  # the reported seeds are never used for tuning


def test_stability_bounds_from_the_feature_norms():
    assert tune.stability_bound("tabular", GRID) == 2.0
    assert tune.stability_bound("quadratic", GRID) == pytest.approx(2.0)  # a corner portfolio: one product = 1
    assert tune.stability_bound("spline", GRID) == pytest.approx(2.0 / 45)  # 9 marginals + 36 pair products = 1


def test_pass2_sweeps_l2_at_the_given_fractions():
    cands = tune.pass2_candidates([0.004, 8e-4])
    assert {(c["fraction"], c["l2"]) for c in cands} == {(f, l) for f in (0.004, 8e-4) for l in tune.L2_VALUES}
    assert all(c["model"] == "spline" and c["id"].startswith("spline_f") and "_l2" in c["id"] for c in cands)


@pytest.mark.parametrize("candidate", [
    {"id": "tabular_f0.02", "model": "tabular", "fraction": 0.02, "l2": 0.0},
    {"id": "quadratic_f0.1", "model": "quadratic", "fraction": 0.1, "l2": 0.0},
    {"id": "spline_f0.02_l20.001", "model": "spline", "fraction": 0.02, "l2": 1e-3},
    {"id": "tabular_1overN", "model": "tabular", "fraction": None, "l2": 0.0},
])
def test_a_tuning_seed_trains_exactly_like_run_experiment(tmp_path, candidate):
    """Same config, same seed layout: tuning seed 1 must equal run 1 of
    run_experiment, so the tuning runs are the real training path."""
    n = 2_000
    rec = tune.run_candidate(candidate, 1, n, tmp_path / "tune")
    alpha = "sample_average" if candidate["fraction"] is None \
        else candidate["fraction"] * tune.stability_bound(candidate["model"], GRID)
    assert rec["alpha"] == alpha and rec["status"] == "ok"
    run_experiment(tune._config(candidate["model"], alpha, candidate["l2"], n), tmp_path / "ref")
    with np.load(tmp_path / "ref" / "run_001.npz") as logs:
        assert rec["final_action"] == int(np.argmax(logs["final_q"]))
        assert rec["final_q_of_action"] == float(logs["final_q"].max())
        assert rec["visits_of_action"] == int(logs["final_visit_counts"][rec["final_action"]])
    # Settings checked against config.py and the notes, not against tune._config itself.
    cfg = rec["config"]
    assert cfg["master_seed"] == 20261103 and cfg["n_runs"] == 6 and cfg["gamma"] == 5.0
    assert cfg["increment"] == 0.2 and cfg["episode_length"] == 20 and cfg["n_episodes"] == n
    assert cfg["beta"] == pytest.approx(beta_for(n, EPSILON_AT_HALFWAY)) and EPSILON_AT_HALFWAY == 0.01
    assert cfg["epsilon0"] == 1.0 and cfg["alpha_rule"] == alpha
    if candidate["model"] == "tabular":
        assert "agent" not in cfg  # tabular records match the baseline manifests
    else:
        assert cfg["agent"] == "linear" and cfg["feature_map"] == candidate["model"]
        assert cfg["l2"] == candidate["l2"] and cfg["greedy_refresh_every"] == 100


def test_an_extra_seed_trains_like_run_experiment_and_leaves_base_seeds_unchanged(tmp_path):
    """Seed 4 (an extra seed) equals run 4 of run_experiment, and seed 1 is the
    same whether the plan declares 3 or 6 runs (streams ignore n_runs)."""
    cand = {"id": "tabular_f0.02", "model": "tabular", "fraction": 0.02, "l2": 0.0}
    rec4 = tune.run_candidate(cand, 4, 1_000, tmp_path / "tune")
    alpha = 0.02 * 2.0
    run_experiment(tune._config("tabular", alpha, 0.0, 1_000), tmp_path / "ref6")
    with np.load(tmp_path / "ref6" / "run_004.npz") as logs:
        assert rec4["final_action"] == int(np.argmax(logs["final_q"]))
    three = tune._config("tabular", alpha, 0.0, 1_000)
    object.__setattr__(three, "n_runs", 3)
    run_experiment(three, tmp_path / "ref3")
    with np.load(tmp_path / "ref3" / "run_001.npz") as a, np.load(tmp_path / "ref6" / "run_001.npz") as b:
        np.testing.assert_array_equal(a["final_q"], b["final_q"])


def test_finished_records_are_skipped_on_rerun(tmp_path, monkeypatch):
    cand = {"id": "tabular_f0.1", "model": "tabular", "fraction": 0.1, "l2": 0.0}
    first = tune.run_candidate(cand, 0, 1_000, tmp_path)

    def must_not_train(*args, **kwargs):
        raise AssertionError("a finished run was trained again")

    monkeypatch.setattr(tune, "train_one_run", must_not_train)
    assert tune.run_candidate(cand, 0, 1_000, tmp_path) == first
    assert not list((tmp_path / "records").glob("*.tmp"))


def test_a_record_from_other_settings_is_refused_not_reused(tmp_path):
    """A smoke-test record in the same folder must not stand in for a full run."""
    cand = {"id": "tabular_f0.1", "model": "tabular", "fraction": 0.1, "l2": 0.0}
    tune.run_candidate(cand, 0, 1_000, tmp_path)
    with pytest.raises(ValueError, match="different settings"):
        tune.run_candidate(cand, 0, 2_000, tmp_path)
    with pytest.raises(ValueError, match="different settings"):  # also before any work is queued
        tune.run_all([cand], 2_000, tmp_path, jobs=1)


def test_divergence_is_recorded_and_excluded(tmp_path, monkeypatch):
    def blow_up(*args, **kwargs):
        raise FloatingPointError("theta is no longer finite: above the stability bound")

    monkeypatch.setattr(tune, "train_one_run", blow_up)
    cand = {"id": "quadratic_f0.5", "model": "quadratic", "fraction": 0.5, "l2": 0.0}
    rec = tune.run_candidate(cand, 0, 1_000, tmp_path)
    assert rec["status"].startswith("diverged") and rec["final_action"] is None


def _independent_rewards(actions, n_eval):
    """Every portfolio's reward on each evaluation episode, one episode at a
    time with the training reward function: a different route from
    reward_moments' chunked, vectorized one."""
    mu, sigma = load_calibration()
    sim = ReturnSimulator(mu, sigma, SeedPlan(tune.TUNING_MASTER_SEED, tune.N_TUNING_SEEDS).evaluation_rng())
    out = np.empty((n_eval, len(actions)))
    for e in range(n_eval):
        batch = sim.sample_episode(20)
        out[e] = [episodic_mean_variance_reward(GRID[a], batch, 5.0) for a in actions]
    return out


def test_reward_moments_match_one_episode_at_a_time():
    actions = [0, 500, 1286]
    r = _independent_rewards(actions, 300)
    mean, cov = tune.reward_moments(actions, 300, chunk=128)  # 128 + 128 + 44
    np.testing.assert_allclose(mean, r.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(cov, np.cov(r, rowvar=False), rtol=1e-8)


def _write(out, cand_id, model, fraction, seed, action, l2=0.0, status="ok", n_episodes=1_000):
    path = tune.record_path(out, cand_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": cand_id, "model": model, "fraction": fraction, "l2": l2, "alpha": 0.1,
                                "seed": seed, "status": status, "final_action": action, "seconds": 60.0,
                                "config": {"n_episodes": n_episodes}, "git_commit": "abc"}))


def test_summarize_scores_paired_shortfalls_from_scratch(tmp_path):
    """Two quadratic candidates with known final portfolios; the criterion,
    the choice and the paired shortfall are rebuilt from per-episode rewards."""
    picks = {"quadratic_f0.1": [3, 700, 700], "quadratic_f0.02": [700, 900, 1286]}
    for cand_id, actions in picks.items():
        for seed, a in enumerate(actions):
            _write(tmp_path, cand_id, "quadratic", float(cand_id.split("_f")[1]), seed, a)
    _write(tmp_path, "spline_f0.1", "spline", 0.1, 0, 5)  # only one seed: incomplete
    n_eval = 400
    summary = tune.summarize(tmp_path, n_eval)

    actions = sorted({a for acts in picks.values() for a in acts})
    r = dict(zip(actions, _independent_rewards(actions, n_eval).T))
    score = {cid: np.mean([r[a] for a in acts], axis=0) for cid, acts in picks.items()}  # per episode
    best = max(score, key=lambda cid: score[cid].mean())
    other = next(cid for cid in score if cid != best)
    rows = summary["candidates"]
    for cid in picks:
        assert rows[cid]["criterion_daily"] == pytest.approx(score[cid].mean(), rel=1e-10)
    assert summary["chosen"]["quadratic"]["id"] == best
    d = score[best] - score[other]
    assert rows[other]["shortfall_to_model_best_bps"] == pytest.approx(252e4 * d.mean(), rel=1e-8)
    assert rows[other]["shortfall_eval_se_bps"] == pytest.approx(252e4 * d.std(ddof=1) / np.sqrt(n_eval), rel=1e-6)
    assert rows[best]["shortfall_to_model_best_bps"] == 0.0
    # Paired by training seed: seed s of each candidate against seed s of the best.
    by_seed = np.array([252e4 * (r[a].mean() - r[b].mean()) for a, b in zip(picks[best], picks[other])])
    assert rows[other]["shortfall_seed_se_bps"] == pytest.approx(by_seed.std(ddof=1) / np.sqrt(3), rel=1e-8)
    assert rows[other]["seeds_at_least_as_good"] == int((by_seed <= 0).sum())
    assert rows[other]["per_seed_reward_bps"] == pytest.approx([252e4 * r[a].mean() for a in picks[other]])
    assert summary["incomplete"] == ["spline_f0.1"] and summary["diverged"] == []
    assert summary["n_episodes"] == 1_000 and summary["git_commits"] == ["abc"]
    assert json.loads((tmp_path / "fa_tuning.json").read_text())["chosen"] == summary["chosen"]


def test_exact_ties_prefer_smaller_l2_then_larger_alpha(tmp_path):
    """Candidates ending on the same portfolios score identically; the choice
    must not depend on file order. A tie is a close call (shortfall 0 within
    a seed SE of 0), so the decision waits for seeds 3-5, which tie again."""
    cands = [("spline_f0.004_l20.001", 0.004, 1e-3), ("spline_f0.004", 0.004, 0.0), ("spline_f0.0008", 8e-4, 0.0)]
    for cid, f, l2 in cands:
        for seed in range(3):
            _write(tmp_path, cid, "spline", f, seed, 700, l2=l2)
    pending = tune.summarize(tmp_path, 50)
    assert pending["chosen"]["spline"]["pending"] and "id" not in pending["chosen"]["spline"]
    assert pending["extra_seeds_needed"]["spline"] == ["spline_f0.004", "spline_f0.0008", "spline_f0.004_l20.001"]
    for cid, f, l2 in cands:
        for seed in range(3, 6):
            _write(tmp_path, cid, "spline", f, seed, 700, l2=l2)
    chosen = tune.summarize(tmp_path, 50)["chosen"]["spline"]
    assert chosen["id"] == "spline_f0.004" and chosen["decided_on_seeds"] == 6
    assert chosen["tied_with"] == ["spline_f0.0008", "spline_f0.004_l20.001"]


def _ranked_actions(n_eval, candidates=(3, 500, 700, 1000, 1286)):
    """Grid portfolios ordered by their realized reward on the tuning
    evaluation sample, best first, from the independent route."""
    r = _independent_rewards(list(candidates), n_eval).mean(axis=0)
    return [a for _, a in sorted(zip(-r, candidates))]


def test_a_clear_winner_is_decided_on_three_seeds(tmp_path):
    good, poor = _ranked_actions(200)[:2]
    for seed in range(3):
        _write(tmp_path, "quadratic_f0.1", "quadratic", 0.1, seed, good)
        _write(tmp_path, "quadratic_f0.02", "quadratic", 0.02, seed, poor)
    summary = tune.summarize(tmp_path, 200)
    chosen = summary["chosen"]["quadratic"]
    assert chosen["id"] == "quadratic_f0.1" and chosen["decided_on_seeds"] == 3 and not chosen["close_call"]
    assert summary["extra_seeds_needed"] == {}
    assert summary["candidates"]["quadratic_f0.02"]["shortfall_seed_se_bps"] == 0.0  # same gap on every seed


def test_a_close_call_extends_the_top_three_and_can_change_the_winner(tmp_path):
    """Seeds 0-2: A wins one seed, B another, so the runner-up sits inside
    one seed SE. Seeds 3-5 favour the base runner-up, which then wins on 6."""
    best, second, third, worst = _ranked_actions(200)[:4]
    base = {"tabular_f0.1": [best, second, best], "tabular_f0.02": [second, best, second],
            "tabular_f0.004": [third, third, third], "tabular_f0.0008": [worst, worst, worst]}
    for cid, acts in base.items():
        for seed, a in enumerate(acts):
            _write(tmp_path, cid, "tabular", float(cid.split("_f")[1]), seed, a)
    summary = tune.summarize(tmp_path, 200)
    assert summary["chosen"]["tabular"]["close_call"] and summary["chosen"]["tabular"]["pending"]
    top3 = ["tabular_f0.1", "tabular_f0.02", "tabular_f0.004"]
    assert summary["extra_seeds_needed"] == {"tabular": top3}
    assert [c["id"] for c in tune.extra_seed_candidates(summary)] == top3
    extra = {"tabular_f0.1": [second] * 3, "tabular_f0.02": [best] * 3, "tabular_f0.004": [third] * 3}
    for cid, acts in extra.items():
        for seed, a in zip(range(3, 6), acts):
            _write(tmp_path, cid, "tabular", float(cid.split("_f")[1]), seed, a)
    summary = tune.summarize(tmp_path, 200)
    chosen = summary["chosen"]["tabular"]
    assert chosen["id"] == "tabular_f0.02" and chosen["decided_on_seeds"] == 6
    assert chosen["base_best"] == "tabular_f0.1"
    e = summary["candidates"]["tabular_f0.1"]["extended"]
    assert len(e["per_seed_reward_bps"]) == 6 and e["seeds_at_least_as_good"] == 2
    assert "extended" not in summary["candidates"]["tabular_f0.0008"]  # outside the top 3


def test_diverged_candidates_are_listed_and_never_scored(tmp_path):
    for seed in range(3):
        _write(tmp_path, "quadratic_f0.5", "quadratic", 0.5, seed, None if seed == 1 else 700,
               status="diverged: theta" if seed == 1 else "ok")
        _write(tmp_path, "quadratic_f0.1", "quadratic", 0.1, seed, 3)
    summary = tune.summarize(tmp_path, 50)
    assert summary["diverged"] == ["quadratic_f0.5"] and "quadratic_f0.5" not in summary["candidates"]
    assert summary["chosen"]["quadratic"]["id"] == "quadratic_f0.1"


def test_the_one_over_n_reference_is_never_chosen(tmp_path):
    """The reference ends on a better portfolio than the constant-alpha
    candidate, yet the constant-alpha candidate is chosen and the reference
    is shown with a negative shortfall."""
    mu, sigma = load_calibration()
    good, poor = 700, 3
    for seed in range(3):
        _write(tmp_path, tune.REFERENCE, "tabular", None, seed, good)
        _write(tmp_path, "tabular_f0.1", "tabular", 0.1, seed, poor)
    summary = tune.summarize(tmp_path, 2_000)
    assert summary["chosen"]["tabular"]["id"] == "tabular_f0.1"
    assert summary["candidates"][tune.REFERENCE]["shortfall_to_model_best_bps"] < 0


def test_mixed_training_budgets_are_refused(tmp_path):
    for seed in range(3):
        _write(tmp_path, "quadratic_f0.1", "quadratic", 0.1, seed, 3, n_episodes=20_000 if seed else 13_000_000)
    with pytest.raises(ValueError, match="mix training budgets"):
        tune.summarize(tmp_path, 50)


def _spline_summary(scores, diverged=()):
    rows = {f"spline_f{f:g}": {"model": "spline", "l2": 0.0, "fraction": f, "criterion_daily": c}
            for f, c in zip(tune.FRACTIONS, scores) if f"spline_f{f:g}" not in diverged}
    return {"candidates": rows, "diverged": list(diverged)}


def test_pass2_uses_the_two_best_spline_fractions_and_needs_all_of_pass1():
    summary = _spline_summary([1, 5, 3, 9, 2, 0, 4, 8])
    assert tune.best_spline_fractions(summary) == [tune.FRACTIONS[3], tune.FRACTIONS[7]]
    summary["candidates"].pop("spline_f0.5")
    with pytest.raises(RuntimeError, match="pass 1"):
        tune.best_spline_fractions(summary)


def test_pass2_breaks_ties_like_the_final_choice_and_tolerates_divergence():
    tied = _spline_summary([7] * 8)
    tied["candidates"] = dict(reversed(list(tied["candidates"].items())))  # smallest alpha first
    assert tune.best_spline_fractions(tied) == [0.5, 0.1]  # larger alpha first, not insertion order
    one_diverged = _spline_summary([1, 5, 3, 9, 2, 0, 4, 8], diverged=["spline_f0.004"])
    assert tune.best_spline_fractions(one_diverged) == [tune.FRACTIONS[7], tune.FRACTIONS[1]]


def test_tuning_never_touches_benchmarks_or_the_true_objective():
    """The choice must use realized reward only. The script's own imports
    exclude the benchmark and evaluation modules, and it computes no
    closed-form objective. (src.config imports evaluate.py for EvalConfig, so
    those modules are loaded, but nothing from them is called here.)"""
    source = Path(tune.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any(m.startswith(("src.benchmarks", "src.evaluate")) for m in imported)
    for forbidden in ("objective(", "best_discrete", "constrained_optimal", "@ sigma", "sigma @"):
        assert forbidden not in source, forbidden


def test_the_extend_step_trains_seeds_three_to_five_only(tmp_path, monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(tune, "run_all", lambda cands, n, out, jobs, seeds=range(3): calls.append(
        ([c["id"] for c in cands], list(seeds))))
    row = {"model": "quadratic", "fraction": 0.1, "l2": 0.0, "alpha": 0.2, "criterion_daily": 1e-4,
           "criterion_bps": 252.0, "per_seed_reward_bps": [252.0] * 3, "minutes_per_seed": 1.0}
    fake = {"n_episodes": 1_000, "n_eval_episodes": 10, "candidates": {"quadratic_f0.1": row}, "incomplete": [],
            "diverged": [], "extra_seeds_needed": {"quadratic": ["quadratic_f0.1"]}, "chosen": {}}
    monkeypatch.setattr(tune, "summarize", lambda out, n: fake)
    monkeypatch.setattr(sys, "argv", ["tune_fa", "--extend", "--episodes", "1000", "--out", str(tmp_path)])
    tune.main()
    assert calls == [(["quadratic_f0.1"], [3, 4, 5])]
