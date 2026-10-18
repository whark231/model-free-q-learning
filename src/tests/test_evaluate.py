"""Checks on the evaluation (src/evaluate.py).

Pure calculations are checked on hand-made inputs with known answers. The
end-to-end checks train tiny experiments with agents whose behavior is known
in advance (always the same portfolio, or a greedy action that switches at a
known episode), so every loss, distance, regret and convergence episode has an
exact expected value computed independently here.
"""

import csv
import json

import numpy as np
import pytest

import src.evaluate as ev
from src.action_space.grid import generate_weight_grid
from src.benchmarks.mean_variance import constrained_optimal_weights
from src.environment import ReturnSimulator, SeedPlan, load_calibration
from src.rewards import episodic_mean_variance_reward
from src.train import TrainConfig, run_experiment

GAMMA = 5.0
EVAL = ev.EvalConfig(n_eval_episodes=2_000, converge_k=600, dq_tol=1e-3, near_optimal_bps=25.0)


class FixedAgent:
    """Always picks one action; Q is the running mean of its rewards. Other
    actions sit at Q = -1, far below any reward, so the greedy action is the
    chosen one from episode 0 and the logs are known exactly."""

    def __init__(self, n_actions, action):
        self.action = action
        self._q = np.full(n_actions, -1.0)
        self._q[action] = 0.0
        self._n = np.zeros(n_actions, dtype=int)

    def select_action(self):
        return self.action

    def update(self, action, reward):
        self._n[action] += 1
        self._q[action] += (reward - self._q[action]) / self._n[action]

    @property
    def q_values(self):
        return self._q.copy()

    @property
    def visit_counts(self):
        return self._n.copy()


class SwitchingAgent(FixedAgent):
    """Greedy (and chosen) action is `first` until `switch_at` updates, then
    `second`: a known, non-constant greedy path."""

    def __init__(self, n_actions, first, second, switch_at):
        super().__init__(n_actions, first)
        self.second, self.switch_at, self._t = second, switch_at, 0

    def update(self, action, reward):
        self._t += 1
        if self._t == self.switch_at:
            self._q[:] = -1.0
            self._q[self.second] = 0.0
            self.action = self.second
        else:
            super().update(action, reward)


def make_config(n_runs, n_episodes=2_000, increment=0.25, gamma=GAMMA, log_every=500):
    return TrainConfig(gamma=gamma, episode_length=20, n_episodes=n_episodes, master_seed=7, n_runs=n_runs,
                       beta=1e-3, increment=increment, log_every=log_every, block_size=500)


def train_with(tmp_path, agents, name="exp", **config_overrides):
    """`agents` is a list of factories f(n_actions) -> agent, one per run."""
    config = make_config(len(agents), **config_overrides)
    made = []

    def factory(cfg, n_actions, rng):
        made.append(1)
        return agents[len(made) - 1](n_actions)

    run_experiment(config, tmp_path / name, agent_factory=factory)
    return tmp_path / name, config


def fixed(action):
    return lambda n: FixedAgent(n, action)


@pytest.fixture(scope="module")
def market():
    mu, sigma = load_calibration()
    grid = generate_weight_grid(9, 0.25)
    values = grid @ mu - 0.5 * GAMMA * ((grid @ sigma) * grid).sum(axis=1)
    return mu, sigma, grid, values


# ---------------------------------------------------------------------------
# Pure calculations, hand-checked
# ---------------------------------------------------------------------------

def test_objective_and_ce_loss_by_hand():
    mu = np.array([0.01, 0.02])
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert ev.objective(np.array([0.5, 0.5]), mu, sigma, 5.0) == pytest.approx(0.015 - 2.5 * 0.0375)
    np.testing.assert_allclose(ev.objective(np.eye(2), mu, sigma, 5.0), [0.01 - 2.5 * 0.04, 0.02 - 2.5 * 0.09])
    assert ev.ce_loss_bps(2e-4, 1.5e-4) == pytest.approx(252 * 0.5e-4 * 1e4)  # 12.6 bps/yr
    assert ev.ce_loss_bps(-1e-4, -3e-4) == pytest.approx(252 * 2e-4 * 1e4)  # sign-safe below zero


def test_relative_gap_divides_by_magnitude_and_refuses_near_zero():
    assert ev.relative_gap(2e-4, 1e-4) == pytest.approx(0.5)
    assert ev.relative_gap(-2e-4, -3e-4) == pytest.approx(0.5)  # worse portfolio still has a positive gap
    assert ev.relative_gap(1e-6, 0.0) is None


def test_expected_regret_by_hand():
    counts = np.array([[3, 1, 0], [0, 2, 2]])
    np.testing.assert_allclose(ev.expected_regret(counts, np.array([1.0, 0.5, 0.0]), 1.0), [0.5, 3.5])


def test_greedy_stability_rules_by_hand():
    episodes, actions = np.array([0, 5, 30, 40]), np.array([0, 3, 7, 3])
    # segments: [0,5) a0, [5,30) a3, [30,40) a7, [40,100) a3
    assert ev.first_greedy_stable(episodes, actions, 100, 20) == (25, 3)
    assert ev.first_greedy_stable(episodes, actions, 100, 61) == (None, None)
    assert ev.greedy_stable_through_end(episodes, 100, 60) == 100
    assert ev.greedy_stable_through_end(episodes, 100, 61) is None


def test_full_windows_drops_a_short_final_window():
    np.testing.assert_array_equal(ev.full_windows([500, 1000, 1030], 500), [True, True, False])


def test_first_for_good_requires_persistence_and_respects_the_mask():
    episodes = [10, 20, 30, 40]
    assert ev.first_for_good(episodes, [False, True, False, True]) == 40
    assert ev.first_for_good(episodes, [True, True, True, True]) == 10
    assert ev.first_for_good(episodes, [True, True, True, False]) is None
    # a short final window that happens to look good must not decide the answer
    assert ev.first_for_good(episodes, [False, False, False, True], mask=[True, True, True, False]) is None


def test_realized_stats_by_direct_computation():
    rng = np.random.default_rng(0)
    episodes = rng.normal(0.0005, 0.01, size=(400, 20, 3))
    w, ref_w = np.array([0.2, 0.3, 0.5]), np.array([1.0, 0.0, 0.0])
    ref_rewards = np.array([episodic_mean_variance_reward(ref_w, e, 4.0) for e in episodes])
    rewards = np.array([episodic_mean_variance_reward(w, e, 4.0) for e in episodes])
    out = ev.realized_stats(w, episodes, 4.0, ref_rewards)
    daily = (episodes @ w).ravel()
    assert out["mean_annual"] == pytest.approx(252 * daily.mean())
    assert out["vol_annual"] == pytest.approx(np.sqrt(252) * daily.std(ddof=1))
    assert out["objective_daily"] == pytest.approx(rewards.mean())
    assert out["objective_se"] == pytest.approx(rewards.std(ddof=1) / 20)
    assert out["shortfall_vs_best_discrete_daily"] == pytest.approx((ref_rewards - rewards).mean())
    assert out["shortfall_vs_best_discrete_se"] == pytest.approx((ref_rewards - rewards).std(ddof=1) / 20)


@pytest.mark.parametrize("bad", [dict(n_eval_episodes=1), dict(converge_k=0), dict(dq_tol=0.0), dict(near_optimal_bps=-1.0)])
def test_eval_config_rejects_invalid_settings(bad):
    settings = dict(n_eval_episodes=10, converge_k=10, dq_tol=1e-3, near_optimal_bps=25.0)
    settings.update(bad)
    with pytest.raises(ValueError):
        ev.EvalConfig(**settings)


# ---------------------------------------------------------------------------
# End to end with known behavior
# ---------------------------------------------------------------------------

def test_agent_that_always_holds_the_best_grid_portfolio(tmp_path, market):
    mu, sigma, grid, values = market
    best = int(np.argmax(values))
    run_dir, _ = train_with(tmp_path, [fixed(best), fixed(best)])
    summary = ev.evaluate_experiment(run_dir, EVAL)
    for seed in summary["per_seed"]:
        assert seed["ce_loss_vs_best_discrete_bps"] == 0.0
        assert seed["relative_learning_gap"] == 0.0
        assert seed["final_cumulative_regret"] == 0.0
        assert seed["is_best_discrete"] and seed["learned_weights"] == grid[best].tolist()
    assert summary["learned"]["share_exactly_best_discrete"] == 1.0


def test_losses_distances_and_regret_for_a_suboptimal_agent(tmp_path, market):
    mu, sigma, grid, values = market
    best, worse = int(np.argmax(values)), int(np.argsort(values)[-10])
    run_dir, config = train_with(tmp_path, [fixed(worse)])
    summary = ev.evaluate_experiment(run_dir, EVAL)
    seed = summary["per_seed"][0]
    w_cont = constrained_optimal_weights(mu, sigma, GAMMA)
    u_cont = w_cont @ mu - 0.5 * GAMMA * w_cont @ sigma @ w_cont

    assert summary["discretization_gap"]["ce_loss_bps"] == pytest.approx(252e4 * (u_cont - values[best]))
    assert summary["discretization_gap"]["relative"] == pytest.approx((u_cont - values[best]) / u_cont)
    assert seed["ce_loss_vs_best_discrete_bps"] == pytest.approx(252e4 * (values[best] - values[worse]))
    assert seed["relative_learning_gap"] == pytest.approx((values[best] - values[worse]) / values[best])
    assert seed["l1_to_continuous"] == pytest.approx(np.abs(grid[worse] - w_cont).sum())
    assert seed["euclidean_to_continuous"] == pytest.approx(np.linalg.norm(grid[worse] - w_cont))
    assert seed["l1_to_best_discrete"] == pytest.approx(np.abs(grid[worse] - grid[best]).sum())
    np.testing.assert_allclose(seed["abs_diff_vs_continuous"], np.abs(grid[worse] - w_cont))
    np.testing.assert_allclose(seed["abs_diff_vs_best_discrete"], np.abs(grid[worse] - grid[best]))
    assert seed["final_cumulative_regret"] == pytest.approx(config.n_episodes * (values[best] - values[worse]))
    assert seed["visits_of_learned_action"] == config.n_episodes


def test_each_seed_is_reported_and_the_mean_of_weights_is_descriptive_only(tmp_path, market):
    """Averaging weights across seeds builds a portfolio no seed holds, and by
    concavity it can beat the best grid action. It must never be scored as
    'the learned portfolio'."""
    mu, sigma, grid, values = market
    order = np.argsort(values)
    picks = [int(order[-1]), int(order[-3]), int(order[-7])]
    run_dir, _ = train_with(tmp_path, [fixed(a) for a in picks])
    summary = ev.evaluate_experiment(run_dir, EVAL)
    losses = sorted(252e4 * (values[picks[0]] - values[a]) for a in picks)
    assert sorted(s["ce_loss_vs_best_discrete_bps"] for s in summary["per_seed"]) == pytest.approx(losses)
    spread = summary["learned"]["ce_loss_vs_best_discrete_bps"]
    assert spread["median"] == pytest.approx(losses[1]) and spread["max"] == pytest.approx(losses[2])
    assert summary["learned"]["median_seed_run"] == 1
    assert "learned" not in summary["benchmarks"]
    np.testing.assert_allclose(summary["learned"]["mean_of_seed_weights_descriptive_only"], grid[picks].mean(axis=0))


def test_switching_greedy_path_gives_exact_convergence_episodes(tmp_path, market):
    mu, sigma, grid, values = market
    best, worse = int(np.argmax(values)), int(np.argsort(values)[0])
    config_k = ev.EvalConfig(n_eval_episodes=500, converge_k=200, dq_tol=1e-3, near_optimal_bps=1.0)
    run_dir, _ = train_with(tmp_path, [lambda n: SwitchingAgent(n, worse, best, 1_200)], n_episodes=2_000)
    seed = ev.evaluate_experiment(run_dir, config_k)["per_seed"][0]
    assert seed["greedy_stable_episode"] == 200  # the bad action, first held 200 episodes
    assert seed["greedy_stable_action_is_final"] is False
    assert seed["greedy_stable_through_end_episode"] == 1_400
    assert seed["greedy_stable_through_end_observations"] == 1_400 * 20
    # checkpoints at 500, 1000, 1500, 2000; the greedy action is best from 1,200 on
    assert seed["near_optimal_for_good_episode"] == 1_500
    assert seed["learned_action"] == best


def test_negative_objectives_keep_losses_sign_safe(tmp_path):
    """At gamma = 60 every grid portfolio has U < 0. The worst portfolio must
    show a large positive loss, never a 'near-optimal' label."""
    mu, sigma = load_calibration()
    grid = generate_weight_grid(9, 0.25)
    values = grid @ mu - 30.0 * ((grid @ sigma) * grid).sum(axis=1)
    assert values.max() < 0
    best, worst = int(np.argmax(values)), int(np.argmin(values))
    run_dir, _ = train_with(tmp_path, [fixed(best), fixed(worst)], gamma=60.0)
    summary = ev.evaluate_experiment(run_dir, EVAL)
    by_action = {s["learned_action"]: s for s in summary["per_seed"]}
    assert by_action[best]["ce_loss_vs_best_discrete_bps"] == 0.0
    assert by_action[worst]["ce_loss_vs_best_discrete_bps"] == pytest.approx(252e4 * (values[best] - values[worst]))
    assert by_action[worst]["ce_loss_vs_best_discrete_bps"] > 1_000
    assert by_action[worst]["near_optimal_for_good_episode"] is None
    assert summary["learned"]["relative_gap_unreliable"] is True


def test_evaluation_sample_is_the_shared_evaluation_stream(tmp_path, market):
    mu, sigma, grid, values = market
    best = int(np.argmax(values))
    run_dir, config = train_with(tmp_path, [fixed(best)])
    summary = ev.evaluate_experiment(run_dir, EVAL)
    plan = SeedPlan(config.master_seed, config.n_runs)
    episodes = ReturnSimulator(mu, sigma, plan.evaluation_rng()).sample_episodes(EVAL.n_eval_episodes, config.episode_length)
    rewards = np.array([episodic_mean_variance_reward(grid[best], e, GAMMA) for e in episodes])
    realized = summary["benchmarks"]["best_discrete"]["evaluation_sample"]
    assert realized["objective_daily"] == pytest.approx(rewards.mean(), rel=1e-12)
    assert realized["shortfall_vs_best_discrete_daily"] == 0.0
    training = ReturnSimulator(mu, sigma, plan.run_streams(0)[0]).sample_episodes(EVAL.n_eval_episodes, config.episode_length)
    assert not np.allclose(training, episodes)
    assert abs(np.corrcoef(training.ravel(), episodes.ravel())[0, 1]) < 0.02


def test_csv_tables_match_the_summary(tmp_path, market):
    mu, sigma, grid, values = market
    picks = [int(np.argmax(values)), int(np.argsort(values)[-5])]
    run_dir, _ = train_with(tmp_path, [fixed(a) for a in picks])
    summary = ev.evaluate_experiment(run_dir, EVAL)
    out = run_dir / "evaluation"
    with open(out / "weights.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["ticker"] for r in rows] == summary["tickers"]
    for i, row in enumerate(rows):
        assert float(row["continuous"]) == pytest.approx(summary["benchmarks"]["continuous"]["weights"][i], abs=5e-5)
        assert float(row["learned_run_1"]) == pytest.approx(grid[picks[1]][i])
        assert float(row["absdiff_vs_best_discrete_run_1"]) == pytest.approx(abs(grid[picks[1]][i] - grid[picks[0]][i]))
    with open(out / "per_seed.csv") as f:
        per_seed = list(csv.DictReader(f))
    assert [int(r["learned_action"]) for r in per_seed] == picks
    with open(out / "performance.csv") as f:
        perf = {r["portfolio"]: r for r in csv.DictReader(f)}
    assert set(perf) == {"continuous", "best_discrete", "equal_weight", "learned_run_0", "learned_run_1"}
    assert float(perf["learned_run_0"]["ce_loss_vs_best_discrete_bps"]) == 0.0
    for name in ("summary.json", "weights_grouped_bar.png", "learning_curve.png", "cumulative_regret.png", "ce_loss_by_seed.png"):
        assert (out / name).stat().st_size > 0, name


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_refuses_when_the_calibration_changed(tmp_path, market):
    run_dir, _ = train_with(tmp_path, [fixed(0)])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["calibration_sha256"]["covariance_daily.csv"] = "0" * 64
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed since training"):
        ev.evaluate_experiment(run_dir, EVAL)


def test_refuses_when_the_grid_changed(tmp_path, market):
    run_dir, _ = train_with(tmp_path, [fixed(0)])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["grid_sha256"] = "0" * 64
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid differs"):
        ev.evaluate_experiment(run_dir, EVAL)


def test_refuses_partial_experiments_unless_asked(tmp_path, market):
    run_dir, _ = train_with(tmp_path, [fixed(0), fixed(1)])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["runs"] = manifest["runs"][:1]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="1 of 2 runs"):
        ev.evaluate_experiment(run_dir, EVAL)
    assert ev.evaluate_experiment(run_dir, EVAL, allow_partial=True)["n_seeds"] == 1


def test_refuses_to_overwrite_an_earlier_evaluation(tmp_path, market):
    run_dir, _ = train_with(tmp_path, [fixed(0)])
    ev.evaluate_experiment(run_dir, EVAL)
    with pytest.raises(FileExistsError):
        ev.evaluate_experiment(run_dir, EVAL)
    ev.evaluate_experiment(run_dir, EVAL, overwrite=True)


# ---------------------------------------------------------------------------
# Comparing grid resolutions
# ---------------------------------------------------------------------------

def test_compare_two_resolutions(tmp_path, market):
    mu, sigma, grid25, values25 = market
    grid20 = generate_weight_grid(9, 0.2)
    values20 = grid20 @ mu - 0.5 * GAMMA * ((grid20 @ sigma) * grid20).sum(axis=1)
    worse20 = int(np.argsort(values20)[-4])
    dir25, _ = train_with(tmp_path, [fixed(int(np.argmax(values25)))], name="g25", increment=0.25)
    dir20, _ = train_with(tmp_path, [fixed(worse20)], name="g20", increment=0.2)
    results = ev.compare_experiments([dir25, dir20], ["25%", "20%"], EVAL, tmp_path / "cmp")
    assert results["25%"]["n_actions"] == 495 and results["20%"]["n_actions"] == 1287
    assert results["25%"]["ce_loss_vs_best_discrete_bps"]["median"] == 0.0
    assert results["20%"]["ce_loss_vs_best_discrete_bps"]["median"] == pytest.approx(252e4 * (values20.max() - values20[worse20]))
    assert results["20%"]["greedy_stable"]["share_of_seeds"] == 1.0
    assert (tmp_path / "cmp" / "convergence_by_resolution.png").stat().st_size > 0


def test_compare_refuses_experiments_that_differ_in_more_than_the_grid(tmp_path, market):
    dir_a, _ = train_with(tmp_path, [fixed(0)], name="a", increment=0.25)
    dir_b, _ = train_with(tmp_path, [fixed(0)], name="b", increment=0.2, gamma=3.0)
    with pytest.raises(ValueError, match="gamma"):
        ev.compare_experiments([dir_a, dir_b], ["a", "b"], EVAL, tmp_path / "cmp")
