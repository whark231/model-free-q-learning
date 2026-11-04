"""Checks on the training runner (src/train.py).

TabularQAgent's algorithm is still being written on feature/agent, so these
tests drive the runner with stand-in agents that follow the same interface.
The checks are about the runner itself: every reward and every logged number
is rebuilt independently and compared exactly, so a wiring or bookkeeping
mistake cannot hide behind sampling noise.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import src.train as train
from src.action_space.grid import generate_weight_grid
from src.environment import ReturnSimulator, SeedPlan, load_calibration
from src.fa_agent import LinearQAgent
from src.features import quadratic_features
from src.rewards import episodic_mean_variance_reward
from src.train import TrainConfig, alpha_callable, run_experiment


class EpsilonGreedyAverager:
    """Stand-in agent: epsilon-greedy over sample averages, so its choices
    depend on its Q-values (a runner bug that changed any reward would change
    its whole trajectory)."""

    def __init__(self, n_actions, rng, epsilon=0.3):
        self._rng = rng
        self._q = np.zeros(n_actions)
        self._n = np.zeros(n_actions, dtype=int)
        self._epsilon = epsilon

    def select_action(self):
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(self._q.size))
        return int(np.argmax(self._q))

    def update(self, action, reward):
        self._n[action] += 1
        self._q[action] += (reward - self._q[action]) / self._n[action]

    @property
    def q_values(self):
        return self._q.copy()

    @property
    def visit_counts(self):
        return self._n.copy()

    @property
    def epsilon(self):
        return self._epsilon


class Recorder(EpsilonGreedyAverager):
    """Records every (action, reward) and the Q-table after every update."""

    def __init__(self, n_actions, rng):
        super().__init__(n_actions, rng)
        self.actions, self.rewards, self.q_history, self.update_types = [], [], [], []

    def update(self, action, reward):
        super().update(action, reward)
        self.actions.append(action)
        self.rewards.append(reward)
        self.q_history.append(self._q.copy())
        self.update_types.append((type(action), type(reward)))


def averager_factory(config, n_actions, rng):
    return EpsilonGreedyAverager(n_actions, rng)


def recording_factory(store):
    def factory(config, n_actions, rng):
        agent = Recorder(n_actions, rng)
        store.append(agent)
        return agent

    return factory


def small_config(**overrides):
    settings = dict(
        gamma=5.0, episode_length=20, n_episodes=2_500, master_seed=123, n_runs=2,
        beta=1e-3, increment=0.25, log_every=1_000, block_size=400,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def load_run(out_dir, run=0):
    with np.load(out_dir / f"run_{run:03d}.npz") as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Every reward is the chosen portfolio's score on that run's training draws
# ---------------------------------------------------------------------------

def test_every_reward_matches_an_independent_rebuild(tmp_path):
    """Rebuild each reward from scratch: the run's own returns stream, drawn
    one episode at a time, the grid row of the logged action, and the reward
    function. Catches training on the evaluation or agent stream, wrong
    columns, wrong gamma, wrong episode length and wrong grid rows."""
    config = small_config()
    agents = []
    run_experiment(config, tmp_path, agent_factory=recording_factory(agents))

    mu, sigma = load_calibration()
    grid = generate_weight_grid(9, config.increment)
    plan = SeedPlan(config.master_seed, config.n_runs)
    for run, agent in enumerate(agents):
        sim = ReturnSimulator(mu, sigma, plan.run_streams(run)[0])
        expected = [
            episodic_mean_variance_reward(grid[a], sim.sample_episode(config.episode_length), config.gamma)
            for a in agent.actions
        ]
        assert len(agent.rewards) == config.n_episodes
        np.testing.assert_array_equal(agent.rewards, expected)


def test_agent_receives_only_integer_actions_and_float_rewards(tmp_path):
    agents = []
    run_experiment(small_config(n_runs=1, n_episodes=200), tmp_path, agent_factory=recording_factory(agents))
    assert set(agents[0].update_types) == {(int, float)}


# ---------------------------------------------------------------------------
# Every logged number matches an independent recomputation
# ---------------------------------------------------------------------------

def test_logs_match_independent_recomputation(tmp_path):
    config = small_config(n_runs=1, n_episodes=2_500, log_every=1_000)  # last window is partial
    agents = []
    run_experiment(config, tmp_path, agent_factory=recording_factory(agents))
    agent, logs = agents[0], load_run(tmp_path)
    rewards = np.array(agent.rewards)
    actions = np.array(agent.actions)
    q_before = [np.zeros(len(agent.q_history[0]))] + agent.q_history[:-1]
    dq = np.array([abs(after[a] - before[a]) for a, before, after in zip(actions, q_before, agent.q_history)])
    greedy = np.array([np.argmax(q) for q in agent.q_history])

    ends = np.array([1_000, 2_000, 2_500])
    starts = np.array([0, 1_000, 2_000])
    np.testing.assert_array_equal(logs["checkpoint_episode"], ends)
    for i, (s, e) in enumerate(zip(starts, ends)):
        assert logs["checkpoint_mean_reward"][i] == pytest.approx(rewards[s:e].mean(), rel=1e-12)
        assert logs["checkpoint_max_abs_dq"][i] == dq[s:e].max()
        np.testing.assert_array_equal(logs["checkpoint_action_counts"][i], np.bincount(actions[s:e], minlength=495))
        assert logs["checkpoint_greedy"][i] == greedy[e - 1]
        np.testing.assert_array_equal(logs["checkpoint_q"][i], agent.q_history[e - 1].astype(np.float32))
        assert logs["checkpoint_next_epsilon"][i] == agent.epsilon  # read from the agent, not the config

    changes = [(0, 0)] + [(t + 1, g) for t, g in enumerate(greedy) if g != (greedy[t - 1] if t else 0)]
    np.testing.assert_array_equal(logs["greedy_change_episode"], [c[0] for c in changes])
    np.testing.assert_array_equal(logs["greedy_change_action"], [c[1] for c in changes])
    np.testing.assert_array_equal(logs["final_q"], agent.q_history[-1])
    assert logs["final_greedy"] == greedy[-1]


def test_logged_epsilon_is_the_real_agents_schedule_for_the_next_episode(tmp_path):
    """With the real agent, the logged rate after e episodes must be
    epsilon0 * exp(-beta * e), recomputed here from the formula. Reading the
    agent one selection early or late would be off by a factor exp(beta)."""
    config = small_config(n_runs=1, n_episodes=2_500, log_every=1_000)
    run_experiment(config, tmp_path)
    logs = load_run(tmp_path)
    expected = [config.epsilon0 * np.exp(-config.beta * e) for e in (1_000, 2_000, 2_500)]
    np.testing.assert_allclose(logs["checkpoint_next_epsilon"], expected, rtol=1e-12)


def test_max_q_change_is_per_window_and_measures_the_change(tmp_path):
    """One portfolio chosen every time: its Q-value settles, so the largest
    |change in Q| must fall from window to window. Never resetting the window,
    or logging |Q| instead of |change in Q|, would break this."""

    class OneAction(Recorder):
        def select_action(self):
            return 0

    agents = []
    config = small_config(n_runs=1, n_episodes=3_000, log_every=1_000)
    run_experiment(config, tmp_path, agent_factory=lambda c, n, rng: agents.append(OneAction(n, rng)) or agents[-1])
    q0 = np.array([q[0] for q in agents[0].q_history])
    dq = np.abs(np.diff(np.concatenate([[0.0], q0])))
    expected = [dq[0:1_000].max(), dq[1_000:2_000].max(), dq[2_000:3_000].max()]
    np.testing.assert_array_equal(load_run(tmp_path)["checkpoint_max_abs_dq"], expected)
    assert expected[2] < expected[1] < expected[0]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_config_gives_identical_logs(tmp_path):
    run_experiment(small_config(), tmp_path / "a", agent_factory=averager_factory)
    run_experiment(small_config(), tmp_path / "b", agent_factory=averager_factory)
    for run in range(2):
        a, b = load_run(tmp_path / "a", run), load_run(tmp_path / "b", run)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("overrides", [dict(block_size=1), dict(block_size=7), dict(log_every=333)])
def test_block_size_and_log_frequency_do_not_change_training(tmp_path, overrides):
    run_experiment(small_config(n_runs=1), tmp_path / "ref", agent_factory=averager_factory)
    run_experiment(small_config(n_runs=1, **overrides), tmp_path / "x", agent_factory=averager_factory)
    ref, other = load_run(tmp_path / "ref"), load_run(tmp_path / "x")
    for key in ("final_q", "final_visit_counts", "greedy_change_episode", "greedy_change_action"):
        np.testing.assert_array_equal(other[key], ref[key], err_msg=key)


def test_adding_runs_leaves_earlier_runs_unchanged(tmp_path):
    run_experiment(small_config(n_runs=1), tmp_path / "one", agent_factory=averager_factory)
    run_experiment(small_config(n_runs=3), tmp_path / "three", agent_factory=averager_factory)
    np.testing.assert_array_equal(load_run(tmp_path / "one")["final_q"], load_run(tmp_path / "three")["final_q"])


def test_runs_differ_from_each_other(tmp_path):
    run_experiment(small_config(), tmp_path, agent_factory=averager_factory)
    assert not np.array_equal(load_run(tmp_path, 0)["final_q"], load_run(tmp_path, 1)["final_q"])


# ---------------------------------------------------------------------------
# The real agent is configured exactly as the config says
# ---------------------------------------------------------------------------

def test_default_factory_passes_every_setting_to_tabular_q_agent(monkeypatch):
    captured = {}

    class Capture:
        def __init__(self, n_actions, **kwargs):
            captured.update(kwargs, n_actions=n_actions)

    monkeypatch.setattr(train, "TabularQAgent", Capture)
    config = small_config(epsilon0=0.9, beta=3e-4, q_init=0.25, discount=0.3, alpha_rule=0.05)
    rng = np.random.default_rng(0)
    train.default_agent_factory(config, 495, rng)
    assert captured == dict(
        n_actions=495, alpha_rule=0.05, epsilon0=0.9, beta=3e-4, q_init=0.25, discount=0.3, rng=rng
    )


def test_sample_average_rule_counts_the_current_update():
    rule = alpha_callable("sample_average")
    assert [rule(n) for n in (1, 2, 4)] == [1.0, 0.5, 0.25]
    with pytest.raises(ValueError):
        rule(np.int64(0))
    assert alpha_callable(0.01) == 0.01


def test_a_non_finite_q_value_stops_the_run(tmp_path):
    class Exploding(EpsilonGreedyAverager):
        def update(self, action, reward):
            self._q[action] = np.inf

    with pytest.raises(FloatingPointError, match="learning-rate rule"):
        run_experiment(small_config(n_runs=1), tmp_path, agent_factory=lambda c, n, rng: Exploding(n, rng))


# ---------------------------------------------------------------------------
# Output folder and manifest
# ---------------------------------------------------------------------------

def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    manifest = run_experiment(small_config(), tmp_path, agent_factory=averager_factory)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == json.loads(json.dumps(manifest))
    assert on_disk["config"]["gamma"] == 5.0 and on_disk["config"]["episode_length"] == 20
    assert on_disk["seed_plan"]["master_seed"] == 123
    assert set(on_disk["calibration_sha256"]) == {"summary_statistics.csv", "covariance_daily.csv"}
    assert set(on_disk["code_sha256"]) == set(train.CODE_FILES)
    assert on_disk["grid_sha256"] == train.grid_fingerprint(generate_weight_grid(9, 0.25))
    assert on_disk["agent_class"].endswith("test_train.EpsilonGreedyAverager")
    assert [r["simulated_days"] for r in on_disk["runs"]] == [50_000, 50_000]
    assert json.loads(str(load_run(tmp_path)["config_json"]))["gamma"] == 5.0


def test_refuses_a_non_empty_output_folder(tmp_path):
    (tmp_path / "old_run.npz").write_text("stale")
    with pytest.raises(FileExistsError):
        run_experiment(small_config(), tmp_path, agent_factory=averager_factory)


def test_a_crash_mid_experiment_leaves_a_manifest_for_the_finished_runs(tmp_path):
    calls = []

    def factory(config, n_actions, rng):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash in run 1")
        return EpsilonGreedyAverager(n_actions, rng)

    with pytest.raises(RuntimeError):
        run_experiment(small_config(), tmp_path, agent_factory=factory)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["run"] for r in manifest["runs"]] == [0]


def test_snapshot_size_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MAX_SNAPSHOT_VALUES", 1_000)
    with pytest.raises(MemoryError):
        run_experiment(small_config(), tmp_path, agent_factory=averager_factory)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (dict(episode_length=1), ValueError),
        (dict(n_episodes=0), ValueError),
        (dict(n_episodes=1000.0), TypeError),
        (dict(epsilon0=1.5), ValueError),
        (dict(beta=-1.0), ValueError),
        (dict(gamma=float("nan")), ValueError),
        (dict(discount=1.0), ValueError),
        (dict(alpha_rule="adam"), ValueError),
        (dict(alpha_rule=0.0), ValueError),
        (dict(alpha_rule=1.5), ValueError),
        (dict(alpha_rule=True), ValueError),
    ],
)
def test_config_rejects_invalid_settings(overrides, error):
    with pytest.raises(error):
        small_config(**overrides)


# ---------------------------------------------------------------------------
# Leakage rules
# ---------------------------------------------------------------------------

def test_training_never_touches_the_evaluation_stream(tmp_path, monkeypatch):
    """Complements the exact rebuild above, which catches any other route to
    the evaluation draws."""

    def forbidden(self):
        raise AssertionError("train.py must not draw from the evaluation stream")

    monkeypatch.setattr(SeedPlan, "evaluation_rng", forbidden)
    run_experiment(small_config(n_runs=1), tmp_path, agent_factory=averager_factory)


def test_train_module_does_not_load_the_benchmarks():
    """The runner holds mu and Sigma; it must not also hold the optimum, or
    something derived from it could steer training."""
    code = "import src.train, sys; print('\\n'.join(sys.modules))"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.split()
    assert "src.train" in loaded
    assert not [m for m in loaded if m.startswith("src.benchmarks")]


# ---------------------------------------------------------------------------
# train.py's own Q-table copy (REVIEW_HISTORY.md §5.3): used only when
# update() returns the new Q(action), and must change nothing in the logs
# ---------------------------------------------------------------------------

class ReturningAverager(EpsilonGreedyAverager):
    def update(self, action, reward):
        super().update(action, reward)
        return self._q[action]


class ReturningTabularQAgent(train.TabularQAgent):
    """William's proposed update(): the same agent, returning the new Q(action)."""

    def update(self, action, reward):
        super().update(action, reward)
        return float(self._q[action])


def returning_tabular_factory(config, n_actions, rng):
    agent = train.default_agent_factory(config, n_actions, rng)
    agent.__class__ = ReturningTabularQAgent
    return agent


@pytest.mark.parametrize("plain, returning", [
    (averager_factory, lambda c, n, rng: ReturningAverager(n, rng)),
    (train.default_agent_factory, returning_tabular_factory),
])
def test_returned_q_value_gives_bit_identical_logs(tmp_path, plain, returning):
    run_experiment(small_config(), tmp_path / "copy", agent_factory=plain)
    run_experiment(small_config(), tmp_path / "mirror", agent_factory=returning)
    for run in range(2):
        a, b = load_run(tmp_path / "copy", run), load_run(tmp_path / "mirror", run)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


class WrongReturn(EpsilonGreedyAverager):
    def update(self, action, reward):
        super().update(action, reward)
        return self._q[action] + 1e-12


class TouchesAnotherEntry(EpsilonGreedyAverager):
    def update(self, action, reward):
        super().update(action, reward)
        self._q[(action + 1) % self._q.size] += 1e-9
        return self._q[action]


@pytest.mark.parametrize("cls", [WrongReturn, TouchesAnotherEntry])
def test_a_copy_that_drifts_from_the_agent_stops_the_run(tmp_path, cls):
    with pytest.raises(RuntimeError, match="copy of the Q-table differs"):
        run_experiment(small_config(), tmp_path, agent_factory=lambda c, n, rng: cls(n, rng))


# ---------------------------------------------------------------------------
# Function-approximation path (post hoc): train_one_run_fa
# ---------------------------------------------------------------------------


def fa_config(**overrides):
    settings = dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, greedy_refresh_every=10)
    settings.update(overrides)
    return small_config(**settings)


class RecordingLinear(LinearQAgent):
    """LinearQAgent that records every (action, reward) it is given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions, self.rewards = [], []

    def update(self, action, reward):
        super().update(action, reward)
        self.actions.append(action)
        self.rewards.append(reward)


def test_logged_greedy_matches_the_learned_portfolio_in_every_run(tmp_path):
    """The reviewer's failure case: before refreshes had to divide the budget,
    6 of 16 small runs ended on a stale greedy. Now every run must agree."""
    config = fa_config(n_runs=6, alpha_rule=0.2, greedy_refresh_every=50)
    run_experiment(config, tmp_path)
    for run in range(6):
        logs = load_run(tmp_path, run)
        assert int(logs["final_greedy"]) == int(np.argmax(logs["final_q"]))


def test_a_factory_that_builds_the_wrong_agent_type_is_refused(tmp_path):
    F = quadratic_features(generate_weight_grid(9, 0.25))

    def factory(c, n, rng):
        return LinearQAgent(F, alpha=0.1, epsilon0=1.0, beta=0.0, rng=rng)

    with pytest.raises(TypeError, match="wrong agent"):
        run_experiment(small_config(n_runs=1), tmp_path, agent_factory=factory)


def test_fa_rewards_match_an_independent_rebuild(tmp_path):
    """The FA path must draw the same returns stream and grid rows as the
    tabular path: rebuild every reward from scratch and compare exactly."""
    config = fa_config(n_runs=1)
    F = quadratic_features(generate_weight_grid(9, config.increment))
    agents = []

    def factory(c, n, rng):
        agents.append(RecordingLinear(F, alpha=0.1, epsilon0=c.epsilon0, beta=c.beta, rng=rng,
                                      greedy_refresh_every=c.greedy_refresh_every))
        return agents[-1]

    run_experiment(config, tmp_path, agent_factory=factory)
    mu, sigma = load_calibration()
    grid = generate_weight_grid(9, config.increment)
    sim = ReturnSimulator(mu, sigma, SeedPlan(config.master_seed, 1).run_streams(0)[0])
    expected = [episodic_mean_variance_reward(grid[a], sim.sample_episode(config.episode_length), config.gamma)
                for a in agents[0].actions]
    assert len(expected) == config.n_episodes
    np.testing.assert_array_equal(agents[0].rewards, expected)


def test_fa_run_logs_exact_q_values_and_theta(tmp_path):
    config = fa_config(n_runs=1)
    manifest = run_experiment(config, tmp_path)
    logs = load_run(tmp_path)
    F = quadratic_features(generate_weight_grid(9, config.increment))
    np.testing.assert_allclose(logs["final_q"], F @ logs["final_theta"], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(logs["checkpoint_q"][-1], logs["final_q"].astype(np.float32))
    assert manifest["agent_class"] == "src.fa_agent.LinearQAgent"
    assert manifest["n_features"] == 45
    assert manifest["config"]["agent"] == "linear" and manifest["config"]["feature_map"] == "quadratic"
    assert {"src/fa_agent.py", "src/features.py"} <= set(manifest["code_sha256"])


def test_fa_greedy_changes_only_at_refreshes(tmp_path):
    """The cached greedy can change only on an update that triggers a
    refresh, so every logged change falls on a multiple of the interval."""
    config = fa_config(n_runs=1, greedy_refresh_every=10)
    run_experiment(config, tmp_path)
    logs = load_run(tmp_path)
    changes = logs["greedy_change_episode"][1:]
    assert changes.size > 0
    assert np.all(changes % 10 == 0)
    # The run ends on a refresh, so the logged greedy is the portfolio evaluate.py reports.
    assert int(logs["final_greedy"]) == int(np.argmax(logs["final_q"]))


def test_fa_run_is_reproducible_from_its_seed(tmp_path):
    run_experiment(fa_config(n_runs=1), tmp_path / "a")
    run_experiment(fa_config(n_runs=1), tmp_path / "b")
    a, b = load_run(tmp_path / "a"), load_run(tmp_path / "b")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_tabular_record_and_hashes_are_unchanged_by_the_fa_fields(tmp_path):
    """A tabular run's manifest must not gain the FA keys or FA code hashes,
    so it still matches the baseline-v1 manifests."""
    manifest = run_experiment(small_config(n_runs=1), tmp_path, agent_factory=averager_factory)
    assert not set(train.FA_FIELD_DEFAULTS) & set(manifest["config"])
    assert not {"src/fa_agent.py", "src/features.py"} & set(manifest["code_sha256"])
    assert "n_features" not in manifest


@pytest.mark.parametrize("overrides", [
    dict(feature_map="quadratic"),  # tabular run with an FA setting
    dict(l2=0.1),
    dict(greedy_refresh_every=5),
    dict(agent="spline"),  # not an agent type
    dict(agent="linear", alpha_rule=0.1),  # no feature map
    dict(agent="linear", feature_map="cubic", alpha_rule=0.1),
    dict(agent="linear", feature_map="quadratic"),  # sample-average alpha
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, discount=0.5),
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, q_init=1.0),
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, l2=-1.0),
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, greedy_refresh_every=0),
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, greedy_refresh_every=7),  # 2,500 not a multiple
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, greedy_refresh_every=500, log_every=750),
    dict(agent="linear", feature_map="quadratic", alpha_rule=0.1, l2=True),
])
def test_inconsistent_fa_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        small_config(**overrides)
