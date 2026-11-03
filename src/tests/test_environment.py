"""Checks on the Part I simulator (src/environment.py).

Two kinds of evidence, because each misses what the other catches:

- Statistical tests compare 400,000 simulated days with mu and Sigma re-derived
  independently from excess_returns.csv. Tolerances are multiples of the
  sampling standard error. A 5-SE bound on one statistic fails by chance with
  probability about 6e-7 (the seeds are fixed, so the suite is deterministic
  unless a seed or numpy's normal generator changes).
- Daily Sharpe ratios are small, so even 400,000 days cannot detect a 15% error
  in mu (3.3 SE for SPY). An exact comparison against numpy's own
  multivariate_normal(method="cholesky"), from the same generator state,
  catches errors of that size and anything that reshuffles the draws.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.universe import TICKERS
from src.environment import (
    MIN_EPISODE_LENGTH,
    ReturnSimulator,
    SeedPlan,
    _cholesky_factor,
    load_calibration,
    simulate_returns,
)
from src.rewards import episodic_mean_variance_reward

PROCESSED = Path("data/processed")
N_SIM_DAYS = 400_000
EXACT = dict(rtol=0, atol=1e-15)  # returns are ~1e-2; allows BLAS rounding differences only


@pytest.fixture(scope="module")
def calibration():
    return load_calibration()


@pytest.fixture(scope="module")
def rederived_moments():
    """mu and Sigma computed independently from the modeling dataset."""
    excess = pd.read_csv(PROCESSED / "excess_returns.csv", index_col=0, float_precision="round_trip")
    return excess.mean().to_numpy(), excess.cov(ddof=1).to_numpy()


@pytest.fixture(scope="module")
def big_sample(calibration):
    mu, sigma = calibration
    return simulate_returns(mu, sigma, N_SIM_DAYS, np.random.default_rng(20261002))


# ---------------------------------------------------------------------------
# load_calibration
# ---------------------------------------------------------------------------

def test_loader_matches_moments_rederived_from_excess_returns(calibration, rederived_moments):
    mu, sigma = calibration
    mu_ref, sigma_ref = rederived_moments
    assert mu.shape == (9,) and sigma.shape == (9, 9)
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-14, atol=0)
    np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-14, atol=0)


def test_loader_columns_follow_tickers(calibration):
    """A reorder between the CSVs and TICKERS must not pass silently. Per
    DATA_CLEANING.md §6, SPY has the highest mean, VNQ the largest variance
    and TIP the smallest."""
    mu, sigma = calibration
    assert TICKERS[int(np.argmax(mu))] == "SPY"
    assert TICKERS[int(np.argmax(np.diag(sigma)))] == "VNQ"
    assert TICKERS[int(np.argmin(np.diag(sigma)))] == "TIP"


def test_calibrated_sigma_is_symmetric_positive_definite(calibration):
    _, sigma = calibration
    np.testing.assert_array_equal(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0


@pytest.mark.parametrize("which", ["covariance_columns", "summary_rows"])
def test_loader_rejects_files_not_in_ticker_order(tmp_path, which):
    stats = pd.read_csv(PROCESSED / "summary_statistics.csv", index_col="ticker")
    cov = pd.read_csv(PROCESSED / "covariance_daily.csv", index_col=0)
    swapped = TICKERS[1:2] + TICKERS[:1] + TICKERS[2:]
    if which == "covariance_columns":
        cov = cov[swapped]
    else:
        stats = stats.loc[swapped]
    stats.to_csv(tmp_path / "summary_statistics.csv")
    cov.to_csv(tmp_path / "covariance_daily.csv")
    with pytest.raises(ValueError, match="TICKERS"):
        load_calibration(tmp_path)


# ---------------------------------------------------------------------------
# The draws are exactly the canonical Gaussian sampler's
# ---------------------------------------------------------------------------

def numpy_reference(mu, sigma, n_days, seed):
    return np.random.default_rng(seed).multivariate_normal(mu, sigma, size=n_days, method="cholesky")


def test_simulate_returns_matches_numpy_reference_exactly(calibration):
    mu, sigma = calibration
    np.testing.assert_allclose(
        simulate_returns(mu, sigma, 2_000, np.random.default_rng(5)), numpy_reference(mu, sigma, 2_000, 5), **EXACT
    )


def test_simulator_episodes_match_numpy_reference_exactly(calibration):
    """Covers both sampling methods, so a per-episode transformation (for
    example de-meaning each episode) cannot hide behind the statistics."""
    mu, sigma = calibration
    reference = numpy_reference(mu, sigma, 100 * 20, 6)
    block = ReturnSimulator(mu, sigma, np.random.default_rng(6)).sample_episodes(100, 20)
    np.testing.assert_allclose(block.reshape(-1, 9), reference, **EXACT)

    sim = ReturnSimulator(mu, sigma, np.random.default_rng(6))
    one_at_a_time = np.concatenate([sim.sample_episode(20) for _ in range(100)])
    np.testing.assert_allclose(one_at_a_time, reference, **EXACT)


# ---------------------------------------------------------------------------
# The draws are i.i.d. N(mu, Sigma): statistical checks
# ---------------------------------------------------------------------------

def test_shape_dtype_and_finiteness(big_sample):
    assert big_sample.shape == (N_SIM_DAYS, 9)
    assert big_sample.dtype == np.float64
    assert np.isfinite(big_sample).all()


def test_sample_mean_converges_to_mu(big_sample, rederived_moments):
    mu, sigma = rederived_moments
    standard_error = np.sqrt(np.diag(sigma) / N_SIM_DAYS)
    z = (big_sample.mean(axis=0) - mu) / standard_error
    assert np.abs(z).max() < 5, z


def test_sample_covariance_converges_to_sigma(big_sample, rederived_moments):
    """For Gaussian data, Var(sample cov_ij) ~ (Sigma_ij^2 + Sigma_ii Sigma_jj) / N."""
    _, sigma = rederived_moments
    diag = np.diag(sigma)
    standard_error = np.sqrt((sigma**2 + np.outer(diag, diag)) / N_SIM_DAYS)
    z = (np.cov(big_sample, rowvar=False) - sigma) / standard_error
    assert np.abs(z).max() < 5, z


def test_days_are_serially_independent(big_sample):
    """Stateless baseline relies on i.i.d. days: every lag-1 correlation,
    own and cross-asset, should be zero up to sampling noise (SE ~ 1/sqrt(N))."""
    standardized = (big_sample - big_sample.mean(axis=0)) / big_sample.std(axis=0)
    lag1 = standardized[:-1].T @ standardized[1:] / (N_SIM_DAYS - 1)
    assert np.abs(lag1).max() < 5 / np.sqrt(N_SIM_DAYS)


def test_cholesky_factor_reproduces_sigma_exactly(calibration):
    _, sigma = calibration
    chol = _cholesky_factor(sigma)
    np.testing.assert_array_equal(np.triu(chol, k=1), 0.0)
    np.testing.assert_allclose(chol @ chol.T, sigma, rtol=1e-12, atol=1e-20)


def test_same_seed_same_draws_and_different_seed_different_draws(calibration):
    mu, sigma = calibration
    a = simulate_returns(mu, sigma, 50, np.random.default_rng(1))
    b = simulate_returns(mu, sigma, 50, np.random.default_rng(1))
    c = simulate_returns(mu, sigma, 50, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_works_for_any_number_of_assets():
    """The three-asset comparison (bonus credit) reuses the same code."""
    mu = np.array([3e-4, 1e-4, 2e-4])
    sigma = np.diag([1e-4, 2e-5, 9e-5])
    draws = simulate_returns(mu, sigma, 10, np.random.default_rng(0))
    assert draws.shape == (10, 3)


# ---------------------------------------------------------------------------
# Input validation: fail loudly, never guess
# ---------------------------------------------------------------------------

GOOD_MU = np.array([1e-4, 2e-4])
GOOD_SIGMA = np.array([[1e-4, 2e-5], [2e-5, 4e-4]])


@pytest.mark.parametrize(
    "mu, sigma, error, match",
    [
        (np.ones((2, 1)), GOOD_SIGMA, ValueError, "1-D"),
        (GOOD_MU, np.eye(3), ValueError, "2x2"),
        (np.array([np.nan, 1e-4]), GOOD_SIGMA, ValueError, "finite"),
        # lower triangle is positive definite, so np.linalg.cholesky alone would accept it
        (GOOD_MU, np.array([[1e-4, 9e-5], [2e-5, 4e-4]]), ValueError, "symmetric"),
        (GOOD_MU, np.array([[1.0, 2.0], [2.0, 1.0]]), ValueError, "positive definite"),
        (pd.Series(GOOD_MU), GOOD_SIGMA, TypeError, "numpy arrays"),
    ],
)
def test_rejects_invalid_moments(mu, sigma, error, match):
    with pytest.raises(error, match=match):
        simulate_returns(mu, sigma, 5, np.random.default_rng(0))
    with pytest.raises(error, match=match):
        ReturnSimulator(mu, sigma, np.random.default_rng(0))


def asymmetric_by(relative_gap):
    sigma = GOOD_SIGMA.copy()
    sigma[0, 1] *= 1 + relative_gap
    return sigma


def test_symmetry_tolerance_admits_rounding_and_rejects_real_asymmetry():
    """Shrinkage estimators can leave rounding-level asymmetry, which must
    pass; a gap a hundred times the tolerance must not."""
    simulate_returns(GOOD_MU, asymmetric_by(1e-15), 5, np.random.default_rng(0))
    with pytest.raises(ValueError, match="symmetric"):
        simulate_returns(GOOD_MU, asymmetric_by(1e-8), 5, np.random.default_rng(0))


@pytest.mark.parametrize("rng", [None, 0, np.random.RandomState(0)])
def test_rejects_anything_but_a_numpy_generator(rng):
    with pytest.raises(TypeError, match="Generator"):
        simulate_returns(GOOD_MU, GOOD_SIGMA, 5, rng)
    with pytest.raises(TypeError, match="Generator"):
        ReturnSimulator(GOOD_MU, GOOD_SIGMA, rng)


@pytest.mark.parametrize("n_days, error", [(0, ValueError), (-3, ValueError), (2.5, TypeError), (True, TypeError)])
def test_rejects_invalid_day_counts(n_days, error):
    with pytest.raises(error):
        simulate_returns(GOOD_MU, GOOD_SIGMA, n_days, np.random.default_rng(0))


def test_episodes_shorter_than_two_days_are_refused():
    """A one-day batch makes the reward's ddof=1 variance NaN (see MIN_EPISODE_LENGTH)."""
    assert MIN_EPISODE_LENGTH == 2
    sim = ReturnSimulator(GOOD_MU, GOOD_SIGMA, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least 2"):
        sim.sample_episode(1)
    with pytest.raises(ValueError, match="at least 2"):
        sim.sample_episodes(10, 1)
    with pytest.raises(ValueError, match="at least 1"):
        sim.sample_episodes(0, 20)


# ---------------------------------------------------------------------------
# ReturnSimulator
# ---------------------------------------------------------------------------

def test_public_surface_exposes_no_moments(calibration):
    """Anything public on the simulator is reachable by the agent, so the
    public surface is pinned to sampling methods, counters and the asset count."""
    sim = ReturnSimulator(*calibration, np.random.default_rng(0))
    public = {name for name in dir(sim) if not name.startswith("_")}
    assert public == {"n_assets", "days_simulated", "episodes_simulated", "sample_episode", "sample_episodes"}


def test_simulator_and_function_give_identical_draws(calibration):
    mu, sigma = calibration
    from_function = simulate_returns(mu, sigma, 20, np.random.default_rng(7))
    from_simulator = ReturnSimulator(mu, sigma, np.random.default_rng(7)).sample_episode(20)
    np.testing.assert_allclose(from_function, from_simulator, **EXACT)


def test_block_sampling_equals_sequential_sampling(calibration):
    """sample_episodes is a speed-up only; it must not change the draws."""
    block = ReturnSimulator(*calibration, np.random.default_rng(11)).sample_episodes(200, 20)
    sim = ReturnSimulator(*calibration, np.random.default_rng(11))
    sequential = np.stack([sim.sample_episode(20) for _ in range(200)])
    assert block.shape == (200, 20, 9)
    np.testing.assert_allclose(block, sequential, **EXACT)


def test_counters_track_simulated_days_and_episodes(calibration):
    sim = ReturnSimulator(*calibration, np.random.default_rng(0))
    sim.sample_episode(20)
    sim.sample_episodes(5, 10)
    assert sim.days_simulated == 70
    assert sim.episodes_simulated == 6


def test_simulator_is_unaffected_by_later_edits_to_the_caller_mu(calibration):
    mu, sigma = calibration
    mu_copy = mu.copy()
    sim = ReturnSimulator(mu_copy, sigma, np.random.default_rng(3))
    mu_copy[:] = 1.0
    reference = simulate_returns(mu, sigma, 20, np.random.default_rng(3))
    np.testing.assert_allclose(sim.sample_episode(20), reference, **EXACT)


def test_reward_on_simulated_episodes_is_unbiased_for_the_true_objective(calibration):
    """End-to-end interface check with src/rewards.py on the real calibration:
    the average episodic reward converges to w'mu - (gamma/2) w'Sigma w.
    Equal weights and gamma = 5 are arbitrary test values, not project choices."""
    mu, sigma = calibration
    w = np.full(9, 1 / 9)
    gamma = 5.0
    episodes = ReturnSimulator(mu, sigma, np.random.default_rng(99)).sample_episodes(20_000, 20)
    rewards = np.array([episodic_mean_variance_reward(w, batch, gamma) for batch in episodes])
    true_objective = w @ mu - 0.5 * gamma * (w @ sigma @ w)
    standard_error = rewards.std(ddof=1) / np.sqrt(rewards.size)
    assert abs(rewards.mean() - true_objective) < 5 * standard_error


# ---------------------------------------------------------------------------
# SeedPlan
# ---------------------------------------------------------------------------

def first_draws(rng, n=8):
    return rng.standard_normal(n)


def test_run_streams_are_reproducible():
    a_returns, a_agent = SeedPlan(123, 3).run_streams(1)
    b_returns, b_agent = SeedPlan(123, 3).run_streams(1)
    np.testing.assert_array_equal(first_draws(a_returns), first_draws(b_returns))
    np.testing.assert_array_equal(first_draws(a_agent), first_draws(b_agent))


def test_all_streams_are_distinct():
    plan = SeedPlan(123, 3)
    draws = [first_draws(plan.evaluation_rng())]
    for run in range(plan.n_runs):
        returns_rng, agent_rng = plan.run_streams(run)
        draws += [first_draws(returns_rng), first_draws(agent_rng)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.allclose(draws[i], draws[j])


def test_plain_numpy_seeding_from_the_master_seed_cannot_reproduce_a_plan_stream():
    """Someone seeding a side task with default_rng(master_seed), or spawning
    from it, must not end up sharing draws with training or evaluation."""
    master = 123
    plan = SeedPlan(master, 3)
    plan_draws = [first_draws(plan.evaluation_rng())]
    plan_draws += [first_draws(rng) for run in range(3) for rng in plan.run_streams(run)]
    outside = [first_draws(np.random.default_rng(master))]
    outside += [first_draws(child) for child in np.random.default_rng(master).spawn(4)]
    for a in plan_draws:
        for b in outside:
            assert not np.allclose(a, b)


def test_evaluation_stream_restarts_on_every_call():
    """Common random numbers: every policy is evaluated on the same draws."""
    plan = SeedPlan(123, 3)
    np.testing.assert_array_equal(first_draws(plan.evaluation_rng()), first_draws(plan.evaluation_rng()))


def test_adding_runs_later_leaves_existing_runs_unchanged():
    small, large = SeedPlan(123, 3), SeedPlan(123, 10)
    for run in range(3):
        for a, b in zip(small.run_streams(run), large.run_streams(run)):
            np.testing.assert_array_equal(first_draws(a), first_draws(b))
    np.testing.assert_array_equal(first_draws(small.evaluation_rng()), first_draws(large.evaluation_rng()))


def test_different_master_seeds_give_different_streams():
    assert not np.allclose(first_draws(SeedPlan(1, 1).run_streams(0)[0]), first_draws(SeedPlan(2, 1).run_streams(0)[0]))


@pytest.mark.parametrize("master_seed, n_runs, error", [(-1, 1, ValueError), (1.5, 1, TypeError), (None, 1, TypeError), (1, 0, ValueError)])
def test_seed_plan_rejects_invalid_settings(master_seed, n_runs, error):
    with pytest.raises(error):
        SeedPlan(master_seed, n_runs)


@pytest.mark.parametrize("run", [-1, 3])
def test_run_index_must_be_in_range(run):
    with pytest.raises(ValueError):
        SeedPlan(123, 3).run_streams(run)


@pytest.mark.parametrize("master_seed, n_runs", [(123, 3), (np.int64(123), np.int64(3))])
def test_seed_record_is_json_serializable(master_seed, n_runs):
    """numpy integers (e.g. from a config array) must not break results logging."""
    record = json.loads(json.dumps(SeedPlan(master_seed, n_runs).as_dict()))
    assert record["master_seed"] == 123 and record["n_runs"] == 3


# ---------------------------------------------------------------------------
# Leakage guard: the agent module must not be able to reach mu, Sigma or the
# benchmarks. Assignment Part I: those objects "are reserved for benchmark
# construction and evaluation." Two checks, because each sees what the other
# cannot: a source scan (data-file paths, names) and the actual import graph
# (helpers that import the benchmarks on the agent's behalf). Not covered, and
# left to code review: values the training loop passes to the agent by hand,
# and deliberate bypasses such as reading ReturnSimulator._mu. If the agent is
# split across more modules later, add them to AGENT_MODULES.
# ---------------------------------------------------------------------------

AGENT_MODULES = ["src.agent", "src.features", "src.fa_agent"]  # the last two: function-approximation extension
FORBIDDEN_MODULES = ("src.benchmarks", "src.data.clean")
FORBIDDEN_NAMES = {"load_calibration", "simulate_returns"}
# A path into the repo's data directory (either slash), or a processed file name.
FORBIDDEN_PATH = re.compile(
    r"(^|[/\\])data[/\\]|(excess_returns|returns_simple|adj_close|rf_daily|covariance_daily"
    r"|correlation_matrix|summary_statistics)\.csv"
)


def leakage_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative import from inside src/
                module = f"src.{module}" if module else "src"
            targets = [module] + [f"{module}.{alias.name}" for alias in node.names]
            found += [alias.name for alias in node.names if alias.name in FORBIDDEN_NAMES]
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append(node.attr)
            continue
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(node.id)
            continue
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if FORBIDDEN_PATH.search(node.value):
                found.append(node.value)
            continue
        else:
            continue
        found += [t for t in targets if t.startswith(FORBIDDEN_MODULES)]
    return found


def modules_loaded_by(statement: str) -> set[str]:
    """Every module in sys.modules after running `statement` in a fresh interpreter."""
    code = f"{statement}\nimport sys\nprint('\\n'.join(sys.modules))"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return set(result.stdout.split())


@pytest.mark.parametrize("module", AGENT_MODULES)
def test_agent_source_cannot_reach_moments_or_benchmarks(module):
    path = Path(*module.split(".")).with_suffix(".py")
    assert leakage_violations(path.read_text()) == []


@pytest.mark.parametrize("module", AGENT_MODULES)
def test_agent_import_graph_excludes_benchmarks_and_cleaning(module):
    loaded = modules_loaded_by(f"import {module}")
    assert module in loaded
    assert sorted(m for m in loaded if m.startswith(FORBIDDEN_MODULES)) == []


def test_import_graph_check_sees_transitive_imports(tmp_path):
    """agent -> helper -> benchmarks is invisible to the source scan of the
    agent, so prove the import-graph check catches it."""
    (tmp_path / "helper_that_leaks.py").write_text("import src.benchmarks.mean_variance\n")
    loaded = modules_loaded_by(f"import sys; sys.path.append({str(tmp_path)!r}); import helper_that_leaks")
    assert "src.benchmarks.mean_variance" in loaded


@pytest.mark.parametrize(
    "snippet",
    [
        "from src.benchmarks.mean_variance import best_discrete_weights",
        "import src.benchmarks.mean_variance as mv",
        "from src import benchmarks",
        "from .benchmarks.mean_variance import closed_form_optimal_weights",
        "from src.data.clean import main",
        "from src.environment import load_calibration",
        "import src.environment as env\nmu, sigma = env.load_calibration()",
        "from src.environment import simulate_returns",
        "import pandas as pd\nsigma = pd.read_csv('data/processed/covariance_daily.csv')",
        "prices = open('data/raw/prices/SPY.csv').read()",
        "path = '..\\\\data\\\\processed\\\\returns_simple.csv'",
        "frame = load(PROCESSED / 'excess_returns.csv')",
    ],
)
def test_leakage_check_catches_each_route(snippet):
    """The agent module is a placeholder today, so the guards above pass
    trivially. These cases prove the source scan would fail a real leak."""
    assert leakage_violations(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "from src.environment import ReturnSimulator",
        "import numpy as np\nfrom src.rewards import episodic_mean_variance_reward",
        '"""Never read data/processed from here."""\nimport numpy as np',
        "log.info('excess returns seen so far: %d', n)",
        "config = {'metadata/': 1, 'data': 2}",
    ],
)
def test_leakage_check_allows_legitimate_agent_code(snippet):
    assert leakage_violations(snippet) == []
