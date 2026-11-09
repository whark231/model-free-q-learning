"""Part I ground-truth environment: the simulator the agent learns from.

The assignment fixes the data-generating process. Daily excess returns are
i.i.d. draws from a stationary Gaussian,

    r_{t+1} = mu + eps_{t+1},    eps_{t+1} ~ N(0, Sigma),

with mu and Sigma held at the values calibrated in Part I (data/processed/).
They are ground truth only inside this experiment (assignment Part I).

Leakage rule (assignment Part I, CLAUDE.md §5): the agent may see simulated
returns and rewards, never mu, Sigma or the theoretical optimum. So there are
two layers here:

- load_calibration() and simulate_returns() return or take mu and Sigma. They
  are for the experiment runner, the benchmarks and the tests.
- ReturnSimulator is the only object the agent should be handed. It keeps mu
  and the Cholesky factor of Sigma private and exposes only simulated returns.
  src/tests/test_environment.py checks that src/agent.py does not import the
  loader, the benchmarks or the data files, directly or through other modules.

Seeds: SeedPlan derives every random stream in an experiment from one recorded
master seed, and keeps evaluation draws on a separate branch from training
draws.

Usage, from the repo root:

    from src.environment import ReturnSimulator, SeedPlan, load_calibration

    mu, sigma = load_calibration()
    plan = SeedPlan(master_seed=..., n_runs=10)
    returns_rng, agent_rng = plan.run_streams(0)
    sim = ReturnSimulator(mu, sigma, returns_rng)
    batch = sim.sample_episode(20)        # (20, 9) daily excess returns

Design notes and decisions not taken are in ENVIRONMENT.md.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.universe import TICKERS

PROCESSED_DIR = Path("data/processed")

# src.rewards.episodic_mean_variance_reward uses a ddof=1 variance, which needs
# at least two days; it raises ValueError for a one-day batch (it used to return
# NaN). Refusing short episodes here as well stops a run before training starts.
# The one-period quadratic reward the assignment also permits would need this
# lowered to 1.
MIN_EPISODE_LENGTH = 2

# SeedSequence spawn-key layout used by SeedPlan. Every key starts with
# _PLAN_TAG so that plain numpy spawning from the same master seed
# (default_rng(seed).spawn(n) gives keys (0,), (1,), ...) can never reproduce
# one of these streams.
_PLAN_TAG = 20261002
_TRAINING, _EVALUATION = 0, 1
_RETURNS, _AGENT = 0, 1


def load_calibration(processed_dir: Path = PROCESSED_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Daily mu and Sigma from Part I, as positional arrays in TICKERS order.

    mu is the `mean_daily` column of summary_statistics.csv and Sigma is
    covariance_daily.csv (sample covariance, ddof=1). clean.py estimates both
    from excess_returns.csv. The ticker labels in both files are checked
    against TICKERS and then dropped: everything downstream is positional, so
    a silent reorder would pair weights with the wrong assets.

    float_precision="round_trip" returns exactly the float64 values clean.py
    wrote. pandas' default parser is off by up to 1e-12 relative.
    """
    processed_dir = Path(processed_dir)
    stats = pd.read_csv(
        processed_dir / "summary_statistics.csv", index_col="ticker", float_precision="round_trip"
    )
    cov = pd.read_csv(processed_dir / "covariance_daily.csv", index_col=0, float_precision="round_trip")

    if list(stats.index) != TICKERS:
        raise ValueError(
            f"summary_statistics.csv rows are {list(stats.index)}, expected TICKERS order {TICKERS}"
        )
    if list(cov.index) != TICKERS or list(cov.columns) != TICKERS:
        raise ValueError(
            f"covariance_daily.csv is labeled {list(cov.index)} x {list(cov.columns)}, "
            f"expected TICKERS order {TICKERS} on both axes"
        )

    return _validate_moments(stats["mean_daily"].to_numpy(dtype=float), cov.to_numpy(dtype=float))


def simulate_returns(
    mu: np.ndarray, sigma: np.ndarray, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """n_days i.i.d. daily excess-return vectors drawn from N(mu, Sigma).

    Returns shape (n_days, n_assets), columns in the order of mu. This is the
    interface NEXT_STEPS_FOR_ALBERT.md specifies. With n_days >= 2 the output
    is what src.rewards.episodic_mean_variance_reward takes as `returns_batch`.

    Sigma is factored on every call. Inside a training loop use
    ReturnSimulator, which factors once; from the same generator state the two
    return the same draws.
    """
    mu, sigma = _validate_moments(mu, sigma)
    n_days = _check_count(n_days, "n_days", minimum=1)
    _check_rng(rng)
    shocks = rng.standard_normal((n_days, mu.shape[0]))
    return _apply_moments(mu, _cholesky_factor(sigma), shocks)


class ReturnSimulator:
    """The environment the agent interacts with.

    Holds mu and the Cholesky factor of Sigma privately and hands out only
    simulated returns. Python cannot enforce privacy, so the guarantee is
    structural: build the simulator in the experiment runner, pass only the
    object to the agent, and keep the agent module away from the loader, the
    benchmarks and the data files (tested in test_environment.py). Nothing
    stops a runner from passing mu or Sigma to the agent by hand; that is for
    code review.

    Give each simulator its own generator (SeedPlan.run_streams). The draws do
    not depend on which portfolio the agent picks, so with a dedicated
    generator the k-th episode's returns are the same whatever the agent does.
    Runs that differ only in grid resolution or learning rate then face
    identical return paths (common random numbers).

    Only Gaussian shocks are implemented. For the Part VI fat-tail check,
    override _standard_shocks with draws of zero mean and identity covariance,
    for example multivariate Student-t shocks rescaled by sqrt((nu - 2) / nu),
    which keeps the covariance at Sigma and every benchmark unchanged. Draw any
    extra randomness (such as the t distribution's chi-square mixing variable)
    from a second generator. Otherwise block sampling stops matching
    one-at-a-time sampling, and the Gaussian part of the shocks stops matching
    the baseline run.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator):
        mu, sigma = _validate_moments(mu, sigma)
        _check_rng(rng)
        self._mu = mu.copy()
        self._chol = _cholesky_factor(sigma)
        self._rng = rng
        self._days_simulated = 0
        self._episodes_simulated = 0

    @property
    def n_assets(self) -> int:
        return self._mu.shape[0]

    @property
    def days_simulated(self) -> int:
        """Simulated days drawn so far.

        With sample_episodes this counts days drawn in advance, which can exceed
        the days the agent actually used. The Part IV observation count should
        come from the training loop (episodes used times episode length).
        """
        return self._days_simulated

    @property
    def episodes_simulated(self) -> int:
        return self._episodes_simulated

    def sample_episode(self, episode_length: int) -> np.ndarray:
        """One episode of daily excess returns, shape (episode_length, n_assets)."""
        episode_length = _check_count(episode_length, "episode_length", minimum=MIN_EPISODE_LENGTH)
        shocks = self._standard_shocks((episode_length, self.n_assets))
        self._days_simulated += episode_length
        self._episodes_simulated += 1
        return _apply_moments(self._mu, self._chol, shocks)

    def sample_episodes(self, n_episodes: int, episode_length: int) -> np.ndarray:
        """n_episodes episodes at once, shape (n_episodes, episode_length, n_assets).

        Gives the same numbers as n_episodes consecutive
        sample_episode(episode_length) calls, generated in one block for speed.
        Memory is n_episodes * episode_length * n_assets * 8 bytes, so draw
        large budgets in chunks.
        """
        n_episodes = _check_count(n_episodes, "n_episodes", minimum=1)
        episode_length = _check_count(episode_length, "episode_length", minimum=MIN_EPISODE_LENGTH)
        shocks = self._standard_shocks((n_episodes, episode_length, self.n_assets))
        self._days_simulated += n_episodes * episode_length
        self._episodes_simulated += n_episodes
        return _apply_moments(self._mu, self._chol, shocks)

    def _standard_shocks(self, shape: tuple[int, ...]) -> np.ndarray:
        """Draws with zero mean and identity covariance across the last axis."""
        return self._rng.standard_normal(shape)


@dataclass(frozen=True)
class SeedPlan:
    """Every random stream in one experiment, derived from one master seed.

    Streams are numpy SeedSequence children, identified by spawn key (T is
    _PLAN_TAG):

        (T, 0, run, 0)   returns for training run `run`, for its ReturnSimulator
        (T, 0, run, 1)   the agent's own randomness in that run (exploration, tie-breaks)
        (T, 1)           evaluation returns, shared by every run

    Training and evaluation sit on different branches, so evaluation draws are
    never used to update Q-values (assignment Part IV). evaluation_rng() starts
    the evaluation stream from the beginning on every call, so every learned
    policy and every benchmark is scored on identical draws. A run's streams do
    not depend on n_runs, so adding seeds later leaves earlier runs unchanged.
    Record as_dict() with the results.
    """

    master_seed: int
    n_runs: int

    def __post_init__(self):
        # Store plain ints: numpy integers would make as_dict() fail json.dumps
        # at results-writing time, after training has already run.
        object.__setattr__(self, "master_seed", _check_count(self.master_seed, "master_seed", minimum=0))
        object.__setattr__(self, "n_runs", _check_count(self.n_runs, "n_runs", minimum=1))

    def run_streams(self, run: int) -> tuple[np.random.Generator, np.random.Generator]:
        """(returns_rng, agent_rng) for training run `run`."""
        run = _check_count(run, "run", minimum=0)
        if run >= self.n_runs:
            raise ValueError(f"run must be below n_runs={self.n_runs}, got {run}")
        return (
            self._generator((_TRAINING, run, _RETURNS)),
            self._generator((_TRAINING, run, _AGENT)),
        )

    def evaluation_rng(self) -> np.random.Generator:
        """A fresh generator at the start of the shared evaluation stream."""
        return self._generator((_EVALUATION,))

    def as_dict(self) -> dict:
        """JSON-serializable record of the seeding scheme, for results logs."""
        return {
            "master_seed": self.master_seed,
            "n_runs": self.n_runs,
            "bit_generator": "PCG64 via numpy.random.default_rng(SeedSequence)",
            "streams": {
                "training_returns": f"SeedSequence(master_seed, spawn_key=({_PLAN_TAG}, 0, run, 0))",
                "training_agent": f"SeedSequence(master_seed, spawn_key=({_PLAN_TAG}, 0, run, 1))",
                "evaluation_returns": f"SeedSequence(master_seed, spawn_key=({_PLAN_TAG}, 1))",
            },
            "numpy_version": np.__version__,
        }

    def _generator(self, key: tuple[int, ...]) -> np.random.Generator:
        seed_sequence = np.random.SeedSequence(self.master_seed, spawn_key=(_PLAN_TAG, *key))
        return np.random.default_rng(seed_sequence)


def _apply_moments(mu: np.ndarray, chol: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    # Rows of `shocks` have identity covariance, so rows of shocks @ chol.T have
    # covariance chol @ chol.T = Sigma. matmul broadcasts over leading axes.
    return mu + shocks @ chol.T


def _cholesky_factor(sigma: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L @ L.T == Sigma."""
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Sigma is not positive definite, so it is not a valid covariance matrix"
        ) from err


def _validate_moments(mu, sigma) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mu, (pd.Series, pd.DataFrame)) or isinstance(sigma, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "pass mu and Sigma as numpy arrays in TICKERS order (see load_calibration); "
            "converting pandas objects drops their labels, so a reordered input would go unnoticed"
        )
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if mu.ndim != 1 or mu.shape[0] == 0:
        raise ValueError(f"mu must be a non-empty 1-D array, got shape {mu.shape}")
    n_assets = mu.shape[0]
    if sigma.shape != (n_assets, n_assets):
        raise ValueError(f"Sigma must be {n_assets}x{n_assets} to match mu, got shape {sigma.shape}")
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
        raise ValueError("mu and Sigma must contain only finite values")
    # np.linalg.cholesky reads only the lower triangle, so it would accept an
    # asymmetric matrix without complaint. The tolerance admits rounding-level
    # asymmetry from shrinkage estimators and rejects anything larger.
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12 * np.abs(sigma).max()):
        raise ValueError("Sigma must be symmetric")
    return mu, sigma


def _check_count(value, name: str, minimum: int) -> int:
    # bool is a subclass of int; True as a day count is almost certainly a bug.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def _check_rng(rng) -> None:
    # This can only check the type. Whether the generator was seeded is up to
    # the caller, which is why SeedPlan exists.
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator from SeedPlan, got {type(rng).__name__}"
        )
