"""Part IV training runner: runs the agent through simulated episodes and
saves what evaluation needs.

One episode (the contract in src/agent.py):

    a = agent.select_action()
    w = grid[a]
    batch = sim.sample_episode(B)
    r = episodic_mean_variance_reward(w, batch, gamma)
    agent.update(a, r)

This file holds mu and Sigma only to build the simulator, and it does not
import the benchmarks, so nothing computed from mu, Sigma or the optimum can
reach the agent. Every run trains for a fixed episode budget with no early
stopping. Whether and when a run converged is judged afterwards from the logs,
by evaluation code, using quantities the agent can see.

Logs are compact, because a run can be tens of millions of episodes:

- every `log_every` episodes: the exploration rate for the next episode (read
  from the agent's `epsilon` property, so any agent must expose one), mean
  reward over the window, the largest |change in Q| in the window, the logged
  greedy action, how often each action was chosen in the window, and a
  Q-table snapshot;
- every change of the logged greedy action, with its episode number,
  starting with the initial greedy action at episode 0, so "greedy action
  unchanged for K episodes" can be checked exactly at any K.

The logged greedy action is the lowest-index argmax of the Q-values. It is
for logging only, never feeds back into training, and never calls the agent's
own greedy_action (which may draw from the agent's random stream). The
function-approximation path logs the agent's cached greedy instead (below).

If agent.update() returns the new Q(action), train.py keeps its own copy of the
Q-table and changes that one entry, instead of copying the whole table every
episode (REVIEW_HISTORY.md §5.3). At the 10% grid that copy is about 6 µs of
the roughly 91 µs per episode; the per-episode argmax below is about 36 µs.
The copy is checked against agent.q_values at every checkpoint. An update()
that returns None, as TabularQAgent's does until William's half of the fix
lands, gets the full copy as before.

Function-approximation agents (src/fa_agent.py, post hoc) take a separate
path, train_one_run_fa, chosen by the agent's updates_all_actions flag. One
update there moves every portfolio's Q-value, and computing all of them
(F @ theta) is expensive, so that path reads the agent's cached greedy and
its last_abs_dq every episode and the full Q-values only at checkpoints. The
tabular path below is unchanged. FA_INTEGRATION_NOTES.md lists every change.

Usage, from the repo root:

    python -m src.train --gamma 5 --episode-length 20 --episodes 200000 \\
        --master-seed 2026 --runs 2 --beta 2e-5 --out results/demo
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.agent import TabularQAgent
from src.environment import PROCESSED_DIR, ReturnSimulator, SeedPlan, load_calibration
from src.fa_agent import LinearQAgent
from src.features import quadratic_features, spline_additive_pairwise_features
from src.hashing import file_sha256
from src.rewards import episodic_mean_variance_reward

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILES = ("summary_statistics.csv", "covariance_daily.csv")
CODE_FILES = (
    "src/train.py",
    "src/agent.py",
    "src/environment.py",
    "src/rewards.py",
    "src/action_space/grid.py",
    "src/hashing.py",
    "src/config.py",
)
FA_CODE_FILES = ("src/fa_agent.py", "src/features.py")  # hashed only for function-approximation runs

# Feature maps a function-approximation run can name in TrainConfig.feature_map.
# Each builds F from the grid alone (never from mu, Sigma or data).
FEATURE_MAPS = {"quadratic": quadratic_features, "spline": spline_additive_pairwise_features}
AGENT_TYPES = ("tabular", "linear")
# The FA-only fields, with the values a tabular run must keep. config_record()
# leaves them out of a tabular manifest, so tabular manifests stay identical to
# the committed baseline-v1 ones.
FA_FIELD_DEFAULTS = {"agent": "tabular", "feature_map": None, "l2": 0.0, "greedy_refresh_every": 1}
# Q snapshots are checkpoints x actions float32 values, and the action counts
# are the same size again in int32. At the 10% grid (43,758 actions) with 26M
# episodes and log_every=10,000 that is 114M values each (455 MB, so about
# 910 MB for both); the 5% grid would be far larger. Refuse rather than run out
# of memory. Raised from 100M on 2026-10-27 for post-hoc run grid10-robustness.
MAX_SNAPSHOT_VALUES = 120_000_000


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines a training experiment.

    The fields without defaults must be stated explicitly. The team's chosen
    values for every field, including these, are in src/config.py.
    """

    gamma: float
    episode_length: int
    n_episodes: int
    master_seed: int
    n_runs: int
    beta: float
    increment: float = 0.2
    alpha_rule: str | float = "sample_average"  # "sample_average" means alpha = 1/N(a)
    epsilon0: float = 1.0
    q_init: float = 0.0
    discount: float = 0.0
    log_every: int = 10_000
    block_size: int = 10_000  # episodes drawn per simulator call, for speed only
    # Function-approximation extension (post hoc). The defaults mean "tabular".
    agent: str = "tabular"  # "tabular" (TabularQAgent) or "linear" (LinearQAgent)
    feature_map: str | None = None  # for "linear": a key of FEATURE_MAPS
    l2: float = 0.0  # ridge penalty, "linear" only
    greedy_refresh_every: int = 1  # "linear" only: updates between greedy recomputations

    def __post_init__(self):
        for name in ("episode_length", "n_episodes", "master_seed", "n_runs", "log_every", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if self.episode_length < 2:
            raise ValueError("episode_length must be at least 2 (the reward's variance needs two days)")
        for name in ("n_episodes", "n_runs", "log_every", "block_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("gamma", "beta", "epsilon0", "q_init", "discount", "increment"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.gamma < 0 or self.beta < 0:
            raise ValueError("gamma and beta must be nonnegative")
        if not 0.0 <= self.epsilon0 <= 1.0:
            raise ValueError("epsilon0 must be in [0, 1]")
        if not 0.0 <= self.discount < 1.0:
            raise ValueError("discount must be in [0, 1)")
        if isinstance(self.alpha_rule, str):
            if self.alpha_rule != "sample_average":
                raise ValueError("alpha_rule must be 'sample_average' or a constant in (0, 1]")
        elif isinstance(self.alpha_rule, bool) or not 0.0 < self.alpha_rule <= 1.0:
            raise ValueError("a constant alpha_rule must be in (0, 1]")
        if self.agent not in AGENT_TYPES:
            raise ValueError(f"agent must be one of {AGENT_TYPES}, got {self.agent!r}")
        if self.agent == "tabular":
            # Refuse FA settings on a tabular run rather than silently ignore them.
            for name, default in FA_FIELD_DEFAULTS.items():
                if getattr(self, name) != default:
                    raise ValueError(f"{name} applies only to agent='linear'")
        else:
            if self.feature_map not in FEATURE_MAPS:
                raise ValueError(f"feature_map must be one of {sorted(FEATURE_MAPS)}, got {self.feature_map!r}")
            if isinstance(self.alpha_rule, str):
                raise ValueError("agent='linear' needs a constant alpha_rule (alpha does not use visit counts)")
            if self.discount != 0.0 or self.q_init != 0.0:
                raise ValueError("agent='linear' has rho = 0 and theta starts at 0: discount and q_init must be 0")
            if isinstance(self.l2, bool) or not isinstance(self.l2, (int, float)) \
                    or not math.isfinite(self.l2) or self.l2 < 0:
                raise ValueError("l2 must be a finite, nonnegative number")
            refresh = self.greedy_refresh_every
            if isinstance(refresh, bool) or not isinstance(refresh, (int, np.integer)) or refresh < 1:
                raise ValueError("greedy_refresh_every must be a positive integer")
            # The run must end, and each checkpoint fall, right after a refresh.
            # Otherwise the last cached greedy can be stale rather than just
            # differing from the final policy because of ties or rounding.
            if self.n_episodes % refresh or self.log_every % refresh:
                raise ValueError("n_episodes and log_every must be multiples of greedy_refresh_every")


def config_record(config: TrainConfig) -> dict:
    """The config as written to manifest.json and each run's .npz. A tabular
    run leaves out the four FA fields, so its record matches the manifests
    written before they existed; run_experiments compares against this."""
    record = asdict(config)
    if config.agent == "tabular":
        for name in FA_FIELD_DEFAULTS:
            del record[name]
    return record


def build_features(config: TrainConfig, grid: np.ndarray) -> np.ndarray:
    """F for a function-approximation run, from the grid alone."""
    return FEATURE_MAPS[config.feature_map](grid)


def alpha_callable(rule: str | float) -> float | Callable[[int], float]:
    """Turn the JSON-friendly config value into what TabularQAgent expects.

    For "sample_average", n is the visit count INCLUDING the update being
    made, so the first update of an action uses alpha = 1/1 and replaces q_init
    with the first reward. n < 1 raises, because 1/0 on a numpy integer is a
    silent inf rather than an error.
    """
    if rule == "sample_average":
        def sample_average(n: int) -> float:
            if n < 1:
                raise ValueError(f"sample-average alpha needs a visit count >= 1, got {n}")
            return 1.0 / n

        return sample_average
    return float(rule)


def default_agent_factory(config: TrainConfig, n_actions: int, rng: np.random.Generator,
                          features: np.ndarray | None = None):
    """TabularQAgent, or LinearQAgent when config.agent == "linear". The
    linear agent needs the feature matrix, which run_experiment builds once
    from the grid and passes in; it never gets the grid itself."""
    if config.agent == "linear":
        if features is None or features.shape[0] != n_actions:
            raise ValueError("agent='linear' needs a feature matrix with one row per action")
        return LinearQAgent(
            features,
            alpha=float(config.alpha_rule),
            epsilon0=config.epsilon0,
            beta=config.beta,
            rng=rng,
            l2=config.l2,
            greedy_refresh_every=config.greedy_refresh_every,
        )
    return TabularQAgent(
        n_actions,
        alpha_rule=alpha_callable(config.alpha_rule),
        epsilon0=config.epsilon0,
        beta=config.beta,
        q_init=config.q_init,
        discount=config.discount,
        rng=rng,
    )


def train_one_run(config: TrainConfig, grid: np.ndarray, simulator: ReturnSimulator, agent) -> dict:
    """Run config.n_episodes episodes and return the compact logs."""
    if getattr(agent, "updates_all_actions", False):
        return train_one_run_fa(config, grid, simulator, agent)
    n_actions = grid.shape[0]
    n_checkpoints = -(-config.n_episodes // config.log_every)  # ceiling division

    checkpoint_episode = np.zeros(n_checkpoints, dtype=np.int64)
    checkpoint_next_epsilon = np.zeros(n_checkpoints)
    checkpoint_mean_reward = np.zeros(n_checkpoints)
    checkpoint_max_abs_dq = np.zeros(n_checkpoints)
    checkpoint_greedy = np.zeros(n_checkpoints, dtype=np.int64)
    checkpoint_action_counts = np.zeros((n_checkpoints, n_actions), dtype=np.int32)
    checkpoint_q = np.zeros((n_checkpoints, n_actions), dtype=np.float32)

    q = agent.q_values
    logged_greedy = int(np.argmax(q))
    greedy_change_episode, greedy_change_action = [0], [logged_greedy]
    window_rewards, window_max_dq, checkpoint = 0.0, 0.0, 0
    mirrored = False  # True once update() has returned a value
    window_counts = np.zeros(n_actions, dtype=np.int32)

    done = 0
    while done < config.n_episodes:
        n_block = min(config.block_size, config.n_episodes - done)
        block = simulator.sample_episodes(n_block, config.episode_length)
        for batch in block:
            action = int(agent.select_action())
            reward = float(episodic_mean_variance_reward(grid[action], batch, config.gamma))
            old_value = q[action]
            returned = agent.update(action, reward)
            done += 1

            if returned is None:
                q = agent.q_values  # full copy
            else:
                q[action] = returned  # q is train.py's own copy; update() changed only Q(action)
                mirrored = True
            if not np.isfinite(q[action]):
                raise FloatingPointError(
                    f"Q-value of action {action} became {q[action]} at episode {done}; "
                    "check the learning-rate rule"
                )
            window_max_dq = max(window_max_dq, abs(q[action] - old_value))
            window_rewards += reward
            window_counts[action] += 1

            greedy = int(np.argmax(q))
            if greedy != logged_greedy:
                logged_greedy = greedy
                greedy_change_episode.append(done)
                greedy_change_action.append(greedy)

            if done % config.log_every == 0 or done == config.n_episodes:
                if mirrored and not np.array_equal(q, agent.q_values):
                    raise RuntimeError(
                        f"train.py's copy of the Q-table differs from the agent's at episode {done}: "
                        "update() must return the new Q(action) and change no other entry"
                    )
                checkpoint_episode[checkpoint] = done
                # Read from the agent, after `done` selections: the rate the next episode uses.
                checkpoint_next_epsilon[checkpoint] = agent.epsilon
                checkpoint_mean_reward[checkpoint] = window_rewards / window_counts.sum()
                checkpoint_max_abs_dq[checkpoint] = window_max_dq
                checkpoint_greedy[checkpoint] = logged_greedy
                checkpoint_action_counts[checkpoint] = window_counts
                checkpoint_q[checkpoint] = q
                checkpoint += 1
                window_rewards, window_max_dq = 0.0, 0.0
                window_counts = np.zeros(n_actions, dtype=np.int32)

    return {
        "checkpoint_episode": checkpoint_episode,
        "checkpoint_next_epsilon": checkpoint_next_epsilon,
        "checkpoint_mean_reward": checkpoint_mean_reward,
        "checkpoint_max_abs_dq": checkpoint_max_abs_dq,
        "checkpoint_greedy": checkpoint_greedy,
        "checkpoint_action_counts": checkpoint_action_counts,
        "checkpoint_q": checkpoint_q,
        "greedy_change_episode": np.array(greedy_change_episode, dtype=np.int64),
        "greedy_change_action": np.array(greedy_change_action, dtype=np.int64),
        "final_q": q,
        "final_visit_counts": agent.visit_counts,
        "final_greedy": np.int64(logged_greedy),
    }


def train_one_run_fa(config: TrainConfig, grid: np.ndarray, simulator: ReturnSimulator, agent) -> dict:
    """The function-approximation version of train_one_run (post hoc).

    Same episodes, same random streams and the same log arrays, so
    evaluate.py reads the result unchanged. Three quantities are logged
    differently, because one update moves every portfolio's Q-value and
    scoring all of them each episode would dominate the run time:

    - the logged greedy is the agent's own cached greedy (float32 scores,
      random tie-break), read every episode. It changes only when the agent
      refreshes it, every greedy_refresh_every updates, so "greedy unchanged
      for K episodes" is exact to that resolution;
    - |change in Q| is the change in the chosen portfolio's Q-value, from the
      agent (last_abs_dq), the same definition as the tabular path. Other
      portfolios' Q-values also move and are not in this number;
    - the full Q-values (float64) are read only at checkpoints and at the end.
      final_q is therefore exact and current. final_greedy, the manifest and
      evaluation use its lowest-index argmax. The cached training history is
      preserved even when a tie or float32 rounding makes its last choice
      differ from this final policy.
    """
    n_actions = grid.shape[0]
    n_checkpoints = -(-config.n_episodes // config.log_every)

    checkpoint_episode = np.zeros(n_checkpoints, dtype=np.int64)
    checkpoint_next_epsilon = np.zeros(n_checkpoints)
    checkpoint_mean_reward = np.zeros(n_checkpoints)
    checkpoint_max_abs_dq = np.zeros(n_checkpoints)
    checkpoint_greedy = np.zeros(n_checkpoints, dtype=np.int64)
    checkpoint_action_counts = np.zeros((n_checkpoints, n_actions), dtype=np.int32)
    checkpoint_q = np.zeros((n_checkpoints, n_actions), dtype=np.float32)

    logged_greedy = int(agent.greedy_action)
    greedy_change_episode, greedy_change_action = [0], [logged_greedy]
    window_rewards, window_max_dq, checkpoint = 0.0, 0.0, 0
    window_counts = np.zeros(n_actions, dtype=np.int32)

    done = 0
    while done < config.n_episodes:
        n_block = min(config.block_size, config.n_episodes - done)
        block = simulator.sample_episodes(n_block, config.episode_length)
        for batch in block:
            action = int(agent.select_action())
            reward = float(episodic_mean_variance_reward(grid[action], batch, config.gamma))
            agent.update(action, reward)  # raises FloatingPointError itself if theta diverges
            done += 1

            window_max_dq = max(window_max_dq, agent.last_abs_dq)
            window_rewards += reward
            window_counts[action] += 1

            greedy = int(agent.greedy_action)
            if greedy != logged_greedy:
                logged_greedy = greedy
                greedy_change_episode.append(done)
                greedy_change_action.append(greedy)

            if done % config.log_every == 0 or done == config.n_episodes:
                checkpoint_episode[checkpoint] = done
                checkpoint_next_epsilon[checkpoint] = agent.epsilon
                checkpoint_mean_reward[checkpoint] = window_rewards / window_counts.sum()
                checkpoint_max_abs_dq[checkpoint] = window_max_dq
                checkpoint_greedy[checkpoint] = logged_greedy
                checkpoint_action_counts[checkpoint] = window_counts
                checkpoint_q[checkpoint] = agent.q_values
                checkpoint += 1
                window_rewards, window_max_dq = 0.0, 0.0
                window_counts = np.zeros(n_actions, dtype=np.int32)

    # The run ends on a refresh (TrainConfig enforces it), so the logged greedy
    # is current. It can still differ from the float64 argmax that evaluate.py
    # uses when two portfolios tie to float32 rounding; anything larger is a bug.
    final_q = agent.q_values
    tolerance = 1e-6 * max(float(np.abs(final_q).max()), 1e-300)
    if final_q.max() - final_q[logged_greedy] > tolerance:
        raise RuntimeError(
            f"logged greedy {logged_greedy} is not the best final Q-value ({int(np.argmax(final_q))}); "
            "the greedy refresh and the end of the run are out of step"
        )

    return {
        "checkpoint_episode": checkpoint_episode,
        "checkpoint_next_epsilon": checkpoint_next_epsilon,
        "checkpoint_mean_reward": checkpoint_mean_reward,
        "checkpoint_max_abs_dq": checkpoint_max_abs_dq,
        "checkpoint_greedy": checkpoint_greedy,
        "checkpoint_action_counts": checkpoint_action_counts,
        "checkpoint_q": checkpoint_q,
        "greedy_change_episode": np.array(greedy_change_episode, dtype=np.int64),
        "greedy_change_action": np.array(greedy_change_action, dtype=np.int64),
        "final_q": final_q,
        "final_visit_counts": agent.visit_counts,
        "final_greedy": np.int64(np.argmax(final_q)),
        "final_theta": agent.theta,
    }


def run_experiment(
    config: TrainConfig,
    out_dir: Path,
    agent_factory=default_agent_factory,
    processed_dir: Path = PROCESSED_DIR,
) -> dict:
    """Train config.n_runs independent runs into a new, empty out_dir: one
    .npz per run plus manifest.json. The manifest is written before the first
    run and rewritten after each one, so a crash still leaves a record of the
    config. Returns the final manifest."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; use a new folder so old and new runs cannot mix")
    out_dir.mkdir(parents=True, exist_ok=True)

    mu, sigma = load_calibration(processed_dir)
    grid = generate_weight_grid(mu.shape[0], increment=config.increment)
    n_checkpoints = -(-config.n_episodes // config.log_every)
    if n_checkpoints * grid.shape[0] > MAX_SNAPSHOT_VALUES:
        raise MemoryError(
            f"Q snapshots would hold {n_checkpoints * grid.shape[0]:,} values; raise log_every"
        )
    plan = SeedPlan(config.master_seed, config.n_runs)
    features = None
    if config.agent == "linear":
        features = build_features(config, grid)  # once per experiment, from the grid only
    if agent_factory is default_agent_factory:
        agent_factory = partial(default_agent_factory, features=features)
    code_files = CODE_FILES + (FA_CODE_FILES if config.agent == "linear" else ())

    manifest = {
        "config": config_record(config),
        "seed_plan": plan.as_dict(),
        "n_actions": int(grid.shape[0]),
        "grid_sha256": grid_fingerprint(grid),
        "calibration_sha256": _hashes(Path(processed_dir), CALIBRATION_FILES),
        "code_sha256": _hashes(REPO_ROOT, code_files),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _version("scipy"),
        "pandas": _version("pandas"),
        "log_schema": {
            "checkpoint_*": "one row per log_every episodes; the last row covers the final, possibly shorter, window",
            "checkpoint_next_epsilon": "agent.epsilon after this many episodes, the rate for the next episode; "
            "epsilon0 * exp(-beta * episodes done) for TabularQAgent",
            "greedy_change_*": "episode number and new logged greedy action at every change; first entry is episode 0",
            "logged_greedy": "lowest-index argmax of the Q-values; logging only",
            "observations": "episodes x episode_length simulated days",
        },
        "runs": [],
    }
    if config.agent == "linear":
        manifest["n_features"] = int(features.shape[1])
        manifest["log_schema"].update({
            "logged_greedy": "the agent's cached greedy (float32 scores, random tie-break), refreshed every "
            "greedy_refresh_every updates; logging only",
            "checkpoint_max_abs_dq": "largest |change in Q(chosen action)| in the window; other actions' Q also move",
            "final_theta": "the learned parameters, one per feature column",
            "final_greedy": "lowest-index argmax of final_q (float64), also used by evaluation; "
            "may differ from the cached training greedy on ties or float32 rounding",
        })
    _write_manifest(out_dir, manifest)

    for run in range(config.n_runs):
        returns_rng, agent_rng = plan.run_streams(run)
        simulator = ReturnSimulator(mu, sigma, returns_rng)
        agent = agent_factory(config, grid.shape[0], agent_rng)
        if bool(getattr(agent, "updates_all_actions", False)) != (config.agent == "linear"):
            raise TypeError(
                f"config.agent is {config.agent!r} but the factory built {type(agent).__name__}; "
                "the manifest would record the wrong agent"
            )
        manifest["agent_class"] = f"{type(agent).__module__}.{type(agent).__qualname__}"

        started = time.perf_counter()
        logs = train_one_run(config, grid, simulator, agent)
        seconds = time.perf_counter() - started

        path = out_dir / f"run_{run:03d}.npz"
        np.savez_compressed(path, config_json=np.array(json.dumps(config_record(config))), **logs)
        final = int(logs["final_greedy"])
        manifest["runs"].append(
            {
                "run": run,
                "file": path.name,
                "seconds": round(seconds, 2),
                "episodes": config.n_episodes,
                "simulated_days": config.n_episodes * config.episode_length,
                "final_greedy_action": final,
                "final_greedy_weights": grid[final].tolist(),
                "greedy_changes": int(logs["greedy_change_episode"].size - 1),
            }
        )
        _write_manifest(out_dir, manifest)
        print(f"run {run}: {seconds:.1f}s, final greedy {grid[final]}")

    return manifest


def grid_fingerprint(grid: np.ndarray) -> str:
    """SHA-256 of the action grid's values, so evaluation can confirm that
    action index a still means the same portfolio."""
    return hashlib.sha256(np.ascontiguousarray(grid, dtype=np.float64).tobytes()).hexdigest()


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _hashes(root: Path, names) -> dict:
    """Line-ending-normalized digests (src/hashing.py), so the same file gives
    the same digest on Windows and on the Mac."""
    return {name: file_sha256(root / name) for name in names}


def _version(module: str) -> str | None:
    """A library's version, or None when it is not installed."""
    try:
        return __import__(module).__version__
    except ImportError:
        return None


def _git_commit() -> str | None:
    """HEAD of the repo this file lives in, with -dirty if tracked files differ.
    code_sha256 in the manifest identifies the exact code either way."""
    try:
        def git(*args):
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
            ).stdout.strip()

        dirty = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--episode-length", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--increment", type=float, default=0.2)
    parser.add_argument("--alpha", default="sample_average", help="'sample_average' or a constant, e.g. 0.01")
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    alpha = args.alpha if args.alpha == "sample_average" else float(args.alpha)
    config = TrainConfig(
        gamma=args.gamma,
        episode_length=args.episode_length,
        n_episodes=args.episodes,
        master_seed=args.master_seed,
        n_runs=args.runs,
        beta=args.beta,
        increment=args.increment,
        alpha_rule=alpha,
        log_every=args.log_every,
    )
    run_experiment(config, args.out)


if __name__ == "__main__":
    main()
