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

- every `log_every` episodes: the exploration rate for the next episode, mean
  reward over the window, the largest |change in Q| in the window, the logged
  greedy action, how often each action was chosen in the window, and a
  Q-table snapshot;
- every change of the logged greedy action, with its episode number,
  starting with the initial greedy action at episode 0, so "greedy action
  unchanged for K episodes" can be checked exactly at any K.

The logged greedy action is the lowest-index argmax of the Q-values. It is
for logging only, never feeds back into training, and never calls the agent's
own greedy_action (which may draw from the agent's random stream).

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
from pathlib import Path
from typing import Callable

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.agent import TabularQAgent
from src.environment import PROCESSED_DIR, ReturnSimulator, SeedPlan, load_calibration
from src.rewards import episodic_mean_variance_reward

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILES = ("summary_statistics.csv", "covariance_daily.csv")
CODE_FILES = (
    "src/train.py",
    "src/agent.py",
    "src/environment.py",
    "src/rewards.py",
    "src/action_space/grid.py",
)
# Q snapshots are checkpoints x actions float32 values. At the 10% grid
# (43,758 actions) with 13M episodes and log_every=10,000 that is 57M values
# (227 MB); the 5% grid would be far larger. Refuse rather than run out of memory.
MAX_SNAPSHOT_VALUES = 100_000_000


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines a training experiment.

    The fields without defaults are open team decisions (γ, episode length,
    budget, seeds, exploration decay) and must be stated explicitly. The
    defaults are the values proposed in REVIEW_NOTES_FOR_WILLIAM.md §3.
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


def default_agent_factory(config: TrainConfig, n_actions: int, rng: np.random.Generator):
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
    window_counts = np.zeros(n_actions, dtype=np.int32)

    done = 0
    while done < config.n_episodes:
        n_block = min(config.block_size, config.n_episodes - done)
        block = simulator.sample_episodes(n_block, config.episode_length)
        for batch in block:
            action = int(agent.select_action())
            reward = float(episodic_mean_variance_reward(grid[action], batch, config.gamma))
            agent.update(action, reward)
            done += 1

            new_q = agent.q_values
            if not np.isfinite(new_q[action]):
                raise FloatingPointError(
                    f"Q-value of action {action} became {new_q[action]} at episode {done}; "
                    "check the learning-rate rule"
                )
            window_max_dq = max(window_max_dq, abs(new_q[action] - q[action]))
            q = new_q
            window_rewards += reward
            window_counts[action] += 1

            greedy = int(np.argmax(q))
            if greedy != logged_greedy:
                logged_greedy = greedy
                greedy_change_episode.append(done)
                greedy_change_action.append(greedy)

            if done % config.log_every == 0 or done == config.n_episodes:
                checkpoint_episode[checkpoint] = done
                # The agent's schedule after `done` selections: the rate the next episode uses.
                checkpoint_next_epsilon[checkpoint] = config.epsilon0 * np.exp(-config.beta * done)
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

    manifest = {
        "config": asdict(config),
        "seed_plan": plan.as_dict(),
        "n_actions": int(grid.shape[0]),
        "calibration_sha256": _hashes(Path(processed_dir), CALIBRATION_FILES),
        "code_sha256": _hashes(REPO_ROOT, CODE_FILES),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "log_schema": {
            "checkpoint_*": "one row per log_every episodes; the last row covers the final, possibly shorter, window",
            "checkpoint_next_epsilon": "epsilon0 * exp(-beta * episodes done): the rate for the next episode",
            "greedy_change_*": "episode number and new logged greedy action at every change; first entry is episode 0",
            "logged_greedy": "lowest-index argmax of the Q-values; logging only",
            "observations": "episodes x episode_length simulated days",
        },
        "runs": [],
    }
    _write_manifest(out_dir, manifest)

    for run in range(config.n_runs):
        returns_rng, agent_rng = plan.run_streams(run)
        simulator = ReturnSimulator(mu, sigma, returns_rng)
        agent = agent_factory(config, grid.shape[0], agent_rng)
        manifest["agent_class"] = f"{type(agent).__module__}.{type(agent).__qualname__}"

        started = time.perf_counter()
        logs = train_one_run(config, grid, simulator, agent)
        seconds = time.perf_counter() - started

        path = out_dir / f"run_{run:03d}.npz"
        np.savez_compressed(path, config_json=np.array(json.dumps(asdict(config))), **logs)
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


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _hashes(root: Path, names) -> dict:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


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
