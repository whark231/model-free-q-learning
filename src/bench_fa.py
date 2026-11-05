"""Timing benchmark for the function-approximation agent (post hoc, Stage D)

In plain terms: how long does LinearQAgent itself take per 100,000 episodes,
for each feature model, grid and greedy refresh interval? Rewards are fake
(Gaussian noise), so this measures the agent only: no return simulation, no
reward computation, no train.py logging. For a real run, also time the
tabular runner on the same machine in the same session (--with-runner) and
compare: machine load changes absolute timings by a factor of 3 or more.

Each case runs a fixed number of episodes chosen so it finishes in seconds,
then scales to 100,000. Refreshing the greedy every episode on the 10% grid
is slow enough that it is timed on fewer episodes.

Usage, from the repo root:

    python -m src.bench_fa
"""

import argparse
import time

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.fa_agent import LinearQAgent
from src.features import quadratic_features, spline_additive_pairwise_features

MODELS = {"quadratic": quadratic_features, "spline": spline_additive_pairwise_features}


def time_agent(F: np.ndarray, refresh: int, n_episodes: int, seed: int = 0) -> float:
    """Seconds per 100,000 episodes of select_action + update on fake rewards.
    alpha is a tenth of the stability bound so theta stays finite on noise."""
    alpha = 0.1 * 2.0 / float((F.astype(float) ** 2).sum(axis=1).max())
    agent = LinearQAgent(F, alpha=alpha, epsilon0=1.0, beta=np.log(100) / 6_500_000,
                         rng=np.random.default_rng(seed), greedy_refresh_every=refresh)
    rewards = np.random.default_rng(seed + 1).normal(2e-4, 1.5e-3, size=n_episodes)
    start = time.perf_counter()
    for r in rewards:
        agent.update(agent.select_action(), float(r))
    return (time.perf_counter() - start) / n_episodes * 100_000


def time_runner(n_episodes: int) -> None:
    """Seconds per 100,000 episodes of the real train.py loop (simulated
    returns, rewards, logging) for the tabular agent and both FA models at
    greedy_refresh_every = 100, B = 20, on the 20% and 10% grids."""
    from src.environment import ReturnSimulator, SeedPlan, load_calibration
    from src.train import TrainConfig, build_features, default_agent_factory, train_one_run

    mu, sigma = load_calibration()
    print(f"{'grid':<5} {'runner':<10} {'s per 100k':>11} {'min per 13M':>12}")
    for increment in (0.2, 0.1):
        grid = generate_weight_grid(9, increment)
        for name in ("tabular", "quadratic", "spline"):
            fa = {} if name == "tabular" else dict(agent="linear", feature_map=name, greedy_refresh_every=100,
                                                   alpha_rule=0.01 if name == "quadratic" else 5e-4)
            config = TrainConfig(gamma=5.0, episode_length=20, n_episodes=n_episodes, master_seed=1, n_runs=1,
                                 beta=np.log(100) / 6_500_000, increment=increment, log_every=10_000, **fa)
            returns_rng, agent_rng = SeedPlan(1, 1).run_streams(0)
            features = build_features(config, grid) if fa else None
            agent = default_agent_factory(config, grid.shape[0], agent_rng, features=features)
            start = time.perf_counter()
            train_one_run(config, grid, ReturnSimulator(mu, sigma, returns_rng), agent)
            per_100k = (time.perf_counter() - start) / n_episodes * 100_000
            print(f"{increment:<5.0%} {name:<10} {per_100k:>11.2f} {per_100k * 130 / 60:>12.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20_000, help="episodes per case (fewer for the slowest)")
    parser.add_argument("--with-runner", action="store_true",
                        help="also time the full train.py loop (simulator, reward, logging) per model")
    args = parser.parse_args()
    if args.with_runner:
        time_runner(args.episodes)
    print(f"{'grid':<5} {'model':<10} {'refresh':>7} {'s per 100k':>11} {'min per 13M':>12}")
    for increment in (0.2, 0.1):
        W = generate_weight_grid(9, increment)
        for name, build in MODELS.items():
            F = build(W)
            for refresh in (1, 100, 1000):
                slow = increment == 0.1 and refresh == 1
                n = max(args.episodes // 10, 500) if slow else args.episodes
                per_100k = time_agent(F, refresh, n)
                print(f"{increment:<5.0%} {name:<10} {refresh:>7} {per_100k:>11.2f} {per_100k * 130 / 60:>12.1f}",
                      flush=True)


if __name__ == "__main__":
    main()
