"""Run a registered transaction-cost comparison on the previous-target grid."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.config import MASTER_SEED, beta_for
from src.environment import ReturnSimulator, SeedPlan, load_calibration
from src.evaluate import objective
from src.rewards import episodic_mean_variance_reward
from src.train import _git_commit, grid_fingerprint


@dataclass(frozen=True)
class Settings:
    reference_cost_bps: float
    discount: float = 0.95
    alpha: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.reference_cost_bps) or self.reference_cost_bps < 0:
            raise ValueError("reference_cost_bps must be finite and nonnegative")
        if not 0 <= self.discount < 1:
            raise ValueError("discount must be in [0, 1)")
        if self.alpha is not None and not 0 < self.alpha <= 1:
            raise ValueError("alpha must be null (1/N) or in (0, 1]")


def switching_cost(grid, reference_cost_bps):
    """Cost as a fraction of wealth. A single 20-point transfer sets the scale."""
    norms = np.einsum("ij,ij->i", grid, grid)
    distance_squared = norms[:, None] + norms[None, :] - 2 * grid @ grid.T
    distance_squared = np.maximum(distance_squared, 0)
    np.fill_diagonal(distance_squared, 0)
    return reference_cost_bps * 1e-4 * distance_squared / (2 * 0.2**2)


def optimal_policy(expected_reward, daily_cost, discount):
    """Bellman solution. Choosing action a sets the next state to a."""
    value = np.zeros_like(expected_reward)
    while True:
        scores = (expected_reward + discount * value)[None, :] - daily_cost
        updated = scores.max(axis=1)
        if np.max(np.abs(updated - value)) <= 1e-12 * (1 - discount):
            return scores.argmax(axis=1)
        value = updated


def train_policy(grid, daily_cost, settings, simulator, rng, start_rng, steps, block_length, gamma,
                 restart_every=100):
    n = len(grid)
    q = np.zeros((n, n))
    visits = np.zeros((n, n), dtype=np.int64)
    beta = beta_for(steps)
    # Compare complete greedy policy snapshots on the grid.
    previous = q.argmax(axis=1)
    last_change = 0
    state = 0
    for offset in range(0, steps, 1000):
        batches = simulator.sample_episodes(min(1000, steps - offset), block_length)
        for step, returns in enumerate(batches, offset):
            if step % restart_every == 0:
                state = int(start_rng.integers(n))
            if rng.random() < np.exp(-beta * step):
                action = int(rng.integers(n))
            else:
                action = int(q[state].argmax())
            reward = episodic_mean_variance_reward(grid[action], returns, gamma) - daily_cost[state, action]
            visits[state, action] += 1
            alpha = settings.alpha if settings.alpha is not None else 1 / visits[state, action]
            q[state, action] += alpha * (reward + settings.discount * q[action].max() - q[state, action])
            state = action
        policy = q.argmax(axis=1)
        if np.any(policy != previous):
            last_change = offset + len(batches)
        previous = policy
    return policy, {
        "visited_pairs": int(np.count_nonzero(visits)),
        "possible_pairs": int(visits.size),
        "checkpoint_stable_through_end": steps - last_change >= steps // 10,
        "last_checkpoint_policy_change": last_change,
    }


def evaluate_policy(policy, starts, paths, grid, daily_cost, expected_reward, gamma, discount):
    """Score each policy on identical paths, including the first rebalance."""
    state = starts.copy()
    realized = np.zeros(len(starts))
    expected = np.zeros(len(starts))
    turnover = np.zeros(len(starts))
    trades = np.zeros(len(starts), dtype=int)
    for step, returns in enumerate(paths):
        action = policy[state]
        weights = grid[action]
        daily = np.einsum("pba,pa->pb", returns, weights)
        cost = daily_cost[state, action]
        weight = discount**step
        realized += weight * (daily.mean(axis=1) - gamma / 2 * daily.var(axis=1, ddof=1) - cost)
        expected += weight * (expected_reward[action] - cost)
        turnover += 0.5 * np.abs(weights - grid[state]).sum(axis=1)
        trades += action != state
        state = action
    annualize = 252e4 / sum(discount**step for step in range(len(paths)))
    return {
        "realized_net_ce_bps_per_year": float(realized.mean() * annualize),
        "expected_net_ce_bps_per_year": float(expected.mean() * annualize),
        "one_way_turnover_per_path": float(turnover.mean()),
        "trades_per_path": float(trades.mean()),
    }


def run_comparison(variants, steps, seeds, evaluation_paths):
    gamma, block_length, horizon, evaluation_discount = 5.0, 20, 100, 0.95
    mu, sigma = load_calibration()
    grid = generate_weight_grid(len(mu), 0.2)
    expected_reward = objective(grid, mu, sigma, gamma)
    plan = SeedPlan(MASTER_SEED, seeds)
    eval_rng = plan.evaluation_rng()
    starts = eval_rng.integers(len(grid), size=evaluation_paths)
    paths = ReturnSimulator(mu, sigma, eval_rng).sample_episodes(
        horizon * evaluation_paths, block_length
    ).reshape(horizon, evaluation_paths, block_length, len(mu))
    calibration_hash = hashlib.sha256(np.concatenate([mu, sigma.ravel()]).tobytes()).hexdigest()
    report = {
        "git_commit": _git_commit(), "grid_sha256": grid_fingerprint(grid),
        "numpy_version": np.__version__, "seed_plan": plan.as_dict(),
        "calibration_sha256": calibration_hash,
        "gamma": gamma, "block_length": block_length,
        "steps_per_seed": steps, "evaluation_paths": evaluation_paths,
        "evaluation_blocks": horizon, "evaluation_discount": evaluation_discount,
        "exploration_beta": beta_for(steps), "training_restart_blocks": 100,
        "evaluation_start_indices": starts.tolist(), "variants": {},
    }
    for name, settings in variants.items():
        cost = switching_cost(grid, settings.reference_cost_bps) / block_length
        oracle = optimal_policy(expected_reward, cost, evaluation_discount)
        myopic = optimal_policy(expected_reward, cost, 0)
        no_cost = np.full(len(grid), expected_reward.argmax())

        def score(policy):
            return evaluate_policy(policy, starts, paths, grid, cost, expected_reward, gamma, evaluation_discount)

        benchmarks = {"grid_oracle": score(oracle), "myopic": score(myopic), "ignore_cost": score(no_cost)}
        results = []
        for seed in range(seeds):
            # Recreate the plan for each seed and variant so child spawning is paired.
            returns_rng, action_rng = SeedPlan(MASTER_SEED, seeds).run_streams(seed)
            start_rng = action_rng.spawn(1)[0]
            policy, coverage = train_policy(
                grid, cost, settings, ReturnSimulator(mu, sigma, returns_rng), action_rng, start_rng,
                steps, block_length, gamma,
            )
            print(f'{name}: completed seed {seed + 1}/{seeds}', flush=True)
            metrics = score(policy)
            gap = benchmarks["grid_oracle"]["expected_net_ce_bps_per_year"] - metrics["expected_net_ce_bps_per_year"]
            results.append({"seed": seed, **coverage, **metrics, "oracle_gap_bps_per_year": gap,
                            "policy": policy.tolist()})
        report["variants"][name] = {"settings": asdict(settings), "benchmarks": benchmarks, "seeds": results}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=13_000_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--evaluation-paths", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.evaluation_paths) < 1:
        parser.error("steps, seeds and evaluation paths must be positive")
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless using --dry-run")
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} already exists")
    variants = {name: Settings(**settings) for name, settings in json.loads(args.config.read_text()).items()}
    if not variants:
        parser.error("config must define at least one variant")
    if args.dry_run:
        print(json.dumps({name: asdict(settings) for name, settings in variants.items()}, indent=2))
        return
    report = run_comparison(variants, args.steps, args.seeds, args.evaluation_paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
