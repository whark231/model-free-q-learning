"""Part IV-V evaluation: how close did the agent get, and how fast?

Reads one experiment folder written by src/train.py (run_*.npz plus
manifest.json) and compares each seed's learned policy with the benchmarks.
This is the only place where mu, Sigma and the optimum meet the agent's
results. The assignment reserves those objects "for benchmark construction
and evaluation", and nothing computed here feeds back into training.

What it computes, mapped to the assignment:

- Part V weights. A seed's learned policy is the greedy action of its final
  Q-table, held with exploration off and no learning. Its time-averaged
  weights w_bar over the evaluation period are that action's weights, because
  the policy picks the same portfolio every period. Each seed is compared with
  the continuous long-only optimum w* and the best grid portfolio: grouped bar
  chart, absolute difference by asset, L1 and Euclidean distance.
- Part V economic performance. For each seed's learned portfolio, the
  continuous optimum, the best grid portfolio and equal weight: annualized
  mean and volatility and the daily objective U(w) = w'mu - (gamma/2) w'Sigma w.
  Exact values come from mu and Sigma. Realized values come from a separate
  evaluation sample (SeedPlan's evaluation stream, never used in training),
  with every portfolio scored on the same draws so paired differences have
  small standard errors.
- Gaps. The primary measure is the certainty-equivalent loss, the return per
  year the investor gives up by holding one portfolio instead of another:
  252 * (U_best - U) in basis points. It keeps its sign and meaning at any
  gamma. Relative gaps, (U_best - U) / |U_best|, are reported alongside but
  flagged as unreliable once U_best is near or below zero (from gamma of
  about 23 on this calibration).
- Part V convergence speed. Learning curve, expected cumulative regret against
  the best grid action, and the observations needed to come within the
  near-optimal line and stay there.
- Part IV convergence. When each agent-observable rule is met, and how often
  across seeds. K, the |change in Q| tolerance and the near-optimal line are
  inputs here; the team's fixed values are in src/config.py.
- Q-value fit (added 2026-11-04 for the function-approximation comparison,
  post hoc). Did the agent learn the whole reward landscape, or only its top
  pick? The learned Q-values are compared with each portfolio's true expected
  reward, which is U(w) exactly because the reward's variance uses ddof = 1:
  Pearson and Spearman correlation, top-10 recall, and how far the learned
  portfolio's Q-value overstates its true value.

Usage, from the repo root:

    python -m src.evaluate results/demo --eval-episodes 10000 \\
        --converge-k 200000 --dq-tol 1e-5 --near-optimal-bps 25
    python -m src.evaluate --compare results/g25 results/g20 --labels 25% 20% ...
"""

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src.action_space.grid import generate_weight_grid  # noqa: E402
from src.benchmarks.mean_variance import (  # noqa: E402
    best_discrete_weights,
    constrained_optimal_weights,
    equal_weights,
)
from src.data.universe import TICKERS  # noqa: E402
from src.environment import PROCESSED_DIR, ReturnSimulator, SeedPlan, load_calibration  # noqa: E402
from src.hashing import file_sha256, matches  # noqa: E402
from src.train import REPO_ROOT, _git_commit, grid_fingerprint  # noqa: E402

TRADING_DAYS = 252
BPS = 1e4
BENCHMARKS = ("continuous", "best_discrete", "equal_weight")
# Below this |U_best| (daily), a relative gap divides by almost nothing.
RELATIVE_GAP_FLOOR = 1e-5
TOP_K = 10  # portfolios compared by the top-k recall
# Evaluation-side code whose digest goes into summary.json; training code is
# already in each manifest's code_sha256 (RUN_ISSUES_2026-10-22.md Finding 7).
EVAL_CODE_FILES = ("src/evaluate.py", "src/config.py", "src/run_experiments.py", "src/hashing.py",
                   "src/benchmarks/mean_variance.py", "src/environment.py", "src/action_space/grid.py")
RUN_KEYS = (
    "checkpoint_episode", "checkpoint_mean_reward", "checkpoint_max_abs_dq", "checkpoint_greedy",
    "checkpoint_action_counts", "checkpoint_next_epsilon", "greedy_change_episode",
    "greedy_change_action", "final_q", "final_visit_counts",
)

# Validated reference palette (light). Portfolios keep one color everywhere;
# grid resolutions use two other slots so they never read as a portfolio.
COLORS = {"learned": "#2a78d6", "continuous": "#eb6834", "best_discrete": "#1baf7a", "equal_weight": "#eda100"}
RESOLUTION_COLORS = ("#6250d6", "#e87ba4")
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LABELS = {
    "learned": "Learned (median seed)",
    "continuous": "Continuous optimum",
    "best_discrete": "Best on the grid",
    "equal_weight": "Equal weight",
}


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings. None of these change what the agent learned."""

    n_eval_episodes: int
    converge_k: int  # greedy action unchanged for this many episodes
    dq_tol: float  # largest |change in Q| per full window below this, for good
    near_optimal_bps: float  # certainty-equivalent loss vs best grid action, bps per year

    def __post_init__(self):
        if self.n_eval_episodes < 2 or self.converge_k < 1:
            raise ValueError("n_eval_episodes must be at least 2 and converge_k at least 1")
        if not (self.dq_tol > 0 and self.near_optimal_bps >= 0):
            raise ValueError("dq_tol must be > 0 and near_optimal_bps >= 0")


# ---------------------------------------------------------------------------
# Pure calculations (each tested on hand-made inputs)
# ---------------------------------------------------------------------------

def objective(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gamma: float):
    """U(w) per day: a float for one portfolio (n,), an array for (k, n)."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        return float(w @ mu - 0.5 * gamma * (w @ sigma @ w))
    return w @ mu - 0.5 * gamma * np.einsum("ij,jk,ik->i", w, sigma, w)


def ce_loss_bps(u_reference, u):
    """Certainty-equivalent loss in basis points per year: 252 * (U_ref - U)."""
    return TRADING_DAYS * BPS * (np.asarray(u_reference) - np.asarray(u))


def relative_gap(u_reference: float, u: float) -> float | None:
    """(U_ref - U) / |U_ref|, or None when |U_ref| is too close to zero."""
    if abs(u_reference) < RELATIVE_GAP_FLOOR:
        return None
    return float((u_reference - u) / abs(u_reference))


def exact_stats(w, mu, sigma, gamma) -> dict:
    return {
        "mean_annual": float(TRADING_DAYS * w @ mu),
        "vol_annual": float(np.sqrt(TRADING_DAYS * w @ sigma @ w)),
        "objective_daily": objective(w, mu, sigma, gamma),
    }


def episode_rewards(w: np.ndarray, episodes: np.ndarray, gamma: float) -> np.ndarray:
    """The training reward (src/rewards.py) on every episode of shape (E, B, n)."""
    daily = episodes @ w
    return daily.mean(axis=1) - 0.5 * gamma * daily.var(axis=1, ddof=1)


def realized_stats(w: np.ndarray, episodes: np.ndarray, gamma: float, reference_rewards: np.ndarray) -> dict:
    """Performance on the evaluation sample, and the paired difference in
    mean reward against a reference portfolio scored on the same draws."""
    daily = (episodes @ w).ravel()
    rewards = episode_rewards(w, episodes, gamma)
    diff = reference_rewards - rewards
    n = rewards.size
    return {
        "mean_annual": float(TRADING_DAYS * daily.mean()),
        "vol_annual": float(np.sqrt(TRADING_DAYS) * daily.std(ddof=1)),
        "objective_daily": float(rewards.mean()),
        "objective_se": float(rewards.std(ddof=1) / np.sqrt(n)),
        "shortfall_vs_best_discrete_daily": float(diff.mean()),
        "shortfall_vs_best_discrete_se": float(diff.std(ddof=1) / np.sqrt(n)),
    }


def expected_regret(action_counts: np.ndarray, action_values: np.ndarray, best_value: float) -> np.ndarray:
    """Cumulative expected regret after each checkpoint window, in daily
    objective units summed over episodes: sum of U(best) - U(chosen)."""
    return np.cumsum(action_counts @ (best_value - action_values))


def q_fit(q_hat, true_q, k: int = TOP_K) -> dict:
    """How well learned Q-values track the true expected rewards, across the
    portfolios passed in. Correlations are None when either side is constant
    or there are fewer than two portfolios. Top-k recall is the share of the
    true top k that are also in the learned top k; ties in q_hat go to the
    lowest index, as for the learned portfolio."""
    q_hat, true_q = np.asarray(q_hat, dtype=float), np.asarray(true_q, dtype=float)
    n = q_hat.size

    def corr(a, b):
        if n < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    k = min(k, n)
    top_hat = np.argsort(-q_hat, kind="stable")[:k]
    top_true = np.argsort(-true_q, kind="stable")[:k]
    return {
        "n_portfolios": int(n),
        "pearson": corr(q_hat, true_q),
        "spearman": corr(rankdata(q_hat), rankdata(true_q)),
        "top_k": int(k),
        "top_k_recall": len(set(top_hat.tolist()) & set(top_true.tolist())) / k if k else None,
    }


def first_greedy_stable(change_episode, change_action, n_episodes: int, k: int):
    """First episode at which the greedy action has been unchanged for k
    consecutive episodes, and that action. (None, None) if never."""
    ends = list(change_episode[1:]) + [n_episodes]
    for start, end, action in zip(change_episode, ends, change_action):
        if end - start >= k:
            return int(start + k), int(action)
    return None, None


def greedy_stable_through_end(change_episode, n_episodes: int, k: int):
    """Episode at which the FINAL greedy action had been held for k episodes,
    if its run to the end of training lasted at least k. Otherwise None."""
    last = int(change_episode[-1])
    return last + k if n_episodes - last >= k else None


def full_windows(checkpoint_episode, log_every: int) -> np.ndarray:
    """Mask of checkpoint windows that cover a full log_every episodes. A short
    final window has few updates and would look artificially calm."""
    sizes = np.diff(np.concatenate([[0], np.asarray(checkpoint_episode)]))
    return sizes == log_every


def first_for_good(checkpoint_episode, ok, mask=None):
    """First checkpoint from which `ok` holds at every later checkpoint
    (considering only checkpoints where mask is True). None if it never does."""
    episodes = np.asarray(checkpoint_episode)
    ok = np.asarray(ok, dtype=bool)
    if mask is not None:
        episodes, ok = episodes[mask], ok[mask]
    if ok.size == 0 or not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return int(episodes[bad[-1] + 1 if bad.size else 0])


def settled_from(checkpoint_episode, checkpoint_greedy, final: int):
    """First checkpoint from which the final action is greedy at every later
    checkpoint, or None if it is not greedy at the final checkpoint.
    Post-hoc and descriptive (RUN_ISSUES_2026-10-22.md Finding 1)."""
    if len(checkpoint_episode) == 0 or checkpoint_greedy[-1] != final:
        return None
    bad = np.flatnonzero(np.asarray(checkpoint_greedy) != final)
    return int(checkpoint_episode[bad[-1] + 1]) if bad.size else int(checkpoint_episode[0])


def near_optimal_every_episode(change_episode, change_action, loss_by_action, threshold: float):
    """First episode after which every greedy action, brief switches included,
    stays within `threshold` bps/yr of the best. None if never. The every-episode
    counterpart of near_optimal_for_good, which is read at checkpoints only
    (RUN_ISSUES_2026-10-22.md Finding 2)."""
    over = np.flatnonzero(np.asarray(loss_by_action)[change_action] > threshold)
    if over.size == 0:
        return int(change_episode[0])
    i = over[-1]
    return int(change_episode[i + 1]) if i + 1 < len(change_episode) else None


def provenance() -> dict:
    """What produced this evaluation: commit, evaluation-code digests and the
    library versions that can change tables or figures."""
    import matplotlib as mpl
    import scipy
    try:
        import pandas
        pandas_version = pandas.__version__
    except ImportError:  # pandas is not needed to evaluate
        pandas_version = None
    return {
        "git_commit": _git_commit(),
        "code_sha256": {name: file_sha256(REPO_ROOT / name) for name in EVAL_CODE_FILES},
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": mpl.__version__, "pandas": pandas_version,
    }


def _spread(values) -> dict:
    values = np.array([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()), "n": int(values.size)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(run_dir: Path, processed_dir: Path, grid: np.ndarray | None = None, allow_partial=False) -> dict:
    """Read the manifest and refuse anything that would make the comparison
    describe a different experiment from the one that ran."""
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["calibration_sha256"].items():
        if not matches(Path(processed_dir) / name, digest):  # either line ending (src/hashing.py)
            raise ValueError(f"{name} has changed since training (sha256 mismatch); re-train or restore the data")
    if grid is not None:
        if grid.shape[0] != manifest["n_actions"]:
            raise ValueError("grid size does not match the training manifest")
        if "grid_sha256" in manifest and manifest["grid_sha256"] != grid_fingerprint(grid):
            raise ValueError("the action grid differs from the one used in training")
    if not manifest["runs"]:
        raise ValueError(f"{run_dir} has no completed runs")
    if len(manifest["runs"]) < manifest["config"]["n_runs"] and not allow_partial:
        raise ValueError(
            f"only {len(manifest['runs'])} of {manifest['config']['n_runs']} runs finished; "
            "pass allow_partial=True to evaluate them anyway"
        )
    return manifest


def load_run(run_dir: Path, entry: dict) -> dict:
    """Only the arrays evaluation uses; the Q snapshots are skipped."""
    with np.load(Path(run_dir) / entry["file"]) as data:
        return {key: data[key] for key in RUN_KEYS}


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment(run_dir: Path, config: EvalConfig, out_dir: Path | None = None,
                        processed_dir: Path = PROCESSED_DIR, overwrite: bool = False,
                        allow_partial: bool = False) -> dict:
    summary, _ = _evaluate(run_dir, config, out_dir, processed_dir, overwrite, allow_partial)
    return summary


def _evaluate(run_dir, config, out_dir, processed_dir, overwrite, allow_partial):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "evaluation"
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass overwrite=True or choose another folder")
    out_dir.mkdir(parents=True, exist_ok=True)

    mu, sigma = load_calibration(processed_dir)
    peek = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    train = peek["config"]
    gamma, episode_length, log_every = train["gamma"], train["episode_length"], train["log_every"]
    grid = generate_weight_grid(mu.shape[0], increment=train["increment"])
    manifest = load_manifest(run_dir, processed_dir, grid, allow_partial)

    # Benchmarks (evaluation only).
    w_cont = constrained_optimal_weights(mu, sigma, gamma)
    w_disc = best_discrete_weights(mu, sigma, gamma, grid=grid)
    action_values = objective(grid, mu, sigma, gamma)
    u_cont, u_disc = objective(w_cont, mu, sigma, gamma), float(action_values.max())
    benchmarks = {"continuous": w_cont, "best_discrete": w_disc, "equal_weight": equal_weights(mu.shape[0])}
    relative_unreliable = abs(u_disc) < RELATIVE_GAP_FLOOR or u_disc <= 0

    # One evaluation sample, shared by every seed and every portfolio.
    plan = SeedPlan(manifest["seed_plan"]["master_seed"], manifest["seed_plan"]["n_runs"])
    eval_episodes = ReturnSimulator(mu, sigma, plan.evaluation_rng()).sample_episodes(
        config.n_eval_episodes, episode_length
    )
    disc_rewards = episode_rewards(w_disc, eval_episodes, gamma)

    seeds, curves = [], {"obs": None, "reward": [], "loss_bps": [], "regret": []}
    for entry in manifest["runs"]:
        logs = load_run(run_dir, entry)
        action = int(np.argmax(logs["final_q"]))  # final policy: float64 scores, lowest index on ties
        w = grid[action]
        u = float(action_values[action])
        greedy_loss = ce_loss_bps(u_disc, action_values[logs["checkpoint_greedy"]])
        stable_at, stable_action = first_greedy_stable(
            logs["greedy_change_episode"], logs["greedy_change_action"], train["n_episodes"], config.converge_k
        )
        mask = full_windows(logs["checkpoint_episode"], log_every)
        visited = logs["final_visit_counts"] > 0
        through_end = greedy_stable_through_end(
            logs["greedy_change_episode"], train["n_episodes"], config.converge_k
        ) if logs["greedy_change_action"][-1] == action else None
        loss_by_action = ce_loss_bps(u_disc, action_values)
        settled = settled_from(logs["checkpoint_episode"], logs["checkpoint_greedy"], action)
        record = {
            "run": entry["run"],
            "learned_action": action,
            "learned_weights": w.tolist(),
            "visits_of_learned_action": int(logs["final_visit_counts"][action]),
            "final_epsilon": float(logs["checkpoint_next_epsilon"][-1]),
            "exact": exact_stats(w, mu, sigma, gamma),
            "evaluation_sample": realized_stats(w, eval_episodes, gamma, disc_rewards),
            "ce_loss_vs_best_discrete_bps": float(ce_loss_bps(u_disc, u)),
            "ce_loss_vs_continuous_bps": float(ce_loss_bps(u_cont, u)),
            "relative_learning_gap": relative_gap(u_disc, u),
            "is_best_discrete": bool(np.array_equal(w, w_disc)),
            "abs_diff_vs_continuous": np.abs(w - w_cont).tolist(),
            "abs_diff_vs_best_discrete": np.abs(w - w_disc).tolist(),
            "l1_to_continuous": float(np.abs(w - w_cont).sum()),
            "euclidean_to_continuous": float(np.linalg.norm(w - w_cont)),
            "l1_to_best_discrete": float(np.abs(w - w_disc).sum()),
            "final_cumulative_regret": None,
            "greedy_stable_episode": stable_at,
            "greedy_stable_action_is_final": None if stable_at is None else stable_action == action,
            "greedy_stable_through_end_episode": through_end,
            "dq_below_tol_for_good_episode": first_for_good(
                logs["checkpoint_episode"], logs["checkpoint_max_abs_dq"] < config.dq_tol, mask
            ),
            "near_optimal_for_good_episode": first_for_good(
                logs["checkpoint_episode"], greedy_loss <= config.near_optimal_bps
            ),
            # Post hoc and descriptive, beside the fixed rules above (RUN_ISSUES_2026-10-22.md
            # Findings 1 and 2): when the final action became greedy at every later
            # checkpoint, and the near-optimal rule read at every episode.
            "settled_episode": settled,
            "near_optimal_every_episode_episode": near_optimal_every_episode(
                logs["greedy_change_episode"], logs["greedy_change_action"], loss_by_action, config.near_optimal_bps
            ),
            # Never-visited portfolios keep their initial Q in the tabular
            # agent, so the fit is also reported over visited portfolios only.
            "q_fit_all": q_fit(logs["final_q"], action_values),
            "q_fit_visited": q_fit(logs["final_q"][visited], action_values[visited]),
            "optimism_of_learned_bps": float(TRADING_DAYS * BPS * (logs["final_q"][action] - u)),
        }
        regret = expected_regret(logs["checkpoint_action_counts"], action_values, u_disc)
        record["final_cumulative_regret"] = float(regret[-1])
        for key in ("greedy_stable_episode", "greedy_stable_through_end_episode",
                    "dq_below_tol_for_good_episode", "near_optimal_for_good_episode",
                    "settled_episode", "near_optimal_every_episode_episode"):
            value = record[key]
            record[key.removesuffix("_episode") + "_observations"] = None if value is None else value * episode_length
        seeds.append(record)
        curves["obs"] = np.concatenate([[0], logs["checkpoint_episode"]]) * episode_length
        curves["reward"].append(logs["checkpoint_mean_reward"])
        curves["loss_bps"].append(greedy_loss)
        curves["regret"].append(np.concatenate([[0.0], regret]))

    losses = [s["ce_loss_vs_best_discrete_bps"] for s in seeds]
    # The "median seed" is the upper median (the 6th of 10); _spread's median
    # averages the two middle seeds, so it need not be any seed's value.
    median_seed = seeds[int(np.argsort(losses)[len(losses) // 2])]
    convergence_rules = {
        "greedy_stable": f"greedy action unchanged for {config.converge_k} episodes (first time)",
        "greedy_stable_through_end": f"the final greedy action held for at least {config.converge_k} episodes",
        "dq_below_tol_for_good": f"max |change in Q| below {config.dq_tol} in every full {log_every}-episode window from then on",
        "near_optimal_for_good": f"greedy action within {config.near_optimal_bps} bps/yr (CE) of the best grid action at every {log_every}-episode checkpoint from then on",
    }
    # Post hoc, descriptive: reported beside the fixed rules, never in place of them.
    descriptive_rules = {
        "settled": f"the final greedy action is greedy at every {log_every}-episode checkpoint from then on",
        "near_optimal_every_episode": f"every greedy action, brief switches included, within {config.near_optimal_bps} "
                                      "bps/yr (CE) of the best grid action from then on",
    }
    summary = {
        "run_dir": str(run_dir),
        "provenance": provenance(),
        "eval_config": asdict(config),
        "training_config": train,
        "n_seeds": len(seeds),
        "n_actions": int(grid.shape[0]),
        "tickers": TICKERS,
        "benchmarks": {
            name: {
                "weights": w.tolist(),
                "exact": exact_stats(w, mu, sigma, gamma),
                "evaluation_sample": realized_stats(w, eval_episodes, gamma, disc_rewards),
            }
            for name, w in benchmarks.items()
        },
        "discretization_gap": {
            "ce_loss_bps": float(ce_loss_bps(u_cont, u_disc)),
            "relative": relative_gap(u_cont, u_disc),
            "l1_continuous_vs_best_discrete": float(np.abs(w_cont - w_disc).sum()),
        },
        "learned": {
            "ce_loss_vs_best_discrete_bps": _spread(losses),
            "ce_loss_vs_continuous_bps": _spread([s["ce_loss_vs_continuous_bps"] for s in seeds]),
            "relative_learning_gap": _spread([s["relative_learning_gap"] for s in seeds]),
            "relative_gap_unreliable": bool(relative_unreliable),
            "objective_daily": _spread([s["exact"]["objective_daily"] for s in seeds]),
            "mean_annual": _spread([s["exact"]["mean_annual"] for s in seeds]),
            "vol_annual": _spread([s["exact"]["vol_annual"] for s in seeds]),
            "l1_to_continuous": _spread([s["l1_to_continuous"] for s in seeds]),
            "euclidean_to_continuous": _spread([s["euclidean_to_continuous"] for s in seeds]),
            "share_exactly_best_discrete": float(np.mean([s["is_best_discrete"] for s in seeds])),
            "median_seed_run": median_seed["run"],
            "mean_of_seed_weights_descriptive_only": np.mean([s["learned_weights"] for s in seeds], axis=0).tolist(),
        },
        "q_fit": {
            **{
                population: {
                    stat: _spread([s[f"q_fit_{population}"][stat] for s in seeds])
                    for stat in ("pearson", "spearman", "top_k_recall", "top_k", "n_portfolios")
                }
                for population in ("all", "visited")
            },
            "optimism_of_learned_bps": _spread([s["optimism_of_learned_bps"] for s in seeds]),
        },
        "convergence": {
            "rules": convergence_rules,
            **{
                f"{rule}": {
                    "share_of_seeds": float(np.mean([s[f"{rule}_episode"] is not None for s in seeds])),
                    "observations": _spread([s[f"{rule}_observations"] for s in seeds]),
                }
                for rule in convergence_rules
            },
            "descriptive_rules": descriptive_rules,
            **{
                rule: {
                    "share_of_seeds": float(np.mean([s[f"{rule}_episode"] is not None for s in seeds])),
                    "observations": _spread([s[f"{rule}_observations"] for s in seeds]),
                }
                for rule in descriptive_rules
            },
        },
        "per_seed": seeds,
        "notes": {
            "primary_gap": "certainty-equivalent loss, 252 * (U_best - U) in basis points per year",
            "relative_gap": "(U_best - U) / |U_best|; None when |U_best| < 1e-5 per day",
            "evaluation_sample": f"{config.n_eval_episodes} episodes x {episode_length} days from SeedPlan's evaluation stream; exact values are primary, realized values a check",
            "mean_of_seed_weights": "average of the seeds' weights; not a portfolio any seed holds, so never compared with the benchmarks",
            "regret_units": "daily objective units summed over episodes; divide by the episode count and "
                            "multiply by 252 * 1e4 for the average loss per episode in bps/yr",
            "total_gap": "ce_loss_vs_continuous_bps: discretization gap plus learning gap, per seed",
            "median_seed": "the weights chart's median seed is the upper median (6th of 10 by learning gap); "
                           "summary medians average the two middle seeds",
            "observations": "simulated days = episodes x B; greedy_stable_through_end includes its K-episode "
                            "waiting window",
            "q_fit": f"final Q-values vs the true expected reward U(w) of every portfolio (all) or of those the "
            f"agent tried (visited); top-{TOP_K} recall is the share of the true top {TOP_K} in the learned top "
            f"{TOP_K}; optimism is 252 * (Qhat - U) of the learned portfolio in bps/yr. Evaluation only",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    _write_tables(out_dir, summary)
    _plot_weights(out_dir / "weights_grouped_bar.png", summary, median_seed)
    _plot_learning(out_dir / "learning_curve.png", curves, u_disc, u_cont, config.near_optimal_bps)
    _plot_regret(out_dir / "cumulative_regret.png", curves)
    _plot_loss_by_seed(out_dir / "ce_loss_by_seed.png", seeds, config.near_optimal_bps)
    return summary, curves


def compare_experiments(run_dirs, labels, config: EvalConfig, out_dir: Path,
                        processed_dir: Path = PROCESSED_DIR, overwrite: bool = False) -> dict:
    """Part V and Part IV: convergence under two or more grid resolutions.
    Refuses unless the experiments differ only in the grid increment."""
    if len(run_dirs) > len(RESOLUTION_COLORS):  # checked before any evaluation work (audit F-L4)
        raise ValueError(f"at most {len(RESOLUTION_COLORS)} grids per comparison plot (audit F-L4)")
    configs = [json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["config"] for d in run_dirs]
    base = {k: v for k, v in configs[0].items() if k != "increment"}
    for label, cfg in zip(labels, configs):
        other = {k: v for k, v in cfg.items() if k != "increment"}
        if other != base:
            diff = sorted(k for k in base if base[k] != other.get(k))
            raise ValueError(f"experiment {label} differs in more than the grid: {diff}")

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass overwrite=True or choose another folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for run_dir, label in zip(run_dirs, labels):
        summary, c = _evaluate(run_dir, config, out_dir / f"eval_{label}", processed_dir, overwrite, False)
        curves[label] = (c["obs"][1:], np.median(np.array(c["loss_bps"]), axis=0),
                         summary["discretization_gap"]["ce_loss_bps"])
        conv = summary["convergence"]
        results[label] = {
            "n_actions": summary["n_actions"],
            "discretization_gap_ce_bps": summary["discretization_gap"]["ce_loss_bps"],
            "ce_loss_vs_best_discrete_bps": summary["learned"]["ce_loss_vs_best_discrete_bps"],
            "ce_loss_vs_continuous_bps": summary["learned"]["ce_loss_vs_continuous_bps"],
            "share_exactly_best_discrete": summary["learned"]["share_exactly_best_discrete"],
            "q_fit": summary["q_fit"],
            **{rule: conv[rule] for rule in conv["rules"]},
            **{rule: conv[rule] for rule in conv["descriptive_rules"]},
        }
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    _plot_comparison(out_dir / "convergence_by_resolution.png", curves, config.near_optimal_bps)
    return results


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def _write_tables(out_dir: Path, summary: dict) -> None:
    seeds = summary["per_seed"]
    with open(out_dir / "weights.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF on every machine (audit M2)
        writer.writerow(["ticker", *BENCHMARKS, *(f"learned_run_{s['run']}" for s in seeds),
                         *(f"absdiff_vs_continuous_run_{s['run']}" for s in seeds),
                         *(f"absdiff_vs_best_discrete_run_{s['run']}" for s in seeds)])
        for i, ticker in enumerate(TICKERS):
            writer.writerow([
                ticker,
                *(f"{summary['benchmarks'][b]['weights'][i]:.4f}" for b in BENCHMARKS),
                *(f"{s['learned_weights'][i]:.4f}" for s in seeds),
                *(f"{s['abs_diff_vs_continuous'][i]:.4f}" for s in seeds),
                *(f"{s['abs_diff_vs_best_discrete'][i]:.4f}" for s in seeds),
            ])
    rows = [(name, summary["benchmarks"][name]) for name in BENCHMARKS]
    rows += [(f"learned_run_{s['run']}", s) for s in seeds]
    with open(out_dir / "performance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF on every machine (audit M2)
        writer.writerow(["portfolio", "mean_annual", "vol_annual", "objective_daily", "ce_loss_vs_best_discrete_bps",
                         "realized_objective_daily", "realized_objective_se",
                         "realized_shortfall_vs_best_discrete_daily", "realized_shortfall_se"])
        u_disc = summary["benchmarks"]["best_discrete"]["exact"]["objective_daily"]
        for name, row in rows:
            exact, realized = row["exact"], row["evaluation_sample"]
            writer.writerow([
                name, f"{exact['mean_annual']:.6f}", f"{exact['vol_annual']:.6f}", f"{exact['objective_daily']:.6e}",
                f"{float(ce_loss_bps(u_disc, exact['objective_daily'])):.2f}",
                f"{realized['objective_daily']:.6e}", f"{realized['objective_se']:.2e}",
                f"{realized['shortfall_vs_best_discrete_daily']:.3e}", f"{realized['shortfall_vs_best_discrete_se']:.2e}",
            ])
    keys = ["run", "learned_action", "visits_of_learned_action", "final_epsilon", "ce_loss_vs_best_discrete_bps",
            "ce_loss_vs_continuous_bps", "relative_learning_gap", "is_best_discrete", "l1_to_continuous", "euclidean_to_continuous",
            "l1_to_best_discrete", "final_cumulative_regret", "greedy_stable_observations",
            "greedy_stable_through_end_observations", "dq_below_tol_for_good_observations",
            "near_optimal_for_good_observations", "settled_observations",
            "near_optimal_every_episode_observations"]
    fit_keys = [(population, stat) for population in ("all", "visited")
                for stat in ("pearson", "spearman", "top_k_recall", "n_portfolios")]
    with open(out_dir / "per_seed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF on every machine (audit M2)
        writer.writerow(keys + [f"q_{stat}_{population}" for population, stat in fit_keys] + ["optimism_of_learned_bps"])
        for s in seeds:
            writer.writerow([s[k] for k in keys] + [s[f"q_fit_{population}"][stat] for population, stat in fit_keys]
                            + [s["optimism_of_learned_bps"]])


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width=9.0, height=4.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _clip_top(ax, curves_2d, threshold):
    """Early learning produces huge losses that flatten the interesting part.
    Clip the y-axis to the second half of training and say so."""
    later = np.asarray(curves_2d)[:, np.asarray(curves_2d).shape[1] // 2:]
    top = max(4 * threshold, 1.5 * float(np.max(later)), 1.0)
    ax.set_ylim(0, top)
    ax.text(0.99, 0.97, "y-axis clipped; early values run higher", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MUTED)


def _plot_weights(path, summary, median_seed):
    fig, ax = _figure()
    x = np.arange(len(TICKERS))
    width = 0.26
    series = [("learned", median_seed["learned_weights"]),
              ("continuous", summary["benchmarks"]["continuous"]["weights"]),
              ("best_discrete", summary["benchmarks"]["best_discrete"]["weights"])]
    for i, (name, weights) in enumerate(series):
        ax.bar(x + (i - 1) * width, weights, width * 0.92, color=COLORS[name], label=LABELS[name])
    seed_weights = np.array([s["learned_weights"] for s in summary["per_seed"]])
    for row in seed_weights:
        ax.scatter(x - width, row, s=14, color=INK_2, zorder=3, linewidths=0)
    ax.scatter([], [], s=14, color=INK_2, label="Each seed's learned weight")
    ax.set_xticks(x, TICKERS)
    ax.set_ylim(0, 1.0)
    _style(ax, f"Portfolio weights (median seed = run {median_seed['run']})", "", "Weight")
    ax.legend(frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_learning(path, curves, u_disc, u_cont, threshold):
    obs = curves["obs"][1:] / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    rewards = np.array([_moving_average(r, 5) for r in curves["reward"]])
    for row in rewards:
        axes[0].plot(obs, row, color=MUTED, linewidth=0.8, alpha=0.6)
    axes[0].plot(obs, np.median(rewards, axis=0), color=COLORS["learned"], linewidth=2, label="Median across seeds")
    axes[0].axhline(u_disc, color=COLORS["best_discrete"], linewidth=1.5, linestyle="--", label="Best on the grid, U")
    axes[0].axhline(u_cont, color=COLORS["continuous"], linewidth=1.5, linestyle=":", label="Continuous optimum, U")
    _style(axes[0], "Learning curve (5-window moving average)", "Simulated days (millions)", "Mean reward (daily objective)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")

    losses = np.array(curves["loss_bps"])
    for row in losses:
        axes[1].plot(obs, row, color=MUTED, linewidth=0.8, alpha=0.6)
    axes[1].plot(obs, np.median(losses, axis=0), color=COLORS["learned"], linewidth=2, label="Median across seeds")
    axes[1].axhline(threshold, color=INK_2, linewidth=1, linestyle="--", label=f"Near-optimal line ({threshold:g} bps/yr)")
    _style(axes[1], "Loss of the current greedy portfolio", "Simulated days (millions)", "CE loss vs best on grid (bps/yr)")
    _clip_top(axes[1], losses, threshold)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_regret(path, curves):
    fig, ax = _figure()
    obs = curves["obs"] / 1e6
    regrets = np.array(curves["regret"])
    for row in regrets:
        ax.plot(obs, row, color=MUTED, linewidth=0.8, alpha=0.6)
    ax.plot(obs, np.median(regrets, axis=0), color=COLORS["learned"], linewidth=2, label="Median across seeds")
    _style(ax, "Cumulative expected regret vs. best on the grid", "Simulated days (millions)",
           "Sum over episodes of U(best) − U(chosen)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_loss_by_seed(path, seeds, threshold):
    losses = np.array([s["ce_loss_vs_best_discrete_bps"] for s in seeds])
    runs = [s["run"] for s in seeds]
    order = np.argsort(losses)
    fig, ax = _figure(7, 4.2)
    ax.bar(np.arange(losses.size), losses[order], 0.7, color=COLORS["learned"])
    for i, value in enumerate(losses[order]):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.axhline(threshold, color=INK_2, linewidth=1, linestyle="--")
    ax.text(losses.size - 0.5, threshold, f" near-optimal line ({threshold:g})", va="bottom", ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(losses.size), [f"run {runs[i]}" for i in order], fontsize=8)
    _style(ax, "Final loss of the learned portfolio, by run (sorted)", "", "CE loss vs best on grid (bps/yr)")
    ax.set_ylim(0, max(1.2 * float(losses.max()), 2 * threshold, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_comparison(path, curves, threshold):
    """Solid: CE loss against the best portfolio on that grid (learning gap).
    Dashed: the same curve plus that grid's discretization gap, i.e. the loss
    against the continuous optimum (total gap), so grids can be compared on
    one reference (audit F-M6)."""
    if len(curves) > len(RESOLUTION_COLORS):
        raise ValueError(f"at most {len(RESOLUTION_COLORS)} grids per comparison plot (audit F-L4)")
    fig, ax = _figure()
    for (label, (obs, loss, disc)), hue in zip(curves.items(), RESOLUTION_COLORS):
        ax.plot(obs / 1e6, loss, color=hue, linewidth=2, label=f"{label} grid, vs best on that grid")
        ax.plot(obs / 1e6, loss + disc, color=hue, linewidth=1.5, linestyle=(0, (4, 2)),
                label=f"{label} grid, vs continuous optimum")
    ax.axhline(threshold, color=INK_2, linewidth=1, linestyle="--",
               label=f"Near-optimal line ({threshold:g} bps/yr, for the solid lines)")
    _style(ax, "Convergence by grid resolution (median across seeds)", "Simulated days (millions)",
           "CE loss (bps/yr)")
    _clip_top(ax, [loss + disc for _, loss, disc in curves.values()], threshold)
    ax.legend(frameon=False, labelcolor=INK_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(values.size)])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # absent under Jupyter or a redirected stdout
        sys.stdout.reconfigure(encoding="utf-8")  # γ in output; Windows consoles default to cp1252 (audit M1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--compare", nargs="+", type=Path, help="two or more run folders differing only in the grid")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--converge-k", type=int, required=True)
    parser.add_argument("--dq-tol", type=float, required=True)
    parser.add_argument("--near-optimal-bps", type=float, required=True)
    args = parser.parse_args()
    config = EvalConfig(args.eval_episodes, args.converge_k, args.dq_tol, args.near_optimal_bps)
    if args.compare:
        labels = args.labels or [p.name for p in args.compare]
        out = args.out or Path("results/comparison")
        print(json.dumps(compare_experiments(args.compare, labels, config, out, overwrite=args.overwrite), indent=2))
    for run_dir in args.run_dirs:
        summary = evaluate_experiment(run_dir, config, args.out, overwrite=args.overwrite)
        loss = summary["learned"]["ce_loss_vs_best_discrete_bps"]
        print(f"{run_dir}: CE loss vs best on grid, median {loss['median']:.1f} bps/yr "
              f"(range {loss['min']:.1f} to {loss['max']:.1f}); discretization "
              f"{summary['discretization_gap']['ce_loss_bps']:.1f} bps/yr")


if __name__ == "__main__":
    main()
