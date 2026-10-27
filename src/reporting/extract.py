"""Turn one run's results folder into a compact, committed data file.

A "run" is one results folder made by `python -m src.run_experiments
--results-dir <folder>` with one set of settings, holding one or more
experiments (for example the 2026-10-22 baseline: four experiments in
results/). This module reads each experiment's manifest and training logs
(run_*.npz), the calibration, and the folder's settings.json, and writes
reports/<run_id>/data.json.

Why a separate data file: the training logs are about 100 MB per run and stay
local (.gitignore), so only the machine that ran a run can read them. The
data file is under ~2 MB and committed, so any later report can compare
against this run without the logs. Extract each run once, on the machine that
ran it, right after it finishes.

Evaluation-side code: it uses mu, Sigma and the benchmark portfolios, which
the agent never sees. Nothing here is imported by src/agent.py or src/train.py.

Run from the repo root:
    python -m src.reporting.extract <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.benchmarks.mean_variance import constrained_optimal_weights
from src.data.universe import TICKERS
from src.environment import PROCESSED_DIR, load_calibration
from src.evaluate import (exact_stats, first_for_good, first_greedy_stable, full_windows, greedy_stable_through_end,
                          objective)
from src.train import grid_fingerprint

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REGISTRY = REPORTS / "registry.json"
BPS = 252 * 1e4            # daily objective difference -> bps per year
MAX_POINTS = 650           # time-series points kept per curve (binned beyond this)
SCHEMA_VERSION = 2   # 2: cross-seed medians computed before binning


# ---------------------------------------------------------------------------
# Small, separately tested pieces
# ---------------------------------------------------------------------------

def reward_sd(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gamma: float, episode_length: int) -> float:
    """Exact standard deviation of one episode's reward for portfolio w.

    The reward is the realized mean minus (gamma/2) times the ddof=1 realized
    variance of B i.i.d. normal daily returns with mean m and variance s^2.
    The sample mean is N(m, s^2/B), the sample variance is s^2 chi2_{B-1}/(B-1),
    and the two are independent, so
        Var(reward) = s^2/B + (gamma/2)^2 * 2 s^4 / (B - 1).
    """
    s2 = float(w @ sigma @ w)
    B = episode_length
    return float(np.sqrt(s2 / B + (gamma / 2) ** 2 * 2 * s2 ** 2 / (B - 1)))


def tries_to_separate(sd_best: float, gap_daily: float) -> float:
    """Tries of EACH of two portfolios before their sample means differ by two
    standard errors, when their rewards come from different episodes
    (unpaired): 2 * sqrt(2 sd^2 / n) = gap  ->  n = 8 sd^2 / gap^2."""
    if gap_daily <= 0:
        return float("inf")
    return float(8 * sd_best ** 2 / gap_daily ** 2)


def settled_from(ckpt_episode: np.ndarray, ckpt_greedy: np.ndarray, final: int) -> int:
    """First checkpoint from which the final action is greedy at every later
    checkpoint. Post-hoc and descriptive (RUN_ISSUES_2026-10-22.md Finding 1)."""
    bad = np.flatnonzero(ckpt_greedy != final)
    return int(ckpt_episode[bad[-1] + 1]) if bad.size else int(ckpt_episode[0])


def excursions(ch_ep, ch_act, final: int, after: int, n_episodes: int, loss_by_action) -> list[tuple[int, int, float]]:
    """Distinct excursions after episode `after`: maximal stretches during which
    the greedy action is not the final one. Returns (start, length, worst loss)."""
    ends = np.append(np.asarray(ch_ep)[1:], n_episodes)
    out, start, worst = [], None, 0.0
    for e0, e1, a in zip(ch_ep, ends, ch_act):
        if e0 <= after:
            continue
        if a != final:
            if start is None:
                start, worst = int(e0), 0.0
            worst = max(worst, float(loss_by_action[a]))
        elif start is not None:
            out.append((start, int(e0) - start, worst))
            start = None
    if start is not None:
        out.append((start, int(n_episodes) - start, worst))
    return out


def near_optimal_every_episode(ch_ep, ch_act, loss_by_action, threshold: float):
    """First episode after which every greedy action, brief switches included,
    stays within `threshold` bps/yr of the best. None if never."""
    over = np.flatnonzero(np.asarray(loss_by_action)[ch_act] > threshold)
    if over.size == 0:
        return int(ch_ep[0])
    i = over[-1]
    return int(ch_ep[i + 1]) if i + 1 < len(ch_ep) else None


def bin_series(x: np.ndarray, ys: dict[str, np.ndarray], how: dict[str, str], max_points: int = MAX_POINTS):
    """Shorten aligned series to at most max_points by grouping consecutive
    points. `how` per series: 'max' keeps spikes (loss), 'mean' averages,
    'last' takes the value at the end of the bin (running averages)."""
    n = len(x)
    if n <= max_points:
        return np.asarray(x), {k: np.asarray(v) for k, v in ys.items()}
    edges = np.linspace(0, n, max_points + 1).astype(int)
    xb = np.array([x[e - 1] for e in edges[1:]])
    out = {}
    for k, v in ys.items():
        v = np.asarray(v, dtype=float)
        f = {"max": np.max, "mean": np.mean, "last": lambda a: a[-1]}[how[k]]
        out[k] = np.array([f(v[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return xb, out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def _spread(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"median": None, "mean": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(v)), "mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size)}


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

def extract_experiment(exp_dir: Path, settings: dict, mu: np.ndarray, sigma: np.ndarray,
                       processed_dir: Path, keep_seed_series: bool = True) -> dict:
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    for name, digest in manifest["calibration_sha256"].items():
        if _sha256(processed_dir / name) != digest:
            raise ValueError(
                f"{exp_dir.name}: {name} differs from the calibration this experiment was trained on. "
                "Point --processed-dir at the calibration it used (for example a calibration-window run)."
            )
    cfg = manifest["config"]
    if len(manifest["runs"]) < cfg["n_runs"]:
        raise ValueError(f"{exp_dir} has {len(manifest['runs'])} of {cfg['n_runs']} runs; finish or remove it first")
    gamma, B, n_ep = cfg["gamma"], cfg["episode_length"], cfg["n_episodes"]
    near = float(settings["near_optimal_bps"])
    k = int(settings.get("convergence_k") or np.ceil(0.10 * n_ep))
    dq_tol = float(settings["dq_tol"])

    grid = generate_weight_grid(len(TICKERS), increment=cfg["increment"])
    if grid.shape[0] != manifest["n_actions"]:
        raise ValueError(f"{exp_dir.name}: grid size {grid.shape[0]} does not match the manifest")
    if "grid_sha256" in manifest and manifest["grid_sha256"] != grid_fingerprint(grid):
        raise ValueError(f"{exp_dir.name}: the action grid differs from the one used in training")
    u = objective(grid, mu, sigma, gamma)
    order = np.argsort(-u, kind="stable")
    best = int(order[0])
    u_best = float(u[best])
    w_cont = constrained_optimal_weights(mu, sigma, gamma)
    u_cont = float(objective(w_cont, mu, sigma, gamma))
    loss = (u_best - u) * BPS
    gap12 = float(u[order[0]] - u[order[1]])
    sd_best = reward_sd(grid[best], mu, sigma, gamma, B)

    seeds, raw = [], {"loss": [], "reward": [], "regret": [], "share": []}
    ep_grid = None
    for entry in manifest["runs"]:
        r = entry["run"]
        d = np.load(exp_dir / entry["file"])
        ep = d["checkpoint_episode"]
        greedy = d["checkpoint_greedy"]
        counts = d["checkpoint_action_counts"]  # int32; the products below promote to float64
        ch_ep, ch_act = d["greedy_change_episode"], d["greedy_change_action"]
        final = int(np.argmax(d["final_q"]))
        w = grid[final]
        settle = settled_from(ep, greedy, final)
        exc = excursions(ch_ep, ch_act, final, settle, n_ep, loss)
        half_exc = [e for e in exc if e[0] > n_ep // 2]
        window = np.diff(np.concatenate([[0], ep]))
        regret_avg = np.cumsum(counts @ (u_best - u)) / ep * BPS
        reward = d["checkpoint_mean_reward"]
        reward_ma = np.array([reward[max(0, i - 4): i + 1].mean() for i in range(reward.size)])
        stable_at, _ = first_greedy_stable(ch_ep, ch_act, n_ep, k)
        through_end = greedy_stable_through_end(ch_ep, n_ep, k)
        dq_at = first_for_good(ep, d["checkpoint_max_abs_dq"] < dq_tol, full_windows(ep, cfg["log_every"]))
        near_ck = first_for_good(ep, loss[greedy] <= near)
        near_ev = near_optimal_every_episode(ch_ep, ch_act, loss, near)
        stats = exact_stats(w, mu, sigma, gamma)
        obs = lambda e: None if e is None else int(e) * B
        seed = {
            "run": r,
            "action": final,
            "weights": [round(float(x), 4) for x in w],
            "rank": int(np.flatnonzero(order == final)[0]) + 1,
            "is_best": final == best,
            "loss_best": float(loss[final]),
            "loss_cont": float((u_cont - u[final]) * BPS),
            "mean_annual": stats["mean_annual"],
            "vol_annual": stats["vol_annual"],
            "l1_cont": float(np.abs(w - w_cont).sum()),
            "l2_cont": float(np.linalg.norm(w - w_cont)),
            "l1_best": float(np.abs(w - grid[best]).sum()),
            "visits_final": int(d["final_visit_counts"][final]),
            "visits_best": int(d["final_visit_counts"][best]),
            "regret_avg_bps": float(regret_avg[-1]),
            "greedy_changes": int(ch_ep.size - 1),
            "settled_obs": obs(settle),
            "excursions": len(exc),
            "excursions_after_half": len(half_exc),
            "longest_excursion_after_half": max((e[1] for e in half_exc), default=0),
            "first_stable_obs": obs(stable_at),
            "primary_obs": obs(through_end),
            "dq_obs": obs(dq_at),
            "near_obs_ckpt": obs(near_ck),
            "near_obs_episode": obs(near_ev),
            "seconds": entry.get("seconds"),
        }
        if ep_grid is None:
            ep_grid = ep
        elif not np.array_equal(ep_grid, ep):
            raise ValueError(f"{exp_dir.name}: seeds have different checkpoint grids")
        per_seed = {"loss": loss[greedy], "reward": reward_ma * 252 * 100, "regret": regret_avg, "share": counts[:, final] / window}
        for key in raw:
            raw[key].append(per_seed[key])
        # Per-seed curves keep each bin's worst loss so brief spikes stay visible in the small multiples.
        xb, yb = bin_series(ep, per_seed, {"loss": "max", "reward": "mean", "regret": "last", "share": "mean"})
        if keep_seed_series:
            seed["series"] = {"loss": np.round(yb["loss"], 2).tolist(), "reward_pct": np.round(yb["reward"], 3).tolist(),
                              "share_final": np.round(yb["share"], 4).tolist()}
            seed["excursion_losses"] = [round(e[2], 1) for e in exc]
        seeds.append(seed)
        del d, counts

    # Cross-seed summaries are taken at full checkpoint resolution and only then
    # binned by averaging, so runs with different checkpoint counts stay
    # comparable (max-binning first would inflate longer runs' curves).
    stack = {k2: np.vstack(v) for k2, v in raw.items()}
    med = {"loss_median": np.median(stack["loss"], 0), "loss_q1": np.quantile(stack["loss"], 0.25, axis=0),
           "loss_q3": np.quantile(stack["loss"], 0.75, axis=0), "reward_median": np.median(stack["reward"], 0),
           "regret_median": np.median(stack["regret"], 0), "share_median": np.median(stack["share"], 0)}
    x_series, medb = bin_series(ep_grid, med, {k2: ("last" if k2 == "regret_median" else "mean") for k2 in med})
    eps = cfg["epsilon0"] * np.exp(-cfg["beta"] * x_series)
    params = {
        "gamma": gamma, "increment": cfg["increment"], "n_actions": int(grid.shape[0]),
        "episode_length": B, "n_episodes": n_ep, "n_runs": cfg["n_runs"], "master_seed": cfg["master_seed"],
        "simulated_days_per_seed": n_ep * B,
        "epsilon0": cfg["epsilon0"], "beta": cfg["beta"],
        "epsilon_at_half": float(cfg["epsilon0"] * np.exp(-cfg["beta"] * n_ep / 2)),
        "epsilon_final": float(cfg["epsilon0"] * np.exp(-cfg["beta"] * n_ep)),
        "random_tries_per_portfolio": float(cfg["epsilon0"] * (1 - np.exp(-cfg["beta"] * n_ep)) / cfg["beta"] / grid.shape[0]),
        "alpha_rule": cfg["alpha_rule"], "q_init": cfg["q_init"], "discount": cfg["discount"],
        "log_every": cfg["log_every"], "convergence_k": k, "near_optimal_bps": near, "dq_tol": dq_tol,
        "n_eval_episodes": settings.get("n_eval_episodes"),
    }
    loss_best = [s["loss_best"] for s in seeds]
    summary = {
        "disc_gap_bps": (u_cont - u_best) * BPS,
        "learning_gap": _spread(loss_best),
        "total_gap": _spread([s["loss_cont"] for s in seeds]),
        "exact_best": sum(s["is_best"] for s in seeds),
        "n_seeds": len(seeds),
        "distinct_final_portfolios": len({tuple(s["weights"]) for s in seeds}),
        "settled": _spread([s["settled_obs"] for s in seeds]),
        "primary": _spread([s["primary_obs"] for s in seeds]),
        "first_stable": _spread([s["first_stable_obs"] for s in seeds]),
        "dq": _spread([s["dq_obs"] for s in seeds]),
        "near_ckpt": _spread([s["near_obs_ckpt"] for s in seeds]),
        "near_episode": _spread([s["near_obs_episode"] for s in seeds]),
        "regret_avg": _spread([s["regret_avg_bps"] for s in seeds]),
        "within_near_of_best": sum(s["loss_best"] <= near for s in seeds),
        "seconds_per_seed": _median([s["seconds"] for s in seeds]),
    }
    return {
        "name": exp_dir.name,
        "family": family_key(gamma, cfg["increment"]),
        "params": params,
        "benchmarks": {
            "best_weights": grid[best].round(4).tolist(),
            "cont_weights": np.round(w_cont, 4).tolist(),
            "u_best_daily": u_best, "u_cont_daily": u_cont,
            "rivals_within_near": int(np.sum(loss <= near) - 1),
            "gap_best_second_bps": gap12 * BPS,
            "reward_sd_best_daily": sd_best,
            "tries_to_separate_top_two": tries_to_separate(sd_best, gap12),
        },
        "summary": summary,
        "seeds": seeds,
        "series": {
            "x_episodes": x_series.astype(int).tolist(),
            "epsilon": np.round(eps, 6).tolist(),
            "loss_median": np.round(medb["loss_median"], 2).tolist(),
            "loss_q1": np.round(medb["loss_q1"], 2).tolist(),
            "loss_q3": np.round(medb["loss_q3"], 2).tolist(),
            "reward_median": np.round(medb["reward_median"], 3).tolist(),
            "regret_median": np.round(medb["regret_median"], 2).tolist(),
            "share_median": np.round(medb["share_median"], 4).tolist(),
        },
        "code": {"git_commit": manifest.get("git_commit"), "numpy": manifest.get("seed_plan", {}).get("numpy_version"),
                 "code_sha256": manifest.get("code_sha256", {})},
        "calibration_sha256": manifest["calibration_sha256"],
    }


def family_key(gamma: float, increment: float) -> str:
    """Experiments that answer the same question across runs: same risk
    aversion and same grid. Everything else (B, budget, exploration) is what a
    comparison varies."""
    return f"gamma{gamma:g}_grid{round(increment * 100):d}"


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY) -> dict:
    reg = json.loads(path.read_text(encoding="utf-8"))
    ids = [r["id"] for r in reg["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("registry run ids must be unique")
    return reg


def registry_entry(run_id: str, registry: dict) -> dict:
    for r in registry["runs"]:
        if r["id"] == run_id:
            if "pre_registered" not in r:
                raise ValueError(f"registry entry {run_id!r} must state pre_registered (true only for settings fixed "
                                 "before any results were seen)")
            return r
    raise KeyError(f"{run_id!r} is not in reports/registry.json; add an entry first (see reports/README.md)")


def extract_run(entry: dict, root: Path = ROOT, processed_dir: Path = ROOT / PROCESSED_DIR) -> dict:
    results_dir = root / entry["results_dir"]
    settings = json.loads((results_dir / "settings.json").read_text())
    mu, sigma = load_calibration(processed_dir)
    exp_dirs = sorted(p.parent for p in results_dir.glob("*/manifest.json"))
    if entry.get("experiments"):
        wanted = set(entry["experiments"])
        exp_dirs = [p for p in exp_dirs if p.name in wanted]
        missing = wanted - {p.name for p in exp_dirs}
        if missing:
            raise ValueError(f"experiments listed in the registry but not found in {results_dir}: {sorted(missing)}")
    if not exp_dirs:
        raise ValueError(f"no experiment folders with a manifest.json in {results_dir}")
    experiments = [extract_experiment(p, settings, mu, sigma, processed_dir) for p in exp_dirs]
    index_path = results_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {k: entry.get(k) for k in ("id", "title", "date", "results_dir", "git_ref", "scale", "what_changed", "pre_registered")},
        "settings": settings,
        "index": {k: index.get(k) for k in ("written_utc", "git_commit", "python", "numpy", "full_scale")},
        "tickers": list(TICKERS),
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / PROCESSED_DIR,
                        help="calibration the run was trained on (default: data/processed)")
    args = parser.parse_args()
    entry = registry_entry(args.run_id, load_registry())
    data = extract_run(entry, processed_dir=args.processed_dir)
    out = REPORTS / args.run_id / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB, {len(data['experiments'])} experiments)")


if __name__ == "__main__":
    main()
