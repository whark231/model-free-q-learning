"""Extract the numbers behind the 2026-10-22 full-run report.

Reads the saved training logs (results/<experiment>/run_*.npz), the
evaluation summaries and the calibration, and writes data.json next to this
file. build_report.py then inlines data.json into report.html.

Evaluation-side code: it uses mu, Sigma and the benchmark portfolios, which
the agent never sees. Nothing here is imported by the agent or train.py.

Run from the repo root:
    python reports/full_run_2026-10-22/build_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `src` imports work

from src.action_space.grid import generate_weight_grid
from src.benchmarks.mean_variance import constrained_optimal_weights
from src.data.universe import TICKERS
from src.environment import load_calibration

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "data.json"

GRIDS = {"g20": "baseline_g5_grid20", "g25": "resolution_g5_grid25"}
ROBUST = {"g3": "robust_g3_grid20", "g5": "baseline_g5_grid20", "g8": "robust_g8_grid20"}
BPS = 252 * 1e4          # daily objective difference -> bps per year
NEAR = 25.0              # near-optimal line, bps/yr (src/config.py)
THIN = 1                 # keep every checkpoint (1,300 points); thinning hid brief switches caught at a checkpoint


def objective(w, mu, sigma, gamma):
    w = np.atleast_2d(w)
    return w @ mu - 0.5 * gamma * np.einsum("ij,jk,ik->i", w, sigma, w)


def settled_from(ckpt_episode, ckpt_greedy, final):
    """First checkpoint from which the final action is greedy at every later checkpoint."""
    bad = np.flatnonzero(ckpt_greedy != final)
    return int(ckpt_episode[bad[-1] + 1]) if bad.size else int(ckpt_episode[0])


def near_optimal_episode_level(change_ep, change_act, loss_by_action):
    """First episode after which every greedy action, including brief switches, is within NEAR bps."""
    over = np.flatnonzero(loss_by_action[change_act] > NEAR)
    if over.size == 0:
        return int(change_ep[0])
    i = over[-1]
    return int(change_ep[i + 1]) if i + 1 < change_ep.size else None


def experiment(name, mu, sigma, keep_series):
    run_dir = RESULTS / name
    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    gamma, B, n_ep = cfg["gamma"], cfg["episode_length"], cfg["n_episodes"]
    grid = generate_weight_grid(len(TICKERS), increment=cfg["increment"])
    u = objective(grid, mu, sigma, gamma)
    best = int(np.argmax(u))
    u_best = float(u[best])
    w_cont = constrained_optimal_weights(mu, sigma, gamma)
    u_cont = float(objective(w_cont, mu, sigma, gamma)[0])
    loss = (u_best - u) * BPS                       # CE loss vs best grid portfolio, every action
    order = np.argsort(-u)

    perf = pd.read_csv(run_dir / "evaluation" / "performance.csv").set_index("portfolio")
    seeds = []
    for rec in summary["per_seed"]:
        r = rec["run"]
        d = np.load(run_dir / f"run_{r:03d}.npz")
        ep = d["checkpoint_episode"]
        greedy = d["checkpoint_greedy"]
        counts = d["checkpoint_action_counts"].astype(np.int64)
        final = int(np.argmax(d["final_q"]))
        assert final == rec["learned_action"] == int(d["final_greedy"])
        ch_ep, ch_act = d["greedy_change_episode"], d["greedy_change_action"]
        settle = settled_from(ep, greedy, final)
        exc = excursions(ch_ep, ch_act, final, settle, n_ep, loss)
        regret_cum = np.cumsum(counts @ (u_best - u))          # daily-U units summed over episodes
        avg_regret_bps = regret_cum / ep * BPS
        window = np.diff(np.concatenate([[0], ep]))
        share_final = counts[:, final] / window
        reward = d["checkpoint_mean_reward"]
        reward_ma = np.convolve(reward, np.ones(5) / 5, mode="full")[: reward.size]
        reward_ma[:4] = np.cumsum(reward[:4]) / np.arange(1, 5)
        # greedy switches per 100k episodes
        switches = np.histogram(ch_ep[1:], bins=np.arange(0, n_ep + 1, 100_000))[0]
        near_ep = near_optimal_episode_level(ch_ep, ch_act, loss)
        p = perf.loc[f"learned_run_{r}"]
        seed = {
            "run": r,
            "action": final,
            "weights": [round(x, 4) for x in rec["learned_weights"]],
            "rank": int(np.flatnonzero(order == final)[0]) + 1,
            "loss_best": rec["ce_loss_vs_best_discrete_bps"],
            "loss_cont": rec["ce_loss_vs_continuous_bps"],
            "is_best": rec["is_best_discrete"],
            "visits": rec["visits_of_learned_action"],
            "visits_best": int(d["final_visit_counts"][best]),
            "visits_top5": [int(d["final_visit_counts"][i]) for i in order[:5]],
            "q_gap_final_vs_best": float((d["final_q"][final] - d["final_q"][best]) * BPS),
            "l1_cont": rec["l1_to_continuous"],
            "l2_cont": rec["euclidean_to_continuous"],
            "l1_best": rec["l1_to_best_discrete"],
            "mean_annual": rec["exact"]["mean_annual"],
            "vol_annual": rec["exact"]["vol_annual"],
            "u_daily": rec["exact"]["objective_daily"],
            "realized_shortfall_bps": float(p["realized_shortfall_vs_best_discrete_daily"]) * BPS,
            "realized_shortfall_se_bps": float(p["realized_shortfall_se"]) * BPS,
            "regret_avg_bps": float(avg_regret_bps[-1]),
            "regret_cum_daily": rec["final_cumulative_regret"],
            "settled_from_ep": settle,
            "greedy_changes": int(ch_ep.size - 1),
            "switches_after_settle": len(exc),
            "excursion_durations": [e[1] for e in exc],
            "excursion_starts": [e[0] for e in exc],
            "first_stable_obs": rec["greedy_stable_observations"],
            "primary_obs": rec["greedy_stable_through_end_observations"],
            "dq_obs": rec["dq_below_tol_for_good_observations"],
            "near_obs_ckpt": rec["near_optimal_for_good_observations"],
            "near_obs_episode": None if near_ep is None else near_ep * B,
            "last_change_ep": int(ch_ep[-1]),
        }
        if keep_series:
            sl = slice(THIN - 1, None, THIN)
            seed["series"] = {
                "loss": np.round(loss[greedy][sl], 2).tolist(),
                "reward_pct": np.round(reward_ma[sl] * 252 * 100, 3).tolist(),
                "regret_avg_bps": np.round(avg_regret_bps[sl], 2).tolist(),
                "share_final": np.round(share_final[sl], 4).tolist(),
                "switches_per_100k": switches.tolist(),
            }
            # Worst portfolio held during each excursion after settling: shows Finding 1 directly.
            seed["late_switch_losses"] = [round(e[2], 1) for e in exc]
        seeds.append(seed)

    out = {
        "name": name,
        "gamma": gamma,
        "increment": cfg["increment"],
        "n_actions": int(grid.shape[0]),
        "n_episodes": n_ep,
        "episode_length": B,
        "best_action": best,
        "best_weights": grid[best].round(4).tolist(),
        "cont_weights": np.round(w_cont, 4).tolist(),
        "u_best": u_best,
        "u_cont": u_cont,
        "disc_gap_bps": (u_cont - u_best) * BPS,
        "n_within_near": int(np.sum(loss <= NEAR)),   # includes the best portfolio itself
        "gap_best_second_bps": float(loss[order[1]]),
        "top5": [{"weights": grid[i].round(2).tolist(), "loss": round(float(loss[i]), 2)} for i in order[:5]],
        "benchmarks": {k: {"mean_annual": v["exact"]["mean_annual"], "vol_annual": v["exact"]["vol_annual"],
                           "u_daily": v["exact"]["objective_daily"],
                           "loss_best": (u_best - v["exact"]["objective_daily"]) * BPS,
                           "realized_u_pct": v["evaluation_sample"]["objective_daily"] * 252 * 100,
                           "realized_se_pct": v["evaluation_sample"]["objective_se"] * 252 * 100}
                       for k, v in summary["benchmarks"].items()},
        "convergence": {k: summary["convergence"][k]["share_of_seeds"] for k in summary["convergence"]["rules"]},
        "seeds": seeds,
    }
    if keep_series:
        ckpt = np.load(run_dir / "run_000.npz")["checkpoint_episode"]
        out["x_days_m"] = np.round(ckpt[THIN - 1::THIN] * B / 1e6, 2).tolist()
        out["epsilon"] = np.round(np.load(run_dir / "run_000.npz")["checkpoint_next_epsilon"][THIN - 1::THIN], 5).tolist()
    return out


def excursions(ch_ep, ch_act, final, settle, n_ep, loss):
    """Distinct excursions after settling: maximal runs during which the greedy
    action is not the final one. Returns (start episode, duration, worst loss)."""
    ends = np.append(ch_ep[1:], n_ep)
    out, start, worst = [], None, 0.0
    for e0, e1, a in zip(ch_ep, ends, ch_act):
        if e0 <= settle:
            continue
        if a != final:
            if start is None:
                start, worst = int(e0), 0.0
            worst = max(worst, float(loss[a]))
        elif start is not None:
            out.append((start, int(e0) - start, worst))
            start = None
    if start is not None:   # cannot happen: the final action is greedy at the end
        out.append((start, int(n_ep) - start, worst))
    return out


def reward_noise(mu, sigma, gamma, w_best, w_second, B, n=200_000, seed=7):
    """SD of one episode's reward for the best portfolio, and the paired SD of the
    best-minus-second difference, from a fixed-seed simulation (evaluation side)."""
    rng = np.random.default_rng(seed)
    r = rng.multivariate_normal(mu, sigma, size=(n, B))
    def rew(w):
        p = r @ w
        return p.mean(1) - 0.5 * gamma * p.var(1, ddof=1)
    a, b = rew(w_best), rew(w_second)
    return float(a.std()), float((a - b).std())


def main():
    mu, sigma = load_calibration()
    data = {"tickers": list(TICKERS)}
    data["grids"] = {k: experiment(v, mu, sigma, keep_series=True) for k, v in GRIDS.items()}
    data["robust"] = {k: experiment(v, mu, sigma, keep_series=False) for k, v in ROBUST.items()}
    for k, g in data["grids"].items():
        grid = generate_weight_grid(9, g["increment"])
        ug = objective(grid, mu, sigma, g["gamma"])
        og = np.argsort(-ug)
        sd_b, _ = reward_noise(mu, sigma, g["gamma"], grid[og[0]], grid[og[1]], g["episode_length"])
        g["reward_sd_best_daily"] = sd_b
        g["visits_each_to_separate_2se"] = float((2 * np.sqrt(2) * sd_b / (ug[og[0]] - ug[og[1]])) ** 2)

    # Calibration facts for the discussion section.
    stats = pd.read_csv(ROOT / "data/processed/summary_statistics.csv", index_col=0).loc[list(TICKERS)]
    T = len(pd.read_csv(ROOT / "data/processed/excess_returns.csv"))
    se_mean_annual = stats["std_daily"].values / np.sqrt(T) * 252
    eig = np.linalg.eigvalsh(sigma)
    g20 = data["grids"]["g20"]
    grid20 = generate_weight_grid(9, 0.2)
    u20 = objective(grid20, mu, sigma, 5.0)
    o = np.argsort(-u20)
    sd_best, sd_diff = reward_noise(mu, sigma, 5.0, grid20[o[0]], grid20[o[1]], 20)
    gap12 = float(u20[o[0]] - u20[o[1]])
    data["calibration"] = {
        "n_obs": T,
        "mean_annual_pct": (stats["mean_annualized"].values * 100).round(2).tolist(),
        "vol_annual_pct": (stats["vol_annualized"].values * 100).round(2).tolist(),
        "se_mean_annual_pct": (se_mean_annual * 100).round(2).tolist(),
        "cond_number": float(eig.max() / eig.min()),
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "reward_sd_best_daily": sd_best,
        "reward_sd_diff_daily": sd_diff,
        "gap_best_second_daily": gap12,
        "gap_best_second_bps": gap12 * BPS,
        # The agent never sees paired draws: each action's rewards come from
        # different episodes, so the relevant SD of a difference is sqrt(2) * sd_best.
        "visits_each_to_separate_2se_unpaired": float((2 * np.sqrt(2) * sd_best / gap12) ** 2),
        "visits_each_to_separate_2se_paired": float((2 * sd_diff / gap12) ** 2),
        "u_best_daily": g20["u_best"],
    }
    data["settings"] = json.loads((RESULTS / "settings.json").read_text())
    data["index"] = {k: json.loads((RESULTS / "index.json").read_text())[k] for k in ("written_utc", "git_commit", "python", "numpy")}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
