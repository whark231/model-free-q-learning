"""Step-size tuning for the constant-alpha agents (post hoc, FA_INTEGRATION_NOTES.md §5)

In plain terms: before the function-approximation (FA) runs, pick each
model's constant learning rate alpha (and, for the spline model, its l2
penalty) on separate tuning seeds. Each candidate is trained in full, and the
winner is the one whose final portfolio earns the most realized reward on a
fresh evaluation sample. This is held-out validation by simulation: the
returns are simulated from mu and Sigma, as in training, but no closed-form
objective, benchmark or optimal weight is computed, and the reported seeds
and their evaluation draws are never touched. The criterion is an unbiased
estimate of each final portfolio's objective, which is what a validation
score should be. The agent itself could not compute it: the evaluation
scores every final portfolio on the same full return draws.

Three models are tuned, each at fractions of its own stability bound
2 / max||phi||^2 (from the grid's features alone):
- tabular: the existing agent with a constant alpha instead of 1/N(a), for the
  post-hoc constant-alpha run ("tabular-const"). Its bound is 2 (one-hot);
- quadratic: the correctly specified reference model;
- spline: the main model; pass 2 also sweeps l2 at its best two alphas.
The tabular agent with alpha = 1/N(a) is trained on the same tuning seeds as
a reference row. It is never a candidate.

Settings match the reported runs (gamma = 5, 20% grid, B = 20, 13M episodes,
the baseline epsilon schedule, greedy refresh 100 for FA) except the master
seed: 20261103 here, never the reported 20261016. Equal alpha fractions are
not equal effective steps across models, so each model is compared only with
itself.

Resumable: each (candidate, seed) result is written to
<out>/records/<candidate>_seed<i>.json as soon as it finishes, and a rerun
skips every record already on disk. A crash or a closed laptop loses only
the runs in progress. A record written with different settings (another
training budget, for example a smoke test in the same folder) is refused
rather than reused. Commit the code before a full run, so each record's
git_commit contains the code that wrote it.

Training-seed luck dominates close calls. All candidates share the same
training streams for a given seed index, so the summary also reports each
candidate's shortfall to its model's best paired by seed: the standard error
over the seeds, and on how many seeds it did at least as well. The
evaluation-episode standard error is much smaller and only says the final
portfolios were scored precisely.

Close-call rule (fixed 2026-11-04, before any full-scale result): within each
model the highest mean over seeds 0-2 wins, unless the runner-up's shortfall
is at most one seed-paired standard error. Then the model's top 3 are trained
on 3 more seeds (3-5) and the winner is the best of those 3 over all 6 seeds.
SeedPlan streams do not depend on the number of runs, so seeds 0-2 are the
same either way.

Usage, from the repo root:

    python -m src.tune_fa --all --jobs 4         # everything below, in order
    python -m src.tune_fa --pass 1 --jobs 4      # 25 candidates x 3 seeds
    python -m src.tune_fa --pass 2 --jobs 4      # spline l2 sweep, after pass 1
    python -m src.tune_fa --extend --jobs 4      # seeds 3-5 for close calls, after pass 2
    python -m src.tune_fa --summarize            # criterion table -> fa_tuning.json

    # quick end-to-end check on a tiny budget, into a scratch folder:
    python -m src.tune_fa --pass 1 --episodes 20000 --eval-episodes 20000 --out <scratch>
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.config import EPSILON_AT_HALFWAY, EPISODE_LENGTH, GAMMA_BASELINE, INCREMENT_BASELINE, LOG_EVERY, \
    N_EPISODES, beta_for
from src.environment import ReturnSimulator, SeedPlan, load_calibration
from src.train import TrainConfig, _git_commit, build_features, config_record, default_agent_factory, \
    train_one_run

TUNING_MASTER_SEED = 20261103  # never the reported 20261016
BPS_YEAR = 252e4  # daily reward -> bps per year
N_TUNING_SEEDS = 3  # seeds 0-2, every candidate
N_EXTRA_SEEDS = 3  # seeds 3-5, only for a model's top 3 when its choice is a close call
TOP_EXTENDED = 3
N_EVAL_EPISODES = 200_000
EVAL_CHUNK = 5_000  # evaluation episodes generated at a time; same draws as one block
GREEDY_REFRESH = 100  # FA_INTEGRATION_NOTES.md §3
FRACTIONS = (0.5, 0.1, 0.02, 0.004, 8e-4, 1.6e-4, 3.2e-5, 6.4e-6)  # of the stability bound
L2_VALUES = (1e-5, 1e-4, 1e-3, 1e-2)  # pass 2; l2 = 0 comes from pass 1
MODELS = ("tabular", "quadratic", "spline")
REFERENCE = "tabular_1overN"
OUT_DIR = Path("results/tuning")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def stability_bound(model: str, grid: np.ndarray) -> float:
    """2 / max||phi||^2 over the grid's feature rows. One-hot rows have norm 1."""
    if model == "tabular":
        return 2.0
    F = build_features(_config(model, 1.0, 0.0, N_EPISODES), grid).astype(float)
    return 2.0 / float((F ** 2).sum(axis=1).max())


def candidate_id(model: str, fraction: float | None, l2: float = 0.0) -> str:
    if fraction is None:
        return REFERENCE
    return f"{model}_f{fraction:g}" + (f"_l2{l2:g}" if l2 else "")


def pass1_candidates() -> list[dict]:
    """The 1/N reference, then every model at every fraction with l2 = 0."""
    out = [{"id": REFERENCE, "model": "tabular", "fraction": None, "l2": 0.0}]
    out += [{"id": candidate_id(m, f), "model": m, "fraction": f, "l2": 0.0} for m in MODELS for f in FRACTIONS]
    return out


def pass2_candidates(best_spline_fractions) -> list[dict]:
    return [{"id": candidate_id("spline", f, l2), "model": "spline", "fraction": f, "l2": l2}
            for f in best_spline_fractions for l2 in L2_VALUES]


def _config(model: str, alpha, l2: float, n_episodes: int) -> TrainConfig:
    """The reported runs' settings with the tuning master seed."""
    fa = {} if model == "tabular" else dict(agent="linear", feature_map=model, l2=l2,
                                            greedy_refresh_every=GREEDY_REFRESH)
    return TrainConfig(gamma=GAMMA_BASELINE, episode_length=EPISODE_LENGTH, n_episodes=n_episodes,
                       master_seed=TUNING_MASTER_SEED, n_runs=N_TUNING_SEEDS + N_EXTRA_SEEDS,
                       beta=beta_for(n_episodes, EPSILON_AT_HALFWAY), increment=INCREMENT_BASELINE,
                       alpha_rule=alpha, log_every=min(LOG_EVERY, n_episodes), **fa)


# ---------------------------------------------------------------------------
# Training one (candidate, seed)
# ---------------------------------------------------------------------------

def record_path(out_dir: Path, cand_id: str, seed: int) -> Path:
    return Path(out_dir) / "records" / f"{cand_id}_seed{seed}.json"


def candidate_config(candidate: dict, n_episodes: int, grid: np.ndarray) -> TrainConfig:
    """The training config for one candidate: alpha = 1/N for the reference,
    otherwise its fraction of the model's stability bound."""
    if candidate["fraction"] is None:
        alpha = "sample_average"
    else:
        alpha = candidate["fraction"] * stability_bound(candidate["model"], grid)
    return _config(candidate["model"], alpha, candidate["l2"], n_episodes)


def existing_record(path: Path, config: TrainConfig) -> dict | None:
    """The finished record at `path`, or None. A record written with other
    settings is refused, so a smoke test can never stand in for a full run."""
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["config"] != config_record(config):
        raise ValueError(f"{path} was written with different settings (n_episodes "
                         f"{record['config']['n_episodes']:,} vs {config.n_episodes:,}?); use another --out folder")
    return record


def run_candidate(candidate: dict, seed: int, n_episodes: int, out_dir: Path) -> dict:
    """Train one tuning seed exactly as run_experiment would train run `seed`
    of the same config, and save the agent-observable outcome. Returns the
    existing record instead if this run already finished with these settings."""
    path = record_path(out_dir, candidate["id"], seed)
    mu, sigma = load_calibration()
    grid = generate_weight_grid(mu.shape[0], increment=INCREMENT_BASELINE)
    config = candidate_config(candidate, n_episodes, grid)
    alpha = config.alpha_rule
    record = existing_record(path, config)
    if record is not None:
        return record
    returns_rng, agent_rng = SeedPlan(config.master_seed, config.n_runs).run_streams(seed)
    features = build_features(config, grid) if config.agent == "linear" else None
    agent = default_agent_factory(config, grid.shape[0], agent_rng, features=features)

    start = time.perf_counter()
    status, logs = "ok", None
    try:
        logs = train_one_run(config, grid, ReturnSimulator(mu, sigma, returns_rng), agent)
    except FloatingPointError as exc:  # theta blew up: alpha too large for this model
        status = f"diverged: {exc}"
    record = {
        **candidate,
        "seed": seed,
        "alpha": alpha,
        "status": status,
        "final_action": None if logs is None else int(np.argmax(logs["final_q"])),
        "final_q_of_action": None if logs is None else float(logs["final_q"].max()),
        "visits_of_action": None if logs is None else int(logs["final_visit_counts"][int(np.argmax(logs["final_q"]))]),
        "seconds": time.perf_counter() - start,
        "config": config_record(config),
        "git_commit": _git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)  # a record is either complete or absent
    return record


def run_all(candidates, n_episodes: int, out_dir: Path, jobs: int, seeds=range(N_TUNING_SEEDS)) -> None:
    grid = generate_weight_grid(9, increment=INCREMENT_BASELINE)
    todo = []
    for c in candidates:
        config = candidate_config(c, n_episodes, grid)
        todo += [(c, s) for s in seeds if existing_record(record_path(out_dir, c["id"], s), config) is None]
    print(f"{len(todo)} runs to do, {len(candidates) * len(seeds) - len(todo)} already on disk", flush=True)
    if jobs <= 1:
        for c, s in todo:
            r = run_candidate(c, s, n_episodes, out_dir)
            print(f"{r['id']} seed {s}: {r['status']}, {r['seconds'] / 60:.1f} min", flush=True)
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
    pool = ProcessPoolExecutor(max_workers=jobs)
    failed = []
    try:
        futures = {pool.submit(run_candidate, c, s, n_episodes, out_dir): (c["id"], s) for c, s in todo}
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as exc:  # report and keep going; no record is written, so a rerun retries it
                failed.append(futures[future])
                print(f"{futures[future][0]} seed {futures[future][1]}: FAILED, {exc!r}", flush=True)
                continue
            print(f"{r['id']} seed {r['seed']}: {r['status']}, {r['seconds'] / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted: cancelling queued runs; finished records are kept", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    if failed:
        raise RuntimeError(f"{len(failed)} run(s) failed: {failed}")


# ---------------------------------------------------------------------------
# The criterion: realized reward of the final portfolio on fresh draws
# ---------------------------------------------------------------------------

def reward_moments(actions, n_eval: int, chunk: int = EVAL_CHUNK):
    """Mean and covariance, across evaluation episodes, of the episodic reward
    of each grid portfolio in `actions`. Every portfolio is scored on the same
    episodes (common random numbers) from the tuning plan's evaluation
    stream, which training never touches."""
    mu, sigma = load_calibration()
    grid = generate_weight_grid(mu.shape[0], increment=INCREMENT_BASELINE)
    actions = np.asarray(actions, dtype=int)
    W = grid[actions]
    sim = ReturnSimulator(mu, sigma, SeedPlan(TUNING_MASTER_SEED, N_TUNING_SEEDS).evaluation_rng())
    s1, s2, done = np.zeros(len(actions)), np.zeros((len(actions), len(actions))), 0
    while done < n_eval:
        n = min(chunk, n_eval - done)
        daily = sim.sample_episodes(n, EPISODE_LENGTH) @ W.T  # (n, B, k)
        r = daily.mean(axis=1) - 0.5 * GAMMA_BASELINE * daily.var(axis=1, ddof=1)  # the training reward
        s1 += r.sum(axis=0)
        s2 += r.T @ r
        done += n
    mean = s1 / n_eval
    cov = (s2 - n_eval * np.outer(mean, mean)) / (n_eval - 1)
    return mean, cov


def summarize(out_dir: Path, n_eval: int) -> dict:
    """Score every finished candidate and apply the close-call rule.

    A candidate's criterion is the realized reward of its final portfolios,
    averaged over seeds 0-2. Within each model, each candidate's shortfall to
    the model's best is reported with two standard errors: paired over the
    training seeds (which decides close calls) and over evaluation episodes.
    If the runner-up is within one seed SE of the best, the model's top 3 are
    compared again over seeds 0-5 once those runs exist; until then the
    model's choice is pending and `extra_seeds_needed` lists the candidates."""
    records = [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "records").glob("*.json"))]
    budgets = {r["config"]["n_episodes"] for r in records}
    if len(budgets) > 1:
        raise ValueError(f"records under {out_dir} mix training budgets {sorted(budgets)}; keep one per folder")
    by_id: dict[str, dict] = {}
    for r in records:
        by_id.setdefault(r["id"], {})[r["seed"]] = r
    base_seeds = range(N_TUNING_SEEDS)
    all_seeds = range(N_TUNING_SEEDS + N_EXTRA_SEEDS)
    finished = {i for i, rs in by_id.items() if all(s in rs for s in base_seeds)}
    complete = {i for i in finished if all(by_id[i][s]["status"] == "ok" for s in base_seeds)}
    extended = {i for i in complete if all(s in by_id[i] and by_id[i][s]["status"] == "ok" for s in all_seeds)}
    actions = sorted({r["final_action"] for i in complete for r in by_id[i].values() if r["status"] == "ok"})
    if not actions:
        raise RuntimeError(f"no complete candidates under {out_dir}")
    mean, cov = reward_moments(actions, n_eval)
    col = {a: j for j, a in enumerate(actions)}

    def score(cand_id, seeds):
        """Seed-average weights c (criterion = c' mean) and each seed's reward in bps."""
        rs = [by_id[cand_id][s] for s in seeds]
        c = np.zeros(len(actions))
        for r in rs:
            c[col[r["final_action"]]] += 1.0 / len(rs)
        return c, np.array([BPS_YEAR * mean[col[r["final_action"]]] for r in rs])

    def compare(best, others, seeds):
        """Shortfall of each candidate to `best`, with its two standard errors."""
        c_best, seed_best = score(best, seeds)
        out = {}
        for i in others:
            c, per_seed = score(i, seeds)
            d, by_seed = c_best - c, seed_best - per_seed  # seeds share training streams
            out[i] = {
                "shortfall_to_model_best_bps": float(BPS_YEAR * d @ mean),
                "shortfall_seed_se_bps": float(by_seed.std(ddof=1) / np.sqrt(by_seed.size)),
                "shortfall_eval_se_bps": float(BPS_YEAR * np.sqrt(max(d @ cov @ d, 0.0) / n_eval)),
                "seeds_at_least_as_good": int((by_seed <= 0).sum()),
            }
        return out

    def rank(ids, key):
        """Best first. An exact tie means the same set of final portfolios
        across the seeds: prefer the smaller l2, then the larger alpha."""
        return sorted(ids, key=lambda i: (-rows[i][key], rows[i]["l2"], -rows[i]["fraction"]))

    rows = {}
    # `complete` is a set of strings, whose order changes with Python's hash
    # seed; sorting keeps fa_tuning.json byte-stable across runs (audit F-L12).
    for cand_id in sorted(complete):
        r0 = by_id[cand_id][0]
        c, per_seed = score(cand_id, base_seeds)
        rows[cand_id] = {
            "model": r0["model"], "fraction": r0["fraction"], "l2": r0["l2"], "alpha": r0["alpha"],
            "criterion_daily": float(c @ mean),
            "criterion_bps": float(BPS_YEAR * c @ mean),
            "per_seed_reward_bps": per_seed.tolist(),
            "final_actions": [by_id[cand_id][s]["final_action"] for s in base_seeds],
            "minutes_per_seed": float(np.mean([by_id[cand_id][s]["seconds"] for s in base_seeds]) / 60),
        }

    chosen, extra_needed = {}, {}
    for model in MODELS:
        ids = [i for i, row in rows.items() if row["model"] == model and row["fraction"] is not None]
        if not ids:
            continue
        ranked = rank(ids, "criterion_daily")
        best = ranked[0]
        shown = ids + ([REFERENCE] if model == "tabular" and REFERENCE in rows else [])
        for i, stats in compare(best, shown, base_seeds).items():
            rows[i].update(stats)
        # The shortfall and its seed SE come from different routes (c' mean and
        # the per-seed vector), so rounding can split a mathematical equality:
        # differing on exactly one of three seeds gives shortfall == SE. The
        # tolerance keeps "at most one SE" true to the rule (bug found 2026-11-05).
        runner_up = rows[ranked[1]] if len(ranked) > 1 else None
        close = runner_up is not None and (runner_up["shortfall_to_model_best_bps"]
                                           <= runner_up["shortfall_seed_se_bps"] * (1 + 1e-9) + 1e-12)
        top = ranked[:TOP_EXTENDED]
        decision = {"close_call": bool(close), "base_best": best}
        if close and not all(i in extended for i in top):
            decision.update(pending=True, extra_seeds_for=top)
            extra_needed[model] = top
        elif close:
            for i in top:
                c, per_seed = score(i, all_seeds)
                rows[i]["extended"] = {"criterion_daily": float(c @ mean), "criterion_bps": float(BPS_YEAR * c @ mean),
                                       "per_seed_reward_bps": per_seed.tolist()}
                rows[i]["criterion6_daily"] = rows[i]["extended"]["criterion_daily"]
            best = rank(top, "criterion6_daily")[0]
            for i, stats in compare(best, top, all_seeds).items():
                rows[i]["extended"].update(stats)
            decision["decided_on_seeds"] = len(all_seeds)
        else:
            decision["decided_on_seeds"] = len(base_seeds)
        if not decision.get("pending"):
            key = "criterion6_daily" if close else "criterion_daily"
            pool = top if close else ids
            decision["tied_with"] = sorted(i for i in pool if i != best and rows[i][key] == rows[best][key])
            chosen[model] = {k: rows[best][k] for k in ("fraction", "l2", "alpha")} | {"id": best} | decision
        else:
            chosen[model] = decision
    for row in rows.values():
        row.pop("criterion6_daily", None)

    summary = {
        "note": "post hoc; tuning seeds only (master seed 20261103); criterion is the realized reward of each "
                "seed's final greedy portfolio on fresh evaluation draws, averaged over seeds; no closed-form "
                "objective or benchmark is computed. shortfall_seed_se_bps is the paired standard error over "
                "training seeds and decides close calls; shortfall_eval_se_bps covers evaluation draws only. "
                "Close-call rule: if the runner-up is within one seed SE, the top 3 are compared on seeds 0-5",
        "n_episodes": budgets.pop(),
        "git_commits": sorted({str(r["git_commit"]) for r in records}),
        "n_eval_episodes": n_eval,
        "n_seeds": N_TUNING_SEEDS,
        "incomplete": sorted(set(by_id) - finished),
        "diverged": sorted(finished - complete),
        "extra_seeds_needed": extra_needed,
        "chosen": chosen,
        "candidates": rows,
    }
    (Path(out_dir) / "fa_tuning.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def extra_seed_candidates(summary: dict) -> list[dict]:
    """The candidates that need seeds 3-5 under the close-call rule."""
    rows = summary["candidates"]
    return [{"id": i, "model": rows[i]["model"], "fraction": rows[i]["fraction"], "l2": rows[i]["l2"]}
            for ids in summary["extra_seeds_needed"].values() for i in ids]


def best_spline_fractions(summary: dict, n: int = 2) -> list[float]:
    """The n spline fractions with the highest criterion at l2 = 0, ties to
    the larger alpha as in the final choice. Every pass-1 spline candidate
    must be finished; one that diverged counts as finished but is not chosen."""
    pass1 = {candidate_id("spline", f) for f in FRACTIONS}
    missing = pass1 - set(summary["candidates"]) - set(summary.get("diverged", []))
    if missing:
        raise RuntimeError(f"pass 2 needs every pass-1 spline candidate finished; run --pass 1 first "
                           f"(missing {sorted(missing)})")
    rows = [r for i, r in summary["candidates"].items() if i in pass1]
    return [r["fraction"] for r in sorted(rows, key=lambda r: (-r["criterion_daily"], -r["fraction"]))[:n]]


def _print_table(summary: dict) -> None:
    print(f"{summary['n_episodes']:,} training episodes, {summary['n_eval_episodes']:,} evaluation episodes; "
          "realized reward and shortfalls in bps/yr; seed SE is paired over the training seeds")
    print(f"{'candidate':<26} {'alpha':>10} {'reward':>8} {'per seed':>20} {'shortfall':>9} {'seed SE':>8} "
          f"{'eval SE':>8} {'seeds >=':>8} {'min/seed':>8}")
    for cand_id, r in sorted(summary["candidates"].items(), key=lambda kv: (kv[1]["model"], -kv[1]["criterion_daily"])):
        alpha = r["alpha"] if isinstance(r["alpha"], str) else f"{r['alpha']:.3g}"
        cells = ["", "", "", ""]
        if "shortfall_to_model_best_bps" in r:
            cells = [f"{r['shortfall_to_model_best_bps']:.1f}", f"{r['shortfall_seed_se_bps']:.1f}",
                     f"{r['shortfall_eval_se_bps']:.1f}", str(r["seeds_at_least_as_good"])]
        per_seed = " ".join(f"{v:.0f}" for v in r["per_seed_reward_bps"])
        print(f"{cand_id:<26} {alpha:>10} {r['criterion_bps']:>8.1f} {per_seed:>20} {cells[0]:>9} {cells[1]:>8} "
              f"{cells[2]:>8} {cells[3]:>8} {r['minutes_per_seed']:>8.1f}")
    extended = {i: r["extended"] for i, r in summary["candidates"].items() if "extended" in r}
    if extended:
        print("close calls, top 3 compared on seeds 0-5:")
        for cand_id, e in sorted(extended.items(), key=lambda kv: -kv[1]["criterion_daily"]):
            per_seed = " ".join(f"{v:.0f}" for v in e["per_seed_reward_bps"])
            print(f"  {cand_id:<24} reward {e['criterion_bps']:.1f}  per seed {per_seed}  shortfall "
                  f"{e['shortfall_to_model_best_bps']:.1f}  seed SE {e['shortfall_seed_se_bps']:.1f}  "
                  f"seeds >= {e['seeds_at_least_as_good']}")
    for key in ("incomplete", "diverged", "extra_seeds_needed"):
        if summary[key]:
            print(f"{key}: {summary[key]}")
    print("chosen:", json.dumps(summary["chosen"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pass", dest="which", type=int, choices=(1, 2))
    parser.add_argument("--extend", action="store_true", help="seeds 3-5 for close calls, after pass 2")
    parser.add_argument("--all", action="store_true", help="pass 1, pass 2, --extend and the summary, in order")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help="training episodes (smoke tests only)")
    parser.add_argument("--eval-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.which is None and not (args.summarize or args.extend or args.all):
        parser.error("choose --all, --pass 1, --pass 2, --extend or --summarize")
    if args.all or args.which == 1:
        run_all(pass1_candidates(), args.episodes, args.out, args.jobs)
    if args.all or args.which == 2:
        fractions = best_spline_fractions(summarize(args.out, args.eval_episodes))
        print(f"pass 2 at the best spline fractions from pass 1: {fractions}", flush=True)
        run_all(pass2_candidates(fractions), args.episodes, args.out, args.jobs)
    if args.all or args.extend:
        extra = extra_seed_candidates(summarize(args.out, args.eval_episodes))
        print(f"close calls: seeds 3-5 for {[c['id'] for c in extra] or 'none'}", flush=True)
        run_all(extra, args.episodes, args.out, args.jobs,
                seeds=range(N_TUNING_SEEDS, N_TUNING_SEEDS + N_EXTRA_SEEDS))
    if args.summarize or args.which or args.extend or args.all:
        summary = summarize(args.out, args.eval_episodes)
        if summary["n_episodes"] != args.episodes:
            raise SystemExit(f"{args.out} holds {summary['n_episodes']:,}-episode runs, not {args.episodes:,}; "
                             "pass the matching --episodes")
        _print_table(summary)


if __name__ == "__main__":
    main()
