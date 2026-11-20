"""Final transaction-cost-with-drift runs: train, score, summarize (TC_DRIFT_DESIGN.md §6, §7, §9).

In plain terms: train the agent with the tuned settings for 10 seeds at each of
the four cost levels, then score every trained agent against the oracle and the
reference policies on one common set of simulated paths per level, and write a
summary per level. Training jobs are independent, so they run in parallel;
scoring holds one level's paths (544 MB at m = 4,000) in memory, so levels are
scored one at a time.

    python -m src.tc_drift.run queue --workers 6          # train every missing (level, seed), headline level first
    python -m src.tc_drift.run train --level aum_1b --seed 0
    python -m src.tc_drift.run evaluate --level aum_1b    # after that level's seeds are trained
    python -m src.tc_drift.run report                     # one table over the levels that are scored

Written 2026-11-20 by William (with Claude) for Albert's review: the docs
assign the run driver to Albert (AUDIT_2026-11-06.md TC-2); he agreed the
interface in NEXT_STEPS_FOR_ALBERT.md.

What each step does:
- train: config.HYPERPARAMETERS, config.N_BLOCKS, config.CHECKPOINT_EVERY and
  the reported seed streams (config.run_rngs(seed), spec.MASTER_SEED). Saves the
  training log with theta at every checkpoint, the tail-averaged theta (design
  §6) and D12 for the averaged and the last-iterate policies (tune.convergence,
  every version's verdict). Never reads mu, Sigma, the costs or the oracle
  beyond what train() hands the agent (net rewards and holdings).
- evaluate: paths and starts from config.evaluation_rng() (the reported
  evaluation stream; the baseline used the same stream, but no TC-drift run
  did before these). Scores per seed the averaged
  policy (the headline), the last-iterate policy, and always rebalancing to the
  averaged agent's favourite target (TC-3); once per level the oracle, always
  rebalancing to the best target, trading once then holding, and myopic. The
  oracle and myopic are the lattice tables saved by `python -m
  src.tc_drift.oracle_runs` (results/tc-drift/oracle/<level>.npz), played on
  continuous holdings through the nearest 1% lattice point, as in check 5.
- report: the per-level summaries side by side.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.environment import SeedPlan
from src.hashing import file_sha256
from src.tc_drift import config, env, evaluate, spec, train, tune
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "tc-drift" / "runs"
ORACLE_DIR = REPO / "results" / "tc-drift" / "oracle"
QUEUE_ORDER = ("aum_1b", "aum_10b", "aum_100m", "zero")    # the headline level first, if time runs short
CODE_FILES = tune.CODE_FILES + ("src/tc_drift/run.py",)
GAP_KEYS = ("agent_minus_oracle", "agent_minus_always_rebalance", "agent_minus_trade_once", "agent_minus_myopic",
            "agent_minus_favourite_rebalance", "favourite_rebalance_minus_always_rebalance",
            "last_iterate_minus_oracle", "averaged_minus_last_iterate")
PINNED = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _require_settled() -> None:
    if config.pending():
        raise RuntimeError(f"config settings still pending: {config.pending()}")


def _provenance(extra_files=()) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    files = {f: file_sha256(REPO / f) for f in CODE_FILES}
    for f in map(Path, extra_files):
        name = f.resolve().relative_to(REPO).as_posix() if f.resolve().is_relative_to(REPO) else f.name
        files[name] = hashlib.sha256(f.read_bytes()).hexdigest()       # binary: raw bytes
    return {"commit": git("rev-parse", "HEAD"), "src_dirty": bool(git("status", "--porcelain", "--", "src")),
            "sha256": files, "python": platform.python_version(), "numpy": np.__version__,
            "seed_plan": SeedPlan(spec.MASTER_SEED, config.N_SEEDS).as_dict(),
            "threads": {k: os.environ.get(k) for k in PINNED},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _write_json(path: Path, obj) -> None:
    """Write through a temporary file and rename, so a killed job never
    leaves a truncated record that the queue would take as done."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _check_record(record: dict, n_blocks: int, checkpoint_every: int) -> None:
    """A saved training record must be a final run under the current settings."""
    expected = {"n_blocks": n_blocks, "checkpoint_every": checkpoint_every,
                "hyperparameters": config.HYPERPARAMETERS, "policy": tune.POLICY}
    wrong = {k: record.get(k) for k, v in expected.items() if record.get(k) != v}
    if wrong:
        raise ValueError(f"{record.get('level')} seed {record.get('seed')} was trained with other settings: {wrong}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def flat_log(log: dict, agent: DriftAgent) -> dict:
    """The arrays tune.convergence and tune.averaged_policy read, from a
    train() log: theta, the costs and the favourite target at every checkpoint."""
    d = log["diagnostics"]
    theta = d["theta"]
    favourite = [agent.with_parameters(t).favourite_target() for t in theta]
    return {"checkpoint_block": log["checkpoint_block"], "theta": theta, "c1": d["cost_linear"],
            "c2": d["cost_quadratic"], "epsilon": d["epsilon"], "favourite_target": np.array(favourite),
            "probe_greedy": log["probe_greedy"].astype(np.int16), "hold_fraction": log["hold_fraction"],
            "mean_cost_bps": log["mean_cost_bps"], "mean_turnover": log["mean_turnover"],
            "mean_net_reward": log["mean_net_reward"]}


def train_one(level: str, seed: int, out_dir: Path = OUT_DIR, n_blocks: int | None = None,
              checkpoint_every: int | None = None) -> dict:
    """Train one (level, seed) and save seed<N>.npz and seed<N>.json under
    out_dir/level. n_blocks and checkpoint_every exist for the tests' small
    runs; the reported runs use config's."""
    _require_settled()
    if level not in config.LEVELS or not 0 <= seed < config.N_SEEDS:
        raise ValueError(f"level must be one of {config.LEVELS} and seed in [0, {config.N_SEEDS})")
    n_blocks = n_blocks or config.N_BLOCKS
    checkpoint_every = checkpoint_every or config.CHECKPOINT_EVERY
    market = env.load_market()
    settings = cost_settings()[level]
    returns_rng, agent_rng, start_rng = config.run_rngs(seed)
    agent = DriftAgent(agent_rng, **config.HYPERPARAMETERS)
    probes = config.probe_states()

    started = _provenance()                       # the code this process imported, before the hours of training
    t0 = time.perf_counter()
    log = train.train(agent, market, settings, n_blocks=n_blocks, returns_rng=returns_rng, start_rng=start_rng,
                      start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                      checkpoint_every=checkpoint_every, probe_states=probes)
    train_seconds = time.perf_counter() - t0

    flat = flat_log(log, agent)
    averaged, conv_averaged, _ = tune.averaged_policy(agent, flat, n_blocks, probes)
    conv_last = tune.convergence(flat, n_blocks, probes)
    targets = spec.target_grid()

    out = Path(out_dir) / level
    out.mkdir(parents=True, exist_ok=True)
    finished = _provenance()
    if finished["sha256"] != started["sha256"]:
        changed = sorted(f for f in started["sha256"] if started["sha256"][f] != finished["sha256"].get(f))
        raise RuntimeError(f"code changed on disk while {level} seed {seed} trained: {changed}; not saved")
    np.savez_compressed(out / f"seed{seed}.npz", **flat, averaged_theta=averaged.theta,
                        final_theta=agent.theta, final_cost=np.array([agent.cost_linear, agent.cost_quadratic]))
    record = {
        "level": level, "seed": seed, "hyperparameters": config.HYPERPARAMETERS,
        "n_blocks": n_blocks, "checkpoint_every": checkpoint_every, "restart_every": config.RESTART_EVERY,
        "simulated_days": log["simulated_days"], "n_restarts": log["n_restarts"],
        "final_c1": agent.cost_linear, "final_c2": agent.cost_quadratic, "final_epsilon": agent.epsilon,
        "policy": tune.POLICY,
        "favourite_target": targets[averaged.favourite_target()].round(2).tolist(),
        "favourite_index": averaged.favourite_target(),
        "convergence": conv_averaged,
        "last_iterate": {"favourite_target": targets[agent.favourite_target()].round(2).tolist(),
                         "favourite_index": agent.favourite_target(), "convergence": conv_last},
        "train_seconds": round(train_seconds, 1),
        "averaged_theta": averaged.theta.tolist(), "final_theta": agent.theta.tolist(),
        "provenance": started,
    }
    _write_json(out / f"seed{seed}.json", record)
    return record


def queue(workers: int, out_dir: Path = OUT_DIR, levels=QUEUE_ORDER, seeds=range(config.N_SEEDS)) -> list:
    """Train every (level, seed) without a saved record, `workers` at a time,
    each in its own process with BLAS pinned to one thread. Logs go to
    out_dir/logs. Returns the (level, seed, exit code) of each job."""
    _require_settled()
    out_dir = Path(out_dir).resolve()
    jobs = []
    for lv in levels:
        for s in seeds:
            path = out_dir / lv / f"seed{s}.json"
            if path.exists():
                _check_record(json.loads(path.read_text(encoding="utf-8")), config.N_BLOCKS, config.CHECKPOINT_EVERY)
            else:
                jobs.append((lv, s))
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    child_env = {**os.environ, **PINNED, "PYTHONIOENCODING": "utf-8"}

    def one(job):
        level, seed = job
        with open(logs / f"{level}_seed{seed}.log", "a", encoding="utf-8") as fh:
            fh.write(f"--- started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            fh.flush()
            code = subprocess.run([sys.executable, "-m", "src.tc_drift.run", "train", "--level", level,
                                   "--seed", str(seed), "--out", str(out_dir)],
                                  cwd=REPO, env=child_env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"{datetime.now():%H:%M:%S} {level} seed {seed}: exit {code}", flush=True)
        return level, seed, code

    print(f"{len(jobs)} jobs, {workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def batch_policy(agent: DriftAgent) -> evaluate.Policy:
    """evaluate.agent_policy, with the agent's batched greedy (the same actions, tested)."""
    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return agent.greedy_actions(holdings)
    return policy


def lattice_policy(table: np.ndarray) -> evaluate.Policy:
    """A 1% lattice table played on continuous holdings through the nearest lattice point."""
    def policy(holdings: np.ndarray, t: int) -> np.ndarray:
        return table[spec.nearest_lattice_index(holdings, spec.LATTICE_INCREMENT)].astype(np.int64)
    return policy


def _stats(result: dict) -> dict:
    return {"mean_value_bps": result["mean_value_bps"], "hold_fraction": result["mean_hold_fraction"],
            "turnover_per_block": float(result["turnover_per_block"].mean()),
            "trades_per_path": float(result["trades"].mean()),
            "mean_trade_size": None if np.isnan(result["mean_trade_size"]) else float(result["mean_trade_size"]),
            "mean_weights": np.asarray(result["mean_weights"]).round(6).tolist()}


def evaluate_level(level: str, out_dir: Path = OUT_DIR, m: int | None = None, oracle_dir: Path = ORACLE_DIR,
                   require_all_seeds: bool = True) -> dict:
    """Score every trained seed of `level` and the reference policies on one
    common set of paths; write out_dir/level/summary.json, per_seed.csv and
    values.npz (every policy's per-path values, for re-analysis)."""
    _require_settled()
    m = m or config.M_EVAL
    level_dir = Path(out_dir) / level
    records = {int(p.stem[4:]): json.loads(p.read_text(encoding="utf-8")) for p in level_dir.glob("seed*.json")}
    missing = sorted(set(range(config.N_SEEDS)) - set(records))
    if require_all_seeds and missing:
        raise RuntimeError(f"{level}: seeds {missing} are not trained yet")
    if records:
        first = next(iter(records.values()))
        for r in records.values():
            _check_record(r, first["n_blocks"], first["checkpoint_every"])
        if require_all_seeds:
            _check_record(first, config.N_BLOCKS, config.CHECKPOINT_EVERY)
        commits = {r["provenance"]["commit"] for r in records.values()}
        if len(commits) > 1:
            raise ValueError(f"{level}: seeds were trained from different commits {sorted(commits)}")
    probes = config.probe_states()
    table_path = Path(oracle_dir) / f"{level}.npz"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} is missing; run `python -m src.tc_drift.oracle_runs` first")
    tables = np.load(table_path)

    t0 = time.perf_counter()
    market = env.load_market()
    settings = cost_settings()[level]
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(), config.H_EVAL, m)
    best = evaluate.best_static_target(market, settings.gamma)
    refs = {"oracle": lattice_policy(tables["policy_table"]),
            "always_rebalance": evaluate.always_rebalance_policy(best),
            "trade_once": evaluate.never_trade_policy(best),
            "myopic": lattice_policy(tables["myopic_policy_table"])}
    ref = {name: evaluate.evaluate_policy(p, starts, paths, settings) for name, p in refs.items()}
    gap = lambda a, b: evaluate.paired_gap(a, b, config.H_EVAL)
    values = {f"ref_{k}": v["value"] for k, v in ref.items()}

    rows = []
    for seed in sorted(records):
        r = records[seed]
        z = np.load(level_dir / f"seed{seed}.npz")
        base = DriftAgent(np.random.default_rng(0), **r["hyperparameters"])
        cost = tuple(z["final_cost"])
        averaged = base.with_parameters(z["averaged_theta"], cost)
        last = base.with_parameters(z["final_theta"], cost)
        if averaged.favourite_target() != r["favourite_index"]:
            raise RuntimeError(f"{level} seed {seed}: the rebuilt agent's favourite target is not the saved one")
        if not np.array_equal(last.greedy_actions(probes), z["probe_greedy"][-1]):
            raise RuntimeError(f"{level} seed {seed}: the rebuilt agent does not reproduce the logged probe actions")
        res_avg = evaluate.evaluate_policy(batch_policy(averaged), starts, paths, settings)
        res_last = evaluate.evaluate_policy(batch_policy(last), starts, paths, settings)
        res_ft = evaluate.evaluate_policy(evaluate.own_target_rebalance_policy(averaged), starts, paths, settings)
        v, vl, vf = res_avg["value"], res_last["value"], res_ft["value"]
        o, ar = ref["oracle"]["value"], ref["always_rebalance"]["value"]
        values.update({f"seed{seed}_averaged": v, f"seed{seed}_last_iterate": vl, f"seed{seed}_favourite_rebalance": vf})
        c = r["convergence"]
        rows.append({
            "seed": seed,
            "agent": _stats(res_avg), "last_iterate": _stats(res_last), "favourite_rebalance": _stats(res_ft),
            "agent_minus_oracle": gap(v, o), "agent_minus_always_rebalance": gap(v, ar),
            "agent_minus_trade_once": gap(v, ref["trade_once"]["value"]), "agent_minus_myopic": gap(v, ref["myopic"]["value"]),
            "agent_minus_favourite_rebalance": gap(v, vf), "favourite_rebalance_minus_always_rebalance": gap(vf, ar),
            "last_iterate_minus_oracle": gap(vl, o), "averaged_minus_last_iterate": gap(v, vl),
            "favourite_target": r["favourite_target"], "final_c1": r["final_c1"], "final_c2": r["final_c2"],
            "converged": c["converged"], "converged_from_block": c["converged_from_block"],
            "converged_from_episode": c["converged_from_episode"],
            "converged_from_daily_observations": c["converged_from_daily_observations"],
            "verdicts": {
                # the 2026-11-12 primary (probe) rule only; its secondary (within 25 bps of the oracle) is not computed
                "d12_2026_11_12_original_primary": r["last_iterate"]["convergence"]["original_all_probes_unchanged"],
                "d12_2026_11_19_last_iterate": (r["last_iterate"]["convergence"]["primary_favourite_within_one_step"]
                                                and r["last_iterate"]["convergence"]["secondary_exact_action_95pct_2026_11_19"]),
                "d12_2026_11_19_tail_average": (c["primary_favourite_within_one_step"]
                                                and c["secondary_exact_action_95pct_2026_11_19"]),
                "d12_2026_11_20": c["converged"]},
        })

    def across(key):
        x = np.array([row[key]["mean"] for row in rows])
        return {"mean": float(x.mean()), "sd_across_seeds": float(x.std(ddof=1)) if len(x) > 1 else None,
                "se_across_seeds": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                "per_seed": x.round(4).tolist()}

    gap_keys = GAP_KEYS if rows else []

    def pooled(a_key, b):
        """The seed-averaged policy value path by path, against b: its paired SE
        is the path-sampling error the across-seed SE cannot see, because every
        seed is scored on the same paths."""
        a = np.mean([values[f"seed{s}_{a_key}"] for s in sorted(records)], axis=0)
        return gap(a, b)
    pooled_gaps = {} if not rows else {
        "agent_minus_oracle": pooled("averaged", ref["oracle"]["value"]),
        "agent_minus_always_rebalance": pooled("averaged", ref["always_rebalance"]["value"]),
        "favourite_rebalance_minus_always_rebalance": pooled("favourite_rebalance", ref["always_rebalance"]["value"]),
        "agent_minus_trade_once": pooled("averaged", ref["trade_once"]["value"])}
    converged = [row for row in rows if row["converged"]]
    c_true = (settings.kappa / spec.BLOCK_LENGTH, settings.lam / (2 * spec.BLOCK_LENGTH))
    summary = {
        "what": f"TC drift final runs at {level}: the tail-averaged agent and the reference policies on common paths "
                "(TC_DRIFT_DESIGN.md §6, §7). Gaps are paired over paths, bps/yr; across-seed figures treat each "
                "seed's mean gap as one observation. c_true and the oracle are benchmarks, never given to the agent.",
        "level": level, "kappa": settings.kappa, "lam": settings.lam, "gamma": settings.gamma, "c_true": c_true,
        "m_eval": m, "horizon": config.H_EVAL, "seeds": sorted(records), "missing_seeds": missing,
        "best_target": spec.target_grid()[best].round(2).tolist(),
        "references": {k: _stats(v) for k, v in ref.items()},
        "oracle_minus_always_rebalance": gap(ref["oracle"]["value"], ref["always_rebalance"]["value"]),
        "across_seeds": {k: across(k) for k in gap_keys},
        "seed_averaged_path_gaps": pooled_gaps,
        "convergence": {
            "n_converged_2026_11_20": len(converged), "n_seeds": len(rows),
            "verdict_counts": {v: int(sum(row["verdicts"][v] for row in rows)) for v in (rows[0]["verdicts"] if rows else {})},
            "converged_from_episode_median": (float(np.median([row["converged_from_episode"] for row in converged]))
                                              if converged else None),
            "converged_from_daily_observations_median": (
                float(np.median([row["converged_from_daily_observations"] for row in converged])) if converged else None)},
        "per_seed": rows,
        "eval_seconds": round(time.perf_counter() - t0, 1),
        "provenance": _provenance(extra_files=[table_path]),
    }
    np.savez_compressed(level_dir / "values.npz", starts=starts, **values)
    _write_json(level_dir / "summary.json", summary)
    with open(level_dir / "per_seed.csv", "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["seed", "agent_value_bps", "agent_minus_oracle", "agent_minus_oracle_se",
                    "agent_minus_always_rebalance", "agent_minus_always_rebalance_se",
                    "agent_minus_favourite_rebalance", "agent_minus_favourite_rebalance_se",
                    "favourite_rebalance_minus_always_rebalance", "favourite_rebalance_minus_always_rebalance_se",
                    "hold_fraction", "turnover_per_block", *(f"favourite_{t.lower()}" for t in spec.ASSETS),
                    "c1", "c2", "converged", "converged_from_episode"])
        for row in rows:
            g = lambda k: (round(row[k]["mean"], 4), round(row[k]["se"], 4))
            w.writerow([row["seed"], round(row["agent"]["mean_value_bps"], 4), *g("agent_minus_oracle"),
                        *g("agent_minus_always_rebalance"), *g("agent_minus_favourite_rebalance"),
                        *g("favourite_rebalance_minus_always_rebalance"), round(row["agent"]["hold_fraction"], 4),
                        round(row["agent"]["turnover_per_block"], 6), *row["favourite_target"],
                        row["final_c1"], row["final_c2"], row["converged"], row["converged_from_episode"]])
    return summary


def report(out_dir: Path = OUT_DIR) -> list:
    """One row per scored level: the headline gaps across seeds and the convergence count."""
    rows = []
    for level in config.LEVELS:
        path = Path(out_dir) / level / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        a = s["across_seeds"]
        rows.append({"level": level, "n_seeds": len(s["seeds"]),
                     "agent_minus_oracle": a["agent_minus_oracle"]["mean"],
                     "agent_minus_oracle_se_across_seeds": a["agent_minus_oracle"]["se_across_seeds"],
                     "agent_minus_always_rebalance": a["agent_minus_always_rebalance"]["mean"],
                     "favourite_rebalance_minus_always_rebalance": a["favourite_rebalance_minus_always_rebalance"]["mean"],
                     "oracle_minus_always_rebalance": s["oracle_minus_always_rebalance"]["mean"],
                     "agent_hold_fraction": float(np.mean([r["agent"]["hold_fraction"] for r in s["per_seed"]])),
                     "oracle_hold_fraction": s["references"]["oracle"]["hold_fraction"],
                     "n_converged": s["convergence"]["n_converged_2026_11_20"]})
    _write_json(Path(out_dir) / "report.json", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--level", choices=config.LEVELS, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", type=Path, default=OUT_DIR)
    q = sub.add_parser("queue")
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--out", type=Path, default=OUT_DIR)
    q.add_argument("--levels", nargs="+", choices=config.LEVELS, default=list(QUEUE_ORDER))
    e = sub.add_parser("evaluate")
    e.add_argument("--level", choices=config.LEVELS, required=True)
    e.add_argument("--out", type=Path, default=OUT_DIR)
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.command == "train":
        rec = train_one(args.level, args.seed, args.out)
        print(json.dumps({k: rec[k] for k in ("level", "seed", "favourite_target", "final_c1", "final_c2",
                                              "train_seconds")}), flush=True)
    elif args.command == "queue":
        results = queue(args.workers, args.out, args.levels)
        sys.exit(1 if any(code for _, _, code in results) else 0)
    elif args.command == "evaluate":
        print(json.dumps(evaluate_level(args.level, args.out)["across_seeds"], indent=2), flush=True)
    else:
        print(json.dumps(report(args.out), indent=2), flush=True)
