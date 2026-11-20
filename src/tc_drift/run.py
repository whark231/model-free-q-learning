"""Final agent runs for transaction costs with drift (Albert; TC_DRIFT_DESIGN.md §7, §9).

In plain terms: train the agent at every cost level on every reported seed,
save what it learned, then score each level's agents on one shared set of
simulated paths next to the oracle and the reference policies. The agent never
sees mu, Sigma, the costs or the oracle; they are used here only to draw the
evaluation paths and to score.

    python -m src.tc_drift.run all                     # 4 levels x 10 seeds, 7 at a time, then scoring
    python -m src.tc_drift.run train --level aum_1b --seed 0
    python -m src.tc_drift.run evaluate --level aum_1b

`all` skips runs whose record already exists, so it resumes after a stop. It
starts every job as its own process with BLAS pinned to one thread.

Needs the oracle tables in results/tc-drift/oracle/<level>.npz (git-ignored):
regenerate with `python -m src.tc_drift.oracle_runs`, which also writes the
committed summary.json they are checked against.

What is scored (per level, on the reported seed's evaluation paths, m = config.M_EVAL
starts, H = 90 blocks, every policy on the same paths):
- the agent's reported policy: greedy on the tail-averaged theta (design §6,
  2026-11-19), with the costs it ended with;
- the last-iterate policy (reported alongside);
- rebalancing every block to the averaged agent's favourite target (TC-3),
  which splits agent minus oracle into the agent's own decisions, the
  favourite-target error and the oracle's timing lead;
- the oracle, the myopic oracle (discount 0), always rebalancing to the best
  target and trading once to it then holding.
D12 (design §6, frozen 2026-11-20) is applied with tune.convergence, and all
four versions' verdicts are reported. Benchmark-derived numbers (the true c1
and c2, the favourite target's utility loss) are diagnostics only.

Outputs under results/tc-drift/runs/<level>/: seed<i>.json (committed) and
seed<i>.npz (git-ignored training log with every theta snapshot), then
summary.json and per_seed.csv (committed) and values.npz (git-ignored per-path
values of every policy).
"""

import argparse
import csv
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.environment import SeedPlan
from src.hashing import file_sha256
from src.tc_drift import config, env, evaluate, spec, train, tune
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/tc-drift/runs"
ORACLE_DIR = REPO / "results/tc-drift/oracle"
CODE_FILES = ("src/tc_drift/run.py", "src/tc_drift/tune.py", "src/tc_drift/agent.py", "src/tc_drift/features.py",
              "src/tc_drift/train.py", "src/tc_drift/env.py", "src/tc_drift/evaluate.py", "src/tc_drift/config.py",
              "src/tc_drift/spec.py", "src/tc_drift/calibration.py", "src/environment.py")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
LOG_KEYS = ("checkpoint_block", "theta", "favourite_target", "probe_greedy", "hold_fraction",
            "explicit_hold_fraction", "mean_cost_bps", "mean_turnover", "mean_net_reward")
H = config.H_EVAL


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def provenance(master_seed: int = spec.MASTER_SEED) -> dict:
    """Commit, whether src/ had uncommitted changes, LF-normalized sha256 of
    the files the run's logic imports, versions, threads and the seed plan."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "src_dirty": bool(git("status", "--porcelain", "--", "src")),
            "sha256": {f: file_sha256(REPO / f) for f in CODE_FILES},
            "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
            "seed_plan": SeedPlan(master_seed, config.N_SEEDS).as_dict(),
            "threads": {k: os.environ.get(k) for k in THREAD_VARS}}


def _check_level_seed(level: str, seed: int) -> None:
    if level not in config.LEVELS:
        raise ValueError(f"level must be one of {config.LEVELS}, got {level!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < config.N_SEEDS:
        raise ValueError(f"seed must be in [0, {config.N_SEEDS}), got {seed!r}")


def _abs(path) -> Path:
    """Paths relative to the repo root, wherever the command was started."""
    path = Path(path)
    return path if path.is_absolute() else (REPO / path).resolve()


def run_files(out_dir: Path, level: str, seed: int) -> tuple:
    folder = _abs(out_dir) / level
    return folder / f"seed{seed}.npz", folder / f"seed{seed}.json"


def train_one(level: str, seed: int, out_dir: Path = OUT_DIR, n_blocks: int | None = None,
              checkpoint_every: int | None = None, hyperparameters: dict | None = None,
              master_seed: int = spec.MASTER_SEED) -> dict:
    """Train one reported run and save its log (with every theta snapshot) and
    its record. The record is written last, so its presence marks a finished run."""
    _check_level_seed(level, seed)
    if config.pending():
        raise RuntimeError(f"config values still pending: {config.pending()}")
    n_blocks = config.N_BLOCKS if n_blocks is None else int(n_blocks)
    checkpoint_every = config.CHECKPOINT_EVERY if checkpoint_every is None else int(checkpoint_every)
    hp = dict(config.HYPERPARAMETERS if hyperparameters is None else hyperparameters)

    market = env.load_market()
    settings = cost_settings()[level]
    returns_rng, agent_rng, start_rng = config.run_rngs(seed, master_seed)
    agent = DriftAgent(agent_rng, **hp)
    t0 = time.perf_counter()
    log = train.train(agent, market, settings, n_blocks=n_blocks, returns_rng=returns_rng, start_rng=start_rng,
                      start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                      checkpoint_every=checkpoint_every, probe_states=config.probe_states())
    seconds = time.perf_counter() - t0
    if not np.array_equal(log["theta"][-1], agent.theta):
        raise RuntimeError("the last theta snapshot is not the trained agent's theta")

    arrays = {k: log[k] for k in LOG_KEYS}
    arrays.update(c1=log["diagnostics"]["cost_linear"], c2=log["diagnostics"]["cost_quadratic"],
                  epsilon=log["diagnostics"]["epsilon"], n_updates=log["diagnostics"]["n_updates"])
    npz_path, json_path = run_files(out_dir, level, seed)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)

    fav = agent.favourite_target()
    record = {
        "level": level, "seed": seed, "master_seed": master_seed, "hyperparameters": hp,
        "n_blocks": n_blocks, "checkpoint_every": checkpoint_every, "restart_every": config.RESTART_EVERY,
        "block_length": spec.BLOCK_LENGTH, "simulated_days": log["simulated_days"], "n_restarts": log["n_restarts"],
        "episodes": log["n_restarts"] + 1,      # starts, including the first (44,445 at 4M blocks)
        "final_c1": agent.cost_linear, "final_c2": agent.cost_quadratic, "final_epsilon": agent.epsilon,
        "final_favourite_index": fav, "final_favourite_target": spec.target_grid()[fav].round(2).tolist(),
        "train_seconds": round(seconds, 1), "ms_per_block": round(seconds / n_blocks * 1e3, 4),
        "npz_sha256": file_sha256(npz_path), "provenance": provenance(master_seed),
    }
    if n_blocks != config.N_BLOCKS or checkpoint_every != config.CHECKPOINT_EVERY or hyperparameters is not None:
        record["smoke"] = True        # not a reported run: beta and the burn-in stay on the 4M-block schedule
    _write_text(json_path, json.dumps(record, indent=2) + "\n")
    return record


def load_run(out_dir: Path, level: str, seed: int) -> tuple:
    """(record, log arrays, the rebuilt last-iterate agent). Checks that the
    rebuilt agent is the trained one: the same favourite target and the same
    greedy action on every probe state as the log's last checkpoint."""
    npz_path, json_path = run_files(out_dir, level, seed)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if file_sha256(npz_path) != record["npz_sha256"]:
        raise RuntimeError(f"{npz_path} does not match its record's hash")
    z = dict(np.load(npz_path))
    base = DriftAgent(np.random.default_rng(0), **record["hyperparameters"])
    last = base.with_parameters(z["theta"][-1], (z["c1"][-1], z["c2"][-1]))
    if not last.favourite_target() == int(z["favourite_target"][-1]) == record["final_favourite_index"]:
        raise RuntimeError(f"{json_path}: the rebuilt agent's favourite target is not the logged one")
    probes = config.probe_states()
    if not np.array_equal([last.greedy_action(x) for x in probes], z["probe_greedy"][-1]):
        raise RuntimeError(f"{json_path}: the rebuilt agent's probe actions differ from the log")
    return record, z, last


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_oracle_tables(level: str, oracle_dir: Path = ORACLE_DIR) -> dict:
    """The oracle's and the myopic oracle's policy tables for a level, with the
    file's hash and the oracle summary's provenance for the record."""
    oracle_dir = _abs(oracle_dir)
    path = oracle_dir / f"{level}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m src.tc_drift.oracle_runs` first")
    z = np.load(path)
    summary = json.loads((oracle_dir / "summary.json").read_text(encoding="utf-8"))
    # The tables are git-ignored and regenerated locally, so check them against the
    # committed summary: the same costs, and the same mean value over the 2% starts.
    entry, settings = summary["settings"][level], cost_settings()[level]
    for name in ("kappa", "lam"):
        if not np.isclose(getattr(settings, name), entry[name], rtol=1e-12, atol=0):
            raise RuntimeError(f"{path}: {name} differs from the oracle summary's")
    value = (1 - spec.DISCOUNT) * z["V"][spec.target_lattice_index()].mean() * spec.TRADING_DAYS * spec.BPS
    if not np.isclose(value, entry["oracle_value_bps"], rtol=0, atol=1e-8):
        raise RuntimeError(f"{path}: its V does not reproduce the oracle summary's value ({value} vs "
                           f"{entry['oracle_value_bps']}); regenerate the tables")
    return {"policy_table": z["policy_table"], "myopic_policy_table": z["myopic_policy_table"],
            "npz_sha256": file_sha256(path), "npz_modified_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
            "checked_against_summary_created_utc": summary.get("created_utc"), "value_check_bps": value}


def _gap(a: np.ndarray, b: np.ndarray) -> dict:
    return evaluate.paired_gap(a, b, H)


def _stats(result: dict) -> dict:
    return {"mean_value_bps": result["mean_value_bps"], "hold_fraction": result["mean_hold_fraction"],
            "turnover_per_block": float(result["turnover_per_block"].mean()),
            "cost_bps_per_block": float(result["cost_bps_per_block"].mean()),
            "mean_trade_size": result["mean_trade_size"],
            "mean_weights": np.round(result["mean_weights"], 4).tolist()}


def d12_versions(averaged_conv: dict, last_conv: dict) -> dict:
    """The four D12 verdicts design §6 asks the final runs to report.
    2026-11-12: every probe keeps its exact greedy action (last iterate; the
    rule's primary only: its evaluation-only secondary, within 25 bps/yr of
    the oracle from some checkpoint on, is not computed).
    2026-11-19, exact action: favourite within one step and 95% of probes keep
    the exact action (last iterate). 2026-11-19, tail average: the same two
    rules on the tail-averaged policy. 2026-11-20 (the operative rule): the
    favourite within one step and 95% of probes' post-decision holdings within
    one step, on the tail-averaged policy."""
    return {
        "2026_11_12_original_primary_last_iterate": last_conv["original_all_probes_unchanged"],
        "2026_11_19_exact_action_last_iterate": bool(last_conv["primary_favourite_within_one_step"]
                                                     and last_conv["secondary_exact_action_95pct_2026_11_19"]),
        "2026_11_19_tail_average": bool(averaged_conv["primary_favourite_within_one_step"]
                                        and averaged_conv["secondary_exact_action_95pct_2026_11_19"]),
        "2026_11_20_operative": bool(averaged_conv["converged"]),
    }


def score_seed(record: dict, z: dict, last: DriftAgent, starts, paths, settings, refs: dict,
               best: int, U: np.ndarray, probes: np.ndarray) -> tuple:
    """One seed's policies on the level's paths: (row for the summary, per-path values)."""
    n_blocks = record["n_blocks"]
    averaged, averaged_conv, _ = tune.averaged_policy(last, z, n_blocks, probes)
    last_conv = tune.convergence(z, n_blocks, probes)
    runs = {
        "agent": evaluate.evaluate_policy(evaluate.agent_policy(averaged), starts, paths, settings),
        "last_iterate": evaluate.evaluate_policy(evaluate.agent_policy(last), starts, paths, settings),
        "own_target": evaluate.evaluate_policy(evaluate.own_target_rebalance_policy(averaged), starts, paths,
                                               settings),
    }
    v = {k: r["value"] for k, r in runs.items()}
    ref = {k: r["value"] for k, r in refs.items()}

    # TC-3 split; an identity on every path, so check it.
    split = (v["agent"] - v["own_target"]) + (v["own_target"] - ref["always_rebalance"]) \
        - (ref["oracle"] - ref["always_rebalance"])
    if not np.allclose(split, v["agent"] - ref["oracle"], rtol=0, atol=1e-9):
        raise RuntimeError("the TC-3 split does not add up")

    targets = spec.target_grid()
    fav = averaged.favourite_target()
    c_true = (settings.kappa / spec.BLOCK_LENGTH, settings.lam / (2 * spec.BLOCK_LENGTH))
    row = {
        "seed": record["seed"],
        "agent": _stats(runs["agent"]), "last_iterate": _stats(runs["last_iterate"]),
        "own_target": _stats(runs["own_target"]),
        "gaps_bps": {
            "agent_minus_oracle": _gap(v["agent"], ref["oracle"]),
            "agent_minus_always_rebalance": _gap(v["agent"], ref["always_rebalance"]),
            "agent_minus_trade_once_then_hold": _gap(v["agent"], ref["trade_once_then_hold"]),
            "agent_minus_myopic": _gap(v["agent"], ref["myopic"]),
            "agent_minus_last_iterate": _gap(v["agent"], v["last_iterate"]),
            "last_iterate_minus_oracle": _gap(v["last_iterate"], ref["oracle"]),
            "split_agent_minus_own_target": _gap(v["agent"], v["own_target"]),
            "split_own_target_minus_always_rebalance": _gap(v["own_target"], ref["always_rebalance"]),
        },
        "favourite_target_index": fav,
        "favourite_target": targets[fav].round(2).tolist(),
        "last_iterate_favourite_target": targets[last.favourite_target()].round(2).tolist(),
        "d12": d12_versions(averaged_conv, last_conv),
        "convergence_tail_average": averaged_conv,
        "convergence_last_iterate": last_conv,
        "final_c1": record["final_c1"], "final_c2": record["final_c2"],
        "diagnostics_benchmark": {
            "c1_over_true": record["final_c1"] / c_true[0] if c_true[0] else None,
            "c2_over_true": record["final_c2"] / c_true[1] if c_true[1] else None,
            "favourite_target_utility_loss_bps": float((U[best] - U[fav]) * spec.TRADING_DAYS * spec.BPS),
        },
        "train_seconds": record["train_seconds"], "ms_per_block": record["ms_per_block"],
    }
    return row, v


def _across(rows: list, get) -> dict:
    x = np.array([get(r) for r in rows], dtype=float)
    return {"mean": float(x.mean()), "sd_across_seeds": float(x.std(ddof=1)) if len(x) > 1 else None,
            "se_across_seeds": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
            "min": float(x.min()), "max": float(x.max())}


def evaluate_level(level: str, out_dir: Path = OUT_DIR, m: int | None = None, seeds=None,
                   oracle_tables: dict | None = None, master_seed: int = spec.MASTER_SEED) -> dict:
    """Score every finished seed of a level and write summary.json,
    per_seed.csv and values.npz. Needs all N_SEEDS seeds unless `seeds` is given."""
    if level not in config.LEVELS:
        raise ValueError(f"level must be one of {config.LEVELS}, got {level!r}")
    m = config.M_EVAL if m is None else int(m)
    seeds = list(range(config.N_SEEDS)) if seeds is None else list(seeds)
    missing = [s for s in seeds if not run_files(out_dir, level, s)[1].exists()]
    if missing:
        raise FileNotFoundError(f"{level}: no finished run for seeds {missing}")
    loaded = [load_run(out_dir, level, s) for s in seeds]
    for key in ("n_blocks", "checkpoint_every", "hyperparameters", "master_seed"):
        if len({json.dumps(r[key], sort_keys=True) for r, _, _ in loaded}) > 1:
            raise ValueError(f"{level}: runs differ in {key}")
    for key in ("commit", "src_dirty", "sha256"):
        if len({json.dumps(r["provenance"][key], sort_keys=True) for r, _, _ in loaded}) > 1:
            raise ValueError(f"{level}: runs were trained on different code ({key} differs)")
    if _abs(out_dir) == OUT_DIR and any(r.get("smoke") for r, _, _ in loaded):
        raise ValueError(f"{level}: a smoke run is in the reported results folder")
    if any(r["master_seed"] != master_seed for r, _, _ in loaded):
        raise ValueError(f"{level}: the runs were trained under another master seed")

    t0 = time.perf_counter()
    market = env.load_market()
    settings = cost_settings()[level]
    tables = load_oracle_tables(level) if oracle_tables is None else oracle_tables
    best = evaluate.best_static_target(market, settings.gamma)
    targets = spec.target_grid()
    U = targets @ market.mu - 0.5 * settings.gamma * np.einsum("ai,ij,aj->a", targets, market.sigma, targets)
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(master_seed), H, m)
    refs = {
        "oracle": evaluate.evaluate_policy(evaluate.oracle_table_policy(tables["policy_table"]), starts, paths,
                                           settings),
        "myopic": evaluate.evaluate_policy(evaluate.oracle_table_policy(tables["myopic_policy_table"]), starts,
                                           paths, settings),
        "always_rebalance": evaluate.evaluate_policy(evaluate.always_rebalance_policy(best), starts, paths,
                                                     settings),
        "trade_once_then_hold": evaluate.evaluate_policy(evaluate.never_trade_policy(best), starts, paths,
                                                         settings),
    }
    ref_values = {k: r["value"] for k, r in refs.items()}
    oracle_lead = _gap(ref_values["oracle"], ref_values["always_rebalance"])

    probes = config.probe_states()
    rows, values = [], {f"ref_{k}": v for k, v in ref_values.items()}
    for record, z, last in loaded:
        row, v = score_seed(record, z, last, starts, paths, settings, refs, best, U, probes)
        rows.append(row)
        values.update({f"seed{record['seed']}_{k}": x for k, x in v.items()})
        print(f"[{level}] seed {record['seed']}: agent - oracle "
              f"{row['gaps_bps']['agent_minus_oracle']['mean']:+.2f} "
              f"({row['gaps_bps']['agent_minus_oracle']['se']:.2f}), "
              f"converged {row['d12']['2026_11_20_operative']}", flush=True)

    gap_names = list(rows[0]["gaps_bps"])
    se_agent_oracle = float(np.mean([r["gaps_bps"]["agent_minus_oracle"]["se"] for r in rows]))
    summary = {
        "what": "TC-drift final agent runs, one cost level (TC_DRIFT_DESIGN.md §7, §9). Post hoc relative to "
                "baseline-v1. Gaps are paired over the m evaluation paths, bps/yr (spec.annualized_bps with "
                "H = 90). 'agent' is the tail-averaged policy (design §6); 'last_iterate' is reported alongside. "
                "diagnostics_benchmark fields use mu, Sigma, kappa and lambda and are not results the agent saw.",
        "level": level, "kappa": settings.kappa, "lam": settings.lam, "gamma": settings.gamma,
        "m": m, "horizon": H, "seeds": seeds, "n_blocks": loaded[0][0]["n_blocks"],
        "hyperparameters": loaded[0][0]["hyperparameters"],
        "best_target": targets[best].round(2).tolist(),
        "references": {k: _stats(r) for k, r in refs.items()},
        "oracle_minus_always_rebalance": oracle_lead,
        "oracle_minus_trade_once_then_hold": _gap(ref_values["oracle"], ref_values["trade_once_then_hold"]),
        "oracle_minus_myopic": _gap(ref_values["oracle"], ref_values["myopic"]),
        "across_seeds": {
            **{f"{g}_mean": _across(rows, lambda r, g=g: r["gaps_bps"][g]["mean"]) for g in gap_names},
            "agent_hold_fraction": _across(rows, lambda r: r["agent"]["hold_fraction"]),
            "agent_turnover_per_block": _across(rows, lambda r: r["agent"]["turnover_per_block"]),
            "agent_cost_bps_per_block": _across(rows, lambda r: r["agent"]["cost_bps_per_block"]),
            "favourite_target_utility_loss_bps": _across(
                rows, lambda r: r["diagnostics_benchmark"]["favourite_target_utility_loss_bps"]),
            "d12_converged_count": {k: int(sum(r["d12"][k] for r in rows)) for k in rows[0]["d12"]},
            "train_hours_mean": float(np.mean([r["train_seconds"] for r in rows]) / 3600),
        },
        "d10": {"rule": "agent-minus-oracle paired SE below a third of the oracle's lead over always rebalancing",
                "target_se": oracle_lead["mean"] / 3, "mean_agent_minus_oracle_se": se_agent_oracle,
                "applicable": bool(oracle_lead["mean"] > 2 * oracle_lead["se"]),
                "met": bool(oracle_lead["mean"] > 0 and se_agent_oracle < oracle_lead["mean"] / 3)},
        "per_seed": rows,
        "oracle_tables": {k: tables.get(k) for k in ("npz_sha256", "npz_modified_utc",
                                                     "checked_against_summary_created_utc", "value_check_bps")},
        "eval_seconds": round(time.perf_counter() - t0, 1),
        "provenance": provenance(master_seed),
    }
    if any(r.get("smoke") for r, _, _ in loaded):
        summary["smoke"] = True

    folder = _abs(out_dir) / level
    tmp = folder / "values.tmp.npz"
    np.savez_compressed(tmp, **values)
    os.replace(tmp, folder / "values.npz")
    _write_text(folder / "summary.json", json.dumps(summary, indent=2) + "\n")
    _write_text(folder / "per_seed.csv", _per_seed_csv(rows))
    return summary


def _per_seed_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    gaps = list(rows[0]["gaps_bps"])
    w.writerow(["seed", *[f"{g}_{s}" for g in gaps for s in ("mean", "se")], "agent_hold_fraction",
                "agent_turnover_per_block", "agent_cost_bps_per_block", "favourite_target", "final_c1", "final_c2",
                "c1_over_true", "c2_over_true", "favourite_target_utility_loss_bps", *rows[0]["d12"],
                "converged_from_block", "converged_from_episode", "converged_from_daily_observations",
                "train_seconds"])
    for r in rows:
        conv = r["convergence_tail_average"]
        d = r["diagnostics_benchmark"]
        w.writerow([r["seed"], *[r["gaps_bps"][g][s] for g in gaps for s in ("mean", "se")],
                    r["agent"]["hold_fraction"], r["agent"]["turnover_per_block"], r["agent"]["cost_bps_per_block"],
                    " ".join(map(str, r["favourite_target"])), r["final_c1"], r["final_c2"], d["c1_over_true"],
                    d["c2_over_true"], d["favourite_target_utility_loss_bps"], *r["d12"].values(),
                    conv.get("converged_from_block"), conv.get("converged_from_episode"),
                    conv.get("converged_from_daily_observations"), r["train_seconds"]])
    return buf.getvalue()


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Everything, in parallel processes
# ---------------------------------------------------------------------------

def _pinned_env() -> dict:
    return {**os.environ, **{k: "1" for k in THREAD_VARS}}


def _run_jobs(commands: list, workers: int, log_dir: Path) -> list:
    """Run each (name, argv) as its own process, at most `workers` at a time;
    return the names that failed."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(commands), {}, []
    while pending or running:
        while pending and len(running) < workers:
            name, argv = pending.pop(0)
            handle = open(log_dir / f"{name}.log", "w", encoding="utf-8", newline="\n")
            running[name] = (subprocess.Popen(argv, cwd=REPO, env=_pinned_env(), stdout=handle,
                                              stderr=subprocess.STDOUT), handle)
            print(f"{time.strftime('%H:%M:%S')} started {name}", flush=True)
        time.sleep(5)
        for name, (proc, handle) in list(running.items()):
            if proc.poll() is not None:
                handle.close()
                del running[name]
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {status} {name}", flush=True)
    return failed


def run_all(levels=None, seeds=None, workers: int = 7, out_dir: Path = OUT_DIR) -> None:
    """Train every unfinished (level, seed), the headline level first, then
    score every level. Stops before scoring if any training job failed."""
    if config.pending():
        raise RuntimeError(f"config values still pending: {config.pending()}")
    levels = list(levels or sorted(config.LEVELS, key=lambda lv: lv != config.HEADLINE_LEVEL))
    seeds = list(range(config.N_SEEDS)) if seeds is None else list(seeds)
    for level in levels:
        load_oracle_tables(level)                      # fail now, not after hours of training
    py = [sys.executable, "-m", "src.tc_drift.run"]
    out = ["--out", str(out_dir)]
    jobs = [(f"train_{lv}_seed{s}", py + ["train", "--level", lv, "--seed", str(s)] + out)
            for s in seeds for lv in levels if not run_files(out_dir, lv, s)[1].exists()]
    out_dir = _abs(out_dir)
    log_dir = out_dir / "logs"
    print(f"{len(jobs)} training runs to do, {workers} at a time; logs in {log_dir}", flush=True)
    failed = _run_jobs(jobs, workers, log_dir)
    if failed:
        raise RuntimeError(f"training failed: {failed}; see {log_dir}")
    seed_args = ["--seeds", *map(str, seeds)]
    jobs = [(f"evaluate_{lv}", py + ["evaluate", "--level", lv] + seed_args + out) for lv in levels]
    failed = _run_jobs(jobs, min(workers, len(jobs)), log_dir)
    if failed:
        raise RuntimeError(f"scoring failed: {failed}; see {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--level", required=True, choices=config.LEVELS)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", type=Path, default=OUT_DIR)
    e = sub.add_parser("evaluate")
    e.add_argument("--level", required=True, choices=config.LEVELS)
    e.add_argument("--seeds", type=int, nargs="*", default=None)
    e.add_argument("--out", type=Path, default=OUT_DIR)
    a = sub.add_parser("all")
    a.add_argument("--levels", nargs="*", default=None, choices=config.LEVELS)
    a.add_argument("--seeds", type=int, nargs="*", default=None)
    a.add_argument("--workers", type=int, default=7)
    a.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.command == "train":
        rec = train_one(args.level, args.seed, args.out)
        print(json.dumps({k: rec[k] for k in ("level", "seed", "train_seconds", "ms_per_block", "final_c1",
                                              "final_c2", "final_favourite_target")}), flush=True)
    elif args.command == "evaluate":
        s = evaluate_level(args.level, args.out, seeds=args.seeds)
        print(json.dumps({"level": s["level"], "across_seeds": s["across_seeds"], "d10": s["d10"]}, indent=2))
    else:
        run_all(args.levels, args.seeds, args.workers, args.out)
