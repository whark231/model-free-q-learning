"""Run every experiment in src/config.py: train, evaluate, compare, index.

    python -m src.run_experiments                      # everything, full scale (about 3.5 hours)
    python -m src.run_experiments --only baseline_g5_grid20
    python -m src.run_experiments --episodes 200000 --runs 2 --results-dir results/smoke   # quick check
    python -m src.run_experiments --dry-run            # print the plan only
    python -m src.run_experiments --run b63 --results-dir results/b63   # a post-hoc variant (config.POST_HOC_RUNS)

Each experiment trains into <results-dir>/<name>/ (one .npz per seed plus
manifest.json) and is evaluated into <results-dir>/<name>/evaluation/.

A results folder holds one set of settings only. The first call writes
<results-dir>/settings.json; a later call with any different setting is
refused, so a smoke test can never overwrite or relabel a full-scale run.
Within that, an experiment whose training finished with exactly the current
settings is reused, so an interrupted batch restarts with the same command.
A partly trained folder is refused with instructions.

Without --run, every setting is the pre-registered baseline and the settings
written are exactly those of tag baseline-v1. With --run, the variant's B,
training budget, ε schedule and grid replace the baseline's, its id is added
to settings.json, and its experiments carry its suffix.

The grid-resolution comparison goes to <results-dir>/<comparison>/, and
INDEX.md plus index.json summarize every evaluated experiment with links.
"""

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config as cfg
from src.evaluate import compare_experiments, evaluate_experiment
from src.train import FA_FIELD_DEFAULTS, _git_commit, config_record, run_experiment

COMPARISON_NAME = "comparison_g5_grid20_vs_grid25"


def batch_settings(n_episodes: int, n_runs: int, n_eval_episodes: int, log_every: int,
                   run: cfg.RunVariant | None = None) -> dict:
    """Everything that must be identical for experiments to share a results folder.
    Without a variant this is exactly the baseline-v1 settings.json, so the
    committed baseline results still match it."""
    settings = {
        "gammas": list(cfg.GAMMAS), "gamma_baseline": cfg.GAMMA_BASELINE,
        "episode_length": cfg.EPISODE_LENGTH, "n_episodes": n_episodes, "n_runs": n_runs,
        "master_seed": cfg.MASTER_SEED, "epsilon0": cfg.EPSILON0, "beta": cfg.beta_for(n_episodes),
        "alpha_rule": cfg.ALPHA_RULE, "q_init": cfg.Q_INIT, "discount": cfg.DISCOUNT,
        "log_every": log_every, "convergence_k": cfg.convergence_k(n_episodes),
        "primary_convergence_rule": cfg.PRIMARY_CONVERGENCE_RULE, "dq_tol": cfg.DQ_TOL,
        "near_optimal_bps": cfg.NEAR_OPTIMAL_BPS, "n_eval_episodes": n_eval_episodes,
        "increments": [cfg.INCREMENT_BASELINE, cfg.INCREMENT_COMPARISON],
    }
    if run is not None:
        settings.update({
            "episode_length": run.episode_length, "beta": cfg.beta_for(n_episodes, run.epsilon_at_halfway),
            "increments": [run.increment], "post_hoc_run": run.run_id,
            "epsilon_at_halfway": run.epsilon_at_halfway,
        })
    return settings


def check_results_dir(results_dir: Path, settings: dict) -> None:
    """Refuse a results folder that already holds different settings."""
    path = results_dir / "settings.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(existing) | set(settings) if existing.get(k) != settings.get(k))
        if diff:
            raise ValueError(
                f"{results_dir} already holds results made with different settings ({', '.join(diff)}). "
                "Use a different --results-dir; nothing was changed."
            )
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def training_status(run_dir: Path, train_cfg) -> str:
    """'new', 'finished' (same settings, all runs), or raise for anything else."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"{run_dir} has files but no manifest; delete that folder and run again")
        return "new"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = json.loads(json.dumps(config_record(train_cfg)))  # tabular: same keys as baseline-v1's manifests
    have = manifest["config"]
    # Compare every key on either side; a missing function-approximation key
    # means its tabular default, so a tabular config never matches an FA folder.
    diff = sorted(k for k in set(wanted) | set(have)
                  if have.get(k, FA_FIELD_DEFAULTS.get(k)) != wanted.get(k, FA_FIELD_DEFAULTS.get(k)))
    if diff:
        raise ValueError(f"{run_dir} was trained with different settings ({', '.join(diff)}); nothing was changed")
    done, expected = len(manifest["runs"]), manifest["config"]["n_runs"]
    if done < expected:
        raise ValueError(
            f"{run_dir} is partly trained ({done} of {expected} runs). Delete that one folder and run again; "
            "other experiments are unaffected"
        )
    return "finished"


def run_all(results_dir: Path = Path("results"), only=None, n_episodes: int | None = None,
            n_runs: int = cfg.N_RUNS, n_eval_episodes: int = cfg.N_EVAL_EPISODES,
            log_every: int = cfg.LOG_EVERY, dry_run: bool = False, agent_factory=None,
            run: cfg.RunVariant | None = None) -> dict:
    """run: a post-hoc variant from cfg.POST_HOC_RUNS, or None for the baseline.
    n_episodes defaults to the variant's (or the baseline's) training budget."""
    results_dir = Path(results_dir)
    full_episodes = run.n_episodes if run else cfg.N_EPISODES
    n_episodes = full_episodes if n_episodes is None else n_episodes
    all_experiments = cfg.experiments(run)
    names = [e.name for e in all_experiments]
    unknown = sorted(set(only or []) - set(names))
    if unknown:
        raise ValueError(f"unknown experiment(s) {unknown}; choose from {names}")
    selected = [e for e in all_experiments if not only or e.name in only]
    eval_cfg = cfg.eval_config(n_episodes, n_eval_episodes)
    plan = [(e, cfg.train_config(e, n_episodes, n_runs, log_every)) for e in selected]
    if dry_run:
        for e, tc in plan:
            print(f"{e.name}: gamma={tc.gamma}, grid={tc.increment}, {tc.n_runs} runs x {tc.n_episodes:,} "
                  f"episodes x B={tc.episode_length}; epsilon {e.epsilon_at_halfway:g} at halfway "
                  f"(beta={tc.beta:.4g}); K={eval_cfg.converge_k:,}; {e.purpose}")
        return {"planned": [e.name for e, _ in plan]}

    settings = batch_settings(n_episodes, n_runs, n_eval_episodes, log_every, run)
    check_results_dir(results_dir, settings)
    for experiment, train_cfg in plan:
        run_dir = results_dir / experiment.name
        if training_status(run_dir, train_cfg) == "finished":
            print(f"{experiment.name}: already trained with these settings, skipping training")
        else:
            kwargs = {"agent_factory": agent_factory} if agent_factory else {}
            run_experiment(train_cfg, run_dir, **kwargs)
        evaluate_experiment(run_dir, eval_cfg, overwrite=True)

    # The index covers every experiment evaluated in this folder so far. The
    # settings check above guarantees they all share one set of settings.
    entries = []
    for experiment in all_experiments:
        summary_path = results_dir / experiment.name / "evaluation" / "summary.json"
        if summary_path.exists():
            entries.append(_index_entry(experiment, json.loads(summary_path.read_text(encoding="utf-8"))))

    comparison = None
    if {"baseline_g5_grid20", "resolution_g5_grid25"} <= {e["name"] for e in entries}:
        out = results_dir / COMPARISON_NAME
        staging = results_dir / f".{COMPARISON_NAME}.new"
        if staging.exists():
            shutil.rmtree(staging)
        comparison = compare_experiments(
            [results_dir / "resolution_g5_grid25", results_dir / "baseline_g5_grid20"], ["25%", "20%"], eval_cfg, staging
        )
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)

    index = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "settings": settings,
        "full_scale": settings == batch_settings(full_episodes, cfg.N_RUNS, cfg.N_EVAL_EPISODES, cfg.LOG_EVERY, run),
        "post_hoc_run": None if run is None else {"run_id": run.run_id, "registered": run.registered,
                                                   "expected": run.expected},
        "experiments": entries,
        "comparison": {"folder": COMPARISON_NAME, "results": comparison} if comparison else None,
    }
    (results_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (results_dir / "INDEX.md").write_text(_index_markdown(index), encoding="utf-8")
    return index


def _index_entry(experiment, summary: dict) -> dict:
    learned = summary["learned"]
    conv = summary["convergence"]
    train = summary["training_config"]
    return {
        "name": experiment.name,
        "purpose": experiment.purpose,
        "gamma": train["gamma"],
        "increment": train["increment"],
        "n_actions": summary["n_actions"],
        "n_episodes": train["n_episodes"],
        "n_runs": train["n_runs"],
        "converge_k": summary["eval_config"]["converge_k"],
        "folder": experiment.name,
        "discretization_gap_bps": summary["discretization_gap"]["ce_loss_bps"],
        "learning_gap_bps": learned["ce_loss_vs_best_discrete_bps"],
        "share_exactly_best_discrete": learned["share_exactly_best_discrete"],
        "convergence": {rule: conv[rule] for rule in conv["rules"]},
    }


def _fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _index_markdown(index: dict) -> str:
    s = index["settings"]
    primary = s["primary_convergence_rule"]
    lines = [
        "# Results index",
        "",
        f"Written {index['written_utc']} from commit `{index['git_commit']}`. "
        f"{'Full-scale settings.' if index['full_scale'] else '**Reduced settings: a pipeline check, not report results.**'}",
        "",
        *([f"**Post-hoc run `{index['post_hoc_run']['run_id']}`**, registered {index['post_hoc_run']['registered']} "
           f"after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: "
           f"{index['post_hoc_run']['expected']}", ""] if index["post_hoc_run"] else []),
        f"γ ∈ {s['gammas']} (baseline {s['gamma_baseline']:g}), B = {s['episode_length']}, "
        f"{s['n_runs']} seeds × {s['n_episodes']:,} episodes, master seed {s['master_seed']}, "
        f"K = {s['convergence_k']:,} episodes (convergence rule: `{primary}`), ΔQ tolerance {s['dq_tol']:g}, "
        f"near-optimal line {s['near_optimal_bps']:g} bps/yr, {s['n_eval_episodes']:,} evaluation episodes.",
        "",
        "Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. "
        "For each rule: the share of seeds that met it, and in brackets the median simulated days "
        "(millions) among the seeds that did.",
        "",
        "| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | "
        "Converged (primary rule) | Near-optimal for good | Folder |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in index["experiments"]:
        gap = e["learning_gap_bps"]

        def obs(rule):
            median = rule["observations"]["median"]
            when = "never" if median is None else f"{median / 1e6:.1f}M"
            return f"{rule['share_of_seeds']:.0%} ({when})"

        lines.append(
            f"| {e['name']} | {e['gamma']:g} | {e['increment']:.0%} ({e['n_actions']:,}) | "
            f"{_fmt(e['discretization_gap_bps'])} | {_fmt(gap['median'])} ({_fmt(gap['min'])} to {_fmt(gap['max'])}) | "
            f"{e['share_exactly_best_discrete']:.0%} | {obs(e['convergence'][primary])} | "
            f"{obs(e['convergence']['near_optimal_for_good'])} | "
            f"[{e['folder']}]({e['folder']}/evaluation/summary.json) |"
        )
    if index["comparison"]:
        lines += ["", f"Resolution comparison: [{index['comparison']['folder']}]"
                      f"({index['comparison']['folder']}/convergence_by_resolution.png)."]
    lines += ["", "Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="default: results/ for the baseline, results/<run_id>/ with --run")
    parser.add_argument("--run", choices=sorted(cfg.POST_HOC_RUNS),
                        help="a post-hoc variant from config.POST_HOC_RUNS; omit for the baseline")
    parser.add_argument("--only", nargs="+", help="experiment names; --dry-run lists them")
    parser.add_argument("--episodes", type=int, default=None, help="default: the run's full training budget")
    parser.add_argument("--runs", type=int, default=cfg.N_RUNS)
    parser.add_argument("--eval-episodes", type=int, default=cfg.N_EVAL_EPISODES)
    parser.add_argument("--log-every", type=int, default=cfg.LOG_EVERY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = cfg.POST_HOC_RUNS[args.run] if args.run else None
    if args.results_dir is None:
        args.results_dir = Path("results") / run.run_id if run else Path("results")
    run_all(args.results_dir, args.only, args.episodes, args.runs, args.eval_episodes, args.log_every, args.dry_run,
            run=run)


if __name__ == "__main__":
    main()
