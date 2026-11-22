# Model-Free Q-Learning and Optimal Portfolio Choice with Nine Assets

Wharton, Data Science and AI for Finance, Fall 2026 · William Harkless and Albert Aldrich

Can a model-free Q-learning agent recover a near-optimal nine-ETF portfolio when
it only observes simulated rewards, never the expected returns or covariances
behind them? We calibrate a Gaussian market from 20 years of ETF data, compute
the mean-variance optimum, train a tabular Q-learning agent on simulated
returns, and measure how close it gets. Two bonus extensions build on this:
function approximation (a linear Q-function over portfolio features) and
transaction costs (a stateful agent that pays to rebalance).

## Requirements

Tested on two combinations. On 2026-11-20 all 758 tests passed on the Mac
(Python 3.13). On 2026-11-09 all 442 tests then in the suite passed on Windows
with Python 3.10, and on 2026-11-19 all 672 tests then in the suite passed there
(`docs/reviews/AUDIT_2026-11-06.md` §9.5); the tests added since have not been
run on Windows:

- Python 3.13 with numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, matplotlib 3.10.6,
  yfinance 1.7.0 and pytest 8.4.2 (the versions that produced the committed data)
- Python 3.10 with the versions pinned in `requirements.txt` (numpy 2.2.6,
  the newest available for Python 3.10)

`requirements.txt` installs the Python 3.10 combination:

```bash
pip install -r requirements.txt
```

**Windows:** `src.run_experiments` and `src.evaluate` now set UTF-8 output
themselves. `src.tc_sweep_report` does not yet, so before running it set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Reproduce

From the repo root:

```bash
python -m pytest src/tests -q          # all tests pass
python -m src.data.clean               # rebuild data/processed/ from the committed raw data
python -m src.run_experiments --results-dir results/repro   # baseline-v1: train and evaluate into a new folder
```

A fresh clone has each run's manifest and summaries but not its training
logs (`run_*.npz`, too large to commit), so `src.run_experiments` refuses to
re-evaluate a committed folder. To reproduce a run, train into a new folder
and compare with the committed summaries, for example
`python -m src.run_experiments --results-dir results/repro` against
`results/INDEX.md`. Training is seeded, so the numbers should match.

Do not re-run `src.data.download_raw` to reproduce results. The raw data
(downloaded 2026-09-24) is committed because Yahoo restates history, and a new
download would change the calibration.

**Post-hoc runs** use `--run <run_id>` into their own results folder, for
example `python -m src.run_experiments --run b63 --results-dir results/b63`.
`docs/methods/EXPERIMENTS.md` §7 lists the tabular variants (`eps-decay`, `b63`, `n26m`,
`all-three`, `grid10-robustness`) and §8 the constant-learning-rate runs
(`tabular-const`, `fa-quadratic`, `fa-spline`), each with its command,
expected direction and run time.

**Transaction costs with drift** (the reported TC extension,
`docs/methods/TC_DRIFT_DESIGN.md`) has its own commands. The oracle tables
(`results/tc-drift/oracle/*.npz`) are not committed, so rebuild them first:

```bash
python -m src.tc_drift.oracle_runs              # 20 min to 2.5 h, depending on the machine
git checkout results/tc-drift/oracle/summary.json   # keep the committed summary (see below)
python -m src.tc_drift.run all --workers 7      # 4 cost levels × 10 seeds, then scoring; about 3 h on a 10-core Mac
python -m src.tc_drift.run train --level aum_1b --seed 0   # one run
python -m src.tc_drift.run evaluate --level aum_1b         # score one level (all 10 seeds; --seeds 0 1 2 for fewer)
```

`oracle_runs` rewrites `summary.json`. Before restoring it, `git diff` should
show changes only in `provenance`, `created_utc` and `runtime_seconds`; any
other change means the tables did not reproduce. `run.py` checks the tables
against the committed summary, so restore it first. Levels are `zero`,
`aum_100m`, `aum_1b` and `aum_10b`. Results go to
`results/tc-drift/runs/<level>/`: `summary.json`, `per_seed.csv` and one
`seed<i>.json` record per run are small; `values.npz` and the per-seed
`.npz` training logs stay local. Console logs go to `results/tc-drift/runs/logs/`. Training and evaluation use the project's
master seed 20261016 (`src/tc_drift/spec.py`; tuning used 20261112), and `run all` skips any run that already
has a finished record, so an interrupted batch can be restarted.

**The first transaction-cost design** (kept as a negative result) runs through
its own script with a configuration file (`docs/methods/TRANSACTION_COSTS.md`):

```bash
python -m src.transaction_costs experiments/transaction-costs/quadratic-cost.json --dry-run
python -m src.tc_sweep_report      # rebuilds reports/tc-quadratic-cost/ from results/tc-quadratic-cost/
```

**Reports** are rebuilt from committed files only:

```bash
python -m src.reporting.build --all    # every reports/<run_id>/report.html
```

Extracting a new run (`python -m src.reporting.extract <run_id>`) needs that
run's training logs, which stay on the machine that trained it
(`reports/README.md`).

## Seeds

All reported randomness comes from master seed **20261016** (`src/config.py`),
split into separate streams per run for returns, the agent's exploration, and
evaluation (`SeedPlan` in `src/environment.py`). Each run's manifest records
the seed plan. Seed i draws the same simulated returns in every run, so runs
can be compared seed by seed. The learning rates for the constant-α runs were
tuned on a separate master seed, **20261103**, never on the reported one
(`docs/methods/EXPERIMENTS.md` §8.1). The transaction-cost agent was tuned on
**20261112** (`src/tc_drift/spec.py`, `TC_DRIFT_DESIGN.md` §9).

## Expected outputs

| Where | What |
|---|---|
| `results/INDEX.md` | Summary of `baseline-v1`'s four experiments |
| `results/<run_id>/INDEX.md` | Summary of each post-hoc run |
| `results/<experiment>/evaluation/` | `summary.json`, CSV tables (per seed, weights, performance) and figures (grouped weight bars, learning curve, cumulative regret, losses by seed) |
| `results/comparison_g5_grid20_vs_grid25/`, `results/grid10-robustness/comparison_*` | The action-space resolution comparisons |
| `results/tc-quadratic-cost/` | The first transaction-cost design's sweep, one JSON per cost level |
| `results/tc-drift/` | Transaction costs with drift: `oracle/summary.json`, `checks/`, `tuning/`, and `runs/<level>/` once the agent runs finish |
| `reports/<run_id>/report.html` | One self-contained report per run, compared with earlier runs (`reports/registry.json`) |

Training logs (`run_*.npz`) are not committed; every summary and report is.
See `docs/methods/EXPERIMENTS.md` §4 for the folder layout.

## Pipeline

| Step | Code | Documentation |
|---|---|---|
| Download and clean nine ETFs and the risk-free rate | `src/data/` | `docs/methods/DATA_CLEANING.md` |
| Simulated market from the calibrated μ and Σ | `src/environment.py` | `docs/methods/ENVIRONMENT.md` |
| Benchmarks: closed form, long-only optimum, best grid portfolio | `src/benchmarks/mean_variance.py` | |
| Action grid and reward | `src/action_space/grid.py`, `src/rewards.py` | |
| Tabular Q-learning agent | `src/agent.py` | |
| Training runner | `src/train.py` | |
| Evaluation against the benchmarks | `src/evaluate.py` | `docs/methods/EVALUATION.md` |
| Experiment settings and batch runner | `src/config.py`, `src/run_experiments.py` | `docs/methods/EXPERIMENTS.md` |
| Function approximation: features, agent, step-size tuning | `src/features.py`, `src/fa_agent.py`, `src/tune_fa.py` | `docs/methods/FA_INTEGRATION_NOTES.md`, `docs/methods/EXPERIMENTS.md` §8 |
| Transaction costs with drift: environment, oracle, agent, tuning, run driver | `src/tc_drift/` | `docs/methods/TC_DRIFT_DESIGN.md` |
| Transaction costs, first design (negative result) | `src/transaction_costs.py`, `src/tc_sweep_report.py` | `docs/methods/TRANSACTION_COSTS.md` |
| Run reports | `src/reporting/` | `reports/README.md` |

`docs/PROJECT_PROSPECTUS.md` has the plan and task ownership.
`docs/reviews/AUDIT_2026-11-06.md` holds the 2026-11-06 audit and the later status reviews; §9.6 lists everything still open.
`docs/README.md` lists every document.

## Repository layout

| Folder | What |
|---|---|
| `src/` | All code; tests in `src/tests/` |
| `data/` | `raw/` (committed, never re-downloaded), `interim/`, `processed/` (rebuilt by `src.data.clean`) |
| `experiments/` | Run configurations for the transaction-cost script |
| `results/` | Manifests, evaluation summaries, tables and figures. `baseline-v1` is the top level (`INDEX.md` and its four experiment folders); each later run has `results/<run_id>/` (the two `fa-spline-grid10-eps*` runs are in `results/<run_id>_2026-11-11/`); `tuning/` holds the FA step-size search |
| `reports/` | One HTML report per run with its extracted `data.json`, the run list `registry.json`, and the hand-built `full_run_2026-10-22/` report |
| `docs/` | Methods, reviews and audits, team notes, and the prospectus (`docs/README.md`) |
| `notebooks/` | An empty placeholder notebook; no reported result uses it |
| `no-ai-slop-main/` | The writing-style checklist the docs follow (`CLAUDE.md` §3.1) |

`assignment.pdf` (the assignment) and `CLAUDE.md` (the team's working agreement
with Claude) stay at the root.
