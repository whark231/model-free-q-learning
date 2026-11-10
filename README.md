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

Tested on two combinations (425 tests on 2026-11-09; 424 pass, and
`test_tune_fa.py::test_a_clear_winner_is_decided_on_three_seeds` fails on a
5e-15 rounding difference, reported to William):

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
python -m pytest src/tests -q          # see Requirements for the one known failure
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
`EXPERIMENTS.md` §7 lists the tabular variants (`eps-decay`, `b63`, `n26m`,
`all-three`, `grid10-robustness`) and §8 the constant-learning-rate runs
(`tabular-const`, `fa-quadratic`, `fa-spline`), each with its command,
expected direction and run time.

**Transaction costs** run through their own script with a configuration file
(`TRANSACTION_COSTS.md`):

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
(`EXPERIMENTS.md` §8.1).

## Expected outputs

| Where | What |
|---|---|
| `results/INDEX.md` | Summary of `baseline-v1`'s four experiments |
| `results/<run_id>/INDEX.md` | Summary of each post-hoc run |
| `results/<experiment>/evaluation/` | `summary.json`, CSV tables (per seed, weights, performance) and figures (grouped weight bars, learning curve, cumulative regret, losses by seed) |
| `results/comparison_g5_grid20_vs_grid25/`, `results/grid10-robustness/comparison_*` | The action-space resolution comparisons |
| `results/tc-quadratic-cost/` | The transaction-cost sweep, one JSON per cost level |
| `reports/<run_id>/report.html` | One self-contained report per run, compared with earlier runs (`reports/registry.json`) |

Training logs (`run_*.npz`) are not committed; every summary and report is.
See `EXPERIMENTS.md` §4 for the folder layout.

## Pipeline

| Step | Code | Documentation |
|---|---|---|
| Download and clean nine ETFs and the risk-free rate | `src/data/` | `DATA_CLEANING.md` |
| Simulated market from the calibrated μ and Σ | `src/environment.py` | `ENVIRONMENT.md` |
| Benchmarks: closed form, long-only optimum, best grid portfolio | `src/benchmarks/mean_variance.py` | |
| Action grid and reward | `src/action_space/grid.py`, `src/rewards.py` | |
| Tabular Q-learning agent | `src/agent.py` | |
| Training runner | `src/train.py` | |
| Evaluation against the benchmarks | `src/evaluate.py` | `EVALUATION.md` |
| Experiment settings and batch runner | `src/config.py`, `src/run_experiments.py` | `EXPERIMENTS.md` |
| Function approximation: features, agent, step-size tuning | `src/features.py`, `src/fa_agent.py`, `src/tune_fa.py` | `FA_INTEGRATION_NOTES.md`, `EXPERIMENTS.md` §8 |
| Transaction costs: stateful agent, grid oracle, sweep report | `src/transaction_costs.py`, `src/tc_sweep_report.py` | `TRANSACTION_COSTS.md` |
| Run reports | `src/reporting/` | `reports/README.md` |

`PROJECT_PROSPECTUS.md` has the plan and task ownership.
`AUDIT_2026-11-06.md` lists the open issues found in the 2026-11-06 audit.
