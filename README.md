# Model-Free Q-Learning and Optimal Portfolio Choice with Nine Assets

Wharton, Data Science and AI for Finance, Fall 2026 · William Harkless and Albert Aldrich

Can a model-free Q-learning agent recover a near-optimal nine-ETF portfolio when
it only observes simulated rewards, never the expected returns or covariances
behind them? We calibrate a Gaussian market from 20 years of ETF data, compute
the mean-variance optimum, train a tabular Q-learning agent on simulated
returns, and measure how close it gets.

## Requirements

Python 3.13 with numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, matplotlib 3.10.6,
yfinance 1.7.0 and pytest 8.4.2 (the versions that produced the committed data):

```bash
pip install -r requirements.txt
```

## Reproduce

From the repo root:

```bash
python -m pytest src/tests -q          # all tests pass
python -m src.data.clean               # rebuild data/processed/ from the committed raw data
python -m src.run_experiments          # train and evaluate every experiment (about 3.5 hours)
```

Do not re-run `src.data.download_raw` to reproduce results. The raw data
(downloaded 2026-09-24) is committed because Yahoo restates history, and a new
download would change the calibration.

## Seeds

All randomness comes from master seed **20261016** (`src/config.py`), split
into separate streams per run for returns, the agent's exploration, and
evaluation (`SeedPlan` in `src/environment.py`). Each run's manifest records
the seed plan.

## Expected outputs

`results/INDEX.md` summarizes every experiment and links to its files. Each
experiment folder holds the training logs (`run_*.npz`, `manifest.json`) and an
`evaluation/` folder with `summary.json`, CSV tables and figures. See
`EXPERIMENTS.md` for the full layout.

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

`PROJECT_PROSPECTUS.md` has the plan and task ownership.
