# Results

**TL;DR / In plain terms:** the report's experiment results come from files in
this folder; the data statistics in report §2 come from `data/processed/`. The four baseline experiments sit at the top level; each
later run has its own subfolder. Large training logs (`run_*.npz`) are not
committed, but manifests, evaluation summaries, CSV tables and figures are.

## Baseline (`baseline-v1`: settings fixed 2026-10-16, run 2026-10-22)

| Path | What | Report |
|---|---|---|
| `INDEX.md`, `index.json`, `settings.json` | Summary and settings of the four experiments | §5 |
| `baseline_g5_grid20/` | γ = 5, 20% grid: the headline result | §5 |
| `resolution_g5_grid25/` | γ = 5, 25% grid: the second action-space resolution | §5.5 |
| `robust_g3_grid20/`, `robust_g8_grid20/` | γ = 3 and γ = 8: the risk-aversion robustness check | §5, §6.2 |
| `comparison_g5_grid20_vs_grid25/` | The 20% against 25% grid comparison | §5.5 |
| `benchmarks/` | Budget-only closed-form weights and their check against a numerical solve (`python -m src.benchmarks.closed_form_table`) | §3 |
| `full_run_*.log` | Console logs of the full-scale runs | |

Each experiment folder has `manifest.json` and an `evaluation/` folder with
`summary.json`, per-seed CSV tables and the figures (weights, learning curve,
cumulative regret, losses by seed). `docs/methods/EVALUATION.md` defines every
field.

## Post hoc runs (each labeled post hoc in the report)

| Path | What | Report |
|---|---|---|
| `eps-decay/`, `b63/`, `n26m/`, `all-three/` | Exploration schedule, 63-day episodes, 26M episodes, and all three together | §6.2 |
| `grid10-robustness/` | The 10% grid (43,758 portfolios) at γ = 3, 5, 8, with comparisons against the 20% grid | §6.1 |
| `tabular-const/` | Tabular agent with a constant step size, the like-for-like reference for function approximation | §6.3 |
| `fa-quadratic/`, `fa-spline/` | Function approximation on the 20% grid | §6.3 |
| `fa-spline-grid10-eps3_2026-11-11/`, `fa-spline-grid10-eps10_2026-11-11/` | Spline function approximation on the 10% grid | §6.3 |
| `tuning/` | Step-size search for the constant-step runs, on master seed 20261103 | §6.3 |
| `tc-quadratic-cost/` | First transaction-cost design (negative result) | §6.4, Table A2, Figure A13 |
| `tc-drift/` | Transaction costs with drift: `oracle/` (tables and summary), `checks/`, `tuning/`, and `runs/<level>/` (the 40 final runs; [`tc-drift/runs/README.md`](tc-drift/runs/README.md) has the results and the verdicts on the twelve registered expectations) | §6.4, Appendix C |

Each post hoc tabular or function-approximation folder has its own `INDEX.md`.
Commands and registered expectations for every run are in
`docs/methods/EXPERIMENTS.md` §7 and §8 and `docs/methods/TC_DRIFT_DESIGN.md`.
