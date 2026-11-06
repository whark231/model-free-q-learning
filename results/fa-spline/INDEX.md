# Results index

Written 2026-11-06T10:42:55+00:00 from commit `bc311a387bd18ee8ab98b4e5415e49bd9e7497f0`. Full-scale settings.

**Post-hoc run `fa-spline`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a lower median Spearman correlation and top-10 recall over all portfolios than fa-quadratic at every γ, since most spline directions learn far more slowly than 13M episodes allow; no direction for the learning gap against fa-quadratic (that argument points to larger, the γ = 5 tuning seeds to smaller), nor against tabular-const or baseline-v1

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 7.111111111111112e-06; agent: linear (spline features, l2 0).

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_FAspline | 5 | 20% (1,287) | 5.7 | 5.3 (0.0 to 78.5) | 50% | 100% (173.9M) | 80% (62.9M) | [baseline_g5_grid20_FAspline](baseline_g5_grid20_FAspline/evaluation/summary.json) |
| robust_g3_grid20_FAspline | 3 | 20% (1,287) | 2.5 | 62.3 (0.0 to 68.3) | 30% | 100% (142.3M) | 30% (36.4M) | [robust_g3_grid20_FAspline](robust_g3_grid20_FAspline/evaluation/summary.json) |
| robust_g8_grid20_FAspline | 8 | 20% (1,287) | 6.1 | 51.6 (50.0 to 61.2) | 0% | 90% (219.5M) | 0% (never) | [robust_g8_grid20_FAspline](robust_g8_grid20_FAspline/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
