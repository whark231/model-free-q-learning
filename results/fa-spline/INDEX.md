# Results index

Written 2026-11-10T00:41:58+00:00 from commit `badbf923f7f637757ac6c758ece64f6438824bea`. Full-scale settings.

**Post-hoc run `fa-spline`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a lower median Spearman correlation and top-10 recall over all portfolios than fa-quadratic at every γ, since most spline directions learn far more slowly than 13M episodes allow; no direction for the learning gap against fa-quadratic (that argument points to larger, the γ = 5 tuning seeds to smaller), nor against tabular-const or baseline-v1

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 7.111111111111112e-06; agent: linear (spline features, l2 0).

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_FAspline | 5 | 20% (1,287) | 5.7 | 5.3 (0.0 to 78.5) | 11.0 (5.7 to 84.2) | 50% | 100% (173.9M) | 146.9M (53.6M to 206.4M) | 80% (62.9M) | 80% (73.1M) | 0.950 | [baseline_g5_grid20_FAspline](baseline_g5_grid20_FAspline/evaluation/summary.json) |
| robust_g3_grid20_FAspline | 3 | 20% (1,287) | 2.5 | 62.3 (0.0 to 68.3) | 64.8 (2.5 to 70.8) | 30% | 100% (142.3M) | 102.0M (24.0M to 233.0M) | 30% (36.4M) | 30% (73.8M) | 0.945 | [robust_g3_grid20_FAspline](robust_g3_grid20_FAspline/evaluation/summary.json) |
| robust_g8_grid20_FAspline | 8 | 20% (1,287) | 6.1 | 51.6 (50.0 to 61.2) | 57.7 (56.1 to 67.3) | 0% | 90% (219.5M) | 165.2M (93.0M to 229.4M) | 0% (never) | 0% (never) | 0.957 | [robust_g8_grid20_FAspline](robust_g8_grid20_FAspline/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
