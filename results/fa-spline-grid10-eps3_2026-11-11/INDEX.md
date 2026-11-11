# Results index

Written 2026-11-11T16:18:09+00:00 from commit `c0e549454f256b69cf7b97756b36ae658bd81949`. Full-scale settings.

**Post-hoc run `fa-spline-grid10-eps3`**, registered 2026-11-11 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: against grid10-robustness (same B, episodes, ε and seeds; paired, 2 SE): a median Spearman correlation over all 43,758 portfolios of at least 0.9 at every γ; at least 6 of 10 seeds meeting the primary rule at every γ (a constant alpha settles the greedy portfolio, so this measures stability); no direction for the learning gap at any γ, top-10 recall, exactly-best share or regret; discretization gap identical

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 2.7775746373405417e-05; agent: linear (spline features, l2 0).

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid10_B63_N26M_eps3_FAspline | 5 | 10% (43,758) | 1.3 | 66.4 (58.2 to 140.9) | 67.6 (59.5 to 142.2) | 0% | 0% (never) | 1586.7M (1144.1M to 1634.2M) | 0% (never) | 0% (never) | 0.976 | [baseline_g5_grid10_B63_N26M_eps3_FAspline](baseline_g5_grid10_B63_N26M_eps3_FAspline/evaluation/summary.json) |
| robust_g3_grid10_B63_N26M_eps3_FAspline | 3 | 10% (43,758) | 2.5 | 137.6 (94.2 to 207.4) | 140.1 (96.7 to 209.9) | 0% | 0% (never) | 1601.1M (1507.0M to 1634.2M) | 0% (never) | 0% (never) | 0.963 | [robust_g3_grid10_B63_N26M_eps3_FAspline](robust_g3_grid10_B63_N26M_eps3_FAspline/evaluation/summary.json) |
| robust_g8_grid10_B63_N26M_eps3_FAspline | 8 | 10% (43,758) | 6.1 | 59.6 (1.3 to 108.0) | 65.7 (7.4 to 114.1) | 0% | 0% (never) | 1557.4M (1360.8M to 1629.8M) | 10% (1535.3M) | 10% (1534.9M) | 0.983 | [robust_g8_grid10_B63_N26M_eps3_FAspline](robust_g8_grid10_B63_N26M_eps3_FAspline/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
