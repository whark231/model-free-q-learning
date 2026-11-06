# Results index

Written 2026-11-06T10:10:15+00:00 from commit `bc311a387bd18ee8ab98b4e5415e49bd9e7497f0`. Full-scale settings.

**Post-hoc run `fa-quadratic`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a smaller mean learning gap than tabular-const at every γ, and a higher median Spearman correlation and top-10 recall over all portfolios; against baseline-v1 no direction is predicted. Reference model, never the discovery result

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 0.00032; agent: linear (quadratic features, l2 0).

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_FAquad | 5 | 20% (1,287) | 5.7 | 22.6 (15.9 to 29.4) | 0% | 20% (243.1M) | 50% (257.8M) | [baseline_g5_grid20_FAquad](baseline_g5_grid20_FAquad/evaluation/summary.json) |
| robust_g3_grid20_FAquad | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 60% | 0% (never) | 100% (135.1M) | [robust_g3_grid20_FAquad](robust_g3_grid20_FAquad/evaluation/summary.json) |
| robust_g8_grid20_FAquad | 8 | 20% (1,287) | 6.1 | 55.7 (55.7 to 70.8) | 0% | 90% (246.1M) | 0% (never) | [robust_g8_grid20_FAquad](robust_g8_grid20_FAquad/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
