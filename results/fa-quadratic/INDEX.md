# Results index

Written 2026-11-10T02:06:11+00:00 from commit `3322e03b04b837f6a9cff1a0485a7cc1704039e7`. Full-scale settings.

**Post-hoc run `fa-quadratic`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a smaller mean learning gap than tabular-const at every γ, and a higher median Spearman correlation and top-10 recall over all portfolios; against baseline-v1 no direction is predicted. Reference model, never the discovery result

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 0.00032; agent: linear (quadratic features, l2 0).

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_FAquad | 5 | 20% (1,287) | 5.7 | 22.6 (15.9 to 29.4) | 28.4 (21.6 to 35.1) | 0% | 20% (243.1M) | 248.5M (183.6M to 260.0M) | 50% (257.8M) | 50% (257.8M) | 0.915 | [baseline_g5_grid20_FAquad](baseline_g5_grid20_FAquad/evaluation/summary.json) |
| robust_g3_grid20_FAquad | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 2.5 (2.5 to 22.5) | 60% | 0% (never) | 254.0M (217.8M to 260.0M) | 100% (135.1M) | 100% (242.8M) | 0.854 | [robust_g3_grid20_FAquad](robust_g3_grid20_FAquad/evaluation/summary.json) |
| robust_g8_grid20_FAquad | 8 | 20% (1,287) | 6.1 | 55.7 (55.7 to 70.8) | 61.8 (61.8 to 76.9) | 0% | 90% (246.1M) | 140.1M (90.2M to 231.4M) | 0% (never) | 0% (never) | 0.951 | [robust_g8_grid20_FAquad](robust_g8_grid20_FAquad/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
