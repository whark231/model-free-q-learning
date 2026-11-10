# Results index

Written 2026-11-10T00:41:05+00:00 from commit `badbf923f7f637757ac6c758ece64f6438824bea`. Full-scale settings.

**Post-hoc run `n26m`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a slightly smaller learning gap and a slightly higher exactly-best share at every γ, likely within noise

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_N26M | 5 | 20% (1,287) | 5.7 | 10.5 (0.0 to 15.9) | 16.3 (5.7 to 21.6) | 30% | 80% (348.9M) | 73.9M (24.8M to 151.2M) | 100% (57.3M) | 100% (269.7M) | 0.895 | [baseline_g5_grid20_N26M](baseline_g5_grid20_N26M/evaluation/summary.json) |
| robust_g3_grid20_N26M | 3 | 20% (1,287) | 2.5 | 20.0 (0.0 to 62.3) | 22.5 (2.5 to 64.8) | 40% | 90% (302.6M) | 58.5M (25.2M to 185.0M) | 90% (57.6M) | 90% (278.9M) | 0.823 | [robust_g3_grid20_N26M](robust_g3_grid20_N26M/evaluation/summary.json) |
| robust_g8_grid20_N26M | 8 | 20% (1,287) | 6.1 | 0.0 (0.0 to 27.2) | 6.1 (6.1 to 33.3) | 70% | 90% (298.2M) | 58.3M (16.0M to 152.6M) | 90% (57.2M) | 90% (246.2M) | 0.956 | [robust_g8_grid20_N26M](robust_g8_grid20_N26M/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
