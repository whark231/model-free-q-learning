# Results index

Written 2026-11-10T02:05:45+00:00 from commit `3322e03b04b837f6a9cff1a0485a7cc1704039e7`. Full-scale settings.

**Post-hoc run `b63`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a smaller learning gap at every γ; exactly-best share, training regret and discretization gap about unchanged

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_B63 | 5 | 20% (1,287) | 5.7 | 0.0 (0.0 to 15.9) | 5.7 (5.7 to 21.6) | 60% | 90% (434.9M) | 77.5M (24.6M to 625.0M) | 100% (44.1M) | 100% (217.5M) | 0.931 | [baseline_g5_grid20_B63](baseline_g5_grid20_B63/evaluation/summary.json) |
| robust_g3_grid20_B63 | 3 | 20% (1,287) | 2.5 | 10.0 (0.0 to 20.0) | 12.5 (2.5 to 22.5) | 50% | 90% (327.2M) | 47.6M (18.9M to 127.9M) | 100% (26.5M) | 100% (233.9M) | 0.872 | [robust_g3_grid20_B63](robust_g3_grid20_B63/evaluation/summary.json) |
| robust_g8_grid20_B63 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 29.5) | 11.9 (6.1 to 35.6) | 50% | 100% (301.1M) | 42.2M (18.9M to 190.3M) | 80% (36.2M) | 80% (219.2M) | 0.971 | [robust_g8_grid20_B63](robust_g8_grid20_B63/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
