# Results index

Written 2026-11-09T21:03:22+00:00 from commit `1e8f80f76d8c987e66b2c647e175aeaf358441e0-dirty`. Full-scale settings.

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Near-optimal is read at the 10,000-episode checkpoints; "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20 | 5 | 20% (1,287) | 5.7 | 14.1 (0.0 to 47.3) | 19.8 (5.7 to 53.1) | 30% | 90% (185.1M) | 47.1M (18.0M to 80.6M) | 70% (25.8M) | 0.821 | [baseline_g5_grid20](baseline_g5_grid20/evaluation/summary.json) |
| resolution_g5_grid25 | 5 | 25% (495) | 11.7 | 2.5 (0.0 to 30.7) | 14.3 (11.7 to 42.5) | 50% | 100% (101.2M) | 18.5M (7.6M to 44.2M) | 90% (9.6M) | 0.926 | [resolution_g5_grid25](resolution_g5_grid25/evaluation/summary.json) |
| robust_g3_grid20 | 3 | 20% (1,287) | 2.5 | 10.0 (0.0 to 81.1) | 12.5 (2.5 to 83.6) | 50% | 80% (192.6M) | 26.4M (12.0M to 58.0M) | 60% (22.2M) | 0.710 | [robust_g3_grid20](robust_g3_grid20/evaluation/summary.json) |
| robust_g8_grid20 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 53.1) | 11.9 (6.1 to 59.2) | 50% | 90% (153.2M) | 27.9M (12.2M to 86.8M) | 70% (25.8M) | 0.920 | [robust_g8_grid20](robust_g8_grid20/evaluation/summary.json) |

Resolution comparison: [comparison_g5_grid20_vs_grid25](comparison_g5_grid20_vs_grid25/convergence_by_resolution.png).

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
