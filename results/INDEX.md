# Results index

Written 2026-10-22T09:00:18+00:00 from commit `80a6e9c3f7b79d6d94a7c8d134836f7046d652b3`. Full-scale settings.

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20 | 5 | 20% (1,287) | 5.7 | 14.1 (0.0 to 47.3) | 30% | 90% (185.1M) | 70% (25.8M) | [baseline_g5_grid20](baseline_g5_grid20/evaluation/summary.json) |
| resolution_g5_grid25 | 5 | 25% (495) | 11.7 | 2.5 (0.0 to 30.7) | 50% | 100% (101.2M) | 90% (9.6M) | [resolution_g5_grid25](resolution_g5_grid25/evaluation/summary.json) |
| robust_g3_grid20 | 3 | 20% (1,287) | 2.5 | 10.0 (0.0 to 81.1) | 50% | 80% (192.6M) | 60% (22.2M) | [robust_g3_grid20](robust_g3_grid20/evaluation/summary.json) |
| robust_g8_grid20 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 53.1) | 50% | 90% (153.2M) | 70% (25.8M) | [robust_g8_grid20](robust_g8_grid20/evaluation/summary.json) |

Resolution comparison: [comparison_g5_grid20_vs_grid25](comparison_g5_grid20_vs_grid25/convergence_by_resolution.png).

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
