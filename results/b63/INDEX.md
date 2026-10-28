# Results index

Written 2026-10-28T22:31:55+00:00 from commit `81814e667c2c199d6337bb30bcf7cc9ea2e71f0c`. Full-scale settings.

**Post-hoc run `b63`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: a smaller learning gap at every γ; exactly-best share, training regret and discretization gap about unchanged

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_B63 | 5 | 20% (1,287) | 5.7 | 0.0 (0.0 to 15.9) | 60% | 90% (434.9M) | 100% (44.1M) | [baseline_g5_grid20_B63](baseline_g5_grid20_B63/evaluation/summary.json) |
| robust_g3_grid20_B63 | 3 | 20% (1,287) | 2.5 | 10.0 (0.0 to 20.0) | 50% | 90% (327.2M) | 100% (26.5M) | [robust_g3_grid20_B63](robust_g3_grid20_B63/evaluation/summary.json) |
| robust_g8_grid20_B63 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 29.5) | 50% | 100% (301.1M) | 80% (36.2M) | [robust_g8_grid20_B63](robust_g8_grid20_B63/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
