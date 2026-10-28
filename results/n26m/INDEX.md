# Results index

Written 2026-10-28T11:34:44+00:00 from commit `81814e667c2c199d6337bb30bcf7cc9ea2e71f0c`. Full-scale settings.

**Post-hoc run `n26m`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: a slightly smaller learning gap and a slightly higher exactly-best share at every γ, likely within noise

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_N26M | 5 | 20% (1,287) | 5.7 | 10.5 (0.0 to 15.9) | 30% | 80% (348.9M) | 100% (57.3M) | [baseline_g5_grid20_N26M](baseline_g5_grid20_N26M/evaluation/summary.json) |
| robust_g3_grid20_N26M | 3 | 20% (1,287) | 2.5 | 20.0 (0.0 to 62.3) | 40% | 90% (302.6M) | 90% (57.6M) | [robust_g3_grid20_N26M](robust_g3_grid20_N26M/evaluation/summary.json) |
| robust_g8_grid20_N26M | 8 | 20% (1,287) | 6.1 | 0.0 (0.0 to 27.2) | 70% | 90% (298.2M) | 90% (57.2M) | [robust_g8_grid20_N26M](robust_g8_grid20_N26M/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
