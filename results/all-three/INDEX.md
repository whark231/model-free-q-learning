# Results index

Written 2026-10-28T12:32:43+00:00 from commit `81814e667c2c199d6337bb30bcf7cc9ea2e71f0c`. Full-scale settings.

**Post-hoc run `all-three`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: the smallest learning gap of runs 1 to 4 at every γ, mostly from B; training regret a little above b63's

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_B63_N26M_eps3 | 5 | 20% (1,287) | 5.7 | 6.2 (0.0 to 29.4) | 50% | 100% (705.2M) | 90% (85.0M) | [baseline_g5_grid20_B63_N26M_eps3](baseline_g5_grid20_B63_N26M_eps3/evaluation/summary.json) |
| robust_g3_grid20_B63_N26M_eps3 | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 80% | 100% (371.0M) | 100% (56.1M) | [robust_g3_grid20_B63_N26M_eps3](robust_g3_grid20_B63_N26M_eps3/evaluation/summary.json) |
| robust_g8_grid20_B63_N26M_eps3 | 8 | 20% (1,287) | 6.1 | 0.0 (0.0 to 11.5) | 90% | 100% (517.1M) | 100% (64.3M) | [robust_g8_grid20_B63_N26M_eps3](robust_g8_grid20_B63_N26M_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
