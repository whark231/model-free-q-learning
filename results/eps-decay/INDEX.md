# Results index

Written 2026-10-28T06:42:15+00:00 from commit `81814e667c2c199d6337bb30bcf7cc9ea2e71f0c`. Full-scale settings.

**Post-hoc run `eps-decay`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: learning gap and exactly-best share unchanged within noise; training regret about 15 bps/yr higher at γ = 5

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_eps3 | 5 | 20% (1,287) | 5.7 | 10.5 (0.0 to 37.2) | 40% | 60% (243.2M) | 60% (59.5M) | [baseline_g5_grid20_eps3](baseline_g5_grid20_eps3/evaluation/summary.json) |
| robust_g3_grid20_eps3 | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 80% | 80% (194.0M) | 100% (46.1M) | [robust_g3_grid20_eps3](robust_g3_grid20_eps3/evaluation/summary.json) |
| robust_g8_grid20_eps3 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 51.6) | 50% | 60% (160.5M) | 80% (62.1M) | [robust_g8_grid20_eps3](robust_g8_grid20_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
