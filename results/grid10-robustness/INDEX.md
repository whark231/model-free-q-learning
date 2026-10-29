# Results index

Written 2026-10-29T14:03:24+00:00 from commit `e13a9a6d1f63788d80a8f0d93168d3f3142df38f`. Full-scale settings.

**Post-hoc run `grid10-robustness`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7). Expected beforehand: a larger learning gap against the best grid portfolio than run all-three's at every γ, exactly-best share near zero, discretization gap 5.7 -> 1.3 bps/yr at γ = 5 and unchanged at γ = 3 and 8

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid10_B63_N26M_eps3 | 5 | 10% (43,758) | 1.3 | 10.8 (4.4 to 19.0) | 0% | 0% (never) | 100% (797.0M) | [baseline_g5_grid10_B63_N26M_eps3](baseline_g5_grid10_B63_N26M_eps3/evaluation/summary.json) |
| robust_g3_grid10_B63_N26M_eps3 | 3 | 10% (43,758) | 2.5 | 25.6 (0.0 to 62.3) | 40% | 0% (never) | 50% (819.6M) | [robust_g3_grid10_B63_N26M_eps3](robust_g3_grid10_B63_N26M_eps3/evaluation/summary.json) |
| robust_g8_grid10_B63_N26M_eps3 | 8 | 10% (43,758) | 6.1 | 6.4 (0.0 to 10.6) | 10% | 0% (never) | 100% (643.5M) | [robust_g8_grid10_B63_N26M_eps3](robust_g8_grid10_B63_N26M_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
