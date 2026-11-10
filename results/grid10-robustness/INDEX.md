# Results index

Written 2026-11-10T00:41:54+00:00 from commit `badbf923f7f637757ac6c758ece64f6438824bea`. Full-scale settings.

**Post-hoc run `grid10-robustness`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a larger learning gap against the best grid portfolio than run all-three's at every γ, exactly-best share near zero, discretization gap 5.7 -> 1.3 bps/yr at γ = 5 and unchanged at γ = 3 and 8

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid10_B63_N26M_eps3 | 5 | 10% (43,758) | 1.3 | 10.8 (4.4 to 19.0) | 12.1 (5.7 to 20.3) | 0% | 0% (never) | 797.0M (567.6M to 1413.1M) | 100% (797.0M) | 100% (1634.0M) | 0.543 | [baseline_g5_grid10_B63_N26M_eps3](baseline_g5_grid10_B63_N26M_eps3/evaluation/summary.json) |
| robust_g3_grid10_B63_N26M_eps3 | 3 | 10% (43,758) | 2.5 | 25.6 (0.0 to 62.3) | 28.0 (2.5 to 64.8) | 40% | 0% (never) | 834.1M (585.3M to 1407.4M) | 50% (819.6M) | 50% (1635.0M) | 0.406 | [robust_g3_grid10_B63_N26M_eps3](robust_g3_grid10_B63_N26M_eps3/evaluation/summary.json) |
| robust_g8_grid10_B63_N26M_eps3 | 8 | 10% (43,758) | 6.1 | 6.4 (0.0 to 10.6) | 12.5 (6.1 to 16.7) | 10% | 0% (never) | 643.5M (489.5M to 1413.1M) | 100% (643.5M) | 100% (1632.1M) | 0.725 | [robust_g8_grid10_B63_N26M_eps3](robust_g8_grid10_B63_N26M_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
