# Results index

Written 2026-11-10T00:41:11+00:00 from commit `badbf923f7f637757ac6c758ece64f6438824bea`. Full-scale settings.

**Post-hoc run `all-three`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: the smallest learning gap of runs 1 to 4 at every γ, mostly from B; training regret a little above b63's

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_B63_N26M_eps3 | 5 | 20% (1,287) | 5.7 | 6.2 (0.0 to 29.4) | 11.9 (5.7 to 35.1) | 50% | 100% (705.2M) | 126.6M (80.6M to 292.9M) | 90% (85.0M) | 90% (302.4M) | 0.971 | [baseline_g5_grid20_B63_N26M_eps3](baseline_g5_grid20_B63_N26M_eps3/evaluation/summary.json) |
| robust_g3_grid20_B63_N26M_eps3 | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 2.5 (2.5 to 22.5) | 80% | 100% (371.0M) | 62.4M (28.4M to 277.2M) | 100% (56.1M) | 100% (207.2M) | 0.946 | [robust_g3_grid20_B63_N26M_eps3](robust_g3_grid20_B63_N26M_eps3/evaluation/summary.json) |
| robust_g8_grid20_B63_N26M_eps3 | 8 | 20% (1,287) | 6.1 | 0.0 (0.0 to 11.5) | 6.1 (6.1 to 17.6) | 90% | 100% (517.1M) | 64.3M (29.6M to 687.3M) | 100% (64.3M) | 100% (329.8M) | 0.988 | [robust_g8_grid20_B63_N26M_eps3](robust_g8_grid20_B63_N26M_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
