# Results index

Written 2026-11-10T02:05:52+00:00 from commit `3322e03b04b837f6a9cff1a0485a7cc1704039e7`. Full-scale settings.

**Post-hoc run `eps-decay`**, registered 2026-10-27 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: learning gap and exactly-best share unchanged within noise; training regret about 15 bps/yr higher at γ = 5

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: sample_average; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_eps3 | 5 | 20% (1,287) | 5.7 | 10.5 (0.0 to 37.2) | 16.3 (5.7 to 42.9) | 40% | 60% (243.2M) | 74.0M (30.6M to 146.4M) | 60% (59.5M) | 60% (222.2M) | 0.862 | [baseline_g5_grid20_eps3](baseline_g5_grid20_eps3/evaluation/summary.json) |
| robust_g3_grid20_eps3 | 3 | 20% (1,287) | 2.5 | 0.0 (0.0 to 20.0) | 2.5 (2.5 to 22.5) | 80% | 80% (194.0M) | 46.1M (22.6M to 103.4M) | 100% (46.1M) | 100% (189.9M) | 0.758 | [robust_g3_grid20_eps3](robust_g3_grid20_eps3/evaluation/summary.json) |
| robust_g8_grid20_eps3 | 8 | 20% (1,287) | 6.1 | 5.7 (0.0 to 51.6) | 11.9 (6.1 to 57.7) | 50% | 60% (160.5M) | 56.0M (30.0M to 230.8M) | 80% (62.1M) | 80% (174.7M) | 0.938 | [robust_g8_grid20_eps3](robust_g8_grid20_eps3/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
