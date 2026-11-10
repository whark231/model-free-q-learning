# Results index

Written 2026-11-10T02:06:04+00:00 from commit `3322e03b04b837f6a9cff1a0485a7cc1704039e7`. Full-scale settings.

**Post-hoc run `tabular-const`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a larger mean learning gap than baseline-v1 at every γ, and fewer seeds meeting the primary convergence rule at every γ (baseline-v1: 9, 8, 9 at γ = 5, 3, 8), because a constant alpha never averages the reward noise away (Q-value noise about 49 bps/yr)

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 0.00032; agent: tabular.

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_TABconst | 5 | 20% (1,287) | 5.7 | 42.0 (10.5 to 109.9) | 47.8 (16.3 to 115.7) | 0% | 100% (53.2M) | 27.3M (1.0M to 183.8M) | 40% (19.1M) | 40% (19.1M) | 0.813 | [baseline_g5_grid20_TABconst](baseline_g5_grid20_TABconst/evaluation/summary.json) |
| robust_g3_grid20_TABconst | 3 | 20% (1,287) | 2.5 | 102.6 (0.0 to 205.2) | 105.1 (2.5 to 207.6) | 20% | 80% (50.9M) | 37.2M (2.0M to 236.4M) | 20% (119.2M) | 20% (119.0M) | 0.703 | [robust_g3_grid20_TABconst](robust_g3_grid20_TABconst/evaluation/summary.json) |
| robust_g8_grid20_TABconst | 8 | 20% (1,287) | 6.1 | 51.5 (11.5 to 61.2) | 57.7 (17.6 to 67.3) | 0% | 100% (109.7M) | 83.8M (1.4M to 187.2M) | 10% (87.6M) | 10% (87.5M) | 0.917 | [robust_g8_grid20_TABconst](robust_g8_grid20_TABconst/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
