# Results index

Written 2026-11-12T00:20:32+00:00 from commit `8a840e2c111075989ec4195e9b8555d0c967f1dc`. Full-scale settings.

**Post-hoc run `fa-spline-grid10-eps10`**, registered 2026-11-11 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: against fa-spline-grid10-eps3 (same alpha and seeds; only ε at halfway differs, 10% against 3%; paired, 2 SE): a smaller mean learning gap at γ = 5; higher training regret at γ = 3 and 8; a higher median Spearman correlation at every γ; no direction for the gap at γ = 3 and 8 or regret at γ = 5. Against grid10-robustness descriptive only

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 63, 10 seeds × 26,000,000 episodes, master seed 20261016, K = 2,600,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 2.7775746373405417e-05; agent: linear (spline features, l2 0).

Losses are certainty-equivalent, in basis points per year. The learning gap is measured against the best portfolio on that grid, the total gap against the continuous long-only optimum (discretization plus learning, per seed). For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did. The primary rule's count includes its K-episode waiting window. Simulated days = episodes x B. Near-optimal for good is read at the 10,000-episode checkpoints; the every-episode version (post hoc) also counts brief switches between checkpoints, so it is later. "settled" (post hoc, descriptive) is the first checkpoint from which the final portfolio was greedy at every later checkpoint; a tabular seed always settles by the last checkpoint, so that column gives the median and range in millions of days, not a share. Q fit is the median Spearman correlation between the learned Q-values and every portfolio's true value.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Total gap, median (range) | Exactly best | Converged (primary rule) | Settled (post hoc), median (range) | Near-optimal for good (checkpoints) | Near-optimal every episode (post hoc) | Q fit | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid10_B63_N26M_eps10_FAspline | 5 | 10% (43,758) | 1.3 | 82.6 (62.0 to 112.8) | 83.8 (63.3 to 114.1) | 0% | 0% (never) | 1620.4M (1567.4M to 1636.1M) | 0% (never) | 0% (never) | 0.975 | [baseline_g5_grid10_B63_N26M_eps10_FAspline](baseline_g5_grid10_B63_N26M_eps10_FAspline/evaluation/summary.json) |
| robust_g3_grid10_B63_N26M_eps10_FAspline | 3 | 10% (43,758) | 2.5 | 154.7 (130.3 to 182.8) | 157.2 (132.8 to 185.3) | 0% | 0% (never) | 1607.1M (1501.9M to 1638.0M) | 0% (never) | 0% (never) | 0.964 | [robust_g3_grid10_B63_N26M_eps10_FAspline](robust_g3_grid10_B63_N26M_eps10_FAspline/evaluation/summary.json) |
| robust_g8_grid10_B63_N26M_eps10_FAspline | 8 | 10% (43,758) | 6.1 | 66.7 (27.6 to 85.5) | 72.8 (33.7 to 91.6) | 0% | 0% (never) | 1625.7M (1551.1M to 1638.0M) | 0% (never) | 0% (never) | 0.985 | [robust_g8_grid10_B63_N26M_eps10_FAspline](robust_g8_grid10_B63_N26M_eps10_FAspline/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
