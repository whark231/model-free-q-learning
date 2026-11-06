# Results index

Written 2026-11-06T02:49:27+00:00 from commit `bc311a387bd18ee8ab98b4e5415e49bd9e7497f0`. Full-scale settings.

**Post-hoc run `tabular-const`**, registered 2026-11-05 after the baseline results were seen (EXPERIMENTS.md §7 and §8). Expected beforehand: a larger mean learning gap than baseline-v1 at every γ, and fewer seeds meeting the primary convergence rule at every γ (baseline-v1: 9, 8, 9 at γ = 5, 3, 8), because a constant alpha never averages the reward noise away (Q-value noise about 49 bps/yr)

γ ∈ [3.0, 5.0, 8.0] (baseline 5), B = 20, 10 seeds × 13,000,000 episodes, master seed 20261016, K = 1,300,000 episodes (convergence rule: `greedy_stable_through_end`), ΔQ tolerance 1e-05, near-optimal line 25 bps/yr, 10,000 evaluation episodes. Learning rate α: 0.00032; agent: tabular.

Losses are certainty-equivalent, in basis points per year, against the best portfolio on that grid. For each rule: the share of seeds that met it, and in brackets the median simulated days (millions) among the seeds that did.

| Experiment | γ | Grid | Discretization gap | Learning gap, median (range) | Exactly best | Converged (primary rule) | Near-optimal for good | Folder |
|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20_TABconst | 5 | 20% (1,287) | 5.7 | 42.0 (10.5 to 109.9) | 0% | 100% (53.2M) | 40% (19.1M) | [baseline_g5_grid20_TABconst](baseline_g5_grid20_TABconst/evaluation/summary.json) |
| robust_g3_grid20_TABconst | 3 | 20% (1,287) | 2.5 | 102.6 (0.0 to 205.2) | 20% | 80% (50.9M) | 20% (119.2M) | [robust_g3_grid20_TABconst](robust_g3_grid20_TABconst/evaluation/summary.json) |
| robust_g8_grid20_TABconst | 8 | 20% (1,287) | 6.1 | 51.5 (11.5 to 61.2) | 0% | 100% (109.7M) | 10% (87.6M) | [robust_g8_grid20_TABconst](robust_g8_grid20_TABconst/evaluation/summary.json) |

Each experiment's figures and tables are in `<folder>/evaluation/`; see EVALUATION.md for definitions.
