# 05 Evaluation and convergence (working notes)

**TL;DR / In plain terms:** `src/evaluate.py` scores each seed's final greedy portfolio against the continuous long-only optimum, the best grid portfolio and equal weight, using exact μ and Σ for the headline numbers and a separate evaluation stream as a check. Convergence rules were fixed on 2026-10-16, before the first full run on 2026-10-22.

Source: `docs/methods/EVALUATION.md` (279 lines), `results/INDEX.md`.

## Definitions (EVALUATION.md §2)
- U(w) = w'μ − (γ/2)w'Σw, daily. CE loss vs X = 252 × (U(X) − U(w)) × 10,000 bps/yr.
- Discretization gap = CE loss of w_disc vs w*. Learning gap = learned vs w_disc. Total gap = learned vs w* = sum of the two (key `ce_loss_vs_continuous_bps`).
- Learned portfolio = greedy action of the final Q-table, ties to lowest index. w̄ equals that portfolio's weights, since the policy is evaluated with exploration off (stated assumption about "steady-state evaluation sample").
- Per seed, never averaged across seeds (a blend can beat the best grid action).
- Regret = Σ over episodes of U(w_disc) − U(chosen), daily objective units summed.
- Observations = episodes × B. The primary rule's count includes the K waiting window.
- Median = usual median (mean of 5th and 6th of 10). "Median seed" in the weights chart = upper median (6th of 10).

## Rules
| Rule | Pre-registered? | Uses |
|---|---|---|
| `greedy_stable_through_end` (primary) | yes, 2026-10-16 | agent only, resolved to the episode |
| `greedy_stable` (secondary) | yes | agent only |
| `dq_below_tol_for_good` (1e-5, 10,000-episode windows) | yes | agent only, checkpoints |
| `near_optimal_for_good` (25 bps/yr) | yes | benchmark, evaluation only, checkpoints |
| `settled` | post hoc 2026-11-09 | descriptive |
| `near_optimal_every_episode` | post hoc 2026-11-09 | descriptive |
| Q fit (Spearman, Pearson, top-10 recall, optimism) | post hoc 2026-11-04 | evaluation only |

25 bps line = 3.5%, 4.8%, 6.7% of best grid value at γ = 3, 5, 8; it contains 2, 4, 2 of the 1,287 portfolios.

## Headline numbers quoted in `results/INDEX.md` (to tie to each `evaluation/summary.json`)
| Experiment | Disc gap | Learning gap median (range) | Total gap median (range) | Exactly best | Primary rule (median days) | Settled median (range) | Near-opt checkpoints | Near-opt every episode | Q fit |
|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20 | 5.7 | 14.1 (0.0 to 47.3) | 19.8 (5.7 to 53.1) | 30% | 90% (185.1M) | 47.1M (18.0M to 80.6M) | 70% (25.8M) | 70% (155.7M) | 0.821 |
| resolution_g5_grid25 | 11.7 | 2.5 (0.0 to 30.7) | 14.3 (11.7 to 42.5) | 50% | 100% (101.2M) | 18.5M (7.6M to 44.2M) | 90% (9.6M) | 90% (66.8M) | 0.926 |
| robust_g3_grid20 | 2.5 | 10.0 (0.0 to 81.1) | 12.5 (2.5 to 83.6) | 50% | 80% (192.6M) | 26.4M (12.0M to 58.0M) | 60% (22.2M) | 60% (147.8M) | 0.710 |
| robust_g8_grid20 | 6.1 | 5.7 (0.0 to 53.1) | 11.9 (6.1 to 59.2) | 50% | 90% (153.2M) | 27.9M (12.2M to 86.8M) | 70% (25.8M) | 70% (123.5M) | 0.920 |

## Things the report must say (from EVALUATION.md and the audits)
- Illustrative demo ran before settings were fixed (EVALUATION.md §6; EXPERIMENTS.md §5).
- Near-optimal read at checkpoints is about 6 times faster than every episode (25.8M vs 155.7M days at γ = 5).
- Resolution comparison: the 25% grid has a smaller learning gap but a larger discretization gap; total gap median 14.3 vs 19.8.
- ΔQ rule is uninformative under a constant α.
- F-L10: weights "after convergence" are reported for every seed, converged or not; caveat `grid10-robustness` (0 of 30 met the primary rule).
- F-L9: outputs give simulated days; episodes = days / B.

## Doc inconsistency seen
- EVALUATION.md §7 says one constant-α update moves a value "by about 5e-7"; EXPERIMENTS.md §8.7 (corrected 2026-11-10) says the largest single update in a window is 1e-6 to 5e-6 and 5e-7 is a typical update. Both are below the 1e-5 tolerance, so the conclusion is the same. Low.
