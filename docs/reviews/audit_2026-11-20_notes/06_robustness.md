# 06 Robustness and post-hoc runs (working notes)

**TL;DR / In plain terms:** this note records what each run after `baseline-v1` changed, what was expected, and the numbers the docs quote, so they can be tied to the result files. Source read: `docs/methods/EXPERIMENTS.md` §1 to §8.8 (908 lines).

## Baseline settings (EXPERIMENTS.md §1, src/config.py)
γ ∈ {3, 5, 8}, baseline 5. B = 20. 10 seeds, master seed 20261016. 13,000,000 episodes per seed (260M simulated days). ε0 = 1, β so ε = 1% at halfway (β ≈ 7.08e-7). α = 1/N(a), ρ = 0, Q0 = 0. K = 10% of episodes = 1,300,000. ΔQ tol 1e-5 per 10,000-episode window. Near-optimal line 25 bps/yr. 10,000 evaluation episodes. Grids: 20% (1,287) baseline, 25% (495) comparison.

Baseline experiments: `baseline_g5_grid20`, `resolution_g5_grid25`, `robust_g3_grid20`, `robust_g8_grid20`, `comparison_g5_grid20_vs_grid25`.

Pre-registration disclosure: an illustrative demo (γ = 5, B = 20, 2M episodes, EVALUATION.md §6) ran before settings were fixed (EXPERIMENTS.md §5).

## Post-hoc tabular runs (registered 2026-10-27, run 10-28/29)
| run | change vs baseline | random tries per portfolio | K |
|---|---|---|---|
| b63 | B = 63 | 1,097 | 1.3M |
| n26m | 26M episodes, β ≈ 3.54e-7 | 2,193 | 2.6M |
| eps-decay | ε 3% at halfway, β ≈ 5.39e-7 | 1,439 | 1.3M |
| all-three | B 63, 26M, ε 3% (β ≈ 2.70e-7) | 2,878 | 2.6M |
| grid10-robustness | all-three + 10% grid | 85 | 2.6M |

grid10 outcome table (EXPERIMENTS.md §7.7, mean learning gap all-three → grid10): γ5 7.9 → 11.7, exactly best 5 → 0, total 13.6 → 13.0 (paired −0.6, SE 3.7), primary rule 10 → 0. γ3 4.0 → 22.0, 8 → 4, 6.5 → 24.5 (+18.0, 7.5), 10 → 0. γ8 1.1 → 5.8, 9 → 1, 7.3 → 11.9 (+4.6, 1.6), 10 → 0. Disc gap 20% → 10%: 5.7 → 1.3 (γ5), 2.5 → 2.5, 6.1 → 6.1.

## Constant-α runs (registered 2026-11-05, run 2026-11-06)
α: tabular 3.2e-4; quadratic 3.2e-4; spline 7.11e-6 (each 1.6e-4 of bound 2/max‖φ‖²). Tuned on master seed 20261103 at γ = 5 only. Table of outcomes in EXPERIMENTS.md §8.7 (mean gap γ5: baseline 17.6, tabular-const 43.5, fa-quadratic 22.6, fa-spline 15.7; γ3: 30.6, 107.2, 8.0, 45.4; γ8: 13.0, 43.3, 58.7, 54.6).

## FA 10% grid (registered 2026-11-11)
`fa-spline-grid10-eps3`, `-eps10`; α = 2.778e-5 (noise-matched, not re-tuned). Folders `results/fa-spline-grid10-eps{3,10}_2026-11-11`. Mean learning gap FA/table γ5 78.8/11.7, γ3 140.6/22.0, γ8 59.6/5.8. eps10: 81.4, 155.6, 60.2.

(Verification of these numbers against result files: see `10_reports_results_inventory.md`.)
