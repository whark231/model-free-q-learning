# 10 Results and reports inventory, and the number tie-out (working notes)

**TL;DR / In plain terms:** I read every committed `evaluation/summary.json` and `per_seed.csv` with my own script and compared them with the numbers quoted in `results/INDEX.md` and `EXPERIMENTS.md` §7.7, §8.7 and §8.8. Every number checked matched. The inventory of folders is at the end.

## Tie-out table (from each `results/<...>/evaluation/summary.json`, 10 seeds each, master seed 20261016 everywhere)
Columns: discretization gap; learning gap mean / median (min to max); total gap mean / median; share exactly best; primary rule share (median days); near-optimal-for-good share; settled median; median Spearman; median top-10 recall. bps/yr.

| Experiment | disc | gap mean / med (range) | total mean / med | best | primary | near-opt | settled med | Spearman | recall |
|---|---|---|---|---|---|---|---|---|---|
| baseline_g5_grid20 | 5.74 | 17.6 / 14.1 (0 to 47.3) | 23.3 / 19.8 | 0.3 | 0.9 (185.1M) | 0.7 | 47.1M | 0.821 | 0.35 |
| resolution_g5_grid25 | 11.74 | 5.1 / 2.5 (0 to 30.7) | 16.9 / 14.3 | 0.5 | 1.0 (101.2M) | 0.9 | 18.5M | 0.926 | 0.50 |
| robust_g3_grid20 | 2.47 | 30.6 / 10.0 (0 to 81.1) | 33.0 / 12.5 | 0.5 | 0.8 (192.6M) | 0.6 | 26.4M | 0.710 | 0.20 |
| robust_g8_grid20 | 6.13 | 13.0 / 5.7 (0 to 53.1) | 19.2 / 11.9 | 0.5 | 0.9 (153.2M) | 0.7 | 27.9M | 0.920 | 0.20 |
| b63 g5 | 5.74 | 5.1 / 0.0 (0 to 15.9) | 10.8 / 5.7 | 0.6 | 0.9 (434.9M) | 1.0 | 77.5M | 0.931 | 0.55 |
| b63 g3 | 2.47 | 10.0 / 10.0 (0 to 20.0) | 12.5 / 12.5 | 0.5 | 0.9 (327.2M) | 1.0 | 47.6M | 0.872 | 0.45 |
| b63 g8 | 6.13 | 9.1 / 5.7 (0 to 29.5) | 15.2 / 11.9 | 0.5 | 1.0 (301.1M) | 0.8 | 42.2M | 0.971 | 0.50 |
| n26m g5 | 5.74 | 8.3 / 10.5 (0 to 15.9) | 14.0 / 16.3 | 0.3 | 0.8 (348.9M) | 1.0 | 73.9M | 0.895 | 0.40 |
| n26m g3 | 2.47 | 16.2 / 20.0 (0 to 62.3) | 18.7 / 22.5 | 0.4 | 0.9 (302.6M) | 0.9 | 58.5M | 0.823 | 0.40 |
| n26m g8 | 6.13 | 5.0 / 0.0 (0 to 27.2) | 11.1 / 6.1 | 0.7 | 0.9 (298.2M) | 0.9 | 58.3M | 0.956 | 0.40 |
| eps-decay g5 | 5.74 | 16.1 / 10.5 (0 to 37.2) | 21.9 / 16.3 | 0.4 | 0.6 (243.2M) | 0.6 | 74.0M | 0.862 | 0.40 |
| eps-decay g3 | 2.47 | 4.0 / 0.0 (0 to 20.0) | 6.5 / 2.5 | 0.8 | 0.8 (194.0M) | 1.0 | 46.1M | 0.758 | 0.30 |
| eps-decay g8 | 6.13 | 11.6 / 5.7 (0 to 51.6) | 17.8 / 11.9 | 0.5 | 0.6 (160.5M) | 0.8 | 56.0M | 0.938 | 0.40 |
| all-three g5 | 5.74 | 7.9 / 6.2 (0 to 29.4) | 13.6 / 11.9 | 0.5 | 1.0 (705.2M) | 0.9 | 126.6M | 0.971 | 0.65 |
| all-three g3 | 2.47 | 4.0 / 0.0 (0 to 20.0) | 6.5 / 2.5 | 0.8 | 1.0 (371.0M) | 1.0 | 62.4M | 0.946 | 0.60 |
| all-three g8 | 6.13 | 1.1 / 0.0 (0 to 11.5) | 7.3 / 6.1 | 0.9 | 1.0 (517.1M) | 1.0 | 64.3M | 0.988 | 0.50 |
| grid10-robustness g5 | 1.29 | 11.7 / 10.8 (4.4 to 19.0) | 13.0 / 12.1 | 0.0 | 0.0 | 1.0 | 797.0M | 0.543 | 0.10 |
| grid10-robustness g3 | 2.47 | 22.0 / 25.6 (0 to 62.3) | 24.5 / 28.0 | 0.4 | 0.0 | 0.5 | 834.1M | 0.406 | 0.10 |
| grid10-robustness g8 | 6.13 | 5.8 / 6.4 (0 to 10.6) | 11.9 / 12.5 | 0.1 | 0.0 | 1.0 | 643.5M | 0.725 | 0.10 |
| tabular-const g5 | 5.74 | 43.5 / 42.0 (10.5 to 109.9) | | 0.0 | 1.0 | 0.4 | | 0.813 | 0.25 |
| tabular-const g3 | 2.47 | 107.2 / 102.6 (0 to 205.2) | 109.6 / 105.1 | 0.2 | 0.8 (50.9M) | 0.2 | 37.2M | 0.703 | 0.15 |
| tabular-const g8 | 6.13 | 43.3 / 51.5 (11.5 to 61.2) | 49.4 / 57.7 | 0.0 | 1.0 (109.7M) | 0.1 | 83.8M | 0.917 | 0.20 |
| fa-quadratic g5 | 5.74 | 22.6 / 22.6 (15.9 to 29.4) | 28.4 / 28.4 | 0.0 | 0.2 (243.1M) | 0.5 | 248.5M | 0.915 | 0.55 |
| fa-quadratic g3 | 2.47 | 8.0 / 0.0 (0 to 20.0) | 10.5 / 2.5 | 0.6 | 0.0 | 1.0 | 254.0M | 0.854 | 0.80 |
| fa-quadratic g8 | 6.13 | 58.7 / 55.7 (55.7 to 70.8) | 64.8 / 61.8 | 0.0 | 0.9 (246.1M) | 0.0 | 140.1M | 0.951 | 0.20 |
| fa-spline g5 | 5.74 | 15.7 / 5.3 (0 to 78.5) | 21.5 / 11.0 | 0.5 | 1.0 (173.9M) | 0.8 | 146.9M | 0.950 | 0.45 |
| fa-spline g3 | 2.47 | 45.4 / 62.3 (0 to 68.3) | 47.9 / 64.8 | 0.3 | 1.0 (142.3M) | 0.3 | 102.0M | 0.945 | 0.65 |
| fa-spline g8 | 6.13 | 54.6 / 51.6 (50.0 to 61.2) | 60.8 / 57.7 | 0.0 | 0.9 (219.5M) | 0.0 | 165.2M | 0.957 | 0.20 |
| fa-spline-grid10-eps3 g5 | 1.29 | 78.8 / 66.4 (58.2 to 140.9) | 80.0 / 67.6 | 0.0 | 0.0 | 0.0 | 1586.7M | 0.976 | 0.00 |
| fa-spline-grid10-eps3 g3 | 2.47 | 140.6 / 137.6 (94.2 to 207.4) | 143.1 / 140.1 | 0.0 | 0.0 | 0.0 | 1601.1M | 0.963 | 0.00 |
| fa-spline-grid10-eps3 g8 | 6.13 | 59.6 / 59.6 (1.3 to 108.0) | 65.7 / 65.7 | 0.0 | 0.0 | 0.1 | 1557.4M | 0.983 | 0.10 |
| fa-spline-grid10-eps10 g5 | 1.29 | 81.4 / 82.6 (62.0 to 112.8) | 82.7 / 83.8 | 0.0 | 0.0 | 0.0 | 1620.4M | 0.975 | 0.00 |
| fa-spline-grid10-eps10 g3 | 2.47 | 155.6 / 154.7 (130.3 to 182.8) | 158.1 / 157.2 | 0.0 | 0.0 | 0.0 | 1607.1M | 0.964 | 0.00 |
| fa-spline-grid10-eps10 g8 | 6.13 | 60.2 / 66.7 (27.6 to 85.5) | 66.3 / 72.8 | 0.0 | 0.0 | 0.0 | 1625.7M | 0.985 | 0.00 |

Experiment folder names: `<run>/{baseline_g5,robust_g3,robust_g8}_grid{20,10}_<suffix>` with suffixes B63, N26M, eps3, B63_N26M_eps3, TABconst, FAquad, FAspline, B63_N26M_eps3_FAspline, B63_N26M_eps10_FAspline.

## Paired mean differences in the learning gap (my recomputation from `per_seed.csv`, SE of the paired mean, 10 seeds)
| Pair | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|
| tabular-const − baseline | +25.9 (11.5) | +76.6 (24.6) | +30.2 (8.0) |
| fa-quadratic − tabular-const | −20.9 (9.8) | −99.2 (23.7) | +15.4 (6.5) |
| fa-spline − tabular-const | −27.8 (11.9) | −61.8 (24.2) | +11.4 (5.1) |
| fa-spline − fa-quadratic | −6.9 (8.3) | +37.4 (10.2) | −4.1 (2.0) |
| fa-quadratic − baseline | +5.0 (5.2) | −22.6 (12.8) | +45.7 (5.6) |
| fa-spline − baseline | −1.8 (9.5) | +14.8 (17.3) | +41.6 (6.0) |
| grid10 − all-three (total gap) | −0.6 (3.7) | +18.0 (7.5) | +4.6 (1.6) |
| fa-grid10-eps3 − grid10 | +67.1 (9.1) | +118.6 (12.2) | +53.8 (9.8) |
| fa-grid10-eps10 − eps3 | +2.6 (11.8) | +15.0 (14.8) | +0.6 (10.1) |
| fa-grid10-eps10 − grid10 | +69.7 (4.7) | +133.6 (12.8) | +54.4 (6.4) |
| b63 − baseline | −12.5 (6.2) | −20.6 (10.4) | −3.9 (6.6) |
| n26m − baseline | −9.3 (5.3) | −14.3 (14.7) | −8.0 (7.2) |
| eps-decay − baseline | −1.5 (7.2) | −26.6 (11.1) | −1.4 (9.2) |
| all-three − baseline | −9.7 (5.9) | −26.6 (11.7) | −11.9 (5.4) |

All agree with `EXPERIMENTS.md` §7.7, §8.7, §8.8 and `AUDIT_2026-11-06.md` §2.3. Caveat on the last four rows: runs with a different B or ε use the seed streams differently, so those pairs share a start but are not paired draw for draw (EXPERIMENTS.md §7.1).

## Benchmarks re-derived independently (scratch script; long-only optimum by enumerating all 511 supports)
| γ | w* (SPY, IEF, GLD) | ann. return | ann. vol | U(w*) bps/yr | 20% best | disc gap 25% / 20% / 10% | 2nd best behind (20%) | EW loss vs w* |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.5502, 0, 0.4498 | 10.00% | 13.89% | 710.8 | (0.6, 0, 0.4) | 2.51 / 2.47 / 2.47 | 20.0 | 354.2 |
| 5 | 0.4688, 0.1363, 0.3948 | 8.86% | 11.92% | 530.5 | (0.4, 0.2, 0.4) | 11.74 / 5.74 / 1.29 | 10.5 | 310.9 |
| 8 | 0.3562, 0.3899, 0.2539 | 6.79% | 8.66% | 379.2 | (0.4, 0.4, 0.2) | 22.60 / 6.13 / 6.13 | 11.5 | 365.2 |

Budget-only closed form at γ = 5: weights (1.82, −0.88, −0.19, 0.27, −0.02, −0.27, 0.69, −0.13, −0.29), U = 925.9 bps/yr. So the long-only constraint costs 925.9 − 530.5 = 395.3 bps/yr at γ = 5.
Grid counts by C(u + 8, 8): 495, 1,287, 43,758, 3,108,105. Mean U over the 20% grid at γ = 5: 56.7 bps/yr (the "random portfolio" figure in the first TC design).

## Inventory classification (every folder under `results/` and `reports/`)
| Path | Class | Note |
|---|---|---|
| `results/INDEX.md`, `index.json`, `settings.json` | Current, reported | `baseline-v1` index |
| `results/baseline_g5_grid20/`, `resolution_g5_grid25/`, `robust_g3_grid20/`, `robust_g8_grid20/` | Current, reported (headline) | Pre-registered |
| `results/comparison_g5_grid20_vs_grid25/` | Current, reported | Part V two-resolution chart |
| `results/b63/`, `n26m/`, `eps-decay/`, `all-three/` | Current, reported (Part VI, post hoc) | Coverage depth undecided (prospectus §6) |
| `results/grid10-robustness/` | Current, reported (Part VI action-space check, post hoc) | 0 of 30 seeds meet the primary rule |
| `results/tabular-const/`, `fa-quadratic/`, `fa-spline/` | Current, reported (FA extension, post hoc) | |
| `results/fa-spline-grid10-eps3_2026-11-11/`, `fa-spline-grid10-eps10_2026-11-11/` | Current, reported (FA extension, post hoc) | Dated folder names differ from the `results/<run_id>/` pattern |
| `results/tuning/` | Tuning record, not a result | Includes two tracked empty `.err` files |
| `results/tc-quadratic-cost/` | Kept as negative result | Reporting form is D-G |
| `results/tc-drift/oracle/`, `checks/`, `tuning/` | Current | `src_dirty: true` on the oracle summary and all six tuning records |
| `results/tc-drift/runs/` | Absent | The 40 agent runs have not run |
| `results/*.log` | Run logs | |
| `results/smoke*`, `results/repro` | Local only, git-ignored | Not in the repo |
| `reports/registry.json` and its 11 run folders (`baseline-v1` to `fa-spline-grid10-eps10`) | Current | One HTML and `data.json` each |
| `reports/full_run_2026-10-22/` | Superseded, kept as record | Hand-built; REVIEW_HISTORY §7.1 cites its `data.json` |
| `reports/tc-quadratic-cost/` | Kept as negative result, outside the pipeline | Text overstates (M8) |

No orphan result folder found. Orphans elsewhere: `experiments/transaction-costs/discount.json` and `learning-rate.json` (registered, never run), `notebooks/exploration.ipynb` (0 cells), `HANDOFF.md` (untracked, dated 2026-10-09).
