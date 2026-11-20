# TC with drift: final agent runs (2026-11-20, post hoc)

**TL;DR / In plain terms:** the agent loses to the oracle at every cost level, mostly because it heads for the wrong portfolio. At zero cost, $100M and $1B it is 10 to 13 bps/yr behind and also loses to plain rebalancing. At $10B, where trading costs are largest, it learns when to trade: it beats always-rebalancing by 5.0 bps/yr and is 2.8 behind the oracle. Of the 12 expectations registered before the runs, 8 are met, 3 partly and 1 not.

**Key concepts to understand**
- **Favourite-target error versus timing.** The agent's gap to the oracle splits exactly into (agent − rebalance to its own favourite target) + (own target − rebalance to the best target) − (oracle − rebalance to the best target). The second term measures where the agent heads, the first when it trades (`TC_DRIFT_DESIGN.md` §7, TC-3).
- **Paired SE versus across-seed SD.** Every seed is scored on the same 4,000 paths, so the paired SE (about 1 bps/yr) measures path noise for one seed. Seeds differ by 4 to 6 bps/yr at the lower cost levels, which is the larger uncertainty.

**Key takeaways**
- At $10B the agent's cost-aware timing is worth +6.2 bps/yr against rebalancing to its own target, and its favourite target loses 1.8 bps/yr.
- At the lower cost levels its favourite target loses 10 to 13 bps/yr, which is larger than the $1B pilot's 2 to 6. The pilot understated the target error.

## How these were produced

`python -m src.tc_drift.run all --workers 7` on `run/tc-drift-final` at commit `1787f0d` (clean tree, `src_dirty: false` in every record), Albert's Mac (arm64, Python 3.13.9, numpy 2.3.5), BLAS pinned to one thread, 2026-11-20 15:27 to 17:41. 40 runs (4 levels × 10 seeds, master seed 20261016), 4M blocks each, about 20 minutes a run at 0.32 ms a block with seven in parallel; scoring about 13 minutes a level. Settings in `src/tc_drift/config.py`; console output in `logs/`.

Per level folder: `summary.json` and `per_seed.csv` (all numbers below), `seed<i>.json` (each run's record and provenance), `seed<i>.npz` (training log with θ at every checkpoint) and `values.npz` (per-path discounted values of every policy, raw units; multiply by (1 − 0.95)/(1 − 0.95⁹⁰) × 252 × 10⁴ for bps/yr).

**Checked independently (2026-11-20):** every seed's agent − oracle mean and paired SE, the oracle − always-rebalance gap and the oracle's value were recomputed from `values.npz` with the conversion above, not `evaluate.annualized_bps`; all match `per_seed.csv` and `summary.json` within 1e-13. The TC-3 split holds on every path.

## Results (bps/yr, mean over 10 seeds; m = 4,000 paths, H = 90 blocks)

| | zero | $100M | $1B | $10B |
|---|---|---|---|---|
| Agent − oracle (across-seed SD) | −13.14 (4.58) | −13.09 (5.52) | −9.99 (4.32) | −2.77 (1.70) |
| Seed-averaged agent − oracle, path SE | ±1.03 | ±1.02 | ±0.83 | ±0.35 |
| Agent − always rebalance | −13.11 | −12.67 | −8.21 | +5.01 |
| Agent − myopic | −13.14 | −13.10 | −9.86 | −0.50 |
| Agent − trade once, then hold (lowest seed) | 2.68 | 2.30 | 3.93 | 21.19 |
| Split: agent − own target | 0.00 | 0.08 | 0.83 | 6.16 |
| Split: own target − always rebalance | −13.11 | −12.75 | −9.04 | −1.14 |
| Favourite target's utility loss | 12.91 | 12.95 | 9.75 | 1.79 |
| Agent hold fraction (seed range) | 0.35 (0.03 to 0.73) | 0.25 (0.00 to 0.71) | 0.18 (0.01 to 0.70) | 0.17 (0.06 to 0.21) |
| Oracle − always rebalance (paired SE) | 0.04 (0.04) | 0.42 (0.22) | 1.78 (0.36) | 7.79 (0.64) |
| Oracle hold fraction | 0.15 | 0.28 | 0.32 | 0.45 |
| Converged, operative D12 (of 10) | 5 | 7 | 7 | 4 |
| Converged, original and exact-action D12 | 0 | 0 | 0 | 0 |
| Tail average − last iterate | −0.35 | 0.42 | 0.20 | 0.13 |
| D10: mean paired SE vs target | 1.10 vs 0.01, n/a | 1.10 vs 0.14, n/a | 0.94 vs 0.59, missed | 0.43 vs 2.60, met |

Learned costs against the truth, per seed (`c1_over_true`, `c2_over_true` in `per_seed.csv`): c2 is within a factor of 2 on 9 of 10 seeds at $100M and 10 of 10 at $1B and $10B. c1 is poorly learned: at or below 0.15 of the truth on 6 of 10 seeds at $1B and below 0.2 on all 10 at $10B (often clamped at zero by D11), and up to 15 times too high on others (5.4 at $1B, 2.8 to 14.9 on five seeds at $100M).

## Registered expectations (`TC_DRIFT_DESIGN.md` §7, registered before the runs)

| # | Verdict | What happened |
|---|---|---|
| E1 | Partly met | Negative on 10 of 10 seeds at every level, but the mean is outside −1 to −8 at zero, $100M and $1B (−13.1, −13.1, −10.0); inside at $10B (−2.8) |
| E2 | Met | Own target − always rebalance (−13.1, −12.8, −9.0) is negative and larger than agent − own target (0.0, 0.1, 0.8) |
| E3 | Met | Agent below always rebalance at $100M (−12.7) and $1B (−8.2); higher at $10B (+5.0) |
| E4 | Met | Agent − own target rises from 0.8 ($1B) to 6.2 ($10B); 0.0 and 0.1 at zero and $100M |
| E5 | Met | Beats trade once, then hold on every seed at every level (lowest 2.3) |
| E6 | Met | Below myopic at every level; at $10B only by 0.5 |
| E7 | Partly met | Spread across seeds is wide at zero to $1B (0.0 to 0.73) and at zero it holds more than the oracle (0.35 against 0.15); but the mean hold fraction is lower at $10B (0.17) than at zero (0.35), the opposite of the prediction |
| E8 | Met | c2 within a factor of 2 on at least 7 of 10 at $1B and $10B (10 and 10); c1 off by more than 2 on at least 3 of 10 at every cost level |
| E9 | Partly met | 0 converge under the original and exact-action rules, as predicted; under the operative rule 5, 7, 7 and 4 converge, so "fewer than 5" holds only at $10B |
| E10 | Met | Tail average − last iterate within ±1 at every level (−0.35 to 0.42) |
| E11 | Not met as stated | D10 is missed at $1B and met at $10B, and does not apply at zero and $100M, as predicted; but the $1B SE is 0.94, outside the predicted 0.5 to 0.8, and the replay's oracle lead (1.78) puts the target at 0.59, not 0.43 |
| E12 | Met | Across-seed SD at $1B is 4.32 |

Under "what would change the story" (§7): E1 failed away from zero, not toward it, and at $10B E4 holds while E2's favourite-target error is small (−1.1), so there the agent learned both when to trade and roughly where to head.

Not yet done: the report's interpretation, the "trade once, then hold" naming in tables, and why the favourite-target error is so much smaller at $10B (not investigated).
