# 09 Transaction costs with drift (working notes)

**TL;DR / In plain terms:** the reported TC extension uses three ETFs (SPY, IEF, GLD), 63-day blocks in which holdings drift, a linear plus quadratic trading cost, a numerical oracle and a spline FA agent that learns the cost size from net rewards. Oracle, checks and tuning are done and committed. The 40 agent runs have not run (`results/tc-drift/runs/` absent).

Source: `docs/methods/TC_DRIFT_DESIGN.md` (630 lines), §1 to §7 read first; §8 to §14 notes are further down.

## Design facts (design §2 to §4)
- Universe SPY, IEF, GLD = positions (0, 3, 6) of `TICKERS`. γ = 5.
- Actions: 1,326 targets on the 2% grid plus HOLD (index 1,326) = 1,327.
- State: continuous holdings; oracle rounds to the 1% lattice (5,151 points).
- B = 63 days per block. ρ = 0.95 per block (about 0.81 per year). Assignment asks a positive discount to be explained (C15).
- Cost c = κ‖a − s‖₁ + (λ/2)‖a − s‖₂², fraction of wealth; HOLD costs 0. Net reward r = R(x) − c/B (daily units).
- Reward: buy-and-hold within block, realized mean − (γ/2) realized variance, ddof = 1.
- Terminology table (design §4): block = one decision; episode = 90 blocks (`config.RESTART_EVERY`); one training run = 4M blocks ≈ 44,400 episodes = 252M days.
- Ties: HOLD wins.
- Seeds: reported `MASTER_SEED` 20261016; tuning 20261112; oracle fixed paths `spec.ORACLE_PATH_SEED`.

## Oracle (design §5)
- M = 2,000 paths; control variate cuts reward SD from 2,390 to 103 bps/yr at the optimum.
- Best target (0.46, 0.14, 0.40) at every level.
- Table in §5 (in-model, bps/yr), oracle lead over always rebalancing: zero 0.008; $100M 0.214; $1B 1.319; $10B 6.939; κ only 1.542; λ only 1.488. Source `results/tc-drift/oracle/summary.json`.
- Check 3 restated on the policy (post hoc, 2026-11-17); check 4 lattice; tolerance 0.5 bps/yr.
- Check 5 (`results/tc-drift/checks/oracle_replay_2026-11-17.json`), replay 1,000 starts, H = 90: oracle lead over always-rebalance −0.03 (0.08), −0.17 (0.43), 1.30 (0.70), 7.10 (1.26); check 5 replay minus model −0.03 (0.08), −0.38 (0.43), 0.01 (0.70), 0.31 (1.25). Hold fraction 0.15 / 0.28 / 0.32 / 0.45. Turnover oracle 2.9 / 2.2 / 2.1 / 1.5% vs 3.0% always-rebalance. Reported as a two-SE bound at cost levels (D-I, post hoc).
- In-sample V about 4.9 ± 0.7 bps/yr too high at M = 2,000; gaps never use it.

## Agent (design §6)
- Q̂(s, a) = θ·φ(target) − θ₁‖u‖₁ − θ₂‖u‖₂²; HOLD: θ·φ(s). 87 spline columns, rank 45 on the 2% grid, max ‖φ‖² = 6.
- Cost coefficients clamped at ≥ 0 (D11). Disclose.
- ε-greedy, ε_t = e^(−βt); exploratory step uniform over all 1,327 actions.
- Frozen target copy; semi-gradient.
- D12 convergence rule: four versions (2026-11-12 original probe rule; 2026-11-19 favourite target within L1 0.04 plus 95% exact-action probes; 2026-11-19 tail average; 2026-11-20 secondary restated on post-decision holdings). Frozen 2026-11-20. All changes before any final run, each labelled; later ones made after seeing pilot benchmark diagnostics (disclose).
- Reported policy: greedy on the tail average of θ over checkpoints in the final 10% (11 snapshots).

## Evaluation (design §7)
- H = 90 blocks, m = 4,000 starts (D-E, 2026-11-20), starts uniform on the 2% grid.
- References: oracle, always rebalance to best target, trade once then hold, myopic. Plus own-target (favourite target) policy for the split.
- Run driver `src/tc_drift/run.py`; outputs `results/tc-drift/runs/<level>/{summary.json, per_seed.csv, seed<i>.json}`.

## Registered expectations E1 to E12 (design §7) with the summary.json field that tests each
| # | Short form | Field |
|---|---|---|
| E1 | agent − oracle mean between −1 and −8 at each level; negative on ≥ 8 of 10 seeds | `agent_minus_oracle_mean`; per seed in `per_seed.csv` |
| E2 | favourite-target error is most of the gap at zero, $100M, $1B | the two `split_*` means |
| E3 | agent loses to always-rebalance at $100M and $1B; relatively better at $10B than $1B | `agent_minus_always_rebalance_mean` |
| E4 | agent − own target rises from $1B to $10B, positive at $10B; within ±1 at zero and $100M | `split_agent_minus_own_target_mean` |
| E5 | agent beats trade-once-then-hold at every level and seed | `agent_minus_trade_once_then_hold` per seed |
| E6 | agent loses to myopic at every level | `agent_minus_myopic_mean` |
| E7 | hold fraction varies across seeds; mean higher at $10B than zero; at zero cost holds more than the oracle | `agent_hold_fraction` |
| E8 | c2 learned better than c1 (c2 within ×2 on ≥ 7 of 10 seeds at $1B, $10B; c1 off by > ×2 on ≥ 3 of 10 at each cost level) | `c1_over_true`, `c2_over_true` |
| E9 | fewer than 5 of 10 seeds converge per level under the 2026-11-20 rule; none under the original and exact-action rules on the last iterate | `d12_converged_count` |
| E10 | tail average vs last iterate differ by < 1 bps/yr in the mean | `agent_minus_last_iterate_mean` |
| E11 | D10: missed at $1B (SE 0.5 to 0.8 vs target about 0.43); met at $10B; not applicable at zero and $100M | `d10` |
| E12 | across-seed SD of agent − oracle at $1B ≥ 0.5 | `sd_across_seeds` |

Basis disclosure: the expectations lean on the $1B tuning pilot's benchmark diagnostics (same agent, other seeds).

## Cost levels (design §8; `src/tc_drift/calibration.py`; test `test_tc_drift_calibration.py` pins the table)
Spread 0.5 bp per dollar traded; square-root impact with Y = 1; liquidity from own raw data over the 12 months to 2026-07-31 (SPY $49.4B and 0.81%, IEF $0.81B and 0.29%, GLD $5.20B and 1.78%). Fit within 27% of the square-root cost at the 8 fitted trade sizes. One κ, λ for every pair, so the typical SPY-vs-GLD drift trade is undercharged by about a fifth (disclose).

| Level | κ (bp per unit ‖u‖₁) | λ | 20-pt transfer (bp) | 40-pt (bp) | undo one block's drift (bp) |
|---|---|---|---|---|---|
| Zero | 0 | 0 | 0 | 0 | 0 |
| $100M | 1.63 | 4.4e-3 | 2.40 | 8.32 | 0.115 |
| $1B (headline) | 3.90 | 1.43e-2 | 7.27 | 25.97 | 0.302 |
| $10B | 11.04 | 4.57e-2 | 22.69 | 81.93 | 0.892 |

## Oracle sweep in-model (design §8; `results/tc-drift/oracle/summary.json`)
| Setting | Hold frac | Turnover/block | Trades in 90 | Mean trade | Lead over always-reb | over trade-once | over never | over myopic | Myopic hold |
|---|---|---|---|---|---|---|---|---|---|
| Zero | 0.156 | 2.95% | 75.9 | 3.50% | 0.008 | 24.83 | 142.35 | 0.000 | 0.156 |
| $100M | 0.272 | 2.26% | 65.5 | 3.10% | 0.214 | 24.55 | 140.83 | 0.014 | 0.291 |
| $1B | 0.308 | 2.13% | 62.3 | 3.07% | 1.319 | 24.86 | 138.50 | 0.243 | 0.388 |
| $10B | 0.443 | 1.57% | 50.2 | 2.82% | 6.939 | 27.96 | 133.25 | 2.586 | 0.599 |
| κ only | 0.577 | 1.57% | 38.1 | 3.70% | 1.542 | 22.67 | 137.69 | 1.691 | 0.735 |
| λ only | 0.259 | 2.21% | 66.6 | 2.99% | 1.488 | 25.69 | 138.95 | 0.129 | 0.248 |

Registered oracle directions not met (2): trade-once-then-hold does not come closer to the oracle as size grows (24.55, 24.86, 27.96); myopic holds less than the oracle under λ only (0.248 vs 0.259).

## Tuning (design §9; `results/tc-drift/tuning/`)
- Tuning seed 20261112, $1B only, 3 seeds, 4M blocks. Criterion: realized discounted net reward on fresh paths, never the oracle gap.
- Rule: RLS on the cost block after constant-α burn-in (until ε reaches 0.5, 301,030 updates); θ constant α_value. Candidates α_value ∈ {3e-4, 1e-3}. Chosen 3e-4: mean 505.4 vs 496.1, paired diff 9.31, paired SE 6.32.
- β = ln 100 / 2,000,000 ≈ 2.30e-6, about 434,000 exploring blocks. target_refresh_every 1,000; l2 0.
- Pilot diagnostics at 3e-4: agent − oracle −3.9 ± 1.5, −2.8 ± 1.1, −3.2 ± 1.3; agent − always-reb −2.6, −1.5, −1.9; favourite target utility loss 5.6, 2.3, 4.1; c1/true 0.41, 3.19, 0.00; c2/true 0.94, 0.49, 1.01; hold fraction 12%, 55%, 16%; 1 of 3 converges (seed 0 from block 3,284,100 = 36,490 episodes = 206.9M daily observations).
- A first set of six pilot runs (clamp inside RLS) was discarded and not reported. Pilot used uncommitted code on `feature/tc-drift-rls`.

## Design numbers (design §12; `python -m src.tc_drift.design_numbers`, seed 20261113)
Nine-asset long-only optimum γ3: SPY 0.5502, GLD 0.4498. γ5: SPY 0.4688, IEF 0.1363, GLD 0.3948. γ8: SPY 0.3562, IEF 0.3899, GLD 0.2539. Block reward SD at optimum 2,385 (constant) / 2,390 (buy-and-hold) / 103 (difference). Drift loss 1.33 after 1 block, 5.40 after 4, 25.9 after 20. Continuous optimum minus best 2% target 0.10 bps/yr. Median one-block drift 2.26%, mean 2.62%. rf effect on drift ≤ 7.1e-6.

## Disclosures the report must carry (design §13 C1, C2, C15; §14 risk 5; audit §9.6)
- Universe = the three ETFs in the nine-asset optimum's support: benchmark information in the design (C1).
- Drift on excess returns (bound 7.1e-6).
- ρ = 0.95 per block explained (D8: half the weight at 3.4 years).
- D11 clamp; cost features build in the cost family's shape.
- Tuning at $1B only on tuning seed; learning rule, tail average and restated secondary chosen with benchmark diagnostics in view.
- D12 changed three times before any final run; all four verdicts reported; original secondary (within 25 bps of oracle) not computed.
- Check 3 restated post hoc; check 5 a two-SE bound at cost levels.
- One cost for every pair undercharges drift trades by about a fifth.
- Probes read at checkpoints only.

## Checked against the committed files (2026-11-20)
- `results/tc-drift/oracle/summary.json`: every value in the design's §5 and §8 tables matches (leads 0.0078, 0.2140, 1.3189, 6.9390, 1.5416, 1.4881; hold fractions 0.1565, 0.2723, 0.3080, 0.4427, 0.5766, 0.2595; κ, λ for $100M 1.627e-4 / 4.384e-3, $1B 3.896e-4 / 1.428e-2, $10B 1.104e-3 / 4.569e-2). Keys: `settings.<level>.{oracle_value_bps, lead_over_always_rebalance_bps, lead_over_trade_once_bps, lead_over_never_trade_bps, lead_over_myopic_bps, oracle.hold_fraction, oracle.turnover_per_block, check3.*, check4.*}`. Levels: `zero`, `aum_100m`, `aum_1b`, `aum_10b`, `kappa_only`, `lambda_only`.
- Provenance of that file: commit `0a8d180`, `src_dirty: true`, Python 3.10.2, numpy 2.2.6 (William's machine), runtime 8,452 s. `oracle_path_seed` [20261016, 3], `check3_fresh_seed` [20261016, 4].
- `results/tc-drift/checks/oracle_replay_2026-11-17.json`: check-5 table matches (keys `levels.<level>.{lead_replay, lead_model, check5, rounding, level_replay_minus_model, oracle_hold_fraction, oracle_turnover_per_block}`). m = 1,000, H = 90, tuning seed's evaluation stream. Its `note` says the control-variate mean was printed 20 times too small in the run and recomputed afterwards.
- `results/tc-drift/tuning/summary.json`: `choice.means` a3e-4 505.398, a1e-3 496.085; `paired_difference` 9.313; `paired_se` 6.321; decision a3e-4 on 3 seeds. `diagnostics_not_used_to_choose.c_true` = [6.184e-6, 1.1334e-4]. Per-run records `tuning/a3e-4/seed{0,1,2}.json`, `tuning/a1e-3/seed{0,1,2}.json`; seed 0 at 3e-4: `convergence.converged` true, `converged_from_block` 3,284,100, `converged_from_episode` 36,490, `converged_from_daily_observations` 206,898,300; `hold_fraction` 0.119. All six records have `src_dirty: true` at commit `bed6d17` and three different `tune.py` hashes (train, rescore, recheck), as the design says ("uncommitted code on `feature/tc-drift-rls`").
- Preflight for the final runs (read-only call of `run.load_oracle_tables` on this Mac): all four levels' local `.npz` tables reproduce the committed `oracle_value_bps` within 1e-8 (534.786, 533.260, 530.933, 525.683). `config.pending()` is empty. `config.HYPERPARAMETERS` = alpha_value 3e-4, alpha_cost 3e-4, beta 2.3026e-6, target_refresh_every 1000, l2 0, cost_rls_after 301,030, cost_rls_prior 1e-3. So nothing in code or data blocks `python -m src.tc_drift.run all` here.
- `results/tc-drift/runs/` does not exist. No agent result exists.
- The six oracle `.npz` tables are 152 to 191 KB each (about 1.05 MB in all) and are git-ignored; a grader has to spend 20 minutes to 2.5 hours rebuilding them before any TC scoring.
