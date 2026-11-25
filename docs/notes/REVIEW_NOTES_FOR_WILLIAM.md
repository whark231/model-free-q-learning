# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09, updated 2026-11-22
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md`, `AUDIT_2026-11-06.md` §2 and the "Closed since 2026-11-20" list below. Albert's audit (`AUDIT_2026-11-20.md` §6) is the correction plan for what is left. Audit IDs: a bare ID (M8, F-L9, L7) is from `AUDIT_2026-11-06.md`; an ID prefixed A20 (A20-H1, A20-M2) is from `AUDIT_2026-11-20.md`.

---

## TL;DR / In plain terms

The 40 transaction-cost runs are done and scored (Albert, 2026-11-20), and nothing else needs running. What is left is writing (the report, with section drafts Nov 24 and the full draft Nov 26, the slides, and the documentation gaps Albert's audit found) plus small code and repo tasks. Three of the gaps are yours: the Part II derivation in prose, a script for the closed-form benchmark table, and the FA analyses that have no committed script.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions have not finished learning (wording per `AUDIT_2026-11-06.md`). This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.
- **Where the TC agent heads versus when it trades.** Its gap to the oracle splits exactly into the cost of its favourite target and the value of its timing, minus the oracle's own lead over always-rebalancing (`TC_DRIFT_DESIGN.md` §7, TC-3).

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.
- For the TC section: agent − oracle is −13.1 / −13.1 / −10.0 / −2.8 bps/yr at zero / $100M / $1B / $10B, all 10 seeds below the oracle at every level. Where it heads (split: own target − always rebalance) is −13.1 / −12.8 / −9.0 / −1.1; its timing is worth +6.2 at $10B, where it beats always-rebalancing by 5.0. A separate diagnostic, the favourite target's utility loss (computed with μ and Σ), is 12.9 / 13.0 / 9.75 / 1.79. E1 to E12: 8 met, 4 partly met (`results/tc-drift/runs/README.md`).

---

## Still open (2026-11-22)

| Item | What | Who |
|---|---|---|
| Report | Not started. Outline `docs/report/REPORT_SKELETON.md`, every number tied to a committed file. Section drafts Nov 24, full draft Nov 26 (D-H) | Both |
| Your report sections | Theory/Benchmark (the derivation), Q-learning (including constant α versus 1/N) and the FA extension (prospectus §4). FA from `AUDIT_2026-11-06.md` §2.3 and `EXPERIMENTS.md` §8.7 and §8.8, with the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; M12) | You |
| A20-H2 | No methods prose for Part II: the closed-form derivation, its three verification routes, and the identical constraints for theory and learning. Write `docs/methods/BENCHMARKS.md` or go straight to report §3 | You draft the derivation, Albert the rest |
| A20-H3 | The budget-only closed-form portfolio is in no committed table. A read-only script (for example `python -m src.benchmarks.report_table`) writing the budget-only and long-only portfolios for γ = 3, 5, 8 to a new file, with a test. γ = 5: U = 925.9 bps/yr budget-only against 530.5 long-only (Albert's audit; re-checked by an independent KKT solve in the 2026-11-22 review) | You |
| A20-H4 | The surrogate study table, the γ = 8 mean-field calculation and the shrinkage slopes have no committed script. Commit them, or the report words them as hypotheses (skeleton's default if not committed by Nov 24) | You |
| A20-M1 | The TC oracle summary and the seven tuning files (six seed records and `tuning/summary.json`) carry `src_dirty: true` (produced from uncommitted code on this machine). Disclose in the report. Rerunning the oracle from a clean commit (20 minutes to 2.5 hours (8,452 s once, not a clean timing)) is optional; with no further runs planned, I recommend disclosure only | You |
| TC section | Who writes it (prospectus §4: proposed for Albert). The interpretation is not written. Open questions: why the favourite-target error is so much smaller at $10B (not investigated); E1's range was missed away from zero (the pilot's 2 to 6 bps/yr utility loss understated the runs' 9.75 at $1B); c1 is poorly learned (at or below 0.15 of the truth on 6 of 10 seeds at $1B, below 0.2 on all 10 at $10B, up to 15 times too high at $100M) while c2 is within a factor of 2 on 10 of 10 at $1B and $10B. Text to carry: the two oracle directions not met (trade once, then hold does not approach the oracle as costs grow; myopic holds less than the oracle under λ only); the oracle's in-sample V is about 5 bps/yr high at M = 2,000; D12 tracks the target the agent trades to and says nothing about when it holds; D12's 2026-11-12 original rule is reported for its probe primary only (its secondary, within 25 bps of the oracle, is not computed); probes are read only at checkpoints, so a flip that reverts between two is not seen (design §7); check 5 as a two-SE bound (D-I); TC-13 and the tail-average change (design §14 risk 5); every disclosure in skeleton §6.5 | Both |
| Report text, non-TC | L9 (take the Part V items from `evaluation/`); L10 (the γ = 8 cause is untested); `RUN_ISSUES_2026-10-22.md` Findings 4, 5, 8 and 9; A20-M5 (why gaps use the known moments) | William and Albert by section |
| TC-8, D-G | How the first TC design is reported. M7, M8, M9, M11, the TC halves of M1 and F-L12, and showing `ignore_cost` in the old tables (H2) wait on it. Albert's audit recommends one paragraph and the four-row table, with M7 to M11 closed as not reported (A20-M4) | Both |
| A20-M2 | Commit the six TC oracle tables (about 1.05 MB) so scoring needs no rebuild (20 minutes to 2.5 hours (8,452 s once, not a clean timing)). The runs' per-seed logs and per-path values are committed already | Both |
| Post-hoc coverage | Which post-hoc runs the report covers in full (A20-M8, skeleton §6; it matches your 2026-10-28 paragraph at the end of History) | Both |
| M12, F-L10 | Report text: the FA tuning disclosure; caveats on weights where no seed converged | William (FA) and Albert (convergence), in the draft |
| F-L9 | Episodes column in `performance.csv` (partly fixed: `INDEX.md` gives days = episodes × B). Say it in the report. Low priority | Albert |
| L7 | Whether `all-three`'s report should compare with its single-change runs | With Albert |
| Presentation | Not started | Both |
| `dev` into `main` | `dev` is 5 commits ahead of `main` (PRs #46 and #47). Merge after the last change, then a fresh-clone check (tests, data rebuild and `run_experiments --dry-run`; no retraining) | Both |

## Your next steps

1. **Your report sections**, for the section drafts due Nov 24.
2. **H3 then H2.** The script is small and gives report §3 its table; the derivation text follows from it.
3. **H4 by Nov 24**: commit the three scripts or accept hypothesis wording.
4. **With Albert:** D-G, who writes the TC section, M2, post-hoc coverage and L7.
5. **Before submission, with Albert:** merge `dev` into `main`, then the fresh-clone check.

## Closed since 2026-11-20

**TL;DR / In plain terms:** the TC runs are done and checked, your three suggestions to Albert are settled, and two small items need no work.

- **TC agent runs** (Albert, 2026-11-20, PR #47): 40 runs from clean commit `1787f0d`, scored on m = 4,000 paths, with the HTML report (`src/tc_drift/report.py`, tested; `reports/tc-drift/report.html`). Albert recomputed the per-seed gaps from `values.npz` (within 1e-13). On 2026-11-22 I re-derived each level's agent − oracle mean and across-seed SD, the seed-averaged path SE, agent − always-rebalance, agent − myopic, the lowest seed against trade once then hold, the split and the oracle's lead with its SE from the per-path values; every one matches `results/tc-drift/runs/README.md` to the printed digit, and the split sums to agent − oracle on every path within 6e-14.
- **TC go/no-go (Nov 24):** met.
- **Your three `run.py` suggestions:** Albert took the first two (the seed `.npz` files and `values.npz` are committed; the seed-averaged gap's path SE is reported). The third, batched scoring, was not needed: scoring took about 13 minutes a level.
- **PR #43** (`feature/tc-drift-runs`) closed unmerged 2026-11-20. Its commits stay on GitHub under `refs/pull/43/head`.
- **Final non-TC runs:** none (2026-11-22). C2, C3, P1 and P2 are not run.
- **`update()` return value** (`REVIEW_HISTORY.md` §5.3): not needed, since no further 10% grid tabular run is planned.
- **Albert's audit edits to your files** (`AUDIT_2026-11-20.md` §3.1, §3.2): the test count (757 to 758), the PR #45 merge, PR #43 marked closed, and "barely move from θ = 0" reworded to "not finished learning" (his audit's L7, not ours). Seen 2026-11-22: the merge matches `origin/main`; PR #43's closed state comes from Albert's `gh` check and is not re-checked here (`gh` is not installed on this machine).
- **Local branches:** the seven whose remotes were deleted are pruned on this machine (2026-11-22). `origin/feature/tc-drift-run` and `origin/run/tc-drift-final` are fully merged into `dev` and can be deleted on GitHub.
- **Windows tests:** 760 of 761 passed on Windows (Python 3.10.2, numpy 2.2.6, matplotlib 3.10.9) on 2026-11-22; the one failure, `test_committed_report_matches_a_rebuild`, comes from the matplotlib version changing the SVG bytes, and `data.json` rebuilds identically (detail in `NEXT_STEPS_FOR_ALBERT.md`).

## History

**From Albert, 2026-11-20: the final runs are ready** (`feature/tc-drift-run`).
- **Built.** `src/tc_drift/run.py` (TC-2): `python -m src.tc_drift.run all` trains 4 levels × 10 seeds, 7 at a time with BLAS pinned, then scores each level on m = 4,000 paths against the oracle, myopic, always rebalance, trade once then hold, and the own-target policy (TC-3, with the split checked as an identity on every path). It reports the tail-averaged policy as the headline and the last iterate alongside. All four D12 verdicts and the Part IV counts come from your `tune.convergence` and `tune.averaged_policy`. `train()` now logs θ and the favourite target at every checkpoint (TC-6). A test checks that one call gives exactly your `train_segmented` run.
- **Decided by Albert:** m = 4,000, fixed before any agent result (TC-7, D-E).
- **Checkpoint interval: 40,050, not the 40,000 you suggested.** It equals `tune.SEGMENT`, so the final runs have the layout design §6 registered and the pilot used: 11 tail-average snapshots and a 12-checkpoint D12 window from block 3,564,450. With 40,000 there would be 10 and 11. Found in the independent review.
- **Expected directions (D-F).** My preliminary E1 to E12 are in `TC_DRIFT_DESIGN.md` §7, written before the runs. Please add or amend them. Anything added after the runs start gets labelled post hoc.
- **Timing on my Mac:** 0.39 ms a block with 7 runs at once, so about 26 minutes a run and about 3 hours for all 40.
- **Your files:** I did not edit `tune.py`. Its `run_one` defaults to `m = config.M_EVAL`, which is now 4,000, while your pilot used 1,000. A rerun of `tune train` would therefore differ from the pilot unless it passes `m=config.M_EVAL_PILOTS`. Your call.
- **D7:** the pointer in `TRANSACTION_COSTS.md` is added.

*William, 2026-11-20:* all answered. E1 to E12 stay as registered, unchanged (D-F). `tune.run_one` now defaults to `m = config.M_EVAL_PILOTS`, with a test. 40,050 is right. Check 5's form: yes to both (D-I, design §5).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
